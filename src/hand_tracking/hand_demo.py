"""
hand_demo.py — 最小手部追蹤示範（筆電攝影機即時版）

取材自朋友專案的 rps_game.py（MediaPipe Hands 的用法），但：
  - 不依賴 Jetson / GStreamer / ultralytics / loguru
  - 手指判斷改成「旋轉不變」：用指尖到手腕的距離，而不是 tip.y < pip.y，
    所以手斜著、橫著、躺著用也不會整個失效
  - 畫面上顯示 FPS 與每張的推論時間，讓你知道這台筆電跑得動多少

用法：
  python hand_demo.py                 # 自動找鏡頭 0/1/2
  python hand_demo.py --camera 1      # 指定鏡頭
  python hand_demo.py --complexity 1  # 較準但較慢（預設 0 = 最輕量）
  python hand_demo.py --hands 1       # 只追蹤一隻手（預設 2，雙手）
  python hand_demo.py --selftest      # 不開鏡頭，只測試判斷邏輯

離開：先用滑鼠點一下影像視窗（讓它取得鍵盤焦點），再按 Q 或 Esc；
      或在終端機按 Ctrl+C。
"""

import argparse
import math
import sys
import time

import cv2

try:
    import mediapipe as mp
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
except (ImportError, AttributeError):
    sys.exit(
        "找不到 mediapipe.solutions。新版 mediapipe（0.10.30 之後）已移除這套舊 API。\n"
        "請用 Python 3.12，並安裝：pip install mediapipe==0.10.21"
    )

# ── MediaPipe 手部 21 個關節點的編號 ──────────────────────────────────────────
WRIST, THUMB_IP, THUMB_TIP = 0, 3, 4
INDEX_TIP, MIDDLE_MCP, PINKY_MCP = 8, 9, 17
# 四根手指：(指尖編號, 第二關節 PIP 編號)
FINGERS = {
    'index':  (8, 6),
    'middle': (12, 10),
    'ring':   (16, 14),
    'pinky':  (20, 18),
}


def _dist(lm, a: int, b: int, w: int, h: int) -> float:
    """兩個關節點的像素距離（x、y 要各乘寬高，否則長寬比會失真）。"""
    return math.hypot((lm[a].x - lm[b].x) * w, (lm[a].y - lm[b].y) * h)


def finger_states(lm, w: int, h: int) -> dict[str, bool]:
    """回傳每根手指是否伸直。

    四指：指尖離手腕比 PIP 關節離手腕更遠 → 伸直（與手的朝向無關）。
    拇指：指尖離小指根部比拇指 IP 關節更遠 → 伸出（拇指收進手掌時會比較近）。
    """
    states = {
        name: _dist(lm, WRIST, tip, w, h) > _dist(lm, WRIST, pip, w, h)
        for name, (tip, pip) in FINGERS.items()
    }
    states['thumb'] = (
        _dist(lm, THUMB_TIP, PINKY_MCP, w, h) > _dist(lm, THUMB_IP, PINKY_MCP, w, h)
    )
    return states


def classify(states: dict[str, bool]) -> str:
    """把手指狀態轉成手勢名稱。想加新手勢就改這裡。"""
    n = sum(states[k] for k in FINGERS)  # 伸直的四指數量
    if n == 4:
        return 'OPEN'
    if n == 0:
        return 'THUMB' if states['thumb'] else 'FIST'
    if n == 1 and states['index']:
        return 'POINT'
    if n == 2 and states['index'] and states['middle']:
        return 'PEACE'
    return '?'


def pinch_ratio(lm, w: int, h: int) -> float:
    """拇指尖到食指尖的距離 ÷ 手掌大小。越小越接近「捏合」（之後做點擊會用到）。"""
    palm = _dist(lm, WRIST, MIDDLE_MCP, w, h)
    return _dist(lm, THUMB_TIP, INDEX_TIP, w, h) / palm if palm > 1e-6 else 1.0


# ── 攝影機 ────────────────────────────────────────────────────────────────────

def open_camera(index: int, width: int = 640, height: int = 480):
    """開啟攝影機。Windows 用 DirectShow 開得比較快、比較穩。"""
    backend = cv2.CAP_DSHOW if sys.platform.startswith('win') else cv2.CAP_ANY
    cap = cv2.VideoCapture(index, backend)
    if not cap.isOpened():
        cap.release()
        return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 少囤幾張畫面，降低延遲（部分驅動無效）
    ok, _ = cap.read()
    if not ok:
        cap.release()
        return None
    return cap


def find_camera(preferred: int):
    indices = (preferred,) if preferred >= 0 else (0, 1, 2)
    for i in indices:
        cap = open_camera(i)
        if cap is not None:
            print(f'[camera] 使用 index {i}')
            return cap
    return None


# ── 主程式 ────────────────────────────────────────────────────────────────────

def run(camera_index: int, complexity: int, num_hands: int) -> None:
    cap = find_camera(camera_index)
    if cap is None:
        sys.exit('找不到可用攝影機：確認鏡頭沒被其他程式佔用，或用 --camera 指定編號。')

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=num_hands,
        model_complexity=complexity,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.5,
    )

    fps, prev_t = 0.0, time.perf_counter()
    stamps: list[float] = []      # 每張畫面的時間戳，結束時用來算平均 FPS
    infer_hist: list[float] = []  # 每張的推論時間（毫秒）
    frame_size = None
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print('[camera] 讀不到畫面，結束')
                break

            frame = cv2.flip(frame, 1)  # 鏡像，操作起來比較直覺
            h, w = frame.shape[:2]

            t0 = time.perf_counter()
            res = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            infer_ms = (time.perf_counter() - t0) * 1000
            infer_hist.append(infer_ms)
            frame_size = (w, h)

            lines = []
            if res.multi_hand_landmarks:
                for i, hand in enumerate(res.multi_hand_landmarks):
                    mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
                    lm = hand.landmark
                    # 畫面已先鏡像，MediaPipe 的左右手標籤就是使用者自己的左/右手
                    side = res.multi_handedness[i].classification[0].label

                    states = finger_states(lm, w, h)
                    gesture = classify(states)
                    fingers = ' '.join(
                        f"{k[0].upper()}{'1' if v else '0'}" for k, v in states.items()
                    )
                    lines += [
                        f'{side}: {gesture}',
                        f'  fingers: {fingers}',
                        f'  pinch: {pinch_ratio(lm, w, h):.2f}'
                        f'  tip: ({lm[INDEX_TIP].x:.2f}, {lm[INDEX_TIP].y:.2f})',
                    ]
            else:
                lines.append('no hand')

            now = time.perf_counter()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev_t, 1e-6))
            prev_t = now
            stamps.append(now)
            lines.insert(0, f'FPS {fps:4.1f}   infer {infer_ms:4.1f} ms')

            for i, text in enumerate(lines):
                cv2.putText(frame, text, (10, 26 + i * 26),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
                cv2.putText(frame, text, (10, 26 + i * 26),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 255, 120), 2, cv2.LINE_AA)

            cv2.imshow('hand_demo  (click here, then Q / Esc to quit)', frame)
            # waitKey 只收得到「影像視窗」的按鍵；焦點在終端機時按 Q 沒用
            if cv2.waitKey(1) & 0xFF in (ord('q'), ord('Q'), 27):
                break
    except KeyboardInterrupt:
        print('\n[中斷] 收到 Ctrl+C，結束')
    finally:
        hands.close()
        cap.release()
        cv2.destroyAllWindows()
        report(stamps, infer_hist, frame_size, complexity)


def report(stamps, infer_hist, frame_size, complexity) -> None:
    """結束時印出效能摘要，並存成 last_run.txt（把這份內容貼給 Claude 就好）。"""
    warm = 20  # 跳過前 20 張（模型暖機，數字不準）
    if len(stamps) <= warm + 10:
        print('畫面太少，沒有足夠資料產生效能摘要（請多跑幾秒再按 Q）')
        return
    s, inf = stamps[warm:], sorted(infer_hist[warm:])
    avg_fps = (len(s) - 1) / (s[-1] - s[0])
    p50, p95 = inf[len(inf) // 2], inf[int(len(inf) * 0.95)]
    import os
    import platform
    text = (
        f'avg FPS        : {avg_fps:.1f}\n'
        f'infer median   : {p50:.1f} ms\n'
        f'infer p95      : {p95:.1f} ms\n'
        f'frames counted : {len(s)}\n'
        f'frame size     : {frame_size[0]}x{frame_size[1]}\n'
        f'model_complexity: {complexity}\n'
        f'mediapipe      : {mp.__version__}\n'
        f'python         : {platform.python_version()}\n'
        f'cpu            : {platform.processor()}\n'
    )
    print('\n===== 效能摘要 =====\n' + text)
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'last_run.txt')
    try:
        with open(path, 'w', encoding='utf-8') as f:
            f.write(text)
        print(f'已存到 {path}')
    except OSError as e:
        print(f'無法寫入 last_run.txt：{e}')


# ── 自我測試（不需要鏡頭）─────────────────────────────────────────────────────

class _P:
    def __init__(self, x, y):
        self.x, self.y = x, y


def _fake_hand(open_fingers: set[str], thumb_open: bool, angle_deg: float = 0.0):
    """用極座標捏一隻假手（21 點），再整隻旋轉 angle_deg，用來測試旋轉不變性。"""
    pts = [(0.0, 0.0)] * 21

    def polar(r, deg):
        a = math.radians(deg)
        return (r * math.sin(a), -r * math.cos(a))  # 0 度 = 朝上

    spread = {'index': -12, 'middle': 0, 'ring': 12, 'pinky': 24}
    pts[0] = (0.0, 0.0)
    for name, (tip, pip) in FINGERS.items():
        mcp, dip = pip - 1, pip + 1
        d = spread[name]
        pts[mcp] = polar(0.25, d)
        pts[pip] = polar(0.40, d)
        if name in open_fingers:
            pts[dip], pts[tip] = polar(0.50, d), polar(0.58, d)
        else:  # 彎下來：指尖縮回比 PIP 更靠近手腕
            pts[dip], pts[tip] = polar(0.33, d), polar(0.22, d)
    pts[PINKY_MCP] = polar(0.25, 24)
    pts[MIDDLE_MCP] = polar(0.25, 0)
    pts[2], pts[3] = polar(0.15, -50), polar(0.25, -55)
    pts[THUMB_TIP] = polar(0.36, -75) if thumb_open else polar(0.12, 5)

    c, s = math.cos(math.radians(angle_deg)), math.sin(math.radians(angle_deg))
    return [_P(0.5 + (x * c - y * s), 0.5 + (x * s + y * c)) for x, y in pts]


def selftest() -> None:
    cases = [
        ('OPEN',  {'index', 'middle', 'ring', 'pinky'}, True),
        ('FIST',  set(), False),
        ('THUMB', set(), True),
        ('POINT', {'index'}, False),
        ('PEACE', {'index', 'middle'}, False),
    ]
    bad = 0
    for angle in (0, 45, 90, 180, 270):  # 手朝上、斜、橫、倒著
        for want, fingers, thumb in cases:
            got = classify(finger_states(_fake_hand(fingers, thumb, angle), 640, 480))
            flag = 'ok ' if got == want else 'BAD'
            bad += got != want
            print(f'[{flag}] angle={angle:3d}  want={want:5s} got={got}')
    print('\n--- MediaPipe 模型載入測試（空白畫面）---')
    import numpy as np
    with mp_hands.Hands(model_complexity=0) as hands:
        r = hands.process(np.zeros((480, 640, 3), dtype=np.uint8))
        print('process ok, hands found:', bool(r.multi_hand_landmarks))
    print('\n全部通過' if bad == 0 else f'\n有 {bad} 項失敗')
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='筆電即時手部追蹤示範')
    ap.add_argument('--camera', type=int, default=-1, help='攝影機編號（預設自動找）')
    ap.add_argument('--complexity', type=int, choices=[0, 1], default=0,
                    help='模型大小：0 最輕量（預設）、1 較準但較慢')
    ap.add_argument('--hands', type=int, choices=[1, 2], default=2,
                    help='最多同時追蹤幾隻手（預設 2）')
    ap.add_argument('--selftest', action='store_true', help='只測試判斷邏輯，不開鏡頭')
    args = ap.parse_args()
    selftest() if args.selftest else run(args.camera, args.complexity, args.hands)
