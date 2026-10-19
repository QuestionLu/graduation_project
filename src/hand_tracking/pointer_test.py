"""
pointer_test.py — 「指點測試」：用食指當游標、拇指+食指捏合當點擊，量測指向效能

影像視窗裡會隨機出現圓形目標。把游標移進圓內，再捏合點擊。
每一題都會記錄：花多少時間、目標多遠、點了幾次才中。
結束（按 Q 或做完）後印出摘要，存成 last_pointer.txt，
並把每一題的資料「附加」到 pointer_results.csv（多次實驗累積在同一個檔案，方便比較）。

這個程式不會控制真正的滑鼠，只在視窗內測試。

用法：
  python pointer_test.py                  # 絕對模式：手的位置 = 游標位置
  python pointer_test.py --mode rel       # 相對模式：像觸控板，握拳 = 暫停移動（手不用一直舉高）
  python pointer_test.py --no-filter      # 關閉游標平滑，比較抖動差異
  python pointer_test.py --anchor mcp     # 用食指根部當定位點（捏合時比較不會漂）
  python pointer_test.py --selftest       # 不開鏡頭，只測試邏輯

離開：先用滑鼠點一下影像視窗，再按 Q 或 Esc；或在終端機按 Ctrl+C。
"""

import argparse
import csv
import math
import os
import random
import statistics
import sys
import time
from collections import deque
from datetime import datetime

import cv2

from hand_demo import (INDEX_TIP, classify, find_camera, finger_states,  # noqa: F401
                       mp_draw, mp_hands, pinch_ratio)

INDEX_MCP = 5
HERE = os.path.dirname(os.path.abspath(__file__))
FONT = cv2.FONT_HERSHEY_SIMPLEX


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


# ── 游標平滑：One Euro Filter ─────────────────────────────────────────────────

class OneEuro:
    """單一軸的 One Euro Filter：慢速時強力去抖，快速移動時自動降低延遲。

    min_cutoff 越小 → 靜止時越穩（但慢速移動會有延遲）
    beta       越大 → 快速移動時越跟手
    （這裡的座標單位是像素，beta 0.01 左右是合理的起點。）
    """

    def __init__(self, min_cutoff=1.0, beta=0.01, d_cutoff=1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.reset()

    def reset(self):
        self.t = self.raw = self.x = self.dx = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x, t):
        if self.t is None:
            self.t, self.raw, self.x, self.dx = t, x, x, 0.0
            return x
        dt = max(t - self.t, 1e-3)
        a_d = self._alpha(self.d_cutoff, dt)
        self.dx = a_d * ((x - self.raw) / dt) + (1 - a_d) * self.dx
        a = self._alpha(self.min_cutoff + self.beta * abs(self.dx), dt)
        self.x = a * x + (1 - a) * self.x
        self.t, self.raw = t, x
        return self.x


# ── 手 → 游標 + 點擊（與攝影機、畫面無關，方便單獨測試）────────────────────────

class Pointer:
    """把每張畫面的手部資訊變成「游標位置」與「點擊事件」。

    mode='abs'：手在畫面中央一塊區域（box，佔畫面的比例）內的位置，對應到整個畫布。
    mode='rel'：像觸控板，游標位移 = 手的位移 × gain；握拳時暫停（像手指離開觸控板）。

    點擊 = 拇指與食指的距離比（pinch ratio）降到 pinch_on 以下；
    升到 pinch_off 以上才算放開（兩個門檻之間不會重複觸發）。
    捏合時手指會晃動、游標會漂移，所以點擊位置取「lookback 秒前」的游標位置，
    並在捏合期間凍結游標。
    """

    def __init__(self, w, h, mode='abs', box=0.6, gain=2.5, smooth=True,
                 min_cutoff=1.0, beta=0.01, pinch_on=0.35, pinch_off=0.60,
                 lookback=0.15):
        self.w, self.h, self.mode = w, h, mode
        self.box, self.gain, self.smooth = box, gain, smooth
        self.pinch_on, self.pinch_off, self.lookback = pinch_on, pinch_off, lookback
        self.fx = OneEuro(min_cutoff, beta)
        self.fy = OneEuro(min_cutoff, beta)
        self.cursor = (w / 2, h / 2)
        self.hist = deque(maxlen=120)   # (時間, x, y)：最近的游標軌跡
        self.pinched = False
        self.present = False
        self.prev = None                # 相對模式用：上一張的定位點

    def lose(self):
        """手不見了：清掉暫存狀態（游標留在原位）。"""
        self.present = self.pinched = False
        self.prev = None
        self.hist.clear()
        self.fx.reset()
        self.fy.reset()

    def _lookback(self, t):
        target, pos = t - self.lookback, None
        for ts, x, y in self.hist:
            if ts <= target:
                pos = (x, y)
            else:
                break
        if pos is None:
            pos = self.hist[0][1:] if self.hist else self.cursor
        return pos

    def update(self, t, anchor, ratio, fist=False):
        """anchor：定位點的像素座標 (x, y)，沒偵測到手給 None。
        回傳：這張畫面剛發生的點擊位置 (x, y)，沒有點擊則為 None。"""
        if anchor is None:
            self.lose()
            return None
        self.present = True
        ax, ay = anchor
        if self.smooth:
            ax, ay = self.fx(ax, t), self.fy(ay, t)

        if not self.pinched:
            if self.mode == 'abs':
                lo = 0.5 - self.box / 2
                nx = (ax / self.w - lo) / self.box
                ny = (ay / self.h - lo) / self.box
                self.cursor = (clamp(nx, 0, 1) * self.w, clamp(ny, 0, 1) * self.h)
            elif self.prev is not None and not fist:
                cx, cy = self.cursor
                self.cursor = (clamp(cx + (ax - self.prev[0]) * self.gain, 0, self.w),
                               clamp(cy + (ay - self.prev[1]) * self.gain, 0, self.h))
            self.hist.append((t, *self.cursor))
        self.prev = (ax, ay)   # 相對模式：握拳或捏合期間也持續更新，放開後才不會跳

        click = None
        if not self.pinched and ratio < self.pinch_on:
            self.pinched = True
            self.cursor = click = self._lookback(t)
        elif self.pinched and ratio > self.pinch_off:
            self.pinched = False
        return click


# ── 題目、命中判定、記錄 ──────────────────────────────────────────────────────

class Session:
    """隨機產生目標，判定點中／點歪，記錄每一題的資料。
    前 warmup 題是暖身（要先把手舉進畫面、熟悉操作），會記錄但不列入統計。"""

    def __init__(self, w, h, n_trials=20, radius=32, seed=1, min_dist=160, warmup=1):
        self.w, self.h, self.r = w, h, radius
        self.n, self.warmup, self.min_dist = n_trials, warmup, min_dist
        self.rng = random.Random(seed)   # 固定種子 → 每次的目標順序相同，方便公平比較
        self.rows = []
        self.target = None
        self.prev_target = (w / 2, h / 2)
        self.t_start = None
        self.attempts = 0
        self.started = False

    @property
    def total(self):
        return self.n + self.warmup

    @property
    def done(self):
        return len(self.rows) >= self.total

    def _new_target(self):
        m = self.r + 20
        while True:
            x = self.rng.uniform(m, self.w - m)
            y = self.rng.uniform(m, self.h - m)
            if math.hypot(x - self.prev_target[0], y - self.prev_target[1]) >= self.min_dist:
                self.target = (x, y)
                return

    def start(self, t):
        if not self.started:
            self.started = True
            self._new_target()
            self.t_start = t

    def inside(self, pos):
        return (self.target is not None and
                math.hypot(pos[0] - self.target[0], pos[1] - self.target[1]) <= self.r)

    def click(self, t, pos):
        """回傳 'hit' / 'miss'；還沒開始或已做完則回傳 None。"""
        if not self.started or self.done:
            return None
        self.attempts += 1
        if not self.inside(pos):
            return 'miss'
        d = math.hypot(self.target[0] - self.prev_target[0], self.target[1] - self.prev_target[1])
        width = 2 * self.r
        self.rows.append({
            'warmup': int(len(self.rows) < self.warmup),
            'trial': len(self.rows) + 1,
            'dist_px': round(d, 1),
            'width_px': width,
            'id_bits': round(math.log2(d / width + 1), 3),   # Fitts 難度指數
            'time_ms': round((t - self.t_start) * 1000),
            'attempts': self.attempts,
            'offset_px': round(math.hypot(pos[0] - self.target[0], pos[1] - self.target[1]), 1),
        })
        self.prev_target, self.attempts = self.target, 0
        if self.done:
            self.target = None
        else:
            self._new_target()
            self.t_start = t
        return 'hit'


# ── 統計與存檔 ────────────────────────────────────────────────────────────────

FIELDS = ['run', 'mode', 'anchor', 'filter', 'box', 'gain', 'min_cutoff', 'beta',
          'pinch_on', 'pinch_off', 'lookback', 'complexity',
          'warmup', 'trial', 'dist_px', 'width_px', 'id_bits', 'time_ms',
          'attempts', 'offset_px']


def make_params(args, run_id):
    return {
        'run': run_id, 'mode': args.mode, 'anchor': args.anchor,
        'filter': 'off' if args.no_filter else 'on', 'box': args.box, 'gain': args.gain,
        'min_cutoff': args.min_cutoff, 'beta': args.beta, 'pinch_on': args.pinch_on,
        'pinch_off': args.pinch_off, 'lookback': args.lookback, 'complexity': args.complexity,
    }


def summarize(ses, params) -> str:
    head = (f"mode={params['mode']}  anchor={params['anchor']}  filter={params['filter']}  "
            f"complexity={params['complexity']}\n")
    rows = [r for r in ses.rows if not r['warmup']]
    if not rows:
        return head + '沒有完成任何一題（暖身題不計），沒有統計資料\n'
    times = [r['time_ms'] / 1000 for r in rows]
    pending = ses.attempts if len(ses.rows) >= ses.warmup else 0   # 還沒點中的那題
    clicks = sum(r['attempts'] for r in rows) + pending
    misses = clicks - len(rows)
    return head + (
        f'trials done     : {len(rows)} / {ses.n}\n'
        f'time mean/median: {statistics.mean(times):.2f} s / {statistics.median(times):.2f} s\n'
        f'clicks / misses : {clicks} / {misses}  (miss rate {misses / clicks:.0%})\n'
        f'mean difficulty : {statistics.mean(r["id_bits"] for r in rows):.2f} bits\n'
        f'throughput      : {sum(r["id_bits"] for r in rows) / sum(times):.2f} bits/s'
        f'  (簡化版，不是 ISO 的有效寬度算法)\n'
    )


def save_results(rows, params, path):
    new = not os.path.exists(path)
    with open(path, 'a', newline='', encoding='utf-8') as f:
        wr = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            wr.writeheader()
        for r in rows:
            wr.writerow({**params, **r})


# ── 畫面 ─────────────────────────────────────────────────────────────────────

def put(frame, text, org, scale=0.6, color=(80, 255, 120)):
    text = text.encode('ascii', 'ignore').decode()   # OpenCV 內建字型不支援中文，畫面上只放英文
    cv2.putText(frame, text, org, FONT, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame, text, org, FONT, scale, color, 2, cv2.LINE_AA)


def draw(frame, ptr, ses, ratio, args, flash=None, summary=None):
    h, w = frame.shape[:2]
    if args.mode == 'abs':   # 手要在這個框裡移動，才對得到整個畫面
        lo, hi = 0.5 - args.box / 2, 0.5 + args.box / 2
        cv2.rectangle(frame, (int(lo * w), int(lo * h)), (int(hi * w), int(hi * h)),
                      (200, 200, 200), 1)

    if ses.target is not None:
        inside = ptr.present and ses.inside(ptr.cursor)
        col = (0, 200, 0) if inside else (0, 0, 255)
        tx, ty = int(ses.target[0]), int(ses.target[1])
        cv2.circle(frame, (tx, ty), ses.r, col, 3)
        cv2.circle(frame, (tx, ty), 3, col, -1)

    cx, cy = int(ptr.cursor[0]), int(ptr.cursor[1])
    ccol = (0, 255, 255) if ptr.pinched else ((255, 255, 255) if ptr.present else (128, 128, 128))
    cv2.drawMarker(frame, (cx, cy), ccol, cv2.MARKER_CROSS, 28, 2)
    cv2.circle(frame, (cx, cy), 9, ccol, 2)

    lines = [f"mode {args.mode}  filter {'off' if args.no_filter else 'on'}  anchor {args.anchor}"]
    if not ses.started:
        lines.append('show your hand to start')
    elif ses.done:
        lines.append('DONE   (click window, then Q / Esc)')
    else:
        k = len(ses.rows)
        label = 'warm-up' if k < ses.warmup else f'trial {k - ses.warmup + 1}/{ses.n}'
        lines.append(f'{label}   misses on this one: {max(ses.attempts, 0)}')
    if ses.rows:
        lines.append(f"last: {ses.rows[-1]['time_ms'] / 1000:.2f} s")
    if ratio is not None:
        lines.append(f'pinch {ratio:.2f}  (click<{args.pinch_on:.2f}  release>{args.pinch_off:.2f})')
    for i, text in enumerate(lines):
        put(frame, text, (10, 24 + i * 24), 0.55)

    if summary:
        rows = summary.strip().split('\n')
        for i, text in enumerate(rows):
            put(frame, text, (10, h - 12 - (len(rows) - 1 - i) * 22), 0.5, (255, 220, 120))
    if flash:
        put(frame, flash[0], (w // 2 - 50, 60), 1.2, flash[1])


# ── 主程式 ────────────────────────────────────────────────────────────────────

def run(args):
    cap = find_camera(args.camera)
    if cap is None:
        sys.exit('找不到可用攝影機：確認鏡頭沒被其他程式佔用，或用 --camera 指定編號。')
    ok, frame = cap.read()
    if not ok:
        cap.release()
        sys.exit('讀不到攝影機畫面')
    h, w = frame.shape[:2]

    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                           model_complexity=args.complexity,
                           min_detection_confidence=0.6, min_tracking_confidence=0.5)
    ptr = Pointer(w, h, args.mode, args.box, args.gain, not args.no_filter,
                  args.min_cutoff, args.beta, args.pinch_on, args.pinch_off, args.lookback)
    ses = Session(w, h, args.trials, args.radius, args.seed)
    params = make_params(args, datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    flash, summary = None, None
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print('[camera] 讀不到畫面，結束')
                break
            frame = cv2.flip(frame, 1)   # 鏡像，操作起來比較直覺
            t = time.perf_counter()

            res = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            anchor = ratio = None
            fist = False
            if res.multi_hand_landmarks:
                hand = res.multi_hand_landmarks[0]
                lm = hand.landmark
                a = lm[INDEX_TIP if args.anchor == 'tip' else INDEX_MCP]
                anchor = (a.x * w, a.y * h)
                ratio = pinch_ratio(lm, w, h)
                states = finger_states(lm, w, h)
                fist = not any(states[k] for k in ('index', 'middle', 'ring', 'pinky'))
                if not args.no_skeleton:
                    mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

            click = ptr.update(t, anchor, ratio if ratio is not None else 1.0, fist)
            if anchor is not None:
                ses.start(t)
            if click is not None:
                result = ses.click(t, click)
                if result == 'hit':
                    flash = ('HIT', (0, 220, 0), t + 0.4)
                elif result == 'miss':
                    flash = ('MISS', (0, 0, 255), t + 0.4)
            if ses.done and summary is None:
                summary = summarize(ses, params)

            draw(frame, ptr, ses, ratio, args,
                 flash if flash and flash[2] > t else None, summary)
            cv2.imshow('pointer_test  (click here, then Q / Esc to quit)', frame)
            if cv2.waitKey(1) & 0xFF in (ord('q'), ord('Q'), 27):
                break
    except KeyboardInterrupt:
        print('\n[中斷] 收到 Ctrl+C，結束')
    finally:
        hands.close()
        cap.release()
        cv2.destroyAllWindows()
        finish(ses, params)


def finish(ses, params):
    text = summarize(ses, params)
    print('\n===== 指點測試摘要 =====\n' + text)
    try:
        if ses.rows:
            save_results(ses.rows, params, os.path.join(HERE, 'pointer_results.csv'))
            print('每題資料已附加到 pointer_results.csv')
        with open(os.path.join(HERE, 'last_pointer.txt'), 'w', encoding='utf-8') as f:
            f.write(text)
        print('摘要已存到 last_pointer.txt')
    except OSError as e:
        print(f'無法寫入結果檔：{e}')


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description='指點測試：食指游標 + 捏合點擊')
    ap.add_argument('--camera', type=int, default=-1, help='攝影機編號（預設自動找）')
    ap.add_argument('--complexity', type=int, choices=[0, 1], default=0, help='MediaPipe 模型大小')
    ap.add_argument('--mode', choices=['abs', 'rel'], default='abs',
                    help='abs=手的位置即游標位置；rel=像觸控板，握拳暫停')
    ap.add_argument('--anchor', choices=['tip', 'mcp'], default='tip',
                    help='定位點：食指指尖(tip)或食指根部(mcp)')
    ap.add_argument('--box', type=float, default=0.6,
                    help='abs 模式：畫面中央多大一塊對應整個畫布（0.3~1，預設 0.6）')
    ap.add_argument('--gain', type=float, default=2.5, help='rel 模式：游標位移 / 手的位移')
    ap.add_argument('--no-filter', action='store_true', help='關閉 One Euro 平滑')
    ap.add_argument('--min-cutoff', type=float, default=1.0, help='One Euro：越小越穩、越慢')
    ap.add_argument('--beta', type=float, default=0.01, help='One Euro：越大快速移動越跟手')
    ap.add_argument('--pinch-on', type=float, default=0.35, help='捏合比例低於此值 = 點擊')
    ap.add_argument('--pinch-off', type=float, default=0.60, help='捏合比例高於此值 = 放開')
    ap.add_argument('--lookback', type=float, default=0.15, help='點擊位置取幾秒前的游標位置')
    ap.add_argument('--trials', type=int, default=20, help='正式題數（另有 1 題暖身）')
    ap.add_argument('--radius', type=int, default=32, help='目標半徑（像素）')
    ap.add_argument('--seed', type=int, default=1, help='目標順序的亂數種子（固定 = 每次相同）')
    ap.add_argument('--no-skeleton', action='store_true', help='不畫手部骨架')
    ap.add_argument('--selftest', action='store_true', help='只測試邏輯，不開鏡頭')
    args = ap.parse_args(argv)
    if not args.pinch_on < args.pinch_off:
        ap.error('--pinch-on 必須小於 --pinch-off')
    if not 0.3 <= args.box <= 1.0:
        ap.error('--box 必須在 0.3 到 1 之間')
    return args


# ── 自我測試（不需要鏡頭）─────────────────────────────────────────────────────

def selftest():
    import tempfile

    import numpy as np
    bad = []

    def check(name, cond):
        print(f"[{'ok ' if cond else 'BAD'}] {name}")
        if not cond:
            bad.append(name)

    def near(a, b, tol=0.5):
        return all(abs(x - y) <= tol for x, y in zip(a, b))

    W, H = 640, 480

    # One Euro Filter
    rng = random.Random(0)
    raw = [200 + rng.uniform(-4, 4) for _ in range(120)]
    f = OneEuro(1.0, 0.01)
    out = [f(v, i / 14) for i, v in enumerate(raw)]
    check('平滑：手靜止時抖動至少減半',
          statistics.pstdev(out[20:]) < 0.5 * statistics.pstdev(raw[20:]))
    f = OneEuro(1.0, 0.01)
    for i in range(10):
        f(0.0, i / 14)
    seq = [f(300.0, (10 + i) / 14) for i in range(6)]
    check('平滑：快速移動 4 張內追上 95%', abs(seq[3] - 300) < 15)

    # 絕對模式的對應
    p = Pointer(W, H, 'abs', box=0.6, smooth=False)
    p.update(0.0, (W / 2, H / 2), 1.0)
    check('abs：畫面中央 → 游標中央', near(p.cursor, (W / 2, H / 2)))
    p.update(0.1, (0.2 * W, 0.2 * H), 1.0)
    check('abs：框的左上角 → 畫布左上角', near(p.cursor, (0, 0)))
    p.update(0.2, (0.8 * W, 0.8 * H), 1.0)
    check('abs：框的右下角 → 畫布右下角', near(p.cursor, (W, H)))
    p.update(0.3, (5, 5), 1.0)
    check('abs：框外會夾在邊緣', near(p.cursor, (0, 0)))

    # 點擊：回溯位置、凍結、遲滯
    def cur_x(k):
        return ((150 + 8 * k) / W - 0.2) / 0.6 * W

    p = Pointer(W, H, 'abs', box=0.6, smooth=False, lookback=0.15)
    clicks, frozen_ok, moved_after = [], True, False
    for k in range(40):
        ratio = 0.2 if 20 <= k < 26 else (0.5 if 26 <= k < 28 else 0.9)
        c = p.update(k / 30, (150 + 8 * k, 240), ratio)
        if c:
            clicks.append((k, c))
        if 20 <= k < 26 and not near(p.cursor, (cur_x(15), 240)):
            frozen_ok = False
        if k >= 30 and near(p.cursor, (cur_x(k), 240)):
            moved_after = True
    check('點擊：只觸發一次（捏合期間與遲滯區間不重複）', len(clicks) == 1 and clicks[0][0] == 20)
    check('點擊：位置取 0.15 秒前，不是捏合當下', near(clicks[0][1], (cur_x(15), 240)))
    check('點擊：捏合期間游標凍結', frozen_ok)
    check('點擊：放開後游標恢復跟手', moved_after)

    # 相對模式與握拳
    p = Pointer(W, H, 'rel', gain=2.0, smooth=False)
    for k in range(5):
        p.update(k / 30, (100 + 10 * k, 100), 1.0)
    check('rel：位移 × gain（第一張只記錄起點）', near(p.cursor, (400, 240)))
    for k in range(5, 10):
        p.update(k / 30, (150 + 50 * (k - 5), 100), 1.0, fist=True)
    check('rel：握拳時游標不動', near(p.cursor, (400, 240)))
    p.update(10 / 30, (360, 100), 1.0)
    check('rel：握拳張開後不會跳', near(p.cursor, (420, 240)))
    p.update(11 / 30, None, 1.0)
    check('手消失：游標留在原位', near(p.cursor, (420, 240)) and not p.present)

    # 題目與記錄
    s = Session(W, H, n_trials=2, radius=30, seed=1, min_dist=150, warmup=1)
    check('尚未開始時點擊無效', s.click(0.0, (10, 10)) is None)
    s.start(0.0)
    tx, ty = s.target
    check('點歪 → miss', s.click(0.5, (tx + 100, ty)) == 'miss')
    check('點中 → hit', s.click(1.0, (tx + 10, ty)) == 'hit')
    r = s.rows[0]
    check('記錄：1000 ms、點了 2 次、暖身', r['time_ms'] == 1000 and r['attempts'] == 2 and r['warmup'] == 1)
    s.click(2.0, s.target)
    s.click(3.5, s.target)
    check('做完 1 暖身 + 2 正式題', s.done and len(s.rows) == 3 and s.target is None)
    check('做完後不再接受點擊', s.click(9.0, (0, 0)) is None)
    check('每題目標至少相距 min_dist', all(row['dist_px'] >= 150 for row in s.rows))
    check('秒數正確（1.0 s、1.5 s）', [row['time_ms'] for row in s.rows[1:]] == [1000, 1500])

    # 摘要與存檔
    args = parse_args([])
    params = make_params(args, 'test')
    text = summarize(s, params)
    check('摘要：不含暖身題（2 題）', 'trials done     : 2 / 2' in text and 'throughput' in text)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, 'r.csv')
        save_results(s.rows, params, path)
        save_results(s.rows, params, path)
        with open(path, encoding='utf-8') as fh:
            n_lines = len(fh.read().strip().split('\n'))
    check('CSV：附加寫入（1 個標題 + 6 列）', n_lines == 7)

    # 畫面繪製（只確認不會出錯）
    frame = np.zeros((H, W, 3), np.uint8)
    s2 = Session(W, H)
    s2.start(0.0)
    draw(frame, Pointer(W, H), s2, 0.8, args, ('HIT', (0, 220, 0), 1.0), text)
    check('畫面繪製不出錯', frame.any())

    print('\n全部通過' if not bad else f'\n有 {len(bad)} 項失敗')
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    a = parse_args()
    selftest() if a.selftest else run(a)
