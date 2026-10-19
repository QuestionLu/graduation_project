# 手勢控制電腦專題

用鏡頭辨識手勢，達到「不碰滑鼠也能操作電腦」。靈感來自躺在椅子上不想坐起來看影片的情境。

> 目前狀態：進行中（探索階段），細節會隨實驗結果調整。

## 展示影片

[Demo 影片（YouTube）]

原始影片檔放在 `media/videos/`：

| 檔案 | 內容 |
|---|---|
| `2026-10-19_demo1_anchor_mcp.webm` | 用食指根部當游標的定位點 |
| `2026-10-19_demo2_mode_rel.webm` | 相對移動模式（像觸控板） |
| `2026-10-19_demo3_no-filter.webm` | 不加平滑處理的對照組 |

測試截圖在 `media/screenshots/`（正常光線、關燈、雙手同時偵測）。

## 目前進度

已完成：
- 用 MediaPipe（Google 的手部追蹤套件）做即時手部追蹤，畫面顯示骨架、每秒幀數與推論時間，可同時追蹤雙手
- 辨識 5 種手勢：張開、握拳、比讚、食指、比 YA。判斷方式不受手的旋轉角度影響，躺著也能用
- 指點測試：用食指當游標、拇指＋食指捏合當點擊。含游標平滑、兩種游標模式（絕對／相對）、點擊位置修正，每題成績自動存成 CSV
- 測試情境：正常光線、關燈、雙手同時偵測

尚未完成：
- 躺姿與坐姿的比較實驗
- 啟動手勢（避免誤觸）
- 控制真正的滑鼠（目前只在視窗內測試）

## 如何執行

需要 **Python 3.12**（新版 MediaPipe 移除了舊的介面，所以版本需固定；3.13 以上裝不起來）。

### Linux

Ubuntu 需先安裝 `python3.12-venv`（`sudo apt install python3.12-venv`）。

```bash
bash scripts/setup.sh           # 建立 .venv、安裝套件、自我測試；看到 Setup OK 就成功
bash scripts/run.sh             # 手部追蹤示範
bash scripts/run_pointer.sh     # 指點測試
```

### Windows（尚未在 Windows 實機驗證）

安裝 Python 3.12 時勾選 *Add python.exe to PATH* 與 *py launcher*。

```bat
scripts\setup.bat
scripts\run.bat
scripts\run_pointer.bat
```

### 使用方式

- **離開**：先用滑鼠點一下影像視窗（讓它取得鍵盤焦點），再按 **Q** 或 **Esc**。焦點在終端機時按 Q 沒有作用。也可以在終端機按 Ctrl+C。
- **傳參數**：直接接在後面，例如 `bash scripts/run.sh --complexity 1`、`bash scripts/run_pointer.sh --mode rel`。完整參數用 `--help` 查看。

## 兩個程式在做什麼

程式都在 `src/hand_tracking/`。

### `hand_demo.py`：手部追蹤示範

開鏡頭，找出手的 21 個關節點，畫出骨架並判斷手勢。畫面會顯示：

| 項目 | 意思 |
|---|---|
| `FPS` / `infer` | 幀率 / 模型每張的推論毫秒數 |
| `Left:` / `Right:` + 手勢 | 哪一隻手（Left 就是你自己的左手），以及 OPEN、FIST、THUMB、POINT、PEACE；認不出來顯示 `?` |
| `fingers` | I M R P T = 食指、中指、無名指、小指、拇指，1 = 伸直 |
| `pinch` | 拇指與食指的距離 ÷ 手掌大小，越小越接近捏合 |

結束後印出效能摘要（平均 FPS、推論時間）。

手勢判斷用「指尖離手腕的距離是否比第二關節更遠」，而不是常見的「指尖 y 座標比關節高」。後者只在手直立時成立，手轉 180° 就整個相反。

常用參數：`--complexity 0/1`（模型大小，0 較快、1 較準）、`--hands 1/2`（最多追蹤幾隻手）、`--selftest`（不開鏡頭，只測試邏輯）。

### `pointer_test.py`：指點測試

視窗裡會隨機出現圓形目標。把游標移進圓內，再用拇指＋食指捏合點擊。第 1 題是暖身，不列入統計，之後預設 20 題。**不會控制真正的滑鼠**，只在視窗內測試。

運作方式：

1. **定位點**：用食指指尖（`tip`）或食指根部（`mcp`）當游標位置。
2. **平滑**：用 One Euro Filter。慢速移動時強力去抖，快速移動時自動放寬以降低延遲。
3. **兩種模式**：`abs` 是畫面中央的灰框對應整個畫布；`rel` 像觸控板，游標位移 = 手的位移 × 倍率，握拳就暫停移動。
4. **捏合點擊**：拇指與食指的距離比低於 `--pinch-on` 算點擊，高於 `--pinch-off` 才算放開。
5. **防止點擊時游標漂移**：捏合時手指會動，所以點擊位置取 `--lookback` 秒前的游標位置，捏合期間游標凍結。

| 常用參數 | 預設 | 意思 |
|---|---|---|
| `--mode abs/rel` | abs | 游標模式 |
| `--anchor tip/mcp` | tip | 定位點 |
| `--no-filter` | 開平滑 | 關閉平滑（用來對照） |
| `--pinch-on` / `--pinch-off` | 0.35 / 0.60 | 點擊／放開的捏合門檻，畫面上會即時顯示 `pinch` 數值方便調整 |
| `--lookback` | 0.15 | 點擊位置取幾秒前的游標位置 |
| `--trials` | 20 | 正式題數 |

**結果檔**：結束後摘要存成 `last_pointer.txt`，每一題的資料**附加**到 `pointer_results.csv`（多次實驗累積在同一個檔案）。欄位包含：實驗設定、目標距離與寬度、Fitts 難度（`id_bits`）、花費時間（`time_ms`）、點了幾次才中（`attempts`，1 = 一次就中）、點中時離中心多遠（`offset_px`）。這些檔案不會上傳 GitHub。

摘要裡的 `throughput` 是簡化算法，不是 ISO 9241-9 的標準算法，只適合比較自己的不同設定。

### 程式碼導覽

| 名稱 | 作用 |
|---|---|
| `hand_demo.py` 的 `finger_states()` / `classify()` | 判斷手指是否伸直、轉成手勢名稱（想新增手勢改 `classify`） |
| `hand_demo.py` 的 `pinch_ratio()` | 計算捏合程度 |
| `pointer_test.py` 的 `OneEuro` | 游標平滑 |
| `pointer_test.py` 的 `Pointer` | 把「手的位置＋捏合程度」變成「游標位置＋點擊」；與鏡頭無關，可單獨測試 |
| `pointer_test.py` 的 `Session` | 產生目標、判定點中或點歪、記錄每一題 |
| `scripts/` | 安裝與執行腳本（Linux `.sh`、Windows `.bat`），會自動處理 venv、版本檢查，Linux 版預設走 X11 以避開 Wayland 下 OpenCV 視窗開不起來的問題 |

兩個程式都有 `--selftest`，不需要鏡頭就能測試邏輯。

## 實測參考

筆電：Intel i5-7200U（940MX 2G）、12 GB RAM、Ubuntu 24.04、Python 3.12.3、mediapipe 0.10.21，640×480。

`hand_demo.py` 效能：

| | 平均 FPS | 推論中位數 | 推論 p95 |
|---|---|---|---|
| `--complexity 0` | 14.1 | 37.6 ms | 63.9 ms |
| `--complexity 1` | 13.2 | 53.2 ms | 96.9 ms |

complexity 1 的推論慢了約 40%，FPS 卻幾乎沒變，推測瓶頸在鏡頭（約 15 FPS 上限），不是模型。這只是推測，沒有另外驗證。

`pointer_test.py`（每種設定合併 2～4 次測試，樣本少、次與次之間差異大，僅供參考）：

| 設定 | 題數 | 首次命中 | 中位數 | 最慢一題 |
|---|---|---|---|---|
| abs + tip + 平滑 | 40 | 88% | 1.79 s | 11.6 s |
| abs + tip + 無平滑 | 80 | 79% | 1.51 s | 31.8 s |
| abs + mcp + 平滑 | 57 | 82% | 1.51 s | 4.6 s |
| rel + tip + 平滑 | 40 | 80% | 1.56 s | 17.1 s |

## 常見問題

- **找不到攝影機**：關掉其他用到鏡頭的程式（Zoom、Teams 等），或加 `--camera 1` 換編號。Linux 可用 `ls /dev/video*` 確認裝置存在。
- **按 Q 沒反應**：先用滑鼠點一下影像視窗再按。
- **視窗開不起來（Linux，Qt／xcb 相關錯誤）**：可嘗試 `sudo apt install libxcb-cursor0`。
- **`找不到 mediapipe.solutions`**：Python 版本不是 3.12，或 mediapipe 版本不對，請用 3.12 重新跑 setup。
- **終端機一堆 `WARNING`／`W0000` 訊息**：是 MediaPipe 的內部日誌，不影響使用。

## 資料夾說明

| 資料夾 | 內容 |
|---|---|
| `src/hand_tracking/` | 程式碼 |
| `scripts/` | 安裝與執行腳本 |
| `docs/` | 筆記與實驗紀錄 |
| `media/` | 測試截圖與 demo 影片 |

## 作者

姓名：盧奎勳
學號：C113151137
指導教授：鐘文鈺 教授

手部追蹤的用法參考了朋友的專案（已徵得同意）。
