#!/usr/bin/env bash
# 手部追蹤示範 (Linux)，參數原樣傳給 hand_demo.py
# 例: bash scripts/run.sh --complexity 1
cd "$(dirname "$(readlink -f "$0")")/.."

if [ ! -x .venv/bin/python ]; then
    echo "請先執行: bash scripts/setup.sh"
    exit 1
fi

# OpenCV 內建的 Qt 在 Wayland 下常找不到外掛，預設走 X11（XWayland）；要改可自行設定此變數
export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-xcb}"

.venv/bin/python src/hand_tracking/hand_demo.py "$@"
