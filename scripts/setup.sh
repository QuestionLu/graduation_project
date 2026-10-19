#!/usr/bin/env bash
# 安裝 (Linux)：建立 .venv、安裝套件、跑自我測試。從哪裡執行都可以：bash scripts/setup.sh
set -e
cd "$(dirname "$(readlink -f "$0")")/.."
echo "=== setup ==="

if ! command -v python3.12 >/dev/null 2>&1; then
    echo "[ERROR] 找不到 python3.12。"
    echo "目前的 python3 是: $(python3 --version 2>&1)"
    echo "請安裝 Python 3.12 後再執行（mediapipe 0.10.21 不支援 3.13 以上）。"
    exit 1
fi
echo "使用 $(python3.12 --version)"

if [ ! -x .venv/bin/python ]; then
    python3.12 -m venv .venv || {
        echo "[ERROR] 建立 venv 失敗，試試: sudo apt install python3.12-venv"
        exit 1
    }
fi

.venv/bin/python -m pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
.venv/bin/python src/hand_tracking/hand_demo.py --selftest

echo
echo "Setup OK. 接著執行: bash scripts/run.sh"
