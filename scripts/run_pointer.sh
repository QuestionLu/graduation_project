#!/usr/bin/env bash
# 指點測試 (Linux)，參數原樣傳給 pointer_test.py
# 例: bash scripts/run_pointer.sh --mode rel
cd "$(dirname "$(readlink -f "$0")")/.."

if [ ! -x .venv/bin/python ]; then
    echo "請先執行: bash scripts/setup.sh"
    exit 1
fi

export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-xcb}"

.venv/bin/python src/hand_tracking/pointer_test.py "$@"
