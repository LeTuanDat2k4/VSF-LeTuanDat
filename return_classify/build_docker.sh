#!/bin/bash
# ===================================================
#   Return Prediction API - Build Docker Image Script
# ===================================================
set -e

echo "==================================================="
echo "  Return Prediction API - Build Docker Image Script"
echo "==================================================="
echo

echo "[Step 1] Chuẩn bị thư mục model..."
python3 -X utf8 prepare_model.py
if [ $? -ne 0 ]; then
    echo "[ERROR] Chuẩn bị model thất bại!"
    exit 1
fi

echo
echo "[Step 2] Tiến hành build Docker Image..."
docker build -t return-prediction-api .
if [ $? -ne 0 ]; then
    echo "[ERROR] docker build thất bại!"
    exit 1
fi

echo
echo "[Step 3] Dọn dẹp thư mục tạm..."
rm -rf model_artifact
echo "[SUCCESS] Đã hoàn thành build Docker Image: return-prediction-api!"
echo
