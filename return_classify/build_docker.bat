@echo off
echo ===================================================
echo   Return Prediction API - Build Docker Image Script
echo ===================================================
echo.

echo [Step 1] Chuan bi thu muc model...
python -X utf8 prepare_model.py
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Chuan bi model that bai!
    exit /b 1
)

echo.
echo [Step 2] Tien hanh build Docker Image...
docker build -t return-prediction-api .
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] docker build that bai!
    exit /b 1
)

echo.
echo [Step 3] Don dep thu muc tam...
rmdir /s /q model_artifact
echo [SUCCESS] Da hoan thanh build Docker Image: return-prediction-api!
echo.
pause
