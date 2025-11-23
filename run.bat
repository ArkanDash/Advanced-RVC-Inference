@echo off
chcp 65001 >nul
set PYTHONIOENCODING=utf-8

echo Starting Advanced RVC Inference - KRVC Kernel...
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Python is not installed or not in PATH. Please install Python 3.8 or higher.
    pause
    exit /b 1
)

REM Check if requirements are installed
echo Checking dependencies...
pip list | findstr -i -c:"gradio" >nul 2>&1
if errorlevel 1 (
    echo Installing requirements...
    pip install -r requirements.txt
)

REM Start the application
echo Starting Advanced RVC Inference...
python -m advanced_rvc_inference.main %*

pause