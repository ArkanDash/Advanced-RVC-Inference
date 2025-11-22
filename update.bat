@echo off
chcp 65001 >nul
set PYTHONIOENCODING=utf-8

echo Updating Advanced RVC Inference - KRVC Kernel...
echo.

REM Check if git is installed
git --version >nul 2>&1
if errorlevel 1 (
    echo Git is not installed or not in PATH. Please install Git.
    pause
    exit /b 1
)

REM Pull the latest changes
echo Pulling latest changes from repository...
git pull origin main

REM Update requirements
echo Updating dependencies...
pip install -r requirements.txt --upgrade

echo Update completed successfully!
pause