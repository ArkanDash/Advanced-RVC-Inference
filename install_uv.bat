@echo off
REM Advanced RVC Inference - UV Installer Script (Windows)
REM This script installs the project using UV package manager for faster dependency resolution

echo 🚀 Advanced RVC Inference - UV Installation
echo ============================================

REM Check if Python is available
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python is not installed. Please install Python 3.10+ first.
    pause
    exit /b 1
)

REM Check Python version
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set python_version=%%i
echo ✅ Python %python_version% detected

REM Install UV if not already installed
uv --version >nul 2>&1
if %errorlevel% neq 0 (
    echo 📦 Installing UV package manager...
    powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
    
    REM Verify UV installation
    uv --version >nul 2>&1
    if %errorlevel% neq 0 (
        echo ❌ Failed to install UV. Please install manually: https://docs.astral.sh/uv/getting-started/installation/
        pause
        exit /b 1
    )
)

echo ✅ UV package manager is available

REM Create virtual environment and install dependencies
echo 🔧 Creating virtual environment and installing dependencies...
uv sync

echo ✅ Installation completed successfully!
echo.
echo 🎯 To run Advanced RVC Inference:
echo    uv run python -m advanced_rvc_inference.main
echo.
echo 🔧 To activate the virtual environment:
echo    .venv\Scripts\activate
echo.
echo 📚 For more information, see README.md
pause