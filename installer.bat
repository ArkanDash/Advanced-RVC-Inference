@echo off
REM Advanced RVC Inference - Windows Installation Script
REM This script installs all necessary dependencies for Advanced RVC Inference

echo ===========================================
echo Advanced RVC Inference Installation Script
echo ===========================================

REM Set up environment variables
set PIP_PREFER_BINARY=1
set PYTHONPATH=%CD%;%PYTHONPATH%

echo Setting up Python environment...

REM Install uv for fast package management
echo Installing uv...
powershell -Command "Invoke-RestMethod -Uri https://astral.sh/uv/install.ps1 | Invoke-Expression"

REM Add uv to PATH for current session
set PATH=%LOCALAPPDATA%\uv;%PATH%

REM Create virtual environment using uv
echo Creating virtual environment...
uv venv

REM Activate the virtual environment
call .venv\Scripts\activate.bat

echo Virtual environment activated.

REM Install torch with CUDA support
echo Installing PyTorch with CUDA support...
uv pip install --upgrade "torch>=2.0.0" "torchvision>=0.15.0" "torchaudio>=2.0.0" --index-url https://download.pytorch.org/whl/cu121

REM Install dependencies from requirements.txt
echo Installing requirements...
uv pip install -r requirements.txt --index-strategy unsafe-best-match

REM Install this package in development mode - only install if dependencies are available
echo Installing Advanced RVC Inference package...
uv pip install -e . || echo Warning: Development install failed, continuing with basic setup...

REM Install prerequisites for RVC
echo Installing RVC prerequisites...
python -c "from advanced_rvc_inference.core import run_prerequisites_script; run_prerequisites_script(pretraineds_hifigan=True, models=True, exe=True)"

echo ===========================================
echo Installation completed successfully!
echo ===========================================

echo To run the application, use one of the following commands:
echo   python -m advanced_rvc_inference.app              // Run with default settings
echo   python -m advanced_rvc_inference.app --share      // Run with public sharing
echo   python -m advanced_rvc_inference.app --listen     // Run with external access

echo ===========================================

pause