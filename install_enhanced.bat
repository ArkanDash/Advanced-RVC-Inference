@echo off
chcp 65001 > nul
setlocal enabledelayedexpansion

echo ======================================
echo Advanced RVC Inference Enhanced Installer
echo Based on Vietnamese-RVC improvements
echo ======================================

set GPU_NVIDIA=0
set GPU_AMD=0
set DETECTED_GPU=None

echo Detecting GPU hardware...

:: GPU Detection
for /f "tokens=*" %%i in ('wmic path win32_VideoController get Name') do (
    echo %%i | find /i "GTX" >nul && set GPU_NVIDIA=1 && set DETECTED_GPU=NVIDIA
    echo %%i | find /i "RTX" >nul && set GPU_NVIDIA=1 && set DETECTED_GPU=NVIDIA
    echo %%i | find /i "NVIDIA" >nul && set GPU_NVIDIA=1 && set DETECTED_GPU=NVIDIA
    echo %%i | find /i "Quadro" >nul && set GPU_NVIDIA=1 && set DETECTED_GPU=NVIDIA
    echo %%i | find /i "GeForce" >nul && set GPU_NVIDIA=1 && set DETECTED_GPU=NVIDIA
    echo %%i | find /i "RX" >nul && set GPU_AMD=1 && set DETECTED_GPU=AMD
    echo %%i | find /i "AMD" >nul && set GPU_AMD=1 && set DETECTED_GPU=AMD
    echo %%i | find /i "Vega" >nul && set GPU_AMD=1 && set DETECTED_GPU=AMD
    echo %%i | find /i "Radeon" >nul && set GPU_AMD=1 && set DETECTED_GPU=AMD
    echo %%i | find /i "FirePro" >nul && set GPU_AMD=1 && set DETECTED_GPU=AMD
)

:: Handle multiple GPUs (prioritize NVIDIA)
if %GPU_NVIDIA%==1 if %GPU_AMD%==1 (
    echo Detected both NVIDIA and AMD. Prioritizing NVIDIA GPU.
    set GPU_NVIDIA=1
    set GPU_AMD=0
    set DETECTED_GPU=NVIDIA
)

:: Set GPU type
if %GPU_NVIDIA%==1 (
    echo GPU detected: NVIDIA
    set INSTALL_TYPE=nvidia_gpu
) else if %GPU_AMD%==1 (
    echo GPU detected: AMD
    set INSTALL_TYPE=amd_gpu
) else (
    echo GPU detected: None (CPU only)
    set INSTALL_TYPE=cpu
)

echo Installation type: %INSTALL_TYPE%
echo.

:: Python version check
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Please install Python 3.8+ from https://python.org
    pause
    exit /b 1
)

for /f "tokens=2" %%v in ('python --version 2^>^&1') do (
    set PYTHON_VERSION=%%v
)
echo Detected Python version: %PYTHON_VERSION%

:: Create virtual environment
echo Creating virtual environment...
if not exist "rvc_env" (
    python -m venv rvc_env
    echo Virtual environment created.
) else (
    echo Virtual environment already exists.
)

:: Activate virtual environment
echo Activating virtual environment...
call rvc_env\Scripts\activate.bat

:: Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

:: Install base requirements
echo Installing base requirements...
python -m pip install -r requirements.txt

:: Install GPU-specific packages
if "%INSTALL_TYPE%"=="nvidia_gpu" (
    echo Installing NVIDIA GPU support...
    python -m pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cu118
) else if "%INSTALL_TYPE%"=="amd_gpu" (
    echo Installing AMD GPU support...
    :: Note: AMD GPU support is limited and experimental
    echo AMD GPU support requires additional setup. Consider using DirectML or ROCm.
) else (
    echo CPU-only installation - no additional packages needed.
)

echo.
echo ======================================
echo Installation completed!
echo ======================================
echo.
echo To run the application:
echo   1. Activate the environment: rvc_env\Scripts\activate
echo   2. Run: python app.py
echo.
echo If using shared mode: python app.py --share
echo.
pause