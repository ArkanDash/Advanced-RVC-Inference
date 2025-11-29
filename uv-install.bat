@echo off
REM uv-install.bat - Script to install and setup the package with uv on Windows

echo Checking if uv is installed...

REM Check if uv is available
where uv >nul 2>&1
if errorlevel 1 (
    echo Installing uv...
    powershell -Command "Invoke-RestMethod -Uri https://astral.sh/uv/install.ps1 | Invoke-Expression"
    REM Refresh PATH to include uv
    call refreshenv 2>nul || echo Please restart your terminal after installation
)

REM Create a virtual environment using uv
echo Creating virtual environment...
uv venv

REM Activate the virtual environment
call .venv\Scripts\activate.bat

REM Install the package in development mode (this will create/update uv.lock)
echo Installing dependencies...
uv pip install -e .

REM Install additional development dependencies
uv pip install pytest black flake8 mypy

echo Setup complete! You can now run the application with:
echo   python -m advanced_rvc_inference.app
echo.
pause