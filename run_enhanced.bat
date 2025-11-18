@echo off
chcp 65001 > nul
setlocal enabledelayedexpansion

title Advanced RVC Inference V3.2 Enhanced
echo Starting Advanced RVC Inference...
echo.

:: Check if virtual environment exists
if not exist "rvc_env" (
    echo Virtual environment not found. Please run install_enhanced.bat first.
    pause
    exit /b 1
)

:: Activate virtual environment
echo Activating virtual environment...
call rvc_env\Scripts\activate.bat

:: Check for command line arguments
set SHARE_MODE=false
set DEBUG_MODE=false
set CUSTOM_PORT=7860

:parse_args
if "%1"=="" goto start_app
if "%1"=="--share" set SHARE_MODE=true
if "%1"=="--debug" set DEBUG_MODE=true
if "%1"=="--port" (
    set CUSTOM_PORT=%2
    shift
)
if "%1"=="--help" (
    echo Usage: run_enhanced.bat [--share] [--debug] [--port PORT]
    echo   --share     Enable public sharing
    echo   --debug     Enable debug logging
    echo   --port PORT Set custom port (default: 7860)
    pause
    exit /b 0
)
shift
goto parse_args

:start_app
echo.
echo ======================================
echo Advanced RVC Inference V3.2 Enhanced
echo Configuration:
echo   - Share Mode: %SHARE_MODE%
echo   - Debug Mode: %DEBUG_MODE%
echo   - Port: %CUSTOM_PORT%
echo ======================================
echo.

:: Set environment variables
set PYTHONPATH=%CD%
set RVC_CONFIG=config_enhanced.json

:: Launch application
if "%SHARE_MODE%"=="true" (
    if "%DEBUG_MODE%"=="true" (
        python app.py --share --debug --port %CUSTOM_PORT%
    ) else (
        python app.py --share --port %CUSTOM_PORT%
    )
) else (
    if "%DEBUG_MODE%"=="true" (
        python app.py --debug --port %CUSTOM_PORT%
    ) else (
        python app.py --port %CUSTOM_PORT%
    )
)

:: Keep console open
echo.
echo Application stopped. Press any key to exit...
pause >nul