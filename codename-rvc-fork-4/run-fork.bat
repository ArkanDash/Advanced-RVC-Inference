@echo off

if /i "%cd%"=="C:\Windows\System32" (
    color 0C
    echo The fork shouldn't be run with admin perms. Don't do that.
    echo.
    pause
    exit /b 1
)

setlocal
title Codename-RVC-Fork-4

if not exist env (
    echo Please run 'run-install.bat' first to set up the environment.
    pause
    exit /b 1
)

env\python.exe app.py --open
echo.
pause
