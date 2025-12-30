@echo off
cd /d %~dp0
call conda activate "%cd%\env"
cmd /k