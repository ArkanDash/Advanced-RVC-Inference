@echo off
setlocal
title Vietnamese RVC By Anh [ZLUDA]

set HIP_VISIBLE_DEVICES="0"
set ZLUDA_COMGR_LOG_LEVEL=1
SET DISABLE_ADDMM_CUDA_LT=1

zluda\zluda.exe -- env\\Scripts\\python.exe main\\app\\app.py --open --allow_all_disk
echo.
pause