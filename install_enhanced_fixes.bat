@echo off
REM Advanced RVC Inference - Enhanced Installation Script

echo === Advanced RVC Inference - Enhanced Import Fixes ===
echo Installing improved import handling system...

REM Backup original files
echo Creating backup...
set BACKUP_DIR=backup_%DATE:~-4,4%%DATE:~-10,2%%DATE:~-7,2%_%TIME:~0,2%%TIME:~3,2%%TIME:~6,2%
set BACKUP_DIR=%BACKUP_DIR: =0%
mkdir %BACKUP_DIR%
copy * %BACKUP_DIR%\ >nul 2>&1

REM Create lib directory structure
echo Creating lib directory structure...
if not exist "lib" mkdir lib
cd lib
mkdir algorithm embedders onnx predictors speaker_diarization tools
cd ..

REM Run enhanced import fixes
echo Running enhanced import fixes...
python enhanced_fix_imports.py

REM Install dependencies
echo Installing/updating dependencies...
python -m pip install --upgrade pip
pip install -r requirements_enhanced.txt

echo === Installation completed successfully! ===
echo Run 'python enhanced_fix_imports.py --test' to test imports
pause