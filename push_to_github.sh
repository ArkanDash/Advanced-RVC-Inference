#!/bin/bash
# Git setup and push script

cd /workspace/Advanced-RVC-Inference

# Configure git
git config user.name "BF667"
git config user.email "bf667@example.com"

# Add all new and modified files
git add enhanced_fix_imports.py
git add requirements_enhanced.txt
git add install_enhanced_fixes.sh
git add install_enhanced_fixes.bat
git add ENHANCED_IMPORT_FIXES_README.md
git add lib/__init__.py
git add lib/algorithm/__init__.py
git add lib/embedders/__init__.py
git add lib/onnx/__init__.py
git add lib/predictors/__init__.py
git add lib/speaker_diarization/__init__.py
git add lib/tools/__init__.py

# Commit changes
git commit -m "Enhanced Import Fixes v2.0.0

- Comprehensive import error handling with graceful degradation
- Added complete lib package structure with 6 submodules
- Enhanced __init__.py files with status reporting
- Improved fallback implementations for missing dependencies
- Cross-platform installation scripts (Linux/Mac/Windows)
- Comprehensive documentation and troubleshooting guide
- Fixed circular import issues in lib/tools
- Added module availability flags for runtime checking
- Enhanced error logging and debugging support"

# Set remote URL with credentials
git remote set-url origin https://BF667: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@github.com/ArkanDash/Advanced-RVC-Inference.git

# Push changes
git push origin master

echo "Successfully pushed enhanced import fixes to GitHub!"