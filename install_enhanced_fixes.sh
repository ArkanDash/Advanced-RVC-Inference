#!/bin/bash
# Advanced RVC Inference - Enhanced Installation Script

echo "=== Advanced RVC Inference - Enhanced Import Fixes ==="
echo "Installing improved import handling system..."

# Backup original files
echo "Creating backup..."
mkdir -p backup_$(date +%Y%m%d_%H%M%S)
cp -r * backup_$(date +%Y%m%d_%H%M%S)/ 2>/dev/null || true

# Create lib directory structure
echo "Creating lib directory structure..."
mkdir -p lib/{algorithm,embedders,onnx,predictors,speaker_diarization,tools}

# Run enhanced import fixes
echo "Running enhanced import fixes..."
python enhanced_fix_imports.py

# Install dependencies
echo "Installing/updating dependencies..."
pip install --upgrade pip
pip install -r requirements_enhanced.txt

echo "=== Installation completed successfully! ==="
echo "Run 'python enhanced_fix_imports.py --test' to test imports"