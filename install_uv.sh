#!/bin/bash

# Advanced RVC Inference - UV Installer Script
# This script installs the project using UV package manager for faster dependency resolution

set -e

echo "🚀 Advanced RVC Inference - UV Installation"
echo "============================================"

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.10+ first."
    exit 1
fi

# Check Python version
python_version=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
required_version="3.10"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ Python $python_version detected. This project requires Python $required_version or higher."
    exit 1
fi

echo "✅ Python $python_version detected"

# Install UV if not already installed
if ! command -v uv &> /dev/null; then
    echo "📦 Installing UV package manager..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
    
    # Verify UV installation
    if ! command -v uv &> /dev/null; then
        echo "❌ Failed to install UV. Please install manually: https://docs.astral.sh/uv/getting-started/installation/"
        exit 1
    fi
fi

echo "✅ UV package manager is available"

# Create virtual environment and install dependencies
echo "🔧 Creating virtual environment and installing dependencies..."
uv sync

echo "✅ Installation completed successfully!"
echo ""
echo "🎯 To run Advanced RVC Inference:"
echo "   uv run python -m advanced_rvc_inference.main"
echo ""
echo "🔧 To activate the virtual environment:"
echo "   source .venv/bin/activate  # Linux/macOS"
echo "   .venv\\Scripts\\activate     # Windows"
echo ""
echo "📚 For more information, see README.md"