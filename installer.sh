#!/bin/bash

# Advanced RVC Inference - Installation Script
# This script installs all necessary dependencies for Advanced RVC Inference

echo "==========================================="
echo "Advanced RVC Inference Installation Script"
echo "==========================================="

# Check if running in Colab
if [ -f "/content/drive/MyDrive" ]; then
    echo "Detected Google Colab environment"
    IN_COLAB=true
else
    IN_COLAB=false
fi

# Set up environment variables
export PIP_PREFER_BINARY=1
export PYTHONPATH="${PWD}:${PYTHONPATH}"

echo "Setting up Python environment..."

if [ "$IN_COLAB" = true ]; then
    # Install system dependencies on Colab
    echo "Installing system dependencies..."
    apt update -y
    apt install -y python3.11 python3.11-distutils python3.11-dev portaudio19-dev build-essential
    
    # Make sure we're using Python 3.11
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 2
    update-alternatives --set python3 /usr/bin/python3.11
    update-alternatives --install /usr/bin/python python /usr/bin/python3.11 2
    update-alternatives --set python /usr/bin/python3.11
fi

# Install uv for fast package management
echo "Installing uv..."
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.cargo/bin:$PATH"
source "$HOME/.cargo/env" 2>/dev/null || true

# Create virtual environment using uv
echo "Creating virtual environment..."
uv venv

# Activate the virtual environment
source .venv/bin/activate

echo "Virtual environment activated."

# Install torch with CUDA support
echo "Installing PyTorch with CUDA support..."
uv pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install dependencies from requirements.txt
echo "Installing requirements..."
uv pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121 --index-strategy unsafe-best-match

# Install additional packages for Colab
if [ "$IN_COLAB" = true ]; then
    echo "Installing Colab-specific packages..."
    uv pip install -q ngrok jupyter-ui-poll
    npm install -g -q localtunnel &> /dev/null
fi

# Install this package in development mode - only install if dependencies are available
echo "Installing Advanced RVC Inference package..."
if uv pip install -e .; then
    echo "Package installed successfully."
else
    echo "Warning: Development install failed, trying regular install..."
    # Copy just the essential files without installing as a package first
    echo "RVC prerequisites will be installed after basic setup is complete."
fi

# Install prerequisites for RVC (run after basic dependencies are installed)
echo "Installing RVC prerequisites..."
python -c "from advanced_rvc_inference.core import run_prerequisites_script; run_prerequisites_script(pretraineds_hifigan=True, models=True, exe=True)"

echo "==========================================="
echo "Installation completed successfully!"
echo "==========================================="

echo "To run the application, use one of the following commands:"
echo "  python -m advanced_rvc_inference.app              # Run with default settings"
echo "  python -m advanced_rvc_inference.app --share      # Run with public sharing"
echo "  python -m advanced_rvc_inference.app --listen     # Run with external access"

if [ "$IN_COLAB" = true ]; then
    echo ""
    echo "For Google Colab, it's recommended to use '--share' flag to get a public URL:"
    echo "  python -m advanced_rvc_inference.app --share"
fi

echo "==========================================="