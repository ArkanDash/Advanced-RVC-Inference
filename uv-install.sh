#!/bin/bash
# uv-install.sh - Script to install and setup the package with uv

# Install uv if not already available
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Add uv to PATH for current session
    export PATH="$HOME/.cargo/bin:$PATH"
fi

# Create a virtual environment using uv
echo "Creating virtual environment..."
uv venv

# Activate the virtual environment
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Sync dependencies (this will create/update uv.lock)
echo "Installing dependencies..."
uv pip install -e .

# Install additional development dependencies
uv pip install pytest black flake8 mypy

echo "Setup complete! You can now run the application with:"
echo "  python -m advanced_rvc_inference.app"