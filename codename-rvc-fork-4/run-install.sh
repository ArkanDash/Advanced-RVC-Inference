#!/bin/bash
set -e

printf "\033]0;Codename-RVC-Fork Installer\007"
clear

INSTALL_DIR="$(pwd)"
MINICONDA_DIR="$HOME/Miniconda3"
ENV_DIR="$INSTALL_DIR/env"
MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-py310_24.7.1-0-Linux-x86_64.sh"
CONDA_EXE="$MINICONDA_DIR/bin/conda"

SECONDS=0

log_message() {
    local msg="$1"
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $msg"
}

cleanup() {
    log_message "Cleaning up unnecessary files..."
    rm -f Makefile Dockerfile docker-compose.yaml *.bat
    log_message "Cleanup complete."
}

install_miniconda() {
    if [ -x "$CONDA_EXE" ]; then
        log_message "Miniconda already installed. Skipping installation."
        return
    fi

    log_message "Miniconda not found. Starting download and installation..."
    curl -fsSLo miniconda.sh "$MINICONDA_URL"

    if [ ! -f miniconda.sh ]; then
        log_message "Download failed. Please check your internet connection and try again."
        exit 1
    fi

    bash miniconda.sh -b -p "$MINICONDA_DIR"
    rm -f miniconda.sh

    log_message "Miniconda installation complete."
}

create_conda_env() {
    log_message "Creating Conda environment..."
    "$MINICONDA_DIR/bin/conda" create --no-shortcuts -y -k --prefix "$ENV_DIR" python=3.10.18

    if [ -x "$ENV_DIR/bin/python" ]; then
        log_message "Installing uv package installer..."
        "$ENV_DIR/bin/python" -m pip install uv
        log_message "uv installation complete."
    fi
}

install_dependencies() {
    log_message "Installing dependencies..."

    # shellcheck disable=SC1091
    source "$MINICONDA_DIR/etc/profile.d/conda.sh"
    conda activate "$ENV_DIR"

    # Ensure uv operates on the env's Python, not any global environment
    export UV_PYTHON="$ENV_DIR/bin/python"

    "$ENV_DIR/bin/python" -m pip install --upgrade pip setuptools
    "$ENV_DIR/bin/python" -m pip install uv

    "$ENV_DIR/bin/uv" pip install --upgrade setuptools
    "$ENV_DIR/bin/uv" pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --upgrade --index-url https://download.pytorch.org/whl/cu128
    "$ENV_DIR/bin/uv" pip install -r "$INSTALL_DIR/requirements.txt"
    "$ENV_DIR/bin/uv" pip install pesq ring-attention-pytorch

    unset UV_PYTHON

    conda deactivate

    log_message "Dependencies installation complete."
}

cleanup
install_miniconda
create_conda_env
install_dependencies

elapsed=$SECONDS
hours=$((elapsed / 3600))
minutes=$(((elapsed % 3600) / 60))
seconds=$((elapsed % 60))

echo "Installation time: ${hours} hours, ${minutes} minutes, ${seconds} seconds."
echo
echo "Codename-RVC-Fork has been installed successfully!"
echo "To start Codename-RVC-Fork, please run 'run-fork.sh'."
