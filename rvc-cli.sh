#!/bin/bash
# Advanced RVC Inference CLI Wrapper
# Usage: ./rvc-cli <command> [options]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed or not in PATH"
    exit 1
fi

# Run the CLI module
python3 -m advanced_rvc_inference.api.cli "$@"
