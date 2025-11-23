#!/bin/bash

echo "Starting Advanced RVC Inference - KRVC Kernel..."
echo

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Python3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

# Check if requirements are installed
if ! python3 -c "import gradio" &> /dev/null; then
    echo "Installing requirements..."
    pip3 install -r requirements.txt
fi

# Start the application
echo "Starting Advanced RVC Inference..."
python3 -m advanced_rvc_inference.main "$@"

# Keep terminal open if running directly
read -p "Press Enter to continue..."