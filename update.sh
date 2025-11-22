#!/bin/bash

echo "Updating Advanced RVC Inference - KRVC Kernel..."
echo

# Check if git is installed
if ! command -v git &> /dev/null; then
    echo "Git is not installed. Please install Git."
    exit 1
fi

# Pull the latest changes
echo "Pulling latest changes from repository..."
git pull origin main

# Update requirements
echo "Updating dependencies..."
pip3 install -r requirements.txt --upgrade

echo "Update completed successfully!"
read -p "Press Enter to continue..."