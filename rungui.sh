#!/bin/bash

# Switch to the directory where the script is located
cd "$(dirname "$0")"

# Check if the virtual environment exists
if [ ! -d ".venv" ]; then
    echo "Virtual environment .venv not found, preparing to create it automatically..."
    python3 -m venv .venv
    echo "Activating virtual environment and installing required packages..."
    . .venv/bin/activate
    pip install -r requirements.txt
else
    # Activate virtual environment
    echo "Activating virtual environment..."
    . .venv/bin/activate
fi

# Set PYTHONPATH so the GUI can directly load shared functions from the sys directory
export PYTHONPATH="$PYTHONPATH:$(pwd)/sys"

# Run GUI application
echo "Starting DNN-HA GUI application..."
python gui/web_app.py

# Deactivate virtual environment after execution
deactivate
echo "Execution completed."
