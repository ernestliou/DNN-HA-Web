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

# Execute main Python program
echo "Running DNN-HA test script..."
python sys/test_DNN-HA_wavfile.py

# Deactivate virtual environment after execution
deactivate
echo "Execution completed."
