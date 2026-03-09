#!/bin/bash

# Exit if any command fails
set -e

# Activate conda environment
echo "Activating conda environment 'download'..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate download

# Path to your Python script and config
SCRIPT_PATH="./main.py"
CONFIG_PATH="./config.json"

echo "Running the download script..."
python "$SCRIPT_PATH" "$CONFIG_PATH"

echo "Done DK."

