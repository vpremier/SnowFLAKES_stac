#!/bin/bash

# Exit if any command fails
set -e


# ------- Query the data ------------------

# Activate conda environment
echo "Activating conda environment 'download'..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate download

# Path to your Python script and config
SCRIPT_PATH="./data_download/main.py"
CONFIG_PATH="./data_download/config.json"

echo "Running the download script..."
python "$SCRIPT_PATH" "$CONFIG_PATH"

echo "Done DK."

