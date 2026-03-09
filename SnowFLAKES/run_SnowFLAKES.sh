#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Activate conda environment
echo "Activating conda environment 'snowmap'..."
source $(conda info --base)/home/vpremier/anaconda3/etc/profile.d/conda.sh
conda activate snowmap

# Define paths
SCRIPT_PATH="./main_SnowFLAKES.py"
CONFIG_PATH="./input_json/fram3s.json"  # <-- Change this to your actual JSON path

BASE="/mnt/CEPH_PROJECTS/FRAM3S/Sentinel-2"

# List of specific subfolders to process
FOLDERS=("12" "13" "20" "21" "28" "29")

echo "Starting SnowFLAKES selective processing..."

for f in "${FOLDERS[@]}"; do
    folder="$BASE/$f"

    echo "Processing folder: $folder"

    # Update JSON working_folder field
    jq --arg wf "$folder" '.working_folder = $wf' "$CONFIG_PATH" > tmp.json && mv tmp.json "$CONFIG_PATH"

    # Run the script
    python "$SCRIPT_PATH" "$CONFIG_PATH"
done


# Run the script
#echo "Running SnowFLAKES classification..."
#python "$SCRIPT_PATH" "$CONFIG_PATH"

echo "SnowFLAKES classification completed."

