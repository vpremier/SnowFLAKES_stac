#!/bin/bash

# ==============================================================
# Sentinel-2/Landsat Downloader Environment Setup Script
# ==============================================================

# Environment name and Python version
ENV_NAME="download"
PYTHON_VERSION="3.10"

echo "--------------------------------------------------------------"
echo "🔧 Setting up conda environment: $ENV_NAME (Python $PYTHON_VERSION)"
echo "--------------------------------------------------------------"

# 1️⃣ Remove old environment if it exists
if conda env list | grep -q "$ENV_NAME"; then
    echo "Removing existing environment: $ENV_NAME"
    conda remove -y -n $ENV_NAME --all
fi

# 2️⃣ Configure conda channels
echo "Configuring conda channels..."
conda config --add channels conda-forge
conda config --add channels defaults
conda config --set channel_priority strict

# 3️⃣ Create a new environment
echo "Creating environment $ENV_NAME..."
conda create -y -n $ENV_NAME python=$PYTHON_VERSION

# 4️⃣ Activate the environment
echo "Activating environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $ENV_NAME

# 5️⃣ Install required packages
echo "Installing required packages ..."

pip install \
    geopandas \
    pandas \
    shapely \
    matplotlib \
    tqdm \
    requests \
    python-dotenv

# 6️⃣ (Optional) Install Spyder IDE
# Uncomment this line if you want Spyder in the environment
pip install spyder

echo "--------------------------------------------------------------"
echo "✅ Environment '$ENV_NAME' is ready with Python $PYTHON_VERSION."
echo "   Installed packages: geopandas, pandas, shapely, matplotlib, tqdm, requests, python-dotenv"
echo "--------------------------------------------------------------"

