#!/bin/bash
# Improved installation script for FusionFrame
set -e  # Exit on error

echo "=== FusionFrame Installation ==="
echo "This script will install all dependencies required for FusionFrame."

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Install CMake and build tools first - important for dlib
echo "[0/8] Installing build tools..."
apt-get update
apt-get install -y cmake build-essential libboost-all-dev

# Ensure pip is up to date
echo "[1/8] Updating pip..."
python -m pip install --upgrade pip

# Install core dependencies from requirements.txt
echo "[2/8] Installing core dependencies..."
pip install -r requirements.txt

# Install PyTorch with CUDA support if CUDA is available
if command_exists nvcc; then
    echo "[3/8] CUDA detected, installing PyTorch with CUDA support..."
    pip install torch==2.0.1 torchvision==0.15.2 --extra-index-url https://download.pytorch.org/whl/cu118
    
    # Install optimizations for CUDA
    echo "[4/8] Installing xformers for GPU optimization..."
    pip install xformers==0.0.22.post7 --extra-index-url https://download.pytorch.org/whl/cu118
else
    echo "[3/8] CUDA not detected, installing PyTorch CPU version..."
    pip install torch==2.0.1 torchvision==0.15.2 --extra-index-url https://download.pytorch.org/whl/cpu
    echo "[4/8] Skipping xformers (GPU only)..."
fi

# Install face-recognition dependencies
echo "[5/8] Setting up face recognition support..."
pip install dlib
pip install face_recognition
pip install hf_xet

# Install IP-Adapter directly from source
echo "[6/8] Installing IP-Adapter..."
pip install git+https://github.com/tencent-ailab/IP-Adapter.git

# Download the IP-Adapter models
echo "[7/8] Downloading IP-Adapter models..."
mkdir -p ~/.fusionframe/IP-Adapter/models
mkdir -p ~/.fusionframe/IP-Adapter/sdxl_models

# Check if git-lfs is installed
if ! command_exists git-lfs; then
    echo "Installing git-lfs..."
    if command_exists apt-get; then
        apt-get install -y git-lfs
    elif command_exists brew; then
        brew install git-lfs
    elif command_exists yum; then
        yum install -y git-lfs
    else
        echo "Please install git-lfs manually: https://git-lfs.github.com/"
    fi
    git lfs install
fi

# Clone the full IP-Adapter repository
echo "[8/8] Downloading IP-Adapter models..."

# Method 1: Clone main repo first
git clone https://huggingface.co/h94/IP-Adapter ~/.fusionframe/IP-Adapter/models

# Method 2: Download SDXL models using HuggingFace Hub
pip install huggingface_hub
python -c "from huggingface_hub import snapshot_download; snapshot_download('h94/IP-Adapter', allow_patterns='sdxl_models/*', local_dir='/root/.fusionframe/IP-Adapter', local_dir_use_symlinks=False)"

echo ""
echo "=== Installation Complete! ==="
echo "You can now run FusionFrame with: python -m fusionframe"
echo ""