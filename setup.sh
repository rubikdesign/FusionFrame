#!/bin/bash
# Improved installation script for FusionFrame

set -e  # Exit on error

echo "=== FusionFrame Installation ==="
echo "This script will install all dependencies required for FusionFrame."

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Ensure pip is up to date
echo "[1/7] Updating pip..."
python -m pip install --upgrade pip

# Install core dependencies from requirements.txt
echo "[2/7] Installing core dependencies..."
pip install -r requirements.txt

# Install PyTorch with CUDA support if CUDA is available
if command_exists nvcc; then
    echo "[3/7] CUDA detected, installing PyTorch with CUDA support..."
    pip install torch==2.0.1 torchvision==0.15.2 --extra-index-url https://download.pytorch.org/whl/cu118
    
    # Install optimizations for CUDA
    echo "[4/7] Installing xformers for GPU optimization..."
    pip install xformers==0.0.22.post7 --extra-index-url https://download.pytorch.org/whl/cu118
else
    echo "[3/7] CUDA not detected, installing PyTorch CPU version..."
    pip install torch==2.0.1 torchvision==0.15.2 --extra-index-url https://download.pytorch.org/whl/cpu
    echo "[4/7] Skipping xformers (GPU only)..."
fi

# Install face-recognition dependencies if needed
echo "[5/7] Setting up face recognition support..."
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Linux - ensure build dependencies are installed
    if command_exists apt-get; then
        echo "Installing dlib dependencies..."
        sudo apt-get update
        sudo apt-get install -y cmake build-essential libboost-all-dev
    elif command_exists yum; then
        echo "Installing dlib dependencies..."
        sudo yum update -y
        sudo yum install -y cmake gcc-c++ boost-devel
    fi
fi

# Try to install dlib and face-recognition
pip install dlib
pip install face_recognition

# Install IP-Adapter directly from source
echo "[6/7] Installing IP-Adapter..."
pip install git+https://github.com/tencent-ailab/IP-Adapter.git

# Download the IP-Adapter models
echo "[7/7] Downloading IP-Adapter models..."
mkdir -p ~/.fusionframe/IP-Adapter/models
mkdir -p ~/.fusionframe/IP-Adapter/sdxl_models

# Check if git-lfs is installed
if ! command_exists git-lfs; then
    echo "Installing git-lfs..."
    if command_exists apt-get; then
        sudo apt-get install -y git-lfs
    elif command_exists brew; then
        brew install git-lfs
    elif command_exists yum; then
        sudo yum install -y git-lfs
    else
        echo "Please install git-lfs manually: https://git-lfs.github.com/"
    fi
    git lfs install
fi

# Clone the IP-Adapter models
git clone https://huggingface.co/h94/IP-Adapter ~/.fusionframe/IP-Adapter/models
git clone https://huggingface.co/h94/IP-Adapter/tree/main/sdxl_models ~/.fusionframe/IP-Adapter/sdxl_models

echo ""
echo "=== Installation Complete! ==="
echo "You can now run FusionFrame with: python -m fusionframe"
echo ""