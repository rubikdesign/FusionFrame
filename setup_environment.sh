#!/bin/bash
# Script for installing all dependencies required for FusionFrame

echo "Installing FusionFrame dependencies..."

# Update pip
pip install --upgrade pip
pip install controlnet_aux diffusers>=0.25 opencv-python
pip install 'mediapipe'
# Install core dependencies
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
pip install diffusers==0.24.0
pip install transformers==4.35.0
pip install accelerate==0.25.0
pip install safetensors==0.4.0
pip install gradio==3.50.2
pip install huggingface_hub==0.17.3

# Install image processing libraries
pip install opencv-python
pip install numpy>=1.24.0
pip install Pillow>=10.0.0
pip install tqdm>=4.66.0

# Install optimizations
pip install xformers==0.0.22.post7 --index-url https://download.pytorch.org/whl/cu118


# Install face recognition support
apt-get update
apt-get install -y cmake
apt-get install -y build-essential
pip install dlib
pip install face_recognition

# Install IP-Adapter directly from source
# Replace with these lines for a more robust installation
pip install git+https://github.com/tencent-ailab/IP-Adapter.git
# Install additional dependencies
pip install timm==0.6.13
pip install open_clip_torch==2.20.0



echo "Installation Complete!"