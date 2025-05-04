#!/bin/bash
# Script pentru instalarea tuturor dependențelor necesare pentru FusionFrame

echo "Install FusionFrame..."

# Actualizare pip
pip install --upgrade pip

# Instalare dependențe de bază
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
pip install diffusers==0.24.0
pip install transformers==4.35.0
pip install accelerate==0.25.0
pip install safetensors==0.4.0
pip install gradio==3.50.2
pip install huggingface_hub==0.17.3

# Instalare biblioteci pentru procesarea imaginilor
pip install opencv-python
pip install numpy>=1.24.0
pip install Pillow>=10.0.0
pip install tqdm>=4.66.0

# Instalare optimizări
pip install xformers==0.0.22.post7

# Instalare suport pentru face recognition
apt-get update
apt-get install -y cmake
apt-get install -y build-essential
pip install dlib
pip install face_recognition

# Instalare IP-Adapter direct din sursă
pip install git+https://github.com/tencent-ailab/IP-Adapter.git

echo "Install Complete"