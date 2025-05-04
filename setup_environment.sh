#!/bin/bash
# Script pentru configurarea mediului FusionFrame cu suport pentru modele de ultimă generație
apt-get update
apt-get install -y cmake
apt-get install -y build-essential
# Actualizăm pip
pip install --upgrade pip

# Instalăm PyTorch cu suport CUDA
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118

# Instalăm dependențele pentru modele avansate
pip install diffusers==0.24.0
pip install transformers==4.35.0
pip install accelerate==0.25.0
pip install safetensors==0.4.0
pip install xformers==0.0.22.post7
pip install triton==2.1.0
pip install opencv-python
pip install face_recognition

# Instalăm Gradio și alte utilități
pip install gradio==3.50.2
pip install numpy>=1.24.0
pip install Pillow>=10.0.0
pip install tqdm>=4.66.0
pip install huggingface_hub>=0.16.0

echo "Mediul pentru FusionFrame a fost configurat cu succes!"
