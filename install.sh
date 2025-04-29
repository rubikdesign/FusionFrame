#!/bin/bash
# Script de instalare și configurare pentru Platforma de Generare Imagini

# Creăm un mediu virtual
echo "Creăm mediu virtual Python..."
python -m venv clothing_generator_env

# Activăm mediul virtual (pentru Linux/Mac)
source clothing_generator_env/bin/activate

# Pentru Windows, utilizați:
# clothing_generator_env\Scripts\activate

# Instalăm PyTorch cu suport CUDA
echo "Instalăm PyTorch cu suport CUDA..."
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118

# Instalăm dependențele de bază
echo "Instalăm dependențele necesare..."
pip install diffusers==0.22.1 transformers accelerate safetensors gradio
pip install numpy pillow opencv-python tqdm requests typing-extensions packaging

# Instalăm dependențele pentru ControlNet
echo "Instalăm dependențele pentru ControlNet..."
pip install controlnet-aux timm mediapipe

# Instalăm IP-Adapter și alte dependențe avansate
echo "Instalăm IP-Adapter..."
pip install git+https://github.com/tencent-ailab/IP-Adapter.git

# Instalăm librării opționale pentru optimizarea performanței
echo "Instalăm librării pentru optimizarea performanței..."
pip install xformers bitsandbytes

# Configurăm structura directorului
echo "Creăm structura director pentru aplicație..."
mkdir -p ~/.cache/clothing_app/loras
mkdir -p ~/.cache/clothing_app/uploads
mkdir -p ~/.cache/clothing_app/outputs
mkdir -p ~/.cache/clothing_app/controlnet
mkdir -p ~/.cache/clothing_app/ipadapter
mkdir -p ~/.cache/clothing_app/logs

echo "Instalarea s-a finalizat cu succes!"
echo "Pentru a rula aplicația, folosiți comanda: python app.py"
