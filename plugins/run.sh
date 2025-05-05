#!/bin/bash

# Script pentru a rezolva problema cu versiunile bibliotecilor

# Dezinstalează versiunile curente (potențial incompatibile)
pip uninstall -y torch transformers diffusers accelerate

# Instalează versiuni specifice compatibile
pip install torch==2.0.1 
pip install transformers==4.30.2
pip install diffusers==0.21.4
pip install accelerate==0.20.3
pip install gradio==3.36.1
pip install safetensors==0.3.1

# Verifică instalarea 
python -c "import torch; print('PyTorch version:', torch.__version__)"
python -c "import transformers; print('Transformers version:', transformers.__version__)"
python -c "import diffusers; print('Diffusers version:', diffusers.__version__)"

echo "Instalare completă!"
