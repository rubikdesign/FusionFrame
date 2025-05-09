#!/bin/bash

# Dezinstalează versiunea actuală de Gradio
pip uninstall -y gradio gradio-client

# Instalează o versiune stabilă anterioară de Gradio
pip install gradio==3.50.2

# Verifică versiunea instalată
pip show gradio

# Pornește aplicația
cd /workspace/FusionFrame
source fusionframe_env/bin/activate
./run_fusionframe.sh --low-vram