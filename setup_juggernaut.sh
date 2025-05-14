#!/bin/bash
# Script pentru configurarea modelului Juggernaut cu link direct

# Asigură-te că există directorul pentru modelul descărcat
mkdir -p /workspace/FusionFrame/models/juggernaut

# Descarcă modelul safetensors direct dacă nu există
if [ ! -f "/workspace/FusionFrame/models/juggernaut/Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors" ]; then
    echo "Descărcare model Juggernaut-XL_v9..."
    wget -O /workspace/FusionFrame/models/juggernaut/Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors "https://huggingface.co/RunDiffusion/Juggernaut-XL-v9/resolve/main/Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors?download=true"
    echo "Descărcare completă."
else
    echo "Modelul Juggernaut-XL_v9 există deja."
fi

echo "Acum poți reporni aplicația: ./run_fusionframe.sh"
