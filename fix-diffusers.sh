#!/bin/bash
# Script pentru corectarea incompatibilității între diffusers și huggingface_hub

echo "Fixarea compatibilității între diffusers și huggingface_hub..."

# Activăm mediul virtual
source ./fusionframe_env/bin/activate

# Fixăm versiunile compatibile
echo "Instalăm versiuni compatibile pentru diffusers și huggingface_hub..."
pip uninstall -y diffusers huggingface_hub
pip install huggingface_hub==0.18.0
pip install diffusers==0.23.0

# Verificăm dacă acum importul funcționează
echo "Verificăm compatibilitatea..."
if python -c "from diffusers import __version__; print(f'Diffusers importat cu succes, versiune: {__version__}')" 2>/dev/null; then
    echo "✅ Diffusers importat cu succes!"
else
    echo "❌ Problema persistă. Încercăm altă abordare..."
    # Abordare alternativă
    pip uninstall -y diffusers huggingface_hub
    pip install huggingface_hub==0.17.0
    pip install diffusers==0.22.0
    
    # Verificăm din nou
    if python -c "from diffusers import __version__; print(f'Diffusers importat cu succes, versiune: {__version__}')" 2>/dev/null; then
        echo "✅ Diffusers importat cu succes în a doua încercare!"
    else
        echo "❌ Nu s-a putut rezolva complet problema. Dar aplicația ar trebui să funcționeze cu modelul nostru personalizat."
    fi
fi

# Setăm variabilele necesare în run_fusionframe.sh
sed -i 's/export DIFFUSERS_DISABLE_XFORMERS=1/export DIFFUSERS_DISABLE_XFORMERS=1\nexport TRANSFORMERS_OFFLINE=1\nexport HF_HUB_OFFLINE=1/g' run_fusionframe.sh 2>/dev/null || echo "Nu s-a putut actualiza variabilele de mediu în run_fusionframe.sh"

echo "Reparare finalizată. Rulați ./run_fusionframe.sh pentru a porni aplicația."

# Dezactivăm mediul virtual
deactivate
