#!/bin/bash
# Script pentru corectarea incompatibilității între transformers și huggingface_hub

echo "Fixarea compatibilității între transformers și huggingface_hub..."

# Activăm mediul virtual
source ./fusionframe_env/bin/activate

# Verificam versiunile actuale
echo "Versiuni curente:"
pip list | grep -E "transformers|huggingface-hub|diffusers"

# Dezinstalăm pentru o reconfigurare completă
echo "Dezinstalăm pachetele incompatibile..."
pip uninstall -y transformers diffusers huggingface-hub

# Instalăm versiuni compatibile în ordinea corectă
echo "Instalăm versiuni compatibile în ordinea corectă..."
# Începem cu o versiune specifică de huggingface_hub care funcționează cu ambele pachete
pip install huggingface_hub==0.16.4
# Apoi instalăm o versiune mai veche de transformers compatibilă cu această versiune
pip install transformers==4.28.1
# Și în final diffusers
pip install diffusers==0.18.2

# Verificăm dacă acum importul funcționează
echo "Verificăm compatibilitatea..."
python -c "from transformers import AutoModelForImageClassification; print('Transformers importat cu succes')"
python -c "from diffusers import __version__; print(f'Diffusers importat cu succes, versiune: {__version__}')"

echo "Adăugăm variabile suplimentare de mediu în run_fusionframe.sh pentru a evita probleme..."
# Adăugăm variabile de mediu la scriptul de rulare
if grep -q "export TRANSFORMERS_OFFLINE=1" run_fusionframe.sh; then
    echo "Variabilele de mediu sunt deja setate."
else
    # Adăugăm variabilele după linia BNB_CUDA_VERSION
    sed -i '/export BNB_CUDA_VERSION=120/a export TRANSFORMERS_OFFLINE=1\nexport HF_HUB_OFFLINE=1\nexport DIFFUSERS_OFFLINE=1' run_fusionframe.sh
    echo "Variabile de mediu adăugate."
fi

echo "Reparare finalizată. Rulați ./run_fusionframe.sh pentru a porni aplicația."

# Dezactivăm mediul virtual
deactivate
