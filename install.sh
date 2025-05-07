#!/bin/bash
# Script pentru crearea unui mediu virtual cu dependențe compatibile pentru FusionFrame 2.0

echo "===== Configurare mediu pentru FusionFrame 2.0 ====="

VENV_DIR="fusionframe_env"

# Verificăm dacă directorul mediului virtual există deja
if [ -d "$VENV_DIR" ]; then
    echo "Directorul mediului virtual '$VENV_DIR' există deja."
    read -p "Doriți să îl ștergeți și să creați unul nou? (y/n): " choice
    if [ "$choice" == "y" ] || [ "$choice" == "Y" ]; then
        echo "Șterg mediul virtual existent..."
        rm -rf "$VENV_DIR"
        echo "Creez un nou mediu virtual '$VENV_DIR'..."
        python3 -m venv "$VENV_DIR"
    else
        echo "Se utilizează mediul virtual existent. Asigurați-vă că este curat dacă întâmpinați probleme."
    fi
else
    echo "Creez mediul virtual '$VENV_DIR'..."
    python3 -m venv "$VENV_DIR"
fi

# Activăm mediul virtual
source "$VENV_DIR/bin/activate"

# Upgrade pip și instalare unelte de build esențiale
echo "Actualizez pip și instalez wheel, setuptools..."
pip install --upgrade pip wheel setuptools

echo "Instalez dependențele de bază din requirements.txt..."
# Este important ca requirements.txt să NU fixeze versiunile pentru torch, torchvision, torchaudio,
# flash-attn, xformers, diffusers, transformers, accelerate, ultralytics, bitsandbytes
# deoarece acestea vor fi gestionate mai jos.
pip install -r requirements.txt

# Instalăm PyTorch cu suport CUDA 11.8 (compatibil cu Torch 2.7.0+)
echo "Instalez PyTorch, torchvision, torchaudio pentru CUDA 11.8..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Reinstalăm/Actualizăm bibliotecile cheie pentru a asigura compatibilitatea cu noul PyTorch
echo "Actualizez/Reinstalez bibliotecile cheie..."

# accelerate (o versiune minimă mai recentă sau ultima)
# Este important să fie după requirements.txt și PyTorch pentru a suprascrie corect.
echo "Instalez/Actualizez accelerate..."
pip uninstall accelerate -y # Dezinstalăm întâi pentru o actualizare curată
pip install -U "accelerate>=0.27.0" # Specificăm o versiune minimă rezonabilă

# flash-attn (pip va alege una compatibilă cu PyTorch 2.7+ și CUDA)
pip install ninja
python -m pip install --upgrade pip wheel setuptools
echo "Instalez flash-attn..."
pip uninstall flash-attn -y
MAX_JOBS=4 python -m pip -v install flash-attn --no-build-isolation 
# Dacă --no-build-isolation continuă să dea erori, încercați fără el,
# dar cu wheel instalat ar trebui să fie ok.

# xformers (ultima versiune compatibilă)
echo "Instalez xformers..."
pip uninstall xformers -y
pip install -U xformers 

# diffusers (ultima versiune de pe GitHub pentru compatibilitate maximă cu HiDream)
echo "Instalez diffusers de pe GitHub..."
pip uninstall diffusers -y
pip install git+https://github.com/huggingface/diffusers.git

# transformers (ultima versiune)
echo "Instalez/Actualizez transformers..."
pip uninstall transformers -y # Dezinstalăm întâi pentru o actualizare curată
pip install -U transformers

# ultralytics (ultima versiune)
echo "Instalez/Actualizez ultralytics..."
pip uninstall ultralytics -y
pip install -U ultralytics

# bitsandbytes (actualizăm pentru compatibilitate cu noul PyTorch)
echo "Instalez/Actualizez bitsandbytes..."
pip uninstall bitsandbytes -y
pip install -U bitsandbytes

echo "Verificarea versiunilor instalate pentru pachetele cheie..."
pip show torch torchvision torchaudio flash-attn xformers diffusers transformers accelerate ultralytics bitsandbytes

echo "Creez script de execuție 'run_fusionframe.sh'..."
# Creăm scriptul de execuție
cat > run_fusionframe.sh << 'EOL'
#!/bin/bash
# Script pentru rularea FusionFrame

# Directorul unde se află acest script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
VENV_PATH="$SCRIPT_DIR/fusionframe_env/bin/activate"

if [ ! -f "$VENV_PATH" ]; then
    echo "Mediul virtual nu a fost găsit la $VENV_PATH"
    echo "Rulați mai întâi scriptul install.sh pentru a crea mediul."
    exit 1
fi

source "$VENV_PATH"

# Adăugăm directorul curent (rădăcina proiectului FusionFrame) la PYTHONPATH
export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"

echo "Pornesc aplicația FusionFrame..."
# Punctul de intrare al aplicației Gradio
python interface/ui.py "$@"

EOL

chmod +x run_fusionframe.sh

echo "===== Instalare și configurare completă! ====="
echo "Pentru a rula aplicația, folosește comanda din directorul proiectului: ./run_fusionframe.sh"
echo "NOTĂ: La prima rulare a aplicației, modelele AI vor fi descărcate automat."
echo "Asigurați-vă că aveți suficient spațiu pe disc."

