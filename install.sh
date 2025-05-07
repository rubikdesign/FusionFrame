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
        echo "Se utilizează mediul virtual existent."
    fi
else
    echo "Creez mediul virtual '$VENV_DIR'..."
    python3 -m venv "$VENV_DIR"
fi

# Activăm mediul virtual
source "$VENV_DIR/bin/activate"

# Upgrade pip
echo "Actualizez pip..."
pip install --upgrade pip

echo "Instalez dependențele de bază din requirements.txt..."
pip install -r requirements.txt

# Instalăm PyTorch cu suport CUDA 11.8 (compatibil cu Torch 2.7.0+)
# Aceasta va instala PyTorch 2.7.0+cu118 sau cea mai recentă versiune stabilă pentru cu118
echo "Instalez PyTorch, torchvision, torchaudio pentru CUDA 11.8..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Reinstalăm/Actualizăm bibliotecile cheie pentru a asigura compatibilitatea cu noul PyTorch
echo "Actualizez/Reinstalez bibliotecile cheie..."

# flash-attn (fără versiune specifică, pip va alege una compatibilă cu PyTorch 2.7+)
echo "Instalez flash-attn..."
pip uninstall flash-attn -y
pip install flash-attn --no-build-isolation

# xformers (ultima versiune compatibilă)
echo "Instalez xformers..."
pip uninstall xformers -y
pip install -U xformers # -U pentru upgrade dacă există deja o versiune

# diffusers (ultima versiune de pe GitHub pentru compatibilitate maximă cu HiDream)
echo "Instalez diffusers de pe GitHub..."
pip uninstall diffusers -y
pip install git+https://github.com/huggingface/diffusers.git

# transformers (ultima versiune)
echo "Instalez/Actualizez transformers..."
pip install -U transformers

# accelerate (ultima versiune, importantă pentru diffusers/transformers)
echo "Instalez/Actualizez accelerate..."
pip install -U accelerate

# ultralytics (ultima versiune)
echo "Instalez/Actualizez ultralytics..."
pip install -U ultralytics

# bitsandbytes (actualizăm pentru compatibilitate cu noul PyTorch)
echo "Instalez/Actualizez bitsandbytes..."
pip install -U bitsandbytes

echo "Verificarea dependențelor instalate (primele 10)..."
pip list | head -n 10

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
# pentru a permite importurile corecte ale modulelor custom (config, core, etc.)
export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"

echo "Pornesc aplicația FusionFrame..."
# Presupunem că scriptul principal al aplicației este app.py sau interface/ui.py
# Modificați mai jos dacă este altul.
# python app.py "$@"
python interface/ui.py "$@" # Sau oricare este punctul de intrare al aplicației Gradio

EOL

chmod +x run_fusionframe.sh

echo "===== Instalare și configurare completă! ====="
echo "Pentru a rula aplicația, folosește comanda din directorul proiectului: ./run_fusionframe.sh"
echo "NOTĂ: La prima rulare a aplicației, modelele AI vor fi descărcate automat (poate dura câteva minute în funcție de conexiunea la internet)."
echo "Asigurați-vă că aveți suficient spațiu pe disc."

