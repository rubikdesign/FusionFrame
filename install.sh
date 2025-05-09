#!/bin/bash
# Script pentru crearea unui mediu virtual cu dependențe compatibile pentru FusionFrame 2.0
# Optimizat pentru CUDA 12.x și compatibilitate între pachete

set -e  # Oprește script-ul la prima eroare

echo "===== Configurare mediu pentru FusionFrame 2.0 cu CUDA 12.x ====="

VENV_DIR="fusionframe_env"
ROOT_DIR=$(pwd)
LOG_FILE="$ROOT_DIR/install_log.txt"

# Creăm și resetăm log file
echo "===== Log instalare FusionFrame $(date) =====" > "$LOG_FILE"

# Funcție pentru logging
log() {
  echo "$(date +"%Y-%m-%d %H:%M:%S") - $1" | tee -a "$LOG_FILE"
}

# Verificăm GPU și CUDA
log "Verificarea GPU NVIDIA și CUDA disponibil..."
CUDA_AVAILABLE=false
CUDA_VERSION=""

if command -v nvidia-smi &> /dev/null; then
  log "GPU NVIDIA detectat:"
  nvidia-smi | tee -a "$LOG_FILE"
  
  if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $6}' | cut -c2-)
    CUDA_MAJOR=$(echo $CUDA_VERSION | cut -d. -f1)
    CUDA_MINOR=$(echo $CUDA_VERSION | cut -d. -f2)
    log "Versiune CUDA detectată: $CUDA_VERSION (Major: $CUDA_MAJOR, Minor: $CUDA_MINOR)"
    CUDA_AVAILABLE=true
  else
    log "CUDA toolkit detectat prin nvidia-smi, dar nvcc nu este disponibil"
    # Încercăm să detectăm CUDA prin biblioteca
    if [ -d "/usr/local/cuda/lib64" ]; then
      log "Director CUDA găsit: /usr/local/cuda/lib64"
      CUDA_AVAILABLE=true
      # Setăm variabile CUDA
      export CUDA_HOME=/usr/local/cuda
      export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
      export PATH=$CUDA_HOME/bin:$PATH
    fi
  fi
else
  log "AVERTISMENT: Nu a fost detectat un GPU NVIDIA sau nvidia-smi nu este instalat"
  log "FusionFrame va rula pe CPU, dar performanța va fi semnificativ redusă"
fi

# Verificăm dacă directorul mediului virtual există deja
if [ -d "$VENV_DIR" ]; then
  log "Directorul mediului virtual '$VENV_DIR' există deja"
  read -p "Doriți să îl ștergeți și să creați unul nou? (y/n): " choice
  if [ "$choice" == "y" ] || [ "$choice" == "Y" ]; then
    log "Șterg mediul virtual existent..."
    rm -rf "$VENV_DIR"
    log "Creez un nou mediu virtual '$VENV_DIR'..."
    python3 -m venv "$VENV_DIR"
  else
    log "Se utilizează mediul virtual existent. Asigurați-vă că este curat dacă întâmpinați probleme."
  fi
else
  log "Creez mediul virtual '$VENV_DIR'..."
  python3 -m venv "$VENV_DIR"
fi

# Activăm mediul virtual
source "$VENV_DIR/bin/activate"

# Upgrade pip și instalare unelte de build esențiale
log "Actualizez pip și instalez wheel, setuptools..."
pip install --upgrade pip wheel setuptools

# Verifică versiunea Python
PYTHON_VERSION=$(python --version 2>&1 | cut -d " " -f 2)
log "Versiune Python: $PYTHON_VERSION"

# Determinăm versiunea potrivită CUDA pentru PyTorch
PYTORCH_CUDA_VERSION="cpu"  # Implicit CPU
if $CUDA_AVAILABLE; then
  if [ -n "$CUDA_MAJOR" ] && [ -n "$CUDA_MINOR" ]; then
    if [ $CUDA_MAJOR -eq 12 ]; then
      if [ $CUDA_MINOR -ge 1 ]; then
        PYTORCH_CUDA_VERSION="cu121"
        log "Setare PyTorch pentru CUDA 12.1"
      else
        PYTORCH_CUDA_VERSION="cu118"
        log "Setare PyTorch pentru CUDA 11.8 (compatibil cu CUDA $CUDA_VERSION)"
      fi
    elif [ $CUDA_MAJOR -eq 11 ]; then
      PYTORCH_CUDA_VERSION="cu118"
      log "Setare PyTorch pentru CUDA 11.8 (compatibil cu CUDA $CUDA_VERSION)"
    else
      log "AVERTISMENT: Versiune CUDA $CUDA_VERSION poate să nu fie compatibilă, folosim ultima versiune PyTorch cu CUDA 11.8"
      PYTORCH_CUDA_VERSION="cu118"
    fi
  else
    log "Nu s-a putut determina versiunea CUDA major/minor, folosim CUDA 11.8 pentru PyTorch"
    PYTORCH_CUDA_VERSION="cu118"
  fi
fi
# Se instaleaza separat pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# Instalează PyTorch cu suport CUDA sau CPU
if [ "$PYTORCH_CUDA_VERSION" == "cpu" ]; then
  log "Instalez PyTorch CPU..."
  pip install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 --index-url https://download.pytorch.org/whl/cpu
elif [ "$PYTORCH_CUDA_VERSION" == "cu121" ]; then
  log "Instalez PyTorch pentru CUDA 12.1..."
  pip install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 --index-url https://download.pytorch.org/whl/cu121
else
  log "Instalez PyTorch pentru CUDA 11.8..."
  pip install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 --index-url https://download.pytorch.org/whl/cu118
fi

# Verifică dacă PyTorch a fost instalat cu suport CUDA
log "Verificare PyTorch cu CUDA..."
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA disponibil: {torch.cuda.is_available()}, Versiune CUDA: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}');" | tee -a "$LOG_FILE"

# Instalează dependențele din requirements.txt
log "Instalez dependențele de bază din requirements.txt..."
pip install -r requirements.txt

# Instalăm diffusers direct din GitHub - CRUCIAL pentru HiDream
log "Instalez diffusers direct din GitHub (necesar pentru HiDream)..."
pip install git+https://github.com/huggingface/diffusers.git

# Determinăm versiunea potrivită pentru bitsandbytes
BITSANDBYTES_VERSION="0.43.0"  # Actualizat la versiunea din requirements.txt

log "Instalez bitsandbytes $BITSANDBYTES_VERSION..."
# Setăm variabila de mediu pentru a forța compilarea pentru versiunea CUDA detectată
if $CUDA_AVAILABLE; then
  export BNB_CUDA_VERSION=$CUDA_MAJOR$CUDA_MINOR
  log "Setare variabilă BNB_CUDA_VERSION=$BNB_CUDA_VERSION pentru bitsandbytes"
  
  # Forțăm instalarea unei versiuni mai stabile a bitsandbytes
  pip install bitsandbytes==$BITSANDBYTES_VERSION
else
  log "Instalez bitsandbytes versiunea CPU..."
  pip install bitsandbytes==$BITSANDBYTES_VERSION
fi

# Instalăm xformers cu versiunea corectă (0.0.24 conform requirements.txt)
if [ "$PYTORCH_CUDA_VERSION" == "cu121" ]; then
  log "Instalez xformers pentru CUDA 12.1..."
  pip install xformers==0.0.24 --index-url https://download.pytorch.org/whl/cu121
elif [ "$PYTORCH_CUDA_VERSION" == "cu118" ]; then
  log "Instalez xformers pentru CUDA 11.8..."
  pip install xformers==0.0.24 --index-url https://download.pytorch.org/whl/cu118
else
  log "Sar peste instalarea xformers (CPU sau versiune CUDA incompatibilă)"
fi

# Instalez accelerate cu versiunea corectă
log "Instalez accelerate (versiune exactă)..."
pip install accelerate==0.30.1  # Actualizat la versiunea din requirements.txt

# Setează variabile de mediu pentru flash-attention
log "Configurez variabile pentru flash-attention..."
export FLASH_ATTENTION_SKIP_CUDA_BUILD=1
log "Instalez flash-attn cu opțiuni de compatibilitate..."
pip install flash-attn==2.5.6 --no-build-isolation  # Actualizat la versiunea din requirements.txt

# Reinstalăm transformers (ultimă versiune) pentru a asigura compatibilitatea cu diffusers
log "Actualizez transformers la última versiune..."
pip install -U transformers

# Verificarea versiunilor instalate pentru pachetele cheie
log "Verificarea versiunilor instalate pentru pachetele cheie..."
pip list | grep -E "torch|vision|audio|diffusers|transformers|accelerate|xformers|flash-attn|bitsandbytes|ultralytics" | tee -a "$LOG_FILE"
pip install --upgrade accelerate bitsandbytes
# Creează script de execuție
log "Creez script de execuție 'run_fusionframe.sh'..."
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

# Adăugăm directoarele lib64 și lib64/python relativi la LD_LIBRARY_PATH
if [ -d "/usr/local/cuda/lib64" ]; then
    export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
fi

# Adăugăm toate directoarele CUDA cunoscute
for cuda_dir in /usr/local/cuda-1[0-9].[0-9]/lib64 /usr/local/cuda-1[0-9]/lib64 /usr/local/cuda/lib64; do
    if [ -d "$cuda_dir" ]; then
        export LD_LIBRARY_PATH="$cuda_dir:$LD_LIBRARY_PATH"
    fi
done

# Dezactivăm compilarea bitsandbytes
export BNB_CUDA_VERSION=120
export FLASH_ATTENTION_SKIP_CUDA_BUILD=1
export XFORMERS_FORCE_DISABLE_TRITON=1
export CUDA_VISIBLE_DEVICES=0

# Afișează informații despre mediu
echo "Informații sistem:"
echo "Python: $(python --version)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA disponibil: $(python -c 'import torch; print(torch.cuda.is_available())')"
if python -c 'import torch; print(torch.cuda.is_available())' | grep -q "True"; then
    echo "CUDA versiune: $(python -c 'import torch; print(torch.version.cuda)')"
    if command -v nvidia-smi &> /dev/null; then
        echo "GPU disponibil:"
        nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
    fi
fi

# Setăm opțiunea pentru difussers să nu folosească xformers dacă rulează CUDA 11.x
if python -c 'import torch; torch_cuda = torch.version.cuda; print(torch_cuda and torch_cuda.startswith("11"))' | grep -q "True"; then
    export DIFFUSERS_DISABLE_XFORMERS=1
    echo "Dezactivat xformers pentru CUDA 11.x"
fi

echo "Pornesc aplicația FusionFrame..."
# Punctul de intrare al aplicației Gradio

python app.py --share "$@"

EOL

chmod +x run_fusionframe.sh

# Generează un script pentru verificarea mediului
log "Creez script pentru verificarea mediului 'check_environment.py'..."
cat > check_environment.py << 'EOL'
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script pentru verificarea mediului FusionFrame
"""

import os
import sys
import platform
import subprocess
from importlib import util

def check_module(name):
    """Verifică dacă modulul Python este instalat și afișează versiunea."""
    is_installed = util.find_spec(name) is not None
    version = "N/A"
    
    if is_installed:
        try:
            module = __import__(name)
            if hasattr(module, '__version__'):
                version = module.__version__
            elif hasattr(module, 'version'):
                version = module.version.__version__
            else:
                version = "Instalat (versiune necunoscută)"
        except ImportError:
            version = "Problemă la import"
    
    return is_installed, version

def check_cuda_libraries():
    """Verifică dacă există biblioteci CUDA în LD_LIBRARY_PATH."""
    ld_path = os.environ.get('LD_LIBRARY_PATH', '')
    paths = ld_path.split(':')
    cuda_libs = []
    
    for path in paths:
        if not path:
            continue
        if not os.path.exists(path):
            continue
        
        try:
            files = os.listdir(path)
            for file in files:
                if file.startswith('libcudart.so.'):
                    cuda_libs.append(os.path.join(path, file))
        except Exception as e:
            print(f"Eroare la verificarea {path}: {e}")
    
    return cuda_libs

def check_bitsandbytes():
    """Verifică dacă bitsandbytes funcționează corect."""
    try:
        import bitsandbytes as bnb
        
        # Verificare bibliotecă
        loaded_lib = getattr(bnb.cuda_setup, 'lib', None)
        lib_path = getattr(loaded_lib, 'path', 'Unknown') if loaded_lib else 'Not loaded'
        
        # Verificare CUDA
        cuda_setup = getattr(bnb, 'cuda_setup', None)
        if cuda_setup:
            compiled_with_cuda = getattr(bnb.cextension, 'COMPILED_WITH_CUDA', False)
            return True, f"Bibliotecă încărcată: {lib_path}, CUDA: {compiled_with_cuda}"
        return True, f"Bibliotecă încărcată: {lib_path}"
    except Exception as e:
        return False, f"Eroare: {str(e)}"

def check_hidream():
    """Verifică dacă HiDreamImagePipeline este disponibil."""
    try:
        # Încercăm să importăm pipeline-ul specific HiDream
        from diffusers.pipelines.hidream_image.pipeline_hidream_image import HiDreamImagePipeline
        return True, f"HiDreamImagePipeline disponibil"
    except ImportError:
        return False, "HiDreamImagePipeline nu este disponibil. Asigurați-vă că ați instalat diffusers din GitHub."
    except Exception as e:
        return False, f"Eroare: {str(e)}"

def main():
    """Funcția principală de verificare a mediului."""
    print("=" * 50)
    print("VERIFICARE MEDIU FUSIONFRAME")
    print("=" * 50)
    
    # Informații sistem
    print(f"\n[Informații sistem]")
    print(f"Sistem de operare: {platform.system()} {platform.release()}")
    print(f"Python: {platform.python_version()}")
    
    # Verificare variabile de mediu
    print(f"\n[Variabile de mediu]")
    env_vars = ["LD_LIBRARY_PATH", "CUDA_HOME", "CUDA_VISIBLE_DEVICES", 
                "PYTHONPATH", "BNB_CUDA_VERSION", "DIFFUSERS_DISABLE_XFORMERS"]
    for var in env_vars:
        print(f"{var}: {os.environ.get(var, 'Nedefinit')}")
    
    # Verificare CUDA și GPU
    print("\n[CUDA și GPU]")
    cuda_available = False
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        print(f"PyTorch: {torch.__version__}")
        print(f"CUDA disponibil prin PyTorch: {cuda_available}")
        if cuda_available:
            print(f"CUDA versiune PyTorch: {torch.version.cuda}")
            print(f"Versiune cuDNN: {torch.backends.cudnn.version()}")
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"Memorie GPU totală: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    except ImportError:
        print("PyTorch nu este instalat")
    
    # Verificare NVIDIA driver și librării
    print("\n[Librării CUDA]")
    cuda_libs = check_cuda_libraries()
    if cuda_libs:
        print("Librării CUDA găsite:")
        for lib in cuda_libs:
            print(f"  - {lib}")
    else:
        print("Nu s-au găsit librării CUDA în LD_LIBRARY_PATH")
    
    # Verifică bitsandbytes
    print("\n[Stare bitsandbytes]")
    bnb_ok, bnb_status = check_bitsandbytes()
    print(f"Status: {bnb_status}")
    
    # Verifică HiDream
    print("\n[Stare HiDream]")
    hidream_ok, hidream_status = check_hidream()
    print(f"Status: {hidream_status}")
    
    try:
        nvidia_smi = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if nvidia_smi.returncode == 0:
            print("\n[Informații NVIDIA Driver]")
            first_line = nvidia_smi.stdout.split('\n')[0].strip()
            second_line = nvidia_smi.stdout.split('\n')[1].strip()
            print(f"{first_line}\n{second_line}\n...")
            print("(Rulați nvidia-smi separat pentru informații complete)")
        else:
            print("nvidia-smi nu a putut fi executat")
    except FileNotFoundError:
        print("nvidia-smi nu este disponibil")
    
    # Verificare pachete Python
    print("\n[Pachete Python esențiale]")
    packages = [
        "torch", "torchvision", "diffusers", "transformers", 
        "accelerate", "xformers", "bitsandbytes", "flash_attn",
        "ultralytics", "segment_anything", "mediapipe", "gradio"
    ]
    
    for package in packages:
        installed, version = check_module(package)
        status = f"{'✅' if installed else '❌'} {package}: {version}"
        print(status)
    
    print("\n" + "=" * 50)
    if not cuda_available and not cuda_libs:
        print("⚠️  AVERTISMENT: CUDA nu este disponibil sau nu a fost detectat!")
        print("   FusionFrame va rula pe CPU, dar performanța va fi semnificativ redusă.")
    elif not cuda_available and cuda_libs:
        print("⚠️  AVERTISMENT: CUDA este instalat, dar PyTorch nu-l detectează!")
        print("   Verificați compatibilitatea dintre versiunea PyTorch și CUDA.")
    
    if not bnb_ok:
        print("⚠️  AVERTISMENT: bitsandbytes nu funcționează corect!")
        print("   Unele funcționalități CUDA ar putea fi limitate.")
        
    if not hidream_ok:
        print("⚠️  AVERTISMENT: HiDreamImagePipeline nu este disponibil!")
        print("   HiDream nu va funcționa corect. Reinstalați diffusers din GitHub:")
        print("   pip install git+https://github.com/huggingface/diffusers.git")
    
    print("=" * 50)

if __name__ == "__main__":
    main()
EOL

chmod +x check_environment.py

# Configurăm variabile de mediu pentru primul test
log "Configurăm variabile de mediu pentru test..."
export CUDA_VISIBLE_DEVICES=0
export BNB_CUDA_VERSION=120
export FLASH_ATTENTION_SKIP_CUDA_BUILD=1
export XFORMERS_FORCE_DISABLE_TRITON=1
export TRANSFORMERS_OFFLINE=0  
# Scriem un mic test pentru bitsandbytes
log "Creez test pentru bitsandbytes..."
cat > test_bnb.py << 'EOL'
#!/usr/bin/env python

"""Test pentru bitsandbytes"""

import os
import sys

print("Test bitsandbytes:")
print(f"Python: {sys.version}")
print(f"BNB_CUDA_VERSION: {os.environ.get('BNB_CUDA_VERSION', 'Nedefinit')}")

try:
    import torch
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
except ImportError:
    print("PyTorch nu este instalat")

try:
    import bitsandbytes as bnb
    print(f"bitsandbytes versiune: {bnb.__version__}")
    
    # Verifică detalii
    cuda_setup = getattr(bnb, 'cuda_setup', None)
    if cuda_setup:
        print("BNB CUDA setup disponibil")
        try:
            from bitsandbytes.cextension import COMPILED_WITH_CUDA
            print(f"Compilat cu CUDA: {COMPILED_WITH_CUDA}")
        except ImportError:
            print("Nu s-a putut importa COMPILED_WITH_CUDA")
        except Exception as e:
            print(f"Eroare la import COMPILED_WITH_CUDA: {str(e)}")
    
    # Încercăm să creăm un layer 8-bit
    try:
        import torch.nn as nn
        # Creăm un layer liniar 8-bit
        test_input = torch.randn(1, 10)
        layer_8bit = bnb.nn.Linear8bitLt(10, 10)
        output = layer_8bit(test_input)
        print("✅ Linear8bitLt funcționează!")
    except Exception as e:
        print(f"❌ Linear8bitLt eșuat: {str(e)}")
        
except ImportError:
    print("bitsandbytes nu este instalat")
except Exception as e:
    print(f"Eroare la inițializarea bitsandbytes: {str(e)}")

print("Test complet.")
EOL