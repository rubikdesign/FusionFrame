#!/bin/bash
# Launcher pentru FusionFrame pe RunPod
# Acest script descarcă install_runpod.py și apoi îl execută

echo "🚀 FusionFrame Launcher pentru RunPod"
echo "-----------------------------------"

# Verifică dacă repo-ul există sau încă trebuie clonat
if [ ! -d "FusionFrame" ]; then
  echo "📥 Clonez repository-ul FusionFrame..."
  git clone https://github.com/YOUR_USERNAME/FusionFrame.git
  # Înlocuiește YOUR_USERNAME cu numele tău real de utilizator GitHub
fi

cd FusionFrame

# Verifică dacă scriptul de instalare există
if [ ! -f "install_runpod.py" ]; then
  echo "📥 Descărcând scriptul de instalare..."
  
  # Creează scriptul de instalare (versiunea prescurtată pentru launcher)
  cat > install_runpod.py << 'ENDOFSCRIPT'
#!/usr/bin/env python3
"""Script de instalare pentru FusionFrame pe RunPod"""

import os
import sys
import subprocess
import argparse
import logging
from pathlib import Path
import time

# Configurare logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("installer")

# Parse arguments
parser = argparse.ArgumentParser(description="FusionFrame Installer")
parser.add_argument("--fix-cuda", action="store_true", help="Fix CUDA compatibility issues")
parser.add_argument("--shared", action="store_true", help="Enable public access")
parser.add_argument("--continue-on-error", action="store_true", help="Continue even if some packages fail")
args = parser.parse_args()

def run_command(cmd, desc=None, check=False, shell=True):
    if desc:
        logger.info(f"🔄 {desc}")
    try:
        result = subprocess.run(cmd, shell=shell, check=check, 
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return result
    except Exception as e:
        logger.error(f"Error: {e}")
        return None

def main():
    start_time = time.time()
    logger.info("Starting FusionFrame installation")
    
    # Fix CUDA environment
    os.environ["NCCL_P2P_DISABLE"] = "1"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
    
    # Install PyTorch first with proper CUDA
    logger.info("Installing PyTorch with CUDA support")
    if args.fix_cuda:
        run_command("pip uninstall -y torch torchvision torchaudio")
    
    run_command("pip install torch==1.13.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117")
    
    # Core packages with fixed versions for stability
    packages = [
        "huggingface_hub==0.16.4",
        "transformers==4.30.2",
        "accelerate==0.20.3",
        "safetensors==0.3.1",
        "gradio==3.33.1",
        "numpy==1.24.3",
        "pillow==9.5.0",
        "opencv-python==4.6.0.66",
        "tqdm==4.65.0",
        "requests==2.31.0",
        "diffusers==0.21.4"
    ]
    
    # Install packages
    for pkg in packages:
        logger.info(f"Installing {pkg}")
        result = run_command(f"pip install {pkg}")
        if result and result.returncode != 0:
            logger.warning(f"Failed to install {pkg}, trying alternative...")
            # Try without version constraint if failed
            pkg_name = pkg.split("==")[0]
            run_command(f"pip install {pkg_name}")
    
    # Optional packages - won't stop on errors
    optionals = [
        "controlnet-aux==0.0.6",
        "timm==0.9.2",
        "xformers==0.0.17", 
        "bitsandbytes"
    ]
    
    for pkg in optionals:
        logger.info(f"Installing optional: {pkg}")
        run_command(f"pip install {pkg}")
    
    # Try IP-Adapter installation
    logger.info("Installing IP-Adapter")
    result = run_command("pip install git+https://github.com/tencent-ailab/IP-Adapter.git")
    if result and result.returncode != 0:
        logger.warning("Failed to install IP-Adapter from git, trying pip...")
        run_command("pip install ip-adapter")
    
    # Copy app.py to a safe location if it exists
    if os.path.exists("app.py"):
        logger.info("Backing up app.py")
        run_command("cp app.py app_original.py")
    
    # Create wrapper script
    logger.info("Creating launcher script")
    with open("start.sh", "w") as f:
        f.write("""#!/bin/bash
export NCCL_P2P_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
python app.py --share
""")
    run_command("chmod +x start.sh")
    
    elapsed = time.time() - start_time
    logger.info(f"Installation completed in {elapsed:.1f} seconds")
    logger.info("To start the application, run: ./start.sh")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("Installation interrupted by user")
    except Exception as e:
        logger.error(f"Installation failed: {e}")
        import traceback
        traceback.print_exc()
ENDOFSCRIPT

  chmod +x install_runpod.py
  echo "✅ Script de instalare creat"
fi

# Rulează scriptul de instalare cu parametrii specificați
echo "🔄 Rulează scriptul de instalare..."
python install_runpod.py --fix-cuda --shared --continue-on-error

# Verifică dacă instalarea s-a finalizat cu succes
if [ -f "start.sh" ]; then
  echo "✅ Instalare finalizată cu succes!"
  echo "🚀 Pentru a porni aplicația, rulează: ./start.sh"
else
  echo "❌ Instalarea pare să fi eșuat. Verifică erorile de mai sus."
  echo "   Încearcă să rulezi manual: python install_runpod.py --fix-cuda --shared --continue-on-error"
fi

# Afișează informații despre sistemul RunPod
echo ""
echo "📊 Informații sistem RunPod:"
nvidia-smi
echo ""
echo "🐍 Versiune Python: $(python --version)"
echo "🔥 Versiune PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "🎮 CUDA disponibil: $(python -c 'import torch; print(torch.cuda.is_available())')"
if python -c 'import torch; exit(0 if torch.cuda.is_available() else 1)' ; then
  echo "💾 Memorie GPU: $(python -c 'import torch; print(f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")')"
fi
echo ""share=True)
"""
        with open(APP_DIR / "app.py", "w") as f:
            f.write(minimal_app)
        logger.info("✅ App minimal creat")

def create_startup_script():
    """Creează script de pornire pentru aplicație"""
    logger.info("🔄 Creez script de pornire...")
    
    # Script bash pentru pornire
    startup_script = f"""#!/bin/bash
# Script de pornire FusionFrame

# Setare variabile de mediu
export HF_HOME="{CACHE_DIR}"
export NCCL_P2P_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Mesaj de start
echo "-----------------------------------"
echo "🚀 Pornesc FusionFrame..."
echo "📂 Director aplicație: {APP_DIR}"
echo "🔗 Interfață web: http://localhost:{args.port}"
echo "-----------------------------------"

# Pornire aplicație
cd {APP_DIR}
python app_wrapper.py --port {args.port} {'--share' if args.shared else ''}
"""
    
    # Salvăm și marcăm ca executabil
    start_script_path = APP_DIR / "start.sh"
    with open(start_script_path, "w") as f:
        f.write(startup_script)
    
    os.chmod(start_script_path, 0o755)
    logger.info(f"✅ Script de pornire creat: {start_script_path}")
    
    # Script Python pentru pornire (alternativă)
    py_script = f"""#!/usr/bin/env python3
import os
import subprocess
import sys

# Setare variabile de mediu
os.environ["HF_HOME"] = "{CACHE_DIR}"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

# Schimbare director
os.chdir("{APP_DIR}")

# Afișare mesaj de start
print("-----------------------------------")
print("🚀 Pornesc FusionFrame...")
print("📂 Director aplicație: {APP_DIR}")
print("🔗 Interfață web: http://localhost:{args.port}")
print("-----------------------------------")

# Pornire aplicație
cmd = ["python", "app_wrapper.py", "--port", "{args.port}"]
{'cmd.append("--share")' if args.shared else '# Fără share'}

subprocess.run(cmd)
"""
    
    # Salvăm și marcăm ca executabil
    py_script_path = APP_DIR / "start.py"
    with open(py_script_path, "w") as f:
        f.write(py_script)
    
    os.chmod(py_script_path, 0o755)
    logger.info(f"✅ Script Python de pornire creat: {py_script_path}")

def create_wrapper():
    """Creează un wrapper pentru app.py care rezolvă problemele de compatibilitate"""
    logger.info("🔄 Creez wrapper de compatibilitate pentru app.py...")
    
    # Salvăm app.py original cu alt nume dacă există
    app_path = APP_DIR / "app.py"
    if app_path.exists():
        shutil.copy(app_path, APP_DIR / "app_original.py")
    
    # Script wrapper care importă corect
    wrapper_script = """#!/usr/bin/env python3
# Wrapper pentru app.py care rezolvă probleme de compatibilitate CUDA/NCCL

import os
import sys
import argparse

# Parsare argumente
parser = argparse.ArgumentParser(description="FusionFrame App Wrapper")
parser.add_argument("--port", type=int, default=7860, help="Port pentru interfața web")
parser.add_argument("--share", action="store_true", help="Permite accesul public")
args = parser.parse_args()

# Setare variabile de mediu esențiale pentru compatibilitate
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

# Asigură-te că directorul curent este în path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Verificare PyTorch înainte de import pentru a evita crash-ul
try:
    # Încercăm să importăm torch
    import torch
    print(f"PyTorch {torch.__version__} încărcat cu succes")
    if torch.cuda.is_available():
        print(f"CUDA disponibil: {torch.cuda.get_device_name(0)}")
    else:
        print("AVERTISMENT: CUDA nu este disponibil, aplicația va rula pe CPU (foarte lent)")
except ImportError as e:
    print(f"Eroare la importul PyTorch: {e}")
    print("Încercăm să reinstalăm PyTorch...")
    import subprocess
    subprocess.run(["pip", "uninstall", "-y", "torch", "torchvision", "torchaudio"])
    subprocess.run(["pip", "install", "torch==1.13.1", "torchvision", "torchaudio", 
                    "--index-url", "https://download.pytorch.org/whl/cu117"])
    # Reîncercăm importul
    try:
        import torch
        print(f"PyTorch reinstalat și încărcat: {torch.__version__}")
    except ImportError as e:
        print(f"Nu s-a putut instala PyTorch: {e}")
        sys.exit(1)

# Verificăm dacă există app_original.py
original_app = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app_original.py")
if os.path.exists(original_app):
    print("Încărcăm aplicația originală...")
    try:
        # Înlocuim argv pentru a transmite parametrii
        sys.argv = [original_app]
        if args.port != 7860:
            sys.argv.extend(["--port", str(args.port)])
        if args.share:
            sys.argv.append("--share")
        
        # Executăm codul din app_original.py
        with open(original_app) as f:
            code = compile(f.read(), original_app, 'exec')
            exec(code, globals())
    except Exception as e:
        print(f"Eroare la pornirea aplicației originale: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
else:
    print("Eroare: Nu găsesc fișierul app_original.py")
    sys.exit(1)
