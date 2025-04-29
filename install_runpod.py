#!/usr/bin/env python3
"""
Script de instalare îmbunătățit pentru FusionFrame pe RunPod
- Rezolvă problemele de compatibilitate CUDA
- Gestionează erorile de instalare a pachetelor
- Implementează o strategie de failover pentru dependențe problematice
"""

import os
import sys
import subprocess
import argparse
import shutil
import logging
from pathlib import Path
import time
import json

# Configurare logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f"install_log_{time.strftime('%Y%m%d_%H%M%S')}.log")
    ]
)
logger = logging.getLogger("installer")

# Setare parser argumente
parser = argparse.ArgumentParser(description="FusionFrame - Platformă de generare imagini pentru îmbrăcare virtuală")
parser.add_argument("--app-dir", type=str, default="FusionFrame", 
                    help="Numele directorului aplicației (default: FusionFrame)")
parser.add_argument("--port", type=int, default=7860, 
                    help="Port pentru interfața web Gradio")
parser.add_argument("--cuda-version", type=str, default="118", 
                    choices=["116", "117", "118", "121"],
                    help="Versiunea CUDA pentru PyTorch")
parser.add_argument("--torch-version", type=str, default="2.0.1", 
                    choices=["1.13.1", "2.0.1", "2.1.2", "2.2.0"],
                    help="Versiunea PyTorch (pentru compatibilitate CUDA)")
parser.add_argument("--repo-path", type=str, default=".", 
                    help="Calea către directorul repo-ului (default: director curent)")
parser.add_argument("--no-xformers", action="store_true", 
                    help="Dezactivează instalarea xformers")
parser.add_argument("--download-models", action="store_true", 
                    help="Descarcă modele predefinite (durează mult)")
parser.add_argument("--shared", action="store_true", 
                    help="Permite accesul public la interfață")
parser.add_argument("--fix-cuda", action="store_true", 
                    help="Forțează reinstalarea CUDA pentru a rezolva probleme de compatibilitate")
parser.add_argument("--continue-on-error", action="store_true", 
                    help="Continuă instalarea chiar dacă unele pachete eșuează")

args = parser.parse_args()

# Configurarea directoarelor
WORKSPACE_DIR = Path("/workspace") if os.path.exists("/workspace") else Path.home()
REPO_PATH = Path(args.repo_path).absolute()
APP_DIR = WORKSPACE_DIR / args.app_dir
CACHE_DIR = APP_DIR / "cache"
LOG_DIR = APP_DIR / "logs"
MODEL_DIR = APP_DIR / "models"

# Pachete necesare cu versiuni fixate pentru stabilitate și alternative
# Format: [(pachet, este_esențial, alternative)]
PACKAGES = [
    ("huggingface_hub==0.16.4", True, ["huggingface_hub==0.15.1"]),
    ("transformers==4.30.2", True, ["transformers==4.29.2"]),
    ("accelerate==0.20.3", True, ["accelerate==0.19.0"]),
    ("safetensors==0.3.1", True, ["safetensors==0.3.0"]), 
    ("numpy==1.24.3", True, ["numpy==1.23.5"]),
    ("pillow==9.5.0", True, ["pillow==9.4.0"]),
    ("tqdm==4.65.0", False, ["tqdm"]), 
    ("requests==2.31.0", False, ["requests"]),
    # Pachete non-esențiale pot fi omise
    ("gradio==3.33.1", False, ["gradio==3.32.0", "gradio==3.31.0", "gradio==3.30.0"]),
    ("opencv-python==4.7.0.72", False, ["opencv-python==4.6.0.66", "opencv-python"]),
    ("controlnet-aux==0.0.6", False, ["controlnet-aux==0.0.5", "controlnet-aux"]),
    ("timm==0.9.2", False, ["timm==0.6.12", "timm"]),
    # mediapipe e problematic pe RunPod - încercăm versiuni alternative
    ("mediapipe==0.10.0", False, ["mediapipe==0.9.0.1", "mediapipe==0.8.10", ""]),
    ("bitsandbytes==0.35.4", False, ["bitsandbytes", ""]),
]

# Versiuni pentru evitarea conflictelor NCCL
CUDA_FIXES = {
    "118": {
        "torch_version": "2.0.1",
        "cuda_toolkit": "11.8.0",
        "cuda_extra": "--index-url https://download.pytorch.org/whl/cu118"
    },
    "117": {
        "torch_version": "1.13.1",
        "cuda_toolkit": "11.7.1", 
        "cuda_extra": "--index-url https://download.pytorch.org/whl/cu117"
    },
    "116": {
        "torch_version": "1.13.1",
        "cuda_toolkit": "11.6.2",
        "cuda_extra": "--index-url https://download.pytorch.org/whl/cu116"
    }
}

def run_command(command, desc=None, check=True, shell=False, env=None, verbose=False):
    """Execută o comandă și loghează output-ul"""
    if desc:
        logger.info(f"🔄 {desc}...")
    
    cmd_list = command if shell else command.split()
    try:
        process = subprocess.run(
            cmd_list, 
            check=False,  # Gestionăm noi erorile pentru flexibilitate 
            shell=shell,
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            text=True,
            env=env
        )
        if process.stdout and (verbose or process.returncode != 0):
            for line in process.stdout.strip().split('\n'):
                if line:
                    logger.debug(line)
        
        if process.returncode != 0:
            if process.stderr:
                logger.error(f"Eroare: {process.stderr.strip()}")
            if check:
                raise subprocess.CalledProcessError(process.returncode, command)
        
        return process
    except subprocess.CalledProcessError as e:
        if check:
            logger.error(f"Comandă eșuată: {e}")
            raise
        return None
    except Exception as e:
        logger.error(f"Excepție la rularea comenzii: {e}")
        if check:
            raise
        return None

def setup_directories():
    """Creează directoarele necesare"""
    logger.info(f"🔄 Creez directoare în {APP_DIR}...")
    for dir_path in [APP_DIR, CACHE_DIR, LOG_DIR, MODEL_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"✅ Creat director: {dir_path}")

def check_gpu():
    """Verifică disponibilitatea GPU și CUDA"""
    logger.info("🔄 Verific GPU...")
    try:
        nvidia_smi = run_command("nvidia-smi", desc="Rulare nvidia-smi", check=False)
        if nvidia_smi and nvidia_smi.returncode == 0:
            logger.info("✅ GPU detectat")
            
            # Script pentru verificarea CUDA cu PyTorch
            check_script = """
import torch
if torch.cuda.is_available():
    print(f"CUDA disponibil: {torch.cuda.get_device_name(0)}")
    print(f"Versiune CUDA: {torch.version.cuda}")
    try:
        print(f"Memorie GPU: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    except:
        print("Nu pot obține informații despre memorie")
else:
    print("CUDA nu este disponibil")
    exit(1)
"""
            with open(APP_DIR / "check_cuda.py", "w") as f:
                f.write(check_script)
            
            return True
        else:
            logger.warning("⚠️ GPU nu a fost detectat. Aplicația va rula în mod CPU (foarte lent).")
            return False
    except Exception as e:
        logger.warning(f"⚠️ Nu pot verifica GPU: {e}")
        return False

def fix_cuda_environment():
    """Corectează variabilele de mediu CUDA"""
    logger.info("🔄 Configurez variabile de mediu CUDA...")
    
    # Setăm variabile de mediu pentru a evita conflicte NCCL
    env_vars = {
        "NCCL_P2P_DISABLE": "1",
        "NCCL_DEBUG": "INFO",
        "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:512"
    }
    
    # Actualizăm ~/.bashrc pentru a păstra aceste setări
    bashrc_path = os.path.expanduser("~/.bashrc")
    try:
        with open(bashrc_path, "a") as f:
            f.write("\n# FusionFrame CUDA environment settings\n")
            for var, value in env_vars.items():
                f.write(f"export {var}={value}\n")
                # Setăm și pentru sesiunea curentă
                os.environ[var] = value
    except Exception as e:
        logger.warning(f"Nu pot actualiza ~/.bashrc: {e}")
        # Continuăm, deoarece acest lucru nu este critic
    
    logger.info("✅ Variabile de mediu CUDA configurate")
    return env_vars

def install_pytorch(cuda_version, torch_version=None, fix_cuda=False):
    """Instalează PyTorch cu suport CUDA corespunzător"""
    
    if fix_cuda or args.fix_cuda:
        # Dezinstalăm PyTorch existent pentru a evita conflicte
        logger.info("🔄 Dezinstalez PyTorch existent...")
        run_command("pip uninstall -y torch torchvision torchaudio", 
                   desc="Dezinstalare PyTorch", shell=True, check=False)
    
    # Folosim configurația CUDA din dicționar
    cuda_config = CUDA_FIXES.get(cuda_version, CUDA_FIXES["118"])
    
    # Utilizăm versiunea specificată sau cea din config
    if torch_version:
        torch_ver = torch_version
    else:
        torch_ver = cuda_config["torch_version"]
    
    cuda_extra = cuda_config["cuda_extra"]
    
    logger.info(f"🔄 Instalez PyTorch {torch_ver} cu CUDA {cuda_version}...")
    
    # Instalăm PyTorch cu versiunea specificată
    install_cmd = f"pip install torch=={torch_ver} torchvision torchaudio {cuda_extra}"
    run_command(install_cmd, desc=f"Instalare PyTorch {torch_ver}", shell=True)
    
    # Verificăm instalarea CUDA
    if os.path.exists(APP_DIR / "check_cuda.py"):
        logger.info("🔄 Verific instalarea CUDA...")
        cuda_check = run_command(f"python {APP_DIR / 'check_cuda.py'}", 
                                desc="Verificare CUDA PyTorch", check=False, shell=True)
        
        if cuda_check and cuda_check.returncode == 0:
            logger.info("✅ PyTorch + CUDA instalat corect")
        else:
            logger.warning("⚠️ Verificare CUDA eșuată. Încerc o altă versiune...")
            if cuda_version == "118":
                logger.info("🔄 Încercăm cu CUDA 117...")
                install_pytorch("117", torch_version="1.13.1", fix_cuda=True)

def install_package(package, is_essential=True, alternatives=None):
    """Instalează un pachet Python cu alternativele ca backup"""
    logger.info(f"🔄 Instalez {package}...")
    
    try:
        # Încercăm prima dată versiunea principală
        result = run_command(f"pip install {package}", 
                          desc=f"Instalare {package}", 
                          shell=True, 
                          check=False)
        
        if result and result.returncode == 0:
            logger.info(f"✅ {package} instalat cu succes")
            return True
        
        logger.warning(f"⚠️ Nu am putut instala {package}, încerc alternative...")
        
        # Dacă prima versiune eșuează, încercăm alternativele
        if alternatives:
            for alt in alternatives:
                if not alt:  # Opțiune goală înseamnă că putem sări peste
                    logger.info(f"🔶 {package} este opțional, continuăm fără el")
                    return True
                
                logger.info(f"🔄 Încerc alternativa: {alt}")
                alt_result = run_command(f"pip install {alt}", 
                                      desc=f"Instalare {alt}", 
                                      shell=True, 
                                      check=False)
                
                if alt_result and alt_result.returncode == 0:
                    logger.info(f"✅ Alternativa {alt} instalată cu succes")
                    return True
        
        # Dacă ajungem aici, toate încercările au eșuat
        if is_essential:
            logger.error(f"❌ Nu am putut instala pachetul esențial {package} și alternativele")
            if not args.continue_on_error:
                raise Exception(f"Instalare eșuată pentru pachetul esențial {package}")
            return False
        else:
            logger.warning(f"⚠️ Nu am putut instala pachetul opțional {package}, dar continuăm instalarea")
            return True
            
    except Exception as e:
        logger.error(f"❌ Eroare la instalarea {package}: {e}")
        if is_essential and not args.continue_on_error:
            raise
        return False

def install_dependencies():
    """Instalează dependențele Python necesare"""
    logger.info("🔄 Instalez pachete Python cu strategii de failover...")
    
    success_count = 0
    failure_count = 0
    
    for package_info in PACKAGES:
        package, is_essential, alternatives = package_info
        if install_package(package, is_essential, alternatives):
            success_count += 1
        else:
            failure_count += 1
    
    # IP-Adapter-ul necesită o instalare specială
    logger.info("🔄 Instalez IP-Adapter...")
    try:
        run_command(
            "pip install git+https://github.com/tencent-ailab/IP-Adapter.git@main",
            desc="Instalare IP-Adapter",
            shell=True,
            check=False
        )
        success_count += 1
    except:
        logger.warning("⚠️ Instalare IP-Adapter eșuată, încerc alternativa...")
        try:
            run_command(
                "pip install ip-adapter",
                desc="Instalare IP-Adapter (alternativ)",
                shell=True,
                check=False
            )
            success_count += 1
        except:
            logger.error("❌ Nu am putut instala IP-Adapter")
            failure_count += 1
    
    # Instalăm diffusers direct din GitHub pentru compatibilitate maximă
    logger.info("🔄 Instalez Diffusers...")
    try:
        # Încercăm întâi o versiune fixată pentru stabilitate
        run_command(
            "pip install diffusers==0.21.4",
            desc="Instalare Diffusers (versiune fixată)",
            shell=True,
            check=False
        )
        success_count += 1
    except:
        logger.warning("⚠️ Instalare Diffusers eșuată, încerc din GitHub...")
        try:
            run_command(
                "pip install git+https://github.com/huggingface/diffusers.git@v0.21.4",
                desc="Instalare Diffusers din GitHub",
                shell=True,
                check=False
            )
            success_count += 1
        except:
            logger.error("❌ Nu am putut instala Diffusers")
            failure_count += 1
    
    # Xformers pentru optimizare (opțional)
    if not args.no_xformers:
        logger.info("🔄 Instalez xformers pentru optimizare...")
        try:
            run_command(
                "pip install xformers==0.0.20", 
                desc="Instalare xformers", 
                shell=True,
                check=False
            )
            success_count += 1
        except:
            logger.warning("⚠️ Instalare xformers eșuată, încerc o versiune mai veche...")
            try:
                run_command(
                    "pip install xformers==0.0.17", 
                    desc="Instalare xformers (versiune alternativă)", 
                    shell=True,
                    check=False
                )
                success_count += 1
            except:
                logger.error("❌ Nu am putut instala xformers, dar continuăm fără el")
                failure_count += 1
    
    logger.info(f"📊 Instalare dependențe completă: {success_count} succese, {failure_count} eșecuri")
    return success_count, failure_count

def copy_application_files():
    """Copiază fișierele aplicației în directorul de instalare"""
    logger.info(f"🔄 Copiez fișierele aplicației din {REPO_PATH} în {APP_DIR}...")
    
    # Verificăm dacă app.py există în repo
    app_py_path = REPO_PATH / "app.py"
    if app_py_path.exists():
        shutil.copy(app_py_path, APP_DIR / "app.py")
        logger.info("✅ app.py copiat")
    else:
        logger.warning(f"⚠️ app.py nu există în {REPO_PATH}")
        
        # Căutăm app.py în toate subdirectoarele
        found = False
        for root, dirs, files in os.walk(REPO_PATH):
            if "app.py" in files:
                src_path = Path(root) / "app.py"
                shutil.copy(src_path, APP_DIR / "app.py")
                logger.info(f"✅ app.py găsit și copiat din {src_path}")
                found = True
                break
        
        if not found:
            # Dacă nu găsim app.py, căutăm un fișier Python care ar putea fi principal
            py_files = list(REPO_PATH.glob("*.py"))
            if py_files:
                main_py = py_files[0]
                shutil.copy(main_py, APP_DIR / "app.py")
                logger.info(f"✅ Am copiat {main_py} ca app.py")
            else:
                logger.error("❌ Nu am găsit niciun fișier Python principal")

    # Verificăm și copiem runpod_utils.py dacă există
    utils_path = REPO_PATH / "runpod_utils.py"
    if utils_path.exists():
        shutil.copy(utils_path, APP_DIR / "runpod_utils.py")
        logger.info("✅ runpod_utils.py copiat")
    
    # Copiem și alte fișiere necesare
    for file in ["requirements.txt", "README.md", "LICENSE"]:
        src_path = REPO_PATH / file
        if src_path.exists():
            shutil.copy(src_path, APP_DIR / file)
            logger.info(f"✅ {file} copiat")

def create_fallback_app():
    """Creează un fișier app.py minimal dacă nu există unul"""
    if not (APP_DIR / "app.py").exists():
        logger.warning("⚠️ Nu am găsit app.py în repo, creez un fișier minimal")
        
        minimal_app = """
import os
import sys
import gradio as gr
import torch

# Verificăm PyTorch
print(f"PyTorch {torch.__version__}")
if torch.cuda.is_available():
    print(f"CUDA: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA nu este disponibil. Performanța va fi redusă.")

# Interfața minimală
def generate_image(prompt):
    # Aici ar trebui să fie codul real
    return f"Simulare generare imagine pentru: {prompt}"

# Creăm interfața Gradio
with gr.Blocks(title="FusionFrame Minimal") as demo:
    gr.Markdown("# FusionFrame - Instalare minimală")
    gr.Markdown("⚠️ Aceasta este o versiune minimală. Copiați app.py original în acest director.")
    
    with gr.Row():
        with gr.Column():
            prompt = gr.Textbox(label="Prompt")
            generate_btn = gr.Button("Generează")
        
        with gr.Column():
            output = gr.Textbox(label="Rezultat")
    
    generate_btn.click(generate_image, inputs=[prompt], outputs=[output])

if __name__ == "__main__":
    demo.launch(share=True)
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
"""
    
    # Salvăm wrapper-ul ca app_wrapper.py
    with open(APP_DIR / "app_wrapper.py", "w") as f:
        f.write(wrapper_script)
    
    logger.info("✅ Wrapper de compatibilitate creat")

def create_info_file():
    """Creează un fișier JSON cu informații despre instalare"""
    info = {
        "install_dir": str(APP_DIR),
        "cache_dir": str(CACHE_DIR),
        "port": args.port,
        "shared": args.shared,
        "cuda_version": args.cuda_version,
        "torch_version": args.torch_version,
        "install_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "python_version": sys.version
    }
    
    info_path = APP_DIR / "install_info.json"
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)
    
    logger.info(f"✅ Informații instalare salvate în: {info_path}")

def main():
    """Funcția principală de instalare"""
    start_time = time.time()
    logger.info(f"🚀 Începere instalare FusionFrame pe RunPod")
    logger.info(f"📂 Director instalare: {APP_DIR}")
    logger.info(f"📂 Director repo: {REPO_PATH}")
    
    # Creăm directoarele
    setup_directories()
    
    # Verificăm GPU-ul
    has_gpu = check_gpu()
    
    # Corectăm variabilele de mediu CUDA
    if has_gpu:
        fix_cuda_environment()
    
    # Instalăm PyTorch
    install_pytorch(args.cuda_version, args.torch_version, fix_cuda=args.fix_cuda)
    
    # Instalăm dependențele
    success_count, failure_count = install_dependencies()
    
    # Copiem fișierele aplicației
    copy_application_files()
    
    # Creăm un app.py minimal dacă nu există
    create_fallback_app()
    
    # Creăm wrapper cu fix-uri pentru CUDA
    create_wrapper()
    
    # Creăm script de pornire
    create_startup_script()
    
    # Creăm fișierul cu informații
    create_info_file()
    
    # Afișăm sumar
    elapsed_time = time.time() - start_time
    logger.info(f"✨ Instalare completă în {elapsed_time:.1f} secunde!")
    logger.info(f"📋 Sumar instalare:")
    logger.info(f"   - Director instalare: {APP_DIR}")
    logger.info(f"   - Director cache: {CACHE_DIR}")
    logger.info(f"   - Pachete instalate cu succes: {success_count}")
    if failure_count > 0:
        logger.warning(f"   - Pachete cu probleme: {failure_count} (unele pot fi opționale)")
    logger.info(f"   - Pentru pornire, rulați: {APP_DIR}/start.sh")
    logger.info(f"   - Interfața web va fi disponibilă la: http://localhost:{args.port}")
    if args.shared:
        logger.info(f"   - Link public va fi afișat la pornire")
    
    logger.info(f"🎉 Instalare finalizată cu succes!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("⚠️ Instalare întreruptă de utilizator")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Eroare în timpul instalării: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
