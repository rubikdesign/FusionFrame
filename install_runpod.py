#!/usr/bin/env python3
"""
Script de instalare pentru Platforma de Generare Imagini pe RunPod
Acest script instalează toate dependențele necesare și configurează mediul pentru rularea aplicației.
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

# Configurare parser argumente
parser = argparse.ArgumentParser(description="Instalare Platformă de Generare Imagini pe RunPod")
parser.add_argument("--install-dir", type=str, default="clothing_generator", 
                    help="Director pentru instalarea aplicației")
parser.add_argument("--port", type=int, default=7860, 
                    help="Port pentru interfața web Gradio")
parser.add_argument("--cuda-version", type=str, default="118", 
                    choices=["116", "117", "118", "121"], 
                    help="Versiunea CUDA pentru PyTorch")
parser.add_argument("--no-xformers", action="store_true", 
                    help="Dezactivează instalarea xformers (pentru compatibilitate)")
parser.add_argument("--download-models", action="store_true", 
                    help="Descarcă modele predefinite (poate dura ceva timp)")
parser.add_argument("--shared", action="store_true", 
                    help="Permite accesul public la interfață")
args = parser.parse_args()

# Definire constante
INSTALL_DIR = Path(args.install_dir)
APP_DIR = INSTALL_DIR / "app"
CACHE_DIR = INSTALL_DIR / "cache"
LOG_DIR = INSTALL_DIR / "logs"
MODEL_DIR = INSTALL_DIR / "models"
REQUIRED_PACKAGES = [
    "torch", "torchvision", "torchaudio", 
    "diffusers==0.22.1", "transformers", "accelerate", "safetensors",
    "gradio", "numpy", "pillow", "opencv-python", "tqdm", "requests",
    "controlnet-aux", "timm", "mediapipe"
]

def run_command(command, desc=None, check=True, shell=False):
    """Execute a command and log output"""
    if desc:
        logger.info(f"🔄 {desc}...")
    
    cmd_list = command if shell else command.split()
    try:
        process = subprocess.run(
            cmd_list, 
            check=check, 
            shell=shell,
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            text=True
        )
        if process.stdout and len(process.stdout) > 0:
            logger.debug(process.stdout)
        if process.returncode != 0 and process.stderr:
            logger.error(f"Command failed: {process.stderr}")
        return process
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed: {e}")
        if not check:
            return None
        sys.exit(1)

def setup_directories():
    """Create necessary directories"""
    logger.info("🔄 Creez directoare necesare...")
    for dir_path in [APP_DIR, CACHE_DIR, LOG_DIR, MODEL_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"✅ Creat director: {dir_path}")

def check_gpu():
    """Check if CUDA is available and get GPU info"""
    logger.info("🔄 Verific disponibilitatea GPU...")
    try:
        nvidia_smi = run_command("nvidia-smi", desc="Rulare nvidia-smi", check=False)
        if nvidia_smi and nvidia_smi.returncode == 0:
            logger.info("✅ GPU detectat. Verificare CUDA...")
            
            # Create a small Python script to check CUDA with PyTorch
            check_script = """
import torch
if torch.cuda.is_available():
    print(f"CUDA available: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("CUDA not available")
    exit(1)
"""
            with open(INSTALL_DIR / "check_cuda.py", "w") as f:
                f.write(check_script)
            
            # Run the script after torch is installed
            return True
        else:
            logger.warning("⚠️ GPU not detected. The application will run in CPU mode (very slow).")
            return False
    except Exception as e:
        logger.warning(f"⚠️ Could not check GPU: {e}")
        return False

def install_pytorch(cuda_version):
    """Install PyTorch with appropriate CUDA version"""
    cuda_url = f"https://download.pytorch.org/whl/cu{cuda_version}"
    logger.info(f"🔄 Instalez PyTorch cu suport CUDA {cuda_version}...")
    
    command = f"pip install torch torchvision torchaudio --extra-index-url {cuda_url}"
    run_command(command, desc="Instalare PyTorch", shell=True)
    
    # Verify PyTorch installation
    if os.path.exists(INSTALL_DIR / "check_cuda.py"):
        cuda_check = run_command(f"python {INSTALL_DIR / 'check_cuda.py'}", desc="Verificare CUDA PyTorch", check=False, shell=True)
        if cuda_check and cuda_check.returncode == 0:
            logger.info("✅ PyTorch cu CUDA instalat corect")
        else:
            logger.warning("⚠️ PyTorch instalat, dar CUDA nu funcționează corect")

def install_dependencies():
    """Install required Python packages"""
    logger.info("🔄 Instalez dependențe Python...")
    
    # Main packages
    packages = " ".join(REQUIRED_PACKAGES)
    run_command(f"pip install {packages}", desc="Instalare pachete principale", shell=True)
    
    # IP-Adapter
    logger.info("🔄 Instalez IP-Adapter...")
    run_command(
        "pip install git+https://github.com/tencent-ailab/IP-Adapter.git",
        desc="Instalare IP-Adapter",
        shell=True
    )
    
    # Xformers for optimization (if enabled)
    if not args.no_xformers:
        logger.info("🔄 Instalez xformers pentru optimizare...")
        run_command("pip install xformers", desc="Instalare xformers", shell=True)
    
    # Memory efficiency optimization
    logger.info("🔄 Instalez bitsandbytes pentru optimizare memorie...")
    run_command("pip install bitsandbytes", desc="Instalare bitsandbytes", shell=True)

def download_models():
    """Download pre-trained models"""
    if not args.download_models:
        logger.info("⏩ Descărcarea modelelor predefinite este dezactivată")
        return
    
    logger.info("🔄 Descărcarea modelelor predefinite (poate dura ceva timp)...")
    
    # Set HF_HOME to our cache directory for model downloads
    os.environ["HF_HOME"] = str(CACHE_DIR)
    
    # Create a script to download models
    download_script = """
import os
from diffusers import StableDiffusionXLPipeline, StableDiffusionPipeline, AutoencoderKL
from huggingface_hub import snapshot_download

# Configure cache directory
os.environ["HF_HOME"] = "%s"

# Download base models
print("Downloading SDXL 1.0 (base only, not full pipeline)...")
StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype="auto",
    use_safetensors=True,
    variant="fp16"
)

print("Downloading SD 1.5 (base only, not full pipeline)...")
StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype="auto",
)

# Download VAEs
print("Downloading VAEs...")
AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")

# Download ControlNet models (just a couple for testing)
print("Downloading sample ControlNet models...")
snapshot_download(repo_id="lllyasviel/control_v11p_sd15_openpose", repo_type="model")
snapshot_download(repo_id="lllyasviel/control_v11p_sd15_canny", repo_type="model")
""" % str(CACHE_DIR)
    
    with open(INSTALL_DIR / "download_models.py", "w") as f:
        f.write(download_script)
    
    # Run the download script
    run_command(f"python {INSTALL_DIR / 'download_models.py'}", desc="Descărcare modele", shell=True)
    logger.info("✅ Modele descărcate")

def copy_application_code():
    """Copy the application code to the installation directory"""
    logger.info("🔄 Copiez codul aplicației...")
    
    # Check if app.py exists in the current directory
    if os.path.exists("app.py"):
        shutil.copy("app.py", APP_DIR / "app.py")
        logger.info("✅ app.py copiat în directorul de instalare")
    else:
        # If not, create a placeholder that will direct user to download the actual code
        logger.warning("⚠️ app.py nu a fost găsit. Creez un placeholder...")
        placeholder = """
print("Aplicația nu a fost găsită. Vă rugăm să descărcați codul aplicației și să-l salvați ca app.py în directorul %s")
""" % str(APP_DIR)
        with open(APP_DIR / "app.py", "w") as f:
            f.write(placeholder)

def create_startup_script():
    """Create a startup script for easy launching"""
    logger.info("🔄 Creez script de pornire...")
    
    startup_script = """#!/bin/bash
# Script de pornire pentru Platforma de Generare Imagini

# Setare variabile de mediu
export HF_HOME="%s"
export PYTHONPATH="$PYTHONPATH:%s"

# Pornire aplicație
cd %s
python app.py --port %d %s

""" % (
    str(CACHE_DIR),
    str(APP_DIR),
    str(APP_DIR),
    args.port,
    "--share" if args.shared else ""
)
    
    startup_path = INSTALL_DIR / "start.sh"
    with open(startup_path, "w") as f:
        f.write(startup_script)
    
    # Make executable
    os.chmod(startup_path, 0o755)
    logger.info(f"✅ Script de pornire creat: {startup_path}")
    
    # Create a Python version too
    py_startup = """#!/usr/bin/env python3
import os
import subprocess
import sys

# Set environment variables
os.environ["HF_HOME"] = "%s"
os.environ["PYTHONPATH"] = os.environ.get("PYTHONPATH", "") + ":%s"

# Change to app directory
os.chdir("%s")

# Start the application
cmd = [sys.executable, "app.py", "--port", "%d"]
%s
subprocess.run(cmd)
""" % (
    str(CACHE_DIR),
    str(APP_DIR),
    str(APP_DIR),
    args.port,
    'cmd.append("--share")' if args.shared else '# No share flag'
)
    
    py_startup_path = INSTALL_DIR / "start.py"
    with open(py_startup_path, "w") as f:
        f.write(py_startup)
    
    # Make executable
    os.chmod(py_startup_path, 0o755)
    logger.info(f"✅ Script Python de pornire creat: {py_startup_path}")

def create_runpod_handler():
    """Create a RunPod serverless handler"""
    logger.info("🔄 Creez handler RunPod...")
    
    handler_script = """#!/usr/bin/env python3
'''
RunPod handler for Clothing Try-On Image Generator
'''
import os
import sys
import runpod
import base64
import io
from PIL import Image
import json
import subprocess
import time

# Get the directory of this script
HANDLER_DIR = os.path.dirname(os.path.abspath(__file__))
APP_DIR = os.path.join("%s", "app")
sys.path.append(APP_DIR)

# Import app functions (will fail gracefully if not found)
try:
    from app import generate_images, prepare_image, load_model
    DIRECT_IMPORT = True
except ImportError:
    DIRECT_IMPORT = False
    print("Warning: Could not import app directly. Will use subprocess.")

def save_encoded_image(b64_image, output_path):
    """Save base64 encoded image to file"""
    with open(output_path, "wb") as f:
        f.write(base64.b64decode(b64_image))
    return output_path

def handler(event):
    '''Handler function for RunPod serverless requests'''
    try:
        # Get input data
        input_data = event.get("input", {})
        
        # Extract parameters
        model_name = input_data.get("model", "SDXL 1.0")
        woman_b64 = input_data.get("woman_image")
        clothing_b64 = input_data.get("clothing_image")
        background_b64 = input_data.get("background_image")
        positive_prompt = input_data.get("prompt", "high quality, photorealistic")
        negative_prompt = input_data.get("negative_prompt", "deformed, bad anatomy")
        
        # ControlNet settings
        controlnet_type = input_data.get("controlnet_type", "None")
        controlnet_scale = float(input_data.get("controlnet_scale", 0.5))
        
        # IP-Adapter settings
        ip_adapter_model = input_data.get("ip_adapter_model", "None")
        ip_adapter_scale = float(input_data.get("ip_adapter_scale", 0.6))
        
        # Generation settings
        steps = int(input_data.get("steps", 30))
        guidance_scale = float(input_data.get("guidance_scale", 7.5))
        seed = int(input_data.get("seed", -1))
        width = int(input_data.get("width", 768))
        height = int(input_data.get("height", 768))
        num_outputs = int(input_data.get("num_outputs", 1))
        
        # Create temp directory for inputs and outputs
        job_id = event.get("id", str(int(time.time())))
        temp_dir = f"/tmp/runpod_job_{job_id}"
        os.makedirs(temp_dir, exist_ok=True)
        
        # Save images if provided
        woman_path = None
        clothing_path = None
        background_path = None
        
        if woman_b64:
            woman_path = os.path.join(temp_dir, "woman.png")
            save_encoded_image(woman_b64, woman_path)
            
        if clothing_b64:
            clothing_path = os.path.join(temp_dir, "clothing.png")
            save_encoded_image(clothing_b64, clothing_path)
            
        if background_b64:
            background_path = os.path.join(temp_dir, "background.png")
            save_encoded_image(background_b64, background_path)
        
        # Generate images
        if DIRECT_IMPORT:
            # Direct method using imported functions
            results, used_seed, _ = generate_images(
                model_name=model_name,
                woman_image=woman_path,
                clothing_image=clothing_path,
                background_image=background_path,
                positive_prompt=positive_prompt,
                negative_prompt=negative_prompt,
                lora1="None", lora1_weight=0.7,
                lora2="None", lora2_weight=0.7,
                lora3="None", lora3_weight=0.7,
                lora4="None", lora4_weight=0.7,
                lora5="None", lora5_weight=0.7,
                controlnet_type=controlnet_type,
                controlnet_conditioning_scale=controlnet_scale,
                ip_adapter_name=ip_adapter_model,
                ip_adapter_scale=ip_adapter_scale,
                denoising_strength=0.75,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                image_width=width,
                image_height=height,
                scheduler_name="DPM++ 2M Karras",
                vae_name="Default",
                num_outputs=num_outputs,
                seed=seed
            )
        else:
            # Subprocess method
            cmd = [
                sys.executable, os.path.join(APP_DIR, "app.py"),
                "--headless",
                "--model", model_name,
                "--prompt", positive_prompt,
                "--negative_prompt", negative_prompt,
                "--steps", str(steps),
                "--guidance_scale", str(guidance_scale),
                "--seed", str(seed),
                "--width", str(width),
                "--height", str(height),
                "--output_dir", temp_dir,
                "--num_outputs", str(num_outputs)
            ]
            
            if woman_path:
                cmd.extend(["--woman_image", woman_path])
            if clothing_path:
                cmd.extend(["--clothing_image", clothing_path])
            if background_path:
                cmd.extend(["--background_image", background_path])
            if controlnet_type != "None":
                cmd.extend(["--controlnet_type", controlnet_type, "--controlnet_scale", str(controlnet_scale)])
            if ip_adapter_model != "None":
                cmd.extend(["--ip_adapter_model", ip_adapter_model, "--ip_adapter_scale", str(ip_adapter_scale)])
            
            # Run the command
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                return {"error": f"Generation failed: {result.stderr}"}
            
            # Parse results
            results = []
            for file in os.listdir(temp_dir):
                if file.startswith("generated_") and file.endswith(".png"):
                    results.append(os.path.join(temp_dir, file))
            
            used_seed = seed  # This will be incorrect if seed was -1, but no easy way to know
        
        # Convert the generated images to base64
        output_images = []
        for img_path in results:
            with open(img_path, "rb") as img_file:
                img_data = base64.b64encode(img_file.read()).decode('utf-8')
                output_images.append(img_data)
        
        # Return results
        return {
            "id": job_id,
            "output": {
                "images": output_images,
                "seed": used_seed,
                "count": len(output_images)
            }
        }
    
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        return {"error": str(e), "details": error_details}

# Start the serverless handler if running as script
if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
""" % INSTALL_DIR
    
    handler_path = INSTALL_DIR / "runpod_handler.py"
    with open(handler_path, "w") as f:
        f.write(handler_script)
    logger.info(f"✅ RunPod handler creat: {handler_path}")

def create_info_file():
    """Create an info file with installation details"""
    info = {
        "install_dir": str(INSTALL_DIR),
        "app_dir": str(APP_DIR),
        "cache_dir": str(CACHE_DIR),
        "port": args.port,
        "shared": args.shared,
        "cuda_version": args.cuda_version,
        "install_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "python_version": sys.version,
    }
    
    info_path = INSTALL_DIR / "install_info.json"
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)
    
    logger.info(f"✅ Informații instalare salvate în: {info_path}")

def main():
    """Main installation function"""
    start_time = time.time()
    logger.info(f"🚀 Începere instalare Platformă de Generare Imagini pe RunPod")
    logger.info(f"📂 Director instalare: {INSTALL_DIR}")
    
    # Create directories
    setup_directories()
    
    # Check for GPU and CUDA
    has_gpu = check_gpu()
    
    # Install PyTorch
    install_pytorch(args.cuda_version)
    
    # Install dependencies
    install_dependencies()
    
    # Copy application code
    copy_application_code()
    
    # Download models if requested
    download_models()
    
    # Create startup scripts
    create_startup_script()
    
    # Create RunPod handler
    create_runpod_handler()
    
    # Create info file
    create_info_file()
    
    # Print summary
    elapsed_time = time.time() - start_time
    logger.info(f"✨ Instalare completă în {elapsed_time:.1f} secunde!")
    logger.info(f"📋 Sumar instalare:")
    logger.info(f"   - Director instalare: {INSTALL_DIR}")
    logger.info(f"   - Director cache: {CACHE_DIR}")
    logger.info(f"   - Pentru pornire, rulați: {INSTALL_DIR}/start.sh")
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
        logger.error(traceback.format_exc())
        sys.exit(1)
