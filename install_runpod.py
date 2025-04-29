#!/usr/bin/env python3

import os
import sys
import subprocess
import argparse
import shutil
import logging
from pathlib import Path
import time
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f"install_log_{time.strftime('%Y%m%d_%H%M%S')}.log")
    ]
)
logger = logging.getLogger("installer")

# Argument parser setup
parser = argparse.ArgumentParser(description="FusionFrame - Installation Script for Image Generation Platform")
parser.add_argument("--install-dir", type=str, default="fusionframe",
                    help="Directory for installing the application")
parser.add_argument("--port", type=int, default=7860,
                    help="Port for Gradio web interface")
parser.add_argument("--cuda-version", type=str, default="118",
                    choices=["116", "117", "118", "121"],
                    help="CUDA version for PyTorch")
parser.add_argument("--no-xformers", action="store_true",
                    help="Disable xformers installation (for compatibility)")
parser.add_argument("--download-models", action="store_true",
                    help="Download pre-trained models (may take time)")
parser.add_argument("--shared", action="store_true",
                    help="Allow public sharing of the web UI")
args = parser.parse_args()

# Constants
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
        if process.stdout:
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
    """Create required directories"""
    logger.info("🔄 Creating necessary directories...")
    for dir_path in [APP_DIR, CACHE_DIR, LOG_DIR, MODEL_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"✅ Created directory: {dir_path}")

def check_gpu():
    """Check CUDA and GPU availability"""
    logger.info("🔄 Checking GPU availability...")
    try:
        nvidia_smi = run_command("nvidia-smi", desc="Running nvidia-smi", check=False)
        if nvidia_smi and nvidia_smi.returncode == 0:
            logger.info("✅ GPU detected. Checking CUDA compatibility...")

            # Small script to check CUDA with PyTorch
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

            return True
        else:
            logger.warning("⚠️ No GPU detected. The application will run in CPU mode (very slow).")
            return False
    except Exception as e:
        logger.warning(f"⚠️ Could not check GPU: {e}")
        return False

def install_pytorch(cuda_version):
    """Install PyTorch with the specified CUDA version"""
    cuda_url = f"https://download.pytorch.org/whl/cu{cuda_version}"
    logger.info(f"🔄 Installing PyTorch with CUDA {cuda_version} support...")

    command = f"pip install torch torchvision torchaudio --extra-index-url {cuda_url}"
    run_command(command, desc="Installing PyTorch", shell=True)

    # Verify installation
    if os.path.exists(INSTALL_DIR / "check_cuda.py"):
        cuda_check = run_command(f"python {INSTALL_DIR / 'check_cuda.py'}", desc="Verifying CUDA in PyTorch", check=False, shell=True)
        if cuda_check and cuda_check.returncode == 0:
            logger.info("✅ PyTorch with CUDA installed successfully")
        else:
            logger.warning("⚠️ PyTorch installed, but CUDA check failed.")

def install_dependencies():
    """Install all required Python packages"""
    logger.info("🔄 Installing Python dependencies...")

    packages = " ".join(REQUIRED_PACKAGES)
    run_command(f"pip install {packages}", desc="Installing core packages", shell=True)

    logger.info("🔄 Installing IP-Adapter...")
    run_command("pip install git+https://github.com/tencent-ailab/IP-Adapter.git", desc="Installing IP-Adapter", shell=True)

    if not args.no_xformers:
        logger.info("🔄 Installing xformers for optimization...")
        run_command("pip install xformers", desc="Installing xformers", shell=True)

    logger.info("🔄 Installing bitsandbytes for memory optimization...")
    run_command("pip install bitsandbytes", desc="Installing bitsandbytes", shell=True)

def main():
    """Main installation function"""
    start_time = time.time()
    logger.info(f"🚀 Starting FusionFrame installation")
    logger.info(f"📂 Installation directory: {INSTALL_DIR}")

    setup_directories()
    check_gpu()
    install_pytorch(args.cuda_version)
    install_dependencies()

    elapsed_time = time.time() - start_time
    logger.info(f"✨ Installation completed in {elapsed_time:.1f} seconds!")
    logger.info(f"📋 Summary:")
    logger.info(f"   - Installation Directory: {INSTALL_DIR}")
    logger.info(f"   - Cache Directory: {CACHE_DIR}")
    logger.info(f"   - To start the application: {INSTALL_DIR}/start.sh")
    logger.info(f"   - Web UI will be available at: http://localhost:{args.port}")
    if args.shared:
        logger.info(f"   - Public share link will be displayed on startup")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("⚠️ Installation interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Installation error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)
