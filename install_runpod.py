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
# Pinned, compatible versions to avoid import errors
REQUIRED_PACKAGES = [
    "torch", "torchvision", "torchaudio",
    "diffusers==0.21.0", "huggingface_hub==0.14.1", "transformers==4.33.0", "accelerate", "safetensors",
    "gradio", "numpy", "pillow", "opencv-python", "tqdm", "requests",
    "controlnet-aux", "timm", "mediapipe"
]

# Helper to run shell commands
def run_command(command, desc=None, check=True, shell=False):
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

# Create necessary directories
def setup_directories():
    logger.info("🔄 Creating necessary directories...")
    for d in [APP_DIR, CACHE_DIR, LOG_DIR, MODEL_DIR]:
        d.mkdir(parents=True, exist_ok=True)
        logger.info(f"✅ Created directory: {d}")

# Check for GPU and CUDA
def check_gpu():
    logger.info("🔄 Checking GPU availability...")
    try:
        proc = run_command("nvidia-smi", desc="Running nvidia-smi", check=False)
        if proc and proc.returncode == 0:
            logger.info("✅ GPU detected. Checking CUDA compatibility...")
            script = """
import torch
if torch.cuda.is_available():
    print(f"CUDA available: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("CUDA not available")
    exit(1)
"""
            (INSTALL_DIR / "check_cuda.py").write_text(script)
            return True
        else:
            logger.warning("⚠️ No GPU detected. Running in CPU mode.")
            return False
    except Exception as e:
        logger.warning(f"⚠️ Could not check GPU: {e}")
        return False

# Install PyTorch with specified CUDA
def install_pytorch(cuda_version):
    url = f"https://download.pytorch.org/whl/cu{cuda_version}"
    logger.info(f"🔄 Installing PyTorch with CUDA {cuda_version} support...")
    run_command(f"pip install torch torchvision torchaudio --extra-index-url {url}", desc="Installing PyTorch", shell=True)
    if (INSTALL_DIR / "check_cuda.py").exists():
        chk = run_command(f"python {INSTALL_DIR / 'check_cuda.py'}", desc="Verifying CUDA in PyTorch", check=False, shell=True)
        if chk and chk.returncode == 0:
            logger.info("✅ PyTorch with CUDA installed successfully")
        else:
            logger.warning("⚠️ PyTorch installed, but CUDA check failed.")

# Install all dependencies, including pinned versions
def install_dependencies():
    logger.info("🔄 Installing Python dependencies...")
    pkgs = " ".join(REQUIRED_PACKAGES)
    run_command(f"pip install {pkgs}", desc="Installing core packages", shell=True)
    logger.info("🔄 Installing IP-Adapter...")
    run_command("pip install git+https://github.com/tencent-ailab/IP-Adapter.git", desc="Installing IP-Adapter", shell=True)
    if not args.no_xformers:
        logger.info("🔄 Installing xformers for optimization...")
        run_command("pip install xformers", desc="Installing xformers", shell=True)
    logger.info("🔄 Installing bitsandbytes for memory optimization...")
    run_command("pip install bitsandbytes", desc="Installing bitsandbytes", shell=True)

# Main installation flow
def main():
    start = time.time()
    logger.info("🚀 Starting FusionFrame installation")
    logger.info(f"📂 Installation directory: {INSTALL_DIR}")

    setup_directories()
    check_gpu()
    install_pytorch(args.cuda_version)
    install_dependencies()

    elapsed = time.time() - start
    logger.info(f"✨ Installation completed in {elapsed:.1f} seconds!")
    logger.info("📋 Summary:")
    logger.info(f"   - Installation Directory: {INSTALL_DIR}")
    logger.info(f"   - Cache Directory: {CACHE_DIR}")
    logger.info(f"   - To start the application: {INSTALL_DIR}/start.sh")
    logger.info(f"   - Web UI at: http://localhost:{args.port}")
    if args.shared:
        logger.info("   - Public share link will be displayed on startup")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("⚠️ Installation interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Installation error: {e}")
        import traceback; traceback.print_exc()
        sys.exit(1)
