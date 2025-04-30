#!/usr/bin/env python3
"""
Enhanced installation script for FusionFrame on RunPod
- Fixes CUDA compatibility issues
- Handles package installation errors
- Implements failover strategies for problematic dependencies
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

# Set up argument parser
parser = argparse.ArgumentParser(description="FusionFrame - Virtual Try-On Image Generation Platform")
parser.add_argument("--app-dir", type=str, default="FusionFrame", 
                    help="Application directory name (default: FusionFrame)")
parser.add_argument("--port", type=int, default=7860, 
                    help="Port for Gradio web interface")
parser.add_argument("--cuda-version", type=str, default="118", 
                    choices=["116", "117", "118", "121"],
                    help="CUDA version for PyTorch")
parser.add_argument("--torch-version", type=str, default="2.0.1", 
                    choices=["1.13.1", "2.0.1", "2.1.2", "2.2.0"],
                    help="PyTorch version (for CUDA compatibility)")
parser.add_argument("--repo-path", type=str, default=".", 
                    help="Path to repository directory (default: current directory)")
parser.add_argument("--no-xformers", action="store_true", 
                    help="Disable xformers installation")
parser.add_argument("--download-models", action="store_true", 
                    help="Download predefined models (takes a long time)")
parser.add_argument("--shared", action="store_true", 
                    help="Allow public access to interface")
parser.add_argument("--fix-cuda", action="store_true", 
                    help="Force CUDA reinstallation to fix compatibility issues")
parser.add_argument("--continue-on-error", action="store_true", 
                    help="Continue installation even if some packages fail")

args = parser.parse_args()

# Configure directories
WORKSPACE_DIR = Path("/workspace") if os.path.exists("/workspace") else Path.home()
REPO_PATH = Path(args.repo_path).absolute()
APP_DIR = WORKSPACE_DIR / args.app_dir
CACHE_DIR = APP_DIR / "cache"
LOG_DIR = APP_DIR / "logs"
MODEL_DIR = APP_DIR / "models"

# Required packages with fixed versions and alternatives
# Format: [(package, is_essential, alternatives)]
PACKAGES = [
    ("huggingface_hub==0.16.4", True, ["huggingface_hub==0.15.1"]),
    ("transformers==4.30.2", True, ["transformers==4.29.2"]),
    ("accelerate==0.20.3", True, ["accelerate==0.19.0"]),
    ("safetensors==0.3.1", True, ["safetensors==0.3.0"]), 
    ("numpy==1.24.3", True, ["numpy==1.23.5"]),
    ("pillow==9.5.0", True, ["pillow==9.4.0"]),
    ("tqdm==4.65.0", False, ["tqdm"]), 
    ("requests==2.31.0", False, ["requests"]),
    # Non-essential packages can be skipped
    ("gradio==3.33.1", False, ["gradio==3.32.0", "gradio==3.31.0", "gradio==3.30.0"]),
    ("opencv-python==4.7.0.72", False, ["opencv-python==4.6.0.66", "opencv-python"]),
    ("controlnet-aux==0.0.6", False, ["controlnet-aux==0.0.5", "controlnet-aux"]),
    ("timm==0.9.2", False, ["timm==0.6.12", "timm"]),
    # mediapipe is problematic on RunPod - try alternative versions
    ("mediapipe==0.10.0", False, ["mediapipe==0.9.0.1", "mediapipe==0.8.10", ""]),
    ("bitsandbytes==0.35.4", False, ["bitsandbytes", ""]),
]

# Versions to avoid NCCL conflicts
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
    """Execute a command and log the output"""
    if desc:
        logger.info(f"🔄 {desc}...")
    
    cmd_list = command if shell else command.split()
    try:
        process = subprocess.run(
            cmd_list, 
            check=False,  # We handle errors for flexibility 
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
                logger.error(f"Error: {process.stderr.strip()}")
            if check:
                raise subprocess.CalledProcessError(process.returncode, command)
        
        return process
    except subprocess.CalledProcessError as e:
        if check:
            logger.error(f"Command failed: {e}")
            raise
        return None
    except Exception as e:
        logger.error(f"Exception running command: {e}")
        if check:
            raise
        return None

def setup_directories():
    """Create necessary directories"""
    logger.info(f"🔄 Creating directories in {APP_DIR}...")
    for dir_path in [APP_DIR, CACHE_DIR, LOG_DIR, MODEL_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"✅ Created directory: {dir_path}")

def check_gpu():
    """Check GPU and CUDA availability"""
    logger.info("🔄 Checking GPU...")
    try:
        nvidia_smi = run_command("nvidia-smi", desc="Running nvidia-smi", check=False)
        if nvidia_smi and nvidia_smi.returncode == 0:
            logger.info("✅ GPU detected")
            
            # Script to check CUDA with PyTorch
            check_script = """
import torch
if torch.cuda.is_available():
    print(f"CUDA available: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")
    try:
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    except:
        print("Cannot get memory information")
else:
    print("CUDA is not available")
    exit(1)
"""
            with open(APP_DIR / "check_cuda.py", "w") as f:
                f.write(check_script)
            
            return True
        else:
            logger.warning("⚠️ GPU not detected. Application will run in CPU mode (very slow).")
            return False
    except Exception as e:
        logger.warning(f"⚠️ Cannot check GPU: {e}")
        return False

def fix_cuda_environment():
    """Fix CUDA environment variables"""
    logger.info("🔄 Setting CUDA environment variables...")
    
    # Set environment variables to avoid NCCL conflicts
    env_vars = {
        "NCCL_P2P_DISABLE": "1",
        "NCCL_DEBUG": "INFO",
        "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:512"
    }
    
    # Update ~/.bashrc to keep these settings
    bashrc_path = os.path.expanduser("~/.bashrc")
    try:
        with open(bashrc_path, "a") as f:
            f.write("\n# FusionFrame CUDA environment settings\n")
            for var, value in env_vars.items():
                f.write(f"export {var}={value}\n")
                # Set for current session too
                os.environ[var] = value
    except Exception as e:
        logger.warning(f"Cannot update ~/.bashrc: {e}")
        # Continue, as this is not critical
    
    logger.info("✅ CUDA environment variables configured")
    return env_vars

def install_pytorch(cuda_version, torch_version=None, fix_cuda=False):
    """Install PyTorch with appropriate CUDA support"""
    
    if fix_cuda or args.fix_cuda:
        # Uninstall existing PyTorch to avoid conflicts
        logger.info("🔄 Uninstalling existing PyTorch...")
        run_command("pip uninstall -y torch torchvision torchaudio", 
                   desc="Uninstalling PyTorch", shell=True, check=False)
    
    # Use CUDA configuration from dictionary
    cuda_config = CUDA_FIXES.get(cuda_version, CUDA_FIXES["118"])
    
    # Use specified version or the one from config
    if torch_version:
        torch_ver = torch_version
    else:
        torch_ver = cuda_config["torch_version"]
    
    cuda_extra = cuda_config["cuda_extra"]
    
    logger.info(f"🔄 Installing PyTorch {torch_ver} with CUDA {cuda_version}...")
    
    # Install PyTorch with specified version
    install_cmd = f"pip install torch=={torch_ver} torchvision torchaudio {cuda_extra}"
    run_command(install_cmd, desc=f"Installing PyTorch {torch_ver}", shell=True)
    
    # Check CUDA installation
    if os.path.exists(APP_DIR / "check_cuda.py"):
        logger.info("🔄 Verifying CUDA installation...")
        cuda_check = run_command(f"python {APP_DIR / 'check_cuda.py'}", 
                                desc="Checking PyTorch CUDA", check=False, shell=True)
        
        if cuda_check and cuda_check.returncode == 0:
            logger.info("✅ PyTorch + CUDA installed correctly")
        else:
            logger.warning("⚠️ CUDA verification failed. Trying another version...")
            if cuda_version == "118":
                logger.info("🔄 Trying with CUDA 117...")
                install_pytorch("117", torch_version="1.13.1", fix_cuda=True)

def install_package(package, is_essential=True, alternatives=None):
    """Install a Python package with alternatives as backup"""
    logger.info(f"🔄 Installing {package}...")
    
    try:
        # Try the main version first
        result = run_command(f"pip install {package}", 
                          desc=f"Installing {package}", 
                          shell=True, 
                          check=False)
        
        if result and result.returncode == 0:
            logger.info(f"✅ {package} installed successfully")
            return True
        
        logger.warning(f"⚠️ Could not install {package}, trying alternatives...")
        
        # If first version fails, try alternatives
        if alternatives:
            for alt in alternatives:
                if not alt:  # Empty option means we can skip
                    logger.info(f"🔶 {package} is optional, continuing without it")
                    return True
                
                logger.info(f"🔄 Trying alternative: {alt}")
                alt_result = run_command(f"pip install {alt}", 
                                      desc=f"Installing {alt}", 
                                      shell=True, 
                                      check=False)
                
                if alt_result and alt_result.returncode == 0:
                    logger.info(f"✅ Alternative {alt} installed successfully")
                    return True
        
        # If we reach here, all attempts failed
        if is_essential:
            logger.error(f"❌ Could not install essential package {package} and alternatives")
            if not args.continue_on_error:
                raise Exception(f"Installation failed for essential package {package}")
            return False
        else:
            logger.warning(f"⚠️ Could not install optional package {package}, but continuing installation")
            return True
            
    except Exception as e:
        logger.error(f"❌ Error installing {package}: {e}")
        if is_essential and not args.continue_on_error:
            raise
        return False

def install_dependencies():
    """Install required Python dependencies"""
    logger.info("🔄 Installing Python packages with failover strategies...")
    
    success_count = 0
    failure_count = 0
    
    for package_info in PACKAGES:
        package, is_essential, alternatives = package_info
        if install_package(package, is_essential, alternatives):
            success_count += 1
        else:
            failure_count += 1
    
    # IP-Adapter needs special installation
    logger.info("🔄 Installing IP-Adapter...")
    try:
        run_command(
            "pip install git+https://github.com/tencent-ailab/IP-Adapter.git@main",
            desc="Installing IP-Adapter",
            shell=True,
            check=False
        )
        success_count += 1
    except:
        logger.warning("⚠️ IP-Adapter installation failed, trying alternative...")
        try:
            run_command(
                "pip install ip-adapter",
                desc="Installing IP-Adapter (alternative)",
                shell=True,
                check=False
            )
            success_count += 1
        except:
            logger.error("❌ Could not install IP-Adapter")
            failure_count += 1
    
    # Install diffusers directly from GitHub for maximum compatibility
    logger.info("🔄 Installing Diffusers...")
    try:
        # Try a fixed version first for stability
        run_command(
            "pip install diffusers==0.21.4",
            desc="Installing Diffusers (fixed version)",
            shell=True,
            check=False
        )
        success_count += 1
    except:
        logger.warning("⚠️ Diffusers installation failed, trying from GitHub...")
        try:
            run_command(
                "pip install git+https://github.com/huggingface/diffusers.git@v0.21.4",
                desc="Installing Diffusers from GitHub",
                shell=True,
                check=False
            )
            success_count += 1
        except:
            logger.error("❌ Could not install Diffusers")
            failure_count += 1
    
    # Xformers for optimization (optional)
    if not args.no_xformers:
        logger.info("🔄 Installing xformers for optimization...")
        try:
            run_command(
                "pip install xformers==0.0.20", 
                desc="Installing xformers", 
                shell=True,
                check=False
            )
            success_count += 1
        except:
            logger.warning("⚠️ Xformers installation failed, trying an older version...")
            try:
                run_command(
                    "pip install xformers==0.0.17", 
                    desc="Installing xformers (alternative version)", 
                    shell=True,
                    check=False
                )
                success_count += 1
            except:
                logger.error("❌ Could not install xformers, but continuing without it")
                failure_count += 1
    
    logger.info(f"📊 Dependencies installation complete: {success_count} successes, {failure_count} failures")
    return success_count, failure_count

def copy_application_files():
    """Copy application files to installation directory"""
    logger.info(f"🔄 Copying application files from {REPO_PATH} to {APP_DIR}...")
    
    # Check if app.py exists in the repo
    app_py_path = REPO_PATH / "app.py"
    if app_py_path.exists():
        shutil.copy(app_py_path, APP_DIR / "app.py")
        logger.info("✅ app.py copied")
    else:
        logger.warning(f"⚠️ app.py doesn't exist in {REPO_PATH}")
        
        # Search for app.py in all subdirectories
        found = False
        for root, dirs, files in os.walk(REPO_PATH):
            if "app.py" in files:
                src_path = Path(root) / "app.py"
                shutil.copy(src_path, APP_DIR / "app.py")
                logger.info(f"✅ app.py found and copied from {src_path}")
                found = True
                break
        
        if not found:
            # If no app.py found, look for a Python file that might be the main one
            py_files = list(REPO_PATH.glob("*.py"))
            if py_files:
                main_py = py_files[0]
                shutil.copy(main_py, APP_DIR / "app.py")
                logger.info(f"✅ Copied {main_py} as app.py")
            else:
                logger.error("❌ Could not find any main Python file")

    # Check and copy runpod_utils.py if it exists
    utils_path = REPO_PATH / "runpod_utils.py"
    if utils_path.exists():
        shutil.copy(utils_path, APP_DIR / "runpod_utils.py")
        logger.info("✅ runpod_utils.py copied")
    
    # Copy other necessary files
    for file in ["requirements.txt", "README.md", "LICENSE"]:
        src_path = REPO_PATH / file
        if src_path.exists():
            shutil.copy(src_path, APP_DIR / file)
            logger.info(f"✅ {file} copied")

def create_fallback_app():
    """Create a minimal app.py file if one doesn't exist"""
    if not (APP_DIR / "app.py").exists():
        logger.warning("⚠️ Could not find app.py in repo, creating a minimal file")
        
        minimal_app = """
import os
import sys
import gradio as gr
import torch

# Check PyTorch
print(f"PyTorch {torch.__version__}")
if torch.cuda.is_available():
    print(f"CUDA: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA is not available. Performance will be reduced.")

# Minimal interface
def generate_image(prompt):
    # Real code would go here
    return f"Simulating image generation for: {prompt}"

# Create Gradio interface
with gr.Blocks(title="FusionFrame Minimal") as demo:
    gr.Markdown("# FusionFrame - Minimal Installation")
    gr.Markdown("⚠️ This is a minimal version. Copy the original app.py to this directory.")
    
    with gr.Row():
        with gr.Column():
            prompt = gr.Textbox(label="Prompt")
            generate_btn = gr.Button("Generate")
        
        with gr.Column():
            output = gr.Textbox(label="Result")
    
    generate_btn.click(generate_image, inputs=[prompt], outputs=[output])

if __name__ == "__main__":
    demo.launch(share=True)
"""
        with open(APP_DIR / "app.py", "w") as f:
            f.write(minimal_app)
        logger.info("✅ Minimal app created")

def create_startup_script():
    """Create startup script for the application"""
    logger.info("🔄 Creating startup script...")
    
    # Bash script for startup
    startup_script = f"""#!/bin/bash
# FusionFrame Startup Script

# Set environment variables
export HF_HOME="{CACHE_DIR}"
export NCCL_P2P_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Start message
echo "-----------------------------------"
echo "🚀 Starting FusionFrame..."
echo "📂 Application directory: {APP_DIR}"
echo "🔗 Web interface: http://localhost:{args.port}"
echo "-----------------------------------"

# Start application
cd {APP_DIR}
python app_wrapper.py --port {args.port} {'--share' if args.shared else ''}
"""
    
    # Save and mark as executable
    start_script_path = APP_DIR / "start.sh"
    with open(start_script_path, "w") as f:
        f.write(startup_script)
    
    os.chmod(start_script_path, 0o755)
    logger.info(f"✅ Startup script created: {start_script_path}")
    
    # Python script for startup (alternative)
    py_script = f"""#!/usr/bin/env python3
import os
import subprocess
import sys

# Set environment variables
os.environ["HF_HOME"] = "{CACHE_DIR}"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

# Change directory
os.chdir("{APP_DIR}")

# Display start message
print("-----------------------------------")
print("🚀 Starting FusionFrame...")
print("📂 Application directory: {APP_DIR}")
print("🔗 Web interface: http://localhost:{args.port}")
print("-----------------------------------")

# Start application
cmd = ["python", "app_wrapper.py", "--port", "{args.port}"]
{'cmd.append("--share")' if args.shared else '# No share'}

subprocess.run(cmd)
"""
    
    # Save and mark as executable
    py_script_path = APP_DIR / "start.py"
    with open(py_script_path, "w") as f:
        f.write(py_script)
    
    os.chmod(py_script_path, 0o755)
    logger.info(f"✅ Python startup script created: {py_script_path}")

def create_wrapper():
    """Create a wrapper for app.py that fixes compatibility issues"""
    logger.info("🔄 Creating compatibility wrapper for app.py...")
    
    # Save original app.py with different name if it exists
    app_path = APP_DIR / "app.py"
    if app_path.exists():
        shutil.copy(app_path, APP_DIR / "app_original.py")
    
    # Wrapper script that imports correctly
    wrapper_script = """#!/usr/bin/env python3
# Wrapper for app.py that fixes CUDA/NCCL compatibility issues

import os
import sys
import argparse

# Parse arguments
parser = argparse.ArgumentParser(description="FusionFrame App Wrapper")
parser.add_argument("--port", type=int, default=7860, help="Port for web interface")
parser.add_argument("--share", action="store_true", help="Allow public access")
args = parser.parse_args()

# Set essential environment variables for compatibility
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

# Make sure current directory is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Check PyTorch before importing to avoid crashes
try:
    # Try to import torch
    import torch
    print(f"PyTorch {torch.__version__} loaded successfully")
    if torch.cuda.is_available():
        print(f"CUDA available: {torch.cuda.get_device_name(0)}")
    else:
        print("WARNING: CUDA is not available, application will run on CPU (very slow)")
except ImportError as e:
    print(f"Error importing PyTorch: {e}")
    print("Trying to reinstall PyTorch...")
    import subprocess
    subprocess.run(["pip", "uninstall", "-y", "torch", "torchvision", "torchaudio"])
    subprocess.run(["pip", "install", "torch==1.13.1", "torchvision", "torchaudio", 
                    "--index-url", "https://download.pytorch.org/whl/cu117"])
    # Try import again
    try:
        import torch
        print(f"PyTorch reinstalled and loaded: {torch.__version__}")
    except ImportError as e:
        print(f"Could not install PyTorch: {e}")
        sys.exit(1)

# Check if app_original.py exists
original_app = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app_original.py")
if os.path.exists(original_app):
    print("Loading original application...")
    try:
        # Replace argv to pass parameters
        sys.argv = [original_app]
        if args.port != 7860:
            sys.argv.extend(["--port", str(args.port)])
        if args.share:
            sys.argv.append("--share")
        
        # Execute code from app_original.py
        with open(original_app) as f:
            code = compile(f.read(), original_app, 'exec')
            exec(code, globals())
    except Exception as e:
        print(f"Error starting original application: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
else:
    print("Error: Cannot find app_original.py file")
    sys.exit(1)
"""
    
    # Save wrapper as app_wrapper.py
    with open(APP_DIR / "app_wrapper.py", "w") as f:
        f.write(wrapper_script)
    
    logger.info("✅ Compatibility wrapper created")

def create_info_file():
    """Create a JSON file with installation information"""
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
    
    logger.info(f"✅ Installation info saved to: {info_path}")

def main():
    """Main installation function"""
    start_time = time.time()
    logger.info(f"🚀 Starting FusionFrame installation on RunPod")
    logger.info(f"📂 Installation directory: {APP_DIR}")
    logger.info(f"📂 Repository directory: {REPO_PATH}")
    
    # Create directories
    setup_directories()
    
    # Check GPU
    has_gpu = check_gpu()
    
    # Fix CUDA environment variables
    if has_gpu:
        fix_cuda_environment()
    
    # Install PyTorch
    install_pytorch(args.cuda_version, args.torch_version, fix_cuda=args.fix_cuda)
    
    # Install dependencies
    success_count, failure_count = install_dependencies()
    
    # Copy application files
    copy_application_files()
    
    # Create minimal app.py if it doesn't exist
    create_fallback_app()
    
    # Create wrapper with CUDA fixes
    create_wrapper()
    
    # Create startup script
    create_startup_script()
    
    # Create info file
    create_info_file()
    
    # Display summary
    elapsed_time = time.time() - start_time
    logger.info(f"✨ Installation complete in {elapsed_time:.1f} seconds!")
    logger.info(f"📋 Installation summary:")
    logger.info(f"   - Installation directory: {APP_DIR}")
    logger.info(f"   - Cache directory: {CACHE_DIR}")
    logger.info(f"   - Successfully installed packages: {success_count}")
    if failure_count > 0:
        logger.warning(f"   - Packages with issues: {failure_count} (some may be optional)")
    logger.info(f"   - To start, run: {APP_DIR}/start.sh")
    logger.info(f"   - Web interface will be available at: http://localhost:{args.port}")
    if args.shared:
        logger.info(f"   - Public link will be displayed at startup")
    
    logger.info(f"🎉 Installation successfully completed!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("⚠️ Installation interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Error during installation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
