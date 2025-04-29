import os
import sys
import torch
import gradio as gr
import numpy as np
from PIL import Image, ImageOps
import requests
from io import BytesIO
from safetensors.torch import load_file
from tqdm.auto import tqdm
import time
import gc
import random
from typing import List, Dict, Tuple, Optional, Union, Any
import logging
import tempfile

# Set up basic logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Better storage paths for RunPod environments
if os.path.exists("/workspace"):
    # We're on RunPod
    BASE_DIR = "/workspace/FusionFrame"
    # Explicitly create the directory
    os.makedirs(BASE_DIR, exist_ok=True)
else:
    # We're on a local environment
    BASE_DIR = os.getcwd()

# Configure HuggingFace cache and other environment variables
os.environ["HF_HOME"] = os.path.join(BASE_DIR, "cache")
os.environ["TRANSFORMERS_CACHE"] = os.path.join(BASE_DIR, "cache", "transformers")
os.environ["DIFFUSERS_CACHE"] = os.path.join(BASE_DIR, "cache", "diffusers")

# Configuration - explicitly use paths under workspace for RunPod
CACHE_DIR = os.path.join(BASE_DIR, "cache")
MODELS_DIR = os.path.join(BASE_DIR, "models")
LORA_DIR = os.path.join(BASE_DIR, "loras")
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
CONTROLNET_DIR = os.path.join(BASE_DIR, "controlnet")
IPADAPTER_DIR = os.path.join(BASE_DIR, "ipadapter")
LOGS_DIR = os.path.join(BASE_DIR, "logs")

# Create necessary directories
for directory in [CACHE_DIR, MODELS_DIR, LORA_DIR, UPLOAD_DIR, OUTPUT_DIR, CONTROLNET_DIR, IPADAPTER_DIR, LOGS_DIR]:
    os.makedirs(directory, exist_ok=True)
    logger.info(f"Created directory: {directory}")

# Set up file logging in addition to console logging
file_handler = logging.FileHandler(os.path.join(LOGS_DIR, f"app_{time.strftime('%Y%m%d_%H%M%S')}.log"))
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

# Check if CUDA is available
if torch.cuda.is_available():
    logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
    logger.info(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    logger.warning("CUDA not available. Performance will be severely limited.")

# Show the directories being used
logger.info(f"Using cache directory: {CACHE_DIR}")
logger.info(f"Using models directory: {MODELS_DIR}")
logger.info(f"Using output directory: {OUTPUT_DIR}")

# --- Import AI frameworks with improved error handling ---
try:
    from diffusers import (
        StableDiffusionXLPipeline,
        StableDiffusionPipeline, 
        StableDiffusionXLControlNetPipeline, 
        StableDiffusionControlNetPipeline,
        DDIMScheduler, 
        EulerAncestralDiscreteScheduler, 
        DPMSolverMultistepScheduler,
        AutoencoderKL, 
        ControlNetModel
    )
    from diffusers.pipelines.controlnet import MultiControlNetModel
    from diffusers.utils import logging as diffusers_logging
    
    # Set diffusers logging to INFO to show download progress
    diffusers_logging.set_verbosity_info()
    
    from transformers import (
        CLIPImageProcessor, 
        AutoProcessor, 
        CLIPVisionModelWithProjection
    )
    from huggingface_hub import HfFolder, snapshot_download
    
    # Configure huggingface cache location
    #HfFolder.save_cache_file('cli/cache_dir', CACHE_DIR)
    os.environ['HF_HOME'] = CACHE_DIR
    logger.info("Successfully imported AI frameworks")
except ImportError as e:
    logger.error(f"Error importing AI dependencies: {e}")
    logger.error("Installing missing dependencies...")
    # Auto-install required packages
    os.system("pip install diffusers==0.21.4 transformers==4.30.2 accelerate==0.20.3 safetensors==0.3.1")
    os.system("pip install huggingface_hub==0.16.4")
    
    # Retry imports
    try:
        from diffusers import (
            StableDiffusionXLPipeline,
            StableDiffusionPipeline, 
            StableDiffusionXLControlNetPipeline, 
            StableDiffusionControlNetPipeline,
            DDIMScheduler, 
            EulerAncestralDiscreteScheduler, 
            DPMSolverMultistepScheduler,
            AutoencoderKL, 
            ControlNetModel
        )
        from diffusers.pipelines.controlnet import MultiControlNetModel
        from diffusers.utils import logging as diffusers_logging
        
        # Set diffusers logging to INFO to show download progress
        diffusers_logging.set_verbosity_info()
        
        from transformers import (
            CLIPImageProcessor, 
            AutoProcessor, 
            CLIPVisionModelWithProjection
        )
        logger.info("Successfully installed and imported AI frameworks")
    except ImportError as e:
        logger.error(f"Critical error importing AI dependencies after installation: {e}")
        logger.error("Please run the installer script to set up the environment properly.")
        sys.exit(1)
# --------------------------------------------------------

# Try to import IP-Adapter
try:
    from ip_adapter import IPAdapterPlus, IPAdapterPlusXL
    IP_ADAPTER_AVAILABLE = True
    logger.info("IP-Adapter successfully imported")
except ImportError:
    logger.warning("IP-Adapter not available. Attempting to install...")
    os.system("pip install git+https://github.com/tencent-ailab/IP-Adapter.git@main")
    try:
        from ip_adapter import IPAdapterPlus, IPAdapterPlusXL
        IP_ADAPTER_AVAILABLE = True
        logger.info("IP-Adapter successfully installed and imported")
    except ImportError as e:
        logger.error(f"Could not install IP-Adapter: {e}")
        logger.warning("IP-Adapter features will be disabled")
        IP_ADAPTER_AVAILABLE = False

# Try to import ControlNet preprocessors
try:
    from controlnet_aux import OpenposeDetector, HEDdetector, MidasDetector, LineartDetector
    import cv2
    CONTROLNET_AUX_AVAILABLE = True
    logger.info("ControlNet auxiliaries successfully imported")
except ImportError:
    logger.warning("ControlNet auxiliary modules not available. Attempting to install...")
    os.system("pip install controlnet_aux==0.0.6")
    try:
        from controlnet_aux import OpenposeDetector, HEDdetector, MidasDetector, LineartDetector
        import cv2
        CONTROLNET_AUX_AVAILABLE = True
        logger.info("ControlNet auxiliaries successfully installed and imported")
    except ImportError as e:
        logger.error(f"Could not install ControlNet auxiliaries: {e}")
        logger.warning("ControlNet pre-processing features will be disabled")
        CONTROLNET_AUX_AVAILABLE = False

# Available models with better file path handling
AVAILABLE_MODELS = {
    "SDXL 1.0": "stabilityai/stable-diffusion-xl-base-1.0",
    "SDXL Refiner": "stabilityai/stable-diffusion-xl-refiner-1.0",
    "SD 2.1": "stabilityai/stable-diffusion-2-1",
    "SD 1.5": "runwayml/stable-diffusion-v1-5",  # Smaller, use this for testing
    "Juggernaut XL": "RunDiffusion/Juggernaut-XL-v9",
    "DreamShaper XL": "Lykon/dreamshaper-xl-1-0",
    "RealisticVision XL": "SG161222/RealVisXL_V4.0",
}

SCHEDULER_OPTIONS = {
    "DPM++ 2M Karras": lambda config: DPMSolverMultistepScheduler.from_config(
        config, use_karras_sigmas=True, algorithm_type="dpmsolver++", solver_order=2
    ),
    "DPM++ SDE Karras": lambda config: DPMSolverMultistepScheduler.from_config(
        config, use_karras_sigmas=True, algorithm_type="sde-dpmsolver++", solver_order=2
    ),
    "Euler A": lambda config: EulerAncestralDiscreteScheduler.from_config(config),
    "DDIM": lambda config: DDIMScheduler.from_config(config)
}

VAE_OPTIONS = {
    "Default": None,
    "SDXL VAE": "madebyollin/sdxl-vae-fp16-fix",
    "SD 1.5 VAE": "stabilityai/sd-vae-ft-mse"
}

# ControlNet models
CONTROLNET_MODELS = {
    "None": None,
    "Pose (OpenPose)": "lllyasviel/control_v11p_sd15_openpose",
    "Pose (SDXL)": "thibaud/controlnet-openpose-sdxl-1.0",
    "Canny Edge": "lllyasviel/control_v11p_sd15_canny",
    "Canny Edge (SDXL)": "diffusers/controlnet-canny-sdxl-1.0",
    "Depth": "lllyasviel/control_v11f1p_sd15_depth",
    "Depth (SDXL)": "diffusers/controlnet-depth-sdxl-1.0",
    "Lineart": "lllyasviel/control_v11p_sd15_lineart",
    "Soft Edge": "lllyasviel/control_v11p_sd15_softedge"
}

# IP-Adapter models
IP_ADAPTER_MODELS = {
    "None": None,
    "IP-Adapter Plus (SD 1.5)": {
        "model_id": "h94/IP-Adapter",
        "subfolder": "models",
        "weight_name": "ip-adapter-plus_sd15.bin"
    },
    "IP-Adapter Plus Face (SD 1.5)": {
        "model_id": "h94/IP-Adapter",
        "subfolder": "models",
        "weight_name": "ip-adapter-plus-face_sd15.bin"
    },
    "IP-Adapter Plus (SDXL)": {
        "model_id": "h94/IP-Adapter",
        "subfolder": "sdxl_models",
        "weight_name": "ip-adapter-plus_sdxl_vit-h.bin"
    },
    "IP-Adapter Plus Face (SDXL)": {
        "model_id": "h94/IP-Adapter",
        "subfolder": "sdxl_models",
        "weight_name": "ip-adapter-plus-face_sdxl_vit-h.bin"
    }
}

DEFAULT_LORAS = ["None"]
lora_files = {}  # To store uploaded LoRA files

# Global variables for loaded components with improved tracking
loaded_components = {
    "model": None,
    "model_name": None,
    "vae": None,
    "controlnet": {},
    "ip_adapter": None,
    "ip_adapter_name": None,
    "scheduler": None,
    "scheduler_name": None,
    "download_progress": None
}

# Status message for the UI
global_status_message = "Ready"

def update_status(message):
    """Update the global status message"""
    global global_status_message
    global_status_message = message
    logger.info(message)
    return message

def download_model_with_progress(model_id, local_dir, progress=None):
    """Download a model with progress tracking for the UI"""
    try:
        update_status(f"Downloading model {model_id}...")
        if progress:
            progress(0.1, desc=f"Downloading {model_id}...")
        
        # Use snapshot_download to download the model
        # This function shows progress in the console
        snapshot_path = snapshot_download(
            repo_id=model_id,
            local_dir=os.path.join(local_dir, model_id.split("/")[-1]),
            local_dir_use_symlinks=False
        )
        
        if progress:
            progress(0.9, desc=f"Download of {model_id} complete")
        
        update_status(f"Downloaded model {model_id} to {snapshot_path}")
        return snapshot_path
    except Exception as e:
        error_msg = f"Error downloading model {model_id}: {e}"
        update_status(error_msg)
        logger.error(error_msg)
        return None

def download_file(url, save_path, progress=None):
    """Download a file from a URL and save it locally with progress"""
    try:
        update_status(f"Downloading file from {url}...")
        if progress:
            progress(0.1, desc=f"Downloading file from {url}...")
            
        response = requests.get(url, stream=True)
        total_size_in_bytes = int(response.headers.get('content-length', 0))
        block_size = 1024
        progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
        
        with open(save_path, 'wb') as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)
                if progress and total_size_in_bytes > 0:
                    current_progress = min(0.1 + 0.8 * (progress_bar.n / total_size_in_bytes), 0.9)
                    progress(current_progress, desc=f"Downloading {url.split('/')[-1]}...")
        
        progress_bar.close()
        
        if progress:
            progress(0.95, desc="Download complete")
            
        if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
            error_msg = f"Error downloading file from {url}"
            update_status(error_msg)
            logger.error(error_msg)
            return False
            
        update_status(f"Downloaded file to {save_path}")
        return True
    except Exception as e:
        error_msg = f"Error downloading file: {e}"
        update_status(error_msg)
        logger.error(error_msg)
        return False

def load_vae(vae_name, progress=None):
    """Load and return VAE if specified"""
    if vae_name == "Default" or vae_name is None:
        return None
    
    update_status(f"Loading VAE: {vae_name}")
    if progress:
        progress(0.1, desc=f"Loading VAE: {vae_name}...")
        
    vae_id = VAE_OPTIONS[vae_name]
    try:
        # Set download directory explicitly
        cache_dir = os.path.join(MODELS_DIR, "vae", vae_name.replace(" ", "_").lower())
        os.makedirs(cache_dir, exist_ok=True)
        
        vae = AutoencoderKL.from_pretrained(
            vae_id,
            torch_dtype=torch.float16,
            cache_dir=cache_dir
        )
        
        if progress:
            progress(0.8, desc=f"Moving VAE to GPU...")
            
        if torch.cuda.is_available():
            vae = vae.to("cuda")
            
        if progress:
            progress(0.9, desc=f"VAE loaded successfully")
            
        update_status(f"VAE {vae_name} loaded successfully")
        return vae
    except Exception as e:
        error_msg = f"Error loading VAE: {e}"
        update_status(error_msg)
        logger.error(error_msg)
        return None

def load_controlnet(controlnet_name, model_type="SD", progress=None):
    """Load ControlNet model if specified"""
    if controlnet_name == "None" or controlnet_name is None:
        return None
    
    model_id = CONTROLNET_MODELS[controlnet_name]
    if model_id is None:
        return None
    
    update_status(f"Loading ControlNet: {controlnet_name}")
    if progress:
        progress(0.1, desc=f"Loading ControlNet: {controlnet_name}...")
    
    cache_path = os.path.join(CONTROLNET_DIR, controlnet_name.replace(" ", "_").lower())
    os.makedirs(cache_path, exist_ok=True)
    
    try:
        controlnet = ControlNetModel.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            cache_dir=cache_path
        )
        
        if progress:
            progress(0.8, desc=f"Moving ControlNet to GPU...")
            
        if torch.cuda.is_available():
            controlnet = controlnet.to("cuda")
            
        if progress:
            progress(0.9, desc=f"ControlNet loaded successfully")
            
        update_status(f"ControlNet {controlnet_name} loaded successfully")
        return controlnet
    except Exception as e:
        error_msg = f"Error loading ControlNet: {e}"
        update_status(error_msg)
        logger.error(error_msg)
        return None

def load_ip_adapter(ip_adapter_name, base_model_path, is_xl=False, progress=None):
    """Load IP-Adapter if specified"""
    if not IP_ADAPTER_AVAILABLE or ip_adapter_name == "None" or ip_adapter_name is None:
        return None
    
    adapter_config = IP_ADAPTER_MODELS.get(ip_adapter_name)
    if adapter_config is None:
        return None
    
    update_status(f"Loading IP-Adapter: {ip_adapter_name}")
    if progress:
        progress(0.1, desc=f"Loading IP-Adapter: {ip_adapter_name}...")
    
    cache_path = os.path.join(IPADAPTER_DIR, ip_adapter_name.replace(" ", "_").lower())
    os.makedirs(cache_path, exist_ok=True)
    
    try:
        if is_xl:
            ip_adapter = IPAdapterPlusXL(
                base_model_path,
                adapter_config["model_id"],
                adapter_config["subfolder"],
                adapter_config["weight_name"],
                device="cuda" if torch.cuda.is_available() else "cpu",
                cache_dir=cache_path
            )
        else:
            ip_adapter = IPAdapterPlus(
                base_model_path,
                adapter_config["model_id"],
                adapter_config["subfolder"],
                adapter_config["weight_name"],
                device="cuda" if torch.cuda.is_available() else "cpu",
                cache_dir=cache_path
            )
            
        if progress:
            progress(0.9, desc=f"IP-Adapter loaded successfully")
            
        update_status(f"IP-Adapter {ip_adapter_name} loaded successfully")
        return ip_adapter
    except Exception as e:
        error_msg = f"Error loading IP-Adapter: {e}"
        update_status(error_msg)
        logger.error(error_msg)
        import traceback
        traceback.print_exc()
        return None

def load_model(model_name, scheduler_name="DPM++ 2M Karras", vae_name="Default", 
               controlnet_name="None", ip_adapter_name="None", force_reload=False,
               progress=None):
    """Load the selected model and its components with visible progress"""
    global loaded_components
    
    # Check if model is already loaded and no reload is forced
    if (loaded_components["model"] is not None and 
        loaded_components["model_name"] == model_name and 
        loaded_components["scheduler_name"] == scheduler_name and
        not force_reload):
        update_status(f"Using already loaded model: {model_name}")
        return loaded_components["model"]
    
    # Free memory from previously loaded model
    if loaded_components["model"] is not None:
        update_status(f"Unloading previous model: {loaded_components['model_name']}")
        if progress:
            progress(0.05, desc="Clearing GPU memory...")
            
        del loaded_components["model"]
        if loaded_components["vae"] is not None:
            del loaded_components["vae"]
        for cn_name in loaded_components["controlnet"]:
            if loaded_components["controlnet"][cn_name] is not None:
                del loaded_components["controlnet"][cn_name]
        if loaded_components["ip_adapter"] is not None:
            del loaded_components["ip_adapter"]
        loaded_components["controlnet"] = {}
        torch.cuda.empty_cache()
        gc.collect()
    
    model_id = AVAILABLE_MODELS[model_name]
    is_xl = "XL" in model_name
    
    update_status(f"Loading model: {model_name} ({model_id})")
    if progress:
        progress(0.1, desc=f"Preparing to load {model_name}...")
    
    # Set explicit model cache directory
    model_cache_dir = os.path.join(MODELS_DIR, model_name.replace(" ", "_").lower())
    os.makedirs(model_cache_dir, exist_ok=True)
    
    # Load VAE first if specified
    if progress:
        progress(0.15, desc=f"Loading VAE...")
    vae = load_vae(vae_name, progress)
    loaded_components["vae"] = vae
    
    # Load controlnet if specified
    if controlnet_name != "None" and progress:
        progress(0.2, desc=f"Loading ControlNet...")
    controlnet = None
    if controlnet_name != "None":
        controlnet = load_controlnet(controlnet_name, "SDXL" if is_xl else "SD", progress)
        if controlnet:
            loaded_components["controlnet"][controlnet_name] = controlnet
    
    # Initialize the appropriate pipeline based on model type
    try:
        if progress:
            progress(0.3, desc=f"Downloading and loading {model_name}...")
        
        if is_xl:
            if controlnet:
                pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
                    model_id,
                    controlnet=controlnet,
                    vae=vae,
                    torch_dtype=torch.float16,
                    use_safetensors=True,
                    variant="fp16",
                    cache_dir=model_cache_dir
                )
            else:
                pipe = StableDiffusionXLPipeline.from_pretrained(
                    model_id,
                    vae=vae,
                    torch_dtype=torch.float16,
                    use_safetensors=True,
                    variant="fp16",
                    cache_dir=model_cache_dir
                )
        else:
            if controlnet:
                pipe = StableDiffusionControlNetPipeline.from_pretrained(
                    model_id,
                    controlnet=controlnet,
                    vae=vae,
                    torch_dtype=torch.float16,
                    safety_checker=None,
                    requires_safety_checker=False,
                    cache_dir=model_cache_dir
                )
            else:
                pipe = StableDiffusionPipeline.from_pretrained(
                    model_id,
                    vae=vae,
                    torch_dtype=torch.float16,
                    safety_checker=None,
                    requires_safety_checker=False,
                    cache_dir=model_cache_dir
                )
        
        if progress:
            progress(0.6, desc=f"Setting up scheduler...")
        
        # Set up scheduler
        scheduler_fn = SCHEDULER_OPTIONS.get(scheduler_name)
        if scheduler_fn:
            pipe.scheduler = scheduler_fn(pipe.scheduler.config)
        
        if progress:
            progress(0.7, desc=f"Moving model to GPU...")
        
        # Move to GPU if available
        if torch.cuda.is_available():
            pipe = pipe.to("cuda")
            pipe.enable_attention_slicing()
            
        if progress:
            progress(0.8, desc=f"Optimizing memory usage...")
            
        # Optional memory optimization
        if torch.cuda.is_available() and hasattr(pipe, "enable_model_cpu_offload"):
            pipe.enable_model_cpu_offload()
        
        # Load IP-Adapter if specified
        if ip_adapter_name != "None" and progress:
            progress(0.85, desc=f"Loading IP-Adapter...")
            
        if ip_adapter_name != "None":
            ip_adapter = load_ip_adapter(ip_adapter_name, model_id, is_xl, progress)
            loaded_components["ip_adapter"] = ip_adapter
            loaded_components["ip_adapter_name"] = ip_adapter_name
        
        # Update loaded components
        loaded_components["model"] = pipe
        loaded_components["model_name"] = model_name
        loaded_components["scheduler_name"] = scheduler_name
        
        if progress:
            progress(0.95, desc=f"Model loaded successfully!")
            
        update_status(f"Model {model_name} loaded successfully")
        return pipe
    
    except Exception as e:
        error_msg = f"Error loading model: {e}"
        update_status(error_msg)
        logger.error(error_msg)
        import traceback
        traceback.print_exc()
        return None

def save_uploaded_file(file):
    """Save uploaded file and return the path"""
    if file is None:
        return None
    
    filename = os.path.join(UPLOAD_DIR, file.name)
    with open(filename, "wb") as f:
        f.write(file.read())
    
    update_status(f"Saved uploaded file: {filename}")
    return filename

def save_uploaded_lora(file):
    """Save uploaded LoRA file and return the updated dropdown"""
    if file is None:
        return gr.Dropdown(choices=DEFAULT_LORAS)
    
    filename = os.path.join(LORA_DIR, file.name)
    with open(filename, "wb") as f:
        f.write(file.read())
    
    # Add to global LoRA files dict
    lora_files[file.name] = filename
    
    # Update dropdown choices
    updated_choices = DEFAULT_LORAS + list(lora_files.keys())
    
    update_status(f"Saved uploaded LoRA: {filename}")
    return gr.Dropdown(choices=updated_choices)

def update_lora_dropdowns(lora_file):
    """Update all LoRA dropdowns when a new LoRA is uploaded"""
    if lora_file is None:
        return [gr.Dropdown(choices=DEFAULT_LORAS) for _ in range(5)]
    
    filename = os.path.join(LORA_DIR, lora_file.name)
    with open(filename, "wb") as f:
        f.write(lora_file.read())
    
    # Add to global LoRA files dict
    lora_files[lora_file.name] = filename
    
    # Update dropdown choices
    updated_choices = DEFAULT_LORAS + list(lora_files.keys())
    
    update_status(f"Updated LoRA dropdowns with: {lora_file.name}")
    return [gr.Dropdown(choices=updated_choices) for _ in range(5)]

def load_lora_weights(pipe, lora_file_path, lora_weight=0.7):
    """Load LoRA weights into the model pipe"""
    if lora_file_path is None or lora_file_path == "None":
        return pipe
    
    update_status(f"Loading LoRA: {lora_file_path} with weight {lora_weight}")
    
    try:
        # Different loading method based on model type
        if hasattr(pipe, "unet") and hasattr(pipe.unet, "load_attn_procs"):
            pipe.load_lora_weights(lora_file_path)
            
            # Apply LoRA weight scaling if the method exists
            if hasattr(pipe, "set_adapters"):
                pipe.set_adapters(["default"], adapter_weights=[lora_weight])
                
        update_status(f"LoRA {lora_file_path} loaded successfully")
    except Exception as e:
        error_msg = f"Error loading LoRA: {e}"
        update_status(error_msg)
        logger.error(error_msg)
    
    return pipe

def prepare_image(img: Optional[Image.Image], target_size=(768,768)):
    if img is None:
        return None
    # Ensure RGB
    if img.mode == "RGBA":
        bg = Image.new("RGB", img.size, (255,255,255))
        bg.paste(img, mask=img)
        img = bg
    # Resize while keeping aspect
    img.thumbnail(target_size, Image.Resampling.LANCZOS)
    # Center on white canvas
    canvas = Image.new("RGB", target_size, (255,255,255))
    offset = ((target_size[0] - img.width)//2, (target_size[1] - img.height)//2)
    canvas.paste(img, offset)
    return canvas


def generate_controlnet_conditioning(image, controlnet_type, progress=None):
    """Generate conditioning image for ControlNet"""
    if not CONTROLNET_AUX_AVAILABLE or image is None or controlnet_type == "None":
        return None
    
    try:
        if progress:
            progress(0.1, desc=f"Preparing {controlnet_type} conditioning...")
            
        # Convert PIL image to numpy array
        numpy_image = np.array(image)
        
        if "Pose" in controlnet_type:
            update_status("Generating OpenPose conditioning...")
            if progress:
                progress(0.2, desc="Loading OpenPose detector...")
                
            openpose = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
            
            if progress:
                progress(0.5, desc="Applying OpenPose detection...")
                
            result = openpose(numpy_image)
            
            if progress:
                progress(0.9, desc="OpenPose conditioning complete")
                
            return result
        
        elif "Canny Edge" in controlnet_type:
            update_status("Generating Canny Edge conditioning...")
            if progress:
                progress(0.3, desc="Applying Canny edge detection...")
                
            # Apply Canny edge detection
            low_threshold = 100
            high_threshold = 200
            image_gray = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(image_gray, low_threshold, high_threshold)
            edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
            
            if progress:
                progress(0.9, desc="Canny edge detection complete")
                
            return Image.fromarray(edges_colored)
        
        elif "Depth" in controlnet_type:
            update_status("Generating Depth conditioning...")
            if progress:
                progress(0.2, desc="Loading depth estimator...")
                
            depth_estimator = MidasDetector.from_pretrained("lllyasviel/ControlNet")
            
            if progress:
                progress(0.5, desc="Calculating depth map...")
                
            result = depth_estimator(numpy_image)
            
            if progress:
                progress(0.9, desc="Depth estimation complete")
                
            return result
        
        elif "Lineart" in controlnet_type:
            update_status("Generating Lineart conditioning...")
            if progress:
                progress(0.2, desc="Loading lineart detector...")
                
            lineart = LineartDetector.from_pretrained("lllyasviel/ControlNet")
            
            if progress:
                progress(0.5, desc="Creating lineart...")
                
            result = lineart(numpy_image)
            
            if progress:
                progress(0.9, desc="Lineart generation complete")
                
            return result
        
        elif "Soft Edge" in controlnet_type:
            update_status("Generating Soft Edge conditioning...")
            if progress:
                progress(0.2, desc="Loading HED detector...")
                
            hed = HEDdetector.from_pretrained("lllyasviel/ControlNet")
            
            if progress:
                progress(0.5, desc="Detecting edges...")
                
            result = hed(numpy_image)
            
            if progress:
                progress(0.9, desc="Soft edge detection complete")
                
            return result
        
        else:
            return None
    
    except Exception as e:
        error_msg = f"Error generating controlnet conditioning: {e}"
        update_status(error_msg)
        logger.error(error_msg)
        return None

def generate_images(
    model_name,
    woman_image,
    clothing_image,
    background_image,
    positive_prompt,
    negative_prompt,
    lora1, lora1_weight,
    lora2, lora2_weight,
    lora3, lora3_weight,
    lora4, lora4_weight,
    lora5, lora5_weight,
    controlnet_type,
    controlnet_conditioning_scale,
    ip_adapter_name,
    ip_adapter_scale,
    denoising_strength,
    num_inference_steps,
    guidance_scale,
    image_width,
    image_height,
    scheduler_name,
    vae_name,
    num_outputs,
    seed,
    progress=gr.Progress()
):
    """Generate images based on the provided inputs with visible progress"""
    try:
        # Track overall progress
        progress(0, desc="Initializing...")
        update_status("Starting image generation...")
        
        # Print model info to debug downloaded models
        logger.info(f"Checking model directories:")
        logger.info(f"MODELS_DIR: {os.listdir(MODELS_DIR) if os.path.exists(MODELS_DIR) else 'Directory does not exist'}")
        logger.info(f"CACHE_DIR: {os.listdir(CACHE_DIR) if os.path.exists(CACHE_DIR) else 'Directory does not exist'}")
        
        # Load the model with progress updates
        pipe = load_model(
            model_name, 
            scheduler_name=scheduler_name, 
            vae_name=vae_name, 
            controlnet_name=controlnet_type,
            ip_adapter_name=ip_adapter_name,
            progress=progress
        )
        
        if pipe is None:
            error_msg = "Failed to load model. Check logs for details."
            update_status(error_msg)
            return [], seed, error_msg
        
        # Apply seed for reproducibility
        progress(0.35, desc="Setting up generation parameters...")
        if seed == -1:
            seed = torch.randint(0, 2**32, (1,)).item()
            update_status(f"Using random seed: {seed}")
        else:
            update_status(f"Using specified seed: {seed}")
        
        generator = torch.Generator("cuda").manual_seed(seed)
        
        # Prepare inputs
        has_woman = woman_image is not None
        has_clothing = clothing_image is not None
        has_background = background_image is not None
        
        # Convert uploaded images to PIL and prepare them
        progress(0.4, desc="Processing input images...")
        update_status("Processing input images...")
        
        woman_img = prepare_image(woman_image) if has_woman else None
        clothing_img = prepare_image(clothing_image) if has_clothing else None
        background_img = prepare_image(background_image) if has_background else None
        
        # Enhanced prompt based on inputs
        base_prompt = positive_prompt.strip() if positive_prompt else ""
        
        # Add more context based on available images
        if has_woman and has_clothing and has_background:
            enhanced_prompt = f"{base_prompt}, woman wearing the clothing, in the background scene"
        elif has_woman and has_clothing:
            enhanced_prompt = f"{base_prompt}, woman wearing the clothing"
        elif has_woman and has_background:
            enhanced_prompt = f"{base_prompt}, woman in the background scene"
        elif has_woman:
            enhanced_prompt = f"{base_prompt}, woman"
        else:
            enhanced_prompt = base_prompt
        
        logger.info(f"Enhanced prompt: {enhanced_prompt}")
        logger.info(f"Negative prompt: {negative_prompt}")
        update_status(f"Using prompt: {enhanced_prompt}")
        
        # Load LoRAs
        progress(0.45, desc="Loading LoRAs...")
        loras = [
            (lora1, lora1_weight),
            (lora2, lora2_weight),
            (lora3, lora3_weight),
            (lora4, lora4_weight),
            (lora5, lora5_weight)
        ]
        
        # Apply each selected LoRA
        for lora_name, lora_weight in loras:
            if lora_name and lora_name != "None":
                lora_path = lora_files.get(lora_name)
                if lora_path:
                    pipe = load_lora_weights(pipe, lora_path, lora_weight)
        
        # Prepare ControlNet input if needed
        progress(0.5, desc="Preparing ControlNet conditioning...")
        controlnet_image = None
        if controlnet_type != "None" and woman_img is not None:
            update_status(f"Generating {controlnet_type} conditioning...")
            controlnet_image = generate_controlnet_conditioning(woman_img, controlnet_type, progress)
            if controlnet_image:
                update_status(f"ControlNet conditioning generated successfully")
            else:
                update_status(f"Failed to generate ControlNet conditioning, proceeding without it")
        
        # Prepare IP-Adapter input if needed
        if ip_adapter_name != "None" and loaded_components["ip_adapter"] is not None:
            progress(0.55, desc="Preparing IP-Adapter...")
            update_status(f"Setting up IP-Adapter with scale {ip_adapter_scale}...")
            
            ip_adapter = loaded_components["ip_adapter"]
            
            # Determine which image to use for IP-Adapter
            reference_image = None
            if has_woman:
                reference_image = woman_img
                update_status("Using woman image for IP-Adapter")
            elif has_clothing:
                reference_image = clothing_img
                update_status("Using clothing image for IP-Adapter")
            elif has_background:
                reference_image = background_img
                update_status("Using background image for IP-Adapter")
            
            if reference_image:
                ip_adapter.set_ip_adapter_scale(ip_adapter_scale)
        
        # Generate images
        progress(0.6, desc="Generating images...")
        update_status(f"Starting generation of {num_outputs} images...")
        
        outputs = []
        
        # We'll use img2img if any of the input images are provided
        use_img2img = has_woman or has_clothing or has_background
        
        # Determine if we're using an XL model
        is_xl = "XL" in model_name
        
        for i in range(num_outputs):
            progress(0.6 + (0.3 * i / num_outputs), desc=f"Generating image {i+1}/{num_outputs}")
            update_status(f"Generating image {i+1}/{num_outputs}...")
            
            # Configure generation parameters
            generation_params = {
                "prompt": enhanced_prompt,
                "negative_prompt": negative_prompt,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "generator": generator,
            }
            
            # Add width and height for text-to-image
            if not use_img2img:
                generation_params.update({
                    "width": image_width,
                    "height": image_height,
                })
            
            # Add ControlNet parameters if applicable
            if controlnet_type != "None" and controlnet_image is not None:
                generation_params.update({
                    "image": controlnet_image,
                    "controlnet_conditioning_scale": controlnet_conditioning_scale,
                })
            
            # Add IP-Adapter parameters if applicable
            if ip_adapter_name != "None" and reference_image is not None and loaded_components["ip_adapter"] is not None:
                # IP-Adapter generates inside the modified pipeline
                update_status(f"Using IP-Adapter for generation...")
                try:
                    result = ip_adapter.generate(
                        pil_image=reference_image,
                        prompt=enhanced_prompt,
                        negative_prompt=negative_prompt,
                        num_samples=1,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        width=image_width,
                        height=image_height,
                        generator=generator,
                    )
                    outputs.append(result[0])
                    continue
                except Exception as e:
                    error_msg = f"Error using IP-Adapter: {e}. Falling back to standard generation."
                    update_status(error_msg)
                    logger.error(error_msg)
            
            # Standard generation when no IP-Adapter is used or IP-Adapter failed
            update_status(f"Running standard diffusion process...")
            result = pipe(**generation_params)
            outputs.extend(result.images)
        
        # Save generated images
        progress(0.95, desc="Saving generated images...")
        update_status("Saving generated images...")
        
        saved_paths = []
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        for i, img in enumerate(outputs):
            output_path = os.path.join(OUTPUT_DIR, f"generated_{timestamp}_{i}_{seed}.png")
            img.save(output_path)
            saved_paths.append(output_path)
            logger.info(f"Saved image to {output_path}")
        
        # Clear LoRAs from pipeline for next run
        progress(0.99, desc="Cleaning up...")
        update_status("Cleaning up...")
        
        if hasattr(pipe, "unload_lora_weights"):
            pipe.unload_lora_weights()
        
        success_message = f"Successfully generated {len(saved_paths)} images with seed {seed}"
        update_status(success_message)
        return saved_paths, seed, success_message
        
    except Exception as e:
        error_msg = f"Error generating images: {e}"
        update_status(error_msg)
        logger.error(error_msg)
        import traceback
        traceback.print_exc()
        return [], seed, error_msg

def create_model_info():
    """Create model information markdown"""
    info = "## Model Information\n\n"
    
    # List available models
    info += "### Available Models\n"
    for name, model_id in AVAILABLE_MODELS.items():
        info += f"- **{name}**: `{model_id}`\n"
    
    # List available schedulers
    info += "\n### Available Schedulers\n"
    for name in SCHEDULER_OPTIONS.keys():
        info += f"- {name}\n"
    
    # List ControlNet models
    info += "\n### Available ControlNet Models\n"
    for name, model_id in CONTROLNET_MODELS.items():
        if model_id:
            info += f"- **{name}**: `{model_id}`\n"
    
    # List IP-Adapter models if available
    if IP_ADAPTER_AVAILABLE:
        info += "\n### Available IP-Adapter Models\n"
        for name, config in IP_ADAPTER_MODELS.items():
            if config:
                info += f"- **{name}**\n"
    
    # Add model storage information
    info += f"\n### Storage Information\n"
    info += f"- Models are stored in: `{MODELS_DIR}`\n"
    info += f"- Cache directory: `{CACHE_DIR}`\n"
    info += f"- Output images: `{OUTPUT_DIR}`\n"
    
    return info

def display_status():
    """Function to display the current status message"""
    return global_status_message

def create_interface():
    """Create the Gradio interface with improved status display"""
    
    with gr.Blocks(title="Clothing Try-On Image Generator", theme=gr.themes.Soft()) as app:
        gr.Markdown("# 👗 Platformă Avansată de Generare Imagini: Îmbrăcare + Integrare în Fundal")
        gr.Markdown("Încărcați imagini pentru a genera combinații între model, îmbrăcăminte și fundal, cu setări personalizate.")
        
        # Add status display at the top
        status_display = gr.Textbox(
            label="Status",
            value="Ready to start. Select a model and click Generate to download models on first use.",
            interactive=False
        )
        
        # Setup periodic status refresh
        status_display.change(display_status, inputs=None, outputs=[status_display])
        
        with gr.Tab("Generare Imagini"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 📸 1. Încărcare Imagini")
                    
                    woman_image    = gr.Image(type="pil", label="Imagine Model")
                    clothing_image = gr.Image(type="pil", label="Imagine Îmbrăcăminte (Opțional)")
                    background_image = gr.Image(type="pil", label="Imagine Fundal (Opțional)")
                    
                    gr.Markdown("### 🧩 2. LoRA Management")
                    lora_upload = gr.File(label="Încarcă Fișier LoRA (.safetensors)", file_types=[".safetensors"])
                    
                with gr.Column(scale=1):
                    gr.Markdown("### 📝 3. Prompt Settings")
                    
                    positive_prompt = gr.Textbox(
                        label="Prompt Pozitiv",
                        placeholder="Descrieți ce doriți în imaginea generată",
                        lines=3,
                        value="high quality, photorealistic, masterpiece, best quality, intricate details, professional photo, realistic, detailed clothing"
                    )
                    
                    negative_prompt = gr.Textbox(
                        label="Prompt Negativ", 
                        placeholder="Descrieți ce doriți să evitați în imaginea generată",
                        value="deformed, bad anatomy, disfigured, poorly drawn face, mutation, mutated, extra limb, ugly, poorly drawn hands, missing limb, floating limbs, disconnected limbs, malformed hands, blurry, ((((mutated hands and fingers)))), watermark, watermarked, oversaturated, censored, distorted hands, amputation, missing hands, obese, doubled face, double hands, (((nude)))",
                        lines=2
                    )
                    
                    with gr.Row():
                        with gr.Column(scale=1):
                            model_selector = gr.Dropdown(
                                label="Model",
                                choices=list(AVAILABLE_MODELS.keys()),
                                value="SD 1.5"  # Starting with SD 1.5 as it's smaller for testing
                            )
                            
                            vae_selector = gr.Dropdown(
                                label="VAE",
                                choices=list(VAE_OPTIONS.keys()),
                                value="Default"
                            )
                        
                        with gr.Column(scale=1):
                            scheduler_selector = gr.Dropdown(
                                label="Scheduler",
                                choices=list(SCHEDULER_OPTIONS.keys()),
                                value="DPM++ 2M Karras"
                            )
                            
                            seed = gr.Number(
                                label="Seed (-1 pentru aleatoriu)",
                                value=-1,
                                precision=0
                            )
                
                with gr.Column(scale=1):
                    gr.Markdown("### 🧠 4. Setări Avansate")
                    
                    with gr.Accordion("ControlNet", open=False):
                        controlnet_selector = gr.Dropdown(
                            label="Tip ControlNet",
                            choices=list(CONTROLNET_MODELS.keys()),
                            value="None"
                        )
                        
                        controlnet_scale = gr.Slider(
                            label="ControlNet Conditioning Scale",
                            minimum=0.0,
                            maximum=1.0,
                            value=0.5,
                            step=0.05
                        )
                    
                    with gr.Accordion("IP-Adapter", open=False):
                        ip_adapter_selector = gr.Dropdown(
                            label="IP-Adapter Model",
                            choices=list(IP_ADAPTER_MODELS.keys()),
                            value="None"
                        )
                        
                        ip_adapter_scale = gr.Slider(
                            label="IP-Adapter Scale",
                            minimum=0.0,
                            maximum=1.0,
                            value=0.6,
                            step=0.05
                        )
                    
                    with gr.Accordion("LoRA", open=False):
                        lora1 = gr.Dropdown(label="LoRA 1", choices=DEFAULT_LORAS, value="None")
                        lora1_weight = gr.Slider(label="Weight", minimum=0.0, maximum=1.0, value=0.7, step=0.05)
                        
                        lora2 = gr.Dropdown(label="LoRA 2", choices=DEFAULT_LORAS, value="None")
                        lora2_weight = gr.Slider(label="Weight", minimum=0.0, maximum=1.0, value=0.7, step=0.05)
                        
                        lora3 = gr.Dropdown(label="LoRA 3", choices=DEFAULT_LORAS, value="None")
                        lora3_weight = gr.Slider(label="Weight", minimum=0.0, maximum=1.0, value=0.7, step=0.05)
                        
                        lora4 = gr.Dropdown(label="LoRA 4", choices=DEFAULT_LORAS, value="None")
                        lora4_weight = gr.Slider(label="Weight", minimum=0.0, maximum=1.0, value=0.7, step=0.05)
                        
                        lora5 = gr.Dropdown(label="LoRA 5", choices=DEFAULT_LORAS, value="None")
                        lora5_weight = gr.Slider(label="Weight", minimum=0.0, maximum=1.0, value=0.7, step=0.05)
            
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### ⚙️ 5. Setări Generare")
                    
                    with gr.Row():
                        with gr.Column(scale=1):
                            denoising_strength = gr.Slider(
                                label="Denoising Strength",
                                minimum=0.0,
                                maximum=1.0,
                                value=0.75,
                                step=0.05
                            )
                            
                            num_inference_steps = gr.Slider(
                                label="Inference Steps",
                                minimum=20,
                                maximum=100,
                                value=30,
                                step=5
                            )
                        
                        with gr.Column(scale=1):
                            guidance_scale = gr.Slider(
                                label="Guidance Scale",
                                minimum=1.0,
                                maximum=20.0,
                                value=7.5,
                                step=0.5
                            )
                            
                            num_outputs = gr.Slider(
                                label="Număr Imagini",
                                minimum=1,
                                maximum=4,
                                value=1,
                                step=1
                            )
                    
                    with gr.Row():
                        image_width = gr.Slider(
                            label="Lățime Imagine",
                            minimum=512,
                            maximum=1024,
                            value=512,  # Lower default for faster testing
                            step=64
                        )
                        
                        image_height = gr.Slider(
                            label="Înălțime Imagine",
                            minimum=512,
                            maximum=1024,
                            value=512,  # Lower default for faster testing
                            step=64
                        )
                    
                    with gr.Row():
                        generate_button = gr.Button("🚀 Generează Imagini", variant="primary")
                        clear_button = gr.Button("🧹 Resetează", variant="secondary")
                
                with gr.Column(scale=2):
                    with gr.Row():
                        output_gallery = gr.Gallery(
                            label="Imagini Generate",
                            show_label=True,
                            elem_id="gallery",
                            height=500
                        )
                        
                    with gr.Row():
                        output_seed = gr.Number(label="Seed Utilizat", interactive=False)
                        output_message = gr.Textbox(label="Mesaj Status", interactive=False)
        
        with gr.Tab("Informații"):
            gr.Markdown(create_model_info())
            
            # Add a section showing downloaded models
            gr.Markdown("### Downloaded Models")
            downloaded_models = gr.Textbox(
                label="List of downloaded models",
                value="No models downloaded yet. Models will be downloaded on first use.",
                lines=10
            )
            
            # Button to refresh list of downloaded models
            refresh_models_btn = gr.Button("Refresh Downloaded Models List")
            
            # Function to list downloaded models
            def list_downloaded_models():
                try:
                    models_list = "Downloaded Models:\n\n"
                    
                    # Check models directory
                    if os.path.exists(MODELS_DIR):
                        models_list += "Models directory contents:\n"
                        for item in os.listdir(MODELS_DIR):
                            item_path = os.path.join(MODELS_DIR, item)
                            if os.path.isdir(item_path):
                                size = sum(os.path.getsize(os.path.join(dirpath, filename)) 
                                        for dirpath, _, filenames in os.walk(item_path)
                                        for filename in filenames)
                                models_list += f"- {item}: {size / (1024*1024*1024):.2f} GB\n"
                    else:
                        models_list += "Models directory does not exist yet.\n\n"
                    
                    # Check cache directory
                    if os.path.exists(CACHE_DIR):
                        models_list += "\nCache directory contents:\n"
                        for item in os.listdir(CACHE_DIR):
                            item_path = os.path.join(CACHE_DIR, item)
                            if os.path.isdir(item_path):
                                size = sum(os.path.getsize(os.path.join(dirpath, filename)) 
                                        for dirpath, _, filenames in os.walk(item_path)
                                        for filename in filenames)
                                models_list += f"- {item}: {size / (1024*1024*1024):.2f} GB\n"
                    else:
                        models_list += "Cache directory does not exist yet.\n"
                    
                    return models_list
                except Exception as e:
                    return f"Error listing models: {e}"
            
            refresh_models_btn.click(list_downloaded_models, inputs=None, outputs=[downloaded_models])
        
        # Set up event handlers
        # woman_image.change(save_uploaded_file, inputs=[woman_image], outputs=[woman_image])
        # clothing_image.change(save_uploaded_file, inputs=[clothing_image], outputs=[clothing_image])
        # background_image.change(save_uploaded_file, inputs=[background_image], outputs=[background_image])
        
        # LoRA uploads should update all LoRA dropdowns
        lora_upload.change(
            save_uploaded_lora,
            inputs=[lora_upload],
            outputs=[lora1]
        ).then(
            lambda x: gr.update(choices=DEFAULT_LORAS + list(lora_files.keys())),
            inputs=None,
            outputs=[lora2]
        ).then(
            lambda x: gr.update(choices=DEFAULT_LORAS + list(lora_files.keys())),
            inputs=None,
            outputs=[lora3]
        ).then(
            lambda x: gr.update(choices=DEFAULT_LORAS + list(lora_files.keys())),
            inputs=None,
            outputs=[lora4]
        ).then(
            lambda x: gr.update(choices=DEFAULT_LORAS + list(lora_files.keys())),
            inputs=None,
            outputs=[lora5]
        )
        
        # Clear function
        def clear_inputs():
            return [
                None, None, None,  # Images
                "high quality, photorealistic, masterpiece, best quality, intricate details, professional photo, realistic, detailed clothing",  # Positive prompt
                "deformed, bad anatomy, disfigured, poorly drawn face, mutation, mutated, extra limb, ugly, poorly drawn hands, missing limb, floating limbs, disconnected limbs, malformed hands, blurry, ((((mutated hands and fingers)))), watermark, watermarked, oversaturated, censored, distorted hands, amputation, missing hands, obese, doubled face, double hands, (((nude)))",  # Negative prompt
                "None", 0.7, "None", 0.7, "None", 0.7, "None", 0.7, "None", 0.7,  # LoRAs
                "None", 0.5, "None", 0.6,  # ControlNet & IP-Adapter
                0.75, 30, 7.5, 512, 512, 1, -1,  # Generation params
                [], -1, ""  # Outputs
            ]
        
        clear_button.click(
            clear_inputs,
            inputs=None,
            outputs=[
                woman_image, clothing_image, background_image,
                positive_prompt, negative_prompt,
                lora1, lora1_weight, lora2, lora2_weight, lora3, lora3_weight, 
                lora4, lora4_weight, lora5, lora5_weight,
                controlnet_selector, controlnet_scale, ip_adapter_selector, ip_adapter_scale,
                denoising_strength, num_inference_steps, guidance_scale, 
                image_width, image_height, num_outputs, seed,
                output_gallery, output_seed, output_message
            ]
        )
        
        # Update status display function
        def update_status_display():
            return global_status_message
        
        # Generation process with status update
        def generate_with_status_updates(*args):
            # First update status
            status_display.update(value="Starting generation process...")
            
            # Then call the generate_images function
            results = generate_images(*args)
            
            # Update status again
            status_display.update(value=global_status_message)
            
            return results
        
        generate_button.click(
            generate_with_status_updates,
            inputs=[
                model_selector,
                woman_image,
                clothing_image,
                background_image,
                positive_prompt,
                negative_prompt,
                lora1, lora1_weight,
                lora2, lora2_weight,
                lora3, lora3_weight,
                lora4, lora4_weight,
                lora5, lora5_weight,
                controlnet_selector,
                controlnet_scale,
                ip_adapter_selector,
                ip_adapter_scale,
                denoising_strength,
                num_inference_steps,
                guidance_scale,
                image_width,
                image_height,
                scheduler_selector,
                vae_selector,
                num_outputs,
                seed
            ],
            outputs=[output_gallery, output_seed, output_message]
        )
        
        # Update status display periodically
        gr.on(
            triggers=[gr.deps.Every(5)],
            fn=update_status_display,
            inputs=None,
            outputs=status_display,
        )
    
    return app

# Function to download and preload models for testing
def download_test_model():
    """Download a small test model for initial setup verification"""
    # Verifică dacă modelul există deja
    test_model_path = os.path.join(MODELS_DIR, "test_model")
    if os.path.exists(test_model_path) and os.listdir(test_model_path):
        logger.info("Model deja downloadat. Sărim peste download.")
        return {
            "model_path": test_model_path,
            "test_image_path": os.path.join(OUTPUT_DIR, "test_model_generation.png"),
            "success": True
        }
    
    try:
        # Select a small, lightweight model for testing
        test_model_id = "runwayml/stable-diffusion-v1-5"
        
        # Create model download directory
        os.makedirs(MODELS_DIR, exist_ok=True)
        
        # Use snapshot_download for comprehensive model download
        update_status(f"Downloading test model: {test_model_id}")
        
        model_path = snapshot_download(
            repo_id=test_model_id,
            local_dir=os.path.join(MODELS_DIR, "test_model"),
            local_dir_use_symlinks=False,
            resume_download=True,
            cache_dir=CACHE_DIR
        )
        
        # Verify basic model loading
        update_status("Verifying model loading...")
        pipe = StableDiffusionPipeline.from_pretrained(
            model_path, 
            torch_dtype=torch.float16
        )
        
        # Move to GPU if available
        if torch.cuda.is_available():
            pipe = pipe.to("cuda")
        
        # Simple test generation to validate model
        generator = torch.Generator("cuda").manual_seed(42)
        test_prompt = "a small cat sitting on a couch"
        
        result = pipe(
            prompt=test_prompt, 
            num_inference_steps=20, 
            guidance_scale=7.5,
            generator=generator,
            num_images_per_prompt=1
        )
        
        # Save test image
        test_image_path = os.path.join(OUTPUT_DIR, "test_model_generation.png")
        result.images[0].save(test_image_path)
        
        update_status(f"Test model downloaded and verified successfully. Test image saved to {test_image_path}")
        
        return {
            "model_path": model_path,
            "test_image_path": test_image_path,
            "success": True
        }
    
    except Exception as e:
        error_msg = f"Error downloading test model: {e}"
        update_status(error_msg)
        logger.error(error_msg)
        
        return {
            "model_path": None,
            "test_image_path": None,
            "success": False,
            "error": str(e)
        }

def main_test_model_download():
    """Main function to test model download"""
    result = download_test_model()
    
    if result["success"]:
        print("Test model download successful!")
        print(f"Model path: {result['model_path']}")
        print(f"Test image path: {result['test_image_path']}")
    else:
        print("Test model download failed.")
        print(f"Error: {result.get('error', 'Unknown error')}")

if __name__ == "__main__":
    # Parsare argumente pentru port și share
    import argparse
    
    parser = argparse.ArgumentParser(description="FusionFrame App")
    parser.add_argument("--port", type=int, default=7860, help="Port pentru interfața web")
    parser.add_argument("--share", action="store_true", help="Permite share public")
    args = parser.parse_args()

    # Creează și lansează interfața
    try:
        app = create_interface()
        app.launch(
            server_name='0.0.0.0', 
            port=args.port, 
            share=args.share
        )
    except Exception as e:
        logger.error(f"Eroare la lansarea aplicației: {e}")
        import traceback
        traceback.print_exc()
