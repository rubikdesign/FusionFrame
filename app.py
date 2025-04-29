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

# Set up basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Check if CUDA is available
if torch.cuda.is_available():
    logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
    logger.info(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    logger.warning("CUDA not available. Performance will be severely limited.")

# Diffusers and related imports
try:
    from diffusers import (
        StableDiffusionXLPipeline, StableDiffusionPipeline, 
        StableDiffusionXLImg2ImgPipeline, StableDiffusionImg2ImgPipeline,
        StableDiffusionXLControlNetPipeline, StableDiffusionControlNetPipeline,
        DDIMScheduler, EulerAncestralDiscreteScheduler, DPMSolverMultistepScheduler,
        AutoencoderKL, ControlNetModel, UNet2DConditionModel
    )
    from diffusers.pipelines.controlnet import MultiControlNetModel
    from diffusers.utils import load_image
    from transformers import CLIPImageProcessor, AutoProcessor, CLIPVisionModelWithProjection
except ImportError:
    logger.error("Required packages not found. Installing dependencies...")
    # Auto-install required packages
    os.system("pip install diffusers==0.22.1 transformers accelerate safetensors controlnet-aux timm mediapipe")
    os.system("pip install git+https://github.com/huggingface/diffusers.git@main")
    os.system("pip install ip-adapter")
    # Retry imports
    from diffusers import (
        StableDiffusionXLPipeline, StableDiffusionPipeline, 
        StableDiffusionXLImg2ImgPipeline, StableDiffusionImg2ImgPipeline,
        StableDiffusionXLControlNetPipeline, StableDiffusionControlNetPipeline,
        DDIMScheduler, EulerAncestralDiscreteScheduler, DPMSolverMultistepScheduler,
        AutoencoderKL, ControlNetModel
    )
    from diffusers.pipelines.controlnet import MultiControlNetModel
    from diffusers.utils import load_image
    from transformers import CLIPImageProcessor, AutoProcessor, CLIPVisionModelWithProjection

# Try to import IP-Adapter
try:
    from ip_adapter import IPAdapterPlus, IPAdapterPlusXL
    IP_ADAPTER_AVAILABLE = True
except ImportError:
    logger.warning("IP-Adapter not available. Installing...")
    os.system("pip install git+https://github.com/tencent-ailab/IP-Adapter.git")
    try:
        from ip_adapter import IPAdapterPlus, IPAdapterPlusXL
        IP_ADAPTER_AVAILABLE = True
    except ImportError:
        logger.error("Could not install IP-Adapter. Some features will be disabled.")
        IP_ADAPTER_AVAILABLE = False

# Try to import ControlNet preprocessors
try:
    from controlnet_aux import OpenposeDetector, HEDdetector, MidasDetector, LineartDetector
    import cv2
    CONTROLNET_AUX_AVAILABLE = True
except ImportError:
    logger.warning("ControlNet auxiliary modules not available. Installing...")
    os.system("pip install controlnet_aux")
    try:
        from controlnet_aux import OpenposeDetector, HEDdetector, MidasDetector, LineartDetector
        import cv2
        CONTROLNET_AUX_AVAILABLE = True
    except ImportError:
        logger.error("Could not install ControlNet auxiliary modules. Some features will be disabled.")
        CONTROLNET_AUX_AVAILABLE = False

# Configuration
CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "clothing_app")
LORA_DIR = os.path.join(CACHE_DIR, "loras")
UPLOAD_DIR = os.path.join(CACHE_DIR, "uploads")
OUTPUT_DIR = os.path.join(CACHE_DIR, "outputs")
CONTROLNET_DIR = os.path.join(CACHE_DIR, "controlnet")
IPADAPTER_DIR = os.path.join(CACHE_DIR, "ipadapter")
LOGS_DIR = os.path.join(CACHE_DIR, "logs")

# Create necessary directories
for directory in [CACHE_DIR, LORA_DIR, UPLOAD_DIR, OUTPUT_DIR, CONTROLNET_DIR, IPADAPTER_DIR, LOGS_DIR]:
    os.makedirs(directory, exist_ok=True)

# Set up file logging in addition to console logging
file_handler = logging.FileHandler(os.path.join(LOGS_DIR, f"app_{time.strftime('%Y%m%d_%H%M%S')}.log"))
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

# Available models
AVAILABLE_MODELS = {
    "SDXL 1.0": "stabilityai/stable-diffusion-xl-base-1.0",
    "SDXL Refiner": "stabilityai/stable-diffusion-xl-refiner-1.0",
    "SD 2.1": "stabilityai/stable-diffusion-2-1",
    "SD 1.5": "runwayml/stable-diffusion-v1-5",
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

# Global variables for loaded components
loaded_components = {
    "model": None,
    "model_name": None,
    "vae": None,
    "controlnet": {},
    "ip_adapter": None,
    "ip_adapter_name": None,
    "scheduler": None,
    "scheduler_name": None
}

def download_file(url, save_path):
    """Download a file from a URL and save it locally"""
    response = requests.get(url, stream=True)
    total_size_in_bytes = int(response.headers.get('content-length', 0))
    block_size = 1024
    progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
    
    with open(save_path, 'wb') as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    
    progress_bar.close()
    if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
        logger.error("Error downloading file")
        return False
    return True

def load_vae(vae_name):
    """Load and return VAE if specified"""
    if vae_name == "Default" or vae_name is None:
        return None
    
    logger.info(f"Loading VAE: {vae_name}")
    vae_id = VAE_OPTIONS[vae_name]
    try:
        vae = AutoencoderKL.from_pretrained(
            vae_id,
            torch_dtype=torch.float16
        )
        if torch.cuda.is_available():
            vae = vae.to("cuda")
        return vae
    except Exception as e:
        logger.error(f"Error loading VAE: {e}")
        return None

def load_controlnet(controlnet_name, model_type="SD"):
    """Load ControlNet model if specified"""
    if controlnet_name == "None" or controlnet_name is None:
        return None
    
    model_id = CONTROLNET_MODELS[controlnet_name]
    if model_id is None:
        return None
    
    cache_path = os.path.join(CONTROLNET_DIR, controlnet_name.replace(" ", "_").lower())
    os.makedirs(cache_path, exist_ok=True)
    
    logger.info(f"Loading ControlNet: {controlnet_name}")
    try:
        controlnet = ControlNetModel.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            cache_dir=cache_path
        )
        if torch.cuda.is_available():
            controlnet = controlnet.to("cuda")
        return controlnet
    except Exception as e:
        logger.error(f"Error loading ControlNet: {e}")
        return None

def load_ip_adapter(ip_adapter_name, base_model_path, is_xl=False):
    """Load IP-Adapter if specified"""
    if not IP_ADAPTER_AVAILABLE or ip_adapter_name == "None" or ip_adapter_name is None:
        return None
    
    adapter_config = IP_ADAPTER_MODELS.get(ip_adapter_name)
    if adapter_config is None:
        return None
    
    cache_path = os.path.join(IPADAPTER_DIR, ip_adapter_name.replace(" ", "_").lower())
    os.makedirs(cache_path, exist_ok=True)
    
    logger.info(f"Loading IP-Adapter: {ip_adapter_name}")
    try:
        if is_xl:
            ip_adapter = IPAdapterPlusXL(
                base_model_path,
                adapter_config["model_id"],
                adapter_config["subfolder"],
                adapter_config["weight_name"],
                device="cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            ip_adapter = IPAdapterPlus(
                base_model_path,
                adapter_config["model_id"],
                adapter_config["subfolder"],
                adapter_config["weight_name"],
                device="cuda" if torch.cuda.is_available() else "cpu"
            )
        return ip_adapter
    except Exception as e:
        logger.error(f"Error loading IP-Adapter: {e}")
        import traceback
        traceback.print_exc()
        return None

def load_model(model_name, scheduler_name="DPM++ 2M Karras", vae_name="Default", 
               controlnet_name="None", ip_adapter_name="None", force_reload=False):
    """Load the selected model and its components"""
    global loaded_components
    
    # Check if model is already loaded and no reload is forced
    if (loaded_components["model"] is not None and 
        loaded_components["model_name"] == model_name and 
        loaded_components["scheduler_name"] == scheduler_name and
        not force_reload):
        return loaded_components["model"]
    
    # Free memory from previously loaded model
    if loaded_components["model"] is not None:
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
    logger.info(f"Loading model: {model_name} ({model_id})")
    
    # Load VAE first if specified
    vae = load_vae(vae_name)
    loaded_components["vae"] = vae
    
    # Load controlnet if specified
    controlnet = None
    if controlnet_name != "None":
        controlnet = load_controlnet(controlnet_name, "SDXL" if is_xl else "SD")
        if controlnet:
            loaded_components["controlnet"][controlnet_name] = controlnet
    
    # Initialize the appropriate pipeline based on model type
    try:
        if is_xl:
            if controlnet:
                pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
                    model_id,
                    controlnet=controlnet,
                    vae=vae,
                    torch_dtype=torch.float16,
                    use_safetensors=True,
                    variant="fp16"
                )
            else:
                pipe = StableDiffusionXLPipeline.from_pretrained(
                    model_id,
                    vae=vae,
                    torch_dtype=torch.float16,
                    use_safetensors=True,
                    variant="fp16"
                )
        else:
            if controlnet:
                pipe = StableDiffusionControlNetPipeline.from_pretrained(
                    model_id,
                    controlnet=controlnet,
                    vae=vae,
                    torch_dtype=torch.float16,
                    safety_checker=None,
                    requires_safety_checker=False
                )
            else:
                pipe = StableDiffusionPipeline.from_pretrained(
                    model_id,
                    vae=vae,
                    torch_dtype=torch.float16,
                    safety_checker=None,
                    requires_safety_checker=False
                )
    
        # Set up scheduler
        scheduler_fn = SCHEDULER_OPTIONS.get(scheduler_name)
        if scheduler_fn:
            pipe.scheduler = scheduler_fn(pipe.scheduler.config)
        
        # Move to GPU if available
        if torch.cuda.is_available():
            pipe = pipe.to("cuda")
            pipe.enable_attention_slicing()
            
        # Optional memory optimization
        if torch.cuda.is_available() and hasattr(pipe, "enable_model_cpu_offload"):
            pipe.enable_model_cpu_offload()
        
        # Load IP-Adapter if specified
        if ip_adapter_name != "None":
            ip_adapter = load_ip_adapter(ip_adapter_name, model_id, is_xl)
            loaded_components["ip_adapter"] = ip_adapter
            loaded_components["ip_adapter_name"] = ip_adapter_name
        
        # Update loaded components
        loaded_components["model"] = pipe
        loaded_components["model_name"] = model_name
        loaded_components["scheduler_name"] = scheduler_name
        
        return pipe
    
    except Exception as e:
        logger.error(f"Error loading model: {e}")
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
    
    return gr.Dropdown(choices=updated_choices)

def update_lora_dropdowns(lora_file):
    """Update all LoRA dropdowns when a new LoRA is uploaded"""
    if lora_file is None:
        return [gr.Dropdown(choices=DEFAULT_LORAS) for _ in range(5)]
    
    filename = os.path.join(LORA_DIR, lora_file.name)
    with open(filename, "wb") as f:
        f.write(lora_file.read())  # Changed from file_handler.read() to lora_file.read()
    
    # Add to global LoRA files dict
    lora_files[lora_file.name] = filename
    
    # Update dropdown choices
    updated_choices = DEFAULT_LORAS + list(lora_files.keys())
    
    return [gr.Dropdown(choices=updated_choices) for _ in range(5)]

def load_lora_weights(pipe, lora_file_path, lora_weight=0.7):
    """Load LoRA weights into the model pipe"""
    if lora_file_path is None or lora_file_path == "None":
        return pipe
    
    logger.info(f"Loading LoRA: {lora_file_path} with weight {lora_weight}")
    
    try:
        # Different loading method based on model type
        if hasattr(pipe, "unet") and hasattr(pipe.unet, "load_attn_procs"):
            pipe.load_lora_weights(lora_file_path)
            
            # Apply LoRA weight scaling if the method exists
            if hasattr(pipe, "set_adapters"):
                pipe.set_adapters(["default"], adapter_weights=[lora_weight])
    except Exception as e:
        logger.error(f"Error loading LoRA: {e}")
    
    return pipe

def prepare_image(image_path, target_size=(768, 768)):
    """Load and prepare image for processing"""
    if image_path is None:
        return None
    
    try:
        # Open the image file
        image = Image.open(image_path)
        
        # Convert RGBA to RGB if needed
        if image.mode == 'RGBA':
            # Create a white background
            background = Image.new('RGB', image.size, (255, 255, 255))
            # Paste the image using itself as mask
            background.paste(image, (0, 0), image)
            image = background
        
        # Resize to target size while preserving aspect ratio
        image.thumbnail(target_size, Image.Resampling.LANCZOS)
        
        # Create a new image with the target size and paste the resized image centered
        new_image = Image.new('RGB', target_size, (255, 255, 255))
        offset = ((target_size[0] - image.width) // 2, (target_size[1] - image.height) // 2)
        new_image.paste(image, offset)
        
        return new_image
    
    except Exception as e:
        logger.error(f"Error preparing image: {e}")
        return None

def generate_controlnet_conditioning(image, controlnet_type):
    """Generate conditioning image for ControlNet"""
    if not CONTROLNET_AUX_AVAILABLE or image is None or controlnet_type == "None":
        return None
    
    try:
        # Convert PIL image to numpy array
        numpy_image = np.array(image)
        
        if "Pose" in controlnet_type:
            openpose = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
            return openpose(numpy_image)
        
        elif "Canny Edge" in controlnet_type:
            # Apply Canny edge detection
            low_threshold = 100
            high_threshold = 200
            image_gray = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(image_gray, low_threshold, high_threshold)
            edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
            return Image.fromarray(edges_colored)
        
        elif "Depth" in controlnet_type:
            depth_estimator = MidasDetector.from_pretrained("lllyasviel/ControlNet")
            return depth_estimator(numpy_image)
        
        elif "Lineart" in controlnet_type:
            lineart = LineartDetector.from_pretrained("lllyasviel/ControlNet")
            return lineart(numpy_image)
        
        elif "Soft Edge" in controlnet_type:
            hed = HEDdetector.from_pretrained("lllyasviel/ControlNet")
            return hed(numpy_image)
        
        else:
            return None
    
    except Exception as e:
        logger.error(f"Error generating controlnet conditioning: {e}")
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
    """Generate images based on the provided inputs"""
    try:
        # Track progress
        progress(0, desc="Initializing...")
        
        # Load the model
        pipe = load_model(
            model_name, 
            scheduler_name=scheduler_name, 
            vae_name=vae_name, 
            controlnet_name=controlnet_type,
            ip_adapter_name=ip_adapter_name
        )
        
        if pipe is None:
            return [], seed, "Failed to load model"
        
        # Apply seed for reproducibility
        if seed == -1:
            seed = torch.randint(0, 2**32, (1,)).item()
        generator = torch.Generator("cuda").manual_seed(seed)
        
        # Prepare inputs
        has_woman = woman_image is not None
        has_clothing = clothing_image is not None
        has_background = background_image is not None
        
        # Convert uploaded images to PIL and prepare them
        progress(0.1, desc="Processing input images...")
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
        
        # Load LoRAs
        progress(0.15, desc="Loading LoRAs...")
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
        progress(0.2, desc="Preparing ControlNet...")
        controlnet_image = None
        if controlnet_type != "None" and woman_img is not None:
            controlnet_image = generate_controlnet_conditioning(woman_img, controlnet_type)
        
        # Prepare IP-Adapter input if needed
        if ip_adapter_name != "None" and loaded_components["ip_adapter"] is not None:
            progress(0.25, desc="Preparing IP-Adapter...")
            ip_adapter = loaded_components["ip_adapter"]
            
            # Determine which image to use for IP-Adapter
            reference_image = None
            if has_woman:
                reference_image = woman_img
            elif has_clothing:
                reference_image = clothing_img
            elif has_background:
                reference_image = background_img
            
            if reference_image:
                ip_adapter.set_ip_adapter_scale(ip_adapter_scale)
        
        # Generate images
        progress(0.3, desc="Generating images...")
        outputs = []
        
        # We'll use img2img if any of the input images are provided
        use_img2img = has_woman or has_clothing or has_background
        
        # Determine if we're using an XL model
        is_xl = "XL" in model_name
        
        for i in range(num_outputs):
            progress(0.3 + (0.7 * i / num_outputs), desc=f"Generating image {i+1}/{num_outputs}")
            
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
            
            # Standard generation when no IP-Adapter is used
            result = pipe(**generation_params)
            outputs.extend(result.images)
        
        # Save generated images
        progress(0.95, desc="Saving generated images...")
        saved_paths = []
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        for i, img in enumerate(outputs):
            output_path = os.path.join(OUTPUT_DIR, f"generated_{timestamp}_{i}_{seed}.png")
            img.save(output_path)
            saved_paths.append(output_path)
        
        # Clear LoRAs from pipeline for next run
        progress(0.99, desc="Cleaning up...")
        if hasattr(pipe, "unload_lora_weights"):
            pipe.unload_lora_weights()
        
        success_message = f"Successfully generated {len(saved_paths)} images with seed {seed}"
        logger.info(success_message)
        return saved_paths, seed, success_message
        
    except Exception as e:
        logger.error(f"Error generating images: {e}")
        import traceback
        traceback.print_exc()
        return [], seed, f"Error: {str(e)}"

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
    
    return info

def create_interface():
    """Create the Gradio interface"""
    
    with gr.Blocks(title="Clothing Try-On Image Generator", theme=gr.themes.Soft()) as app:
        gr.Markdown("# 👗 Platformă Avansată de Generare Imagini: Îmbrăcare + Integrare în Fundal")
        gr.Markdown("Încărcați imagini pentru a genera combinații între model, îmbrăcăminte și fundal, cu setări personalizate.")
        
        with gr.Tab("Generare Imagini"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 📸 1. Încărcare Imagini")
                    
                    woman_image = gr.File(label="Imagine Model", file_types=["image"])
                    clothing_image = gr.File(label="Imagine Îmbrăcăminte (Opțional)", file_types=["image"])
                    background_image = gr.File(label="Imagine Fundal (Opțional)", file_types=["image"])
                    
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
                                value="SDXL 1.0"
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
                            value=768,
                            step=64
                        )
                        
                        image_height = gr.Slider(
                            label="Înălțime Imagine",
                            minimum=512,
                            maximum=1024,
                            value=768,
                            step=64
                        )
                    
                    with gr.Row():
                        generate_button = gr.Button("🚀 Generează Imagini", variant="primary", size="lg")
                        clear_button = gr.Button("🧹 Resetează", variant="secondary")
                
                with gr.Column(scale=2):
                    with gr.Row():
                        output_gallery = gr.Gallery(
                            label="Imagini Generate",
                            show_label=True,
                            elem_id="gallery",
                            columns=2,
                            height=600
                        )
                        
                    with gr.Row():
                        output_seed = gr.Number(label="Seed Utilizat", interactive=False)
                        output_message = gr.Textbox(label="Mesaj Status", interactive=False)
        
        with gr.Tab("Informații"):
            gr.Markdown(create_model_info())
        
        # Set up event handlers
        woman_image.change(save_uploaded_file, inputs=[woman_image], outputs=[woman_image])
        clothing_image.change(save_uploaded_file, inputs=[clothing_image], outputs=[clothing_image])
        background_image.change(save_uploaded_file, inputs=[background_image], outputs=[background_image])
        
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
                0.75, 30, 7.5, 768, 768, 1, -1,  # Generation params
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
        
        # Generation process
        generate_button.click(
            generate_images,
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
    
    return app

if __name__ == "__main__":
    # Print system information
    logger.info(f"Python version: {sys.version}")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Create and launch the app
    app = create_interface()
    app.queue(concurrency_count=1, max_size=20)
    app.launch(share=True, server_port=7860)
