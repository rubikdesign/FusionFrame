"""
Configuration settings for FusionFrame application.

This module contains all configurable parameters and settings that were previously
hardcoded in the FusionFrame class. Moving these to a dedicated config module
allows for easier modification without changing core code.
"""

import os
import torch
from pathlib import Path

# System paths
CACHE_DIR = os.path.join(os.path.expanduser("~"), ".fusionframe")
LORAS_DIR = os.path.join(os.getcwd(), "Loras")
OUTPUTS_DIR = os.path.join(os.getcwd(), "outputs")

# Create required directories
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(LORAS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_MIXED_PRECISION = True  # Use float16 if on GPU
DTYPE = torch.float16 if USE_MIXED_PRECISION and torch.cuda.is_available() else torch.float32

# Default models
DEFAULT_MODEL_ID = "stabilityai/stable-diffusion-xl-refiner-1.0"
DEFAULT_REFINER_ID = "stabilityai/stable-diffusion-xl-refiner-1.0"
DEFAULT_REFINER_NAME = "SDXL Refiner 1.0 (Default)"

# Model repositories
AVAILABLE_MODELS = {
    "stabilityai/stable-diffusion-xl-refiner-1.0": "SDXL Refiner 1.0 (Default)",
    "runwayml/stable-diffusion-v1-5": "Stable Diffusion 1.5",
    "SG161222/Realistic_Vision_V5.1_noVAE": "Realistic Vision 5.1",
    "emilianJR/epiCRealism": "EpicRealism",
    "Lykon/dreamshaper-xl-1-0": "DreamShaper XL",
    "segmind/SSD-1B": "Segmind SSD-1B",
    "gsdf/Counterfeit-V2.5": "Counterfeit V2.5 (Realist)",
    "digiplay/AbsoluteReality_v1.8.1": "Absolute Reality 1.8.1",
}

# ControlNet configurations
CONTROLNET_SOURCES = [
    "lllyasviel/sd-controlnet-openpose",  # First choice for SDXL OpenPose
    "thibaud/controlnet-openpose-sdxl-1.0",
    "diffusers/controlnet-openpose-sdxl",
]

# IP-Adapter configurations
IP_ADAPTER_REPO = "h94/IP-Adapter" 
IP_ADAPTER_FACE_ID = "IP-Adapter/models/face-id/model.ckpt"
IP_ADAPTER_SDXL = "IP-Adapter/sdxl_models/ip-adapter-plus-face_sdxl_vit-h.bin"

# Processing parameters
DEFAULT_SAMPLER = "DPM++ 2M Karras" 
AVAILABLE_SAMPLERS = [
    "DPM++ 2M Karras",
    "Euler a",
    "DDIM"
]

# Face processing parameters
FACE_ENHANCEMENT_DEFAULT = True
FACE_PRESERVATION_STRENGTH = 0.8
FACE_TRANSFER_PARTS = ("eyes", "nose", "mouth")
FACE_TRANSFER_BLEND = 0.85

# Generation parameters
DEFAULT_STEPS = 30
DEFAULT_GUIDANCE_SCALE = 7.5
DEFAULT_STRENGTH = 0.75
DEFAULT_WIDTH = 1024
DEFAULT_HEIGHT = 1024

# Refiner parameters
REFINER_ENABLED_DEFAULT = False
REFINER_STRENGTH_DEFAULT = 0.3

# ControlNet parameters
CONTROLNET_ENABLED_DEFAULT = False
CONTROLNET_STRENGTH_DEFAULT = 1.0

# Save parameters
AUTO_SAVE_DEFAULT = True
SAVE_FORMAT_DEFAULT = "png"

# Prompting
DEFAULT_NEGATIVE_PROMPT = (
    "deformed face, ugly, bad proportions, bad anatomy, disfigured, mutations, "
    "poorly drawn, blurry, low quality, cartoon, anime, illustration, painting, "
    "drawing, different person, wrong face, two faces, multiple faces, mutation, "
    "deformed iris, deformed pupils, morbid, mutilated, extra fingers, extra limbs, "
    "disfigured"
)

# Logging
LOGGING_LEVEL = "INFO"  # Options: DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_TO_FILE = False
LOG_FILE = os.path.join(CACHE_DIR, "fusionframe.log")
