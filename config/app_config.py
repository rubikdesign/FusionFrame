#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
General configurations for FusionFrame 2.0 application
"""
import os
import torch
import logging  # Added logging import
from pathlib import Path

class AppConfig:
    """Global configuration for the application"""
    # Version information
    VERSION = "2.0.0"
    APP_NAME = "FusionFrame"
    
    # Device and data types
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

    MEDIAPIPE_SELFIE_MODEL_SELECTION = 1 # 0 for landscape, 1 for general
    MEDIAPIPE_FACE_MODEL_SELECTION = 0 # 0 for short-range, 1 for full-range
    MEDIAPIPE_FACE_MIN_CONFIDENCE = 0.5
    REMBG_MODEL_NAME = "u2net" # Other options: "u2net_human_seg", "isnet-general-use", etc.
    
    # CUDA check and optimization settings
    CUDA_AVAILABLE = torch.cuda.is_available()
    if CUDA_AVAILABLE:
        CUDA_VERSION = torch.version.cuda
        if CUDA_VERSION and CUDA_VERSION.startswith("11."):
            os.environ["DIFFUSERS_DISABLE_XFORMERS"] = "1"
        else:
            os.environ["DIFFUSERS_DISABLE_XFORMERS"] = "0"
    
    # Memory optimizations
    MAX_WORKERS = min(4, os.cpu_count() or 2)
    TILE_SIZE = 512
    LOW_VRAM_MODE = torch.cuda.is_available() and \
                    torch.cuda.get_device_properties(0).total_memory < 6e9
    
    # Application directories
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODEL_DIR = os.path.join(BASE_DIR, "models_cache")
    CACHE_DIR = os.path.join(MODEL_DIR, "cache")
    LORA_DIR = os.path.join(BASE_DIR, "loras")
    LOGS_DIR = os.path.join(BASE_DIR, "logs")
    OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
    
    # Model download URLs
    MODEL_URLS = {
        "sam": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
        "modnet": "https://github.com/ZHKKKe/MODNet/releases/download/v1.0/modnet_photographic_portrait_matting.pth",
        "gpen": "https://github.com/yangxy/GPEN/releases/download/v1.0/GPEN-BFR-512.pth",
        "codeformer": "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth",
        "esrgan": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        # Adding URLs for HiDream-I1 models if needed
        "hidream-i1": "https://huggingface.co/HiDream-ai/HiDream-I1-Full/resolve/main/pytorch_model.safetensors",
    }
    
    # Quality and generation parameters
    QUALITY_THRESHOLD = 0.75
    MAX_REGENERATION_ATTEMPTS = 3
    DEFAULT_GUIDANCE_SCALE = 7.5
    DEFAULT_STEPS = 50
    MAX_STEPS = 80
    
    # Refiner settings
    USE_REFINER = True  # Enable refiner by default
    REFINER_STRENGTH = 0.3  # Default value for refiner intensity
    
    @classmethod
    def ensure_dirs(cls):
        """Ensure all directories exist"""
        Path(cls.MODEL_DIR).mkdir(parents=True, exist_ok=True)
        Path(cls.CACHE_DIR).mkdir(parents=True, exist_ok=True)
        Path(cls.LORA_DIR).mkdir(parents=True, exist_ok=True)
        Path(cls.LOGS_DIR).mkdir(parents=True, exist_ok=True)
        Path(cls.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def setup_logging(cls, level=logging.INFO):
        """Configure logging system"""
        # Ensure log directory exists
        Path(cls.LOGS_DIR).mkdir(parents=True, exist_ok=True)
        
        # Configure general logging
        log_file = os.path.join(cls.LOGS_DIR, "fusionframe.log")
        logging.basicConfig(
            level=level,  # Use received parameter
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        return logging.getLogger(__name__)