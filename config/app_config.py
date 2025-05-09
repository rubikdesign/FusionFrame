#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
General configurations for FusionFrame 2.0 application with enhanced memory management
"""
import os
import torch
import logging  # Added logging import
from pathlib import Path
import gc

class AppConfig:
    """Global configuration for the application"""
    # Version information
    VERSION = "2.0.0"
    APP_NAME = "FusionFrame"
    
    # Device and data types
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DTYPE = DTYPE = torch.float32 

    # --- Memory Management and Optimization Settings ---
    # Automatic detection of memory constraints
    if torch.cuda.is_available():
        try:
            gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            # Automatic settings based on available GPU memory
            if gpu_mem_gb < 6:  # Less than 6GB VRAM
                LOW_VRAM_MODE = True
                FORCE_CPU_FOR_MASK_MODELS = True
                ENABLE_VAE_TILING = True
                USE_REFINER = False
                DISABLE_DEPTH_ESTIMATION = True
                ANALYZER_LIGHTWEIGHT_MODE = True
                USE_LIGHTWEIGHT_MODELS = True
            elif gpu_mem_gb < 10:  # 6-10GB VRAM
                LOW_VRAM_MODE = True
                ENABLE_VAE_TILING = True
                USE_REFINER = False
                ANALYZER_LIGHTWEIGHT_MODE = True
                USE_LIGHTWEIGHT_MODELS = True
                FORCE_CPU_FOR_MASK_MODELS = False
                DISABLE_DEPTH_ESTIMATION = False
            else:  # 10GB+ VRAM (high-end GPUs)
                LOW_VRAM_MODE = False
                USE_REFINER = True
                ANALYZER_LIGHTWEIGHT_MODE = False
                USE_LIGHTWEIGHT_MODELS = False
                FORCE_CPU_FOR_MASK_MODELS = False
                DISABLE_DEPTH_ESTIMATION = False
        except Exception as e:
            # Fallback if GPU detection fails
            logging.warning(f"GPU memory detection failed: {e}. Using conservative memory settings.")
            LOW_VRAM_MODE = True
            FORCE_CPU_FOR_MASK_MODELS = True
            ENABLE_VAE_TILING = True
            USE_REFINER = False
            ANALYZER_LIGHTWEIGHT_MODE = True
            USE_LIGHTWEIGHT_MODELS = True
            DISABLE_DEPTH_ESTIMATION = True
    else:
        # CPU-only mode settings
        LOW_VRAM_MODE = True
        FORCE_CPU_FOR_MASK_MODELS = True
        ENABLE_VAE_TILING = True
        USE_REFINER = False
        ANALYZER_LIGHTWEIGHT_MODE = True
        USE_LIGHTWEIGHT_MODELS = True
        DISABLE_DEPTH_ESTIMATION = True

    # --- Advanced Memory Management Settings ---
    # Model Loading Policy - Determines how models are managed
    MODEL_LOADING_POLICY = "AGGRESSIVE_UNLOAD"  # Options: KEEP_LOADED, UNLOAD_UNUSED, AGGRESSIVE_UNLOAD
    
    # Minimal set of models to keep loaded (others will be unloaded when not in use)
    ESSENTIAL_MODELS = ["main"]
    
    # Controls which models are forced to CPU even in high VRAM conditions
    FORCE_CPU_MODELS = ["depth_estimator", "image_classifier"]
    
    # How much VRAM (in MB) should be kept free at all times
    MIN_FREE_VRAM_MB = 1000
    
    # Time (in seconds) after which unused models are unloaded
    MODEL_UNLOAD_TIMEOUT = 30
    
    # Set of models that should use lightweight variants
    LIGHTWEIGHT_MODEL_KEYS = ["yolo", "depth_estimator", "image_classifier"]

    # --- Model Processing Settings ---
    MEDIAPIPE_SELFIE_MODEL_SELECTION = 1 # 0 for landscape, 1 for general
    MEDIAPIPE_FACE_MODEL_SELECTION = 0 # 0 for short-range, 1 for full-range
    MEDIAPIPE_FACE_MIN_CONFIDENCE = 0.5
    REMBG_MODEL_NAME = "u2netp" if LOW_VRAM_MODE else "u2net" # Smaller model in low VRAM mode
    
    # CUDA optimizations
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

    # --- Image Analysis Settings ---
    # Controls which features are enabled in analyzer.py
    ANALYZER_LIGHTWEIGHT_MODE = LOW_VRAM_MODE  # Use lightweight mode when in LOW_VRAM_MODE
    FORCE_CPU_FOR_ANALYSIS_MODELS = FORCE_CPU_FOR_MASK_MODELS
    DISABLE_DEPTH_ESTIMATION = LOW_VRAM_MODE  # Disable depth estimation in LOW_VRAM_MODE
    DISABLE_HEAVY_ANALYZER_MODELS = LOW_VRAM_MODE  # Generally disable heavy models

    # --- Application directories ---
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
    DEFAULT_STEPS = 30 if LOW_VRAM_MODE else 50  # Reduce steps in LOW_VRAM_MODE
    MAX_STEPS = 50 if LOW_VRAM_MODE else 80  # Reduce max steps in LOW_VRAM_MODE
    
    # Refiner settings
    USE_REFINER = not LOW_VRAM_MODE  # Disable refiner in LOW_VRAM_MODE
    REFINER_STRENGTH = 0.3
    
    # --- CLIPSeg Segmentation Thresholds ---
    CLIPSEG_DEFAULT_THRESHOLD = 0.35
    CLIPSEG_HAIR_THRESHOLD = 0.4
    CLIPSEG_HEAD_THRESHOLD = 0.4
    CLIPSEG_EYES_THRESHOLD = 0.4
    CLIPSEG_FACE_THRESHOLD = 0.4
    CLIPSEG_CLOTHING_THRESHOLD = 0.4
    CLIPSEG_PERSON_THRESHOLD = 0.4
    CLIPSEG_SKY_THRESHOLD = 0.4
    CLIPSEG_TREE_THRESHOLD = 0.4
    CLIPSEG_OBJECT_THRESHOLD = 0.4
    HYBRID_MASK_THRESHOLD = 0.4
    
    # Mask Generator Settings
    FORCE_CPU_FOR_MASK_MODELS = LOW_VRAM_MODE  # Force mask models to CPU in LOW_VRAM_MODE
    UNLOAD_MODELS_AGGRESSIVELY = LOW_VRAM_MODE  # Unload models aggressively in LOW_VRAM_MODE
    
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
        
    @classmethod
    def get(cls, key, default=None):
        """Get configuration value with fallback"""
        return getattr(cls, key, default)
        
    @classmethod
    def clear_gpu_memory(cls):
        """Clear GPU memory cache"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            
    @classmethod
    def get_available_vram_mb(cls):
        """Get available VRAM in MB"""
        if not torch.cuda.is_available():
            return 0
            
        try:
            free_vram = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()
            free_vram_mb = free_vram / (1024 * 1024)
            return free_vram_mb
        except Exception:
            return 0
