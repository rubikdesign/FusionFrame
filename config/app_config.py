#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Configurații generale pentru aplicația FusionFrame 2.0
"""
import os
import torch
import logging  # Adăugat importul pentru logging
from pathlib import Path

class AppConfig:
    """Configurare globală pentru aplicație"""
    # Informații versiune
    VERSION = "2.0.0"
    APP_NAME = "FusionFrame"
    
    # Dispozitiv și tipuri de date
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32
    
    # Verificare CUDA și setare optimizări
    CUDA_AVAILABLE = torch.cuda.is_available()
    if CUDA_AVAILABLE:
        CUDA_VERSION = torch.version.cuda
        if CUDA_VERSION and CUDA_VERSION.startswith("11."):
            os.environ["DIFFUSERS_DISABLE_XFORMERS"] = "1"
        else:
            os.environ["DIFFUSERS_DISABLE_XFORMERS"] = "0"
    
    # Optimizări memorie
    MAX_WORKERS = min(4, os.cpu_count() or 2)
    TILE_SIZE = 512
    LOW_VRAM_MODE = torch.cuda.is_available() and \
                    torch.cuda.get_device_properties(0).total_memory < 6e9
    
    # Directoare aplicație
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODEL_DIR = os.path.join(BASE_DIR, "models_cache")
    CACHE_DIR = os.path.join(MODEL_DIR, "cache")
    LORA_DIR = os.path.join(BASE_DIR, "loras")
    LOGS_DIR = os.path.join(BASE_DIR, "logs")
    OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
    
    # URL-uri pentru descărcare modele
    MODEL_URLS = {
        "sam": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
        "modnet": "https://github.com/ZHKKKe/MODNet/releases/download/v1.0/modnet_photographic_portrait_matting.pth",
        "gpen": "https://github.com/yangxy/GPEN/releases/download/v1.0/GPEN-BFR-512.pth",
        "codeformer": "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth",
        "esrgan": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        # Adăugăm URL-uri pentru modelele HiDream-I1 dacă sunt necesare
        "hidream-i1": "https://huggingface.co/HiDream-ai/HiDream-I1-Full/resolve/main/pytorch_model.safetensors",
    }
    
    # Parametri pentru calitate și generare
    QUALITY_THRESHOLD = 0.75
    MAX_REGENERATION_ATTEMPTS = 3
    DEFAULT_GUIDANCE_SCALE = 7.5
    DEFAULT_STEPS = 50
    MAX_STEPS = 80
    
    # Setări pentru refiner
    USE_REFINER = True  # Activăm refiner-ul implicit
    REFINER_STRENGTH = 0.3  # Valoarea implicită pentru intensitatea refiner-ului
    
    @classmethod
    def ensure_dirs(cls):
        """Asigură că toate directoarele există"""
        Path(cls.MODEL_DIR).mkdir(parents=True, exist_ok=True)
        Path(cls.CACHE_DIR).mkdir(parents=True, exist_ok=True)
        Path(cls.LORA_DIR).mkdir(parents=True, exist_ok=True)
        Path(cls.LOGS_DIR).mkdir(parents=True, exist_ok=True)
        Path(cls.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def setup_logging(cls, level=logging.INFO):
        """Configurează sistemul de logging"""
        # Asigură existența directorului pentru log-uri
        Path(cls.LOGS_DIR).mkdir(parents=True, exist_ok=True)
        
        # Configurează logging-ul general
        log_file = os.path.join(cls.LOGS_DIR, "fusionframe.log")
        logging.basicConfig(
            level=level,  # Folosim parametrul primit
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        return logging.getLogger(__name__)