#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
General configurations for FusionFrame 2.0 application - MAX PERFORMANCE SETTINGS
"""
import os
import torch
import logging
from pathlib import Path
import gc

class AppConfig:
    """Global configuration for the application"""
    # Version information
    VERSION = "2.0.1-max_performance" # Am schimbat versiunea pentru a reflecta setările
    APP_NAME = "FusionFrame"

    # --- Device and Data Types ---
    # Utilizează CUDA dacă e disponibil, altfel CPU
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    # Pentru performanță maximă pe GPU-uri moderne, float16 este adesea preferat.
    # Dacă întâmpini probleme de precizie sau artefacte, poți reveni la torch.float32.
    DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

    # --- Memory Management and Optimization Settings - CONFIGURATE PENTRU PERFORMANȚĂ MAXIMĂ ---
    if torch.cuda.is_available():
        # Dezactivează modul VRAM redus și toate optimizările agresive de memorie
        LOW_VRAM_MODE = False
        FORCE_CPU_FOR_MASK_MODELS = False # Rulează modelele de mască pe GPU pentru viteză
        ENABLE_VAE_TILING = False # De obicei nu e necesar cu VRAM suficient; poate încetini
        USE_REFINER = True # Activează refiner-ul pentru calitate îmbunătățită (dacă modelul principal îl suportă)
        DISABLE_DEPTH_ESTIMATION = False # Activează estimarea de adâncime
        ANALYZER_LIGHTWEIGHT_MODE = False # Folosește modul complet pentru analiză
        USE_LIGHTWEIGHT_MODELS = False # Folosește variantele standard, mai mari și mai precise ale modelelor
    else:
        # Setări pentru CPU-only (acestea rămân orientate spre funcționalitate pe CPU)
        LOW_VRAM_MODE = True
        FORCE_CPU_FOR_MASK_MODELS = True
        ENABLE_VAE_TILING = True # Poate fi util pe CPU pentru imagini mari
        USE_REFINER = False
        ANALYZER_LIGHTWEIGHT_MODE = True
        USE_LIGHTWEIGHT_MODELS = True
        DISABLE_DEPTH_ESTIMATION = True # Estimarea de adâncime poate fi lentă pe CPU

    # --- Advanced Memory Management Settings ---
    # Păstrează modelele încărcate pentru acces rapid, dacă memoria permite
    MODEL_LOADING_POLICY = "KEEP_LOADED" # Opțiuni: KEEP_LOADED, UNLOAD_UNUSED, AGGRESSIVE_UNLOAD
    ESSENTIAL_MODELS = ["main", "sam_predictor", "clipseg"] # Modele care să rămână mereu încărcate (dacă se poate)

    # Golește această listă dacă vrei ca toate modelele să încerce GPU-ul
    FORCE_CPU_MODELS = [] # Exemple: ["image_classifier"] dacă vrei ca doar clasificatorul să meargă pe CPU

    # Prag mai mic pentru VRAM liber dacă ai mult VRAM
    MIN_FREE_VRAM_MB = 500 # (ex: 500MB-1GB)
    # Timeout mai mare dacă politica este "UNLOAD_UNUSED"
    MODEL_UNLOAD_TIMEOUT = 600 # 10 minute

    # Devine mai puțin relevantă dacă USE_LIGHTWEIGHT_MODELS este False
    LIGHTWEIGHT_MODEL_KEYS = ["yolo", "depth_estimator", "image_classifier", "sam"]

    # --- Model Processing Settings ---
    MEDIAPIPE_SELFIE_MODEL_SELECTION = 1 # 0 pentru landscape, 1 pentru general
    MEDIAPIPE_FACE_MODEL_SELECTION = 1 # 1 pentru full-range (potențial mai precis)
    MEDIAPIPE_FACE_MIN_CONFIDENCE = 0.5
    REMBG_MODEL_NAME = "u2net" # Folosește modelul standard, mai mare

    # --- CUDA Optimizations ---
    CUDA_AVAILABLE = torch.cuda.is_available()
    if CUDA_AVAILABLE:
        CUDA_VERSION = torch.version.cuda
        # xFormers este de obicei benefic pentru performanță și memorie pe GPU-uri compatibile.
        # Verifică compatibilitatea cu versiunea ta de PyTorch/CUDA/diffusers.
        # "0" înseamnă că xFormers este ACTIVAT (nu dezactivat).
        # Setează la "1" doar dacă întâmpini probleme cu xFormers.
        if CUDA_VERSION and CUDA_VERSION.startswith("11."): # Exemplu: poate xformers are probleme cu CUDA 11 mai vechi
             os.environ.setdefault("DIFFUSERS_DISABLE_XFORMERS", "1") # Dezactivează xFormers
        else:
             os.environ.setdefault("DIFFUSERS_DISABLE_XFORMERS", "0") # Activează xFormers
    else:
        os.environ["DIFFUSERS_DISABLE_XFORMERS"] = "1" # Nu are sens pe CPU

    # Număr mai mare de workers dacă ai multe nuclee CPU și I/O rapid
    MAX_WORKERS = min(8, (os.cpu_count() or 1) + 4) # Ajustează în funcție de sistem
    TILE_SIZE = 1024 # Mărime mai mare pentru tiling dacă VAE Tiling e activat și necesar

    # --- Image Analysis Settings ---
    # Acestea vor fi setate la False dacă LOW_VRAM_MODE este False
    ANALYZER_LIGHTWEIGHT_MODE = LOW_VRAM_MODE
    FORCE_CPU_FOR_ANALYSIS_MODELS = FORCE_CPU_FOR_MASK_MODELS # Adică False
    DISABLE_DEPTH_ESTIMATION = LOW_VRAM_MODE # Adică False
    DISABLE_HEAVY_ANALYZER_MODELS = LOW_VRAM_MODE # Adică False

    # --- Application directories ---
    # Acestea rămân la fel, dar asigură-te că ai suficient spațiu pe disc
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) if __file__ else os.getcwd()
    MODEL_DIR = os.path.join(BASE_DIR, "models_cache")
    CACHE_DIR = os.path.join(MODEL_DIR, "cache") # Cache pentru Hugging Face etc.
    LORA_DIR = os.path.join(BASE_DIR, "loras")
    LOGS_DIR = os.path.join(BASE_DIR, "logs")
    OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

    # Model download URLs (mai mult informative, clasele model ar trebui să gestioneze descărcarea)
    MODEL_URLS = {
        "sam": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth", # Modelul SAM mare
        "modnet": "https://github.com/ZHKKKe/MODNet/releases/download/v1.0/modnet_photographic_portrait_matting.pth",
        "gpen": "https://github.com/yangxy/GPEN/releases/download/v1.0/GPEN-BFR-512.pth",
        "codeformer": "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth",
        "esrgan": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        "flux": "https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/flux1-dev.safetensors", # Referință
        "hidream-i1": "https://huggingface.co/HiDream-ai/HiDream-I1-Fast/resolve/main/pytorch_model.safetensors", # Referință
    }

    # --- Quality and Generation Parameters ---
    QUALITY_THRESHOLD = 0.85 # Prag mai mare pentru calitate
    MAX_REGENERATION_ATTEMPTS = 2 # Mai puține încercări dacă calitatea inițială e mai bună
    DEFAULT_GUIDANCE_SCALE = 7.0 # Valoare comună, poate fi ajustată
    DEFAULT_STEPS = 50 # Număr bun de pași pentru calitate și viteză echilibrate
                        # Poți crește la 60-75 pentru calitate maximă, dar va fi mai lent.
    MAX_STEPS = 100    # Limită superioară pentru pași

    # Refiner settings
    USE_REFINER = not LOW_VRAM_MODE  # Va fi True
    REFINER_STRENGTH = 0.2 # Ajustează între 0.1 și 0.4 pentru efectul dorit al refiner-ului

    # --- CLIPSeg Segmentation Thresholds ---
    # Pot rămâne la fel, sau pot fi ajustate fin pentru precizie
    CLIPSEG_DEFAULT_THRESHOLD = 0.35
    CLIPSEG_HAIR_THRESHOLD = 0.4
    CLIPSEG_HEAD_THRESHOLD = 0.4
    # ... restul pragurilor CLIPSeg

    HYBRID_MASK_THRESHOLD = 0.4

    # Mask Generator Settings
    FORCE_CPU_FOR_MASK_MODELS = LOW_VRAM_MODE # Va fi False
    # Nu mai descărca agresiv dacă politica generală e KEEP_LOADED
    UNLOAD_MODELS_AGGRESSIVELY = False if MODEL_LOADING_POLICY == "KEEP_LOADED" else LOW_VRAM_MODE

    @classmethod
    def ensure_dirs(cls):
        """Ensure all directories exist"""
        Path(cls.MODEL_DIR).mkdir(parents=True, exist_ok=True)
        Path(cls.CACHE_DIR).mkdir(parents=True, exist_ok=True)
        Path(cls.LORA_DIR).mkdir(parents=True, exist_ok=True)
        Path(cls.LOGS_DIR).mkdir(parents=True, exist_ok=True)
        Path(cls.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    @classmethod
    def setup_logging(cls, level=logging.INFO): # Poți seta level=logging.DEBUG pentru mai multe detalii
        """Configure logging system"""
        Path(cls.LOGS_DIR).mkdir(parents=True, exist_ok=True)
        log_file = os.path.join(cls.LOGS_DIR, f"{cls.APP_NAME.lower()}_performance.log")
        logging.basicConfig(
            level=level,
            format="%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger(cls.APP_NAME) # Logger specific aplicației

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
            # logging.debug("GPU memory cache cleared.") # Adaugă logging dacă dorești

    @classmethod
    def get_available_vram_mb(cls) -> float:
        """Get available VRAM in MB from PyTorch's perspective (total - reserved)."""
        if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
            return 0.0
        try:
            # total_memory - memory_reserved by PyTorch
            # memory_allocated este memoria efectiv folosită de tensori în spațiul rezervat.
            # memory_reserved este memoria totală pe care PyTorch a alocat-o de la driverul CUDA.
            # Ceea ce este "liber" pentru noi ca programatori PyTorch este total_memory - memory_reserved.
            total_mem = torch.cuda.get_device_properties(0).total_memory
            reserved_mem = torch.cuda.memory_reserved(0)
            free_for_pytorch_mb = (total_mem - reserved_mem) / (1024 * 1024)
            return free_for_pytorch_mb
        except Exception as e:
            # logging.getLogger(__name__).warning(f"Could not get available VRAM: {e}")
            return 0.0

# Inițializează logging-ul când modulul este importat
AppConfig.setup_logging()