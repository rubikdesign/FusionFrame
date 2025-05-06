#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ICEdit Pro - Aplicație avansată de editare a imaginilor folosind AI
Inspirat din UNO și ICEdit
"""

import torch
import gradio as gr
import numpy as np
import cv2
import re
import json
import os
import random
import requests
import time
import sys
import gc
from io import BytesIO
from pathlib import Path
from skimage.metrics import structural_similarity as ssim
from typing import Optional, Dict, List, Tuple, Union, Any
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import warnings
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")

# Setăm variabile de mediu pentru compatibilitate
os.environ["PYTHONWARNINGS"] = "ignore"

# Verificăm disponibilitatea CUDA și setăm variabilele corespunzătoare
CUDA_AVAILABLE = torch.cuda.is_available()
if CUDA_AVAILABLE:
    CUDA_VERSION = torch.version.cuda
    # Verificăm versiunea CUDA pentru a decide dacă dezactivăm xformers
    if CUDA_VERSION and CUDA_VERSION.startswith("11."):
        os.environ["DIFFUSERS_DISABLE_XFORMERS"] = "1"
    else:
        os.environ["DIFFUSERS_DISABLE_XFORMERS"] = "0"

# Importuri pentru Diffusers și modele
try:
    from diffusers import (
        StableDiffusionXLInpaintPipeline,
        AutoencoderKL,
        EulerAncestralDiscreteScheduler,
        ControlNetModel,
        DPMSolverMultistepScheduler
    )
    from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
except ImportError as e:
    print(f"Error importing AI libraries: {e}")
    print("Please install the required dependencies with: pip install -r requirements.txt")
    sys.exit(1)

# Importuri pentru detectarea fețelor și segmentare
try:
    import mediapipe as mp
except ImportError:
    print("MediaPipe not found. Installing...")
    os.system("pip install mediapipe")
    import mediapipe as mp

# Încercăm să importăm alte biblioteci necesare
try:
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
    SAM_AVAILABLE = True
except ImportError:
    print("SAM model not available. Some functionality will be limited.")
    SAM_AVAILABLE = False

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    print("YOLO model not available. Some functionality will be limited.")
    YOLO_AVAILABLE = False


# Configurații pentru aplicație
class Config:
    """Configurare globală pentru aplicație"""
    # Dispozitiv și tipuri de date
    DEVICE = "cuda" if CUDA_AVAILABLE else "cpu"
    DTYPE = torch.float16 if CUDA_AVAILABLE else torch.float32
    MAX_WORKERS = min(4, os.cpu_count() or 2)
    TILE_SIZE = 512
    LOW_VRAM_MODE = torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory < 6e9
    
    # Directoare pentru modele
    MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    CACHE_DIR = os.path.join(MODEL_DIR, "cache")
    
    # Path-uri pentru modele specializate
    LAMA_MODEL_PATH = os.path.join(MODEL_DIR, "lama")
    MODNET_MODEL_PATH = os.path.join(MODEL_DIR, "modnet")
    GPEN_MODEL_PATH = os.path.join(MODEL_DIR, "gpen")
    CODEFORMER_MODEL_PATH = os.path.join(MODEL_DIR, "codeformer")
    ESRGAN_MODEL_PATH = os.path.join(MODEL_DIR, "esrgan")
    
    # URL-uri pentru descărcare modele
    MODEL_URLS = {
        "lama": "https://github.com/advimman/lama/releases/download/v1.0/big-lama.zip",
        "modnet": "https://github.com/ZHKKKe/MODNet/releases/download/v1.0/modnet_photographic_portrait_matting.pth",
        "gpen": "https://github.com/yangxy/GPEN/releases/download/v1.0/GPEN-BFR-512.pth",
        "codeformer": "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth",
        "esrgan": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
    }
    
    # Parametri pentru calitate și generare
    QUALITY_THRESHOLD = 0.7
    MAX_REGENERATION_ATTEMPTS = 3
    DEFAULT_GUIDANCE_SCALE = 7.5
    DEFAULT_STEPS = 50
    MAX_STEPS = 80
    
    @classmethod
    def ensure_model_dirs(cls):
        """Asigură că toate directoarele pentru modele există"""
        os.makedirs(cls.MODEL_DIR, exist_ok=True)
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.LAMA_MODEL_PATH, exist_ok=True)
        os.makedirs(cls.MODNET_MODEL_PATH, exist_ok=True)
        os.makedirs(cls.GPEN_MODEL_PATH, exist_ok=True)
        os.makedirs(cls.CODEFORMER_MODEL_PATH, exist_ok=True)
        os.makedirs(cls.ESRGAN_MODEL_PATH, exist_ok=True)


# Inițializăm configurația
config = Config()
config.ensure_model_dirs()

# Paths pentru modele
MODEL_PATHS = {
    "sam": os.path.join(config.MODEL_DIR, "sam_vit_h_4b8939.pth"),
    "yolo": os.path.join(config.MODEL_DIR, "yolov8x-seg.pt")
}

# Model ControlNet pentru SDXL
CONTROLNET_MODEL = "diffusers/controlnet-canny-sdxl-1.0"


class ProgressMonitor:
    """Clasă pentru monitorizarea progresului cu tqdm"""
    def __init__(self, total_steps=100, description="Processing"):
        self.total_steps = total_steps
        self.description = description
        self.pbar = None
        
    def start(self):
        """Inițializează bara de progres"""
        self.pbar = tqdm(total=self.total_steps, desc=self.description)
        
    def update(self, step=1):
        """Actualizează progresul"""
        if self.pbar is not None:
            self.pbar.update(step)
            
    def close(self):
        """Închide bara de progres"""
        if self.pbar is not None:
            self.pbar.close()
            
    def set_description(self, description):
        """Schimbă descrierea barei de progres"""
        if self.pbar is not None:
            self.pbar.set_description(description)


class ModelDownloader:
    """Gestionează descărcarea și configurarea modelelor specializate"""
    @staticmethod
    def download_file(url, destination):
        """Descarcă un fișier de la URL către destinație"""
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        with tqdm(total=total_size, unit='B', unit_scale=True, desc=f"Downloading {os.path.basename(destination)}") as pbar:
            with open(destination, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        
        return destination
    
    @staticmethod
    def download_and_extract(url, destination_dir):
        """Descarcă și extrage un fișier zip"""
        import zipfile
        
        # Creăm un fișier temporar pentru zip
        temp_zip = os.path.join(destination_dir, "temp.zip")
        
        # Descărcăm fișierul zip
        ModelDownloader.download_file(url, temp_zip)
        
        # Extragem fișierul zip
        with zipfile.ZipFile(temp_zip, 'r') as zip_ref:
            zip_ref.extractall(destination_dir)
        
        # Ștergem fișierul temporar zip
        os.remove(temp_zip)
        
        return destination_dir
    
    @classmethod
    def ensure_model(cls, model_name):
        """Ne asigurăm că un model specific este descărcat și disponibil"""
        model_path = getattr(config, f"{model_name.upper()}_MODEL_PATH")
        model_url = config.MODEL_URLS.get(model_name)
        
        # Verificăm dacă fișierele modelului există
        model_files = list(Path(model_path).glob("*.pth"))
        if not model_files and model_url:
            print(f"Model {model_name} not found. Downloading...")
            if model_url.endswith(".zip"):
                cls.download_and_extract(model_url, model_path)
            else:
                model_file = os.path.join(model_path, f"{model_name}.pth")
                cls.download_file(model_url, model_file)
                
        return model_path


class AdvancedModelManager:
    """Manager pentru modele AI specializate"""
    def __init__(self):
        self.models = {}
        self.load_core_models()
    
    def load_core_models(self):
        """Încărcăm modelele de bază necesare pentru operațiunea de bază"""
        # Inițializăm dicționarul gol pentru a stoca modelele
        self.models = {}
        
        # Încărcăm modelul SDXL (va fi încărcat în EnhancedImageEditor)
        self.models['sdxl'] = None
        
        # Pregătim path-urile pentru alte modele dar nu le încărcăm încă (lazy loading)
        self.model_paths = {
            'lama': ModelDownloader.ensure_model('lama'),
            'modnet': ModelDownloader.ensure_model('modnet'),
            'gpen': ModelDownloader.ensure_model('gpen'),
            'codeformer': ModelDownloader.ensure_model('codeformer'),
            'esrgan': ModelDownloader.ensure_model('esrgan')
        }
    
    def load_model(self, model_name):
        """Lazy load pentru un model specific când este necesar"""
        if model_name in self.models and self.models[model_name] is not None:
            return self.models[model_name]
        
        print(f"Loading {model_name} model...")
        
        # Încărcăm în funcție de tipul modelului
        if model_name == 'lama':
            # Modelul LaMa pentru inpainting
            try:
                import torch
                from lama_cleaner.model_manager import ModelManager
                
                self.models['lama'] = ModelManager(name="lama", device=config.DEVICE)
                print(f"LaMa model loaded successfully")
            except ImportError:
                print("Could not import lama_cleaner. Installing...")
                os.system("pip install lama-cleaner")
                from lama_cleaner.model_manager import ModelManager
                self.models['lama'] = ModelManager(name="lama", device=config.DEVICE)
        
        elif model_name == 'modnet':
            # MODNet pentru segmentarea portretelor
            try:
                import torch
                from torchvision import transforms
                
                # Adăugăm calea pentru MODNet
                mod_path = os.path.join(config.MODEL_DIR, "MODNet")
                if not os.path.exists(mod_path):
                    os.makedirs(mod_path, exist_ok=True)
                    
                # Clone the repository if it doesn't exist
                if not os.path.exists(os.path.join(mod_path, "src")):
                    os.system(f"git clone https://github.com/ZHKKKe/MODNet.git {mod_path}")
                
                sys.path.append(mod_path)
                try:
                    from src.models.modnet import MODNet
                except ImportError:
                    print("Could not import MODNet. Using fallback...")
                    self.models['modnet'] = None
                    return None
                
                model = MODNet(backbone_pretrained=False)
                model_path = os.path.join(config.MODNET_MODEL_PATH, "modnet_photographic_portrait_matting.pth")
                
                if not os.path.exists(model_path):
                    print("MODNet model not found. Downloading...")
                    ModelDownloader.download_file(
                        config.MODEL_URLS['modnet'],
                        model_path
                    )
                
                model.load_state_dict(torch.load(model_path, map_location=config.DEVICE))
                model.to(config.DEVICE)
                model.eval()
                
                self.models['modnet'] = {
                    'model': model,
                    'transform': transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                    ])
                }
                print(f"MODNet model loaded successfully")
            except Exception as e:
                print(f"Could not load MODNet: {e}")
                self.models['modnet'] = None
        
        elif model_name == 'gpen':
            # GPEN pentru restaurare fețe
            try:
                gpen_path = os.path.join(config.MODEL_DIR, "GPEN")
                if not os.path.exists(gpen_path):
                    os.makedirs(gpen_path, exist_ok=True)
                    os.system(f"git clone https://github.com/yangxy/GPEN.git {gpen_path}")
                
                sys.path.append(gpen_path)
                try:
                    from face_enhancement import FaceEnhancement
                except ImportError:
                    print("Could not import GPEN modules. Using fallback...")
                    self.models['gpen'] = None
                    return None
                
                model_path = os.path.join(config.GPEN_MODEL_PATH, "GPEN-BFR-512.pth")
                if not os.path.exists(model_path):
                    print("GPEN model not found. Downloading...")
                    ModelDownloader.download_file(
                        config.MODEL_URLS['gpen'],
                        model_path
                    )
                
                self.models['gpen'] = FaceEnhancement(
                    model_path,
                    config.DEVICE,
                    aligned=False
                )
                print(f"GPEN model loaded successfully")
            except Exception as e:
                print(f"Could not load GPEN: {e}")
                self.models['gpen'] = None
        
        elif model_name == 'codeformer':
            # CodeFormer pentru restaurare fețe
            try:
                codeformer_path = os.path.join(config.MODEL_DIR, "CodeFormer")
                if not os.path.exists(codeformer_path):
                    os.makedirs(codeformer_path, exist_ok=True)
                    os.system(f"git clone https://github.com/sczhou/CodeFormer.git {codeformer_path}")
                
                sys.path.append(codeformer_path)
                try:
                    from basicsr.archs.codeformer_arch import CodeFormer
                except ImportError:
                    print("Could not import CodeFormer modules. Using fallback...")
                    self.models['codeformer'] = None
                    return None
                
                model = CodeFormer(dim_embd=512, codebook_size=1024, n_head=8, n_layers=9)
                
                model_path = os.path.join(config.CODEFORMER_MODEL_PATH, "codeformer.pth")
                if not os.path.exists(model_path):
                    print("CodeFormer model not found. Downloading...")
                    ModelDownloader.download_file(
                        config.MODEL_URLS['codeformer'],
                        model_path
                    )
                
                checkpoint = torch.load(model_path, map_location=config.DEVICE)
                if 'params' in checkpoint:
                    model.load_state_dict(checkpoint['params'])
                else:
                    model.load_state_dict(checkpoint)
                    
                model.to(config.DEVICE)
                model.eval()
                
                self.models['codeformer'] = model
                print(f"CodeFormer model loaded successfully")
            except Exception as e:
                print(f"Could not load CodeFormer: {e}")
                self.models['codeformer'] = None
        
        elif model_name == 'esrgan':
            # Real-ESRGAN super-resolution
            try:
                esrgan_path = os.path.join(config.MODEL_DIR, "Real-ESRGAN")
                if not os.path.exists(esrgan_path):
                    os.makedirs(esrgan_path, exist_ok=True)
                    os.system(f"git clone https://github.com/xinntao/Real-ESRGAN.git {esrgan_path}")
                
                sys.path.append(esrgan_path)
                try:
                    from basicsr.archs.rrdbnet_arch import RRDBNet
                    from realesrgan import RealESRGANer
                except ImportError:
                    print("Could not import Real-ESRGAN modules. Using fallback...")
                    self.models['esrgan'] = None
                    return None
                
                model = RRDBNet(3, 3, 64, 23)
                
                model_path = os.path.join(config.ESRGAN_MODEL_PATH, "RealESRGAN_x4plus.pth")
                if not os.path.exists(model_path):
                    print("Real-ESRGAN model not found. Downloading...")
                    ModelDownloader.download_file(
                        config.MODEL_URLS['esrgan'],
                        model_path
                    )
                
                self.models['esrgan'] = RealESRGANer(
                    scale=4,
                    model_path=model_path,
                    model=model,
                    device=config.DEVICE
                )
                print(f"Real-ESRGAN model loaded successfully")
            except Exception as e:
                print(f"Could not load Real-ESRGAN: {e}")
                self.models['esrgan'] = None
        
        elif model_name == 'rembg':
            # REMBG pentru eliminarea fundalului
            try:
                import rembg
                self.models['rembg'] = rembg
                print(f"REMBG model loaded successfully")
            except ImportError:
                print("Could not import rembg. Installing...")
                os.system("pip install rembg")
                try:
                    import rembg
                    self.models['rembg'] = rembg
                except ImportError:
                    print("Failed to load rembg. Using fallback...")
                    self.models['rembg'] = None
        
        return self.models.get(model_name)
    
    def unload_model(self, model_name):
        """Descarcă un model pentru a elibera memorie"""
        if model_name in self.models and self.models[model_name] is not None:
            self.models[model_name] = None
            torch.cuda.empty_cache()
            gc.collect()
            print(f"Unloaded {model_name} model")


class EnhancedImageEditor:
    """Clasa principală pentru editarea imaginilor cu AI"""
    def __init__(self):
        self.models = {}
        self.model_manager = AdvancedModelManager()
        self.progress = ProgressMonitor(total_steps=4, description="Loading models")
        self.load_models()
        
    def load_models(self):
        """Încarcă modelele necesare"""
        try:
            self.progress.start()
            
            # 1. VAE (Autoencoder pentru SDXL)
            self.progress.set_description("Loading VAE")
            vae = AutoencoderKL.from_pretrained(
                "stabilityai/sdxl-vae", 
                torch_dtype=config.DTYPE,
                cache_dir=config.CACHE_DIR
            ).to(config.DEVICE)
            self.progress.update()

            # 2. ControlNet compatibil cu SDXL
            self.progress.set_description("Loading ControlNet")
            try:
                controlnet = ControlNetModel.from_pretrained(
                    CONTROLNET_MODEL,
                    torch_dtype=config.DTYPE,
                    use_safetensors=True,
                    variant="fp16" if config.DTYPE == torch.float16 else None,
                    cache_dir=config.CACHE_DIR
                ).to(config.DEVICE)
            except Exception as e:
                print(f"Error loading ControlNet: {e}")
                print("Using default configuration instead...")
                controlnet = None
            self.progress.update()

            # 3. SDXL Inpaint Pipeline
            self.progress.set_description("Loading SDXL")
            try:
                self.models['sdxl'] = StableDiffusionXLInpaintPipeline.from_pretrained(
                    "stabilityai/stable-diffusion-xl-base-1.0",
                    vae=vae,
                    torch_dtype=config.DTYPE,
                    variant="fp16" if config.DTYPE == torch.float16 else None,
                    use_safetensors=True,
                    cache_dir=config.CACHE_DIR
                )
                
                # Adăugăm controlnet la pipeline dacă este disponibil
                if controlnet is not None:
                    self.models['sdxl'].controlnet = controlnet

                # Optimizări de performanță (dacă e suficient VRAM)
                if config.LOW_VRAM_MODE:
                    self.models['sdxl'].enable_model_cpu_offload()
                else:
                    self.models['sdxl'].to(config.DEVICE)

                # Optimizarea scheduler-ului
                self.models['sdxl'].scheduler = DPMSolverMultistepScheduler.from_config(
                    self.models['sdxl'].scheduler.config,
                    algorithm_type="sde-dpmsolver++",
                    use_karras_sigmas=True
                )
            except Exception as e:
                print(f"Error loading SDXL: {e}")
                print("Using a simpler configuration...")
                
                # Încercăm cu o configurație mai simplă
                self.models['sdxl'] = StableDiffusionXLInpaintPipeline.from_pretrained(
                    "stabilityai/stable-diffusion-xl-base-1.0",
                    torch_dtype=config.DTYPE,
                    cache_dir=config.CACHE_DIR
                ).to(config.DEVICE)
                
                # Optimizare simplă scheduler
                self.models['sdxl'].scheduler = EulerAncestralDiscreteScheduler.from_config(
                    self.models['sdxl'].scheduler.config
                )
                
            self.progress.update()

            # 4. SAM - pentru segmentare avansată
            self.progress.set_description("Loading segmentation models")
            if SAM_AVAILABLE:
                self.models['sam'] = self._load_sam()
            else:
                self.models['sam'] = None

            # 5. YOLO - pentru detecția obiectelor
            if YOLO_AVAILABLE:
                try:
                    self.models['yolo'] = YOLO(MODEL_PATHS["yolo"])
                    # Forțăm YOLO să folosească același device
                    if hasattr(self.models['yolo'].model, 'to'):
                        self.models['yolo'].model.to(device=config.DEVICE)
                except Exception as e:
                    print(f"Error loading YOLO: {e}")
                    self.models['yolo'] = None
            else:
                self.models['yolo'] = None

            # 6. CLIPSeg - pentru segmentare bazată pe text
            self.models['clipseg'] = self._load_clipseg()

            # 7. MediaPipe - pentru segmentarea corpului
            try:
                self.models['mediapipe'] = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)
            except Exception as e:
                print(f"Error loading MediaPipe selfie segmentation: {e}")
                self.models['mediapipe'] = None
            
            # 8. Detector de față pentru segmentare îmbunătățită
            try:
                self.models['face_detector'] = mp.solutions.face_detection.FaceDetection(
                    model_selection=1,
                    min_detection_confidence=0.5
                )
            except Exception as e:
                print(f"Error loading MediaPipe face detection: {e}")
                self.models['face_detector'] = None
            
            self.progress.update()
            
        except Exception as e:
            raise RuntimeError(f"Model loading failed: {str(e)}")
        finally:
            self.progress.close()

    def _load_sam(self):
        """Încarcă modelul SAM cu optimizări de memorie sau folosește o alternativă"""
        # URL direct de descărcare
        SAM_MODEL_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
        
        # Verificăm dacă modelul există
        if not os.path.exists(MODEL_PATHS["sam"]):
            try:
                # Creăm directorul pentru model dacă nu există
                model_dir = os.path.dirname(MODEL_PATHS["sam"])
                if model_dir and not os.path.exists(model_dir):
                    os.makedirs(model_dir)
                
                # Descărcăm modelul
                print(f"[INFO] Downloading SAM model from {SAM_MODEL_URL}...")
                
                import urllib.request
                urllib.request.urlretrieve(SAM_MODEL_URL, MODEL_PATHS["sam"])
                
                print(f"[INFO] SAM model successfully downloaded to {MODEL_PATHS['sam']}")
            except Exception as e:
                print(f"[WARN] Couldn't download SAM model: {str(e)}")
                print(f"[WARN] Continuing without SAM capabilities")
                return None
        
        try:
            # Încărcăm modelul
            sam = sam_model_registry["vit_h"](checkpoint=MODEL_PATHS["sam"])
            sam.to(config.DEVICE)

            return SamAutomaticMaskGenerator(
                sam,
                points_per_side=32,
                pred_iou_thresh=0.95,
                stability_score_thresh=0.97,
                crop_n_layers=1,
                crop_n_points_downscale_factor=2,
                min_mask_region_area=100
            )
        except Exception as e:
            print(f"[WARN] Error loading SAM model: {str(e)}")
            print(f"[WARN] Continuing without SAM capabilities")
            return None

    def _load_clipseg(self):
        """Încarcă CLIPSeg pentru segmentare bazată pe text cu rezolvarea erorilor de tip"""
        try:
            processor = CLIPSegProcessor.from_pretrained(
                "CIDAS/clipseg-rd64-refined",
                cache_dir=config.CACHE_DIR
            )
            model = CLIPSegForImageSegmentation.from_pretrained(
                "CIDAS/clipseg-rd64-refined", 
                torch_dtype=config.DTYPE,
                cache_dir=config.CACHE_DIR
            ).to(config.DEVICE)
            
            # Ne asigurăm că modelul și parametrii folosesc același tip de date
            for param in model.parameters():
                if param.requires_grad:
                    param.data = param.data.to(dtype=config.DTYPE)
                    
            return {'processor': processor, 'model': model}
        except Exception as e:
            print(f"Error loading CLIPSeg: {e}")
            return None
    
    def get_specialized_model(self, model_name):
        """Obține un model specializat din managerul de modele"""
        return self.model_manager.load_model(model_name)


class ImageAnalyzer:
    """Clasă pentru analizarea contextului și conținutului imaginii"""
    @staticmethod
    def analyze_image_context(image):
        """Analizează contextul imaginii pentru o mai bună formulare a prompt-ului"""
        if isinstance(image, Image.Image):
            image_np = np.array(image)
        else:
            image_np = image
            
        # Analizează iluminarea
        lighting = ImageAnalyzer.analyze_lighting(image_np)
        
        # Analizează tipul de scenă
        scene_type = ImageAnalyzer.detect_scene_type(image_np)
        
        # Analizează stilul
        style = ImageAnalyzer.detect_photo_style(image_np)
        
        # Combinăm în descrierea contextului
        context = f"{scene_type} scene with {lighting} lighting in {style} style"
        
        return context
    
    @staticmethod
    def analyze_lighting(image_np):
        """Analizează condițiile de iluminare din imagine"""
        # Convertim la tonuri de gri
        if len(image_np.shape) == 3:
            gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
        else:
            gray = image_np
        
        # Calculăm luminozitatea medie
        mean_brightness = np.mean(gray)
        
        # Calculăm variația luminozității
        brightness_var = np.var(gray)
        
        # Determinăm tipul de iluminare bazat pe luminozitate și variație
        if mean_brightness < 80:
            if brightness_var > 1500:
                return "low-key dramatic"
            else:
                return "dim"
        elif mean_brightness > 180:
            if brightness_var < 1000:
                return "flat bright"
            else:
                return "bright"
        else:
            if brightness_var > 2000:
                return "high contrast"
            else:
                return "balanced"
    
    @staticmethod
    def detect_scene_type(image_np):
        """Detectează dacă scena este interior, exterior, portret, etc."""
        # Euristică simplă bazată pe distribuțiile de culori
        hsv = cv2.cvtColor(image_np, cv2.COLOR_BGR2HSV)
        
        # Extragem canalele
        h, s, v = cv2.split(hsv)
        
        # Calculăm saturația și valoarea medie
        mean_s = np.mean(s)
        mean_v = np.mean(v)
        
        # Verificăm prezența feței
        face_detector = mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.5)
        results = face_detector.process(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))
        
        if results and hasattr(results, 'detections') and results.detections:
            # Față detectată, probabil un portret
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                face_area = bbox.width * bbox.height
                if face_area > 0.15:  # Dacă fața ocupă mai mult de 15% din imagine
                    return "portrait"
            return "portrait"
        
        # Fără față sau față mică, determinăm interior/exterior
        if mean_s < 50:  # Saturație scăzută indică deseori scene de interior
            return "indoor"
        else:
            return "outdoor"
    
    @staticmethod
    def detect_photo_style(image_np):
        """Detectează stilul fotografic al imaginii"""
        # Euristică simplă bazată pe caracteristicile de culoare
        hsv = cv2.cvtColor(image_np, cv2.COLOR_BGR2HSV)
        
        # Extragem canalele
        h, s, v = cv2.split(hsv)
        
        # Calculăm metrici
        mean_s = np.mean(s)
        std_h = np.std(h)
        mean_v = np.mean(v)
        std_v = np.std(v)
        
        # Determinăm stilul bazat pe metrici
        if std_h < 20 and mean_s < 40:
            return "minimalist"
        elif std_v > 60:
            return "high contrast"
        elif mean_v > 200:
            return "bright and airy"
        elif mean_v < 100:
            return "moody"
        else:
            return "natural"
    
    @staticmethod
    def calculate_masked_ssim(original, generated, mask):
        """Calculează SSIM pentru regiunea mascată"""
        if isinstance(original, Image.Image):
            original = np.array(original)
        if isinstance(generated, Image.Image):
            generated = np.array(generated)
        if isinstance(mask, Image.Image):
            mask = np.array(mask)
            
        # Convertim la tonuri de gri dacă e necesar
        if len(original.shape) == 3:
            original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        else:
            original_gray = original
            
        if len(generated.shape) == 3:
            generated_gray = cv2.cvtColor(generated, cv2.COLOR_BGR2GRAY)
        else:
            generated_gray = generated
            
        # Ne asigurăm că masca este binară și de aceeași dimensiune cu imaginile
        if mask.shape != original_gray.shape:
            mask = cv2.resize(mask, (original_gray.shape[1], original_gray.shape[0]))
        
        if mask.max() > 1:
            mask = mask / 255.0
            
        # Aplicăm masca
        masked_original = original_gray * mask
        masked_generated = generated_gray * mask
        
        # Calculăm SSIM
        try:
            score, _ = ssim(masked_original, masked_generated, full=True)
            return score
        except Exception as e:
            print(f"Error calculating SSIM: {e}")
            return 0.5
    
    @staticmethod
    def detect_face_anomalies(image):
        """Detectează anomalii în trăsăturile faciale"""
        if isinstance(image, Image.Image):
            image_np = np.array(image)
        else:
            image_np = image
            
        # Inițializăm mediapipe face mesh
        mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            min_detection_confidence=0.5
        )
        
        # Procesăm imaginea
        results = mp_face_mesh.process(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))
        
        # Verificăm punctele de reper ale feței
        if not results or not hasattr(results, 'multi_face_landmarks') or not results.multi_face_landmarks:
            return 0.0  # Nicio față detectată
            
        # Obținem punctele de reper ale feței
        face_landmarks = results.multi_face_landmarks[0]
        
        # Extragem reperele pentru ochi
        left_eye = [(landmark.x, landmark.y) for i, landmark in enumerate(face_landmarks.landmark) if i in [33, 133]]
        right_eye = [(landmark.x, landmark.y) for i, landmark in enumerate(face_landmarks.landmark) if i in [362, 263]]
        
        # Calculăm distanța dintre ochi
        if left_eye and right_eye:
            left_eye_center = np.mean(left_eye, axis=0)
            right_eye_center = np.mean(right_eye, axis=0)
            eye_distance = np.linalg.norm(left_eye_center - right_eye_center)
            
            # Verificăm dacă ochii sunt prea departe sau prea aproape
            if eye_distance > 0.4 or eye_distance < 0.2:
                return 0.8  # Scor de anomalie ridicat
                
        # Mai multe verificări pot fi adăugate aici
        
        return 0.1  # Scor de anomalie scăzut dacă nu sunt detectate probleme


class AdvancedImageProcessor:
    """Clasă pentru procesarea avansată a imaginilor cu diferite pipeline-uri specializate"""
    def __init__(self, editor):
        self.editor = editor
        self.progress = None
        self.image_analyzer = ImageAnalyzer()
        
    def process_image(self, image, prompt, strength, operation_type=None, progress=None):
        """Procesează imaginea cu un pipeline specializat bazat pe tipul operației"""
        # Analizăm operația
        operation = self._classify_operation(prompt)
        
        # Suprascriem operation_type dacă este furnizat
        if operation_type:
            operation['type'] = operation_type
        
        # Selectăm pipeline-ul specializat
        if operation['type'] == 'remove':
            if 'person' in operation['target'].lower():
                return self.remove_person_pipeline(image, operation, strength, progress)
            else:
                return self.remove_object_pipeline(image, operation, strength, progress)
        elif operation['type'] == 'color':
            if 'hair' in operation['target'].lower():
                return self.change_hair_color_pipeline(image, operation, strength, progress)
            else:
                return self.change_color_pipeline(image, operation, strength, progress)
        elif operation['type'] == 'background':
            return self.replace_background_pipeline(image, operation, strength, progress)
        elif operation['type'] == 'add' or 'glasses' in prompt.lower():
            return self.add_object_pipeline(image, operation, strength, progress)
        else:
            # Pipeline implicit pentru alte operații
            return self.general_edit_pipeline(image, operation, strength, progress)
    
    def remove_person_pipeline(self, image, operation, strength, progress=None):
        """Pipeline specializat pentru eliminarea persoanelor"""
        if progress:
            progress(0.1, desc="Analyzing portrait...")
        
        # Convertim la numpy array dacă este PIL
        if isinstance(image, Image.Image):
            image_np = np.array(image)
            pil_image = image
        else:
            image_np = image
            pil_image = Image.fromarray(image_np)
        
        # Folosim MediaPipe pentru segmentarea persoanei
        person_mask = self._process_mediapipe(image_np)
        
        if person_mask is None or np.sum(person_mask) < 100:
            # Folosim CLIPSeg ca alternativă
            person_mask = self._process_clipseg(image_np, "person")
        
        # Ne asigurăm că masca este validă
        if person_mask is None or np.sum(person_mask) < 100:
            # Creăm o mască de rezervă
            h, w = image_np.shape[:2]
            person_mask = np.zeros((h, w), dtype=np.uint8)
            center_x, center_y = w // 2, h // 2
            center_radius = min(w, h) // 3
            cv2.circle(person_mask, (center_x, center_y), center_radius, 255, -1)
        
        # Îmbunătățim masca
        enhanced_mask = self.enhance_mask_for_removal(image_np, person_mask)
        
        if progress:
            progress(0.3, desc="Generating background...")
        
        # Încercăm să folosim LaMa pentru inpainting inițial
        try:
            lama_model = self.editor.get_specialized_model('lama')
            if lama_model:
                # Convertim masca la binar
                binary_mask = (enhanced_mask > 127).astype(np.uint8) * 255
                # Folosim LaMa pentru inpainting inițial
                lama_result = lama_model(image_np, binary_mask)
                initial_result = lama_result
            else:
                initial_result = image_np
        except Exception as e:
            print(f"Error using LaMa: {e}")
            initial_result = image_np
        
        if progress:
            progress(0.5, desc="Refining details...")
        
        # Ajustăm parametrii
        params = {
            'num_inference_steps': 70,
            'guidance_scale': 10.0,
            'strength': max(0.95, strength),  # Foarte puternic pentru eliminare completă
            'controlnet_conditioning_scale': 0.4
        }
        
        # Generăm prompt-ul îmbunătățit
        context = self.image_analyzer.analyze_image_context(image)
        enhanced_prompt = f"empty scene without any person, clean background, {context}"
        
        if progress:
            progress(0.7, desc="Final generation...")
        
        # Rafinare finală cu SDXL
        try:
            # Pregătim pentru SDXL
            pil_initial = Image.fromarray(initial_result)
            pil_mask = Image.fromarray(enhanced_mask)
            
            # Generăm cu SDXL
            result = self.editor.models['sdxl'](
                prompt=enhanced_prompt,
                negative_prompt="person, human, face, body, distortion, artifact, blurry",
                image=pil_initial,
                mask_image=pil_mask,
                num_inference_steps=params['num_inference_steps'],
                guidance_scale=params['guidance_scale'],
                strength=params['strength'],
                controlnet_conditioning_scale=params.get('controlnet_conditioning_scale', 0.8) 
                if hasattr(self.editor.models['sdxl'], 'controlnet') else None
            ).images[0]
        except Exception as e:
            print(f"Error in final generation: {e}")
            result = Image.fromarray(initial_result)
        
        if progress:
            progress(1.0, desc="Processing complete!")
        
        return result, Image.fromarray(enhanced_mask), operation, "Processing complete!"
    
    def remove_object_pipeline(self, image, operation, strength, progress=None):
        """Pipeline pentru eliminarea obiectelor"""
        if progress:
            progress(0.1, desc="Analyzing object...")
        
        # Convertim la numpy array dacă este PIL
        if isinstance(image, Image.Image):
            image_np = np.array(image)
            pil_image = image
        else:
            image_np = image
            pil_image = Image.fromarray(image_np)
        
        # Generăm masca hibridă care vizează obiectul
        mask = self.generate_hybrid_mask(image_np, operation['target'], operation['type'])
        
        if progress:
            progress(0.3, desc="Refining mask...")
        
        # Rafinăm masca
        refined_mask = self.refine_mask(mask, image_np)
        
        if progress:
            progress(0.5, desc="Initial inpainting...")
        
        # Încercăm să folosim LaMa pentru inpainting inițial
        try:
            lama_model = self.editor.get_specialized_model('lama')
            if lama_model:
                # Folosim LaMa pentru inpainting inițial
                initial_result = lama_model(image_np, refined_mask)
            else:
                initial_result = image_np
        except Exception as e:
            print(f"Error using LaMa: {e}")
            initial_result = image_np
        
        if progress:
            progress(0.7, desc="Final refinement...")
        
        # Generăm prompt-ul conștient de context
        context = self.image_analyzer.analyze_image_context(image)
        enhanced_prompt = f"scene without {operation['target']}, clean area, {context}"
        
        # Ajustăm parametrii
        params = {
            'num_inference_steps': 60,
            'guidance_scale': 9.0,
            'strength': max(0.9, strength),
            'controlnet_conditioning_scale': 0.6
        }
        
        # Rafinare finală cu SDXL
        try:
            # Pregătim pentru SDXL
            pil_initial = Image.fromarray(initial_result)
            pil_mask = Image.fromarray(refined_mask)
            
            # Generăm cu SDXL
            result = self.editor.models['sdxl'](
                prompt=enhanced_prompt,
                negative_prompt=f"{operation['target']}, distortion, artifact, blurry",
                image=pil_initial,
                mask_image=pil_mask,
                num_inference_steps=params['num_inference_steps'],
                guidance_scale=params['guidance_scale'],
                strength=params['strength'],
                controlnet_conditioning_scale=params.get('controlnet_conditioning_scale', 0.8)
                if hasattr(self.editor.models['sdxl'], 'controlnet') else None
            ).images[0]
        except Exception as e:
            print(f"Error in final generation: {e}")
            result = Image.fromarray(initial_result)
        
        if progress:
            progress(1.0, desc="Processing complete!")
        
        return result, Image.fromarray(refined_mask), operation, "Processing complete!"
    
    def change_hair_color_pipeline(self, image, operation, strength, progress=None):
        """Pipeline specializat pentru schimbarea culorii părului"""
        if progress:
            progress(0.1, desc="Analyzing hair...")
        
        # Convertim la numpy array dacă este PIL
        if isinstance(image, Image.Image):
            image_np = np.array(image)
            pil_image = image
        else:
            image_np = image
            pil_image = Image.fromarray(image_np)
        
        # Folosim CLIPSeg pentru a obține masca părului
        hair_mask = self._process_clipseg(image_np, "hair")
        
        if hair_mask is None or np.sum(hair_mask) < 100:
            # Creăm o mască de rezervă
            h, w = image_np.shape[:2]
            hair_mask = np.zeros((h, w), dtype=np.uint8)
            # Mască simplă pentru partea de sus a capului
            cv2.rectangle(hair_mask, (0, 0), (w, h//3), 255, -1)
        
        if progress:
            progress(0.3, desc="Refining hair mask...")
        
        # Rafinăm masca
        refined_mask = self.refine_mask(hair_mask, image_np)
        
        if progress:
            progress(0.5, desc="Color transformation...")
        
        # Culoarea țintă
        target_color = operation['attribute']
        
        # Generăm prompt-ul conștient de context
        enhanced_prompt = f"{target_color} hair, natural looking, realistic {target_color} hair, matching lighting"
        
        # Ajustăm parametrii
        params = {
            'num_inference_steps': 50,
            'guidance_scale': 7.0,
            'strength': min(0.7, strength + 0.1),  # Valoare mai mică pentru a păstra trăsăturile
            'controlnet_conditioning_scale': 0.9  # Control mai mare pentru a se potrivi cu originalul
        }
        
        if progress:
            progress(0.7, desc="Applying color...")
        
        # Generăm cu SDXL
        try:
            # Pregătim pentru SDXL
            pil_mask = Image.fromarray(refined_mask)
            
            # Generăm cu SDXL
            result = self.editor.models['sdxl'](
                prompt=enhanced_prompt,
                negative_prompt="unrealistic hair, wig, bad hair, distortion, blurry",
                image=pil_image,
                mask_image=pil_mask,
                num_inference_steps=params['num_inference_steps'],
                guidance_scale=params['guidance_scale'],
                strength=params['strength'],
                controlnet_conditioning_scale=params.get('controlnet_conditioning_scale', 0.8)
                if hasattr(self.editor.models['sdxl'], 'controlnet') else None
            ).images[0]
            
            # Încercăm să îmbunătățim fața dacă este posibil
            try:
                gpen_model = self.editor.get_specialized_model('gpen')
                if gpen_model:
                    result = gpen_model.process(np.array(result))
                    result = Image.fromarray(result)
            except Exception as e:
                print(f"Error enhancing face: {e}")
                
        except Exception as e:
            print(f"Error in hair color generation: {e}")
            result = pil_image
        
        if progress:
            progress(1.0, desc="Processing complete!")
        
        return result, Image.fromarray(refined_mask), operation, "Processing complete!"
    
    def change_color_pipeline(self, image, operation, strength, progress=None):
        """Pipeline pentru schimbarea culorii obiectelor"""
        if progress:
            progress(0.1, desc="Analyzing target...")
        
        # Convertim la numpy array dacă este PIL
        if isinstance(image, Image.Image):
            image_np = np.array(image)
            pil_image = image
        else:
            image_np = image
            pil_image = Image.fromarray(image_np)
        
        # Generăm masca pentru obiectul țintă
        target_mask = self.generate_hybrid_mask(image_np, operation['target'], operation['type'])
        
        if progress:
            progress(0.3, desc="Refining mask...")
        
        # Rafinăm masca
        refined_mask = self.refine_mask(target_mask, image_np)
        
        if progress:
            progress(0.5, desc="Preparing color change...")
        
        # Culoarea țintă
        target_color = operation['attribute']
        
        # Generăm prompt-ul îmbunătățit
        enhanced_prompt = f"{target_color} {operation['target']}, realistic {target_color} color, natural texture"
        
        # Ajustăm parametrii
        params = {
            'num_inference_steps': 45,
            'guidance_scale': 7.0,
            'strength': min(0.6, strength),  # Valoare mai mică pentru a păstra structura
            'controlnet_conditioning_scale': 0.85  # Control mai mare pentru a se potrivi cu originalul
        }
        
        if progress:
            progress(0.7, desc="Applying color...")
        
        # Generăm cu SDXL
        try:
            # Pregătim pentru SDXL
            pil_mask = Image.fromarray(refined_mask)
            
            # Generăm cu SDXL
            result = self.editor.models['sdxl'](
                prompt=enhanced_prompt,
                negative_prompt="unrealistic color, distortion, blurry",
                image=pil_image,
                mask_image=pil_mask,
                num_inference_steps=params['num_inference_steps'],
                guidance_scale=params['guidance_scale'],
                strength=params['strength'],
                controlnet_conditioning_scale=params.get('controlnet_conditioning_scale', 0.8)
                if hasattr(self.editor.models['sdxl'], 'controlnet') else None
            ).images[0]
        except Exception as e:
            print(f"Error in color generation: {e}")
            result = pil_image
        
        if progress:
            progress(1.0, desc="Processing complete!")
        
        return result, Image.fromarray(refined_mask), operation, "Processing complete!"
    
    def replace_background_pipeline(self, image, operation, strength, progress=None):
        """Pipeline pentru înlocuirea fundalului"""
        if progress:
            progress(0.1, desc="Segmenting subject...")
        
        # Convertim la numpy array dacă este PIL
        if isinstance(image, Image.Image):
            image_np = np.array(image)
            pil_image = image
        else:
            image_np = image
            pil_image = Image.fromarray(image_np)
        
        # Folosim REMBG pentru extracția precisă a subiectului
        try:
            rembg_model = self.editor.get_specialized_model('rembg')
            if rembg_model and rembg_model is not None:
                # Extragem subiectul
                subject_result = rembg_model.remove(image_np)
                # Creăm masca din canalul alpha
                if subject_result.shape[2] == 4:  # Are canal alpha
                    subject_mask = subject_result[:,:,3]
                else:
                    # Fallback la MediaPipe
                    subject_mask = self._process_mediapipe(image_np)
            else:
                # Fallback la MediaPipe
                subject_mask = self._process_mediapipe(image_np)
        except Exception as e:
            print(f"Error using REMBG: {e}")
            # Fallback la MediaPipe
            subject_mask = self._process_mediapipe(image_np)
        
        if subject_mask is None or np.sum(subject_mask) < 100:
            # Fallback la YOLO
            subject_mask = self._process_yolo(image_np, "person")
            
        if subject_mask is None or np.sum(subject_mask) < 100:
            # Creăm o mască de rezervă
            h, w = image_np.shape[:2]
            subject_mask = np.zeros((h, w), dtype=np.uint8)
            center_x, center_y = w // 2, h // 2
            center_radius = min(w, h) // 4
            cv2.circle(subject_mask, (center_x, center_y), center_radius, 255, -1)
        
        # Creăm masca de fundal (inversăm masca subiectului)
        background_mask = 255 - subject_mask
        
        if progress:
            progress(0.3, desc="Refining mask...")
        
        # Rafinăm masca de fundal
        refined_mask = self.refine_mask(background_mask, image_np)
        
        if progress:
            progress(0.5, desc="Generating background...")
        
        # Tema fundalului
        background_theme = operation['attribute']
        
        # Generăm prompt-ul pentru noul fundal
        enhanced_prompt = f"{background_theme} background, detailed {background_theme} scene, professional photography"
        
        # Ajustăm parametrii
        params = {
            'num_inference_steps': 65,
            'guidance_scale': 8.5,
            'strength': min(0.9, strength + 0.1),
            'controlnet_conditioning_scale': 0.7
        }
        
        if progress:
            progress(0.7, desc="Integrating background...")
        
        # Generăm cu SDXL
        try:
            # Pregătim pentru SDXL
            pil_mask = Image.fromarray(refined_mask)
            
            # Generăm cu SDXL
            result = self.editor.models['sdxl'](
                prompt=enhanced_prompt,
                negative_prompt="bad background, distortion, blurry, inconsistent lighting",
                image=pil_image,
                mask_image=pil_mask,
                num_inference_steps=params['num_inference_steps'],
                guidance_scale=params['guidance_scale'],
                strength=params['strength'],
                controlnet_conditioning_scale=params.get('controlnet_conditioning_scale', 0.8)
                if hasattr(self.editor.models['sdxl'], 'controlnet') else None
            ).images[0]
        except Exception as e:
            print(f"Error in background generation: {e}")
            result = pil_image
        
        if progress:
            progress(1.0, desc="Processing complete!")
        
        return result, Image.fromarray(refined_mask), operation, "Processing complete!"
    
    def add_object_pipeline(self, image, operation, strength, progress=None):
        """Pipeline pentru adăugarea obiectelor precum ochelari"""
        if progress:
            progress(0.1, desc="Analyzing face...")
            
        # Convertim la numpy array dacă este PIL
        if isinstance(image, Image.Image):
            image_np = np.array(image)
            pil_image = image
        else:
            image_np = image
            pil_image = Image.fromarray(image_np)
        
        # Detectăm regiunea feței
        face_mask = self._process_face_detection(image_np)
        
        if face_mask is None or np.sum(face_mask) < 100:
            # Fallback la MediaPipe
            face_mask = self._process_mediapipe(image_np)
            
        if face_mask is None or np.sum(face_mask) < 100:
            # Creăm o mască de rezervă
            h, w = image_np.shape[:2]
            face_mask = np.zeros((h, w), dtype=np.uint8)
            center_x, center_y = w // 2, h // 3  # Treimea superioară pentru față
            center_radius = min(w, h) // 5
            cv2.circle(face_mask, (center_x, center_y), center_radius, 255, -1)
        
        if progress:
            progress(0.3, desc="Creating mask...")
            
        # Rafinăm pentru a ne concentra pe regiunea ochilor pentru ochelari
        if "glasses" in operation.get('target', '').lower() or "glasses" in str(operation).lower():
            # Ne concentrăm pe regiunea ochilor
            h, w = face_mask.shape[:2]
            eye_region_mask = np.zeros_like(face_mask)
            
            # Regiunea ochilor este aproximativ în jumătatea superioară a feței
            face_bbox = cv2.boundingRect(face_mask.astype(np.uint8))
            x, y, fw, fh = face_bbox
            
            # Creăm masca pentru regiunea ochilor
            eye_y = y + fh // 3  # Ochii sunt la aproximativ 1/3 de sus față de fața
            eye_h = fh // 3      # Ochii ocupă aproximativ 1/3 din înălțimea feței
            cv2.rectangle(eye_region_mask, (x, eye_y), (x + fw, eye_y + eye_h), 255, -1)
            
            # Combinăm cu masca feței
            eye_mask = cv2.bitwise_and(face_mask, eye_region_mask)
            refined_mask = eye_mask
        else:
            refined_mask = face_mask
        
        if progress:
            progress(0.5, desc="Generating object...")
        
        # Generăm prompt-ul bazat pe ce trebuie adăugat
        if "glasses" in operation.get('target', '').lower() or "glasses" in str(operation).lower():
            enhanced_prompt = "face with stylish glasses, realistic glasses, detailed eyewear"
        else:
            enhanced_prompt = f"face with {operation.get('attribute', 'added object')}, realistic, high quality"
        
        # Ajustăm parametrii
        params = {
            'num_inference_steps': 55,
            'guidance_scale': 8.0,
            'strength': min(0.6, strength),  # Valoare mai mică pentru a păstra fața
            'controlnet_conditioning_scale': 0.85
        }
        
        if progress:
            progress(0.7, desc="Integrating object...")
        
        # Generăm cu SDXL
        try:
            # Pregătim pentru SDXL
            pil_mask = Image.fromarray(refined_mask)
            
            # Generăm cu SDXL
            result = self.editor.models['sdxl'](
                prompt=enhanced_prompt,
                negative_prompt="unrealistic, deformed, distorted, blurry, bad quality",
                image=pil_image,
                mask_image=pil_mask,
                num_inference_steps=params['num_inference_steps'],
                guidance_scale=params['guidance_scale'],
                strength=params['strength'],
                controlnet_conditioning_scale=params.get('controlnet_conditioning_scale', 0.8)
                if hasattr(self.editor.models['sdxl'], 'controlnet') else None
            ).images[0]
            
            # Încercăm să îmbunătățim fața dacă este posibil
            try:
                gpen_model = self.editor.get_specialized_model('gpen')
                if gpen_model:
                    result = gpen_model.process(np.array(result))
                    result = Image.fromarray(result)
            except Exception as e:
                print(f"Error enhancing face: {e}")
                
        except Exception as e:
            print(f"Error in object generation: {e}")
            result = pil_image
        
        if progress:
            progress(1.0, desc="Processing complete!")
        
        return result, Image.fromarray(refined_mask), operation, "Processing complete!"
    
    def general_edit_pipeline(self, image, operation, strength, progress=None):
        """Pipeline general pentru editări care nu sunt acoperite de pipeline-uri specializate"""
        if progress:
            progress(0.1, desc="Analyzing image...")
        
        # Convertim la numpy array dacă este PIL
        if isinstance(image, Image.Image):
            image_np = np.array(image)
            pil_image = image
        else:
            image_np = image
            pil_image = Image.fromarray(image_np)
        
        # Generăm masca bazată pe tipul operației
        mask = self.generate_hybrid_mask(image_np, operation['target'], operation['type'])
        
        if progress:
            progress(0.3, desc="Refining mask...")
        
        # Rafinăm masca
        refined_mask = self.refine_mask(mask, image_np)
        
        if progress:
            progress(0.5, desc="Enhancing prompt...")
        
        # Generăm prompt-ul îmbunătățit
        enhanced_prompt = self._enhance_prompt(operation)
        
        # Ajustăm parametrii bazați pe tipul operației
        params = self._get_generation_params(operation['type'])
        
        if progress:
            progress(0.7, desc="Generating result...")
        
        # Generăm cu SDXL
        try:
            # Pregătim pentru SDXL
            pil_mask = Image.fromarray(refined_mask)
            
            # Generăm cu SDXL
            result = self.editor.models['sdxl'](
                prompt=enhanced_prompt,
                negative_prompt="deformed, distorted, blurry, low quality, bad anatomy",
                image=pil_image,
                mask_image=pil_mask,
                num_inference_steps=params['num_inference_steps'],
                guidance_scale=params['guidance_scale'],
                strength=min(params['strength'], strength + 0.1),
                controlnet_conditioning_scale=params.get('controlnet_conditioning_scale', 0.8)
                if hasattr(self.editor.models['sdxl'], 'controlnet') else None
            ).images[0]
        except Exception as e:
            print(f"Error in generation: {e}")
            result = pil_image
        
        if progress:
            progress(1.0, desc="Processing complete!")
        
        return result, Image.fromarray(refined_mask), operation, "Processing complete!"
    
    def generate_hybrid_mask(self, image_np, prompt, operation_type):
        """Generează o mască hibridă optimizată cu gestionare specifică operației"""
        h, w = image_np.shape[:2]
        mask_weights = {
            'sam': 0.4,
            'yolo': 0.3,
            'clipseg': 0.2,
            'mediapipe': 0.1,
            'face': 0.2,
            'background': 0.5  # Pondere mai mare pentru detecția fundalului
        }
        final_mask = np.zeros((h, w), dtype=np.float32)
        
        # Inițializăm cu o mască de bază în caz că toate modelele eșuează
        backup_mask = np.ones((h, w), dtype=np.float32) * 0.5
        center_x, center_y = w // 2, h // 2
        center_radius = min(w, h) // 4
        cv2.circle(backup_mask, (center_x, center_y), center_radius, 1.0, -1)
        
        # Urmărim modelele de succes
        success_count = 0
        
        # Configurăm monitorizarea progresului
        self.progress = ProgressMonitor(total_steps=5, description="Generating mask")
        self.progress.start()
        
        # Gestionare specială pentru diferite tipuri de operații
        if operation_type == 'background':
            # Pentru fundal, folosim detecția fundalului ca o componentă cu pondere ridicată
            background_mask = self._process_background_detection(image_np)
            if background_mask is not None:
                # Deoarece operația de fundal inversează masca, trebuie să o inversăm și aici
                background_mask = 1.0 - background_mask  # Inversăm pentru selectarea subiectului
                final_mask += background_mask * mask_weights['background']
                success_count += 1
        elif operation_type == 'color' or operation_type == 'replace':
            # Pentru operațiile de culoare, creștem ponderea detecției fețelor dacă sunt menționate elemente faciale
            face_keywords = ['face', 'hair', 'eye', 'eyes', 'lips', 'mouth', 'nose', 'skin']
            if any(keyword in prompt.lower() for keyword in face_keywords):
                mask_weights['face'] = 0.6  # Prioritizăm detecția feței pentru editări legate de față
                # Folosim SAM cu o pondere mai mică pentru aceste operații pentru a evita supra-segmentarea
                mask_weights['sam'] = 0.2
        
        # Folosim thread pool pentru procesare paralelă
        with ThreadPoolExecutor(max_workers=config.MAX_WORKERS) as executor:
            futures = []
            future_to_model = {}
            
            # Trimitem sarcina SAM
            self.progress.set_description("Processing SAM")
            if self.editor.models['sam']:
                future = executor.submit(self._process_sam, image_np)
                futures.append(future)
                future_to_model[future] = 'sam'
            
            # Trimitem sarcina YOLO
            self.progress.set_description("Processing YOLO")
            future = executor.submit(self._process_yolo, image_np, prompt)
            futures.append(future)
            future_to_model[future] = 'yolo'
            
            # Trimitem sarcina CLIPSeg cu prompt-ul adecvat bazat pe operație
            self.progress.set_description("Processing CLIPSeg")
            # Folosim ținta sau atributul în funcție de tipul operației
            text_prompt = prompt
            if operation_type == 'replace' and ' with ' in prompt:
                # Pentru înlocuire, ne concentrăm pe țintă, nu pe înlocuire
                target_part = prompt.split(' with ')[0]
                if 'replace' in target_part:
                    text_prompt = target_part.split('replace ')[1].strip()
            elif operation_type == 'color' and ' to ' in prompt:
                # Pentru culoare, ne concentrăm pe obiect, nu pe culoare
                text_prompt = prompt.split(' to ')[0]
                if 'color' in text_prompt:
                    text_prompt = text_prompt.split('color ')[1].strip()
                elif 'make' in text_prompt:
                    text_prompt = text_prompt.split('make ')[1].strip()
                    
            future = executor.submit(self._process_clipseg, image_np, text_prompt)
            futures.append(future)
            future_to_model[future] = 'clipseg'
            
            # Trimitem sarcina MediaPipe
            self.progress.set_description("Processing MediaPipe")
            future = executor.submit(self._process_mediapipe, image_np)
            futures.append(future)
            future_to_model[future] = 'mediapipe'
            
            # Trimitem sarcina detecției fețelor
            self.progress.set_description("Processing face detection")
            future = executor.submit(self._process_face_detection, image_np)
            futures.append(future)
            future_to_model[future] = 'face'
            
            # Colectăm rezultatele și actualizăm masca
            for i, future in enumerate(futures):
                model_name = future_to_model[future]
                try:
                    mask = future.result(timeout=10)
                    if mask is not None and mask.shape[:2] == (h, w):
                        weight = mask_weights.get(model_name, 0.1)
                        final_mask += mask * weight
                        success_count += 1
                    self.progress.update()
                except Exception as e:
                    print(f"Error processing {model_name}: {str(e)}")
                    self.progress.update()
        
        # Dacă toate modelele au eșuat, folosim masca de rezervă
        if success_count == 0:
            final_mask = backup_mask
        
        # Normalizăm și stabilim pragul
        if final_mask.max() > 0:
            final_mask = (final_mask - final_mask.min()) / (final_mask.max() - final_mask.min())
        
        # Prag adaptiv cu ajustări specifice operației
        if operation_type == 'background':
            # Prag mai agresiv pentru fundal pentru a asigura margini curate
            dynamic_threshold = 0.35 + 0.45 * (1 - np.mean(final_mask))
        elif operation_type == 'color':
            # Prag mai scăzut pentru culoare pentru a asigura acoperirea completă
            dynamic_threshold = 0.25 + 0.35 * (1 - np.mean(final_mask))
        else:
            # Prag standard pentru alte operații
            dynamic_threshold = 0.3 + 0.4 * (1 - np.mean(final_mask))
            
        binary_mask = (final_mask > dynamic_threshold).astype(np.uint8) * 255
        
        # Operații morfologice cu ajustări specifice operației
        if operation_type == 'background':
            # Nucleu mai mare pentru operațiile de fundal
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
            binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
        elif operation_type == 'color':
            # Nucleu mai mic pentru operațiile de culoare pentru a păstra detaliile
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        else:
            # Nucleu standard pentru alte operații
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # Pentru operațiile de fundal, inversăm masca
        if operation_type == 'background':
            binary_mask = 255 - binary_mask
        
        self.progress.close()
        self.progress = None
        
        return binary_mask
    
    def enhance_mask_for_removal(self, image, mask):
        """Îmbunătățește masca specific pentru operațiile de eliminare"""
        # Convertim masca la binar dacă nu este deja
        if mask.max() > 1:
            binary_mask = (mask > 127).astype(np.uint8) * 255
        else:
            binary_mask = (mask > 0.5).astype(np.uint8) * 255
        
        # Dilatăm masca pentru a asigura acoperirea completă
        kernel = np.ones((15, 15), np.uint8)
        dilated_mask = cv2.dilate(binary_mask, kernel, iterations=1)
        
        # Netezim marginile
        blurred_mask = cv2.GaussianBlur(dilated_mask, (15, 15), 0)
        smoothed_mask = (blurred_mask > 127).astype(np.uint8) * 255
        
        return smoothed_mask
    
    def _process_sam(self, image_np):
        """Procesează generarea măștii SAM"""
        if self.editor.models['sam'] is None:
            return np.zeros_like(image_np[..., 0], dtype=np.float32)
            
        try:
            masks = self.editor.models['sam'].generate(image_np)
            combined_mask = np.zeros_like(image_np[..., 0], dtype=np.float32)
            
            for mask in masks:
                if mask['stability_score'] > 0.9:
                    resized_mask = self._safe_resize(mask['segmentation'].astype(np.float32), 
                                            (image_np.shape[1], image_np.shape[0]),
                                            interpolation=cv2.INTER_LINEAR)
                    combined_mask += resized_mask
                    
            return combined_mask
        except Exception as e:
            print(f"Error in SAM processing: {e}")
            return np.zeros_like(image_np[..., 0], dtype=np.float32)
    
    def _process_yolo(self, image_np, prompt):
        """Procesează segmentarea YOLO cu gestionarea erorilor"""
        try:
            # Sărim procesarea YOLO dacă există o încredere scăzută în capacitatea de a gestiona sarcina
            if prompt == '' or len(prompt) < 3 or self.editor.models['yolo'] is None:
                return np.zeros_like(image_np[..., 0], dtype=np.float32)
            
            # Încercăm să rulăm detecția YOLO
            results = self.editor.models['yolo'](
                image_np, 
                imgsz=960, 
                conf=0.3,
                iou=0.4,
                agnostic_nms=True,
                verbose=False
            )
            
            combined_mask = np.zeros_like(image_np[..., 0], dtype=np.float32)
            
            for result in results:
                if hasattr(result, 'masks') and result.masks is not None:
                    for mask, box in zip(result.masks.xy, result.boxes):
                        # Verificăm dacă se potrivește cu prompt-ul - corectat pentru a gestiona tipuri de date diferite
                        if self._is_relevant_detection(prompt, box.cls.item() if hasattr(box.cls, 'item') else box.cls):
                            poly = np.array(mask, np.int32).reshape((-1, 1, 2))
                            cv2.fillPoly(combined_mask, [poly], 1.0)
                            
            return combined_mask
        
        except Exception as e:
            print(f"Error processing YOLO: {str(e)}")
            # Returnăm o mască goală în caz de eroare
            return np.zeros_like(image_np[..., 0], dtype=np.float32)
    
    def _is_relevant_detection(self, prompt, class_id):
        """Verifică dacă clasa de detecție se potrivește cu prompt-ul cu gestionare sigură a tipului"""
        try:
            # Mapăm ID-urile de clasă YOLO la nume de clasă comune
            if self.editor.models['yolo'] is None or not hasattr(self.editor.models['yolo'], 'names'):
                return True  # Fallback sigur
                
            class_names = self.editor.models['yolo'].names
            
            # Verificăm dacă class_id este valid (ar putea fi un float)
            if isinstance(class_id, (int, float)) and int(class_id) in class_names:
                class_name = class_names[int(class_id)]
                # Verificăm dacă numele clasei apare în prompt
                prompt_lower = prompt.lower()
                class_lower = class_name.lower()
                
                return class_lower in prompt_lower or any(
                    word in prompt_lower for word in class_lower.split('_')
                )
            return True  # Implicit la True dacă nu se găsește nicio potrivire
        except Exception as e:
            print(f"Error in relevance detection: {str(e)}")
            return True  # Fallback sigur
    
    def _process_clipseg(self, image_np, prompt):
        """Procesează segmentarea CLIPSeg cu rezolvarea completă a erorilor de tip și gestionare robustă a redimensionării"""
        if self.editor.models['clipseg'] is None:
            return np.zeros_like(image_np[..., 0], dtype=np.float32)
            
        try:
            inputs = self.editor.models['clipseg']['processor'](
                text=prompt,
                images=Image.fromarray(image_np),
                return_tensors="pt",
                padding=True
            )
            
            # Rezolvă problemele de tip mai aprofundat - convertim la CPU, apoi la dtype corect, apoi înapoi la GPU
            input_fixed = {}
            for key, tensor in inputs.items():
                if torch.is_tensor(tensor):
                    # Mutăm la CPU, convertim la dtype corect, apoi înapoi la GPU cu dtype corect
                    if key in ['input_ids', 'attention_mask', 'position_ids'] or 'indices' in key:
                        # Aceste tensori trebuie să fie Long
                        input_fixed[key] = tensor.cpu().long().to(device=config.DEVICE)
                    else:
                        # Alți tensori pot folosi dtype-ul configurat
                        input_fixed[key] = tensor.cpu().to(dtype=config.DTYPE, device=config.DEVICE)
                else:
                    input_fixed[key] = tensor
            
            with torch.no_grad():
                outputs = self.editor.models['clipseg']['model'](**input_fixed)
                mask = torch.sigmoid(outputs.logits).squeeze().cpu().numpy()
            
            # Gestionare robustă a redimensionării - verificăm dacă masca este validă
            if mask is None or mask.size == 0:
                print("Warning: Empty mask from CLIPSeg")
                return np.zeros((image_np.shape[0], image_np.shape[1]), dtype=np.float32)
            
            # Ne asigurăm că masca are dimensiunile corecte pentru redimensionare
            if len(mask.shape) != 2:
                print(f"Warning: Unexpected mask shape: {mask.shape}")
                # Încercăm să corectăm forma măștii dacă este posibil
                if len(mask.shape) > 2:
                    mask = mask[0] if mask.shape[0] == 1 else np.mean(mask, axis=0)
                else:
                    return np.zeros((image_np.shape[0], image_np.shape[1]), dtype=np.float32)
            
            # Redimensionare sigură cu gestionarea erorilor
            return self._safe_resize(mask, (image_np.shape[1], image_np.shape[0]))
                
        except Exception as e:
            print(f"Error in CLIPSeg processing: {str(e)}")
            return np.zeros((image_np.shape[0], image_np.shape[1]), dtype=np.float32)  # Returnăm o mască goală în caz de eroare
    
    def _process_background_detection(self, image_np):
        """Creează o mască care separă subiectul de fundal"""
        h, w = image_np.shape[:2]
        
        # 1. Creăm o mască de fundal de bază conștientă de margini
        gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (15, 15), 0)
        edges = cv2.Canny(blurred, 50, 150)
        
        # Dilatăm marginile pentru a crea limite mai complete
        kernel = np.ones((5,5), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=2)
        
        # 2. Folosim grabcut pentru o separare mai avansată a fundalului
        mask = np.zeros(image_np.shape[:2], np.uint8)
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        
        # Creăm un dreptunghi care este puțin mai mic decât imaginea
        margin = int(min(w, h) * 0.1)
        rect = (margin, margin, w - margin*2, h - margin*2)
        
        try:
            # Aplicăm algoritmul grabcut
            cv2.grabCut(image_np, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
            
            # Creăm masca unde fundalul sigur și probabil sunt setate la 0, altfel 1
            mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('float32')
            
            # Combinăm cu marginile
            combined = mask2 * (1 - np.clip(edges.astype('float32') / 255.0, 0, 1) * 0.7)
            
            return combined
        except Exception as e:
            print(f"Error in background detection: {str(e)}")
            # Fallback la o mască simplă
            mask = np.ones((h, w), dtype=np.float32) * 0.5
            cv2.rectangle(mask, (margin, margin), (w - margin, h - margin), 1.0, -1)
            return mask
    
    def _process_mediapipe(self, image_np):
        """Procesează segmentarea MediaPipe"""
        if self.editor.models['mediapipe'] is None:
            return None
            
        try:
            results = self.editor.models['mediapipe'].process(
                cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
            )
            
            if results.segmentation_mask is not None:
                return self._safe_resize(
                    results.segmentation_mask.astype(np.float32),
                    (image_np.shape[1], image_np.shape[0])
                )
            return None
        except Exception as e:
            print(f"Error in MediaPipe processing: {e}")
            return None
    
    def _process_face_detection(self, image_np):
        """Procesează detecția fețelor pentru generarea măștii îmbunătățite"""
        if self.editor.models['face_detector'] is None:
            return None
            
        try:
            rgb_image = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
            results = self.editor.models['face_detector'].process(rgb_image)
            
            h, w = image_np.shape[:2]
            face_mask = np.zeros((h, w), dtype=np.float32)
            
            if hasattr(results, 'detections') and results.detections:
                for detection in results.detections:
                    bbox = detection.location_data.relative_bounding_box
                    x, y = int(bbox.xmin * w), int(bbox.ymin * h)
                    width, height = int(bbox.width * w), int(bbox.height * h)
                    
                    # Extindem ușor regiunea feței
                    expansion = 0.2
                    x_expanded = max(0, int(x - width * expansion))
                    y_expanded = max(0, int(y - height * expansion))
                    width_expanded = min(w - x_expanded, int(width * (1 + 2 * expansion)))
                    height_expanded = min(h - y_expanded, int(height * (1 + 2 * expansion)))
                    
                    # Creăm o mască eliptică pentru o selecție mai naturală a feței
                    center = (x_expanded + width_expanded // 2, y_expanded + height_expanded // 2)
                    axes = (width_expanded // 2, height_expanded // 2)
                    cv2.ellipse(face_mask, center, axes, 0, 0, 360, 1.0, -1)
            
            return face_mask
        except Exception as e:
            print(f"Error in face detection: {e}")
            return None
    
    def _safe_resize(self, image, size, interpolation=cv2.INTER_LINEAR):
        """Redimensionare sigură a imaginii cu multiple fallback-uri"""
        try:
            # Convertim la numpy dacă este PIL
            if isinstance(image, Image.Image):
                image_np = np.array(image)
            else:
                image_np = image
                
            # Verificăm dacă imaginea este validă
            if image_np is None or image_np.size == 0:
                # Creăm o imagine goală de dimensiunea țintă
                if len(size) == 2:
                    return np.zeros((size[1], size[0]), dtype=np.uint8)
                return None
                
            # Încercăm redimensionarea cv2
            return cv2.resize(image_np, size, interpolation=interpolation)
        except cv2.error as e:
            print(f"OpenCV resize error: {e}")
            try:
                # Fallback la PIL
                pil_img = Image.fromarray(image_np)
                resized = pil_img.resize(size, Image.LANCZOS)
                return np.array(resized)
            except Exception as e2:
                print(f"PIL resize error: {e2}")
                # Creăm o imagine goală de dimensiunea țintă ca ultimă soluție
                if len(image_np.shape) == 3:
                    return np.zeros((size[1], size[0], image_np.shape[2]), dtype=image_np.dtype)
                else:
                    return np.zeros((size[1], size[0]), dtype=image_np.dtype)
        except Exception as e:
            print(f"Unexpected resize error: {e}")
            # Creăm o imagine goală de dimensiunea țintă ca ultimă soluție
            return np.zeros((size[1], size[0]), dtype=np.uint8)

    def refine_mask(self, mask, image):
        """Rafinare avansată a măștii cu conștientizare a marginilor și algoritm îmbunătățit"""
        if self.progress is None:
            self.progress = ProgressMonitor(total_steps=5, description="Refining mask")
            self.progress.start()
        
        try:
            # Convertim la float32 pentru procesare
            if mask.max() > 1:
                mask = mask.astype(np.float32) / 255.0
            self.progress.update()
            
            # Detectia marginilor
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            edges = cv2.GaussianBlur(edges, (5, 5), 0) / 255.0
            self.progress.update()
            
            # Combinăm masca cu marginile
            refined = mask * (1 - edges * 0.3) + edges * 0.7
            self.progress.update()
            
            # Convertim la formatul potrivit pentru filtrul bilateral (8-bit unsigned integer)
            refined_uint8 = (refined * 255).astype(np.uint8)
            
            # Aplicăm filtrul bilateral pentru netezire cu păstrarea marginilor
            refined_filtered = cv2.bilateralFilter(refined_uint8, d=9, sigmaColor=75, sigmaSpace=75)
            self.progress.update()
            
            # Threshold
            try:
                adaptive_thresh = cv2.adaptiveThreshold(
                    refined_filtered,
                    255,
                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY,
                    101,
                    3
                )
            except cv2.error:
                # Fallback la threshold simplu dacă cel adaptiv eșuează
                _, adaptive_thresh = cv2.threshold(refined_filtered, 127, 255, cv2.THRESH_BINARY)
            
            # Eliminăm obiectele mici și umplem găurile
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            cleaned = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_OPEN, kernel, iterations=1)
            cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=3)
            
            # Netezire finală
            cleaned = cv2.GaussianBlur(cleaned, (3, 3), 0)
            _, cleaned = cv2.threshold(cleaned, 127, 255, cv2.THRESH_BINARY)
            self.progress.update()
            
        except Exception as e:
            print(f"Error in mask refinement: {str(e)}")
            # Returnăm masca originală dacă rafinarea eșuează
            if isinstance(mask, np.ndarray):
                if mask.max() <= 1:
                    cleaned = (mask * 255).astype(np.uint8)
                else:
                    cleaned = mask.astype(np.uint8)
            else:
                # Creăm o mască goală dacă totul eșuează
                cleaned = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        
        finally:
            if self.progress is not None:
                self.progress.close()
                self.progress = None
        
        return cleaned
    
    def _prepare_control_image(self, image, mask):
        """Pregătește imaginea de control bazată pe mască"""
        # Convertim imaginea la numpy array
        if isinstance(image, Image.Image):
            image_np = np.array(image)
        else:
            image_np = image
        
        # Convertim masca la numpy dacă este necesară
        if isinstance(mask, Image.Image):
            mask_np = np.array(mask)
        else:
            mask_np = mask
        
        # Ne asigurăm că masca este binară
        if mask_np.max() > 1 and mask_np.dtype != np.bool_:
            mask_np = mask_np > 127
        
        # Detecție de margini Canny
        gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
        
        # Praguri adaptive bazate pe conținutul imaginii
        median_value = np.median(gray)
        lower_threshold = max(0, int(median_value * 0.7))
        upper_threshold = min(255, int(median_value * 1.3))
        
        edges = cv2.Canny(gray, lower_threshold, upper_threshold)
        
        # Aplicăm masca la margini - ne concentrăm pe marginile din zona de interes
        masked_edges = cv2.bitwise_and(edges, edges, mask=mask_np.astype(np.uint8))
        
        # Blur ușor pentru a reduce zgomotul
        masked_edges = cv2.GaussianBlur(masked_edges, (3, 3), 0)
        
        # Convertim la imagine PIL
        return Image.fromarray(masked_edges)
    
    def _create_callback_fn(self):
        """Creează o funcție de callback pentru monitorizarea progresului de generare"""
        def callback_fn(step, timestep, latents):
            if self.progress is not None:
                self.progress.set_description(f"Generating image (step {step})")
        return callback_fn
    
    def _get_generation_params(self, operation_type):
        """Obține parametrii optimizați de generare"""
        base_params = {
            'num_inference_steps': 50,
            'guidance_scale': 7.5,
            'strength': 0.75,
            'controlnet_conditioning_scale': 0.8
        }
        
        # Ajustăm parametrii bazați pe tipul operației
        if operation_type == 'remove':
            base_params.update({
                'num_inference_steps': 60,
                'guidance_scale': 9.0,
                'strength': 0.85,
                'controlnet_conditioning_scale': 0.6
            })
        elif operation_type == 'replace':
            base_params.update({
                'num_inference_steps': 70,
                'guidance_scale': 10.0,
                'strength': 0.9,
                'controlnet_conditioning_scale': 0.9
            })
        elif operation_type == 'background':
            base_params.update({
                'num_inference_steps': 65,
                'guidance_scale': 8.5,
                'strength': 0.8,
                'controlnet_conditioning_scale': 0.7
            })
        elif operation_type == 'color':
            base_params.update({
                'num_inference_steps': 45,
                'guidance_scale': 7.0,
                'strength': 0.7,
                'controlnet_conditioning_scale': 0.5
            })
        
        return base_params
    
    def _build_negative_prompt(self):
        """Construiește un prompt negativ îmbunătățit"""
        return (
            "blurry, distorted, deformed, low quality, artifacts, "
            "extra limbs, extra digits, watermarks, text, signatures, "
            "poorly drawn face, bad anatomy, duplicate, mosaic, "
            "low resolution, grainy, noisy, overexposed, underexposed"
        )
        
    def _classify_operation(self, prompt):
        """Sistem de clasificare îmbunătățit pentru operații"""
        prompt = prompt.lower().strip()
        operation_map = {
            'remove': [
                (r"(remove|delete|erase)\s+(the\s+)?(?P<target>[a-z\s]+?)(\s+from\s+the\s+image)?$", "remove"),
                (r"eliminate\s+(the\s+)?(?P<target>[a-z\s]+)", "remove")
            ],
            'replace': [
                (r"(replace|swap|change)\s+(the\s+)?(?P<target>[a-z\s]+?)\s+with\s+(a\s+)?(?P<attr>[a-z\s]+)", "replace"),
                (r"substitute\s+(the\s+)?(?P<target>[a-z\s]+?)\s+for\s+(a\s+)?(?P<attr>[a-z\s]+)", "replace")
            ],
            'color': [
                (r"(color|recolor|change\s+color)\s+(the\s+)?(?P<target>[a-z\s]+?)\s+to\s+(?P<attr>[a-z]+)", "color"),
                (r"make\s+(the\s+)?(?P<target>[a-z\s]+?)\s+(?P<attr>[a-z]+)", "color")
            ],
            'background': [
                (r"(change|alter)\s+(the\s+)?background\s+to\s+(?P<attr>[a-z\s]+)", "background"),
                (r"new\s+background\s+(of|as)\s+(?P<attr>[a-z\s]+)", "background")
            ],
            'add': [
                (r"(add|place|put)\s+(a\s+)?(?P<attr>[a-z\s]+)", "add"),
                (r"(wear|wearing)\s+(a\s+)?(?P<attr>[a-z\s]+)", "add")
            ]
        }
        
        for op_type, patterns in operation_map.items():
            for pattern, match_type in patterns:
                match = re.search(pattern, prompt)
                if match:
                    groups = match.groupdict()
                    return {
                        'type': op_type,
                        'target': groups.get('target', '').strip(),
                        'attribute': groups.get('attr', '').strip(),
                        'confidence': 0.95
                    }
        
        return {'type': 'general', 'target': '', 'attribute': prompt, 'confidence': 0.5}
        
    def _enhance_prompt(self, operation):
        """Generator avansat de prompt-uri"""
        prompt_enhancers = {
            'remove': [
                "highly detailed",
                "seamless integration",
                "perfect edges",
                "no artifacts",
                "professional retouching"
            ],
            'replace': [
                "photorealistic",
                "perfect lighting matching",
                "accurate shadows",
                "consistent perspective",
                "high resolution detail"
            ],
            'color': [
                "vibrant colors",
                "natural gradients",
                "realistic textures",
                "accurate lighting",
                "high quality"
            ],
            'background': [
                "cinematic lighting",
                "ultra detailed",
                "professional photography",
                "8k resolution",
                "realistic environment"
            ],
            'add': [
                "realistic integration",
                "perfect placement",
                "natural appearance",
                "high quality details",
                "professional look"
            ],
            'general': [
                "high quality",
                "detailed",
                "professional",
                "sharp focus",
                "realistic"
            ]
        }
        
        base = ""
        if operation['attribute']:
            base = f"{operation['attribute']}, "
        elif operation['type'] == 'remove' and 'person' in operation['target'].lower():
            # Caz special pentru eliminarea persoanelor
            base = "empty scene, clean background, natural lighting, "
        
        enhancers = prompt_enhancers.get(operation['type'], prompt_enhancers['general'])
        return base + ", ".join(enhancers[:3])


# CSS pentru interfața Gradio
CSS_STYLES = """
.container { max-width: 1200px; margin: auto; }
.image-preview { height: 500px; }
.error { color: red; }
.progress-area { margin: 10px 0; padding: 10px; background: #f5f5f5; border-radius: 4px; }
.controls { display: flex; gap: 10px; margin-bottom: 15px; }
.info-panel { background: #e6f7ff; padding: 10px; border-radius: 4px; margin-top: 10px; }
.example-btn { margin: 5px; }
"""


def create_interface(editor):
    """Crează interfața Gradio îmbunătățită"""
    processor = AdvancedImageProcessor(editor)
    
    def process_image(image, prompt, strength, progress=gr.Progress()):
        if image is None:
            return None, None, {"error": "No image provided"}, "Processing cannot start without an image"
        
        try:
            progress(0, desc="Initializing...")
            
            # Convertim imaginea la numpy array
            image_np = np.array(image)
            
            # Analizăm promptul
            operation = processor._classify_operation(prompt)
            progress(0.1, desc="Analyzing prompt...")
            
            # Procesăm cu pipeline specializat
            result = processor.process_image(image, prompt, strength, progress)
            
            # Returnăm rezultatul
            return result
        except Exception as e:
            error_detail = {"error": str(e), "type": type(e).__name__}
            print(f"Error processing image: {str(e)}")
            return image, None, error_detail, f"Error: {str(e)}"
    
    # Creăm interfața
    with gr.Blocks(theme=gr.themes.Soft(), css=CSS_STYLES) as demo:
        gr.Markdown("# 🚀 ICEdit Pro - Advanced AI Image Editor")
        
        with gr.Row(equal_height=True):
            with gr.Column():
                image_input = gr.Image(type="pil", label="Upload Image", elem_classes="image-preview")
                
                with gr.Row(elem_classes="controls"):
                    prompt = gr.Textbox(
                        label="Edit Instructions", 
                        placeholder="E.g., 'Remove the car', 'Change hair color to blonde'",
                        elem_id="prompt-input"
                    )
                    strength = gr.Slider(
                        0.1, 1.0, 0.75, 
                        label="Edit Strength",
                        info="Higher values create more dramatic changes"
                    )
                
                run_btn = gr.Button("Generate Edit", variant="primary", elem_id="generate-btn")
                status_area = gr.Textbox(
                    label="Status", 
                    value="Ready", 
                    elem_classes="progress-area",
                    interactive=False
                )
                
            with gr.Column():
                image_output = gr.Image(label="Edited Result", elem_classes="image-preview")
                mask_output = gr.Image(label="Generated Mask")
                
                with gr.Accordion("Operation Details", open=False):
                    info = gr.JSON(label="Operation Analysis")
        
        with gr.Row():
            gr.Markdown("## Example Prompts")
        
        # Exemple de prompt-uri cu opțiuni extinse
        example_prompts = [
            # Operații de bază
            ["remove the car from the street", 0.8],
            ["change hair color to bright pink", 0.7],
            ["replace background with futuristic city", 0.85],
            ["remove watermark from top right corner", 0.75],
            # Exemple avansate adiționale
            ["remove person from the image", 0.9],
            ["change the shirt color to blue", 0.65],
            ["replace the sky with sunset colors", 0.8],
            ["erase all text from the image", 0.85],
            ["make the eyes green", 0.6],
            ["replace glasses with sunglasses", 0.75],
            ["add glasses", 0.65]
        ]
        
        # Creăm butoane pentru exemple direct, în loc să folosim gr.Examples
        example_rows = [example_prompts[i:i+4] for i in range(0, len(example_prompts), 4)]
        
        for row_examples in example_rows:
            with gr.Row():
                for ex_prompt, ex_strength in row_examples:
                    ex_btn = gr.Button(ex_prompt, elem_classes="example-btn")
                    # Definim o funcție separată pentru fiecare buton
                    ex_btn.click(
                        fn=lambda p=ex_prompt, s=ex_strength: [p, s],
                        inputs=None,
                        outputs=[prompt, strength]
                    )
        
        # Panou de informații
        with gr.Accordion("Tips & Info", open=False):
            gr.Markdown("""
            ### Tips for better results:
            - Be specific in your instructions (e.g., "remove the red car on the left" instead of just "remove car")
            - For replacing objects, specify what to replace them with
            - For color changes, specify the exact color (e.g., "bright pink", "deep blue")
            - Adjust strength slider for more or less dramatic changes
            - Check the generated mask to see what area will be edited
            
            ### Common operations:
            - **Remove**: "remove [object]"
            - **Replace**: "replace [object] with [new object]"
            - **Color Change**: "change color of [object] to [color]"
            - **Background Change**: "change background to [scene]"
            - **Add**: "add [object]" (e.g., "add glasses")
            """)
        
        run_btn.click(
            fn=process_image,
            inputs=[image_input, prompt, strength],
            outputs=[image_output, mask_output, info, status_area]
        )
    
    return demo


# Funcția principală
def main():
    """Funcția principală pentru a rula aplicația"""
    print("Initializing ICEdit Pro...")
    
    create_requirements_file()
    create_install_script()
    
    try:
        # Încercăm să inițializăm editorul
        editor = EnhancedImageEditor()
        app = create_interface(editor)
        
        # Rulăm aplicația
        app.launch(share=True, server_name="0.0.0.0", server_port=7860)
        
    except Exception as e:
        print(f"Error initializing application: {e}")
        print("Please ensure all dependencies are installed by running ./install.sh")


# Punct de intrare pentru script
if __name__ == "__main__":
    main()