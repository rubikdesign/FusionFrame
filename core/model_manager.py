#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Manager pentru modelele AI folosite în FusionFrame 2.0
"""

import os
import gc
import torch
import logging
import requests
import numpy as np
from tqdm.auto import tqdm
from pathlib import Path
from typing import Optional, Dict, Any, List, Union

from config.app_config import AppConfig
from config.model_config import ModelConfig

# Setăm logger-ul
logger = logging.getLogger(__name__)

class ModelManager:
    """
    Manager pentru modelele AI
    
    Responsabil pentru încărcarea, descărcarea și gestionarea modelelor AI
    folosite în aplicație. Implementează pattern-ul Singleton pentru a asigura
    o singură instanță a managerului în întreaga aplicație.
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self.models = {}
        self.model_paths = {}
        self.config = AppConfig
        self.model_config = ModelConfig
        
        # Asigură că directoarele pentru modele există
        self.config.ensure_dirs()
        
        # Inițializează dicționarul de modele
        self._initialized = True
        logger.info("ModelManager initialized")
    
    def download_file(self, url: str, destination: str) -> str:
        """
        Descarcă un fișier de la URL către destinație cu bare de progres
        
        Args:
            url: URL-ul fișierului de descărcat
            destination: Calea unde va fi salvat fișierul
            
        Returns:
            Calea către fișierul descărcat
        """
        logger.info(f"Downloading file from {url} to {destination}")
        
        # Creăm directorul destinație dacă nu există
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        
        # Descărcăm fișierul cu bare de progres
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        with tqdm(total=total_size, unit='B', unit_scale=True, 
                  desc=f"Downloading {os.path.basename(destination)}") as pbar:
            with open(destination, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        
        logger.info(f"Download complete: {destination}")
        return destination
    
    def download_and_extract(self, url: str, destination_dir: str) -> str:
        """
        Descarcă și extrage un fișier zip
        
        Args:
            url: URL-ul fișierului zip
            destination_dir: Directorul unde va fi extras fișierul
            
        Returns:
            Calea către directorul cu fișierele extrase
        """
        import zipfile
        
        logger.info(f"Downloading and extracting zip from {url} to {destination_dir}")
        
        # Creăm directorul destinație dacă nu există
        os.makedirs(destination_dir, exist_ok=True)
        
        # Descărcăm fișierul zip
        temp_zip = os.path.join(destination_dir, "temp.zip")
        self.download_file(url, temp_zip)
        
        # Extragem fișierul zip
        with zipfile.ZipFile(temp_zip, 'r') as zip_ref:
            zip_ref.extractall(destination_dir)
        
        # Ștergem fișierul temporar zip
        os.remove(temp_zip)
        
        logger.info(f"Extraction complete: {destination_dir}")
        return destination_dir
    
    def ensure_model(self, model_name: str) -> str:
        """
        Se asigură că un model specific este descărcat și disponibil
        
        Args:
            model_name: Numele modelului
            
        Returns:
            Calea către directorul modelului
        """
        model_path = os.path.join(self.config.MODEL_DIR, model_name)
        model_url = self.config.MODEL_URLS.get(model_name)
        
        if not model_url:
            logger.warning(f"No URL defined for model: {model_name}")
            return model_path
        
        # Verificăm dacă fișierele modelului există
        model_files = list(Path(model_path).glob("*.pth"))
        if not model_files:
            logger.info(f"Model {model_name} not found. Downloading...")
            os.makedirs(model_path, exist_ok=True)
            
            if model_url.endswith(".zip"):
                self.download_and_extract(model_url, model_path)
            else:
                model_file = os.path.join(model_path, f"{model_name}.pth")
                self.download_file(model_url, model_file)
        
        return model_path
    
    def load_main_model(self) -> None:
        """
        Încarcă modelul principal (HiDream) pentru editare
        """
        logger.info("Loading main HiDream model...")
        
        try:
            from diffusers import (
                StableDiffusionXLInpaintPipeline,
                AutoencoderKL,
                DPMSolverMultistepScheduler,
                ControlNetModel
            )
            
            # Încarcă VAE
            vae = AutoencoderKL.from_pretrained(
                self.model_config.HIDREAM_CONFIG["vae_name_or_path"],
                torch_dtype=self.config.DTYPE,
                cache_dir=self.config.CACHE_DIR
            ).to(self.config.DEVICE)
            
            # Încarcă ControlNet dacă este necesar
            controlnet = None
            if self.model_config.CONTROLNET_CONFIG:
                try:
                    controlnet = ControlNetModel.from_pretrained(
                        self.model_config.CONTROLNET_CONFIG["model_id"],
                        torch_dtype=self.config.DTYPE,
                        use_safetensors=True,
                        variant="fp16" if self.config.DTYPE == torch.float16 else None,
                        cache_dir=self.config.CACHE_DIR
                    ).to(self.config.DEVICE)
                except Exception as e:
                    logger.error(f"Error loading ControlNet: {e}")
                    logger.info("Continuing without ControlNet")
            
            # Încarcă modelul principal
            pipeline = StableDiffusionXLInpaintPipeline.from_pretrained(
                self.model_config.HIDREAM_CONFIG["pretrained_model_name_or_path"],
                vae=vae,
                torch_dtype=self.config.DTYPE,
                variant="fp16" if self.config.DTYPE == torch.float16 else None,
                use_safetensors=True,
                cache_dir=self.config.CACHE_DIR
            )
            
            # Adaugă controlnet la pipeline dacă este disponibil
            if controlnet is not None:
                pipeline.controlnet = controlnet
            
            # Optimizări pentru VRAM scăzut
            if self.config.LOW_VRAM_MODE:
                pipeline.enable_model_cpu_offload()
            else:
                pipeline.to(self.config.DEVICE)
            
            # Setează scheduler optimizat
            pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
                pipeline.scheduler.config,
                algorithm_type="sde-dpmsolver++",
                use_karras_sigmas=True
            )
            
            # Adaugă modelul la dicționar
            self.models['main'] = pipeline
            logger.info("Main model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading main model: {e}")
            raise RuntimeError(f"Failed to load main model: {str(e)}")
    
    def load_sam_model(self) -> None:
        """
        Încarcă modelul SAM pentru segmentare
        """
        logger.info("Loading SAM model...")
        
        try:
            from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
            
            # Asigură că modelul este descărcat
            model_path = self.ensure_model("sam")
            sam_checkpoint = os.path.join(model_path, "sam_vit_h_4b8939.pth")
            
            if not os.path.exists(sam_checkpoint):
                logger.info("SAM checkpoint not found, downloading...")
                self.download_file(self.config.MODEL_URLS["sam"], sam_checkpoint)
            
            # Încarcă modelul SAM
            sam = sam_model_registry[self.model_config.SAM_CONFIG["model_type"]](
                checkpoint=sam_checkpoint
            )
            sam.to(self.config.DEVICE)
            
            # Creează generatorul automat de măști
            mask_generator = SamAutomaticMaskGenerator(
                sam,
                points_per_side=self.model_config.SAM_CONFIG["points_per_side"],
                pred_iou_thresh=self.model_config.SAM_CONFIG["pred_iou_thresh"],
                stability_score_thresh=self.model_config.SAM_CONFIG["stability_score_thresh"],
                min_mask_region_area=self.model_config.SAM_CONFIG["min_mask_region_area"]
            )
            
            # Adaugă modelul la dicționar
            self.models['sam'] = mask_generator
            logger.info("SAM model loaded successfully")
            
        except ImportError:
            logger.warning("segment_anything not installed. SAM capabilities will be limited.")
        except Exception as e:
            logger.error(f"Error loading SAM model: {e}")
            logger.warning("Continuing without SAM capabilities")
    
    def load_clipseg_model(self) -> None:
        """
        Încarcă modelul CLIPSeg pentru segmentare bazată pe text
        """
        logger.info("Loading CLIPSeg model...")
        
        try:
            from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
            
            # Încarcă procesorul și modelul CLIPSeg
            processor = CLIPSegProcessor.from_pretrained(
                self.model_config.CLIP_CONFIG["model_id"],
                cache_dir=self.config.CACHE_DIR
            )
            
            model = CLIPSegForImageSegmentation.from_pretrained(
                self.model_config.CLIP_CONFIG["model_id"],
                torch_dtype=self.config.DTYPE,
                cache_dir=self.config.CACHE_DIR
            ).to(self.config.DEVICE)
            
            # Ne asigurăm că modelul și parametrii folosesc același tip de date
            for param in model.parameters():
                if param.requires_grad:
                    param.data = param.data.to(dtype=self.config.DTYPE)
            
            # Adaugă modelul la dicționar
            self.models['clipseg'] = {
                'processor': processor,
                'model': model
            }
            logger.info("CLIPSeg model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading CLIPSeg model: {e}")
            logger.warning("Continuing without CLIPSeg capabilities")
    
    def load_specialized_model(self, model_name: str) -> Any:
        """
        Încarcă un model specializat la cerere (lazy loading)
        
        Args:
            model_name: Numele modelului de încărcat
            
        Returns:
            Modelul încărcat sau None dacă încărcarea eșuează
        """
        logger.info(f"Loading specialized model: {model_name}")
        
        if model_name in self.models and self.models[model_name] is not None:
            return self.models[model_name]
        
        # Implementăm încărcarea pentru fiecare tip de model specializat
        if model_name == 'yolo':
            return self._load_yolo_model()
        elif model_name == 'mediapipe':
            return self._load_mediapipe_model()
        elif model_name == 'face_detector':
            return self._load_face_detector()
        elif model_name == 'rembg':
            return self._load_rembg_model()
        else:
            logger.warning(f"Unknown model: {model_name}")
            return None
    
    def _load_yolo_model(self):
        """Încarcă modelul YOLO pentru detectarea obiectelor"""
        try:
            from ultralytics import YOLO
            
            # Definim path-ul către modelul YOLO
            yolo_path = os.path.join(self.config.MODEL_DIR, "yolov8x-seg.pt")
            
            # Verificăm dacă modelul există, altfel îl descărcăm
            if not os.path.exists(yolo_path):
                logger.info("YOLO model not found, downloading...")
                os.makedirs(os.path.dirname(yolo_path), exist_ok=True)
                
                # Folosim YOLO pentru a descărca modelul
                model = YOLO("yolov8x-seg.pt")
            else:
                model = YOLO(yolo_path)
            
            # Forțăm YOLO să folosească același device
            if hasattr(model.model, 'to'):
                model.model.to(device=self.config.DEVICE)
            
            self.models['yolo'] = model
            logger.info("YOLO model loaded successfully")
            return model
            
        except ImportError:
            logger.warning("ultralytics not installed. YOLO capabilities will be limited.")
            return None
        except Exception as e:
            logger.error(f"Error loading YOLO model: {e}")
            logger.warning("Continuing without YOLO capabilities")
            return None
    
    def _load_mediapipe_model(self):
        """Încarcă modelul MediaPipe pentru segmentarea persoanelor"""
        try:
            import mediapipe as mp
            
            model = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)
            self.models['mediapipe'] = model
            logger.info("MediaPipe model loaded successfully")
            return model
            
        except ImportError:
            logger.warning("mediapipe not installed. Trying to install it...")
            try:
                import subprocess
                subprocess.check_call(["pip", "install", "mediapipe"])
                import mediapipe as mp
                model = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)
                self.models['mediapipe'] = model
                logger.info("MediaPipe installed and loaded successfully")
                return model
            except Exception as e:
                logger.error(f"Error installing/loading MediaPipe: {e}")
                logger.warning("Continuing without MediaPipe capabilities")
                return None
        except Exception as e:
            logger.error(f"Error loading MediaPipe model: {e}")
            logger.warning("Continuing without MediaPipe capabilities")
            return None
    
    def _load_face_detector(self):
        """Încarcă detectorul de fețe pentru segmentare îmbunătățită"""
        try:
            import mediapipe as mp
            
            model = mp.solutions.face_detection.FaceDetection(
                model_selection=1,
                min_detection_confidence=0.5
            )
            self.models['face_detector'] = model
            logger.info("Face detector loaded successfully")
            return model
            
        except ImportError:
            logger.warning("mediapipe not installed. Face detection capabilities will be limited.")
            return None
        except Exception as e:
            logger.error(f"Error loading face detector: {e}")
            logger.warning("Continuing without face detection capabilities")
            return None
    
    def _load_rembg_model(self):
        """Încarcă modelul rembg pentru eliminarea fundalului"""
        try:
            import rembg
            self.models['rembg'] = rembg
            logger.info("Rembg model loaded successfully")
            return rembg
            
        except ImportError:
            logger.warning("rembg not installed. Trying to install it...")
            try:
                import subprocess
                subprocess.check_call(["pip", "install", "rembg"])
                import rembg
                self.models['rembg'] = rembg
                logger.info("Rembg installed and loaded successfully")
                return rembg
            except Exception as e:
                logger.error(f"Error installing/loading rembg: {e}")
                logger.warning("Continuing without rembg capabilities")
                return None
        except Exception as e:
            logger.error(f"Error loading rembg model: {e}")
            logger.warning("Continuing without rembg capabilities")
            return None
    
    def load_all_models(self) -> None:
        """
        Încarcă toate modelele principale necesare pentru aplicație
        """
        logger.info("Loading all main models...")
        
        # Încarcă modelul principal
        self.load_main_model()
        
        # Încarcă modelul SAM
        self.load_sam_model()
        
        # Încarcă modelul CLIPSeg
        self.load_clipseg_model()
        
        logger.info("All main models loaded")
    
    def unload_model(self, model_name: str) -> None:
        """
        Descarcă un model pentru a elibera memorie
        
        Args:
            model_name: Numele modelului de descărcat
        """
        if model_name in self.models and self.models[model_name] is not None:
            logger.info(f"Unloading model: {model_name}")
            self.models[model_name] = None
            torch.cuda.empty_cache()
            gc.collect()
            logger.info(f"Model {model_name} unloaded")
    
    def get_model(self, model_name: str) -> Any:
        """
        Obține un model din manager, încărcându-l dacă este necesar
        
        Args:
            model_name: Numele modelului
            
        Returns:
            Modelul cerut sau None dacă modelul nu poate fi încărcat
        """
        # Verificăm dacă modelul este deja încărcat
        if model_name in self.models and self.models[model_name] is not None:
            return self.models[model_name]
        
        # Altfel încercăm să încărcăm modelul
        if model_name == 'main':
            self.load_main_model()
        elif model_name == 'sam':
            self.load_sam_model()
        elif model_name == 'clipseg':
            self.load_clipseg_model()
        else:
            # Pentru alte modele specializate folosim încărcarea la cerere
            return self.load_specialized_model(model_name)
        
        return self.models.get(model_name)