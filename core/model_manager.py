#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Ai Models Manager used in FusionFrame 2.0
"""

import os
import gc
import torch
import logging
import requests
import zipfile
import time  # Adăugat import pentru time
from tqdm.auto import tqdm
from pathlib import Path
from typing import Optional, Dict, Any, Union, List

# Import BaseModel local (with fallback)
try:
    from models.base_model import BaseModel
except ImportError:
    BaseModel = object

from config.app_config import AppConfig
from config.model_config import ModelConfig

# Imports Transformers (with fallback)
try:
    from transformers import (
        AutoImageProcessor, AutoModelForImageClassification, AutoModelForDepthEstimation
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    logging.getLogger(__name__).warning("Transformers library not found/outdated.")
    AutoImageProcessor, AutoModelForImageClassification, AutoModelForDepthEstimation = None, None, None
    TRANSFORMERS_AVAILABLE = False

# SAM Predictor Imports (with fallback)
try:
    from segment_anything import sam_model_registry, SamPredictor
    SAM_AVAILABLE = True
except ImportError:
    logging.getLogger(__name__).warning("'segment_anything' library not found. SAM features disabled.")
    sam_model_registry, SamPredictor = None, None
    SAM_AVAILABLE = False

logger = logging.getLogger(__name__)

class ModelManager:
    _instance = None
    
    # Lista modelelor care pot fi încărcate pe CPU pentru a economisi VRAM
    CPU_FRIENDLY_MODELS = [
        'image_classifier', 'depth_estimator', 'yolo', 
        'mediapipe', 'face_detector', 'rembg'
    ]
    
    # Lista modelelor care sunt esențiale și ar trebui să rămână pe GPU
    ESSENTIAL_GPU_MODELS = ['main', 'sam_predictor', 'clipseg']
    
    def __new__(cls): # Singleton
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self.models: Dict[str, Any] = {}
        self.config = AppConfig  # Folosim clasa direct, nu instanțiem
        self.model_config = ModelConfig
        self.config.ensure_dirs()
        self._initialized = True
        
        # Adăugăm configurări pentru gestionarea memoriei
        self.memory_stats = {
            "last_check": 0,
            "loaded_models": [],
            "memory_usage": {}
        }
        
        logger.info("ModelManager initialized")

    # --- Metode pentru gestionarea memoriei ---
    def _clear_gpu_memory(self):
        """Eliberează memoria GPU."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            
            # Logarea utilizării memoriei
            allocated = torch.cuda.memory_allocated() / (1024**3)
            reserved = torch.cuda.memory_reserved() / (1024**3)
            logger.info(f"GPU memory after cleanup: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")
    
    def _log_memory_stats(self):
        """Înregistrează statisticile memoriei."""
        if not torch.cuda.is_available():
            return
        
        current_time = time.time()
        # Logăm doar o dată la 10 secunde pentru a evita spam-ul
        if current_time - self.memory_stats["last_check"] < 10:
            return
            
        self.memory_stats["last_check"] = current_time
        allocated = torch.cuda.memory_allocated() / (1024**3)
        reserved = torch.cuda.memory_reserved() / (1024**3)
        
        if torch.cuda.is_available():
            try:
                total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                free = total - reserved
                percent_used = (reserved / total) * 100 if total > 0 else 0
                
                logger.info(f"Memory Stats: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved, "
                          f"{free:.2f}GB free ({percent_used:.1f}% used)")
                
                self.memory_stats["memory_usage"] = {
                    "allocated_gb": allocated,
                    "reserved_gb": reserved,
                    "total_gb": total,
                    "free_gb": free,
                    "percent_used": percent_used
                }
            except Exception as e:
                logger.error(f"Error getting memory stats: {e}")
    
    def _is_memory_critical(self) -> bool:
        """Verifică dacă memoria GPU este aproape de a fi epuizată."""
        if not torch.cuda.is_available():
            return False
            
        try:
            total = torch.cuda.get_device_properties(0).total_memory
            reserved = torch.cuda.memory_reserved()
            percent_used = (reserved / total) * 100 if total > 0 else 0
            
            # Considerăm critică utilizarea peste 90%
            return percent_used > 90
        except Exception as e:
            logger.error(f"Error checking memory status: {e}")
            return False
    
    def _should_use_cpu_for_model(self, model_name: str) -> bool:
        """Determină dacă un model ar trebui încărcat pe CPU pentru a economisi VRAM."""
        # Verifică dacă avem modo LOW_VRAM activat
        low_vram_mode = getattr(self.config, "LOW_VRAM_MODE", False)
        
        # Dacă este un model care poate rula pe CPU și suntem în LOW_VRAM_MODE, folosim CPU
        if model_name in self.CPU_FRIENDLY_MODELS and low_vram_mode:
            return True
            
        # Dacă memoria este critică, forțăm modelele non-esențiale pe CPU
        if self._is_memory_critical() and model_name not in self.ESSENTIAL_GPU_MODELS:
            logger.warning(f"Memory is critical. Forcing model '{model_name}' to CPU.")
            return True
            
        return False
    
    def _get_device_for_model(self, model_name: str) -> str:
        """Determină dispozitivul potrivit pentru un model."""
        if self._should_use_cpu_for_model(model_name):
            return "cpu"
        return self.config.DEVICE

    # --- Metode Utilitare ---
    def _get_filename_from_url(self, url: str) -> str:
        try:
            path = requests.utils.urlparse(url).path
            name = os.path.basename(path)
            return name if name else (path.split('/')[-1] if '/' in path else "downloaded_file")
        except:
            logger.warning(f"URL parse failed: {url}")
            return "downloaded_file"

    def download_file(self, url: str, dest: str, desc: Optional[str]=None) -> Optional[str]:
        name = desc or os.path.basename(dest)
        logger.info(f"Downloading {name} to {dest}")
        total_size = 0 # Inițializăm total_size
        try:
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            r = requests.get(url, stream=True, timeout=120)
            r.raise_for_status()
            total_size = int(r.headers.get('content-length', 0))
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=f"DL {name}", leave=False) as p, open(dest, 'wb') as f:
                for chunk in r.iter_content(8192*4):
                    f.write(chunk)
                    p.update(len(chunk))
            logger.info(f"Download complete: {dest}")
            return dest
        except Exception as e:
            logger.error(f"Download error {name}: {e}", exc_info=True)
            if os.path.exists(dest):
                try:
                    # Verificăm mărimea doar dacă descărcarea a început (total_size > 0)
                    # sau dacă fișierul are mărime 0
                    if os.path.getsize(dest) == 0 or \
                       (total_size > 0 and os.path.getsize(dest) != total_size):
                        logger.warning(f"Removing potentially incomplete/zero-byte file: {dest}")
                        os.remove(dest)
                except OSError as e_remove:
                    logger.warning(f"Could not remove potentially corrupted file: {dest} - {e_remove}")
            return None

    def download_and_extract_zip(self, url: str, dest_dir: str) -> Optional[str]:
        logger.info(f"DL/Extract zip {url} to {dest_dir}")
        os.makedirs(dest_dir, exist_ok=True)
        os.makedirs(self.config.CACHE_DIR, exist_ok=True)
        zip_name = self._get_filename_from_url(url)
        zip_name += ".zip" if not zip_name.lower().endswith(".zip") else ""
        tmp_path = os.path.join(self.config.CACHE_DIR, zip_name)
        tmp_obj = Path(tmp_path)
        try:
            if not (tmp_obj.exists() and tmp_obj.stat().st_size > 0):
                if not self.download_file(url, tmp_path, desc=zip_name):
                    return None
            else:
                logger.info(f"Zip {tmp_path} exists.")
                
            logger.info(f"Extracting {tmp_path} to {dest_dir}...")
            with zipfile.ZipFile(tmp_path, 'r') as zf:
                zf.extractall(dest_dir)
            logger.info(f"Extracted to: {dest_dir}")
            return dest_dir
        except zipfile.BadZipFile as e_zip:
            logger.error(f"Bad zip {tmp_path}: {e_zip}")
            if tmp_obj.exists():
                try:
                    os.remove(tmp_path)
                    logger.info(f"Removed corrupted zip: {tmp_path}")
                except OSError as e:
                    logger.warning(f"Could not remove bad zip: {e}")
        except Exception as e:
            logger.error(f"Zip DL/Extract error: {e}", exc_info=True)
        return None

    def ensure_model(self, model_key: str) -> str:
        assets_dir = os.path.join(self.config.MODEL_DIR, model_key)
        os.makedirs(assets_dir, exist_ok=True)
        model_urls = getattr(self.config, 'MODEL_URLS', getattr(self.model_config, 'MODEL_URLS', {}))
        url = model_urls.get(model_key)
        if not url:
            logger.warning(f"No URL for '{model_key}'.")
            return assets_dir
            
        if model_key == "sam":
            chk_name = self.model_config.SAM_CONFIG.get("checkpoint")
            chk_path = os.path.join(assets_dir, chk_name)
            if not (os.path.exists(chk_path) and os.path.getsize(chk_path) > 0):
                logger.info(f"SAM .pth missing, DL {url}...")
                if not self.download_file(url, chk_path, description=chk_name):
                    logger.error("Failed DL SAM .pth.")
            else:
                logger.debug(f"SAM .pth found: {chk_path}")
        return assets_dir

    # --- Model loading methods ---
    def load_main_model(self) -> None:
        logger.debug("Request main model load...")
        if 'main' in self.models and isinstance(self.models.get('main'), BaseModel) and self.models['main'].is_loaded:
            return
            
        # Curățăm memoria înainte de a încărca modelul principal
        self._clear_gpu_memory()
        
        try:
            from models.hidream_model import HiDreamModel
            logger.info("Instantiating HiDreamModel...")
            inst = HiDreamModel()
            if inst.load():
                self.models['main'] = inst
                logger.info("Main model loaded.")
                self._log_memory_stats()  # Logăm utilizarea memoriei după încărcare
            else:
                logger.error("HiDream load() failed.")
                if 'main' in self.models:
                    del self.models['main']
        except ImportError:
            logger.error("HiDreamModel class not found.", exc_info=True)
        except Exception as e:
            logger.error(f"Main load error: {e}", exc_info=True)
            self.unload_model('main')

    def load_sam_model(self) -> None:
        logger.debug("Request SAM predictor load...")
        if 'sam_predictor' in self.models:
            return
        if not SAM_AVAILABLE:
            logger.error("SAM library unavailable.")
            return
            
        # Curățăm memoria înainte de a încărca SAM
        self._clear_gpu_memory()
        
        try:
            assets_dir = self.ensure_model("sam")
            cfg = self.model_config.SAM_CONFIG
            chk_path = os.path.join(assets_dir, cfg.get("checkpoint"))
            m_type = cfg.get("model_type")
            if not (os.path.exists(chk_path) and os.path.getsize(chk_path) > 0):
                logger.error(f"SAM .pth not found: {chk_path}")
                return
                
            logger.info(f"Loading SAM model ({m_type}) for predictor: {chk_path}")
            
            # SAM are nevoie de GPU pentru a fi eficient, deci îl încărcăm pe GPU indiferent de LOW_VRAM_MODE
            sam_model = sam_model_registry[m_type](checkpoint=chk_path).to(self.config.DEVICE).eval()
            predictor = SamPredictor(sam_model)
            self.models['sam_predictor'] = predictor
            logger.info(f"SAM model & Predictor loaded.")
            self._log_memory_stats()
        except KeyError as e:
            logger.error(f"SAM Config key error: {e}", exc_info=True)
        except Exception as e:
            logger.error(f"Error loading SAM/predictor: {e}", exc_info=True)

    def load_clipseg_model(self) -> None:
        logger.debug("Request CLIPSeg...")
        if 'clipseg' in self.models:
            return
        if not TRANSFORMERS_AVAILABLE:
            return
            
        # Curățăm memoria înainte de a încărca CLIPSeg
        self._clear_gpu_memory()
        
        try:
            from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
            mid = self.model_config.CLIP_CONFIG.get("model_id")
            logger.info(f"Loading CLIPSeg: {mid}")
            p = CLIPSegProcessor.from_pretrained(mid, cache_dir=self.config.CACHE_DIR)
            
            # CLIPSeg ar trebui să fie pe GPU pentru eficiență
            device = self.config.DEVICE
            m = CLIPSegForImageSegmentation.from_pretrained(
                mid, 
                torch_dtype=self.config.DTYPE,
                cache_dir=self.config.CACHE_DIR
            ).to(device).eval()
            
            self.models['clipseg'] = {'processor': p, 'model': m, 'device': device}
            logger.info(f"CLIPSeg loaded on {device}.")
            self._log_memory_stats()
        except Exception as e:
            logger.error(f"CLIPSeg load error: {e}", exc_info=True)

    def _load_yolo_model(self) -> Optional[Any]:
        logger.debug("Request YOLO...")
        if 'yolo' in self.models:
            return self.models['yolo']
            
        # Curățăm memoria și determinăm dispozitivul
        self._clear_gpu_memory()
        device = self._get_device_for_model('yolo')
        
        try:
            from ultralytics import YOLO
            
            # Folosim varianta lite pentru a economisi memorie
            if device == "cpu" or self.config.LOW_VRAM_MODE:
                name = "yolov8n-seg.pt"  # Varianta nano, economică din punct de vedere al memoriei
                logger.info("Using YOLOv8-nano for low VRAM mode")
            else:
                name = "yolov8x-seg.pt"  # Varianta completă
            
            path = Path(self.config.MODEL_DIR) / "YOLO" / name
            target = str(path) if path.is_file() else name
            
            logger.info(f"Loading YOLO: {target} on {device}")
            m = YOLO(target)
            
            # YOLO se va încărca automat pe GPU la prima utilizare,
            # dar putem înregistra dispozitivul dorit
            self.models['yolo'] = {'model': m, 'device': device}
            logger.info(f"YOLO loaded (will run on {device}).")
            self._log_memory_stats()
            return self.models['yolo']
        except Exception as e:
            logger.error(f"YOLO load error: {e}", exc_info=True)
            return None

    def _load_mediapipe_model(self) -> Optional[Any]:
        logger.debug("Request MP Selfie...")
        if 'mediapipe' in self.models:
            return self.models['mediapipe']
            
        # MediaPipe merge doar pe CPU, nu necesită curățarea memoriei GPU
        try:
            import mediapipe as mp
            sel = getattr(self.model_config, "MEDIAPIPE_SELFIE_MODEL_SELECTION", 1)
            seg = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=sel)
            self.models['mediapipe'] = {'model': seg, 'device': 'cpu'}
            logger.info("MP Selfie loaded (CPU only).")
            return self.models['mediapipe']
        except Exception as e:
            logger.error(f"MP Selfie load error: {e}", exc_info=True)
            return None

    def _load_face_detector(self) -> Optional[Any]:
        logger.debug("Request MP FaceDet...")
        if 'face_detector' in self.models:
            return self.models['face_detector']
            
        # MediaPipe merge doar pe CPU, nu necesită curățarea memoriei GPU
        try:
            import mediapipe as mp
            sel = getattr(self.model_config, "MEDIAPIPE_FACE_MODEL_SELECTION", 0)
            conf = getattr(self.model_config, "MEDIAPIPE_FACE_MIN_CONFIDENCE", 0.5)
            det = mp.solutions.face_detection.FaceDetection(
                model_selection=sel,
                min_detection_confidence=conf
            )
            self.models['face_detector'] = {'model': det, 'device': 'cpu'}
            logger.info("MP FaceDet loaded (CPU only).")
            return self.models['face_detector']
        except Exception as e:
            logger.error(f"MP FaceDet load error: {e}", exc_info=True)
            return None

    def _load_rembg_model(self) -> Optional[Any]:
        logger.debug("Request Rembg session...")
        if 'rembg' in self.models:
            return self.models['rembg']
            
        device = self._get_device_for_model('rembg')
        
        try:
            from rembg import new_session
            name = getattr(self.model_config, "REMBG_MODEL_NAME", "u2net")
            
            # Dacă suntem în LOW_VRAM_MODE, folosim un model mai mic
            if device == "cpu" or self.config.LOW_VRAM_MODE:
                name = "u2netp"  # Varianta mai mică a u2net
                logger.info("Using smaller Rembg model (u2netp) for low VRAM mode")
                
            sess = new_session(model_name=name)
            self.models['rembg'] = {'model': sess, 'device': device}
            logger.info(f"Rembg session '{name}' created on {device}.")
            return self.models['rembg']
        except Exception as e:
            logger.error(f"Rembg load error: {e}", exc_info=True)
            return None

    def _load_image_classifier(self) -> Optional[Dict[str, Any]]:
        logger.debug("Request ImgClassifier...")
        if 'image_classifier' in self.models:
            return self.models['image_classifier']
        if not TRANSFORMERS_AVAILABLE:
            return None
            
        # Curățăm memoria și determinăm dispozitivul
        self._clear_gpu_memory()
        device = self._get_device_for_model('image_classifier')
        
        try:
            cfg = getattr(self.model_config, "IMAGE_CLASSIFIER_CONFIG", {})
            mid = cfg.get("model_id")
            
            # În LOW_VRAM_MODE, folosim un model mai mic
            if device == "cpu" or self.config.LOW_VRAM_MODE:
                mid = "google/vit-base-patch16-224-in21k"  # Un model mai mic dar similar
                
            logger.info(f"Loading ImgClass: {mid} on {device}")
            p = AutoImageProcessor.from_pretrained(mid, cache_dir=self.config.CACHE_DIR)
            m = AutoModelForImageClassification.from_pretrained(
                mid,
                torch_dtype=torch.float32 if device == "cpu" else self.config.DTYPE,
                cache_dir=self.config.CACHE_DIR
            ).to(device).eval()
            
            bundle = {'processor': p, 'model': m, 'device': device}
            self.models['image_classifier'] = bundle
            logger.info(f"ImgClass loaded on {device}.")
            self._log_memory_stats()
            return bundle
        except Exception as e:
            logger.error(f"ImgClass load error: {e}", exc_info=True)
            return None

    def _load_depth_estimator(self) -> Optional[Dict[str, Any]]:
        logger.debug("Request DepthEst...")
        if 'depth_estimator' in self.models:
            return self.models['depth_estimator']
        if not TRANSFORMERS_AVAILABLE:
            return None
            
        # Curățăm memoria și determinăm dispozitivul
        self._clear_gpu_memory()
        device = self._get_device_for_model('depth_estimator')
        
        try:
            cfg = getattr(self.model_config, "DEPTH_ESTIMATOR_CONFIG", {})
            mid = cfg.get("model_id")
            
            # În LOW_VRAM_MODE, folosim un model mai mic
            if device == "cpu" or self.config.LOW_VRAM_MODE:
                mid = "Intel/dpt-large"  # Un model mai mic dar similar
                
            logger.info(f"Loading DepthEst: {mid} on {device}")
            p = AutoImageProcessor.from_pretrained(mid, cache_dir=self.config.CACHE_DIR)
            m = AutoModelForDepthEstimation.from_pretrained(
                mid,
                torch_dtype=torch.float32 if device == "cpu" else self.config.DTYPE,
                cache_dir=self.config.CACHE_DIR
            ).to(device).eval()
            
            bundle = {'processor': p, 'model': m, 'device': device}
            self.models['depth_estimator'] = bundle
            logger.info(f"DepthEst loaded on {device}.")
            self._log_memory_stats()
            return bundle
        except Exception as e:
            logger.error(f"DepthEst load error: {e}", exc_info=True)
            return None

    def load_all_models(self) -> None:
        logger.info("Loading/Checking essential models (main, sam_predictor, clipseg)...")
        self.get_model('main')
        self.get_model('sam_predictor')
        self.get_model('clipseg')
        logger.info("Essential models check/load initiated.")

    def unload_model(self, model_name: str) -> None:
        model_obj = self.models.pop(model_name, None)
        if model_obj:
            logger.info(f"Unloading model: '{model_name}'...")
            try:
                actual_model = model_obj.get('model') if isinstance(model_obj, dict) else model_obj
                if model_name == 'main' and hasattr(actual_model, 'unload'):
                    actual_model.unload()
                elif model_name in ('mediapipe', 'face_detector') and hasattr(actual_model, 'close'):
                    actual_model.close()
                elif model_name == 'sam_predictor':
                    logger.debug("Unload SAM predictor.")
                    
                del actual_model
                del model_obj
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                logger.info(f"Model '{model_name}' unloaded.")
                self._log_memory_stats()
            except Exception as e:
                logger.error(f"Unload {model_name} error: {e}")
        else:
            logger.debug(f"Unload: Model '{model_name}' not found.")

    def get_model(self, model_name: str) -> Any:
        """Obține un model, încărcându-l leneș."""
        if model_name in self.models and self.models[model_name] is not None:
            # Verificare main model
            if model_name == 'main':
                main_model = self.models['main']
                if isinstance(main_model, BaseModel) and not main_model.is_loaded:
                    logger.warning("Main cached but not loaded. Reloading.")
                    self.load_main_model()
                elif not isinstance(main_model, BaseModel):
                    logger.error(f"Invalid cache for 'main': {type(main_model)}. Reloading.")
                    self.unload_model('main')
                    self.load_main_model()
                return self.models.get(model_name)
                
            # Pentru modele multi-componente (dicționare)
            if isinstance(self.models[model_name], dict) and "model" in self.models[model_name]:
                return self.models[model_name]
                
            return self.models[model_name]

        logger.info(f"Model '{model_name}' not loaded. Lazy loading...")
        
        # Verificăm spațiul disponibil înainte de a încărca un model nou
        if self._is_memory_critical() and model_name in self.CPU_FRIENDLY_MODELS:
            logger.warning(f"Memory is critical! Loading '{model_name}' on CPU.")
        
        # Mapare actualizată pentru modele
        loader_map = {
            'main': self.load_main_model,
            'sam_predictor': self.load_sam_model,
            'clipseg': self.load_clipseg_model,
            'yolo': self._load_yolo_model,
            'mediapipe': self._load_mediapipe_model,
            'face_detector': self._load_face_detector,
            'rembg': self._load_rembg_model,
            'image_classifier': self._load_image_classifier,
            'depth_estimator': self._load_depth_estimator
        }
        
        # Curățăm memoria înainte de a încărca un model nou
        self._clear_gpu_memory()
        
        loader_func = loader_map.get(model_name)
        if loader_func:
            loader_func()
        else:
            logger.warning(f"No loader for key: '{model_name}'.")
            return None

        final_model = self.models.get(model_name)
        if final_model is None:
            logger.error(f"Failed to load model '{model_name}'.")
        return final_model
    
    def get_memory_status(self) -> Dict[str, Any]:

        memory_info = {
            "cuda_available": torch.cuda.is_available(),
            "loaded_models": list(self.models.keys()),
            "system_ram_available_gb": 0,
            "cuda_info": {}
        }
    
        # System RAM info
        try:
            import psutil
            memory_info["system_ram_available_gb"] = psutil.virtual_memory().available / (1024**3)
            memory_info["system_ram_percent"] = psutil.virtual_memory().percent
        except ImportError:
            pass
        
        # CUDA memory info
        if torch.cuda.is_available():
            try:
                current_device = torch.cuda.current_device()
                device_name = torch.cuda.get_device_name(current_device)
                total_memory = torch.cuda.get_device_properties(current_device).total_memory
                allocated_memory = torch.cuda.memory_allocated(current_device)
                reserved_memory = torch.cuda.memory_reserved(current_device)
                free_memory = total_memory - allocated_memory
                
                memory_info["cuda_info"] = {
                    "device_name": device_name,
                    "current_device": current_device,
                    "total_memory_gb": total_memory / (1024**3),
                    "allocated_memory_gb": allocated_memory / (1024**3),
                    "reserved_memory_gb": reserved_memory / (1024**3),
                    "free_memory_gb": free_memory / (1024**3),
                    "percent_used": (allocated_memory / total_memory) * 100,
                    "is_memory_critical": (free_memory / total_memory) < 0.1  # Less than 10% free
                }
            except Exception as e:
                logger.error(f"Error getting CUDA memory status: {e}")
        
        return memory_info

    def emergency_memory_recovery(self) -> bool:
        """
        Perform emergency memory recovery when CUDA out-of-memory occurs.
        
        Returns:
            True if recovery actions were taken, False otherwise
        """
        if not torch.cuda.is_available():
            return False
        
        logger.warning("Performing emergency memory recovery")
        
        # 1. Move non-essential models to CPU
        for model_name in list(self.models.keys()):
            if model_name in self.CPU_FRIENDLY_MODELS and model_name not in self.ESSENTIAL_GPU_MODELS:
                self._move_model_to_cpu(model_name)
        
        # 2. Unload all non-essential models
        essential_models = getattr(self.config, "ESSENTIAL_MODELS", ["main"])
        for model_name in list(self.models.keys()):
            if model_name not in essential_models:
                logger.info(f"Emergency unloading model: {model_name}")
                self.unload_model(model_name)
        
        # 3. Force garbage collection and cache clearing
        self._clear_gpu_memory()
        
        # 4. Return success
        return True

    def _move_model_to_cpu(self, model_name: str) -> bool:
        """
        Move a model from GPU to CPU to free VRAM.
        
        Args:
            model_name: Name of the model to move
            
        Returns:
            True if successful, False otherwise
        """
        if model_name not in self.models:
            return False
        
        model_obj = self.models[model_name]
        
        try:
            # Handle dictionary case (processor + model)
            if isinstance(model_obj, dict) and "model" in model_obj:
                model = model_obj["model"]
                if hasattr(model, "to") and callable(model.to):
                    logger.info(f"Moving {model_name} to CPU")
                    model.to("cpu")
                    model_obj["device"] = "cpu"
                    return True
            # Handle direct model case
            elif hasattr(model_obj, "to") and callable(model_obj.to):
                logger.info(f"Moving {model_name} to CPU")
                model_obj.to("cpu")
                return True
            return False
        except Exception as e:
            logger.error(f"Error moving {model_name} to CPU: {e}")
            return False

    def monitor_memory_and_recover(self) -> None:
        """
        Monitor memory usage and perform recovery actions if needed.
        Call this method periodically in long-running operations.
        """
        if not torch.cuda.is_available():
            return
        
        try:
            mem_status = self.get_memory_status()
            cuda_info = mem_status.get("cuda_info", {})
            
            # Check if memory is critical
            if cuda_info.get("is_memory_critical", False):
                logger.warning(f"CRITICAL MEMORY STATE: {cuda_info.get('free_memory_gb', 0):.2f}GB free, "
                            f"{cuda_info.get('percent_used', 0):.1f}% used")
                
                # Identify models that can be moved to CPU or unloaded
                self._free_memory_proactively()
        except Exception as e:
            logger.error(f"Error in memory monitoring: {e}")

    def _free_memory_proactively(self) -> None:
        """
        Proactively free memory when approaching critical levels.
        """
        # Get minimum free VRAM threshold from config
        min_free_vram_mb = getattr(self.config, "MIN_FREE_VRAM_MB", 1000)
        
        # Check current free VRAM
        free_vram_mb = 0
        if torch.cuda.is_available():
            free_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()
            free_vram_mb = free_memory / (1024 * 1024)
        
        # If we have enough free VRAM, nothing to do
        if free_vram_mb >= min_free_vram_mb:
            return
        
        logger.warning(f"Low VRAM: {free_vram_mb:.0f}MB free (below {min_free_vram_mb}MB threshold)")
        
        # Define unload priority (least important first)
        unload_priority = [
            'depth_estimator',  # Heavy but not critical
            'image_classifier', # Useful but not critical
            'yolo',             # Object detector - heavy
            'clipseg',          # Used only for mask generation
            'sam_predictor',    # Very heavy, used for segmentation
            'gpen',             # Used only for face enhancement
            'esrgan',           # Used only for detail enhancement
            'codeformer',       # Used only for face enhancement
            'rembg',            # Used for background removal
            'mediapipe',        # Lightweight media processor
            'face_detector',    # Lightweight detector
        ]
        
        # Remove essential models from unload priority
        essential_models = getattr(self.config, "ESSENTIAL_MODELS", ["main"])
        unload_priority = [m for m in unload_priority if m not in essential_models]
        
        # First try to move models to CPU
        for model_name in unload_priority:
            if model_name in self.models:
                # Skip if it's already at an equivalent memory state
                if isinstance(self.models[model_name], dict) and self.models[model_name].get("device") == "cpu":
                    continue
                    
                # Try to move the model to CPU
                if self._move_model_to_cpu(model_name):
                    logger.info(f"Moved {model_name} to CPU to reduce VRAM usage")
                    
                    # Check if we've freed enough memory
                    if torch.cuda.is_available():
                        free_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()
                        free_vram_mb = free_memory / (1024 * 1024)
                        if free_vram_mb >= min_free_vram_mb:
                            logger.info(f"Successfully freed memory by moving models to CPU: {free_vram_mb:.0f}MB free")
                            return
        
        # If moving to CPU wasn't enough, unload models
        for model_name in unload_priority:
            if model_name in self.models:
                logger.info(f"Unloading {model_name} to free memory")
                self.unload_model(model_name)
                
                # Check if we've freed enough memory
                if torch.cuda.is_available():
                    free_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()
                    free_vram_mb = free_memory / (1024 * 1024)
                    if free_vram_mb >= min_free_vram_mb:
                        logger.info(f"Successfully freed memory by unloading models: {free_vram_mb:.0f}MB free")
                        return

    def optimize_for_inference(self) -> None:
        """
        Optimize memory usage before running the main inference process.
        Call this before starting a pipeline.
        """
        # Check if we need to optimize
        if not torch.cuda.is_available() or not getattr(self.config, "LOW_VRAM_MODE", False):
            return
        
        logger.info("Optimizing memory for inference...")
        
        # 1. Ensure main model is loaded and has priority
        main_model = self.get_model('main')
        if main_model is None:
            logger.error("Main model not loaded!")
            return
        
        # 2. Unload unnecessary models
        for model_name in list(self.models.keys()):
            if model_name != 'main' and model_name not in getattr(self.config, "ESSENTIAL_MODELS", ["main"]):
                self.unload_model(model_name)
        
        # 3. Clear CUDA cache
        self._clear_gpu_memory()
        
        # 4. Apply memory reduction techniques to main model if it's HiDreamModel
        main_model_obj = self.models['main']
        if hasattr(main_model_obj, 'pipeline'):
            try:
                pipeline = main_model_obj.pipeline
                
                # Check if we can move text encoders to CPU
                for component_name in ['text_encoder', 'text_encoder_2']:
                    if hasattr(pipeline, component_name):
                        component = getattr(pipeline, component_name)
                        if component is not None and next(component.parameters(), None) is not None:
                            device = next(component.parameters()).device
                            if device.type == 'cuda':
                                logger.info(f"Moving {component_name} to CPU to save VRAM")
                                component.to('cpu')
                
                # Enable attention slicing if available
                if hasattr(pipeline, 'enable_attention_slicing'):
                    logger.info("Enabling attention slicing")
                    pipeline.enable_attention_slicing()
                    
                # Enable VAE slicing if available
                if hasattr(pipeline, 'enable_vae_slicing'):
                    logger.info("Enabling VAE slicing")
                    pipeline.enable_vae_slicing()
                    
                # Enable VAE tiling if enabled in config
                if hasattr(pipeline, 'enable_vae_tiling') and getattr(self.config, 'ENABLE_VAE_TILING', False):
                    logger.info("Enabling VAE tiling")
                    pipeline.enable_vae_tiling()
            except Exception as e:
                logger.error(f"Error optimizing main model: {e}")
        
        logger.info("Memory optimization for inference completed")

    def handle_oom_error(self, func):
        """
        Decorator to handle OOM errors gracefully.
        
        Example usage:
            @handle_oom_error
            def load_model(self, model_name):
                # Your code here
        """
        import functools
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    logger.error(f"CUDA OOM in {func.__name__}: {str(e)}")
                    # Try emergency recovery
                    self.emergency_memory_recovery()
                    # Retry with CPU fallback if possible
                    try:
                        # If there's a force_cpu parameter, set it to True
                        if 'force_cpu' in kwargs:
                            kwargs['force_cpu'] = True
                        # If there's a device parameter, set it to 'cpu'
                        if 'device' in kwargs:
                            kwargs['device'] = 'cpu'
                        logger.info(f"Retrying {func.__name__} with CPU fallback")
                        return func(*args, **kwargs)
                    except Exception as retry_e:
                        logger.error(f"Retry failed in {func.__name__}: {str(retry_e)}")
                        # Propagate the original error
                        raise e
                # Re-raise other errors
                raise
        
        return wrapper

    # --- Enhanced Version of load_model_with_memory_management ---

    def load_model_with_memory_management(self, model_name: str, model_params: Dict[str, Any] = None) -> Any:
        """
        Enhanced version with better memory handling and error recovery.
        """
        # If model is already loaded, return it
        if model_name in self.models and self.models[model_name] is not None:
            return self.models[model_name]
        
        # Initialize params if needed
        model_params = model_params or {}
        
        # Check memory status and perform proactive cleanup if needed
        if torch.cuda.is_available():
            self.monitor_memory_and_recover()
        
        # Determine if we should use a lightweight variant
        use_lightweight = getattr(self.config, "USE_LIGHTWEIGHT_MODELS", False)
        if use_lightweight and model_name in getattr(self.config, "LIGHTWEIGHT_MODEL_KEYS", []):
            logger.info(f"Using lightweight variant for {model_name}")
            model_params['lightweight'] = True
        
        # Determine proper device (CPU or CUDA)
        force_cpu = model_params.get('force_cpu', False)
        if not force_cpu:
            force_cpu = self._should_use_cpu_for_model(model_name)
        
        device = "cpu" if force_cpu else self.config.DEVICE
        model_params['device'] = device
        
        logger.info(f"Loading {model_name} on {device}")
        
        try:
            # Try to load the model
            return self.get_model(model_name)
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                logger.error(f"CUDA OOM while loading {model_name}. Attempting recovery...")
                
                # Try emergency recovery
                self.emergency_memory_recovery()
                
                # Retry with CPU
                if device != "cpu":
                    logger.info(f"Retrying {model_name} load on CPU")
                    model_params['device'] = "cpu"
                    model_params['force_cpu'] = True
                    try:
                        return self.get_model(model_name)
                    except Exception as e2:
                        logger.error(f"CPU fallback failed for {model_name}: {e2}")
                
                # If we got here, both attempts failed
                raise RuntimeError(f"Could not load {model_name} due to memory constraints: {e}")
            else:
                # Re-raise other errors
                raise