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
from tqdm.auto import tqdm
from pathlib import Path
from typing import Optional, Dict, Any, Union

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
    
    def __new__(cls): # Singleton
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self.models: Dict[str, Any] = {}
        self.config = AppConfig
        self.model_config = ModelConfig
        self.config.ensure_dirs()
        self._initialized = True
        logger.info("ModelManager initialized")

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
            # --- CORECȚIE SINTAXĂ AICI ---
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
            # --- SFÂRȘIT CORECȚIE ---
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
                logger.debug(f"SAM .pth found: {chk_path}")  # Changed to debug
        return assets_dir

    # --- Start upload specific methods ---
    def load_main_model(self) -> None:
        logger.debug("Request main model load...")
        if 'main' in self.models and isinstance(self.models.get('main'), BaseModel) and self.models['main'].is_loaded:
            return
        try:
            from models.hidream_model import HiDreamModel
            logger.info("Instantiating HiDreamModel...")
            inst = HiDreamModel()
            if inst.load():
                self.models['main'] = inst
                logger.info("Main model loaded.")
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
        try:
            assets_dir = self.ensure_model("sam")
            cfg = self.model_config.SAM_CONFIG
            chk_path = os.path.join(assets_dir, cfg.get("checkpoint"))
            m_type = cfg.get("model_type")
            if not (os.path.exists(chk_path) and os.path.getsize(chk_path) > 0):
                logger.error(f"SAM .pth not found: {chk_path}")
                return
                
            logger.info(f"Loading SAM model ({m_type}) for predictor: {chk_path}")
            sam_model = sam_model_registry[m_type](checkpoint=chk_path).to(self.config.DEVICE).eval()
            predictor = SamPredictor(sam_model)
            self.models['sam_predictor'] = predictor
            logger.info(f"SAM model & Predictor loaded.")
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
        try:
            from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
            mid = self.model_config.CLIP_CONFIG.get("model_id")
            logger.info(f"Loading CLIPSeg: {mid}")
            p = CLIPSegProcessor.from_pretrained(mid, cache_dir=self.config.CACHE_DIR)
            m = CLIPSegForImageSegmentation.from_pretrained(
                mid, 
                torch_dtype=self.config.DTYPE,
                cache_dir=self.config.CACHE_DIR
            ).to(self.config.DEVICE).eval()
            self.models['clipseg'] = {'processor': p, 'model': m}
            logger.info("CLIPSeg loaded.")
        except Exception as e:
            logger.error(f"CLIPSeg load error: {e}", exc_info=True)

    def _load_yolo_model(self) -> Optional[Any]:
        logger.debug("Request YOLO...")
        if 'yolo' in self.models:
            return self.models['yolo']
        try:
            from ultralytics import YOLO
            name = "yolov8x-seg.pt"
            path = Path(self.config.MODEL_DIR) / "YOLO" / name
            target = str(path) if path.is_file() else name
            logger.info(f"Loading YOLO: {target}")
            m = YOLO(target)
            self.models['yolo'] = m
            logger.info("YOLO loaded.")
            return m
        except Exception as e:
            logger.error(f"YOLO load error: {e}", exc_info=True)
            return None

    def _load_mediapipe_model(self) -> Optional[Any]:
        logger.debug("Request MP Selfie...")
        if 'mediapipe' in self.models:
            return self.models['mediapipe']
        try:
            import mediapipe as mp
            sel = getattr(self.model_config, "MEDIAPIPE_SELFIE_MODEL_SELECTION", 1)
            seg = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=sel)
            self.models['mediapipe'] = seg
            logger.info("MP Selfie loaded.")
            return seg
        except Exception as e:
            logger.error(f"MP Selfie load error: {e}", exc_info=True)
            return None

    def _load_face_detector(self) -> Optional[Any]:
        logger.debug("Request MP FaceDet...")
        if 'face_detector' in self.models:
            return self.models['face_detector']
        try:
            import mediapipe as mp
            sel = getattr(self.model_config, "MEDIAPIPE_FACE_MODEL_SELECTION", 0)
            conf = getattr(self.model_config, "MEDIAPIPE_FACE_MIN_CONFIDENCE", 0.5)
            det = mp.solutions.face_detection.FaceDetection(
                model_selection=sel,
                min_detection_confidence=conf
            )
            self.models['face_detector'] = det
            logger.info("MP FaceDet loaded.")
            return det
        except Exception as e:
            logger.error(f"MP FaceDet load error: {e}", exc_info=True)
            return None

    def _load_rembg_model(self) -> Optional[Any]:
        logger.debug("Request Rembg session...")
        if 'rembg' in self.models:
            return self.models['rembg']
        try:
            from rembg import new_session
            name = getattr(self.model_config, "REMBG_MODEL_NAME", "u2net")
            sess = new_session(model_name=name)
            self.models['rembg'] = sess
            logger.info(f"Rembg session '{name}' created.")
            return sess
        except Exception as e:
            logger.error(f"Rembg load error: {e}", exc_info=True)
            return None

    def _load_image_classifier(self) -> Optional[Dict[str, Any]]:
        logger.debug("Request ImgClassifier...")
        if 'image_classifier' in self.models:
            return self.models['image_classifier']
        if not TRANSFORMERS_AVAILABLE:
            return None
        try:
            cfg = getattr(self.model_config, "IMAGE_CLASSIFIER_CONFIG", {})
            mid = cfg.get("model_id")
            logger.info(f"Loading ImgClass: {mid}")
            p = AutoImageProcessor.from_pretrained(mid, cache_dir=self.config.CACHE_DIR)
            m = AutoModelForImageClassification.from_pretrained(
                mid,
                torch_dtype=self.config.DTYPE,
                cache_dir=self.config.CACHE_DIR
            ).to(self.config.DEVICE).eval()
            bundle = {'processor': p, 'model': m}
            self.models['image_classifier'] = bundle
            logger.info("ImgClass loaded.")
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
        try:
            cfg = getattr(self.model_config, "DEPTH_ESTIMATOR_CONFIG", {})
            mid = cfg.get("model_id")
            logger.info(f"Loading DepthEst: {mid}")
            p = AutoImageProcessor.from_pretrained(mid, cache_dir=self.config.CACHE_DIR)
            m = AutoModelForDepthEstimation.from_pretrained(
                mid,
                torch_dtype=self.config.DTYPE,
                cache_dir=self.config.CACHE_DIR
            ).to(self.config.DEVICE).eval()
            bundle = {'processor': p, 'model': m}
            self.models['depth_estimator'] = bundle
            logger.info("DepthEst loaded.")
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
            except Exception as e:
                logger.error(f"Unload {model_name} error: {e}")
        else:
            logger.debug(f"Unload: Model '{model_name}' not found.")

    def get_model(self, model_name: str) -> Any:
        """Obține un model, încărcându-l leneș."""
        if model_name in self.models and self.models[model_name] is not None:
            # Verificare main model (ca în versiunea ta)
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
            return self.models[model_name]

        logger.info(f"Model '{model_name}' not loaded. Lazy loading...")
        # Mapare actualizată pentru SAM
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