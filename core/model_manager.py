#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Manager pentru modelele AI folosite în FusionFrame 2.0
(Actualizat pentru SAM Predictor)
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

# Import BaseModel local
try:
     from models.base_model import BaseModel
except ImportError:
     BaseModel = object

from config.app_config import AppConfig
from config.model_config import ModelConfig

# Importuri pentru modelele transformers auxiliare
try:
    from transformers import ( AutoImageProcessor, AutoModelForImageClassification, AutoModelForDepthEstimation )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    logging.getLogger(__name__).warning("Transformers library not found/outdated.")
    AutoImageProcessor, AutoModelForImageClassification, AutoModelForDepthEstimation = None, None, None
    TRANSFORMERS_AVAILABLE = False

# >>> MODIFICARE: Importuri pentru SAM Predictor <<<
try:
    from segment_anything import sam_model_registry, SamPredictor # Importăm SamPredictor
    SAM_AVAILABLE = True
except ImportError:
    logging.getLogger(__name__).warning("'segment_anything' library not found. SAM features disabled.")
    sam_model_registry, SamPredictor = None, None # Setăm la None dacă lipsește
    SAM_AVAILABLE = False
# <<< SFÂRȘIT MODIFICARE >>>

logger = logging.getLogger(__name__)

class ModelManager:
    _instance = None
    def __new__(cls): # Singleton
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized: return
        self.models: Dict[str, Any] = {}
        self.config = AppConfig
        self.model_config = ModelConfig
        self.config.ensure_dirs()
        self._initialized = True
        logger.info("ModelManager initialized")

    # --- Metode Utilitare (Neschimbate față de versiunea ta) ---
    def _get_filename_from_url(self, url: str) -> str:
        try: path = requests.utils.urlparse(url).path; name = os.path.basename(path); return name if name else (path.split('/')[-1] if '/' in path else "downloaded_file")
        except: return "downloaded_file"

    def download_file(self, url: str, dest: str, desc: Optional[str]=None) -> Optional[str]:
        name = desc or os.path.basename(dest); logger.info(f"DL {name} to {dest}")
        try:
            os.makedirs(os.path.dirname(dest), exist_ok=True); r = requests.get(url, stream=True, timeout=60); r.raise_for_status()
            total = int(r.headers.get('content-length', 0))
            with tqdm(total=total, unit='B', unit_scale=True, desc=f"DL {name}", leave=False) as p, open(dest, 'wb') as f:
                 for chunk in r.iter_content(8192): f.write(chunk); p.update(len(chunk))
            logger.info(f"DL complete: {dest}"); return dest
        except Exception as e: logger.error(f"DL error {name}: {e}", exc_info=True); # ... (cleanup) ...
        return None

    def download_and_extract_zip(self, url: str, dest_dir: str) -> Optional[str]:
        logger.info(f"DL/Extract zip {url} to {dest_dir}")
        os.makedirs(dest_dir, exist_ok=True); os.makedirs(self.config.CACHE_DIR, exist_ok=True)
        zip_name=self._get_filename_from_url(url); zip_name+= ".zip" if not zip_name.lower().endswith(".zip") else ""
        tmp_path=os.path.join(self.config.CACHE_DIR, zip_name); tmp_obj=Path(tmp_path)
        try:
            if not (tmp_obj.exists() and tmp_obj.stat().st_size > 0):
                 if not self.download_file(url, tmp_path, description=zip_name): return None
            else: logger.info(f"Zip {tmp_path} exists.")
            logger.info(f"Extracting {tmp_path} to {dest_dir}...");
            with zipfile.ZipFile(tmp_path, 'r') as zf: zf.extractall(dest_dir)
            logger.info(f"Extracted to: {dest_dir}"); return dest_dir
        except Exception as e: logger.error(f"Zip DL/Extract error: {e}", exc_info=True); #... (cleanup) ...
        return None

    def ensure_model(self, model_key: str) -> str:
        # Păstrăm pentru SAM .pth download
        assets_dir = os.path.join(self.config.MODEL_DIR, model_key); os.makedirs(assets_dir, exist_ok=True)
        url = self.config.MODEL_URLS.get(model_key)
        if not url: logger.warning(f"No URL for '{model_key}'."); return assets_dir
        if model_key == "sam":
            chk_name = self.model_config.SAM_CONFIG.get("checkpoint"); chk_path = os.path.join(assets_dir, chk_name)
            if not (os.path.exists(chk_path) and os.path.getsize(chk_path) > 0):
                 logger.info(f"SAM .pth missing, downloading {url}...")
                 if not self.download_file(url, chk_path, description=chk_name): logger.error("Failed DL SAM .pth.")
            else: logger.info(f"SAM .pth found: {chk_path}")
        return assets_dir

    # --- Metode de Încărcare Specifice ---

    def load_main_model(self) -> None: # Neschimbat
        logger.debug("Request main model load..."); # ... (cod identic cu cel furnizat de tine) ...
        if 'main' in self.models and isinstance(self.models.get('main'), BaseModel) and self.models['main'].is_loaded: logger.debug("Main already loaded."); return
        try: from models.hidream_model import HiDreamModel; logger.info("Instantiating HiDream..."); inst = HiDreamModel();
             if inst.load(): self.models['main'] = inst; logger.info("Main model loaded.")
             else: logger.error("HiDream load() failed."); del self.models['main'] if 'main' in self.models else None
        except Exception as e: logger.error(f"Main load error: {e}", exc_info=True); self.unload_model('main')


    # >>> MODIFICARE: load_sam_model încarcă predictorul <<<
    def load_sam_model(self) -> None:
        """Încarcă modelul SAM și creează instanța SamPredictor."""
        logger.debug("Request to load SAM model and predictor...")
        # Verificăm dacă predictorul este deja încărcat
        if 'sam_predictor' in self.models and self.models['sam_predictor']:
            logger.debug("SAM predictor already loaded."); return

        if not SAM_AVAILABLE: # Verificăm dacă importul a reușit
             logger.error("Segment Anything library not available. Cannot load SAM.")
             return

        try:
            sam_assets_dir = self.ensure_model("sam") # Asigură descărcarea .pth
            cfg = self.model_config.SAM_CONFIG
            sam_checkpoint_path = os.path.join(sam_assets_dir, cfg.get("checkpoint", "sam_vit_h_4b8939.pth"))
            sam_model_type = cfg.get("model_type", "vit_h")

            if not (os.path.exists(sam_checkpoint_path) and os.path.getsize(sam_checkpoint_path) > 0):
                logger.error(f"SAM checkpoint not found: {sam_checkpoint_path}"); return

            logger.info(f"Loading SAM model ({sam_model_type}) from: {sam_checkpoint_path}")
            # 1. Încarcă modelul SAM
            sam_model = sam_model_registry[sam_model_type](checkpoint=sam_checkpoint_path)
            sam_model.to(device=self.config.DEVICE)
            sam_model.eval() # Important pentru inferență

            # 2. Creează predictorul SAM
            sam_predictor = SamPredictor(sam_model)

            # 3. Stocăm predictorul sub cheia 'sam_predictor'
            self.models['sam_predictor'] = sam_predictor
            logger.info(f"SAM model ('{sam_model_type}') and SamPredictor loaded successfully.")

        except KeyError as e_key: logger.error(f"SAM Config key error (type='{sam_model_type}'?): {e_key}", exc_info=True)
        except Exception as e: logger.error(f"Error loading SAM/predictor: {e}", exc_info=True)
    # <<< SFÂRȘIT MODIFICARE >>>


    # ... (Metodele _load_clipseg, _load_yolo, etc. rămân neschimbate ca în versiunea ta) ...
    def load_clipseg_model(self) -> None: # Neschimbat
        logger.debug("Request CLIPSeg..."); #... (cod identic) ...
        if 'clipseg' in self.models and self.models['clipseg']: return
        if not TRANSFORMERS_AVAILABLE: return
        try: from transformers import CLIPSegProcessor,CLIPSegForImageSegmentation; mid=self.model_config.CLIP_CONFIG.get("model_id");logger.info(f"Loading CLIPSeg:{mid}");p=CLIPSegProcessor.from_pretrained(mid,cache_dir=self.config.CACHE_DIR);m=CLIPSegForImageSegmentation.from_pretrained(mid,torch_dtype=self.config.DTYPE,cache_dir=self.config.CACHE_DIR).to(self.config.DEVICE).eval();self.models['clipseg']={'processor':p,'model':m};logger.info("CLIPSeg loaded.")
        except Exception as e: logger.error(f"CLIPSeg load error: {e}",exc_info=True)

    def _load_yolo_model(self) -> Optional[Any]: # Neschimbat
        logger.debug("Request YOLO..."); #... (cod identic) ...
        if 'yolo' in self.models and self.models['yolo']: return self.models['yolo']
        try: from ultralytics import YOLO; name="yolov8x-seg.pt"; path=Path(self.config.MODEL_DIR)/"YOLO"/name; target=str(path) if path.is_file() else name; logger.info(f"Loading YOLO: {target}"); m=YOLO(target); self.models['yolo']=m; logger.info("YOLO loaded."); return m
        except Exception as e: logger.error(f"YOLO load error: {e}",exc_info=True); return None

    def _load_mediapipe_model(self) -> Optional[Any]: # Neschimbat
        logger.debug("Request MP Selfie..."); #... (cod identic) ...
        if 'mediapipe' in self.models and self.models['mediapipe']: return self.models['mediapipe']
        try: import mediapipe as mp; sel=getattr(self.model_config,"MEDIAPIPE_SELFIE_MODEL_SELECTION",1); seg=mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=sel); self.models['mediapipe']=seg; logger.info("MP Selfie loaded."); return seg
        except Exception as e: logger.error(f"MP Selfie load error: {e}",exc_info=True); return None

    def _load_face_detector(self) -> Optional[Any]: # Neschimbat
        logger.debug("Request MP FaceDet..."); #... (cod identic) ...
        if 'face_detector' in self.models and self.models['face_detector']: return self.models['face_detector']
        try: import mediapipe as mp; sel=getattr(self.model_config,"MEDIAPIPE_FACE_MODEL_SELECTION",0); conf=getattr(self.model_config,"MEDIAPIPE_FACE_MIN_CONFIDENCE",0.5); det=mp.solutions.face_detection.FaceDetection(model_selection=sel,min_detection_confidence=conf); self.models['face_detector']=det; logger.info("MP FaceDet loaded."); return det
        except Exception as e: logger.error(f"MP FaceDet load error: {e}",exc_info=True); return None

    def _load_rembg_model(self) -> Optional[Any]: # Neschimbat
        logger.debug("Request Rembg session..."); #... (cod identic) ...
        if 'rembg' in self.models and self.models['rembg']: return self.models['rembg']
        try: from rembg import new_session; name=getattr(self.model_config,"REMBG_MODEL_NAME","u2net"); sess=new_session(model_name=name); self.models['rembg']=sess; logger.info(f"Rembg session '{name}' created."); return sess
        except Exception as e: logger.error(f"Rembg load error: {e}",exc_info=True); return None

    def _load_image_classifier(self) -> Optional[Dict[str, Any]]: # Neschimbat
        logger.debug("Request ImgClassifier..."); #... (cod identic) ...
        if 'image_classifier' in self.models and self.models['image_classifier']: return self.models['image_classifier']
        if not TRANSFORMERS_AVAILABLE: return None
        try: cfg=getattr(self.model_config,"IMAGE_CLASSIFIER_CONFIG",{}); mid=cfg.get("model_id","google/vit-base-patch16-224");logger.info(f"Loading ImgClass: {mid}");p=AutoImageProcessor.from_pretrained(mid,cache_dir=self.config.CACHE_DIR);m=AutoModelForImageClassification.from_pretrained(mid,torch_dtype=self.config.DTYPE,cache_dir=self.config.CACHE_DIR).to(self.config.DEVICE).eval();bundle={'processor':p,'model':m};self.models['image_classifier']=bundle;logger.info("ImgClass loaded.");return bundle
        except Exception as e: logger.error(f"ImgClass load error: {e}",exc_info=True); return None

    def _load_depth_estimator(self) -> Optional[Dict[str, Any]]: # Neschimbat
        logger.debug("Request DepthEst..."); #... (cod identic) ...
        if 'depth_estimator' in self.models and self.models['depth_estimator']: return self.models['depth_estimator']
        if not TRANSFORMERS_AVAILABLE: return None
        try: cfg=getattr(self.model_config,"DEPTH_ESTIMATOR_CONFIG",{}); mid=cfg.get("model_id","Intel/dpt-hybrid-midas");logger.info(f"Loading DepthEst: {mid}");p=AutoImageProcessor.from_pretrained(mid,cache_dir=self.config.CACHE_DIR);m=AutoModelForDepthEstimation.from_pretrained(mid,torch_dtype=self.config.DTYPE,cache_dir=self.config.CACHE_DIR).to(self.config.DEVICE).eval();bundle={'processor':p,'model':m};self.models['depth_estimator']=bundle;logger.info("DepthEst loaded.");return bundle
        except Exception as e: logger.error(f"DepthEst load error: {e}",exc_info=True); return None

    # --- Metode de Gestionare ---

    def load_specialized_model(self, model_name: str) -> Any: # Neschimbat
        return self.get_model(model_name) # Simplificat

    def load_all_models(self) -> None: # MODIFICAT SĂ VERIFICE SAM Predictor
        logger.info("Loading/Checking essential models (main, sam_predictor, clipseg)...")
        self.get_model('main'); self.get_model('sam_predictor'); self.get_model('clipseg') # Cheia pentru SAM e acum 'sam_predictor'
        logger.info("Essential models checked/load initiated.")

    def unload_model(self, model_name: str) -> None: # MODIFICAT SĂ TRATEZE sam_predictor
        model_obj = self.models.pop(model_name, None)
        if model_obj is not None:
            logger.info(f"Unloading model: '{model_name}'...")
            try:
                actual_model = model_obj.get('model') if isinstance(model_obj, dict) else model_obj
                # ... (logica unload/close) ...
                if model_name == 'main' and hasattr(actual_model, 'unload'): actual_model.unload()
                elif model_name in ('mediapipe', 'face_detector') and hasattr(actual_model, 'close'): actual_model.close()
                elif model_name == 'sam_predictor': logger.debug("Unloading SAM predictor.") # Log specific
                # ... (del și gc) ...
                del actual_model; del model_obj; gc.collect()
                if torch.cuda.is_available(): torch.cuda.empty_cache()
                logger.info(f"Model '{model_name}' unloaded.")
            except Exception as e: logger.error(f"Unload error {model_name}: {e}", exc_info=True)
        else: logger.info(f"Model '{model_name}' not loaded/found.")

    def get_model(self, model_name: str) -> Any: # MODIFICAT SĂ ÎNCARCE sam_predictor
        """Obține un model, încărcându-l leneș."""
        if model_name in self.models and self.models[model_name] is not None:
             # ... (verificare main model neschimbată) ...
            if model_name == 'main':
                 main_model = self.models['main']
                 if isinstance(main_model, BaseModel) and not main_model.is_loaded: logger.warning("Main cached but not loaded. Reloading."); self.load_main_model()
                 elif not isinstance(main_model, BaseModel): logger.error(f"Invalid cache for 'main': {type(main_model)}. Reloading."); self.unload_model('main'); self.load_main_model()
                 return self.models.get(model_name)
            return self.models[model_name]

        logger.info(f"Model '{model_name}' not loaded. Attempting lazy load...")
        loader_map = { # Mapare actualizată
            'main': self.load_main_model,
            'sam_predictor': self.load_sam_model, # Actualizat cheia SAM
            'clipseg': self.load_clipseg_model,
            'yolo': self._load_yolo_model,
            'mediapipe': self._load_mediapipe_model,
            'face_detector': self._load_face_detector,
            'rembg': self._load_rembg_model,
            'image_classifier': self._load_image_classifier,
            'depth_estimator': self._load_depth_estimator
        }
        loader_func = loader_map.get(model_name)
        if loader_func: loader_func()
        else: logger.warning(f"No loader for key: '{model_name}'."); return None

        final_model = self.models.get(model_name)
        if final_model is None: logger.error(f"Failed to load model '{model_name}'.")
        return final_model