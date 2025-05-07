#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Manager pentru modelele AI folosite în FusionFrame 2.0 (Master Branch Corectat)
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

try:
    from config.app_config import AppConfig
    from config.model_config import ModelConfig
    # Importăm clasa modelului specific pentru master (chiar dacă e numită HiDreamModel)
    from models.hidream_model import HiDreamModel 
except ImportError as e:
    print(f"ERROR: Failed to import necessary modules in model_manager.py: {e}")
    raise e

logger = logging.getLogger(__name__)
if not logger.hasHandlers(): # Basic logging if not configured
    _ch = logging.StreamHandler()
    _ch.setFormatter(logging.Formatter("%(asctime)s-%(name)s-%(levelname)s-%(message)s"))
    logger.addHandler(_ch)
    logger.setLevel(logging.INFO)

class ModelManager:
    _instance = None

    def __new__(cls):
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

    def _get_filename_from_url(self, url: str) -> str:
        try:
            parsed_path = requests.utils.urlparse(url).path
            filename = os.path.basename(parsed_path)
            if not filename: 
                parts = [part for part in parsed_path.split('/') if part]
                if parts: 
                    filename = parts[-1]
            return filename if filename else "downloaded_file"
        except Exception: 
            return "downloaded_file"

    def download_file(self, url: str, destination_path: str, description: Optional[str] = None) -> Optional[str]:
        desc_name = description or os.path.basename(destination_path)
        logger.info(f"Attempting download: {url} -> {destination_path}")
        try:
            os.makedirs(os.path.dirname(destination_path), exist_ok=True)
            response = requests.get(url, stream=True, timeout=60)
            response.raise_for_status() 
            total_size = int(response.headers.get('content-length', 0))
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=f"Downloading {desc_name}", leave=False) as pbar:
                with open(destination_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192): 
                        f.write(chunk)
                        pbar.update(len(chunk))
            logger.info(f"Download complete: {destination_path}")
            return destination_path
        except Exception as e:
            logger.error(f"Download failed for {desc_name}: {e}", exc_info=True)
            if os.path.exists(destination_path):
                try: 
                    if os.path.getsize(destination_path) == 0 or (total_size > 0 and os.path.getsize(destination_path) != total_size):
                        logger.warning(f"Removing incomplete file: {destination_path}")
                        os.remove(destination_path)
                except OSError: 
                    logger.warning(f"Could not remove potentially corrupted file: {destination_path}")
            return None

    def download_and_extract_zip(self, url: str, destination_extract_dir: str) -> Optional[str]:
        logger.info(f"Attempting zip download/extract: {url} -> {destination_extract_dir}")
        os.makedirs(destination_extract_dir, exist_ok=True)
        os.makedirs(self.config.CACHE_DIR, exist_ok=True)
        zip_filename = self._get_filename_from_url(url)
        zip_filename = zip_filename if zip_filename.lower().endswith(".zip") else zip_filename + ".zip"
        temp_zip_path = os.path.join(self.config.CACHE_DIR, zip_filename)
        temp_zip_path_obj = Path(temp_zip_path)

        try:
            if not (temp_zip_path_obj.exists() and temp_zip_path_obj.is_file() and temp_zip_path_obj.stat().st_size > 0):
                if not self.download_file(url, temp_zip_path, description=zip_filename): 
                    return None
            else:
                logger.info(f"Zip file {temp_zip_path} found in cache.")
            
            logger.info(f"Extracting {temp_zip_path} to {destination_extract_dir}...")
            with zipfile.ZipFile(temp_zip_path, 'r') as zip_ref:
                zip_ref.extractall(destination_extract_dir)
            logger.info(f"Extraction complete into: {destination_extract_dir}")
            return destination_extract_dir
        except zipfile.BadZipFile as e_zip:
            logger.error(f"Invalid zip file {temp_zip_path}: {e_zip}", exc_info=True)
            if temp_zip_path_obj.exists():
                try: 
                    os.remove(temp_zip_path)
                    logger.info(f"Removed corrupted zip: {temp_zip_path}")
                except OSError: 
                    pass
        except Exception as e:
            logger.error(f"Zip download/extraction error: {e}", exc_info=True)
        return None

    def ensure_model(self, model_name_key: str) -> str:
        """PĂSTRAT PENTRU COMPATIBILITATE (ex: SAM)."""
        model_assets_dir = os.path.join(self.config.MODEL_DIR, model_name_key)
        model_url = self.config.MODEL_URLS.get(model_name_key)
        os.makedirs(model_assets_dir, exist_ok=True)
        
        if not model_url:
            logger.warning(f"No URL for '{model_name_key}'. Ensured dir: {model_assets_dir}")
            return model_assets_dir
        
        needs_download = True
        if os.path.isdir(model_assets_dir):
            if model_url.endswith(".zip"):
                if len(os.listdir(model_assets_dir)) > 0:
                    logger.info(f"Assets for '{model_name_key}' seem to exist (from zip).")
                    needs_download = False
            else: 
                if any(Path(model_assets_dir).glob(f"{model_name_key}.*")) or any(Path(model_assets_dir).glob("*.pth")) or any(Path(model_assets_dir).glob("*.safetensors")):
                    logger.info(f"Asset file for '{model_name_key}' found.")
                    needs_download = False
        
        if needs_download:
            logger.info(f"Assets for {model_name_key} missing/incomplete. Processing URL...")
            if model_url.endswith(".zip"):
                if not self.download_and_extract_zip(model_url, model_assets_dir):
                    logger.error(f"Zip download/extract failed for {model_name_key}.")
            else:
                filename = self._get_filename_from_url(model_url)
                if model_name_key == "sam" and self.model_config.SAM_CONFIG.get("checkpoint"): 
                    filename = self.model_config.SAM_CONFIG["checkpoint"]
                elif not filename.lower().endswith((".pth", ".safetensors", ".pt", ".bin", ".onnx")):
                    filename = f"{model_name_key}.pth"
                target_file_path = os.path.join(model_assets_dir, filename)
                if not (os.path.exists(target_file_path) and os.path.getsize(target_file_path) > 0):
                    if not self.download_file(model_url, target_file_path, description=filename):
                        logger.error(f"File download failed for {model_name_key}.")
                else:
                    logger.info(f"File {target_file_path} already exists.")
        return model_assets_dir

    def load_main_model(self) -> None:
        """Încarcă modelul principal (SDXL Inpaint pe master)."""
        logger.info("Attempting to load main model (Master Branch - SDXL Inpaint)...")
        current_model = self.models.get('main')
        if isinstance(current_model, HiDreamModel) and current_model.is_loaded:
            logger.info("Main model already loaded.")
            return
        
        try:
            logger.info("Instantiating main model class...")
            main_model_instance = HiDreamModel() 
            if main_model_instance.load():
                self.models['main'] = main_model_instance
                logger.info("Main model loaded successfully via instance.load().")
            else:
                logger.error("Main model instance.load() returned False.")
                if 'main' in self.models: 
                    del self.models['main'] 
        except Exception as e:
            logger.error(f"Error during main model instantiation or load: {e}", exc_info=True)
            if 'main' in self.models: 
                self.unload_model('main') 

    def load_sam_model(self) -> None:
        """Încarcă modelul SAM."""
        logger.info("Attempting to load SAM model...")
        try:
            from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
            sam_checkpoint_filename = self.model_config.SAM_CONFIG.get("checkpoint", "sam_vit_h_4b8939.pth")
            sam_assets_dir = os.path.join(self.config.MODEL_DIR, "sam") 
            sam_checkpoint_path = os.path.join(sam_assets_dir, sam_checkpoint_filename)
            
            if not (os.path.exists(sam_checkpoint_path) and os.path.getsize(sam_checkpoint_path) > 0):
                logger.info(f"SAM checkpoint {sam_checkpoint_path} missing. Downloading...")
                model_url = self.config.MODEL_URLS.get("sam")
                if not model_url: 
                    logger.error("SAM URL missing.")
                    return
                if not self.download_file(model_url, sam_checkpoint_path, description=sam_checkpoint_filename): 
                    logger.error("SAM download failed.")
                    return
            
            logger.info(f"Loading SAM from: {sam_checkpoint_path}")
            sam_instance = sam_model_registry[self.model_config.SAM_CONFIG["model_type"]](checkpoint=sam_checkpoint_path)
            sam_instance.to(self.config.DEVICE)
            
            mask_generator = SamAutomaticMaskGenerator(sam_instance, **{k:v for k,v in self.model_config.SAM_CONFIG.items() if k != 'model_type' and k != 'checkpoint'})
            predictor = SamPredictor(sam_instance)
            self.models['sam'] = mask_generator
            self.models['sam_predictor'] = predictor
            logger.info("SAM Auto Mask Generator and Predictor loaded successfully.")
        except ImportError: 
            logger.error("Please install 'segment_anything': pip install segment-anything")
        except Exception as e: 
            logger.error(f"Error loading SAM model: {e}", exc_info=True)

    def load_clipseg_model(self) -> None:
        """Încarcă modelul CLIPSeg."""
        logger.info("Attempting to load CLIPSeg model...")
        try:
            from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
            model_id = self.model_config.CLIP_CONFIG.get("model_id", "CIDAS/clipseg-rd64-refined")
            processor = CLIPSegProcessor.from_pretrained(model_id, cache_dir=self.config.CACHE_DIR)
            model = CLIPSegForImageSegmentation.from_pretrained(model_id, torch_dtype=self.config.DTYPE, cache_dir=self.config.CACHE_DIR).to(self.config.DEVICE).eval()
            self.models['clipseg'] = {'processor': processor, 'model': model}
            logger.info(f"CLIPSeg model ('{model_id}') loaded successfully.")
        except ImportError: 
            logger.error("Please install/update 'transformers': pip install -U transformers")
        except Exception as e: 
            logger.error(f"Error loading CLIPSeg model '{model_id}': {e}", exc_info=True)

    def _load_yolo_model(self) -> Optional[Any]:
        logger.info("Attempting to load YOLO model (yolov8x-seg.pt)...")
        try:
            from ultralytics import YOLO
            yolo_model_name = "yolov8x-seg.pt"
            model_instance = YOLO(yolo_model_name) 
            self.models['yolo'] = model_instance
            logger.info(f"YOLO model ('{yolo_model_name}') loaded successfully.")
            return model_instance
        except ImportError: 
            logger.error("Please install 'ultralytics': pip install ultralytics")
            return None
        except Exception as e: 
            logger.error(f"Error loading YOLO model: {e}", exc_info=True)
            return None

    def _load_mediapipe_model(self) -> Optional[Any]: 
        logger.info("Attempting to load MediaPipe Selfie Segmentation model...")
        try:
            import mediapipe as mp
            mp_opts = getattr(self.model_config, "MEDIAPIPE_SELFIE_OPTIONS", {"model_selection": 1})
            segmenter = mp.solutions.selfie_segmentation.SelfieSegmentation(**mp_opts)
            self.models['mediapipe'] = segmenter 
            logger.info(f"MediaPipe Selfie Segmentation model loaded (options: {mp_opts}).")
            return segmenter
        except ImportError: 
            logger.error("Please install 'mediapipe': pip install mediapipe")
            return None
        except Exception as e: 
            logger.error(f"Error loading MediaPipe Selfie Segmentation: {e}", exc_info=True)
            return None

    def _load_face_detector(self) -> Optional[Any]:
        logger.info("Attempting to load MediaPipe Face Detection model...")
        try:
            import mediapipe as mp
            mp_opts = getattr(self.model_config, "MEDIAPIPE_FACE_OPTIONS", {"model_selection": 0, "min_detection_confidence": 0.5})
            detector = mp.solutions.face_detection.FaceDetection(**mp_opts)
            self.models['face_detector'] = detector
            logger.info(f"MediaPipe Face Detection model loaded (options: {mp_opts}).")
            return detector
        except ImportError: 
            logger.error("Please install 'mediapipe': pip install mediapipe")
            return None
        except Exception as e: 
            logger.error(f"Error loading MediaPipe Face Detection: {e}", exc_info=True)
            return None

    def _load_rembg_model(self) -> Optional[Any]: 
        logger.info("Attempting to create rembg session...")
        try:
            from rembg import new_session 
            session_model_name = getattr(self.model_config, "REMBG_MODEL_NAME", "u2net")
            session = new_session(model_name=session_model_name) 
            self.models['rembg'] = session 
            logger.info(f"Rembg session for model '{session_model_name}' created successfully.")
            return session
        except ImportError: 
            logger.error("Please install 'rembg': pip install rembg[gpu] or pip install rembg")
            return None
        except Exception as e: 
            logger.error(f"Error creating rembg session: {e}", exc_info=True)
            return None

    def load_specialized_model(self, model_name: str) -> Any:
        if model_name in self.models and self.models[model_name] is not None:
            return self.models[model_name]
        
        logger.info(f"Explicit call to load_specialized_model for: '{model_name}'.")
        if model_name == 'yolo': 
            return self._load_yolo_model()
        elif model_name == 'mediapipe': 
            return self._load_mediapipe_model()
        elif model_name == 'face_detector': 
            return self._load_face_detector()
        elif model_name == 'rembg': 
            return self._load_rembg_model()
        else: 
            logger.warning(f"Unknown specialized model name for direct loading: '{model_name}'.")
            return None

    def load_all_models(self) -> None:
        """Încarcă modelele esențiale la pornire."""
        logger.info("Loading all essential models (main, sam, clipseg)...")
        self.load_main_model()
        self.load_sam_model()
        self.load_clipseg_model()
        logger.info("Essential models loading process initiated.")

    def unload_model(self, model_name: str) -> None:
        """Descarcă un model specificat."""
        model_instance = self.models.get(model_name)
        if model_instance is not None:
            logger.info(f"Attempting to unload model: '{model_name}'...")
            unload_successful = False
            try:
                if model_name == 'main' and hasattr(model_instance, 'unload') and callable(model_instance.unload):
                    logger.debug(f"Calling custom 'unload()' for main model.")
                    model_instance.unload()
                elif (model_name == 'mediapipe' or model_name == 'face_detector') and hasattr(model_instance, 'close') and callable(model_instance.close):
                    logger.debug(f"Calling 'close()' for MediaPipe model '{model_name}'.")
                    model_instance.close()
                del self.models[model_name]
                unload_successful = True
            except Exception as e:
                logger.error(f"Error during specific unload/close for {model_name}: {e}", exc_info=True)
                if model_name in self.models: 
                    del self.models[model_name]
            
            if unload_successful:
                if torch.cuda.is_available(): 
                    logger.debug("Clearing CUDA cache.")
                    torch.cuda.empty_cache()
                logger.debug("Collecting garbage.")
                gc.collect()
                logger.info(f"Model '{model_name}' unloaded and memory cleanup attempted.")
            else:
                logger.warning(f"Model '{model_name}' reference removed, but unload/close might have failed.")
        else:
            logger.info(f"Model '{model_name}' not found or already unloaded.")

    def get_model(self, model_name: str) -> Any:
        """Obține un model, încărcându-l leneș dacă este necesar."""
        if model_name == 'main':
            main_model = self.models.get('main')
            if main_model and getattr(main_model, 'is_loaded', False):
                return main_model
        elif model_name in self.models and self.models[model_name] is not None:
            return self.models[model_name]
        
        logger.info(f"Model '{model_name}' not loaded or invalid. Attempting to load...")
        if model_name == 'main': 
            self.load_main_model()
        elif model_name == 'sam': 
            self.load_sam_model()
        elif model_name == 'sam_predictor': 
            if 'sam_predictor' not in self.models: 
                self.load_sam_model()
        elif model_name == 'clipseg': 
            self.load_clipseg_model()
        elif model_name == 'yolo': 
            self._load_yolo_model() 
        elif model_name == 'mediapipe': 
            self._load_mediapipe_model()
        elif model_name == 'face_detector': 
            self._load_face_detector()
        elif model_name == 'rembg': 
            self._load_rembg_model()
        else: 
            logger.warning(f"No specific loading logic in get_model for key: '{model_name}'.")
            return None
        
        loaded_model = self.models.get(model_name)
        if loaded_model is None and model_name != 'main': 
            logger.error(f"Failed to load model '{model_name}'. Check logs.")
        elif model_name == 'main' and (not loaded_model or not getattr(loaded_model, 'is_loaded', False)):
            logger.error(f"Failed to load main model '{model_name}'. Check logs.")
            return None 
        return loaded_model