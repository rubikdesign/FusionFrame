#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Manager pentru modelele AI folosite în FusionFrame 2.0
(Versiune Refăcută Inteligent - ControlNet gestionat intern de HiDreamModel)
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

from config.app_config import AppConfig
from config.model_config import ModelConfig

# Importuri pentru modelele transformers auxiliare
try:
    from transformers import (
        AutoImageProcessor,
        AutoModelForImageClassification,
        AutoModelForDepthEstimation # Adăugat pentru Depth Estimation
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    logging.getLogger(__name__).warning(
        "Transformers library not found or outdated. Image classification and/or Depth Estimation might not work. "
        "Please ensure 'transformers' is installed: pip install transformers"
    )
    AutoImageProcessor, AutoModelForImageClassification, AutoModelForDepthEstimation = None, None, None
    TRANSFORMERS_AVAILABLE = False

# Setăm logger-ul
logger = logging.getLogger(__name__)

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
        """Extrage numele fișierului dintr-un URL, cu fallback."""
        try:
            parsed_path = requests.utils.urlparse(url).path
            filename = os.path.basename(parsed_path)
            if not filename:
                parts = [part for part in parsed_path.split('/') if part]
                if parts:
                    filename = parts[-1]
            return filename if filename else "downloaded_file"
        except Exception:
            logger.warning(f"Could not parse filename from URL: {url}. Using default.")
            return "downloaded_file"

    def download_file(self, url: str, destination_path: str, description: Optional[str] = None) -> Optional[str]:
        """Descarcă un fișier cu bară de progres și gestionare erori."""
        desc_name = description or os.path.basename(destination_path)
        logger.info(f"Attempting to download file from {url} to {destination_path}")

        try:
            os.makedirs(os.path.dirname(destination_path), exist_ok=True)
            response = requests.get(url, stream=True, timeout=60)
            response.raise_for_status()
            total_size = int(response.headers.get('content-length', 0))

            with tqdm(total=total_size, unit='B', unit_scale=True, desc=f"Downloading {desc_name}", leave=False) as pbar:
                with open(destination_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            logger.info(f"Download complete: {destination_path}")
            return destination_path
        except requests.exceptions.Timeout:
            logger.error(f"Timeout while downloading {desc_name} from {url}.", exc_info=True)
        except requests.exceptions.RequestException as e_req:
            logger.error(f"Request failed for {desc_name} from {url}: {e_req}", exc_info=True)
        except IOError as e_io:
            logger.error(f"IO error writing {desc_name} to {destination_path}: {e_io}", exc_info=True)
        except Exception as e:
            logger.error(f"An unexpected error occurred during download of {desc_name}: {e}", exc_info=True)

        if os.path.exists(destination_path):
            try:
                if os.path.getsize(destination_path) == 0 or (total_size > 0 and os.path.getsize(destination_path) != total_size):
                    logger.warning(f"Removing incomplete or zero-byte file: {destination_path}")
                    os.remove(destination_path)
            except OSError:
                logger.warning(f"Could not remove potentially corrupted file: {destination_path}")
        return None

    def download_and_extract_zip(self, url: str, destination_extract_dir: str) -> Optional[str]:
        """Descarcă și extrage o arhivă zip în directorul specificat."""
        logger.info(f"Attempting to download and extract zip from {url} to {destination_extract_dir}")
        os.makedirs(destination_extract_dir, exist_ok=True)
        os.makedirs(self.config.CACHE_DIR, exist_ok=True)

        zip_filename = self._get_filename_from_url(url)
        if not zip_filename.lower().endswith(".zip"):
            zip_filename += ".zip"

        temp_zip_path = os.path.join(self.config.CACHE_DIR, zip_filename)
        temp_zip_path_obj = Path(temp_zip_path)

        try:
            if not (temp_zip_path_obj.exists() and temp_zip_path_obj.is_file() and temp_zip_path_obj.stat().st_size > 0):
                 if not self.download_file(url, temp_zip_path, description=zip_filename):
                    logger.error(f"Failed to download zip file for extraction from {url}.")
                    return None
            else:
                logger.info(f"Zip file {temp_zip_path} already exists in cache. Skipping download.")

            logger.info(f"Extracting {temp_zip_path} to {destination_extract_dir}...")
            with zipfile.ZipFile(temp_zip_path, 'r') as zip_ref:
                zip_ref.extractall(destination_extract_dir)
            logger.info(f"Extraction complete into: {destination_extract_dir}")
            return destination_extract_dir
        except zipfile.BadZipFile as e_zip:
            logger.error(f"Failed to extract zip file {temp_zip_path}: Invalid zip file. {e_zip}", exc_info=True)
            if temp_zip_path_obj.exists():
                try:
                    os.remove(temp_zip_path)
                    logger.info(f"Removed corrupted zip file: {temp_zip_path}")
                except OSError:
                    logger.warning(f"Could not remove corrupted zip file: {temp_zip_path}")
        except Exception as e:
            logger.error(f"An unexpected error during zip download/extraction: {e}", exc_info=True)
        return None

    def ensure_model(self, model_name_key: str) -> str:
        """Metodă păstrată pentru compatibilitate, dar descărcarea efectivă ar trebui gestionată prin metodele de load specifice."""
        # (Codul metodei ensure_model rămâne neschimbat față de versiunea anterioară)
        model_assets_dir = os.path.join(self.config.MODEL_DIR, model_name_key)
        model_url = self.config.MODEL_URLS.get(model_name_key)

        os.makedirs(model_assets_dir, exist_ok=True)

        if not model_url:
            logger.warning(f"No URL for model key '{model_name_key}'. Only ensured directory: {model_assets_dir}")
            return model_assets_dir

        if model_name_key == "sam":
            sam_checkpoint_filename = self.model_config.SAM_CONFIG.get("checkpoint", "sam_vit_h_4b8939.pth")
            expected_sam_path = os.path.join(model_assets_dir, sam_checkpoint_filename)
            if os.path.exists(expected_sam_path) and os.path.getsize(expected_sam_path) > 0:
                logger.info(f"SAM checkpoint {expected_sam_path} already exists.")
                return model_assets_dir

        if os.path.isdir(model_assets_dir) and len(os.listdir(model_assets_dir)) > 0 and model_url.endswith(".zip"):
            logger.info(f"Assets for '{model_name_key}' seem to exist in {model_assets_dir} (likely from previous zip extraction).")
            return model_assets_dir

        logger.info(f"Assets for '{model_name_key}' in {model_assets_dir} might be missing or incomplete. Processing URL: {model_url}")
        if model_url.endswith(".zip"):
            if not self.download_and_extract_zip(model_url, model_assets_dir):
                logger.error(f"Failed to download and extract zip for {model_name_key} into {model_assets_dir}.")
        else:
            filename = self._get_filename_from_url(model_url)
            if model_name_key == "sam" and self.model_config.SAM_CONFIG.get("checkpoint"):
                filename = self.model_config.SAM_CONFIG["checkpoint"]
            elif not filename.lower().endswith((".pth", ".safetensors", ".pt", ".bin", ".onnx")):
                default_ext = ".pth"
                logger.warning(f"URL for '{model_name_key}' ('{filename}') lacks a clear model extension. Assuming '{default_ext}'.")
                actual_filename_parts = os.path.splitext(filename)
                filename = actual_filename_parts[0] + default_ext

            target_file_path = os.path.join(model_assets_dir, filename)
            if not (os.path.exists(target_file_path) and os.path.getsize(target_file_path) > 0):
                if not self.download_file(model_url, target_file_path, description=filename):
                    logger.error(f"Failed to download model file for {model_name_key} to {target_file_path}.")
            else:
                logger.info(f"File {target_file_path} for '{model_name_key}' already exists.")

        return model_assets_dir


    # --- Metode de Încărcare Specifice ---

    def load_main_model(self) -> None:
        """Încarcă modelul principal de editare (HiDream)."""
        logger.info("Attempting to load main HiDream model...")
        try:
            from models.hidream_model import HiDreamModel # Import local pentru a evita dependențe circulare

            if 'main' in self.models and isinstance(self.models.get('main'), HiDreamModel) and self.models['main'].is_loaded:
                logger.info("Main HiDream model is already loaded.")
                return

            logger.info("Instantiating HiDreamModel for the main model...")
            main_model_instance = HiDreamModel() # Configurația este preluată în __init__-ul HiDreamModel

            if main_model_instance.load():
                self.models['main'] = main_model_instance
                logger.info("Main HiDream model (HiDreamModel class) loaded successfully.")
            else:
                logger.error("HiDreamModel.load() returned False. Main model not loaded.")
                if 'main' in self.models: del self.models['main']
        except ImportError:
            logger.error("HiDreamModel class not found. Ensure 'models.hidream_model.py' is correct.", exc_info=True)
        except Exception as e:
            logger.error(f"An unexpected error occurred while loading the main HiDream model: {e}", exc_info=True)
            if 'main' in self.models: self.unload_model('main')

    def load_sam_model(self) -> None:
        """Încarcă modelul SAM și generatorul de măști automate."""
        logger.info("Attempting to load SAM model...")
        try:
            from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
        except ImportError:
             logger.error("The 'segment_anything' library is not installed. Please install it: pip install segment-anything")
             return

        try:
            sam_assets_dir = self.ensure_model("sam")
            sam_checkpoint_filename = self.model_config.SAM_CONFIG.get("checkpoint", "sam_vit_h_4b8939.pth")
            sam_checkpoint_path = os.path.join(sam_assets_dir, sam_checkpoint_filename)

            if not (os.path.exists(sam_checkpoint_path) and os.path.getsize(sam_checkpoint_path) > 0):
                logger.error(f"SAM checkpoint '{sam_checkpoint_path}' not found or empty even after ensure_model call.")
                return

            logger.info(f"Loading SAM model from checkpoint: {sam_checkpoint_path}")
            sam_instance = sam_model_registry[self.model_config.SAM_CONFIG["model_type"]](
                checkpoint=sam_checkpoint_path
            )
            sam_instance.to(self.config.DEVICE)

            mask_generator = SamAutomaticMaskGenerator(
                sam_instance,
                points_per_side=self.model_config.SAM_CONFIG.get("points_per_side", 32),
                pred_iou_thresh=self.model_config.SAM_CONFIG.get("pred_iou_thresh", 0.95),
                stability_score_thresh=self.model_config.SAM_CONFIG.get("stability_score_thresh", 0.97),
                min_mask_region_area=self.model_config.SAM_CONFIG.get("min_mask_region_area", 100)
            )
            self.models['sam'] = mask_generator
            logger.info("SAM model and mask generator loaded successfully.")

        except KeyError as e_key:
            logger.error(f"Configuration key error for SAM: {e_key}. Check ModelConfig.", exc_info=True)
        except Exception as e:
            logger.error(f"An unexpected error occurred while loading SAM model: {e}", exc_info=True)

    def load_clipseg_model(self) -> None:
        """Încarcă modelul CLIPSeg și procesorul său."""
        logger.info("Attempting to load CLIPSeg model...")
        if not TRANSFORMERS_AVAILABLE:
            logger.error("Transformers library not available for CLIPSeg.")
            return
        try:
            from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation

            model_id = self.model_config.CLIP_CONFIG.get("model_id", "CIDAS/clipseg-rd64-refined")

            processor = CLIPSegProcessor.from_pretrained(model_id, cache_dir=self.config.CACHE_DIR)
            model = CLIPSegForImageSegmentation.from_pretrained(
                model_id,
                torch_dtype=self.config.DTYPE,
                cache_dir=self.config.CACHE_DIR
            ).to(self.config.DEVICE).eval()

            self.models['clipseg'] = {'processor': processor, 'model': model}
            logger.info(f"CLIPSeg model ('{model_id}') loaded successfully.")
        except ImportError:
            logger.error("Could not import CLIPSeg components from transformers.")
        except Exception as e:
            logger.error(f"An error occurred while loading CLIPSeg model from '{model_id}': {e}", exc_info=True)

    def _load_yolo_model(self) -> Optional[Any]:
        """Încarcă modelul YOLOv8 pentru segmentare."""
        logger.info("Attempting to load YOLO model (yolov8x-seg.pt)...")
        try:
            from ultralytics import YOLO
        except ImportError:
            logger.error("The 'ultralytics' library is not installed. Please install it: pip install ultralytics")
            return None

        try:
            yolo_model_name = "yolov8x-seg.pt"
            yolo_preferred_path = Path(self.config.MODEL_DIR) / "YOLO" / yolo_model_name

            model_to_load_path_or_name = str(yolo_preferred_path) if yolo_preferred_path.is_file() else yolo_model_name
            if yolo_preferred_path.is_file():
                 logger.info(f"Loading YOLO from preferred path: {yolo_preferred_path}")
            else:
                 logger.info(f"YOLO preferred path {yolo_preferred_path} not found. Using model name '{yolo_model_name}'.")

            model_instance = YOLO(model_to_load_path_or_name)
            # YOLO ar trebui să gestioneze device-ul, dar putem forța dacă e necesar și suportat
            # model_instance.to(self.config.DEVICE)
            self.models['yolo'] = model_instance
            logger.info(f"YOLO model ('{yolo_model_name}') loaded successfully.")
            return model_instance
        except Exception as e:
            logger.error(f"Error loading YOLO model: {e}", exc_info=True)
            return None

    def _load_mediapipe_model(self) -> Optional[Any]:
        """Încarcă modelul MediaPipe pentru segmentarea selfie."""
        logger.info("Attempting to load MediaPipe Selfie Segmentation model...")
        try:
            import mediapipe as mp
            mp_solutions = mp.solutions.selfie_segmentation
            mp_model_selection = getattr(self.model_config, "MEDIAPIPE_SELFIE_MODEL_SELECTION", 1)
            segmenter = mp_solutions.SelfieSegmentation(model_selection=mp_model_selection)
            self.models['mediapipe'] = segmenter
            logger.info(f"MediaPipe Selfie Segmentation model (selection={mp_model_selection}) loaded.")
            return segmenter
        except ImportError:
            logger.error("The 'mediapipe' library is not installed. Please install it: pip install mediapipe")
            return None
        except Exception as e:
            logger.error(f"Error loading MediaPipe Selfie Segmentation model: {e}", exc_info=True)
            return None

    def _load_face_detector(self) -> Optional[Any]:
        """Încarcă modelul MediaPipe pentru detecția feței."""
        logger.info("Attempting to load MediaPipe Face Detection model...")
        try:
            import mediapipe as mp
            mp_solutions = mp.solutions.face_detection
            mp_model_selection = getattr(self.model_config, "MEDIAPIPE_FACE_MODEL_SELECTION", 0)
            mp_min_confidence = getattr(self.model_config, "MEDIAPIPE_FACE_MIN_CONFIDENCE", 0.5)
            detector = mp_solutions.FaceDetection(
                model_selection=mp_model_selection,
                min_detection_confidence=mp_min_confidence
            )
            self.models['face_detector'] = detector
            logger.info(f"MediaPipe Face Detection model (selection={mp_model_selection}, confidence={mp_min_confidence}) loaded.")
            return detector
        except ImportError:
            logger.error("The 'mediapipe' library is not installed. Please install it: pip install mediapipe")
            return None
        except Exception as e:
            logger.error(f"Error loading MediaPipe Face Detection model: {e}", exc_info=True)
            return None

    def _load_rembg_model(self) -> Optional[Any]:
        """Creează o sesiune Rembg pentru eliminarea fundalului."""
        logger.info("Attempting to create rembg session...")
        try:
            from rembg import new_session
        except ImportError:
            logger.error("The 'rembg' library is not installed. Please install it: pip install rembg")
            return None

        try:
            session_model_name = getattr(self.model_config, "REMBG_MODEL_NAME", "u2net")
            # Rembg gestionează descărcarea intern
            session = new_session(model_name=session_model_name)
            self.models['rembg'] = session
            logger.info(f"Rembg session for model '{session_model_name}' created successfully.")
            return session
        except Exception as e:
            logger.error(f"Error creating rembg session (model: {session_model_name if 'session_model_name' in locals() else 'default'}): {e}", exc_info=True)
            return None

    def _load_image_classifier(self) -> Optional[Dict[str, Any]]:
        """Încarcă modelul de clasificare a imaginilor și procesorul său."""
        logger.info("Attempting to load Image Classification model...")
        if not TRANSFORMERS_AVAILABLE or AutoImageProcessor is None or AutoModelForImageClassification is None:
            logger.error("Transformers library components not available for Image Classification.")
            return None
        try:
            classifier_config = getattr(self.model_config, "IMAGE_CLASSIFIER_CONFIG", {})
            classifier_model_id = classifier_config.get("model_id", "google/vit-base-patch16-224")

            processor = AutoImageProcessor.from_pretrained(
                classifier_model_id,
                cache_dir=self.config.CACHE_DIR
            )
            # Încărcăm modelul pe CPU inițial dacă suntem în low VRAM, apoi îl mutăm la nevoie
            # Sau lăsăm offload-ul să gestioneze? Pentru simplitate, încărcăm pe device-ul principal.
            model = AutoModelForImageClassification.from_pretrained(
                classifier_model_id,
                torch_dtype=self.config.DTYPE,
                cache_dir=self.config.CACHE_DIR
            ).to(self.config.DEVICE).eval()

            bundle = {'processor': processor, 'model': model}
            self.models['image_classifier'] = bundle
            logger.info(f"Image Classification model ('{classifier_model_id}') loaded successfully.")
            return bundle
        except Exception as e:
            default_id = "google/vit-base-patch16-224"
            logger.error(f"Error loading Image Classification model ('{classifier_model_id if 'classifier_model_id' in locals() else default_id}'): {e}", exc_info=True)
            return None

    # NOU: Metodă pentru încărcarea modelului de estimare a adâncimii
    def _load_depth_estimator(self) -> Optional[Dict[str, Any]]:
        """Încarcă modelul de estimare a adâncimii și procesorul său."""
        logger.info("Attempting to load Depth Estimation model...")
        if not TRANSFORMERS_AVAILABLE or AutoImageProcessor is None or AutoModelForDepthEstimation is None:
            logger.error("Transformers library components not available for Depth Estimation.")
            return None
        try:
            depth_config = getattr(self.model_config, "DEPTH_ESTIMATOR_CONFIG", {})
            depth_model_id = depth_config.get("model_id", "Intel/dpt-hybrid-midas")

            processor = AutoImageProcessor.from_pretrained(
                depth_model_id,
                cache_dir=self.config.CACHE_DIR
            )
            model = AutoModelForDepthEstimation.from_pretrained(
                depth_model_id,
                torch_dtype=self.config.DTYPE, # Folosim float16 dacă e disponibil
                cache_dir=self.config.CACHE_DIR
            ).to(self.config.DEVICE).eval()

            bundle = {'processor': processor, 'model': model}
            self.models['depth_estimator'] = bundle
            logger.info(f"Depth Estimation model ('{depth_model_id}') loaded successfully.")
            return bundle
        except Exception as e:
            default_id = "Intel/dpt-hybrid-midas"
            logger.error(f"Error loading Depth Estimation model ('{depth_model_id if 'depth_model_id' in locals() else default_id}'): {e}", exc_info=True)
            return None


    # --- Metode de Gestionare ---

    def load_specialized_model(self, model_name: str) -> Any:
        """Încarcă un model auxiliar la cerere (folosit mai rar, prefer get_model)."""
        if model_name in self.models and self.models[model_name] is not None:
            logger.debug(f"Model '{model_name}' already loaded, returning existing instance.")
            return self.models[model_name]

        logger.info(f"Explicit call to load_specialized_model for: '{model_name}'.")

        loader_map = {
            'yolo': self._load_yolo_model,
            'mediapipe': self._load_mediapipe_model, # Selfie segmentation
            'face_detector': self._load_face_detector,
            'rembg': self._load_rembg_model,
            'image_classifier': self._load_image_classifier,
            'depth_estimator': self._load_depth_estimator, # Adăugat
            # Modelele 'main', 'sam', 'clipseg' sunt considerate esențiale și încărcate prin get_model sau load_all_models
        }

        loader_func = loader_map.get(model_name)
        if loader_func:
            return loader_func()
        else:
            logger.warning(f"Unknown specialized model name for direct loading: '{model_name}'. Use get_model for standard models.")
            return None

    def load_all_models(self) -> None:
        """Încarcă modelele considerate esențiale la pornire."""
        logger.info("Loading essential models (main, sam, clipseg)...")
        # Acestea vor fi încărcate dacă nu sunt deja prezente
        self.get_model('main')
        self.get_model('sam')
        self.get_model('clipseg')
        logger.info("Essential models loading process initiated/checked.")

    def unload_model(self, model_name: str) -> None:
        """Descarcă un model specificat pentru a elibera memorie."""
        model_instance_or_bundle = self.models.pop(model_name, None) # Folosim pop pentru a șterge și a obține valoarea

        if model_instance_or_bundle is not None:
            logger.info(f"Attempting to unload model: '{model_name}'...")

            try:
                # Tratează cazul în care modelul este un dicționar (bundle)
                actual_model_instance = model_instance_or_bundle
                processor_instance = None
                if isinstance(model_instance_or_bundle, dict):
                    actual_model_instance = model_instance_or_bundle.get('model')
                    processor_instance = model_instance_or_bundle.get('processor')

                # Apelăm metode specifice de unload/close dacă există
                if model_name == 'main' and hasattr(actual_model_instance, 'unload') and callable(getattr(actual_model_instance, 'unload', None)):
                    logger.debug(f"Calling custom 'unload()' method for main model ({actual_model_instance.__class__.__name__}).")
                    actual_model_instance.unload()
                elif (model_name == 'mediapipe' or model_name == 'face_detector') and hasattr(actual_model_instance, 'close') and callable(getattr(actual_model_instance, 'close', None)):
                    logger.debug(f"Calling 'close()' method for MediaPipe model '{model_name}'.")
                    actual_model_instance.close()

                # Eliberăm referințele explicit
                del actual_model_instance
                if processor_instance:
                    del processor_instance
                del model_instance_or_bundle

                # Eliberăm memoria CUDA și colectăm gunoiul Python
                if torch.cuda.is_available():
                    logger.debug("Clearing CUDA cache after unloading model.")
                    torch.cuda.empty_cache()
                logger.debug("Collecting garbage after unloading model.")
                gc.collect()
                logger.info(f"Model '{model_name}' unloaded and memory cleanup attempted.")

            except Exception as e_unload:
                 logger.error(f"Error during unload/cleanup for {model_name}: {e_unload}", exc_info=True)
                 # Chiar dacă apare o eroare, modelul a fost scos din dicționarul self.models

        else:
            logger.info(f"Model '{model_name}' not found in manager or already unloaded.")

    def get_model(self, model_name: str) -> Any:
        """
        Obține un model din manager, încărcându-l leneș dacă nu este deja încărcat.
        """
        if model_name in self.models and self.models[model_name] is not None:
            # Verificare suplimentară pentru HiDreamModel dacă e încărcat corect
            if model_name == 'main':
                 main_model = self.models['main']
                 if isinstance(main_model, BaseModel) and not main_model.is_loaded:
                     logger.warning("Main model found in cache but not marked as loaded. Attempting reload.")
                     self.load_main_model() # Încercăm reîncărcarea
                 elif not isinstance(main_model, BaseModel): # Dacă nu e instanță BaseModel
                     logger.error(f"Cached object for 'main' is not a BaseModel instance ({type(main_model)}). Removing cache and attempting reload.")
                     self.unload_model('main') # Curățăm cache-ul invalid
                     self.load_main_model()
                 # Returnează modelul (posibil reîncărcat) sau None dacă reîncărcarea eșuează
                 return self.models.get(model_name)
            # Pentru alte modele, returnăm direct
            return self.models[model_name]

        logger.info(f"Model '{model_name}' not loaded. Attempting lazy load...")

        loader_func = None
        # Mapăm numele modelului la funcția sa de încărcare specifică
        if model_name == 'main':
            loader_func = self.load_main_model
        elif model_name == 'sam':
            loader_func = self.load_sam_model
        elif model_name == 'clipseg':
            loader_func = self.load_clipseg_model
        elif model_name == 'yolo':
            loader_func = self._load_yolo_model
        elif model_name == 'mediapipe': # Selfie segmentation
            loader_func = self._load_mediapipe_model
        elif model_name == 'face_detector':
            loader_func = self._load_face_detector
        elif model_name == 'rembg':
            loader_func = self._load_rembg_model
        elif model_name == 'image_classifier':
            loader_func = self._load_image_classifier
        elif model_name == 'depth_estimator': # Adăugat
            loader_func = self._load_depth_estimator
        # ControlNet este gestionat intern de HiDreamModel, deci nu are loader propriu aici

        if loader_func:
            loader_func() # Apelăm funcția de încărcare (care populează self.models)
        else:
            logger.warning(f"No specific loading function defined in get_model for key: '{model_name}'.")
            return None

        # Returnăm modelul proaspăt încărcat (sau None dacă încărcarea a eșuat)
        loaded_model = self.models.get(model_name)
        if loaded_model is None:
            logger.error(f"Failed to load model '{model_name}' during lazy load attempt. Check specific loader logs.")
        return loaded_model