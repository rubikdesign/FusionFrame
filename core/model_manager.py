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
        
        self.config.ensure_dirs() # Asigură că directoarele MODEL_DIR, CACHE_DIR etc. există
        
        self._initialized = True
        logger.info("ModelManager initialized")

    def _get_filename_from_url(self, url: str) -> str:
        """Extrage numele fișierului dintr-un URL, cu fallback."""
        try:
            parsed_path = requests.utils.urlparse(url).path
            filename = os.path.basename(parsed_path)
            if not filename: # Dacă path-ul se termină cu /
                # Încercăm să luăm ultima componentă nenulă
                parts = [part for part in parsed_path.split('/') if part]
                if parts:
                    filename = parts[-1]
            return filename if filename else "downloaded_file" # Fallback dacă tot e gol
        except Exception:
            logger.warning(f"Could not parse filename from URL: {url}. Using default.")
            return "downloaded_file"

    def download_file(self, url: str, destination_path: str, description: Optional[str] = None) -> Optional[str]:
        """Descarcă un fișier cu bară de progres și gestionare erori."""
        desc_name = description or os.path.basename(destination_path)
        logger.info(f"Attempting to download file from {url} to {destination_path}")
        
        try:
            os.makedirs(os.path.dirname(destination_path), exist_ok=True)
            response = requests.get(url, stream=True, timeout=60) # Timeout mărit la 60s
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
        except Exception as e: # Prindem orice altă excepție
            logger.error(f"An unexpected error occurred during download of {desc_name}: {e}", exc_info=True)
        
        # Cleanup în caz de eșec
        if os.path.exists(destination_path): # nosemgrep: standard-library-memory-resource-management-issues
            try:
                if os.path.getsize(destination_path) == 0 or (total_size > 0 and os.path.getsize(destination_path) != total_size):
                    logger.warning(f"Removing incomplete or zero-byte file: {destination_path}")
                    os.remove(destination_path) # nosemgrep: standard-library-memory-resource-management-issues
            except OSError:
                logger.warning(f"Could not remove potentially corrupted file: {destination_path}")
        return None

    def download_and_extract_zip(self, url: str, destination_extract_dir: str) -> Optional[str]:
        """Descarcă și extrage o arhivă zip în directorul specificat."""
        logger.info(f"Attempting to download and extract zip from {url} to {destination_extract_dir}")
        # Directorul temporar pentru descărcarea zip-ului poate fi diferit de cel de extragere
        # sau putem folosi CACHE_DIR pentru descărcări temporare.
        # Aici, vom descărca în CACHE_DIR și extrage în destination_extract_dir.
        
        # Asigurăm că directorul de extragere există
        os.makedirs(destination_extract_dir, exist_ok=True)
        # Asigurăm că directorul cache pentru descărcare există
        os.makedirs(self.config.CACHE_DIR, exist_ok=True)

        zip_filename = self._get_filename_from_url(url)
        if not zip_filename.lower().endswith(".zip"):
            zip_filename += ".zip" # Adaugă extensia dacă lipsește din URL parsato
            
        temp_zip_path = os.path.join(self.config.CACHE_DIR, zip_filename) # Descărcăm în CACHE_DIR
        
        temp_zip_path_obj = Path(temp_zip_path)

        try:
            # Verificăm dacă zip-ul există deja și e valid (verificare simplistă prin existență)
            # O verificare mai bună ar fi checksum dacă serverul îl oferă.
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
            if temp_zip_path_obj.exists(): # Șterge zip-ul corupt
                try:
                    os.remove(temp_zip_path) # nosemgrep: standard-library-memory-resource-management-issues
                    logger.info(f"Removed corrupted zip file: {temp_zip_path}")
                except OSError:
                    logger.warning(f"Could not remove corrupted zip file: {temp_zip_path}")
        except Exception as e:
            logger.error(f"An unexpected error during zip download/extraction: {e}", exc_info=True)
        # Nu mai ștergem zip-ul din CACHE_DIR automat, poate fi util pentru rulări ulterioare.
        # Se poate adăuga o logică de curățare a cache-ului periodic.
        return None
    
    def ensure_model(self, model_name_key: str) -> str:
        """
        PĂSTRAT PENTRU COMPATIBILITATE cu `load_sam_model` original.
        Se asigură că un director pentru model_name_key există în MODEL_DIR.
        Dacă model_url există și e .zip, îl descarcă și extrage în MODEL_DIR/model_name_key/.
        Dacă model_url există și NU e .zip, descarcă fișierul în MODEL_DIR/model_name_key/nume_fisier_din_url_sau_default.pth.
        Returnează calea către MODEL_DIR/model_name_key/.
        """
        model_assets_dir = os.path.join(self.config.MODEL_DIR, model_name_key)
        model_url = self.config.MODEL_URLS.get(model_name_key)

        os.makedirs(model_assets_dir, exist_ok=True) # Asigurăm că directorul base există

        if not model_url:
            logger.warning(f"No URL for model key '{model_name_key}'. Only ensured directory: {model_assets_dir}")
            return model_assets_dir

        # Verificare specifică pentru SAM (dacă fișierul .pth e deja acolo)
        if model_name_key == "sam":
            sam_checkpoint_filename = self.model_config.SAM_CONFIG.get("checkpoint", "sam_vit_h_4b8939.pth")
            expected_sam_path = os.path.join(model_assets_dir, sam_checkpoint_filename)
            if os.path.exists(expected_sam_path) and os.path.getsize(expected_sam_path) > 0:
                logger.info(f"SAM checkpoint {expected_sam_path} already exists.")
                return model_assets_dir # Directorul asset-urilor SAM

        # Logica generală de descărcare dacă nu e SAM sau SAM lipsește
        # Verificăm dacă directorul conține deja ceva, ca o heuristică pentru zip-uri extrase
        if os.path.isdir(model_assets_dir) and len(os.listdir(model_assets_dir)) > 0 and model_url.endswith(".zip"):
            logger.info(f"Assets for '{model_name_key}' seem to exist in {model_assets_dir} (likely from previous zip extraction).")
            return model_assets_dir
        
        logger.info(f"Assets for '{model_name_key}' in {model_assets_dir} might be missing or incomplete. Processing URL: {model_url}")
        if model_url.endswith(".zip"):
            if not self.download_and_extract_zip(model_url, model_assets_dir): # Extrage direct în model_assets_dir
                logger.error(f"Failed to download and extract zip for {model_name_key} into {model_assets_dir}.")
        else: # Nu e zip, descărcăm un singur fișier
            filename = self._get_filename_from_url(model_url)
            # Pentru SAM, folosim numele specificat în config
            if model_name_key == "sam" and self.model_config.SAM_CONFIG.get("checkpoint"):
                filename = self.model_config.SAM_CONFIG["checkpoint"]
            elif not filename.lower().endswith((".pth", ".safetensors", ".pt", ".bin", ".onnx")): # Extensii comune
                default_ext = ".pth" # Un default rezonabil
                logger.warning(f"URL for '{model_name_key}' ('{filename}') lacks a clear model extension. Assuming '{default_ext}'.")
                actual_filename_parts = os.path.splitext(filename)
                filename = actual_filename_parts[0] + default_ext


            target_file_path = os.path.join(model_assets_dir, filename)
            # Descărcăm doar dacă fișierul nu există sau e gol
            if not (os.path.exists(target_file_path) and os.path.getsize(target_file_path) > 0):
                if not self.download_file(model_url, target_file_path, description=filename):
                    logger.error(f"Failed to download model file for {model_name_key} to {target_file_path}.")
            else:
                logger.info(f"File {target_file_path} for '{model_name_key}' already exists.")
        
        return model_assets_dir

    def load_main_model(self) -> None:
        logger.info("Attempting to load main HiDream model...")
        try:
            from models.hidream_model import HiDreamModel 

            if 'main' in self.models and isinstance(self.models.get('main'), HiDreamModel) and self.models['main'].is_loaded:
                logger.info("Main HiDream model is already loaded.")
                return

            logger.info("Instantiating HiDreamModel for the main model...")
            # HiDreamModel își va prelua configurația (inclusiv MAIN_MODEL ID) din ModelConfig în __init__
            main_model_instance = HiDreamModel() 
            
            if main_model_instance.load(): # Metoda load() din HiDreamModel gestionează încărcarea reală
                self.models['main'] = main_model_instance
                logger.info("Main HiDream model (HiDreamModel class) loaded successfully.")
            else:
                logger.error("HiDreamModel.load() returned False. Main model not loaded.")
                if 'main' in self.models: del self.models['main'] # Curățăm referința dacă încărcarea eșuează
                # Nu ridicăm excepție aici, get_model va returna None, lăsând apelantul să decidă
        except ImportError:
            logger.error("HiDreamModel class not found. Ensure 'models.hidream_model.py' is correct and dependencies (like diffusers) are installed.", exc_info=True)
            # Nu ridicăm excepție, get_model va returna None
        except Exception as e:
            logger.error(f"An unexpected error occurred while loading the main HiDream model: {e}", exc_info=True)
            if 'main' in self.models: self.unload_model('main')
            # Nu ridicăm excepție, get_model va returna None
    
    def load_sam_model(self) -> None:
        logger.info("Attempting to load SAM model...")
        try:
            from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
            
            # Folosim ensure_model pentru a gestiona directorul și descărcarea fișierului .pth pentru SAM
            # ensure_model va crea MODEL_DIR/sam/ și va descărca checkpoint-ul acolo
            sam_assets_dir = self.ensure_model("sam") # "sam" este cheia din MODEL_URLS
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
        except ImportError:
            logger.error("The 'segment_anything' library is not installed. Please install it: pip install segment-anything")
        except KeyError as e_key: # Pentru SAM_CONFIG sau MODEL_URLS lipsă
            logger.error(f"Configuration key error for SAM: {e_key}. Check ModelConfig and AppConfig.", exc_info=True)
        except Exception as e:
            logger.error(f"An unexpected error occurred while loading SAM model: {e}", exc_info=True)
    
    def load_clipseg_model(self) -> None:
        logger.info("Attempting to load CLIPSeg model...")
        try:
            from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
            
            model_id = self.model_config.CLIP_CONFIG.get("model_id", "CIDAS/clipseg-rd64-refined")
            
            processor = CLIPSegProcessor.from_pretrained(model_id, cache_dir=self.config.CACHE_DIR)
            model = CLIPSegForImageSegmentation.from_pretrained(
                model_id,
                torch_dtype=self.config.DTYPE, # Specificăm dtype la încărcare
                cache_dir=self.config.CACHE_DIR
            ).to(self.config.DEVICE).eval() # Setăm în modul evaluare

            self.models['clipseg'] = {'processor': processor, 'model': model}
            logger.info(f"CLIPSeg model ('{model_id}') loaded successfully.")
        except ImportError:
            logger.error("The 'transformers' library (or CLIPSeg components) is not installed. Please install/update it: pip install transformers")
        except Exception as e: # Poate fi eroare de la HuggingFace (rețea, model ID greșit etc.)
            logger.error(f"An error occurred while loading CLIPSeg model from '{model_id}': {e}", exc_info=True)

    # Metodele _load_... vor popula self.models['nume_model'] și vor returna modelul
    def _load_yolo_model(self) -> Optional[Any]:
        logger.info("Attempting to load YOLO model (yolov8x-seg.pt)...")
        try:
            from ultralytics import YOLO
            yolo_model_name = "yolov8x-seg.pt" # Model specific pentru segmentare
            # YOLO va descărca în propriul cache dacă nu este găsit.
            # Calea specificată aici e mai mult o preferință dacă fișierul există deja acolo.
            yolo_preferred_path = Path(self.config.MODEL_DIR) / "YOLO" / yolo_model_name
            
            model_to_load_path_or_name = str(yolo_preferred_path) if yolo_preferred_path.is_file() else yolo_model_name
            if yolo_preferred_path.is_file():
                 logger.info(f"Loading YOLO from preferred path: {yolo_preferred_path}")
            else:
                 logger.info(f"YOLO preferred path {yolo_preferred_path} not found. Using model name '{yolo_model_name}' (YOLO will attempt download/cache).")

            model_instance = YOLO(model_to_load_path_or_name)
            # YOLO se ocupă de obicei de mutarea pe device.
            # model_instance.to(self.config.DEVICE) # Dacă e necesar și YOLO suportă direct.

            self.models['yolo'] = model_instance
            logger.info(f"YOLO model ('{yolo_model_name}') loaded successfully.")
            return model_instance
        except ImportError:
            logger.error("The 'ultralytics' library is not installed. Please install it: pip install ultralytics")
            return None
        except Exception as e:
            logger.error(f"Error loading YOLO model: {e}", exc_info=True)
            return None
    
    def _load_mediapipe_model(self) -> Optional[Any]: # Numele original, pentru selfie_segmentation
        logger.info("Attempting to load MediaPipe Selfie Segmentation model...")
        try:
            import mediapipe as mp
            mp_model_selection = getattr(self.model_config, "MEDIAPIPE_SELFIE_MODEL_SELECTION", 1) # 1 pentru general
            # Documentația MediaPipe recomandă folosirea `with` pentru soluții
            # Deoarece stocăm instanța, utilizatorul trebuie să fie conștient de gestionarea resurselor
            # (de ex., `close()` dacă e apelat la unload)
            segmenter = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=mp_model_selection)
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
        logger.info("Attempting to load MediaPipe Face Detection model...")
        try:
            import mediapipe as mp
            mp_model_selection = getattr(self.model_config, "MEDIAPIPE_FACE_MODEL_SELECTION", 0) # 0 for short-range, 1 for full-range
            mp_min_confidence = getattr(self.model_config, "MEDIAPIPE_FACE_MIN_CONFIDENCE", 0.5)
            detector = mp.solutions.face_detection.FaceDetection(
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
    
    def _load_rembg_model(self) -> Optional[Any]: # Va returna sesiunea rembg
        logger.info("Attempting to create rembg session...")
        try:
            from rembg import new_session # Funcția corectă pentru a obține o sesiune
            
            session_model_name = getattr(self.model_config, "REMBG_MODEL_NAME", "u2net") # u2net e un default bun
            # new_session va descărca modelul dacă nu e în cache-ul rembg
            session = new_session(model_name=session_model_name)
            self.models['rembg'] = session # Stocăm sesiunea, nu modulul
            logger.info(f"Rembg session for model '{session_model_name}' created successfully.")
            return session
        except ImportError:
            logger.error("The 'rembg' library is not installed. Please install it: pip install rembg (or rembg[gpu])")
            return None
        except Exception as e: # Poate fi eroare de la descărcarea modelului de către rembg
            logger.error(f"Error creating rembg session (model: {session_model_name if 'session_model_name' in locals() else 'default'}): {e}", exc_info=True)
            return None

    # Păstrăm load_specialized_model pentru compatibilitate cu structura originală
    # Aceasta va încerca să încarce modelul dacă nu este deja încărcat.
    def load_specialized_model(self, model_name: str) -> Any:
        # Verificăm dacă este deja încărcat
        if model_name in self.models and self.models[model_name] is not None:
            logger.debug(f"Model '{model_name}' already loaded, returning existing instance from load_specialized_model.")
            return self.models[model_name]

        logger.info(f"Call to load_specialized_model for: '{model_name}' (will attempt to load if not present).")
        
        # Mapăm numele la metoda de încărcare corespunzătoare
        # Această metodă practic devine un alias pentru get_model, dar forțează încărcarea dacă nu există.
        if model_name == 'yolo':
            return self._load_yolo_model()
        elif model_name == 'mediapipe': # Pentru selfie_segmentation
            return self._load_mediapipe_model()
        elif model_name == 'face_detector':
            return self._load_face_detector()
        elif model_name == 'rembg':
            return self._load_rembg_model()
        # **NU INCLUDEM 'controlnet' AICI** deoarece este gestionat de HiDreamModel
        else:
            logger.warning(f"Unknown specialized model name for direct loading: '{model_name}'. Use get_model for lazy loading of standard models (main, sam, clipseg) or check name.")
            return None

    def load_all_models(self) -> None:
        """Încarcă modelele considerate esențiale la pornire."""
        logger.info("Loading all essential models (main, sam, clipseg)...")
        self.load_main_model()
        self.load_sam_model()
        self.load_clipseg_model()
        # Alte modele (YOLO, MediaPipe, Rembg) sunt încărcate leneș prin get_model()
        # sau explicit prin load_specialized_model(nume) dacă se dorește.
        logger.info("Essential models loading process initiated.")
    
    def unload_model(self, model_name: str) -> None:
        """Descarcă un model specificat pentru a elibera memorie."""
        model_instance = self.models.get(model_name)

        if model_instance is not None:
            logger.info(f"Attempting to unload model: '{model_name}'...")
            
            # Logică specifică de unload/close
            if model_name == 'main' and hasattr(model_instance, 'unload') and callable(model_instance.unload):
                logger.debug(f"Calling custom 'unload()' method for main model ({model_instance.__class__.__name__}).")
                try: model_instance.unload()
                except Exception as e_unload: logger.error(f"Error during custom unload for {model_name}: {e_unload}", exc_info=True)
            elif (model_name == 'mediapipe' or model_name == 'face_detector') and hasattr(model_instance, 'close') and callable(model_instance.close):
                logger.debug(f"Calling 'close()' method for MediaPipe model '{model_name}'.")
                try: model_instance.close()
                except Exception as e_close: logger.error(f"Error during close for {model_name}: {e_close}", exc_info=True)
            # Pentru rembg (sesiune), nu există o metodă standard de 'close' expusă pe obiectul sesiune.
            # Ștergerea referinței și gc.collect() ar trebui să fie suficiente.
            
            # Ștergem referința din dicționarul managerului
            del self.models[model_name] 
            
            # Eliberăm memoria CUDA și colectăm gunoiul Python
            if torch.cuda.is_available():
                logger.debug("Clearing CUDA cache after unloading model.")
                torch.cuda.empty_cache()
            logger.debug("Collecting garbage after unloading model.")
            gc.collect()
            logger.info(f"Model '{model_name}' unloaded and memory cleanup attempted.")
        else:
            logger.info(f"Model '{model_name}' not found in manager or already unloaded.")
    
    def get_model(self, model_name: str) -> Any:
        """
        Obține un model din manager, încărcându-l leneș dacă nu este deja încărcat.
        Folosește numele exact al cheii așa cum este definit în metodele de încărcare.
        """
        if model_name in self.models and self.models[model_name] is not None:
            return self.models[model_name]
        
        logger.info(f"Model '{model_name}' not pre-loaded. Attempting to load via specific loader...")
        
        # Rutină de încărcare specifică bazată pe numele modelului (cheia)
        if model_name == 'main':
            self.load_main_model()
        elif model_name == 'sam':
            self.load_sam_model()
        elif model_name == 'clipseg':
            self.load_clipseg_model()
        elif model_name == 'yolo':
            self._load_yolo_model() 
        elif model_name == 'mediapipe': # Acesta încarcă selfie_segmentation
             self._load_mediapipe_model()
        elif model_name == 'face_detector':
             self._load_face_detector()
        elif model_name == 'rembg': # Acesta creează sesiunea rembg
             self._load_rembg_model()
        # **NU EXISTĂ CAZ PENTRU 'controlnet' AICI**
        else:
            logger.warning(f"No specific loading branch in get_model for key: '{model_name}'. Model will not be loaded by this call.")
            return None # Important să returnăm None dacă nu știm să încărcăm
            
        loaded_model = self.models.get(model_name)
        if loaded_model is None:
            logger.error(f"Failed to load model '{model_name}' even after specific loading attempt. Check logs for previous errors during load.")
        return loaded_model