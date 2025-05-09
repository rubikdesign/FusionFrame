#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Ai Models Manager used in FusionFrame 2.0
"""


import os
import gc
import torch
import logging
import requests # Asigură-te că e instalat
import zipfile # Asigură-te că e instalat
import time
from tqdm.auto import tqdm # Asigură-te că e instalat
from pathlib import Path
from typing import Optional, Dict, Any, Union, List

# --- Importuri Clase Model ---
try:
    from models.base_model import BaseModel
except ImportError:
    class BaseModel: #type: ignore
        def __init__(self, model_id, device): self.model_id=model_id; self.device=device; self.is_loaded=False
        def load(self): return False
        def unload(self): return True
        def get_info(self): return {}
    logging.getLogger(__name__).error("BaseModel not found. Using mock.")


try:
    from models.flux_model import FluxModel # Dacă vrei să-l păstrezi ca opțiune
except ImportError:
    FluxModel = None #type: ignore
    logging.getLogger(__name__).warning("FluxModel not found.")

try:
    from models.hidream_model import HiDreamModel # Pentru backup
except ImportError:
    HiDreamModel = None #type: ignore
    logging.getLogger(__name__).warning("HiDreamModel not found.")

try:
    from models.sdxl_inpaint_model import SDXLInpaintModel # Noul model principal
except ImportError:
    SDXLInpaintModel = None #type: ignore
    logging.getLogger(__name__).error("SDXLInpaintModel not found! Critical for Juggernaut-XL.")


# --- Importuri Configurații ---
try:
    from config.app_config import AppConfig
    from config.model_config import ModelConfig
except ImportError:
    class AppConfig: DEVICE="cpu"; LOW_VRAM_MODE=True; CACHE_DIR="cache"; MODEL_DIR="models"; DTYPE=torch.float32; ESSENTIAL_MODELS=["main"]; MIN_FREE_VRAM_MB=500; MODEL_LOADING_POLICY="KEEP_LOADED"; ensure_dirs=lambda:None #type: ignore
    class ModelConfig: MAIN_MODEL="mock"; CONTROLNET_CONFIG={}; SAM_CONFIG={}; CLIP_CONFIG={}; OBJECT_DETECTOR_CONFIG={}; IMAGE_CLASSIFIER_CONFIG={}; DEPTH_ESTIMATOR_CONFIG={} #type: ignore
    logging.getLogger(__name__).error("AppConfig/ModelConfig not found. Using mocks.")


# --- Importuri Biblioteci Terțe ---
try:
    from transformers import (
        AutoImageProcessor, AutoModelForImageClassification, AutoModelForDepthEstimation,
        CLIPSegProcessor, CLIPSegForImageSegmentation # Adăugat pentru _load_clipseg_model
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    logging.getLogger(__name__).warning("Transformers library not found or some components missing.")
    AutoImageProcessor, AutoModelForImageClassification, AutoModelForDepthEstimation = None, None, None #type: ignore
    CLIPSegProcessor, CLIPSegForImageSegmentation = None, None #type: ignore
    TRANSFORMERS_AVAILABLE = False

try:
    from segment_anything import sam_model_registry, SamPredictor
    SAM_AVAILABLE = True
except ImportError:
    logging.getLogger(__name__).warning("'segment_anything' library not found. SAM features disabled.")
    sam_model_registry, SamPredictor = None, None #type: ignore
    SAM_AVAILABLE = False

try:
    from ultralytics import YOLO # Pentru _load_yolo_model
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    YOLO = None #type: ignore
    ULTRALYTICS_AVAILABLE = False
    logging.getLogger(__name__).warning("Ultralytics YOLO library not found.")

try:
    import mediapipe as mp # Pentru modelele MediaPipe
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    mp = None #type: ignore
    MEDIAPIPE_AVAILABLE = False
    logging.getLogger(__name__).warning("MediaPipe library not found.")

try:
    from rembg import new_session as new_rembg_session # Alias pentru a evita conflicte
    REMBG_AVAILABLE = True
except ImportError:
    new_rembg_session = None #type: ignore
    REMBG_AVAILABLE = False
    logging.getLogger(__name__).warning("Rembg library not found.")


logger = logging.getLogger(__name__)

class ModelManager:
    _instance: Optional['ModelManager'] = None
    _initialized: bool = False

    CPU_FRIENDLY_MODELS = [
        'image_classifier', 'depth_estimator', 'yolo',
        'mediapipe', 'face_detector', 'rembg', 'clipseg' # Adăugat clipseg
    ]
    ESSENTIAL_GPU_MODELS = ['main', 'sam_predictor'] # SAM e mai bun pe GPU

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
        if hasattr(self.config, 'ensure_dirs') and callable(self.config.ensure_dirs):
            self.config.ensure_dirs()
        self._initialized = True
        self.memory_stats: Dict[str, Any] = {"last_check": 0, "loaded_models": [], "memory_usage": {}}
        logger.info("ModelManager initialized")

        self.memory_stats: Dict[str, Any] = {
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

            allocated = torch.cuda.memory_allocated() / (1024**3)
            reserved = torch.cuda.memory_reserved() / (1024**3)
            logger.info(f"GPU memory after cleanup: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")

    def _log_memory_stats(self):
        """Înregistrează statisticile memoriei."""
        if not torch.cuda.is_available():
            return

        current_time = time.time()
        if current_time - self.memory_stats.get("last_check", 0) < 10:
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
            # Considerăm critică utilizarea peste 90% sau dacă memoria liberă este sub un prag
            # (de exemplu, MIN_FREE_VRAM_MB din AppConfig)
            min_free_mb = getattr(self.config, "MIN_FREE_VRAM_MB", 500) # Default 500MB
            free_mb = (total - reserved) / (1024**2)

            percent_used = (reserved / total) * 100 if total > 0 else 0

            if percent_used > 90 or free_mb < min_free_mb:
                logger.warning(f"Memory critical: {percent_used:.1f}% used, {free_mb:.0f}MB free.")
                return True
            return False
        except Exception as e:
            logger.error(f"Error checking memory status: {e}")
            return False # Presupunem că nu e critică în caz de eroare

    def _should_use_cpu_for_model(self, model_name: str) -> bool:
        """Determină dacă un model ar trebui încărcat pe CPU pentru a economisi VRAM."""
        low_vram_mode = getattr(self.config, "LOW_VRAM_MODE", False)

        if model_name in self.CPU_FRIENDLY_MODELS and low_vram_mode:
            logger.info(f"LOW_VRAM_MODE: Model '{model_name}' will use CPU.")
            return True

        force_cpu_models_list = getattr(self.config, "FORCE_CPU_MODELS", [])
        if model_name in force_cpu_models_list:
            logger.info(f"FORCE_CPU_MODELS: Model '{model_name}' forced to CPU.")
            return True

        if self._is_memory_critical() and model_name not in self.ESSENTIAL_GPU_MODELS:
            logger.warning(f"Memory is critical. Forcing model '{model_name}' to CPU.")
            return True

        return False

    def _get_device_for_model(self, model_name: str) -> str:
        """Determină dispozitivul potrivit pentru un model."""
        if self._should_use_cpu_for_model(model_name):
            return "cpu"
        return str(self.config.DEVICE)


    # --- Metode Utilitare ---
    def _get_filename_from_url(self, url: str) -> str:
        try:
            # Verifică dacă requests.utils este disponibil
            if hasattr(requests, 'utils') and hasattr(requests.utils, 'urlparse'):
                path = requests.utils.urlparse(url).path #type: ignore
            else:
                # Fallback simplu dacă requests.utils nu e disponibil (puțin probabil)
                from urllib.parse import urlparse
                path = urlparse(url).path
            name = os.path.basename(path)
            return name if name else (path.split('/')[-1] if '/' in path else "downloaded_file")
        except Exception as e: # Prinde excepții mai specifice dacă e posibil
            logger.warning(f"URL parse failed for {url}: {e}")
            return "downloaded_file"


    def download_file(self, url: str, dest: str, desc: Optional[str]=None) -> Optional[str]:
        name = desc or os.path.basename(dest)
        logger.info(f"Downloading {name} to {dest}")
        total_size = 0
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
        except requests.exceptions.RequestException as e_req:
            logger.error(f"Download request error for {name}: {e_req}", exc_info=True)
        except IOError as e_io:
            logger.error(f"File I/O error for {name} at {dest}: {e_io}", exc_info=True)
        except Exception as e:
            logger.error(f"Generic download error for {name}: {e}", exc_info=True)

        if os.path.exists(dest):
            try:
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
        if not os.path.exists(self.config.CACHE_DIR): #type: ignore
            os.makedirs(self.config.CACHE_DIR, exist_ok=True) #type: ignore

        zip_name = self._get_filename_from_url(url)
        zip_name += ".zip" if not zip_name.lower().endswith(".zip") else ""
        tmp_path = os.path.join(self.config.CACHE_DIR, zip_name) #type: ignore
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
                except OSError as e_rm_zip:
                    logger.warning(f"Could not remove bad zip: {e_rm_zip}")
        except Exception as e:
            logger.error(f"Zip DL/Extract error: {e}", exc_info=True)
        return None

    def ensure_model(self, model_key: str) -> str:
        assets_dir = os.path.join(self.config.MODEL_DIR, model_key) #type: ignore
        os.makedirs(assets_dir, exist_ok=True)
        model_urls = getattr(self.config, 'MODEL_URLS', getattr(self.model_config, 'MODEL_URLS', {}))
        url = model_urls.get(model_key)
        if not url:
            logger.warning(f"No URL for '{model_key}'. Assets directory: {assets_dir}")
            return assets_dir

        if model_key == "sam":
            sam_cfg = getattr(self.model_config, "SAM_CONFIG", {})
            chk_name = sam_cfg.get("checkpoint", "sam_vit_h_4b8939.pth") # Default checkpoint
            chk_path = os.path.join(assets_dir, chk_name)
            if not (os.path.exists(chk_path) and os.path.getsize(chk_path) > 0):
                logger.info(f"SAM .pth missing, DL {url} to {chk_path}...")
                if not self.download_file(url, chk_path, desc=chk_name): # Folosește desc în loc de description
                    logger.error(f"Failed DL SAM .pth from {url}.")
            else:
                logger.debug(f"SAM .pth found: {chk_path}")
        # Adaugă logica similară pentru alte modele dacă necesită descărcare manuală de fișiere specifice
        # De exemplu, pentru FLUX dacă safetensors nu e direct in repo-ul pipeline-ului:
        elif model_key == ModelConfig.MAIN_MODEL and model_key == "FLUX.1-dev": # Verifică dacă modelul principal este FLUX
            flux_cfg = getattr(self.model_config, "FLUX_CONFIG", {})
            model_path = flux_cfg.get("pretrained_model_name_or_path")
            # Dacă model_path este un URL direct către un fișier .safetensors, descarcă-l
            if url.endswith(".safetensors") and not os.path.exists(os.path.join(assets_dir, os.path.basename(url))):
                 logger.info(f"FLUX .safetensors missing, DL {url}...")
                 dest_file = os.path.join(assets_dir, os.path.basename(url))
                 if not self.download_file(url, dest_file, desc=os.path.basename(url)):
                     logger.error(f"Failed DL FLUX .safetensors from {url}.")
                 else: # Actualizează calea în config dacă descărcăm local
                     flux_cfg["pretrained_model_name_or_path"] = dest_file


        return assets_dir

    # --- Model loading methods ---
    def load_main_model(self) -> None:
        """
        Loads the main generation model based on ModelConfig.MAIN_MODEL.
        Supports SDXLInpaintModel (e.g., Juggernaut-XL), FluxModel, and HiDreamModel.
        """
        main_model_id_from_config = getattr(ModelConfig, 'MAIN_MODEL', 'NOT_CONFIGURED')
        logger.debug(f"Request main model load. Configured MAIN_MODEL: '{main_model_id_from_config}'")

        ModelClassToLoad: Optional[type[BaseModel]] = None
        expected_config_attr: Optional[str] = None

        # Determine which model class and config to use
        if main_model_id_from_config == "RunDiffusion/Juggernaut-XL-v9" and SDXLInpaintModel is not None:
            ModelClassToLoad = SDXLInpaintModel
            expected_config_attr = "SDXL_INPAINT_CONFIG"
            logger.info(f"Preparing to load SDXLInpaintModel for '{main_model_id_from_config}'.")
        elif main_model_id_from_config == "FLUX.1-dev" and FluxModel is not None:
            ModelClassToLoad = FluxModel
            expected_config_attr = "FLUX_CONFIG"
            logger.info(f"Preparing to load FluxModel for '{main_model_id_from_config}'.")
        elif main_model_id_from_config == "HiDream-I1-Fast" and HiDreamModel is not None: # Sau alt ID HiDream
            ModelClassToLoad = HiDreamModel
            expected_config_attr = "HIDREAM_CONFIG" # Asigură-te că HIDREAM_CONFIG există în ModelConfig
            logger.info(f"Preparing to load HiDreamModel for '{main_model_id_from_config}'.")
        # Adaugă aici alte `elif` pentru alte modele principale pe care vrei să le suporti

        if ModelClassToLoad is None:
            logger.error(f"Main model_id '{main_model_id_from_config}' is configured, but a corresponding "
                         f"ModelClass is not available or not imported correctly in ModelManager. "
                         f"(SDXLInpaintModel: {SDXLInpaintModel is not None}, FluxModel: {FluxModel is not None}, HiDreamModel: {HiDreamModel is not None})")
            return

        if expected_config_attr and not hasattr(ModelConfig, expected_config_attr):
            logger.error(f"Configuration attribute '{expected_config_attr}' expected for model '{main_model_id_from_config}' "
                         f"is missing in ModelConfig. Cannot load main model.")
            return


        # Check if the correct type of model is already loaded
        if 'main' in self.models and isinstance(self.models.get('main'), BaseModel) and self.models['main'].is_loaded:
            current_model_instance = self.models['main']
            if isinstance(current_model_instance, ModelClassToLoad) and current_model_instance.model_id == main_model_id_from_config:
                logger.info(f"{ModelClassToLoad.__name__} for '{main_model_id_from_config}' is already loaded.")
                return
            else:
                logger.info(f"Current main model (type: {type(current_model_instance).__name__}, id: {current_model_instance.model_id}) "
                              f"does not match configured MAIN_MODEL (type: {ModelClassToLoad.__name__}, id: {main_model_id_from_config}). Unloading to reload.")
                self.unload_model('main') # Unload a instanței vechi/incorecte

        self._clear_gpu_memory()

        try:
            logger.info(f"Instantiating {ModelClassToLoad.__name__} for main model (id: '{main_model_id_from_config}')...")
            # Clasa modelului (ex: SDXLInpaintModel) va prelua model_id-ul din ModelConfig.MAIN_MODEL
            # și configurația sa specifică (ex: SDXL_INPAINT_CONFIG) din interiorul constructorului său.
            inst = ModelClassToLoad(model_id=main_model_id_from_config) # Pasează explicit model_id-ul

            if inst.load():
                self.models['main'] = inst
                logger.info(f"Main model ({ModelClassToLoad.__name__} for '{main_model_id_from_config}') loaded successfully.")
                self._log_memory_stats()
            else:
                logger.error(f"{ModelClassToLoad.__name__} load() failed for main model '{main_model_id_from_config}'.")
                if 'main' in self.models: # Asigură că nu rămâne o instanță eșuată
                    del self.models['main']
        except Exception as e:
            logger.error(f"Error during main model ({ModelClassToLoad.__name__} for '{main_model_id_from_config}') load: {e}", exc_info=True)
            self.unload_model('main') # Asigură curățarea în caz de eroare la instanțiere/load
    

    def load_sam_model(self) -> None:
        logger.debug("Request SAM predictor load...")
        if 'sam_predictor' in self.models:
            return
        if not SAM_AVAILABLE or sam_model_registry is None or SamPredictor is None:
            logger.error("SAM library or components (sam_model_registry, SamPredictor) unavailable.")
            return

        self._clear_gpu_memory()

        try:
            assets_dir = self.ensure_model("sam")
            sam_cfg = getattr(self.model_config, "SAM_CONFIG", {})
            chk_name = sam_cfg.get("checkpoint", "sam_vit_h_4b8939.pth")
            chk_path = os.path.join(assets_dir, chk_name)
            m_type = sam_cfg.get("model_type", "vit_h") # Default la "vit_h"

            if not (os.path.exists(chk_path) and os.path.getsize(chk_path) > 0):
                logger.error(f"SAM .pth not found or is empty: {chk_path}. Attempt ensure_model again or check URL.")
                # Opțional, încearcă să descarci din nou dacă ensure_model nu a reușit
                # self.ensure_model("sam") # Ar putea fi redundant dacă ensure_model e robust
                return

            logger.info(f"Loading SAM model ({m_type}) for predictor from: {chk_path}")

            device_to_use = self._get_device_for_model('sam_predictor') # SAM e pe GPU de obicei
            logger.info(f"SAM will be loaded on device: {device_to_use}")

            sam_model = sam_model_registry[m_type](checkpoint=chk_path).to(device_to_use).eval() #type: ignore
            predictor = SamPredictor(sam_model) #type: ignore
            self.models['sam_predictor'] = predictor
            logger.info(f"SAM model & Predictor loaded on {device_to_use}.")
            self._log_memory_stats()
        except KeyError as e:
            logger.error(f"SAM Config key error or invalid model_type '{m_type}': {e}", exc_info=True)
        except Exception as e:
            logger.error(f"Error loading SAM/predictor: {e}", exc_info=True)


    def load_clipseg_model(self) -> None:
        logger.debug("Request CLIPSeg...")
        if 'clipseg' in self.models:
            return
        if not TRANSFORMERS_AVAILABLE:
            logger.error("Transformers library not available for CLIPSeg.")
            return

        self._clear_gpu_memory()

        try:
            from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation # Import local
            clip_cfg = getattr(self.model_config, "CLIP_CONFIG", {})
            mid = clip_cfg.get("model_id", "CIDAS/clipseg-rd64-refined") # Default model ID
            logger.info(f"Loading CLIPSeg: {mid}")

            device_to_use = self._get_device_for_model('clipseg') # CLIPSeg e de obicei pe GPU

            p = CLIPSegProcessor.from_pretrained(mid, cache_dir=self.config.CACHE_DIR)
            m = CLIPSegForImageSegmentation.from_pretrained(
                mid,
                torch_dtype=self.config.DTYPE if device_to_use != "cpu" else torch.float32, #type: ignore
                cache_dir=self.config.CACHE_DIR
            ).to(device_to_use).eval()

            self.models['clipseg'] = {'processor': p, 'model': m, 'device': device_to_use}
            logger.info(f"CLIPSeg loaded on {device_to_use}.")
            self._log_memory_stats()
        except Exception as e:
            logger.error(f"CLIPSeg load error: {e}", exc_info=True)


    def _load_yolo_model(self) -> Optional[Any]:
        logger.debug("Request YOLO...")
        if 'yolo' in self.models:
            return self.models['yolo']

        device = self._get_device_for_model('yolo')
        self._clear_gpu_memory() # Curăță înainte de a încărca YOLO

        try:
            from ultralytics import YOLO # Import local
            yolo_cfg = getattr(self.model_config, "OBJECT_DETECTOR_CONFIG", {})

            if device == "cpu" or getattr(self.config, "LOW_VRAM_MODE", False):
                name = yolo_cfg.get("lightweight_model_id", "yolov8n-seg.pt")
                logger.info(f"Using lightweight YOLO ({name}) for low VRAM mode or CPU.")
            else:
                name = yolo_cfg.get("model_id", "yolov8x-seg.pt")
                logger.info(f"Using standard YOLO ({name}).")

            # YOLO descarcă automat modelul dacă nu îl găsește local la 'name'
            # Nu este necesar să construim calea completă decât dacă vrem să forțăm un fișier local
            # target = str(Path(self.config.MODEL_DIR) / "YOLO" / name) if Path(self.config.MODEL_DIR / "YOLO" / name).is_file() else name
            target = name # YOLO se ocupă de descărcare/cale

            logger.info(f"Loading YOLO: {target} (intended device: {device})")
            # YOLO își gestionează propriul dispozitiv intern, dar putem specifica la inferență
            m = YOLO(target)
            # m.to(device) # YOLO face asta automat la prima inferență sau poate fi setat global

            self.models['yolo'] = {'model': m, 'device': device} # Stocăm dispozitivul intenționat
            logger.info(f"YOLO '{target}' loaded (will attempt to use {device} on inference).")
            self._log_memory_stats()
            return self.models['yolo']
        except ImportError:
            logger.error("Ultralytics YOLO library not found.")
            return None
        except Exception as e:
            logger.error(f"YOLO load error: {e}", exc_info=True)
            return None


    def _load_mediapipe_model(self) -> Optional[Any]:
        logger.debug("Request MP Selfie Segmentation...")
        if 'mediapipe' in self.models: # Nume mai specific, ex: 'mediapipe_selfie_segmentation'
            return self.models['mediapipe']

        # MediaPipe rulează pe CPU, nu necesită curățarea memoriei GPU în mod specific pentru el
        try:
            import mediapipe as mp # Import local
            # Asigură-te că solutions și selfie_segmentation sunt disponibile
            if not (hasattr(mp, 'solutions') and hasattr(mp.solutions, 'selfie_segmentation')):
                logger.error("MediaPipe solutions or selfie_segmentation not found.")
                return None

            sel_cfg_key = "MEDIAPIPE_SELFIE_MODEL_SELECTION"
            model_selection_val = getattr(self.config, sel_cfg_key, #type: ignore
                                       getattr(self.model_config, sel_cfg_key, 1)) # Default 1

            seg = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=model_selection_val)
            self.models['mediapipe'] = {'model': seg, 'device': 'cpu'} # MediaPipe rulează pe CPU
            logger.info(f"MP Selfie Segmentation (selection={model_selection_val}) loaded (CPU only).")
            return self.models['mediapipe']
        except ImportError:
            logger.error("MediaPipe library not found.")
            return None
        except Exception as e:
            logger.error(f"MP Selfie Segmentation load error: {e}", exc_info=True)
            return None


    def _load_face_detector(self) -> Optional[Any]:
        logger.debug("Request MP Face Detection...")
        if 'face_detector' in self.models:
            return self.models['face_detector']

        try:
            import mediapipe as mp # Import local
            if not (hasattr(mp, 'solutions') and hasattr(mp.solutions, 'face_detection')):
                logger.error("MediaPipe solutions or face_detection not found.")
                return None

            sel_cfg_key = "MEDIAPIPE_FACE_MODEL_SELECTION"
            conf_cfg_key = "MEDIAPIPE_FACE_MIN_CONFIDENCE"

            model_selection_val = getattr(self.config, sel_cfg_key, #type: ignore
                                       getattr(self.model_config, sel_cfg_key, 0)) # Default 0
            min_confidence_val = getattr(self.config, conf_cfg_key, #type: ignore
                                      getattr(self.model_config, conf_cfg_key, 0.5)) # Default 0.5


            det = mp.solutions.face_detection.FaceDetection(
                model_selection=model_selection_val,
                min_detection_confidence=min_confidence_val
            )
            self.models['face_detector'] = {'model': det, 'device': 'cpu'} # MediaPipe rulează pe CPU
            logger.info(f"MP Face Detection (selection={model_selection_val}, confidence={min_confidence_val}) loaded (CPU only).")
            return self.models['face_detector']
        except ImportError:
            logger.error("MediaPipe library not found.")
            return None
        except Exception as e:
            logger.error(f"MP Face Detection load error: {e}", exc_info=True)
            return None

    def _load_rembg_model(self) -> Optional[Any]:
        logger.debug("Request Rembg session...")
        if 'rembg' in self.models:
            return self.models['rembg']

        device = self._get_device_for_model('rembg') # Rembg poate folosi GPU

        try:
            from rembg import new_session # Import local

            rembg_model_name_key = "REMBG_MODEL_NAME"
            rembg_light_model_name_key = "REMBG_LIGHTWEIGHT_MODEL_NAME" # Din model_config

            name = getattr(self.model_config, rembg_model_name_key, "u2net") # Default

            if device == "cpu" or getattr(self.config, "LOW_VRAM_MODE", False):
                name = getattr(self.model_config, rembg_light_model_name_key, "u2netp") # Model mai mic
                logger.info(f"Using smaller Rembg model ({name}) for low VRAM mode or CPU.")

            # new_session poate accepta providers dacă onnxruntime-gpu e instalat
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if device == 'cuda' and torch.cuda.is_available() else ['CPUExecutionProvider']
            logger.info(f"Loading Rembg session '{name}' with providers: {providers}")
            sess = new_session(model_name=name, providers=providers)

            self.models['rembg'] = {'model': sess, 'device': device} # Stocăm dispozitivul țintă
            logger.info(f"Rembg session '{name}' created (intended device: {device}).")
            return self.models['rembg']
        except ImportError:
            logger.error("Rembg library not found.")
            return None
        except Exception as e:
            logger.error(f"Rembg load error: {e}", exc_info=True)
            return None


    def _load_image_classifier(self) -> Optional[Dict[str, Any]]:
        logger.debug("Request ImgClassifier...")
        if 'image_classifier' in self.models:
            return self.models['image_classifier']
        if not TRANSFORMERS_AVAILABLE or AutoImageProcessor is None or AutoModelForImageClassification is None:
            logger.error("Transformers library or components not available for Image Classifier.")
            return None

        device = self._get_device_for_model('image_classifier')
        self._clear_gpu_memory()

        try:
            img_cls_cfg = getattr(self.model_config, "IMAGE_CLASSIFIER_CONFIG", {})
            mid = img_cls_cfg.get("model_id", "google/vit-base-patch16-224") # Default

            if device == "cpu" or getattr(self.config, "LOW_VRAM_MODE", False):
                mid = img_cls_cfg.get("lightweight_model_id", "google/vit-base-patch16-224-in21k")
                logger.info(f"Using lightweight Image Classifier ({mid}) for low VRAM or CPU.")


            logger.info(f"Loading ImgClass: {mid} on {device}")
            p = AutoImageProcessor.from_pretrained(mid, cache_dir=self.config.CACHE_DIR)
            m = AutoModelForImageClassification.from_pretrained(
                mid,
                torch_dtype=self.config.DTYPE if device != "cpu" else torch.float32, #type: ignore
                cache_dir=self.config.CACHE_DIR
            ).to(device).eval()

            bundle = {'processor': p, 'model': m, 'device': device}
            self.models['image_classifier'] = bundle
            logger.info(f"ImgClass '{mid}' loaded on {device}.")
            self._log_memory_stats()
            return bundle
        except Exception as e:
            logger.error(f"ImgClass load error: {e}", exc_info=True)
            return None


    def _load_depth_estimator(self) -> Optional[Dict[str, Any]]:
        logger.debug("Request DepthEst...")
        if 'depth_estimator' in self.models:
            return self.models['depth_estimator']
        if not TRANSFORMERS_AVAILABLE or AutoImageProcessor is None or AutoModelForDepthEstimation is None:
            logger.error("Transformers library or components not available for Depth Estimator.")
            return None

        device = self._get_device_for_model('depth_estimator')
        self._clear_gpu_memory()

        try:
            depth_cfg = getattr(self.model_config, "DEPTH_ESTIMATOR_CONFIG", {})
            mid = depth_cfg.get("model_id", "Intel/dpt-hybrid-midas") # Default

            if device == "cpu" or getattr(self.config, "LOW_VRAM_MODE", False) or getattr(self.config, "DISABLE_DEPTH_ESTIMATION", False): #type: ignore
                if getattr(self.config, "DISABLE_DEPTH_ESTIMATION", False): #type: ignore
                    logger.info("Depth estimation is disabled in AppConfig. Skipping load.")
                    return None
                mid = depth_cfg.get("lightweight_model_id", "Intel/dpt-large") # Model mai mic
                logger.info(f"Using lightweight Depth Estimator ({mid}) for low VRAM or CPU.")


            logger.info(f"Loading DepthEst: {mid} on {device}")
            p = AutoImageProcessor.from_pretrained(mid, cache_dir=self.config.CACHE_DIR)
            m = AutoModelForDepthEstimation.from_pretrained(
                mid,
                torch_dtype=self.config.DTYPE if device != "cpu" else torch.float32, #type: ignore
                cache_dir=self.config.CACHE_DIR
            ).to(device).eval()

            bundle = {'processor': p, 'model': m, 'device': device}
            self.models['depth_estimator'] = bundle
            logger.info(f"DepthEst '{mid}' loaded on {device}.")
            self._log_memory_stats()
            return bundle
        except Exception as e:
            logger.error(f"DepthEst load error: {e}", exc_info=True)
            return None


    def load_all_models(self) -> None:
        """Încarcă modelele esențiale definite în config."""
        essential_models_to_load = getattr(self.config, "ESSENTIAL_MODELS", ["main"]) # Default la ['main']
        logger.info(f"Loading/Checking essential models: {essential_models_to_load}...")
        for model_name in essential_models_to_load:
            self.get_model(model_name)
        logger.info("Essential models check/load initiated.")


    def unload_model(self, model_name: str) -> None:
        model_entry = self.models.pop(model_name, None)
        if model_entry:
            logger.info(f"Unloading model: '{model_name}'...")
            try:
                # Cazul pentru modelele principale care moștenesc BaseModel (FluxModel, HiDreamModel)
                if isinstance(model_entry, BaseModel) and hasattr(model_entry, 'unload'):
                    model_entry.unload() #type: ignore
                # Cazul pentru modelele stocate ca dicționare (CLIPSeg, ImgClassifier, etc.)
                elif isinstance(model_entry, dict) and 'model' in model_entry:
                    actual_model_component = model_entry['model']
                    if hasattr(actual_model_component, 'to'): # Mută pe CPU înainte de del
                        try:
                            actual_model_component.to('cpu')
                        except Exception as e_to_cpu:
                            logger.warning(f"Could not move model component of '{model_name}' to CPU before unload: {e_to_cpu}")
                    del actual_model_component
                    if 'processor' in model_entry: # Pentru modelele cu procesor
                        del model_entry['processor']
                # Cazul pentru SAM (care e direct predictorul) sau MediaPipe (care are .close())
                elif model_name == 'sam_predictor' and hasattr(model_entry, 'reset_image'): # SamPredictor
                     # SamPredictor nu are o metodă 'unload' evidentă, modelul e în spate
                     # Încercăm să ștergem modelul SAM din GPU dacă e posibil (mai complex)
                     # Pentru simplitate, doar ștergem referința.
                    logger.debug("Unloading SAM predictor (reference only).")
                elif model_name in ('mediapipe', 'face_detector') and hasattr(model_entry.get('model') if isinstance(model_entry, dict) else model_entry, 'close'):
                    if isinstance(model_entry, dict):
                        model_entry.get('model').close()
                    else:
                        model_entry.close() #type: ignore
                else: # Cazul general pentru alte tipuri de obiecte model
                    logger.debug(f"No specific unload for {model_name}, deleting reference.")

                del model_entry # Șterge intrarea din self.models
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                logger.info(f"Model '{model_name}' unloaded.")
                self._log_memory_stats()
            except Exception as e:
                logger.error(f"Error during unload of '{model_name}': {e}", exc_info=True)
                # Chiar dacă e eroare, modelul a fost scos din self.models, deci e "unloaded" din perspectiva managerului
        else:
            logger.debug(f"Unload: Model '{model_name}' not found or already unloaded.")


    def get_model(self, model_name: str) -> Any:
        """Obține un model, încărcându-l leneș."""
        if model_name in self.models and self.models[model_name] is not None:
            # Verificare specială pentru modelul principal (care moștenește BaseModel)
            if model_name == 'main':
                main_model_instance = self.models['main']
                if isinstance(main_model_instance, BaseModel):
                    if not main_model_instance.is_loaded: #type: ignore
                        logger.warning(f"Main model '{main_model_instance.model_id}' cached but not loaded. Reloading.") #type: ignore
                        self.load_main_model() # Va reîncărca modelul principal corect
                    # Verifică dacă tipul modelului principal încărcat corespunde cu ModelConfig.MAIN_MODEL
                    expected_model_class = None
                    if ModelConfig.MAIN_MODEL == "FLUX.1-dev": expected_model_class = FluxModel
                    elif ModelConfig.MAIN_MODEL == "HiDream-I1-Fast": expected_model_class = HiDreamModel #type: ignore

                    if expected_model_class and not isinstance(main_model_instance, expected_model_class):
                        logger.warning(f"Main model is type {type(main_model_instance)} but config expects {ModelConfig.MAIN_MODEL} ({expected_model_class}). Reloading.")
                        self.unload_model('main')
                        self.load_main_model()

                elif not isinstance(main_model_instance, BaseModel): # Ceva e greșit
                    logger.error(f"Invalid cache for 'main': type is {type(main_model_instance)}, expected BaseModel. Reloading.")
                    self.unload_model('main')
                    self.load_main_model()
                return self.models.get(model_name) # Returnează modelul (re)încărcat

            # Pentru alte modele (majoritatea stocate ca dicționare sau obiecte directe)
            return self.models[model_name]


        logger.info(f"Model '{model_name}' not loaded. Lazy loading...")

        # Verifică politica de încărcare/descărcare înainte de a încărca un model nou
        self._apply_model_loading_policy(model_name_to_load=model_name)


        loader_map = {
            'main': self.load_main_model,
            'sam_predictor': self.load_sam_model,
            'clipseg': self.load_clipseg_model,
            'yolo': self._load_yolo_model,
            'mediapipe': self._load_mediapipe_model, # Specific pt selfie segmentation
            'face_detector': self._load_face_detector,
            'rembg': self._load_rembg_model,
            'image_classifier': self._load_image_classifier,
            'depth_estimator': self._load_depth_estimator
            # Adaugă alte mapări aici dacă e necesar
        }

        # self._clear_gpu_memory() # Mutat în interiorul funcțiilor de încărcare unde e relevant (GPU-heavy models)

        loader_func = loader_map.get(model_name)
        if loader_func:
            loader_func() # Apelăm funcția de încărcare specifică
        else:
            logger.warning(f"No specific loader function defined for model key: '{model_name}'. Cannot load.")
            return None

        final_model = self.models.get(model_name)
        if final_model is None:
            logger.error(f"Failed to load model '{model_name}' using its loader function.")
        return final_model


    def get_memory_status(self) -> Dict[str, Any]:
        memory_info: Dict[str, Any] = {
            "cuda_available": torch.cuda.is_available(),
            "loaded_models_count": len(self.models),
            "loaded_models_list": list(self.models.keys()),
            "system_ram_available_gb": 0,
            "system_ram_percent": 0,
            "cuda_info": {}
        }

        try:
            import psutil # Import local
            mem = psutil.virtual_memory()
            memory_info["system_ram_available_gb"] = round(mem.available / (1024**3), 2)
            memory_info["system_ram_total_gb"] = round(mem.total / (1024**3), 2)
            memory_info["system_ram_percent"] = mem.percent
        except ImportError:
            logger.warning("psutil not installed, cannot get system RAM info.")
        except Exception as e:
            logger.warning(f"Could not get system RAM info: {e}")


        if torch.cuda.is_available():
            try:
                # Asigură-te că ID-ul dispozitivului este valid
                device_id = torch.cuda.current_device() if torch.cuda.device_count() > 0 else 0
                if device_id < torch.cuda.device_count():
                    device_name = torch.cuda.get_device_name(device_id)
                    total_memory = torch.cuda.get_device_properties(device_id).total_memory
                    allocated_memory = torch.cuda.memory_allocated(device_id)
                    reserved_memory = torch.cuda.memory_reserved(device_id) # Cât a rezervat PyTorch
                    # Memoria liberă pe GPU din perspectiva PyTorch este total - rezervat
                    free_pytorch_reserved = total_memory - reserved_memory


                    memory_info["cuda_info"] = {
                        "device_name": device_name,
                        "current_device_id": device_id,
                        "total_memory_gb": round(total_memory / (1024**3), 2),
                        "allocated_memory_gb": round(allocated_memory / (1024**3), 2), # Alocat efectiv de tensori
                        "reserved_memory_gb": round(reserved_memory / (1024**3), 2), # Total rezervat de PyTorch
                        "free_within_reserved_gb": round((reserved_memory - allocated_memory) / (1024**3), 2), # Liber în cache-ul PyTorch
                        "free_total_pytorch_perspective_gb": round(free_pytorch_reserved / (1024**3), 2), # Total - Rezervat
                        "percent_allocated_of_total": round((allocated_memory / total_memory) * 100 if total_memory > 0 else 0, 1),
                        "percent_reserved_of_total": round((reserved_memory / total_memory) * 100 if total_memory > 0 else 0, 1),
                        "is_memory_critical": self._is_memory_critical() # Reutilizează logica existentă
                    }
                else:
                    logger.warning(f"CUDA device ID {device_id} is out of range. Device count: {torch.cuda.device_count()}")
                    memory_info["cuda_info"]["error"] = "Invalid CUDA device ID"

            except Exception as e:
                logger.error(f"Error getting CUDA memory status: {e}", exc_info=True)
                memory_info["cuda_info"]["error"] = str(e)

        return memory_info


    def emergency_memory_recovery(self) -> bool:
        """
        Perform emergency memory recovery when CUDA out-of-memory occurs.
        Returns: True if recovery actions were taken, False otherwise
        """
        if not torch.cuda.is_available():
            return False

        logger.warning("EMERGENCY MEMORY RECOVERY triggered!")
        recovery_actions_taken = False

        # 1. Încearcă să muți modelele non-esențiale, prietenoase cu CPU, pe CPU
        models_to_move_or_unload = [
            m for m in self.CPU_FRIENDLY_MODELS if m in self.models and m not in self.ESSENTIAL_GPU_MODELS
        ]
        logger.info(f"Attempting to move to CPU: {models_to_move_or_unload}")
        for model_name in models_to_move_or_unload:
            if self._move_model_to_cpu(model_name):
                recovery_actions_taken = True

        self._clear_gpu_memory() # Curăță cache după mutări

        # 2. Dacă memoria e încă critică, descarcă modelele non-esențiale
        # Lista modelelor esențiale de păstrat, citită din config
        essential_to_keep = getattr(self.config, "ESSENTIAL_MODELS", ["main"])
        models_currently_loaded = list(self.models.keys()) # Copie pentru a itera în siguranță

        # Definește o prioritate de descărcare (cele mai puțin importante primele)
        unload_priority = [
            'depth_estimator', 'image_classifier', 'yolo', 'rembg',
            'face_detector', 'mediapipe', 'clipseg', 'sam_predictor',
            # Adaugă alte modele aici în ordinea inversă a importanței
        ]
        # Filtrează pentru a păstra doar modelele încărcate și non-esențiale
        models_to_consider_unloading = [
            m for m in unload_priority if m in models_currently_loaded and m not in essential_to_keep
        ]

        logger.info(f"Models to consider for unloading (if memory still critical): {models_to_consider_unloading}")

        for model_name in models_to_consider_unloading:
            if self._is_memory_critical() or getattr(self.config, "MODEL_LOADING_POLICY", "KEEP_LOADED") == "AGGRESSIVE_UNLOAD": #type: ignore
                logger.warning(f"Emergency unloading model: {model_name} due to critical memory or aggressive policy.")
                self.unload_model(model_name)
                recovery_actions_taken = True
                self._clear_gpu_memory() # Curăță după fiecare descărcare majoră
            else: # Dacă memoria nu mai e critică, oprește descărcările de urgență
                break


        # 3. Dacă modelul principal este încărcat și are o metodă de curățare de urgență, apeleaz-o
        if 'main' in self.models and isinstance(self.models['main'], BaseModel) and hasattr(self.models['main'], 'emergency_cleanup'):
            logger.warning("Performing emergency cleanup on main model.")
            try:
                self.models['main'].emergency_cleanup() #type: ignore
                recovery_actions_taken = True
            except Exception as e_main_cleanup:
                logger.error(f"Error during main model emergency_cleanup: {e_main_cleanup}")

        # 4. Forțează colectarea gunoiului și golirea cache-ului din nou
        self._clear_gpu_memory()

        if recovery_actions_taken:
            logger.info("Emergency memory recovery actions performed.")
        else:
            logger.info("No specific emergency recovery actions were performed (or needed beyond standard cleanup).")

        return recovery_actions_taken


    def _move_model_to_cpu(self, model_name: str) -> bool:
        """
        Move a model from GPU to CPU to free VRAM.
        Returns: True if successful, False otherwise
        """
        if model_name not in self.models or self.models[model_name] is None:
            logger.debug(f"Cannot move '{model_name}' to CPU: not loaded.")
            return False

        model_entry = self.models[model_name]
        current_device_of_model = "unknown"

        try:
            # Caz pentru modelele care moștenesc BaseModel (ex: FluxModel, HiDreamModel)
            if isinstance(model_entry, BaseModel):
                # Verifică dacă modelul principal are componente pe GPU
                if hasattr(model_entry, 'pipeline') and model_entry.pipeline is not None: #type: ignore
                    # Încearcă să muți componentele principale ale pipeline-ului (unet, text_encoders dacă nu sunt deja pe CPU)
                    # Aceasta este o logică mai complexă și depinde de structura modelului.
                    # FluxModel/HiDreamModel ar trebui să aibă propria logică de CPU offload parțial.
                    # Aici doar semnalăm că modelul 'main' e considerat mutat dacă e configurat pt CPU în LOW_VRAM.
                    if hasattr(model_entry, 'device') and model_entry.device == 'cpu': #type: ignore
                        logger.info(f"Main model '{model_name}' already configured for CPU or components are offloaded.")
                        return True # Considerat mutat dacă device-ul său e CPU
                    # Dacă nu, o mutare completă a unui pipeline mare e riscantă și grea.
                    # Mai bine se bazează pe logica internă de offload a modelului.
                    logger.warning(f"Moving full pipeline of main model '{model_name}' to CPU is not directly supported here. Relies on model's internal offload.")
                    # Forțăm o curățare de urgență dacă există, poate ajută
                    if hasattr(model_entry, 'emergency_cleanup'):
                        model_entry.emergency_cleanup() #type: ignore
                    return False # Nu confirmăm mutarea completă aici

            # Caz pentru modelele stocate ca dicționare (CLIPSeg, ImgClassifier, DepthEstimator etc.)
            elif isinstance(model_entry, dict) and "model" in model_entry and "device" in model_entry:
                actual_model_component = model_entry["model"]
                current_device_of_model = model_entry["device"]
                if current_device_of_model == "cpu":
                    logger.info(f"Model '{model_name}' component already on CPU.")
                    return True # Deja pe CPU

                if hasattr(actual_model_component, "to") and callable(actual_model_component.to):
                    logger.info(f"Moving model component of '{model_name}' from {current_device_of_model} to CPU...")
                    actual_model_component.to("cpu")
                    model_entry["device"] = "cpu" # Actualizează starea dispozitivului
                    # Dacă există și procesor, și el ar trebui mutat dacă e un model nn.Module
                    if "processor" in model_entry and hasattr(model_entry["processor"], "to") and callable(model_entry["processor"].to) \
                       and not isinstance(model_entry["processor"], (str, int, float)): # Verifică să nu fie un simplu tip de date
                        try:
                            model_entry["processor"].to("cpu")
                            logger.info(f"Moved processor of '{model_name}' to CPU.")
                        except Exception as e_proc_cpu:
                             logger.warning(f"Could not move processor of '{model_name}' to CPU: {e_proc_cpu}")
                    logger.info(f"Model '{model_name}' component moved to CPU.")
                    return True
                else:
                    logger.warning(f"Model component of '{model_name}' does not have a 'to' method. Cannot move.")
                    return False
            # Caz pentru YOLO (modelul e în model_entry['model'], device-ul e stocat separat)
            elif model_name == 'yolo' and isinstance(model_entry, dict) and 'model' in model_entry and 'device' in model_entry:
                if model_entry['device'] == 'cpu':
                    logger.info(f"YOLO model '{model_name}' already intended for CPU.")
                    return True
                # YOLO își gestionează dispozitivul la inferență. Doar actualizăm starea.
                logger.info(f"Setting intended device for YOLO model '{model_name}' to CPU.")
                model_entry['device'] = 'cpu'
                # Nu există o comandă YOLO().to('cpu') directă care să descarce de pe GPU în mod explicit aici.
                # Mutarea se face prin a nu specifica GPU la următoarea inferență.
                return True # Considerăm că am setat intenția
            # Alte modele (ex: Rembg, care își setează providers la creare)
            elif model_name == 'rembg' and isinstance(model_entry, dict) and 'device' in model_entry:
                 if model_entry['device'] == 'cpu':
                    logger.info(f"Rembg model '{model_name}' already intended for CPU.")
                    return True
                 logger.info(f"Setting intended device for Rembg model '{model_name}' to CPU. Will require reload with CPU providers.")
                 # Pentru Rembg, o schimbare reală de dispozitiv necesită re-crearea sesiunii.
                 # Aici doar marcăm și la următoarea încărcare se va folosi CPU.
                 self.unload_model(model_name) # Forțează reîncărcarea la următoarea cerere
                 logger.info(f"Unloaded Rembg model '{model_name}' to allow reloading on CPU.")
                 return True


            else:
                logger.warning(f"Model '{model_name}' (type: {type(model_entry)}) does not have a standard structure for CPU moving. Cannot move automatically.")
                return False

        except Exception as e:
            logger.error(f"Error moving model '{model_name}' to CPU: {e}", exc_info=True)
            return False
        return False


    def monitor_memory_and_recover(self) -> None:
        """
        Monitor memory usage and perform recovery actions if needed.
        Call this method periodically in long-running operations.
        """
        if not torch.cuda.is_available():
            return

        try:
            if self._is_memory_critical(): # Folosește metoda actualizată _is_memory_critical
                logger.warning(f"CRITICAL MEMORY STATE DETECTED. Initiating proactive freeing.")
                self._free_memory_proactively()
            else:
                # Verifică politica de descărcare dacă nu e memorie critică
                policy = getattr(self.config, "MODEL_LOADING_POLICY", "KEEP_LOADED") #type: ignore
                if policy == "UNLOAD_UNUSED" or policy == "AGGRESSIVE_UNLOAD":
                    self._apply_model_loading_policy()


        except Exception as e:
            logger.error(f"Error in memory monitoring/recovery: {e}", exc_info=True)


    def _apply_model_loading_policy(self, model_name_to_load: Optional[str] = None) -> None:
        """Aplică politica de încărcare/descărcare a modelelor."""
        policy = getattr(self.config, "MODEL_LOADING_POLICY", "KEEP_LOADED") #type: ignore
        essential_models = getattr(self.config, "ESSENTIAL_MODELS", ["main"]) #type: ignore
        model_unload_timeout = getattr(self.config, "MODEL_UNLOAD_TIMEOUT", 300) #type: ignore # 5 minute default

        if policy == "KEEP_LOADED":
            return # Nu face nimic

        models_currently_loaded = list(self.models.keys())
        for model_name in models_currently_loaded:
            if model_name in essential_models:
                continue # Nu descărca modelele esențiale

            # Logica pentru UNLOAD_UNUSED / AGGRESSIVE_UNLOAD
            # AGGRESSIVE_UNLOAD: descarcă tot ce nu e esențial și nu e modelul curent cerut
            if policy == "AGGRESSIVE_UNLOAD":
                if model_name_to_load is None or model_name != model_name_to_load:
                    logger.info(f"Aggressive policy: Unloading '{model_name}'.")
                    self.unload_model(model_name)

            # UNLOAD_UNUSED: necesită urmărirea timpului de neutilizare (mai complex, nu implementat complet aici)
            # Pentru simplitate, dacă e UNLOAD_UNUSED și nu e modelul curent, descarcă-l
            # O implementare corectă ar avea un timestamp last_used pentru fiecare model.
            elif policy == "UNLOAD_UNUSED":
                 if model_name_to_load is None or model_name != model_name_to_load:
                    logger.info(f"Unload unused policy (simplified): Unloading '{model_name}'.")
                    self.unload_model(model_name)
        if policy != "KEEP_LOADED":
            self._clear_gpu_memory()


    def _free_memory_proactively(self) -> None:
        """Proactively free memory when approaching critical levels."""
        if not torch.cuda.is_available(): return

        min_free_vram_mb = getattr(self.config, "MIN_FREE_VRAM_MB", 500) #type: ignore
        current_cuda_info = self.get_memory_status().get("cuda_info", {})
        # Folosim free_total_pytorch_perspective_gb care e total_gpu - reserved_by_pytorch
        free_vram_mb_pytorch = current_cuda_info.get("free_total_pytorch_perspective_gb", float('inf')) * 1024

        if free_vram_mb_pytorch >= min_free_vram_mb:
            logger.debug(f"Proactive free: Sufficient VRAM ({free_vram_mb_pytorch:.0f}MB free vs {min_free_vram_mb}MB threshold). No action.")
            return

        logger.warning(f"Low VRAM: {free_vram_mb_pytorch:.0f}MB free (below {min_free_vram_mb}MB threshold). Proactively freeing memory.")

        # Definește prioritatea de descărcare (cele mai puțin importante/mai mari primele)
        unload_priority = [
            'depth_estimator', 'image_classifier', 'yolo', 'clipseg', 'sam_predictor',
            'rembg', 'face_detector', 'mediapipe',
            # Adaugă aici alte modele în ordinea descrescătoare a importanței sau crescătoare a mărimii
        ]
        essential_models = getattr(self.config, "ESSENTIAL_MODELS", ["main"]) #type: ignore
        unload_priority = [m for m in unload_priority if m not in essential_models and m in self.models]

        # 1. Încearcă să muți pe CPU modelele compatibile
        logger.info(f"Proactive: Attempting to move models to CPU from: {unload_priority}")
        for model_name in list(unload_priority): # Copie pentru a modifica în timpul iterării
            if model_name in self.CPU_FRIENDLY_MODELS: # Verifică dacă e eligibil pentru CPU
                if self._move_model_to_cpu(model_name):
                    logger.info(f"Proactively moved '{model_name}' to CPU.")
                    current_cuda_info = self.get_memory_status().get("cuda_info", {})
                    free_vram_mb_pytorch = current_cuda_info.get("free_total_pytorch_perspective_gb", float('inf')) * 1024
                    if free_vram_mb_pytorch >= min_free_vram_mb:
                        logger.info(f"Sufficient VRAM freed by moving models to CPU: {free_vram_mb_pytorch:.0f}MB free.")
                        self._clear_gpu_memory()
                        return
                unload_priority.remove(model_name) # Elimină din lista de descărcare dacă a fost mutat

        self._clear_gpu_memory() # Curăță după mutări

        # 2. Dacă mutarea pe CPU nu a fost suficientă, descarcă modelele conform priorității
        logger.info(f"Proactive: Attempting to unload models from: {unload_priority}")
        for model_name in unload_priority:
            if model_name in self.models: # Verifică dacă mai e încărcat
                logger.warning(f"Proactively unloading '{model_name}' to free VRAM.")
                self.unload_model(model_name)
                current_cuda_info = self.get_memory_status().get("cuda_info", {})
                free_vram_mb_pytorch = current_cuda_info.get("free_total_pytorch_perspective_gb", float('inf')) * 1024
                if free_vram_mb_pytorch >= min_free_vram_mb:
                    logger.info(f"Sufficient VRAM freed by unloading models: {free_vram_mb_pytorch:.0f}MB free.")
                    self._clear_gpu_memory()
                    return
        logger.warning(f"Proactive freeing finished. Current free VRAM: {free_vram_mb_pytorch:.0f}MB.")
        self._clear_gpu_memory()


    def optimize_for_inference(self) -> None:
        """
        Optimize memory usage before running the main inference process.
        Call this before starting a generation pipeline.
        """
        logger.info("Optimizing memory for inference...")
        self._apply_model_loading_policy(model_name_to_load='main') # Descarcă tot ce nu e 'main' dacă politica o cere

        main_model_instance = self.get_model('main') # Asigură încărcarea modelului principal
        if main_model_instance is None or not (isinstance(main_model_instance, BaseModel) and main_model_instance.is_loaded): #type: ignore
            logger.error("Main model not loaded or failed to load! Cannot optimize for inference.")
            return

        # Aplică optimizări specifice modelului principal (dacă e cazul și LOW_VRAM_MODE)
        if getattr(self.config, "LOW_VRAM_MODE", False):
            if isinstance(main_model_instance, BaseModel) and hasattr(main_model_instance, 'apply_low_vram_optimizations'):
                logger.info("Applying LOW_VRAM_MODE optimizations to main model for inference.")
                try:
                    main_model_instance.apply_low_vram_optimizations() #type: ignore # Presupunem că modelele au această metodă
                except Exception as e_opt:
                    logger.error(f"Error applying low VRAM optimizations to main model: {e_opt}")
            # Logica specifică pentru HiDream (dacă mai e relevantă) a fost mutată în clasa HiDreamModel.
            # FluxModel are deja logica de CPU offload în metoda sa load() pentru LOW_VRAM_MODE.

        self._clear_gpu_memory()
        logger.info("Memory optimization for inference completed.")
        self._log_memory_stats()


    def handle_oom_error(self, func):
        """
        Decorator to handle OOM errors gracefully.
        """
        import functools

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except RuntimeError as e:
                if "CUDA out of memory" in str(e) or "HIP out of memory" in str(e): # Suport și pt AMD ROCm
                    logger.error(f"CUDA/HIP OOM error in '{func.__name__}': {str(e)}")
                    self.emergency_memory_recovery() # Încearcă recuperarea de urgență

                    # Logica de reîncercare cu CPU (opțional și dependent de funcție)
                    # Aceasta ar trebui să fie specifică funcției decorate,
                    # deoarece nu toate funcțiile pot rula pe CPU sau au parametri 'device'/'force_cpu'.
                    # Aici doar semnalăm eroarea și faptul că s-a încercat recuperarea.
                    # Re-aruncăm excepția pentru ca apelantul să știe că operațiunea a eșuat inițial.
                    logger.warning(f"Operation '{func.__name__}' failed due to OOM. Recovery attempted. Consider retrying with CPU if applicable.")
                    raise e # Re-aruncă excepția originală după încercarea de recuperare
                else:
                    # Re-aruncă alte erori RuntimeError care nu sunt OOM
                    raise
            except Exception as ex_other: # Prinde și alte excepții pentru logging
                logger.error(f"Unexpected error in '{func.__name__}': {ex_other}", exc_info=True)
                raise # Re-aruncă excepția

        return wrapper

    # load_model_with_memory_management nu mai este necesară dacă get_model și politicile sunt robuste.
    # Am integrat logica similară în get_model și _apply_model_loading_policy / _free_memory_proactively.