#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Pipeline de bază pentru procesare imagine în FusionFrame 2.0
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union, Tuple, Callable
import numpy as np
from PIL import Image
import sys 
import cv2 # Import cv2 pentru _create_canny_control

# --- Importuri Critice pentru Configurare ---
try:
    from config.app_config import AppConfig
    from config.model_config import ModelConfig # <-- IMPORTUL ESENȚIAL PENTRU ModelConfig
except ImportError as e_cfg:
    logging.basicConfig(level=logging.CRITICAL)
    logging.critical(f"CRITICAL ERROR: Failed to import AppConfig or ModelConfig in base_pipeline.py: {e_cfg}", exc_info=True)
    sys.exit(f"Critical configuration import error: {e_cfg}")

# --- Alte Importuri Core ---
try:
    from core.model_manager import ModelManager
    from processing.analyzer import ImageAnalyzer
    from processing.mask_generator import MaskGenerator 
except ImportError as e_core:
     # Folosim un logger de bază dacă cel configurat nu e disponibil încă
     logging.basicConfig(level=logging.ERROR)
     logger_fallback = logging.getLogger(__name__) 
     logger_fallback.error(f"ERROR: Failed to import core modules (ModelManager, ImageAnalyzer, MaskGenerator) in base_pipeline.py: {e_core}", exc_info=True)
     # Setăm la None pentru a putea verifica existența lor în __init__
     ModelManager = None
     ImageAnalyzer = None
     MaskGenerator = None


# Setăm logger-ul (ar trebui să fie deja configurat de AppConfig.setup_logging() la rularea aplicației)
logger = logging.getLogger(__name__)
if not logger.hasHandlers() or not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
    _ch = logging.StreamHandler(); _f = logging.Formatter("%(asctime)s-%(name)s-%(levelname)s-%(message)s"); _ch.setFormatter(_f)
    logger.addHandler(_ch); 
    if logger.level == logging.NOTSET: logger.setLevel(logging.INFO)


class BasePipeline(ABC):
    """
    Clasă abstractă de bază pentru toate pipeline-urile de procesare.
    """
    
    def __init__(self):
        """Inițializează pipeline-ul de bază."""
        self.config_class = AppConfig 
        
        if ModelManager is None: # Verificăm dacă importul a reușit
            raise RuntimeError("ModelManager could not be imported. BasePipeline cannot initialize.")
        self.model_manager = ModelManager()
        
        try:
            if MaskGenerator is None: raise RuntimeError("MaskGenerator could not be imported.")
            self.mask_generator = MaskGenerator()
            if ImageAnalyzer is None: raise RuntimeError("ImageAnalyzer could not be imported.")
            self.image_analyzer = ImageAnalyzer()
        except Exception as e_init_comp:
            logger.critical(f"Error initializing components (MaskGenerator, ImageAnalyzer) in BasePipeline: {e_init_comp}", exc_info=True)
            raise e_init_comp

        self.progress_callback: Optional[Callable[[float, Optional[str]], None]] = None 
    
    @abstractmethod
    def process(self, 
               image: Union[Image.Image, np.ndarray],
               prompt: str,
               strength: float = 0.75,
               progress_callback: Optional[Callable[[float, Optional[str]], None]] = None, 
               **kwargs) -> Dict[str, Any]:
        """Metodă abstractă pentru procesarea imaginii."""
        pass 
    
    def _update_progress(self, progress: float, desc: Optional[str] = None): 
        """Actualizează callback-ul de progres."""
        if self.progress_callback is not None:
            try:
                self.progress_callback(progress, desc) 
            except TypeError as te:
                 logger.error(f"TypeError calling progress callback: {te}. Args: progress={progress}, desc='{desc}'", exc_info=False)
            except Exception as e_cb:
                 logger.error(f"Error executing progress callback: {e_cb}", exc_info=True)
    
    def _enhance_prompt(self, prompt: str, operation: Optional[Dict[str, Any]] = None) -> str:
        """Îmbunătățește promptul pentru editare."""
        prompt_enhancers = {
            'remove': ["highly detailed", "seamless integration", "perfect edges", "no artifacts", "professional retouching"],
            'replace': ["photorealistic", "perfect lighting matching", "accurate shadows", "consistent perspective", "high resolution detail"],
            'color': ["vibrant colors", "natural gradients", "realistic textures", "accurate lighting", "high quality"],
            'background': ["cinematic lighting", "ultra detailed", "professional photography", "8k resolution", "realistic environment"],
            'add': ["realistic integration", "perfect placement", "natural appearance", "high quality details", "professional look"],
            'general': ["high quality", "detailed", "professional", "sharp focus", "realistic"]
        }
        op_type = operation.get('type', 'general') if operation else 'general'
        enhancers = prompt_enhancers.get(op_type, prompt_enhancers['general'])
        num_enhancers_to_add = 3 
        if len(prompt) < 150: 
             return f"{prompt}, {', '.join(enhancers[:num_enhancers_to_add])}"
        return prompt

    def _get_generation_params(self, operation_type: Optional[str] = None) -> Dict[str, Any]:
        """Obține parametrii optimizați de generare."""
        base_params = {
            'num_inference_steps': self.config_class.DEFAULT_STEPS,
            'guidance_scale': self.config_class.DEFAULT_GUIDANCE_SCALE,
            'strength': 0.75, 
            # ModelConfig este acum importat și disponibil în scope-ul modulului
            'controlnet_conditioning_scale': getattr(ModelConfig, 'CONTROLNET_CONFIG', {}).get('conditioning_scale', 0.8) 
        }
        
        if operation_type == 'remove':
            base_params.update({'num_inference_steps': 60, 'guidance_scale': 9.0, 'strength': 0.9, 'controlnet_conditioning_scale': 0.6})
        elif operation_type == 'replace':
            base_params.update({'num_inference_steps': 70, 'guidance_scale': 10.0, 'strength': 0.9, 'controlnet_conditioning_scale': 0.9})
        elif operation_type == 'background':
            base_params.update({'num_inference_steps': 65, 'guidance_scale': 8.5, 'strength': 0.85, 'controlnet_conditioning_scale': 0.7})
        elif operation_type == 'color':
            base_params.update({'num_inference_steps': 45, 'guidance_scale': 7.0, 'strength': 0.7, 'controlnet_conditioning_scale': 0.5})
        
        return base_params
    
    def _prepare_control_image(self, 
                              image: Union[Image.Image, np.ndarray],
                              control_mode: str = "canny") -> Optional[Image.Image]:
        logger.debug(f"Preparing control image using mode: {control_mode}")
        if control_mode.lower() == "canny":
            return self._create_canny_control(image)
        else:
            logger.warning(f"Control mode '{control_mode}' not implemented in base _prepare_control_image.")
            return None
    
    def _create_canny_control(self, 
                            image: Union[Image.Image, np.ndarray]
                            ) -> Optional[Image.Image]:
        try:
            if isinstance(image, Image.Image): image_np = np.array(image.convert("RGB"))
            else: image_np = image.copy() 

            if image_np.ndim == 3 and image_np.shape[2] == 3: 
                 if isinstance(image, Image.Image): image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
                 else: image_bgr = image_np 
            elif image_np.ndim == 2: image_bgr = cv2.cvtColor(image_np, cv2.COLOR_GRAY2BGR)
            else: logger.error(f"Cannot convert image shape {image_np.shape} to BGR for Canny."); return None

            gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
            low_threshold = 100; high_threshold = 200
            edges = cv2.Canny(gray, low_threshold, high_threshold)
            edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB) 
            return Image.fromarray(edges_rgb)
        except Exception as e:
            logger.error(f"Error creating Canny control image: {str(e)}", exc_info=True)
            return None

