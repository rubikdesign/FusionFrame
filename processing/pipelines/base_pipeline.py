#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Pipeline de bază pentru procesare imagine în FusionFrame 2.0
(Actualizat pentru a folosi PromptEnhancer contextual)
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union, Tuple, Callable
import numpy as np
from PIL import Image
import cv2 # Adăugat pentru funcțiile de conversie
import time # Adăugat
from config.app_config import AppConfig
from config.model_config import ModelConfig # Importăm ModelConfig pentru negative_prompt default
from core.model_manager import ModelManager
from processing.analyzer import ImageAnalyzer
from processing.mask_generator import MaskGenerator
# NOU: Importăm PromptEnhancer
from processing.prompt_enhancer import PromptEnhancer

# Setăm logger-ul
logger = logging.getLogger(__name__)

class BasePipeline(ABC):
    """
    Clasă abstractă de bază pentru toate pipeline-urile de procesare.
    Include acum logica centralizată pentru îmbunătățirea prompturilor.
    """

    def __init__(self):
        """Inițializează pipeline-ul de bază"""
        self.config = AppConfig
        self.model_config = ModelConfig # Stocăm și config model
        self.model_manager = ModelManager()
        self.mask_generator = MaskGenerator()
        self.image_analyzer = ImageAnalyzer()
        # NOU: Inițializăm PromptEnhancer
        self.prompt_enhancer = PromptEnhancer()
        self.progress_callback = None

    @abstractmethod
    def process(self,
               image: Union[Image.Image, np.ndarray],
               prompt: str,
               strength: float = 0.75,
               progress_callback: Optional[Callable] = None, # Optional
               operation: Optional[Dict[str, Any]] = None,
               image_context: Optional[Dict[str, Any]] = None,
               num_inference_steps: Optional[int] = None,
               guidance_scale: Optional[float] = None,
               use_controlnet_if_available: bool = True,
               use_refiner_if_available: Optional[bool] = None,
               refiner_strength: Optional[float] = None,
               enhance_details: bool = False, # Adăugat pt semnătură standard
               fix_faces: bool = False,       # Adăugat pt semnătură standard
               remove_artifacts: bool = False,# Adăugat pt semnătură standard
               **kwargs) -> Dict[str, Any]:
        """
        Metodă abstractă pentru procesarea imaginii.
        Include acum parametrii comuni pentru a standardiza semnătura.

        Returns:
            Dicționar standardizat: {'result_image': PIL.Image | None,
                                    'mask_image': PIL.Image | None,
                                    'operation': Dict,
                                    'message': str,
                                    'success': bool}
        """
        pass

    def _update_progress(self, progress: float, desc: str = None):
        """Actualizează callback-ul de progres."""
        if self.progress_callback is not None:
            try:
                self.progress_callback(progress, desc=desc)
            except Exception as e:
                 logger.warning(f"Progress callback failed: {e}")
                 self.progress_callback = None

    # --- Metode Noi pentru Prompt Enhancement ---



    # În BasePipeline, adaugă aceste metode:

    def _enhance_prompt(self, prompt: str, operation: Optional[Dict[str, Any]] = None, 
                    image_context: Optional[Dict[str, Any]] = None) -> str:
        """Îmbunătățește promptul pozitiv folosind PromptEnhancer."""
        if not self.prompt_enhancer:
            logger.warning("PromptEnhancer not initialized. Returning original prompt.")
            return prompt
        try:
            op_type = operation.get("type") if operation else None
            return self.prompt_enhancer.enhance_prompt(prompt, operation_type=op_type, image_context=image_context)
        except Exception as e:
            logger.error(f"Error enhancing prompt: {e}. Returning original.", exc_info=True)
            return prompt

    def _get_negative_prompt(self, prompt: str, operation: Optional[Dict[str, Any]] = None,
                            image_context: Optional[Dict[str, Any]] = None) -> str:
        """Generează promptul negativ folosind PromptEnhancer."""
        fallback_negative = self.model_config.GENERATION_PARAMS.get("negative_prompt", "low quality, blurry")
        if not self.prompt_enhancer:
            logger.warning("PromptEnhancer not initialized. Returning default negative prompt.")
            return fallback_negative
        try:
            op_type = operation.get("type") if operation else None
            negative = self.prompt_enhancer.generate_negative_prompt(prompt=prompt, operation_type=op_type, image_context=image_context)
            return negative if negative else fallback_negative
        except Exception as e:
            logger.error(f"Error generating negative prompt: {e}. Returning default.", exc_info=True)
            return fallback_negative

    def _enhance_prompt(self,
                        prompt: str,
                        operation: Optional[Dict[str, Any]] = None,
                        image_context: Optional[Dict[str, Any]] = None) -> str:
        """Îmbunătățește promptul pozitiv folosind PromptEnhancer."""
        if not self.prompt_enhancer:
            logger.warning("PromptEnhancer not initialized. Returning original prompt.")
            return prompt
        try:
            op_type = operation.get('type') if operation else None
            return self.prompt_enhancer.enhance_prompt(prompt, operation_type=op_type, image_context=image_context)
        except Exception as e:
            logger.error(f"Error enhancing prompt: {e}. Returning original.", exc_info=True)
            return prompt

    def _get_negative_prompt(self,
                             prompt: str, # Promptul pozitiv original
                             operation: Optional[Dict[str, Any]] = None,
                             image_context: Optional[Dict[str, Any]] = None) -> str:
        """Generează promptul negativ folosind PromptEnhancer."""
        fallback_negative = self.model_config.GENERATION_PARAMS.get("negative_prompt", "low quality, blurry")
        if not self.prompt_enhancer:
            logger.warning("PromptEnhancer not initialized. Returning default negative prompt.")
            return fallback_negative
        try:
            op_type = operation.get('type') if operation else None
            negative = self.prompt_enhancer.generate_negative_prompt(prompt=prompt, operation_type=op_type, image_context=image_context)
            return negative if negative else fallback_negative
        except Exception as e:
            logger.error(f"Error generating negative prompt: {e}. Returning default.", exc_info=True)
            return fallback_negative

    # --- Metode Utilitare ---

    def _get_generation_params(self, operation_type: str = None) -> Dict[str, Any]:
        """Obține parametrii de generare default specifici operației."""
        base_params = {
            'num_inference_steps': self.model_config.GENERATION_PARAMS['default_steps'],
            'guidance_scale': self.model_config.GENERATION_PARAMS['guidance_scale'],
            'strength': 0.75,
            'controlnet_conditioning_scale': self.model_config.CONTROLNET_CONFIG.get('conditioning_scale', 0.7)
        }
        if operation_type == 'remove': base_params.update({'strength': 0.85, 'num_inference_steps': 60})
        elif operation_type == 'replace': base_params.update({'strength': 0.90, 'num_inference_steps': 70})
        elif operation_type == 'color': base_params.update({'strength': 0.65, 'num_inference_steps': 45})
        elif operation_type == 'background': base_params.update({'strength': 0.80, 'num_inference_steps': 65})
        logger.debug(f"Base generation params for op '{operation_type}': {base_params}")
        return base_params

    def _convert_to_pil(self, image: Union[Image.Image, np.ndarray], mode: str = "RGB") -> Image.Image:
        """Converteste inputul in PIL Image (RGB sau alt mod specificat)."""
        if isinstance(image, Image.Image):
            return image.convert(mode) if image.mode != mode else image
        elif isinstance(image, np.ndarray):
            if image.ndim == 2: # Grayscale
                 if mode == "L": return Image.fromarray(image)
                 elif mode == "RGB": return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_GRAY2RGB))
                 else: raise ValueError(f"Cannot convert grayscale NumPy to PIL mode {mode}")
            elif image.shape[2] == 4: # RGBA
                 img_rgba = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA) # Asigurăm RGBA order
                 pil_img = Image.fromarray(img_rgba, 'RGBA')
                 return pil_img.convert(mode) if mode != "RGBA" else pil_img
            elif image.shape[2] == 3: # Presupunem BGR
                 img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                 pil_img = Image.fromarray(img_rgb, 'RGB')
                 return pil_img.convert(mode) if mode != "RGB" else pil_img
            else: raise ValueError(f"Unsupported NumPy shape for PIL conversion: {image.shape}")
        else: raise TypeError(f"Unsupported type for PIL conversion: {type(image)}")

    def _convert_to_cv2(self, image: Union[Image.Image, np.ndarray]) -> np.ndarray:
        """Converteste inputul in NumPy array BGR."""
        if isinstance(image, np.ndarray):
            if image.ndim == 2: return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            elif image.shape[2] == 4: return cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
            elif image.shape[2] == 3: return image # Asumăm BGR
            else: raise ValueError(f"Unsupported NumPy shape for CV2 conversion: {image.shape}")
        elif isinstance(image, Image.Image):
            mode = image.mode
            if mode == 'L': return cv2.cvtColor(np.array(image), cv2.COLOR_GRAY2BGR)
            elif mode == 'RGBA': return cv2.cvtColor(np.array(image.convert('RGBA')), cv2.COLOR_RGBA2BGR) # Convert to RGBA first
            elif mode == 'RGB': return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            else: # Încercăm conversia la RGB ca fallback
                 logger.warning(f"Converting PIL image from mode {mode} to RGB before CV2 BGR.")
                 return cv2.cvtColor(np.array(image.convert('RGB')), cv2.COLOR_RGB2BGR)
        else: raise TypeError(f"Unsupported type for CV2 conversion: {type(image)}")

    def _ensure_pil_mask(self, mask: Optional[Union[Image.Image, np.ndarray]]) -> Optional[Image.Image]:
        """Asigură că masca este PIL Image în mod 'L'."""
        if mask is None: return None
        if isinstance(mask, Image.Image):
             return mask.convert("L") if mask.mode != "L" else mask
        elif isinstance(mask, np.ndarray):
             if mask.ndim == 3 and mask.shape[2] == 1: mask = mask.squeeze(axis=2)
             if mask.ndim != 2: raise ValueError("Mask NumPy array must be 2D for PIL conversion.")
             # Normalizăm la 0-255 dacă e necesar
             if mask.dtype != np.uint8:
                  if np.max(mask) <= 1.0 and (mask.dtype == np.float32 or mask.dtype == np.float64):
                       mask = (mask * 255).astype(np.uint8)
                  else:
                       mask = np.clip(mask, 0, 255).astype(np.uint8)
             return Image.fromarray(mask, 'L')
        else: raise TypeError(f"Unsupported mask type: {type(mask)}")