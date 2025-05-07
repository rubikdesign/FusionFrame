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

from config.app_config import AppConfig
from core.model_manager import ModelManager
from processing.analyzer import ImageAnalyzer
from processing.mask_generator import MaskGenerator

# Setăm logger-ul
logger = logging.getLogger(__name__)

class BasePipeline(ABC):
    """
    Clasă abstractă de bază pentru toate pipeline-urile de procesare
    
    Toate pipeline-urile specifice vor moșteni această clasă și vor
    implementa metodele abstracte.
    """
    
    def __init__(self):
        """Inițializează pipeline-ul de bază"""
        self.config = AppConfig
        self.model_manager = ModelManager()
        self.mask_generator = MaskGenerator()
        self.image_analyzer = ImageAnalyzer()
        self.progress_callback = None
    
    @abstractmethod
    def process(self, 
               image: Union[Image.Image, np.ndarray],
               prompt: str,
               strength: float = 0.75,
               progress_callback: Callable = None,
               **kwargs) -> Dict[str, Any]:
        """
        Procesează imaginea folosind pipeline-ul specific
        
        Args:
            image: Imaginea de procesat
            prompt: Promptul pentru editare
            strength: Intensitatea editării (0.0-1.0)
            progress_callback: Funcție de callback pentru progres
            **kwargs: Argumentele adiționale pentru procesare
            
        Returns:
            Dicționar cu rezultatele procesării
        """
        pass
    
    def _update_progress(self, progress: float, desc: str = None):
        """
        Actualizează callback-ul de progres dacă există
        
        Args:
            progress: Progresul curent (0.0-1.0)
            desc: Descrierea progresului (opțional)
        """
        if self.progress_callback is not None:
            self.progress_callback(progress, desc=desc)
    
    def _enhance_prompt(self, prompt: str, operation: Dict[str, Any] = None) -> str:
        """
        Îmbunătățește promptul pentru editare
        
        Args:
            prompt: Promptul original
            operation: Detalii despre operație
            
        Returns:
            Promptul îmbunătățit
        """
        # Prompt enhancers în funcție de tipul operației
        prompt_enhancers = {
            'remove': [
                "highly detailed",
                "seamless integration",
                "perfect edges",
                "no artifacts",
                "professional retouching"
            ],
            'replace': [
                "photorealistic",
                "perfect lighting matching",
                "accurate shadows",
                "consistent perspective",
                "high resolution detail"
            ],
            'color': [
                "vibrant colors",
                "natural gradients",
                "realistic textures",
                "accurate lighting",
                "high quality"
            ],
            'background': [
                "cinematic lighting",
                "ultra detailed",
                "professional photography",
                "8k resolution",
                "realistic environment"
            ],
            'add': [
                "realistic integration",
                "perfect placement",
                "natural appearance",
                "high quality details",
                "professional look"
            ],
            'general': [
                "high quality",
                "detailed",
                "professional",
                "sharp focus",
                "realistic"
            ]
        }
        
        # Selectăm enhancers în funcție de operație
        op_type = operation.get('type', 'general') if operation else 'general'
        enhancers = prompt_enhancers.get(op_type, prompt_enhancers['general'])
        
        # Construim promptul îmbunătățit
        return f"{prompt}, {', '.join(enhancers[:3])}"
    
    def _get_generation_params(self, operation_type: str = None) -> Dict[str, Any]:
        """
        Obține parametrii optimizați de generare
        
        Args:
            operation_type: Tipul operației (opțional)
            
        Returns:
            Dicționar cu parametrii de generare
        """
        # Parametri de bază
        base_params = {
            'num_inference_steps': 50,
            'guidance_scale': 7.5,
            'strength': 0.75,
            'controlnet_conditioning_scale': 0.8
        }
        
        # Ajustăm parametrii bazați pe tipul operației
        if operation_type == 'remove':
            base_params.update({
                'num_inference_steps': 60,
                'guidance_scale': 9.0,
                'strength': 0.85,
                'controlnet_conditioning_scale': 0.6
            })
        elif operation_type == 'replace':
            base_params.update({
                'num_inference_steps': 70,
                'guidance_scale': 10.0,
                'strength': 0.9,
                'controlnet_conditioning_scale': 0.9
            })
        elif operation_type == 'background':
            base_params.update({
                'num_inference_steps': 65,
                'guidance_scale': 8.5,
                'strength': 0.8,
                'controlnet_conditioning_scale': 0.7
            })
        elif operation_type == 'color':
            base_params.update({
                'num_inference_steps': 45,
                'guidance_scale': 7.0,
                'strength': 0.7,
                'controlnet_conditioning_scale': 0.5
            })
        
        return base_params
    
    def _prepare_control_image(self, 
                              image: Union[Image.Image, np.ndarray],
                              mask: Union[Image.Image, np.ndarray],
                              control_mode: str = "canny") -> Optional[Image.Image]:
        """
        Pregătește imaginea de control pentru ControlNet
        
        Args:
            image: Imaginea originală
            mask: Masca pentru editare
            control_mode: Modul de control (canny, depth, normal)
            
        Returns:
            Imaginea de control pregătită sau None în caz de eroare
        """
        try:
            # Obținem handler-ul ControlNet
            controlnet_handler = self.model_manager.get_model('controlnet')
            if controlnet_handler is None:
                # Fallback la procesare simplă
                return self._create_canny_control(image, mask)
            
            # Procesăm cu handler-ul ControlNet
            result = controlnet_handler.process(image, control_mode)
            if result['success'] and result['control_image'] is not None:
                return result['control_image']
            else:
                # Fallback la procesare simplă
                return self._create_canny_control(image, mask)
                
        except Exception as e:
            logger.error(f"Error preparing control image: {str(e)}")
            # Fallback la procesare simplă
            return self._create_canny_control(image, mask)
    
    def _create_canny_control(self, 
                            image: Union[Image.Image, np.ndarray],
                            mask: Union[Image.Image, np.ndarray]) -> Optional[Image.Image]:
        """
        Creează o imagine de control Canny simplă
        
        Args:
            image: Imaginea originală
            mask: Masca pentru editare
            
        Returns:
            Imaginea de control Canny sau None în caz de eroare
        """
        try:
            # Convertim la numpy dacă este PIL
            if isinstance(image, Image.Image):
                image_np = np.array(image)
            else:
                image_np = image
            
            # Convertim masca la numpy dacă este PIL
            if isinstance(mask, Image.Image):
                mask_np = np.array(mask)
            else:
                mask_np = mask
            
            # Ne asigurăm că masca este binară
            if mask_np.max() > 1 and mask_np.dtype != np.bool_:
                mask_np = mask_np > 127
            
            # Convertim la tonuri de gri
            if len(image_np.shape) == 3:
                gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
            else:
                gray = image_np
            
            # Praguri adaptive bazate pe conținutul imaginii
            median_value = np.median(gray)
            lower_threshold = max(0, int(median_value * 0.7))
            upper_threshold = min(255, int(median_value * 1.3))
            
            # Aplicăm detecția de margini Canny
            edges = cv2.Canny(gray, lower_threshold, upper_threshold)
            
            # Aplicăm masca la margini
            masked_edges = cv2.bitwise_and(edges, edges, mask=mask_np.astype(np.uint8))
            
            # Blur ușor pentru a reduce zgomotul
            masked_edges = cv2.GaussianBlur(masked_edges, (3, 3), 0)
            
            # Convertim la imagine PIL
            return Image.fromarray(masked_edges)
            
        except Exception as e:
            logger.error(f"Error creating Canny control image: {str(e)}")
            return None
    
    def _select_model(self, operation_type: str = None) -> str:
        """
        Selectează modelul potrivit pentru operație
        
        Args:
            operation_type: Tipul operației
            
        Returns:
            Numele modelului selectat
        """
        # Implicit folosim modelul principal (HiDream)
        model_name = 'main'
        
        # Pentru operații speciale, putem decide să folosim alt model
        if operation_type == 'color' and operation_type == 'background':
            # FLUX poate fi mai potrivit pentru unele operații
            model_name = 'flux'
        
        return model_name