#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Pipeline pentru înlocuirea fundalurilor în FusionFrame 2.0
"""

import logging
import cv2
import numpy as np
from typing import Dict, Any, Optional, Union, Tuple, Callable
from PIL import Image

from fusionframe.processing.pipelines.base_pipeline import BasePipeline
from fusionframe.processing.analyzer import OperationAnalyzer

# Setăm logger-ul
logger = logging.getLogger(__name__)

class BackgroundPipeline(BasePipeline):
    """
    Pipeline specializat pentru înlocuirea fundalurilor
    
    Implementează algoritmi avansați pentru înlocuirea completă
    sau parțială a fundalului cu păstrarea subiectului principal.
    """
    
    def __init__(self):
        """Inițializează pipeline-ul pentru înlocuirea fundalurilor"""
        super().__init__()
        self.operation_analyzer = OperationAnalyzer()
    
    def process(self,
               image: Union[Image.Image, np.ndarray],
               prompt: str,
               strength: float = 0.75,
               progress_callback: Callable = None,
               **kwargs) -> Dict[str, Any]:
        """
        Procesează imaginea pentru a înlocui fundalul
        
        Args:
            image: Imaginea de procesat
            prompt: Promptul pentru înlocuirea fundalului
            strength: Intensitatea înlocuirii (0.0-1.0)
            progress_callback: Funcție de callback pentru progres
            **kwargs: Argumentele adiționale pentru procesare
            
        Returns:
            Dicționar cu rezultatele procesării
        """
        self.progress_callback = progress_callback
        
        # Analizăm operația pentru a determina ce fundal adăugăm
        operation = self.operation_analyzer.analyze_operation(prompt)
        
        # Convertim la format potrivit
        if isinstance(image, np.ndarray):
            image_np = image
            pil_image = Image.fromarray(image)
        else:
            image_np = np.array(image)
            pil_image = image
        
        # 1. Analizăm imaginea și contextul
        self._update_progress(0.1, desc="Analiză imagine...")
        image_context = self.image_analyzer.analyze_image_context(image_np)
        
        # 2. Generăm masca pentru segmentarea subiect/fundal
        self._update_progress(0.2, desc="Segmentare subiect/fundal...")
        
        # Încercăm mai întâi cu REMBG pentru extracția subiectului
        try:
            rembg_model = self.model_manager.get_specialized_model('rembg')
            if rembg_model is not None:
                self._update_progress(0.3, desc="Extragere subiect...")
                # Extragem subiectul
                subject_result = rembg_model.remove(image_np)
                # Creăm masca din canalul alpha
                if subject_result.shape[2] == 4:  # Are canal alpha
                    subject_mask = subject_result[:,:,3]
                else:
                    # Fallback la MediaPipe
                    mediapipe = self.model_manager.get_model('mediapipe')
                    if mediapipe is not None:
                        results = mediapipe.process(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))
                        if results.segmentation_mask is not None:
                            subject_mask = (results.segmentation_mask * 255).astype(np.uint8)
                        else:
                            subject_mask = self._generate_fallback_mask(image_np)
                    else:
                        subject_mask = self._generate_fallback_mask(image_np)
            else:
                # Încercăm cu MediaPipe dacă REMBG nu este disponibil
                mediapipe = self.model_manager.get_model('mediapipe')
                if mediapipe is not None:
                    results = mediapipe.process(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))
                    if results.segmentation_mask is not None:
                        subject_mask = (results.segmentation_mask * 255).astype(np.uint8)
                    else:
                        subject_mask = self._generate_fallback_mask(image_np)
                else:
                    subject_mask = self._generate_fallback_mask(image_np)
        except Exception as e:
            logger.error(f"Error in subject extraction: {str(e)}")
            subject_mask = self._generate_fallback_mask(image_np)
        
        # Inversăm masca pentru a obține masca fundalului
        background_mask = 255 - subject_mask
        
        # Rafinăm masca de fundal pentru a asigura tranziții netede
        self._update_progress(0.4, desc="Rafinare mască...")
        background_mask = self.mask_generator.refine_mask(background_mask, image_np)
        
        # 3. Procesăm înlocuirea fundalului folosind modelul principal
        self._update_progress(0.6, desc="Generare fundal nou...")
        
        # Obținem modelul principal
        main_model = self.model_manager.get_model('main')
        if main_model is None:
            logger.error("Main model not available")
            return {
                'result': pil_image,
                'mask': Image.fromarray(background_mask),
                'operation': operation,
                'message': "Modelul principal nu este disponibil"
            }
        
        # Obținem tema fundalului
        background_theme = operation.get('attribute', 'natural')
        
        # Îmbunătățim prompt-ul cu contextul imaginii
        lighting_desc = image_context.get('lighting', 'balanced')
        style_desc = image_context.get('style', 'natural')
        enhanced_prompt = f"{background_theme} background scene, {lighting_desc} lighting, {style_desc} style, detailed {background_theme} scene, professional photography"
        
        # Ajustăm parametrii pentru înlocuirea fundalului
        params = self._get_generation_params('background')
        
        # Procesăm cu modelul
        self._update_progress(0.7, desc="Aplicare fundal nou...")
        try:
            result = main_model.process(
                image=pil_image,
                mask_image=Image.fromarray(background_mask),
                prompt=enhanced_prompt,
                negative_prompt="bad background, distortion, blurry, inconsistent lighting",
                strength=params['strength'],
                num_inference_steps=params['num_inference_steps'],
                guidance_scale=params['guidance_scale'],
                controlnet_conditioning_scale=params.get('controlnet_conditioning_scale')
            )
            
            if result['success']:
                self._update_progress(1.0, desc="Procesare completă!")
                return {
                    'result': result['result'],
                    'mask': Image.fromarray(background_mask),
                    'operation': operation,
                    'message': f"Fundal înlocuit cu {background_theme} cu succes"
                }
            else:
                logger.error(f"Error in processing: {result['message']}")
                return {
                    'result': pil_image,
                    'mask': Image.fromarray(background_mask),
                    'operation': operation,
                    'message': result['message']
                }
                
        except Exception as e:
            logger.error(f"Error in background replacement: {str(e)}")
            return {
                'result': pil_image,
                'mask': Image.fromarray(background_mask),
                'operation': operation,
                'message': f"Eroare: {str(e)}"
            }
    
    def _generate_fallback_mask(self, image_np: np.ndarray) -> np.ndarray:
        """
        Generează o mască de rezervă pentru segmentarea subiect/fundal
        
        Args:
            image_np: Imaginea de procesat
            
        Returns:
            Masca generată
        """
        h, w = image_np.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # Creăm o mască centrală pentru subiect
        center_x, center_y = w // 2, h // 2
        radius_x = w // 3
        radius_y = h // 2
        
        # Desenăm o elipsă pentru a aproxima un subiect central
        cv2.ellipse(
            mask,
            (center_x, center_y),
            (radius_x, radius_y),
            0,
            0,
            360,
            255,
            -1
        )
        
        return mask