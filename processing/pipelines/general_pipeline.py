#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Pipeline general pentru procesare în FusionFrame 2.0
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

class GeneralPipeline(BasePipeline):
    """
    Pipeline general pentru procesarea imaginilor
    
    Implementează un pipeline general pentru diverse operații
    care nu se încadrează în celelalte categorii specializate.
    """
    
    def __init__(self):
        """Inițializează pipeline-ul general"""
        super().__init__()
        self.operation_analyzer = OperationAnalyzer()
    
    def process(self,
               image: Union[Image.Image, np.ndarray],
               prompt: str,
               strength: float = 0.75,
               progress_callback: Callable = None,
               **kwargs) -> Dict[str, Any]:
        """
        Procesează imaginea folosind pipeline-ul general
        
        Args:
            image: Imaginea de procesat
            prompt: Promptul pentru procesare
            strength: Intensitatea procesării (0.0-1.0)
            progress_callback: Funcție de callback pentru progres
            **kwargs: Argumentele adiționale pentru procesare
            
        Returns:
            Dicționar cu rezultatele procesării
        """
        self.progress_callback = progress_callback
        
        # Analizăm operația pentru a determina tipul de procesare
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
        
        # 2. Generăm masca pentru regiunea de interes
        self._update_progress(0.2, desc="Generare mască...")
        mask_result = self.mask_generator.generate_mask(
            image=image_np,
            prompt=prompt,
            operation=operation,
            progress_callback=lambda p, desc: self._update_progress(0.2 + p * 0.3, desc=desc)
        )
        
        # Verificăm rezultatul măștii
        if not mask_result['success']:
            logger.error("Failed to generate mask")
            return {
                'result': pil_image,
                'mask': None,
                'operation': operation,
                'message': "Eroare la generarea măștii"
            }
        
        # Obținem masca
        mask = mask_result['mask']
        
        # 3. Procesăm cu modelul principal
        self._update_progress(0.6, desc="Procesare imagine...")
        
        # Obținem modelul principal
        main_model = self.model_manager.get_model('main')
        if main_model is None:
            logger.error("Main model not available")
            return {
                'result': pil_image,
                'mask': Image.fromarray(mask),
                'operation': operation,
                'message': "Modelul principal nu este disponibil"
            }
        
        # Îmbunătățim prompt-ul
        enhanced_prompt = self._enhance_prompt(prompt, operation)
        
        # Setăm parametrii pentru procesare
        params = self._get_generation_params(operation.get('type', 'general'))
        
        # Procesăm cu modelul
        self._update_progress(0.7, desc="Generare rezultat...")
        try:
            result = main_model.process(
                image=pil_image,
                mask_image=Image.fromarray(mask),
                prompt=enhanced_prompt,
                negative_prompt="distortion, deformation, artifact, blurry, low quality",
                strength=strength,
                num_inference_steps=params['num_inference_steps'],
                guidance_scale=params['guidance_scale'],
                controlnet_conditioning_scale=params.get('controlnet_conditioning_scale')
            )
            
            if result['success']:
                self._update_progress(1.0, desc="Procesare completă!")
                return {
                    'result': result['result'],
                    'mask': Image.fromarray(mask),
                    'operation': operation,
                    'message': "Procesare completă cu succes"
                }
            else:
                logger.error(f"Error in processing: {result['message']}")
                return {
                    'result': pil_image,
                    'mask': Image.fromarray(mask),
                    'operation': operation,
                    'message': result['message']
                }
                
        except Exception as e:
            logger.error(f"Error in general processing: {str(e)}")
            return {
                'result': pil_image,
                'mask': Image.fromarray(mask),
                'operation': operation,
                'message': f"Eroare: {str(e)}"
            }