#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Pipeline general pentru procesare în FusionFrame 2.0
"""

import logging
import cv2
import numpy as np
from typing import Dict, Any, Union, Callable
from PIL import Image

from processing.pipelines.base_pipeline import BasePipeline
from processing.analyzer import OperationAnalyzer

# Setăm logger-ul
logger = logging.getLogger(__name__)

class GeneralPipeline(BasePipeline):
    """
    Pipeline general pentru procesarea imaginilor
    """
    def __init__(self):
        super().__init__()
        self.operation_analyzer = OperationAnalyzer()

    def process(self,
                image: Union[Image.Image, np.ndarray],
                prompt: str,
                strength: float = 0.75,
                progress_callback: Callable = None,
                use_refiner: bool = None,
                refiner_strength: float = None,
                **kwargs) -> Dict[str, Any]:
        """
        Procesează imaginea folosind pipeline-ul general
        
        Args:
            image: Imaginea de procesat
            prompt: Promptul pentru procesare
            strength: Intensitatea procesării (0.0-1.0)
            progress_callback: Funcție de callback pentru progres
            use_refiner: Dacă să folosească refiner
            refiner_strength: Intensitatea refiner-ului
            **kwargs: Argumentele adiționale pentru procesare
            
        Returns:
            Dicționar cu rezultatele procesării
        """
        # Salvăm callback-ul de progres
        self.progress_callback = progress_callback
        # Determinăm tipul de operație din prompt
        operation = self.operation_analyzer.analyze_operation(prompt)

        # 1. Convertim inputul în np.ndarray și PIL
        if isinstance(image, np.ndarray):
            image_np = image
            pil_image = Image.fromarray(image)
        else:
            image_np = np.array(image)
            pil_image = image

        # 2. Analiză imagine și context
        self._update_progress(0.1, desc="Analiză imagine...")
        image_context = self.image_analyzer.analyze_image_context(image_np)

        # 3. Generare mască
        self._update_progress(0.2, desc="Generare mască...")
        mask_result = self.mask_generator.generate_mask(
            image=image_np,
            prompt=prompt,
            operation=operation,
            progress_callback=lambda p, desc=None: self._update_progress(0.2 + p * 0.3, desc=desc)
        )
        if not mask_result.get('success'):
            logger.error("Failed to generate mask")
            return {
                'result': pil_image,
                'mask': None,
                'operation': operation,
                'message': "Eroare la generarea măștii"
            }
        mask_image = mask_result['mask']

        # 4. Procesare model principal
        self._update_progress(0.6, desc="Procesare imagine...")
        main_model = self.model_manager.get_model('main')
        if main_model is None:
            logger.error("Main model not available")
            return {
                'result': pil_image,
                'mask': Image.fromarray(mask_image),
                'operation': operation,
                'message': "Modelul principal nu este disponibil"
            }

        # 5. Prompt enhancement și setare parametri
        enhanced_prompt = self._enhance_prompt(prompt, operation)
        params = self._get_generation_params(operation.get('type', 'general'))

        # 6. Apel pipeline
        self._update_progress(0.7, desc="Generare rezultat...")
        try:
            output = main_model(
                image=pil_image,
                mask_image=Image.fromarray(mask_image),
                prompt=enhanced_prompt,
                negative_prompt="distortion, deformation, artifact, blurry, low quality",
                strength=strength,
                num_inference_steps=params['num_inference_steps'],
                guidance_scale=params['guidance_scale'],
                controlnet_conditioning_scale=params.get('controlnet_conditioning_scale'),
                use_refiner=use_refiner,
                refiner_strength=refiner_strength
            )
            # Extragem imaginea procesată
            result_image = output.images[0]
            self._update_progress(1.0, desc="Procesare completă!")
            return {
                'result': result_image,
                'mask': Image.fromarray(mask_image),
                'operation': operation,
                'message': "Procesare completă cu succes"
            }
        except Exception as e:
            logger.error(f"Error in general processing: {e}")
            return {
                'result': pil_image,
                'mask': Image.fromarray(mask_image),
                'operation': operation,
                'message': f"Eroare: {e}"
            }