#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Pipeline pentru schimbarea culorilor în FusionFrame 2.0
(Actualizat pentru a folosi PromptEnhancer din BasePipeline)
"""

import logging
import cv2
import numpy as np
from typing import Dict, Any, Optional, Union, Tuple, Callable
from PIL import Image
import time # Adăugat

from processing.pipelines.base_pipeline import BasePipeline
from processing.analyzer import OperationAnalyzer

logger = logging.getLogger(__name__)

class ColorChangePipeline(BasePipeline):
    """Pipeline specializat pentru schimbarea culorilor."""
    def __init__(self):
        super().__init__()
        self.operation_analyzer = OperationAnalyzer()

    def process(self,
               image: Union[Image.Image, np.ndarray],
               prompt: str,
               strength: float = 0.7, # Default puțin mai mic pt culoare
               progress_callback: Optional[Callable] = None,
               operation: Optional[Dict[str, Any]] = None,
               image_context: Optional[Dict[str, Any]] = None,
               num_inference_steps: Optional[int] = None,
               guidance_scale: Optional[float] = None,
               use_controlnet_if_available: bool = False, # ControlNet rar util pt culoare
               use_refiner_if_available: Optional[bool] = None,
               refiner_strength: Optional[float] = None,
               **kwargs) -> Dict[str, Any]:

        self.progress_callback = progress_callback
        start_time = time.time()

        # 1. Analiză operație și input
        if operation is None: operation = self.operation_analyzer.analyze_operation(prompt)
        op_type = operation.get('type', 'color')
        target_object = operation.get('target_object', 'object')
        target_color = operation.get('attribute', 'specified color')

        try:
            pil_image = self._convert_to_pil(image)
            image_np = self._convert_to_cv2(pil_image)
        except (TypeError, ValueError) as e:
             msg = f"Input image conversion error: {e}"; logger.error(msg)
             return {'result_image': None, 'mask_image': None, 'operation': operation, 'message': msg, 'success': False}

        # 2. Analiză imagine
        if image_context is None:
            self._update_progress(0.1, desc="Analiză imagine...")
            image_context = self.image_analyzer.analyze_image_context(pil_image)
            if "error" in image_context: image_context = {}
        else:
            self._update_progress(0.1, desc="Context imagine primit.")

        # 3. Generare mască
        self._update_progress(0.2, desc=f"Generare mască {target_object}...")
        mask_result = self.mask_generator.generate_mask(
            image=image_np, prompt=f"{target_object}", operation=operation,
            progress_callback=lambda p, desc=None: self._update_progress(0.2 + p * 0.4, desc=desc)
        )
        if not mask_result.get('success'):
            msg = f"Mask generation failed for '{target_object}': {mask_result.get('message')}"
            logger.error(msg); return {'result_image': pil_image, 'mask_image': None, 'operation': operation, 'message': msg, 'success': False}

        mask_np = mask_result.get('mask')
        mask_pil = self._ensure_pil_mask(mask_np)
        if mask_pil is None:
            msg = f"Mask is None after generation for '{target_object}'."; logger.error(msg)
            return {'result_image': pil_image, 'mask_image': None, 'operation': operation, 'message': msg, 'success': False}

        # 4. Îmbunătățire Prompt și Parametri
        self._update_progress(0.6, desc="Pregătire prompt culoare...")
        color_prompt = f"{target_color} {target_object}"
        enhanced_prompt = self._enhance_prompt(color_prompt, operation=operation, image_context=image_context)
        enhanced_prompt += ", realistic color, natural texture, correct lighting" # Specific color change
        negative_prompt = self._get_negative_prompt(color_prompt, operation=operation, image_context=image_context)
        negative_prompt += f", wrong color, unnatural color, {target_object} blurry, {target_object} distorted" # Specific color change negative

        gen_params = self._get_generation_params(op_type)
        final_params = {
            'image': pil_image, 'mask_image': mask_pil, 'prompt': enhanced_prompt, 'negative_prompt': negative_prompt,
            'strength': min(0.75, strength), # Păstrăm detalii
            'num_inference_steps': num_inference_steps if num_inference_steps is not None else gen_params['num_inference_steps'],
            'guidance_scale': guidance_scale if guidance_scale is not None else gen_params['guidance_scale'],
            'controlnet_conditioning_scale': None, # Dezactivat
            'use_refiner': use_refiner_if_available, 'refiner_strength': refiner_strength
        }
        final_params = {k: v for k, v in final_params.items() if v is not None}

        # 5. Apel model principal
        self._update_progress(0.7, desc=f"Schimbare culoare {target_object}...")
        main_model = self.model_manager.get_model('main')
        if not main_model or not getattr(main_model, 'is_loaded', False):
            msg = "Main model not available."; logger.error(msg)
            return {'result_image': pil_image, 'mask_image': mask_pil, 'operation': operation, 'message': msg, 'success': False}

        try:
            output_dict = main_model.process(**final_params)

            if output_dict.get('success'):
                result_pil = self._convert_to_pil(output_dict.get('result'))
                # TODO: Post-procesare (Face Enhance pt păr?)
                self._update_progress(1.0, desc="Procesare completă!")
                total_time = time.time() - start_time
                return {
                    'result_image': result_pil, 'mask_image': mask_pil, 'operation': operation,
                    'message': output_dict.get('message', f"Color changed to {target_color} ({total_time:.2f}s)."),
                    'success': True
                }
            else:
                 msg = f"Color change failed: {output_dict.get('message')}"; logger.error(msg)
                 return {'result_image': pil_image, 'mask_image': mask_pil, 'operation': operation, 'message': msg, 'success': False}
        except Exception as e:
            msg = f"Runtime error during color change pipeline: {e}"; logger.error(msg, exc_info=True)
            return {'result_image': pil_image, 'mask_image': mask_pil, 'operation': operation, 'message': msg, 'success': False}