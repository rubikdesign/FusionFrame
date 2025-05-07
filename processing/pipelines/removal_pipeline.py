#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Pipeline pentru eliminarea obiectelor/persoanelor în FusionFrame 2.0
(Actualizat pentru a folosi PromptEnhancer din BasePipeline)
"""

import logging
import cv2
import numpy as np
from typing import Dict, Any, Optional, Union, Tuple, Callable
from PIL import Image
import re # Importăm re
import time # Adăugat

from processing.pipelines.base_pipeline import BasePipeline
from processing.analyzer import OperationAnalyzer

logger = logging.getLogger(__name__)

class RemovalPipeline(BasePipeline):
    """Pipeline specializat pentru eliminarea obiectelor sau persoanelor."""
    def __init__(self):
        super().__init__()
        self.operation_analyzer = OperationAnalyzer()

    def process(self,
               image: Union[Image.Image, np.ndarray],
               prompt: str, # Promptul original, ex: "remove the red car"
               strength: float = 0.9, # Default mai mare pentru remove
               progress_callback: Optional[Callable] = None,
               operation: Optional[Dict[str, Any]] = None,
               image_context: Optional[Dict[str, Any]] = None,
               num_inference_steps: Optional[int] = None,
               guidance_scale: Optional[float] = None,
               use_controlnet_if_available: bool = True, # ControlNet poate ajuta la inpainting
               use_refiner_if_available: Optional[bool] = None,
               refiner_strength: Optional[float] = None,
               **kwargs) -> Dict[str, Any]:

        self.progress_callback = progress_callback
        start_time = time.time()

        # 1. Analiză operație și input
        if operation is None: operation = self.operation_analyzer.analyze_operation(prompt)
        op_type = operation.get('type', 'remove')
        target_object = operation.get('target_object', 'object to remove')

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

        # 3. Generare mască pentru obiectul de eliminat
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

        # 4. Îmbunătățire Prompt și Parametri pentru Inpainting
        self._update_progress(0.6, desc="Pregătire prompt inpainting...")
        # Cream prompt pozitiv care descrie scena FĂRĂ obiect
        inpainting_prompt_base = image_context.get('full_description_heuristic', 'background scene')
        # Eliminăm mențiunea obiectului (simplu)
        inpainting_prompt_base = re.sub(rf'\b{re.escape(target_object)}\b', '', inpainting_prompt_base, flags=re.IGNORECASE).strip()
        inpainting_prompt_base = re.sub(r'\s{2,}', ' ', inpainting_prompt_base).replace("featuring ,", "featuring").strip(', ')
        if not inpainting_prompt_base or len(inpainting_prompt_base) < 10: inpainting_prompt_base = "empty area, clean background"

        enhanced_prompt = self._enhance_prompt(inpainting_prompt_base, operation=operation, image_context=image_context)
        enhanced_prompt += ", seamless texture, realistic inpainting" # Specific inpainting

        negative_prompt = self._get_negative_prompt(prompt, operation=operation, image_context=image_context)
        negative_prompt += f", {target_object}, visible {target_object}, silhouette of {target_object}, remaining parts, ghosting, blurry patch" # Specific inpainting

        gen_params = self._get_generation_params(op_type)
        final_params = {
            'image': pil_image, 'mask_image': mask_pil, 'prompt': enhanced_prompt, 'negative_prompt': negative_prompt,
            'strength': max(0.85, strength), # Strength mare pentru inpainting
            'num_inference_steps': num_inference_steps if num_inference_steps is not None else gen_params['num_inference_steps'],
            'guidance_scale': guidance_scale if guidance_scale is not None else gen_params['guidance_scale'],
            'controlnet_conditioning_scale': gen_params['controlnet_conditioning_scale'] if use_controlnet_if_available else None, # ControlNet poate ajuta
            'use_refiner': use_refiner_if_available, 'refiner_strength': refiner_strength
        }
        final_params = {k: v for k, v in final_params.items() if v is not None}

        # 5. Apel model principal (Inpainting)
        self._update_progress(0.7, desc=f"Eliminare {target_object}...")
        main_model = self.model_manager.get_model('main')
        if not main_model or not getattr(main_model, 'is_loaded', False):
            msg = "Main model not available."; logger.error(msg)
            return {'result_image': pil_image, 'mask_image': mask_pil, 'operation': operation, 'message': msg, 'success': False}

        try:
            output_dict = main_model.process(**final_params) # Procesul de inpainting

            if output_dict.get('success'):
                result_pil = self._convert_to_pil(output_dict.get('result'))
                # TODO: Post-procesare pentru netezire?
                self._update_progress(1.0, desc="Procesare completă!")
                total_time = time.time() - start_time
                return {
                    'result_image': result_pil, 'mask_image': mask_pil, 'operation': operation,
                    'message': output_dict.get('message', f"{target_object.capitalize()} removed ({total_time:.2f}s)."),
                    'success': True
                }
            else:
                 msg = f"Removal failed: {output_dict.get('message')}"; logger.error(msg)
                 return {'result_image': pil_image, 'mask_image': mask_pil, 'operation': operation, 'message': msg, 'success': False}
        except Exception as e:
            msg = f"Runtime error during removal pipeline: {e}"; logger.error(msg, exc_info=True)
            return {'result_image': pil_image, 'mask_image': mask_pil, 'operation': operation, 'message': msg, 'success': False}