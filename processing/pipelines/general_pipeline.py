#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Pipeline general pentru procesare în FusionFrame 2.0
(Actualizat pentru a folosi PromptEnhancer din BasePipeline)
"""

import logging
import cv2
import numpy as np
from typing import Dict, Any, Union, Callable, Optional
from PIL import Image

from processing.pipelines.base_pipeline import BasePipeline
from processing.analyzer import OperationAnalyzer

logger = logging.getLogger(__name__)

class GeneralPipeline(BasePipeline):
    """Pipeline general pentru procesarea imaginilor folosind context."""
    def __init__(self):
        super().__init__()
        self.operation_analyzer = OperationAnalyzer()

    def process(self,
                image: Union[Image.Image, np.ndarray],
                prompt: str,
                strength: float = 0.75,
                progress_callback: Optional[Callable] = None,
                operation: Optional[Dict[str, Any]] = None,
                image_context: Optional[Dict[str, Any]] = None,
                num_inference_steps: Optional[int] = None,
                guidance_scale: Optional[float] = None,
                use_controlnet_if_available: bool = True,
                use_refiner_if_available: Optional[bool] = None,
                refiner_strength: Optional[float] = None,
                **kwargs # Prindem restul argumentelor din BasePipeline
                ) -> Dict[str, Any]:
        """Procesează imaginea folosind pipeline-ul general și contextul."""
        self.progress_callback = progress_callback
        start_time = time.time() # Import time needed

        # 1. Analiză operație și input
        if operation is None: operation = self.operation_analyzer.analyze_operation(prompt)
        op_type = operation.get('type', 'general')

        try:
            pil_image = self._convert_to_pil(image) # Standardizăm la PIL RGB
            image_np = self._convert_to_cv2(pil_image) # Avem și varianta BGR
        except (TypeError, ValueError) as e:
             msg = f"Input image conversion error: {e}"; logger.error(msg)
             return {'result_image': None, 'mask_image': None, 'operation': operation, 'message': msg, 'success': False}

        # 2. Analiză imagine (dacă nu e deja furnizată)
        if image_context is None:
            self._update_progress(0.1, desc="Analiză imagine...")
            image_context = self.image_analyzer.analyze_image_context(pil_image)
            if "error" in image_context:
                 logger.error(f"Image analysis failed: {image_context['error']}"); image_context = {}
        else:
            self._update_progress(0.1, desc="Context imagine primit.")

        # 3. Generare mască
        self._update_progress(0.2, desc="Generare mască...")
        mask_result = self.mask_generator.generate_mask(
            image=image_np, prompt=prompt, operation=operation,
            progress_callback=lambda p, desc=None: self._update_progress(0.2 + p * 0.4, desc=desc)
        )
        if not mask_result.get('success'):
            msg = f"Mask generation failed: {mask_result.get('message')}"
            logger.error(msg); return {'result_image': pil_image, 'mask_image': None, 'operation': operation, 'message': msg, 'success': False}

        mask_np = mask_result.get('mask')
        mask_pil = self._ensure_pil_mask(mask_np) # Convertim/validăm masca la PIL 'L'
        if mask_pil is None: logger.warning("Mask is None after generation.")

        # 4. Îmbunătățire Prompt și Parametri
        self._update_progress(0.6, desc="Pregătire prompt...")
        enhanced_prompt = self._enhance_prompt(prompt, operation=operation, image_context=image_context)
        negative_prompt = self._get_negative_prompt(prompt, operation=operation, image_context=image_context)

        gen_params = self._get_generation_params(op_type)
        final_params = {
            'image': pil_image, 'mask_image': mask_pil, 'prompt': enhanced_prompt, 'negative_prompt': negative_prompt,
            'strength': strength,
            'num_inference_steps': num_inference_steps if num_inference_steps is not None else gen_params['num_inference_steps'],
            'guidance_scale': guidance_scale if guidance_scale is not None else gen_params['guidance_scale'],
            'controlnet_conditioning_scale': gen_params['controlnet_conditioning_scale'] if use_controlnet_if_available else None,
            'use_refiner': use_refiner_if_available, 'refiner_strength': refiner_strength
        }
        final_params = {k: v for k, v in final_params.items() if v is not None} # Curățăm None

        # 5. Apel model principal
        self._update_progress(0.7, desc="Generare imagine...")
        main_model = self.model_manager.get_model('main')
        if not main_model or not getattr(main_model, 'is_loaded', False):
            msg = "Main model not available or not loaded."; logger.error(msg)
            return {'result_image': pil_image, 'mask_image': mask_pil, 'operation': operation, 'message': msg, 'success': False}

        try:
            output_dict = main_model.process(**final_params) # Apelăm modelul cu kwargs

            if output_dict.get('success'):
                result_pil = self._convert_to_pil(output_dict.get('result'))
                # TODO: Adăugare post-procesare dacă enhance_details etc. sunt True
                self._update_progress(1.0, desc="Procesare completă!")
                total_time = time.time() - start_time
                return {
                    'result_image': result_pil, 'mask_image': mask_pil, 'operation': operation,
                    'message': output_dict.get('message', f"General processing successful ({total_time:.2f}s)."),
                    'success': True
                }
            else:
                 msg = f"Processing failed: {output_dict.get('message')}"; logger.error(msg)
                 return {
                    'result_image': pil_image, 'mask_image': mask_pil, 'operation': operation,
                    'message': msg, 'success': False
                 }
        except Exception as e:
            msg = f"Runtime error during general pipeline: {e}"; logger.error(msg, exc_info=True)
            return {
                'result_image': pil_image, 'mask_image': mask_pil, 'operation': operation,
                'message': msg, 'success': False
            }