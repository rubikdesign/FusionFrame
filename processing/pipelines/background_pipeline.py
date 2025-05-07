#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Pipeline pentru înlocuirea fundalurilor în FusionFrame 2.0
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

class BackgroundPipeline(BasePipeline):
    """Pipeline specializat pentru înlocuirea fundalurilor."""
    def __init__(self):
        super().__init__()
        self.operation_analyzer = OperationAnalyzer()

    def process(self,
               image: Union[Image.Image, np.ndarray],
               prompt: str, # Descrierea noului fundal
               strength: float = 0.8, # Default mai mare pt background
               progress_callback: Optional[Callable] = None,
               operation: Optional[Dict[str, Any]] = None,
               image_context: Optional[Dict[str, Any]] = None,
               num_inference_steps: Optional[int] = None,
               guidance_scale: Optional[float] = None,
               use_controlnet_if_available: bool = False, # ControlNet mai puțin util pt schimbare completă bg
               use_refiner_if_available: Optional[bool] = None,
               refiner_strength: Optional[float] = None,
               **kwargs) -> Dict[str, Any]:

        self.progress_callback = progress_callback
        start_time = time.time()

        # 1. Analiză operație și input
        if operation is None: operation = self.operation_analyzer.analyze_operation(prompt)
        op_type = operation.get('type', 'background')
        background_desc = operation.get('attribute') or prompt # Folosim atributul sau promptul ca descriere

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

        # 3. Generare mască FUNDAL
        self._update_progress(0.2, desc="Generare mască fundal...")
        mask_result = self.mask_generator.generate_mask(
            image=image_np, prompt="background", operation={'type': 'background'},
            progress_callback=lambda p, desc=None: self._update_progress(0.2 + p * 0.4, desc=desc)
        )

        if not mask_result.get('success'): # Încercăm fallback
            logger.warning(f"Primary mask generation failed for background: {mask_result.get('message')}. Trying fallback.")
            subject_mask_np = self._get_subject_mask_fallback(image_np)
            if subject_mask_np is None:
                 msg = "Background mask generation failed (including fallback)."
                 logger.error(msg); return {'result_image': pil_image, 'mask_image': None, 'operation': operation, 'message': msg, 'success': False}
            background_mask_np = cv2.bitwise_not(subject_mask_np)
        else:
            background_mask_np = mask_result.get('mask')

        mask_pil = self._ensure_pil_mask(background_mask_np)
        if mask_pil is None:
             msg = "Background mask is None or invalid."; logger.error(msg)
             return {'result_image': pil_image, 'mask_image': None, 'operation': operation, 'message': msg, 'success': False}

        # 4. Îmbunătățire Prompt și Parametri
        self._update_progress(0.6, desc="Pregătire prompt fundal...")
        # Folosim enhancer-ul doar cu descrierea fundalului + contextul general
        enhanced_prompt = self._enhance_prompt(background_desc, operation=operation, image_context=image_context)
        enhanced_prompt += ", background scene, environmental context" # Adăugăm specificitate
        negative_prompt = self._get_negative_prompt(background_desc, operation=operation, image_context=image_context)
        negative_prompt += ", person, people, subject, foreground object, blurry foreground, watermark, text, signature" # Negativ specific

        gen_params = self._get_generation_params(op_type)
        final_params = {
            'image': pil_image, 'mask_image': mask_pil, 'prompt': enhanced_prompt, 'negative_prompt': negative_prompt,
            'strength': max(0.85, strength), # Strength mare
            'num_inference_steps': num_inference_steps if num_inference_steps is not None else gen_params['num_inference_steps'],
            'guidance_scale': guidance_scale if guidance_scale is not None else gen_params['guidance_scale'],
            'controlnet_conditioning_scale': None, # Dezactivat explicit
            'use_refiner': use_refiner_if_available, 'refiner_strength': refiner_strength
        }
        final_params = {k: v for k, v in final_params.items() if v is not None}

        # 5. Apel model principal
        self._update_progress(0.7, desc="Generare fundal...")
        main_model = self.model_manager.get_model('main')
        if not main_model or not getattr(main_model, 'is_loaded', False):
            msg = "Main model not available."; logger.error(msg)
            return {'result_image': pil_image, 'mask_image': mask_pil, 'operation': operation, 'message': msg, 'success': False}

        try:
            output_dict = main_model.process(**final_params)

            if output_dict.get('success'):
                 result_pil = self._convert_to_pil(output_dict.get('result'))
                 # TODO: Post-procesare blending margini
                 self._update_progress(1.0, desc="Procesare completă!")
                 total_time = time.time() - start_time
                 return {
                     'result_image': result_pil, 'mask_image': mask_pil, 'operation': operation,
                     'message': output_dict.get('message', f"Background replaced ({total_time:.2f}s)."),
                     'success': True
                 }
            else:
                 msg = f"Background replacement failed: {output_dict.get('message')}"; logger.error(msg)
                 return {'result_image': pil_image, 'mask_image': mask_pil, 'operation': operation, 'message': msg, 'success': False}
        except Exception as e:
            msg = f"Runtime error during background pipeline: {e}"; logger.error(msg, exc_info=True)
            return {'result_image': pil_image, 'mask_image': mask_pil, 'operation': operation, 'message': msg, 'success': False}


    # --- Metodă Fallback pentru Masca Subiectului ---
    def _get_subject_mask_fallback(self, image_np: np.ndarray) -> Optional[np.ndarray]:
        """Încearcă să obțină masca subiectului prin rembg sau mediapipe ca fallback."""
        subject_mask = None
        try: # Rembg
            rembg_session = self.model_manager.get_model('rembg')
            if rembg_session:
                # Rembg .predict direct pe NumPy BGR returnează masca alpha ca ultim canal? Verificăm.
                # Sau folosim .remove și extragem alpha? Mai sigur .remove
                out_rgba = rembg_session.remove(image_np) # Presupunem că returnează RGBA
                if out_rgba.shape[2] == 4:
                     subject_mask = out_rgba[:,:,3]; logger.info("Using Rembg fallback for subject mask.")
        except Exception as e_rembg: logger.warning(f"Rembg fallback failed: {e_rembg}")

        if subject_mask is None: # MediaPipe
            try:
                mediapipe_seg = self.model_manager.get_model('mediapipe')
                if mediapipe_seg:
                    results = mediapipe_seg.process(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))
                    if results.segmentation_mask is not None:
                        mask_float = results.segmentation_mask
                        subject_mask = (mask_float > 0.5).astype(np.uint8) * 255
                        logger.info("Using MediaPipe fallback for subject mask.")
            except Exception as e_mp: logger.warning(f"MediaPipe fallback failed: {e_mp}")

        if subject_mask is None: logger.error("All fallback mask methods failed for background."); return None

        # Post-procesare simplă
        try:
             kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
             subject_mask = cv2.morphologyEx(subject_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
             subject_mask = cv2.morphologyEx(subject_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        except Exception as e_morph: logger.warning(f"Morphology on fallback mask failed: {e_morph}")
        return subject_mask