#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Pipeline pentru adăugarea obiectelor în FusionFrame 2.0
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

class AddObjectPipeline(BasePipeline):
    """Pipeline specializat pentru adăugarea obiectelor."""
    def __init__(self):
        super().__init__()
        self.operation_analyzer = OperationAnalyzer()

    def process(self,
               image: Union[Image.Image, np.ndarray],
               prompt: str,
               strength: float = 0.75,
               progress_callback: Callable = None,
               operation: Optional[Dict[str, Any]] = None, # Argumente standard
               image_context: Optional[Dict[str, Any]] = None,
               num_inference_steps: Optional[int] = None,
               guidance_scale: Optional[float] = None,
               use_controlnet_if_available: bool = True,
               use_refiner_if_available: Optional[bool] = None,
               refiner_strength: Optional[float] = None,
               **kwargs) -> Dict[str, Any]:

        self.progress_callback = progress_callback

        # 1. Analiză operație și input
        if operation is None: operation = self.operation_analyzer.analyze_operation(prompt)
        op_type = operation.get('type', 'add') # Asumăm 'add' dacă nu e specificat altfel

        try:
            pil_image = self._convert_to_pil(image)
            image_np = self._convert_to_cv2(pil_image)
        except (TypeError, ValueError) as e:
             logger.error(f"Input image conversion error: {e}"); return {'success': False, 'message': str(e)}

        # 2. Analiză imagine (dacă nu e deja furnizată)
        if image_context is None:
            self._update_progress(0.1, desc="Analiză imagine...")
            image_context = self.image_analyzer.analyze_image_context(pil_image)
            if "error" in image_context: image_context = {} # Continuăm fără context
        else:
            self._update_progress(0.1, desc="Context imagine primit.")


        # 3. Logica specifică pentru adăugare (ex: generare mască specifică)
        # Aici păstrăm logica existentă pentru generarea măștii pentru ochelari/obiect general
        # dar generarea măștii ar putea fi îmbunătățită folosind MaskGenerator
        target_attribute = operation.get('attribute', '').lower()
        if "glasses" in target_attribute or "ochelari" in target_attribute:
             # Generare mască pentru ochelari (folosind logica veche sau MaskGenerator)
             self._update_progress(0.2, desc="Generare mască ochelari...")
             # TODO: Ideal ar fi să folosim MaskGenerator și pentru acest caz specific
             mask_np = self._generate_glasses_mask_fallback(image_np)
        else:
             # Generare mască pentru obiect generic (folosind logica veche sau MaskGenerator)
             self._update_progress(0.2, desc="Generare mască obiect generic...")
             # TODO: Ideal ar fi să folosim MaskGenerator
             mask_np = self._generate_generic_object_mask_fallback(image_np, target_attribute)

        mask_pil = Image.fromarray(mask_np) if mask_np is not None else None
        if mask_pil is None:
             logger.error("Mask generation failed for add object."); return {'success': False, 'message': "Mask generation failed."}


        # 4. Îmbunătățire Prompt și Parametri
        self._update_progress(0.6, desc="Pregătire prompt...")
        enhanced_prompt = self._enhance_prompt(prompt, operation=operation, image_context=image_context)
        negative_prompt = self._get_negative_prompt(prompt, operation=operation, image_context=image_context)

        gen_params = self._get_generation_params(op_type)
        final_params = {
            'image': pil_image, 'mask_image': mask_pil, 'prompt': enhanced_prompt, 'negative_prompt': negative_prompt,
            'strength': min(strength, 0.85), # Limităm strength pentru 'add' pentru a nu afecta prea mult restul
            'num_inference_steps': num_inference_steps if num_inference_steps is not None else gen_params['num_inference_steps'],
            'guidance_scale': guidance_scale if guidance_scale is not None else gen_params['guidance_scale'],
            'controlnet_conditioning_scale': gen_params['controlnet_conditioning_scale'] if use_controlnet_if_available else None,
            'use_refiner': use_refiner_if_available, 'refiner_strength': refiner_strength
        }
        final_params = {k: v for k, v in final_params.items() if v is not None}

        # 5. Apel model principal
        self._update_progress(0.7, desc=f"Adăugare {target_attribute or 'obiect'}...")
        main_model = self.model_manager.get_model('main')
        if not main_model or not main_model.is_loaded:
            logger.error("Main model not available."); return {'success': False, 'message': "Main model not loaded."}

        try:
            output_dict = main_model.process(**final_params)

            if output_dict.get('success'):
                result_pil = self._convert_to_pil(output_dict.get('result'))
                # TODO: Integrare PostProcessor (ex: Face enhancement pentru ochelari)
                self._update_progress(1.0, desc="Procesare completă!")
                return {
                    'result_image': result_pil, 'mask_image': mask_pil, 'operation': operation,
                    'message': output_dict.get('message', "Object added successfully."), 'success': True
                }
            else:
                 logger.error(f"Add object failed: {output_dict.get('message')}")
                 return {
                    'result_image': pil_image, 'mask_image': mask_pil, 'operation': operation,
                    'message': f"Processing failed: {output_dict.get('message')}", 'success': False
                 }
        except Exception as e:
            logger.error(f"Error during add object pipeline: {e}", exc_info=True)
            return {
                'result_image': pil_image, 'mask_image': mask_pil, 'operation': operation,
                'message': f"Runtime error: {e}", 'success': False
            }

    # --- Metode Fallback pentru Măști (păstrate din versiunea originală) ---
    # Acestea ar trebui înlocuite cu apeluri la MaskGenerator în viitor
    def _generate_glasses_mask_fallback(self, image_np):
        face_detector = self._get_face_detector() # Folosim lazy loader din BasePipeline
        if face_detector is None: return None
        h, w = image_np.shape[:2]; mask = np.zeros((h, w), dtype=np.uint8)
        try:
            results = face_detector.process(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))
            if hasattr(results, 'detections') and results.detections:
                for detection in results.detections:
                    bbox = detection.location_data.relative_bounding_box
                    if bbox:
                         x, y = int(bbox.xmin * w), int(bbox.ymin * h)
                         width, height = int(bbox.width * w), int(bbox.height * h)
                         eye_y = y + int(height * 0.3); eye_h = int(height * 0.25)
                         cv2.rectangle(mask, (x, eye_y), (x + width, eye_y + eye_h), 255, -1)
                return mask
        except Exception as e: logger.error(f"Fallback glasses mask error: {e}");
        # Fallback mask dacă detecția eșuează
        c_y, ch = h // 3, h // 6; cv2.rectangle(mask, (w // 4, c_y), (3 * w // 4, c_y + ch), 255, -1); return mask

    def _generate_generic_object_mask_fallback(self, image_np, obj_name):
         h, w = image_np.shape[:2]; mask = np.zeros((h, w), dtype=np.uint8)
         if obj_name in ['hat', 'cap', 'palarie', 'pălărie']: tm = h // 6; cv2.rectangle(mask, (w//4, 0), (3*w//4, tm+h//8), 255, -1)
         elif obj_name in ['necklace', 'pendant', 'colier', 'lanț']: ny = h // 3; cv2.rectangle(mask, (w//3, ny), (2*w//3, ny+h//8), 255, -1)
         else: cx, cy = w//2, h//2; cv2.rectangle(mask, (cx-w//4, cy-h//4), (cx+w//4, cy+h//4), 255, -1)
         return mask