#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Pipeline pentru adăugarea obiectelor în FusionFrame 2.0
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

class AddObjectPipeline(BasePipeline):
    """
    Pipeline specializat pentru adăugarea obiectelor
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
        self.progress_callback = progress_callback
        operation = self.operation_analyzer.analyze_operation(prompt)
        if "glasses" in prompt.lower() or "ochelari" in prompt.lower():
            return self.add_glasses(image, operation, strength, use_refiner=use_refiner, refiner_strength=refiner_strength, **kwargs)
        return self.add_generic_object(image, operation, strength, use_refiner=use_refiner, refiner_strength=refiner_strength, **kwargs)

    def add_glasses(self,
                    image: Union[Image.Image, np.ndarray],
                    operation: Dict[str, Any],
                    strength: float = 0.75,
                    use_refiner: bool = None,
                    refiner_strength: float = None,
                    **kwargs) -> Dict[str, Any]:
        # Convertim la format potrivit
        if isinstance(image, np.ndarray):
            image_np = image
            pil_image = Image.fromarray(image)
        else:
            image_np = np.array(image)
            pil_image = image

        self._update_progress(0.1, desc="Analiză imagine...")
        image_context = self.image_analyzer.analyze_image_context(image_np)

        self._update_progress(0.2, desc="Detectare față și ochi...")
        face_detector = self.model_manager.get_model('face_detector')
        if face_detector is None:
            logger.error("Face detector not available")
            return {'result': pil_image, 'mask': None, 'operation': operation, 'message': "Detectorul de față nu este disponibil"}

        h, w = image_np.shape[:2]
        face_mask = np.zeros((h, w), dtype=np.uint8)
        rgb_image = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        results = face_detector.process(rgb_image)

        face_detected = False
        if hasattr(results, 'detections') and results.detections:
            face_detected = True
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                x, y = int(bbox.xmin * w), int(bbox.ymin * h)
                width, height = int(bbox.width * w), int(bbox.height * h)
                eye_y = y + int(height * 0.3);
                eye_h = int(height * 0.25)
                cv2.rectangle(face_mask, (x, eye_y), (x + width, eye_y + eye_h), 255, -1)
        if not face_detected:
            logger.warning("No face detected, using fallback mask")
            face_mask = np.zeros((h, w), dtype=np.uint8)
            c_y, ch = h // 3, h // 6
            cv2.rectangle(face_mask, (w // 4, c_y), (3 * w // 4, c_y + ch), 255, -1)

        self._update_progress(0.5, desc="Generare ochelari...")
        main_model = self.model_manager.get_model('main')
        if main_model is None:
            logger.error("Main model not available")
            return {'result': pil_image, 'mask': Image.fromarray(face_mask), 'operation': operation, 'message': "Modelul principal nu este disponibil"}

        glasses_type = operation.get('attribute', '')
        if glasses_type:
            prompt_text = f"face with {glasses_type} glasses, realistic glasses, high quality eyewear"
        else:
            prompt_text = "face with stylish glasses, realistic glasses, detailed eyewear"

        params = self._get_generation_params('add')
        params['strength'] = min(0.6, strength)

        self._update_progress(0.7, desc="Aplicare ochelari...")
        try:
            result = main_model.process(
                image=pil_image,
                mask_image=Image.fromarray(face_mask),
                prompt=prompt_text,
                negative_prompt="unrealistic, deformed, distorted, blurry, bad quality",
                strength=params['strength'],
                num_inference_steps=params['num_inference_steps'],
                guidance_scale=params['guidance_scale'],
                controlnet_conditioning_scale=params.get('controlnet_conditioning_scale'),
                use_refiner=use_refiner,
                refiner_strength=refiner_strength
            )
            if result['success']:
                try:
                    self._update_progress(0.9, desc="Îmbunătățire față...")
                    gpen = self.model_manager.get_specialized_model('gpen')
                    if gpen:
                        er = gpen.process(np.array(result['result']))
                        if er['success']:
                            result['result'] = er['result']
                except Exception:
                    pass
                self._update_progress(1.0, desc="Procesare completă!")
            return {'result': result['result'], 'mask': Image.fromarray(face_mask), 'operation': operation, 'message': result.get('message', '')}
        except Exception as e:
            logger.error(f"Error in adding glasses: {e}")
            return {'result': pil_image, 'mask': Image.fromarray(face_mask), 'operation': operation, 'message': f"Eroare: {e}"}

    def add_generic_object(self,
                           image: Union[Image.Image, np.ndarray],
                           operation: Dict[str, Any],
                           strength: float = 0.75,
                           use_refiner: bool = None,
                           refiner_strength: float = None,
                           **kwargs) -> Dict[str, Any]:
        if isinstance(image, np.ndarray):
            image_np = image
            pil_image = Image.fromarray(image)
        else:
            image_np = np.array(image)
            pil_image = image

        self._update_progress(0.1, desc="Analiză imagine...")
        image_context = self.image_analyzer.analyze_image_context(image_np)

        self._update_progress(0.2, desc="Pregătire regiune...")
        h, w = image_np.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        obj = operation.get('attribute', '') or kwargs.get('object_type', 'generic object')
        if obj.lower() in ['hat', 'cap', 'palarie', 'pălărie']:
            tm = h // 6
            cv2.rectangle(mask, (w//4, 0), (3*w//4, tm+h//8), 255, -1)
        elif obj.lower() in ['necklace', 'pendant', 'colier', 'lanț']:
            ny = h // 3
            cv2.rectangle(mask, (w//3, ny), (2*w//3, ny+h//8), 255, -1)
        else:
            cx, cy = w//2, h//2
            cv2.rectangle(mask, (cx-w//4, cy-h//4), (cx+w//4, cy+h//4), 255, -1)

        self._update_progress(0.5, desc=f"Generare {obj}...")
        main_model = self.model_manager.get_model('main')
        if not main_model:
            return {'result': pil_image, 'mask': Image.fromarray(mask), 'operation': operation, 'message': "Modelul principal nu este disponibil"}

        scene = image_context.get('scene_type', 'scene')
        style = image_context.get('style', 'natural')
        prompt_text = f"photo with {obj}, realistic {obj}, high quality, {style} style, {scene}"

        params = self._get_generation_params('add')
        self._update_progress(0.7, desc=f"Aplicare {obj}...")
        try:
            result = main_model.process(
                image=pil_image,
                mask_image=Image.fromarray(mask),
                prompt=prompt_text,
                negative_prompt="unrealistic, deformed, distorted, blurry, bad quality",
                strength=strength,
                num_inference_steps=params['num_inference_steps'],
                guidance_scale=params['guidance_scale'],
                controlnet_conditioning_scale=params.get('controlnet_conditioning_scale'),
                use_refiner=use_refiner,
                refiner_strength=refiner_strength
            )
            if result['success']:
                self._update_progress(1.0, desc="Procesare completă!")
            return {'result': result['result'], 'mask': Image.fromarray(mask), 'operation': operation, 'message': result.get('message', '')}
        except Exception as e:
            logger.error(f"Error in adding object: {e}")
            return {'result': pil_image, 'mask': Image.fromarray(mask), 'operation': operation, 'message': f"Eroare: {e}"}

    def remove_object(self,
                      image: Union[Image.Image, np.ndarray],
                      operation: Dict[str, Any],
                      strength: float = 0.75,
                      use_refiner: bool = None,
                      refiner_strength: float = None,
                      **kwargs) -> Dict[str, Any]:
        if isinstance(image, np.ndarray):
            image_np = image
            pil_image = Image.fromarray(image)
        else:
            image_np = np.array(image)
            pil_image = image

        self._update_progress(0.1, desc="Analiză imagine...")
        image_context = self.image_analyzer.analyze_image_context(image_np)

        self._update_progress(0.2, desc="Generare mască persoană...")
        mask_res = self.mask_generator.generate_mask(
            image=image_np,
            prompt="person",
            operation=operation,
            progress_callback=lambda p, desc: self._update_progress(0.2 + p*0.3, desc=desc)
        )
        if not mask_res['success']:
            return {'result': pil_image, 'mask': None, 'operation': operation, 'message': "Eroare la generarea măștii"}
        mask = mask_res['mask']

        self._update_progress(0.6, desc="Eliminare inițială...")
        main_model = self.model_manager.get_model('main')
        if not main_model:
            return {'result': pil_image, 'mask': Image.fromarray(mask), 'operation': operation, 'message': "Modelul principal nu este disponibil"}

        desc = image_context.get('description', '')
        prompt_text = f"empty scene without any person, clean background, {desc}"
        params = self._get_generation_params('remove')
        params['strength'] = max(0.95, strength)

        self._update_progress(0.7, desc="Generare finală...")
        try:
            result = main_model.process(
                image=pil_image,
                mask_image=Image.fromarray(mask),
                prompt=prompt_text,
                negative_prompt="person, human, face, body, distortion, artifact, blurry",
                strength=params['strength'],
                num_inference_steps=params['num_inference_steps'],
                guidance_scale=params['guidance_scale'],
                controlnet_conditioning_scale=params.get('controlnet_conditioning_scale'),
                use_refiner=use_refiner,
                refiner_strength=refiner_strength
            )
            if result['success']:
                self._update_progress(1.0, desc="Procesare completă!")
            return {'result': result['result'], 'mask': Image.fromarray(mask), 'operation': operation, 'message': result.get('message', '')}
        except Exception as e:
            logger.error(f"Error in person removal: {e}")
            return {'result': pil_image, 'mask': Image.fromarray(mask), 'operation': operation, 'message': f"Eroare: {e}"}