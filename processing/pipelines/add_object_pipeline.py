#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Pipeline pentru adăugarea obiectelor în FusionFrame 2.0
(Actualizat cu integrare completă PostProcessor și PromptEnhancer)
"""

import logging
import cv2
import numpy as np
from typing import Dict, Any, Union, Callable, Optional
from PIL import Image
import time

from processing.pipelines.base_pipeline import BasePipeline
from processing.analyzer import OperationAnalyzer
from processing.post_processor import PostProcessor
from processing.prompt_enhancer import PromptEnhancer

logger = logging.getLogger(__name__)

class AddObjectPipeline(BasePipeline):
    """Pipeline specializat pentru adăugarea obiectelor."""
    def __init__(self):
        super().__init__()
        self.operation_analyzer = OperationAnalyzer()
        # Inițializăm direct PostProcessor și PromptEnhancer
        self.post_processor = PostProcessor()
        self.prompt_enhancer = PromptEnhancer()

    def process(self,
               image: Union[Image.Image, np.ndarray],
               prompt: str,
               strength: float = 0.75,
               progress_callback: Callable = None,
               operation: Optional[Dict[str, Any]] = None,
               image_context: Optional[Dict[str, Any]] = None,
               num_inference_steps: Optional[int] = None,
               guidance_scale: Optional[float] = None,
               use_controlnet_if_available: bool = True,
               use_refiner_if_available: Optional[bool] = None,
               refiner_strength: Optional[float] = None,
               enhance_details: bool = True,  # Default true pentru adăugare obiecte
               fix_faces: bool = False,
               remove_artifacts: bool = False,
               **kwargs) -> Dict[str, Any]:

        self.progress_callback = progress_callback
        start_time = time.time()

        # 1. Analiză operație și input
        if operation is None: 
            operation = self.operation_analyzer.analyze_operation(prompt)
        
        op_type = operation.get('type', 'add')
        target_attribute = operation.get('attribute', '').lower()
        if not target_attribute:
            msg = "No object specified to add."
            logger.error(msg)
            return {'result_image': None, 'mask_image': None, 'operation': operation, 'message': msg, 'success': False}

        try:
            pil_image = self._convert_to_pil(image)
            image_np = self._convert_to_cv2(pil_image)
        except (TypeError, ValueError) as e:
             msg = f"Input image conversion error: {e}"
             logger.error(msg)
             return {'result_image': None, 'mask_image': None, 'operation': operation, 'message': msg, 'success': False}

        # 2. Analiză imagine
        if image_context is None:
            self._update_progress(0.1, desc="Analiză imagine...")
            image_context = self.image_analyzer.analyze_image_context(pil_image)
            if "error" in image_context:
                 logger.error(f"Image analysis failed: {image_context['error']}")
                 image_context = {}
        else:
            self._update_progress(0.1, desc="Context imagine primit.")

        # 3. Generare mască pentru locația obiectului
        self._update_progress(0.2, desc="Generare mască adăugare...")
        
        # Logica specifică pentru diferite tipuri de obiecte adăugate
        if "glasses" in target_attribute or "ochelari" in target_attribute:
            # Generare mască pentru ochelari (folosind logica specializată)
            self._update_progress(0.2, desc="Generare mască ochelari...")
            mask_np = self._generate_glasses_mask_fallback(image_np)
        else:
            # Generare mască pentru obiect generic
            self._update_progress(0.2, desc="Generare mască obiect generic...")
            mask_np = self._generate_generic_object_mask_fallback(image_np, target_attribute)

        mask_pil = self._ensure_pil_mask(mask_np) if mask_np is not None else None
        if mask_pil is None:
             msg = "Mask generation failed for add object."
             logger.error(msg)
             return {'success': False, 'message': msg, 'result_image': pil_image}

        # 4. Îmbunătățire Prompt și Parametri
        self._update_progress(0.6, desc="Pregătire prompt adăugare...")
        
        # Folosim direct PromptEnhancer
        enhanced_prompt = self.prompt_enhancer.enhance_prompt(
            prompt=prompt, 
            operation_type=op_type, 
            image_context=image_context
        )
        enhanced_prompt += f", realistic {target_attribute}, seamless integration, proper lighting"

        negative_prompt = self.prompt_enhancer.generate_negative_prompt(
            prompt=prompt, 
            operation_type=op_type, 
            image_context=image_context
        )
        negative_prompt += f", floating {target_attribute}, unrealistic {target_attribute}, misplaced, bad proportions, distorted"

        gen_params = self._get_generation_params(op_type)
        final_params = {
            'image': pil_image, 
            'mask_image': mask_pil, 
            'prompt': enhanced_prompt, 
            'negative_prompt': negative_prompt,
            'strength': min(strength, 0.85),  # Limităm strength pentru 'add' pentru a nu afecta prea mult restul
            'num_inference_steps': num_inference_steps if num_inference_steps is not None else gen_params['num_inference_steps'],
            'guidance_scale': guidance_scale if guidance_scale is not None else gen_params['guidance_scale'],
            'controlnet_conditioning_scale': gen_params['controlnet_conditioning_scale'] if use_controlnet_if_available else None,
            'use_refiner': use_refiner_if_available, 
            'refiner_strength': refiner_strength
        }
        final_params = {k: v for k, v in final_params.items() if v is not None}

        # 5. Apel model principal cu tratare OOM
        self._update_progress(0.7, desc=f"Adăugare {target_attribute}...")
        main_model = self.model_manager.get_model('main')
        if not main_model or not getattr(main_model, 'is_loaded', False):
            msg = "Main model not available."
            logger.error(msg)
            return {'result_image': pil_image, 'mask_image': mask_pil, 'operation': operation, 'message': msg, 'success': False}

        try:
            # Folosim metoda _safe_model_process pentru tratarea OOM
            output_dict = self._safe_model_process(main_model, final_params)

            if output_dict.get('success'):
                # Obținem imaginea rezultat
                result_img = output_dict.get('result')
                
                # 6. Aplicăm Post-Procesarea (important pentru adăugare obiecte)
                # Setăm fix_faces=True explicit în cazul ochelarilor
                should_fix_faces = fix_faces or ("glasses" in target_attribute) or ("ochelari" in target_attribute)
                
                if enhance_details or should_fix_faces or remove_artifacts:
                    self._update_progress(0.85, desc="Aplicare post-procesare...")
                    try:
                        post_result = self.post_processor.process(
                            image=result_img,
                            original_image=pil_image,
                            mask=mask_pil,
                            operation_type=op_type,
                            enhance_details=enhance_details,
                            fix_faces=should_fix_faces,
                            remove_artifacts=remove_artifacts,
                            seamless_blending=True,
                            color_harmonization=True,
                            progress_callback=lambda p, desc=None: self._update_progress(0.85 + p * 0.15, desc=desc)
                        )
                        
                        if post_result.get('success'):
                            result_pil = post_result.get('result_image')
                            logger.info(f"Post-procesare aplicată cu succes: {post_result.get('message')}")
                        else:
                            result_pil = self._convert_to_pil(result_img)
                            logger.warning(f"Post-procesarea a eșuat: {post_result.get('message')}")
                    except Exception as e_post:
                        result_pil = self._convert_to_pil(result_img)
                        logger.error(f"Eroare în timpul post-procesării: {e_post}", exc_info=True)
                else:
                    result_pil = self._convert_to_pil(result_img)
                
                # Finalizare și returnare
                self._update_progress(1.0, desc="Procesare completă!")
                total_time = time.time() - start_time
                return {
                    'result_image': result_pil, 
                    'mask_image': mask_pil, 
                    'operation': operation,
                    'message': output_dict.get('message', f"{target_attribute.capitalize()} added successfully ({total_time:.2f}s)."),
                    'success': True
                }
            else:
                 msg = f"Add object failed: {output_dict.get('message')}"
                 logger.error(msg)
                 return {
                    'result_image': pil_image, 
                    'mask_image': mask_pil, 
                    'operation': operation,
                    'message': msg, 
                    'success': False
                 }
        except Exception as e:
            msg = f"Runtime error during add object pipeline: {e}"
            logger.error(msg, exc_info=True)
            return {
                'result_image': pil_image, 
                'mask_image': mask_pil, 
                'operation': operation,
                'message': msg, 
                'success': False
            }
    
    # --- Metode Fallback pentru Măști (păstrate din versiunea originală) ---
    def _generate_glasses_mask_fallback(self, image_np):
        """Generează mască pentru adăugarea ochelarilor."""
        face_detector = self._get_face_detector()  # Folosim lazy loader din BasePipeline
        if face_detector is None: 
            return None
            
        h, w = image_np.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        try:
            results = face_detector.process(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))
            if hasattr(results, 'detections') and results.detections:
                for detection in results.detections:
                    bbox = detection.location_data.relative_bounding_box
                    if bbox:
                         x, y = int(bbox.xmin * w), int(bbox.ymin * h)
                         width, height = int(bbox.width * w), int(bbox.height * h)
                         eye_y = y + int(height * 0.3)
                         eye_h = int(height * 0.25)
                         cv2.rectangle(mask, (x, eye_y), (x + width, eye_y + eye_h), 255, -1)
                return mask
        except Exception as e:
            logger.error(f"Fallback glasses mask error: {e}")
            
        # Fallback mask dacă detecția eșuează
        c_y, ch = h // 3, h // 6
        cv2.rectangle(mask, (w // 4, c_y), (3 * w // 4, c_y + ch), 255, -1)
        return mask

    def _generate_generic_object_mask_fallback(self, image_np, obj_name):
        """Generează mască pentru adăugarea obiectelor generice."""
        h, w = image_np.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # Reguli specifice pentru anumite tipuri de obiecte
        if obj_name in ['hat', 'cap', 'palarie', 'pălărie']:
            tm = h // 6
            cv2.rectangle(mask, (w//4, 0), (3*w//4, tm+h//8), 255, -1)
        elif obj_name in ['necklace', 'pendant', 'colier', 'lanț']:
            ny = h // 3
            cv2.rectangle(mask, (w//3, ny), (2*w//3, ny+h//8), 255, -1)
        elif obj_name in ['earrings', 'earring', 'cercei']:
            ey = h // 3
            # Două zone pentru cercei (stânga și dreapta)
            cv2.rectangle(mask, (w//5, ey), (w//5 + w//16, ey+h//8), 255, -1)
            cv2.rectangle(mask, (4*w//5 - w//16, ey), (4*w//5, ey+h//8), 255, -1)
        elif obj_name in ['tattoo', 'tatuaj']:
            # Pentru tatuaje - braț (dreapta)
            cv2.rectangle(mask, (4*w//5, h//3), (w, 2*h//3), 255, -1)
        else:
            # Pentru obiecte generice - centrul imaginii
            cx, cy = w//2, h//2
            cv2.rectangle(mask, (cx-w//4, cy-h//4), (cx+w//4, cy+h//4), 255, -1)
            
        return mask
    
    def _get_face_detector(self):
        """Returnează detectorul de fețe (MediaPipe)."""
        try:
            detector = self.model_manager.get_model('face_detector')
            if isinstance(detector, dict) and 'model' in detector:
                detector = detector['model']
            return detector
        except Exception as e:
            logger.error(f"Error getting face detector: {e}")
            return None
    
    def _safe_model_process(self, model, final_params) -> Dict[str, Any]:
        """Procesează modelul cu tratarea erorilor OOM."""
        try:
            return model.process(**final_params)
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                logger.warning("CUDA out of memory. Încercare de recuperare...")
                self.model_manager.emergency_memory_recovery()
                
                # Reducem parametrii pentru reîncercare
                steps_before = final_params.get('num_inference_steps', 50)
                guidance_before = final_params.get('guidance_scale', 7.5)
                
                final_params['num_inference_steps'] = max(20, steps_before // 2)
                final_params['guidance_scale'] = min(7.0, guidance_before)
                
                logger.info(f"Reîncerc cu parametri reduși: steps {steps_before} -> {final_params['num_inference_steps']}, "
                          f"guidance {guidance_before} -> {final_params['guidance_scale']}")
                
                try:
                    return model.process(**final_params)
                except Exception as retry_e:
                    logger.error(f"Recuperarea din OOM a eșuat: {retry_e}")
                    raise RuntimeError(f"Procesare eșuată după recuperare din OOM: {retry_e}")
            else:
                raise