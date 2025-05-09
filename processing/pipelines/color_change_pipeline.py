#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Pipeline pentru schimbarea culorilor în FusionFrame 2.0
(Actualizat cu integrare completă PostProcessor și PromptEnhancer)
"""

import logging
import cv2
import numpy as np
from typing import Dict, Any, Optional, Union, Tuple, Callable
from PIL import Image
import time

from processing.pipelines.base_pipeline import BasePipeline
from processing.analyzer import OperationAnalyzer
from processing.post_processor import PostProcessor
from processing.prompt_enhancer import PromptEnhancer

logger = logging.getLogger(__name__)

class ColorChangePipeline(BasePipeline):
    """Pipeline specializat pentru schimbarea culorilor."""
    def __init__(self):
        super().__init__()
        self.operation_analyzer = OperationAnalyzer()
        # Inițializăm direct PostProcessor și PromptEnhancer
        self.post_processor = PostProcessor()
        self.prompt_enhancer = PromptEnhancer()

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
               enhance_details: bool = False,  # Parametri expliciti pentru post-procesare
               fix_faces: bool = False,
               remove_artifacts: bool = False,
               **kwargs) -> Dict[str, Any]:

        self.progress_callback = progress_callback
        start_time = time.time()

        # 1. Analiză operație și input
        if operation is None: 
            operation = self.operation_analyzer.analyze_operation(prompt)
        
        op_type = operation.get('type', 'color')
        target_object = operation.get('target_object', 'object')
        target_color = operation.get('attribute', 'specified color')

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

        # 3. Generare mască
        self._update_progress(0.2, desc=f"Generare mască {target_object}...")
        mask_result = self.mask_generator.generate_mask(
            image=image_np, 
            prompt=f"{target_object}", 
            operation=operation,
            progress_callback=lambda p, desc=None: self._update_progress(0.2 + p * 0.4, desc=desc)
        )
        if not mask_result.get('success'):
            msg = f"Mask generation failed for '{target_object}': {mask_result.get('message')}"
            logger.error(msg)
            return {'result_image': pil_image, 'mask_image': None, 'operation': operation, 'message': msg, 'success': False}

        mask_np = mask_result.get('mask')
        mask_pil = self._ensure_pil_mask(mask_np)
        if mask_pil is None:
            msg = f"Mask is None after generation for '{target_object}'."
            logger.error(msg)
            return {'result_image': pil_image, 'mask_image': None, 'operation': operation, 'message': msg, 'success': False}

        # 4. Îmbunătățire Prompt și Parametri
        self._update_progress(0.6, desc="Pregătire prompt culoare...")
        
        # Creăm promptul specific pentru schimbarea culorii
        color_prompt = f"{target_color} {target_object}"
        
        # Folosim direct PromptEnhancer
        enhanced_prompt = self.prompt_enhancer.enhance_prompt(
            prompt=color_prompt, 
            operation_type=op_type, 
            image_context=image_context
        )
        enhanced_prompt += ", realistic color, natural texture, correct lighting"  # Specific color change

        negative_prompt = self.prompt_enhancer.generate_negative_prompt(
            prompt=color_prompt, 
            operation_type=op_type, 
            image_context=image_context
        )
        negative_prompt += f", wrong color, unnatural color, {target_object} blurry, {target_object} distorted"  # Specific color change negative

        gen_params = self._get_generation_params(op_type)
        final_params = {
            'image': pil_image, 
            'mask_image': mask_pil, 
            'prompt': enhanced_prompt, 
            'negative_prompt': negative_prompt,
            'strength': min(0.75, strength),  # Păstrăm detalii
            'num_inference_steps': num_inference_steps if num_inference_steps is not None else gen_params['num_inference_steps'],
            'guidance_scale': guidance_scale if guidance_scale is not None else gen_params['guidance_scale'],
            'controlnet_conditioning_scale': None,  # Dezactivat explicit pentru color change
            'use_refiner': use_refiner_if_available, 
            'refiner_strength': refiner_strength
        }
        final_params = {k: v for k, v in final_params.items() if v is not None}

        # 5. Apel model principal cu tratare OOM
        self._update_progress(0.7, desc=f"Schimbare culoare {target_object}...")
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
                
                # 6. Aplicăm Post-Procesarea dacă este cerută
                if enhance_details or fix_faces or remove_artifacts:
                    self._update_progress(0.85, desc="Aplicare post-procesare...")
                    try:
                        post_result = self.post_processor.process(
                            image=result_img,
                            original_image=pil_image,
                            mask=mask_pil,
                            operation_type=op_type,
                            enhance_details=enhance_details,
                            fix_faces=fix_faces,
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
                    'message': output_dict.get('message', f"Color changed to {target_color} ({total_time:.2f}s)."),
                    'success': True
                }
            else:
                 msg = f"Color change failed: {output_dict.get('message')}"
                 logger.error(msg)
                 return {'result_image': pil_image, 'mask_image': mask_pil, 'operation': operation, 'message': msg, 'success': False}
        except Exception as e:
            msg = f"Runtime error during color change pipeline: {e}"
            logger.error(msg, exc_info=True)
            return {'result_image': pil_image, 'mask_image': mask_pil, 'operation': operation, 'message': msg, 'success': False}
    
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