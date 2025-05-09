#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Pipeline general pentru procesare în FusionFrame 2.0
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
from processing.prompt_enhancer import PromptEnhancer  # Importăm versiunea existentă

logger = logging.getLogger(__name__)

class GeneralPipeline(BasePipeline):
    """Pipeline general pentru procesarea imaginilor folosind context."""
    def __init__(self):
        super().__init__()
        self.operation_analyzer = OperationAnalyzer()
        # Inițializăm direct PostProcessor și PromptEnhancer
        self.post_processor = PostProcessor()
        self.prompt_enhancer = PromptEnhancer()

    def _ensure_dimensions_multiple_of_8(self, image_np_or_pil):
        """Asigură că dimensiunile imaginii sunt divizibile cu 8."""
        if isinstance(image_np_or_pil, Image.Image):
            width, height = image_np_or_pil.size
            new_width = (width // 8) * 8
            new_height = (height // 8) * 8
            
            # Dacă dimensiunile sunt deja divizibile cu 8, returnăm imaginea originală
            if width == new_width and height == new_height:
                return image_np_or_pil
            
            # Altfel, redimensionăm imaginea la dimensiunile corecte
            logger.info(f"Redimensionare imagine de la {width}x{height} la {new_width}x{new_height} pentru a asigura divizibilitatea cu 8")
            return image_np_or_pil.resize((new_width, new_height), Image.LANCZOS)
        
        elif isinstance(image_np_or_pil, np.ndarray):
            height, width = image_np_or_pil.shape[:2]
            new_height = (height // 8) * 8
            new_width = (width // 8) * 8
            
            # Dacă dimensiunile sunt deja divizibile cu 8, returnăm imaginea originală
            if width == new_width and height == new_height:
                return image_np_or_pil
            
            # Altfel, redimensionăm imaginea la dimensiunile corecte
            logger.info(f"Redimensionare numpy array de la {width}x{height} la {new_width}x{new_height} pentru a asigura divizibilitatea cu 8")
            return cv2.resize(image_np_or_pil, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
        
        else:
            raise ValueError(f"Tip de imagine nesuportat: {type(image_np_or_pil)}")

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
                enhance_details: bool = False,  # Parametri expliciti pentru post-procesare
                fix_faces: bool = False,
                remove_artifacts: bool = False,
                **kwargs  # Prindem restul argumentelor
                ) -> Dict[str, Any]:
        """Procesează imaginea folosind pipeline-ul general și contextul."""
        self.progress_callback = progress_callback
        start_time = time.time()

        # 1. Analiză operație și input
        if operation is None: 
            operation = self.operation_analyzer.analyze_operation(prompt)
        op_type = operation.get('type', 'general')

        try:
            pil_image = self._convert_to_pil(image)
            # Asigură că dimensiunile sunt divizibile cu 8
            pil_image = self._ensure_dimensions_multiple_of_8(pil_image)
            image_np = self._convert_to_cv2(pil_image)
        except (TypeError, ValueError) as e:
             msg = f"Input image conversion error: {e}"
             logger.error(msg)
             return {'result_image': None, 'mask_image': None, 'operation': operation, 'message': msg, 'success': False}

        # 2. Analiză imagine (dacă nu e deja furnizată)
        if image_context is None:
            self._update_progress(0.1, desc="Analiză imagine...")
            image_context = self.image_analyzer.analyze_image_context(pil_image)
            if "error" in image_context:
                 logger.error(f"Image analysis failed: {image_context['error']}")
                 image_context = {}
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
            logger.error(msg)
            return {'result_image': pil_image, 'mask_image': None, 'operation': operation, 'message': msg, 'success': False}

        mask_np = mask_result.get('mask')
        # Asigură că masca are dimensiuni divizibile cu 8
        if mask_np is not None:
            mask_np = self._ensure_dimensions_multiple_of_8(mask_np)
            
        mask_pil = self._ensure_pil_mask(mask_np)
        if mask_pil is None: 
            logger.warning("Mask is None after generation.")
        else:
            # Asigură că masca și imaginea au aceleași dimensiuni
            if mask_pil.size != pil_image.size:
                logger.info(f"Redimensionare mască de la {mask_pil.size} la {pil_image.size} pentru a se potrivi cu imaginea")
                mask_pil = mask_pil.resize(pil_image.size, Image.NEAREST)

        # 4. Îmbunătățire Prompt și Parametri
        self._update_progress(0.6, desc="Pregătire prompt...")
        # Folosim direct PromptEnhancer în loc de metoda din BasePipeline
        enhanced_prompt = self.prompt_enhancer.enhance_prompt(
            prompt=prompt, 
            operation_type=op_type, 
            image_context=image_context
        )
        negative_prompt = self.prompt_enhancer.generate_negative_prompt(
            prompt=prompt, 
            operation_type=op_type, 
            image_context=image_context
        )

        gen_params = self._get_generation_params(op_type)
        
        # Asigură că height și width sunt divizibile cu 8 dacă sunt specificate
        width, height = pil_image.size
        effective_width = (width // 8) * 8
        effective_height = (height // 8) * 8
        
        final_params = {
            'image': pil_image, 
            'mask_image': mask_pil, 
            'prompt': enhanced_prompt, 
            'negative_prompt': negative_prompt,
            'strength': strength,
            'num_inference_steps': num_inference_steps if num_inference_steps is not None else gen_params['num_inference_steps'],
            'guidance_scale': guidance_scale if guidance_scale is not None else gen_params['guidance_scale'],
            'controlnet_conditioning_scale': gen_params['controlnet_conditioning_scale'] if use_controlnet_if_available else None,
            'use_refiner': use_refiner_if_available, 
            'refiner_strength': refiner_strength,
            'height': effective_height,  # Asigură că height este divizibil cu 8
            'width': effective_width     # Asigură că width este divizibil cu 8
        }
        final_params = {k: v for k, v in final_params.items() if v is not None}

        # 5. Apel model principal cu tratare OOM
        self._update_progress(0.7, desc="Generare imagine...")
        main_model = self.model_manager.get_model('main')
        if not main_model or not getattr(main_model, 'is_loaded', False):
            msg = "Main model not available or not loaded."
            logger.error(msg)
            return {'result_image': pil_image, 'mask_image': mask_pil, 'operation': operation, 'message': msg, 'success': False}

        try:
            # Apelăm modelul cu tratare OOM
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
                    'message': output_dict.get('message', f"General processing successful ({total_time:.2f}s)."),
                    'success': True
                }
            else:
                 msg = f"Processing failed: {output_dict.get('message')}"
                 logger.error(msg)
                 return {
                    'result_image': pil_image, 
                    'mask_image': mask_pil, 
                    'operation': operation,
                    'message': msg, 
                    'success': False
                 }
        except Exception as e:
            msg = f"Processing failed: {e}"
            logger.error(msg, exc_info=True)
            return {
                'result_image': pil_image, 
                'mask_image': mask_pil, 
                'operation': operation,
                'message': msg, 
                'success': False
            }
    
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
                # Se pare că avem o altă problemă, nu OOM
                if "have to be divisible by 8" in str(e):
                    # Avem probleme cu dimensiunile
                    logger.warning(f"Eroare de dimensiuni: {e}. Încercăm să corectăm...")
                    
                    # Ajustăm dimensiunile imaginii și măștii
                    if 'image' in final_params:
                        img = final_params['image']
                        if isinstance(img, Image.Image):
                            width, height = img.size
                            new_width = (width // 8) * 8
                            new_height = (height // 8) * 8
                            if width != new_width or height != new_height:
                                logger.info(f"Ajustăm dimensiunile imaginii de la {width}x{height} la {new_width}x{new_height}")
                                final_params['image'] = img.resize((new_width, new_height), Image.LANCZOS)
                    
                    if 'mask_image' in final_params and final_params['mask_image'] is not None:
                        mask = final_params['mask_image']
                        if isinstance(mask, Image.Image):
                            width, height = mask.size
                            new_width = (width // 8) * 8
                            new_height = (height // 8) * 8
                            if width != new_width or height != new_height:
                                logger.info(f"Ajustăm dimensiunile măștii de la {width}x{height} la {new_width}x{new_height}")
                                final_params['mask_image'] = mask.resize((new_width, new_height), Image.NEAREST)
                    
                    # Ajustăm explicit parametrii height și width
                    if 'height' in final_params:
                        final_params['height'] = (final_params['height'] // 8) * 8
                    if 'width' in final_params:
                        final_params['width'] = (final_params['width'] // 8) * 8
                    
                    # Încercăm din nou
                    try:
                        return model.process(**final_params)
                    except Exception as dim_retry_e:
                        logger.error(f"Corectarea dimensiunilor a eșuat: {dim_retry_e}")
                        raise RuntimeError(f"Procesare eșuată după corectarea dimensiunilor: {dim_retry_e}")
                else:
                    # Altă eroare
                    raise