#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Pipeline pentru schimbarea culorilor în FusionFrame 2.0
"""

import logging
import cv2
import numpy as np
from typing import Dict, Any, Optional, Union, Tuple, Callable
from PIL import Image

from fusionframe.processing.pipelines.base_pipeline import BasePipeline
from fusionframe.processing.analyzer import OperationAnalyzer

# Setăm logger-ul
logger = logging.getLogger(__name__)

class ColorChangePipeline(BasePipeline):
    """
    Pipeline specializat pentru schimbarea culorilor
    
    Implementează algoritmi avansați pentru schimbarea naturală 
    a culorilor pentru haine, păr, obiecte, etc.
    """
    
    def __init__(self):
        """Inițializează pipeline-ul pentru schimbarea culorilor"""
        super().__init__()
        self.operation_analyzer = OperationAnalyzer()
    
    def process(self,
               image: Union[Image.Image, np.ndarray],
               prompt: str,
               strength: float = 0.75,
               progress_callback: Callable = None,
               **kwargs) -> Dict[str, Any]:
        """
        Procesează imaginea pentru a schimba culoarea unui obiect
        
        Args:
            image: Imaginea de procesat
            prompt: Promptul pentru schimbarea culorii
            strength: Intensitatea schimbării (0.0-1.0)
            progress_callback: Funcție de callback pentru progres
            **kwargs: Argumentele adiționale pentru procesare
            
        Returns:
            Dicționar cu rezultatele procesării
        """
        self.progress_callback = progress_callback
        
        # Analizăm operația pentru a determina ce culori schimbăm
        operation = self.operation_analyzer.analyze_operation(prompt)
        target = operation.get('target', '')
        
        # Decidem procesul în funcție de target
        if 'hair' in target.lower():
            return self.change_hair_color(image, operation, strength, **kwargs)
        else:
            return self.change_object_color(image, operation, strength, **kwargs)
    
    def change_hair_color(self,
                        image: Union[Image.Image, np.ndarray],
                        operation: Dict[str, Any],
                        strength: float = 0.75,
                        **kwargs) -> Dict[str, Any]:
        """
        Schimbă culoarea părului
        
        Args:
            image: Imaginea de procesat
            operation: Detalii despre operație
            strength: Intensitatea schimbării (0.0-1.0)
            **kwargs: Argumentele adiționale pentru procesare
            
        Returns:
            Dicționar cu rezultatele procesării
        """
        # Convertim la format potrivit
        if isinstance(image, np.ndarray):
            image_np = image
            pil_image = Image.fromarray(image)
        else:
            image_np = np.array(image)
            pil_image = image
        
        # 1. Analizăm imaginea și contextul
        self._update_progress(0.1, desc="Analiză imagine...")
        image_context = self.image_analyzer.analyze_image_context(image_np)
        
        # 2. Generăm masca pentru păr
        self._update_progress(0.2, desc="Generare mască păr...")
        mask_result = self.mask_generator.generate_mask(
            image=image_np,
            prompt="hair",
            operation=operation,
            progress_callback=lambda p, desc: self._update_progress(0.2 + p * 0.3, desc=desc)
        )
        
        # Verificăm rezultatul măștii
        if not mask_result['success']:
            logger.error("Failed to generate mask")
            return {
                'result': pil_image,
                'mask': None,
                'operation': operation,
                'message': "Eroare la generarea măștii"
            }
        
        # Obținem masca
        mask = mask_result['mask']
        
        # 3. Procesăm schimbarea de culoare folosind modelul principal
        self._update_progress(0.6, desc="Schimbare culoare păr...")
        
        # Obținem modelul principal
        main_model = self.model_manager.get_model('main')
        if main_model is None:
            logger.error("Main model not available")
            return {
                'result': pil_image,
                'mask': Image.fromarray(mask),
                'operation': operation,
                'message': "Modelul principal nu este disponibil"
            }
        
        # Obținem culoarea țintă
        target_color = operation.get('attribute', 'changed color')
        
        # Îmbunătățim prompt-ul
        enhanced_prompt = f"{target_color} hair, natural looking, realistic {target_color} hair, matching lighting"
        
        # Setăm parametrii pentru schimbare de culoare
        params = self._get_generation_params('color')
        params['strength'] = min(0.7, strength + 0.1)  # Valoare mai mică pentru a păstra trăsăturile
        
        # Procesăm cu modelul
        self._update_progress(0.7, desc="Aplicare culoare...")
        try:
            result = main_model.process(
                image=pil_image,
                mask_image=Image.fromarray(mask),
                prompt=enhanced_prompt,
                negative_prompt="unrealistic hair, wig, bad hair, distortion, blurry",
                strength=params['strength'],
                num_inference_steps=params['num_inference_steps'],
                guidance_scale=params['guidance_scale'],
                controlnet_conditioning_scale=params.get('controlnet_conditioning_scale')
            )
            
            if result['success']:
                # Încercăm să îmbunătățim fața dacă este posibil
                try:
                    self._update_progress(0.9, desc="Îmbunătățire față...")
                    gpen_model = self.model_manager.get_specialized_model('gpen')
                    if gpen_model:
                        enhanced_result = gpen_model.process(np.array(result['result']))
                        if enhanced_result['success']:
                            result['result'] = enhanced_result['result']
                except Exception as e:
                    logger.error(f"Error enhancing face: {str(e)}")
                
                self._update_progress(1.0, desc="Procesare completă!")
                return {
                    'result': result['result'],
                    'mask': Image.fromarray(mask),
                    'operation': operation,
                    'message': f"Culoare păr schimbată în {target_color} cu succes"
                }
            else:
                logger.error(f"Error in processing: {result['message']}")
                return {
                    'result': pil_image,
                    'mask': Image.fromarray(mask),
                    'operation': operation,
                    'message': result['message']
                }
                
        except Exception as e:
            logger.error(f"Error in hair color change: {str(e)}")
            return {
                'result': pil_image,
                'mask': Image.fromarray(mask),
                'operation': operation,
                'message': f"Eroare: {str(e)}"
            }
    
    def change_object_color(self,
                          image: Union[Image.Image, np.ndarray],
                          operation: Dict[str, Any],
                          strength: float = 0.75,
                          **kwargs) -> Dict[str, Any]:
        """
        Schimbă culoarea unui obiect
        
        Args:
            image: Imaginea de procesat
            operation: Detalii despre operație
            strength: Intensitatea schimbării (0.0-1.0)
            **kwargs: Argumentele adiționale pentru procesare
            
        Returns:
            Dicționar cu rezultatele procesării
        """
        # Convertim la format potrivit
        if isinstance(image, np.ndarray):
            image_np = image
            pil_image = Image.fromarray(image)
        else:
            image_np = np.array(image)
            pil_image = image
        
        # 1. Analizăm imaginea și contextul
        self._update_progress(0.1, desc="Analiză imagine...")
        image_context = self.image_analyzer.analyze_image_context(image_np)
        
        # 2. Generăm masca pentru obiect
        self._update_progress(0.2, desc="Generare mască obiect...")
        target = operation.get('target', 'object')
        mask_result = self.mask_generator.generate_mask(
            image=image_np,
            prompt=target,
            operation=operation,
            progress_callback=lambda p, desc: self._update_progress(0.2 + p * 0.3, desc=desc)
        )
        
        # Verificăm rezultatul măștii
        if not mask_result['success']:
            logger.error("Failed to generate mask")
            return {
                'result': pil_image,
                'mask': None,
                'operation': operation,
                'message': "Eroare la generarea măștii"
            }
        
        # Obținem masca
        mask = mask_result['mask']
        
        # 3. Procesăm schimbarea de culoare folosind modelul principal
        self._update_progress(0.6, desc="Schimbare culoare obiect...")
        
        # Obținem modelul principal
        main_model = self.model_manager.get_model('main')
        if main_model is None:
            logger.error("Main model not available")
            return {
                'result': pil_image,
                'mask': Image.fromarray(mask),
                'operation': operation,
                'message': "Modelul principal nu este disponibil"
            }
        
        # Obținem culoarea țintă
        target_color = operation.get('attribute', 'changed color')
        
        # Îmbunătățim prompt-ul
        enhanced_prompt = f"{target_color} {target}, realistic {target_color} color, natural texture"
        
        # Setăm parametrii pentru schimbare de culoare
        params = self._get_generation_params('color')
        params['strength'] = min(0.6, strength)  # Valoare mai mică pentru a păstra structura
        
        # Procesăm cu modelul
        self._update_progress(0.7, desc="Aplicare culoare...")
        try:
            result = main_model.process(
                image=pil_image,
                mask_image=Image.fromarray(mask),
                prompt=enhanced_prompt,
                negative_prompt="unrealistic color, distortion, blurry",
                strength=params['strength'],
                num_inference_steps=params['num_inference_steps'],
                guidance_scale=params['guidance_scale'],
                controlnet_conditioning_scale=params.get('controlnet_conditioning_scale')
            )
            
            if result['success']:
                self._update_progress(1.0, desc="Procesare completă!")
                return {
                    'result': result['result'],
                    'mask': Image.fromarray(mask),
                    'operation': operation,
                    'message': f"Culoare {target} schimbată în {target_color} cu succes"
                }
            else:
                logger.error(f"Error in processing: {result['message']}")
                return {
                    'result': pil_image,
                    'mask': Image.fromarray(mask),
                    'operation': operation,
                    'message': result['message']
                }
                
        except Exception as e:
            logger.error(f"Error in object color change: {str(e)}")
            return {
                'result': pil_image,
                'mask': Image.fromarray(mask),
                'operation': operation,
                'message': f"Eroare: {str(e)}"
            }