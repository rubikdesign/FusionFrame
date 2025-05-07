#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Pipeline pentru eliminarea obiectelor/persoanelor în FusionFrame 2.0
"""

import logging
import cv2
import numpy as np
from typing import Dict, Any, Optional, Union, Tuple, Callable
from PIL import Image

from processing.pipelines.base_pipeline import BasePipeline
from processing.analyzer import OperationAnalyzer

# Setăm logger-ul
logger = logging.getLogger(__name__)

class RemovalPipeline(BasePipeline):
    """
    Pipeline specializat pentru eliminarea obiectelor sau persoanelor
    
    Implementează algoritmi avansați pentru eliminarea completă a
    obiectelor sau persoanelor din imagine cu reconstrucția fundalului.
    """
    
    def __init__(self):
        """Inițializează pipeline-ul pentru eliminare"""
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
        """
        Procesează imaginea pentru a elimina un obiect sau o persoană
        
        Args:
            image: Imaginea de procesat
            prompt: Promptul pentru eliminare
            strength: Intensitatea eliminării (0.0-1.0)
            progress_callback: Funcție de callback pentru progres
            use_refiner: Dacă să folosească refiner
            refiner_strength: Intensitatea refiner-ului
            **kwargs: Argumentele adiționale pentru procesare
            
        Returns:
            Dicționar cu rezultatele procesării
        """
        self.progress_callback = progress_callback
        
        # Analizăm operația pentru a determina ce eliminăm
        operation = self.operation_analyzer.analyze_operation(prompt)
        target = operation.get('target', '')
        
        # Decidem procesul în funcție de target
        if 'person' in target.lower():
            return self.remove_person(image, operation, strength, use_refiner=use_refiner, refiner_strength=refiner_strength, **kwargs)
        else:
            return self.remove_object(image, operation, strength, use_refiner=use_refiner, refiner_strength=refiner_strength, **kwargs)
    
    def remove_person(self,
                     image: Union[Image.Image, np.ndarray],
                     operation: Dict[str, Any],
                     strength: float = 0.75,
                     use_refiner: bool = None,
                     refiner_strength: float = None,
                     **kwargs) -> Dict[str, Any]:
        """
        Elimină o persoană din imagine
        
        Args:
            image: Imaginea de procesat
            operation: Detalii despre operație
            strength: Intensitatea eliminării (0.0-1.0)
            use_refiner: Dacă să folosească refiner
            refiner_strength: Intensitatea refiner-ului
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
        
        # 3. Inpainting folosind modelul principal
        self._update_progress(0.6, desc="Eliminare obiect...")
        
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
        
        # Îmbunătățim prompt-ul cu contextul
        context_desc = image_context['description']
        enhanced_prompt = f"scene without {target}, clean area, {context_desc}"
        
        # Setăm parametrii pentru eliminare
        params = self._get_generation_params('remove')
        params['strength'] = max(0.9, strength)  # Asigurăm eliminare completă
        
        # Procesăm cu modelul
        self._update_progress(0.7, desc="Generare finală...")
        try:
            result = main_model.process(
                image=pil_image,
                mask_image=Image.fromarray(mask),
                prompt=enhanced_prompt,
                negative_prompt=f"{target}, distortion, artifact, blurry",
                strength=params['strength'],
                num_inference_steps=params['num_inference_steps'],
                guidance_scale=params['guidance_scale'],
                controlnet_conditioning_scale=params.get('controlnet_conditioning_scale'),
                use_refiner=use_refiner,
                refiner_strength=refiner_strength
            )
            
            if result['success']:
                self._update_progress(1.0, desc="Procesare completă!")
                return {
                    'result': result['result'],
                    'mask': Image.fromarray(mask),
                    'operation': operation,
                    'message': f"{target} eliminat cu succes"
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
            logger.error(f"Error in object removal: {str(e)}")
            return {
                'result': pil_image,
                'mask': Image.fromarray(mask),
                'operation': operation,
                'message': f"Eroare: {str(e)}"
            }

    def remove_object(self,
                     image: Union[Image.Image, np.ndarray],
                     operation: Dict[str, Any],
                     strength: float = 0.75,
                     use_refiner: bool = None,
                     refiner_strength: float = None,
                     **kwargs) -> Dict[str, Any]:
        """
        Elimină un obiect din imagine
        
        Args:
            image: Imaginea de procesat
            operation: Detalii despre operație
            strength: Intensitatea eliminării (0.0-1.0)
            use_refiner: Dacă să folosească refiner
            refiner_strength: Intensitatea refiner-ului
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
        
        # 3. Inpainting folosind modelul principal
        self._update_progress(0.6, desc="Eliminare obiect...")
        
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
        
        # Îmbunătățim prompt-ul cu contextul
        context_desc = image_context['description']
        enhanced_prompt = f"scene without {target}, clean area, {context_desc}"
        
        # Setăm parametrii pentru eliminare
        params = self._get_generation_params('remove')
        params['strength'] = max(0.9, strength)  # Asigurăm eliminare completă
        
        # Procesăm cu modelul
        self._update_progress(0.7, desc="Generare finală...")
        try:
            result = main_model.process(
                image=pil_image,
                mask_image=Image.fromarray(mask),
                prompt=enhanced_prompt,
                negative_prompt=f"{target}, distortion, artifact, blurry",
                strength=params['strength'],
                num_inference_steps=params['num_inference_steps'],
                guidance_scale=params['guidance_scale'],
                controlnet_conditioning_scale=params.get('controlnet_conditioning_scale'),
                use_refiner=use_refiner,
                refiner_strength=refiner_strength
            )
            
            if result['success']:
                self._update_progress(1.0, desc="Procesare completă!")
                return {
                    'result': result['result'],
                    'mask': Image.fromarray(mask),
                    'operation': operation,
                    'message': f"{target} eliminat cu succes"
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
            logger.error(f"Error in object removal: {str(e)}")
            return {
                'result': pil_image,
                'mask': Image.fromarray(mask),
                'operation': operation,
                'message': f"Eroare: {str(e)}"
            }