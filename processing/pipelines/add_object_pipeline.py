#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Pipeline pentru adăugarea obiectelor în FusionFrame 2.0
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

class AddObjectPipeline(BasePipeline):
    """
    Pipeline specializat pentru adăugarea obiectelor
    
    Implementează algoritmi avansați pentru adăugarea naturală
    de obiecte (ochelari, haine, accesorii, etc) în imagini.
    """
    
    def __init__(self):
        """Inițializează pipeline-ul pentru adăugarea obiectelor"""
        super().__init__()
        self.operation_analyzer = OperationAnalyzer()
    
    def process(self,
               image: Union[Image.Image, np.ndarray],
               prompt: str,
               strength: float = 0.75,
               progress_callback: Callable = None,
               **kwargs) -> Dict[str, Any]:
        """
        Procesează imaginea pentru a adăuga un obiect
        
        Args:
            image: Imaginea de procesat
            prompt: Promptul pentru adăugarea obiectului
            strength: Intensitatea adăugării (0.0-1.0)
            progress_callback: Funcție de callback pentru progres
            **kwargs: Argumentele adiționale pentru procesare
            
        Returns:
            Dicționar cu rezultatele procesării
        """
        self.progress_callback = progress_callback
        
        # Analizăm operația pentru a determina ce obiect adăugăm
        operation = self.operation_analyzer.analyze_operation(prompt)
        
        # Decidem procesul în funcție de tipul obiectului
        if "glasses" in prompt.lower() or "ochelari" in prompt.lower():
            return self.add_glasses(image, operation, strength, **kwargs)
        else:
            return self.add_generic_object(image, operation, strength, **kwargs)
    
    def add_glasses(self,
                  image: Union[Image.Image, np.ndarray],
                  operation: Dict[str, Any],
                  strength: float = 0.75,
                  **kwargs) -> Dict[str, Any]:
        """
        Adaugă ochelari pe față
        
        Args:
            image: Imaginea de procesat
            operation: Detalii despre operație
            strength: Intensitatea adăugării (0.0-1.0)
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
        
        # 2. Generăm masca pentru regiunea ochilor
        self._update_progress(0.2, desc="Detectare față și ochi...")
        
        # Detectăm fața
        face_detector = self.model_manager.get_model('face_detector')
        if face_detector is None:
            logger.error("Face detector not available")
            return {
                'result': pil_image,
                'mask': None,
                'operation': operation,
                'message': "Detectorul de față nu este disponibil"
            }
        
        # Procesăm imaginea pentru detecția feței
        h, w = image_np.shape[:2]
        face_mask = np.zeros((h, w), dtype=np.uint8)
        
        rgb_image = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        results = face_detector.process(rgb_image)
        
        # Verificăm dacă am detectat fețe
        face_detected = False
        if hasattr(results, 'detections') and results.detections:
            face_detected = True
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                x, y = int(bbox.xmin * w), int(bbox.ymin * h)
                width, height = int(bbox.width * w), int(bbox.height * h)
                
                # Creăm o mască pentru regiunea ochilor
                eye_region_y = y + int(height * 0.3)  # Regiunea ochilor începe la aproximativ 30% din înălțimea feței
                eye_region_height = int(height * 0.25)  # Regiunea ochilor ocupă aproximativ 25% din înălțimea feței
                
                # Desenăm masca pentru regiunea ochilor
                cv2.rectangle(
                    face_mask,
                    (x, eye_region_y),
                    (x + width, eye_region_y + eye_region_height),
                    255,
                    -1
                )
        
        # Verificăm dacă am găsit vreo față
        if not face_detected:
            logger.warning("No face detected, using fallback mask")
            # Creăm o mască de rezervă
            face_mask = np.zeros((h, w), dtype=np.uint8)
            center_y = h // 3
            center_height = h // 6
            cv2.rectangle(
                face_mask,
                (w // 4, center_y),
                (3 * w // 4, center_y + center_height),
                255,
                -1
            )
        
        self._update_progress(0.5, desc="Generare ochelari...")
        
        # 3. Procesăm adăugarea ochelarilor folosind modelul principal
        main_model = self.model_manager.get_model('main')
        if main_model is None:
            logger.error("Main model not available")
            return {
                'result': pil_image,
                'mask': Image.fromarray(face_mask),
                'operation': operation,
                'message': "Modelul principal nu este disponibil"
            }
        
        # Îmbunătățim prompt-ul
        glasses_type = operation.get('attribute', '')
        if glasses_type:
            enhanced_prompt = f"face with {glasses_type} glasses, realistic glasses, high quality eyewear"
        else:
            enhanced_prompt = "face with stylish glasses, realistic glasses, detailed eyewear"
        
        # Setăm parametrii pentru adăugare
        params = self._get_generation_params('add')
        params['strength'] = min(0.6, strength)  # Valoare mai mică pentru a păstra fața
        
        # Procesăm cu modelul
        self._update_progress(0.7, desc="Aplicare ochelari...")
        try:
            result = main_model.process(
                image=pil_image,
                mask_image=Image.fromarray(face_mask),
                prompt=enhanced_prompt,
                negative_prompt="unrealistic, deformed, distorted, blurry, bad quality",
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
                    'mask': Image.fromarray(face_mask),
                    'operation': operation,
                    'message': "Ochelari adăugați cu succes"
                }
            else:
                logger.error(f"Error in processing: {result['message']}")
                return {
                    'result': pil_image,
                    'mask': Image.fromarray(face_mask),
                    'operation': operation,
                    'message': result['message']
                }
                
        except Exception as e:
            logger.error(f"Error in adding glasses: {str(e)}")
            return {
                'result': pil_image,
                'mask': Image.fromarray(face_mask),
                'operation': operation,
                'message': f"Eroare: {str(e)}"
            }
    
    def add_generic_object(self,
                         image: Union[Image.Image, np.ndarray],
                         operation: Dict[str, Any],
                         strength: float = 0.75,
                         **kwargs) -> Dict[str, Any]:
        """
        Adaugă un obiect generic
        
        Args:
            image: Imaginea de procesat
            operation: Detalii despre operație
            strength: Intensitatea adăugării (0.0-1.0)
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
        
        # 2. Generăm masca pentru adăugarea obiectului
        self._update_progress(0.2, desc="Pregătire regiune...")
        
        # Pentru obiectele generice, putem folosi o mască centrală sau specifică
        # în funcție de tipul obiectului
        h, w = image_np.shape[:2]
        object_mask = np.zeros((h, w), dtype=np.uint8)
        
        # Obținem tipul de obiect
        object_type = operation.get('attribute', '')
        if not object_type:
            object_type = kwargs.get('object_type', 'generic object')
        
        # Ajustăm masca în funcție de tipul obiectului
        if object_type.lower() in ['hat', 'cap', 'palarie', 'pălărie']:
            # Pentru pălării, masca ar trebui să fie în partea de sus a imaginii
            top_margin = h // 6
            cv2.rectangle(
                object_mask,
                (w // 4, 0),
                (3 * w // 4, top_margin + h // 8),
                255,
                -1
            )
        elif object_type.lower() in ['necklace', 'pendant', 'colier', 'lanț']:
            # Pentru coliere, masca ar trebui să fie în zona gâtului
            neck_y = h // 3
            cv2.rectangle(
                object_mask,
                (w // 3, neck_y),
                (2 * w // 3, neck_y + h // 8),
                255,
                -1
            )
        else:
            # Pentru alte obiecte, folosim o mască centrală
            center_x, center_y = w // 2, h // 2
            cv2.rectangle(
                object_mask,
                (center_x - w // 4, center_y - h // 4),
                (center_x + w // 4, center_y + h // 4),
                255,
                -1
            )
        
        self._update_progress(0.5, desc=f"Generare {object_type}...")
        
        # 3. Procesăm adăugarea obiectului folosind modelul principal
        main_model = self.model_manager.get_model('main')
        if main_model is None:
            logger.error("Main model not available")
            return {
                'result': pil_image,
                'mask': Image.fromarray(object_mask),
                'operation': operation,
                'message': "Modelul principal nu este disponibil"
            }
        
        # Îmbunătățim prompt-ul
        scene_desc = image_context.get('scene_type', 'scene')
        style_desc = image_context.get('style', 'natural')
        enhanced_prompt = f"photo with {object_type}, realistic {object_type}, high quality, {style_desc} style, {scene_desc}"
        
        # Setăm parametrii pentru adăugare
        params = self._get_generation_params('add')
        
        # Procesăm cu modelul
        self._update_progress(0.7, desc=f"Aplicare {object_type}...")
        try:
            result = main_model.process(
                image=pil_image,
                mask_image=Image.fromarray(object_mask),
                prompt=enhanced_prompt,
                negative_prompt="unrealistic, deformed, distorted, blurry, bad quality",
                strength=strength,
                num_inference_steps=params['num_inference_steps'],
                guidance_scale=params['guidance_scale'],
                controlnet_conditioning_scale=params.get('controlnet_conditioning_scale')
            )
            
            if result['success']:
                self._update_progress(1.0, desc="Procesare completă!")
                return {
                    'result': result['result'],
                    'mask': Image.fromarray(object_mask),
                    'operation': operation,
                    'message': f"{object_type} adăugat cu succes"
                }
            else:
                logger.error(f"Error in processing: {result['message']}")
                return {
                    'result': pil_image,
                    'mask': Image.fromarray(object_mask),
                    'operation': operation,
                    'message': result['message']
                }
                
        except Exception as e:
            logger.error(f"Error in adding object: {str(e)}")
            return {
                'result': pil_image,
                'mask': Image.fromarray(object_mask),
                'operation': operation,
                'message': f"Eroare: {str(e)}"
            }
```image_analyzer.analyze_image_context(image_np)
        
        # 2. Generăm masca pentru persoană
        self._update_progress(0.2, desc="Generare mască persoană...")
        mask_result = self.mask_generator.generate_mask(
            image=image_np,
            prompt="person",
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
        
        # 3. Inpainting inițial folosind modelul principal
        self._update_progress(0.6, desc="Eliminare inițială...")
        
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
        enhanced_prompt = f"empty scene without any person, clean background, {context_desc}"
        
        # Setăm parametrii pentru eliminare
        params = self._get_generation_params('remove')
        params['strength'] = max(0.95, strength)  # Asigurăm eliminare completă
        
        # Procesăm cu modelul
        self._update_progress(0.7, desc="Generare finală...")
        try:
            result = main_model.process(
                image=pil_image,
                mask_image=Image.fromarray(mask),
                prompt=enhanced_prompt,
                negative_prompt="person, human, face, body, distortion, artifact, blurry",
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
                    'message': "Persoană eliminată cu succes"
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
            logger.error(f"Error in person removal: {str(e)}")
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
                     **kwargs) -> Dict[str, Any]:
        """
        Elimină un obiect din imagine
        
        Args:
            image: Imaginea de procesat
            operation: Detalii despre operație
            strength: Intensitatea eliminării (0.0-1.0)
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
        image_context = self.