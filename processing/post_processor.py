#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Post-procesare pentru rezultate în FusionFrame 2.0
"""

import logging
import cv2
import numpy as np
from typing import Dict, Any, Optional, Union, Tuple, Callable
from PIL import Image

from config.app_config import AppConfig
from core.model_manager import ModelManager

# Setăm logger-ul
logger = logging.getLogger(__name__)

class PostProcessor:
    """
    Modulul de post-procesare pentru îmbunătățirea rezultatelor
    
    Responsabil pentru eliminarea artefactelor, îmbunătățirea
    detaliilor și finisarea finală a rezultatelor generate.
    """
    
    def __init__(self):
        """Inițializează procesorul de post-procesare"""
        self.config = AppConfig
        self.model_manager = ModelManager()
        self.progress_callback = None
    
    def process(self,
               image: Union[Image.Image, np.ndarray],
               original_image: Union[Image.Image, np.ndarray] = None,
               mask: Union[Image.Image, np.ndarray] = None,
               operation_type: str = None,
               enhance_details: bool = True,
               fix_faces: bool = True,
               remove_artifacts: bool = True,
               progress_callback: Callable = None) -> Dict[str, Any]:
        """
        Procesează imaginea pentru a îmbunătăți rezultatul final
        
        Args:
            image: Imaginea de post-procesat
            original_image: Imaginea originală pentru referință (opțional)
            mask: Masca folosită pentru procesare (opțional)
            operation_type: Tipul operației efectuate
            enhance_details: Dacă se aplică îmbunătățirea detaliilor
            fix_faces: Dacă se aplică corectarea fețelor
            remove_artifacts: Dacă se aplică eliminarea artefactelor
            progress_callback: Funcție de callback pentru progres
            
        Returns:
            Dicționar cu rezultatele procesării
        """
        self.progress_callback = progress_callback
        
        # Convertim la format potrivit
        if isinstance(image, Image.Image):
            image_np = np.array(image)
            pil_image = image
        else:
            image_np = image
            pil_image = Image.fromarray(image)
        
        # Convertim imaginea originală la numpy dacă este furnizată
        if original_image is not None:
            if isinstance(original_image, Image.Image):
                original_np = np.array(original_image)
            else:
                original_np = original_image
        else:
            original_np = None
        
        # Convertim masca la numpy dacă este furnizată
        if mask is not None:
            if isinstance(mask, Image.Image):
                mask_np = np.array(mask)
            else:
                mask_np = mask
        else:
            mask_np = None
        
        # Aplicăm post-procesarea în funcție de parametrii
        processed_image = image_np.copy()
        
        # 1. Eliminăm artefactele dacă este necesar
        if remove_artifacts:
            self._update_progress(0.1, desc="Eliminare artefacte...")
            processed_image = self._remove_artifacts(processed_image, mask_np)
        
        # 2. Îmbunătățim detaliile dacă este necesar
        if enhance_details:
            self._update_progress(0.4, desc="Îmbunătățire detalii...")
            processed_image = self._enhance_details(processed_image, original_np, mask_np)
        
        # 3. Corectăm fețele dacă este necesar
        if fix_faces and operation_type != 'remove':
            self._update_progress(0.7, desc="Corectare fețe...")
            processed_image = self._fix_faces(processed_image)
        
        # 4. Aplicăm operații finale de netezire și curățare
        self._update_progress(0.9, desc="Finisare finală...")
        processed_image = self._final_polish(processed_image, original_np, mask_np)
        
        # Convertim rezultatul înapoi la PIL
        result_pil = Image.fromarray(processed_image)
        
        self._update_progress(1.0, desc="Post-procesare completă!")
        
        return {
            'result': result_pil,
            'success': True,
            'message': "Post-procesare completă cu succes"
        }
    
    def _update_progress(self, progress: float, desc: str = None):
        """
        Actualizează callback-ul de progres dacă există
        
        Args:
            progress: Progresul curent (0.0-1.0)
            desc: Descrierea progresului (opțional)
        """
        if self.progress_callback is not None:
            self.progress_callback(progress, desc=desc)
    
    def _remove_artifacts(self,
                         image: np.ndarray,
                         mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Elimină artefactele din imagine
        
        Args:
            image: Imaginea de procesat
            mask: Masca pentru procesare (opțional)
            
        Returns:
            Imaginea prelucrată
        """
        try:
            # Aplicăm un filtru bilateral pentru a elimina zgomotul păstrând marginile
            smoothed = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)
            
            # Dacă avem o mască, combinăm imaginea smoothed cu cea originală doar în regiunea măștii
            if mask is not None:
                # Ne asigurăm că masca are valorile potrivite (0-255)
                if mask.max() <= 1:
                    mask_for_blend = (mask * 255).astype(np.uint8)
                else:
                    mask_for_blend = mask.astype(np.uint8)
                
                # Normalizăm masca
                mask_for_blend = cv2.GaussianBlur(mask_for_blend, (5, 5), 0)
                
                # Creăm o mască pentru blend
                blend_mask = mask_for_blend / 255.0
                
                # Adaptăm dimensiunile măștii dacă e necesar
                if blend_mask.shape[:2] != image.shape[:2]:
                    blend_mask = cv2.resize(blend_mask, (image.shape[1], image.shape[0]))
                
                # Extinde masca la 3 canale dacă e necesar
                if len(blend_mask.shape) == 2 and len(image.shape) == 3:
                    blend_mask = np.repeat(blend_mask[:, :, np.newaxis], 3, axis=2)
                
                # Combinăm imaginile
                result = smoothed * blend_mask + image * (1 - blend_mask)
                result = result.astype(np.uint8)
            else:
                # Fără mască, folosim smoothed direct
                result = smoothed
            
            return result
            
        except Exception as e:
            logger.error(f"Error removing artifacts: {str(e)}")
            return image
    
    def _enhance_details(self,
                        image: np.ndarray,
                        original_image: Optional[np.ndarray] = None,
                        mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Îmbunătățește detaliile din imagine
        
        Args:
            image: Imaginea de procesat
            original_image: Imaginea originală pentru referință (opțional)
            mask: Masca pentru procesare (opțional)
            
        Returns:
            Imaginea prelucrată
        """
        try:
            # Încercăm să folosim modelul ESRGAN pentru îmbunătățirea detaliilor
            esrgan_model = self.model_manager.get_specialized_model('esrgan')
            if esrgan_model is not None:
                # Folosim ESRGAN pentru a îmbunătăți detaliile
                try:
                    enhanced = esrgan_model.process(image)
                    
                    # Dacă dimensiunile s-au schimbat, redimensionăm înapoi
                    if enhanced.shape[:2] != image.shape[:2]:
                        enhanced = cv2.resize(enhanced, (image.shape[1], image.shape[0]))
                    
                    # Dacă avem o mască, combinăm imaginea enhanced cu cea originală doar în regiunea măștii
                    if mask is not None:
                        # Ne asigurăm că masca are valorile potrivite (0-255)
                        if mask.max() <= 1:
                            mask_for_blend = (mask * 255).astype(np.uint8)
                        else:
                            mask_for_blend = mask.astype(np.uint8)
                        
                        # Normalizăm masca
                        mask_for_blend = cv2.GaussianBlur(mask_for_blend, (5, 5), 0)
                        
                        # Creăm o mască pentru blend
                        blend_mask = mask_for_blend / 255.0
                        
                        # Adaptăm dimensiunile măștii dacă e necesar
                        if blend_mask.shape[:2] != image.shape[:2]:
                            blend_mask = cv2.resize(blend_mask, (image.shape[1], image.shape[0]))
                        
                        # Extinde masca la 3 canale dacă e necesar
                        if len(blend_mask.shape) == 2 and len(image.shape) == 3:
                            blend_mask = np.repeat(blend_mask[:, :, np.newaxis], 3, axis=2)
                        
                        # Combinăm imaginile
                        result = enhanced * blend_mask + image * (1 - blend_mask)
                        result = result.astype(np.uint8)
                    else:
                        # Fără mască, folosim enhanced direct
                        result = enhanced
                    
                    return result
                except Exception as e:
                    logger.error(f"Error in ESRGAN processing: {str(e)}")
            
            # Fallback: folosim tehnici simple pentru îmbunătățirea contrastului și detaliilor
            # Implementăm un algoritm de unsharp masking
            gaussian = cv2.GaussianBlur(image, (0, 0), 3)
            unsharp_image = cv2.addWeighted(image, 1.5, gaussian, -0.5, 0)
            
            return unsharp_image
            
        except Exception as e:
            logger.error(f"Error enhancing details: {str(e)}")
            return image
    
    def _fix_faces(self, image: np.ndarray) -> np.ndarray:
        """
        Corectează fețele din imagine
        
        Args:
            image: Imaginea de procesat
            
        Returns:
            Imaginea prelucrată
        """
        try:
            # Detectăm fețele
            face_detector = self.model_manager.get_model('face_detector')
            if face_detector is None:
                return image
            
            # Procesăm imaginea pentru detecția feței
            h, w = image.shape[:2]
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_detector.process(rgb_image)
            
            # Verificăm dacă am detectat fețe
            face_detected = False
            if hasattr(results, 'detections') and results.detections:
                face_detected = True
                
                # Încercăm să folosim un model specializat pentru îmbunătățirea fețelor
                gpen_model = self.model_manager.get_specialized_model('gpen')
                if gpen_model is not None:
                    try:
                        # Procesăm întreaga imagine pentru a păstra consistența
                        enhanced_image = gpen_model.process(image)
                        return enhanced_image
                    except Exception as e:
                        logger.error(f"Error in GPEN processing: {str(e)}")
                
                # Fallback: folosim CodeFormer dacă GPEN nu este disponibil
                codeformer_model = self.model_manager.get_specialized_model('codeformer')
                if codeformer_model is not None:
                    try:
                        # Procesăm întreaga imagine pentru a păstra consistența
                        enhanced_image = codeformer_model.process(image)
                        return enhanced_image
                    except Exception as e:
                        logger.error(f"Error in CodeFormer processing: {str(e)}")
            
            # Dacă nu am detectat fețe sau dacă modelele specializate au eșuat, returnăm imaginea originală
            return image
            
        except Exception as e:
            logger.error(f"Error fixing faces: {str(e)}")
            return image
    
    def _final_polish(self,
                     image: np.ndarray,
                     original_image: Optional[np.ndarray] = None,
                     mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Aplică finisarea finală a imaginii
        
        Args:
            image: Imaginea de procesat
            original_image: Imaginea originală pentru referință (opțional)
            mask: Masca pentru procesare (opțional)
            
        Returns:
            Imaginea prelucrată
        """
        try:
            # 1. Aplicăm o netezire fină pentru a elimina zgomotul rezidual
            polished = cv2.GaussianBlur(image, (3, 3), 0)
            
            # 2. Îmbunătățim contrastul și luminozitatea pentru a păstra naturalețea
            lab = cv2.cvtColor(polished, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Aplicăm CLAHE doar pe canalul L (luminozitate)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            
            # Combinăm canalele
            lab = cv2.merge((cl, a, b))
            polished = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            
            # 3. Dacă avem o mască și imaginea originală, asigurăm o tranziție netedă la margini
            if mask is not None and original_image is not None:
                # Ne asigurăm că masca are valorile potrivite (0-255)
                if mask.max() <= 1:
                    mask_for_blend = (mask * 255).astype(np.uint8)
                else:
                    mask_for_blend = mask.astype(np.uint8)
                
                # Erodăm masca pentru a crea o zonă de tranziție
                kernel = np.ones((5, 5), np.uint8)
                eroded_mask = cv2.erode(mask_for_blend, kernel, iterations=1)
                
                # Calculăm zona de tranziție
                transition_zone = cv2.subtract(mask_for_blend, eroded_mask)
                
                # Aplicăm un blur pe zona de tranziție
                transition_zone = cv2.GaussianBlur(transition_zone, (15, 15), 0)
                
                # Normalizăm masca
                blended_mask = transition_zone / 255.0
                
                # Adaptăm dimensiunile dacă e necesar
                if blended_mask.shape[:2] != polished.shape[:2]:
                    blended_mask = cv2.resize(blended_mask, (polished.shape[1], polished.shape[0]))
                
                # Extinde masca la 3 canale dacă e necesar
                
                if len(blended_mask.shape) == 2 and len(polished.shape) == 3:
                    blended_mask = np.repeat(blended_mask[:, :, np.newaxis], 3, axis=2)
                
                # Combinăm imaginea originală cu cea prelucrată în zona de tranziție
                result = polished.copy()
                result = original_image * (1 - blended_mask) + polished * blended_mask
                result = result.astype(np.uint8)
            else:
                result = polished
            
            return result
            
        except Exception as e:
            logger.error(f"Error in final polish: {str(e)}")
            return image