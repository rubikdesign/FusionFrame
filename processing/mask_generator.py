#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Generator de măști pentru FusionFrame 2.0
"""

import cv2
import numpy as np
import logging
import torch
from typing import Dict, Any, List, Optional, Union, Tuple, Callable
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

from config.app_config import AppConfig
from core.model_manager import ModelManager

# Setăm logger-ul
logger = logging.getLogger(__name__)

class MaskGenerator:
    """
    Generator de măști pentru segmentare în FusionFrame 2.0
    
    Responsabil pentru generarea de măști precise pentru operațiile
    de editare a imaginilor.
    """
    
    def __init__(self):
        """Inițializează generatorul de măști"""
        self.model_manager = ModelManager()
        self.config = AppConfig
        self.progress_callback = None
    
    def generate_mask(self, 
                     image: Union[Image.Image, np.ndarray],
                     prompt: str = None,
                     operation: Dict[str, Any] = None,
                     progress_callback: Callable = None) -> Dict[str, Any]:
        """
        Generează o mască optimizată pentru operația specificată
        
        Args:
            image: Imaginea de procesat
            prompt: Promptul pentru generarea măștii
            operation: Detalii despre operație
            progress_callback: Funcție de callback pentru progres
            
        Returns:
            Dicționar cu masca și informații adiționale
        """
        self.progress_callback = progress_callback
        
        # Convertim la numpy array dacă este PIL
        if isinstance(image, Image.Image):
            image_np = np.array(image)
        else:
            image_np = image
        
        # Inițializăm parametrii pentru generarea măștii
        h, w = image_np.shape[:2]
        mask_weights = {
            'sam': 0.4,
            'yolo': 0.3,
            'clipseg': 0.2,
            'mediapipe': 0.1,
            'face': 0.2,
            'background': 0.5  # Pondere mai mare pentru detecția fundalului
        }
        final_mask = np.zeros((h, w), dtype=np.float32)
        
        # Inițializăm cu o mască de bază în caz că toate modelele eșuează
        backup_mask = np.ones((h, w), dtype=np.float32) * 0.5
        center_x, center_y = w // 2, h // 2
        center_radius = min(w, h) // 4
        cv2.circle(backup_mask, (center_x, center_y), center_radius, 1.0, -1)
        
        # Urmărim modelele de succes
        success_count = 0
        
        # Gestionare specială pentru diferite tipuri de operații
        if operation and operation.get('type') == 'background':
            # Pentru fundal, folosim detecția fundalului ca o componentă cu pondere ridicată
            background_mask = self._process_background_detection(image_np)
            if background_mask is not None:
                # Deoarece operația de fundal inversează masca, trebuie să o inversăm și aici
                background_mask = 1.0 - background_mask  # Inversăm pentru selectarea subiectului
                final_mask += background_mask * mask_weights['background']
                success_count += 1
        elif operation and (operation.get('type') == 'color' or operation.get('type') == 'replace'):
            # Pentru operațiile de culoare, creștem ponderea detecției fețelor dacă sunt menționate elemente faciale
            face_keywords = ['face', 'hair', 'eye', 'eyes', 'lips', 'mouth', 'nose', 'skin']
            target = operation.get('target', '')
            if any(keyword in (target or '').lower() for keyword in face_keywords):
                mask_weights['face'] = 0.6  # Prioritizăm detecția feței pentru editări legate de față
                # Folosim SAM cu o pondere mai mică pentru aceste operații pentru a evita supra-segmentarea
                mask_weights['sam'] = 0.2
        
        if self.progress_callback:
            self.progress_callback(0.1, desc="Generare mască hibridă...")
        
        # Folosim thread pool pentru procesare paralelă
        with ThreadPoolExecutor(max_workers=self.config.MAX_WORKERS) as executor:
            futures = []
            future_to_model = {}
            
            # Pregătim parametrii pentru prompt
            text_prompt = prompt
            if operation:
                # Ajustăm promptul în funcție de operație
                if operation.get('type') == 'replace' and operation.get('target'):
                    text_prompt = operation.get('target')
                elif operation.get('type') == 'color' and operation.get('target'):
                    text_prompt = operation.get('target')
                elif operation.get('type') == 'remove' and operation.get('target'):
                    text_prompt = operation.get('target')
            
            # Trimitem sarcina SAM
            if self.progress_callback:
                self.progress_callback(0.2, desc="Procesare SAM...")
            future = executor.submit(self._process_sam, image_np)
            futures.append(future)
            future_to_model[future] = 'sam'
            
            # Trimitem sarcina YOLO
            if self.progress_callback:
                self.progress_callback(0.3, desc="Procesare YOLO...")
            future = executor.submit(self._process_yolo, image_np, text_prompt)
            futures.append(future)
            future_to_model[future] = 'yolo'
            
            # Trimitem sarcina CLIPSeg
            if self.progress_callback:
                self.progress_callback(0.4, desc="Procesare CLIPSeg...")
            future = executor.submit(self._process_clipseg, image_np, text_prompt)
            futures.append(future)
            future_to_model[future] = 'clipseg'
            
            # Trimitem sarcina MediaPipe
            if self.progress_callback:
                self.progress_callback(0.5, desc="Procesare MediaPipe...")
            future = executor.submit(self._process_mediapipe, image_np)
            futures.append(future)
            future_to_model[future] = 'mediapipe'
            
            # Trimitem sarcina detecției fețelor
            if self.progress_callback:
                self.progress_callback(0.6, desc="Procesare detecție față...")
            future = executor.submit(self._process_face_detection, image_np)
            futures.append(future)
            future_to_model[future] = 'face'
            
            # Colectăm rezultatele și actualizăm masca
            for i, future in enumerate(futures):
                model_name = future_to_model[future]
                try:
                    mask = future.result(timeout=10)
                    if mask is not None and mask.shape[:2] == (h, w):
                        weight = mask_weights.get(model_name, 0.1)
                        final_mask += mask * weight
                        success_count += 1
                    if self.progress_callback:
                        self.progress_callback(0.6 + 0.2 * (i+1) / len(futures), 
                                              desc=f"Procesare {model_name} completă")
                except Exception as e:
                    logger.error(f"Error processing {model_name}: {str(e)}")
                    if self.progress_callback:
                        self.progress_callback(0.6 + 0.2 * (i+1) / len(futures), 
                                              desc=f"Eroare procesare {model_name}")
        
        # Dacă toate modelele au eșuat, folosim masca de rezervă
        if success_count == 0:
            final_mask = backup_mask
        
        # Normalizăm și stabilim pragul
        if final_mask.max() > 0:
            final_mask = (final_mask - final_mask.min()) / (final_mask.max() - final_mask.min())
        
        # Prag adaptiv cu ajustări specifice operației
        if operation and operation.get('type') == 'background':
            # Prag mai agresiv pentru fundal pentru a asigura margini curate
            dynamic_threshold = 0.35 + 0.45 * (1 - np.mean(final_mask))
        elif operation and operation.get('type') == 'color':
            # Prag mai scăzut pentru culoare pentru a asigura acoperirea completă
            dynamic_threshold = 0.25 + 0.35 * (1 - np.mean(final_mask))
        else:
            # Prag standard pentru alte operații
            dynamic_threshold = 0.3 + 0.4 * (1 - np.mean(final_mask))
            
        binary_mask = (final_mask > dynamic_threshold).astype(np.uint8) * 255
        
        # Operații morfologice cu ajustări specifice operației
        if operation and operation.get('type') == 'background':
            # Nucleu mai mare pentru operațiile de fundal
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
            binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
        elif operation and operation.get('type') == 'color':
            # Nucleu mai mic pentru operațiile de culoare pentru a păstra detaliile
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        else:
            # Nucleu standard pentru alte operații
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # Pentru operațiile de fundal, inversăm masca
        if operation and operation.get('type') == 'background':
            binary_mask = 255 - binary_mask
        
        if self.progress_callback:
            self.progress_callback(0.9, desc="Rafinare mască...")
        
        # Rafinăm masca
        refined_mask = self.refine_mask(binary_mask, image_np)
        
        if self.progress_callback:
            self.progress_callback(1.0, desc="Generare mască completă")
        
        return {
            'mask': refined_mask,
            'raw_mask': binary_mask,
            'success': True,
            'message': "Generare mască completă"
        }
    
    def _process_sam(self, image_np: np.ndarray) -> Optional[np.ndarray]:
        """
        Procesează generarea măștii SAM
        
        Args:
            image_np: Imaginea de procesat
            
        Returns:
            Masca generată sau None în caz de eroare
        """
        try:
            # Obținem modelul SAM
            sam = self.model_manager.get_model('sam')
            if sam is None:
                logger.warning("SAM model not available")
                return None
                
            # Generăm măștile
            masks = sam.generate(image_np)
            combined_mask = np.zeros_like(image_np[..., 0], dtype=np.float32)
            
            for mask in masks:
                if mask['stability_score'] > 0.9:
                    resized_mask = self._safe_resize(
                        mask['segmentation'].astype(np.float32), 
                        (image_np.shape[1], image_np.shape[0]),
                        interpolation=cv2.INTER_LINEAR
                    )
                    combined_mask += resized_mask
                    
            return combined_mask
            
        except Exception as e:
            logger.error(f"Error in SAM processing: {str(e)}")
            return None
    
    def _process_yolo(self, image_np: np.ndarray, prompt: Optional[str] = None) -> Optional[np.ndarray]:
        """
        Procesează segmentarea YOLO
        
        Args:
            image_np: Imaginea de procesat
            prompt: Promptul pentru filtrarea detectărilor
            
        Returns:
            Masca generată sau None în caz de eroare
        """
        try:
            # Obținem modelul YOLO
            yolo = self.model_manager.get_model('yolo')
            if yolo is None:
                logger.warning("YOLO model not available")
                return None
                
            # Rulăm detecția YOLO
            results = yolo(
                image_np, 
                imgsz=960, 
                conf=0.3,
                iou=0.4,
                agnostic_nms=True,
                verbose=False
            )
            
            combined_mask = np.zeros_like(image_np[..., 0], dtype=np.float32)
            
            for result in results:
                if hasattr(result, 'masks') and result.masks is not None:
                    for mask, box in zip(result.masks.xy, result.boxes):
                        # Verificăm dacă se potrivește cu prompt-ul
                        cls_id = box.cls.item() if hasattr(box.cls, 'item') else box.cls
                        if not prompt or self._is_relevant_detection(prompt, cls_id, yolo.names):
                            poly = np.array(mask, np.int32).reshape((-1, 1, 2))
                            cv2.fillPoly(combined_mask, [poly], 1.0)
                            
            return combined_mask
            
        except Exception as e:
            logger.error(f"Error in YOLO processing: {str(e)}")
            return None
    
    def _is_relevant_detection(self, prompt: str, class_id: Union[int, float], 
                              class_names: Dict[int, str]) -> bool:
        """
        Verifică dacă clasa de detecție se potrivește cu prompt-ul
        
        Args:
            prompt: Promptul de verificat
            class_id: ID-ul clasei detectate
            class_names: Dicționar cu ID-uri și nume de clase
            
        Returns:
            True dacă detecția este relevantă, False altfel
        """
        try:
            # Verificăm dacă class_id este valid
            if isinstance(class_id, (int, float)) and int(class_id) in class_names:
                class_name = class_names[int(class_id)]
                # Verificăm dacă numele clasei apare în prompt
                prompt_lower = prompt.lower()
                class_lower = class_name.lower()
                
                return class_lower in prompt_lower or any(
                    word in prompt_lower for word in class_lower.split('_')
                )
            return True  # Implicit la True dacă nu se găsește nicio potrivire
        except Exception as e:
            logger.error(f"Error in relevance detection: {str(e)}")
            return True
    
    def _process_clipseg(self, image_np: np.ndarray, 
                        text_prompt: Optional[str] = None) -> Optional[np.ndarray]:
        """
        Procesează segmentarea CLIPSeg
        
        Args:
            image_np: Imaginea de procesat
            text_prompt: Promptul de text pentru segmentare
            
        Returns:
            Masca generată sau None în caz de eroare
        """
        try:
            # Obținem modelul CLIPSeg
            clipseg = self.model_manager.get_model('clipseg')
            if clipseg is None:
                logger.warning("CLIPSeg model not available")
                return None
                
            # Dacă nu avem prompt, folosim "object" ca prompt generic
            if not text_prompt:
                text_prompt = "object"
                
            # Procesăm imaginea
            inputs = clipseg['processor'](
                text=text_prompt,
                images=Image.fromarray(image_np),
                return_tensors="pt",
                padding=True
            )
            
            # Rezolvăm problemele de tip
            input_fixed = {}
            # Implementare FusionFrame 2.0
            # Încercăm să corectăm forma măștii
            if len(mask.shape) > 2:
                mask = mask[0] if mask.shape[0] == 1 else np.mean(mask, axis=0)
            else:
                return None
            
            # Redimensionare sigură
            return self._safe_resize(mask, (image_np.shape[1], image_np.shape[0]))
            
        except Exception as e:
            logger.error(f"Error in CLIPSeg processing: {str(e)}")
            return None
    
    def _process_mediapipe(self, image_np: np.ndarray) -> Optional[np.ndarray]:
        """
        Procesează segmentarea MediaPipe
        
        Args:
            image_np: Imaginea de procesat
            
        Returns:
            Masca generată sau None în caz de eroare
        """
        try:
            # Obținem modelul MediaPipe
            mediapipe = self.model_manager.get_model('mediapipe')
            if mediapipe is None:
                logger.warning("MediaPipe model not available")
                return None
            
            # Procesăm imaginea
            results = mediapipe.process(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))
            
            # Verificăm rezultatele
            if results.segmentation_mask is not None:
                # Convertim la float32 și redimensionăm dacă este necesar
                mask = results.segmentation_mask.astype(np.float32)
                if mask.shape[:2] != image_np.shape[:2]:
                    mask = self._safe_resize(mask, (image_np.shape[1], image_np.shape[0]))
                return mask
            
            return None
            
        except Exception as e:
            logger.error(f"Error in MediaPipe processing: {str(e)}")
            return None
    
    def _process_face_detection(self, image_np: np.ndarray) -> Optional[np.ndarray]:
        """
        Procesează detecția fețelor
        
        Args:
            image_np: Imaginea de procesat
            
        Returns:
            Masca generată sau None în caz de eroare
        """
        try:
            # Obținem modelul de detecție a feței
            face_detector = self.model_manager.get_model('face_detector')
            if face_detector is None:
                logger.warning("Face detector not available")
                return None
            
            # Procesăm imaginea
            rgb_image = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
            results = face_detector.process(rgb_image)
            
            # Verificăm rezultatele
            h, w = image_np.shape[:2]
            face_mask = np.zeros((h, w), dtype=np.float32)
            
            if hasattr(results, 'detections') and results.detections:
                for detection in results.detections:
                    bbox = detection.location_data.relative_bounding_box
                    x, y = int(bbox.xmin * w), int(bbox.ymin * h)
                    width, height = int(bbox.width * w), int(bbox.height * h)
                    
                    # Extindem ușor regiunea feței
                    expansion = 0.2
                    x_expanded = max(0, int(x - width * expansion))
                    y_expanded = max(0, int(y - height * expansion))
                    width_expanded = min(w - x_expanded, int(width * (1 + 2 * expansion)))
                    height_expanded = min(h - y_expanded, int(height * (1 + 2 * expansion)))
                    
                    # Creăm o mască eliptică pentru o selecție mai naturală a feței
                    center = (x_expanded + width_expanded // 2, y_expanded + height_expanded // 2)
                    axes = (width_expanded // 2, height_expanded // 2)
                    cv2.ellipse(face_mask, center, axes, 0, 0, 360, 1.0, -1)
            
            return face_mask
            
        except Exception as e:
            logger.error(f"Error in face detection: {str(e)}")
            return None
    
    def _process_background_detection(self, image_np: np.ndarray) -> Optional[np.ndarray]:
        """
        Creează o mască care separă subiectul de fundal
        
        Args:
            image_np: Imaginea de procesat
            
        Returns:
            Masca generată sau None în caz de eroare
        """
        try:
            h, w = image_np.shape[:2]
            
            # 1. Creăm o mască de fundal de bază conștientă de margini
            gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (15, 15), 0)
            edges = cv2.Canny(blurred, 50, 150)
            
            # Dilatăm marginile pentru a crea limite mai complete
            kernel = np.ones((5, 5), np.uint8)
            edges = cv2.dilate(edges, kernel, iterations=2)
            
            # 2. Folosim grabcut pentru o separare mai avansată a fundalului
            mask = np.zeros(image_np.shape[:2], np.uint8)
            bgd_model = np.zeros((1, 65), np.float64)
            fgd_model = np.zeros((1, 65), np.float64)
            
            # Creăm un dreptunghi care este puțin mai mic decât imaginea
            margin = int(min(w, h) * 0.1)
            rect = (margin, margin, w - margin*2, h - margin*2)
            
            # Aplicăm algoritmul grabcut
            cv2.grabCut(image_np, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
            
            # Creăm masca unde fundalul sigur și probabil sunt setate la 0, altfel 1
            mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('float32')
            
            # Combinăm cu marginile
            combined = mask2 * (1 - np.clip(edges.astype('float32') / 255.0, 0, 1) * 0.7)
            
            return combined
            
        except Exception as e:
            logger.error(f"Error in background detection: {str(e)}")
            
            # Fallback la o mască simplă
            h, w = image_np.shape[:2]
            mask = np.ones((h, w), dtype=np.float32) * 0.5
            margin = int(min(w, h) * 0.1)
            cv2.rectangle(mask, (margin, margin), (w - margin, h - margin), 1.0, -1)
            
            return mask
    
    def _safe_resize(self, image: np.ndarray, size: Tuple[int, int], 
                   interpolation: int = cv2.INTER_LINEAR) -> np.ndarray:
        """
        Redimensionare sigură a imaginii cu multiple fallback-uri
        
        Args:
            image: Imaginea de redimensionat
            size: Dimensiunea țintă (width, height)
            interpolation: Metoda de interpolare
            
        Returns:
            Imaginea redimensionată
        """
        try:
            # Convertim la numpy dacă este PIL
            if isinstance(image, Image.Image):
                image_np = np.array(image)
            else:
                image_np = image
            
            # Verificăm dacă imaginea este validă
            if image_np is None or image_np.size == 0:
                # Creăm o imagine goală de dimensiunea țintă
                if len(size) == 2:
                    return np.zeros((size[1], size[0]), dtype=np.float32)
                return None
            
            # Încercăm redimensionarea cv2
            return cv2.resize(image_np, size, interpolation=interpolation)
            
        except cv2.error as e:
            logger.error(f"OpenCV resize error: {e}")
            
            try:
                # Fallback la PIL
                pil_img = Image.fromarray(image_np)
                resized = pil_img.resize(size, Image.LANCZOS)
                return np.array(resized)
                
            except Exception as e2:
                logger.error(f"PIL resize error: {e2}")
                
                # Creăm o imagine goală de dimensiunea țintă ca ultimă soluție
                if len(image_np.shape) == 3:
                    return np.zeros((size[1], size[0], image_np.shape[2]), dtype=image_np.dtype)
                else:
                    return np.zeros((size[1], size[0]), dtype=image_np.dtype)
                    
        except Exception as e:
            logger.error(f"Unexpected resize error: {e}")
            
            # Creăm o imagine goală de dimensiunea țintă ca ultimă soluție
            return np.zeros((size[1], size[0]), dtype=np.float32)
    
    def refine_mask(self, mask: np.ndarray, image: np.ndarray) -> np.ndarray:
        """
        Rafinare avansată a măștii cu conștientizare a marginilor
        
        Args:
            mask: Masca de rafinat
            image: Imaginea originală pentru analiza marginilor
            
        Returns:
            Masca rafinată
        """
        try:
            # Convertim la float32 pentru procesare
            if mask.max() > 1:
                mask = mask.astype(np.float32) / 255.0
            
            # Detectia marginilor
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            edges = cv2.GaussianBlur(edges, (5, 5), 0) / 255.0
            
            # Combinăm masca cu marginile
            refined = mask * (1 - edges * 0.3) + edges * 0.7
            
            # Convertim la formatul potrivit pentru filtrul bilateral
            refined_uint8 = (refined * 255).astype(np.uint8)
            
            # Aplicăm filtrul bilateral pentru netezire cu păstrarea marginilor
            refined_filtered = cv2.bilateralFilter(refined_uint8, d=9, sigmaColor=75, sigmaSpace=75)
            
            # Threshold
            try:
                adaptive_thresh = cv2.adaptiveThreshold(
                    refined_filtered,
                    255,
                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY,
                    101,
                    3
                )
            except cv2.error:
                # Fallback la threshold simplu dacă cel adaptiv eșuează
                _, adaptive_thresh = cv2.threshold(refined_filtered, 127, 255, cv2.THRESH_BINARY)
            
            # Eliminăm obiectele mici și umplem găurile
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            cleaned = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_OPEN, kernel, iterations=1)
            cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=3)
            
            # Netezire finală
            cleaned = cv2.GaussianBlur(cleaned, (3, 3), 0)
            _, cleaned = cv2.threshold(cleaned, 127, 255, cv2.THRESH_BINARY)
            
            return cleaned
            
        except Exception as e:
            logger.error(f"Error in mask refinement: {str(e)}")
            
            # Returnăm masca originală dacă rafinarea eșuează
            if isinstance(mask, np.ndarray):
                if mask.max() <= 1:
                    return (mask * 255).astype(np.uint8)
                else:
                    return mask.astype(np.uint8)
            else:
                # Creăm o mască goală dacă totul eșuează
                return np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)