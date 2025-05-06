#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Analiză context și conținut imagine pentru FusionFrame 2.0
"""

import re
import cv2
import numpy as np
import logging
from typing import Dict, Any, List, Optional, Union, Tuple
from PIL import Image

# Setăm logger-ul
logger = logging.getLogger(__name__)

class OperationAnalyzer:
    """
    Analizator pentru operațiile de editare
    
    Responsabil pentru interpretarea prompturilor și clasificarea
    operațiilor de editare pentru a selecta pipeline-ul adecvat.
    """
    
    def __init__(self):
        """Inițializează analizatorul de operații"""
        # Maparea regex-urilor pentru tipuri de operații
        self.operation_patterns = {
            'remove': [
                (r"(remove|delete|erase)\s+(the\s+)?(?P<target>[a-z\s]+?)(\s+from\s+the\s+image)?$", "remove"),
                (r"(remove|delete|erase)\s+(the\s+)?(?P<target>[a-z\s]+?)(\s+from\s+the\s+image)?$", "remove"),
                (r"eliminate\s+(the\s+)?(?P<target>[a-z\s]+)", "remove")
             ],
            'replace': [
                (r"(replace|swap|change)\s+(the\s+)?(?P<target>[a-z\s]+?)\s+with\s+(a\s+)?(?P<attr>[a-z\s]+)", "replace"),
                (r"substitute\s+(the\s+)?(?P<target>[a-z\s]+?)\s+for\s+(a\s+)?(?P<attr>[a-z\s]+)", "replace")
            ],
            'color': [
                (r"(color|recolor|change\s+color)\s+(the\s+)?(?P<target>[a-z\s]+?)\s+to\s+(?P<attr>[a-z]+)", "color"),
                (r"make\s+(the\s+)?(?P<target>[a-z\s]+?)\s+(?P<attr>[a-z]+)", "color")
            ],
            'background': [
                (r"(change|alter)\s+(the\s+)?background\s+to\s+(?P<attr>[a-z\s]+)", "background"),
                (r"new\s+background\s+(of|as)\s+(?P<attr>[a-z\s]+)", "background"),
                (r"replace\s+(the\s+)?background\s+with\s+(?P<attr>[a-z\s]+)", "background")
            ],
            'add': [
                (r"(add|place|put)\s+(a\s+)?(?P<attr>[a-z\s]+)", "add"),
                (r"(wear|wearing)\s+(a\s+)?(?P<attr>[a-z\s]+)", "add")
            ]
                   
       }

def analyze_operation(self, prompt: str) -> Dict[str, Any]:
    """
    Analizează promptul pentru a determina tipul operației
    
    Args:
        prompt: Promptul de analizat
        
    Returns:
        Dicționar cu informații despre operație
    """
    # Normalizăm promptul
    prompt_lower = prompt.lower().strip()
    
    # Iterăm prin fiecare tip de operație și pattern-urile asociate
    for op_type, patterns in self.operation_patterns.items():
        for pattern, match_type in patterns:
            match = re.search(pattern, prompt_lower)
            if match:
                groups = match.groupdict()
                return {
                    'type': op_type,
                    'target': groups.get('target', '').strip(),
                    'attribute': groups.get('attr', '').strip(),
                    'confidence': 0.95
                }
    
    # Dacă nu am găsit o potrivire exactă, facem o potrivire mai generală
    general_match = self._general_match(prompt_lower)
    if general_match:
        return general_match
    
    # Returnăm o operație generală dacă nu putem determina tipul
    return {
        'type': 'general',
        'target': '',
        'attribute': prompt_lower,
        'confidence': 0.5
    }

def _general_match(self, prompt: str) -> Optional[Dict[str, Any]]:
    """
    Efectuează o potrivire mai relaxată pentru prompt
    
    Args:
        prompt: Promptul de analizat
        
    Returns:
        Dicționar cu informații despre operație sau None
    """
    # Cuvinte cheie pentru fiecare tip de operație
    keywords = {
        'remove': ['remove', 'delete', 'erase', 'eliminate', 'get rid', 'take out'],
        'replace': ['replace', 'swap', 'change', 'substitute', 'switch'],
        'color': ['color', 'recolor', 'hue', 'tint', 'shade'],
        'background': ['background', 'backdrop', 'scene', 'setting', 'environment'],
        'add': ['add', 'place', 'put', 'insert', 'include', 'attach', 'wear', 'glasses']
    }
    
    # Verificăm fiecare set de cuvinte cheie
    for op_type, key_words in keywords.items():
        for word in key_words:
            if word in prompt:
                # Încercăm să extragem ținta și atributul
                target = ''
                attribute = ''
                
                if op_type == 'remove':
                    # Pentru remove, încercăm să găsim ce urmează după cuvântul cheie
                    match = re.search(f"{word}\\s+(the\\s+)?([a-z\\s]+)", prompt)
                    if match:
                        target = match.group(2).strip()
                elif op_type == 'color':
                    # Pentru color, încercăm să găsim obiectul și culoarea
                    match = re.search(f"{word}\\s+(the\\s+)?([a-z\\s]+)\\s+to\\s+([a-z\\s]+)", prompt)
                    if match:
                        target = match.group(2).strip()
                        attribute = match.group(3).strip()
                elif op_type == 'replace' or op_type == 'background':
                    # Pentru replace/background, încercăm să găsim "with" sau "to"
                    match = re.search(f"(with|to)\\s+([a-z\\s]+)", prompt)
                    if match:
                        attribute = match.group(2).strip()
                elif op_type == 'add':
                    # Pentru add, tot ce urmează după cuvântul cheie
                    match = re.search(f"{word}\\s+(a\\s+)?([a-z\\s]+)", prompt)
                    if match:
                        attribute = match.group(2).strip()
                
                return {
                    'type': op_type,
                    'target': target,
                    'attribute': attribute,
                    'confidence': 0.7
                }
    
    return None
class ImageAnalyzer:
"""
Analizator pentru imaginile de intrare
Responsabil pentru analiza conținutului și contextului imaginilor
pentru a îmbunătăți editarea și generarea de prompturi.
"""

def __init__(self):
    """Inițializează analizatorul de imagini"""
    self.face_detector = None
    self.person_detector = None

def analyze_image_context(self, image: Union[Image.Image, np.ndarray]) -> Dict[str, Any]:
    """
    Analizează contextul imaginii pentru o mai bună formulare a prompturilor
    
    Args:
        image: Imaginea de analizat
        
    Returns:
        Dicționar cu informații despre context
    """
    # Convertim la numpy array dacă este PIL
    if isinstance(image, Image.Image):
        image_np = np.array(image)
    else:
        image_np = image
        
    # Analizăm iluminarea
    lighting = self.analyze_lighting(image_np)
    
    # Analizăm tipul de scenă
    scene_type = self.detect_scene_type(image_np)
    
    # Analizăm stilul
    style = self.detect_photo_style(image_np)
    
    # Analizăm detaliile la nivel de pixel
    pixel_info = self.analyze_pixel_data(image_np)
    
    # Combinăm în descrierea contextului
    context = {
        'lighting': lighting,
        'scene_type': scene_type,
        'style': style,
        'pixel_data': pixel_info,
        'description': f"{scene_type} scene with {lighting} lighting in {style} style"
    }
    
    return context

def analyze_lighting(self, image_np: np.ndarray) -> str:
    """
    Analizează condițiile de iluminare din imagine
    
    Args:
        image_np: Imaginea de analizat în format numpy
        
    Returns:
        Descrierea iluminării
    """
    # Convertim la tonuri de gri
    if len(image_np.shape) == 3:
        gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    else:
        gray = image_np
    
    # Calculăm luminozitatea medie
    mean_brightness = np.mean(gray)
    
    # Calculăm variația luminozității
    brightness_var = np.var(gray)
    
    # Determinăm tipul de iluminare bazat pe luminozitate și variație
    if mean_brightness < 80:
        if brightness_var > 1500:
            return "low-key dramatic"
        else:
            return "dim"
    elif mean_brightness > 180:
        if brightness_var < 1000:
            return "flat bright"
        else:
            return "bright"
    else:
        if brightness_var > 2000:
            return "high contrast"
        else:
            return "balanced"

def detect_scene_type(self, image_np: np.ndarray) -> str:
    """
    Detectează dacă scena este interior, exterior, portret, etc.
    
    Args:
        image_np: Imaginea de analizat în format numpy
        
    Returns:
        Tipul de scenă
    """
    # Inițializăm detectorul de fețe dacă nu există
    if self.face_detector is None:
        try:
            import mediapipe as mp
            self.face_detector = mp.solutions.face_detection.FaceDetection(
                min_detection_confidence=0.5
            )
        except ImportError:
            logger.warning("MediaPipe not available for face detection")
    
    # Verificăm prezența feței
    face_detected = False
    face_area_ratio = 0.0
    
    if self.face_detector is not None:
        try:
            results = self.face_detector.process(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))
            if results and hasattr(results, 'detections') and results.detections:
                face_detected = True
                # Calculăm raportul ariei feței
                for detection in results.detections:
                    bbox = detection.location_data.relative_bounding_box
                    face_area_ratio = bbox.width * bbox.height
        except Exception as e:
            logger.error(f"Error in face detection: {str(e)}")
    
    # Euristică simplă bazată pe distribuțiile de culori
    hsv = cv2.cvtColor(image_np, cv2.COLOR_BGR2HSV)
    
    # Extragem canalele
    h, s, v = cv2.split(hsv)
    
    # Calculăm saturația și valoarea medie
    mean_s = np.mean(s)
    mean_v = np.mean(v)
    
    # Determinăm tipul scenei
    if face_detected and face_area_ratio > 0.15:
        return "portrait"
    elif mean_s < 50:  # Saturație scăzută indică deseori scene de interior
        return "indoor"
    else:
        return "outdoor"

def detect_photo_style(self, image_np: np.ndarray) -> str:
    """
    Detectează stilul fotografic al imaginii
    
    Args:
        image_np: Imaginea de analizat în format numpy
        
    Returns:
        Stilul fotografic
    """
    # Euristică simplă bazată pe caracteristicile de culoare
    hsv = cv2.cvtColor(image_np, cv2.COLOR_BGR2HSV)
    
    # Extragem canalele
    h, s, v = cv2.split(hsv)
    
    # Calculăm metrici
    mean_s = np.mean(s)
    std_h = np.std(h)
    mean_v = np.mean(v)
    std_v = np.std(v)
    
    # Determinăm stilul bazat pe metrici
    if std_h < 20 and mean_s < 40:
        return "minimalist"
    elif std_v > 60:
        return "high contrast"
    elif mean_v > 200:
        return "bright and airy"
    elif mean_v < 100:
        return "moody"
    else:
        return "natural"

def analyze_pixel_data(self, image_np: np.ndarray) -> Dict[str, Any]:
    """
    Analizează datele la nivel de pixel pentru a obține informații statistice
    
    Args:
        image_np: Imaginea de analizat în format numpy
        
    Returns:
        Dicționar cu informații statistice despre pixeli
    """
    # Asigurăm că imaginea este în format BGR sau RGB
    if len(image_np.shape) != 3:
        return {
            'channels': 1,
            'mean_value': float(np.mean(image_np)),
            'std_value': float(np.std(image_np)),
            'histogram': None
        }
    
    # Calculăm statistici pentru fiecare canal
    channels = image_np.shape[2]
    channel_means = [float(np.mean(image_np[:, :, i])) for i in range(channels)]
    channel_stds = [float(np.std(image_np[:, :, i])) for i in range(channels)]
    
    # Calculăm histograma pentru fiecare canal
    hist_data = []
    for i in range(channels):
        hist = cv2.calcHist([image_np], [i], None, [256], [0, 256])
        hist = hist.flatten().tolist()
        hist_data.append(hist)
    
    return {
        'channels': channels,
        'channel_means': channel_means,
        'channel_stds': channel_stds,
        'histogram': hist_data
    }