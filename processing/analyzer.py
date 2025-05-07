#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Analiză context și conținut imagine pentru FusionFrame 2.0
"""

import re
import cv2
import numpy as np
import logging
from typing import Dict, Any, List, Optional, Union

# Importuri necesare pentru noile funcționalități
import torch
import torch.nn.functional as F
from PIL import Image

from core.model_manager import ModelManager # Asigurăm calea corectă

# Setăm logger-ul
logger = logging.getLogger(__name__)

class OperationAnalyzer:
    """
    Analizator pentru operațiile de editare din prompturi.
    """
    def __init__(self):
        """Inițializează analizatorul de operații"""
        self.operation_patterns = {
             'remove': [
                (r"(remove|delete|erase)\s+(the\s+)?(?P<target>[a-z\s.,0-9'-]+?)(\s+from\s+the\s+image)?$", "remove"),
                (r"eliminate\s+(the\s+)?(?P<target>[a-z\s.,0-9'-]+)", "remove")
             ],
            'replace': [
                (r"(replace|swap|change)\s+(the\s+)?(?P<target>[a-z\s.,0-9'-]+?)\s+with\s+(a\s+)?(?P<attr>[a-z\s.,0-9'-]+)", "replace"),
                (r"substitute\s+(the\s+)?(?P<target>[a-z\s.,0-9'-]+?)\s+for\s+(a\s+)?(?P<attr>[a-z\s.,0-9'-]+)", "replace")
            ],
            'color': [
                (r"(color|recolor|change\s+color)\s+(the\s+)?(?P<target>[a-z\s.,0-9'-]+?)\s+to\s+(?P<attr>[a-z\s'-]+)", "color"),
                (r"make\s+(the\s+)?(?P<target>[a-z\s.,0-9'-]+?)\s+(?P<attr>[a-z\s'-]+)\b", "color") # \b pentru a evita potriviri parțiale la sfârșit
            ],
            'background': [
                (r"(change|alter)\s+(the\s+)?background\s+to\s+(?P<attr>[a-z\s.,0-9'-]+)", "background"),
                (r"new\s+background\s+(of|as)\s+(?P<attr>[a-z\s.,0-9'-]+)", "background"),
                (r"replace\s+(the\s+)?background\s+with\s+(?P<attr>[a-z\s.,0-9'-]+)", "background")
            ],
            'add': [
                (r"(add|place|put|insert)\s+(an?\s+)?(?P<attr>[a-z\s.,0-9'-]+)", "add"),
                (r"(wear|wearing)\s+(an?\s+)?(?P<attr>[a-z\s.,0-9'-]+)", "add")
            ]
        }

    def analyze_operation(self, prompt: str) -> Dict[str, Any]:
        """
        Analizează promptul pentru a determina tipul operației, ținta și atributul.
        """
        prompt_lower = prompt.lower().strip()

        for op_type, patterns_list in self.operation_patterns.items():
            for pattern_tuple in patterns_list:
                pattern_regex = pattern_tuple[0]
                match = re.search(pattern_regex, prompt_lower)
                if match:
                    groups = match.groupdict()
                    return {
                        'type': op_type,
                        'target_object': groups.get('target', '').strip(),
                        'attribute': groups.get('attr', '').strip(),
                        'full_prompt': prompt,
                        'confidence': 0.95
                    }

        general_match_result = self._general_match(prompt_lower)
        if general_match_result:
            general_match_result['full_prompt'] = prompt
            return general_match_result

        # Fallback la operație generală
        return {
            'type': 'general',
            'target_object': '',
            'attribute': prompt_lower, # Folosim întregul prompt ca "instrucțiune"
            'full_prompt': prompt,
            'confidence': 0.5
        }

    def _general_match(self, prompt: str) -> Optional[Dict[str, Any]]:
        """Efectuează o potrivire mai relaxată bazată pe cuvinte cheie."""
        # Simplificare: detectăm doar tipul și extragem textul relevant ca 'attribute' sau 'target'
        if any(k in prompt for k in ['remove', 'delete', 'erase', 'eliminate']):
            match = re.search(r"(?:remove|delete|erase|eliminate)\s+(?:the\s+|an?\s+)?(?P<target>.+)", prompt)
            target = match.group('target').strip() if match else prompt # Fallback la tot promptul
            return {'type': 'remove', 'target_object': target, 'attribute': '', 'confidence': 0.7}
        elif any(k in prompt for k in ['replace', 'swap', 'substitute']):
             match = re.search(r"(?:replace|swap|substitute)\s+(?:the\s+|an?\s+)?(?P<target>.+?)\s+(?:with|for)\s+(?:an?\s+)?(?P<attr>.+)", prompt)
             if match:
                 return {'type': 'replace', 'target_object': match.group('target').strip(), 'attribute': match.group('attr').strip(), 'confidence': 0.7}
        elif any(k in prompt for k in ['color', 'recolor', 'change color', 'make .+ color']):
            match_to = re.search(r"(?:color|recolor|change color)\s+(?:of\s+)?(?:the\s+|an?\s+)?(?P<target>.+?)\s+to\s+(?P<attr>[a-z\s'-]+)", prompt)
            match_make = re.search(r"make\s+(?:the\s+|an?\s+)?(?P<target>.+?)\s+(?P<attr>[a-z\s'-]+)", prompt)
            if match_to:
                 return {'type': 'color', 'target_object': match_to.group('target').strip(), 'attribute': match_to.group('attr').strip(), 'confidence': 0.7}
            if match_make:
                 return {'type': 'color', 'target_object': match_make.group('target').strip(), 'attribute': match_make.group('attr').strip(), 'confidence': 0.65} # Puțin mai puțin sigur
        elif any(k in prompt for k in ['background', 'backdrop', 'scene']):
            match = re.search(r"(?:background|backdrop|scene)\s+(?:to|with|as|of)\s+(?:an?\s+)?(?P<attr>.+)", prompt)
            attr = match.group('attr').strip() if match else prompt # Fallback
            return {'type': 'background', 'target_object': 'background', 'attribute': attr, 'confidence': 0.7}
        elif any(k in prompt for k in ['add', 'place', 'put', 'insert', 'wear']):
            match = re.search(r"(?:add|place|put|insert|wear|wearing)\s+(?:an?\s+)?(?P<attr>.+)", prompt)
            attr = match.group('attr').strip() if match else prompt # Fallback
            return {'type': 'add', 'target_object': '', 'attribute': attr, 'confidence': 0.7}

        return None


class ImageAnalyzer:
    """
    Analizator pentru conținutul și contextul imaginilor.
    """
    def __init__(self):
        """Inițializează analizatorul de imagini și încarcă modelele necesare."""
        self.model_manager = ModelManager()
        self.face_detector = None # Lazy loaded

        # Lazy load pentru modelele auxiliare de analiză
        self.image_classifier_bundle = None
        self.depth_estimator_bundle = None

        # Preia setările din ModelConfig
        self.classifier_settings = getattr(self.model_manager.model_config, "IMAGE_CLASSIFIER_CONFIG", {})
        self.classifier_top_n = self.classifier_settings.get("top_n_results", 5)
        self.depth_estimator_settings = getattr(self.model_manager.model_config, "DEPTH_ESTIMATOR_CONFIG", {})

        logger.info("ImageAnalyzer initialized. Models will be loaded on first use.")

    # --- Lazy Loaders pentru Modele ---
    def _get_face_detector(self):
        """Lazy loader pentru detectorul de fețe MediaPipe."""
        if self.face_detector is None:
            # Folosim ModelManager pentru a încărca și gestiona modelul
            self.face_detector = self.model_manager.get_model('face_detector')
            if self.face_detector is None:
                 logger.warning("MediaPipe Face Detector could not be loaded via ModelManager.")
        return self.face_detector

    def _get_image_classifier(self):
        """Lazy loader pentru clasificatorul de imagini."""
        if self.image_classifier_bundle is None:
            self.image_classifier_bundle = self.model_manager.get_model('image_classifier')
            if self.image_classifier_bundle is None:
                 logger.warning("Image Classifier could not be loaded via ModelManager.")
        return self.image_classifier_bundle

    def _get_depth_estimator(self):
        """Lazy loader pentru estimatorul de adâncime."""
        if self.depth_estimator_bundle is None:
            self.depth_estimator_bundle = self.model_manager.get_model('depth_estimator')
            if self.depth_estimator_bundle is None:
                 logger.warning("Depth Estimator could not be loaded via ModelManager.")
        return self.depth_estimator_bundle

    # --- Metode de Analiză ---
    def _get_scene_classification_tags(self, image_np_rgb: np.ndarray) -> List[Dict[str, Any]]:
        """Clasifică imaginea (așteaptă RGB)."""
        classifier_bundle = self._get_image_classifier()
        if not classifier_bundle: return []

        processor = classifier_bundle.get('processor')
        model = classifier_bundle.get('model')
        if not processor or not model: return []

        try:
            pil_image = Image.fromarray(image_np_rgb)
            inputs = processor(images=pil_image, return_tensors="pt").to(model.device)

            with torch.no_grad():
                outputs = model(**inputs)
            logits = outputs.logits

            probabilities = torch.nn.functional.softmax(logits, dim=-1)
            top_probs, top_indices = torch.topk(probabilities, self.classifier_top_n, dim=-1)

            results = []
            for i in range(min(self.classifier_top_n, top_indices.size(1))): # Asigurăm că nu depășim indecșii
                label_id = top_indices[0, i].item()
                label_name = model.config.id2label.get(label_id, f"unknown_id_{label_id}")
                score = top_probs[0, i].item()
                results.append({'label': label_name, 'score': round(score, 4)})

            logger.debug(f"Image classification results (top {len(results)}): {results}")
            return results
        except Exception as e:
            logger.error(f"Error during scene classification: {e}", exc_info=True)
            return []

    def _get_depth_map(self, image_np_rgb: np.ndarray) -> Optional[np.ndarray]:
        """Estimează harta de adâncime (așteaptă RGB). Returnează float32 0-1 (1=aproape)."""
        depth_bundle = self._get_depth_estimator()
        if not depth_bundle: return None

        processor = depth_bundle.get('processor')
        model = depth_bundle.get('model')
        if not processor or not model: return None

        try:
            original_height, original_width = image_np_rgb.shape[:2]
            pil_image = Image.fromarray(image_np_rgb)
            inputs = processor(images=pil_image, return_tensors="pt").to(model.device, dtype=model.dtype)

            with torch.no_grad():
                outputs = model(**inputs)
            # Verificăm dacă output-ul are atributul așteptat
            if not hasattr(outputs, "predicted_depth"):
                 logger.error("Depth model output does not contain 'predicted_depth'.")
                 return None
                 
            predicted_depth = outputs.predicted_depth

            # Redimensionare la dimensiunea originală folosind interpolate
            prediction = F.interpolate(
                predicted_depth.unsqueeze(1),
                size=(original_height, original_width),
                mode="bicubic", # Poate fi schimbat în "bilinear" dacă bicubic e prea lent/resurse
                align_corners=False,
            ).squeeze()

            # Normalizare (adâncime inversă relativă: 1=aproape, 0=departe)
            depth_min = torch.min(prediction)
            depth_max = torch.max(prediction)
            if depth_max - depth_min > 1e-6:
                normalized_depth = (prediction - depth_min) / (depth_max - depth_min)
            else:
                normalized_depth = torch.zeros_like(prediction)

            depth_map_np = normalized_depth.cpu().numpy().astype(np.float32)
            logger.debug(f"Depth map generated with shape: {depth_map_np.shape}, min: {depth_map_np.min():.2f}, max: {depth_map_np.max():.2f}")
            return depth_map_np
        except Exception as e:
            logger.error(f"Error during depth estimation: {e}", exc_info=True)
            return None

    def analyze_image_context(self, image: Union[Image.Image, np.ndarray]) -> Dict[str, Any]:
        """Analizează complet imaginea."""
        image_np_rgb, image_np_bgr = None, None
        # Conversie și validare input
        try:
            if isinstance(image, Image.Image):
                pil_image_rgb = image.convert("RGB") if image.mode != "RGB" else image
                image_np_rgb = np.array(pil_image_rgb)
                image_np_bgr = cv2.cvtColor(image_np_rgb, cv2.COLOR_RGB2BGR)
            elif isinstance(image, np.ndarray):
                if image.ndim == 2:
                    image_np_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                    image_np_rgb = cv2.cvtColor(image_np_bgr, cv2.COLOR_BGR2RGB)
                elif image.shape[2] == 4:
                    image_np_rgb = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
                    image_np_bgr = cv2.cvtColor(image_np_rgb, cv2.COLOR_RGB2BGR)
                elif image.shape[2] == 3:
                    # Presupunem BGR ca input NumPy standard, dar creăm și RGB
                    image_np_bgr = image
                    image_np_rgb = cv2.cvtColor(image_np_bgr, cv2.COLOR_BGR2RGB)
                else: raise ValueError(f"Unsupported NumPy image shape: {image.shape}")
            else: raise TypeError(f"Unsupported image type: {type(image)}")
        except Exception as e:
             logger.error(f"Error processing input image in analyze_image_context: {e}", exc_info=True)
             return {"error": f"Image processing error: {e}"}

        # Executăm analizele
        lighting_info = self.analyze_lighting(image_np_bgr)
        legacy_scene_type = self.detect_scene_type_heuristic(image_np_bgr) # Renumit pt claritate
        photo_style = self.detect_photo_style_heuristic(image_np_bgr) # Renumit pt claritate
        pixel_stats = self.analyze_pixel_data(image_np_bgr)

        classification_tags = self._get_scene_classification_tags(image_np_rgb)
        depth_map_array = self._get_depth_map(image_np_rgb)

        # Procesăm rezultatele
        primary_scene_tag = classification_tags[0]['label'] if classification_tags else ""
        secondary_scene_tags = [tag['label'] for tag in classification_tags[1:]] if classification_tags else []

        depth_characteristics = None
        if depth_map_array is not None:
             depth_std = np.std(depth_map_array)
             if depth_std < 0.08: depth_characteristics = "relatively_flat" # Ajustat prag
             elif np.mean(depth_map_array > 0.75) > 0.4: depth_characteristics = "dominant_foreground" # Ajustat
             elif depth_std > 0.22: depth_characteristics = "good_fg_bg_separation" # Ajustat
             else: depth_characteristics = "medium_depth_variation"

        desc_parts = []
        if primary_scene_tag: desc_parts.append(f"{primary_scene_tag.replace('_', ' ')}") # Înlocuim underscore din etichete
        elif legacy_scene_type: desc_parts.append(f"{legacy_scene_type}")
        if lighting_info: desc_parts.append(f"({lighting_info} lighting)")
        if photo_style: desc_parts.append(f"({photo_style} style)")
        heuristic_description = " ".join(filter(None,desc_parts)).capitalize()
        if not heuristic_description: heuristic_description = "General image"


        context = {
            'lighting_conditions': {
                'overall_brightness_heuristic': lighting_info,
            },
            'scene_info': {
                'primary_scene_tag_ml': primary_scene_tag,
                'secondary_scene_tags_ml': secondary_scene_tags,
                'legacy_scene_type_heuristic': legacy_scene_type,
                'detected_objects': [], # Placeholder
            },
            'style_and_quality': {
                'visual_style_heuristic': photo_style,
            },
            'spatial_info': {
                'depth_map_available': depth_map_array is not None,
                'depth_map': depth_map_array, # Stocăm array-ul (poate fi mare!)
                'depth_characteristics': depth_characteristics,
            },
            'raw_pixel_stats': pixel_stats,
            'full_description_heuristic': heuristic_description
        }

        return context

    # --- Metode Heuristice (Păstrate ca fallback/comparație) ---
    def analyze_lighting(self, image_np_bgr: np.ndarray) -> str:
        """Analizează euristic iluminarea (așteaptă BGR)."""
        # (Codul rămâne neschimbat față de versiunea anterioară)
        if image_np_bgr.ndim == 3 and image_np_bgr.shape[2] == 3:
            gray = cv2.cvtColor(image_np_bgr, cv2.COLOR_BGR2GRAY)
        elif image_np_bgr.ndim == 2:
            gray = image_np_bgr
        else:
            return "unknown lighting"

        mean_brightness = np.mean(gray)
        brightness_var = np.var(gray)

        if mean_brightness < 70:
            return "dim" if brightness_var < 1800 else "low-key dramatic"
        elif mean_brightness > 190:
            return "bright" if brightness_var > 1200 else "flat bright"
        else:
            return "high contrast" if brightness_var > 2200 else "balanced"


    def detect_scene_type_heuristic(self, image_np_bgr: np.ndarray) -> str:
        """Detectează euristic tipul de scenă (așteaptă BGR)."""
        # (Codul rămâne neschimbat față de versiunea anterioară, dar folosește _get_face_detector)
        face_detector_instance = self._get_face_detector()
        face_detected = False
        face_area_ratio = 0.0

        if face_detector_instance:
            try:
                image_np_rgb = cv2.cvtColor(image_np_bgr, cv2.COLOR_BGR2RGB)
                results = face_detector_instance.process(image_np_rgb)
                if results and hasattr(results, 'detections') and results.detections:
                    face_detected = True
                    for detection in results.detections:
                         if hasattr(detection, 'location_data') and detection.location_data and \
                            hasattr(detection.location_data, 'relative_bounding_box'):
                            bbox = detection.location_data.relative_bounding_box
                            if bbox and hasattr(bbox, 'width') and hasattr(bbox, 'height'):
                                face_area_ratio += bbox.width * bbox.height
            except Exception as e:
                logger.error(f"Error in heuristic face detection: {str(e)}")

        hsv = cv2.cvtColor(image_np_bgr, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        mean_s = np.mean(s)

        if face_detected and face_area_ratio > 0.1:
            return "portrait"
        elif mean_s < 60:
            return "indoor or low saturation outdoor"
        else:
            return "outdoor"

    def detect_photo_style_heuristic(self, image_np_bgr: np.ndarray) -> str:
        """Detectează euristic stilul fotografic (așteaptă BGR)."""
        # (Codul rămâne neschimbat față de versiunea anterioară)
        hsv = cv2.cvtColor(image_np_bgr, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        std_h = np.std(h)
        mean_s = np.mean(s)
        mean_v = np.mean(v)
        std_v = np.std(v)

        if std_h < 25 and mean_s < 50:
            return "minimalist or desaturated"
        elif std_v > 65:
            return "high contrast"
        elif mean_v > 195:
            return "bright and airy"
        elif mean_v < 85:
            return "moody or dark"
        elif mean_s > 100 and std_h > 40 :
            return "vibrant"
        else:
            return "natural"

    def analyze_pixel_data(self, image_np_bgr: np.ndarray) -> Dict[str, Any]:
        """Analizează statisticile pixelilor (așteaptă BGR)."""
        # (Codul rămâne neschimbat față de versiunea anterioară)
        if image_np_bgr.ndim != 3 or image_np_bgr.shape[2] != 3 :
            if image_np_bgr.ndim == 2:
                gray_image = image_np_bgr
                channels = 1
                channel_means = [float(np.mean(gray_image))]
                channel_stds = [float(np.std(gray_image))]
                hist_data = [cv2.calcHist([gray_image], [0], None, [256], [0, 256]).flatten().tolist()]
            else:
                logger.warning(f"analyze_pixel_data received unexpected image shape: {image_np_bgr.shape}")
                return {'channels': 0, 'mean_value': 0, 'std_value': 0, 'histogram': None}
        else:
            channels = image_np_bgr.shape[2]
            channel_means = [float(np.mean(image_np_bgr[:, :, i])) for i in range(channels)]
            channel_stds = [float(np.std(image_np_bgr[:, :, i])) for i in range(channels)]

            hist_data = []
            colors = ('b', 'g', 'r')
            for i, col in enumerate(colors):
                hist = cv2.calcHist([image_np_bgr], [i], None, [256], [0, 256])
                hist_data.append(hist.flatten().tolist())

        return {
            'channels': channels,
            'channel_means_bgr_or_gray': channel_means,
            'channel_stds_bgr_or_gray': channel_stds,
            'histogram_bgr_or_gray': hist_data
        }