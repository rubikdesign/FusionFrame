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

import torch
import torch.nn.functional as F
from PIL import Image

from core.model_manager import ModelManager

# Setăm logger-ul
logger = logging.getLogger(__name__)

# --- Clasa OperationAnalyzer (Rămâne neschimbată față de versiunea anterioară) ---
class OperationAnalyzer:
    """Analizator pentru operațiile de editare din prompturi."""
    def __init__(self):
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
                (r"make\s+(the\s+)?(?P<target>[a-z\s.,0-9'-]+?)\s+(?P<attr>[a-z\s'-]+)\b", "color")
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
        return {
            'type': 'general', 'target_object': '', 'attribute': prompt_lower,
            'full_prompt': prompt, 'confidence': 0.5
        }

    def _general_match(self, prompt: str) -> Optional[Dict[str, Any]]:
        make_color_match = re.search(r"make\s+(the\s+)?(?P<target>.+?)\s+(?P<attr>[a-z\s'-]+)\s*$", prompt)
        if make_color_match:
            target = make_color_match.group('target').strip()
            possible_color = make_color_match.group('attr').strip()
            if "remove" not in prompt and "add" not in prompt and "replace" not in prompt and "background" not in prompt:
                 return {'type': 'color', 'target_object': target, 'attribute': possible_color, 'confidence': 0.75}

        keywords = {
            'remove': ['remove', 'delete', 'erase', 'eliminate', 'get rid of', 'take out'],
            'replace': ['replace', 'swap', 'change with', 'substitute with', 'switch for'],
            'color': ['color of', 'recolor', 'change color of', 'hue of', 'tint of', 'shade of'],
            'background': ['background to', 'backdrop as', 'scene with', 'setting for', 'environment of', 'replace background with'],
            'add': ['add a', 'add an', 'place a', 'place an', 'put a', 'put an', 'insert a', 'insert an', 'include a', 'include an', 'attach a', 'attach an', 'wear a', 'wearing a']
        }
        for op_type, key_phrases in keywords.items():
             for phrase_base in key_phrases:
                 # Logica de potrivire generală (simplificată aici pentru lizibilitate, codul anterior era OK)
                 if phrase_base in prompt:
                     # Încercare simplă de extracție (poate fi îmbunătățită)
                     relevant_text = prompt.split(phrase_base, 1)[-1].strip()
                     if op_type == 'remove':
                         return {'type': op_type, 'target_object': relevant_text, 'attribute': '', 'confidence': 0.7}
                     else: # add, color, replace, background
                         return {'type': op_type, 'target_object': '', 'attribute': relevant_text, 'confidence': 0.7}
        return None

# --- Clasa ImageAnalyzer (Actualizată) ---
class ImageAnalyzer:
    """Analizator pentru conținutul și contextul imaginilor."""
    def __init__(self):
        self.model_manager = ModelManager()
        self.face_detector = None
        self.image_classifier_bundle = None
        self.depth_estimator_bundle = None
        self.object_detector = None # NOU: Pentru YOLO

        # Preia setările din ModelConfig
        self.classifier_settings = getattr(self.model_manager.model_config, "IMAGE_CLASSIFIER_CONFIG", {})
        self.classifier_top_n = self.classifier_settings.get("top_n_results", 5)
        self.depth_estimator_settings = getattr(self.model_manager.model_config, "DEPTH_ESTIMATOR_CONFIG", {})
        # NOU: Preia setările pentru detectorul de obiecte
        self.object_detector_settings = getattr(self.model_manager.model_config, "OBJECT_DETECTOR_CONFIG", {})
        self.object_detector_conf_thresh = self.object_detector_settings.get("confidence_threshold", 0.4)

        logger.info("ImageAnalyzer initialized. Models will be loaded on first use.")

    # --- Lazy Loaders pentru Modele ---
    def _get_face_detector(self):
        if self.face_detector is None:
            self.face_detector = self.model_manager.get_model('face_detector')
            if self.face_detector is None: logger.warning("MediaPipe Face Detector could not be loaded.")
        return self.face_detector

    def _get_image_classifier(self):
        if self.image_classifier_bundle is None:
            self.image_classifier_bundle = self.model_manager.get_model('image_classifier')
            if self.image_classifier_bundle is None: logger.warning("Image Classifier could not be loaded.")
        return self.image_classifier_bundle

    def _get_depth_estimator(self):
        if self.depth_estimator_bundle is None:
            self.depth_estimator_bundle = self.model_manager.get_model('depth_estimator')
            if self.depth_estimator_bundle is None: logger.warning("Depth Estimator could not be loaded.")
        return self.depth_estimator_bundle

    # NOU: Lazy Loader pentru Detectorul de Obiecte (YOLO)
    def _get_object_detector(self):
        if self.object_detector is None:
             # Presupunem că YOLO este încărcat sub cheia 'yolo'
            self.object_detector = self.model_manager.get_model('yolo')
            if self.object_detector is None:
                 logger.warning("YOLO Object Detector could not be loaded.")
        return self.object_detector


    # --- Metode de Analiză ---

    def _get_scene_classification_tags(self, image_np_rgb: np.ndarray) -> List[Dict[str, Any]]:
        """Clasifică imaginea (așteaptă RGB)."""
        # (Codul metodei rămâne neschimbat față de versiunea anterioară)
        classifier_bundle = self._get_image_classifier()
        if not classifier_bundle: return []
        processor = classifier_bundle.get('processor')
        model = classifier_bundle.get('model')
        if not processor or not model: return []
        try:
            pil_image = Image.fromarray(image_np_rgb)
            inputs = processor(images=pil_image, return_tensors="pt").to(model.device)
            with torch.no_grad(): outputs = model(**inputs)
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=-1)
            top_probs, top_indices = torch.topk(probabilities, self.classifier_top_n, dim=-1)
            results = []
            for i in range(min(self.classifier_top_n, top_indices.size(1))):
                label_id = top_indices[0, i].item()
                label_name = model.config.id2label.get(label_id, f"unknown_id_{label_id}")
                score = top_probs[0, i].item()
                results.append({'label': label_name, 'score': round(score, 4)})
            logger.debug(f"Image classification results: {results}")
            return results
        except Exception as e:
            logger.error(f"Error during scene classification: {e}", exc_info=True)
            return []


    def _get_depth_map(self, image_np_rgb: np.ndarray) -> Optional[np.ndarray]:
        """Estimează harta de adâncime (așteaptă RGB). Returnează float32 0-1 (1=aproape)."""
        # (Codul metodei rămâne neschimbat față de versiunea anterioară)
        depth_bundle = self._get_depth_estimator()
        if not depth_bundle: return None
        processor = depth_bundle.get('processor')
        model = depth_bundle.get('model')
        if not processor or not model: return None
        try:
            original_height, original_width = image_np_rgb.shape[:2]
            pil_image = Image.fromarray(image_np_rgb)
            inputs = processor(images=pil_image, return_tensors="pt").to(model.device, dtype=model.dtype)
            with torch.no_grad(): outputs = model(**inputs)
            if not hasattr(outputs, "predicted_depth"): return None
            predicted_depth = outputs.predicted_depth
            prediction = F.interpolate(
                predicted_depth.unsqueeze(1), size=(original_height, original_width),
                mode="bicubic", align_corners=False,
            ).squeeze()
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

    # NOU: Metodă pentru Detecția de Obiecte
    def _get_detected_objects(self, image_np_bgr: np.ndarray) -> List[Dict[str, Any]]:
        """Detectează obiecte folosind YOLO (așteaptă BGR)."""
        yolo_model = self._get_object_detector()
        if not yolo_model:
            logger.debug("YOLO model not available for object detection.")
            return []

        detected_objects = []
        try:
            # Rulăm predicția YOLO (poate necesita ajustarea parametrilor: imgsz, conf)
            # Folosim stream=False pentru a obține rezultatele direct
            results = yolo_model.predict(
                source=image_np_bgr, # YOLO preferă BGR NumPy
                stream=False,
                conf=self.object_detector_conf_thresh, # Folosim pragul din config
                iou=self.object_detector_settings.get("iou_threshold", 0.5), # Folosim prag IoU pentru NMS
                verbose=False # Reducem output-ul YOLO în consolă
            )

            # Procesăm rezultatele (presupunând formatul ultralytics YOLOv8)
            if results and isinstance(results, list):
                 # Pentru predict pe o singură imagine, results este o listă cu un singur element Results
                 res = results[0]
                 if hasattr(res, 'boxes') and res.boxes is not None:
                     boxes = res.boxes.data.cpu().numpy() # Tensor to NumPy [x1, y1, x2, y2, conf, cls]
                     class_names = getattr(res, 'names', None) # Obținem numele claselor din obiectul Results

                     if class_names is None: # Fallback dacă names nu sunt direct în results
                         if hasattr(yolo_model, 'names'):
                             class_names = yolo_model.names
                         else:
                             logger.warning("Could not retrieve class names from YOLO model.")
                             class_names = {} # Folosim un dicționar gol

                     for box in boxes:
                         x1, y1, x2, y2, conf, cls_id = box
                         label = class_names.get(int(cls_id), f"class_{int(cls_id)}")
                         
                         # Coordonate normalizate (YOLOv8 poate returna diverse formate, boxes.data dă de obicei xyxy)
                         # Verificăm dacă sunt deja normalizate sau nu
                         h, w = image_np_bgr.shape[:2]
                         if x1 > 1.0 or y1 > 1.0 or x2 > 1.0 or y2 > 1.0: # Probabil coordonate absolute
                            xn1, yn1, xn2, yn2 = x1/w, y1/h, x2/w, y2/h
                         else: # Probabil deja normalizate
                            xn1, yn1, xn2, yn2 = x1, y1, x2, y2
                             
                         detected_objects.append({
                             'label': label,
                             'confidence': round(float(conf), 4),
                             'box_normalized': [round(float(coord), 4) for coord in [xn1, yn1, xn2, yn2]] # [x_min, y_min, x_max, y_max] normalizat
                         })

            logger.debug(f"Detected {len(detected_objects)} objects with conf >= {self.object_detector_conf_thresh}")
            return detected_objects

        except AttributeError as ae:
             logger.error(f"YOLO results format might have changed or model is not loaded correctly: {ae}", exc_info=True)
             return []
        except Exception as e:
            logger.error(f"Error during object detection: {e}", exc_info=True)
            return []

    # NOU: Metodă pentru analiza îmbunătățită a iluminării
    def analyze_lighting_advanced(self, image_np_bgr: np.ndarray) -> Dict[str, Any]:
        """Analiză mai detaliată a iluminării (așteaptă BGR)."""
        results = {
            'brightness_heuristic': "unknown",
            'temperature_heuristic': "neutral",
            'highlights_pct': 0.0,
            'shadows_pct': 0.0,
            'contrast_heuristic': "medium" # Adăugat
        }
        if image_np_bgr.ndim != 3 or image_np_bgr.shape[2] != 3:
            logger.warning("analyze_lighting_advanced requires a 3-channel BGR image.")
            return results

        try:
            # 1. Luminozitate și Contrast (similar cu metoda veche, dar stocat separat)
            gray = cv2.cvtColor(image_np_bgr, cv2.COLOR_BGR2GRAY)
            mean_brightness = np.mean(gray)
            std_dev = np.std(gray) # Deviația standard e un indicator de contrast

            if mean_brightness < 70: results['brightness_heuristic'] = "dim"
            elif mean_brightness > 190: results['brightness_heuristic'] = "bright"
            else: results['brightness_heuristic'] = "balanced"

            # Praguri euristice pentru contrast bazate pe deviația standard
            if std_dev < 30: results['contrast_heuristic'] = "low"
            elif std_dev > 65: results['contrast_heuristic'] = "high"
            else: results['contrast_heuristic'] = "medium"
            # Combinăm cu evaluarea anterioară pt cazuri specifice
            if results['brightness_heuristic'] == "dim" and results['contrast_heuristic'] == "high":
                results['brightness_heuristic'] = "low-key dramatic"
            elif results['brightness_heuristic'] == "bright" and results['contrast_heuristic'] == "low":
                 results['brightness_heuristic'] = "flat bright"


            # 2. Temperatură de Culoare (Euristică BGR Mean)
            mean_b = np.mean(image_np_bgr[:, :, 0])
            mean_g = np.mean(image_np_bgr[:, :, 1]) # Ignorat momentan
            mean_r = np.mean(image_np_bgr[:, :, 2])

            # Comparație simplă Blue vs Red
            if mean_b > mean_r * 1.05: # Dacă albastru e semnificativ mai mare
                results['temperature_heuristic'] = "cool"
            elif mean_r > mean_b * 1.05: # Dacă roșu e semnificativ mai mare
                results['temperature_heuristic'] = "warm"
            else:
                results['temperature_heuristic'] = "neutral"

            # 3. Procent Highlights și Shadows (din histograma grayscale)
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            total_pixels = gray.size
            highlight_threshold = 245 # Prag pentru pixeli foarte luminoși
            shadow_threshold = 10   # Prag pentru pixeli foarte întunecați

            highlight_pixels = np.sum(hist[highlight_threshold:])
            shadow_pixels = np.sum(hist[:shadow_threshold])

            results['highlights_pct'] = round((highlight_pixels / total_pixels) * 100, 2)
            results['shadows_pct'] = round((shadow_pixels / total_pixels) * 100, 2)

            logger.debug(f"Advanced lighting analysis: {results}")

        except Exception as e:
            logger.error(f"Error during advanced lighting analysis: {e}", exc_info=True)

        return results


    def analyze_image_context(self, image: Union[Image.Image, np.ndarray]) -> Dict[str, Any]:
        """Analizează complet imaginea, incluzând noile funcționalități."""
        image_np_rgb, image_np_bgr = None, None
        try:
            # (Logica de conversie imagine rămâne aceeași ca în versiunea anterioară)
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
                    image_np_bgr = image
                    image_np_rgb = cv2.cvtColor(image_np_bgr, cv2.COLOR_BGR2RGB)
                else: raise ValueError(f"Unsupported NumPy image shape: {image.shape}")
            else: raise TypeError(f"Unsupported image type: {type(image)}")
        except Exception as e:
             logger.error(f"Error processing input image in analyze_image_context: {e}", exc_info=True)
             return {"error": f"Image processing error: {e}"}

        # --- Executăm Toate Analizele ---
        logger.debug("Starting advanced image context analysis...")
        start_time = time.time() # Măsurăm timpul de analiză

        # Analize ML (pot fi mai lente)
        classification_tags = self._get_scene_classification_tags(image_np_rgb)
        depth_map_array = self._get_depth_map(image_np_rgb)
        detected_objects = self._get_detected_objects(image_np_bgr) # YOLO folosește BGR

        # Analize CV (relativ rapide)
        lighting_info_adv = self.analyze_lighting_advanced(image_np_bgr)
        legacy_scene_type = self.detect_scene_type_heuristic(image_np_bgr)
        photo_style = self.detect_photo_style_heuristic(image_np_bgr)
        pixel_stats = self.analyze_pixel_data(image_np_bgr)

        analysis_time = time.time() - start_time
        logger.debug(f"Image analysis completed in {analysis_time:.2f} seconds.")

        # --- Procesăm și Structurăm Rezultatele ---
        primary_scene_tag = classification_tags[0]['label'] if classification_tags else ""
        secondary_scene_tags = [tag['label'] for tag in classification_tags[1:]] if classification_tags else []

        depth_characteristics = None
        if depth_map_array is not None:
             depth_std = np.std(depth_map_array)
             if depth_std < 0.08: depth_characteristics = "relatively_flat"
             elif np.mean(depth_map_array > 0.75) > 0.4: depth_characteristics = "dominant_foreground"
             elif depth_std > 0.22: depth_characteristics = "good_fg_bg_separation"
             else: depth_characteristics = "medium_depth_variation"

        # Construim descrierea euristică finală
        desc_parts = []
        if primary_scene_tag: desc_parts.append(f"{primary_scene_tag.replace('_', ' ')}")
        elif legacy_scene_type: desc_parts.append(f"{legacy_scene_type}")
        # Adăugăm obiectele detectate la descriere (opțional)
        if detected_objects:
             obj_names = [obj['label'] for obj in detected_objects[:2]] # Primele 2 obiecte
             if obj_names: desc_parts.append(f"featuring {', '.join(obj_names)}")
        # Adăugăm info despre iluminare/stil
        lighting_desc = lighting_info_adv.get('brightness_heuristic', '')
        temp_desc = lighting_info_adv.get('temperature_heuristic', '')
        if lighting_desc and lighting_desc != 'unknown': desc_parts.append(f"({lighting_desc} lighting)")
        if temp_desc and temp_desc != 'neutral': desc_parts.append(f"({temp_desc} tones)")
        if photo_style: desc_parts.append(f"({photo_style} style)")
        heuristic_description = " ".join(filter(None,desc_parts)).capitalize()
        if not heuristic_description: heuristic_description = "General image"


        # --- Dicționarul Final de Context ---
        context = {
            'analysis_time_sec': round(analysis_time, 2),
            'lighting_conditions': lighting_info_adv, # Folosim noua analiză detaliată
            'scene_info': {
                'primary_scene_tag_ml': primary_scene_tag,
                'secondary_scene_tags_ml': secondary_scene_tags,
                'legacy_scene_type_heuristic': legacy_scene_type,
                'detected_objects': detected_objects, # Adăugăm lista obiectelor detectate
            },
            'style_and_quality': {
                'visual_style_heuristic': photo_style,
                # Aici se pot adăuga analize de culoare dominantă, zgomot, claritate
            },
            'spatial_info': {
                'depth_map_available': depth_map_array is not None,
                'depth_map': depth_map_array,
                'depth_characteristics': depth_characteristics,
            },
            'raw_pixel_stats': pixel_stats,
            'full_description_heuristic': heuristic_description
        }

        return context

    # --- Metode Heuristice (Păstrate ca fallback/comparație) ---
    def detect_scene_type_heuristic(self, image_np_bgr: np.ndarray) -> str:
        # (Codul rămâne neschimbat față de versiunea anterioară)
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
                         if hasattr(detection, 'location_data') and hasattr(detection.location_data, 'relative_bounding_box'):
                            bbox = detection.location_data.relative_bounding_box
                            if bbox and hasattr(bbox, 'width') and hasattr(bbox, 'height'):
                                face_area_ratio += bbox.width * bbox.height
            except Exception as e: logger.error(f"Error in heuristic face detection: {e}")
        hsv = cv2.cvtColor(image_np_bgr, cv2.COLOR_BGR2HSV)
        mean_s = np.mean(hsv[:,:,1])
        if face_detected and face_area_ratio > 0.1: return "portrait"
        elif mean_s < 60: return "indoor or low saturation outdoor"
        else: return "outdoor"

    def detect_photo_style_heuristic(self, image_np_bgr: np.ndarray) -> str:
        # (Codul rămâne neschimbat față de versiunea anterioară)
        hsv = cv2.cvtColor(image_np_bgr, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        std_h, mean_s, mean_v, std_v = np.std(h), np.mean(s), np.mean(v), np.std(v)
        if std_h < 25 and mean_s < 50: return "minimalist or desaturated"
        elif std_v > 65: return "high contrast"
        elif mean_v > 195: return "bright and airy"
        elif mean_v < 85: return "moody or dark"
        elif mean_s > 100 and std_h > 40 : return "vibrant"
        else: return "natural"

    def analyze_pixel_data(self, image_np_bgr: np.ndarray) -> Dict[str, Any]:
        # (Codul rămâne neschimbat față de versiunea anterioară)
        if image_np_bgr.ndim != 3 or image_np_bgr.shape[2] != 3 :
            if image_np_bgr.ndim == 2:
                gray_image = image_np_bgr; channels = 1
                means = [float(np.mean(gray_image))]; stds = [float(np.std(gray_image))]
                hist = [cv2.calcHist([gray_image], [0], None, [256], [0, 256]).flatten().tolist()]
            else: return {'channels': 0, 'mean_value': 0, 'std_value': 0, 'histogram': None}
        else:
            channels = 3; means=[]; stds=[]; hist=[]
            colors = ('b', 'g', 'r')
            for i, col in enumerate(colors):
                means.append(float(np.mean(image_np_bgr[:, :, i])))
                stds.append(float(np.std(image_np_bgr[:, :, i])))
                h = cv2.calcHist([image_np_bgr], [i], None, [256], [0, 256])
                hist.append(h.flatten().tolist())
        return {'channels': channels, 'channel_means_bgr_or_gray': means,
                'channel_stds_bgr_or_gray': stds, 'histogram_bgr_or_gray': hist}