#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
MaskGenerator v3.2 — Inteligent, operation-aware mask generation
for FusionFrame. Includes more specific rules, refined CLIPSeg prompting,
and placeholders for Segment Anything Model (SAM) integration.
"""

import cv2
import numpy as np
import logging
import torch 
from typing import Dict, Any, Union, Callable, Optional, List, Tuple
from PIL import Image
import re # Pentru expresii regulate în extracția de cuvinte cheie

try:
    from config.app_config import AppConfig
except ImportError:
    print("WARNING: AppConfig not found, using AppConfigMock.")
    class AppConfigMock: # Actualizat cu mai multe praguri
        def __init__(self):
            self._config = {
                "CLIPSEG_DEFAULT_THRESHOLD": 0.35,
                "CLIPSEG_BACKGROUND_THRESHOLD": 0.4,
                "CLIPSEG_HAIR_THRESHOLD": 0.45,
                "CLIPSEG_HEAD_THRESHOLD": 0.3,
                "CLIPSEG_FACE_THRESHOLD": 0.35,
                "CLIPSEG_EYES_THRESHOLD": 0.4,
                "CLIPSEG_MOUTH_THRESHOLD": 0.35,
                "CLIPSEG_NOSE_THRESHOLD": 0.3,
                "CLIPSEG_PERSON_THRESHOLD": 0.5,
                "CLIPSEG_CLOTHING_THRESHOLD": 0.4,
                "CLIPSEG_SHIRT_THRESHOLD": 0.45,
                "CLIPSEG_PANTS_THRESHOLD": 0.4,
                "CLIPSEG_SHOES_THRESHOLD": 0.4,
                "CLIPSEG_SKY_THRESHOLD": 0.4,
                "CLIPSEG_TREE_THRESHOLD": 0.4,
                "CLIPSEG_CAR_THRESHOLD": 0.4,
                "CLIPSEG_CAT_THRESHOLD": 0.45,
                "CLIPSEG_DOG_THRESHOLD": 0.45,
                "CLIPSEG_BUILDING_THRESHOLD": 0.35,
                "CLIPSEG_ROAD_THRESHOLD": 0.3,
                "CLIPSEG_WATER_THRESHOLD": 0.3,
                "CLIPSEG_OBJECT_THRESHOLD": 0.35, 
                "SAM_ASSISTED_THRESHOLD": 0.5, # Prag ipotetic pentru măștile SAM
                "HYBRID_MASK_THRESHOLD": 0.35,
            }
        def get(self, key: str, default: Any = None) -> Any:
            return self._config.get(key, default)
    AppConfig = AppConfigMock

from core.model_manager import ModelManager

logger = logging.getLogger(__name__)
if not logger.hasHandlers() or not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
    _console_handler = logging.StreamHandler()
    _console_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    logger.addHandler(_console_handler)
    if logger.level == logging.NOTSET: 
        logger.setLevel(logging.INFO)

RuleCondition = Callable[[str, Dict[str, Any]], bool]
MaskStrategy = Callable[[np.ndarray, str, Dict[str, Any], Dict[str, Any], Callable], Optional[np.ndarray]]

class MaskGenerator:
    def __init__(self):
        logger.debug("Initializing MaskGenerator v3.2...")
        self.models = ModelManager()
        self.config = AppConfig() 
        self.rules: List[Dict[str, Any]] = self._define_rules()
        logger.debug(f"MaskGenerator initialized with {len(self.rules)} rules.")

    def _extract_keyword_from_prompt(self, prompt_lower: str, keywords: List[str], use_regex: bool = True) -> Optional[str]:
        """Helper pentru a extrage primul cuvânt cheie (din lista keywords) găsit în prompt, cu opțiune regex."""
        for kw in keywords:
            if use_regex:
                # Caută cuvântul întreg, insensibil la contextul imediat (ex: "car" nu în "scarf")
                pattern = r'\b' + re.escape(kw) + r'\b'
                if re.search(pattern, prompt_lower):
                    return kw
            else: # Verificare simplă 'in'
                if kw in prompt_lower: 
                    return kw
        return None

    def _get_clip_prompt_from_rule(self, prompt: str, operation: Dict[str, Any], rule_params: Dict[str, Any], param_key_base: str) -> Optional[str]:
        # ... (Rămâne la fel ca în v3.1, dar beneficiază de _extract_keyword_from_prompt îmbunătățit)
        lprompt = prompt.lower()
        target_object_from_op = operation.get("target_object", "").lower()
        # Un default mai specific dacă regula are un keyword principal
        default_keyword_from_rule = rule_params.get(f"{param_key_base}_keyword", 
                                                   rule_params.get("main_clip_prompt_keyword", "object of interest"))


        if rule_params.get(f"{param_key_base}_keyword_from_prompt", False):
            specific_keywords = rule_params.get("keywords_for_extraction", []) # Asigurăm listă
            if specific_keywords: # keywords_for_extraction are prioritate
                extracted = self._extract_keyword_from_prompt(lprompt, specific_keywords)
                if extracted: 
                    logger.debug(f"Extracted '{extracted}' from prompt for {param_key_base} using specific keywords.")
                    return extracted
            
            # Dacă nu s-a extras nimic cu keywords_for_extraction, încercăm target_object din operație
            if target_object_from_op:
                logger.debug(f"Using target_object '{target_object_from_op}' from operation for {param_key_base} as fallback from_prompt.")
                return target_object_from_op
            
            logger.debug(f"Falling back to default keyword '{default_keyword_from_rule}' for {param_key_base} (from_prompt rule).")
            return default_keyword_from_rule

        elif rule_params.get(f"{param_key_base}_keyword_from_op_target", False):
            if target_object_from_op:
                logger.debug(f"Using target_object '{target_object_from_op}' from operation for {param_key_base}.")
                return target_object_from_op
            logger.debug(f"Falling back to default keyword '{default_keyword_from_rule}' for {param_key_base} (from_op_target rule).")
            return default_keyword_from_rule
        
        else: 
            fixed_keyword = rule_params.get(f"{param_key_base}_keyword")
            logger.debug(f"Using fixed keyword '{fixed_keyword}' or default '{default_keyword_from_rule}' for {param_key_base}.")
            return fixed_keyword if fixed_keyword else default_keyword_from_rule

    def _define_rules(self) -> List[Dict[str, Any]]:
        rules = [
            # --- Reguli Specifice (Prioritate Mare) ---
            {
                "name": "Background Interaction", # Acoperă replace, remove, modify
                "condition": lambda prompt, op: "background" in prompt.lower() or op.get("target_object", "").lower() == "background",
                "strategy": self._strategy_background, "params": {}, "message": "Background mask"
            },
            {
                "name": "Hair Interaction",
                "condition": lambda prompt, op: any(kw in prompt.lower() for kw in ["hair", "hairstyle", "wig"]) or op.get("target_object", "").lower() == "hair",
                "strategy": self._strategy_semantic_clipseg,
                "params": {
                    "main_clip_prompt_keyword": "hair", "refine_clip_prompt_keyword": "head",
                    "main_threshold_key": "CLIPSEG_HAIR_THRESHOLD", "refine_threshold_key": "CLIPSEG_HEAD_THRESHOLD", "combine_op": "and"
                }, "message": "Hair mask"
            },
            {
                "name": "Eyes Interaction",
                "condition": lambda prompt, op: any(kw in prompt.lower() for kw in ["eye", "eyes"]) or op.get("target_object", "").lower() in ("eye", "eyes"),
                "strategy": self._strategy_semantic_clipseg,
                "params": {
                    "main_clip_prompt_keyword": "eyes", "refine_clip_prompt_keyword": "face",
                    "main_threshold_key": "CLIPSEG_EYES_THRESHOLD", "refine_threshold_key": "CLIPSEG_FACE_THRESHOLD", "combine_op": "and"
                }, "message": "Eyes mask"
            },
            {
                "name": "Mouth/Lips Interaction",
                "condition": lambda prompt, op: any(kw in prompt.lower() for kw in ["mouth", "lips"]) or op.get("target_object", "").lower() in ("mouth", "lips"),
                "strategy": self._strategy_semantic_clipseg,
                "params": {
                    "main_clip_prompt_keyword": "mouth lips", "refine_clip_prompt_keyword": "face", # "mouth lips" poate fi mai robust
                    "main_threshold_key": "CLIPSEG_MOUTH_THRESHOLD", "refine_threshold_key": "CLIPSEG_FACE_THRESHOLD", "combine_op": "and"
                }, "message": "Mouth/Lips mask"
            },
            {
                "name": "Nose Interaction",
                "condition": lambda prompt, op: "nose" in prompt.lower() or op.get("target_object", "").lower() == "nose",
                "strategy": self._strategy_semantic_clipseg,
                "params": {
                    "main_clip_prompt_keyword": "nose", "refine_clip_prompt_keyword": "face",
                    "main_threshold_key": "CLIPSEG_NOSE_THRESHOLD", "refine_threshold_key": "CLIPSEG_FACE_THRESHOLD", "combine_op": "and"
                }, "message": "Nose mask"
            },
            {
                "name": "Face Interaction (General)", # Pentru "change face expression", "add makeup to face"
                "condition": lambda prompt, op: "face" in prompt.lower() or op.get("target_object", "").lower() == "face",
                "strategy": self._strategy_semantic_clipseg,
                "params": {"main_clip_prompt_keyword": "face", "main_threshold_key": "CLIPSEG_FACE_THRESHOLD"},
                "message": "Face mask"
            },
            {
                "name": "Glasses Interaction (Add, Replace, Remove)",
                "condition": lambda prompt, op: any(kw in prompt.lower() for kw in ["glasses", "sunglasses", "eyewear"]) or \
                                             op.get("target_object", "").lower() in ("glasses", "sunglasses", "eyewear"),
                "strategy": self._strategy_semantic_clipseg,
                "params": { 
                    "main_clip_prompt_keyword": "glasses on face" if "add" in prompt.lower() or op.get("type") == "add" else "glasses", # Context pentru adăugare
                    "refine_clip_prompt_keyword": "eyes area" if "add" in prompt.lower() or op.get("type") == "add" else "face",
                    "main_threshold_key": "CLIPSEG_OBJECT_THRESHOLD", # Folosim un prag de obiect
                    "refine_threshold_key": "CLIPSEG_FACE_THRESHOLD", "combine_op": "and" 
                },
                "message": "Glasses interaction mask"
            },
            {
                "name": "Remove Person/Human",
                "condition": lambda prompt, op: (op.get("type") == "remove" and \
                                                self._extract_keyword_from_prompt(prompt.lower(), ["person", "man", "woman", "boy", "girl", "child", "people", "human", "figure"]) is not None) or \
                                                op.get("target_object", "").lower() in ("person", "man", "woman", "boy", "girl", "child", "people", "human", "figure"),
                # SAM_Integration_Point: Această regulă ar beneficia enorm de SAM.
                # "strategy": self._strategy_sam_assisted_segmentation, 
                "strategy": self._strategy_semantic_clipseg, # Momentan CLIPSeg
                "params": {
                    "main_clip_prompt_keyword_from_prompt": True, 
                    "keywords_for_extraction": ["person", "man", "woman", "boy", "girl", "child", "people", "human", "figure"],
                    "main_threshold_key": "CLIPSEG_PERSON_THRESHOLD",
                    # "sam_fallback_if_clipseg_fails": True # Un parametru ipotetic
                },
                "message": "Person mask for removal"
            },
            {
                "name": "Clothing Interaction (Shirt, Pants, Dress, etc.)",
                "condition": lambda prompt, op: self._extract_keyword_from_prompt(prompt.lower(), ["shirt", "t-shirt", "top", "blouse", "jacket", "coat", "dress", "pants", "trousers", "skirt", "suit", "clothing", "outfit", "shoes", "boots", "hat"]) is not None or \
                                             op.get("target_object", "").lower() in ["shirt", "t-shirt", "top", "blouse", "jacket", "coat", "dress", "pants", "trousers", "skirt", "suit", "clothing", "outfit", "shoes", "boots", "hat"],
                "strategy": self._strategy_semantic_clipseg,
                "params": {
                    "main_clip_prompt_keyword_from_prompt": True, # Va extrage "shirt", "pants", etc.
                    "keywords_for_extraction": ["shirt", "t-shirt", "top", "blouse", "jacket", "coat", "dress", "pants", "trousers", "skirt", "suit", "clothing", "outfit", "shoes", "boots", "hat"],
                    "main_threshold_key": "CLIPSEG_CLOTHING_THRESHOLD", 
                    "refine_clip_prompt_keyword": "person", # Rafinăm cu persoana
                    "refine_threshold_key": "CLIPSEG_PERSON_THRESHOLD", "combine_op": "and"
                },
                "message": "Clothing mask"
            },
            {
                "name": "Sky Interaction",
                "condition": lambda prompt, op: "sky" in prompt.lower() or op.get("target_object", "").lower() == "sky",
                "strategy": self._strategy_semantic_clipseg,
                "params": {"main_clip_prompt_keyword": "sky", "main_threshold_key": "CLIPSEG_SKY_THRESHOLD"},
                "message": "Sky mask"
            },
             {
                "name": "Water Interaction (sea, river, lake, pool)",
                "condition": lambda prompt, op: any(kw in prompt.lower() for kw in ["water", "sea", "ocean", "river", "lake", "pool"]) or \
                                             op.get("target_object", "").lower() in ("water", "sea", "ocean", "river", "lake", "pool"),
                "strategy": self._strategy_semantic_clipseg,
                "params": {"main_clip_prompt_keyword_from_prompt": True, 
                           "keywords_for_extraction": ["water", "sea", "ocean", "river", "lake", "pool"],
                           "main_threshold_key": "CLIPSEG_WATER_THRESHOLD"},
                "message": "Water mask"
            },
            {
                "name": "Road/Street Interaction",
                "condition": lambda prompt, op: any(kw in prompt.lower() for kw in ["road", "street", "path", "pavement"]) or \
                                             op.get("target_object", "").lower() in ("road", "street", "path", "pavement"),
                "strategy": self._strategy_semantic_clipseg,
                "params": {"main_clip_prompt_keyword_from_prompt": True,
                           "keywords_for_extraction": ["road", "street", "path", "pavement"],
                           "main_threshold_key": "CLIPSEG_ROAD_THRESHOLD"},
                "message": "Road/Street mask"
            },
            {
                "name": "Building/House Interaction",
                "condition": lambda prompt, op: any(kw in prompt.lower() for kw in ["building", "house", "architecture", "structure"]) or \
                                             op.get("target_object", "").lower() in ("building", "house", "architecture", "structure"),
                "strategy": self._strategy_semantic_clipseg,
                "params": {"main_clip_prompt_keyword_from_prompt": True,
                           "keywords_for_extraction": ["building", "house", "architecture", "structure"],
                           "main_threshold_key": "CLIPSEG_BUILDING_THRESHOLD"},
                "message": "Building/House mask"
            },
            # --- Reguli pentru Obiecte Comune Specifice ---
            # SAM_Integration_Point: Acestea ar beneficia de SAM dacă CLIPSeg nu e suficient de precis.
            {
                "name": "Cat Interaction",
                "condition": lambda prompt, op: "cat" in prompt.lower() or op.get("target_object", "").lower() == "cat",
                "strategy": self._strategy_semantic_clipseg, # Sau _strategy_sam_assisted_segmentation
                "params": {"main_clip_prompt_keyword": "cat", "main_threshold_key": "CLIPSEG_CAT_THRESHOLD"},
                "message": "Cat mask"
            },
            # ... (similar pentru Dog, Car, Tree ca în v3.1) ...
            {
                "name": "Dog Interaction",
                "condition": lambda prompt, op: "dog" in prompt.lower() or op.get("target_object", "").lower() == "dog",
                "strategy": self._strategy_semantic_clipseg,
                "params": {"main_clip_prompt_keyword": "dog", "main_threshold_key": "CLIPSEG_DOG_THRESHOLD"},
                "message": "Dog mask"
            },
            {
                "name": "Car Interaction",
                "condition": lambda prompt, op: "car" in prompt.lower() or op.get("target_object", "").lower() == "car",
                "strategy": self._strategy_semantic_clipseg,
                "params": {"main_clip_prompt_keyword": "car", "main_threshold_key": "CLIPSEG_CAR_THRESHOLD"},
                "message": "Car mask"
            },
            {
                "name": "Tree Interaction",
                "condition": lambda prompt, op: "tree" in prompt.lower() or op.get("target_object", "").lower() == "tree",
                "strategy": self._strategy_semantic_clipseg,
                "params": {"main_clip_prompt_keyword": "tree", "main_threshold_key": "CLIPSEG_TREE_THRESHOLD"},
                "message": "Tree mask"
            },
            # --- Regula Generică (Fallback înainte de Hibrid Total) ---
            { 
                "name": "Generic Object (from Operation Target if specific)",
                "condition": lambda prompt, op: bool(op.get("target_object")) and \
                                             op.get("target_object") not in [
                                                 "background", "hair", "hairstyle", "wig", "eye", "eyes", "mouth", "lips", "nose", "face", 
                                                 "glasses", "sunglasses", "eyewear", "person", "man", "woman", "boy", "girl", 
                                                 "child", "people", "human", "figure", "shirt", "t-shirt", "top", "blouse", 
                                                 "jacket", "coat", "dress", "pants", "trousers", "skirt", "suit", "clothing", 
                                                 "outfit", "shoes", "boots", "hat", "sky", "water", "sea", "ocean", "river", 
                                                 "lake", "pool", "road", "street", "path", "pavement", "building", "house", 
                                                 "architecture", "structure", "cat", "dog", "car", "tree"
                                             ], # Lista extinsă de cuvinte cheie deja acoperite
                "strategy": self._strategy_semantic_clipseg, # SAM_Integration_Point: Ar putea folosi SAM aici
                "params": {
                    "main_clip_prompt_keyword_from_op_target": True, 
                    "main_threshold_key": "CLIPSEG_OBJECT_THRESHOLD",
                },
                "message": lambda op: f"{op.get('target_object', 'Identified Object').capitalize()} mask"
            },
            # SAM_Integration_Point: O regulă specifică pentru SAM dacă promptul indică o selecție precisă
            # {
            #     "name": "SAM Precise Object (if box/points provided or complex object)",
            #     "condition": lambda prompt, op: op.get("use_sam_directly", False) or \
            #                                  (op.get("target_object") and op.get("type") in ["remove", "isolate"]), # Condiție exemplu
            #     "strategy": self._strategy_sam_assisted_segmentation, # Necesită implementare
            #     "params": {
            #         "clipseg_prompt_for_sam_box": True, # Folosește CLIPSeg pentru a genera box-ul pentru SAM
            #         "sam_threshold_key": "SAM_ASSISTED_THRESHOLD",
            #     },
            #     "message": lambda op: f"SAM mask for {op.get('target_object', 'precise selection')}"
            # },
        ]
        logger.debug(f"Defined {len(rules)} mask generation rules.")
        return rules

    # ... (restul metodelor: _strategy_semantic_clipseg, _strategy_background, generate_mask, 
    #      _strategy_hybrid_fallback, _grabcut_subject, _clipseg_segment, _morphology, 
    #      _dynamic_morphology, _advanced_edge_refine, _edge_refine, _adaptive_threshold
    #      rămân în mare parte la fel ca în v3.1, cu ajustările de robustețe și logging deja făcute.
    #      Am inclus versiunile lor actualizate în artifactul anterior <mask_generator_v3_1>.
    #      Pentru concizie, nu le repet aici dacă sunt identice cu cele din artifactul specificat.
    #      Important: Asigurați-vă că toate aceste metode helper sunt copiate corect din v3.1)

    # Voi include din nou metodele helper din v3.1 pentru completitudine, cu mici ajustări dacă e cazul.

    def _strategy_semantic_clipseg(self, img_np_bgr: np.ndarray, prompt: str, operation: Dict[str, Any], rule_params: Dict[str, Any], upd: Callable) -> Optional[np.ndarray]:
        main_clip_text = self._get_clip_prompt_from_rule(prompt, operation, rule_params, "main_clip_prompt")
        if not main_clip_text or not main_clip_text.strip():
            logger.warning(f"Rule '{rule_params.get('_rule_name_', 'Semantic CLIPSeg')}' failed: main_clip_text empty.")
            return None
            
        upd(0.1, f"CLIPSeg: '{main_clip_text}'")
        main_seg = self._clipseg_segment(img_np_bgr, main_clip_text)
        if main_seg is None:
            logger.warning(f"CLIPSeg failed for main prompt: '{main_clip_text}'")
            return None

        main_thresh_key = rule_params.get("main_threshold_key", "CLIPSEG_DEFAULT_THRESHOLD")
        main_thresh_factor = self.config.get(main_thresh_key, self.config.get("CLIPSEG_DEFAULT_THRESHOLD", 0.35))
        main_thresh_val = int(main_thresh_factor * 255)
        _, main_mask = cv2.threshold(main_seg, main_thresh_val, 255, cv2.THRESH_BINARY)
        logger.debug(f"Applied threshold {main_thresh_val} ({main_thresh_factor*100:.0f}%) for main CLIPSeg mask '{main_clip_text}'.")
        final_combined_mask = main_mask

        refine_clip_text = self._get_clip_prompt_from_rule(prompt, operation, rule_params, "refine_clip_prompt")
        if refine_clip_text and refine_clip_text.strip() and refine_clip_text != main_clip_text: # Evităm rafinarea cu același prompt
            upd(0.3, f"CLIPSeg Refine: '{refine_clip_text}'")
            refine_seg = self._clipseg_segment(img_np_bgr, refine_clip_text)
            if refine_seg is not None:
                refine_thresh_key = rule_params.get("refine_threshold_key", "CLIPSEG_DEFAULT_THRESHOLD")
                refine_thresh_factor = self.config.get(refine_thresh_key, self.config.get("CLIPSEG_DEFAULT_THRESHOLD", 0.35))
                refine_thresh_val = int(refine_thresh_factor * 255)
                _, refine_mask = cv2.threshold(refine_seg, refine_thresh_val, 255, cv2.THRESH_BINARY)
                logger.debug(f"Applied threshold {refine_thresh_val} ({refine_thresh_factor*100:.0f}%) for refine CLIPSeg mask '{refine_clip_text}'.")
                
                combine_op = rule_params.get("combine_op", "and") 
                if combine_op == "and": final_combined_mask = cv2.bitwise_and(main_mask, refine_mask)
                elif combine_op == "or": final_combined_mask = cv2.bitwise_or(main_mask, refine_mask)
                logger.debug(f"Combined main ('{main_clip_text}') and refine ('{refine_clip_text}') masks using '{combine_op}'.")
            else: logger.warning(f"CLIPSeg failed for refinement prompt: '{refine_clip_text}'. Using main mask only.")
        return final_combined_mask

    def _strategy_background(self, img_np_bgr: np.ndarray, prompt: str, operation: Dict[str, Any], rule_params: Dict[str, Any], upd: Callable) -> Optional[np.ndarray]:
        upd(0.1, "GrabCut: Subject for background")
        return self._grabcut_subject(img_np_bgr) 

    def generate_mask(
        self, image: Union[Image.Image, np.ndarray], prompt: str = "",
        operation: Optional[Dict[str, Any]] = None, 
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> Dict[str, Any]:
        operation = operation or {} 
        def upd(pct: float, desc: str):
            if progress_callback: progress_callback(pct, desc)
            logger.debug(f"MaskGen Progress: {pct*100:.0f}% - {desc}")

        img_np_bgr: Optional[np.ndarray] = None
        if isinstance(image, Image.Image):
            try:
                img_pil = image.convert("RGB"); img_np_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
            except Exception as e: logger.error(f"PIL to NumPy BGR error: {e}", exc_info=True); return {"mask": None, "raw_mask": None, "success": False, "message": "Image conversion error."}
        elif isinstance(image, np.ndarray):
            img_o = image.copy()
            if img_o.ndim == 2: img_np_bgr = cv2.cvtColor(img_o, cv2.COLOR_GRAY2BGR)
            elif img_o.shape[2] == 4: img_np_bgr = cv2.cvtColor(img_o, cv2.COLOR_RGBA2BGR)
            elif img_o.shape[2] == 3: img_np_bgr = img_o # Assume BGR or RGB (CLIPSeg handles RGB if needed)
            else: logger.error(f"Unsupported NumPy shape: {img_o.shape}"); return {"mask": None, "raw_mask": None, "success": False, "message": "Unsupported NumPy image format."}
        else: logger.error(f"Unsupported image type: {type(image)}"); return {"mask": None, "raw_mask": None, "success": False, "message": "Unsupported image type."}

        if img_np_bgr.dtype != np.uint8:
            logger.warning(f"Input image not uint8 (type: {img_np_bgr.dtype}). Converting.")
            try:
                if np.max(img_np_bgr) <= 1.0 and img_np_bgr.dtype in [np.float32, np.float64]: img_np_bgr = (img_np_bgr * 255).astype(np.uint8)
                else: img_np_bgr = np.clip(img_np_bgr, 0, 255).astype(np.uint8)
            except Exception as e: logger.error(f"uint8 conversion error: {e}", exc_info=True); return {"mask": None, "raw_mask": None, "success": False, "message": "Image data type conversion error."}

        h, w = img_np_bgr.shape[:2]
        logger.info(f"Generating mask for prompt: '{prompt}', op: {operation}")
        raw_mask_generated, final_message, applied_rule_name = None, "Mask gen failed.", "N/A"

        for rule in self.rules:
            if rule["condition"](prompt, operation):
                applied_rule_name = rule['name']; logger.info(f"Applying rule: {applied_rule_name}"); upd(0.05, f"Rule: {applied_rule_name}")
                strategy_params = rule.get("params", {}); strategy_params["_rule_name_"] = applied_rule_name
                raw_mask_candidate = rule["strategy"](img_np_bgr, prompt, operation, strategy_params, upd)
                if raw_mask_candidate is not None and raw_mask_candidate.size > 0 and raw_mask_candidate.shape[:2] == (h,w):
                    raw_mask_generated = raw_mask_candidate
                    msg_template = rule.get("message", "Mask by rule."); final_message = msg_template(operation) if callable(msg_template) else msg_template
                    if applied_rule_name == "Background Interaction": # Numele actualizat al regulii
                        upd(0.4, "Post-proc: BG subject mask"); morphed_subj = self._dynamic_morphology(raw_mask_generated, img_np_bgr)
                        upd(0.7, "Post-proc: BG edge refine"); refined_subj = self._advanced_edge_refine(morphed_subj, img_np_bgr)
                        final_processed_mask = cv2.bitwise_not(refined_subj); raw_mask_to_return = refined_subj
                        upd(1.0, "Background mask ready")
                        return {"mask": final_processed_mask, "raw_mask": raw_mask_to_return, "success": True, "message": final_message}
                    break 
                else: logger.warning(f"Rule '{applied_rule_name}' no valid mask. Shape: {raw_mask_candidate.shape if raw_mask_candidate is not None else 'None'}")
        
        if raw_mask_generated is None: # Fallback la hibrid
            applied_rule_name = "Hybrid Fallback"; logger.info("No specific rule matched/succeeded. Falling back to hybrid."); upd(0.0, applied_rule_name)
            clip_prompt_hybrid = operation.get("target_object", "").strip()
            if not clip_prompt_hybrid or clip_prompt_hybrid == "subject": # Încercăm să extragem ceva din prompt
                words = [w for w in prompt.lower().replace("remove","").replace("change","").replace("replace","").replace("add","").replace("make","").split() if w not in ["the","a","an","to","color","of","with","on","from","in","into", "style"]]
                clip_prompt_hybrid = " ".join(words[:3]).strip() # Primele 3 cuvinte "relevante"
            if not clip_prompt_hybrid : clip_prompt_hybrid = "area of interest"
            raw_mask_generated = self._strategy_hybrid_fallback(img_np_bgr, clip_prompt_hybrid, upd)
            final_message = f"Hybrid fallback mask for '{clip_prompt_hybrid}'."
            if raw_mask_generated is None or raw_mask_generated.size == 0:
                logger.error("Hybrid fallback also failed."); return {"mask": np.zeros((h,w),np.uint8), "raw_mask":None, "success":False, "message":"All mask strategies failed."}

        logger.info(f"Final post-processing for rule: {applied_rule_name}"); upd(0.65, "Final Morphology")
        morphed_mask = self._dynamic_morphology(raw_mask_generated, img_np_bgr)
        upd(0.85, "Final Edge Refinement"); final_processed_mask = self._advanced_edge_refine(morphed_mask, img_np_bgr)
        upd(1.0, "Mask ready")
        return {"mask": final_processed_mask, "raw_mask": raw_mask_generated, "success": True, "message": final_message}

    def _strategy_hybrid_fallback(self, img_np_bgr: np.ndarray, clip_prompt_text: str, upd: Callable) -> Optional[np.ndarray]:
        h, w = img_np_bgr.shape[:2]; accum = np.zeros((h, w), np.float32); active_model_count = 0
        upd(0.1, "Hybrid: GrabCut"); grabcut_subj = self._grabcut_subject(img_np_bgr)
        if grabcut_subj is not None: accum += grabcut_subj.astype(np.float32)/255.0 * 0.7; active_model_count += 1 # Ponderare redusă

        yolo = self.models.get_model("yolo")
        if yolo:
            upd(0.25, "Hybrid: YOLO"); yolo_masks_combined = np.zeros((h,w),np.float32); yolo_detected_count=0
            try:
                preds = yolo.predict(source=img_np_bgr, stream=False, imgsz=640, conf=0.2, verbose=False) # conf redus
                for r in preds:
                    if getattr(r,"masks",None) and hasattr(r.masks,"data") and r.masks.data.numel() > 0:
                        # SAM_Integration_Point: Aici, măștile YOLO (sau bounding box-urile lor) ar putea fi folosite ca prompt pentru SAM
                        # pentru a rafina fiecare obiect detectat de YOLO, mai ales dacă clasa obiectului YOLO se potrivește cu `clip_prompt_text`.
                        masks_data = r.masks.data.cpu().numpy()
                        for m_yolo in masks_data:
                            if m_yolo.size==0: continue
                            yolo_masks_combined += cv2.resize(m_yolo.astype(np.float32),(w,h), interpolation=cv2.INTER_LINEAR)
                            yolo_detected_count+=1
                if yolo_detected_count > 0: accum += (yolo_masks_combined/yolo_detected_count)*1.0; active_model_count+=1
            except Exception as e: logger.error(f"Hybrid YOLO error: {e}", exc_info=True)

        if any(k in clip_prompt_text.lower() for k in ["person","face","selfie","man","woman","boy","girl","human"]):
            mp_segmenter = self.models.get_model("mediapipe") 
            if mp_segmenter:
                upd(0.5, "Hybrid: MediaPipe"); 
                try:
                    img_rgb_for_mp = cv2.cvtColor(img_np_bgr, cv2.COLOR_BGR2RGB)
                    res_mp = mp_segmenter.process(img_rgb_for_mp)
                    mm = getattr(res_mp, 'segmentation_mask', None)
                    if mm is not None and mm.size > 0:
                        accum += cv2.resize(mm.astype(np.float32),(w,h),interpolation=cv2.INTER_LINEAR)*1.2; active_model_count+=1
                except Exception as e: logger.error(f"Hybrid MediaPipe error: {e}", exc_info=True)
        
        upd(0.75, f"Hybrid: CLIPSeg '{clip_prompt_text}'")
        # SAM_Integration_Point: Dacă `clip_prompt_text` este un obiect specific, am putea folosi SAM aici,
        # posibil ghidat de o mască brută de la acest apel CLIPSeg.
        clip_gen_seg = self._clipseg_segment(img_np_bgr, clip_prompt_text)
        if clip_gen_seg is not None: accum += clip_gen_seg.astype(np.float32)/255.0 * 1.5; active_model_count+=1
        
        if active_model_count == 0: logger.warning("Hybrid: No models contributed."); return np.zeros((h,w),dtype=np.uint8)
        combined_float_mask = np.clip(accum / active_model_count, 0.0, 1.0) 
        return (combined_float_mask > self.config.get("HYBRID_MASK_THRESHOLD",0.35)).astype(np.uint8)*255

    def _grabcut_subject(self, img_bgr_np: np.ndarray, rect_inset_ratio: float = 0.05) -> Optional[np.ndarray]:
        h, w = img_bgr_np.shape[:2]
        if h < 10 or w < 10 : logger.error(f"GrabCut: Image too small ({w}x{h})."); return None # Adăugat verificare dimensiune minimă
        if img_bgr_np.ndim < 3 or img_bgr_np.shape[2] < 3: logger.error(f"GrabCut expects BGR, got {img_bgr_np.shape}."); return None
        mask_gc = np.zeros((h, w), np.uint8); bgd_model, fgd_model = np.zeros((1,65),np.float64), np.zeros((1,65),np.float64)
        dx, dy = int(rect_inset_ratio*w), int(rect_inset_ratio*h)
        rect = (max(1,dx), max(1,dy), max(1,w-2*dx), max(1,h-2*dy))
        if rect[2] <= 0 or rect[3] <= 0: logger.error(f"GrabCut: Invalid rect {rect} for image {w}x{h}."); return None
        try:
            cv2.grabCut(img_bgr_np, mask_gc, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
            return np.where((mask_gc==cv2.GC_FGD)|(mask_gc==cv2.GC_PR_FGD),255,0).astype(np.uint8)
        except Exception as e:
            logger.error(f"GrabCut error: {e}", exc_info=True)
            fb_mask = np.zeros((h,w),np.uint8); cv2.rectangle(fb_mask,(rect[0],rect[1]),(rect[0]+rect[2],rect[1]+rect[3]),255,-1); return fb_mask

    def _clipseg_segment(self, img_bgr_np: np.ndarray, text_prompt: str) -> Optional[np.ndarray]:
        bundle = self.models.get_model("clipseg"); 
        if not bundle or "processor" not in bundle or "model" not in bundle: logger.warning("CLIPSeg model/processor not found."); return None
        processor, model = bundle["processor"], bundle["model"]
        try: img_rgb_np = cv2.cvtColor(img_bgr_np, cv2.COLOR_BGR2RGB); pil_image = Image.fromarray(img_rgb_np)
        except Exception as e: logger.error(f"CLIPSeg PIL conversion error: {e}", exc_info=True); return None
        effective_text = text_prompt.strip() if text_prompt and text_prompt.strip() else "object"
        logger.debug(f"CLIPSeg with text: '{effective_text}' (Device: {model.device})")
        try: inputs = processor(text=[effective_text], images=[pil_image], return_tensors="pt", padding="max_length", truncation=True, max_length=processor.model_max_length if hasattr(processor, 'model_max_length') else 77)
        except Exception as e: logger.error(f"CLIPSeg processor error ('{effective_text}'): {e}", exc_info=True); return None
        processed_inputs = {}; target_device = model.device; target_dtype = model.dtype if hasattr(model, 'dtype') and model.dtype is not None else torch.float32
        try:
            for k,v_tensor in inputs.items(): processed_inputs[k] = v_tensor.to(target_device, dtype=target_dtype if v_tensor.dtype.is_floating_point else v_tensor.dtype)
        except Exception as e: logger.error(f"CLIPSeg device move error ('{effective_text}'): {e}", exc_info=True); return None
        try:
            with torch.no_grad(): outputs = model(**processed_inputs)
            logits = outputs.logits if hasattr(outputs,'logits') else outputs[0]
            if logits.ndim == 4: logits = logits.squeeze(0).squeeze(0) 
            elif logits.ndim == 3: logits = logits.squeeze(0)
            if logits.ndim != 2: logger.error(f"CLIPSeg logits ndim {logits.ndim} for '{effective_text}'."); return None
            probs = torch.sigmoid(logits).cpu().numpy().astype(np.float32)
            if probs.size == 0: logger.error(f"CLIPSeg 'probs' empty for '{effective_text}'."); return None
            return (np.clip(cv2.resize(probs, (img_bgr_np.shape[1],img_bgr_np.shape[0]), interpolation=cv2.INTER_LINEAR),0.0,1.0)*255).astype(np.uint8)
        except Exception as e: logger.error(f"CLIPSeg runtime/resize error ('{effective_text}'): {e}", exc_info=True); return None

    def _morphology(self, mask:np.ndarray, close_k:int, open_k:int, close_iter:int, open_iter:int) -> np.ndarray:
        if mask is None or mask.size==0: return mask
        if close_k<=0 or open_k<=0: return mask
        close_k=close_k if close_k%2!=0 else close_k+1; open_k=open_k if open_k%2!=0 else open_k+1
        try:
            ker_c=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(close_k,close_k)); ker_o=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(open_k,open_k))
            m=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,ker_c,iterations=close_iter); return cv2.morphologyEx(m,cv2.MORPH_OPEN,ker_o,iterations=open_iter)
        except Exception as e: logger.error(f"Morphology error: {e}", exc_info=True); return mask

    def _dynamic_morphology(self, mask:np.ndarray, img:np.ndarray) -> np.ndarray:
        if mask is None or img is None or img.size==0: return mask
        h,w=img.shape[:2]; min_d=min(h,w)
        ck_f,ok_f = (0.015,0.007) if min_d>1000 else (0.02,0.01) if min_d>500 else (0.025,0.015) # Kernel factors ajustați
        ci,oi = (3,2) if min_d>700 else (2,1)
        ck,ok = max(3,int(ck_f*min_d)//2*2+1), max(3,int(ok_f*min_d)//2*2+1)
        logger.debug(f"Dynamic morphology: ck={ck}, ok={ok}, iters={ci},{oi} (min_dim={min_d})")
        return self._morphology(mask,ck,ok,ci,oi)

    def _advanced_edge_refine(self, mask:np.ndarray, img_bgr:np.ndarray) -> np.ndarray:
        if mask is None or img_bgr is None or img_bgr.size==0: return mask
        try:
            if not hasattr(cv2,'ximgproc') or not hasattr(cv2.ximgproc,'guidedFilter'): return self._edge_refine(mask,img_bgr)
        except: return self._edge_refine(mask,img_bgr)
        try:
            mf=mask.astype(np.float32)/255.0; gray=cv2.cvtColor(img_bgr,cv2.COLOR_BGR2GRAY)
            canny=cv2.Canny(gray,30,100).astype(np.float32)/255.0 # Praguri mai sensibile
            gx=cv2.Sobel(gray,cv2.CV_32F,1,0,ksize=3); gy=cv2.Sobel(gray,cv2.CV_32F,0,1,ksize=3)
            sobel_mag=cv2.magnitude(gx,gy); max_s=np.max(sobel_mag); sobel=sobel_mag/max_s if max_s>0 else np.zeros_like(sobel_mag,dtype=np.float32)
            edges=np.maximum(canny,sobel*0.7) # Ponderare Sobel
            influence=0.35; blended_f = mf*(1.0-edges*influence) + edges*(mf*0.1 + 0.05) # Blend mai fin
            blended_f=np.clip(blended_f,0.0,1.0)
            radius=max(3,int(0.004*min(img_bgr.shape[:2]))); eps_v=0.001 # Valori mai mici pentru mai multă netezire
            guide=img_bgr; src_guide=(blended_f*255).astype(np.uint8)
            if src_guide.ndim==3: src_guide=cv2.cvtColor(src_guide,cv2.COLOR_BGR2GRAY)
            filt_m=cv2.ximgproc.guidedFilter(guide,src_guide,radius,eps_v*eps_v*255*255)
            if filt_m.dtype!=np.uint8: filt_m=np.clip(filt_m,0,255).astype(np.uint8)
            _,thresh_m=cv2.threshold(filt_m,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            k=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
            return cv2.morphologyEx(cv2.morphologyEx(thresh_m,cv2.MORPH_OPEN,k,iterations=1),cv2.MORPH_CLOSE,k,iterations=1)
        except Exception as e: logger.error(f"Advanced edge refine error: {e}",exc_info=True); return self._edge_refine(mask,img_bgr)

    def _edge_refine(self, mask:np.ndarray, img:np.ndarray) -> np.ndarray:
        if mask is None or img is None or img.size==0: return mask
        try:
            k_s=max(3,int(0.005*min(img.shape[:2]))//2*2+1); k=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(k_s,k_s))
            return cv2.morphologyEx(cv2.morphologyEx(mask,cv2.MORPH_OPEN,k,iterations=1),cv2.MORPH_CLOSE,k,iterations=1)
        except Exception as e: logger.error(f"Basic edge refine error: {e}",exc_info=True); return mask
    
    def _adaptive_threshold(self, mask_channel: np.ndarray, img: np.ndarray) -> np.ndarray: # Păstrat pentru potențial uz viitor
        if mask_channel is None or img is None or img.size == 0: return mask_channel if mask_channel is not None else np.array([], dtype=np.uint8)
        if mask_channel.dtype != np.uint8: mask_channel = np.clip(mask_channel, 0, 255).astype(np.uint8)
        if mask_channel.ndim != 2:
            if mask_channel.ndim == 3 and mask_channel.shape[2] == 1: mask_channel = mask_channel.squeeze(axis=2)
            else: _, th_mask = cv2.threshold(mask_channel, 127, 255, cv2.THRESH_BINARY); return th_mask
        if mask_channel.size == 0: return np.array([], dtype=np.uint8)
        block_size = min(31, max(3, int(0.05 * min(img.shape[:2])) // 2 * 2 + 1)) 
        C_val = 2 
        try: return cv2.adaptiveThreshold(mask_channel, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, C_val)
        except: _, th_mask = cv2.threshold(mask_channel, 127, 255, cv2.THRESH_BINARY); return th_mask

