#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
MaskGenerator v3.5 — Inteligent, operation-aware mask generation
for FusionFrame. Corrected AppConfig access.
"""

import cv2
import numpy as np
import logging
import torch 
import time 
import re 
import sys 
from typing import Dict, Any, Union, Callable, Optional, List, Tuple
from PIL import Image

# Importuri Config și Manager
try:
    from config.app_config import AppConfig
    APP_CONFIG_IS_MOCK = False
except ImportError:
    print("WARNING: Real AppConfig not found, using AppConfigMock for MaskGenerator.")
    class AppConfigMock: 
        def __init__(self):
            # Atributele sunt definite direct, la fel ca în AppConfig real
            self.CLIPSEG_DEFAULT_THRESHOLD = 0.35
            self.CLIPSEG_BACKGROUND_THRESHOLD = 0.4
            self.CLIPSEG_HAIR_THRESHOLD = 0.45
            self.CLIPSEG_HEAD_THRESHOLD = 0.3
            self.CLIPSEG_FACE_THRESHOLD = 0.35
            self.CLIPSEG_EYES_THRESHOLD = 0.4
            self.CLIPSEG_MOUTH_THRESHOLD = 0.35
            self.CLIPSEG_NOSE_THRESHOLD = 0.3
            self.CLIPSEG_PERSON_THRESHOLD = 0.5
            self.CLIPSEG_CLOTHING_THRESHOLD = 0.4
            self.CLIPSEG_SHIRT_THRESHOLD = 0.45
            self.CLIPSEG_PANTS_THRESHOLD = 0.4
            self.CLIPSEG_SHOES_THRESHOLD = 0.4
            self.CLIPSEG_SKY_THRESHOLD = 0.4
            self.CLIPSEG_TREE_THRESHOLD = 0.4
            self.CLIPSEG_CAR_THRESHOLD = 0.4
            self.CLIPSEG_CAT_THRESHOLD = 0.45
            self.CLIPSEG_DOG_THRESHOLD = 0.45
            self.CLIPSEG_BUILDING_THRESHOLD = 0.35
            self.CLIPSEG_ROAD_THRESHOLD = 0.3
            self.CLIPSEG_WATER_THRESHOLD = 0.3
            self.CLIPSEG_OBJECT_THRESHOLD = 0.35
            self.SAM_ASSISTED_THRESHOLD = 0.5
            self.HYBRID_MASK_THRESHOLD = 0.35
            # Adăugăm și alte atribute pe care AppConfig real le-ar putea avea, dacă sunt necesare aici
            # De exemplu, dacă MaskGenerator ar folosi DEVICE sau DTYPE direct:
            # self.DEVICE = "cpu" 
            # self.DTYPE = torch.float32

        # Metoda get nu mai este necesară dacă accesăm direct atributele
        # def get(self, key: str, default: Any = None) -> Any:
        #     return getattr(self, key, default if default is not None else 0.35)

    AppConfig = AppConfigMock
    APP_CONFIG_IS_MOCK = True


try:
    from core.model_manager import ModelManager
except ImportError:
    print("ERROR: core.model_manager not found. Please ensure it's in the PYTHONPATH.")
    sys.exit(1) 

# Configurare Logger
logger = logging.getLogger(__name__)
if not logger.hasHandlers() or not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
    _ch = logging.StreamHandler(); _f = logging.Formatter("%(asctime)s-%(name)s-%(levelname)s-%(message)s"); _ch.setFormatter(_f)
    logger.addHandler(_ch); 
    if logger.level == logging.NOTSET: logger.setLevel(logging.INFO)

RuleCondition = Callable[[str, Dict[str, Any]], bool]
MaskStrategy = Callable[[np.ndarray, str, Dict[str, Any], Dict[str, Any], Callable], Optional[np.ndarray]]

class MaskGenerator:
    def __init__(self):
        logger.debug("Initializing MaskGenerator v3.5...")
        self.models = ModelManager()
        # self.config va fi o instanță a AppConfig (real sau mock)
        # Atributele vor fi accesate prin self.config.NUME_ATRIBUT
        self.config = AppConfig() 
        self.rules: List[Dict[str, Any]] = self._define_rules()
        logger.debug(f"MaskGenerator initialized with {len(self.rules)} rules.")

    def _get_config_threshold(self, key: str, default_fallback_value: float = 0.35) -> float:
        """Helper pentru a obține un prag din config, cu fallback."""
        # Încearcă să obțină atributul direct.
        # Dacă AppConfigMock este folosit și un atribut lipsește, getattr va da AttributeError.
        # Dacă AppConfig real este folosit, la fel.
        try:
            # Pentru AppConfig real, atributele sunt definite la nivel de clasă
            # Pentru instanța self.config, putem accesa direct
            value = getattr(self.config, key)
            if not isinstance(value, (float, int)):
                 logger.warning(f"Config threshold '{key}' is not a number ({value}). Using fallback {default_fallback_value}.")
                 return default_fallback_value
            return float(value)
        except AttributeError:
            logger.warning(f"Config threshold '{key}' not found in AppConfig. Using fallback {default_fallback_value}.")
            return default_fallback_value

    # --- Metodele _extract_keyword_from_prompt, _get_clip_prompt_from_rule, _define_rules ---
    # --- Rămân identice cu cele din v3.3 / v3.4 ---
    # ... (Includeți aici implementările complete din v3.3/v3.4) ...
    def _extract_keyword_from_prompt(self, prompt_lower: str, keywords: List[str], use_regex: bool = True) -> Optional[str]:
        for kw in keywords:
            if use_regex:
                pattern = r'\b' + re.escape(kw) + r'\b'
                if re.search(pattern, prompt_lower): return kw
            elif kw in prompt_lower: return kw
        return None

    def _get_clip_prompt_from_rule(self, prompt: str, operation: Dict[str, Any], rule_params: Dict[str, Any], param_key_base: str) -> Optional[str]:
        lprompt = prompt.lower()
        op_type = operation.get("type", "").lower()
        target_object_from_op = operation.get("target_object", "").lower()
        rule_name = rule_params.get("_rule_name_", "Unknown Rule") 
        default_keyword_from_rule = rule_params.get(f"{param_key_base}_keyword", rule_params.get("main_clip_prompt_keyword", "object of interest"))

        if rule_name == "Glasses Interaction" and param_key_base == "main_clip_prompt":
             if op_type == "add" or "add" in lprompt: return "glasses on face"
             else: return rule_params.get(f"{param_key_base}_keyword", "glasses")

        if rule_params.get(f"{param_key_base}_keyword_from_prompt", False):
            specific_keywords = rule_params.get("keywords_for_extraction", []) 
            if specific_keywords:
                extracted = self._extract_keyword_from_prompt(lprompt, specific_keywords)
                if extracted: logger.debug(f"Extracted '{extracted}' from prompt for {param_key_base}."); return extracted
            if target_object_from_op: logger.debug(f"Using target_object '{target_object_from_op}' for {param_key_base}."); return target_object_from_op
            logger.debug(f"Falling back to default '{default_keyword_from_rule}' for {param_key_base} (from_prompt)."); return default_keyword_from_rule

        elif rule_params.get(f"{param_key_base}_keyword_from_op_target", False):
            if target_object_from_op: logger.debug(f"Using target_object '{target_object_from_op}' for {param_key_base}."); return target_object_from_op
            logger.debug(f"Falling back to default '{default_keyword_from_rule}' for {param_key_base} (from_op_target)."); return default_keyword_from_rule
        
        else: 
            fixed_keyword = rule_params.get(f"{param_key_base}_keyword")
            final_keyword = fixed_keyword if fixed_keyword else default_keyword_from_rule
            logger.debug(f"Using fixed/default '{final_keyword}' for {param_key_base}.")
            return final_keyword

    def _define_rules(self) -> List[Dict[str, Any]]:
        covered_keywords = [ 
            "background", "hair", "hairstyle", "wig", "eye", "eyes", "mouth", "lips", "nose", "face",
            "glasses", "sunglasses", "eyewear", "person", "man", "woman", "boy", "girl",
            "child", "people", "human", "figure", "shirt", "t-shirt", "top", "blouse",
            "jacket", "coat", "dress", "pants", "trousers", "skirt", "suit", "clothing",
            "outfit", "shoes", "boots", "hat", "sky", "water", "sea", "ocean", "river",
            "lake", "pool", "road", "street", "path", "pavement", "building", "house",
            "architecture", "structure", "cat", "dog", "car", "tree"
        ]
        rules = [
            {"name": "Background Interaction", "condition": lambda p, op: "background" in p.lower() or op.get("target_object", "").lower() == "background", "strategy": self._strategy_background, "params": {}, "message": "Background mask"},
            {"name": "Hair Interaction", "condition": lambda p, op: any(kw in p.lower() for kw in ["hair", "hairstyle", "wig"]) or op.get("target_object", "").lower() == "hair", "strategy": self._strategy_semantic_clipseg, "params": {"main_clip_prompt_keyword": "hair", "refine_clip_prompt_keyword": "head", "main_threshold_key": "CLIPSEG_HAIR_THRESHOLD", "refine_threshold_key": "CLIPSEG_HEAD_THRESHOLD", "combine_op": "and"}, "message": "Hair mask"},
            {"name": "Eyes Interaction", "condition": lambda p, op: any(kw in p.lower() for kw in ["eye", "eyes"]) or op.get("target_object", "").lower() in ("eye", "eyes"), "strategy": self._strategy_semantic_clipseg, "params": {"main_clip_prompt_keyword": "eyes", "refine_clip_prompt_keyword": "face", "main_threshold_key": "CLIPSEG_EYES_THRESHOLD", "refine_threshold_key": "CLIPSEG_FACE_THRESHOLD", "combine_op": "and"}, "message": "Eyes mask"},
            {"name": "Mouth/Lips Interaction", "condition": lambda p, op: any(kw in p.lower() for kw in ["mouth", "lips"]) or op.get("target_object", "").lower() in ("mouth", "lips"), "strategy": self._strategy_semantic_clipseg, "params": {"main_clip_prompt_keyword": "mouth lips", "refine_clip_prompt_keyword": "face", "main_threshold_key": "CLIPSEG_MOUTH_THRESHOLD", "refine_threshold_key": "CLIPSEG_FACE_THRESHOLD", "combine_op": "and"}, "message": "Mouth/Lips mask"},
            {"name": "Nose Interaction", "condition": lambda p, op: "nose" in p.lower() or op.get("target_object", "").lower() == "nose", "strategy": self._strategy_semantic_clipseg, "params": {"main_clip_prompt_keyword": "nose", "refine_clip_prompt_keyword": "face", "main_threshold_key": "CLIPSEG_NOSE_THRESHOLD", "refine_threshold_key": "CLIPSEG_FACE_THRESHOLD", "combine_op": "and"}, "message": "Nose mask"},
            {"name": "Face Interaction (General)", "condition": lambda p, op: "face" in p.lower() or op.get("target_object", "").lower() == "face", "strategy": self._strategy_semantic_clipseg, "params": {"main_clip_prompt_keyword": "face", "main_threshold_key": "CLIPSEG_FACE_THRESHOLD"}, "message": "Face mask"},
            {"name": "Glasses Interaction", "condition": lambda p, op: any(kw in p.lower() for kw in ["glasses", "sunglasses", "eyewear"]) or op.get("target_object", "").lower() in ("glasses", "sunglasses", "eyewear"), "strategy": self._strategy_semantic_clipseg, "params": {"main_clip_prompt_keyword": "glasses", "refine_clip_prompt_keyword": "eyes area", "main_threshold_key": "CLIPSEG_OBJECT_THRESHOLD", "refine_threshold_key": "CLIPSEG_FACE_THRESHOLD", "combine_op": "and"}, "message": "Glasses interaction mask"},
            {"name": "Remove Person/Human", "condition": lambda p, op: (op.get("type") == "remove" and self._extract_keyword_from_prompt(p.lower(), ["person", "man", "woman", "boy", "girl", "child", "people", "human", "figure"]) is not None) or op.get("target_object", "").lower() in ("person", "man", "woman", "boy", "girl", "child", "people", "human", "figure"), "strategy": self._strategy_semantic_clipseg, "params": {"main_clip_prompt_keyword_from_prompt": True, "keywords_for_extraction": ["person", "man", "woman", "boy", "girl", "child", "people", "human", "figure"], "main_threshold_key": "CLIPSEG_PERSON_THRESHOLD"}, "message": "Person mask for removal"},
            {"name": "Clothing Interaction", "condition": lambda p, op: self._extract_keyword_from_prompt(p.lower(), ["shirt", "t-shirt", "top", "blouse", "jacket", "coat", "dress", "pants", "trousers", "skirt", "suit", "clothing", "outfit", "shoes", "boots", "hat"]) is not None or op.get("target_object", "").lower() in ["shirt", "t-shirt", "top", "blouse", "jacket", "coat", "dress", "pants", "trousers", "skirt", "suit", "clothing", "outfit", "shoes", "boots", "hat"], "strategy": self._strategy_semantic_clipseg, "params": {"main_clip_prompt_keyword_from_prompt": True, "keywords_for_extraction": ["shirt", "t-shirt", "top", "blouse", "jacket", "coat", "dress", "pants", "trousers", "skirt", "suit", "clothing", "outfit", "shoes", "boots", "hat"], "main_threshold_key": "CLIPSEG_CLOTHING_THRESHOLD", "refine_clip_prompt_keyword": "person", "refine_threshold_key": "CLIPSEG_PERSON_THRESHOLD", "combine_op": "and"}, "message": "Clothing mask"},
            {"name": "Sky Interaction", "condition": lambda p, op: "sky" in p.lower() or op.get("target_object", "").lower() == "sky", "strategy": self._strategy_semantic_clipseg, "params": {"main_clip_prompt_keyword": "sky", "main_threshold_key": "CLIPSEG_SKY_THRESHOLD"}, "message": "Sky mask"},
            {"name": "Water Interaction", "condition": lambda p, op: any(kw in p.lower() for kw in ["water", "sea", "ocean", "river", "lake", "pool"]) or op.get("target_object", "").lower() in ("water", "sea", "ocean", "river", "lake", "pool"), "strategy": self._strategy_semantic_clipseg, "params": {"main_clip_prompt_keyword_from_prompt": True, "keywords_for_extraction": ["water", "sea", "ocean", "river", "lake", "pool"], "main_threshold_key": "CLIPSEG_WATER_THRESHOLD"}, "message": "Water mask"},
            {"name": "Road/Street Interaction", "condition": lambda p, op: any(kw in p.lower() for kw in ["road", "street", "path", "pavement"]) or op.get("target_object", "").lower() in ("road", "street", "path", "pavement"), "strategy": self._strategy_semantic_clipseg, "params": {"main_clip_prompt_keyword_from_prompt": True, "keywords_for_extraction": ["road", "street", "path", "pavement"], "main_threshold_key": "CLIPSEG_ROAD_THRESHOLD"}, "message": "Road/Street mask"},
            {"name": "Building/House Interaction", "condition": lambda p, op: any(kw in p.lower() for kw in ["building", "house", "architecture", "structure"]) or op.get("target_object", "").lower() in ("building", "house", "architecture", "structure"), "strategy": self._strategy_semantic_clipseg, "params": {"main_clip_prompt_keyword_from_prompt": True, "keywords_for_extraction": ["building", "house", "architecture", "structure"], "main_threshold_key": "CLIPSEG_BUILDING_THRESHOLD"}, "message": "Building/House mask"},
            {"name": "Cat Interaction", "condition": lambda p, op: "cat" in p.lower() or op.get("target_object", "").lower() == "cat", "strategy": self._strategy_semantic_clipseg, "params": {"main_clip_prompt_keyword": "cat", "main_threshold_key": "CLIPSEG_CAT_THRESHOLD"}, "message": "Cat mask"},
            {"name": "Dog Interaction", "condition": lambda p, op: "dog" in p.lower() or op.get("target_object", "").lower() == "dog", "strategy": self._strategy_semantic_clipseg, "params": {"main_clip_prompt_keyword": "dog", "main_threshold_key": "CLIPSEG_DOG_THRESHOLD"}, "message": "Dog mask"},
            {"name": "Car Interaction", "condition": lambda p, op: "car" in p.lower() or op.get("target_object", "").lower() == "car", "strategy": self._strategy_semantic_clipseg, "params": {"main_clip_prompt_keyword": "car", "main_threshold_key": "CLIPSEG_CAR_THRESHOLD"}, "message": "Car mask"},
            {"name": "Tree Interaction", "condition": lambda p, op: "tree" in p.lower() or op.get("target_object", "").lower() == "tree", "strategy": self._strategy_semantic_clipseg, "params": {"main_clip_prompt_keyword": "tree", "main_threshold_key": "CLIPSEG_TREE_THRESHOLD"}, "message": "Tree mask"},
            {"name": "Generic Object (from Operation Target)", "condition": lambda p, op: bool(op.get("target_object")) and op.get("target_object", "").lower() not in covered_keywords, "strategy": self._strategy_semantic_clipseg, "params": {"main_clip_prompt_keyword_from_op_target": True, "main_threshold_key": "CLIPSEG_OBJECT_THRESHOLD"}, "message": lambda op: f"{op.get('target_object', 'Object').capitalize()} mask"},
        ]
        logger.debug(f"Defined {len(rules)} mask generation rules.")
        return rules

    def _strategy_semantic_clipseg(self, img_np_bgr: np.ndarray, prompt: str, operation: Dict[str, Any], rule_params: Dict[str, Any], upd: Callable) -> Optional[np.ndarray]:
        """Strategie bazată pe CLIPSeg, cu rafinare opțională. Primește imagine BGR."""
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
        # CORECȚIE: Folosim _get_config_threshold
        main_thresh_factor = self._get_config_threshold(main_thresh_key, self._get_config_threshold("CLIPSEG_DEFAULT_THRESHOLD", 0.35))
        main_thresh_val = int(main_thresh_factor * 255)
        _, main_mask = cv2.threshold(main_seg, main_thresh_val, 255, cv2.THRESH_BINARY)
        logger.debug(f"Applied threshold {main_thresh_val} ({main_thresh_factor*100:.0f}%) for main CLIPSeg mask '{main_clip_text}'.")
        final_combined_mask = main_mask

        refine_clip_text = self._get_clip_prompt_from_rule(prompt, operation, rule_params, "refine_clip_prompt")
        if refine_clip_text and refine_clip_text.strip() and refine_clip_text != main_clip_text:
            upd(0.3, f"CLIPSeg Refine: '{refine_clip_text}'")
            refine_seg = self._clipseg_segment(img_np_bgr, refine_clip_text)
            if refine_seg is not None:
                refine_thresh_key = rule_params.get("refine_threshold_key", "CLIPSEG_DEFAULT_THRESHOLD")
                # CORECȚIE: Folosim _get_config_threshold
                refine_thresh_factor = self._get_config_threshold(refine_thresh_key, self._get_config_threshold("CLIPSEG_DEFAULT_THRESHOLD", 0.35))
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
        """Strategie pentru fundal."""
        upd(0.1, "GrabCut: Subject for background")
        return self._grabcut_subject(img_np_bgr) 

    def generate_mask(
        self, image: Union[Image.Image, np.ndarray], prompt: str = "",
        operation: Optional[Dict[str, Any]] = None, 
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> Dict[str, Any]:
        """Metoda principală de generare a măștii."""
        operation = operation or {} 
        start_time = time.time() 
        def upd(pct: float, desc: str):
            if progress_callback: progress_callback(pct, desc)
            logger.debug(f"MaskGen Progress: {pct*100:.0f}% - {desc}")
        img_np_bgr: Optional[np.ndarray] = None
        try: 
            if isinstance(image, Image.Image): img_pil = image.convert("RGB"); img_np_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
            elif isinstance(image, np.ndarray):
                img_o = image.copy();
                if img_o.ndim==2: img_np_bgr=cv2.cvtColor(img_o,cv2.COLOR_GRAY2BGR)
                elif img_o.shape[2]==4: img_np_bgr=cv2.cvtColor(img_o,cv2.COLOR_RGBA2BGR)
                elif img_o.shape[2]==3: img_np_bgr=img_o
                else: raise ValueError(f"Unsupported NumPy shape: {img_o.shape}")
            else: raise TypeError(f"Unsupported image type: {type(image)}")
            if img_np_bgr.dtype != np.uint8:
                logger.warning(f"Input not uint8 (type: {img_np_bgr.dtype}). Converting.")
                if np.max(img_np_bgr)<=1.0 and img_np_bgr.dtype in [np.float32,np.float64]: img_np_bgr=(img_np_bgr*255).astype(np.uint8)
                else: img_np_bgr=np.clip(img_np_bgr,0,255).astype(np.uint8)
        except Exception as e: logger.error(f"Image input error: {e}", exc_info=True); return {"mask":None, "raw_mask":None, "success":False, "message":f"Image input error: {e}"}
        h, w = img_np_bgr.shape[:2]
        logger.info(f"Generating mask for prompt: '{prompt}', op: {operation}")
        raw_mask_generated, final_message, applied_rule_name = None, "Mask gen failed.", "N/A"
        for rule in self.rules:
            try: 
                if rule["condition"](prompt, operation):
                    applied_rule_name = rule['name']; logger.info(f"Applying rule: {applied_rule_name}"); upd(0.05, f"Rule: {applied_rule_name}")
                    strategy_params = rule.get("params", {}); strategy_params["_rule_name_"] = applied_rule_name
                    raw_mask_candidate = rule["strategy"](img_np_bgr, prompt, operation, strategy_params, upd)
                    if raw_mask_candidate is not None and raw_mask_candidate.size > 0 and raw_mask_candidate.shape[:2] == (h,w):
                        raw_mask_generated = raw_mask_candidate
                        msg_template = rule.get("message", "Mask by rule."); final_message = msg_template(operation) if callable(msg_template) else msg_template
                        if applied_rule_name == "Background Interaction": 
                            upd(0.4,"Post-proc: BG subject"); morphed_subj=self._dynamic_morphology(raw_mask_generated,img_np_bgr)
                            upd(0.7,"Post-proc: BG edge"); refined_subj=self._advanced_edge_refine(morphed_subj,img_np_bgr)
                            final_processed_mask=cv2.bitwise_not(refined_subj); raw_mask_to_return=refined_subj
                            upd(1.0,"BG mask ready"); logger.info(f"Finished rule '{applied_rule_name}' in {time.time()-start_time:.2f}s")
                            return {"mask":final_processed_mask, "raw_mask":raw_mask_to_return, "success":True, "message":final_message}
                        break 
                    else: logger.warning(f"Rule '{applied_rule_name}' no valid mask. Shape: {raw_mask_candidate.shape if raw_mask_candidate is not None else 'None'}")
            except Exception as e_rule: logger.error(f"Error checking/executing rule '{rule.get('name', 'Unnamed')}': {e_rule}", exc_info=True)
        
        if raw_mask_generated is None:
            applied_rule_name = "Hybrid Fallback"; logger.info("No specific rule matched/succeeded. Falling back to hybrid."); upd(0.0, applied_rule_name)
            clip_prompt_hybrid = operation.get("target_object", "").strip()
            if not clip_prompt_hybrid or clip_prompt_hybrid == "subject":
                words = [w for w in prompt.lower().replace("remove","").replace("change","").replace("replace","").replace("add","").replace("make","").split() if w not in ["the","a","an","to","color","of","with","on","from","in","into", "style"]]
                clip_prompt_hybrid = " ".join(words[:3]).strip()
            if not clip_prompt_hybrid : clip_prompt_hybrid = "area of interest"
            raw_mask_generated = self._strategy_hybrid_fallback(img_np_bgr, clip_prompt_hybrid, upd)
            final_message = f"Hybrid fallback mask for '{clip_prompt_hybrid}'."
            if raw_mask_generated is None or raw_mask_generated.size == 0:
                logger.error("Hybrid fallback also failed."); return {"mask":np.zeros((h,w),np.uint8), "raw_mask":None, "success":False, "message":"All mask strategies failed."}

        logger.info(f"Final post-processing for rule: {applied_rule_name}"); upd(0.65, "Final Morphology")
        morphed_mask = self._dynamic_morphology(raw_mask_generated, img_np_bgr)
        upd(0.85, "Final Edge Refinement"); final_processed_mask = self._advanced_edge_refine(morphed_mask, img_np_bgr)
        
        upd(1.0, "Mask ready"); logger.info(f"Finished rule '{applied_rule_name}' in {time.time()-start_time:.2f}s")
        return {"mask": final_processed_mask, "raw_mask": raw_mask_generated, "success": True, "message": final_message}

    def _strategy_hybrid_fallback(self, img_np_bgr: np.ndarray, clip_prompt_text: str, upd: Callable) -> Optional[np.ndarray]:
        """Hybrid fallback strategy combining multiple models."""
        h, w = img_np_bgr.shape[:2]; accum = np.zeros((h, w), np.float32); weights_sum = 0.0
        upd(0.1, "Hybrid: GrabCut"); grabcut_subj = self._grabcut_subject(img_np_bgr)
        if grabcut_subj is not None: weight=0.7; accum += grabcut_subj.astype(np.float32)/255.0 * weight; weights_sum += weight
        yolo = self.models.get_model("yolo")
        if yolo:
            upd(0.25, "Hybrid: YOLO"); yolo_masks_combined = np.zeros((h,w),np.float32); yolo_detected_count=0
            try:
                preds = yolo.predict(source=img_np_bgr, stream=False, imgsz=640, conf=0.2, verbose=False)
                for r in preds:
                    if getattr(r,"masks",None) and hasattr(r.masks,"data") and r.masks.data.numel()>0:
                        masks_data = r.masks.data.cpu().numpy()
                        for m_yolo in masks_data:
                            if m_yolo.size==0: continue
                            yolo_masks_combined += cv2.resize(m_yolo.astype(np.float32),(w,h),cv2.INTER_LINEAR)
                            yolo_detected_count+=1
                if yolo_detected_count > 0: weight=1.0; accum += (yolo_masks_combined/yolo_detected_count)*weight; weights_sum+=weight
            except Exception as e: logger.error(f"Hybrid YOLO error: {e}", exc_info=True)
        if any(k in clip_prompt_text.lower() for k in ["person","face","selfie","man","woman","boy","girl","human"]):
            mp_segmenter = self.models.get_model("mediapipe") 
            if mp_segmenter:
                upd(0.5, "Hybrid: MediaPipe"); 
                try:
                    img_rgb_for_mp = cv2.cvtColor(img_np_bgr, cv2.COLOR_BGR2RGB)
                    results = mp_segmenter.process(img_rgb_for_mp)
                    mm = getattr(results, 'segmentation_mask', None)
                    if mm is not None and mm.size > 0:
                        weight=1.2; accum += cv2.resize(mm.astype(np.float32),(w,h),cv2.INTER_LINEAR)*weight; weights_sum+=weight
                except Exception as e: logger.error(f"Hybrid MediaPipe error: {e}", exc_info=True)
        upd(0.75, f"Hybrid: CLIPSeg '{clip_prompt_text}'")
        clip_gen_seg = self._clipseg_segment(img_np_bgr, clip_prompt_text)
        if clip_gen_seg is not None: weight=1.5; accum += clip_gen_seg.astype(np.float32)/255.0 * weight; weights_sum+=weight
        if weights_sum <= 0: logger.warning("Hybrid fallback: No models contributed."); return np.zeros((h,w),dtype=np.uint8)
        combined_float_mask = np.clip(accum / weights_sum, 0.0, 1.0) 
        # CORECȚIE: Folosim _get_config_threshold
        hybrid_threshold = self._get_config_threshold("HYBRID_MASK_THRESHOLD",0.35)
        final_hybrid_mask = (combined_float_mask > hybrid_threshold).astype(np.uint8)*255
        logger.debug(f"Hybrid fallback mask generated with threshold {hybrid_threshold*100:.0f}%.")
        return final_hybrid_mask

    def _grabcut_subject(self, img_bgr_np: np.ndarray, rect_inset_ratio: float = 0.05) -> Optional[np.ndarray]:
        h, w = img_bgr_np.shape[:2]
        if h < 10 or w < 10 : logger.error(f"GrabCut: Image too small ({w}x{h})."); return None
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
            fb_mask = np.zeros((h,w),np.uint8); 
            try: cv2.rectangle(fb_mask,(rect[0],rect[1]),(rect[0]+rect[2],rect[1]+rect[3]),255,-1)
            except: pass
            return fb_mask

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
            resized_probs = cv2.resize(probs, (img_bgr_np.shape[1],img_bgr_np.shape[0]), interpolation=cv2.INTER_LINEAR)
            return (np.clip(resized_probs,0.0,1.0)*255).astype(np.uint8)
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
        ck_f,ok_f = (0.012,0.006) if min_d>1000 else (0.018,0.008) if min_d>500 else (0.022,0.012) 
        ci,oi = (2,1) 
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
            canny=cv2.Canny(gray,30,100).astype(np.float32)/255.0
            gx=cv2.Sobel(gray,cv2.CV_32F,1,0,ksize=3); gy=cv2.Sobel(gray,cv2.CV_32F,0,1,ksize=3)
            sobel_mag=cv2.magnitude(gx,gy); max_s=np.max(sobel_mag); sobel=sobel_mag/max_s if max_s>0 else np.zeros_like(sobel_mag,dtype=np.float32)
            edges=np.maximum(canny,sobel*0.7)
            influence=0.3; blended_f = mf*(1.0-edges*influence) + edges*(mf*0.1 + 0.05)
            blended_f=np.clip(blended_f,0.0,1.0)
            radius=max(3,int(0.004*min(img_bgr.shape[:2]))); eps_v=0.001 
            guide=img_bgr; src_guide=(blended_f*255).astype(np.uint8)
            if src_guide.ndim==3: src_guide=cv2.cvtColor(src_guide,cv2.COLOR_BGR2GRAY)
            filt_m=cv2.ximgproc.guidedFilter(guide,src_guide,radius,eps_v*eps_v*255*255)
            if filt_m.dtype!=np.uint8: filt_m=np.clip(filt_m,0,255).astype(np.uint8)
            _,thresh_m=cv2.threshold(filt_m,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            k=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
            return cv2.morphologyEx(cv2.morphologyEx(thresh_m,cv2.MORPH_OPEN,k,iterations=1),cv2.MORPH_CLOSE,k,iterations=1)
        except Exception as e: logger.error(f"Advanced edge refine error: {e}",exc_info=True); return self._edge_refine(mask,img_bgr)

    def _edge_refine(self, mask:np.ndarray, img:np.ndarray) -> np.ndarray: # Fallback
        if mask is None or img is None or img.size==0: return mask
        try:
            k_s=max(3,int(0.005*min(img.shape[:2]))//2*2+1); k=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(k_s,k_s))
            return cv2.morphologyEx(cv2.morphologyEx(mask,cv2.MORPH_OPEN,k,iterations=1),cv2.MORPH_CLOSE,k,iterations=1)
        except Exception as e: logger.error(f"Basic edge refine error: {e}",exc_info=True); return mask
    
    def _adaptive_threshold(self, mask_channel: np.ndarray, img: np.ndarray) -> np.ndarray:
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

