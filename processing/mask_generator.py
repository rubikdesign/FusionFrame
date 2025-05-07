#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
MaskGenerator v3.1 — Inteligent, operation-aware mask generation
for FusionFrame. Prioritizes semantic understanding of the prompt.
Includes more specific rules and refined CLIPSeg prompting.
"""

import cv2
import numpy as np
import logging
import torch # Rămâne pentru tipare, deși CLIPSeg intern folosește torch
from typing import Dict, Any, Union, Callable, Optional, List, Tuple
from PIL import Image

# Presupunem că AppConfig real va fi folosit și va conține aceste chei.
# Pentru testare standalone, AppConfigMock este util.
try:
    from config.app_config import AppConfig
except ImportError:
    # Acest logger s-ar putea să nu fie configurat încă dacă AppConfig nu e găsit
    # și setup_logging nu a rulat.
    print("WARNING: AppConfig not found, using AppConfigMock. Ensure AppConfig is in PYTHONPATH for actual use.")
    class AppConfigMock:
        def __init__(self):
            self._config = {
                "CLIPSEG_DEFAULT_THRESHOLD": 0.35,
                "CLIPSEG_BACKGROUND_THRESHOLD": 0.4,
                "CLIPSEG_HAIR_THRESHOLD": 0.45,
                "CLIPSEG_HEAD_THRESHOLD": 0.3,
                "CLIPSEG_FACE_THRESHOLD": 0.35,
                "CLIPSEG_EYES_THRESHOLD": 0.4,
                "CLIPSEG_PERSON_THRESHOLD": 0.5,
                "CLIPSEG_CLOTHING_THRESHOLD": 0.4, # Prag general pentru haine
                "CLIPSEG_SHIRT_THRESHOLD": 0.45, # Mai specific pentru cămăși/tricouri
                "CLIPSEG_SKY_THRESHOLD": 0.4,
                "CLIPSEG_TREE_THRESHOLD": 0.4,
                "CLIPSEG_CAR_THRESHOLD": 0.4,
                "CLIPSEG_CAT_THRESHOLD": 0.45,
                "CLIPSEG_DOG_THRESHOLD": 0.45,
                "CLIPSEG_OBJECT_THRESHOLD": 0.35, # Prag mai mic pentru obiecte generice, poate necesita ajustare
                "HYBRID_MASK_THRESHOLD": 0.35,
            }
        def get(self, key: str, default: Any = None) -> Any:
            return self._config.get(key, default)
    AppConfig = AppConfigMock


from core.model_manager import ModelManager

logger = logging.getLogger(__name__)
# Configurare de bază a logger-ului dacă nu este deja configurat
# Acest lucru este important mai ales dacă AppConfigMock este folosit și AppConfig.setup_logging() nu a rulat.
if not logger.hasHandlers() or not any(isinstance(h, logging.StreamHandler) for h in logger.handlers): # Verificăm și dacă are deja un StreamHandler
    # Adăugăm un handler de consolă dacă nu există niciunul, pentru a vedea logurile
    # Chiar dacă AppConfig.setup_logging() va rula ulterior, acest lucru asigură vizibilitatea logurilor devreme.
    _console_handler = logging.StreamHandler()
    _console_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    logger.addHandler(_console_handler)
    if logger.level == logging.NOTSET: # Dacă nivelul nu a fost setat deloc
        logger.setLevel(logging.INFO) # Setăm un default rezonabil


RuleCondition = Callable[[str, Dict[str, Any]], bool]
MaskStrategy = Callable[[np.ndarray, str, Dict[str, Any], Dict[str, Any], Callable], Optional[np.ndarray]]


class MaskGenerator:
    def __init__(self):
        logger.debug("Initializing MaskGenerator...")
        self.models = ModelManager()
        self.config = AppConfig() 
        self.rules: List[Dict[str, Any]] = self._define_rules()
        logger.debug(f"MaskGenerator initialized with {len(self.rules)} rules.")

    def _define_rules(self) -> List[Dict[str, Any]]:
        """Definește regulile de prioritate pentru generarea măștilor."""
        
        rules = [
            # Reguli foarte specifice și comune prima dată
            {
                "name": "Background Change/Remove",
                "condition": lambda prompt, op: (op.get("type") in ("replace", "remove") and \
                                                ("background" in prompt.lower() or op.get("target_object", "").lower() == "background")) or \
                                                "background" in op.get("target_object", "").lower(),
                "strategy": self._strategy_background,
                "params": {}, 
                "message": "Background mask (inverted subject)"
            },
            {
                "name": "Hair Color/Style/Modification", 
                "condition": lambda prompt, op: "hair" in prompt.lower() or \
                                             op.get("target_object", "").lower() == "hair" or \
                                             "hairstyle" in prompt.lower(),
                "strategy": self._strategy_semantic_clipseg,
                "params": {
                    "main_clip_prompt_keyword": "hair",
                    "refine_clip_prompt_keyword": "head", 
                    "main_threshold_key": "CLIPSEG_HAIR_THRESHOLD",
                    "refine_threshold_key": "CLIPSEG_HEAD_THRESHOLD",
                    "combine_op": "and"
                },
                "message": "Hair mask"
            },
            {
                "name": "Eyes Color/Details",
                "condition": lambda prompt, op: any(kw in prompt.lower() for kw in ["eye", "eyes"]) or \
                                             op.get("target_object", "").lower() in ("eye", "eyes"),
                "strategy": self._strategy_semantic_clipseg,
                "params": {
                    "main_clip_prompt_keyword": "eyes",
                    "refine_clip_prompt_keyword": "face",
                    "main_threshold_key": "CLIPSEG_EYES_THRESHOLD",
                    "refine_threshold_key": "CLIPSEG_FACE_THRESHOLD",
                    "combine_op": "and"
                },
                "message": "Eyes mask"
            },
            {
                "name": "Add/Replace Glasses",
                "condition": lambda prompt, op: "glasses" in prompt.lower() or \
                                             op.get("target_object", "").lower() in ("glasses", "sunglasses"),
                "strategy": self._strategy_semantic_clipseg,
                "params": { 
                    "main_clip_prompt_keyword": "face", 
                    "refine_clip_prompt_keyword": "eyes area", 
                    "main_threshold_key": "CLIPSEG_FACE_THRESHOLD",
                    "refine_threshold_key": "CLIPSEG_EYES_THRESHOLD", 
                    "combine_op": "and" 
                },
                "message": "Region for glasses"
            },
            {
                "name": "Remove Person/Human",
                "condition": lambda prompt, op: (op.get("type") == "remove" and \
                                                self._extract_keyword_from_prompt(prompt.lower(), ["person", "man", "woman", "boy", "girl", "child", "people", "human"]) is not None) or \
                                                op.get("target_object", "").lower() in ("person", "man", "woman", "boy", "girl", "child", "people", "human"),
                "strategy": self._strategy_semantic_clipseg,
                "params": {
                    "main_clip_prompt_keyword_from_prompt": True, 
                    "keywords_for_extraction": ["person", "man", "woman", "boy", "girl", "child", "people", "human"],
                    "main_threshold_key": "CLIPSEG_PERSON_THRESHOLD",
                },
                "message": "Person mask for removal"
            },
            {
                "name": "Change Shirt/Clothing Color",
                "condition": lambda prompt, op: (op.get("type") == "color" and \
                                                (self._extract_keyword_from_prompt(prompt.lower(), ["shirt", "t-shirt", "top", "blouse", "jacket", "coat", "dress", "clothing"]) is not None or \
                                                 op.get("target_object", "").lower() in ["shirt", "t-shirt", "top", "blouse", "jacket", "coat", "dress"])) or \
                                                 (op.get("target_object", "").lower() == "clothing" and "color" in prompt.lower()),
                "strategy": self._strategy_semantic_clipseg,
                "params": {
                    "main_clip_prompt_keyword_from_prompt": True,
                    "keywords_for_extraction": ["shirt", "t-shirt", "top", "blouse", "jacket", "coat", "dress", "clothing"],
                    "main_threshold_key": "CLIPSEG_CLOTHING_THRESHOLD",
                    "refine_clip_prompt_keyword": "person", 
                    "refine_threshold_key": "CLIPSEG_PERSON_THRESHOLD",
                    "combine_op": "and"
                },
                "message": "Clothing mask for color change"
            },
            {
                "name": "Replace/Modify Sky",
                "condition": lambda prompt, op: "sky" in prompt.lower() or op.get("target_object", "").lower() == "sky",
                "strategy": self._strategy_semantic_clipseg,
                "params": {
                    "main_clip_prompt_keyword": "sky",
                    "main_threshold_key": "CLIPSEG_SKY_THRESHOLD",
                },
                "message": "Sky mask"
            },
            {
                "name": "Cat Interaction",
                "condition": lambda prompt, op: "cat" in prompt.lower() or op.get("target_object", "").lower() == "cat",
                "strategy": self._strategy_semantic_clipseg,
                "params": {"main_clip_prompt_keyword": "cat", "main_threshold_key": "CLIPSEG_CAT_THRESHOLD"},
                "message": "Cat mask"
            },
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
            { # Regula generică pentru obiecte, acum un fallback mai jos în listă
                "name": "Generic Object (from Operation Target)",
                "condition": lambda prompt, op: bool(op.get("target_object")) and \
                                             op.get("target_object") not in ["background", "hair", "eyes", "glasses", "person", "sky", "cat", "dog", "car", "tree", "clothing", "shirt"], # Adăugat clothing/shirt
                "strategy": self._strategy_semantic_clipseg,
                "params": {
                    "main_clip_prompt_keyword_from_op_target": True, # Folosește operation['target_object']
                    "main_threshold_key": "CLIPSEG_OBJECT_THRESHOLD",
                },
                "message": lambda op: f"{op.get('target_object', 'Identified Object').capitalize()} mask"
            },
        ]
        logger.debug(f"Defined {len(rules)} mask generation rules.")
        return rules

    def _extract_keyword_from_prompt(self, prompt_lower: str, keywords: List[str]) -> Optional[str]:
        """Helper pentru a extrage primul cuvânt cheie (din lista keywords) găsit în prompt."""
        for kw in keywords:
            # Căutăm cuvântul cheie ca un cuvânt întreg pentru a evita potriviri parțiale (ex: "car" în "scarf")
            # Folosim limite de cuvânt (\b) dacă nu este la început/sfârșit sau are spații în jur.
            # O verificare simplă `in` poate fi prea largă.
            # Pentru simplitate, păstrăm `in` dar conștientizăm limitarea.
            # O soluție mai bună ar folosi regex: import re; if re.search(r'\b' + re.escape(kw) + r'\b', prompt_lower):
            if kw in prompt_lower: 
                return kw
        return None

    def _get_clip_prompt_from_rule(self, prompt: str, operation: Dict[str, Any], rule_params: Dict[str, Any], param_key_base: str) -> Optional[str]:
        lprompt = prompt.lower()
        target_object_from_op = operation.get("target_object", "").lower()
        default_keyword_from_rule = rule_params.get(f"{param_key_base}_keyword", "object of interest") # Un default mai generic

        if rule_params.get(f"{param_key_base}_keyword_from_prompt", False):
            specific_keywords = rule_params.get("keywords_for_extraction") 
            if specific_keywords:
                extracted = self._extract_keyword_from_prompt(lprompt, specific_keywords)
                if extracted: 
                    logger.debug(f"Extracted '{extracted}' from prompt for {param_key_base}")
                    return extracted
            # Fallback la target_object din operație dacă extracția din prompt eșuează
            if target_object_from_op:
                logger.debug(f"Using target_object '{target_object_from_op}' from operation for {param_key_base}")
                return target_object_from_op
            logger.debug(f"Falling back to default keyword '{default_keyword_from_rule}' for {param_key_base} (from_prompt rule)")
            return default_keyword_from_rule

        elif rule_params.get(f"{param_key_base}_keyword_from_op_target", False):
            if target_object_from_op:
                logger.debug(f"Using target_object '{target_object_from_op}' from operation for {param_key_base}")
                return target_object_from_op
            logger.debug(f"Falling back to default keyword '{default_keyword_from_rule}' for {param_key_base} (from_op_target rule)")
            return default_keyword_from_rule
        
        else: # Caută un cuvânt cheie fix definit în regulă
            fixed_keyword = rule_params.get(f"{param_key_base}_keyword")
            logger.debug(f"Using fixed keyword '{fixed_keyword}' or default for {param_key_base}")
            return fixed_keyword if fixed_keyword else default_keyword_from_rule


    def _strategy_semantic_clipseg(self, img_np_bgr: np.ndarray, prompt: str, operation: Dict[str, Any], rule_params: Dict[str, Any], upd: Callable) -> Optional[np.ndarray]:
        """Strategie bazată pe CLIPSeg, cu rafinare opțională. Primește imagine BGR."""
        
        main_clip_text = self._get_clip_prompt_from_rule(prompt, operation, rule_params, "main_clip_prompt")
        if not main_clip_text or not main_clip_text.strip():
            logger.warning(f"Rule '{rule_params.get('_rule_name_', 'Unknown Semantic CLIPSeg')}' failed: main_clip_text is empty or could not be determined.")
            return None
            
        upd(0.1, f"CLIPSeg: '{main_clip_text}'")
        main_seg = self._clipseg_segment(img_np_bgr, main_clip_text) # Pasăm BGR
        if main_seg is None:
            logger.warning(f"CLIPSeg failed for main prompt: '{main_clip_text}'")
            return None

        main_thresh_key = rule_params.get("main_threshold_key", "CLIPSEG_DEFAULT_THRESHOLD")
        # Folosim un default global dacă cheia specifică nu e în config
        main_thresh_factor = self.config.get(main_thresh_key, self.config.get("CLIPSEG_DEFAULT_THRESHOLD"))
        main_thresh_val = int(main_thresh_factor * 255)
        _, main_mask = cv2.threshold(main_seg, main_thresh_val, 255, cv2.THRESH_BINARY)
        logger.debug(f"Applied threshold {main_thresh_val} ({main_thresh_factor*100:.0f}%) for main CLIPSeg mask '{main_clip_text}'.")

        final_combined_mask = main_mask

        refine_clip_text = self._get_clip_prompt_from_rule(prompt, operation, rule_params, "refine_clip_prompt")

        if refine_clip_text and refine_clip_text.strip():
            upd(0.3, f"CLIPSeg Refine: '{refine_clip_text}'")
            refine_seg = self._clipseg_segment(img_np_bgr, refine_clip_text) # Pasăm BGR
            if refine_seg is not None:
                refine_thresh_key = rule_params.get("refine_threshold_key", "CLIPSEG_DEFAULT_THRESHOLD")
                refine_thresh_factor = self.config.get(refine_thresh_key, self.config.get("CLIPSEG_DEFAULT_THRESHOLD"))
                refine_thresh_val = int(refine_thresh_factor * 255)
                _, refine_mask = cv2.threshold(refine_seg, refine_thresh_val, 255, cv2.THRESH_BINARY)
                logger.debug(f"Applied threshold {refine_thresh_val} ({refine_thresh_factor*100:.0f}%) for refine CLIPSeg mask '{refine_clip_text}'.")
                
                combine_op = rule_params.get("combine_op", "and") 
                if combine_op == "and":
                    final_combined_mask = cv2.bitwise_and(main_mask, refine_mask)
                elif combine_op == "or":
                    final_combined_mask = cv2.bitwise_or(main_mask, refine_mask)
                logger.debug(f"Combined main mask ('{main_clip_text}') and refine mask ('{refine_clip_text}') using '{combine_op}'.")
            else:
                logger.warning(f"CLIPSeg failed for refinement prompt: '{refine_clip_text}'. Using main mask only.")
        
        return final_combined_mask

    def _strategy_background(self, img_np_bgr: np.ndarray, prompt: str, operation: Dict[str, Any], rule_params: Dict[str, Any], upd: Callable) -> Optional[np.ndarray]:
        upd(0.1, "GrabCut: Subject for background")
        raw_subj_mask = self._grabcut_subject(img_np_bgr) 
        return raw_subj_mask # Returnează masca subiectului; inversarea se face în generate_mask

    def generate_mask(
        self,
        image: Union[Image.Image, np.ndarray],
        prompt: str = "",
        operation: Optional[Dict[str, Any]] = None, 
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> Dict[str, Any]:
        
        operation = operation or {} 

        def upd(pct: float, desc: str):
            if progress_callback:
                progress_callback(pct, desc)
            logger.debug(f"MaskGen Progress: {pct*100:.0f}% - {desc}")

        # Standardize input image to BGR uint8 numpy array
        img_np_bgr: Optional[np.ndarray] = None
        if isinstance(image, Image.Image):
            try:
                img_pil = image.convert("RGB") # Asigurăm format RGB pentru PIL
                img_np_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
            except Exception as e_pil_conv:
                logger.error(f"Error converting PIL Image to BGR NumPy: {e_pil_conv}", exc_info=True)
                return {"mask": None, "raw_mask": None, "success": False, "message": "Image conversion error (PIL to NumPy)."}
        elif isinstance(image, np.ndarray):
            img_np_orig = image.copy()
            if img_np_orig.ndim == 2: 
                img_np_bgr = cv2.cvtColor(img_np_orig, cv2.COLOR_GRAY2BGR)
            elif img_np_orig.shape[2] == 4: 
                img_np_bgr = cv2.cvtColor(img_np_orig, cv2.COLOR_RGBA2BGR)
            elif img_np_orig.shape[2] == 3: 
                # Presupunem că e BGR dacă e NumPy cu 3 canale, deoarece OpenCV e sursa comună.
                # Dacă ar fi RGB, _clipseg_segment oricum îl convertește la BGR intern pentru procesare,
                # apoi înapoi la RGB pentru PIL. Aici, păstrăm BGR ca format intern principal.
                img_np_bgr = img_np_orig 
            else:
                logger.error(f"Unsupported numpy image format: {img_np_orig.shape}")
                return {"mask": None, "raw_mask": None, "success": False, "message": "Unsupported NumPy image format."}
        else:
            logger.error(f"Unsupported image type: {type(image)}")
            return {"mask": None, "raw_mask": None, "success": False, "message": "Unsupported image type."}

        if img_np_bgr.dtype != np.uint8:
            logger.warning(f"Input image numpy array is not uint8 (type: {img_np_bgr.dtype}). Converting.")
            if np.max(img_np_bgr) <= 1.0 and (img_np_bgr.dtype == np.float32 or img_np_bgr.dtype == np.float64):
                 img_np_bgr = (img_np_bgr * 255).astype(np.uint8)
            else: # Încercăm o conversie directă, cu riscuri
                 try:
                     img_np_bgr = np.clip(img_np_bgr, 0, 255).astype(np.uint8)
                 except Exception as e_astype:
                     logger.error(f"Failed to convert image to uint8: {e_astype}", exc_info=True)
                     return {"mask": None, "raw_mask": None, "success": False, "message": "Image data type conversion error."}


        h, w = img_np_bgr.shape[:2]
        lprompt = prompt.lower()
        
        # Se presupune că `operation` este deja populat de `OperationAnalyzer`
        # Dacă nu, logica de fallback din interiorul `_define_rules` sau `_get_clip_prompt_from_rule` va încerca să deducă.
        logger.info(f"Generating mask for prompt: '{prompt}', pre-analyzed operation: {operation}")

        raw_mask_generated = None 
        final_message = "Mask generation failed or no specific rule matched."
        applied_rule_name = "N/A"

        for rule in self.rules:
            if rule["condition"](prompt, operation):
                applied_rule_name = rule['name']
                logger.info(f"Applying rule: {applied_rule_name}")
                upd(0.05, f"Rule: {applied_rule_name}")
                
                strategy_params = rule.get("params", {})
                strategy_params["_rule_name_"] = applied_rule_name

                raw_mask_candidate = rule["strategy"](img_np_bgr, prompt, operation, strategy_params, upd)
                
                if raw_mask_candidate is not None and raw_mask_candidate.size > 0 and raw_mask_candidate.shape[:2] == (h,w):
                    raw_mask_generated = raw_mask_candidate
                    current_message = rule.get("message", "Mask generated by rule.")
                    final_message = current_message(operation) if callable(current_message) else current_message
                    
                    if applied_rule_name == "Background Change/Remove":
                        upd(0.4, "Morphology on subject (for background mask)")
                        morphed_subj_mask = self._dynamic_morphology(raw_mask_generated, img_np_bgr)
                        upd(0.7, "Edge refinement on subject (for background mask)")
                        refined_subj_mask = self._advanced_edge_refine(morphed_subj_mask, img_np_bgr)
                        final_processed_mask = cv2.bitwise_not(refined_subj_mask)
                        raw_mask_to_return = refined_subj_mask 
                        upd(1.0, "Background mask ready")
                        return {"mask": final_processed_mask, "raw_mask": raw_mask_to_return, "success": True, "message": final_message}
                    
                    break 
                else:
                    logger.warning(f"Rule '{applied_rule_name}' executed but returned no mask, empty mask, or mismatched shape (expected ({h},{w}), got {raw_mask_candidate.shape if raw_mask_candidate is not None else 'None'}).")
        
        if raw_mask_generated is None:
            logger.info("No specific rule matched or all matching rules failed. Falling back to hybrid strategy.")
            applied_rule_name = "Hybrid Fallback" # Actualizăm numele regulii aplicate
            upd(0.0, applied_rule_name)
            
            clip_prompt_hybrid = operation.get("target_object", "").strip()
            if not clip_prompt_hybrid or clip_prompt_hybrid == "subject":
                words = [w for w in lprompt.replace("remove","").replace("change","").replace("replace","").replace("add","").replace("make","").split() if w not in ["the","a","an","to","color","of","with","on","from","in","into", "style"]]
                clip_prompt_hybrid = " ".join(words[:3]).strip() # Primele 3 cuvinte relevante
            
            if not clip_prompt_hybrid : clip_prompt_hybrid = "area of interest" # Un default mai bun
            
            raw_mask_generated = self._strategy_hybrid_fallback(img_np_bgr, clip_prompt_hybrid, upd)
            final_message = f"Hybrid fallback mask for '{clip_prompt_hybrid}'." # Mesaj mai specific
            if raw_mask_generated is None or raw_mask_generated.size == 0:
                logger.error("Hybrid fallback mask generation also failed.")
                return {"mask": np.zeros((h, w), np.uint8), "raw_mask": None, "success": False, "message": "All mask generation strategies failed."}

        logger.info(f"Performing final post-processing for rule: {applied_rule_name}")
        upd(0.65, "Final Morphology")
        morphed_mask = self._dynamic_morphology(raw_mask_generated, img_np_bgr)
        upd(0.85, "Final Edge Refinement")
        final_processed_mask = self._advanced_edge_refine(morphed_mask, img_np_bgr)
        
        upd(1.0, "Mask ready")
        return {"mask": final_processed_mask, "raw_mask": raw_mask_generated, "success": True, "message": final_message}

    def _strategy_hybrid_fallback(self, img_np_bgr: np.ndarray, clip_prompt_text: str, upd: Callable) -> Optional[np.ndarray]:
        h, w = img_np_bgr.shape[:2]
        accum = np.zeros((h, w), np.float32)
        active_model_count = 0 # Număr de modele care au contribuit efectiv

        upd(0.1, "Hybrid: GrabCut subject")
        grabcut_subj = self._grabcut_subject(img_np_bgr)
        if grabcut_subj is not None:
            accum += grabcut_subj.astype(np.float32) / 255.0 * 0.8 # Ponderare
            active_model_count += 1

        yolo = self.models.get_model("yolo")
        if yolo:
            upd(0.25, "Hybrid: YOLO segmentation")
            try:
                preds = yolo.predict(source=img_np_bgr, stream=False, imgsz=640, conf=0.25, verbose=False)
                yolo_masks_combined = np.zeros((h, w), np.float32)
                yolo_detected_count = 0
                for r in preds:
                    if getattr(r, "masks", None) and hasattr(r.masks, "data") and r.masks.data.numel() > 0:
                        masks_data = r.masks.data.cpu().numpy()
                        for m_yolo in masks_data:
                            if m_yolo.size == 0: continue
                            m_resized = cv2.resize(m_yolo.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
                            yolo_masks_combined += m_resized
                            yolo_detected_count +=1
                if yolo_detected_count > 0:
                    accum += (yolo_masks_combined / yolo_detected_count) * 1.0 # Ponderare
                    active_model_count += 1 
            except Exception as e: logger.error(f"Hybrid YOLO error: {e}", exc_info=True)

        if any(k in clip_prompt_text.lower() for k in ["person", "face", "selfie", "man", "woman", "boy", "girl", "human"]):
            mp_segmenter = self.models.get_model("mediapipe") 
            if mp_segmenter:
                upd(0.5, "Hybrid: MediaPipe segmentation")
                try:
                    img_rgb_for_mp = cv2.cvtColor(img_np_bgr, cv2.COLOR_BGR2RGB)
                    res_mp = mp_segmenter.process(img_rgb_for_mp)
                    mm = getattr(res_mp, 'segmentation_mask', None)
                    if mm is not None and mm.size > 0:
                        mmf = cv2.resize(mm.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
                        accum += mmf * 1.2 # Ponderare
                        active_model_count += 1
                except Exception as e: logger.error(f"Hybrid MediaPipe error: {e}", exc_info=True)
        
        upd(0.75, f"Hybrid: CLIPSeg for '{clip_prompt_text}'")
        clip_gen_seg = self._clipseg_segment(img_np_bgr, clip_prompt_text)
        if clip_gen_seg is not None:
            accum += clip_gen_seg.astype(np.float32) / 255.0 * 1.5 # Ponderare
            active_model_count += 1
        
        if active_model_count == 0: # Schimbat din count în active_model_count
            logger.warning("Hybrid fallback: No models contributed effectively. Returning a blank mask.")
            return np.zeros((h,w), dtype=np.uint8)

        # Normalizăm prin numărul de modele care au contribuit efectiv
        combined_float_mask = np.clip(accum / active_model_count, 0.0, 1.0) 
        hybrid_threshold_val = self.config.get("HYBRID_MASK_THRESHOLD", 0.35)
        final_hybrid_mask = (combined_float_mask > hybrid_threshold_val).astype(np.uint8) * 255
        return final_hybrid_mask

    def _grabcut_subject(self, img_bgr_np: np.ndarray, rect_inset_ratio: float = 0.05) -> Optional[np.ndarray]:
        h, w = img_bgr_np.shape[:2]
        if h == 0 or w == 0: logger.error("GrabCut received an empty image."); return None
        if img_bgr_np.ndim < 3 or img_bgr_np.shape[2] < 3:
            logger.error(f"GrabCut expects a 3-channel BGR image, got {img_bgr_np.shape}."); return None

        mask_gc = np.zeros((h, w), np.uint8)
        bgd_model, fgd_model = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
        
        dx, dy = int(rect_inset_ratio * w), int(rect_inset_ratio * h) 
        rect_w, rect_h = w - 2*dx, h - 2*dy
        rect = (max(1, dx), max(1, dy), max(1, rect_w), max(1, rect_h)) # Asigură valori pozitive
        if rect[2] <= 0 or rect[3] <= 0: logger.error("Image too small for GrabCut rect."); return None

        try:
            cv2.grabCut(img_bgr_np, mask_gc, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
            return np.where((mask_gc == cv2.GC_FGD) | (mask_gc == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
        except Exception as e:
            logger.error(f"GrabCut error: {e}", exc_info=True)
            fb_mask = np.zeros((h, w), np.uint8)
            try: cv2.rectangle(fb_mask, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), 255, -1)
            except: pass
            return fb_mask

    def _clipseg_segment(self, img_bgr_np: np.ndarray, text_prompt: str) -> Optional[np.ndarray]:
        """Rulează CLIPSeg. Input: BGR NumPy. Output: uint8 mask [0–255]."""
        bundle = self.models.get_model("clipseg")
        if not bundle or "processor" not in bundle or "model" not in bundle:
            logger.warning("CLIPSeg model/processor not in ModelManager.")
            return None
        processor, model = bundle["processor"], bundle["model"]
        try:
            img_rgb_np = cv2.cvtColor(img_bgr_np, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(img_rgb_np)
        except Exception as e_pil:
            logger.error(f"PIL conversion error for CLIPSeg: {e_pil}", exc_info=True); return None
            
        effective_text = text_prompt.strip() if text_prompt and text_prompt.strip() else "object"
        logger.debug(f"CLIPSeg with text: '{effective_text}'")
        try:
            inputs = processor(text=[effective_text], images=[pil_image], return_tensors="pt", padding="max_length", truncation=True) # Adăugat max_length și truncation
        except Exception as e_proc:
            logger.error(f"CLIPSeg processor error ('{effective_text}'): {e_proc}", exc_info=True); return None
        processed_inputs = {}
        try:
            target_device = model.device
            target_dtype = model.dtype if hasattr(model, 'dtype') and model.dtype is not None else torch.float32
            for k, v_tensor in inputs.items():
                processed_inputs[k] = v_tensor.to(target_device, dtype=target_dtype if v_tensor.dtype.is_floating_point else v_tensor.dtype)
        except Exception as e_device:
            logger.error(f"Device move error for CLIPSeg ('{effective_text}'): {e_device}", exc_info=True); return None
        try:
            with torch.no_grad(): outputs = model(**processed_inputs)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
            if logits.ndim == 4: logits = logits.squeeze(0).squeeze(0) 
            elif logits.ndim == 3: logits = logits.squeeze(0)
            if logits.ndim != 2: logger.error(f"CLIPSeg logits ndim {logits.ndim} for '{effective_text}'. Shape: {logits.shape}"); return None
            probs = torch.sigmoid(logits).cpu().numpy().astype(np.float32)
            if probs.size == 0 or probs.shape[0] == 0 or probs.shape[1] == 0: logger.error(f"CLIPSeg 'probs' empty for '{effective_text}'."); return None
            target_h, target_w = img_bgr_np.shape[:2]
            mask_resized_float = cv2.resize(probs, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            return (np.clip(mask_resized_float, 0.0, 1.0) * 255).astype(np.uint8)
        except Exception as e_runtime:
            logger.error(f"Runtime/Resize error in CLIPSeg ('{effective_text}'): {e_runtime}", exc_info=True); return None

    def _morphology(self, mask: np.ndarray, close_k:int, open_k:int, close_iter:int, open_iter:int) -> np.ndarray:
        if mask is None or mask.size == 0: return mask
        if close_k <= 0 or open_k <=0: return mask
        close_k = close_k if close_k % 2 != 0 else close_k + 1
        open_k = open_k if open_k % 2 != 0 else open_k + 1
        try:
            ker_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
            ker_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k, open_k))
            m = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, ker_close, iterations=close_iter)
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN, ker_open, iterations=open_iter)
            return m
        except Exception as e_morph:
            logger.error(f"Morphology error: {e_morph}", exc_info=True); return mask

    def _dynamic_morphology(self, mask: np.ndarray, img: np.ndarray) -> np.ndarray:
        if mask is None or img is None or img.size == 0: return mask
        h, w = img.shape[:2]; min_dim = min(h,w)
        close_k = max(3, int((0.015 if min_dim > 1000 else 0.01) * min_dim) // 2 * 2 + 1) 
        open_k = max(3, int((0.007 if min_dim > 1000 else 0.005) * min_dim) // 2 * 2 + 1)
        close_iter = 3 if min_dim > 700 else 2; open_iter = 2 if min_dim > 700 else 1
        logger.debug(f"Dynamic morphology: close_k={close_k}, open_k={open_k}, iters={close_iter},{open_iter}")
        return self._morphology(mask, close_k, open_k, close_iter, open_iter)

    def _advanced_edge_refine(self, mask: np.ndarray, img_bgr: np.ndarray) -> np.ndarray:
        if mask is None or img_bgr is None or img_bgr.size == 0: return mask
        try:
            if not hasattr(cv2, 'ximgproc') or not hasattr(cv2.ximgproc, 'guidedFilter'):
                return self._edge_refine(mask, img_bgr)
        except AttributeError: return self._edge_refine(mask, img_bgr)
        try:
            mask_float = mask.astype(np.float32) / 255.0
            gray_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            edges_canny = cv2.Canny(gray_img, 30, 100).astype(np.float32) / 255.0
            grad_x = cv2.Sobel(gray_img, cv2.CV_32F, 1, 0, ksize=3); grad_y = cv2.Sobel(gray_img, cv2.CV_32F, 0, 1, ksize=3)
            edges_sobel_mag = cv2.magnitude(grad_x, grad_y)
            max_s = np.max(edges_sobel_mag); edges_sobel = edges_sobel_mag/max_s if max_s > 0 else np.zeros_like(edges_sobel_mag, dtype=np.float32)
            combined_edges = np.maximum(edges_canny, edges_sobel)
            edge_influence = 0.3 # Ajustat
            # Blend mai atent: întărește masca unde e puternică și nu sunt muchii, și vice-versa
            blended_float = mask_float * (1.0 - combined_edges * edge_influence) + \
                            combined_edges * (mask_float * 0.2 + 0.1) # Adaugă o componentă mică de muchie
            blended_float = np.clip(blended_float, 0.0, 1.0)

            radius = max(3, int(0.005 * min(img_bgr.shape[:2]))) 
            eps_val = 0.005 
            guide_img = img_bgr # Guided filter se așteaptă la BGR dacă src e monocrom
            
            blended_u8_for_guide = (blended_float * 255).astype(np.uint8)
            # Asigurăm că src pentru guidedFilter este monocrom
            if blended_u8_for_guide.ndim == 3: blended_u8_for_guide = cv2.cvtColor(blended_u8_for_guide, cv2.COLOR_BGR2GRAY)

            filtered_mask = cv2.ximgproc.guidedFilter(guide=guide_img, src=blended_u8_for_guide, radius=radius, eps=eps_val*eps_val*255*255) 
            if filtered_mask.dtype != np.uint8: filtered_mask = np.clip(filtered_mask, 0, 255).astype(np.uint8)
            _, thresholded_mask = cv2.threshold(filtered_mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            k_size = 3 # Kernel mai mic pentru curățare fină
            kernel_final_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
            cleaned_mask = cv2.morphologyEx(thresholded_mask, cv2.MORPH_OPEN, kernel_final_clean, iterations=1)
            cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel_final_clean, iterations=1) # O singură închidere
            return cleaned_mask
        except Exception as e_adv:
            logger.error(f"Advanced edge refine error: {e_adv}. Falling back.", exc_info=True)
            return self._edge_refine(mask, img_bgr) 

    def _edge_refine(self, mask: np.ndarray, img: np.ndarray) -> np.ndarray: # Fallback
        if mask is None or img is None or img.size == 0: return mask
        try:
            kernel_size = max(3, int(0.005 * min(img.shape[:2])) // 2 * 2 + 1)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            opened_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
            closed_mask = cv2.morphologyEx(opened_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
            return closed_mask
        except Exception as e_basic:
            logger.error(f"Basic edge refine error: {e_basic}", exc_info=True); return mask
    
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

