#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
MaskGenerator v4.0 — Integrare SAM + Context ImageAnalyzer
Prioritizează SAM ghidat de YOLO/Context, folosește CLIPSeg/Altele ca fallback.
"""

import cv2
import numpy as np
import logging
import torch # Rămâne pentru tipare și inferența SAM
from typing import Dict, Any, Union, Callable, Optional, List, Tuple
from PIL import Image
import time # Pentru debug timp

try: from config.app_config import AppConfig
except ImportError: print("WARNING: AppConfig not found, using Mock."); from .utils import AppConfigMock; AppConfig = AppConfigMock

from core.model_manager import ModelManager
# NOU: Importăm și ImageAnalyzer pentru a putea folosi contextul dacă nu e pasat
from processing.analyzer import ImageAnalyzer

logger = logging.getLogger(__name__)
if not logger.hasHandlers(): # Basic logging if not configured
     _ch = logging.StreamHandler(); _ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)s:%(name)s:%(message)s"))
     logger.addHandler(_ch); logger.setLevel(logging.INFO)

# Tipuri pentru claritate
RuleCondition = Callable[[str, Dict[str, Any], Optional[Dict[str, Any]]], bool] # Adăugat image_context opțional
MaskStrategy = Callable[[np.ndarray, str, Dict[str, Any], Optional[Dict[str, Any]], Dict[str, Any], Callable], Optional[np.ndarray]] # Adăugat image_context opțional


class MaskGenerator:
    def __init__(self):
        logger.debug("Initializing MaskGenerator v4.0 (SAM Integration)...")
        self.models = ModelManager()
        self.config = AppConfig() if 'AppConfig' in locals() and callable(AppConfig) else None # Folosim AppConfig real
        if self.config is None: logger.error("AppConfig instance could not be created in MaskGenerator!"); # Ar trebui să fie fatal?
        # NOU: Instanțiem ImageAnalyzer pentru a-l folosi dacă contextul nu e furnizat
        self.image_analyzer = ImageAnalyzer()
        self.rules: List[Dict[str, Any]] = self._define_rules()
        logger.debug(f"MaskGenerator initialized with {len(self.rules)} rules.")

    def _define_rules(self) -> List[Dict[str, Any]]:
        """Definește regulile de prioritate, acum cu condiții bazate și pe context și strategii SAM."""

        # --- Helper Conditions ---
        def is_op_type(op, types): return op.get("type") in types
        def has_target(op, targets): return op.get("target_object", "").lower() in targets
        def prompt_contains(prompt, words): return any(w in prompt.lower() for w in words)
        def context_has_object(ctx, labels, min_conf=0.4): # Verifică dacă YOLO a detectat obiectul
             if not ctx or 'detected_objects' not in ctx.get('scene_info',{}): return False
             return any(obj.get('label') in labels and obj.get('confidence', 0) >= min_conf for obj in ctx['scene_info']['detected_objects'])

        rules = [
            # 1. Reguli SAM + YOLO (prioritate maximă dacă YOLO detectează obiectul țintă)
            {
                "name": "SAM + YOLO: Person",
                "condition": lambda p, op, ctx: (is_op_type(op, ["remove", "color", "replace", "add"]) and \
                                                (has_target(op, ["person", "man", "woman", "boy", "girl", "child", "people", "human"]) or \
                                                 prompt_contains(p, ["person", "man", "woman", "boy", "girl", "child", "people", "human"]))) and \
                                                context_has_object(ctx, ["person"]),
                "strategy": self._strategy_sam_yolo,
                "params": {"yolo_labels": ["person"]}, # Label căutat în detecțiile YOLO
                "message": "SAM mask guided by YOLO detection for Person"
            },
            {
                "name": "SAM + YOLO: Car",
                "condition": lambda p, op, ctx: (is_op_type(op, ["remove", "color", "replace", "add"]) and \
                                                (has_target(op, ["car", "vehicle", "automobile"]) or prompt_contains(p, ["car", "vehicle"]))) and \
                                                context_has_object(ctx, ["car", "truck", "bus"]), # Poate prinde și camioane/autobuze
                "strategy": self._strategy_sam_yolo,
                "params": {"yolo_labels": ["car", "truck", "bus"]},
                "message": "SAM mask guided by YOLO detection for Car/Vehicle"
            },
             {
                "name": "SAM + YOLO: Cat",
                "condition": lambda p, op, ctx: (is_op_type(op, ["remove", "color", "replace", "add"]) and \
                                                (has_target(op, ["cat", "kitten"]) or prompt_contains(p, ["cat", "kitten"]))) and \
                                                context_has_object(ctx, ["cat"]),
                "strategy": self._strategy_sam_yolo,
                "params": {"yolo_labels": ["cat"]},
                "message": "SAM mask guided by YOLO detection for Cat"
            },
             {
                "name": "SAM + YOLO: Dog",
                "condition": lambda p, op, ctx: (is_op_type(op, ["remove", "color", "replace", "add"]) and \
                                                (has_target(op, ["dog", "puppy"]) or prompt_contains(p, ["dog", "puppy"]))) and \
                                                context_has_object(ctx, ["dog"]),
                "strategy": self._strategy_sam_yolo,
                "params": {"yolo_labels": ["dog"]},
                "message": "SAM mask guided by YOLO detection for Dog"
            },
            # Adaugă aici alte reguli SAM + YOLO pentru obiecte comune detectabile

            # 2. Reguli Specifice (CLIPSeg sau alte metode, prioritate medie)
             {
                "name": "Background (Inverted Subject)",
                "condition": lambda p, op, ctx: op.get("type") == "background" or has_target(op, ["background", "scene", "backdrop"]),
                "strategy": self._strategy_background, # Folosește Rembg/MediaPipe/Grabcut
                "params": {},
                "message": "Background mask (inverted subject)"
            },
            {   "name": "Hair (CLIPSeg Refined)", "condition": lambda p, op, ctx: has_target(op, ["hair", "hairstyle"]) or prompt_contains(p, ["hair", "hairstyle"]),
                "strategy": self._strategy_semantic_clipseg,
                "params": { "main_clip_prompt_keyword": "hair", "refine_clip_prompt_keyword": "head", "main_threshold_key": "CLIPSEG_HAIR_THRESHOLD", "refine_threshold_key": "CLIPSEG_HEAD_THRESHOLD", "combine_op": "and"}, "message": "Hair mask (CLIPSeg)"
            },
            {   "name": "Eyes (CLIPSeg Refined)", "condition": lambda p, op, ctx: has_target(op, ["eye", "eyes"]) or prompt_contains(p, ["eye", "eyes"]),
                "strategy": self._strategy_semantic_clipseg,
                "params": { "main_clip_prompt_keyword": "eyes", "refine_clip_prompt_keyword": "face", "main_threshold_key": "CLIPSEG_EYES_THRESHOLD", "refine_threshold_key": "CLIPSEG_FACE_THRESHOLD", "combine_op": "and"}, "message": "Eyes mask (CLIPSeg)"
            },
            {   "name": "Glasses Area (CLIPSeg)", "condition": lambda p, op, ctx: has_target(op, ["glasses", "sunglasses"]) or prompt_contains(p, ["glasses", "sunglasses"]),
                 "strategy": self._strategy_semantic_clipseg, # Sau SAM ghidat de 'eyes area'? Poate mai robust.
                 "params": { "main_clip_prompt_keyword": "eyes area", "main_threshold_key": "CLIPSEG_EYES_THRESHOLD", "refine_clip_prompt_keyword": "face", "refine_threshold_key":"CLIPSEG_FACE_THRESHOLD", "combine_op":"and"}, "message": "Glasses area mask (CLIPSeg)"
             },
            {   "name": "Clothing (CLIPSeg Refined)", "condition": lambda p, op, ctx: has_target(op, ["shirt", "t-shirt", "top", "blouse", "jacket", "coat", "dress", "clothing"]) or prompt_contains(p, ["shirt", "jacket", "dress", "clothing"]), # Am simplificat prompt_contains
                "strategy": self._strategy_semantic_clipseg,
                "params": {"main_clip_prompt_keyword_from_op_target": True, "keywords_for_extraction":["shirt","t-shirt","top","blouse","jacket","coat","dress","clothing"], "main_threshold_key":"CLIPSEG_CLOTHING_THRESHOLD", "refine_clip_prompt_keyword":"person", "refine_threshold_key":"CLIPSEG_PERSON_THRESHOLD", "combine_op":"and"}, "message": "Clothing mask (CLIPSeg)"
             },
             {  "name": "Sky (CLIPSeg)", "condition": lambda p, op, ctx: has_target(op, ["sky"]) or prompt_contains(p, ["sky"]),
                "strategy": self._strategy_semantic_clipseg, "params": {"main_clip_prompt_keyword": "sky", "main_threshold_key": "CLIPSEG_SKY_THRESHOLD"}, "message": "Sky mask (CLIPSeg)"
             },
             {  "name": "Tree (CLIPSeg)", "condition": lambda p, op, ctx: has_target(op, ["tree", "trees"]) or prompt_contains(p, ["tree", "trees"]),
                 "strategy": self._strategy_semantic_clipseg, "params": {"main_clip_prompt_keyword": "tree", "main_threshold_key": "CLIPSEG_TREE_THRESHOLD"}, "message": "Tree mask (CLIPSeg)"
             },

            # 3. Regula Generică Obiect (SAM + YOLO dacă e detectat, altfel CLIPSeg)
            {
                "name": "Generic Object (Try SAM+YOLO first, then CLIPSeg)",
                "condition": lambda p, op, ctx: bool(op.get("target_object")) and not has_target(op, ["background", "hair", "eyes", "glasses", "person", "sky", "clothing", "shirt", "car", "cat", "dog", "tree"]), # Exclude cele deja tratate
                "strategy": self._strategy_generic_object, # Strategie nouă care decide intern
                "params": {}, # Parametrii sunt deduși în strategie
                "message": lambda op: f"{op.get('target_object', 'Generic Object').capitalize()} mask (Auto strategy)"
            },
        ]
        logger.debug(f"Defined {len(rules)} mask generation rules.")
        return rules

    # --- Metode Helper (Extracție, Prompting - păstrate și adaptate) ---
    def _extract_keyword_from_prompt(self, prompt_lower: str, keywords: List[str]) -> Optional[str]:
        """Helper simplu pentru extragere keyword."""
        for kw in keywords:
            # Folosim regex pentru potrivire cuvânt întreg
            if re.search(r'\b' + re.escape(kw) + r'\b', prompt_lower): return kw
        return None

    def _get_clip_prompt_from_rule(self, prompt: str, operation: Dict[str, Any], rule_params: Dict[str, Any], param_key_base: str) -> Optional[str]:
        """Determină promptul text pentru CLIPSeg pe baza regulii."""
        lprompt = prompt.lower(); target = operation.get("target_object", "").lower()
        default_kw = rule_params.get(f"{param_key_base}_keyword", "area of interest")

        if rule_params.get(f"{param_key_base}_keyword_from_prompt", False):
            extracted = self._extract_keyword_from_prompt(lprompt, rule_params.get("keywords_for_extraction", []))
            if extracted: return extracted
            if target: return target # Fallback la target din op
            return default_kw
        elif rule_params.get(f"{param_key_base}_keyword_from_op_target", False):
            if target: return target
            return default_kw
        else: return rule_params.get(f"{param_key_base}_keyword", default_kw)


    # --- Strategii de Generare Măști ---

    # NOU: Strategie SAM ghidată de Bounding Box din YOLO
    def _strategy_sam_yolo(self, img_np_bgr: np.ndarray, prompt: str, operation: Dict[str, Any],
                           image_context: Optional[Dict[str, Any]], rule_params: Dict[str, Any], upd: Callable) -> Optional[np.ndarray]:
        """Strategie: Găsește obiectul cu YOLO, apoi segmentează cu SAM folosind box-ul."""
        yolo_labels_to_find = rule_params.get("yolo_labels", [])
        if not yolo_labels_to_find or not image_context:
            logger.warning("SAM+YOLO strategy requires 'yolo_labels' in rule params and image_context."); return None

        # 1. Găsește Bounding Box-ul relevant din context (de la ImageAnalyzer)
        target_object_op = operation.get("target_object", "").lower() # Obiectul specificat în prompt/operație
        best_box = None
        highest_conf = 0.0
        detected_objects = image_context.get('scene_info', {}).get('detected_objects', [])

        if not detected_objects: logger.debug("No detected objects in context for SAM+YOLO."); return None

        for obj in detected_objects:
            label = obj.get('label')
            conf = obj.get('confidence', 0)
            box_norm = obj.get('box_normalized') # [x1, y1, x2, y2]
            if label in yolo_labels_to_find and box_norm and conf > highest_conf:
                # Verificăm dacă obiectul detectat se potrivește cât de cât cu cel din operație (dacă există)
                # Exemplu simplu: dacă op e pt 'person', dar YOLO găsește și 'dog', luăm 'person'
                is_primary_target = (target_object_op and label == target_object_op)
                # Prioritizăm targetul exact, altfel luăm cea mai bună detecție din lista permisă
                if is_primary_target or not best_box or (conf > highest_conf and not (best_box and best_box['label'] == target_object_op)):
                    h, w = img_np_bgr.shape[:2]
                    # Convertim box normalizat în coordonate absolute [x_min, y_min, x_max, y_max]
                    box_abs = np.array([box_norm[0] * w, box_norm[1] * h, box_norm[2] * w, box_norm[3] * h]).astype(int)
                    best_box = {'label': label, 'box': box_abs}
                    highest_conf = conf

        if best_box:
            logger.info(f"SAM+YOLO: Found '{best_box['label']}' (conf: {highest_conf:.2f}). Using its box for SAM.")
            upd(0.2, f"SAM prompt: Box for {best_box['label']}")
            # 2. Rulează SAM cu bounding box-ul găsit
            # Asigurăm că pasăm imaginea în format RGB NumPy la SAM
            try: img_np_rgb = cv2.cvtColor(img_np_bgr, cv2.COLOR_BGR2RGB)
            except Exception as e_conv: logger.error(f"BGR to RGB conversion failed for SAM: {e_conv}"); return None

            sam_mask = self._sam_segment_box(img_np_rgb, best_box['box'])
            if sam_mask is not None: upd(0.6, "SAM segmentation complete"); return sam_mask
            else: logger.warning("SAM segmentation using YOLO box failed."); return None
        else:
            logger.info(f"SAM+YOLO: Target object(s) '{yolo_labels_to_find}' not detected with sufficient confidence by YOLO.")
            return None # Fallback la altă regulă/strategie


    # NOU: Strategie specifică pentru obiecte generice
    def _strategy_generic_object(self, img_np_bgr: np.ndarray, prompt: str, operation: Dict[str, Any],
                                 image_context: Optional[Dict[str, Any]], rule_params: Dict[str, Any], upd: Callable) -> Optional[np.ndarray]:
        """Strategie pentru obiecte generice: încearcă SAM+YOLO, apoi CLIPSeg."""
        target_object = operation.get("target_object", "").lower()
        if not target_object: return None # Nu avem ce căuta

        logger.info(f"Generic Object Strategy for: '{target_object}'")

        # 1. Încearcă SAM + YOLO (folosind label-ul target)
        if image_context:
             upd(0.1, f"Trying SAM+YOLO for {target_object}")
             sam_yolo_params = {"yolo_labels": [target_object]} # Căutăm exact obiectul
             sam_mask = self._strategy_sam_yolo(img_np_bgr, prompt, operation, image_context, sam_yolo_params, lambda p,d: upd(0.1+p*0.4, d))
             if sam_mask is not None:
                  logger.info("Generic Object: SAM+YOLO succeeded.")
                  return sam_mask
             else:
                  logger.info("Generic Object: SAM+YOLO failed or object not detected by YOLO.")

        # 2. Fallback la CLIPSeg
        upd(0.6, f"Trying CLIPSeg for {target_object}")
        clipseg_params = {
            "main_clip_prompt_keyword_from_op_target": True,
            "main_threshold_key": "CLIPSEG_OBJECT_THRESHOLD",
             "_rule_name_": "Generic Object Fallback (CLIPSeg)" # Nume pentru debugging
        }
        clip_mask = self._strategy_semantic_clipseg(img_np_bgr, prompt, operation, clipseg_params, lambda p,d: upd(0.6+p*0.4, d))
        if clip_mask is not None:
             logger.info("Generic Object: CLIPSeg fallback succeeded.")
             return clip_mask
        else:
             logger.warning("Generic Object: CLIPSeg fallback also failed.")
             return None


    # Strategiile existente (CLIPSeg, Background) - adaptate să primească image_context (chiar dacă nu-l folosesc încă)
    def _strategy_semantic_clipseg(self, img_np_bgr: np.ndarray, prompt: str, operation: Dict[str, Any],
                                   image_context: Optional[Dict[str, Any]], rule_params: Dict[str, Any], upd: Callable) -> Optional[np.ndarray]:
        """Strategie CLIPSeg (rămâne similară, dar primește context)."""
        # (Codul intern al strategiei rămâne neschimbat, doar semnătura e adaptată)
        main_clip_text = self._get_clip_prompt_from_rule(prompt, operation, rule_params, "main_clip_prompt")
        if not main_clip_text: logger.warning(f"Rule '{rule_params.get('_rule_name_')}' failed: main_clip_text empty."); return None
        upd(0.1, f"CLIPSeg: '{main_clip_text}'")
        main_seg = self._clipseg_segment(img_np_bgr, main_clip_text)
        if main_seg is None: logger.warning(f"CLIPSeg failed for '{main_clip_text}'."); return None

        thresh_key = rule_params.get("main_threshold_key", "CLIPSEG_DEFAULT_THRESHOLD")
        thresh_factor = self.config.get(thresh_key, self.config.get("CLIPSEG_DEFAULT_THRESHOLD", 0.35))
        thresh_val = int(thresh_factor * 255)
        _, main_mask = cv2.threshold(main_seg, thresh_val, 255, cv2.THRESH_BINARY)
        logger.debug(f"Applied thresh {thresh_val} for '{main_clip_text}'.")

        final_mask = main_mask
        refine_clip_text = self._get_clip_prompt_from_rule(prompt, operation, rule_params, "refine_clip_prompt")
        if refine_clip_text:
            upd(0.3, f"CLIPSeg Refine: '{refine_clip_text}'")
            refine_seg = self._clipseg_segment(img_np_bgr, refine_clip_text)
            if refine_seg is not None:
                ref_thresh_key = rule_params.get("refine_threshold_key", "CLIPSEG_DEFAULT_THRESHOLD")
                ref_thresh_factor = self.config.get(ref_thresh_key, self.config.get("CLIPSEG_DEFAULT_THRESHOLD", 0.35))
                ref_thresh_val = int(ref_thresh_factor * 255)
                _, refine_mask = cv2.threshold(refine_seg, ref_thresh_val, 255, cv2.THRESH_BINARY)
                logger.debug(f"Applied thresh {ref_thresh_val} for '{refine_clip_text}'.")
                op = rule_params.get("combine_op", "and")
                if op == "and": final_mask = cv2.bitwise_and(main_mask, refine_mask)
                elif op == "or": final_mask = cv2.bitwise_or(main_mask, refine_mask)
                logger.debug(f"Combined masks using '{op}'.")
            else: logger.warning(f"CLIPSeg refine failed for '{refine_clip_text}'.")
        return final_mask

    def _strategy_background(self, img_np_bgr: np.ndarray, prompt: str, operation: Dict[str, Any],
                             image_context: Optional[Dict[str, Any]], rule_params: Dict[str, Any], upd: Callable) -> Optional[np.ndarray]:
        """Strategie Background (folosește Rembg/MediaPipe/Grabcut)."""
        # Încercăm întâi Rembg (mai robust de obicei)
        rembg_session = self.models.get_model("rembg")
        if rembg_session:
            upd(0.1, "Background: Rembg subject extraction")
            try:
                # Rembg preferă RGBA sau RGB. Îi dăm BGR, ar trebui să se descurce.
                # .remove returnează RGBA cu alpha=masca
                out_rgba = rembg_session.remove(img_np_bgr)
                if out_rgba.shape[2] == 4:
                     logger.info("Using Rembg for background subject mask.")
                     return out_rgba[:,:,3] # Returnăm canalul alpha ca mască a subiectului
            except Exception as e: logger.warning(f"Rembg failed for background: {e}")

        # Fallback la MediaPipe (dacă Rembg eșuează sau nu e disponibil)
        mp_segmenter = self.models.get_model("mediapipe")
        if mp_segmenter:
             upd(0.1, "Background: MediaPipe subject segmentation")
             try:
                 img_rgb_mp = cv2.cvtColor(img_np_bgr, cv2.COLOR_BGR2RGB)
                 res_mp = mp_segmenter.process(img_rgb_mp)
                 mm = getattr(res_mp, 'segmentation_mask', None)
                 if mm is not None:
                      logger.info("Using MediaPipe for background subject mask.")
                      # Convertim la 0-255
                      return (mm > 0.5).astype(np.uint8) * 255
             except Exception as e: logger.warning(f"MediaPipe failed for background: {e}")

        # Fallback final la GrabCut (mai puțin precis, dar general)
        logger.warning("Falling back to GrabCut for background subject mask.")
        upd(0.1, "Background: GrabCut subject fallback")
        raw_subj_mask = self._grabcut_subject(img_np_bgr)
        return raw_subj_mask # Returnează masca subiectului; inversarea se face în generate_mask


    # --- Metoda Principală de Generare ---
    def generate_mask(
        self,
        image: Union[Image.Image, np.ndarray],
        prompt: str = "",
        operation: Optional[Dict[str, Any]] = None,
        image_context: Optional[Dict[str, Any]] = None, # NOU: Acceptă context
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> Dict[str, Any]:
        """Generează masca optimă bazată pe prompt, operație și context."""
        operation = operation or {}
        start_time_mask = time.time()

        def upd(pct: float, desc: str): # Funcție internă de progres
            if progress_callback:
                try: progress_callback(pct, desc)
                except: pass # Ignorăm erorile din callback
            logger.debug(f"MaskGen: {pct*100:.0f}% - {desc}")

        # 1. Standardizare Input și Context
        upd(0.0, "Preparing inputs...")
        img_np_bgr = self._standardize_input_image(image)
        if img_np_bgr is None: return {"success": False, "message": "Invalid input image."}
        h, w = img_np_bgr.shape[:2]

        # Obținem contextul dacă nu este furnizat
        if image_context is None:
            logger.info("Image context not provided, analyzing image...")
            upd(0.02, "Analyzing image context...")
            try:
                 # Asigurăm PIL RGB pentru analizor
                 img_pil_rgb = self._convert_cv2_to_pil(img_np_bgr)
                 image_context = self.image_analyzer.analyze_image_context(img_pil_rgb)
                 if "error" in image_context: image_context = {} # Folosim context gol dacă analiza eșuează
            except Exception as e_ctx:
                 logger.error(f"Failed to analyze image context internally: {e_ctx}")
                 image_context = {} # Folosim context gol
        else:
             logger.debug("Using provided image context.")

        # 2. Aplicare Reguli
        raw_mask_generated = None; final_message = "No specific rule matched or strategy failed."; applied_rule_name = "N/A"
        rule_matched = False
        for rule in self.rules:
            # Pasăm și contextul la condiție
            if rule["condition"](prompt, operation, image_context):
                rule_matched = True
                applied_rule_name = rule['name']; logger.info(f"Applying rule: {applied_rule_name}")
                upd(0.05, f"Rule: {applied_rule_name}")
                strategy_params = rule.get("params", {}); strategy_params["_rule_name_"] = applied_rule_name
                # Pasăm și contextul la strategie
                raw_mask_candidate = rule["strategy"](img_np_bgr, prompt, operation, image_context, strategy_params, upd)

                if raw_mask_candidate is not None and raw_mask_candidate.size > 0 and raw_mask_candidate.shape[:2] == (h,w):
                    raw_mask_generated = raw_mask_candidate
                    msg_tpl = rule.get("message", "Mask generated by rule.")
                    final_message = msg_tpl(operation) if callable(msg_tpl) else msg_tpl

                    # Tratament special pentru background (inversare)
                    if applied_rule_name == "Background (Inverted Subject)":
                        upd(0.4, "Processing background mask...")
                        morphed_subj = self._dynamic_morphology(raw_mask_generated, img_np_bgr)
                        refined_subj = self._advanced_edge_refine(morphed_subj, img_np_bgr)
                        final_processed_mask = cv2.bitwise_not(refined_subj)
                        raw_mask_to_return = refined_subj # Returnăm masca subiectului ca raw
                        upd(1.0, "Background mask ready")
                        total_time = time.time() - start_time_mask
                        logger.info(f"Mask generation time: {total_time:.2f}s (Rule: {applied_rule_name})")
                        return {"mask": final_processed_mask, "raw_mask": raw_mask_to_return, "success": True, "message": final_message}
                    break # Ieșim din for după prima regulă potrivită (non-background)
                else:
                    logger.warning(f"Rule '{applied_rule_name}' failed to produce a valid mask.")
            # Continuăm la următoarea regulă dacă nu s-a potrivit condiția

        # 3. Fallback Hibrid (dacă nicio regulă specifică nu a funcționat)
        if not rule_matched or raw_mask_generated is None:
             logger.info("No specific rule matched or rule failed. Falling back to hybrid strategy.")
             applied_rule_name = "Hybrid Fallback"; upd(0.0, applied_rule_name)
             # Folosim target_object sau un text extras din prompt pentru CLIPSeg în hibrid
             clip_prompt_hybrid = operation.get("target_object", "").strip() or \
                                  next((w for w in prompt.lower().split() if w not in ["remove","add","change","replace","make","the","a","an","to"]), "area")
             raw_mask_generated = self._strategy_hybrid_fallback(img_np_bgr, clip_prompt_hybrid, image_context, upd)
             final_message = f"Hybrid mask for '{clip_prompt_hybrid}'."
             if raw_mask_generated is None:
                  msg = "All mask generation strategies failed."; logger.error(msg)
                  return {"mask": np.zeros((h, w), np.uint8), "raw_mask": None, "success": False, "message": msg}

        # 4. Post-Procesare Finală (Morfologie + Rafinare Margini)
        logger.info(f"Performing final post-processing (Rule: {applied_rule_name})")
        upd(0.65, "Final Morphology")
        morphed_mask = self._dynamic_morphology(raw_mask_generated, img_np_bgr)
        upd(0.85, "Final Edge Refinement")
        final_processed_mask = self._advanced_edge_refine(morphed_mask, img_np_bgr)

        upd(1.0, "Mask ready")
        total_time = time.time() - start_time_mask
        logger.info(f"Mask generation time: {total_time:.2f}s (Rule: {applied_rule_name})")
        return {"mask": final_processed_mask, "raw_mask": raw_mask_generated, "success": True, "message": final_message}


    # --- Metode Helper pentru Strategii Specifice ---

    def _grabcut_subject(self, img_bgr_np: np.ndarray, rect_inset_ratio: float = 0.05) -> Optional[np.ndarray]:
        """Rulează GrabCut pentru a extrage subiectul central."""
        # (Cod neschimbat, dar verificările de input sunt acum în generate_mask)
        h, w = img_bgr_np.shape[:2]; mask_gc = np.zeros((h, w), np.uint8)
        bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
        dx, dy = int(rect_inset_ratio * w), int(rect_inset_ratio * h)
        rect = (max(1, dx), max(1, dy), max(1, w - 2*dx), max(1, h - 2*dy))
        if rect[2] <= 0 or rect[3] <= 0: return None # Imagine prea mică
        try:
            cv2.grabCut(img_bgr_np, mask_gc, rect, bgd, fgd, 5, cv2.GC_INIT_WITH_RECT)
            return np.where((mask_gc == cv2.GC_FGD) | (mask_gc == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
        except Exception as e: logger.error(f"GrabCut error: {e}"); return None


    def _clipseg_segment(self, img_bgr_np: np.ndarray, text_prompt: str) -> Optional[np.ndarray]:
        """Rulează CLIPSeg. Returnează masca uint8 (0-255)."""
        # (Cod neschimbat, dar verificările modelului sunt acum în get_model)
        bundle = self.models.get_model("clipseg"); # Folosim get_model pentru lazy loading
        if not bundle or not bundle.get('processor') or not bundle.get('model'): logger.warning("CLIPSeg model/processor not loaded."); return None
        processor, model = bundle["processor"], bundle["model"]
        try:
            pil_image = self._convert_cv2_to_pil(img_bgr_np) # Helper pentru conversie sigură
            effective_text = text_prompt.strip() if text_prompt else "object"
            inputs = processor(text=[effective_text], images=[pil_image], return_tensors="pt", padding=True, truncation=True).to(model.device) # Padding/Truncation importante
            with torch.no_grad(): outputs = model(**inputs)
            logits = outputs.logits; # Dimensiuni [1, 1, H_out, W_out] sau [1, H_out, W_out]
            # Redimensionăm logits la dimensiunea imaginii originale *înainte* de sigmoid/thresholding
            target_h, target_w = img_bgr_np.shape[:2]
            logits_resized = F.interpolate(logits.unsqueeze(0), size=(target_h, target_w), mode='bilinear', align_corners=False).squeeze() # [H, W]
            probs = torch.sigmoid(logits_resized).cpu().numpy() # float32 0-1
            return (probs * 255).astype(np.uint8) # Returnăm heatmap-ul uint8
        except Exception as e: logger.error(f"CLIPSeg processing error ('{effective_text}'): {e}", exc_info=True); return None

    # NOU: Metodă pentru a rula SAM cu bounding box
    def _sam_segment_box(self, img_rgb_np: np.ndarray, bbox_xyxy: np.ndarray) -> Optional[np.ndarray]:
        """Rulează SAM predictor folosind un bounding box."""
        sam_predictor = self.models.get_model("sam_predictor")
        if not sam_predictor: logger.warning("SAM Predictor not loaded."); return None
        if img_rgb_np.ndim != 3 or img_rgb_np.shape[2] != 3: logger.error("SAM requires RGB NumPy image."); return None

        try:
             # 1. Setează imaginea în predictor (trebuie făcut o singură dată per imagine)
             # Ideal, ar trebui să verificăm dacă imaginea s-a schimbat de la ultimul apel.
             # Pentru simplitate, o setăm de fiecare dată.
             start_set = time.time()
             sam_predictor.set_image(img_rgb_np)
             logger.debug(f"SAM set_image took: {time.time() - start_set:.3f}s")

             # 2. Rulează predicția cu bounding box
             start_pred = time.time()
             masks, scores, _ = sam_predictor.predict(
                 point_coords=None, # Nu folosim puncte
                 point_labels=None,
                 box=bbox_xyxy[None, :], # Adăugăm dimensiune batch [1, 4]
                 multimask_output=False, # Cerem o singură mască (cea mai bună)
             )
             logger.debug(f"SAM predict (box) took: {time.time() - start_pred:.3f}s")

             # masks este [1, H, W] boolean, scores este [1]
             if masks.shape[0] > 0:
                  # Returnăm prima (și singura) mască ca uint8 (0 sau 255)
                  best_mask = (masks[0] * 255).astype(np.uint8)
                  logger.debug(f"SAM mask generated from box, score: {scores[0]:.3f}")
                  return best_mask
             else:
                  logger.warning("SAM predict with box returned no masks.")
                  return None

        except Exception as e:
            logger.error(f"SAM segmentation with box error: {e}", exc_info=True)
            return None

    # --- Strategie Hibridă (Adaptată să poată folosi SAM) ---
    def _strategy_hybrid_fallback(self, img_np_bgr: np.ndarray, clip_prompt_text: str,
                                   image_context: Optional[Dict[str, Any]], upd: Callable) -> Optional[np.ndarray]:
        """Strategie fallback care combină mai multe metode, inclusiv SAM dacă e posibil."""
        h, w = img_np_bgr.shape[:2]
        accum = np.zeros((h, w), np.float32)
        contrib_count = 0; weights_sum = 0.0

        def add_contribution(mask_np, weight):
             nonlocal accum, contrib_count, weights_sum
             if mask_np is not None and mask_np.shape[:2] == (h,w):
                  # Normalizăm masca la 0-1 float
                  if mask_np.dtype == np.uint8: mask_float = mask_np.astype(np.float32) / 255.0
                  elif np.max(mask_np) > 1.0: mask_float = np.clip(mask_np, 0, 255).astype(np.float32) / 255.0
                  else: mask_float = mask_np.astype(np.float32)
                  accum += mask_float * weight
                  contrib_count += 1
                  weights_sum += weight
                  return True
             return False

        # 1. Încercăm SAM + YOLO pentru obiectul din promptul CLIP (dacă avem context)
        if image_context:
            upd(0.1, f"Hybrid: Try SAM+YOLO for '{clip_prompt_text}'")
            yolo_labels = [clip_prompt_text] # Căutăm obiectul specific
            # Simulăm o operație și o regulă pentru a apela _strategy_sam_yolo
            sam_yolo_op = {"target_object": clip_prompt_text}
            sam_yolo_params = {"yolo_labels": yolo_labels}
            sam_mask = self._strategy_sam_yolo(img_np_bgr, "", sam_yolo_op, image_context, sam_yolo_params, lambda p,d: upd(0.1+p*0.2, d))
            if add_contribution(sam_mask, 2.0): logger.debug("Hybrid: SAM+YOLO contributed.") # Ponderare mare pentru SAM

        # 2. CLIPSeg (dacă SAM nu a contribuit sau ca sursă suplimentară)
        upd(0.35, f"Hybrid: CLIPSeg for '{clip_prompt_text}'")
        clip_seg_mask = self._clipseg_segment(img_np_bgr, clip_prompt_text)
        if add_contribution(clip_seg_mask, 1.0): logger.debug("Hybrid: CLIPSeg contributed.")

        # 3. YOLO (dacă e disponibil și găsește ceva relevant)
        if image_context and clip_prompt_text != "person": # Evităm rularea YOLO din nou dacă am rulat pt SAM
             yolo_model = self._get_object_detector()
             if yolo_model:
                  upd(0.6, "Hybrid: YOLO detection/segmentation")
                  yolo_mask = self._get_yolo_mask_for_prompt(yolo_model, img_np_bgr, clip_prompt_text)
                  if add_contribution(yolo_mask, 0.8): logger.debug("Hybrid: YOLO contributed.")

        # 4. GrabCut (ca bază generală)
        upd(0.8, "Hybrid: GrabCut subject")
        grabcut_mask = self._grabcut_subject(img_np_bgr)
        if add_contribution(grabcut_mask, 0.5): logger.debug("Hybrid: GrabCut contributed.")

        if contrib_count == 0: logger.warning("Hybrid fallback: No models contributed."); return None

        # Normalizare și Thresholding
        # Normalizăm prin suma ponderilor pentru a nu depăși 1.0
        final_float_mask = np.clip(accum / weights_sum if weights_sum > 0 else accum, 0.0, 1.0)
        thresh = self.config.get("HYBRID_MASK_THRESHOLD", 0.4) if self.config else 0.4 # Prag mai mare pentru hibrid
        final_mask = (final_float_mask > thresh).astype(np.uint8) * 255
        logger.debug(f"Hybrid mask generated using {contrib_count} sources.")
        return final_mask

    def _get_yolo_mask_for_prompt(self, yolo_model, img_np_bgr, prompt_text) -> Optional[np.ndarray]:
         """Rulează YOLO și returnează masca combinată pentru obiectele care se potrivesc cu promptul."""
         try:
              h, w = img_np_bgr.shape[:2]
              results = yolo_model.predict(source=img_np_bgr, stream=False, conf=0.3, verbose=False) # Conf mai mic aici
              combined_mask = np.zeros((h, w), dtype=np.uint8)
              count = 0
              if results and isinstance(results, list):
                   res = results[0]
                   if hasattr(res, 'boxes') and res.boxes is not None and hasattr(res, 'masks') and res.masks is not None:
                        boxes_data = res.boxes.data.cpu().numpy() # xyxy, conf, cls
                        masks_data = res.masks.data.cpu().numpy() # [N, H, W]
                        class_names = getattr(res, 'names', {})
                        for i, box in enumerate(boxes_data):
                             cls_id = int(box[5]); label = class_names.get(cls_id, '')
                             # Verificare simplă dacă label-ul apare în prompt
                             if label and label in prompt_text:
                                  if i < len(masks_data):
                                       mask_i = cv2.resize(masks_data[i], (w, h), interpolation=cv2.INTER_NEAREST)
                                       combined_mask = np.maximum(combined_mask, (mask_i > 0.5).astype(np.uint8) * 255)
                                       count += 1
              logger.debug(f"YOLO found {count} mask(s) matching '{prompt_text}' for hybrid strategy.")
              return combined_mask if count > 0 else None
         except Exception as e: logger.error(f"YOLO mask extraction for prompt failed: {e}"); return None


    # --- Metode de Post-Procesare Mască (neschimbate) ---
    def _morphology(self, mask: np.ndarray, close_k:int, open_k:int, close_iter:int, open_iter:int) -> np.ndarray:
        # (Cod neschimbat)
        if mask is None or mask.size == 0 or close_k<=0 or open_k<=0: return mask
        close_k+=1 if close_k%2==0 else 0; open_k+=1 if open_k%2==0 else 0
        try:
            k_close=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(close_k,close_k))
            k_open=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(open_k,open_k))
            m = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close, iterations=close_iter)
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k_open, iterations=open_iter)
            return m
        except Exception as e: logger.error(f"Morphology error: {e}"); return mask

# În fișierul /workspace/FusionFrame/processing/mask_generator.py
    # În clasa MaskGenerator

    def _dynamic_morphology(self, mask: np.ndarray, img: np.ndarray) -> np.ndarray:
        if mask is None:
            logger.debug("Dynamic morphology: Mask is None, returning original mask.")
            return mask # Returnează masca originală dacă este None

        if img is None:
            logger.warning("Dynamic morphology: Image is None, cannot perform calculations. Returning original mask.")
            return mask # Returnează masca originală dacă imaginea este None

        # Asigură că 'img' este un array NumPy valid cu cel puțin 2 dimensiuni
        if not isinstance(img, np.ndarray) or img.ndim < 2:
            logger.error(f"Dynamic morphology: Invalid image type or dimensions. Type: {type(img)}, ndim: {getattr(img, 'ndim', 'N/A')}. Returning original mask.")
            return mask

        try:
            h, w = img.shape[:2]
            min_d = min(h, w)
        except Exception as e:
            logger.error(f"Dynamic morphology: Error getting image dimensions: {e}. Returning original mask.")
            return mask

        # Restul logicii originale a funcției
        close_k = max(3, int((0.015 if min_d > 1000 else 0.01) * min_d) // 2 * 2 + 1)
        open_k = max(3, int((0.007 if min_d > 1000 else 0.005) * min_d) // 2 * 2 + 1)
        close_i = 3 if min_d > 700 else 2
        open_i = 2 if min_d > 700 else 1
        
        logger.debug(f"Dynamic morphology: close_k={close_k}, close_i={close_i}, open_k={open_k}, open_i={open_i}, min_d={min_d}")
        return self._morphology(mask, close_k, open_k, close_i, open_i)

    def _advanced_edge_refine(self, mask: np.ndarray, img_bgr: np.ndarray) -> np.ndarray:
        # (Cod neschimbat)
        if mask is None or img_bgr is None: return mask
        try:
            if not hasattr(cv2, 'ximgproc') or not hasattr(cv2.ximgproc, 'guidedFilter'): return self._edge_refine(mask, img_bgr)
        except AttributeError: return self._edge_refine(mask, img_bgr)
        try:
            mask_f=mask.astype(np.float32)/255.0; gray=cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            edges_c=cv2.Canny(gray,30,100).astype(np.float32)/255.0
            gx=cv2.Sobel(gray,cv2.CV_32F,1,0,ksize=3); gy=cv2.Sobel(gray,cv2.CV_32F,0,1,ksize=3)
            edges_s_mag=cv2.magnitude(gx,gy); max_s=np.max(edges_s_mag); edges_s=edges_s_mag/max_s if max_s>0 else np.zeros_like(edges_s_mag)
            edges=np.maximum(edges_c,edges_s); influence=0.3
            blend_f=mask_f*(1.0-edges*influence)+edges*(mask_f*0.2+0.1); blend_f=np.clip(blend_f,0.0,1.0)
            radius=max(3,int(0.005*min(img_bgr.shape[:2]))); eps=0.005
            blend_u8= (blend_f*255).astype(np.uint8)
            filtered = cv2.ximgproc.guidedFilter(guide=img_bgr, src=blend_u8, radius=radius, eps=eps*eps*255*255)
            if filtered.dtype!=np.uint8: filtered=np.clip(filtered,0,255).astype(np.uint8)
            _,thresh=cv2.threshold(filtered,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            k=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
            clean=cv2.morphologyEx(thresh,cv2.MORPH_OPEN,k,iterations=1); clean=cv2.morphologyEx(clean,cv2.MORPH_CLOSE,k,iterations=1)
            return clean
        except Exception as e: logger.error(f"Advanced edge refine error: {e}. Fallback.",exc_info=True); return self._edge_refine(mask, img_bgr)

    def _edge_refine(self, mask: np.ndarray, img: np.ndarray) -> np.ndarray: # Fallback
        # (Cod neschimbat)
        if mask is None or img is None: return mask
        try:
            k_size=max(3,int(0.005*min(img.shape[:2]))//2*2+1); k=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(k_size,k_size))
            m=cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1); m=cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=1); return m
        except Exception as e: logger.error(f"Basic edge refine error: {e}"); return mask

        # Adaugă această metodă în clasa MaskGenerator (în mask_generator.py)

    def _get_object_detector(self):
        """
        Returnează modelul de detecție a obiectelor (ex: YOLO).
        """
        yolo_bundle = self.models.get_model("yolo") # Sau alt nume relevant pentru modelul YOLO

        if not yolo_bundle:
            logger.warning("Object detection model (YOLO) bundle not loaded.")
            return None

        if not isinstance(yolo_bundle, dict) or 'model' not in yolo_bundle:
            logger.error("YOLO bundle is not a dictionary or does not contain a 'model' key.")
            return None

        yolo_model_obj = yolo_bundle.get('model')

        if yolo_model_obj is None:
            logger.error("The 'model' key in YOLO bundle is None.")
            return None

        if not hasattr(yolo_model_obj, 'predict'):
            logger.error("YOLO model object in bundle does not have a 'predict' method.")
            return None
            
        return yolo_model_obj

    # --- Helpers de Conversie ---
    def _standardize_input_image(self, image: Union[Image.Image, np.ndarray]) -> Optional[np.ndarray]:
         """Convertește inputul la BGR NumPy uint8."""
         try: # Copiem helper-ul din BasePipeline pentru independență
            if isinstance(image, np.ndarray):
                if image.ndim == 2: return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                elif image.shape[2] == 4: return cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
                elif image.shape[2] == 3: return image.astype(np.uint8) if image.dtype!=np.uint8 else image
                else: raise ValueError(f"Unsupported NumPy shape: {image.shape}")
            elif isinstance(image, Image.Image):
                img_rgb = image.convert("RGB"); return cv2.cvtColor(np.array(img_rgb), cv2.COLOR_RGB2BGR)
            else: raise TypeError(f"Unsupported type: {type(image)}")
         except Exception as e: logger.error(f"Image standardization error: {e}"); return None

    def _convert_cv2_to_pil(self, img_np):
        """Convertește BGR NumPy în PIL RGB."""
        if img_np is None: return None
        try: return Image.fromarray(cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB))
        except Exception as e: logger.error(f"CV2 to PIL conversion failed: {e}"); return None