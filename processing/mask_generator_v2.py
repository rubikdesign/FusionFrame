#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
MaskGenerator v3.0 — Inteligent, operation-aware mask generation
for FusionFrame. Prioritizes semantic understanding of the prompt.
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
    logger.warning("AppConfig not found, using AppConfigMock. Ensure AppConfig is in PYTHONPATH for actual use.")
    class AppConfigMock:
        def __init__(self):
            self._config = {
                "CLIPSEG_DEFAULT_THRESHOLD": 0.35, # Un prag general bun
                "CLIPSEG_BACKGROUND_THRESHOLD": 0.4,
                "CLIPSEG_HAIR_THRESHOLD": 0.45, # Puțin mai mare pentru precizie
                "CLIPSEG_HEAD_THRESHOLD": 0.3,
                "CLIPSEG_FACE_THRESHOLD": 0.35,
                "CLIPSEG_EYES_THRESHOLD": 0.4,
                "CLIPSEG_PERSON_THRESHOLD": 0.5, # Persoanele sunt de obicei mai clare
                "CLIPSEG_OBJECT_THRESHOLD": 0.4, # Prag pentru obiecte generice
                "HYBRID_MASK_THRESHOLD": 0.35,
                # Adăugați alte praguri specifice după nevoie
            }
        def get(self, key: str, default: Any = None) -> Any:
            return self._config.get(key, default)
    AppConfig = AppConfigMock # Folosim mock-ul dacă importul real eșuează


from core.model_manager import ModelManager # Asigurați-vă că acest import este valid

logger = logging.getLogger(__name__)

# Tipuri pentru reguli
RuleCondition = Callable[[str, Dict[str, Any]], bool]
MaskStrategy = Callable[[np.ndarray, str, Dict[str, Any], Callable], Optional[np.ndarray]]


class MaskGenerator:
    def __init__(self):
        self.models = ModelManager()
        self.config = AppConfig() # Folosește AppConfig real sau mock-ul definit mai sus
        self.rules: List[Dict[str, Any]] = self._define_rules()

    def _define_rules(self) -> List[Dict[str, Any]]:
        """Definește regulile de prioritate pentru generarea măștilor."""
        
        rules = [
            {
                "name": "Background Change/Remove",
                "condition": lambda prompt, op: (op.get("type") in ("replace", "remove") and \
                                                ("background" in prompt.lower() or op.get("target_object", "").lower() == "background")) or \
                                                "background" in op.get("target_object", "").lower(), # Adăugat pentru cazul când target_object e "background"
                "strategy": self._strategy_background,
                "message": "Background mask (inverted subject)"
            },
            {
                "name": "Hair Color/Style",
                "condition": lambda prompt, op: "hair" in prompt.lower() or op.get("target_object", "").lower() == "hair",
                "strategy": self._strategy_semantic_clipseg,
                "params": {
                    "main_clip_prompt_keyword": "hair", # Cuvântul cheie principal pentru CLIPSeg
                    "refine_clip_prompt_keyword": "head",
                    "main_threshold_key": "CLIPSEG_HAIR_THRESHOLD",
                    "refine_threshold_key": "CLIPSEG_HEAD_THRESHOLD",
                    "combine_op": "and" # 'hair' AND 'head'
                },
                "message": "Hair mask"
            },
            {
                "name": "Eyes Color/Details",
                "condition": lambda prompt, op: "eye" in prompt.lower() or op.get("target_object", "").lower() in ("eye", "eyes"), # "eye" sau "eyes"
                "strategy": self._strategy_semantic_clipseg,
                "params": {
                    "main_clip_prompt_keyword": "eyes", # Folosim "eyes" pentru consistență
                    "refine_clip_prompt_keyword": "face",
                    "main_threshold_key": "CLIPSEG_EYES_THRESHOLD",
                    "refine_threshold_key": "CLIPSEG_FACE_THRESHOLD",
                    "combine_op": "and"
                },
                "message": "Eyes mask"
            },
            {
                "name": "Add/Replace Glasses", # Acoperă "add glasses", "replace glasses with sunglasses"
                "condition": lambda prompt, op: "glasses" in prompt.lower() or op.get("target_object", "").lower() in ("glasses", "sunglasses"),
                "strategy": self._strategy_semantic_clipseg,
                "params": { # Pentru "add glasses", masca este pe regiunea feței/ochilor
                    "main_clip_prompt_keyword": "face", 
                    "refine_clip_prompt_keyword": "eyes", # Opțional, pentru a centra mai bine
                    "main_threshold_key": "CLIPSEG_FACE_THRESHOLD",
                    "refine_threshold_key": "CLIPSEG_EYES_THRESHOLD",
                    "combine_op": "and" # Masca rezultată va fi intersecția, focusând pe zona ochilor
                },
                "message": "Region for glasses mask"
            },
            {
                "name": "Remove Person",
                "condition": lambda prompt, op: (op.get("type") == "remove" and \
                                                any(kw in prompt.lower() for kw in ["person", "man", "woman", "child", "people"])) or \
                                                op.get("target_object", "").lower() in ("person", "man", "woman", "child", "people"),
                "strategy": self._strategy_semantic_clipseg, # Sau o strategie hibridă specifică pentru persoane
                "params": {
                    "main_clip_prompt_keyword_from_prompt": True, # Extrage "person", "man" etc. din prompt
                    "main_threshold_key": "CLIPSEG_PERSON_THRESHOLD",
                },
                "message": "Person mask for removal"
            },
            # Reguli generice pentru obiecte comune (pot fi extinse)
            # Acestea ar trebui să vină după cele mai specifice
            {
                "name": "Generic Object Removal/Modification (Cat, Dog, Car, Tree, etc.)",
                "condition": lambda prompt, op: op.get("type") in ("remove", "color", "replace") and \
                                             op.get("target_object", "") not in ["", "background", "hair", "eyes", "glasses", "person"], # Evităm suprapunerea cu reguli mai specifice
                "strategy": self._strategy_semantic_clipseg,
                "params": {
                    "main_clip_prompt_keyword_from_op_target": True, # Folosește operation['target_object']
                    "main_threshold_key": "CLIPSEG_OBJECT_THRESHOLD",
                },
                "message": lambda op: f"{op.get('target_object', 'Object').capitalize()} mask" # Mesaj dinamic
            },
        ]
        return rules

    def _extract_keyword_from_prompt(self, prompt: str, keywords: List[str]) -> Optional[str]:
        """Helper pentru a extrage primul cuvânt cheie găsit din prompt."""
        lprompt = prompt.lower()
        for kw in keywords:
            if kw in lprompt:
                return kw
        return None

    def _get_clip_prompt_from_rule(self, prompt: str, operation: Dict[str, Any], rule_params: Dict[str, Any], param_name: str) -> Optional[str]:
        """Determină textul pentru CLIPSeg pe baza parametrilor regulii."""
        if rule_params.get(f"{param_name}_keyword_from_prompt"):
            # Extrage din prompt pe baza listei de cuvinte cheie ale regulii (dacă e definită așa)
            # Sau un cuvânt cheie specific dacă e dat în rule_params
            specific_keywords = rule_params.get("keywords_for_extraction") # ex: ["person", "man", "woman"]
            if specific_keywords:
                return self._extract_keyword_from_prompt(prompt, specific_keywords)
            else: # Folosește target_object dacă nu sunt specificate cuvinte cheie pentru extracție
                 return operation.get("target_object", rule_params.get(f"{param_name}_keyword", "object"))
        elif rule_params.get(f"{param_name}_keyword_from_op_target"):
            return operation.get("target_object", rule_params.get(f"{param_name}_keyword", "object"))
        else:
            return rule_params.get(f"{param_name}_keyword")


    def _strategy_semantic_clipseg(self, img_np: np.ndarray, prompt: str, operation: Dict[str, Any], rule_params: Dict[str, Any], upd: Callable) -> Optional[np.ndarray]:
        """Strategie bazată pe CLIPSeg, cu rafinare opțională."""
        
        main_clip_text = self._get_clip_prompt_from_rule(prompt, operation, rule_params, "main_clip_prompt")
        if not main_clip_text:
            logger.warning(f"Rule '{rule_params.get('name', 'Unknown')}' failed: main_clip_text could not be determined.")
            return None
            
        upd(0.1, f"CLIPSeg: '{main_clip_text}'")
        main_seg = self._clipseg_segment(img_np, main_clip_text)
        if main_seg is None:
            logger.warning(f"CLIPSeg failed for main prompt: '{main_clip_text}'")
            return None

        main_thresh_val = self.config.get(rule_params["main_threshold_key"], self.config.get("CLIPSEG_DEFAULT_THRESHOLD", 0.35)) * 255
        _, main_mask = cv2.threshold(main_seg, int(main_thresh_val), 255, cv2.THRESH_BINARY)

        final_combined_mask = main_mask

        refine_clip_text = self._get_clip_prompt_from_rule(prompt, operation, rule_params, "refine_clip_prompt")

        if refine_clip_text:
            upd(0.3, f"CLIPSeg Refine: '{refine_clip_text}'")
            refine_seg = self._clipseg_segment(img_np, refine_clip_text)
            if refine_seg is not None:
                refine_thresh_val = self.config.get(rule_params.get("refine_threshold_key"), self.config.get("CLIPSEG_DEFAULT_THRESHOLD", 0.35)) * 255
                _, refine_mask = cv2.threshold(refine_seg, int(refine_thresh_val), 255, cv2.THRESH_BINARY)
                
                combine_op = rule_params.get("combine_op", "and")
                if combine_op == "and":
                    final_combined_mask = cv2.bitwise_and(main_mask, refine_mask)
                elif combine_op == "or":
                    final_combined_mask = cv2.bitwise_or(main_mask, refine_mask)
                # Adăugați alte operații de combinare dacă e necesar (ex: subtract)
            else:
                logger.warning(f"CLIPSeg failed for refinement prompt: '{refine_clip_text}'")
        
        return final_combined_mask


    def _strategy_background(self, img_np: np.ndarray, prompt: str, operation: Dict[str, Any], upd: Callable) -> Optional[np.ndarray]:
        """Strategie specifică pentru fundal (inversul subiectului GrabCut)."""
        upd(0.1, "GrabCut: Subject for background")
        raw_subj_mask = self._grabcut_subject(img_np)
        if raw_subj_mask is None:
            return None
        # Masca pentru fundal este inversul măștii subiectului
        # Etapele de morfologie și rafinare se aplică pe masca subiectului înainte de inversare
        return raw_subj_mask # Vom inversa după morfologie și rafinare în `generate_mask`


    def generate_mask(
        self,
        image: Union[Image.Image, np.ndarray],
        prompt: str = "",
        operation: Optional[Dict[str, Any]] = None, # OperationAnalyzer ar trebui să populeze asta
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> Dict[str, Any]:
        
        operation = operation or {} # Asigurăm că operation este un dicționar

        def upd(pct: float, desc: str):
            if progress_callback:
                progress_callback(pct, desc)
            logger.debug(f"MaskGen Progress: {pct*100:.0f}% - {desc}")

        img_np_orig = np.asarray(image) if isinstance(image, Image.Image) else image.copy()
        
        # Conversie la BGR uint8, formatul principal pentru OpenCV
        if img_np_orig.ndim == 2:
            img_np = cv2.cvtColor(img_np_orig, cv2.COLOR_GRAY2BGR)
        elif img_np_orig.shape[2] == 4: # RGBA
            img_np = cv2.cvtColor(img_np_orig, cv2.COLOR_RGBA2BGR)
        elif img_np_orig.shape[2] == 3: # RGB
            img_np = cv2.cvtColor(img_np_orig, cv2.COLOR_RGB2BGR)
        else:
            img_np = img_np_orig.copy() # Presupunem că e deja BGR

        if img_np.dtype != np.uint8:
            logger.warning(f"Input image numpy array is not uint8 (type: {img_np.dtype}). Converting.")
            if np.max(img_np) <= 1.0 and img_np.dtype == np.float32 or img_np.dtype == np.float64:
                 img_np = (img_np * 255).astype(np.uint8)
            else:
                 img_np = img_np.astype(np.uint8)

        h, w = img_np.shape[:2]
        lprompt = prompt.lower()
        
        # Analiza promptului (dacă `operation` nu e complet) - simplificat
        # Ideal, `OperationAnalyzer` face asta și populează `operation`
        if not operation.get("target_object") and not operation.get("type"):
            if "remove" in lprompt: operation["type"] = "remove"
            elif "change color" in lprompt: operation["type"] = "color"
            elif "replace" in lprompt: operation["type"] = "replace"
            elif "add" in lprompt: operation["type"] = "add"
            # Extracție simplă de target_object (poate fi îmbunătățită)
            # Aceasta este o versiune foarte simplistă.
            potential_targets = ["hair", "eyes", "face", "person", "cat", "dog", "car", "tree", "background", "sky", "glasses", "shirt"]
            for pt in potential_targets:
                if pt in lprompt:
                    operation["target_object"] = pt
                    break
            if not operation.get("target_object"):
                 # Dacă promptul conține "remove" și un substantiv după, îl considerăm țintă
                 if operation.get("type") == "remove" and len(lprompt.split("remove ")) > 1:
                     possible_target = lprompt.split("remove ")[1].split(" ")[0] # Primul cuvânt după "remove"
                     operation["target_object"] = possible_target

        logger.info(f"Generating mask for prompt: '{prompt}', operation: {operation}")

        raw_mask = None
        final_message = "Mask generation failed or no specific rule matched."

        # Aplicăm regulile definite
        for rule in self.rules:
            if rule["condition"](prompt, operation):
                logger.info(f"Applying rule: {rule['name']}")
                upd(0.05, f"Rule: {rule['name']}")
                
                # Parametrii specifici pentru strategie (dacă există)
                strategy_params = rule.get("params", {})
                strategy_params["name"] = rule["name"] # Adăugăm numele regulii la parametri pentru logging

                raw_mask_candidate = rule["strategy"](img_np, prompt, operation, strategy_params, upd)
                
                if raw_mask_candidate is not None:
                    raw_mask = raw_mask_candidate
                    final_message = rule.get("message", "Mask generated by rule.")
                    if callable(final_message): # Pentru mesaje dinamice
                        final_message = final_message(operation)
                    
                    # Tratament special pentru fundal: inversăm după morfologie și rafinare
                    if rule["name"] == "Background Change/Remove":
                        upd(0.4, "Morphology on subject for background")
                        morphed_subj_mask = self._dynamic_morphology(raw_mask, img_np)
                        upd(0.7, "Edge refinement on subject for background")
                        refined_subj_mask = self._advanced_edge_refine(morphed_subj_mask, img_np)
                        final_mask = cv2.bitwise_not(refined_subj_mask)
                        raw_mask_for_output = refined_subj_mask # Raw mask este masca subiectului
                        upd(1.0, "Background mask ready")
                        return {"mask": final_mask, "raw_mask": raw_mask_for_output, "success": True, "message": final_message}
                    break # Am găsit și aplicat o regulă
        
        # Dacă nicio regulă specifică nu a generat o mască, folosim fallback-ul hibrid
        if raw_mask is None:
            logger.info("No specific rule matched or rule failed. Falling back to hybrid/default strategy.")
            upd(0.0, "Hybrid Fallback: Initializing")
            
            # Extragem un target mai specific pentru CLIPSeg din prompt
            # (poate fi o funcție mai complexă de NLP)
            extracted_target = operation.get("target_object", "subject") # Default la "subject"
            if extracted_target == "subject" and prompt: # Încercăm să fim mai specifici
                # O logică simplă: căutăm substantive comune după verbe de acțiune
                action_verbs = ["remove", "change", "replace", "add", "make", "modify", "turn", "convert"]
                words = prompt.lower().split()
                for i, word in enumerate(words):
                    if word in action_verbs and i + 1 < len(words):
                        # Verificăm dacă următorul cuvânt este un articol sau prepoziție comună
                        if words[i+1] not in ["the", "a", "an", "to", "into", "with", "on", "from"]:
                            extracted_target = words[i+1]
                            # Putem încerca să luăm și următorul cuvânt dacă formează o expresie (ex: "hair color")
                            if i + 2 < len(words) and words[i+2] not in action_verbs + ["to", "into", "with", "on", "from"]:
                                extracted_target += " " + words[i+2]
                            break
                if extracted_target == "subject": # Dacă tot nu am găsit, folosim primul substantiv (foarte simplist)
                    # Această parte necesită o bibliotecă NLP pentru o extracție robustă de substantive
                    # Pentru moment, lăsăm "subject" sau ce a extras OperationAnalyzer
                    pass


            clip_prompt_hybrid = extracted_target if extracted_target and extracted_target != "subject" else prompt # Folosim promptul întreg dacă targetul e vag
            if not clip_prompt_hybrid.strip() : clip_prompt_hybrid = "object of interest" # Fallback final

            raw_mask = self._strategy_hybrid_fallback(img_np, clip_prompt_hybrid, upd)
            final_message = "Hybrid fallback mask generated."
            if raw_mask is None: # Dacă și fallback-ul eșuează
                logger.error("Hybrid fallback mask generation also failed.")
                return {"mask": np.zeros((h, w), np.uint8), "raw_mask": None, "success": False, "message": "All mask generation strategies failed."}

        # Post-procesare comună pentru măștile generate (exceptând cea de fundal care are propria logică)
        upd(0.6, "Morphology on generated mask")
        morphed_mask = self._dynamic_morphology(raw_mask, img_np)
        upd(0.85, "Edge refinement on generated mask")
        final_mask = self._advanced_edge_refine(morphed_mask, img_np)
        
        upd(1.0, "Mask ready")
        return {"mask": final_mask, "raw_mask": raw_mask, "success": True, "message": final_message}

    def _strategy_hybrid_fallback(self, img_np: np.ndarray, clip_prompt_text: str, upd: Callable) -> Optional[np.ndarray]:
        """Strategia hibridă de fallback, combinând mai multe surse."""
        h, w = img_np.shape[:2]
        accum = np.zeros((h, w), np.float32)
        count = 0

        # 1. GrabCut Subject (ca o componentă a hibridului)
        upd(0.1, "Hybrid: GrabCut subject")
        grabcut_subj = self._grabcut_subject(img_np)
        if grabcut_subj is not None:
            accum += grabcut_subj.astype(np.float32) / 255.0
            count += 1
        
        # 2. YOLO segmentation (dacă disponibil și relevant)
        yolo = self.models.get_model("yolo")
        if yolo:
            upd(0.25, "Hybrid: YOLO segmentation")
            try:
                preds = yolo.predict(source=img_np, stream=False, imgsz=640, conf=0.25, verbose=False)
                yolo_masks_combined = np.zeros((h, w), np.float32)
                yolo_mask_count = 0
                for r in preds:
                    if getattr(r, "masks", None) and hasattr(r.masks, "data") and r.masks.data.numel() > 0:
                        masks_data = r.masks.data.cpu().numpy()
                        for m_yolo in masks_data:
                            if m_yolo.size == 0: continue
                            # Aici ar trebui să filtrăm măștile YOLO pe baza promptului, dacă e posibil
                            # Momentan, le combinăm pe toate cele detectate cu încredere
                            m_resized = cv2.resize(m_yolo.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
                            yolo_masks_combined += m_resized
                            yolo_mask_count +=1
                if yolo_mask_count > 0:
                    accum += (yolo_masks_combined / yolo_mask_count) # Adăugăm media măștilor YOLO
                    count += 1 
            except Exception as e:
                logger.error(f"Hybrid YOLO error: {e}", exc_info=True)

        # 3. MediaPipe Selfie Segmentation (dacă disponibil și relevant - ex: dacă promptul implică o persoană)
        # Ar trebui să verificăm relevanța înainte de a-l rula
        if "person" in clip_prompt_text.lower() or "face" in clip_prompt_text.lower() or "selfie" in clip_prompt_text.lower():
            mp_segmenter = self.models.get_model("mediapipe") # Cheia originală
            if mp_segmenter:
                upd(0.5, "Hybrid: MediaPipe segmentation")
                try:
                    res_mp = mp_segmenter.process(cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB))
                    mm = getattr(res_mp, 'segmentation_mask', None)
                    if mm is not None and mm.size > 0:
                        mmf = cv2.resize(mm.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
                        accum += mmf 
                        count += 1
                except Exception as e:
                    logger.error(f"Hybrid MediaPipe error: {e}", exc_info=True)
        
        # 4. CLIPSeg cu promptul extras/generic
        upd(0.75, f"Hybrid: CLIPSeg for '{clip_prompt_text}'")
        clip_gen_seg = self._clipseg_segment(img_np, clip_prompt_text)
        if clip_gen_seg is not None:
            accum += clip_gen_seg.astype(np.float32) / 255.0
            count += 1
        
        if count == 0:
            logger.warning("Hybrid fallback: No models contributed. Returning a blank mask.")
            return np.zeros((h,w), dtype=np.uint8) # Sau None pentru a indica eșec total

        combined_float_mask = accum / count
        hybrid_threshold_val = self.config.get("HYBRID_MASK_THRESHOLD", 0.35)
        final_hybrid_mask = (combined_float_mask > hybrid_threshold_val).astype(np.uint8) * 255
        return final_hybrid_mask

    # --- Metodele helper (_grabcut_subject, _clipseg_segment, _morphology, etc.) rămân în mare parte la fel ---
    # Asigurați-vă că _clipseg_segment și celelalte sunt robuste, așa cum am discutat anterior.
    # Voi include versiunile corectate anterior pentru acestea.

    def _grabcut_subject(self, img: np.ndarray, rect_inset_ratio: float = 0.05) -> Optional[np.ndarray]:
        h, w = img.shape[:2]
        if h == 0 or w == 0:
            logger.error("GrabCut received an empty image.")
            return None
            
        mask_gc = np.zeros((h, w), np.uint8)
        # Verificăm dacă imaginea are cel puțin 3 canale pentru GrabCut
        if img.ndim < 3 or img.shape[2] < 3:
            logger.warning(f"GrabCut expects a 3-channel image, got {img.shape}. Attempting to convert to BGR.")
            if img.ndim == 2 : img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            # Alte conversii ar putea fi necesare dacă imaginea are mai puțin de 3 canale într-un mod neașteptat
            if img.ndim < 3 or img.shape[2] < 3: # Verificare din nou
                logger.error("Could not convert image to 3 channels for GrabCut.")
                return None


        bgd_model, fgd_model = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
        
        dx, dy = int(rect_inset_ratio * w), int(rect_inset_ratio * h) 
        rect_w, rect_h = w - 2*dx, h - 2*dy

        if rect_w <=0 or rect_h <=0: 
            logger.warning(f"GrabCut rectangle is too small or invalid ({dx},{dy},{rect_w},{rect_h}). Using full image (1px inset).")
            rect = (1, 1, max(1, w-2), max(1, h-2)) # Asigurăm rect pozitiv și minim 1x1
            if rect[2] <= 0 or rect[3] <= 0: 
                 logger.error("Image too small for GrabCut even with minimal inset.")
                 return None
        else:
            rect = (dx, dy, rect_w, rect_h)

        try:
            cv2.grabCut(img, mask_gc, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
            subject_mask = np.where((mask_gc == cv2.GC_FGD) | (mask_gc == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
            return subject_mask
        except cv2.error as e_cv:
            logger.error(f"GrabCut OpenCV error: {e_cv}", exc_info=True)
            fb_mask = np.zeros((h, w), np.uint8)
            try: # Încercăm să desenăm fallback-ul chiar dacă rect-ul e problematic
                cv2.rectangle(fb_mask, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), 255, -1)
            except Exception: # Dacă și desenarea eșuează, returnăm masca goală
                logger.error("Failed to draw fallback rectangle for GrabCut.")
            return fb_mask
        except Exception as e:
            logger.error(f"GrabCut generic error: {e}", exc_info=True)
            return None

    def _clipseg_segment(self, img_bgr_np: np.ndarray, text_prompt: str) -> Optional[np.ndarray]:
        bundle = self.models.get_model("clipseg")
        if not bundle or "processor" not in bundle or "model" not in bundle:
            logger.warning("CLIPSeg model or processor not found/incomplete in ModelManager.")
            return None

        processor, model = bundle["processor"], bundle["model"]
        
        try:
            # CLIPSeg se așteaptă la o imagine RGB PIL
            img_rgb_np = cv2.cvtColor(img_bgr_np, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(img_rgb_np)
        except Exception as e_pil:
            logger.error(f"Failed to convert BGR numpy to RGB PIL Image for CLIPSeg: {e_pil}", exc_info=True)
            return None
            
        effective_text = text_prompt if text_prompt and text_prompt.strip() else "object"
        logger.debug(f"CLIPSeg processing with text: '{effective_text}'")

        try:
            inputs = processor(text=[effective_text], images=[pil_image], return_tensors="pt", padding=True)
        except Exception as e_proc:
            logger.error(f"CLIPSeg processor error for text '{effective_text}': {e_proc}", exc_info=True)
            return None
            
        processed_inputs = {}
        try:
            target_device = model.device # Folosim dispozitivul pe care este deja modelul
            target_dtype = model.dtype if hasattr(model, 'dtype') else torch.float32 # Folosim dtype-ul modelului dacă există

            for k, v_tensor in inputs.items():
                if v_tensor.dtype.is_floating_point:
                    processed_inputs[k] = v_tensor.to(target_device, dtype=target_dtype)
                else:
                    processed_inputs[k] = v_tensor.to(target_device)
        except Exception as e_device:
            logger.error(f"Error moving CLIPSeg inputs to device for text '{effective_text}': {e_device}", exc_info=True)
            return None

        try:
            with torch.no_grad():
                outputs = model(**processed_inputs)
            
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
            
            if logits.ndim == 4: logits = logits.squeeze(0).squeeze(0) 
            elif logits.ndim == 3: logits = logits.squeeze(0)
            
            if logits.ndim != 2:
                logger.error(f"CLIPSeg logits unexpected ndim: {logits.ndim} for '{effective_text}'. Shape: {logits.shape}")
                return None

            probs = torch.sigmoid(logits).cpu().numpy().astype(np.float32)
            
            logger.debug(f"CLIPSeg 'probs' before resize: shape={probs.shape}, dtype={probs.dtype}, min={np.min(probs):.2f}, max={np.max(probs):.2f}")

            if probs.size == 0 or probs.shape[0] == 0 or probs.shape[1] == 0:
                logger.error(f"CLIPSeg 'probs' empty/zero-dim for '{effective_text}'. Shape: {probs.shape}")
                return None
            
            target_h, target_w = img_bgr_np.shape[:2]
            target_size_cv = (target_w, target_h)

            mask_resized_float = cv2.resize(probs, target_size_cv, interpolation=cv2.INTER_LINEAR)
            mask_resized_float = np.clip(mask_resized_float, 0.0, 1.0)
            final_mask_uint8 = (mask_resized_float * 255).astype(np.uint8)
            
            logger.debug(f"CLIPSeg mask for '{effective_text}'. Output shape: {final_mask_uint8.shape}")
            return final_mask_uint8
        except cv2.error as e_cv_resize:
            logger.error(f"OpenCV resize error in CLIPSeg for '{effective_text}': {e_cv_resize}", exc_info=True)
            logger.error(f"Details - probs shape: {probs.shape if 'probs' in locals() else 'N/A'}, target_size: {target_size_cv if 'target_size_cv' in locals() else 'N/A'}")
            return None
        except Exception as e_runtime:
            logger.error(f"Runtime error in CLIPSeg for '{effective_text}': {e_runtime}", exc_info=True)
            return None

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
            logger.error(f"Morphology error: {e_morph}", exc_info=True)
            return mask

    def _dynamic_morphology(self, mask: np.ndarray, img: np.ndarray) -> np.ndarray:
        if mask is None or img is None or img.size == 0: return mask
        h, w = img.shape[:2]
        min_dim = min(h,w)
        # Scalare mai agresivă a kernel-urilor pentru imagini mai mari
        close_k_factor = 0.015 if min_dim > 1000 else 0.01
        open_k_factor = 0.007 if min_dim > 1000 else 0.005

        close_k = max(3, int(close_k_factor * min_dim) // 2 * 2 + 1) 
        open_k = max(3, int(open_k_factor * min_dim) // 2 * 2 + 1)
        # Iteratii mai multe pentru măști mai zgomotoase sau imagini mari
        close_iter = 3 if min_dim > 700 else 2
        open_iter = 2 if min_dim > 700 else 1
        logger.debug(f"Dynamic morphology: close_k={close_k}, open_k={open_k}, close_iter={close_iter}, open_iter={open_iter}")
        return self._morphology(mask, close_k, open_k, close_iter, open_iter)

    def _advanced_edge_refine(self, mask: np.ndarray, img: np.ndarray) -> np.ndarray:
        if mask is None or img is None or img.size == 0: return mask
        try:
            if not hasattr(cv2, 'ximgproc') or not hasattr(cv2.ximgproc, 'guidedFilter'):
                logger.warning("cv2.ximgproc.guidedFilter not available. Falling back to basic edge_refine.")
                return self._edge_refine(mask, img)
        except AttributeError:
             logger.warning("cv2.ximgproc module not available. Falling back to basic edge_refine.")
             return self._edge_refine(mask, img)

        try:
            mask_float = mask.astype(np.float32) / 255.0
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            edges_canny = cv2.Canny(gray_img, 30, 100).astype(np.float32) / 255.0 # Praguri mai joase
            
            grad_x = cv2.Sobel(gray_img, cv2.CV_32F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray_img, cv2.CV_32F, 0, 1, ksize=3)
            edges_sobel_mag = cv2.magnitude(grad_x, grad_y)
            max_sobel = np.max(edges_sobel_mag)
            edges_sobel = edges_sobel_mag / max_sobel if max_sobel > 0 else np.zeros_like(edges_sobel_mag, dtype=np.float32)

            combined_edges = np.maximum(edges_canny, edges_sobel)
            edge_influence = 0.4 # Redus puțin
            blended_float = mask_float * (1 - combined_edges * edge_influence) + combined_edges * (mask_float * (1 - edge_influence)) # Blend mai inteligent

            radius = max(3, int(0.005 * min(img.shape[:2]))) # Radius mai mic
            eps_val = 0.005 # Epsilon mai mic pentru mai multă netezire, dar păstrarea muchiilor
            
            guide_img = img if img.dtype == np.uint8 else (img.clip(0,255)).astype(np.uint8)
            if guide_img.ndim == 2: guide_img = cv2.cvtColor(guide_img, cv2.COLOR_GRAY2BGR)

            blended_u8_for_guide = (np.clip(blended_float, 0, 1) * 255).astype(np.uint8)
            if blended_u8_for_guide.ndim != 2:
                 blended_u8_for_guide = cv2.cvtColor(blended_u8_for_guide, cv2.COLOR_BGR2GRAY) if blended_u8_for_guide.ndim == 3 and blended_u8_for_guide.shape[2] > 1 else blended_u8_for_guide.squeeze(axis=2) if blended_u8_for_guide.ndim == 3 else blended_u8_for_guide

            if blended_u8_for_guide.ndim != 2:
                logger.error("Cannot convert mask for guided filter to 2D. Skipping.")
                return mask

            filtered_mask = cv2.ximgproc.guidedFilter(guide=guide_img, src=blended_u8_for_guide, radius=radius, eps=eps_val*eps_val*255*255) 
            if filtered_mask.dtype != np.uint8:
                 filtered_mask = np.clip(filtered_mask, 0, 255).astype(np.uint8)

            _, thresholded_mask = cv2.threshold(filtered_mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) # Otsu pentru prag automat
            
            kernel_final_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            cleaned_mask = cv2.morphologyEx(thresholded_mask, cv2.MORPH_OPEN, kernel_final_clean, iterations=1)
            cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel_final_clean, iterations=2) # O închidere în plus
            return cleaned_mask
        except Exception as e_adv:
            logger.error(f"Advanced edge refine error: {e_adv}. Falling back.", exc_info=True)
            return self._edge_refine(mask, img) 

    def _edge_refine(self, mask: np.ndarray, img: np.ndarray) -> np.ndarray: # Fallback
        if mask is None or img is None or img.size == 0: return mask
        try:
            # Simplificat pentru fallback
            kernel_size = max(3, int(0.005 * min(img.shape[:2])) // 2 * 2 + 1)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            opened_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
            closed_mask = cv2.morphologyEx(opened_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
            return closed_mask
        except Exception as e_basic:
            logger.error(f"Basic edge refine error: {e_basic}", exc_info=True)
            return mask
    
    # _adaptive_threshold nu mai este folosit direct, dar poate fi util în viitor
    def _adaptive_threshold(self, mask_channel: np.ndarray, img: np.ndarray) -> np.ndarray:
        # ... (codul rămâne la fel ca în versiunea anterioară corectată)
        if mask_channel is None or img is None or img.size == 0:
            return mask_channel if mask_channel is not None else np.array([], dtype=np.uint8)
        if mask_channel.dtype != np.uint8:
            mask_channel = np.clip(mask_channel, 0, 255).astype(np.uint8)
        if mask_channel.ndim != 2:
            if mask_channel.ndim == 3 and mask_channel.shape[2] == 1: mask_channel = mask_channel.squeeze(axis=2)
            else: 
                _, th_mask = cv2.threshold(mask_channel, 127, 255, cv2.THRESH_BINARY); return th_mask
        if mask_channel.size == 0: return np.array([], dtype=np.uint8)
        block_size = min(31, max(3, int(0.05 * min(img.shape[:2])) // 2 * 2 + 1)) 
        C_val = 2 
        try:
            return cv2.adaptiveThreshold(mask_channel, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, C_val)
        except Exception: 
            _, th_mask = cv2.threshold(mask_channel, 127, 255, cv2.THRESH_BINARY); return th_mask

