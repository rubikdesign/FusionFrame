#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
MaskGenerator v2.2 — high-precision, operation-aware mask generation
for FusionFrame 2.0 (Versiune Corectată pentru eroarea OpenCV)

Modes:
  1) Background replace/remove → tight GrabCut subject mask (inverted to background mask) + dynamic morphology + advanced edge refine
  2) Hair color change          → CLIPSeg("hair") & CLIPSeg("head") + threshold + dynamic morphology + advanced edge refine
  3) All other edits            → hybrid average of YOLO-seg, MediaPipe, face box, CLIPSeg with fallback + dynamic morphology + advanced edge refine
"""

import cv2
import numpy as np
import logging
import torch
from typing import Dict, Any, Union, Callable, Optional
from PIL import Image

# Assuming AppConfig is a class or dict-like object for configuration
# from config.app_config import AppConfig
# For standalone running, let's mock it:
class AppConfigMock:
    def __init__(self):
        self._config = {
            "CLIPSEG_HAIR_THRESHOLD": 0.4,
            "CLIPSEG_HEAD_THRESHOLD": 0.3,
            "HYBRID_MASK_THRESHOLD": 0.35, # Adjusted from 0.3
        }
    def get(self, key: str, default: Any = None) -> Any:
        return self._config.get(key, default)

from core.model_manager import ModelManager # Asigurați-vă că acest import este valid în structura proiectului dvs.

logger = logging.getLogger(__name__)

class MaskGenerator:
    def __init__(self):
        self.models = ModelManager()
        self.config = AppConfigMock() # Înlocuiți cu `AppConfig()` în mediul real

    def generate_mask(
        self,
        image: Union[Image.Image, np.ndarray],
        prompt: str = "",
        operation: Dict[str, Any] = None,
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> Dict[str, Any]:
        """
        Returns a dict with:
          'mask'     : np.uint8 binary mask (0 or 255) for inpainting (area to be edited is 255)
          'raw_mask' : np.uint8 binary mask before final refinement steps
          'success'  : bool
          'message'  : str
        """
        def upd(pct: float, desc: str):
            if progress_callback:
                progress_callback(pct, desc)

        img_np = np.asarray(image) if isinstance(image, Image.Image) else image.copy()
        if img_np.ndim == 2: # Grayscale image
            img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
        elif img_np.shape[2] == 4: # RGBA image
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGR)
        
        if img_np.dtype != np.uint8: # Asigurăm că imaginea este uint8 pentru majoritatea operațiilor OpenCV
            logger.warning(f"Input image numpy array is not uint8 (type: {img_np.dtype}). Converting to uint8.")
            if np.max(img_np) <= 1.0 and np.min(img_np) >=0.0 : # Posibil float 0-1
                 img_np = (img_np * 255).astype(np.uint8)
            else: # Încercăm o conversie directă, cu riscuri dacă datele nu sunt potrivite
                 img_np = img_np.astype(np.uint8)


        h, w = img_np.shape[:2]

        op_type = (operation or {}).get("type", "").lower()
        target = (operation or {}).get("target", "").lower()
        lprompt = prompt.lower()

        # === Mode 1: Background replace/remove ===
        if op_type in ("replace", "remove") and ("background" in lprompt or target == "background"):
            upd(0.1, "Running GrabCut for subject")
            raw_subj_mask = self._grabcut_subject(img_np)
            if raw_subj_mask is None: # GrabCut poate eșua
                logger.error("GrabCut failed to produce a subject mask.")
                return {"mask": np.zeros((h, w), np.uint8), "raw_mask": None, "success": False, "message": "GrabCut subject segmentation failed"}

            upd(0.4, "Applying dynamic morphology to subject mask")
            morphed_subj_mask = self._dynamic_morphology(raw_subj_mask, img_np)
            upd(0.7, "Refining subject mask edges")
            refined_subj_mask = self._advanced_edge_refine(morphed_subj_mask, img_np)
            
            final_mask = cv2.bitwise_not(refined_subj_mask)
            upd(1.0, "Background mask ready")
            return {"mask": final_mask, "raw_mask": refined_subj_mask, "success": True, "message": "Background mask (inverted subject)"}

        # === Mode 2: Hair color change only ===
        if op_type == "color" and ("hair" in lprompt or target == "hair"):
            upd(0.1, "Segmenting hair via CLIPSeg")
            hair_seg = self._clipseg_segment(img_np, "hair") # `text` este promptul specific
            if hair_seg is None:
                return {"mask": np.zeros((h, w), np.uint8), "raw_mask": None, "success": False, "message": "CLIPSeg hair segmentation failed"}

            hair_threshold = int(self.config.get("CLIPSEG_HAIR_THRESHOLD", 0.4) * 255)
            _, raw_hair_mask = cv2.threshold(hair_seg, hair_threshold, 255, cv2.THRESH_BINARY)

            upd(0.3, "Segmenting head for refinement (optional)")
            head_seg = self._clipseg_segment(img_np, "head") # `text` este promptul specific
            if head_seg is not None:
                head_threshold = int(self.config.get("CLIPSEG_HEAD_THRESHOLD", 0.3) * 255)
                _, head_mask_thresh = cv2.threshold(head_seg, head_threshold, 255, cv2.THRESH_BINARY)
                raw_hair_mask = cv2.bitwise_and(raw_hair_mask, head_mask_thresh)
            
            upd(0.5, "Applying dynamic morphology to hair mask")
            morphed_hair_mask = self._dynamic_morphology(raw_hair_mask, img_np)
            upd(0.8, "Refining hair mask edges")
            final_mask = self._advanced_edge_refine(morphed_hair_mask, img_np)
            upd(1.0, "Hair mask ready")
            return {"mask": final_mask, "raw_mask": raw_hair_mask, "success": True, "message": "Hair mask"}

        # === Mode 3: General hybrid pipeline ===
        upd(0.0, "Hybrid: Initializing")
        fallback_mask = self._grabcut_subject(img_np)
        if fallback_mask is None: # GrabCut poate eșua
            logger.warning("GrabCut fallback mask generation failed. Using a blank mask as fallback.")
            fallback_mask = np.zeros((h,w), dtype=np.uint8) # Un fallback și mai simplu

        accum, count = np.zeros((h, w), np.float32), 0

        yolo = self.models.get_model("yolo")
        if yolo:
            upd(0.15, "YOLO segmentation")
            try:
                preds = yolo.predict(source=img_np, stream=False, imgsz=640, conf=0.25, verbose=False)
                for r in preds:
                    if getattr(r, "masks", None) and hasattr(r.masks, "data") and r.masks.data.numel() > 0:
                        masks_data = r.masks.data.cpu().numpy()
                        for m_yolo in masks_data:
                            if m_yolo.size == 0: continue # Skip empty masks
                            m_resized = cv2.resize(m_yolo.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
                            accum += m_resized
                            count += 1
            except Exception as e:
                logger.error(f"YOLO error: {e}", exc_info=True)

        mp = self.models.get_model("mediapipe")
        if mp:
            upd(0.3, "MediaPipe segmentation")
            try:
                res_mp = mp.process(cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB))
                mm = getattr(res_mp, 'segmentation_mask', None) # Folosim getattr pentru siguranță
                if mm is not None and mm.size > 0:
                    mmf = cv2.resize(mm.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
                    accum += mmf 
                    count += 1
            except Exception as e:
                logger.error(f"MediaPipe error: {e}", exc_info=True)

        fd = self.models.get_model("face_detector")
        if fd and "face" in lprompt:
            upd(0.4, "Face detection mask")
            try:
                dets = fd.process(cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB))
                fm = np.zeros((h, w), np.float32)
                if getattr(dets, "detections", None):
                    for d_face in dets.detections:
                        bb = d_face.location_data.relative_bounding_box
                        fx, fy = int(bb.xmin * w), int(bb.ymin * h)
                        fW, fH = int(bb.width * w), int(bb.height * h)
                        cv2.rectangle(fm, (fx, fy), (fx + fW, fy + fH), 1.0, -1)
                    if fm.sum() > 0:
                        accum += fm
                        count += 1
            except Exception as e:
                logger.error(f"Face detector error: {e}", exc_info=True)
        
        upd(0.5, "CLIPSeg generic mask")
        clip_prompt = prompt if prompt and prompt.strip() else "subject" # Un prompt default mai relevant
        clip_gen_seg = self._clipseg_segment(img_np, clip_prompt) # `text` este promptul
        if clip_gen_seg is not None:
            accum += clip_gen_seg.astype(np.float32) / 255.0 
            count += 1

        if count > 0:
            combined_float_mask = accum / count
        else:
            logger.warning("No segmentation models contributed to the hybrid mask. Using fallback_mask.")
            combined_float_mask = fallback_mask.astype(np.float32) / 255.0

        upd(0.7, "Thresholding hybrid mask")
        hybrid_threshold_val = self.config.get("HYBRID_MASK_THRESHOLD", 0.35)
        raw_hybrid_mask = (combined_float_mask > hybrid_threshold_val).astype(np.uint8) * 255
        
        upd(0.8, "Applying dynamic morphology to hybrid mask")
        morphed_hybrid_mask = self._dynamic_morphology(raw_hybrid_mask, img_np)
        upd(0.9, "Refining hybrid mask edges")
        final_mask = self._advanced_edge_refine(morphed_hybrid_mask, img_np)
        upd(1.0, "Hybrid mask ready")

        return {"mask": final_mask, "raw_mask": raw_hybrid_mask, "success": True, "message": "Hybrid mask"}

    def _grabcut_subject(self, img: np.ndarray, rect_inset_ratio: float = 0.05) -> Optional[np.ndarray]:
        h, w = img.shape[:2]
        if h == 0 or w == 0:
            logger.error("GrabCut received an empty image.")
            return None
            
        mask_gc = np.zeros((h, w), np.uint8)
        bgd_model, fgd_model = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
        
        dx, dy = int(rect_inset_ratio * w), int(rect_inset_ratio * h) 
        rect_w, rect_h = w - 2*dx, h - 2*dy

        if rect_w <=0 or rect_h <=0: 
            logger.warning(f"GrabCut rectangle is too small or invalid ({dx},{dy},{rect_w},{rect_h}). Using full image.")
            rect = (1, 1, w-2 if w > 1 else 1, h-2 if h > 1 else 1) # Asigurăm rect pozitiv
            if rect[2] <= 0 or rect[3] <= 0: # Dacă imaginea e 1x1 sau similar
                 logger.error("Image too small for GrabCut even with full rect attempt.")
                 return None
        else:
            rect = (dx, dy, rect_w, rect_h)

        try:
            cv2.grabCut(img, mask_gc, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
            subject_mask = np.where((mask_gc == cv2.GC_FGD) | (mask_gc == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
            return subject_mask
        except cv2.error as e_cv:
            logger.error(f"GrabCut OpenCV error: {e_cv}", exc_info=True)
            # Fallback la un dreptunghi umplut dacă GrabCut eșuează catastrofal
            fb_mask = np.zeros((h, w), np.uint8)
            # Folosim coordonatele rect-ului calculate, asigurându-ne că sunt valide
            cv2.rectangle(fb_mask, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), 255, -1)
            return fb_mask
        except Exception as e:
            logger.error(f"GrabCut generic error: {e}", exc_info=True)
            return None


    def _clipseg_segment(self, img_rgb_np: np.ndarray, text_prompt: str) -> Optional[np.ndarray]:
        """Runs CLIPSeg, returns uint8 mask [0–255], same H×W as input img_rgb_np."""
        bundle = self.models.get_model("clipseg")
        if not bundle or "processor" not in bundle or "model" not in bundle:
            logger.warning("CLIPSeg model or processor not found/incomplete in ModelManager.")
            return None

        processor, model = bundle["processor"], bundle["model"]
        
        # Imaginea de intrare pentru CLIPSeg ar trebui să fie RGB PIL Image
        try:
            pil_image = Image.fromarray(img_rgb_np) # Presupunem că img_rgb_np este deja BGR np.uint8
        except Exception as e_pil:
            logger.error(f"Failed to convert numpy array to PIL Image for CLIPSeg: {e_pil}", exc_info=True)
            return None
            
        effective_text = text_prompt if text_prompt and text_prompt.strip() else "object"
        logger.debug(f"CLIPSeg processing with text: '{effective_text}'")

        try:
            inputs = processor(
                text=[effective_text], 
                images=[pil_image], 
                return_tensors="pt", 
                padding=True
            )
        except Exception as e_proc:
            logger.error(f"CLIPSeg processor error for text '{effective_text}': {e_proc}", exc_info=True)
            return None
            
        processed_inputs = {}
        try:
            for k, v_tensor in inputs.items():
                if v_tensor.dtype.is_floating_point:
                    processed_inputs[k] = v_tensor.to(model.device, dtype=model.dtype)
                else:
                    processed_inputs[k] = v_tensor.to(model.device)
        except Exception as e_device:
            logger.error(f"Error moving CLIPSeg inputs to device for text '{effective_text}': {e_device}", exc_info=True)
            return None

        try:
            with torch.no_grad():
                outputs = model(**processed_inputs)
            
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
            
            # Asigurăm că logits este 2D (H_model, W_model)
            if logits.ndim == 4: # (batch, num_prompts, H_model, W_model)
                logits = logits.squeeze(0).squeeze(0) 
            elif logits.ndim == 3: # (batch_or_num_prompts, H_model, W_model)
                logits = logits.squeeze(0)
            
            if logits.ndim != 2:
                logger.error(f"CLIPSeg logits have unexpected ndim after squeeze: {logits.ndim} for text '{effective_text}'. Shape: {logits.shape}")
                return None

            probs = torch.sigmoid(logits).cpu().numpy() # Acum `probs` este 2D float32 (H_model, W_model)
            
            # --- Aici începe partea critică pentru cv2.resize ---
            logger.debug(f"CLIPSeg 'probs' before resize: shape={probs.shape}, dtype={probs.dtype}, ndim={probs.ndim}, min={np.min(probs) if probs.size > 0 else 'N/A'}, max={np.max(probs) if probs.size > 0 else 'N/A'}")

            if probs.size == 0 or probs.shape[0] == 0 or probs.shape[1] == 0:
                logger.error(f"CLIPSeg 'probs' is empty or has zero dimension before resize for text '{effective_text}'. Shape: {probs.shape}")
                return None

            # Asigurăm explicit float32, deși ar trebui să fie deja
            probs = probs.astype(np.float32) 
            
            target_h, target_w = img_rgb_np.shape[:2]
            target_size_cv = (target_w, target_h) # OpenCV dsize este (width, height)

            mask_resized_float = cv2.resize(probs, target_size_cv, interpolation=cv2.INTER_LINEAR)
            
            # Normalizăm și convertim la uint8
            # mask_resized_float poate avea valori în afara [0,1] după interpolare, deși rar pt INTER_LINEAR din sigmoid
            mask_resized_float = np.clip(mask_resized_float, 0.0, 1.0) # Asigurăm intervalul [0,1]
            final_mask_uint8 = (mask_resized_float * 255).astype(np.uint8)
            
            logger.debug(f"CLIPSeg successfully generated mask for text '{effective_text}'. Output shape: {final_mask_uint8.shape}")
            return final_mask_uint8

        except cv2.error as e_cv_resize:
            logger.error(f"OpenCV resize error in CLIPSeg for text '{effective_text}': {e_cv_resize}", exc_info=True)
            logger.error(f"Details - probs shape: {probs.shape if 'probs' in locals() else 'N/A'}, dtype: {probs.dtype if 'probs' in locals() else 'N/A'}, target_size_cv: {target_size_cv if 'target_size_cv' in locals() else 'N/A'}")
            # --- Opțional: Încercați INTER_NEAREST ca diagnostic ---
            # try:
            #     logger.warning(f"Retrying CLIPSeg resize with INTER_NEAREST for text '{effective_text}'.")
            #     mask_resized_float = cv2.resize(probs, target_size_cv, interpolation=cv2.INTER_NEAREST)
            #     mask_resized_float = np.clip(mask_resized_float, 0.0, 1.0)
            #     final_mask_uint8 = (mask_resized_float * 255).astype(np.uint8)
            #     logger.info(f"CLIPSeg resize with INTER_NEAREST succeeded for '{effective_text}'.")
            #     return final_mask_uint8
            # except Exception as e_nearest:
            #     logger.error(f"CLIPSeg resize with INTER_NEAREST also failed for '{effective_text}': {e_nearest}", exc_info=True)
            return None
        except Exception as e_runtime: # Alte erori în timpul procesării modelului sau redimensionării
            logger.error(f"Runtime error in CLIPSeg processing text '{effective_text}': {e_runtime}", exc_info=True)
            return None

    def _morphology(self, mask: np.ndarray, close_k:int, open_k:int, close_iter:int, open_iter:int) -> np.ndarray:
        if mask is None or mask.size == 0:
            logger.warning("Morphology received an empty mask.")
            return mask
        if close_k <= 0 or open_k <=0: 
            logger.warning(f"Morphology kernel sizes invalid (close_k={close_k}, open_k={open_k}). Returning original mask.")
            return mask
        
        # Asigurăm că kernel_size este impar
        close_k = close_k if close_k % 2 != 0 else close_k + 1
        open_k = open_k if open_k % 2 != 0 else open_k + 1

        try:
            ker_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
            ker_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k, open_k))
            m = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, ker_close, iterations=close_iter)
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN, ker_open, iterations=open_iter)
            return m
        except cv2.error as e_cv_morph:
            logger.error(f"OpenCV error during morphology: {e_cv_morph}", exc_info=True)
            return mask # Returnăm masca originală în caz de eroare
        except Exception as e_morph:
            logger.error(f"Generic error during morphology: {e_morph}", exc_info=True)
            return mask


    def _dynamic_morphology(self, mask: np.ndarray, img: np.ndarray) -> np.ndarray:
        if mask is None or img is None or img.size == 0:
            logger.warning("Dynamic morphology received empty mask or image.")
            return mask
        h, w = img.shape[:2]
        close_k = max(3, int(0.01 * min(h, w)) // 2 * 2 + 1) 
        open_k = max(3, int(0.005 * min(h, w)) // 2 * 2 + 1)
        close_iter = 2
        open_iter = 1
        return self._morphology(mask, close_k, open_k, close_iter, open_iter)

    def _advanced_edge_refine(self, mask: np.ndarray, img: np.ndarray) -> np.ndarray:
        if mask is None or img is None or img.size == 0:
            logger.warning("Advanced edge refine received empty mask or image.")
            return mask
        
        try:
            # Verificăm dacă ximgproc este disponibil și funcțional
            # Uneori `cv2.ximgproc` poate exista dar `guidedFilter` nu, în funcție de build-ul OpenCV
            if not hasattr(cv2, 'ximgproc') or not hasattr(cv2.ximgproc, 'guidedFilter'):
                logger.warning("cv2.ximgproc.guidedFilter not available. Falling back to basic edge_refine.")
                return self._edge_refine(mask, img)
        except AttributeError: # În cazul în care cv2.ximgproc nu este deloc definit
             logger.warning("cv2.ximgproc module not available. Falling back to basic edge_refine.")
             return self._edge_refine(mask, img)


        try:
            mask_float = mask.astype(np.float32) / 255.0
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            edges_canny = cv2.Canny(gray_img, 50, 150).astype(np.float32) / 255.0
            
            grad_x = cv2.Sobel(gray_img, cv2.CV_32F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray_img, cv2.CV_32F, 0, 1, ksize=3)
            edges_sobel_mag = cv2.magnitude(grad_x, grad_y)
            if np.max(edges_sobel_mag) > 0: # Evităm împărțirea la zero
                edges_sobel = edges_sobel_mag / np.max(edges_sobel_mag)
            else:
                edges_sobel = np.zeros_like(edges_sobel_mag, dtype=np.float32)

            combined_edges = np.maximum(edges_canny, edges_sobel)
            edge_influence = 0.5 
            blended_float = mask_float * (1 - combined_edges * edge_influence) + combined_edges * (1 - edge_influence)
            
            radius = max(5, int(0.01 * min(img.shape[:2]))) 
            eps_val = 0.01 # Nume variabilă schimbat din `eps` pentru a evita conflictul cu `eps` din cv2.ximgproc.guidedFilter
            
            guide_img = img if img.dtype == np.uint8 else (img.clip(0,255)).astype(np.uint8) # Asigurăm uint8 și clip
            if guide_img.ndim == 2: guide_img = cv2.cvtColor(guide_img, cv2.COLOR_GRAY2BGR)

            blended_u8_for_guide = (np.clip(blended_float, 0, 1) * 255).astype(np.uint8)
            if blended_u8_for_guide.ndim == 3 and blended_u8_for_guide.shape[2] ==1:
                 blended_u8_for_guide = blended_u8_for_guide.squeeze(axis=2)
            elif blended_u8_for_guide.ndim ==3 and blended_u8_for_guide.shape[2] > 1:
                 blended_u8_for_guide = cv2.cvtColor(blended_u8_for_guide, cv2.COLOR_BGR2GRAY)

            # Asigurăm că blended_u8_for_guide este 2D
            if blended_u8_for_guide.ndim != 2:
                logger.warning(f"Mask for guided filter is not 2D (shape: {blended_u8_for_guide.shape}). Converting to grayscale if possible.")
                if blended_u8_for_guide.ndim == 3 and blended_u8_for_guide.shape[2] > 1 :
                     blended_u8_for_guide = cv2.cvtColor(blended_u8_for_guide, cv2.COLOR_BGR2GRAY)
                else: # Nu se poate converti la 2D ușor
                    logger.error("Cannot convert mask for guided filter to 2D. Skipping guided filter.")
                    return mask


            filtered_mask = cv2.ximgproc.guidedFilter(
                guide=guide_img, 
                src=blended_u8_for_guide, 
                radius=radius, 
                eps=eps_val*eps_val*255*255 # eps pentru guidedFilter este pătratul celui intuitiv și scalat
            ) 
            
            # Asigurăm că filtered_mask este în intervalul 0-255 și uint8 înainte de threshold
            if filtered_mask.dtype != np.uint8:
                 filtered_mask = np.clip(filtered_mask, 0, 255).astype(np.uint8)

            _, thresholded_mask = cv2.threshold(filtered_mask, 127, 255, cv2.THRESH_BINARY)
            
            kernel_final_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            cleaned_mask = cv2.morphologyEx(thresholded_mask, cv2.MORPH_OPEN, kernel_final_clean, iterations=1)
            cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel_final_clean, iterations=1)
            
            return cleaned_mask
        except cv2.error as e_cv_adv:
             logger.error(f"OpenCV error in advanced edge refine: {e_cv_adv}. Falling back.", exc_info=True)
             return self._edge_refine(mask, img) # Fallback la basic în caz de eroare OpenCV
        except Exception as e_adv:
            logger.error(f"Advanced edge refine generic error: {e_adv}. Falling back to input mask.", exc_info=True)
            return mask 

    def _edge_refine(self, mask: np.ndarray, img: np.ndarray) -> np.ndarray:
        if mask is None or img is None or img.size == 0:
            logger.warning("Basic edge refine received empty mask or image.")
            return mask
        try:
            m_float = mask.astype(np.float32) / 255.0
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150).astype(np.float32) / 255.0
            
            blend = m_float * (1 - edges * 0.3) + edges * 0.1
            u8_blend = (np.clip(blend,0,1) * 255).astype(np.uint8)
            
            filt = cv2.bilateralFilter(u8_blend, d=9, sigmaColor=75, sigmaSpace=75)
            _, th_mask = cv2.threshold(filt, 127, 255, cv2.THRESH_BINARY)
            
            ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            clean = cv2.morphologyEx(th_mask, cv2.MORPH_CLOSE, ker, iterations=1) 
            clean = cv2.morphologyEx(clean, cv2.MORPH_OPEN, ker, iterations=1)
            return clean
        except cv2.error as e_cv_basic:
            logger.error(f"OpenCV error in basic edge refine: {e_cv_basic}", exc_info=True)
            return mask # Returnăm masca originală
        except Exception as e_basic:
            logger.error(f"Basic edge refine generic error: {e_basic}", exc_info=True)
            return mask

    def _adaptive_threshold(self, mask_channel: np.ndarray, img: np.ndarray) -> np.ndarray:
        if mask_channel is None or img is None or img.size == 0:
            logger.warning("Adaptive threshold received empty mask_channel or image.")
            return mask_channel if mask_channel is not None else np.array([], dtype=np.uint8)

        # Asigurăm că mask_channel este uint8 și 2D
        if mask_channel.dtype != np.uint8:
            mask_channel = np.clip(mask_channel, 0, 255).astype(np.uint8)

        if mask_channel.ndim != 2:
            logger.warning(f"Adaptive threshold expects single channel 2D mask. Got ndim={mask_channel.ndim}")
            if mask_channel.ndim == 3 and mask_channel.shape[2] == 1:
                mask_channel = mask_channel.squeeze(axis=2)
            else: 
                logger.error("Cannot convert mask_channel to 2D for adaptive threshold. Applying simple binary threshold.")
                _, th_mask = cv2.threshold(mask_channel, 127, 255, cv2.THRESH_BINARY)
                return th_mask
        
        if mask_channel.size == 0: # Verificare suplimentară după squeeze
            logger.error("Mask_channel is empty after processing for adaptive threshold.")
            return np.array([], dtype=np.uint8)


        block_size = min(31, max(3, int(0.05 * min(img.shape[:2])) // 2 * 2 + 1)) 
        C_val = 2 
        try:
            adapt_mask = cv2.adaptiveThreshold(
                mask_channel, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, C_val
            )
            return adapt_mask
        except cv2.error as e_cv_adapt:
            logger.error(f"OpenCV error in adaptive threshold (block_size={block_size}): {e_cv_adapt}. Falling back to simple threshold.", exc_info=True)
            _, th_mask = cv2.threshold(mask_channel, 127, 255, cv2.THRESH_BINARY)
            return th_mask
        except Exception as e_adapt:
            logger.error(f"Generic error in adaptive threshold: {e_adapt}. Falling back to simple threshold.", exc_info=True)
            _, th_mask = cv2.threshold(mask_channel, 127, 255, cv2.THRESH_BINARY)
            return th_mask