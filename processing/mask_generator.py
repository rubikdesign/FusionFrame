#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
MaskGenerator v2.2 — high-precision, operation-aware mask generation
for FusionFrame 2.0

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

from core.model_manager import ModelManager

logger = logging.getLogger(__name__)

class MaskGenerator:
    def __init__(self):
        self.models = ModelManager()
        self.config = AppConfigMock() # Use AppConfig in your actual environment

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

        h, w = img_np.shape[:2]

        op_type = (operation or {}).get("type", "").lower()
        target = (operation or {}).get("target", "").lower()
        lprompt = prompt.lower()

        # === Mode 1: Background replace/remove ===
        if op_type in ("replace", "remove") and ("background" in lprompt or target == "background"):
            upd(0.1, "Running GrabCut for subject")
            raw_subj_mask = self._grabcut_subject(img_np)
            upd(0.4, "Applying dynamic morphology to subject mask")
            morphed_subj_mask = self._dynamic_morphology(raw_subj_mask, img_np)
            upd(0.7, "Refining subject mask edges")
            refined_subj_mask = self._advanced_edge_refine(morphed_subj_mask, img_np)
            
            # Invert subject mask to get background mask (area to be edited is 255)
            final_mask = cv2.bitwise_not(refined_subj_mask)
            upd(1.0, "Background mask ready")
            return {"mask": final_mask, "raw_mask": refined_subj_mask, "success": True, "message": "Background mask (inverted subject)"}

        # === Mode 2: Hair color change only ===
        if op_type == "color" and ("hair" in lprompt or target == "hair"):
            upd(0.1, "Segmenting hair via CLIPSeg")
            hair_seg = self._clipseg_segment(img_np, "hair")
            if hair_seg is None:
                return {"mask": np.zeros((h, w), np.uint8), "raw_mask": None, "success": False, "message": "CLIPSeg hair segmentation failed"}

            hair_threshold = int(self.config.get("CLIPSEG_HAIR_THRESHOLD", 0.4) * 255)
            _, raw_hair_mask = cv2.threshold(hair_seg, hair_threshold, 255, cv2.THRESH_BINARY)

            # Optional: Refine with head mask
            upd(0.3, "Segmenting head for refinement (optional)")
            head_seg = self._clipseg_segment(img_np, "head")
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
        # Fallback is a subject mask. This mode usually targets an object/region.
        fallback_mask = self._grabcut_subject(img_np) 
        accum, count = np.zeros((h, w), np.float32), 0

        # YOLO segmentation
        yolo = self.models.get_model("yolo")
        if yolo:
            upd(0.15, "YOLO segmentation")
            try:
                preds = yolo.predict(source=img_np, stream=False, imgsz=640, conf=0.25, verbose=False) # smaller imgsz for speed
                for r in preds:
                    if getattr(r, "masks", None) and hasattr(r.masks, "data"):
                        masks_data = r.masks.data.cpu().numpy()
                        for m_yolo in masks_data:
                            m_resized = cv2.resize(m_yolo.astype(np.float32), (w, h), cv2.INTER_LINEAR)
                            accum += m_resized
                            count += 1
            except Exception as e:
                logger.error(f"YOLO error: {e}")

        # MediaPipe segmentation
        mp = self.models.get_model("mediapipe")
        if mp:
            upd(0.3, "MediaPipe segmentation")
            try:
                res_mp = mp.process(cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB))
                mm = res_mp.segmentation_mask
                if mm is not None:
                    mmf = cv2.resize(mm.astype(np.float32), (w, h), cv2.INTER_LINEAR)
                    accum += mmf # MediaPipe mask is 0-1 range
                    count += 1
            except Exception as e:
                logger.error(f"MediaPipe error: {e}")

        # Face detection mask (as a region of interest, not a segmentation)
        fd = self.models.get_model("face_detector")
        if fd and "face" in lprompt: # Only add if "face" is relevant
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
                logger.error(f"Face detector error: {e}")
        
        # Generic CLIPSeg (full prompt)
        upd(0.5, "CLIPSeg generic mask")
        # Use a more specific part of the prompt if possible, e.g., extract nouns
        clip_prompt = prompt if prompt else "object" 
        clip_gen_seg = self._clipseg_segment(img_np, clip_prompt)
        if clip_gen_seg is not None:
            accum += clip_gen_seg.astype(np.float32) / 255.0 # Normalize if it's 0-255
            count += 1

        if count > 0:
            combined_float_mask = accum / count
        else:
            # If all else fails, use the fallback subject mask
            # This assumes the "other edit" is likely on the main subject
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

    # ——— Helper methods ——————————————————————————

    def _grabcut_subject(self, img: np.ndarray, rect_inset_ratio: float = 0.05) -> np.ndarray:
        """Returns 255=subject, 0=background using GrabCut."""
        h, w = img.shape[:2]
        mask_gc = np.zeros((h, w), np.uint8)
        bgd_model, fgd_model = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
        
        # Slightly smaller inset for potentially better subject coverage
        dx, dy = int(rect_inset_ratio * w), int(rect_inset_ratio * h) 
        rect = (dx, dy, w - 2*dx, h - 2*dy)
        if rect[2] <=0 or rect[3] <=0: # Handle very small images
            rect = (0,0,w,h)

        try:
            cv2.grabCut(img, mask_gc, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
            # GC_FGD (1), GC_PR_FGD (3) are foreground
            # GC_BGD (0), GC_PR_BGD (2) are background
            subject_mask = np.where((mask_gc == cv2.GC_FGD) | (mask_gc == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
            return subject_mask
        except Exception as e:
            logger.error(f"GrabCut failed: {e}")
            # Fallback to a filled rectangle if GrabCut fails catastrophically
            fb_mask = np.zeros((h, w), np.uint8)
            cv2.rectangle(fb_mask, (dx, dy), (w - dx, h - dy), 255, -1)
            return fb_mask

    def _clipseg_segment(self, img: np.ndarray, text: str) -> Optional[np.ndarray]:
        """Runs CLIPSeg, returns float mask [0–255] uint8, same H×W."""
        bundle = self.models.get_model("clipseg")
        if not bundle:
            logger.warning("CLIPSeg model not found in ModelManager.")
            return None

        processor, model = bundle["processor"], bundle["model"]
        pil_image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        
        # Ensure text is not empty, provide a default if necessary
        effective_text = text if text and text.strip() else "object"

        inputs = processor(text=[effective_text], images=[pil_image], return_tensors="pt", padding=True)
        
        processed_inputs = {}
        for k, v_tensor in inputs.items():
            # Cast floating point tensors to model's dtype, keep integer tensors as is
            if v_tensor.dtype.is_floating_point:
                processed_inputs[k] = v_tensor.to(model.device, dtype=model.dtype)
            else:
                processed_inputs[k] = v_tensor.to(model.device)
        try:
            with torch.no_grad():
                outputs = model(**processed_inputs)
            
            # Logits can be of different shapes depending on the model version / task
            # For basic CLIPSeg, logits is usually the direct output.
            # Some wrappers might have it under `outputs.logits`
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
            
            # Handle potential batch dimension if text list was > 1 (though we send 1)
            if logits.ndim == 3: # (batch, H, W) or (H, W, batch_somehow)
                logits = logits.squeeze(0) if logits.shape[0] == 1 else logits[0] # Assuming batch first
            elif logits.ndim == 4: # (batch, num_classes_or_prompts, H, W)
                 logits = logits.squeeze(0).squeeze(0) # Assuming single batch, single prompt

            probs = torch.sigmoid(logits).cpu().numpy()
            
            mask_resized = cv2.resize(probs, (img.shape[1], img.shape[0]), cv2.INTER_LINEAR)
            return (mask_resized * 255).astype(np.uint8)
        except Exception as e:
            logger.error(f"CLIPSeg error processing text '{effective_text}': {e}")
            return None

    def _morphology(self, mask: np.ndarray, close_k:int, open_k:int, close_iter:int, open_iter:int) -> np.ndarray:
        """Applies a close then open with elliptical kernels."""
        if close_k <= 0 or open_k <=0: return mask # Kernels must be positive
        ker_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
        ker_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k, open_k))
        m = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, ker_close, iterations=close_iter)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, ker_open, iterations=open_iter)
        return m

    def _dynamic_morphology(self, mask: np.ndarray, img: np.ndarray) -> np.ndarray:
        """Applies dynamic morphology based on image size."""
        h, w = img.shape[:2]
        # Ensure kernel sizes are odd and positive
        close_k = max(3, int(0.01 * min(h, w)) // 2 * 2 + 1) # e.g., 1% of min_dim, odd
        open_k = max(3, int(0.005 * min(h, w)) // 2 * 2 + 1) # e.g., 0.5% of min_dim, odd
        
        # Iterations can also be dynamic, but let's keep them fixed for now
        close_iter = 2
        open_iter = 1
        return self._morphology(mask, close_k, open_k, close_iter, open_iter)

    def _advanced_edge_refine(self, mask: np.ndarray, img: np.ndarray) -> np.ndarray:
        """Refines mask edges using guided filter and Canny/Sobel edges."""
        if not cv2.ximgproc:
            logger.warning("cv2.ximgproc not available. Falling back to basic edge_refine.")
            return self._edge_refine(mask, img)
        try:
            mask_float = mask.astype(np.float32) / 255.0
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            edges_canny = cv2.Canny(gray_img, 50, 150).astype(np.float32) / 255.0
            
            # Sobel edges
            grad_x = cv2.Sobel(gray_img, cv2.CV_32F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray_img, cv2.CV_32F, 0, 1, ksize=3)
            edges_sobel = cv2.magnitude(grad_x, grad_y)
            if edges_sobel.max() > 0:
                edges_sobel = edges_sobel / edges_sobel.max() # Normalize to 0-1

            # Combine Canny and Sobel (stronger edges)
            combined_edges = np.maximum(edges_canny, edges_sobel)

            # Blend mask with edges: make mask slightly thinner at strong edges
            # and slightly thicker where edges are weak but mask is strong
            # Weighting factor for how much edges influence the mask
            edge_influence = 0.5 
            blended_float = mask_float * (1 - combined_edges * edge_influence) + combined_edges * (1 - edge_influence)
            
            # Guided filter for smoothing while preserving edges
            # Radius and epsilon are important parameters
            radius = max(5, int(0.01 * min(img.shape[:2]))) 
            eps = 0.01 # Smaller eps means more adherence to edges, but can be noisy
            
            # Ensure img for guided filter is BGR and uint8
            guide_img = img if img.dtype == np.uint8 else (img * 255).astype(np.uint8)
            if guide_img.ndim == 2: guide_img = cv2.cvtColor(guide_img, cv2.COLOR_GRAY2BGR)


            # Ensure mask for guided filter is single channel
            blended_u8_for_guide = (np.clip(blended_float, 0, 1) * 255).astype(np.uint8)
            if blended_u8_for_guide.ndim == 3 and blended_u8_for_guide.shape[2] ==1:
                 blended_u8_for_guide = blended_u8_for_guide.squeeze(axis=2)
            elif blended_u8_for_guide.ndim ==3 and blended_u8_for_guide.shape[2] > 1: # Should not happen
                 blended_u8_for_guide = cv2.cvtColor(blended_u8_for_guide, cv2.COLOR_BGR2GRAY)


            filtered_mask = cv2.ximgproc.guidedFilter(guide=guide_img, src=blended_u8_for_guide, radius=radius, eps=eps*eps*255*255) # eps for guidedFilter is squared
            
            _, thresholded_mask = cv2.threshold(filtered_mask, 127, 255, cv2.THRESH_BINARY)
            
            # Final small morphological clean-up
            kernel_final_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            cleaned_mask = cv2.morphologyEx(thresholded_mask, cv2.MORPH_OPEN, kernel_final_clean, iterations=1)
            cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel_final_clean, iterations=1)
            
            return cleaned_mask
        except Exception as e:
            logger.error(f"Advanced edge refine error: {e}. Falling back to input mask.")
            return mask # Fallback to the input mask if advanced refinement fails

    def _edge_refine(self, mask: np.ndarray, img: np.ndarray) -> np.ndarray:
        """Basic edge refinement (fallback if ximgproc is not available)."""
        try:
            m_float = mask.astype(np.float32) / 255.0
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150).astype(np.float32) / 255.0
            
            # Simple blending
            blend = m_float * (1 - edges * 0.3) + edges * 0.1 # Less aggressive edge influence
            
            u8_blend = (np.clip(blend,0,1) * 255).astype(np.uint8)
            
            # Bilateral filter for smoothing while preserving edges somewhat
            filt = cv2.bilateralFilter(u8_blend, d=9, sigmaColor=75, sigmaSpace=75)
            
            _, th_mask = cv2.threshold(filt, 127, 255, cv2.THRESH_BINARY)
            
            ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            clean = cv2.morphologyEx(th_mask, cv2.MORPH_CLOSE, ker, iterations=1) # Reduced iterations
            clean = cv2.morphologyEx(clean, cv2.MORPH_OPEN, ker, iterations=1)
            return clean
        except Exception as e:
            logger.error(f"Basic edge refine error: {e}")
            return mask


    def _adaptive_threshold(self, mask_channel: np.ndarray, img: np.ndarray) -> np.ndarray:
        """Applies adaptive thresholding. Expects single channel mask_channel (0-255)."""
        if mask_channel.ndim != 2:
            logger.warning("Adaptive threshold expects single channel mask.")
            if mask_channel.ndim == 3 and mask_channel.shape[2] == 1:
                mask_channel = mask_channel.squeeze(axis=2)
            else: # Cannot easily convert, return original or simple threshold
                _, th_mask = cv2.threshold(mask_channel, 127, 255, cv2.THRESH_BINARY)
                return th_mask

        block_size = min(31, max(3, int(0.05 * min(img.shape[:2])) // 2 * 2 + 1)) # Odd, e.g. 5% of min_dim
        C_val = 2 # Constant subtracted from the mean
        adapt_mask = cv2.adaptiveThreshold(
            mask_channel, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, C_val
        )
        return adapt_mask