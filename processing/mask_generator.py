#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generator de măști pentru FusionFrame 2.0
"""

import cv2
import numpy as np
import logging
import torch
from typing import Dict, Any, List, Optional, Union, Tuple, Callable
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

from config.app_config import AppConfig
from core.model_manager import ModelManager

# Setăm logger-ul
logger = logging.getLogger(__name__)

class MaskGenerator:
    """
    Generator de măști pentru segmentare în FusionFrame 2.0
    """
    def __init__(self):
        self.model_manager = ModelManager()
        self.config = AppConfig
        self.progress_callback: Optional[Callable] = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def generate_mask(
        self,
        image: Union[Image.Image, np.ndarray],
        prompt: Optional[str] = None,
        operation: Optional[Dict[str, Any]] = None,
        progress_callback: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        self.progress_callback = progress_callback
        # convert input
        image_np = np.array(image) if isinstance(image, Image.Image) else image
        h, w = image_np.shape[:2]
        # weights
        mask_weights = {'sam': 0.4, 'yolo': 0.3, 'clipseg': 0.2, 'mediapipe': 0.1, 'face': 0.2, 'background': 0.5}
        final_mask = np.zeros((h, w), dtype=np.float32)
        # backup mask
        backup = np.ones((h, w), np.float32) * 0.5
        cv2.circle(backup,(w//2,h//2),min(w,h)//4,1.0,-1)

        # adjust for background or face
        if operation and operation.get('type') == 'background':
            bg = self._process_background_detection(image_np)
            if bg is not None:
                final_mask += (1-bg) * mask_weights['background']

        # parallel tasks
        tasks = []
        with ThreadPoolExecutor(max_workers=self.config.MAX_WORKERS) as exe:
            mapping = {
                'sam': (self._process_sam, (image_np,)),
                'yolo': (self._process_yolo, (image_np, prompt)),
                'clipseg': (self._process_clipseg, (image_np, prompt)),
                'mediapipe': (self._process_mediapipe, (image_np,)),
                'face': (self._process_face_detection, (image_np,)),
            }
            futures = {exe.submit(fn, *args): name for name, (fn, args) in mapping.items()}
            for i,(fut,name) in enumerate(futures.items()):
                try:
                    mask = fut.result(timeout=15)
                    if mask is not None:
                        final_mask += mask * mask_weights.get(name,0.1)
                except Exception as e:
                    logger.error(f"{name} error: {e}")
                if self.progress_callback:
                    self.progress_callback(0.2+0.6*(i+1)/len(futures), desc=f"{name} done")

        if final_mask.sum() <= 0:
            final_mask = backup
        # normalize
        mn, mx = final_mask.min(), final_mask.max()
        final_mask = (final_mask - mn)/(mx-mn) if mx>mn else final_mask
        # threshold
        thresh = {'background':0.35, 'color':0.25}.get(operation.get('type',''),0.3)
        binary = (final_mask>thresh).astype(np.uint8)*255
        # morphology
        kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,{ 'background':(9,9), 'color':(5,5) }.get(operation.get('type',''),(7,7)))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kern, iterations=2)
        if operation and operation.get('type')=='background':
            binary = 255-binary
        if self.progress_callback:
            self.progress_callback(0.9, desc="Refining mask")
        refined = self.refine_mask(binary, image_np)
        if self.progress_callback:
            self.progress_callback(1.0, desc="Mask ready")
        return {'mask': refined, 'raw_mask': binary, 'success': True}

    def _tensor_to_device(self, tensor: torch.Tensor):
        return tensor.to(self.device, dtype=torch.half if torch.cuda.is_available() else torch.float)

    def _process_sam(self, image_np):
        try:
            sam = self.model_manager.get_model('sam')
            if sam is None: return None
            masks = sam.generate(image_np)
            out = np.zeros(image_np.shape[:2],np.float32)
            for m in masks:
                seg = m['segmentation'].astype(np.float32)
                out += cv2.resize(seg,(image_np.shape[1],image_np.shape[0]))
            return out
        except Exception as e:
            logger.error(f"SAM: {e}");return None

    def _process_yolo(self, image_np, prompt=None):
        try:
            yolo = self.model_manager.get_model('yolo')
            if yolo is None: return None
            results = yolo(image_np, imgsz=960, conf=0.3)
            out = np.zeros(image_np.shape[:2],np.float32)
            for r in results:
                if r.masks is None: continue
                for mask in r.masks.masks:
                    poly = (mask.cpu().numpy()>0.5).astype(np.uint8)
                    out += poly
            return out
        except Exception as e:
            logger.error(f"YOLO: {e}");return None

    def _process_clipseg(self, image_np, text_prompt):
        try:
            clipseg = self.model_manager.get_model('clipseg')
            if clipseg is None: return None
            proc, model = clipseg['processor'], clipseg['model']
            inputs = proc(text=text_prompt or "object", images=Image.fromarray(image_np), return_tensors="pt").to(self.device)
            inputs = {k: self._tensor_to_device(v) for k,v in inputs.items()}
            with torch.no_grad():
                logits = model(**inputs).logits
            mask = torch.sigmoid(logits).cpu().numpy()[0,0]
            return cv2.resize(mask,(image_np.shape[1],image_np.shape[0]))
        except Exception as e:
            logger.error(f"CLIPSeg: {e}");return None

    def _process_mediapipe(self, image_np):
        try:
            mp = self.model_manager.get_model('mediapipe')
            if mp is None: return None
            res = mp.process(cv2.cvtColor(image_np,cv2.COLOR_BGR2RGB))
            m = res.segmentation_mask
            return cv2.resize(m.astype(np.float32),(image_np.shape[1],image_np.shape[0])) if m is not None else None
        except Exception as e:
            logger.error(f"MediaPipe: {e}");return None

    def _process_face_detection(self, image_np):
        try:
            fd = self.model_manager.get_model('face_detector')
            if fd is None: return None
            res = fd.process(cv2.cvtColor(image_np,cv2.COLOR_BGR2RGB))
            h,w=image_np.shape[:2]
            out = np.zeros((h,w),np.float32)
            for d in getattr(res,'detections',[]):
                bb = d.location_data.relative_bounding_box
                x,y=int(bb.xmin*w),int(bb.ymin*h)
                W,H=int(bb.width*w),int(bb.height*h)
                cv2.ellipse(out,(x+W//2,y+H//2),(W//2,H//2),0,0,360,1.0,-1)
            return out
        except Exception as e:
            logger.error(f"Face: {e}");return None

    def _process_background_detection(self, image_np):
        try:
            # simplified grabcut
            mask = np.zeros(image_np.shape[:2],np.uint8)
            bg,fg=np.zeros((1,65),np.float64),np.zeros((1,65),np.float64)
            H,W=image_np.shape[:2]
            rect=(int(W*0.1),int(H*0.1),int(W*0.8),int(H*0.8))
            cv2.grabCut(image_np,mask,rect,bg,fg,5,cv2.GC_INIT_WITH_RECT)
            binm=(mask==2)|(mask==0)
            return (1-binm.astype(np.float32))
        except Exception as e:
            logger.error(f"BG: {e}")
            return None

    def refine_mask(self, mask, image):
        try:
            m = (mask/255.0) if mask.max()>1 else mask
            edges=cv2.Canny(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY),50,150)/255.0
            fused=m*(1-edges*0.3)+edges*0.7
            u=(fused*255).astype(np.uint8)
            clean=cv2.morphologyEx(u,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)))
            return clean
        except Exception as e:
            logger.error(f"Refine: {e}");return mask
