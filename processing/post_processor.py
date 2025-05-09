#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Post-procesare pentru rezultate în FusionFrame 2.0
(Actualizat cu Armonizare LAB și Blending Margini)
"""

import logging
import cv2
import numpy as np
from typing import Dict, Any, Optional, Union, Tuple, Callable
from PIL import Image
import time # Adăugat pentru măsurare timp

# Presupunem că AppConfig și ModelManager sunt accesibile corect
try:
    from config.app_config import AppConfig
    from core.model_manager import ModelManager
except ImportError:
    # Fallback simplu
    logging.warning("Could not import AppConfig/ModelManager normally. Using placeholders.")
    class AppConfig: pass
    class ModelManager:
        def get_specialized_model(self, name): logger.debug(f"Placeholder: get_specialized_model({name})"); return None
        def get_model(self, name): logger.debug(f"Placeholder: get_model({name})"); return None

# Setăm logger-ul
logger = logging.getLogger(__name__)

class PostProcessor:
    """Modulul de post-procesare pentru îmbunătățirea rezultatelor."""

    def __init__(self):
        """Inițializează procesorul de post-procesare"""
        # Folosim getattr pentru siguranță, în caz că rulăm fără AppConfig real
        self.config = getattr(__import__('config.app_config', fromlist=['AppConfig']), 'AppConfig', None)
        if self.config is None: logger.warning("AppConfig could not be loaded in PostProcessor.")
        self.model_manager = ModelManager()
        self.progress_callback = None

    def process(self,
               image: Union[Image.Image, np.ndarray], # Imaginea generată
               original_image: Optional[Union[Image.Image, np.ndarray]] = None, # Originalul
               mask: Optional[Union[Image.Image, np.ndarray]] = None, # Masca folosită
               operation_type: Optional[str] = None,
               enhance_details: bool = False,
               fix_faces: bool = False,
               remove_artifacts: bool = False,
               seamless_blending: bool = True,
               color_harmonization: bool = True,
               progress_callback: Optional[Callable] = None
               ) -> Dict[str, Any]:
        """Procesează imaginea generată pentru a îmbunătăți calitatea și integrarea."""
        self.progress_callback = progress_callback
        start_time = time.time()
        steps_applied = []
        logger.info(f"Starting post-processing: enhance={enhance_details}, fix_faces={fix_faces}, remove_artifacts={remove_artifacts}, blend={seamless_blending}, harmonize={color_harmonization}")

        # --- Standardizare și Validare Input ---
        processed_pil, processed_np, original_np, mask_np_binary = None, None, None, None
        initial_image_pil = image # Păstrăm referința inițială pentru fallback

        try:
            # Procesăm imaginea de input
            processed_pil = self._convert_to_pil(image, "RGB")
            processed_np = self._convert_to_cv2(processed_pil) # Obținem BGR

            # Procesăm imaginea originală (dacă e necesară și furnizată)
            original_needed = seamless_blending or color_harmonization
            if original_needed and original_image is not None:
                original_pil = self._convert_to_pil(original_image, "RGB")
                original_np = self._convert_to_cv2(original_pil)
                if original_np.shape[:2] != processed_np.shape[:2]:
                     logger.warning(f"Resizing original image for post-proc.")
                     original_np = cv2.resize(original_np, (processed_np.shape[1], processed_np.shape[0]), interpolation=cv2.INTER_LANCZOS4)
            elif original_needed:
                logger.warning("Original image needed but not provided. Disabling Blending/Harmonization.")
                seamless_blending = False; color_harmonization = False

            # Procesăm masca (dacă e necesară și furnizată)
            mask_needed = seamless_blending or color_harmonization or remove_artifacts or enhance_details
            if mask_needed and mask is not None:
                mask_pil_l = self._ensure_pil_mask(mask) # Obținem PIL 'L'
                if mask_pil_l is not None:
                    mask_np = np.array(mask_pil_l)
                    if mask_np.shape[:2] != processed_np.shape[:2]:
                         logger.warning(f"Resizing mask for post-proc.")
                         mask_np = cv2.resize(mask_np, (processed_np.shape[1], processed_np.shape[0]), interpolation=cv2.INTER_NEAREST)
                    # Binarizăm corect
                    _, mask_np_binary = cv2.threshold(mask_np, 127, 255, cv2.THRESH_BINARY)
                    # Verificăm dacă masca e utilă
                    if np.all(mask_np_binary == 0) or np.all(mask_np_binary == 255):
                         logger.warning("Mask is fully black or white, ignoring for mask-dependent steps.")
                         mask_np_binary = None # Tratăm ca și cum nu ar exista
                else: mask_np_binary = None # Eroare la conversie mască

                # Dezactivăm pașii dependenți dacă masca e invalidă
                if mask_np_binary is None and (seamless_blending or color_harmonization):
                     logger.warning("Mask invalid, disabling Blending/Harmonization.")
                     seamless_blending = False; color_harmonization = False

            elif not mask and (seamless_blending or color_harmonization):
                logger.warning("Mask required but not provided. Disabling Blending/Harmonization.")
                seamless_blending = False; color_harmonization = False

        except Exception as e:
             msg = f"Error preparing inputs for post-processing: {e}"; logger.error(msg, exc_info=True)
             return {'result_image': initial_image_pil, 'success': False, 'message': msg}


        # --- Aplicare Pași Post-Procesare ---
        current_image_np = processed_np.copy()
        self._update_progress(0.05, desc="Starting post-processing...")

        # Ordinea poate conta. Sugestie: Artefacte -> Detalii/Fețe -> Armonizare -> Blending
        if remove_artifacts:
            step_start_time = time.time()
            self._update_progress(0.1, desc="Removing artifacts...")
            # Aplicăm doar în zona măștii dacă există și e validă
            current_image_np = self._remove_artifacts_cv(current_image_np, mask=mask_np_binary)
            logger.debug(f"Artifact removal step took {time.time() - step_start_time:.2f}s")
            steps_applied.append("Artifact Removal")

        if enhance_details:
            step_start_time = time.time()
            self._update_progress(0.25, desc="Enhancing details...")
            # Aplicăm doar în zona măștii dacă există și e validă
            current_image_np = self._enhance_details_step(current_image_np, original_np=original_np, mask=mask_np_binary)
            logger.debug(f"Detail enhancement step took {time.time() - step_start_time:.2f}s")
            steps_applied.append("Detail Enhancement")

        if fix_faces and operation_type != 'remove':
            step_start_time = time.time()
            self._update_progress(0.5, desc="Fixing faces...")
            current_image_np = self._fix_faces_step(current_image_np) # Acționează pe toată imaginea
            logger.debug(f"Face fixing step took {time.time() - step_start_time:.2f}s")
            steps_applied.append("Face Correction")

        if color_harmonization and original_np is not None and mask_np_binary is not None:
            step_start_time = time.time()
            self._update_progress(0.7, desc="Harmonizing colors...")
            current_image_np = self._harmonize_color_lab(current_image_np, original_np, mask_np_binary)
            logger.debug(f"Color harmonization step took {time.time() - step_start_time:.2f}s")
            steps_applied.append("Color Harmonization")

        if seamless_blending and original_np is not None and mask_np_binary is not None:
            step_start_time = time.time()
            self._update_progress(0.85, desc="Blending edges...")
            current_image_np = self._alpha_blend_edges(current_image_np, original_np, mask_np_binary)
            logger.debug(f"Edge blending step took {time.time() - step_start_time:.2f}s")
            steps_applied.append("Seamless Blending")

        # --- Conversie Finală și Returnare ---
        try:
             result_pil = self._convert_cv2_to_pil(current_image_np) # Folosim helper
        except Exception as e_conv:
             msg = f"Error converting final image to PIL: {e_conv}"; logger.error(msg, exc_info=True)
             return {'result_image': processed_pil, 'success': False, 'message': msg} # Fallback la imaginea de dinainte de PP

        total_time = time.time() - start_time
        final_message = f"Post-processing complete ({total_time:.2f}s). Steps: {', '.join(steps_applied) if steps_applied else 'None'}."
        logger.info(final_message)
        self._update_progress(1.0, desc="Post-processing complete!")

        return {'result_image': result_pil, 'success': True, 'message': final_message}

    # --- Helper Methods (Private) ---
    # (Metodele _update_progress, _remove_artifacts_cv, _enhance_details_step,
    # _fix_faces_step, _harmonize_color_lab, _alpha_blend_edges, _seamless_blend_poisson
    # rămân la fel ca în versiunea anterioară)
    def _update_progress(self, progress: float, desc: str = None):
        """Actualizează callback-ul de progres."""
        if self.progress_callback is not None:
            try: self.progress_callback(progress, desc=desc)
            except: self.progress_callback = None

    def _remove_artifacts_cv(self, image_np: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Aplică smoothing (OpenCV)."""
        logger.debug("Applying bilateral filter.")
        try:
            d=5; sigmaColor=30; sigmaSpace=30
            smoothed = cv2.bilateralFilter(image_np, d, sigmaColor, sigmaSpace)
            if mask is not None and mask.shape[:2] == image_np.shape[:2]:
                 return np.where(mask[:, :, np.newaxis] > 0, smoothed, image_np).astype(np.uint8)
            else: return smoothed
        except Exception as e: logger.error(f"Artifact removal error: {e}"); return image_np

    def _enhance_details_step(self, image_np: np.ndarray, original_np: Optional[np.ndarray] = None, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Îmbunătățește detaliile (ESRGAN/Unsharp Masking)."""
        logger.debug("Attempting detail enhancement...")
        enhanced_result = image_np # Fallback
        model_applied = None
        try: # ESRGAN
            esrgan = self.model_manager.get_specialized_model('esrgan')
            if esrgan and hasattr(esrgan, 'process'):
                 logger.info("Using ESRGAN."); res_dict = esrgan.process(image_np)
                 if res_dict.get('success'):
                     enhanced = res_dict.get('result')
                     if enhanced.shape[:2] != image_np.shape[:2]: enhanced = cv2.resize(enhanced, (image_np.shape[1], image_np.shape[0]), interpolation=cv2.INTER_LANCZOS4)
                     enhanced_result = enhanced; model_applied="ESRGAN"
                 else: logger.warning(f"ESRGAN failed: {res_dict.get('message')}")
        except Exception as e: logger.error(f"ESRGAN error: {e}", exc_info=True)

        if model_applied is None: # Fallback USM
            try:
                logger.info("Applying Unsharp Masking."); gaussian = cv2.GaussianBlur(image_np, (0,0), sigmaX=1.5)
                unsharp = cv2.addWeighted(image_np, 1.5, gaussian, -0.5, 0); enhanced_result = unsharp; model_applied="USM"
            except Exception as e_usm: logger.error(f"Unsharp mask error: {e_usm}", exc_info=True)

        if model_applied: logger.debug(f"Detail enhancement applied using {model_applied}.")
        # Aplicăm selectiv pe mască
        if mask is not None and mask.shape[:2] == image_np.shape[:2]:
             return np.where(mask[:, :, np.newaxis] > 0, enhanced_result, image_np).astype(np.uint8)
        else: return enhanced_result.astype(np.uint8)



    def _convert_cv2_to_pil(self, image_np: np.ndarray) -> Image.Image:
        """Converteste o imagine NumPy BGR in PIL Image RGB."""
        if image_np.ndim == 2:  # Grayscale
            return Image.fromarray(image_np)
        elif image_np.shape[2] == 3:  # BGR
            rgb_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
            return Image.fromarray(rgb_np)
        elif image_np.shape[2] == 4:  # BGRA
            rgba_np = cv2.cvtColor(image_np, cv2.COLOR_BGRA2RGBA)
            return Image.fromarray(rgba_np)
        else:
            raise ValueError(f"Unsupported NumPy shape for PIL conversion: {image_np.shape}")

    def _fix_faces_step(self, image_np: np.ndarray) -> np.ndarray:
        """Corectează fețele (GPEN/CodeFormer)."""
        logger.debug("Attempting face correction...")
        corrected_image = image_np; model_used = None
        try: # GPEN
            gpen = self.model_manager.get_specialized_model('gpen')
            if gpen and hasattr(gpen, 'process'):
                 logger.info("Using GPEN."); res_dict = gpen.process(image_np)
                 if res_dict.get('success'): corrected_image = res_dict.get('result'); model_used = "GPEN"
                 else: logger.warning(f"GPEN failed: {res_dict.get('message')}")
        except Exception as e_gpen: logger.error(f"GPEN error: {e_gpen}", exc_info=True)

        if model_used is None: # CodeFormer Fallback
             try:
                 cf = self.model_manager.get_specialized_model('codeformer')
                 if cf and hasattr(cf, 'process'):
                      logger.info("Using CodeFormer."); res_dict = cf.process(image_np)
                      if res_dict.get('success'): corrected_image = res_dict.get('result'); model_used = "CodeFormer"
                      else: logger.warning(f"CodeFormer failed: {res_dict.get('message')}")
             except Exception as e_cf: logger.error(f"CodeFormer error: {e_cf}", exc_info=True)

        if model_used: logger.info(f"Face correction applied via {model_used}.")
        else: logger.info("No face correction applied.")
        return corrected_image.astype(np.uint8)

    def _harmonize_color_lab(self, edited_np: np.ndarray, original_np: np.ndarray, mask_np: np.ndarray) -> np.ndarray:
        """Armonizează culoarea (LAB Transfer)."""
        logger.debug("Applying LAB color harmonization.")
        if edited_np.shape != original_np.shape or mask_np.ndim != 2 or mask_np.shape[:2] != edited_np.shape[:2]:
             logger.warning("Shape/mask mismatch for harmonization. Skipping."); return edited_np
        if np.all(mask_np == 0): return edited_np # Skip dacă masca e goală
        try:
            edited_lab = cv2.cvtColor(edited_np, cv2.COLOR_BGR2LAB).astype(np.float32)
            original_lab = cv2.cvtColor(original_np, cv2.COLOR_BGR2LAB).astype(np.float32)
            mask_inv = cv2.bitwise_not(mask_np)
            harmonized_lab = edited_lab.copy()
            for i in range(3):
                mean_orig, std_orig = cv2.meanStdDev(original_lab[:, :, i], mask=mask_inv)
                mean_edit, std_edit = cv2.meanStdDev(edited_lab[:, :, i], mask=mask_np)
                mean_orig, std_orig, mean_edit, std_edit = mean_orig.item(), std_orig.item(), mean_edit.item(), std_edit.item()
                if std_edit > 1e-5:
                    channel_data = harmonized_lab[:, :, i]
                    channel_in_mask = channel_data[mask_np > 0]
                    harmonized_values = (channel_in_mask - mean_edit) * (std_orig / std_edit) + mean_orig
                    harmonized_lab[:, :, i][mask_np > 0] = harmonized_values
                else: harmonized_lab[:, :, i][mask_np > 0] = mean_orig # Set to target mean if no variation
            harmonized_lab = np.clip(harmonized_lab, 0, 255)
            harmonized_bgr = cv2.cvtColor(harmonized_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
            final_result = np.where(mask_np[:, :, np.newaxis] > 0, harmonized_bgr, edited_np).astype(np.uint8)
            return final_result
        except Exception as e: logger.error(f"LAB harmonization error: {e}", exc_info=True); return edited_np

    def _alpha_blend_edges(self, edited_np: np.ndarray, original_np: np.ndarray, mask_np: np.ndarray) -> np.ndarray:
         """Netezește marginile prin alpha blending."""
         logger.debug("Applying alpha blending at mask edges.")
         if edited_np.shape != original_np.shape or mask_np.ndim != 2 or mask_np.shape[:2] != edited_np.shape[:2]:
              logger.warning("Shape/mask mismatch for alpha blending. Skipping."); return edited_np
         if np.all(mask_np == 0) or np.all(mask_np == 255): return edited_np # Skip dacă masca e goală sau plină
         try:
              kernel_size = 25; kernel_size += 1 if kernel_size % 2 == 0 else 0
              alpha_mask_float = cv2.GaussianBlur(mask_np.astype(np.float32), (kernel_size, kernel_size), 0) / 255.0
              alpha_mask_3ch = np.repeat(alpha_mask_float[:, :, np.newaxis], 3, axis=2)
              blended = (edited_np.astype(np.float32) * alpha_mask_3ch + original_np.astype(np.float32) * (1.0 - alpha_mask_3ch))
              return np.clip(blended, 0, 255).astype(np.uint8)
         except Exception as e: logger.error(f"Alpha edge blending error: {e}", exc_info=True); return edited_np

    def _seamless_blend_poisson(self, edited_np: np.ndarray, original_np: np.ndarray, mask_np: np.ndarray) -> np.ndarray:
        """(Alternativă) Netezește marginile folosind Poisson Blending."""
        logger.debug("Applying Poisson seamless blending.")
        if edited_np.shape!= original_np.shape or mask_np.ndim != 2 or mask_np.shape[:2] != edited_np.shape[:2]: return edited_np
        if mask_np.max() == 0: return edited_np
        try:
            contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours: return edited_np
            largest_contour = max(contours, key=cv2.contourArea); M = cv2.moments(largest_contour)
            if M["m00"] == 0: x, y, w, h = cv2.boundingRect(largest_contour); center = (int(x + w / 2), int(y + h / 2))
            else: center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            h_img, w_img = edited_np.shape[:2]
            if not (0 <= center[0] < w_img and 0 <= center[1] < h_img): center = (w_img // 2, h_img // 2)
            # Clonăm zona editată (sursă) PESTE original (destinație)
            blended = cv2.seamlessClone(edited_np, original_np, mask_np, center, cv2.NORMAL_CLONE)
            return blended
        except Exception as e: logger.error(f"Poisson blending error: {e}", exc_info=True); return edited_np

    # --- Helpers de Conversie ---
    def _convert_to_pil(self, image: Union[Image.Image, np.ndarray], mode: str = "RGB") -> Image.Image:
        """Converteste inputul in PIL Image."""
        if isinstance(image, Image.Image): return image.convert(mode) if image.mode != mode else image
        elif isinstance(image, np.ndarray):
            if image.ndim == 2: # Grayscale
                 if mode == "L": return Image.fromarray(image)
                 elif mode == "RGB": return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_GRAY2RGB))
                 else: raise ValueError(f"Cannot convert grayscale NumPy to PIL mode {mode}")
            elif image.shape[2] == 4: # RGBA/BGRA
                 # Asumăm BGRA de la OpenCV sau RGBA direct
                 try: img_rgba = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA) # Try BGRA first
                 except: img_rgba = image # Assume RGBA
                 pil_img = Image.fromarray(img_rgba, 'RGBA')
                 return pil_img.convert(mode) if mode != "RGBA" else pil_img
            elif image.shape[2] == 3: # BGR/RGB
                 try: img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Try BGR first
                 except: img_rgb = image # Assume RGB
                 pil_img = Image.fromarray(img_rgb, 'RGB')
                 return pil_img.convert(mode) if mode != "RGB" else pil_img
            else: raise ValueError(f"Unsupported NumPy shape for PIL: {image.shape}")
        else: raise TypeError(f"Unsupported type for PIL: {type(image)}")

    def _convert_to_cv2(self, image: Union[Image.Image, np.ndarray]) -> np.ndarray:
        """Converteste inputul in NumPy array BGR."""
        if isinstance(image, np.ndarray): # Optimizare: dacă e deja BGR NumPy, returnăm direct
            if image.ndim == 3 and image.shape[2] == 3: return image
            elif image.ndim == 2: return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            elif image.shape[2] == 4: return cv2.cvtColor(image, cv2.COLOR_RGBA2BGR) # Asumăm RGBA
            else: raise ValueError(f"Unsupported NumPy shape for CV2 BGR: {image.shape}")
        elif isinstance(image, Image.Image):
            mode = image.mode
            if mode == 'L': return cv2.cvtColor(np.array(image), cv2.COLOR_GRAY2BGR)
            elif mode == 'RGBA': return cv2.cvtColor(np.array(image.convert('RGBA')), cv2.COLOR_RGBA2BGR)
            elif mode == 'RGB': return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            else: return cv2.cvtColor(np.array(image.convert('RGB')), cv2.COLOR_RGB2BGR)
        else: raise TypeError(f"Unsupported type for CV2: {type(image)}")

    def _ensure_pil_mask(self, mask: Optional[Union[Image.Image, np.ndarray]]) -> Optional[Image.Image]:
        """Asigură că masca este PIL Image în mod 'L'."""
        if mask is None: return None
        try:
            if isinstance(mask, Image.Image): return mask.convert("L") if mask.mode != "L" else mask
            elif isinstance(mask, np.ndarray):
                 if mask.ndim == 3 and mask.shape[2] == 1: mask = mask.squeeze(axis=2)
                 if mask.ndim != 2: raise ValueError("Mask NumPy array must be 2D")
                 if mask.dtype != np.uint8:
                      if np.max(mask) <= 1.0: mask = (mask * 255).astype(np.uint8)
                      else: mask = np.clip(mask, 0, 255).astype(np.uint8)
                 return Image.fromarray(mask, 'L')
            else: raise TypeError(f"Unsupported mask type: {type(mask)}")
        except Exception as e: logger.error(f"Failed to ensure PIL mask: {e}"); return None