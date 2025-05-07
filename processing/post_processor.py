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
        # Mock methods to avoid AttributeError if called
        def get_specialized_model(self, name): logger.debug(f"Placeholder: get_specialized_model({name})"); return None
        def get_model(self, name): logger.debug(f"Placeholder: get_model({name})"); return None

# Setăm logger-ul
logger = logging.getLogger(__name__)

class PostProcessor:
    """Modulul de post-procesare pentru îmbunătățirea rezultatelor."""

    def __init__(self):
        """Inițializează procesorul de post-procesare"""
        self.config = AppConfig() if 'AppConfig' in locals() and callable(AppConfig) else None # Instanțiem dacă nu e placeholder
        self.model_manager = ModelManager()
        self.progress_callback = None # Va fi setat de metoda process

    def process(self,
               image: Union[Image.Image, np.ndarray], # Imaginea generată de pipeline
               original_image: Optional[Union[Image.Image, np.ndarray]] = None, # Imaginea originală din UI
               mask: Optional[Union[Image.Image, np.ndarray]] = None, # Masca folosită de pipeline
               operation_type: Optional[str] = None, # Tipul operației
               # Flag-uri de control pentru pașii de post-procesare
               enhance_details: bool = False,
               fix_faces: bool = False,
               remove_artifacts: bool = False,
               seamless_blending: bool = True, # Default True pentru blending
               color_harmonization: bool = True, # Default True pentru armonizare
               progress_callback: Optional[Callable] = None
               ) -> Dict[str, Any]:
        """
        Procesează imaginea generată pentru a îmbunătăți calitatea și integrarea.

        Args:
            image: Imaginea de post-procesat (rezultatul pipeline-ului).
            original_image: Imaginea originală (opțional, necesar pentru blending/armonizare).
            mask: Masca folosită în pipeline (opțional, necesar pentru blending/armonizare).
            operation_type: Tipul operației efectuate (opțional, util pentru fix_faces).
            enhance_details: Activează îmbunătățirea detaliilor (ESRGAN/sharpening).
            fix_faces: Activează corectarea fețelor (GPEN/CodeFormer).
            remove_artifacts: Activează eliminarea artefactelor generale (smoothing).
            seamless_blending: Activează netezirea marginilor măștii.
            color_harmonization: Activează armonizarea culorilor.
            progress_callback: Funcție callback pentru progres (0.0-1.0).

        Returns:
            Dicționar: {'result_image': PIL.Image | None, 'success': bool, 'message': str}
        """
        self.progress_callback = progress_callback
        start_time = time.time()
        steps_applied = [] # Lista pașilor aplicați efectiv
        logger.info(f"Starting post-processing with flags: enhance={enhance_details}, fix_faces={fix_faces}, remove_artifacts={remove_artifacts}, blend={seamless_blending}, harmonize={color_harmonization}")

        # --- Standardizare și Validare Input ---
        processed_pil, processed_np, original_np, mask_np_binary = None, None, None, None
        try:
            # Procesăm imaginea de input (rezultatul pipeline-ului)
            if isinstance(image, Image.Image):
                 processed_pil = image.convert("RGB") if image.mode != "RGB" else image
                 processed_np = cv2.cvtColor(np.array(processed_pil), cv2.COLOR_RGB2BGR)
            elif isinstance(image, np.ndarray):
                 if image.ndim == 2: processed_np = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                 elif image.shape[2] == 4: processed_np = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
                 elif image.shape[2] == 3: processed_np = image # Asumăm BGR
                 else: raise ValueError(f"Unsupported NumPy input image shape: {image.shape}")
                 processed_pil = Image.fromarray(cv2.cvtColor(processed_np, cv2.COLOR_BGR2RGB)) # Creăm și PIL
            else: raise TypeError(f"Unsupported input image type: {type(image)}")

            # Procesăm imaginea originală (dacă e necesară și furnizată)
            if (seamless_blending or color_harmonization) and original_image is not None:
                if isinstance(original_image, Image.Image):
                     img_orig_pil = original_image.convert("RGB") if original_image.mode != "RGB" else original_image
                     original_np = cv2.cvtColor(np.array(img_orig_pil), cv2.COLOR_RGB2BGR)
                elif isinstance(original_image, np.ndarray):
                     if original_image.ndim==2: original_np = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
                     elif original_image.shape[2]==4: original_np = cv2.cvtColor(original_image, cv2.COLOR_RGBA2BGR)
                     elif original_image.shape[2]==3: original_np = original_image # Asumăm BGR
                     else: logger.warning("Unsupported original_image NumPy format.")
                else: logger.warning("Unsupported original_image type.")

                # Validăm și redimensionăm original_np dacă e necesar
                if original_np is not None:
                    if original_np.shape[:2] != processed_np.shape[:2]:
                         logger.warning(f"Resizing original image from {original_np.shape[:2]} to {processed_np.shape[:2]} for post-proc.")
                         original_np = cv2.resize(original_np, (processed_np.shape[1], processed_np.shape[0]), interpolation=cv2.INTER_LANCZOS4)
                else: # Dacă original_image nu a putut fi procesat, dezactivăm pașii dependenți
                     logger.warning("Original image is None after processing, disabling dependent steps.")
                     seamless_blending = False; color_harmonization = False

            # Procesăm masca (dacă e necesară și furnizată)
            if (seamless_blending or color_harmonization or remove_artifacts or enhance_details) and mask is not None:
                 if isinstance(mask, Image.Image): mask_l = mask.convert("L"); mask_np = np.array(mask_l)
                 elif isinstance(mask, np.ndarray):
                      if mask.ndim == 3 and mask.shape[2]==1: mask_np = mask.squeeze(axis=2)
                      elif mask.ndim == 2: mask_np = mask
                      else: mask_np = None; logger.warning("Unsupported mask NumPy format.")
                 else: mask_np = None; logger.warning("Unsupported mask type.")

                 # Validăm, redimensionăm și binarizăm masca
                 if mask_np is not None:
                     if mask_np.shape[:2] != processed_np.shape[:2]:
                          logger.warning(f"Resizing mask from {mask_np.shape[:2]} to {processed_np.shape[:2]}.")
                          mask_np = cv2.resize(mask_np, (processed_np.shape[1], processed_np.shape[0]), interpolation=cv2.INTER_NEAREST)
                     if mask_np.dtype != np.uint8 or mask_np.max() > 1:
                          _, mask_np_binary = cv2.threshold(mask_np, 127, 255, cv2.THRESH_BINARY)
                     else: mask_np_binary = (mask_np > 0).astype(np.uint8) * 255
                     # Verificăm dacă masca nu e complet neagră sau complet albă
                     if np.all(mask_np_binary == 0) or np.all(mask_np_binary == 255):
                          logger.warning("Mask is fully black or white, disabling mask-dependent steps.")
                          mask_np_binary = None # Dezactivăm masca
                 else: mask_np_binary = None # Dezactivăm dacă procesarea măștii a eșuat

                 # Dacă masca e invalidă/goală, dezactivăm pașii dependenți
                 if mask_np_binary is None and (seamless_blending or color_harmonization):
                      logger.warning("Mask is invalid/None, disabling Blending and Harmonization.")
                      seamless_blending = False; color_harmonization = False

            elif not mask and (seamless_blending or color_harmonization):
                # Dacă e nevoie de mască dar nu e furnizată
                logger.warning("Mask required but not provided for Blending/Harmonization. Disabling steps.")
                seamless_blending = False; color_harmonization = False

        except Exception as e:
             msg = f"Error preparing inputs for post-processing: {e}"; logger.error(msg, exc_info=True)
             return {'result_image': image, 'success': False, 'message': msg}


        # --- Aplicare Pași Post-Procesare ---
        current_image_np = processed_np.copy() # Lucrăm pe o copie
        self._update_progress(0.05, desc="Starting post-processing...")

        # 1. Remove Artifacts (Smoothing)
        if remove_artifacts:
            step_start_time = time.time()
            self._update_progress(0.1, desc="Removing artifacts...")
            current_image_np = self._remove_artifacts_cv(current_image_np, mask=mask_np_binary)
            logger.debug(f"Artifact removal step took {time.time() - step_start_time:.2f}s")
            steps_applied.append("Artifact Removal")

        # 2. Enhance Details (ESRGAN / Sharpening)
        if enhance_details:
            step_start_time = time.time()
            self._update_progress(0.25, desc="Enhancing details...")
            # Passăm masca la enhance_details pentru aplicare selectivă (dacă e implementată)
            current_image_np = self._enhance_details_step(current_image_np, original_np=original_np, mask=mask_np_binary)
            logger.debug(f"Detail enhancement step took {time.time() - step_start_time:.2f}s")
            steps_applied.append("Detail Enhancement")

        # 3. Fix Faces (GPEN / CodeFormer)
        if fix_faces and operation_type != 'remove':
            step_start_time = time.time()
            self._update_progress(0.5, desc="Fixing faces...")
            current_image_np = self._fix_faces_step(current_image_np) # Acționează pe toată imaginea
            logger.debug(f"Face fixing step took {time.time() - step_start_time:.2f}s")
            steps_applied.append("Face Correction")

        # 4. Color Harmonization (LAB Transfer)
        if color_harmonization and original_np is not None and mask_np_binary is not None:
            step_start_time = time.time()
            self._update_progress(0.7, desc="Harmonizing colors...")
            current_image_np = self._harmonize_color_lab(current_image_np, original_np, mask_np_binary)
            logger.debug(f"Color harmonization step took {time.time() - step_start_time:.2f}s")
            steps_applied.append("Color Harmonization")

        # 5. Seamless Blending (Alpha Blend Edges)
        if seamless_blending and original_np is not None and mask_np_binary is not None:
            step_start_time = time.time()
            self._update_progress(0.85, desc="Blending edges...")
            # Folosim Alpha Blending pentru a păstra rezultatul editat
            current_image_np = self._alpha_blend_edges(current_image_np, original_np, mask_np_binary)
            # Alternativ, folosim Poisson:
            # current_image_np = self._seamless_blend_poisson(current_image_np, original_np, mask_np_binary)
            logger.debug(f"Edge blending step took {time.time() - step_start_time:.2f}s")
            steps_applied.append("Seamless Blending")


        # --- Conversie Finală și Returnare ---
        try:
             result_pil = Image.fromarray(cv2.cvtColor(current_image_np, cv2.COLOR_BGR2RGB))
        except Exception as e_conv:
             msg = f"Error converting final image to PIL: {e_conv}"
             logger.error(msg, exc_info=True)
             # Fallback la imaginea de dinainte de post-procesare
             result_pil = Image.fromarray(cv2.cvtColor(processed_np, cv2.COLOR_BGR2RGB))
             return {'result_image': result_pil, 'success': False, 'message': msg}

        total_time = time.time() - start_time
        final_message = f"Post-processing complete ({total_time:.2f}s). Steps: {', '.join(steps_applied) if steps_applied else 'None'}."
        logger.info(final_message)
        self._update_progress(1.0, desc="Post-processing complete!")

        return {'result_image': result_pil, 'success': True, 'message': final_message}


    def _update_progress(self, progress: float, desc: str = None):
        """Actualizează callback-ul de progres."""
        if self.progress_callback is not None:
            try: self.progress_callback(progress, desc=desc)
            except: self.progress_callback = None

    # --- Implementări Pași Individuali ---

    def _remove_artifacts_cv(self, image_np: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Aplică smoothing pentru a reduce artefactele (OpenCV)."""
        logger.debug("Applying bilateral filter for artifact removal.")
        try:
            d=5; sigmaColor=30; sigmaSpace=30 # Parametri ajustabili
            smoothed = cv2.bilateralFilter(image_np, d, sigmaColor, sigmaSpace)
            if mask is not None and mask.shape[:2] == image_np.shape[:2]: # Verificăm și dimensiunea măștii
                 # Aplicăm doar unde masca e > 0 (255)
                 return np.where(mask[:, :, np.newaxis] > 0, smoothed, image_np).astype(np.uint8)
            else: return smoothed
        except Exception as e: logger.error(f"Artifact removal error: {e}", exc_info=True); return image_np

    def _enhance_details_step(self, image_np: np.ndarray, original_np: Optional[np.ndarray] = None, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Îmbunătățește detaliile (ESRGAN/Unsharp Masking)."""
        logger.debug("Attempting detail enhancement...")
        enhanced_result = image_np # Fallback la imaginea curentă
        # Încercăm ESRGAN
        try:
            esrgan_model = self.model_manager.get_specialized_model('esrgan')
            if esrgan_model and hasattr(esrgan_model, 'process'):
                 logger.info("Using ESRGAN for detail enhancement.")
                 result_dict = esrgan_model.process(image_np)
                 if result_dict.get('success'):
                     enhanced = result_dict.get('result')
                     if enhanced.shape[:2] != image_np.shape[:2]:
                          enhanced = cv2.resize(enhanced, (image_np.shape[1], image_np.shape[0]), interpolation=cv2.INTER_LANCZOS4)
                     enhanced_result = enhanced # Am obținut rezultatul de la ESRGAN
                     logger.info("ESRGAN applied successfully.")
                 else: logger.warning(f"ESRGAN processing failed: {result_dict.get('message')}")
            # else: logger.debug("ESRGAN model not found/loaded.") # Trecem la fallback
        except Exception as e: logger.error(f"Error during ESRGAN call: {e}", exc_info=True)

        # Dacă ESRGAN nu a funcționat sau nu e disponibil, încercăm Unsharp Masking
        if enhanced_result is image_np: # Verificăm dacă rezultatul nu a fost modificat de ESRGAN
            try:
                logger.info("Applying Unsharp Masking as fallback/alternative.")
                gaussian = cv2.GaussianBlur(image_np, (0,0), sigmaX=1.5, sigmaY=1.5)
                unsharp_image = cv2.addWeighted(image_np, 1.5, gaussian, -0.5, 0)
                enhanced_result = unsharp_image
            except Exception as e_usm: logger.error(f"Unsharp mask error: {e_usm}", exc_info=True)

        # Aplicăm rezultatul (fie de la ESRGAN, fie de la USM) doar în zona măștii, dacă există
        if mask is not None and mask.shape[:2] == image_np.shape[:2]:
             return np.where(mask[:, :, np.newaxis] > 0, enhanced_result, image_np).astype(np.uint8)
        else: # Altfel aplicăm pe toată imaginea
             return enhanced_result.astype(np.uint8)


    def _fix_faces_step(self, image_np: np.ndarray) -> np.ndarray:
        """Corectează fețele (GPEN/CodeFormer)."""
        logger.debug("Attempting face correction...")
        corrected_image = image_np # Fallback
        model_used = None
        # Încercăm GPEN
        try:
            gpen_model = self.model_manager.get_specialized_model('gpen')
            if gpen_model and hasattr(gpen_model, 'process'):
                 logger.info("Using GPEN for face correction.")
                 result_dict = gpen_model.process(image_np)
                 if result_dict.get('success'):
                     corrected_image = result_dict.get('result'); model_used = "GPEN"
                 else: logger.warning(f"GPEN failed: {result_dict.get('message')}")
            # else: logger.debug("GPEN model not loaded.")
        except Exception as e_gpen: logger.error(f"GPEN error: {e_gpen}", exc_info=True)

        # Dacă GPEN nu a funcționat, încercăm CodeFormer
        if model_used is None:
             try:
                 codeformer_model = self.model_manager.get_specialized_model('codeformer')
                 if codeformer_model and hasattr(codeformer_model, 'process'):
                      logger.info("Using CodeFormer for face correction.")
                      result_dict = codeformer_model.process(image_np)
                      if result_dict.get('success'):
                          corrected_image = result_dict.get('result'); model_used = "CodeFormer"
                      else: logger.warning(f"CodeFormer failed: {result_dict.get('message')}")
                 # else: logger.debug("CodeFormer model not loaded.")
             except Exception as e_cf: logger.error(f"CodeFormer error: {e_cf}", exc_info=True)

        if model_used: logger.info(f"Face correction applied using {model_used}.")
        else: logger.info("No face correction model applied.")
        return corrected_image.astype(np.uint8) # Asigurăm tipul corect

    def _harmonize_color_lab(self, edited_np: np.ndarray, original_np: np.ndarray, mask_np: np.ndarray) -> np.ndarray:
        """Armonizează culoarea (LAB Transfer)."""
        logger.debug("Applying LAB color harmonization.")
        if edited_np.shape != original_np.shape or mask_np.ndim != 2 or mask_np.shape[:2] != edited_np.shape[:2]:
             logger.warning("Shape mismatch or invalid mask for harmonization. Skipping.")
             return edited_np
        try:
            edited_lab = cv2.cvtColor(edited_np, cv2.COLOR_BGR2LAB).astype(np.float32)
            original_lab = cv2.cvtColor(original_np, cv2.COLOR_BGR2LAB).astype(np.float32)
            mask_inv = cv2.bitwise_not(mask_np) # Zona nemodificată din original

            harmonized_lab = edited_lab.copy()

            # Iterăm prin canalele LAB (L*, a*, b*)
            for i in range(3):
                # Calculăm statistici pentru zona originală neafectată
                mean_orig, std_orig = cv2.meanStdDev(original_lab[:, :, i], mask=mask_inv)
                # Calculăm statistici pentru zona editată
                mean_edit, std_edit = cv2.meanStdDev(edited_lab[:, :, i], mask=mask_np)

                mean_orig, std_orig = mean_orig.item(), std_orig.item()
                mean_edit, std_edit = mean_edit.item(), std_edit.item()

                if std_edit > 1e-5: # Evităm împărțirea la zero
                    # Aplicăm formula de transfer
                    channel_data = harmonized_lab[:, :, i]
                    channel_in_mask = channel_data[mask_np > 0] # Extragem doar zona măștii
                    # Modificăm valorile doar în zona măștii
                    harmonized_values = (channel_in_mask - mean_edit) * (std_orig / std_edit) + mean_orig
                    # Plasăm valorile modificate înapoi în canalul corespunzător
                    harmonized_lab[:, :, i][mask_np > 0] = harmonized_values
                else:
                     # Dacă deviația standard în zona editată e (aproape) zero, setăm la media originală
                     harmonized_lab[:, :, i][mask_np > 0] = mean_orig


            # Clip la limite valide și conversie înapoi
            harmonized_lab = np.clip(harmonized_lab, 0, 255)
            harmonized_bgr = cv2.cvtColor(harmonized_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)

            # Rezultatul final este imaginea editată originală, cu excepția zonei măștii,
            # unde folosim rezultatul armonizat.
            final_result = np.where(mask_np[:, :, np.newaxis] > 0, harmonized_bgr, edited_np).astype(np.uint8)
            return final_result

        except Exception as e:
            logger.error(f"LAB color harmonization error: {e}", exc_info=True)
            return edited_np

    def _alpha_blend_edges(self, edited_np: np.ndarray, original_np: np.ndarray, mask_np: np.ndarray) -> np.ndarray:
         """Netezește marginile prin alpha blending folosind o mască blurată."""
         logger.debug("Applying alpha blending at mask edges.")
         if edited_np.shape != original_np.shape or mask_np.ndim != 2 or mask_np.shape[:2] != edited_np.shape[:2]:
              logger.warning("Shape mismatch or invalid mask for alpha blending. Skipping.")
              return edited_np
         if np.all(mask_np == 0): return edited_np # Masca e goală
         if np.all(mask_np == 255): return edited_np # Masca acoperă tot

         try:
              # Kernel mai mare pentru blur mai puternic la margini
              kernel_size = 25 # Poate fi ajustat
              if kernel_size % 2 == 0: kernel_size += 1

              # Creăm masca alpha blurată (float32, 0.0-1.0)
              alpha_mask_float = cv2.GaussianBlur(mask_np.astype(np.float32), (kernel_size, kernel_size), 0) / 255.0
              alpha_mask_3ch = np.repeat(alpha_mask_float[:, :, np.newaxis], 3, axis=2)

              # Formula: result = edited * alpha + original * (1 - alpha)
              blended_output = (edited_np.astype(np.float32) * alpha_mask_3ch +
                                original_np.astype(np.float32) * (1.0 - alpha_mask_3ch))

              return np.clip(blended_output, 0, 255).astype(np.uint8)

         except Exception as e:
              logger.error(f"Alpha edge blending error: {e}", exc_info=True)
              return edited_np


    def _seamless_blend_poisson(self, edited_np: np.ndarray, original_np: np.ndarray, mask_np: np.ndarray) -> np.ndarray:
        """(Alternativă) Netezește marginile folosind Poisson Blending."""
        logger.debug("Applying Poisson seamless blending.")
        if edited_np.shape != original_np.shape or mask_np.ndim != 2 or mask_np.shape[:2] != edited_np.shape[:2]:
            logger.warning("Shape mismatch/invalid mask for Poisson blending. Skipping.")
            return edited_np
        if mask_np.max() == 0: return edited_np # Masca e goală

        try:
            contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours: return edited_np # Nu avem ce blenda

            largest_contour = max(contours, key=cv2.contourArea)
            moments = cv2.moments(largest_contour)
            if moments["m00"] == 0:
                 x, y, w, h = cv2.boundingRect(largest_contour)
                 center = (int(x + w / 2), int(y + h / 2))
            else:
                center = (int(moments["m10"] / moments["m00"]), int(moments["m01"] / moments["m00"]))

            h_img, w_img = edited_np.shape[:2]
            if not (0 <= center[0] < w_img and 0 <= center[1] < h_img): center = (w_img // 2, h_img // 2)

            # Aplicăm NORMAL_CLONE: clonează sursa (edited) în destinație (original) folosind masca
            # Acest lucru pierde conținutul original din afara măștii din 'original_np'.
            # Pentru a păstra originalul și a blenda doar editarea, ar trebui inversat.
            # Dar funcția clonează sursa *peste* destinație.
            # Vom returna originalul cu zona editată clonată peste el.
            blended_output = cv2.seamlessClone(edited_np, original_np, mask_np, center, cv2.NORMAL_CLONE)
            return blended_output
        except Exception as e:
            logger.error(f"Poisson blending error: {e}", exc_info=True)
            return edited_np