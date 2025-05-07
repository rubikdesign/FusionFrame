#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Pipeline pentru eliminarea obiectelor/persoanelor în FusionFrame 2.0
(Versiune corectată și simplificată)
"""

import logging
import cv2
import numpy as np
from typing import Dict, Any, Optional, Union, Tuple, Callable
from PIL import Image

# Importuri necesare din proiect
try:
    from processing.pipelines.base_pipeline import BasePipeline
    from processing.analyzer import OperationAnalyzer
    # Importăm ModelConfig dacă avem nevoie de valori default (ex: pentru negative prompt)
    from config.model_config import ModelConfig
except ImportError as e:
    logging.basicConfig(level=logging.ERROR)
    logging.critical(f"ERROR: Failed to import modules in removal_pipeline.py: {e}")
    import sys
    sys.exit(f"Critical import error in removal_pipeline.py: {e}")


# Setăm logger-ul
logger = logging.getLogger(__name__)
if not logger.hasHandlers() or not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
    _ch = logging.StreamHandler()
    _f = logging.Formatter("%(asctime)s-%(name)s-%(levelname)s-%(message)s")
    _ch.setFormatter(_f)
    logger.addHandler(_ch)
    if logger.level == logging.NOTSET: logger.setLevel(logging.INFO)


class RemovalPipeline(BasePipeline):
    """
    Pipeline specializat pentru eliminarea obiectelor sau persoanelor.
    Implementează algoritmi avansați pentru eliminarea completă a
    țintei din imagine cu reconstrucția fundalului (inpainting).
    """

    def __init__(self):
        """Inițializează pipeline-ul pentru eliminare"""
        super().__init__()
        self.operation_analyzer = OperationAnalyzer()

    def process(self,
               image: Union[Image.Image, np.ndarray],
               prompt: str,
               strength: float = 0.85, # Valoare default mai mare pentru remove
               progress_callback: Optional[Callable[[float, Optional[str]], None]] = None,
               **kwargs) -> Dict[str, Any]:
        """
        Procesează imaginea pentru a elimina un obiect sau o persoană specificat(ă) în prompt.

        Args:
            image: Imaginea de procesat (PIL.Image sau np.ndarray).
            prompt: Promptul care descrie ce trebuie eliminat (ex: "remove the red car", "erase the person").
            strength: Intensitatea procesului de inpainting (0.0-1.0). O valoare mai mare e recomandată pentru remove.
            progress_callback: Funcție opțională pentru raportarea progresului.
            **kwargs: Argumente adiționale (ex: num_inference_steps, guidance_scale, seed, use_controlnet).

        Returns:
            Dicționar cu rezultatele procesării ('result', 'mask', 'operation', 'message').
        """
        self.progress_callback = progress_callback

        # 1. Analizăm operația din prompt pentru a identifica ținta
        operation = self.operation_analyzer.analyze_operation(prompt)
        # Extragem ținta; folosim un text generic dacă nu e identificată clar
        target_description = operation.get('target_object') or operation.get('target') or "object to remove"
        logger.info(f"Removal target identified as: '{target_description}'")

        # 2. Apelăm metoda unică de eliminare/inpainting
        # Preluăm parametrii din kwargs pentru a-i pasa mai departe
        # (ex: num_inference_steps, guidance_scale etc. pot fi în kwargs)
        return self._perform_removal(image=image,
                                     operation_details=operation, # Transmitem toate detaliile operației
                                     target_prompt_for_mask=target_description, # Ce să căutăm în imagine
                                     strength=strength,
                                     **kwargs) # Transmitem restul argumentelor

    # Am redenumit metoda pentru claritate semantică
    def _perform_removal(self,
                         image: Union[Image.Image, np.ndarray],
                         operation_details: Dict[str, Any],
                         target_prompt_for_mask: str,
                         strength: float = 0.85,
                         **kwargs) -> Dict[str, Any]:
        """
        Metoda internă care realizează eliminarea țintei specificate prin inpainting.

        Args:
            image: Imaginea de procesat.
            operation_details: Dicționarul cu detaliile operației (din OperationAnalyzer).
            target_prompt_for_mask: Textul folosit pentru a genera masca (ce trebuie eliminat).
            strength: Intensitatea procesului de inpainting.
            **kwargs: Argumente adiționale pentru procesarea modelului (steps, cfg, seed etc.).

        Returns:
            Dicționar cu rezultatele procesării.
        """
        original_input_pil: Optional[Image.Image] = None
        mask_pil_image: Optional[Image.Image] = None # Inițializăm masca PIL

        # a. Conversie și validare input imagine
        try:
            if isinstance(image, Image.Image):
                pil_image = image.convert("RGB")
                image_np = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            elif isinstance(image, np.ndarray):
                if image.ndim == 2:
                    image_np = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                    pil_image = Image.fromarray(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))
                elif image.shape[2] == 4:
                    image_np = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
                    pil_image = Image.fromarray(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))
                elif image.shape[2] == 3:
                    image_np = image # Presupunem BGR pentru MaskGenerator
                    pil_image = Image.fromarray(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)) # RGB pentru model
                else: raise ValueError(f"Unsupported NumPy image shape: {image.shape}")
            else: raise TypeError(f"Unsupported image type: {type(image)}")
            original_input_pil = pil_image.copy() # Salvăm copia originală PIL RGB
        except Exception as e:
             logger.error(f"Error converting input image in removal: {e}", exc_info=True)
             original_pil_if_needed = image if isinstance(image, Image.Image) else (Image.fromarray(image) if isinstance(image, np.ndarray) else None)
             return {'result': original_pil_if_needed, 'mask': None, 'operation': operation_details, 'message': f"Eroare la conversia imaginii de intrare: {e}"}

        # b. Analiză context (opțional, poate fi folosit pentru promptul de inpainting)
        self._update_progress(0.1, desc="Analiză imagine...")
        # image_context = self.image_analyzer.analyze_image_context(image_np) # image_np e BGR

        # c. Generare mască pentru ținta de eliminat
        self._update_progress(0.2, desc=f"Generare mască pentru '{target_prompt_for_mask}'...")
        mask_result = self.mask_generator.generate_mask(
            image=image_np, # BGR
            prompt=f"{target_prompt_for_mask}", # Folosim ținta extrasă
            operation=operation_details, # Transmitem detaliile complete
            progress_callback=lambda p, desc=None: self._update_progress(0.2 + p * 0.3, desc=desc)
        )

        # Verificăm masca
        if not mask_result.get('success') or mask_result.get('mask') is None:
            logger.error(f"Failed to generate mask for removal target: '{target_prompt_for_mask}'")
            return {
                'result': original_input_pil,
                'mask': None,
                'operation': operation_details,
                'message': f"Eroare la generarea măștii pentru '{target_prompt_for_mask}'"
            }
        else:
            mask_np_array = mask_result['mask']
            if not isinstance(mask_np_array, np.ndarray):
                 logger.error(f"Removal mask is not a NumPy array: {type(mask_np_array)}")
                 return {'result': original_input_pil, 'mask': None, 'operation': operation_details, 'message': "Masca generată are un format invalid."}
            # Convertim masca numpy în PIL.Image în modul 'L'
            mask_pil_image = Image.fromarray(mask_np_array).convert('L') # Asigurăm modul 'L'


        # d. Inpainting folosind modelul principal
        self._update_progress(0.6, desc=f"Eliminare/Inpainting '{target_prompt_for_mask}'...")

        main_model = self.model_manager.get_model('main')
        if main_model is None or not getattr(main_model, 'is_loaded', False):
            logger.error("Main model not available or not loaded for removal")
            return {
                'result': original_input_pil,
                'mask': mask_pil_image, # Returnăm masca generată chiar dacă modelul eșuează
                'operation': operation_details,
                'message': "Modelul principal nu este disponibil sau nu a fost încărcat."
            }

        # e. Definire prompturi și parametri pentru inpainting
        # Promptul pozitiv ar trebui să descrie ce să *genereze* în zona măștii
        # context_desc = image_context.get('description', 'realistic background') # Folosim contextul dacă e disponibil
        # enhanced_prompt = f"empty area, clean background, {context_desc}, photorealistic, high detail"
        # Sau un prompt simplu, focusat pe calitate:
        enhanced_prompt = "realistic background, clean area, high quality photorealistic inpaint, seamless"
        # Promptul negativ ar trebui să includă obiectul eliminat și artefacte nedorite
        negative_prompt = f"{target_prompt_for_mask}, object, item, text, words, letters, watermark, signature, blurry, distortion, deformed, low quality, artifacts"

        # Preluăm parametrii de generare specifici pentru 'remove'
        params = self._get_generation_params('remove')
        # Ajustăm strength-ul - pentru inpainting/remove, o valoare mare e adesea necesară
        final_strength = max(0.95, strength) # Asigurăm o valoare mare, dar permitem suprascrierea prin 'strength' dacă e > 0.95

        # Preluăm parametrii specifici din kwargs, suprascriind cele din 'params' dacă sunt prezente
        num_inference_steps = kwargs.get('num_inference_steps', params['num_inference_steps'])
        guidance_scale = kwargs.get('guidance_scale', params['guidance_scale'])
        seed = kwargs.get('seed', -1)
        aesthetic_score = kwargs.get('aesthetic_score', 6.0)
        negative_aesthetic_score = kwargs.get('negative_aesthetic_score', 2.5)
        # Suprascriem promptul negativ dacă este furnizat în kwargs
        negative_prompt = kwargs.get('negative_prompt', negative_prompt)


        # f. Pregătire ControlNet (Opțional)
        control_image_pil: Optional[Image.Image] = None
        use_controlnet_flag = kwargs.get('use_controlnet', False) # Verificăm flag-ul
        controlnet_scale = kwargs.get('controlnet_conditioning_scale', params.get('controlnet_conditioning_scale', 0.7)) # Preluăm scala

        if use_controlnet_flag and main_model.controlnet is not None:
            self._update_progress(0.65, desc="Pregătire imagine ControlNet...")
            # Folosim imaginea originală PIL (RGB) pentru a genera harta Canny
            control_image_pil = self._prepare_control_image(original_input_pil, control_mode="canny")
            if control_image_pil is None:
                logger.warning("Nu s-a putut genera imaginea de control Canny. ControlNet nu va fi folosit.")


        # g. Execuție model principal (Inpainting)
        self._update_progress(0.7, desc="Generare finală (Inpainting)...")
        try:
            model_process_args = {
                "image": original_input_pil, # Imaginea originală (nealterată)
                "mask_image": mask_pil_image, # Masca obiectului de eliminat
                "prompt": enhanced_prompt,
                "negative_prompt": negative_prompt,
                "strength": final_strength, # Folosim strength-ul ajustat pentru remove
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "seed": seed,
                "aesthetic_score": aesthetic_score,
                "negative_aesthetic_score": negative_aesthetic_score,
                # Adăugăm argumentele ControlNet condiționat
                **({"controlnet_conditioning_image": control_image_pil,
                    "controlnet_conditioning_scale": controlnet_scale}
                   if use_controlnet_flag and control_image_pil and main_model.controlnet else {})
            }

            # Apelăm modelul principal
            result_dict = main_model.process(**model_process_args)

            # Verificăm rezultatul
            if result_dict and result_dict.get('success'):
                final_result_image = result_dict.get('result')
                if final_result_image is None or not isinstance(final_result_image, Image.Image):
                     logger.warning("Modelul a raportat succes, dar nu a returnat o imagine PIL validă. Se returnează imaginea originală.")
                     final_result_image = original_input_pil

                final_message = f"{target_prompt_for_mask.capitalize()} eliminat cu succes"
                self._update_progress(1.0, desc=final_message)
                return {
                    'result': final_result_image,
                    'mask': mask_pil_image, # Returnăm masca PIL folosită
                    'operation': operation_details,
                    'message': final_message
                }
            else:
                error_message = result_dict.get('message', "Eroare necunoscută la procesarea modelului")
                logger.error(f"Error in removal processing: {error_message}")
                return {
                    'result': original_input_pil, # Returnează originalul
                    'mask': mask_pil_image,
                    'operation': operation_details,
                    'message': f"Eroare model: {error_message}"
                }

        except Exception as e:
            logger.error(f"Error during object removal execution: {str(e)}", exc_info=True)
            return {
                'result': original_input_pil, # Returnează originalul
                'mask': mask_pil_image, # Returnează masca dacă s-a generat
                'operation': operation_details,
                'message': f"Eroare execuție eliminare: {str(e)}"
            }

    # Metoda remove_object nu mai este definită separat