#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Pipeline pentru adăugarea obiectelor în FusionFrame 2.0
(Versiune corectată pentru TypeError)
"""

import logging
import cv2
import numpy as np
from typing import Dict, Any, Optional, Union, Callable # Adăugat Optional
from PIL import Image

# Importuri necesare din proiect
try:
    from processing.pipelines.base_pipeline import BasePipeline
    from processing.analyzer import OperationAnalyzer
    # Importăm ModelConfig dacă avem nevoie de valori default
    from config.model_config import ModelConfig
except ImportError as e:
    logging.basicConfig(level=logging.ERROR)
    logging.critical(f"ERROR: Failed to import modules in add_object_pipeline.py: {e}")
    import sys
    sys.exit(f"Critical import error in add_object_pipeline.py: {e}")


# Setăm logger-ul
logger = logging.getLogger(__name__)
if not logger.hasHandlers() or not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
    _ch = logging.StreamHandler()
    _f = logging.Formatter("%(asctime)s-%(name)s-%(levelname)s-%(message)s")
    _ch.setFormatter(_f)
    logger.addHandler(_ch)
    if logger.level == logging.NOTSET: logger.setLevel(logging.INFO)


class AddObjectPipeline(BasePipeline):
    """
    Pipeline specializat pentru adăugarea obiectelor.
    """
    def __init__(self):
        """Inițializează pipeline-ul pentru adăugare."""
        super().__init__()
        self.operation_analyzer = OperationAnalyzer()

    def process(self,
                image: Union[Image.Image, np.ndarray],
                prompt: str,
                strength: float = 0.75,
                progress_callback: Optional[Callable[[float, Optional[str]], None]] = None, # Tipar corectat
                **kwargs) -> Dict[str, Any]:
        """
        Procesează imaginea pentru a adăuga un obiect specificat în prompt.

        Args:
            image: Imaginea de procesat.
            prompt: Promptul care descrie ce trebuie adăugat (ex: "add sunglasses", "put a hat on him").
            strength: Intensitatea procesului de adăugare/modificare (0.0-1.0).
            progress_callback: Funcție opțională pentru raportarea progresului.
            **kwargs: Argumente adiționale (ex: num_inference_steps, guidance_scale, seed, use_controlnet).

        Returns:
            Dicționar cu rezultatele procesării ('result', 'mask', 'operation', 'message').
        """
        self.progress_callback = progress_callback
        operation = self.operation_analyzer.analyze_operation(prompt)
        target_object = operation.get('target_object', '').lower()
        attribute = operation.get('attribute', '').lower()

        # Decidem ce metodă specifică să apelăm
        # Folosim target_object sau attribute sau chiar promptul pentru a detecta "glasses"
        if "glasses" in target_object or "ochelari" in target_object or \
           "glasses" in attribute or "ochelari" in attribute or \
           ("glasses" in prompt.lower() or "ochelari" in prompt.lower()):
            logger.info(f"Detected 'glasses' addition. Using add_glasses method.")
            # Apelăm metoda specifică, pasând doar argumentele necesare explicit
            # și restul prin kwargs dacă metoda le suportă și le folosește.
            # Definiția add_glasses acceptă **kwargs, deci le pasăm mai departe.
            return self.add_glasses(image=image,
                                    operation_details=operation,
                                    strength=strength,
                                    **kwargs) # Pasăm kwargs aici
        else:
            logger.info(f"Adding generic object: '{target_object or attribute or 'unknown'}'. Using add_generic_object method.")
            # Apelăm metoda generică
            return self.add_generic_object(image=image,
                                           operation_details=operation,
                                           strength=strength,
                                           **kwargs) # Pasăm kwargs aici

    def add_glasses(self,
                    image: Union[Image.Image, np.ndarray],
                    operation_details: Dict[str, Any], # Renumit din 'operation'
                    strength: float = 0.75,
                    **kwargs) -> Dict[str, Any]:
        """
        Adaugă ochelari pe o față detectată în imagine.

        Args:
            image: Imaginea de procesat.
            operation_details: Detaliile operației (din OperationAnalyzer).
            strength: Intensitatea adăugării.
            **kwargs: Argumente adiționale pentru modelul principal.

        Returns:
            Dicționar cu rezultatele procesării.
        """
        original_input_pil: Optional[Image.Image] = None
        mask_pil_image: Optional[Image.Image] = None

        # a. Conversie și validare input
        try:
            if isinstance(image, Image.Image):
                pil_image = image.convert("RGB")
                image_np = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            elif isinstance(image, np.ndarray):
                # Logică similară cu cea din celelalte pipeline-uri
                if image.ndim == 2: image_np = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                elif image.shape[2] == 4: image_np = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
                elif image.shape[2] == 3: image_np = image
                else: raise ValueError(f"Unsupported NumPy image shape: {image.shape}")
                pil_image = Image.fromarray(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))
            else: raise TypeError(f"Unsupported image type: {type(image)}")
            original_input_pil = pil_image.copy()
        except Exception as e:
             logger.error(f"Error converting input image in add_glasses: {e}", exc_info=True)
             original_pil_if_needed = image if isinstance(image, Image.Image) else (Image.fromarray(image) if isinstance(image, np.ndarray) else None)
             return {'result': original_pil_if_needed, 'mask': None, 'operation': operation_details, 'message': f"Eroare la conversia imaginii de intrare: {e}"}

        # b. Analiză imagine (opțional)
        # self._update_progress(0.1, desc="Analiză imagine...")
        # image_context = self.image_analyzer.analyze_image_context(image_np)

        # c. Detectare față și creare mască pentru zona ochilor/ochelarilor
        self._update_progress(0.2, desc="Detectare față și zonă ochelari...")
        face_detector = self.model_manager.get_model('face_detector')
        if face_detector is None:
            logger.error("Face detector model not available.")
            # Am putea folosi MaskGenerator ca fallback? Sau o mască predefinită?
            # Momentan returnăm eroare.
            return {'result': original_input_pil, 'mask': None, 'operation': operation_details, 'message': "Detectorul de față nu este disponibil"}

        h, w = image_np.shape[:2]
        eyes_area_mask_np = np.zeros((h, w), dtype=np.uint8) # Masca pentru zona ochelarilor
        rgb_image_for_detection = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        results = face_detector.process(rgb_image_for_detection)

        face_detected = False
        if hasattr(results, 'detections') and results.detections:
            face_detected = True
            # Considerăm doar prima față detectată pentru simplitate
            detection = results.detections[0]
            try:
                bbox = detection.location_data.relative_bounding_box
                xmin = int(bbox.xmin * w); ymin = int(bbox.ymin * h)
                width = int(bbox.width * w); height = int(bbox.height * h)
                xmax = xmin + width; ymax = ymin + height

                # Estimăm o zonă mai largă pentru ochelari bazată pe bounding box-ul feței
                # Acoperim zona de la sprâncene până sub ochi, lărgită puțin pe orizontală
                eye_region_y_start = ymin + int(height * 0.20) # Puțin mai sus de unde era înainte
                eye_region_y_end = ymin + int(height * 0.55)  # Puțin mai jos
                eye_region_x_start = xmin - int(width * 0.05) # Lărgim puțin
                eye_region_x_end = xmax + int(width * 0.05)

                # Asigurăm limitele imaginii
                eye_region_x_start = max(0, eye_region_x_start)
                eye_region_y_start = max(0, eye_region_y_start)
                eye_region_x_end = min(w, eye_region_x_end)
                eye_region_y_end = min(h, eye_region_y_end)

                # Desenăm dreptunghiul măștii
                if eye_region_x_end > eye_region_x_start and eye_region_y_end > eye_region_y_start:
                    cv2.rectangle(eyes_area_mask_np,
                                  (eye_region_x_start, eye_region_y_start),
                                  (eye_region_x_end, eye_region_y_end),
                                  255, -1)
                    logger.info(f"Face detected. Mask created for glasses area: [{eye_region_x_start},{eye_region_y_start} - {eye_region_x_end},{eye_region_y_end}]")
                else:
                    logger.warning("Calculated eye region dimensions are invalid.")
                    face_detected = False # Tratăm ca și cum nu s-a detectat

            except Exception as e_bbox:
                logger.error(f"Error processing face detection bounding box: {e_bbox}", exc_info=True)
                face_detected = False

        if not face_detected:
            logger.warning("No face detected or error during detection. Using a fallback mask for center-upper area.")
            # Fallback: o mască generică în zona unde ar putea fi ochelarii
            eyes_area_mask_np = np.zeros((h, w), dtype=np.uint8)
            center_y, region_height = h // 3, h // 6
            center_x, region_width = w // 2, w // 2
            cv2.rectangle(eyes_area_mask_np,
                          (center_x - region_width // 2, center_y - region_height // 2),
                          (center_x + region_width // 2, center_y + region_height // 2),
                          255, -1)

        mask_pil_image = Image.fromarray(eyes_area_mask_np).convert('L')


        # d. Pregătire prompt și parametri pentru modelul principal
        self._update_progress(0.5, desc="Generare ochelari...")
        main_model = self.model_manager.get_model('main')
        if main_model is None or not getattr(main_model, 'is_loaded', False):
            logger.error("Main model not available or not loaded.")
            return {'result': original_input_pil, 'mask': mask_pil_image, 'operation': operation_details, 'message': "Modelul principal nu este disponibil"}

        # Extragem tipul de ochelari din detaliile operației
        glasses_type = operation_details.get('attribute', '').strip()
        if glasses_type:
            # Construim promptul specificând tipul de ochelari
            prompt_text = f"face portrait wearing {glasses_type} glasses, realistic {glasses_type}, high quality photography"
        else:
            # Prompt generic pentru ochelari stilați
            prompt_text = "face portrait wearing stylish glasses, realistic glasses, detailed eyewear, high quality photography"

        negative_prompt = kwargs.get('negative_prompt', "no glasses, deformed face, distorted eyes, blurry, low quality, bad anatomy, extra limbs")

        # Preluăm parametrii de generare specifici pentru 'add'
        params = self._get_generation_params('add')
        # Ajustăm strength-ul - pentru adăugare poate e nevoie de o valoare medie-mică pentru a nu schimba prea mult fața
        final_strength = min(0.65, strength) # Folosim o valoare mai mică, dar permitem suprascrierea
        num_inference_steps = kwargs.get('num_inference_steps', params['num_inference_steps'])
        guidance_scale = kwargs.get('guidance_scale', params['guidance_scale'])
        seed = kwargs.get('seed', -1)
        aesthetic_score = kwargs.get('aesthetic_score', 6.0)
        negative_aesthetic_score = kwargs.get('negative_aesthetic_score', 2.5)

        # e. Pregătire ControlNet (Opțional, poate Canny pe față ajută)
        control_image_pil: Optional[Image.Image] = None
        use_controlnet_flag = kwargs.get('use_controlnet', False)
        controlnet_scale = kwargs.get('controlnet_conditioning_scale', params.get('controlnet_conditioning_scale', 0.5)) # Scală mai mică pentru add?

        if use_controlnet_flag and main_model.controlnet is not None:
            self._update_progress(0.65, desc="Pregătire imagine ControlNet...")
            control_image_pil = self._prepare_control_image(original_input_pil, control_mode="canny")
            if control_image_pil is None:
                logger.warning("Nu s-a putut genera imaginea de control Canny. ControlNet nu va fi folosit.")


        # f. Execuție model principal
        self._update_progress(0.7, desc="Aplicare ochelari...")
        try:
            model_process_args = {
                "image": original_input_pil,
                "mask_image": mask_pil_image,
                "prompt": prompt_text,
                "negative_prompt": negative_prompt,
                "strength": final_strength,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "seed": seed,
                "aesthetic_score": aesthetic_score,
                "negative_aesthetic_score": negative_aesthetic_score,
                **({"controlnet_conditioning_image": control_image_pil,
                    "controlnet_conditioning_scale": controlnet_scale}
                   if use_controlnet_flag and control_image_pil and main_model.controlnet else {})
            }

            result_dict = main_model.process(**model_process_args)

            if result_dict and result_dict.get('success'):
                final_result_image = result_dict.get('result')
                if final_result_image is None or not isinstance(final_result_image, Image.Image):
                     logger.warning("Modelul (add_glasses) a raportat succes, dar nu a returnat o imagine PIL validă.")
                     final_result_image = original_input_pil # Fallback la original

                # g. Îmbunătățire față (Opțional) - GPEN
                # Acest pas ar putea fi scos sau făcut opțional în UI
                try:
                    self._update_progress(0.9, desc="Îmbunătățire față (GPEN)...")
                    # Încercăm să obținem modelul GPEN (presupunând că e înregistrat ca 'gpen')
                    gpen_model = self.model_manager.get_model('gpen') # Sau load_specialized_model
                    if gpen_model and hasattr(gpen_model, 'process'):
                        # GPEN ar putea necesita input numpy BGR
                        result_np_bgr = cv2.cvtColor(np.array(final_result_image), cv2.COLOR_RGB2BGR)
                        enhanced_result_dict = gpen_model.process(result_np_bgr) # Verifică formatul așteptat de GPEN
                        if enhanced_result_dict.get('success') and enhanced_result_dict.get('result') is not None:
                             # Convertim înapoi la PIL RGB
                             enhanced_np = enhanced_result_dict['result']
                             if isinstance(enhanced_np, np.ndarray):
                                 final_result_image = Image.fromarray(cv2.cvtColor(enhanced_np, cv2.COLOR_BGR2RGB))
                                 logger.info("Face enhanced successfully using GPEN.")
                             else: logger.warning("GPEN result was not a NumPy array.")
                        else: logger.warning(f"GPEN face enhancement failed: {enhanced_result_dict.get('message')}")
                    else: logger.info("GPEN model not available or loaded, skipping face enhancement.")
                except Exception as e_enhance:
                    logger.error(f"Error during optional face enhancement: {e_enhance}", exc_info=True)
                    # Continuăm cu imaginea negenerată de GPEN

                final_message = result_dict.get('message', "Ochelari adăugați cu succes.")
                self._update_progress(1.0, desc="Procesare completă!")
                return {'result': final_result_image, 'mask': mask_pil_image, 'operation': operation_details, 'message': final_message}
            else:
                 error_message = result_dict.get('message', "Eroare necunoscută la adăugarea ochelarilor")
                 logger.error(f"Add glasses model processing failed: {error_message}")
                 return {'result': original_input_pil, 'mask': mask_pil_image, 'operation': operation_details, 'message': f"Eroare model: {error_message}"}

        except Exception as e:
            logger.error(f"Error in adding glasses pipeline: {e}", exc_info=True)
            return {'result': original_input_pil, 'mask': mask_pil_image, 'operation': operation_details, 'message': f"Eroare execuție adăugare ochelari: {str(e)}"}


    def add_generic_object(self,
                           image: Union[Image.Image, np.ndarray],
                           operation_details: Dict[str, Any], # Renumit din 'operation'
                           strength: float = 0.75,
                           **kwargs) -> Dict[str, Any]:
        """
        Adaugă un obiect generic într-o regiune specificată sau centrală a imaginii.

        Args:
            image: Imaginea de procesat.
            operation_details: Detaliile operației (din OperationAnalyzer).
            strength: Intensitatea adăugării.
            **kwargs: Argumente adiționale pentru modelul principal.

        Returns:
            Dicționar cu rezultatele procesării.
        """
        original_input_pil: Optional[Image.Image] = None
        mask_pil_image: Optional[Image.Image] = None

        # a. Conversie și validare input
        try:
            # (Similar cu add_glasses)
            if isinstance(image, Image.Image):
                pil_image = image.convert("RGB")
                image_np = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            elif isinstance(image, np.ndarray):
                if image.ndim == 2: image_np = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                elif image.shape[2] == 4: image_np = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
                elif image.shape[2] == 3: image_np = image
                else: raise ValueError(f"Unsupported NumPy image shape: {image.shape}")
                pil_image = Image.fromarray(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))
            else: raise TypeError(f"Unsupported image type: {type(image)}")
            original_input_pil = pil_image.copy()
        except Exception as e:
             logger.error(f"Error converting input image in add_generic_object: {e}", exc_info=True)
             original_pil_if_needed = image if isinstance(image, Image.Image) else (Image.fromarray(image) if isinstance(image, np.ndarray) else None)
             return {'result': original_pil_if_needed, 'mask': None, 'operation': operation_details, 'message': f"Eroare la conversia imaginii de intrare: {e}"}

        # b. Analiză context imagine (opțional, pentru prompt)
        self._update_progress(0.1, desc="Analiză imagine...")
        image_context = self.image_analyzer.analyze_image_context(image_np)

        # c. Creare mască pentru regiunea de adăugare
        # Folosim MaskGenerator pentru a localiza zona, dacă e posibil, altfel fallback la regiune predefinită
        self._update_progress(0.2, desc="Generare mască locație...")
        # Ce adăugăm? Extragem din operation_details
        object_to_add = operation_details.get('target_object') or operation_details.get('attribute') or "object"
        # Putem încerca să generăm o mască pentru 'contextul' unde ar trebui adăugat obiectul,
        # de exemplu, dacă promptul e "add a cat on the sofa", încercăm mască pentru "sofa".
        # Sau, dacă promptul e doar "add a hat", folosim o mască predefinită pentru cap.

        # Încercare cu MaskGenerator (dacă are sens pentru context)
        # context_target = "location for " + object_to_add # Sau o logică mai bună
        # mask_result = self.mask_generator.generate_mask(image_np, prompt=context_target, operation=operation_details, ...)
        # if mask_result['success']:
        #    mask_np_array = mask_result['mask']
        # else: # Fallback la regiuni predefinite

        # --- Fallback la regiuni predefinite (codul tău original) ---
        h, w = image_np.shape[:2]
        mask_np_array = np.zeros((h, w), dtype=np.uint8)
        obj_lower = object_to_add.lower()

        if obj_lower in ['hat', 'cap', 'palarie', 'pălărie', 'tiara', 'crown']:
            # Regiune pentru cap/pălărie
            top_margin = h // 10 # Lăsăm puțin spațiu sus
            region_height = h // 4
            cv2.rectangle(mask_np_array, (w // 4, top_margin), (3 * w // 4, top_margin + region_height), 255, -1)
            logger.info(f"Using predefined mask region for '{obj_lower}' (head area).")
        elif obj_lower in ['necklace', 'pendant', 'colier', 'lanț', 'tie', 'cravată']:
            # Regiune pentru gât/piept
            neck_y_start = h // 3
            region_height = h // 5
            cv2.rectangle(mask_np_array, (w // 3, neck_y_start), (2 * w // 3, neck_y_start + region_height), 255, -1)
            logger.info(f"Using predefined mask region for '{obj_lower}' (neck/chest area).")
        # Adaugă alte regiuni predefinite dacă e necesar (ex: mâini pentru ceas/brățară)
        else:
            # Regiune centrală generică
            center_x, center_y = w // 2, h // 2
            region_width, region_height = w // 2, h // 2 # O regiune centrală mare
            cv2.rectangle(mask_np_array,
                          (center_x - region_width // 2, center_y - region_height // 2),
                          (center_x + region_width // 2, center_y + region_height // 2),
                          255, -1)
            logger.info(f"Using predefined generic center mask region for '{obj_lower}'.")
        # --- Sfârșit Fallback ---

        mask_pil_image = Image.fromarray(mask_np_array).convert('L')

        # d. Pregătire prompt și parametri model
        self._update_progress(0.5, desc=f"Generare {object_to_add}...")
        main_model = self.model_manager.get_model('main')
        if main_model is None or not getattr(main_model, 'is_loaded', False):
             return {'result': original_input_pil, 'mask': mask_pil_image, 'operation': operation_details, 'message': "Modelul principal nu este disponibil"}

        # Construim promptul pentru adăugare
        scene_type = image_context.get('scene_type', 'scene') # Din analiza imaginii
        style = image_context.get('style', 'realistic')     # Din analiza imaginii
        # Promptul trebuie să descrie *întreaga* scenă modificată
        prompt_text = f"photo of {scene_type} with a {object_to_add} added, {style} style, high quality, detailed"
        # Sau putem încerca să descriem doar obiectul în context:
        # prompt_text = f"a realistic {object_to_add} in a {scene_type}, {style} style, high quality"

        negative_prompt = kwargs.get('negative_prompt', f"no {object_to_add}, unrealistic, deformed, distorted, blurry, bad quality, worst quality, lowres, extra objects")

        params = self._get_generation_params('add')
        final_strength = min(0.75, strength) # Strength mediu pentru adăugare
        num_inference_steps = kwargs.get('num_inference_steps', params['num_inference_steps'])
        guidance_scale = kwargs.get('guidance_scale', params['guidance_scale'])
        seed = kwargs.get('seed', -1)
        aesthetic_score = kwargs.get('aesthetic_score', 6.0)
        negative_aesthetic_score = kwargs.get('negative_aesthetic_score', 2.5)

        # e. Pregătire ControlNet (Opțional)
        control_image_pil: Optional[Image.Image] = None
        use_controlnet_flag = kwargs.get('use_controlnet', False)
        controlnet_scale = kwargs.get('controlnet_conditioning_scale', params.get('controlnet_conditioning_scale', 0.6))

        if use_controlnet_flag and main_model.controlnet is not None:
            self._update_progress(0.65, desc="Pregătire imagine ControlNet...")
            control_image_pil = self._prepare_control_image(original_input_pil, control_mode="canny")
            if control_image_pil is None:
                logger.warning("Nu s-a putut genera imaginea de control Canny. ControlNet nu va fi folosit.")


        # f. Execuție model principal
        self._update_progress(0.7, desc=f"Aplicare {object_to_add}...")
        try:
            model_process_args = {
                "image": original_input_pil,
                "mask_image": mask_pil_image,
                "prompt": prompt_text,
                "negative_prompt": negative_prompt,
                "strength": final_strength,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "seed": seed,
                "aesthetic_score": aesthetic_score,
                "negative_aesthetic_score": negative_aesthetic_score,
                 **({"controlnet_conditioning_image": control_image_pil,
                    "controlnet_conditioning_scale": controlnet_scale}
                   if use_controlnet_flag and control_image_pil and main_model.controlnet else {})
            }

            result_dict = main_model.process(**model_process_args)

            if result_dict and result_dict.get('success'):
                final_result_image = result_dict.get('result')
                if final_result_image is None or not isinstance(final_result_image, Image.Image):
                     logger.warning("Modelul (add_generic) a raportat succes, dar nu a returnat o imagine PIL validă.")
                     final_result_image = original_input_pil # Fallback

                final_message = result_dict.get('message', f"{object_to_add.capitalize()} adăugat cu succes.")
                self._update_progress(1.0, desc="Procesare completă!")
                return {'result': final_result_image, 'mask': mask_pil_image, 'operation': operation_details, 'message': final_message}
            else:
                 error_message = result_dict.get('message', f"Eroare necunoscută la adăugarea {object_to_add}")
                 logger.error(f"Add generic object model processing failed: {error_message}")
                 return {'result': original_input_pil, 'mask': mask_pil_image, 'operation': operation_details, 'message': f"Eroare model: {error_message}"}

        except Exception as e:
            logger.error(f"Error in adding generic object pipeline: {e}", exc_info=True)
            return {'result': original_input_pil, 'mask': mask_pil_image, 'operation': operation_details, 'message': f"Eroare execuție adăugare obiect: {str(e)}"}


    # --- Metoda remove_object a fost mutată în removal_pipeline.py ---
    # def remove_object(self, ...): # Eliminată de aici
    #    pass