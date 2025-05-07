#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Pipeline general pentru procesare în FusionFrame 2.0
"""

import logging
import cv2 # Deși nu este folosit direct aici, este importat în base_pipeline
import numpy as np
from typing import Dict, Any, Union, Callable, Optional # Am adăugat Optional
from PIL import Image

from processing.pipelines.base_pipeline import BasePipeline
from processing.analyzer import OperationAnalyzer
# Este o bună practică să imporți și ModelConfig dacă te bazezi pe valorile sale default aici
from config.model_config import ModelConfig

# Setăm logger-ul
logger = logging.getLogger(__name__)

class GeneralPipeline(BasePipeline):
    """
    Pipeline general pentru procesarea imaginilor
    """
    def __init__(self):
        super().__init__()
        self.operation_analyzer = OperationAnalyzer()

    def process(self,
                image: Union[Image.Image, np.ndarray],
                prompt: str,
                strength: float = 0.75,
                progress_callback: Optional[Callable[[float, Optional[str]], None]] = None, # Am adăugat tipajul corect
                **kwargs) -> Dict[str, Any]:
        # Salvăm callback-ul de progres
        self.progress_callback = progress_callback
        # Determinăm tipul de operație din prompt
        operation = self.operation_analyzer.analyze_operation(prompt)

        # 1. Convertim inputul în np.ndarray și PIL.Image
        # Asigurăm că pil_image este RGB și image_np este BGR (dacă e cazul și necesar pentru mask_generator)
        try:
            if isinstance(image, Image.Image):
                pil_image = image.convert("RGB") # Asigurăm RGB
                image_np = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            elif isinstance(image, np.ndarray):
                if image.ndim == 2: # Grayscale
                    image_np = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                    pil_image = Image.fromarray(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))
                elif image.shape[2] == 4: # RGBA
                    image_np = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
                    pil_image = Image.fromarray(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))
                elif image.shape[2] == 3: # Presupunem BGR sau RGB
                    # Verificăm ordinea canalelor dacă e necesar sau standardizăm
                    image_np = image # Să presupunem că mask_generator se așteaptă la BGR
                    pil_image = Image.fromarray(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))
                else:
                    raise ValueError(f"Unsupported NumPy image shape: {image.shape}")
            else:
                raise TypeError(f"Unsupported image type: {type(image)}")
        except Exception as e:
            logger.error(f"Error converting input image: {e}", exc_info=True)
            # Returnăm imaginea originală sub formă de PIL dacă e posibil, altfel None
            original_pil = image if isinstance(image, Image.Image) else (Image.fromarray(image) if isinstance(image, np.ndarray) else None)
            return {
                'result': original_pil, 'mask': None, 'operation': operation,
                'message': f"Eroare la conversia imaginii de intrare: {e}"
            }

        # 2. Analiză imagine și context
        self._update_progress(0.1, desc="Analiză imagine...")
        # image_context = self.image_analyzer.analyze_image_context(image_np) # image_np ar trebui să fie BGR

        # 3. Generare mască
        self._update_progress(0.2, desc="Generare mască...")
        mask_result = self.mask_generator.generate_mask(
            image=image_np, # MaskGenerator se așteaptă la BGR din ce am văzut în implementarea sa
            prompt=prompt,
            operation=operation,
            progress_callback=lambda p, desc=None: self._update_progress(0.2 + p * 0.3, desc=desc)
        )

        mask_pil_image: Optional[Image.Image] = None
        if not mask_result.get('success') or mask_result.get('mask') is None:
            logger.warning("Failed to generate mask or mask is None. Using a full blank mask as fallback.")
            # Creează o mască goală (neagră) de dimensiunea imaginii ca fallback
            # Sau poate o mască complet albă dacă intenția e să se aplice pe toată imaginea
            # Depinde de logica dorită pentru HiDreamModel când masca e invalidă.
            # Aici presupun că o mască neagră (nicio zonă de editat) e mai sigură decât una albă.
            # Dacă HiDreamModel necesită o mască validă, ar trebui să returnezi eroare.
            fallback_mask_np = np.zeros((pil_image.height, pil_image.width), dtype=np.uint8)
            mask_pil_image = Image.fromarray(fallback_mask_np, mode='L')
            # Comentează rândul de mai jos dacă nu vrei să returnezi eroare și continui cu masca goală
            # return {'result': pil_image, 'mask': None, 'operation': operation, 'message': "Eroare la generarea măștii"}
        else:
            mask_np_array = mask_result['mask']
            if not isinstance(mask_np_array, np.ndarray):
                logger.error(f"Mask from generator is not a NumPy array: {type(mask_np_array)}")
                return {'result': pil_image, 'mask': None, 'operation': operation, 'message': "Masca generată are un format invalid."}
            mask_pil_image = Image.fromarray(mask_np_array, mode='L') # Asigură-te că masca e în modul 'L' (grayscale)

        # 4. Procesare model principal
        self._update_progress(0.6, desc="Procesare imagine...")
        main_model = self.model_manager.get_model('main')
        if main_model is None or not getattr(main_model, 'is_loaded', False): # Verifică și dacă e încărcat
            logger.error("Main model not available or not loaded.")
            return {
                'result': pil_image,
                'mask': mask_pil_image,
                'operation': operation,
                'message': "Modelul principal nu este disponibil sau nu a fost încărcat."
            }

        # 5. Prompt enhancement și setare parametri
        enhanced_prompt = self._enhance_prompt(prompt, operation)
        params = self._get_generation_params(operation.get('type', 'general'))

        # Pregătire imagine de control pentru ControlNet (dacă e activat și configurat)
        control_image_pil: Optional[Image.Image] = None
        # Din log-uri: 'use_controlnet': True este trimis în advanced_kwargs
        # Ar trebui să verifici dacă kwargs conține o opțiune pentru a activa ControlNet
        use_controlnet_flag = kwargs.get('use_controlnet', False) # Presupunem că primești acest flag
        if use_controlnet_flag and main_model.controlnet is not None: # Verificăm dacă modelul principal are ControlNet
            self._update_progress(0.65, desc="Pregătire imagine ControlNet...")
            # Folosim metoda din BasePipeline pentru a crea imaginea Canny
            # Transmitem imaginea originală PIL (RGB)
            control_image_pil = self._prepare_control_image(pil_image, control_mode="canny")
            if control_image_pil is None:
                logger.warning("Nu s-a putut genera imaginea de control Canny. ControlNet nu va fi folosit.")


        # 6. Apel metodă 'process' a modelului
        self._update_progress(0.7, desc="Generare rezultat...")
        try:
            # Preluăm parametrii specifici din kwargs dacă există (ex. din advanced_kwargs)
            num_inference_steps = kwargs.get('num_inference_steps', params['num_inference_steps'])
            guidance_scale = kwargs.get('guidance_scale', params['guidance_scale'])
            # Folosim un negativ prompt default mai generic sau cel din config
            negative_prompt = kwargs.get('negative_prompt', ModelConfig.GENERATION_PARAMS.get("negative_prompt", "distortion, deformation, artifact, blurry, low quality, worst quality, lowres"))

            # Construim dicționarul de argumente pentru model.process
            model_process_args = {
                "image": pil_image,
                "mask_image": mask_pil_image,
                "prompt": enhanced_prompt,
                "negative_prompt": negative_prompt,
                "strength": strength, # Strength-ul primit de pipeline
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "seed": kwargs.get('seed', -1), # Permite transmiterea unui seed
                "aesthetic_score": kwargs.get('aesthetic_score', 6.0), # Valori default din HiDreamModel
                "negative_aesthetic_score": kwargs.get('negative_aesthetic_score', 2.5) # Valori default din HiDreamModel
            }

            if use_controlnet_flag and control_image_pil and main_model.controlnet:
                model_process_args["controlnet_conditioning_image"] = control_image_pil
                # Scala pentru ControlNet ar trebui să vină din params sau kwargs
                model_process_args["controlnet_conditioning_scale"] = kwargs.get('controlnet_conditioning_scale', params.get('controlnet_conditioning_scale', 0.7))


            model_output_dict = main_model.process(**model_process_args)

            if model_output_dict and model_output_dict.get('success'):
                result_image = model_output_dict.get('result')
                if result_image is None or not isinstance(result_image, Image.Image): # Verificăm și tipul
                    # Poate modelul returnează None sau altceva dacă imaginea nu s-a schimbat
                    logger.warning("Modelul a raportat succes, dar nu a returnat o imagine PIL validă. Se returnează imaginea originală.")
                    result_image = pil_image # Sau gestionează eroarea mai strict
                    # raise ValueError("Modelul a raportat succes, dar nu a returnat o imagine PIL validă.")
                
                final_message = model_output_dict.get('message', "Procesare completă cu succes")
                self._update_progress(1.0, desc=final_message if final_message else "Procesare completă!")
                return {
                    'result': result_image,
                    'mask': mask_pil_image, # Returnăm masca folosită, convertită în PIL
                    'operation': operation,
                    'message': final_message
                }
            else:
                error_message = model_output_dict.get('message', "Eroare necunoscută în timpul procesării modelului")
                logger.error(f"Model processing failed: {error_message}")
                return {
                    'result': pil_image,
                    'mask': mask_pil_image,
                    'operation': operation,
                    'message': f"Eroare model: {error_message}"
                }

        except Exception as e:
            logger.error(f"Error in general processing: {e}", exc_info=True)
            return {
                'result': pil_image,
                'mask': mask_pil_image, # Chiar și în eroare, e util să vedem masca dacă s-a generat
                'operation': operation,
                'message': f"Eroare execuție pipeline: {str(e)}"
            }