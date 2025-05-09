#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script simplu pentru debugging procesare imagine
"""

import os
import sys
import logging
import time
from PIL import Image
import traceback

# Configurare logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger(__name__)

# Adaugă calea proiectului la sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

def main():
    """Funcție principală pentru debugging"""
    try:
        from config.app_config import AppConfig
        from core.model_manager import ModelManager
        from core.pipeline_manager import PipelineManager
        from processing.analyzer import OperationAnalyzer, ImageAnalyzer
        
        # Asigură-te că directoarele există
        AppConfig.ensure_dirs()
        AppConfig.LOW_VRAM_MODE = True
        
        logger.info("Inițializare componente pentru test...")
        model_manager = ModelManager()
        pipeline_manager = PipelineManager()
        op_analyzer = OperationAnalyzer()
        img_analyzer = ImageAnalyzer()
        
        # Verifică modelul
        main_model = model_manager.get_model('main')
        if not main_model or not getattr(main_model, 'is_loaded', False):
            logger.error("Modelul principal nu este încărcat!")
            return
        
        logger.info("Modelul principal este încărcat. Continuăm cu testul...")
        
        # Crează o imagine test sau folosește una existentă
        test_image_path = os.path.join(AppConfig.BASE_DIR, "test_image.jpg")
        if not os.path.exists(test_image_path):
            logger.info(f"Nu s-a găsit imagine test la {test_image_path}. Te rog încarcă o imagine pentru test.")
            return
        
        image = Image.open(test_image_path)
        # Redimensionare la maxim 512x512 pentru a economisi memorie
        if max(image.width, image.height) > 512:
            ratio = 512 / max(image.width, image.height)
            new_size = (int(image.width * ratio), int(image.height * ratio))
            image = image.resize(new_size, Image.LANCZOS)
            logger.info(f"Imagine redimensionată la {new_size}")
        
        # Test de analiză
        logger.info("Analizăm imaginea...")
        image_context = img_analyzer.analyze_image_context(image)
        logger.info("Analiza imaginii completă.")
        
        # Test de prompt
        test_prompt = "change hair color to bright pink"
        logger.info(f"Analizăm prompt-ul: '{test_prompt}'")
        operation_details = op_analyzer.analyze_operation(test_prompt)
        logger.info(f"Operație detectată: {operation_details.get('type', 'unknown')}")
        
        # Obține pipeline-ul
        op_type = operation_details.get('type', 'general')
        target_obj = operation_details.get('target_object', '')
        pipeline = pipeline_manager.get_pipeline_for_operation(op_type, target_obj)
        
        if not pipeline:
            logger.error(f"Nu s-a găsit pipeline pentru operația '{op_type}'")
            return
        
        logger.info(f"Pipeline găsit: {pipeline.__class__.__name__}")
        
        # Rulează procesarea
        logger.info("Începem procesarea imaginii...")
        start_time = time.time()
        
        pipeline_kwargs = {
            "image": image,
            "prompt": test_prompt,
            "strength": 0.7,
            "operation": operation_details,
            "image_context": image_context,
            "num_inference_steps": 8,  # Redus pentru economie de memorie
            "guidance_scale": 7.5,
            "use_controlnet": True,
            "use_refiner": False,  # Dezactivat pentru economie de memorie
        }
        
        try:
            pipeline_result = pipeline.process(**pipeline_kwargs)
            logger.info("Procesare completă!")
            
            if isinstance(pipeline_result, dict):
                if pipeline_result.get('success', False):
                    result_img = pipeline_result.get('result_image')
                    if result_img:
                        # Salvează rezultatul
                        output_path = os.path.join(AppConfig.OUTPUT_DIR, "test_output.jpg")
                        result_img.save(output_path)
                        logger.info(f"Rezultat salvat la: {output_path}")
                    else:
                        logger.error("Rezultatul nu conține imagine")
                else:
                    logger.error(f"Procesare eșuată: {pipeline_result.get('message', 'Motiv necunoscut')}")
            else:
                logger.error(f"Rezultat neașteptat: {type(pipeline_result)}")
            
            proc_time = time.time() - start_time
            logger.info(f"Timp total de procesare: {proc_time:.2f} secunde")
            
        except Exception as e:
            logger.error(f"Eroare în timpul procesării: {e}", exc_info=True)
    
    except Exception as e:
        logger.error(f"Eroare generală: {e}", exc_info=True)

if __name__ == "__main__":
    main()