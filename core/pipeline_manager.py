#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Manager pentru pipeline-uri de procesare în FusionFrame 2.0
"""

import logging
import time  # <-- IMPORT ADĂUGAT
import sys   # Import sys pentru gestionarea erorilor critice
from typing import Dict, Any, Optional, Callable

# Asigurăm importurile corecte și gestionăm erorile
try:
    from processing.pipelines.base_pipeline import BasePipeline
    # Importăm pipeline-urile specifice aici pentru a le înregistra
    from processing.pipelines.general_pipeline import GeneralPipeline
    from processing.pipelines.removal_pipeline import RemovalPipeline
    from processing.pipelines.color_change_pipeline import ColorChangePipeline
    from processing.pipelines.add_object_pipeline import AddObjectPipeline
    from processing.pipelines.background_pipeline import BackgroundPipeline
    # Importăm OperationAnalyzer aici dacă este folosit în process_image
    from processing.analyzer import OperationAnalyzer 
except ImportError as e:
     logging.basicConfig(level=logging.ERROR) # Configurare minimă pt a vedea eroarea
     logging.critical(f"ERROR: Failed to import pipeline modules in pipeline_manager.py: {e}")
     sys.exit(f"Critical import error in pipeline_manager.py: {e}")


# Setăm logger-ul
logger = logging.getLogger(__name__)
if not logger.hasHandlers(): # Configurare de bază dacă nu e deja făcută
    _ch = logging.StreamHandler(); _f = logging.Formatter("%(asctime)s-%(name)s-%(levelname)s-%(message)s"); _ch.setFormatter(_f)
    logger.addHandler(_ch); 
    if logger.level == logging.NOTSET: logger.setLevel(logging.INFO)

class PipelineManager:
    """
    Manager pentru pipeline-uri de procesare. Gestionează și coordonează 
    pipeline-urile pentru diverse operații de editare. Implementează Singleton.
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(PipelineManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self.pipeline_types: Dict[str, type[BasePipeline]] = {} # Tipare corectă: dicționar de tipuri
        # Registrăm clasele de pipeline
        self.register_pipeline("general_pipeline", GeneralPipeline)
        self.register_pipeline("removal_pipeline", RemovalPipeline)
        self.register_pipeline("color_change_pipeline", ColorChangePipeline)
        self.register_pipeline("add_object_pipeline", AddObjectPipeline)
        self.register_pipeline("background_pipeline", BackgroundPipeline)

        # Mapare operație -> cheie pipeline
        self.operation_to_pipeline = {
            # Cheia este tipul operației (lowercase)
            "remove":   {"default": "removal_pipeline"}, # Folosim default dacă nu e specificat target
            "color":    {"default": "color_change_pipeline"},
            "add":      {"default": "add_object_pipeline"},
            "replace":  {"background": "background_pipeline", "default": "general_pipeline"}, # Replace background are pipeline dedicat
            # Orice altceva (inclusiv 'replace' obiect general) va folosi 'general_pipeline'
        }
        
        # Cache pentru instanțele de pipeline create
        self.pipelines: Dict[str, BasePipeline] = {}
        
        self._initialized = True
        logger.info("PipelineManager initialized")
    
    def register_pipeline(self, name: str, pipeline_class: type[BasePipeline]) -> None:
        """Înregistrează un tip de pipeline."""
        if not issubclass(pipeline_class, BasePipeline):
             logger.error(f"Cannot register '{name}': Class {pipeline_class.__name__} is not a subclass of BasePipeline.")
             return
        self.pipeline_types[name] = pipeline_class
        logger.info(f"Pipeline '{name}' registered successfully with class {pipeline_class.__name__}")
    
    def get_pipeline(self, name: str, **kwargs) -> Optional[BasePipeline]:
        """Obține sau creează o instanță a unui pipeline înregistrat."""
        if name not in self.pipeline_types:
            logger.warning(f"Pipeline type '{name}' not registered.")
            return None
        
        # Verificăm cache-ul de instanțe
        if name not in self.pipelines:
            try:
                # Instanțiem clasa de pipeline
                self.pipelines[name] = self.pipeline_types[name](**kwargs) 
                logger.info(f"Pipeline '{name}' instantiated.")
            except Exception as e_inst:
                 logger.error(f"Failed to instantiate pipeline '{name}' with class {self.pipeline_types[name].__name__}: {e_inst}", exc_info=True)
                 return None # Returnăm None dacă instanțierea eșuează
        
        return self.pipelines[name]
    
    def get_pipeline_for_operation(self, operation_type: Optional[str], target: Optional[str] = None) -> Optional[BasePipeline]:
        """Selectează pipeline-ul potrivit pe baza tipului de operație și a țintei."""
        pipeline_key = "general_pipeline" # Default
        op_type_lower = operation_type.lower() if operation_type else "general"
        target_lower = target.lower() if target else None

        mapping = self.operation_to_pipeline.get(op_type_lower)
        if mapping:
            # Căutăm cheia specifică pentru target, dacă există, altfel folosim default-ul operației
            pipeline_key = mapping.get(target_lower, mapping.get("default", pipeline_key)) # Folosim default-ul general dacă nu există nici target, nici default specific operației

        logger.info(f"Selected pipeline key '{pipeline_key}' for operation '{op_type_lower}' (target: '{target_lower}')")
        return self.get_pipeline(pipeline_key) # Obținem/creăm instanța
    
    def process_image(self, 
                      image, # Tipul va fi verificat în pipeline-ul specific
                      prompt: str, 
                      strength: float = 0.75,
                      operation_details: Optional[Dict[str, Any]] = None, # Primim detaliile pre-analizate
                      progress_callback: Optional[Callable] = None, 
                      **kwargs) -> Dict[str, Any]:
        """
        Procesează o imagine folosind pipeline-ul potrivit.
        Acum primește `operation_details` pre-analizate.
        """
        start_process_time = time.time() # Pornim cronometrul aici
        
        # Extragem tipul și ținta din detaliile operației
        op_type = operation_details.get('type') if operation_details else None
        target = operation_details.get('target_object') if operation_details else None # Folosim cheia corectă
        
        # Dacă detaliile nu sunt furnizate, le analizăm (fallback)
        if not operation_details:
            logger.warning("Operation details not provided to PipelineManager. Analyzing prompt again.")
            analyzer = OperationAnalyzer() # Instanțiem doar dacă e necesar
            operation_details = analyzer.analyze_operation(prompt)
            op_type = operation_details.get('type')
            target = operation_details.get('target_object') # Folosim cheia corectă

        # Obținem pipeline-ul potrivit
        pipeline_instance = self.get_pipeline_for_operation(op_type, target)

        if not pipeline_instance:
            error_msg = f"No pipeline available for operation type '{op_type}' (target: {target})"
            logger.error(error_msg)
            return {
                'result_image': image, 'mask_image': None, 'operation': operation_details,
                'message': error_msg, 'success': False
            }

        try:
            logger.info(f"Processing image with '{pipeline_instance.__class__.__name__}'...")
            
            # Pasăm toți parametrii relevanți către metoda process a pipeline-ului
            # Inclusiv operation_details și kwargs primite din UI
            result = pipeline_instance.process(
                image=image,
                prompt=prompt,
                strength=strength,
                progress_callback=progress_callback,
                operation=operation_details, # Pasăm și detaliile operației
                **kwargs # Pasăm restul argumentelor (num_steps, guidance, etc.)
            )
            
            # Adăugăm timpul total de procesare (opțional, poate fi calculat și în UI)
            end_process_time = time.time()
            processing_duration = end_process_time - start_process_time
            if isinstance(result, dict):
                 # Adăugăm sau actualizăm informațiile de operare
                 op_info = result.get('operation', operation_details or {})
                 op_info['pipeline_processing_time'] = f"{processing_duration:.2f} seconds"
                 op_info['pipeline_used'] = pipeline_instance.__class__.__name__
                 result['operation'] = op_info
                 # Asigurăm că există cheia 'success'
                 if 'success' not in result:
                      result['success'] = True # Presupunem succes dacă nu există eroare explicită
            
            logger.info(f"Finished processing with '{pipeline_instance.__class__.__name__}' in {processing_duration:.2f}s")
            return result
            
        except Exception as e:
            # Logăm eroarea completă cu traceback
            logger.error(f"Error processing image in pipeline '{pipeline_instance.__class__.__name__}': {str(e)}", exc_info=True)
            end_process_time = time.time() # Calculăm timpul chiar și la eroare
            processing_duration = end_process_time - start_process_time
            # Returnăm un dicționar de eroare consistent
            return {
                'result_image': image, # Returnăm imaginea originală
                'mask_image': None,
                'operation': operation_details or {'type': op_type, 'target': target},
                'message': f"Error in {pipeline_instance.__class__.__name__}: {str(e)}",
                'success': False,
                'pipeline_processing_time': f"{processing_duration:.2f} seconds"
            }

