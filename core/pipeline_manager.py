#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Manager pentru pipeline-uri de procesare în FusionFrame 2.0
"""

import logging
from typing import Dict, Any, Optional, Callable

from processing.pipelines.base_pipeline import BasePipeline

# Setăm logger-ul
logger = logging.getLogger(__name__)

class PipelineManager:
    """
    Manager pentru pipeline-uri de procesare
    
    Responsabil pentru gestionarea și coordonarea diferitelor pipeline-uri
    de procesare pentru diverse operații de editare a imaginilor.
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
        
        # Înregistrăm toate pipeline-urile disponibile
        from processing.pipelines.general_pipeline       import GeneralPipeline
        from processing.pipelines.removal_pipeline       import RemovalPipeline
        from processing.pipelines.color_change_pipeline  import ColorChangePipeline
        from processing.pipelines.add_object_pipeline    import AddObjectPipeline
        from processing.pipelines.background_pipeline    import BackgroundPipeline
        
        self.pipeline_types: Dict[str, BasePipeline] = {}
        # Registrăm clasele de pipeline
        self.register_pipeline("general_pipeline", GeneralPipeline)
        self.register_pipeline("removal_pipeline", RemovalPipeline)
        self.register_pipeline("color_change_pipeline", ColorChangePipeline)
        self.register_pipeline("add_object_pipeline", AddObjectPipeline)
        self.register_pipeline("background_pipeline", BackgroundPipeline)

        # Aliasuri pentru operation_type
        # Operații comune mapate la pipeline-uri
        self.operation_to_pipeline = {
            "remove":   {"person": "removal_pipeline", "default": "removal_pipeline"},
            "color":    {"hair":   "color_change_pipeline", "default": "color_change_pipeline"},
            "add":      {"default": "add_object_pipeline"},
            "background": {"default": "background_pipeline"},
            # orice altă operație: va folosi pipeline general
        }
        
        # cache pentru instanțele de pipeline
        self.pipelines: Dict[str, BasePipeline] = {}
        
        self._initialized = True
        logger.info("PipelineManager initialized")
    
    def register_pipeline(self, name: str, pipeline_class: BasePipeline) -> None:
        """
        Înregistrează un tip de pipeline pentru utilizare
        
        Args:
            name: Numele pipeline-ului
            pipeline_class: Clasa de pipeline care va fi instanțiată
        """
        self.pipeline_types[name] = pipeline_class
        logger.info(f"Pipeline '{name}' registered successfully")
    
    def get_pipeline(self, name: str, **kwargs) -> Optional[BasePipeline]:
        """
        Obține o instanță a unui pipeline înregistrat
        
        Args:
            name: Numele pipeline-ului
            **kwargs: Argumente pentru inițializarea pipeline-ului
        
        Returns:
            Instanța pipeline-ului sau None dacă pipeline-ul nu există
        """
        if name not in self.pipeline_types:
            logger.warning(f"Pipeline '{name}' not registered")
            return None
        
        if name not in self.pipelines:
            self.pipelines[name] = self.pipeline_types[name](**kwargs)
            logger.info(f"Pipeline '{name}' instantiated")
        
        return self.pipelines[name]
    
    def get_pipeline_for_operation(self, operation_type: str, target: str = None) -> Optional[BasePipeline]:
        """
        Selectează pipeline-ul potrivit pentru tipul de operație
        
        Args:
            operation_type: Tipul operației (remove, color, add, background, etc.)
            target: Ținta operației (opțional)
        
        Returns:
            Pipeline-ul potrivit pentru operație sau None dacă niciun pipeline nu este potrivit
        """
        pipeline_key = None
        mapping = self.operation_to_pipeline.get(operation_type)
        if mapping:
            pipeline_key = mapping.get(target) if target in mapping else mapping.get("default")

        if not pipeline_key:
            logger.info(f"No specific pipeline for operation '{operation_type}', using general pipeline")
            pipeline_key = "general_pipeline"

        return self.get_pipeline(pipeline_key)
    
    def process_image(self, image, prompt: str, strength: float = 0.75,
                      operation_type: str = None, target: str = None,
                      progress_callback: Callable = None, **kwargs) -> Dict[str, Any]:
        """
        Procesează o imagine folosind pipeline-ul potrivit
        
        Args:
            image: Imaginea de procesat
            prompt: Prompt-ul de editare
            strength: Intensitatea editării (0.0-1.0)
            operation_type: Tipul operației (opțional, va fi detectat automat)
            target: Ținta operației (opțional)
            progress_callback: Funcție de callback pentru progres
        
        Returns:
            Dicționar cu rezultatele procesării:
                - 'result': Imaginea rezultată
                - 'mask': Masca utilizată
                - 'operation': Detalii despre operație
                - 'message': Mesaj despre procesare
        """
        from processing.analyzer import OperationAnalyzer

        if not operation_type:
            analyzer = OperationAnalyzer()
            operation = analyzer.analyze_operation(prompt)
            operation_type = operation.get('type')
            target = operation.get('target')

        pipeline = self.get_pipeline_for_operation(operation_type, target)

        if not pipeline:
            logger.error(f"No pipeline available for operation '{operation_type}'")
            return {
                'result': image,
                'mask': None,
                'operation': {'type': operation_type, 'target': target},
                'message': f"No pipeline available for operation '{operation_type}'"
            }

        try:
            logger.info(f"Processing image with '{pipeline.__class__.__name__}'")
            result = pipeline.process(
                image=image,
                prompt=prompt,
                strength=strength,
                progress_callback=progress_callback,
                **kwargs
            )
            return result
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            return {
                'result': image,
                'mask': None,
                'operation': {'type': operation_type, 'target': target},
                'message': f"Error: {str(e)}"
            }
