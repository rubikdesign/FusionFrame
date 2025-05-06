#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Manager pentru pipeline-uri de procesare în FusionFrame 2.0
"""

import logging
from typing import Dict, Any, Type, List, Optional, Callable

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
        
        self.pipelines = {}
        self.pipeline_types = {}
        
        self._initialized = True
        logger.info("PipelineManager initialized")
    
    def register_pipeline(self, name: str, pipeline_class: Type[BasePipeline]) -> None:
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
        
        # Instanțiază pipeline-ul dacă nu există deja
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
        # Mapăm tipurile de operații la pipeline-uri
        operation_to_pipeline = {
            "remove": {
                "person": "removal_pipeline",
                "default": "removal_pipeline"
            },
            "color": {
                "hair": "color_change_pipeline",
                "default": "color_change_pipeline"
            },
            "add": {
                "default": "add_object_pipeline"
            },
            "background": {
                "default": "background_pipeline"
            }
        }
        
        # Obținem numele pipeline-ului pentru operație
        pipeline_name = None
        if operation_type in operation_to_pipeline:
            if target and target in operation_to_pipeline[operation_type]:
                pipeline_name = operation_to_pipeline[operation_type][target]
            else:
                pipeline_name = operation_to_pipeline[operation_type].get("default")
        
        # Dacă nu găsim un pipeline specific, folosim pipeline-ul general
        if not pipeline_name:
            logger.info(f"No specific pipeline for operation '{operation_type}', using general pipeline")
            pipeline_name = "general_pipeline"
        
        # Returnăm pipeline-ul
        return self.get_pipeline(pipeline_name)
    
    def process_image(self, image, prompt, strength=0.75, operation_type=None, 
                     target=None, progress_callback=None) -> Dict[str, Any]:
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
        
        # Detectăm automat operația dacă nu este specificată
        if not operation_type:
            # Analizăm operația din prompt
            analyzer = OperationAnalyzer()
            operation = analyzer.analyze_operation(prompt)
            operation_type = operation['type']
            target = operation.get('target')
        
        # Obținem pipeline-ul potrivit
        pipeline = self.get_pipeline_for_operation(operation_type, target)
        
        if not pipeline:
            logger.error(f"No pipeline available for operation '{operation_type}'")
            return {
                'result': image,
                'mask': None,
                'operation': {'type': operation_type, 'target': target},
                'message': f"No pipeline available for operation '{operation_type}'"
            }
        
        # Procesăm imaginea cu pipeline-ul
        try:
            logger.info(f"Processing image with '{pipeline.__class__.__name__}'")
            result = pipeline.process(
                image=image,
                prompt=prompt, 
                strength=strength,
                progress_callback=progress_callback
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