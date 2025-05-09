#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Manager for processing pipelines in FusionFrame 2.0
"""

import logging
from typing import Dict, Any, Optional, Callable

from processing.pipelines.base_pipeline import BasePipeline

# Set up logger
logger = logging.getLogger(__name__)

class PipelineManager:
    """
    Manager for processing pipelines
    
    Responsible for managing and coordinating different processing pipelines
    for various image editing operations.
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
        
        # Register all available pipelines
        from processing.pipelines.general_pipeline       import GeneralPipeline
        from processing.pipelines.removal_pipeline       import RemovalPipeline
        from processing.pipelines.color_change_pipeline  import ColorChangePipeline
        from processing.pipelines.add_object_pipeline    import AddObjectPipeline
        from processing.pipelines.background_pipeline    import BackgroundPipeline
        
        self.pipeline_types: Dict[str, BasePipeline] = {}
        # Register pipeline classes
        self.register_pipeline("general_pipeline", GeneralPipeline)
        self.register_pipeline("removal_pipeline", RemovalPipeline)
        self.register_pipeline("color_change_pipeline", ColorChangePipeline)
        self.register_pipeline("add_object_pipeline", AddObjectPipeline)
        self.register_pipeline("background_pipeline", BackgroundPipeline)

        # Aliases for operation_type
        # Common operations mapped to pipelines
        self.operation_to_pipeline = {
            "remove":   {"person": "removal_pipeline", "default": "removal_pipeline"},
            "color":    {"hair":   "color_change_pipeline", "default": "color_change_pipeline"},
            "add":      {"default": "add_object_pipeline"},
            "background": {"default": "background_pipeline"},
            # any other operation: will use general pipeline
        }
        
        # cache for pipeline instances
        self.pipelines: Dict[str, BasePipeline] = {}
        
        self._initialized = True
        logger.info("PipelineManager initialized")
    
    def register_pipeline(self, name: str, pipeline_class: BasePipeline) -> None:
        """
        Register a pipeline type for use
        
        Args:
            name: Name of the pipeline
            pipeline_class: Pipeline class to be instantiated
        """
        self.pipeline_types[name] = pipeline_class
        logger.info(f"Pipeline '{name}' registered successfully")
    
    def get_pipeline(self, name: str, **kwargs) -> Optional[BasePipeline]:
        """
        Get an instance of a registered pipeline
        
        Args:
            name: Name of the pipeline
            **kwargs: Arguments for pipeline initialization
        
        Returns:
            Pipeline instance or None if the pipeline doesn't exist
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
        Select the appropriate pipeline for the operation type
        
        Args:
            operation_type: Type of operation (remove, color, add, background, etc.)
            target: Operation target (optional)
        
        Returns:
            Appropriate pipeline for the operation or None if no pipeline is suitable
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
        Process an image using the appropriate pipeline
        
        Args:
            image: Image to process
            prompt: Editing prompt
            strength: Editing strength (0.0-1.0)
            operation_type: Operation type (optional, will be auto-detected)
            target: Operation target (optional)
            progress_callback: Progress callback function
        
        Returns:
            Dictionary with processing results:
                - 'result': Resulting image
                - 'mask': Used mask
                - 'operation': Operation details
                - 'message': Processing message
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