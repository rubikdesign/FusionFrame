#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Base class for all AI models in FusionFrame 2.0
"""

import torch
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union

from config.app_config import AppConfig

# Set up logger
logger = logging.getLogger(__name__)

class BaseModel(ABC):
    """
    Abstract base class for all AI models
    
    All specific models will inherit this class and
    implement the abstract methods.
    """
    
    def __init__(self, model_id: str, device: Optional[str] = None):
        """
        Base model initialization
        
        Args:
            model_id: Model identifier
            device: Device where the model will run (cpu or cuda)
        """
        self.model_id = model_id
        self.device = device if device else AppConfig.DEVICE
        self.model = None
        self.is_loaded = False
    
    @abstractmethod
    def load(self) -> bool:
        """
        Load model into memory
        
        Returns:
            True if loading succeeded, False otherwise
        """
        pass
    
    @abstractmethod
    def unload(self) -> bool:
        """
        Unload model from memory
        
        Returns:
            True if unloading succeeded, False otherwise
        """
        pass
    
    @abstractmethod
    def process(self, *args, **kwargs) -> Any:
        """
        Process input using the model
        
        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Processing result
        """
        pass
    
    def to(self, device: str) -> 'BaseModel':
        """
        Move model to another device
        
        Args:
            device: Target device (cpu or cuda)
            
        Returns:
            Self, for chaining
        """
        if not self.is_loaded:
            self.device = device
            return self
            
        if self.model is None:
            logger.warning(f"Cannot move model '{self.model_id}' to {device}: model not loaded")
            return self
            
        try:
            self.model.to(device)
            self.device = device
            logger.info(f"Model '{self.model_id}' moved to {device}")
        except Exception as e:
            logger.error(f"Error moving model '{self.model_id}' to {device}: {str(e)}")
        
        return self
    
    def get_info(self) -> Dict[str, Any]:
        """
        Get model information
        
        Returns:
            Dictionary with model information
        """
        return {
            "model_id": self.model_id,
            "device": self.device,
            "is_loaded": self.is_loaded,
            "model_type": self.__class__.__name__
        }