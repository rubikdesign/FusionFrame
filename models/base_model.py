#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Clasă de bază pentru toate modelele AI din FusionFrame 2.0
"""

import torch
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union

from fusionframe.config.app_config import AppConfig

# Setăm logger-ul
logger = logging.getLogger(__name__)

class BaseModel(ABC):
    """
    Clasă abstractă de bază pentru toate modelele AI
    
    Toate modelele specifice vor moșteni această clasă și vor
    implementa metodele abstracte.
    """
    
    def __init__(self, model_id: str, device: Optional[str] = None):
        """
        Inițializare pentru modelul de bază
        
        Args:
            model_id: Identificatorul modelului
            device: Dispozitivul pe care va rula modelul (cpu sau cuda)
        """
        self.model_id = model_id
        self.device = device if device else AppConfig.DEVICE
        self.model = None
        self.is_loaded = False
    
    @abstractmethod
    def load(self) -> bool:
        """
        Încarcă modelul în memorie
        
        Returns:
            True dacă încărcarea a reușit, False altfel
        """
        pass
    
    @abstractmethod
    def unload(self) -> bool:
        """
        Descarcă modelul din memorie
        
        Returns:
            True dacă descărcarea a reușit, False altfel
        """
        pass
    
    @abstractmethod
    def process(self, *args, **kwargs) -> Any:
        """
        Procesează input-ul folosind modelul
        
        Args:
            *args: Argumentele poziționale
            **kwargs: Argumentele numite
            
        Returns:
            Rezultatul procesării
        """
        pass
    
    def to(self, device: str) -> 'BaseModel':
        """
        Mută modelul pe un alt dispozitiv
        
        Args:
            device: Dispozitivul țintă (cpu sau cuda)
            
        Returns:
            Self, pentru chaining
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
        Obține informații despre model
        
        Returns:
            Dicționar cu informații despre model
        """
        return {
            "model_id": self.model_id,
            "device": self.device,
            "is_loaded": self.is_loaded,
            "model_type": self.__class__.__name__
        }