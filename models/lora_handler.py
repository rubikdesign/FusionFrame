#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Implementare pentru gestionarea LoRA în FusionFrame 2.0
"""

import os
import logging
from typing import Dict, Any, List, Optional, Union, Tuple
from pathlib import Path

from fusionframe.config.app_config import AppConfig
from fusionframe.config.model_config import ModelConfig

# Setăm logger-ul
logger = logging.getLogger(__name__)

class LoraHandler:
    """
    Manager pentru gestionarea LoRA-urilor
    
    Responsabil pentru încărcarea, descărcarea și aplicarea LoRA-urilor
    pentru a personaliza modelele de generare.
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LoraHandler, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self.config = AppConfig
        self.model_config = ModelConfig
        self.lora_dir = self.config.LORA_DIR
        self.active_loras = []
        self.max_loras = self.model_config.LORA_CONFIG["max_loras"]
        self.weight_range = self.model_config.LORA_CONFIG["weight_range"]
        
        # Asigurăm că directorul pentru LoRA-uri există
        Path(self.lora_dir).mkdir(parents=True, exist_ok=True)
        
        self._initialized = True
        logger.info("LoraHandler initialized")
        
        # Scanează LoRA-urile disponibile
        self.available_loras = self.scan_available_loras()
    
    def scan_available_loras(self) -> List[Dict[str, Any]]:
        """
        Scanează directorul de LoRA-uri pentru a găsi fișierele disponibile
        
        Returns:
            Lista cu informații despre LoRA-urile disponibile
        """
        lora_files = []
        
        try:
            # Scanăm directorul pentru fișiere safetensors și pt
            lora_paths = list(Path(self.lora_dir).glob("**/*.safetensors")) + \
                         list(Path(self.lora_dir).glob("**/*.pt"))
            
            # Procesăm fiecare fișier găsit
            for lora_path in lora_paths:
                # Obținem informații despre fișier
                lora_name = lora_path.stem
                lora_format = lora_path.suffix[1:]  # Eliminăm punctul
                lora_size = lora_path.stat().st_size
                
                # Adăugăm la lista de LoRA-uri
                lora_files.append({
                    "name": lora_name,
                    "path": str(lora_path),
                    "format": lora_format,
                    "size": lora_size,
                    "size_mb": round(lora_size / (1024 * 1024), 2)
                })
            
            logger.info(f"Found {len(lora_files)} available LoRA files")
            
        except Exception as e:
            logger.error(f"Error scanning LoRA directory: {str(e)}")
        
        return lora_files
    
    def get_available_loras(self) -> List[Dict[str, Any]]:
        """
        Obține lista de LoRA-uri disponibile, rescanând directorul
        
        Returns:
            Lista actualizată cu informații despre LoRA-urile disponibile
        """
        self.available_loras = self.scan_available_loras()
        return self.available_loras
    
    def add_lora_to_pipeline(self, pipeline: Any, lora_info: Dict[str, Any],
                            weight: Optional[float] = None) -> Tuple[Any, bool]:
        """
        Adaugă un LoRA la un pipeline
        
        Args:
            pipeline: Pipeline-ul la care se adaugă LoRA
            lora_info: Informații despre LoRA
            weight: Greutatea de aplicare (opțională)
            
        Returns:
            Tuple cu pipeline-ul actualizat și un boolean care indică succesul
        """
        try:
            # Verificăm dacă pipeline-ul suportă LoRA
            if not hasattr(pipeline, "load_lora_weights"):
                logger.error(f"Pipeline does not support LoRA")
                return pipeline, False
            
            # Verificăm dacă path-ul există
            lora_path = lora_info.get("path")
            if not os.path.exists(lora_path):
                logger.error(f"LoRA path does not exist: {lora_path}")
                return pipeline, False
            
            # Generăm un nume de adaptor unic
            adapter_name = lora_info.get("name", os.path.basename(lora_path))
            
            # Încărcăm LoRA
            pipeline.load_lora_weights(
                lora_path,
                adapter_name=adapter_name,
                weight_name=lora_info.get("weight_name")
            )
            
            # Setăm greutatea dacă este specificată
            if weight is not None:
                # Asigurăm că greutatea este în intervalul permis
                min_weight, max_weight = self.weight_range
                weight = max(min_weight, min(max_weight, weight))
                
                # Setăm greutatea
                pipeline.set_adapters([adapter_name], [weight])
            
            # Adăugăm la lista de LoRA-uri active
            self.active_loras.append({
                "name": adapter_name,
                "path": lora_path,
                "weight": weight
            })
            
            logger.info(f"LoRA '{adapter_name}' added successfully with weight {weight}")
            return pipeline, True
            
        except Exception as e:
            logger.error(f"Error adding LoRA: {str(e)}")
            return pipeline, False
    
    def remove_lora_from_pipeline(self, pipeline: Any, 
                                adapter_name: str) -> Tuple[Any, bool]:
        """
        Elimină un LoRA din pipeline
        
        Args:
            pipeline: Pipeline-ul din care se elimină LoRA
            adapter_name: Numele adaptorului de eliminat
            
        Returns:
            Tuple cu pipeline-ul actualizat și un boolean care indică succesul
        """
        try:
            # Verificăm dacă pipeline-ul suportă LoRA
            if not hasattr(pipeline, "delete_adapters"):
                logger.error(f"Pipeline does not support LoRA")
                return pipeline, False
            
            # Eliminăm LoRA
            pipeline.delete_adapters([adapter_name])
            
            # Actualizăm lista de LoRA-uri active
            self.active_loras = [lora for lora in self.active_loras if lora["name"] != adapter_name]
            
            logger.info(f"LoRA '{adapter_name}' removed successfully")
            return pipeline, True
            
        except Exception as e:
            logger.error(f"Error removing LoRA: {str(e)}")
            return pipeline, False
    
    def update_lora_weight(self, pipeline: Any, adapter_name: str, 
                          weight: float) -> Tuple[Any, bool]:
        """
        Actualizează greutatea unui LoRA în pipeline
        
        Args:
            pipeline: Pipeline-ul în care se actualizează LoRA
            adapter_name: Numele adaptorului
            weight: Noua greutate de aplicare
            
        Returns:
            Tuple cu pipeline-ul actualizat și un boolean care indică succesul
        """
        try:
            # Verificăm dacă pipeline-ul suportă LoRA
            if not hasattr(pipeline, "set_adapters"):
                logger.error(f"Pipeline does not support LoRA")
                return pipeline, False
            
            # Asigurăm că greutatea este în intervalul permis
            min_weight, max_weight = self.weight_range
            weight = max(min_weight, min(max_weight, weight))
            
            # Setăm greutatea
            pipeline.set_adapters([adapter_name], [weight])
            
            # Actualizăm lista de LoRA-uri active
            for lora in self.active_loras:
                if lora["name"] == adapter_name:
                    lora["weight"] = weight
                    break
            
            logger.info(f"LoRA '{adapter_name}' weight updated to {weight}")
            return pipeline, True
            
        except Exception as e:
            logger.error(f"Error updating LoRA weight: {str(e)}")
            return pipeline, False
    
    def get_active_loras(self) -> List[Dict[str, Any]]:
        """
        Obține lista de LoRA-uri active
        
        Returns:
            Lista cu informații despre LoRA-urile active
        """
        return self.active_loras