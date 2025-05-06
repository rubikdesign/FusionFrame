#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Manager pentru resursele și memoria în FusionFrame 2.0
"""

import os
import gc
import logging
import psutil
import torch
from typing import Dict, Any, Optional

from config.app_config import AppConfig

# Setăm logger-ul
logger = logging.getLogger(__name__)

class ResourceManager:
    """
    Manager pentru resursele și memoria
    
    Responsabil pentru monitorizarea și optimizarea utilizării resurselor
    și memoriei pentru a asigura performanța optimă a aplicației.
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ResourceManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self.config = AppConfig
        self.current_gpu_memory = 0
        self.peak_gpu_memory = 0
        self.peak_ram_usage = 0
        
        self._initialized = True
        logger.info("ResourceManager initialized")
    
    def get_system_info(self) -> Dict[str, Any]:
        """
        Obține informații despre sistem și resurse
        
        Returns:
            Dicționar cu informații despre sistem
        """
        info = {
            "cpu": {
                "cores": psutil.cpu_count(logical=False),
                "threads": psutil.cpu_count(logical=True),
                "usage_percent": psutil.cpu_percent()
            },
            "ram": {
                "total": psutil.virtual_memory().total,
                "available": psutil.virtual_memory().available,
                "used": psutil.virtual_memory().used,
                "percent": psutil.virtual_memory().percent
            },
            "disk": {
                "total": psutil.disk_usage('/').total,
                "free": psutil.disk_usage('/').free,
                "used": psutil.disk_usage('/').used,
                "percent": psutil.disk_usage('/').percent
            }
        }
        
        # Adăugăm informații despre GPU dacă este disponibil
        if torch.cuda.is_available():
            info["gpu"] = {
                "name": torch.cuda.get_device_name(0),
                "count": torch.cuda.device_count(),
                "current_memory": self.get_gpu_memory_usage(),
                "peak_memory": self.peak_gpu_memory
            }
        
        return info
    
    def get_gpu_memory_usage(self) -> int:
        """
        Obține utilizarea curentă a memoriei GPU
        
        Returns:
            Memoria utilizată în bytes
        """
        if not torch.cuda.is_available():
            return 0
            
        # Obținem memoria utilizată
        memory_allocated = torch.cuda.memory_allocated()
        memory_reserved = torch.cuda.memory_reserved()
        
        # Actualizăm valorile curente și de vârf
        self.current_gpu_memory = memory_allocated
        self.peak_gpu_memory = max(self.peak_gpu_memory, memory_allocated)
        
        return memory_allocated
    
    def optimize_memory(self, threshold_percent: float = 90.0) -> bool:
        """
        Optimizează utilizarea memoriei dacă depășește un anumit prag
        
        Args:
            threshold_percent: Pragul procentual pentru declanșarea optimizării
            
        Returns:
            True dacă optimizarea a fost efectuată, False altfel
        """
        # Verificăm utilizarea RAM
        ram_percent = psutil.virtual_memory().percent
        
        # Verificăm utilizarea GPU dacă este disponibilă
        gpu_percent = 0
        if torch.cuda.is_available():
            total_memory = torch.cuda.get_device_properties(0).total_memory
            used_memory = self.get_gpu_memory_usage()
            gpu_percent = (used_memory / total_memory) * 100
        
        # Dacă oricare depășește pragul, optimizăm memoria
        if ram_percent > threshold_percent or gpu_percent > threshold_percent:
            logger.info(f"Memory usage high: RAM {ram_percent}%, GPU {gpu_percent}%. Optimizing...")
            
            # Curățăm memoria neutilizată
            gc.collect()
            
            # Curățăm cache-ul CUDA dacă e disponibil
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Verificăm din nou utilizarea
            new_ram_percent = psutil.virtual_memory().percent
            new_gpu_percent = 0
            if torch.cuda.is_available():
                used_memory = self.get_gpu_memory_usage()
                new_gpu_percent = (used_memory / total_memory) * 100
            
            logger.info(f"After optimization: RAM {new_ram_percent}%, GPU {new_gpu_percent}%")
            return True
        
        return False
    
    def should_use_tiling(self, image_size: tuple) -> bool:
        """
        Determină dacă ar trebui să folosim tiling pentru procesarea imaginii
        
        Args:
            image_size: Dimensiunea imaginii (width, height)
            
        Returns:
            True dacă ar trebui să folosim tiling, False altfel
        """
        width, height = image_size
        
        # Calculăm numărul de pixeli
        num_pixels = width * height
        
        # Pragul pentru tiling
        tiling_threshold = 1024 * 1024  # 1 megapixel
        
        # Verificăm dacă memoria GPU este limitată
        if torch.cuda.is_available():
            free_memory = torch.cuda.get_device_properties(0).total_memory - self.get_gpu_memory_usage()
            # Dacă memoria liberă este sub 2 GB, folosim tiling indiferent de dimensiune
            if free_memory < 2 * 1024 * 1024 * 1024:
                return True
        
        # Decizie bazată pe numărul de pixeli
        return num_pixels > tiling_threshold
    
    def estimate_memory_requirements(self, operation_type: str, image_size: tuple) -> Dict[str, Any]:
        """
        Estimează cerințele de memorie pentru o operație
        
        Args:
            operation_type: Tipul operației
            image_size: Dimensiunea imaginii
            
        Returns:
            Dicționar cu estimările de memorie
        """
        width, height = image_size
        num_pixels = width * height
        
        # Memory constants (bytes per pixel)
        BASE_MEMORY_PER_PIXEL = 20  # Aproximativ pentru o imagine în memorie
        
        # Estimări de bază
        base_memory = num_pixels * BASE_MEMORY_PER_PIXEL
        
        # Factori de multiplicare pentru diferite operații
        operation_factors = {
            "remove": 3.0,      # Necesită mai multă memorie pentru reconstrucție
            "color": 1.5,       # Schimbările de culoare sunt mai simple
            "add": 2.0,         # Adăugarea obiectelor necesită memorie moderată
            "background": 2.5,  # Înlocuirea fundalului necesită mai multă memorie
            "default": 2.0      # Factor implicit
        }
        
        # Obținem factorul pentru operație
        factor = operation_factors.get(operation_type, operation_factors["default"])
        
        # Calculăm estimarea finală
        estimated_memory = base_memory * factor
        
        # Verificăm disponibilitatea memoriei
        available_memory = 0
        if torch.cuda.is_available():
            available_memory = torch.cuda.get_device_properties(0).total_memory - self.get_gpu_memory_usage()
        else:
            available_memory = psutil.virtual_memory().available
        
        # Rezultatul estimării
        return {
            "estimated_bytes": estimated_memory,
            "estimated_mb": estimated_memory / (1024 * 1024),
            "available_bytes": available_memory,
            "available_mb": available_memory / (1024 * 1024),
            "is_sufficient": available_memory > estimated_memory,
            "recommended_tiling": available_memory < estimated_memory
        }