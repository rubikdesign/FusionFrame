#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Manager for resources and memory in FusionFrame 2.0
"""

import os
import gc
import logging
import psutil
import torch
from typing import Dict, Any, Optional

from config.app_config import AppConfig

# Set up logger
logger = logging.getLogger(__name__)

class ResourceManager:
    """
    Manager for resources and memory
    
    Responsible for monitoring and optimizing resource and memory usage
    to ensure optimal application performance.
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
        Get system and resource information
        
        Returns:
            Dictionary with system information
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
        
        # Add GPU information if available
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
        Get current GPU memory usage
        
        Returns:
            Memory used in bytes
        """
        if not torch.cuda.is_available():
            return 0
            
        # Get used memory
        memory_allocated = torch.cuda.memory_allocated()
        memory_reserved = torch.cuda.memory_reserved()
        
        # Update current and peak values
        self.current_gpu_memory = memory_allocated
        self.peak_gpu_memory = max(self.peak_gpu_memory, memory_allocated)
        
        return memory_allocated
    
    def optimize_memory(self, threshold_percent: float = 90.0) -> bool:
        """
        Optimize memory usage if it exceeds a certain threshold
        
        Args:
            threshold_percent: Percentage threshold for triggering optimization
            
        Returns:
            True if optimization was performed, False otherwise
        """
        # Check RAM usage
        ram_percent = psutil.virtual_memory().percent
        
        # Check GPU usage if available
        gpu_percent = 0
        if torch.cuda.is_available():
            total_memory = torch.cuda.get_device_properties(0).total_memory
            used_memory = self.get_gpu_memory_usage()
            gpu_percent = (used_memory / total_memory) * 100
        
        # If either exceeds the threshold, optimize memory
        if ram_percent > threshold_percent or gpu_percent > threshold_percent:
            logger.info(f"Memory usage high: RAM {ram_percent}%, GPU {gpu_percent}%. Optimizing...")
            
            # Clean up unused memory
            gc.collect()
            
            # Clear CUDA cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Check usage again
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
        Determine whether to use tiling for image processing
        
        Args:
            image_size: Image dimensions (width, height)
            
        Returns:
            True if tiling should be used, False otherwise
        """
        width, height = image_size
        
        # Calculate number of pixels
        num_pixels = width * height
        
        # Tiling threshold
        tiling_threshold = 1024 * 1024  # 1 megapixel
        
        # Check if GPU memory is limited
        if torch.cuda.is_available():
            free_memory = torch.cuda.get_device_properties(0).total_memory - self.get_gpu_memory_usage()
            # If free memory is below 2 GB, use tiling regardless of size
            if free_memory < 2 * 1024 * 1024 * 1024:
                return True
        
        # Decision based on number of pixels
        return num_pixels > tiling_threshold
    
    def estimate_memory_requirements(self, operation_type: str, image_size: tuple) -> Dict[str, Any]:
        """
        Estimate memory requirements for an operation
        
        Args:
            operation_type: Type of operation
            image_size: Image dimensions
            
        Returns:
            Dictionary with memory estimates
        """
        width, height = image_size
        num_pixels = width * height
        
        # Memory constants (bytes per pixel)
        BASE_MEMORY_PER_PIXEL = 20  # Approximate for an in-memory image
        
        # Base estimates
        base_memory = num_pixels * BASE_MEMORY_PER_PIXEL
        
        # Multiplication factors for different operations
        operation_factors = {
            "remove": 3.0,      # Requires more memory for reconstruction
            "color": 1.5,       # Color changes are simpler
            "add": 2.0,         # Adding objects requires moderate memory
            "background": 2.5,  # Background replacement requires more memory
            "default": 2.0      # Default factor
        }
        
        # Get factor for operation
        factor = operation_factors.get(operation_type, operation_factors["default"])
        
        # Calculate final estimate
        estimated_memory = base_memory * factor
        
        # Check memory availability
        available_memory = 0
        if torch.cuda.is_available():
            available_memory = torch.cuda.get_device_properties(0).total_memory - self.get_gpu_memory_usage()
        else:
            available_memory = psutil.virtual_memory().available
        
        # Estimation result
        return {
            "estimated_bytes": estimated_memory,
            "estimated_mb": estimated_memory / (1024 * 1024),
            "available_bytes": available_memory,
            "available_mb": available_memory / (1024 * 1024),
            "is_sufficient": available_memory > estimated_memory,
            "recommended_tiling": available_memory < estimated_memory
        }