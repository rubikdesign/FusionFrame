"""
IP-Adapter plugin for FusionFrame application.

This module provides functions to integrate IP-Adapter with Stable Diffusion
pipelines for improved face preservation and identity transfer.
"""

import os
import torch
import logging
from typing import Dict, Any, Optional, Union
from PIL import Image

logger = logging.getLogger(__name__)

class IPAdapterPlugin:
    """
    IP-Adapter integration for Stable Diffusion pipelines.
    
    This class provides a clean interface to use IP-Adapter for face-preserving
    image generation with Stable Diffusion.
    """
    
    def __init__(self, cache_dir: str):
        """
        Initialize the IP-Adapter plugin.
        
        Args:
            cache_dir (str): Directory for cached models
        """
        self.cache_dir = cache_dir
        self.ip_adapter = None
        self.is_available = self._check_availability()
        
    def _check_availability(self) -> bool:
        """
        Check if IP-Adapter is available (library installed).
        
        Returns:
            bool: True if IP-Adapter is available, False otherwise
        """
        try:
            import ip_adapter  # noqa
            logger.info("IP-Adapter library is available")
            return True
        except ImportError:
            logger.warning("IP-Adapter library not found. Face preservation will be limited.")
            logger.info("Install with: pip install git+https://github.com/tencent-ailab/IP-Adapter.git")
            return False
    
    def setup(self, pipe, device: str) -> bool:
        """
        Set up IP-Adapter with the given pipeline.
        
        Args:
            pipe: Diffusers pipeline
            device (str): Device to use (cuda or cpu)
            
        Returns:
            bool: True if setup was successful, False otherwise
        """
        if not self.is_available:
            return False
            
        try:
            # Dynamically import to avoid errors if not installed
            from ip_adapter import IPAdapterXL
            
            # Calea către modelul IP-Adapter
            ip_ckpt = os.path.join(self.cache_dir, "IP-Adapter", "models", "ip-adapter-plus-face_sdxl.bin")
            
            # Dacă fișierul nu există local, folosim calea directă de pe Hugging Face
            if not os.path.exists(ip_ckpt):
                ip_ckpt = "h94/IP-Adapter/sdxl_models/ip-adapter-plus-face_sdxl.bin"
            
            # Initialize the IP-Adapter cu parametrul obligatoriu ip_ckpt
            self.ip_adapter = IPAdapterXL(
                pipe, 
                ip_ckpt,
                device=device
            )
            
            logger.info("✅ IP-Adapter successfully initialized")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize IP-Adapter: {e}")
            return False
    
    def setup_plus(self, pipe, device: str) -> bool:
        """
        Set up IP-Adapter Plus with the given pipeline for better face preservation.
        
        Args:
            pipe: Diffusers pipeline
            device (str): Device to use (cuda or cpu)
            
        Returns:
            bool: True if setup was successful, False otherwise
        """
        if not self.is_available:
            return False
            
        try:
            # Dynamically import to avoid errors if not installed
            from ip_adapter import IPAdapterPlusXL
            
            # Definim calea către modelul IP-Adapter
            ip_ckpt = os.path.join(self.cache_dir, "IP-Adapter", "models", "ip-adapter-plus-face_sdxl.bin")
            
            # Dacă fișierul nu există local, folosim calea directă de pe Hugging Face
            if not os.path.exists(ip_ckpt):
                ip_ckpt = "h94/IP-Adapter/sdxl_models/ip-adapter-plus-face_sdxl.bin"
            
            # Initialize the IP-Adapter
            self.ip_adapter = IPAdapterPlusXL(
                pipe,
                ip_ckpt,  # Parametrul obligatoriu
                device=device,
                num_tokens=16
            )
            
            logger.info("✅ IP-Adapter Plus successfully initialized")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize IP-Adapter Plus: {e}")
            return False
    
    def generate(
        self, 
        prompt: str, 
        negative_prompt: str, 
        reference_image: Image.Image, 
        input_image: Image.Image, 
        strength: float = 0.75, 
        guidance_scale: float = 7.5, 
        num_inference_steps: int = 30, 
        generator = None, 
        callback = None
    ) -> Image.Image:
        """
        Generate an image with IP-Adapter for face preservation.
        
        Args:
            prompt (str): Text prompt for generation
            negative_prompt (str): Negative text prompt
            reference_image (PIL.Image): Reference image with the face to preserve
            input_image (PIL.Image): Input image to transform
            strength (float): Transformation strength (0-1)
            guidance_scale (float): Classifier-free guidance scale
            num_inference_steps (int): Number of denoising steps
            generator: Random number generator
            callback: Progress callback function
            
        Returns:
            PIL.Image: Generated image
        """
        if not self.ip_adapter:
            logger.error("IP-Adapter not initialized")
            return input_image
            
        try:
            # Encode reference image to get image embeddings
            ip_image_embeds = self.ip_adapter.encode_image(reference_image)
            
            # Run generation with image embeddings
            result = self.ip_adapter.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=input_image,
                ip_adapter_image_embeds=ip_image_embeds,
                strength=strength,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                generator=generator,
                callback=callback,
                callback_steps=1
            ).images[0]
            
            return result
        except Exception as e:
            logger.error(f"Error during IP-Adapter generation: {e}")
            return input_image