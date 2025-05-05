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
            
            # Initialize the IP-Adapter
            self.ip_adapter = IPAdapterXL(
                pipe,
                "h94/IP-Adapter",
                subfolder="sdxl_models",
                cache_dir=self.cache_dir,
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
            
            # Check if the model files exist locally
            model_dir = os.path.join(self.cache_dir, "IP-Adapter")
            face_id_path = os.path.join(model_dir, "models/face-id/model.ckpt")
            ip_adapter_path = os.path.join(model_dir, "sdxl_models/ip-adapter-plus-face_sdxl_vit-h.bin")
            
            # Check if files exist, otherwise try to download
            if not os.path.exists(face_id_path) or not os.path.exists(ip_adapter_path):
                logger.warning("IP-Adapter model files not found locally")
                logger.info("Using direct model references from Hugging Face")
                
                image_encoder_path = None  # Let the library handle it
                ip_adapter_checkpoint = None  # Let the library handle it
            else:
                logger.info("Using local IP-Adapter model files")
                image_encoder_path = face_id_path
                ip_adapter_checkpoint = ip_adapter_path
            
            # Initialize the IP-Adapter
            self.ip_adapter = IPAdapterPlusXL(
                pipe,
                image_encoder_path=image_encoder_path,
                ip_adapter_checkpoint=ip_adapter_checkpoint,
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


def download_models(cache_dir: str) -> bool:
    """
    Download IP-Adapter models from Hugging Face.
    
    Args:
        cache_dir (str): Directory to save models
        
    Returns:
        bool: True if download was successful, False otherwise
    """
    try:
        from huggingface_hub import snapshot_download
        
        # Create the target directory
        ip_adapter_dir = os.path.join(cache_dir, "IP-Adapter")
        os.makedirs(ip_adapter_dir, exist_ok=True)
        
        # Download the models
        logger.info("Downloading IP-Adapter models...")
        
        # Download base models
        snapshot_download(
            repo_id="h94/IP-Adapter",
            local_dir=os.path.join(ip_adapter_dir, "models"),
            resume_download=True,
            local_files_only=False
        )
        
        # Download SDXL models
        snapshot_download(
            repo_id="h94/IP-Adapter",
            local_dir=os.path.join(ip_adapter_dir, "sdxl_models"),
            subfolder="sdxl_models",
            resume_download=True,
            local_files_only=False
        )
        
        logger.info("IP-Adapter models downloaded successfully")
        return True
    except Exception as e:
        logger.error(f"Error downloading IP-Adapter models: {e}")
        return False
