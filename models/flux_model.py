#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Implementation for FLUX model in FusionFrame 2.0
"""

import os
import torch
import logging
from typing import Dict, Any, Optional, List, Union, Tuple
from PIL import Image
import numpy as np

from config.app_config import AppConfig
from config.model_config import ModelConfig
from models.base_model import BaseModel

# Set up logger
logger = logging.getLogger(__name__)

class FluxModel(BaseModel):
    """
    Implementation for FLUX model
    
    FLUX is an alternative editing model that
    can be used when HiDream doesn't produce
    optimal results.
    """
    
    def __init__(self, 
                model_id: str = ModelConfig.BACKUP_MODEL, 
                device: Optional[str] = None):
        """
        Initialization for FLUX model
        
        Args:
            model_id: Model identifier (default FLUX.1-dev)
            device: Device where the model will run (default from AppConfig)
        """
        super().__init__(model_id, device)
        
        self.config = ModelConfig.FLUX_CONFIG
        self.vae = None
        self.controlnet = None
        self.pipeline = None
        self.lora_weights = []
    
    def load(self) -> bool:
        """
        Load FLUX model
        
        Returns:
            True if loading succeeded, False otherwise
        """
        logger.info(f"Loading FLUX model '{self.model_id}'")
        
        try:
            from diffusers import (
                StableDiffusionXLInpaintPipeline,
                AutoencoderKL,
                EulerAncestralDiscreteScheduler,
                ControlNetModel
            )
            
            # Load VAE
            self.vae = AutoencoderKL.from_pretrained(
                self.config["vae_name_or_path"],
                torch_dtype=AppConfig.DTYPE,
                cache_dir=AppConfig.CACHE_DIR
            ).to(self.device)
            
            # Load ControlNet if configured
            if ModelConfig.CONTROLNET_CONFIG:
                try:
                    self.controlnet = ControlNetModel.from_pretrained(
                        ModelConfig.CONTROLNET_CONFIG["model_id"],
                        torch_dtype=AppConfig.DTYPE,
                        use_safetensors=True,
                        variant="fp16" if AppConfig.DTYPE == torch.float16 else None,
                        cache_dir=AppConfig.CACHE_DIR
                    ).to(self.device)
                except Exception as e:
                    logger.error(f"Error loading ControlNet: {e}")
                    logger.info("Continuing without ControlNet")
            
            # Load main pipeline
            self.pipeline = StableDiffusionXLInpaintPipeline.from_pretrained(
                self.config["pretrained_model_name_or_path"],
                vae=self.vae,
                torch_dtype=AppConfig.DTYPE,
                variant="fp16" if AppConfig.DTYPE == torch.float16 else None,
                use_safetensors=self.config["use_safetensors"],
                cache_dir=AppConfig.CACHE_DIR
            )
            
            # Add controlnet to pipeline if available
            if self.controlnet is not None:
                self.pipeline.controlnet = self.controlnet
            
            # Optimizations for low VRAM
            if AppConfig.LOW_VRAM_MODE:
                self.pipeline.enable_model_cpu_offload()
            else:
                self.pipeline.to(self.device)
            
            # Set scheduler for FLUX
            self.pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
                self.pipeline.scheduler.config
            )
            
            # Load configured LoRAs
            if self.config["lora_weights"]:
                self._load_loras()
            
            self.model = self.pipeline
            self.is_loaded = True
            logger.info(f"FLUX model '{self.model_id}' loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error loading FLUX model '{self.model_id}': {str(e)}")
            self.is_loaded = False
            return False
    
    def _load_loras(self) -> None:
        """Load configured LoRAs"""
        for lora_info in self.config["lora_weights"]:
            try:
                # Load LoRA
                self.pipeline.load_lora_weights(
                    lora_info["path"],
                    adapter_name=lora_info.get("name", os.path.basename(lora_info["path"])),
                    weight_name=lora_info.get("weight_name")
                )
                self.lora_weights.append(lora_info)
                logger.info(f"LoRA '{lora_info.get('name')}' loaded successfully")
            except Exception as e:
                logger.error(f"Error loading LoRA '{lora_info.get('name')}': {str(e)}")
    
    def unload(self) -> bool:
        """
        Unload model from memory
        
        Returns:
            True if unloading succeeded, False otherwise
        """
        if not self.is_loaded:
            return True
            
        try:
            # Unload components
            self.vae = None
            self.controlnet = None
            self.pipeline = None
            self.model = None
            
            # Clear memory
            torch.cuda.empty_cache()
            
            self.is_loaded = False
            logger.info(f"FLUX model '{self.model_id}' unloaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error unloading FLUX model '{self.model_id}': {str(e)}")
            return False
    
    def process(self, 
               image: Union[Image.Image, np.ndarray],
               mask_image: Union[Image.Image, np.ndarray],
               prompt: str,
               negative_prompt: Optional[str] = None,
               strength: float = 0.75,
               num_inference_steps: int = 50,
               guidance_scale: float = 7.5,
               controlnet_conditioning_scale: Optional[float] = None,
               **kwargs) -> Dict[str, Any]:
        """
        Process image using FLUX model
        
        Args:
            image: Image to process
            mask_image: Processing mask
            prompt: Editing prompt
            negative_prompt: Negative prompt (optional)
            strength: Editing strength (0.0-1.0)
            num_inference_steps: Number of inference steps
            guidance_scale: Guidance scale factor
            controlnet_conditioning_scale: ControlNet conditioning scale
            **kwargs: Additional pipeline arguments
            
        Returns:
            Dictionary with processing results
        """
        if not self.is_loaded:
            if not self.load():
                logger.error(f"Cannot process: model '{self.model_id}' failed to load")
                return {
                    'result': image,
                    'success': False,
                    'message': f"Model '{self.model_id}' failed to load"
                }
        
        # Convert image to PIL if numpy array
        if isinstance(image, np.ndarray):
            image_pil = Image.fromarray(image)
        else:
            image_pil = image
        
        # Convert mask to PIL if numpy array
        if isinstance(mask_image, np.ndarray):
            mask_pil = Image.fromarray(mask_image)
        else:
            mask_pil = mask_image
        
        # Prepare parameters
        generator = torch.Generator(device=self.device).manual_seed(kwargs.get('seed', 42))
        
        # Set default negative prompt if not provided
        if negative_prompt is None:
            negative_prompt = ModelConfig.GENERATION_PARAMS["negative_prompt"]
        
        # Prepare controlnet arguments if available
        controlnet_args = {}
        if self.controlnet is not None and controlnet_conditioning_scale is not None:
            controlnet_args["controlnet_conditioning_scale"] = controlnet_conditioning_scale
        
        try:
            # Generate result
            result = self.pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=image_pil,
                mask_image=mask_pil,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                strength=strength,
                generator=generator,
                **controlnet_args,
                **kwargs
            )
            
            # Return result
            return {
                'result': result.images[0],
                'success': True,
                'message': "Processing completed successfully"
            }
            
        except Exception as e:
            logger.error(f"Error processing with FLUX model '{self.model_id}': {str(e)}")
            return {
                'result': image_pil,
                'success': False,
                'message': f"Error: {str(e)}"
            }
    
    def get_info(self) -> Dict[str, Any]:
        """
        Get detailed model information
        
        Returns:
            Dictionary with detailed information
        """
        info = super().get_info()
        
        # Add FLUX-specific information
        info.update({
            "config": self.config,
            "has_controlnet": self.controlnet is not None,
            "lora_weights": self.lora_weights,
            "vram_usage": torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        })
        
        return info