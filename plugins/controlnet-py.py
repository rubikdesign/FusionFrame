"""
ControlNet plugin for FusionFrame application.

This module provides functions to integrate ControlNet with Stable Diffusion
pipelines for pose-guided image generation.
"""

import os
import torch
import logging
from typing import Dict, Any, Tuple, Optional, Union
from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)

class ControlNetPlugin:
    """
    ControlNet integration for pose-guided image generation.
    
    This class provides a clean interface to use ControlNet (especially OpenPose)
    with Stable Diffusion pipelines.
    """
    
    def __init__(self, cache_dir: str):
        """
        Initialize the ControlNet plugin.
        
        Args:
            cache_dir (str): Directory for cached models
        """
        self.cache_dir = cache_dir
        self.controlnet = None
        self.pipe = None
        self.detector = None
        self.is_available = self._check_availability()
        
    def _check_availability(self) -> bool:
        """
        Check if ControlNet is available (libraries installed).
        
        Returns:
            bool: True if ControlNet is available, False otherwise
        """
        try:
            from diffusers import ControlNetModel
            from controlnet_aux import OpenposeDetector
            logger.info("ControlNet libraries are available")
            return True
        except ImportError:
            logger.warning("ControlNet libraries not found. Pose guidance will not be available.")
            logger.info("Install with: pip install controlnet_aux diffusers")
            return False
    
    def setup(self, base_pipeline, device: str, model_id: str = None) -> bool:
        """
        Set up ControlNet with the given base pipeline.
        
        Args:
            base_pipeline: Base Diffusers pipeline to use with ControlNet
            device (str): Device to use (cuda or cpu)
            model_id (str, optional): Model ID for ControlNet. If None, will try recommended sources
            
        Returns:
            bool: True if setup was successful, False otherwise
        """
        if not self.is_available:
            return False
            
        try:
            from diffusers import ControlNetModel, StableDiffusionXLControlNetImg2ImgPipeline
            from controlnet_aux import OpenposeDetector
            
            # If no model_id provided, try these sources in order
            model_sources = [
                "lllyasviel/sd-controlnet-openpose",  # First choice
                "thibaud/controlnet-openpose-sdxl-1.0",
                "diffusers/controlnet-openpose-sdxl",
            ] if model_id is None else [model_id]
            
            # Try each source until one works
            success = False
            for source in model_sources:
                try:
                    logger.info(f"Trying to load ControlNet from: {source}")
                    self.controlnet = ControlNetModel.from_pretrained(
                        source,
                        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                        cache_dir=self.cache_dir,
                        use_safetensors=True
                    ).to(device)
                    
                    # Create pipeline using the current model's components
                    self.pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
                        # Use the same model ID as the base pipeline
                        pretrained_model_name_or_path=base_pipeline.config._name_or_path 
                            if hasattr(base_pipeline, "config") and hasattr(base_pipeline.config, "_name_or_path") 
                            else "stabilityai/stable-diffusion-xl-refiner-1.0",
                        controlnet=self.controlnet,
                        # Reuse components from base pipeline
                        vae=base_pipeline.vae,
                        text_encoder=base_pipeline.text_encoder,
                        tokenizer=base_pipeline.tokenizer,
                        unet=base_pipeline.unet,
                        scheduler=base_pipeline.scheduler,
                        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    ).to(device)
                    
                    # IMPORTANT: Fix for time embedding vector issue
                    # Use register_to_config method to properly disable the aesthetics score
                    self.pipe.register_to_config(requires_aesthetics_score=False)
                    
                    # Create pose detector
                    self.detector = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
                    
                    # Try to enable xformers if available
                    if torch.cuda.is_available():
                        try:
                            self.pipe.enable_xformers_memory_efficient_attention()
                            logger.info("xFormers enabled for ControlNet")
                        except Exception as e:
                            logger.warning(f"Could not enable xFormers for ControlNet: {e}")
                    
                    success = True
                    logger.info(f"Successfully loaded ControlNet from: {source}")
                    break
                except Exception as e:
                    logger.warning(f"Failed to load from {source}: {e}")
            
            if not success:
                logger.error("Failed to load ControlNet from any source")
                return False
                
            return True
        except Exception as e:
            logger.error(f"Error setting up ControlNet: {e}")
            return False
    
    def detect_pose(self, image: Union[Image.Image, np.ndarray]) -> Optional[Image.Image]:
        """
        Detect pose in an image using OpenPose.
        
        Args:
            image: Input image (PIL Image or numpy array)
            
        Returns:
            PIL.Image: Image with detected pose, or None if detection failed
        """
        if self.detector is None:
            logger.error("Pose detector not initialized")
            return None
            
        try:
            # Convert to numpy if PIL Image
            if isinstance(image, Image.Image):
                image_np = np.array(image)
            else:
                image_np = image
                
            # Detect pose
            pose_image = self.detector(image_np)
            logger.info("Pose detection successful")
            
            return pose_image
        except Exception as e:
            logger.error(f"Error during pose detection: {e}")
            return None
    
    def generate(
        self, 
        prompt: str, 
        negative_prompt: str, 
        input_image: Image.Image, 
        pose_image: Image.Image, 
        strength: float = 0.75, 
        guidance_scale: float = 7.5, 
        controlnet_conditioning_scale: Union[float, str] = 1.0,
        num_inference_steps: int = 30, 
        generator = None, 
        callback = None
    ) -> Image.Image:
        """
        Generate an image with ControlNet for pose guidance.
        
        Args:
            prompt (str): Text prompt for generation
            negative_prompt (str): Negative text prompt
            input_image (PIL.Image): Input image to transform
            pose_image (PIL.Image): Pose reference image or detected pose
            strength (float): Transformation strength (0-1)
            guidance_scale (float): Classifier-free guidance scale
            controlnet_conditioning_scale (float or str): How much to follow the pose control
            num_inference_steps (int): Number of denoising steps
            generator: Random number generator
            callback: Progress callback function
            
        Returns:
            PIL.Image: Generated image
        """
        if not self.pipe:
            logger.error("ControlNet pipeline not initialized")
            return input_image
            
        try:
            # Ensure all numerical parameters are correct types
            strength_float = float(strength)
            guidance_scale_float = float(guidance_scale)
            num_steps_int = int(num_inference_steps)
            
            # Ensure controlnet_conditioning_scale is a float
            if isinstance(controlnet_conditioning_scale, str):
                controlnet_conditioning_scale_float = float(controlnet_conditioning_scale)
            else:
                controlnet_conditioning_scale_float = float(controlnet_conditioning_scale)
                
            # Run generation with controlnet
            result = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=input_image,
                control_image=pose_image,
                strength=strength_float,
                guidance_scale=guidance_scale_float,
                controlnet_conditioning_scale=controlnet_conditioning_scale_float,
                num_inference_steps=num_steps_int,
                generator=generator,
                callback=callback,
                callback_steps=1
            ).images[0]
            
            return result
        except Exception as e:
            logger.error(f"Error during ControlNet generation: {e}")
            return input_image