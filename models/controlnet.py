#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ControlNet support implementation for FusionFrame 2.0
"""

import torch
import logging
from typing import Dict, Any, Optional, Union
from PIL import Image
import numpy as np
import cv2

from config.app_config import AppConfig
from config.model_config import ModelConfig
from models.base_model import BaseModel

# Set up logger
logger = logging.getLogger(__name__)

class ControlNetHandler(BaseModel):
    """
    Handler for interacting with ControlNet models
    
    Responsible for loading and applying ControlNet models
    to guide the image generation process.
    """
    
    def __init__(self, 
                model_id: str = ModelConfig.CONTROLNET_CONFIG["model_id"], 
                device: Optional[str] = None):
        """
        Initialization for ControlNet handler
        
        Args:
            model_id: ControlNet model identifier 
            device: Device where the model will run (default from AppConfig)
        """
        super().__init__(model_id, device)
        
        self.config = ModelConfig.CONTROLNET_CONFIG
        self.conditioning_scale = self.config.get("conditioning_scale", 0.7)
    
    def load(self) -> bool:
        """
        Load ControlNet model
        
        Returns:
            True if loading succeeded, False otherwise
        """
        logger.info(f"Loading ControlNet model '{self.model_id}'")
        
        try:
            from diffusers import ControlNetModel
            
            # Load model
            self.model = ControlNetModel.from_pretrained(
                self.model_id,
                torch_dtype=AppConfig.DTYPE,
                use_safetensors=True,
                variant="fp16" if AppConfig.DTYPE == torch.float16 else None,
                cache_dir=AppConfig.CACHE_DIR
            ).to(self.device)
            
            self.is_loaded = True
            logger.info(f"ControlNet model '{self.model_id}' loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error loading ControlNet model '{self.model_id}': {str(e)}")
            self.is_loaded = False
            return False
    
    def unload(self) -> bool:
        """
        Unload model from memory
        
        Returns:
            True if unloading succeeded, False otherwise
        """
        if not self.is_loaded:
            return True
            
        try:
            # Unload model
            self.model = None
            
            # Clear memory
            torch.cuda.empty_cache()
            
            self.is_loaded = False
            logger.info(f"ControlNet model '{self.model_id}' unloaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error unloading ControlNet model '{self.model_id}': {str(e)}")
            return False
    
    def process(self,
               image: Union[Image.Image, np.ndarray],
               control_mode: str = "canny",
               **kwargs) -> Dict[str, Any]:
        """
        Process image to create control image
        
        Args:
            image: Image to process
            control_mode: Processing mode (canny, depth, normal, etc.)
            **kwargs: Additional processing arguments
            
        Returns:
            Dictionary with resulting control image
        """
        if not self.is_loaded:
            if not self.load():
                logger.error(f"Cannot process: model '{self.model_id}' failed to load")
                return {
                    'control_image': None,
                    'success': False,
                    'message': f"Model '{self.model_id}' failed to load"
                }
        
        # Convert image to numpy array
        if isinstance(image, Image.Image):
            image_np = np.array(image)
        else:
            image_np = image
        
        # Process image based on selected mode
        try:
            if control_mode == "canny":
                control_image = self._process_canny(image_np, **kwargs)
            elif control_mode == "depth":
                control_image = self._process_depth(image_np, **kwargs)
            elif control_mode == "normal":
                control_image = self._process_normal(image_np, **kwargs)
            else:
                logger.warning(f"Unknown control mode '{control_mode}'. Using canny as default.")
                control_image = self._process_canny(image_np, **kwargs)
            
            # Convert to PIL for Diffusers compatibility
            if isinstance(control_image, np.ndarray):
                control_image = Image.fromarray(control_image)
            
            return {
                'control_image': control_image,
                'success': True,
                'message': f"Control image created successfully using {control_mode} mode"
            }
            
        except Exception as e:
            logger.error(f"Error creating control image: {str(e)}")
            return {
                'control_image': None,
                'success': False,
                'message': f"Error: {str(e)}"
            }
    
    def _process_canny(self, image: np.ndarray, low_threshold: int = 100, 
                      high_threshold: int = 200) -> np.ndarray:
        """
        Process image using Canny edge detector
        
        Args:
            image: Image to process
            low_threshold: Lower threshold for Canny
            high_threshold: Upper threshold for Canny
            
        Returns:
            Processed control image
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Apply blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply Canny edge detection
        edges = cv2.Canny(blurred, low_threshold, high_threshold)
        
        # Convert back to RGB for compatibility
        edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        
        return edges_rgb
    
    def _process_depth(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """
        Process image for depth map-based control
        
        Args:
            image: Image to process
            **kwargs: Additional processing arguments
            
        Returns:
            Depth map for input image
        """
        try:
            # Import required library
            from transformers import pipeline
            
            # Convert to PIL
            image_pil = Image.fromarray(image)
            
            # Get depth estimation pipeline
            depth_estimator = pipeline("depth-estimation", model="Intel/dpt-large")
            
            # Calculate depth map
            depth_result = depth_estimator(image_pil)
            depth_map = depth_result["depth"]
            
            # Convert to numpy and adjust for visualization
            depth_np = np.array(depth_map)
            depth_np = (depth_np - depth_np.min()) / (depth_np.max() - depth_np.min()) * 255
            depth_np = depth_np.astype(np.uint8)
            
            # Convert to RGB
            depth_rgb = cv2.cvtColor(depth_np, cv2.COLOR_GRAY2RGB)
            
            return depth_rgb
            
        except ImportError:
            logger.warning("transformers depth-estimation not available. Using canny fallback.")
            return self._process_canny(image)
        except Exception as e:
            logger.error(f"Error in depth processing: {str(e)}")
            logger.warning("Using canny fallback.")
            return self._process_canny(image)
    
    def _process_normal(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """
        Process image for normal map-based control
        
        Args:
            image: Image to process
            **kwargs: Additional processing arguments
            
        Returns:
            Normal map for input image
        """
        try:
            # Import required library
            from transformers import pipeline
            
            # Convert to PIL
            image_pil = Image.fromarray(image)
            
            # Get normal estimation pipeline
            normal_estimator = pipeline("depth-estimation", model="Intel/dpt-hybrid-midas")
            
            # Calculate normal map from depth map
            depth_result = normal_estimator(image_pil)
            depth_map = depth_result["depth"]
            
            # Convert to numpy
            depth_np = np.array(depth_map)
            
            # Calculate gradients to get normal map
            sobelx = cv2.Sobel(depth_np, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(depth_np, cv2.CV_64F, 0, 1, ksize=3)
            
            # Normalize gradients
            sobelx = cv2.normalize(sobelx, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            sobely = cv2.normalize(sobely, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            
            # Combine into RGB normal map
            normal_map = np.zeros((depth_np.shape[0], depth_np.shape[1], 3), dtype=np.uint8)
            normal_map[:, :, 0] = sobelx  # R channel
            normal_map[:, :, 1] = sobely  # G channel
            normal_map[:, :, 2] = 255  # B channel (constant)
            
            return normal_map
            
        except ImportError:
            logger.warning("transformers depth-estimation not available. Using canny fallback.")
            return self._process_canny(image)
        except Exception as e:
            logger.error(f"Error in normal processing: {str(e)}")
            logger.warning("Using canny fallback.")
            return self._process_canny(image)