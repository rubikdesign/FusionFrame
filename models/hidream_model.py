#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Implementation for HiDream model in FusionFrame 2.0 (Corrected Version)

Note: HiDream models require a recent version of diffusers installed from source:
pip install git+https://github.com/huggingface/diffusers.git
"""

import os
import torch
import logging
from typing import Dict, Any, Optional, List, Union
from PIL import Image
import numpy as np

from config.app_config import AppConfig
from config.model_config import ModelConfig
from models.base_model import BaseModel

# Required imports for HiDream-I1 and Diffusers
from transformers import LlamaForCausalLM, PreTrainedTokenizerFast # For HiDream-I1 text encoders
from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    DiffusionPipeline, # Used as fallback
    FlowMatchEulerDiscreteScheduler, # Recommended scheduler for HiDream
    StableDiffusionXLImg2ImgPipeline # For SDXL Refiner
)
# HiDreamImageTransformer2DModel is typically an internal component of the HiDream pipeline
# or can be loaded separately if the HiDream structure specifically requires it.

# Set up logger
logger = logging.getLogger(__name__)

class HiDreamModel(BaseModel):
    """
    Wrapper for HiDream-I1 model with optional refiner.
    On first load, models will be automatically downloaded if not present locally.
    
    HiDream models require Llama model to be loaded to work correctly.
    Ensure you have access to "meta-llama/Meta-Llama-3.1-8B-Instruct".
    """
    def __init__(
        self,
        model_id: str = ModelConfig.MAIN_MODEL,
        device: Optional[str] = None,
        use_refiner: Optional[bool] = None,
    ):
        super().__init__(model_id, device)
        # Explicitly add is_loaded to prevent attribute errors
        self.is_loaded = False
        
        self.config = ModelConfig.HIDREAM_CONFIG
        self.refiner_config = ModelConfig.REFINER_CONFIG
        if use_refiner is None:
            use_refiner = AppConfig.USE_REFINER
        self.use_refiner = use_refiner and self.refiner_config.get("enabled", False)

        self.vae = None
        self.controlnet = None
        self.pipeline = None # Base type for pipeline
        self.refiner_pipeline = None # For SDXL Refiner
        self.lora_weights = [] # To track loaded LoRAs
        self.text_encoder_4 = None # Specific to HiDream-I1 (Llama)
        self.tokenizer_4 = None    # Specific to HiDream-I1 (Llama)

    def load(self) -> bool:
        logger.info(f"Loading HiDream model '{self.model_id}' with refiner={self.use_refiner}")
        PipelineClass = DiffusionPipeline # Initialize with fallback

        try:
            # Try to import specific HiDream class
            from diffusers.pipelines.hidream_image.pipeline_hidream_image import HiDreamImagePipeline
            PipelineClass = HiDreamImagePipeline
            logger.info("Successfully imported HiDreamImagePipeline.")
            has_hidream_specific_pipeline = True
        except ImportError as e:
            logger.warning(
                f"Failed to import HiDreamImagePipeline: {e}. "
                f"Falling back to generic DiffusionPipeline. "
                f"Model '{self.model_id}' might not load or work as expected if it requires HiDreamImagePipeline. "
                f"Please install diffusers from source with: pip install git+https://github.com/huggingface/diffusers.git"
            )
            has_hidream_specific_pipeline = False
            # If model_id is specific HiDream and we don't have the pipeline, should we consider it an error?
            if "HiDream-ai/" in self.config.get("pretrained_model_name_or_path", ""):
                 logger.error(f"Model {self.model_id} seems to be a HiDream model, but HiDreamImagePipeline could not be imported. Critical error.")
                 self.is_loaded = False
                 return False


        try:
            # 1. Load VAE
            logger.info(f"Loading VAE from {self.config['vae_name_or_path']}...")
            self.vae = AutoencoderKL.from_pretrained(
                self.config["vae_name_or_path"],
                torch_dtype=AppConfig.DTYPE,
                cache_dir=AppConfig.CACHE_DIR,
            ).to(self.device)
            logger.info("VAE loaded successfully.")

            # 2. Load ControlNet if configured
            if ModelConfig.CONTROLNET_CONFIG and ModelConfig.CONTROLNET_CONFIG.get("model_id"):
                logger.info(f"Loading ControlNet from {ModelConfig.CONTROLNET_CONFIG['model_id']}...")
                try:
                    self.controlnet = ControlNetModel.from_pretrained(
                        ModelConfig.CONTROLNET_CONFIG["model_id"],
                        torch_dtype=AppConfig.DTYPE,
                        use_safetensors=True,
                        variant="fp16" if AppConfig.DTYPE == torch.float16 else None,
                        cache_dir=AppConfig.CACHE_DIR,
                    ).to(self.device)
                    logger.info("ControlNet loaded successfully.")
                except Exception as e_ctrl:
                    logger.warning(f"ControlNet download/load failed: {e_ctrl}, continuing without it.")
                    self.controlnet = None
            else:
                logger.info("ControlNet not configured or model_id missing.")
                self.controlnet = None

            # 3. Load specific Text Encoders for HiDream-I1 (Llama)
            # These are needed ONLY if using HiDreamImagePipeline
            if has_hidream_specific_pipeline and "HiDream-ai/HiDream-I1" in self.config.get("pretrained_model_name_or_path", ""):
                try:
                    # Llama model name should be configurable or inferred
                    llama_model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct" # Ensure this is correct
                    logger.info(f"Loading Llama tokenizer_4 from {llama_model_name} for HiDream-I1...")
                    self.tokenizer_4 = PreTrainedTokenizerFast.from_pretrained(
                        llama_model_name, cache_dir=AppConfig.CACHE_DIR
                    )
                    logger.info(f"Loading Llama text_encoder_4 from {llama_model_name} for HiDream-I1...")
                    self.text_encoder_4 = LlamaForCausalLM.from_pretrained(
                        llama_model_name,
                        output_hidden_states=True,  # Required for HiDream
                        output_attentions=True,     # Required for HiDream
                        torch_dtype=AppConfig.DTYPE, # or torch.bfloat16 if supported and preferred
                        cache_dir=AppConfig.CACHE_DIR,
                    ).to(self.device)
                    
                    logger.info("Llama text encoder and tokenizer for HiDream-I1 loaded successfully.")
                except Exception as e_llama:
                    logger.error(f"Failed to load Llama text encoder/tokenizer for HiDream-I1: {e_llama}. "
                                 "HiDream-I1 pipeline might not function correctly.", exc_info=True)
                    # This is likely critical for HiDream-I1
                    self.is_loaded = False
                    return False

            # 4. Load main pipeline
            logger.info(f"Loading main pipeline '{self.config['pretrained_model_name_or_path']}' using {PipelineClass.__name__}...")
            
            # Prepare pipeline parameters
            pipeline_kwargs = {
                "vae": self.vae,
                "torch_dtype": AppConfig.DTYPE,
                "use_safetensors": self.config.get("use_safetensors", True), # Default to True
                "cache_dir": AppConfig.CACHE_DIR,
            }
            
            # Add tokenizer_4 and text_encoder_4 if available
            if self.tokenizer_4 and self.text_encoder_4:
                pipeline_kwargs["tokenizer_4"] = self.tokenizer_4
                pipeline_kwargs["text_encoder_4"] = self.text_encoder_4
            
            if self.controlnet:
                pipeline_kwargs["controlnet"] = self.controlnet
            
            self.pipeline = PipelineClass.from_pretrained(
                self.config["pretrained_model_name_or_path"],
                **pipeline_kwargs,
            )
            logger.info("Main pipeline loaded successfully.")

            # 5. VRAM optimizations (after loading pipeline)
            if self.pipeline: # Verify pipeline was loaded
                if AppConfig.LOW_VRAM_MODE:
                    logger.info("Enabling model CPU offload for main pipeline.")
                    self.pipeline.enable_model_cpu_offload()
                    
                    # Enable VAE slicing to save memory
                    logger.info("Enabling VAE slicing for memory optimization.")
                    self.pipeline.enable_vae_slicing()
                    
                    # Optionally enable tiling for very large images
                    if hasattr(AppConfig, 'ENABLE_VAE_TILING') and AppConfig.ENABLE_VAE_TILING:
                        logger.info("Enabling VAE tiling for large images.")
                        self.pipeline.enable_vae_tiling()
                else:
                    logger.info("Moving main pipeline to device.")
                    self.pipeline.to(self.device)

                # 6. Configure optimized scheduler
                logger.info("Configuring FlowMatchEulerDiscreteScheduler for main pipeline.")
                # Use FlowMatchEulerDiscreteScheduler instead of DPMSolverMultistepScheduler
                # because this is the recommended scheduler for HiDream
                try:
                    self.pipeline.scheduler = FlowMatchEulerDiscreteScheduler.from_config(
                        self.pipeline.scheduler.config,
                    )
                    logger.info("Scheduler configured as FlowMatchEulerDiscreteScheduler.")
                except Exception as e_scheduler:
                    logger.warning(f"Failed to set FlowMatchEulerDiscreteScheduler: {e_scheduler}. "
                                  "Using default scheduler.")
            else:
                logger.error("Main pipeline is None, cannot apply VRAM optimizations or scheduler.")
                self.is_loaded = False
                return False


            # 7. Load Refiner Pipeline (if enabled and configured)
            if self.use_refiner and self.refiner_config.get("pretrained_model_name_or_path"):
                logger.info(f"Loading Refiner pipeline from {self.refiner_config['pretrained_model_name_or_path']}...")
                try:
                    # VAE can be reused or a refiner-specific one can be loaded if needed
                    refiner_vae = self.vae # Reuse main VAE
                    # refiner_vae = AutoencoderKL.from_pretrained(self.refiner_config["vae_name_or_path"] or self.config["vae_name_or_path"], torch_dtype=AppConfig.DTYPE).to(self.device)

                    self.refiner_pipeline = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                        self.refiner_config["pretrained_model_name_or_path"],
                        vae=refiner_vae,
                        torch_dtype=AppConfig.DTYPE,
                        use_safetensors=self.refiner_config.get("use_safetensors", True),
                        cache_dir=AppConfig.CACHE_DIR,
                    )
                    
                    if AppConfig.LOW_VRAM_MODE:
                        logger.info("Enabling model CPU offload for refiner pipeline.")
                        self.refiner_pipeline.enable_model_cpu_offload()
                        
                        # Enable VAE optimizations for refiner too
                        self.refiner_pipeline.enable_vae_slicing()
                        if hasattr(AppConfig, 'ENABLE_VAE_TILING') and AppConfig.ENABLE_VAE_TILING:
                            self.refiner_pipeline.enable_vae_tiling()
                    else:
                        logger.info("Moving refiner pipeline to device.")
                        self.refiner_pipeline.to(self.device)
                    logger.info("Refiner pipeline loaded successfully.")

                except Exception as e_refiner:
                    logger.error(f"Failed to load Refiner pipeline: {e_refiner}. Disabling refiner.", exc_info=True)
                    self.use_refiner = False
                    self.refiner_pipeline = None
            else:
                logger.info("Refiner not used or not configured with a model path.")
                self.use_refiner = False
                self.refiner_pipeline = None


            # 8. Load configured LoRAs (if main pipeline exists)
            if self.pipeline and self.config.get("lora_weights"):
                self._load_loras()

            self.model = self.pipeline # Main model is the pipeline
            self.is_loaded = True
            logger.info(f"HiDream model '{self.model_id}' and components loaded successfully.")
            return True

        except ImportError as ie:
            logger.error(f"ImportError during model loading: {ie}. "
                         "Ensure 'diffusers' is installed correctly (pip install git+https://github.com/huggingface/diffusers.git) "
                         "and all dependencies (like 'transformers', 'flash-attn') are met.", exc_info=True)
            self.is_loaded = False
            return False
        except Exception as e:
            logger.error(f"Generic error loading HiDream model: {e}", exc_info=True)
            self.is_loaded = False
            return False

    def _load_loras(self) -> None:
        """Load configured LoRAs into the main pipeline."""
        if not self.pipeline:
            logger.warning("Main pipeline not loaded, cannot load LoRAs.")
            return

        logger.info("Loading LoRA weights...")
        for lora_info in self.config.get("lora_weights", []):
            lora_path = lora_info.get("path")
            lora_name = lora_info.get("name", os.path.basename(lora_path) if lora_path else "unknown_lora")
            weight_name = lora_info.get("weight_name") # e.g. "pytorch_lora_weights.safetensors" or None for auto-detect

            if not lora_path:
                logger.warning(f"Skipping LoRA '{lora_name}' due to missing path.")
                continue
            
            logger.info(f"Loading LoRA '{lora_name}' from path '{lora_path}' with weight_name '{weight_name}'.")
            try:
                # load_lora_weights can be directly on pipeline or on unet/text_encoder
                # For Diffusers >0.17, it's directly on pipeline
                self.pipeline.load_lora_weights(
                    lora_path, # Can be a directory or .safetensors file
                    weight_name=weight_name, # Specific filename in directory if lora_path is a directory
                    adapter_name=lora_name # Internal name for adapter
                )
                
                self.lora_weights.append(lora_info)
                logger.info(f"Successfully loaded LoRA '{lora_name}'.")
            except Exception as e_lora:
                logger.warning(f"Could not load LoRA '{lora_name}' from '{lora_path}': {e_lora}", exc_info=True)
        
        if self.lora_weights:
             # After loading all LoRAs, set which adapters are active and with what weights
             # Example: activate all loaded LoRAs with specified weights (default 1.0)
            active_adapters = [lora.get("name", os.path.basename(lora.get("path"))) for lora in self.lora_weights]
            adapter_weights = [lora.get("weight", 1.0) for lora in self.lora_weights]
            try:
                self.pipeline.set_adapters(active_adapters, adapter_weights=adapter_weights)
                logger.info(f"Set active LoRA adapters: {active_adapters} with weights: {adapter_weights}")
            except Exception as e_set_adapters:
                logger.error(f"Failed to set active LoRA adapters: {e_set_adapters}")


    def unload(self) -> bool:
        if not self.is_loaded:
            logger.info("Model already unloaded or was never loaded.")
            return True
        
        logger.info(f"Unloading HiDream model '{self.model_id}' and components...")
        del self.pipeline
        del self.vae
        if self.controlnet:
            del self.controlnet
        if self.refiner_pipeline:
            del self.refiner_pipeline
        if self.text_encoder_4:
            del self.text_encoder_4
        if self.tokenizer_4:
            del self.tokenizer_4
        
        self.pipeline = None
        self.vae = None
        self.controlnet = None
        self.refiner_pipeline = None
        self.text_encoder_4 = None
        self.tokenizer_4 = None
        self.model = None # Ensure self.model is also cleaned
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("CUDA cache emptied.")
            
        self.is_loaded = False
        logger.info(f"HiDream model '{self.model_id}' unloaded successfully.")
        return True

    def process(
        self,
        image: Union[Image.Image, np.ndarray],
        mask_image: Union[Image.Image, np.ndarray],
        prompt: str,
        negative_prompt: Optional[str] = None,
        strength: float = 0.75,
        num_inference_steps: Optional[int] = None,
        guidance_scale: float = 7.5,
        controlnet_conditioning_scale: Optional[float] = None,
        refiner_strength: Optional[float] = None,
        seed: int = 42,
        **kwargs, # Allows passing other pipeline-specific parameters
    ) -> Dict[str, Any]:
        if not self.is_loaded:
            logger.warning("Model not loaded. Attempting to load...")
            if not self.load():
                return {"result": image, "success": False, "message": "Model failed to load on demand."}
        
        if not self.pipeline: # Additional check
             return {"result": image, "success": False, "message": "Main pipeline is not available."}

        image_pil = Image.fromarray(image) if isinstance(image, np.ndarray) else image.convert("RGB")
        mask_pil = Image.fromarray(mask_image) if isinstance(mask_image, np.ndarray) else mask_image.convert("L")

        # Ensure seed is correctly configured
        gen = torch.Generator(device=self.device).manual_seed(seed)
        
        effective_num_inference_steps = (
            num_inference_steps
            if num_inference_steps is not None
            else self.config.get("inference_steps", ModelConfig.GENERATION_PARAMS["default_steps"])
        )
        
        effective_refiner_strength = refiner_strength if refiner_strength is not None else AppConfig.REFINER_STRENGTH
        
        effective_negative_prompt = negative_prompt if negative_prompt is not None else ModelConfig.GENERATION_PARAMS.get("negative_prompt", "")

        # Prepare pipeline arguments
        pipeline_args = {
            "prompt": prompt,
            "negative_prompt": effective_negative_prompt,
            "image": image_pil,
            "mask_image": mask_pil,
            "num_inference_steps": effective_num_inference_steps,
            "guidance_scale": guidance_scale,
            "strength": strength, # For img2img / inpainting
            "generator": gen,
        }

        if self.controlnet and controlnet_conditioning_scale is not None:
            # For ControlNet, add specific parameters
            pipeline_args["controlnet_conditioning_scale"] = controlnet_conditioning_scale
            logger.info(f"Using ControlNet with scale: {controlnet_conditioning_scale}")

        # Add kwargs to pipeline arguments, allowing flexibility
        pipeline_args.update(kwargs)

        try:
            logger.info(f"Processing with main pipeline. Steps: {effective_num_inference_steps}, Guidance: {guidance_scale}, Strength: {strength}")
            out = self.pipeline(**pipeline_args)
            result_img = out.images[0]

            if self.use_refiner and self.refiner_pipeline:
                logger.info(f"Applying Refiner. Strength: {effective_refiner_strength}")
                # For refiner, strength has a different meaning (how much to modify the image from base)
                # Number of steps for refiner can be different
                refiner_steps = self.refiner_config.get("inference_steps", max(10, effective_num_inference_steps // 3))
                
                refiner_args = {
                    "prompt": prompt, # Refiner can use the same prompt
                    "negative_prompt": effective_negative_prompt,
                    "image": result_img, # Image generated by main pipeline
                    "num_inference_steps": refiner_steps,
                    "guidance_scale": guidance_scale, # Can be adjusted for refiner
                    "strength": effective_refiner_strength, # Control refiner effect
                    "generator": gen, # Reuse generator for consistency
                }

                out_refined = self.refiner_pipeline(**refiner_args)
                result_img = out_refined.images[0]
                logger.info("Refiner applied successfully.")

            return {"result": result_img, "success": True, "message": "Processing completed successfully"}
        
        except Exception as e:
            logger.error(f"Processing error: {e}", exc_info=True)
            return {"result": image_pil, "success": False, "message": str(e)}

    def get_info(self) -> Dict[str, Any]:
        info = super().get_info()
        info.update({
            "config_main_model": self.config,
            "config_refiner_model": self.refiner_config if self.use_refiner else "Not Used",
            "has_controlnet": bool(self.controlnet),
            "using_refiner": self.use_refiner and bool(self.refiner_pipeline),
            "active_loras": self.lora_weights,
            "main_pipeline_class": self.pipeline.__class__.__name__ if self.pipeline else "N/A",
            "refiner_pipeline_class": self.refiner_pipeline.__class__.__name__ if self.refiner_pipeline else "N/A",
            "device": self.device,
            "dtype": str(AppConfig.DTYPE),
            "low_vram_mode": AppConfig.LOW_VRAM_MODE,
        })
        if torch.cuda.is_available():
            info["vram_allocated_gb"] = round(torch.cuda.memory_allocated(self.device) / (1024 ** 3), 2)
            info["vram_reserved_gb"] = round(torch.cuda.memory_reserved(self.device) / (1024 ** 3), 2)
        return info