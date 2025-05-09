#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Enhanced Implementation for FLUX model in FusionFrame 2.0
- Optimized for memory efficiency
- Support for ControlNet and LoRA
- Better error handling and memory management
"""

import os
import torch
import logging
import gc
import time
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
    Enhanced implementation for FLUX model
    
    FLUX is a lighter alternative to HiDream that doesn't require
    LLama text encoders and uses significantly less GPU memory while
    maintaining high-quality image generation capabilities.
    """
    
    def __init__(self, 
                model_id: str = ModelConfig.MAIN_MODEL, 
                device: Optional[str] = None):
        """
        Initialization for FLUX model
        
        Args:
            model_id: Model identifier (default from ModelConfig.MAIN_MODEL)
            device: Device where the model will run (default from AppConfig)
        """
        super().__init__(model_id, device)
        
        self.config = ModelConfig.FLUX_CONFIG
        self.vae = None
        self.controlnet = None
        self.pipeline = None
        self.lora_weights = []
        self.is_loaded = False


    def load(self) -> bool:
        """
        Load FLUX model with memory optimizations using FluxPipeline.

        Returns:
            True if loading succeeded, False otherwise
        """
        logger.info(f"Loading FLUX model '{self.model_id}' with FluxPipeline")
        start_time = time.time()

        # Clean GPU memory before loading
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            logger.info(f"CUDA memory before FLUX load: {torch.cuda.memory_allocated(self.device) / 1024**3:.2f} GB used, {torch.cuda.memory_reserved(self.device) / 1024**3:.2f} GB reserved")

        try:
            from diffusers import FluxPipeline, AutoencoderKL, ControlNetModel
            # Scheduler-ul este de obicei gestionat intern de FluxPipeline sau configurat specific.
            # Nu mai importăm EulerAncestralDiscreteScheduler aici pentru a-l seta manual inițial.

            # Determine effective torch_dtype (bfloat16 is preferred for FLUX if supported)
            effective_torch_dtype = AppConfig.DTYPE
            if str(AppConfig.DTYPE).lower() == "torch.bfloat16": # Dacă AppConfig.DTYPE este setat la bfloat16
                if AppConfig.DEVICE == "cuda" and not torch.cuda.is_bf16_supported():
                    logger.warning("Configured DTYPE is bfloat16, but it's not supported on this GPU. Falling back to float32 for FluxPipeline.")
                    effective_torch_dtype = torch.float32
                elif AppConfig.DEVICE == "cpu":
                    logger.warning("Configured DTYPE is bfloat16, which is not recommended for CPU. Using float32 for FluxPipeline.")
                    effective_torch_dtype = torch.float32
                else:
                    effective_torch_dtype = torch.bfloat16 # Confirmă utilizarea bfloat16
            elif str(AppConfig.DTYPE).lower() == "torch.float16":
                 # FLUX e optimizat pentru bfloat16, float16 poate merge dar bfloat16 e preferat.
                 logger.info(f"AppConfig.DTYPE is {AppConfig.DTYPE}. FLUX documentation often suggests bfloat16.")
                 effective_torch_dtype = torch.float16
            else: # Default la float32
                effective_torch_dtype = torch.float32

            logger.info(f"Attempting to load FLUX pipeline '{self.config['pretrained_model_name_or_path']}' with dtype: {effective_torch_dtype}")

            # 1. Load VAE (opțional explicit, FluxPipeline ar putea să-l gestioneze)
            # FluxPipeline ar trebui să poată încărca VAE-ul specificat în configurația sa internă
            # sau din `self.config["vae_name_or_path"]` dacă este necesar și suportat.
            # Pentru FLUX.1-dev, este posibil ca VAE-ul să fie parte din pachetul principal.
            # Vom încerca să încărcăm VAE-ul explicit dacă este specificat în config,
            # și îl vom pasa la pipeline dacă FluxPipeline îl acceptă.
            self.vae = None # Resetează VAE-ul intern
            if self.config.get("vae_name_or_path"):
                try:
                    logger.info(f"Loading VAE explicitly from {self.config['vae_name_or_path']}...")
                    self.vae = AutoencoderKL.from_pretrained(
                        self.config["vae_name_or_path"],
                        torch_dtype=effective_torch_dtype, # Potrivește dtype-ul
                        cache_dir=AppConfig.CACHE_DIR,
                    )
                    # Nu muta pe self.device aici; lasă pipeline-ul să gestioneze sau fă-o după încărcarea pipeline-ului.
                    logger.info("VAE loaded successfully (will be moved to device with pipeline or by offload).")
                except Exception as e_vae_load:
                    logger.error(f"Failed to load VAE from {self.config['vae_name_or_path']}: {e_vae_load}. Proceeding without explicitly passed VAE.")
                    self.vae = None


            # 2. Load ControlNet if configured (atenție la compatibilitatea cu FluxPipeline)
            self.controlnet = None # Resetează ControlNet intern
            if ModelConfig.CONTROLNET_CONFIG and ModelConfig.CONTROLNET_CONFIG.get("model_id"):
                logger.warning("ControlNet is configured, but FluxPipeline may not have direct built-in support for it "
                               "in the same way as StableDiffusion pipelines. ControlNet will be loaded but NOT passed to FluxPipeline by default.")
                try:
                    self.controlnet = ControlNetModel.from_pretrained(
                        ModelConfig.CONTROLNET_CONFIG["model_id"],
                        torch_dtype=effective_torch_dtype,
                        use_safetensors=True,
                        # variant="fp16" if effective_torch_dtype == torch.float16 else None, # Specific SD
                        cache_dir=AppConfig.CACHE_DIR,
                    ) # Nu muta pe device încă
                    logger.info("ControlNet model loaded (not passed to FluxPipeline).")
                except Exception as e_ctrl:
                    logger.warning(f"ControlNet download/load failed: {e_ctrl}, continuing without it.")
                    self.controlnet = None
            else:
                logger.info("ControlNet not configured or model_id missing.")


            # 3. Load main FLUX pipeline
            pipeline_load_args = {
                "torch_dtype": effective_torch_dtype,
                "use_safetensors": self.config.get("use_safetensors", True),
                "cache_dir": AppConfig.CACHE_DIR,
            }

            # Adaugă VAE-ul dacă a fost încărcat și dacă FluxPipeline îl acceptă ca argument.
            # Documentația FluxPipeline.from_pretrained() va clarifica acest lucru.
            # De obicei, pentru pipeline-uri mai noi, integrate, VAE-ul e încărcat din repo-ul principal.
            if self.vae:
                pipeline_load_args["vae"] = self.vae
                logger.info("VAE instance will be passed to FluxPipeline.from_pretrained.")


            self.pipeline = FluxPipeline.from_pretrained(
                self.config["pretrained_model_name_or_path"], # ex: "black-forest-labs/FLUX.1-dev"
                **pipeline_load_args
            )
            logger.info(f"FluxPipeline instance created from '{self.config['pretrained_model_name_or_path']}'.")

            # 4. Aplică optimizările de memorie și mutarea pe dispozitiv
            if AppConfig.LOW_VRAM_MODE:
                logger.info("LOW_VRAM_MODE: Attempting to enable model CPU offload for FluxPipeline.")
                try:
                    # Metoda specifică FLUX pentru offload inteligent
                    self.pipeline.enable_model_cpu_offload()
                    logger.info("FluxPipeline.enable_model_cpu_offload() called successfully.")
                    # VAE și ControlNet (dacă sunt încărcate și nu sunt gestionate de offload)
                    # ar putea necesita mutare manuală dacă nu sunt acoperite de enable_model_cpu_offload.
                    # Totuși, enable_model_cpu_offload ar trebui să fie destul de cuprinzător.
                    if self.vae and hasattr(self.vae, "to") and AppConfig.DEVICE != "cpu": # Dacă VAE e încărcat și nu e pe CPU
                        logger.info("Ensuring VAE is on CPU for LOW_VRAM_MODE after pipeline offload setup.")
                        # self.vae.to("cpu") # Comentat - enable_model_cpu_offload ar trebui să gestioneze
                    if self.controlnet and hasattr(self.controlnet, "to") and AppConfig.DEVICE != "cpu":
                         logger.info("Ensuring ControlNet is on CPU for LOW_VRAM_MODE (if loaded separately).")
                         self.controlnet.to("cpu") # ControlNet nu e parte din pipeline FLUX
                except AttributeError:
                    logger.warning("FluxPipeline instance does not have 'enable_model_cpu_offload'. "
                                   "Attempting manual offload of text encoders if applicable (though FLUX has integrated transformers).")
                    # Fallback la mutarea manuală a componentelor dacă `enable_model_cpu_offload` nu există
                    # sau dacă vrem control mai fin (deși pentru FLUX, transformerele sunt mai integrate).
                    # Acest bloc s-ar putea să nu fie necesar dacă enable_model_cpu_offload funcționează.
                    for component_name in ["text_encoder", "text_encoder_2", "transformer"]: # "transformer" e componenta mare din FLUX
                        if hasattr(self.pipeline, component_name) and getattr(self.pipeline, component_name) is not None:
                            try:
                                component = getattr(self.pipeline, component_name)
                                component.to("cpu")
                                logger.info(f"Manually moved FluxPipeline component '{component_name}' to CPU.")
                            except Exception as e_manual_offload:
                                logger.warning(f"Could not manually move '{component_name}' to CPU: {e_manual_offload}")
                except Exception as e_offload:
                    logger.error(f"Error during FluxPipeline CPU offload: {e_offload}")
            else: # Nu este LOW_VRAM_MODE, mută totul pe dispozitivul principal
                logger.info(f"Moving full FluxPipeline to device: {self.device}")
                self.pipeline.to(self.device)
                if self.vae and hasattr(self.vae, "to"): # Asigură-te că și VAE-ul (dacă e încărcat separat) e pe device
                    self.vae.to(self.device)
                if self.controlnet and hasattr(self.controlnet, "to"): # Și ControlNet (dacă e încărcat separat)
                    self.controlnet.to(self.device)


            # 5. Configurare Scheduler (Opțional - FluxPipeline ar trebui să aibă un default bun)
            # Dacă vrei să suprascrii scheduler-ul default al FluxPipeline:
            # try:
            #     from diffusers import FlowMatchEulerDiscreteScheduler # Sau alt scheduler compatibil
            #     if hasattr(self.pipeline, 'scheduler'):
            #         self.pipeline.scheduler = FlowMatchEulerDiscreteScheduler.from_config(
            #             self.pipeline.scheduler.config
            #         )
            #         logger.info(f"Scheduler for FluxPipeline configured as {self.pipeline.scheduler.__class__.__name__}.")
            #     else:
            #         logger.warning("FluxPipeline instance does not have a 'scheduler' attribute to override.")
            # except Exception as e_scheduler:
            #     logger.warning(f"Failed to set custom scheduler for FluxPipeline: {e_scheduler}. Using default.")
            logger.info(f"FluxPipeline is using its default scheduler: {self.pipeline.scheduler.__class__.__name__}")


            # 6. Load configured LoRAs (if present and FluxPipeline supports them)
            # Verifică dacă FluxPipeline are metode standard `load_lora_weights` și `set_adapters`.
            if self.config.get("lora_weights"):
                if hasattr(self.pipeline, "load_lora_weights") and hasattr(self.pipeline, "set_adapters"):
                    self._load_loras() # Metoda ta existentă _load_loras ar trebui să funcționeze
                else:
                    logger.warning("FluxPipeline instance does not seem to support LoRA loading via standard methods. Skipping LoRA loading.")


            self.model = self.pipeline # Setează modelul principal ca fiind pipeline-ul
            self.is_loaded = True

            load_time = time.time() - start_time
            logger.info(f"FLUX model '{self.model_id}' (FluxPipeline) loaded successfully in {load_time:.2f} seconds.")

            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated(self.device) / (1024**3)
                reserved = torch.cuda.memory_reserved(self.device) / (1024**3)
                logger.info(f"CUDA memory after FLUX load: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")

            return True

        except ImportError as ie:
            logger.error(f"ImportError during FLUX model loading: {ie}. "
                         "Ensure 'diffusers' is installed correctly and FluxPipeline is available.", exc_info=True)
            self.is_loaded = False
            return False
        except Exception as e:
            logger.error(f"Error loading FLUX model '{self.model_id}' with FluxPipeline: {str(e)}", exc_info=True)
            self.is_loaded = False
            return False

    
    
    
    def _load_loras(self) -> None:
        """Load configured LoRAs"""
        if not self.pipeline:
            logger.warning("Main pipeline not loaded, cannot load LoRAs.")
            return

        logger.info("Loading LoRA weights...")
        for lora_info in self.config.get("lora_weights", []):
            lora_path = lora_info.get("path")
            lora_name = lora_info.get("name", os.path.basename(lora_path) if lora_path else "unknown_lora")
            weight_name = lora_info.get("weight_name")

            if not lora_path:
                logger.warning(f"Skipping LoRA '{lora_name}' due to missing path.")
                continue
            
            logger.info(f"Loading LoRA '{lora_name}' from path '{lora_path}'.")
            try:
                self.pipeline.load_lora_weights(
                    lora_path,
                    adapter_name=lora_name,
                    weight_name=weight_name
                )
                self.lora_weights.append(lora_info)
                logger.info(f"LoRA '{lora_name}' loaded successfully")
            except Exception as e:
                logger.error(f"Error loading LoRA '{lora_name}': {str(e)}")
        
        if self.lora_weights:
            # Set active LoRAs
            active_adapters = [lora.get("name", os.path.basename(lora.get("path"))) for lora in self.lora_weights]
            adapter_weights = [lora.get("weight", 1.0) for lora in self.lora_weights]
            try:
                self.pipeline.set_adapters(active_adapters, adapter_weights=adapter_weights)
                logger.info(f"Set active LoRA adapters: {active_adapters} with weights: {adapter_weights}")
            except Exception as e_set_adapters:
                logger.error(f"Failed to set active LoRA adapters: {e_set_adapters}")
    
    def unload(self) -> bool:
        """
        Unload model from memory
        
        Returns:
            True if unloading succeeded, False otherwise
        """
        if not self.is_loaded:
            logger.info("Model already unloaded or was never loaded.")
            return True
            
        logger.info(f"Unloading FLUX model '{self.model_id}'...")
        
        try:
            # Unload components
            del self.pipeline
            del self.vae
            if self.controlnet:
                del self.controlnet
            
            self.pipeline = None
            self.vae = None
            self.controlnet = None
            self.model = None
            
            # Clear memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
                logger.info("CUDA cache emptied.")
            
            self.is_loaded = False
            logger.info(f"FLUX model '{self.model_id}' unloaded successfully.")
            return True
            
        except Exception as e:
            logger.error(f"Error unloading FLUX model '{self.model_id}': {str(e)}")
            return False
    
    # Emergency cleanup for OOM situations
    def emergency_cleanup(self) -> None:
        """
        Aggressive memory cleanup for OOM (Out of Memory) emergencies
        """
        logger.warning(f"Performing emergency memory cleanup for FLUX model '{self.model_id}'")
        
        # Move components to CPU
        if self.pipeline:
            # Move text encoders to CPU
            if hasattr(self.pipeline, "text_encoder") and self.pipeline.text_encoder is not None:
                try:
                    self.pipeline.text_encoder.to("cpu")
                    logger.info("Emergency: Moved text_encoder to CPU")
                except Exception as e_te:
                    logger.warning(f"Could not move text_encoder to CPU: {e_te}")
                
            if hasattr(self.pipeline, "text_encoder_2") and self.pipeline.text_encoder_2 is not None:
                try:
                    self.pipeline.text_encoder_2.to("cpu")
                    logger.info("Emergency: Moved text_encoder_2 to CPU")
                except Exception as e_te2:
                    logger.warning(f"Could not move text_encoder_2 to CPU: {e_te2}")
        
        # Clear cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            
        logger.info("Emergency cleanup completed")
        
        # Log memory state after cleanup
        if torch.cuda.is_available():
            allocated_gb = torch.cuda.memory_allocated(self.device) / (1024**3)
            reserved_gb = torch.cuda.memory_reserved(self.device) / (1024**3)
            logger.info(f"CUDA memory after emergency cleanup: {allocated_gb:.2f} GB allocated, {reserved_gb:.2f} GB reserved")
    
    def process(self, 
               image: Union[Image.Image, np.ndarray],
               mask_image: Union[Image.Image, np.ndarray],
               prompt: str,
               negative_prompt: Optional[str] = None,
               strength: float = 0.75,
               num_inference_steps: Optional[int] = None,
               guidance_scale: float = 7.5,
               controlnet_conditioning_scale: Optional[float] = None,
               seed: int = 42,
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
            seed: Random seed
            **kwargs: Additional pipeline arguments
            
        Returns:
            Dictionary with processing results
        """
        if not self.is_loaded:
            logger.warning("Model not loaded. Attempting to load...")
            if not self.load():
                return {"result": image, "success": False, "message": "Model failed to load on demand."}
        
        if not self.pipeline:
            return {"result": image, "success": False, "message": "Pipeline not available."}
        
        # Measure processing time
        start_time = time.time()

        # Convert input to correct format
        image_pil = Image.fromarray(image) if isinstance(image, np.ndarray) else image.convert("RGB")
        mask_pil = Image.fromarray(mask_image) if isinstance(mask_image, np.ndarray) else mask_image.convert("L")
        
        # Resize images if in LOW_VRAM_MODE and they're too large
        if AppConfig.LOW_VRAM_MODE:
            max_dim = 1024  # Maximum dimension in LOW_VRAM_MODE
            img_w, img_h = image_pil.size
            
            if max(img_w, img_h) > max_dim:
                # Calculate new dimensions preserving aspect ratio
                if img_w > img_h:
                    new_w, new_h = max_dim, int(img_h * max_dim / img_w)
                else:
                    new_w, new_h = int(img_w * max_dim / img_h), max_dim
                    
                # Resize image and mask
                logger.warning(f"Image too large in LOW_VRAM_MODE. Resizing from {img_w}x{img_h} to {new_w}x{new_h}")
                image_pil = image_pil.resize((new_w, new_h), Image.LANCZOS)
                mask_pil = mask_pil.resize((new_w, new_h), Image.NEAREST)
        
        # Configure generator for reproducibility
        generator = torch.Generator(device=self.device).manual_seed(seed)
        
        # Adjust parameters for LOW_VRAM_MODE
        if AppConfig.LOW_VRAM_MODE and hasattr(ModelConfig, "LOW_VRAM_INFERENCE_STEPS"):
            effective_num_inference_steps = ModelConfig.LOW_VRAM_INFERENCE_STEPS.get(self.model_id, 30)
            logger.info(f"Using reduced inference steps in LOW_VRAM_MODE: {effective_num_inference_steps}")
        else:
            effective_num_inference_steps = (
                num_inference_steps
                if num_inference_steps is not None
                else self.config.get("inference_steps", ModelConfig.GENERATION_PARAMS["default_steps"])
            )
        
        # Set default negative prompt if not provided
        effective_negative_prompt = negative_prompt if negative_prompt is not None else ModelConfig.GENERATION_PARAMS.get("negative_prompt", "")
        
        # Prepare pipeline arguments
        pipeline_args = {
            "prompt": prompt,
            "negative_prompt": effective_negative_prompt,
            "image": image_pil,
            "mask_image": mask_pil,
            "num_inference_steps": effective_num_inference_steps,
            "guidance_scale": guidance_scale,
            "strength": strength,
            "generator": generator,
        }
        
        # Add ControlNet parameters if available
        if self.controlnet and controlnet_conditioning_scale is not None:
            pipeline_args["controlnet_conditioning_scale"] = controlnet_conditioning_scale
            logger.info(f"Using ControlNet with scale: {controlnet_conditioning_scale}")
        
        # Add any additional parameters
        pipeline_args.update(kwargs)
        
        # Log memory state before inference
        if torch.cuda.is_available():
            allocated_before = torch.cuda.memory_allocated(self.device) / (1024**3)
            reserved_before = torch.cuda.memory_reserved(self.device) / (1024**3)
            logger.info(f"CUDA memory before inference: {allocated_before:.2f} GB allocated, {reserved_before:.2f} GB reserved")
        
        try:
            # Clean memory before inference
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
            
            logger.info(f"Processing with FLUX. Steps: {effective_num_inference_steps}, Guidance: {guidance_scale}, Strength: {strength}")
            
            # Run inference
            result = self.pipeline(**pipeline_args)
            result_img = result.images[0]
            
            # Measure total processing time
            proc_time = time.time() - start_time
            logger.info(f"Processing completed successfully in {proc_time:.2f} seconds")
            
            # Log memory state after inference
            if torch.cuda.is_available():
                allocated_after = torch.cuda.memory_allocated(self.device) / (1024**3)
                reserved_after = torch.cuda.memory_reserved(self.device) / (1024**3)
                logger.info(f"CUDA memory after inference: {allocated_after:.2f} GB allocated, {reserved_after:.2f} GB reserved")
                logger.info(f"Memory change during inference: {allocated_after-allocated_before:.2f} GB")
            
            return {
                'result': result_img,
                'success': True,
                'message': f"Processing completed successfully in {proc_time:.2f}s"
            }
            
        except RuntimeError as e:
            error_msg = str(e)
            if "CUDA out of memory" in error_msg:
                logger.error(f"CUDA out of memory during inference: {error_msg}")
                
                # Try recovery with emergency cleanup
                logger.warning("Attempting emergency cleanup and retry with reduced parameters")
                self.emergency_cleanup()
                
                # Reduce parameters for retry
                reduced_steps = max(20, effective_num_inference_steps // 2)
                reduced_guidance = min(7.0, guidance_scale)
                
                try:
                    # Prepare arguments for retry
                    retry_args = pipeline_args.copy()
                    retry_args["num_inference_steps"] = reduced_steps
                    retry_args["guidance_scale"] = reduced_guidance
                    
                    logger.info(f"Retrying with reduced parameters: steps={reduced_steps}, guidance={reduced_guidance}")
                    
                    # Retry processing
                    result = self.pipeline(**retry_args)
                    result_img = result.images[0]
                    
                    proc_time = time.time() - start_time
                    logger.info(f"Processing completed with reduced parameters in {proc_time:.2f} seconds")
                    
                    return {
                        "result": result_img, 
                        "success": True, 
                        "message": f"Processing completed with reduced parameters (steps={reduced_steps}) in {proc_time:.2f}s"
                    }
                except Exception as retry_e:
                    logger.error(f"Retry failed: {retry_e}")
                    return {
                        "result": image_pil, 
                        "success": False, 
                        "message": f"CUDA out of memory. Emergency retry also failed: {retry_e}"
                    }
            else:
                # Other types of errors
                logger.error(f"Processing error: {e}", exc_info=True)
                return {"result": image_pil, "success": False, "message": str(e)}
        except Exception as e:
            logger.error(f"Processing error: {e}", exc_info=True)
            return {"result": image_pil, "success": False, "message": str(e)}
    
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
            "device": self.device,
            "dtype": str(AppConfig.DTYPE),
            "low_vram_mode": AppConfig.LOW_VRAM_MODE,
        })
        
        if torch.cuda.is_available():
            info["vram_allocated_gb"] = round(torch.cuda.memory_allocated(self.device) / (1024**3), 2)
            info["vram_reserved_gb"] = round(torch.cuda.memory_reserved(self.device) / (1024**3), 2)
            
        return info