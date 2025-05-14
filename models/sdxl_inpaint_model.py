#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Implementation for SDXL Inpainting models like Juggernaut-XL for FusionFrame 2.0
"""

import os
import torch
import logging
import gc
import time
from typing import Dict, Any, Optional, List, Union
from PIL import Image
import numpy as np

# Referințe la configurații și clasa de bază
try:
    from config.app_config import AppConfig
    from config.model_config import ModelConfig
    from models.base_model import BaseModel
except ImportError:
    # Fallback-uri dacă rulezi acest fișier izolat sau ai probleme cu path-urile
    # Acest lucru nu ar trebui să se întâmple într-o aplicație funcțională.
    class MockBaseModel:
        def __init__(self, model_id, device): self.model_id = model_id; self.device = device
        def get_info(self): return {"model_id": self.model_id, "device": self.device}
    BaseModel = MockBaseModel #type: ignore
    class AppConfig: DTYPE = torch.float32; DEVICE = "cpu"; LOW_VRAM_MODE=True; CACHE_DIR="cache" #type: ignore
    class ModelConfig: MAIN_MODEL="mock"; SDXL_INPAINT_CONFIG={}; GENERATION_PARAMS={"default_steps":30}; LOW_VRAM_INFERENCE_STEPS={} #type: ignore
    logging.basicConfig(level=logging.INFO)


# Importuri specifice Diffusers pentru SDXL Inpainting
try:
    from diffusers import (
        StableDiffusionXLInpaintPipeline,
        AutoencoderKL,
        EulerAncestralDiscreteScheduler, # Sau alt scheduler preferat pentru SDXL
        ControlNetModel
    )
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False
    StableDiffusionXLInpaintPipeline, AutoencoderKL, EulerAncestralDiscreteScheduler, ControlNetModel = None, None, None, None #type: ignore
    logging.getLogger(__name__).error("Diffusers library not found. SDXLInpaintModel will not work.")

# Încercăm să importăm OpenCV pentru procesarea imaginilor
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    logging.getLogger(__name__).warning("OpenCV (cv2) not found. Some image processing functions may not work.")

# Încarcă singleton-ul ModelManager pentru emergency_memory_recovery
try:
    from core.model_manager import ModelManager
    MODEL_MANAGER_AVAILABLE = True
except ImportError:
    MODEL_MANAGER_AVAILABLE = False
    logging.getLogger(__name__).warning("ModelManager not available. Emergency memory recovery may not work properly.")


logger = logging.getLogger(__name__)

class SDXLInpaintModel(BaseModel):
    """
    Wrapper for SDXL-based inpainting models (e.g., Juggernaut-XL).
    """
    def __init__(self,
                 model_id: str = ModelConfig.MAIN_MODEL,
                 device: Optional[str] = None):
        super().__init__(model_id, device or AppConfig.DEVICE) #type: ignore

        # Determină ce configurație să folosească pe baza model_id-ului curent
        # Acest lucru permite flexibilitate dacă ai mai multe configuri SDXL
        if self.model_id == "RunDiffusion/Juggernaut-XL-v9" and hasattr(ModelConfig, 'SDXL_INPAINT_CONFIG'):
            self.config = ModelConfig.SDXL_INPAINT_CONFIG
        # Adaugă alte `elif` aici dacă ai alte modele SDXL cu configuri specifice
        elif hasattr(ModelConfig, 'SDXL_INPAINT_CONFIG'): # Fallback la configul general SDXL_INPAINT
            logger.warning(f"Using generic SDXL_INPAINT_CONFIG for model_id: {self.model_id}")
            self.config = ModelConfig.SDXL_INPAINT_CONFIG
        else:
            logger.error(f"SDXL_INPAINT_CONFIG not found in ModelConfig for model_id: {self.model_id}! Using empty config.")
            self.config: Dict[str, Any] = {} # Asigură că self.config este un dicționar

        self.vae: Optional[AutoencoderKL] = None
        self.controlnet: Optional[ControlNetModel] = None
        self.pipeline: Optional[StableDiffusionXLInpaintPipeline] = None
        self.lora_weights: List[Dict[str, Any]] = []
        self.is_loaded: bool = False

    def load(self) -> bool:
        if not DIFFUSERS_AVAILABLE:
            logger.error("Diffusers library not available. Cannot load SDXLInpaintModel.")
            return False

        model_load_path = self.config.get('pretrained_model_name_or_path', self.model_id)
        logger.info(f"Loading SDXL Inpaint model '{self.model_id}' from path '{model_load_path}'")
        start_time = time.time()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            logger.info(f"CUDA memory before SDXL load: {torch.cuda.memory_allocated(self.device) / 1024**3:.2f} GB used, {torch.cuda.memory_reserved(self.device) / 1024**3:.2f} GB reserved")

        try:
            app_dtype_str = str(AppConfig.DTYPE).lower()
            effective_torch_dtype = AppConfig.DTYPE

            if "bfloat16" in app_dtype_str:
                if AppConfig.DEVICE == "cuda" and hasattr(torch.cuda, 'is_bf16_supported') and not torch.cuda.is_bf16_supported():
                    logger.warning("Configured DTYPE is bfloat16, but not supported on this GPU. Falling back to float32.")
                    effective_torch_dtype = torch.float32
                elif AppConfig.DEVICE == "cpu":
                    logger.warning("bfloat16 not recommended for CPU. Using float32.")
                    effective_torch_dtype = torch.float32
                else:
                    effective_torch_dtype = torch.bfloat16
            elif "float16" in app_dtype_str:
                effective_torch_dtype = torch.float16
            else:
                effective_torch_dtype = torch.float32
            logger.info(f"Effective torch_dtype for SDXL model: {effective_torch_dtype}")

            # ControlNet setup
            self.controlnet = None
            controlnet_model_id = ModelConfig.CONTROLNET_CONFIG.get("model_id") if hasattr(ModelConfig, 'CONTROLNET_CONFIG') else None
            if controlnet_model_id and ControlNetModel is not None:
                try:
                    logger.info(f"Preparing to load ControlNet from {controlnet_model_id} separately...")
                    logger.info(f"ControlNet ({controlnet_model_id}) will be handled separately if needed.")
                except Exception as e_ctrl:
                    logger.warning(f"ControlNet configuration found but could not be prepared: {e_ctrl}")
            else:
                logger.info("ControlNet not configured or ControlNetModel class not available.")

            # VAE setup
            self.vae = None
            explicit_vae_path = self.config.get("vae_name_or_path")
            
            pipeline_load_kwargs = {
                "torch_dtype": effective_torch_dtype,
                "use_safetensors": self.config.get("use_safetensors", True),
                "cache_dir": AppConfig.CACHE_DIR,
                "variant": "fp16" if effective_torch_dtype == torch.float16 and AppConfig.DEVICE == "cuda" else None,
            }

            if explicit_vae_path and str(explicit_vae_path).strip().lower() not in ["", "none", "included", "default"]:
                try:
                    logger.info(f"Attempting to load and pass separate VAE from {explicit_vae_path}...")
                    if AutoencoderKL is not None:
                        self.vae = AutoencoderKL.from_pretrained(
                            explicit_vae_path,
                            torch_dtype=effective_torch_dtype,
                            cache_dir=AppConfig.CACHE_DIR,
                        )
                        pipeline_load_kwargs["vae"] = self.vae
                        logger.info(f"Separate VAE '{explicit_vae_path}' will be passed to the pipeline.")
                    else:
                        logger.error("AutoencoderKL class not available for loading VAE.")
                except Exception as e_vae:
                    logger.error(f"Failed to load separate VAE from {explicit_vae_path}: {e_vae}. Pipeline will use its default VAE.")
                    self.vae = None
                    if "vae" in pipeline_load_kwargs:
                        del pipeline_load_kwargs["vae"]
            else:
                logger.info("No separate VAE specified or VAE is 'included'. Pipeline will attempt to load its default VAE.")

            # Load pipeline using from_single_file or from_pretrained
            logger.info(f"Attempting to load StableDiffusionXLInpaintPipeline with kwargs: "
                       f"{ {k:v.__class__.__name__ if hasattr(v, '__class__') and v is not None else v for k,v in pipeline_load_kwargs.items()} }")
            
            if self.config.get("load_from_single_file", False):
                logger.info(f"Loading SDXL Inpaint model from single file: {model_load_path}")
                assert StableDiffusionXLInpaintPipeline is not None
                self.pipeline = StableDiffusionXLInpaintPipeline.from_single_file(
                    model_load_path,
                    **pipeline_load_kwargs
                )
                logger.info(f"StableDiffusionXLInpaintPipeline instance created from single file: {model_load_path}")
            else:
                logger.info(f"Loading SDXL Inpaint model from pretrained: {model_load_path}")
                assert StableDiffusionXLInpaintPipeline is not None
                self.pipeline = StableDiffusionXLInpaintPipeline.from_pretrained(
                    model_load_path,
                    **pipeline_load_kwargs
                )
                logger.info(f"StableDiffusionXLInpaintPipeline instance created from pretrained: {model_load_path}")

            # Memory optimizations
            if AppConfig.LOW_VRAM_MODE and self.pipeline:
                logger.info("Applying LOW_VRAM_MODE optimizations for SDXLInpaintPipeline")
                if hasattr(self.pipeline, "enable_model_cpu_offload") and callable(self.pipeline.enable_model_cpu_offload):
                    self.pipeline.enable_model_cpu_offload()
                    logger.info("Enabled model CPU offload via pipeline.enable_model_cpu_offload().")
                else:
                    logger.warning("pipeline.enable_model_cpu_offload() not available. Attempting manual offload.")
                    if hasattr(self.pipeline, "text_encoder") and self.pipeline.text_encoder is not None: 
                        self.pipeline.text_encoder.to("cpu")
                        logger.info("Moved text_encoder to CPU.")
                    if hasattr(self.pipeline, "text_encoder_2") and self.pipeline.text_encoder_2 is not None: 
                        self.pipeline.text_encoder_2.to("cpu")
                        logger.info("Moved text_encoder_2 to CPU.")
                if hasattr(self.pipeline, "enable_vae_slicing"): 
                    self.pipeline.enable_vae_slicing()
                    logger.info("Enabled VAE slicing.")
                if hasattr(self.pipeline, "enable_attention_slicing"): 
                    self.pipeline.enable_attention_slicing()
                    logger.info("Enabled attention slicing.")
                elif hasattr(self.pipeline, "enable_xformers_memory_efficient_attention"):
                    try: 
                        self.pipeline.enable_xformers_memory_efficient_attention()
                        logger.info("Enabled xformers.")
                    except Exception as e_xf: 
                        logger.warning(f"Xformers failed: {e_xf}")
                
                if self.controlnet and hasattr(self.controlnet, 'to') and self.controlnet.device.type != 'cpu': 
                    self.controlnet.to("cpu")
                    logger.info("Moved separate ControlNet to CPU for LOW_VRAM.")

            elif self.pipeline:
                logger.info(f"Moving SDXLInpaintPipeline and its components to device: {self.device}")
                self.pipeline.to(self.device)
                if self.controlnet and hasattr(self.controlnet, 'to'): 
                    self.controlnet.to(self.device)
                    logger.info("Moved separate ControlNet to device.")
            else:
                logger.error("Pipeline is None after attempting to load. Cannot proceed.")
                self.is_loaded = False
                return False

            if self.pipeline and hasattr(self.pipeline, 'scheduler') and EulerAncestralDiscreteScheduler is not None:
                try:
                    self.pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(self.pipeline.scheduler.config)
                    logger.info(f"Scheduler re-configured: {self.pipeline.scheduler.__class__.__name__}.")
                except Exception as e_sch: 
                    logger.warning(f"Scheduler config failed: {e_sch}. Using default: {self.pipeline.scheduler.__class__.__name__}")
            elif self.pipeline: 
                logger.info(f"Using default scheduler: {self.pipeline.scheduler.__class__.__name__}")

            if self.config.get("lora_weights"): 
                self._load_loras()

            self.model = self.pipeline
            self.is_loaded = True
            load_time = time.time() - start_time
            logger.info(f"SDXL Inpaint model '{self.model_id}' loaded successfully in {load_time:.2f} seconds.")
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated(self.device) / (1024**3)
                reserved = torch.cuda.memory_reserved(self.device) / 1024**3
                logger.info(f"CUDA memory after SDXL Inpaint load: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")
            return True

        except OSError as oe:
            logger.error(f"OSError loading SDXL Inpaint model '{self.model_id}': {str(oe)}", exc_info=True)
            cache_path_to_delete = os.path.join(str(AppConfig.CACHE_DIR), 'models--' + model_load_path.replace('/', '--'))
            logger.error(
                "This often indicates an incomplete or corrupted model download in the cache. "
                "Please try deleting the model-specific cache directory and retrying: \n"
                f"rm -rf {cache_path_to_delete}"
            )
            self.is_loaded = False
            return False
        except Exception as e:
            logger.error(f"Generic error loading SDXL Inpaint model '{self.model_id}': {str(e)}", exc_info=True)
            self.is_loaded = False
            return False
    
    def _load_loras(self) -> None:
        if not self.pipeline:
            logger.warning("Main pipeline not loaded, cannot load LoRAs.")
            return

        logger.info(f"Loading LoRA weights for {self.model_id}...")
        loaded_lora_names = []
        for lora_info in self.config.get("lora_weights", []):
            lora_path = lora_info.get("path")
            # Numele adaptorului este important pentru activare/dezactivare
            adapter_name = lora_info.get("name", os.path.splitext(os.path.basename(lora_path))[0] if lora_path else f"lora_{len(self.lora_weights)}")
            lora_weight_name = lora_info.get("weight_name") # Pentru fișiere .safetensors individuale în directoare

            if not lora_path:
                logger.warning(f"Skipping LoRA '{adapter_name}' due to missing path.")
                continue

            logger.info(f"Loading LoRA '{adapter_name}' from path '{lora_path}' (weight_name: {lora_weight_name}).")
            try:
                # Pentru SDXL, load_lora_weights este direct pe pipeline
                self.pipeline.load_lora_weights(
                    lora_path,
                    weight_name=lora_weight_name, # Dacă lora_path e un folder, specifică fișierul
                    adapter_name=adapter_name
                )
                self.lora_weights.append({"name": adapter_name, "path": lora_path, "weight": lora_info.get("weight", 1.0)})
                loaded_lora_names.append(adapter_name)
                logger.info(f"Successfully loaded LoRA '{adapter_name}'.")
            except Exception as e_lora:
                logger.error(f"Could not load LoRA '{adapter_name}' from '{lora_path}': {e_lora}", exc_info=True)

        # Setează greutățile pentru LoRA-urile încărcate DUPĂ ce toate sunt încărcate
        # Acest lucru poate fi necesar dacă `set_adapters` le activează
        # Sau poți fuziona direct LoRA-urile dacă asta vrei (self.pipeline.fuse_lora())
        if loaded_lora_names:
            # Pentru a le activa cu greutăți specifice:
            try:
                # Creează dicționarul de greutăți pentru adaptoarele active
                adapter_weights = {lora["name"]: lora["weight"] for lora in self.lora_weights if lora["name"] in loaded_lora_names}
                # self.pipeline.set_adapters(loaded_lora_names, adapter_weights=[adapter_weights[name] for name in loaded_lora_names])
                # Sau pentru a fuziona:
                # for lora_name in loaded_lora_names:
                #    self.pipeline.fuse_lora(lora_name=lora_name, lora_scale=adapter_weights.get(lora_name, 1.0))
                # Momentan, doar le încărcăm. Activarea și setarea greutăților se poate face înainte de inferență
                # sau prin fuzionare. Diffusers API poate varia ușor aici.
                logger.info(f"LoRAs {loaded_lora_names} are loaded. Activation/weight setting might need explicit call before inference or fusing.")
            except Exception as e_set_lora:
                 logger.error(f"Error setting/activating LoRAs: {e_set_lora}")

    def unload(self) -> bool:
        if not self.is_loaded:
            # logger.info(f"SDXL Inpaint model '{self.model_id}' already unloaded or never loaded.")
            return True
        logger.info(f"Unloading SDXL Inpaint model '{self.model_id}'...")
        try:
            # Șterge referințele în ordine inversă încărcării sau cele mai mari întâi
            del self.pipeline # Pipeline-ul poate ține referințe la celelalte
            del self.model
            del self.vae
            if self.controlnet:
                del self.controlnet

            self.pipeline = None
            self.model = None
            self.vae = None
            self.controlnet = None
            self.lora_weights = []

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
            self.is_loaded = False
            logger.info(f"SDXL Inpaint model '{self.model_id}' unloaded.")
            return True
        except Exception as e:
            logger.error(f"Error unloading SDXL Inpaint model '{self.model_id}': {e}", exc_info=True)
            return False

    def _ensure_dimensions_multiple_of_8(self, image_pil):
        """Asigură că dimensiunile imaginii sunt divizibile cu 8."""
        width, height = image_pil.size
        new_width = (width // 8) * 8
        new_height = (height // 8) * 8
        
        # Dacă dimensiunile sunt deja divizibile cu 8, returnăm imaginea originală
        if width == new_width and height == new_height:
            return image_pil
        
        # Altfel, redimensionăm imaginea la dimensiunile corecte
        logger.info(f"Redimensionare imagine de la {width}x{height} la {new_width}x{new_height} pentru a asigura divizibilitatea cu 8")
        return image_pil.resize((new_width, new_height), Image.LANCZOS)
    
    
    def process(self,
               image: Union[Image.Image, np.ndarray],
               mask_image: Union[Image.Image, np.ndarray],
               prompt: str,
               negative_prompt: Optional[str] = None,
               strength: float = 0.85, # Valoare bună default pentru inpainting
               num_inference_steps: Optional[int] = None,
               guidance_scale: float = 7.0, # SDXL adesea preferă valori mai mici de guidance
               controlnet_conditioning_scale: Optional[float] = None,
               seed: int = 42,
               height: Optional[int] = None, # Adăugat pentru consistență cu SDXL
               width: Optional[int] = None,  # Adăugat pentru consistență cu SDXL
               # eta: float = 0.0, # Pentru DDIMScheduler
               # cross_attention_kwargs: Optional[Dict[str, Any]] = None, # Pentru LoRA-uri & altele
               **kwargs) -> Dict[str, Any]:
    
        if not self.is_loaded or not self.pipeline:
            logger.warning(f"SDXL Inpaint model '{self.model_id}' not loaded. Attempting to load...")
            # Convertim imaginea la PIL pentru fallback înainte de a încerca să încărcăm
            try:
                pil_image_fallback = self._ensure_pil_rgb(image)
            except Exception: # Dacă conversia eșuează, nu avem ce returna
                pil_image_fallback = Image.new("RGB", (512,512), "grey") # Imagine goală
    
            if not self.load():
                return {"result": pil_image_fallback, "success": False, "message": "Model failed to load on demand."}
        
        assert self.pipeline is not None # Pentru type checker
    
        start_time_proc = time.time()
        
        processed_image_pil: Optional[Image.Image] = None # Pentru fallback în caz de eroare
    
        try:
            image_pil = self._ensure_pil_rgb(image)
            mask_pil = self._ensure_pil_l_for_mask(mask_image)
            processed_image_pil = image_pil # Salvează pentru fallback
    
            # Asigură că imaginea și masca au dimensiuni divizibile cu 8
            width, height = image_pil.size
            new_width = (width // 8) * 8
            new_height = (height // 8) * 8
            
            # Redimensionează doar dacă e necesar
            if width != new_width or height != new_height:
                logger.info(f"Redimensionare imagine de la {width}x{height} la {new_width}x{new_height} pentru a asigura divizibilitatea cu 8")
                image_pil = image_pil.resize((new_width, new_height), Image.LANCZOS)
            
            # Asigură că imaginea și masca au aceleași dimensiuni
            if mask_pil.size != image_pil.size:
                logger.warning(f"Image size {image_pil.size} and mask size {mask_pil.size} differ. Resizing mask to image size.")
                mask_pil = mask_pil.resize(image_pil.size, Image.NEAREST) # Sau Image.LANCZOS dacă masca e fină
    
            # Pentru SDXL, rezoluția nativă e 1024. Dacă imaginile sunt diferite,
            # pipeline-ul le va redimensiona intern, dar e bine să fim conștienți.
            # Setăm height și width dacă nu sunt furnizate, bazat pe imaginea de input.
            effective_width, effective_height = image_pil.size
            if height is not None and width is not None:
                # Asigură că height și width specificate sunt divizibile cu 8
                effective_height = (height // 8) * 8
                effective_width = (width // 8) * 8
                if effective_height != height or effective_width != width:
                    logger.info(f"Ajustare height/width de la {width}x{height} la {effective_width}x{effective_height} pentru a asigura divizibilitatea cu 8")
    
            # Optimizări pentru LOW_VRAM_MODE (dincolo de cele de la încărcare)
            if AppConfig.LOW_VRAM_MODE:
                # Redu dimensiunea dacă e prea mare, păstrând aspect ratio
                max_dim_low_vram = 768 # Sau 512 pentru VRAM foarte redus
                if max(effective_height, effective_width) > max_dim_low_vram:
                    logger.info(f"LOW_VRAM: Resizing input from {effective_width}x{effective_height} for processing.")
                    aspect_ratio = effective_width / effective_height
                    if effective_width > effective_height:
                        effective_width = max_dim_low_vram
                        effective_height = int(effective_width / aspect_ratio)
                    else:
                        effective_height = max_dim_low_vram
                        effective_width = int(effective_height * aspect_ratio)
                    # Asigură divizibilitate cu 8 după redimensionare
                    effective_width = (effective_width // 8) * 8
                    effective_height = (effective_height // 8) * 8
                    logger.info(f"LOW_VRAM: Resized to {effective_width}x{effective_height}.")
                    image_pil = image_pil.resize((effective_width, effective_height), Image.LANCZOS)
                    mask_pil = mask_pil.resize((effective_width, effective_height), Image.NEAREST)
    
            generator = torch.Generator(device=self.device).manual_seed(seed)
            
            cfg_gen_params = ModelConfig.GENERATION_PARAMS if hasattr(ModelConfig, 'GENERATION_PARAMS') else {}
            default_steps_cfg = cfg_gen_params.get("default_steps", 30)
    
            effective_num_inference_steps = (
                num_inference_steps if num_inference_steps is not None
                else self.config.get("inference_steps", default_steps_cfg)
            )
            if AppConfig.LOW_VRAM_MODE:
                 steps_config = ModelConfig.LOW_VRAM_INFERENCE_STEPS if hasattr(ModelConfig, 'LOW_VRAM_INFERENCE_STEPS') else {}
                 effective_num_inference_steps = steps_config.get(self.model_id, 20)
    
            pipeline_args: Dict[str, Any] = {
                "prompt": prompt,
                "image": image_pil,
                "mask_image": mask_pil,
                "strength": strength,
                "negative_prompt": negative_prompt,
                "num_inference_steps": effective_num_inference_steps,
                "guidance_scale": guidance_scale,
                "generator": generator,
                "height": effective_height, # Pasează dimensiunile ajustate
                "width": effective_width,   # Pasează dimensiunile ajustate
                # "eta": eta, # Dacă folosești DDIMScheduler
                # "cross_attention_kwargs": cross_attention_kwargs, # Pentru LoRA etc.
            }
    
            if self.controlnet and controlnet_conditioning_scale is not None:
                # Pentru SDXL Inpaint cu ControlNet, s-ar putea să trebuiască să pasezi `control_image`
                # și `controlnet_conditioning_scale`. Presupunem că `image_pil` servește și ca
                # imagine de control după procesarea Canny (care ar trebui făcută înainte).
                # Aceasta este o simplificare; un pipeline ControlNet complet e mai complex.
                # pipeline_args["control_image"] = ... preprocessed_control_image ...
                pipeline_args["controlnet_conditioning_scale"] = controlnet_conditioning_scale
                logger.info(f"Using ControlNet with scale: {controlnet_conditioning_scale} (ensure control_image is handled if needed).")
            
            pipeline_args.update(kwargs) # Adaugă orice alți parametri custom
    
            logger.info(f"Processing with SDXLInpaintPipeline. Target size: {effective_width}x{effective_height}, Steps: {effective_num_inference_steps}, Strength: {strength}, Guidance: {guidance_scale}")
            if torch.cuda.is_available(): logger.info(f"VRAM before SDXLInpaint inference: {torch.cuda.memory_allocated(self.device)/(1024**3):.2f} GB")
    
            output = self.pipeline(**pipeline_args)
            result_img = output.images[0]
    
            proc_time = time.time() - start_time_proc
            logger.info(f"SDXL Inpainting completed in {proc_time:.2f} seconds.")
            if torch.cuda.is_available(): logger.info(f"VRAM after SDXLInpaint inference: {torch.cuda.memory_allocated(self.device)/(1024**3):.2f} GB")
    
            return {"result": result_img, "success": True, "message": "Inpainting successful."}
    
        except RuntimeError as e: # Specific pentru CUDA OOM
            if "CUDA out of memory" in str(e):
                logger.error(f"CUDA OOM during SDXL Inpainting: {e}. Attempting emergency cleanup.")
                if hasattr(self, 'emergency_cleanup') and callable(self.emergency_cleanup):
                    self.emergency_cleanup() # Dacă ai o metodă de cleanup în BaseModel/SDXLInpaintModel
                else: # Fallback la model_manager cleanup dacă e accesibil sau gc.collect
                    try:
                        from core.model_manager import ModelManager
                        ModelManager().emergency_memory_recovery() # Accesează singleton-ul
                    except ImportError:
                        # Fallback dacă ModelManager nu e disponibil
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            gc.collect()
                return {"result": processed_image_pil, "success": False, "message": f"CUDA out of memory during inpainting. Try reducing resolution or steps. Details: {str(e)}"}
            else:
                logger.error(f"Runtime error during SDXL Inpainting process: {e}", exc_info=True)
                return {"result": processed_image_pil, "success": False, "message": str(e)}
        except Exception as e:
            logger.error(f"Generic error during SDXL Inpainting process: {e}", exc_info=True)
            return {"result": processed_image_pil, "success": False, "message": str(e)}
    
    


    def get_info(self) -> Dict[str, Any]:
        info = super().get_info()
        info.update({
            "config_used": self.config, # Arată configurația specifică folosită
            "has_controlnet": self.controlnet is not None,
            "active_loras": self.lora_weights, # Schimbat din lora_weights
            "device_in_use": self.device, # Schimbat din device
            "torch_dtype_used": str(AppConfig.DTYPE), # Schimbat din dtype
            "low_vram_mode_active" : AppConfig.LOW_VRAM_MODE,     # Schimbat din low_vram_mode
            "pipeline_class_name": self.pipeline.__class__.__name__ if self.pipeline else "N/A", # Schimbat din pipeline_class
        })
        if torch.cuda.is_available():
            info["vram_allocated_gb"] = round(torch.cuda.memory_allocated(self.device) / (1024**3), 2)
            info["vram_reserved_gb"] = round(torch.cuda.memory_reserved(self.device) / (1024**3), 2)
        return info

    def _ensure_pil_rgb(self, image_data: Union[Image.Image, np.ndarray]) -> Image.Image:
        """Ensure image is PIL RGB."""
        if isinstance(image_data, Image.Image):
            return image_data.convert("RGB") if image_data.mode != "RGB" else image_data
        elif isinstance(image_data, np.ndarray):
            if image_data.ndim == 2:
                # Imagine grayscale - convertim la RGB fără cv2 dacă nu e disponibil
                if CV2_AVAILABLE:
                    return Image.fromarray(cv2.cvtColor(image_data, cv2.COLOR_GRAY2RGB))
                else:
                    # Alternativă fără cv2
                    rgb_array = np.stack((image_data,) * 3, axis=-1)
                    return Image.fromarray(rgb_array.astype(np.uint8))
            elif image_data.shape[2] == 4:  # RGBA
                if CV2_AVAILABLE:
                    return Image.fromarray(cv2.cvtColor(image_data, cv2.COLOR_RGBA2RGB))
                else:
                    # Alternativă fără cv2 - păstrăm doar canalele RGB, ignorând alfa
                    return Image.fromarray(image_data[:, :, :3].astype(np.uint8))
            elif image_data.shape[2] == 3 and image_data.dtype == np.uint8:  # RGB sau BGR
                if CV2_AVAILABLE:
                    # Încercăm să detectăm dacă e BGR (OpenCV default)
                    if image_data[0, 0, 0] > image_data[0, 0, 2] + 10 and image_data[0, 0, 0] > image_data[0, 0, 1] + 10:
                        logger.debug("Detected BGR NumPy array, converting to RGB for PIL.")
                        return Image.fromarray(cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB))
                # Asumăm RGB
                return Image.fromarray(image_data)
            else:
                raise ValueError(f"Unsupported NumPy array shape/type for PIL RGB conversion: {image_data.shape}, {image_data.dtype}")
        raise TypeError(f"Unsupported image type for PIL RGB conversion: {type(image_data)}")

    def _ensure_pil_l_for_mask(self, mask_data: Union[Image.Image, np.ndarray]) -> Image.Image:
        """Ensure mask is PIL L (grayscale) uint8."""
        if isinstance(mask_data, Image.Image):
            return mask_data.convert("L") if mask_data.mode != "L" else mask_data
        elif isinstance(mask_data, np.ndarray):
            # Handle 3-channel masks (e.g., from segmentation models)
            if mask_data.ndim == 3:
                if CV2_AVAILABLE:
                    if mask_data.shape[2] == 3:  # BGR/RGB
                        mask_data = cv2.cvtColor(mask_data, cv2.COLOR_BGR2GRAY)
                    elif mask_data.shape[2] == 4:  # BGRA/RGBA
                        mask_data = cv2.cvtColor(mask_data, cv2.COLOR_BGRA2GRAY)
                    elif mask_data.shape[2] == 1:
                        mask_data = mask_data.squeeze(axis=2)
                    else:
                        raise ValueError(f"Unsupported 3D NumPy mask shape: {mask_data.shape}")
                else:
                    # Alternativă fără cv2 - convertim la gri folosind formula standard
                    if mask_data.shape[2] == 3:  # RGB/BGR
                        # Folosim formula gray = 0.299*R + 0.587*G + 0.114*B pentru conversie
                        # Asumăm RGB dar în general nu e critic pentru maști
                        mask_data = np.dot(mask_data[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)
                    elif mask_data.shape[2] == 4:  # RGBA/BGRA
                        # Ignorăm canalul alpha
                        mask_data = np.dot(mask_data[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)
                    elif mask_data.shape[2] == 1:
                        mask_data = mask_data.squeeze(axis=2)
                    else:
                        raise ValueError(f"Unsupported 3D NumPy mask shape: {mask_data.shape}")

            if mask_data.ndim != 2:
                raise ValueError(f"Mask NumPy array must be 2D or convertible to 2D, got ndim: {mask_data.ndim}")

            # Normalize and convert to uint8
            if mask_data.dtype != np.uint8:
                if np.issubdtype(mask_data.dtype, np.floating) and np.max(mask_data) <= 1.0 and np.min(mask_data) >= 0.0:
                    mask_data = (mask_data * 255).astype(np.uint8)
                elif np.issubdtype(mask_data.dtype, np.bool_):
                    mask_data = mask_data.astype(np.uint8) * 255
                else:  # For other types, clip and convert
                    mask_data = np.clip(mask_data, 0, 255).astype(np.uint8)
            return Image.fromarray(mask_data, mode='L')
        raise TypeError(f"Unsupported mask type for PIL L conversion: {type(mask_data)}")