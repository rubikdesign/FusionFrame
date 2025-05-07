#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Implementare pentru modelul HiDream în FusionFrame 2.0 (Versiune Corectată)
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

# Importuri necesare pentru HiDream-I1 și Diffusers
from transformers import LlamaForCausalLM, PreTrainedTokenizerFast # Pentru text encoders HiDream-I1
from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    DiffusionPipeline, # Folosit ca fallback
    DPMSolverMultistepScheduler,
    StableDiffusionXLImg2ImgPipeline # Pentru SDXL Refiner
)
# HiDreamImageTransformer2DModel este de obicei o componentă internă a pipeline-ului HiDream
# sau poate fi încărcat separat dacă structura HiDream o cere specific.
# Vom încerca să importăm HiDreamImagePipeline și să vedem dacă acesta gestionează intern transformer-ul.

# Setăm logger-ul
logger = logging.getLogger(__name__)

class HiDreamModel(BaseModel):
    """
    Wrapper pentru modelul HiDream-I1 cu refiner opțional.
    La prima încărcare, modelele vor fi descărcate automat dacă nu sunt prezente local.
    """
    def __init__(
        self,
        model_id: str = ModelConfig.MAIN_MODEL,
        device: Optional[str] = None,
        use_refiner: Optional[bool] = None,
    ):
        super().__init__(model_id, device)
        self.config = ModelConfig.HIDREAM_CONFIG
        self.refiner_config = ModelConfig.REFINER_CONFIG
        if use_refiner is None:
            use_refiner = AppConfig.USE_REFINER
        self.use_refiner = use_refiner and self.refiner_config.get("enabled", False)

        self.vae = None
        self.controlnet = None
        self.pipeline: Optional[DiffusionPipeline] = None # Tipul de bază pentru pipeline
        self.refiner_pipeline: Optional[StableDiffusionXLImg2ImgPipeline] = None # Pentru SDXL Refiner
        self.lora_weights: List[Dict[str, Any]] = []
        self.text_encoder_4 = None # Specific pentru HiDream-I1
        self.tokenizer_4 = None    # Specific pentru HiDream-I1

    def load(self) -> bool:
        logger.info(f"Loading HiDream model '{self.model_id}' with refiner={self.use_refiner}")
        PipelineClass = DiffusionPipeline # Inițializare cu fallback

        try:
            # Încercăm să importăm clasa specifică HiDream
            from diffusers.pipelines.hidream_image.pipeline import HiDreamImagePipeline
            PipelineClass = HiDreamImagePipeline
            logger.info("Successfully imported HiDreamImagePipeline.")
            has_hidream_specific_pipeline = True
        except ImportError as e:
            logger.warning(
                f"Failed to import HiDreamImagePipeline: {e}. "
                f"Falling back to generic DiffusionPipeline. "
                f"Model '{self.model_id}' might not load or work as expected if it requires HiDreamImagePipeline."
            )
            has_hidream_specific_pipeline = False
            # Dacă model_id este specific HiDream și nu avem pipeline-ul, ar trebui să considerăm o eroare?
            if "HiDream-ai/" in self.config.get("pretrained_model_name_or_path", ""):
                 logger.error(f"Model {self.model_id} seems to be a HiDream model, but HiDreamImagePipeline could not be imported. Critical error.")
                 self.is_loaded = False
                 return False


        try:
            # 1. Încarcă VAE
            logger.info(f"Loading VAE from {self.config['vae_name_or_path']}...")
            self.vae = AutoencoderKL.from_pretrained(
                self.config["vae_name_or_path"],
                torch_dtype=AppConfig.DTYPE,
                cache_dir=AppConfig.CACHE_DIR,
            ).to(self.device)
            logger.info("VAE loaded successfully.")

            # 2. Încarcă ControlNet dacă e configurat
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


            pipeline_kwargs: Dict[str, Any] = {
                "vae": self.vae,
                "torch_dtype": AppConfig.DTYPE,
                "use_safetensors": self.config.get("use_safetensors", True), # Default to True
                "cache_dir": AppConfig.CACHE_DIR,
            }

            if self.controlnet:
                pipeline_kwargs["controlnet"] = self.controlnet

            # 3. Încarcă Text Encoders specifici pentru HiDream-I1 (Llama)
            # Acestea sunt necesare DOAR dacă folosim HiDreamImagePipeline
            if has_hidream_specific_pipeline and "HiDream-ai/HiDream-I1" in self.config.get("pretrained_model_name_or_path", ""):
                try:
                    # Numele modelului Llama ar trebui să fie configurabil sau dedus
                    llama_model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct" # Asigurați-vă că acesta este corect
                    logger.info(f"Loading Llama tokenizer_4 from {llama_model_name} for HiDream-I1...")
                    self.tokenizer_4 = PreTrainedTokenizerFast.from_pretrained(
                        llama_model_name, cache_dir=AppConfig.CACHE_DIR
                    )
                    logger.info(f"Loading Llama text_encoder_4 from {llama_model_name} for HiDream-I1...")
                    self.text_encoder_4 = LlamaForCausalLM.from_pretrained(
                        llama_model_name,
                        output_hidden_states=True,
                        output_attentions=True,
                        torch_dtype=AppConfig.DTYPE, # sau torch.bfloat16 dacă e suportat și preferat
                        cache_dir=AppConfig.CACHE_DIR,
                    ).to(self.device)
                    
                    pipeline_kwargs["tokenizer_4"] = self.tokenizer_4
                    pipeline_kwargs["text_encoder_4"] = self.text_encoder_4
                    logger.info("Llama text encoder and tokenizer for HiDream-I1 loaded successfully.")
                except Exception as e_llama:
                    logger.error(f"Failed to load Llama text encoder/tokenizer for HiDream-I1: {e_llama}. "
                                 "HiDream-I1 pipeline might not function correctly.", exc_info=True)
                    # Acest lucru este probabil critic pentru HiDream-I1
                    self.is_loaded = False
                    return False
            
            # 4. Încarcă pipeline-ul principal
            logger.info(f"Loading main pipeline '{self.config['pretrained_model_name_or_path']}' using {PipelineClass.__name__}...")
            self.pipeline = PipelineClass.from_pretrained(
                self.config["pretrained_model_name_or_path"],
                **pipeline_kwargs,
            )
            logger.info("Main pipeline loaded successfully.")

            # 5. Optimizări VRAM (după încărcarea pipeline-ului)
            if self.pipeline: # Verificăm dacă pipeline-ul a fost încărcat
                if AppConfig.LOW_VRAM_MODE:
                    logger.info("Enabling model CPU offload for main pipeline.")
                    self.pipeline.enable_model_cpu_offload()
                else:
                    logger.info("Moving main pipeline to device.")
                    self.pipeline.to(self.device)

                # 6. Configurează scheduler optimizat
                logger.info("Configuring DPMSolverMultistepScheduler for main pipeline.")
                self.pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
                    self.pipeline.scheduler.config,
                    algorithm_type="sde-dpmsolver++",
                    use_karras_sigmas=True,
                )
                logger.info("Scheduler configured.")
            else:
                logger.error("Main pipeline is None, cannot apply VRAM optimizations or scheduler.")
                self.is_loaded = False
                return False


            # 7. Încarcă Refiner Pipeline (dacă este activat și configurat)
            if self.use_refiner and self.refiner_config.get("pretrained_model_name_or_path"):
                logger.info(f"Loading Refiner pipeline from {self.refiner_config['pretrained_model_name_or_path']}...")
                try:
                    # VAE-ul poate fi refolosit sau se poate încărca unul specific refiner-ului dacă este necesar
                    refiner_vae = self.vae # Refolosim VAE-ul principal
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


            # 8. Încarcă LoRA-urile configurate (dacă pipeline-ul principal există)
            if self.pipeline and self.config.get("lora_weights"):
                self._load_loras()

            self.model = self.pipeline # Modelul principal este pipeline-ul
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
        """Încarcă LoRA-urile configurate în pipeline-ul principal."""
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
                # load_lora_weights poate fi direct pe pipeline sau pe unet/text_encoder
                # Pentru Diffusers >0.17, este direct pe pipeline
                self.pipeline.load_lora_weights(
                    lora_path, # Poate fi un director sau un fișier .safetensors
                    weight_name=weight_name, # Numele fișierului specific în director, dacă lora_path e un director
                    adapter_name=lora_name # Nume intern pentru adapter
                )
                # Pentru a activa LoRA-ul după încărcare, s-ar putea să fie nevoie de:
                # self.pipeline.set_adapters([lora_name], adapter_weights=[lora_info.get("weight", 1.0)])
                # Sau, dacă se încarcă mai multe și se dorește fuzionarea:
                # self.pipeline.fuse_lora(adapter_names=[lora_name], lora_scale=lora_info.get("weight", 1.0))
                # Comportamentul exact poate depinde de versiunea diffusers și de modul cum se dorește utilizarea LoRA.
                # Pentru început, doar încărcarea este suficientă, activarea se face la inferență sau prin `set_adapters`.

                self.lora_weights.append(lora_info)
                logger.info(f"Successfully loaded LoRA '{lora_name}'.")
            except Exception as e_lora:
                logger.warning(f"Could not load LoRA '{lora_name}' from '{lora_path}': {e_lora}", exc_info=True)
        
        if self.lora_weights:
             # După încărcarea tuturor LoRA-urilor, puteți seta ce adaptoare sunt active și cu ce greutăți
             # Exemplu: activăm toate LoRA-urile încărcate cu greutățile specificate (default 1.0)
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
        del self.controlnet
        del self.refiner_pipeline
        del self.text_encoder_4
        del self.tokenizer_4
        
        self.pipeline = None
        self.vae = None
        self.controlnet = None
        self.refiner_pipeline = None
        self.text_encoder_4 = None
        self.tokenizer_4 = None
        self.model = None # Asigurăm că și self.model este curățat
        
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
        controlnet_conditioning_scale: Optional[float] = None, # Adăugat pentru a fi utilizat
        refiner_strength: Optional[float] = None, # Adăugat pentru a fi utilizat
        seed: int = 42, # Adăugat parametru pentru seed
        **kwargs, # Permite pasarea altor parametri specifici pipeline-ului
    ) -> Dict[str, Any]:
        if not self.is_loaded:
            logger.warning("Model not loaded. Attempting to load...")
            if not self.load():
                return {"result": image, "success": False, "message": "Model failed to load on demand."}
        
        if not self.pipeline: # Verificare suplimentară
             return {"result": image, "success": False, "message": "Main pipeline is not available."}

        image_pil = Image.fromarray(image) if isinstance(image, np.ndarray) else image.convert("RGB")
        mask_pil = Image.fromarray(mask_image) if isinstance(mask_image, np.ndarray) else mask_image.convert("L")

        # Asigurăm că dimensiunile imaginii și măștii sunt compatibile cu ce așteaptă pipeline-ul (de ex. multiplu de 8)
        # Aceasta este o practică bună, deși unele pipeline-uri o fac intern.
        # width, height = image_pil.size
        # new_width = (width // 8) * 8
        # new_height = (height // 8) * 8
        # if new_width != width or new_height != height:
        #     logger.info(f"Resizing input image from ({width},{height}) to ({new_width},{new_height})")
        #     image_pil = image_pil.resize((new_width, new_height), Image.LANCZOS)
        #     mask_pil = mask_pil.resize((new_width, new_height), Image.NEAREST)


        gen = torch.Generator(device=self.device).manual_seed(seed)
        
        effective_num_inference_steps = (
            num_inference_steps
            if num_inference_steps is not None
            else self.config.get("inference_steps", ModelConfig.GENERATION_PARAMS["default_steps"])
        )
        
        effective_refiner_strength = refiner_strength if refiner_strength is not None else AppConfig.REFINER_STRENGTH
        
        effective_negative_prompt = negative_prompt if negative_prompt is not None else ModelConfig.GENERATION_PARAMS.get("negative_prompt", "")

        pipeline_args = {
            "prompt": prompt,
            "negative_prompt": effective_negative_prompt,
            "image": image_pil,
            "mask_image": mask_pil,
            "num_inference_steps": effective_num_inference_steps,
            "guidance_scale": guidance_scale,
            "strength": strength, # Pentru img2img / inpainting
            "generator": gen,
        }

        if self.controlnet and controlnet_conditioning_scale is not None:
            # ControlNet necesită imaginea de condiționare, care poate fi derivată din imaginea originală
            # sau o imagine specifică pentru Canny, Depth etc.
            # Presupunem aici că imaginea de intrare este folosită direct sau pre-procesată pentru ControlNet
            # pipeline_args["controlnet_conditioning_image"] = image_pil # Sau o versiune procesată (ex: Canny edges)
            pipeline_args["controlnet_conditioning_scale"] = controlnet_conditioning_scale
            logger.info(f"Using ControlNet with scale: {controlnet_conditioning_scale}")

        # Adăugăm kwargs la argumentele pipeline-ului, permițând flexibilitate
        pipeline_args.update(kwargs)

        try:
            logger.info(f"Processing with main pipeline. Steps: {effective_num_inference_steps}, Guidance: {guidance_scale}, Strength: {strength}")
            out = self.pipeline(**pipeline_args)
            result_img = out.images[0]

            if self.use_refiner and self.refiner_pipeline:
                logger.info(f"Applying Refiner. Strength: {effective_refiner_strength}")
                # Pentru refiner, strength are un alt sens (cât de mult să modifice imaginea de la base)
                # Numărul de pași pentru refiner poate fi diferit
                refiner_steps = self.refiner_config.get("inference_steps", max(10, effective_num_inference_steps // 3))
                
                refiner_args = {
                    "prompt": prompt, # Refiner-ul poate folosi același prompt
                    "negative_prompt": effective_negative_prompt,
                    "image": result_img, # Imaginea generată de pipeline-ul principal
                    "num_inference_steps": refiner_steps,
                    "guidance_scale": guidance_scale, # Poate fi ajustat pentru refiner
                    "strength": effective_refiner_strength, # Controlul efectului refiner-ului
                    "generator": gen, # Refolosim generatorul pentru consistență (sau unul nou)
                }
                # Dacă refiner-ul este un SDXLImg2ImgPipeline, s-ar putea să nu folosească `mask_image`
                # ci doar `strength` pentru a determina cât de mult se schimbă imaginea de intrare.

                out_refined = self.refiner_pipeline(**refiner_args)
                result_img = out_refined.images[0]
                logger.info("Refiner applied successfully.")

            return {"result": result_img, "success": True, "message": "Procesare completă cu succes"}
        
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
            "active_loras": self.lora_weights, # Sau informații despre LoRA-urile active
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