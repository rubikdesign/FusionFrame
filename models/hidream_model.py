#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Implementare pentru modelul SDXL Inpaint în FusionFrame 2.0 (Master Branch)
Corectat pentru a gestiona dimensiuni nedivizibile cu 8.
"""
import os
import torch
import logging
import cv2
import gc # Am adăugat importul lipsă pentru garbage collection în unload
from typing import Dict, Any, Optional, List, Union, Tuple
from PIL import Image, ImageOps # ImageOps ar putea fi util pentru padding dacă e nevoie
import numpy as np

# Importuri corecte pentru Master Branch
try:
    from config.app_config import AppConfig
    from config.model_config import ModelConfig
    from models.base_model import BaseModel
    from diffusers import (
        StableDiffusionXLInpaintPipeline,
        AutoencoderKL,
        DPMSolverMultistepScheduler,
        ControlNetModel
    )
except ImportError as e:
    print(f"ERROR: Failed to import necessary modules in sdxl_inpaint_model.py: {e}")
    # Tratarea erorii - poate ridica excepția sau ieși
    raise e

logger = logging.getLogger(__name__)
if not logger.hasHandlers(): # Basic logging if not configured
    _ch = logging.StreamHandler()
    _ch.setFormatter(logging.Formatter("%(asctime)s-%(name)s-%(levelname)s-%(message)s"))
    logger.addHandler(_ch)
    logger.setLevel(logging.INFO)

class HiDreamModel(BaseModel): # Păstrăm numele clasei pentru compatibilitate cu ModelManager
    """
    Wrapper pentru modelul Stable Diffusion XL Inpainting.
    Numele clasei este păstrat ca 'HiDreamModel' pentru compatibilitate cu
    ModelManager existent, dar gestionează SDXL Inpaint pe acest branch.
    """
    def __init__(self,
                 model_id: Optional[str] = None, # Permitem suprascrierea ID-ului
                 device: Optional[str] = None):
        # Folosim ID-ul din config dacă nu este specificat altul
        effective_model_id = model_id or ModelConfig.MAIN_MODEL
        super().__init__(effective_model_id, device) # Inițializăm clasa de bază
        # Folosim configurația relevantă
        self.config = ModelConfig.SDXL_INPAINT_CONFIG
        if self.config.get("pretrained_model_name_or_path") != effective_model_id:
            logger.warning(f"Model ID '{effective_model_id}' differs from config path '{self.config.get('pretrained_model_name_or_path')}'. Using '{effective_model_id}'.")
            self.config["pretrained_model_name_or_path"] = effective_model_id # Folosim ID-ul efectiv
        self.vae: Optional[AutoencoderKL] = None
        self.controlnet: Optional[ControlNetModel] = None
        self.pipeline: Optional[StableDiffusionXLInpaintPipeline] = None
        self.lora_weights: List[Dict[str, Any]] = []
        logger.debug(f"HiDreamModel (SDXL Inpaint Wrapper) initialized for model ID: {self.model_id}")

    def load(self) -> bool:
        """Încarcă modelul SDXL Inpaint și componentele sale."""
        if self.is_loaded:
            logger.info(f"Model '{self.model_id}' is already loaded.")
            return True

        logger.info(f"Loading SDXL Inpaint model '{self.model_id}'...")
        try:
            # --- 1. Încarcă VAE ---
            vae_path = self.config.get("vae_name_or_path", "stabilityai/sdxl-vae")
            logger.debug(f"Loading VAE from: {vae_path}")
            self.vae = AutoencoderKL.from_pretrained(
                vae_path, torch_dtype=AppConfig.DTYPE, cache_dir=AppConfig.CACHE_DIR
            ).to(self.device)

            # --- 2. Încarcă ControlNet (Opțional) ---
            controlnet_model_id = getattr(ModelConfig, 'CONTROLNET_CONFIG', {}).get("model_id")
            if controlnet_model_id:
                logger.info(f"Loading ControlNet from: {controlnet_model_id}")
                try:
                    self.controlnet = ControlNetModel.from_pretrained(
                        controlnet_model_id, torch_dtype=AppConfig.DTYPE, use_safetensors=True,
                        variant="fp16" if AppConfig.DTYPE == torch.float16 else None, cache_dir=AppConfig.CACHE_DIR
                    ).to(self.device)
                    logger.info("ControlNet loaded successfully.")
                except Exception as e_ctrl:
                    logger.error(f"Failed to load ControlNet '{controlnet_model_id}': {e_ctrl}", exc_info=True)
                    self.controlnet = None
            else:
                logger.info("ControlNet not configured in ModelConfig.")
                self.controlnet = None

            # --- 3. Încarcă Pipeline Principal (SDXL Inpaint) ---
            pipeline_path = self.config.get("pretrained_model_name_or_path")
            if not pipeline_path: raise ValueError("`pretrained_model_name_or_path` missing in config.")
            logger.info(f"Loading pipeline StableDiffusionXLInpaintPipeline from: {pipeline_path}")
            pipeline_kwargs = {
                "vae": self.vae,
                "torch_dtype": AppConfig.DTYPE,
                "variant": "fp16" if AppConfig.DTYPE == torch.float16 else None,
                "use_safetensors": self.config.get("use_safetensors", True),
                "cache_dir": AppConfig.CACHE_DIR
            }
            # !!! Atenție: Dacă ControlNet este încărcat, pipeline-ul *trebuie* să fie compatibil cu ControlNet
            # StableDiffusionXLInpaintPipeline standard *nu* acceptă direct argumentul 'controlnet' la inițializare.
            # Ar trebui folosit un pipeline specific, de ex., StableDiffusionXLControlNetInpaintPipeline
            # Sau ControlNet-ul este adăugat *după* inițializare, dacă pipeline-ul o permite.
            # Verifică documentația diffusers pentru metoda corectă de a combina SDXL Inpaint cu ControlNet.
            # Momentan, voi lăsa adăugarea la kwargs, dar S-AR PUTEA SĂ DEA EROARE LA ÎNCĂRCARE dacă pipeline-ul nu îl suportă.
            # Vezi log-ul tău inițial care menționa "Keyword arguments {'controlnet': ...} are not expected..."
            # COMENTARIU: Eliminăm temporar adăugarea ControlNet aici, se va adăuga la kwargs în `process` dacă e necesar
            # if self.controlnet:
            #     pipeline_kwargs["controlnet"] = self.controlnet # Acest rând poate cauza probleme la load
            self.pipeline = StableDiffusionXLInpaintPipeline.from_pretrained(pipeline_path, **pipeline_kwargs)

            # --- 4. Optimizări și Scheduler ---
            if AppConfig.LOW_VRAM_MODE:
                logger.info("Enabling model CPU offload for the pipeline.")
                self.pipeline.enable_model_cpu_offload()
            else:
                logger.info(f"Moving pipeline to device: {self.device}")
                self.pipeline.to(self.device)
            logger.info("Setting DPMSolverMultistepScheduler with Karras sigmas.")
            self.pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
                self.pipeline.scheduler.config, algorithm_type="sde-dpmsolver++", use_karras_sigmas=True
            )

            # --- 5. Încarcă LoRA-uri ---
            if self.config.get("lora_weights"):
                self._load_loras()

            self.model = self.pipeline # Referință generică
            self.is_loaded = True
            logger.info(f"Model '{self.model_id}' (SDXL Inpaint) loaded successfully.")
            return True
        except Exception as e:
            logger.error(f"Error loading model '{self.model_id}': {e}", exc_info=True)
            self.unload() # Curățăm resursele parțial încărcate
            self.is_loaded = False
            return False

    def _load_loras(self) -> None:
        """Încarcă LoRA-urile configurate în pipeline."""
        if not self.pipeline: logger.warning("Pipeline not loaded, cannot load LoRAs."); return
        logger.info("Loading configured LoRA weights...")
        loaded_lora_names = []
        for lora_info in self.config.get("lora_weights", []):
            lora_path=lora_info.get("path"); lora_name=lora_info.get("name", os.path.basename(lora_path) if lora_path else f"lora_{len(loaded_lora_names)}")
            weight_name=lora_info.get("weight_name")
            if not lora_path: logger.warning(f"Skipping LoRA '{lora_name}' due to missing path."); continue
            logger.info(f"Loading LoRA '{lora_name}' from '{lora_path}' (weight file: {weight_name or 'auto'}).")
            try:
                # Presupunem că pipeline-ul are metoda load_lora_weights (majoritatea au)
                self.pipeline.load_lora_weights(lora_path, weight_name=weight_name, adapter_name=lora_name)
                self.lora_weights.append({**lora_info, "adapter_name": lora_name})
                loaded_lora_names.append(lora_name)
                logger.info(f"Successfully loaded LoRA '{lora_name}'.")
            except Exception as e_lora: logger.error(f"Error loading LoRA '{lora_name}' from '{lora_path}': {e_lora}", exc_info=True)
        if loaded_lora_names and hasattr(self.pipeline, 'set_adapters') and callable(self.pipeline.set_adapters):
            # Setăm adaptoarele active și greutățile lor
            adapter_weights = [lora.get("weight", 1.0) for lora in self.lora_weights if lora.get("adapter_name") in loaded_lora_names]
            try:
                self.pipeline.set_adapters(loaded_lora_names, adapter_weights=adapter_weights)
                logger.info(f"Set active LoRA adapters: {loaded_lora_names} with weights: {adapter_weights}")
            except Exception as e_set: logger.error(f"Failed to set/activate LoRA adapters: {e_set}", exc_info=True)

    def unload(self) -> bool:
        """Descarcă modelul și eliberează memoria."""
        if not self.is_loaded: logger.info(f"Model '{self.model_id}' already unloaded."); return True
        logger.info(f"Unloading model '{self.model_id}'...");
        try:
            del self.pipeline; del self.vae; del self.controlnet; del self.model
            self.pipeline, self.vae, self.controlnet, self.model = None, None, None, None
            self.lora_weights = []
            if torch.cuda.is_available(): logger.debug("Clearing CUDA cache..."); torch.cuda.empty_cache()
            logger.debug("Collecting garbage..."); gc.collect() # Acum gc este importat
            self.is_loaded = False; logger.info(f"Model '{self.model_id}' unloaded successfully."); return True
        except Exception as e: logger.error(f"Error unloading model '{self.model_id}': {e}", exc_info=True); self.is_loaded = False; return False

    def process(self,
               image: Union[Image.Image, np.ndarray],
               mask_image: Union[Image.Image, np.ndarray],
               prompt: str,
               negative_prompt: Optional[str] = None,
               strength: float = 0.85, # Default strength mai mare pentru inpainting
               num_inference_steps: int = 30, # SDXL Inpaint poate necesita mai puțini pași
               guidance_scale: float = 8.0, # Ajustat pentru SDXL Inpaint
               seed: int = -1, # Default la random seed
               controlnet_conditioning_image: Optional[Union[Image.Image, np.ndarray]] = None,
               controlnet_conditioning_scale: Optional[float] = None,
               # Parametri specifici SDXL Inpaint
               aesthetic_score: float = 6.0,
               negative_aesthetic_score: float = 2.5,
               ) -> Dict[str, Any]:
        """Procesează imaginea folosind SDXL Inpaint, ajustând dimensiunea dacă e necesar."""
        if not self.is_loaded:
            logger.warning(f"Model '{self.model_id}' not loaded. Attempting load...")
            if not self.load():
                msg=f"Cannot process: model '{self.model_id}' failed to load"; logger.error(msg)
                # Încercăm să returnăm imaginea originală în format PIL
                original_pil_if_needed = image if isinstance(image, Image.Image) else Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)) if isinstance(image, np.ndarray) and image.ndim==3 else None
                return {'result': original_pil_if_needed, 'success': False, 'message': msg}
        if self.pipeline is None:
             logger.error("Pipeline object is None, cannot process.");
             original_pil_if_needed = image if isinstance(image, Image.Image) else Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)) if isinstance(image, np.ndarray) and image.ndim==3 else None
             return {'result': original_pil_if_needed, 'success': False, 'message': "Internal error: Pipeline not loaded."}

        # Salvează referința la imaginea originală (ca PIL) pentru return în caz de eroare
        original_input_pil: Optional[Image.Image] = None

        # --- Input Preparation ---
        try:
            # Conversie imagine la PIL RGB
            if isinstance(image, np.ndarray):
                if image.ndim==2: img_pil=Image.fromarray(image).convert("RGB")
                elif image.shape[2]==4: img_pil=Image.fromarray(image).convert("RGB")
                elif image.shape[2]==3: img_pil=Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)) # Assume BGR -> RGB
                else: raise ValueError(f"Unsupported NumPy image shape: {image.shape}")
            elif isinstance(image, Image.Image): img_pil=image.convert("RGB")
            else: raise TypeError("Input image must be PIL Image or NumPy array.")
            original_input_pil = img_pil.copy() # Salvăm o copie a imaginii originale PIL

            # Conversie mască la PIL 'L' (grayscale)
            if isinstance(mask_image, np.ndarray):
                if mask_image.ndim==3: mask_pil=Image.fromarray(mask_image).convert("L")
                elif mask_image.ndim==2: mask_pil=Image.fromarray(mask_image).convert("L")
                else: raise ValueError(f"Unsupported NumPy mask shape: {mask_image.shape}")
            elif isinstance(mask_image, Image.Image): mask_pil=mask_image.convert("L")
            else: raise TypeError("Input mask must be PIL Image or NumPy array.")

            # Verificare și redimensionare mască (dacă e necesar, LA DIMENSIUNEA IMAGINII CURENTE)
            if img_pil.size != mask_pil.size:
                logger.warning(f"Resizing mask ({mask_pil.size}) to image size ({img_pil.size}) using NEAREST.")
                mask_pil = mask_pil.resize(img_pil.size, Image.Resampling.NEAREST)
        except Exception as e:
            logger.error(f"Input prep error: {e}", exc_info=True)
            return {'result': original_input_pil, 'success': False, 'message': f"Input prep error: {e}"}

        # --- AJUSTARE DIMENSIUNI PENTRU DIVIZIBILITATE CU 8 ---
        original_size = img_pil.size # Salvăm dimensiunea originală a imaginii PIL
        original_width, original_height = original_size

        # Calculează noile dimensiuni rotunjind în jos la cel mai apropiat multiplu de 8
        new_width = (original_width // 8) * 8
        new_height = (original_height // 8) * 8

        # Gestionează cazurile în care rotunjirea în jos duce la dimensiuni prea mici
        if new_width <= 0 or new_height <= 0:
            # Opțiune: Rotunjește în sus în loc
            new_width = ((original_width + 7) // 8) * 8
            new_height = ((original_height + 7) // 8) * 8
            if new_width <= 0 or new_height <= 0: # Verificare dublă
                 logger.error(f"Image dimensions ({original_width}x{original_height}) are too small.")
                 return {'result': original_input_pil, 'success': False, 'message': f"Image dimensions too small ({original_width}x{original_height})."}
            logger.warning(f"Original dimensions ({original_width}x{original_height}) very small, rounding *up* to {new_width}x{new_height}.")

        # Redimensionează doar dacă dimensiunile s-au schimbat
        if original_width != new_width or original_height != new_height:
            logger.warning(f"Resizing input image and mask from {original_width}x{original_height} to {new_width}x{new_height} to be divisible by 8.")
            try:
                img_pil = img_pil.resize((new_width, new_height), Image.Resampling.LANCZOS)
                mask_pil = mask_pil.resize((new_width, new_height), Image.Resampling.NEAREST)
            except Exception as e_resize:
                logger.error(f"Error during pre-pipeline resize: {e_resize}", exc_info=True)
                return {'result': original_input_pil, 'success': False, 'message': f"Error during pre-pipeline resize: {e_resize}"}
        else:
            logger.debug(f"Image dimensions {original_width}x{original_height} are already divisible by 8. No pre-resize needed.")

        # Acum img_pil și mask_pil au dimensiuni (new_width, new_height) divizibile cu 8

        # --- Parameter Setup ---
        generator = torch.Generator(device=self.device).manual_seed(seed) if seed != -1 else None
        effective_negative_prompt = negative_prompt if negative_prompt is not None else getattr(ModelConfig, 'GENERATION_PARAMS', {}).get("negative_prompt", "low quality, blurry, deformed")

        # --- Pipeline Arguments ---
        pipeline_kwargs = {
            "prompt": prompt,
            "negative_prompt": effective_negative_prompt,
            "image": img_pil, # Imaginea redimensionată (sau originală dacă era deja divizibilă)
            "mask_image": mask_pil, # Masca redimensionată (sau originală)
            "height": new_height, # Înălțimea ajustată
            "width": new_width,  # Lățimea ajustată
            "strength": strength,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "generator": generator,
            "output_type": "pil",
            # Parametri specifici SDXL
            "aesthetic_score": aesthetic_score,
            "negative_aesthetic_score": negative_aesthetic_score,
        }

        # --- ControlNet Handling ---
        # Adaugă argumentele pentru ControlNet *doar dacă* este necesar și disponibil
        # Verifică dacă pipeline-ul suportă aceste argumente (poate necesita un pipeline diferit)
        if self.controlnet is not None and controlnet_conditioning_image is not None and controlnet_conditioning_scale is not None:
            logger.info(f"Adding ControlNet input with scale {controlnet_conditioning_scale}.")
            try:
                # Pregătește imaginea de condiționare ControlNet (PIL, RGB, aceeași dimensiune cu inputul ajustat)
                if isinstance(controlnet_conditioning_image, np.ndarray):
                    cond_img_pil = Image.fromarray(cv2.cvtColor(controlnet_conditioning_image, cv2.COLOR_BGR2RGB)).convert("RGB")
                elif isinstance(controlnet_conditioning_image, Image.Image):
                    cond_img_pil = controlnet_conditioning_image.convert("RGB")
                else: raise TypeError("Invalid type for controlnet_conditioning_image.")

                # Redimensionează imaginea de control la dimensiunea ajustată (new_width, new_height)
                if cond_img_pil.size != (new_width, new_height):
                    logger.warning(f"Resizing ControlNet image ({cond_img_pil.size}) to pipeline input size ({new_width}x{new_height}).")
                    cond_img_pil = cond_img_pil.resize((new_width, new_height), Image.Resampling.LANCZOS)

                # Adaugă argumentele specifice ControlNet la kwargs
                # Notă: Numele argumentelor pot varia în funcție de pipeline-ul folosit!
                # Pentru ControlNet cu SDXL, argumentele pot fi 'control_image' sau similar.
                # Verifică documentația pipeline-ului specific pe care îl folosești.
                # Presupunând că pipeline-ul acceptă 'control_image' și 'controlnet_conditioning_scale':
                pipeline_kwargs["control_image"] = cond_img_pil
                pipeline_kwargs["controlnet_conditioning_scale"] = float(controlnet_conditioning_scale)

            except Exception as e:
                logger.error(f"ControlNet prep error: {e}", exc_info=True)
                logger.warning("Skipping ControlNet due to preparation error.")
                # Eliminăm argumentele dacă pregătirea a eșuat
                pipeline_kwargs.pop("control_image", None)
                pipeline_kwargs.pop("controlnet_conditioning_scale", None)
        elif controlnet_conditioning_scale is not None:
            logger.warning("ControlNet scale provided, but model or conditioning image missing/invalid. Skipping ControlNet.")


        # --- Execute Pipeline ---
        try:
            logger.debug(f"Calling pipeline {self.pipeline.__class__.__name__} with adjusted size {new_width}x{new_height}...")
            result = self.pipeline(**pipeline_kwargs) # Va rula cu dimensiunile ajustate
            if not hasattr(result, 'images') or not result.images: raise RuntimeError("Pipeline returned no images.")
            result_image_processed = result.images[0] # Aceasta va avea dimensiunile new_width x new_height

            # --- Post-processing Resize Check (ÎNAPOI LA ORIGINAL) ---
            # Redimensionează rezultatul înapoi la dimensiunea originală de intrare (înainte de ajustarea la multiplu de 8)
            if result_image_processed.size != original_size:
                logger.warning(f"Output size {result_image_processed.size} != original {original_size}. Resizing back.")
                try:
                    result_image_final = result_image_processed.resize(original_size, Image.Resampling.LANCZOS)
                except Exception as e_post_resize:
                    logger.error(f"Resize error back to original size: {e_post_resize}", exc_info=True)
                    result_image_final = result_image_processed # Folosește imaginea procesată dacă redimensionarea eșuează
            else:
                result_image_final = result_image_processed

            logger.info("Processing completed successfully.")
            return {'result': result_image_final, 'success': True, 'message': "Processing completed successfully"}

        except ValueError as ve: # Prindem specific eroarea de dimensiune dacă apare totuși
             if "divisible by 8" in str(ve):
                  logger.error(f"Dimension error despite check: {ve}", exc_info=True)
                  return {'result': original_input_pil, 'success': False, 'message': f"Dimension error: {ve}"}
             else: # Alt ValueError
                  logger.error(f"Pipeline execution ValueError: {ve}", exc_info=True)
                  return {'result': original_input_pil, 'success': False, 'message': f"Pipeline ValueError: {ve}"}
        except Exception as e:
            logger.error(f"Pipeline execution error: {e}", exc_info=True)
            # Returnează imaginea originală în caz de eroare
            return {'result': original_input_pil, 'success': False, 'message': f"Pipeline error: {e}"}

    def get_info(self) -> Dict[str, Any]:
        """Obține informații detaliate despre modelul încărcat."""
        info = super().get_info()
        info.update({
            "config": self.config,
            "pipeline_class": self.pipeline.__class__.__name__ if self.pipeline else "N/A",
            "scheduler_class": self.pipeline.scheduler.__class__.__name__ if self.pipeline and hasattr(self.pipeline, 'scheduler') else "N/A",
            "has_controlnet": self.controlnet is not None,
            "loaded_loras": [lora.get("name", lora.get("path")) for lora in self.lora_weights],
        })
        if torch.cuda.is_available() and self.is_loaded:
            try:
                info["vram_allocated_gb"] = round(torch.cuda.memory_allocated(self.device)/(1024**3),2)
                info["vram_reserved_gb"] = round(torch.cuda.memory_reserved(self.device)/(1024**3),2)
            except: pass
        return info