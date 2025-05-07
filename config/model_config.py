#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Configurații pentru modelele AI folosite în FusionFrame 2.0 (Master Branch)
"""

class ModelConfig:
    """Configurare pentru modelele AI"""
    
    # Modelul principal pentru editare pe MASTER este SDXL Inpaint
    # Folosim un ID de model SDXL Inpainting cunoscut. 
    # Puteți alege și altele dacă preferați (ex: stabilityai/stable-diffusion-xl-refiner-1.0 poate face și inpaint, dar e mai mult refiner)
    MAIN_MODEL = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1" 
    
    # Modelul de backup nu este relevant dacă folosim doar SDXL Inpaint pe master
    # BACKUP_MODEL = "FLUX.1-dev" # Comentat sau eliminat pe master

    # Redenumim secțiunea de configurare pentru claritate (Opțional, dar recomandat)
    # Sau păstrăm HIDREAM_CONFIG dar cu valorile corecte pentru SDXL Inpaint
    SDXL_INPAINT_CONFIG = { # Redenumit din HIDREAM_CONFIG
        "pretrained_model_name_or_path": MAIN_MODEL, # Folosim ID-ul corect
        "vae_name_or_path": "stabilityai/sdxl-vae",  # VAE standard pentru SDXL
        # "custom_pipeline": None, # Nu folosim un pipeline custom pentru SDXL Inpaint standard
        "use_safetensors": True,
        "lora_weights": [] 
    }
    # Păstrăm HIDREAM_CONFIG cu valorile corecte dacă ModelManager/HiDreamModel se bazează pe acest nume
    HIDREAM_CONFIG = SDXL_INPAINT_CONFIG # Alias pentru compatibilitate dacă e necesar

    # Configurarea Refiner NU este folosită implicit cu SDXL Inpaint standard
    # REFINER_CONFIG = { ... } # Comentat sau eliminat pe master

    # Configurări pentru modele auxiliare (rămân la fel)
    SAM_CONFIG = {
        "model_type": "vit_h",
        "checkpoint": "sam_vit_h_4b8939.pth",
        "points_per_side": 32,
        "pred_iou_thresh": 0.95,
        "stability_score_thresh": 0.97,
        "min_mask_region_area": 100
    }
    
    CONTROLNET_CONFIG = {
        # Asigurăm compatibilitate SDXL
        "model_id": "diffusers/controlnet-canny-sdxl-1.0", 
        # Scala se setează la inferență, nu aici
        # "conditioning_scale": 0.7 
    }
    
    CLIP_CONFIG = {
        "model_id": "CIDAS/clipseg-rd64-refined"
    }
    
    LORA_CONFIG = {
        "max_loras": 3,
        "weight_range": (-1.0, 2.0)
    }
    
    GENERATION_PARAMS = {
        "default_steps": 30, # SDXL poate necesita mai puțini pași
        "max_steps": 80,
        "guidance_scale": 8.0, # Valoare comună pentru SDXL Inpaint
        "negative_prompt": "blurry, distorted, deformed, low quality, low resolution, bad anatomy, artifacts, watermark, text, words, letters, signature" # Prompt negativ specific
    }

