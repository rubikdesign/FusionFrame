#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Configurații pentru modelele AI folosite în FusionFrame 2.0
"""

class ModelConfig:
    """Configurare pentru modelele AI"""
    
    # Modele principale pentru editare
    MAIN_MODEL = "HiDream-E1-Full"  # model specializat pentru editare bazată pe instrucțiuni
    BACKUP_MODEL = "FLUX.1-dev"     # model alternativ pentru cazuri specifice
    
    # Configurări pentru modelul principal
    HIDREAM_CONFIG = {
        "pretrained_model_name_or_path": "stabilityai/stable-diffusion-xl-base-1.0",
        "vae_name_or_path": "stabilityai/sdxl-vae",
        "custom_pipeline": "hidream_pipeline",
        "use_safetensors": True,
        "lora_weights": []  # Lista de LoRA-uri care vor fi încărcate
    }
    
    # Configurări pentru modelul de backup
    FLUX_CONFIG = {
        "pretrained_model_name_or_path": "stabilityai/stable-diffusion-xl-base-1.0",
        "vae_name_or_path": "stabilityai/sdxl-vae",
        "custom_pipeline": "flux_pipeline",
        "use_safetensors": True,
        "lora_weights": []  # Lista de LoRA-uri care vor fi încărcate
    }
    
    # Configurări pentru modele auxiliare
    SAM_CONFIG = {
        "model_type": "vit_h",
        "checkpoint": "sam_vit_h_4b8939.pth",
        "points_per_side": 32,
        "pred_iou_thresh": 0.95,
        "stability_score_thresh": 0.97,
        "min_mask_region_area": 100
    }
    
    CONTROLNET_CONFIG = {
        "model_id": "diffusers/controlnet-canny-sdxl-1.0",
        "conditioning_scale": 0.7
    }
    
    CLIP_CONFIG = {
        "model_id": "CIDAS/clipseg-rd64-refined"
    }
    
    # Configurări pentru LoRA
    LORA_CONFIG = {
        "max_loras": 3,  # Număr maxim de LoRA-uri active simultan
        "weight_range": (-1.0, 2.0)  # Intervalul permis pentru greutatea LoRA-urilor
    }
    
    # Parametrii pentru generare
    GENERATION_PARAMS = {
        "default_steps": 50,
        "max_steps": 80,
        "guidance_scale": 7.5,
        "negative_prompt": "blurry, distorted, deformed, low quality, low resolution, bad anatomy, artifacts, watermark"
    }