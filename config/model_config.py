#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Configurații pentru modelele AI folosite în FusionFrame 2.0
"""

class ModelConfig:
    """Configurare pentru modelele AI"""
    
    # Modele principale pentru editare
    MAIN_MODEL = "HiDream-I1-Full"  # Schimbat din HiDream-E1-Full la HiDream-I1-Full
    BACKUP_MODEL = "HiDream-I1-Fast"  # Model alternativ cu mai puține pași (16 vs 50)
    
    # Configurări pentru modelul principal
    HIDREAM_CONFIG = {
        "pretrained_model_name_or_path": "HiDream-ai/HiDream-I1-Full",  # URL-ul modelului HiDream-I1-Full
        "vae_name_or_path": "stabilityai/sdxl-vae",  # Se păstrează VAE-ul de la SDXL
        "custom_pipeline": "hidream_pipeline",
        "use_safetensors": True,
        "lora_weights": [],  # Lista de LoRA-uri care vor fi încărcate
        "inference_steps": 50  # Pași de inferență conform tabelului pentru HiDream-I1-Full
    }
    
    # Configurări pentru refiner SDXL
    REFINER_CONFIG = {
        "enabled": True,  # Activăm refiner-ul implicit
        "pretrained_model_name_or_path": "stabilityai/stable-diffusion-xl-refiner-1.0",
        "vae_name_or_path": "stabilityai/sdxl-vae",
        "use_safetensors": True,
        "inference_steps": 25  # Jumătate din pașii modelului principal pentru refiner
    }
    
    # Configurări pentru modelul de backup
    FLUX_CONFIG = {
        "pretrained_model_name_or_path": "HiDream-ai/HiDream-I1-Fast",  # Versiunea Fast pentru backup
        "vae_name_or_path": "stabilityai/sdxl-vae",
        "custom_pipeline": "hidream_pipeline",
        "use_safetensors": True,
        "lora_weights": [],  # Lista de LoRA-uri care vor fi încărcate
        "inference_steps": 16  # Pași de inferență conform tabelului pentru HiDream-I1-Fast
    }
    
    # Configurări pentru modelele auxiliare
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
        "default_steps": 50,  # Conform tabelului pentru HiDream-I1-Full
        "max_steps": 80,
        "guidance_scale": 7.5,
        "negative_prompt": "blurry, distorted, deformed, low quality, low resolution, bad anatomy, artifacts, watermark"
    }