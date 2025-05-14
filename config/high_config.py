#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Configurations for AI models used in FusionFrame 2.0
Enhanced with lightweight model alternatives and memory optimizations
"""

class ModelConfig:
    """Centralized configuration for AI models and their parameters."""

    # --- Main Model Selection ---
    MAIN_MODEL = "RunDiffusion/Juggernaut-XL-v9"
    BACKUP_MODEL_NAME = "HiDream-I1-Fast"

    # --- Configuration for SDXL Inpainting Models (e.g., Juggernaut-XL-v9) ---
    SDXL_INPAINT_CONFIG = {
        # Folosim calea locală către fișierul safetensors (descărcat direct)
        "pretrained_model_name_or_path": "/workspace/FusionFrame/models/juggernaut/Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors",
        # VAE-ul este inclus în model
        "vae_name_or_path": None, 
        "use_safetensors": True,
        "load_from_single_file": True,  # Important: folosim from_single_file pentru modelul descărcat
        "lora_weights": [],
        "inference_steps": 30,
    }

    # --- FLUX Configuration (păstrat dacă vrei să comuți la el ulterior) ---
    FLUX_CONFIG = {
        "pretrained_model_name_or_path": "black-forest-labs/FLUX.1-dev",
        "vae_name_or_path": "stabilityai/sdxl-vae", # FLUX ar putea necesita un VAE specific SDXL
        "use_safetensors": True,
        "lora_weights": [],
        "inference_steps": 30 # FLUX poate necesita mai puțini pași (ex: 8-20 conform documentației)
    }


    # --- HiDream Configuration (pentru backup sau comparație) ---
    HIDREAM_CONFIG = { # Redenumit din BACKUP_CONFIG pentru claritate
        "pretrained_model_name_or_path": "HiDream-ai/HiDream-I1-Fast",
        "vae_name_or_path": "stabilityai/sdxl-vae",
        "use_safetensors": True,
        "lora_weights": [],
        "inference_steps": 16 # HiDream Fast este rapid
    }


    # Refiner (SDXL) - Poate fi folosit cu Juggernaut XL
    REFINER_CONFIG = {
        "enabled": True, # Setează la True în AppConfig dacă vrei să-l folosești
        "pretrained_model_name_or_path": "stabilityai/stable-diffusion-xl-refiner-1.0",
        "vae_name_or_path": "stabilityai/sdxl-vae", # Adesea același VAE ca modelul de bază
        "use_safetensors": True,
        "inference_steps": 20 # Pași pentru refiner
    }

    # --- Lightweight Model Alternatives ---
    LIGHTWEIGHT_MODELS = {
        "yolo": "yolov8n-seg.pt",
        "image_classifier": "google/vit-base-patch16-224-in21k",
        "depth_estimator": "Intel/dpt-large",
        "sam": "vit_b", # Model SAM mai mic (ex: "facebook/sam-vit-base")
        "rembg": "u2netp"
    }

    # --- Memory Optimization Settings ---
    LOW_VRAM_INFERENCE_STEPS = {
        "RunDiffusion/Juggernaut-XL-v9": 20, # Pași reduși pentru Juggernaut în low VRAM
        "HiDream-I1-Fast": 12,
        "FLUX.1-dev": 15, # Pași reduși pentru FLUX în low VRAM
        "refiner": 10
    }

    # --- Auxiliary Models (Segmentation, Detection, Analysis) ---
    SAM_CONFIG = {
        "model_type": "vit_h",  # Standard: "vit_h". În LOW_VRAM_MODE, ModelManager ar trebui să aleagă "vit_b"
        "checkpoint": "sam_vit_h_4b8939.pth", # Numele fișierului descărcat
        # ... restul parametrilor SAM ...
    }

    CONTROLNET_CONFIG = {
        "model_id": "diffusers/controlnet-canny-sdxl-1.0", # Compatibil cu SDXL
        "conditioning_scale": 0.7
    }

    CLIP_CONFIG = {
        "model_id": "CIDAS/clipseg-rd64-refined",
        "cpu_offload": True,
        "quantize": True # Verifică dacă modelul suportă/beneficiază de cuantizare
    }

    IMAGE_CLASSIFIER_CONFIG = {
        "model_id": "google/vit-base-patch16-224",
        "lightweight_model_id": "google/vit-base-patch16-224-in21k",
        "top_n_results": 5,
        "cpu_offload": True
    }

    DEPTH_ESTIMATOR_CONFIG = {
        "model_id": "Intel/dpt-hybrid-midas",
        "lightweight_model_id": "Intel/dpt-large",
        "cpu_offload": True
    }

    OBJECT_DETECTOR_CONFIG = {
        "model_id": "yolov8x-seg.pt",
        "lightweight_model_id": "yolov8n-seg.pt",
        "confidence_threshold": 0.4,
        "iou_threshold": 0.5,
        "img_size": 640,
        "lightweight_img_size": 320
    }

    MEDIAPIPE_SELFIE_MODEL_SELECTION = 1
    MEDIAPIPE_FACE_MODEL_SELECTION = 0 # sau 1 pentru full-range
    MEDIAPIPE_FACE_MIN_CONFIDENCE = 0.5

    REMBG_MODEL_NAME = "u2net"
    REMBG_LIGHTWEIGHT_MODEL_NAME = "u2netp"

    # --- Other Global Configurations ---
    LORA_CONFIG = {
        "max_loras": 5, # Poți crește dacă ai VRAM
        "weight_range": (-2.0, 2.0)
    }

    GENERATION_PARAMS = {
        "default_steps": 40, # Poate fi mai mare pentru SDXL (30-50 e un interval bun)
        "max_steps": 100,
        "guidance_scale": 7.0, # SDXL adesea arată bine cu guidance mai mic (5-8)
        "negative_prompt": (
            "ugly, deformed, noisy, blurry, low contrast, realism, photorealistic, illustration, painting, drawing, "
            "sketch, cartoon, anime, render, 3d, watermark, signature, text, label, duplicate, morbid, mutilated, "
            "extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, disgusting, bad anatomy, "
            "bad proportions, extra limbs, cloned face, gross proportions, malformed limbs, missing arms, missing legs, "
            "extra arms, extra legs, fused fingers, too many fingers, long neck, username, artist name, low quality, worst quality, jpeg artifacts"
        ) # Am adăugat câteva negative prompts comune
    }

    # --- Memory-specific Operation Parameters ---
    # Acestea pot rămâne, dar numărul de pași va fi influențat și de LOW_VRAM_INFERENCE_STEPS
    LOW_VRAM_PARAMS = {
        "remove": {"strength": 0.85, "num_inference_steps": 25, "guidance_scale": 7.0},
        "replace": {"strength": 0.90, "num_inference_steps": 30, "guidance_scale": 7.0},
        "color": {"strength": 0.65, "num_inference_steps": 20, "guidance_scale": 7.0},
        "background": {"strength": 0.80, "num_inference_steps": 25, "guidance_scale": 7.0},
        "add": {"strength": 0.75, "num_inference_steps": 25, "guidance_scale": 7.0},
        "general": {"strength": 0.75, "num_inference_steps": 20, "guidance_scale": 7.0}
    }