#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Configurații pentru modelele AI folosite în FusionFrame 2.0
"""

class ModelConfig:
    """Configurare centralizată pentru modelele AI și parametrii lor."""

    # --- Modele Principale și de Backup ---
    MAIN_MODEL_NAME = "HiDream-I1-Full"  # Numele cheie pentru modelul principal (folosit intern)
    BACKUP_MODEL_NAME = "HiDream-I1-Fast" # Numele cheie pentru modelul de backup (folosit intern)

    # --- Configurări Detaliate pentru Modele Specifice ---

    # Model Principal (HiDream-I1 Full)
    HIDREAM_CONFIG = {
        "pretrained_model_name_or_path": "HiDream-ai/HiDream-I1-Full",
        "vae_name_or_path": "stabilityai/sdxl-vae",
        "use_safetensors": True,
        "lora_weights": [],  # Lista de LoRA-uri active (structură: [{"name": "lora1", "path": "path1", "weight": 0.8}, ...])
        "inference_steps": 50
    }

    # Model de Backup (HiDream-I1 Fast)
    BACKUP_CONFIG = {
        "pretrained_model_name_or_path": "HiDream-ai/HiDream-I1-Fast",
        "vae_name_or_path": "stabilityai/sdxl-vae",
        "use_safetensors": True,
        "lora_weights": [],
        "inference_steps": 16
    }

    # Refiner (SDXL)
    REFINER_CONFIG = {
        "enabled": True,
        "pretrained_model_name_or_path": "stabilityai/stable-diffusion-xl-refiner-1.0",
        "vae_name_or_path": "stabilityai/sdxl-vae",
        "use_safetensors": True,
        "inference_steps": 25
    }

    # --- Modele Auxiliare (Segmentare, Detecție, Analiză) ---

    # Segment Anything Model (SAM)
    SAM_CONFIG = {
        "model_type": "vit_h",
        "checkpoint": "sam_vit_h_4b8939.pth",
        "points_per_side": 32,
        "pred_iou_thresh": 0.95,
        "stability_score_thresh": 0.97,
        "min_mask_region_area": 100
    }

    # ControlNet (pentru ghidare condiționată SDXL)
    CONTROLNET_CONFIG = {
        "model_id": "diffusers/controlnet-canny-sdxl-1.0",
        "conditioning_scale": 0.7
    }

    # CLIPSeg (pentru segmentare bazată pe text)
    CLIP_CONFIG = {
        "model_id": "CIDAS/clipseg-rd64-refined"
    }

    # Image Classifier (ViT)
    IMAGE_CLASSIFIER_CONFIG = {
        "model_id": "google/vit-base-patch16-224",
        "top_n_results": 5 # Numărul de etichete relevante de returnat
    }

    # NOU: Depth Estimator (DPT/MiDaS)
    DEPTH_ESTIMATOR_CONFIG = {
        "model_id": "Intel/dpt-hybrid-midas" # Un bun echilibru între viteză și calitate
        # Alternative: "Intel/dpt-large" (mai precis, mai lent), modele specifice MiDaS
    }

    # MediaPipe (segmentare selfie & detecție față)
    # Notă: Acestea trebuie șterse/comentate din AppConfig dacă sunt definite aici
    MEDIAPIPE_SELFIE_MODEL_SELECTION = 1 # 0: landscape, 1: general
    MEDIAPIPE_FACE_MODEL_SELECTION = 0   # 0: short-range, 1: full-range
    MEDIAPIPE_FACE_MIN_CONFIDENCE = 0.5

    # Rembg (eliminare fundal)
    # Notă: Acesta trebuie șters/comentat din AppConfig dacă este definit aici
    REMBG_MODEL_NAME = "u2net"

    # --- Alte Configurări Globale Legate de Modele ---

    # LoRA
    LORA_CONFIG = {
        "max_loras": 3,
        "weight_range": (-1.0, 2.0)
    }

    # Parametri Generali de Generare
    GENERATION_PARAMS = {
        "default_steps": 50,
        "max_steps": 100,
        "guidance_scale": 7.5,
        "negative_prompt": (
            "ugly, deformed, noisy, blurry, low contrast, realism, photorealistic, illustration, painting, drawing, "
            "sketch, cartoon, anime, render, 3d, watermark, signature, text, label, duplicate, morbid, mutilated, "
            "extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, disgusting, bad anatomy, "
            "bad proportions, extra limbs, cloned face, gross proportions, malformed limbs, missing arms, missing legs, "
            "extra arms, extra legs, fused fingers, too many fingers, long neck, username, artist name"
        )
    }