#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Configurații pentru modelele AI folosite în FusionFrame 2.0
"""

class ModelConfig:
    """Configurare centralizată pentru modelele AI și parametrii lor."""

    # --- Modele Principale și de Backup ---
    MAIN_MODEL_NAME = "HiDream-I1-Full"
    BACKUP_MODEL_NAME = "HiDream-I1-Fast"

    # --- Configurări Detaliate pentru Modele Specifice ---

    # Model Principal (HiDream-I1 Full)
    HIDREAM_CONFIG = {
        "pretrained_model_name_or_path": "HiDream-ai/HiDream-I1-Full",
        "vae_name_or_path": "stabilityai/sdxl-vae",
        "use_safetensors": True,
        "lora_weights": [],
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

    # ControlNet (SDXL Canny)
    CONTROLNET_CONFIG = {
        "model_id": "diffusers/controlnet-canny-sdxl-1.0",
        "conditioning_scale": 0.7
    }

    # CLIPSeg
    CLIP_CONFIG = {
        "model_id": "CIDAS/clipseg-rd64-refined"
    }

    # Image Classifier (ViT)
    IMAGE_CLASSIFIER_CONFIG = {
        "model_id": "google/vit-base-patch16-224",
        "top_n_results": 5
    }

    # Depth Estimator (DPT/MiDaS)
    DEPTH_ESTIMATOR_CONFIG = {
        "model_id": "Intel/dpt-hybrid-midas"
    }

    # Object Detector (YOLO) - Parametri specifici analizei
    OBJECT_DETECTOR_CONFIG = {
        # Presupunem că ModelManager încarcă YOLO sub cheia 'yolo'
        # Poate fi adăugat 'model_name' aici dacă gestionăm mai multe YOLO.
        "confidence_threshold": 0.4, # Prag pentru analiza obiectelor
        "iou_threshold": 0.5          # Prag IoU pentru NMS în YOLO predict
    }

    # MediaPipe (Selfie Seg & Face Detection)
    # Șterge/Comentează din AppConfig dacă sunt definite aici
    MEDIAPIPE_SELFIE_MODEL_SELECTION = 1
    MEDIAPIPE_FACE_MODEL_SELECTION = 0
    MEDIAPIPE_FACE_MIN_CONFIDENCE = 0.5

    # Rembg
    # Șterge/Comentează din AppConfig dacă este definit aici
    REMBG_MODEL_NAME = "u2net"

    # --- Alte Configurări Globale ---

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