#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Configurations for AI models used in FusionFrame 2.0
Enhanced with lightweight model alternatives and memory optimizations
"""

class ModelConfig:
    """Centralized configuration for AI models and their parameters."""

    # --- Main and Backup Models ---
    MAIN_MODEL = "HiDream-I1-Fast"
    BACKUP_MODEL_NAME = "HiDream-I1-Full"

    # --- Lightweight Model Alternatives ---
    # Use these models when in LOW_VRAM_MODE
    LIGHTWEIGHT_MODELS = {
        "yolo": "yolov8n-seg.pt",  # Nano variant is much smaller than standard
        "image_classifier": "google/vit-base-patch16-224-in21k",  # Smaller than default
        "depth_estimator": "Intel/dpt-large",  # Smaller than full model
        "sam": "vit_b",  # Use SAM base model instead of huge
        "rembg": "u2netp"  # Use smaller rembg model
    }

    # --- Memory Optimization Settings ---
    # Inference step reduction for low memory environments
    LOW_VRAM_INFERENCE_STEPS = {
        "HiDream-I1-Full": 30,   # Reduced from 50
        "HiDream-I1-Fast": 12,   # Reduced from 16
        "refiner": 15           # Reduced from 25
    }

    # --- Detailed Configurations for Specific Models ---

    # Main Model (HiDream-I1 Full)
    BACKUP_CONFIG = {
        "pretrained_model_name_or_path": "HiDream-ai/HiDream-I1-Full",
        "vae_name_or_path": "stabilityai/sdxl-vae",
        "use_safetensors": True,
        "lora_weights": [],
        "inference_steps": 50
    }

    # Backup Model (HiDream-I1 Fast)
    HIDREAM_CONFIG = {
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

    # --- Auxiliary Models (Segmentation, Detection, Analysis) ---

    # Segment Anything Model (SAM)
    SAM_CONFIG = {
        "model_type": "vit_h",  # Standard: "vit_h", Lightweight: "vit_b"
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
        "model_id": "CIDAS/clipseg-rd64-refined",  # Smaller: "CIDAS/clipseg-rd16"
        "cpu_offload": True,  # Offload to CPU when not in use
        "quantize": True  # Use quantization to reduce memory footprint
    }

    # Image Classifier (ViT)
    IMAGE_CLASSIFIER_CONFIG = {
        "model_id": "google/vit-base-patch16-224",  # Standard model
        "lightweight_model_id": "google/vit-base-patch16-224-in21k",  # Smaller alternative
        "top_n_results": 5,
        "cpu_offload": True  # Can run on CPU with minimal performance impact
    }

    # Depth Estimator (DPT/MiDaS)
    DEPTH_ESTIMATOR_CONFIG = {
        "model_id": "Intel/dpt-hybrid-midas",  # Standard model
        "lightweight_model_id": "Intel/dpt-large",  # Smaller alternative
        "cpu_offload": True  # Better on CPU than losing other models
    }

    # Object Detector (YOLO) - Specific analysis parameters
    OBJECT_DETECTOR_CONFIG = {
        # Standard model is usually "yolov8x-seg.pt"
        "model_id": "yolov8x-seg.pt",
        "lightweight_model_id": "yolov8n-seg.pt",  # Nano variant - much smaller
        "confidence_threshold": 0.4,  # Threshold for object analysis
        "iou_threshold": 0.5,         # IoU threshold for NMS in YOLO predict
        "img_size": 640,              # Standard size for prediction
        "lightweight_img_size": 320    # Smaller size for low VRAM mode
    }

    # MediaPipe (Selfie Seg & Face Detection)
    MEDIAPIPE_SELFIE_MODEL_SELECTION = 1
    MEDIAPIPE_FACE_MODEL_SELECTION = 0
    MEDIAPIPE_FACE_MIN_CONFIDENCE = 0.5

    # Rembg
    REMBG_MODEL_NAME = "u2net"  # Standard model
    REMBG_LIGHTWEIGHT_MODEL_NAME = "u2netp"  # Smaller model

    # --- Other Global Configurations ---

    # LoRA
    LORA_CONFIG = {
        "max_loras": 3,
        "weight_range": (-1.0, 2.0)
    }

    # General Generation Parameters
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
    
    # --- Memory-specific Operation Parameters ---
    # Default parameter adjustments for low VRAM mode
    LOW_VRAM_PARAMS = {
        "remove": {
            "strength": 0.85,
            "num_inference_steps": 40,  # Reduced from 60
            "guidance_scale": 7.0
        },
        "replace": {
            "strength": 0.90,
            "num_inference_steps": 45,  # Reduced from 70
            "guidance_scale": 7.0
        },
        "color": {
            "strength": 0.65,
            "num_inference_steps": 30,  # Reduced from 45
            "guidance_scale": 7.0
        },
        "background": {
            "strength": 0.80,
            "num_inference_steps": 40,  # Reduced from 65
            "guidance_scale": 7.0
        },
        "add": {
            "strength": 0.75,
            "num_inference_steps": 35,  # Reduced from original
            "guidance_scale": 7.0
        },
        "general": {
            "strength": 0.75,
            "num_inference_steps": 30,  # General reduction
            "guidance_scale": 7.0
        }
    }