#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Gradio interface components for FusionFrame 2.0
"""

import gradio as gr
from typing import List, Tuple, Any

from config.app_config import AppConfig

def create_examples() -> List[List[Any]]:
    """
    Create the list of examples for the interface
    
    Returns:
        List of examples
    """
    examples = [
        # Basic operations
        ["remove the car from the street", 0.8],
        ["change hair color to bright pink", 0.7],
        ["replace background with futuristic city", 0.85],
        ["remove watermark from top right corner", 0.75],
        # Additional advanced examples
        ["remove person from the image", 0.9],
        ["change the shirt color to blue", 0.65],
        ["replace the sky with sunset colors", 0.8],
        ["erase all text from the image", 0.85],
        ["make the eyes green", 0.6],
        ["replace glasses with sunglasses", 0.75],
        ["add glasses", 0.65]
    ]
    
    return examples

def create_advanced_settings_panel() -> List[gr.components.Component]:
    """
    Create the advanced settings panel
    
    Returns:
        List of Gradio components
    """
    with gr.Row():
        num_steps = gr.Slider(
            minimum=20, 
            maximum=100, 
            value=50, 
            step=1, 
            label="Inference Steps",
            info="More steps generally give better quality but take longer"
        )
        guidance = gr.Slider(
            minimum=1.0, 
            maximum=15.0, 
            value=7.5, 
            step=0.5, 
            label="Guidance Scale",
            info="How closely to follow the prompt (higher = more faithful but less creative)"
        )
    
    with gr.Row():
        enhance_details = gr.Checkbox(
            value=True, 
            label="Enhance Details", 
            info="Improve details in the final result"
        )
        fix_faces = gr.Checkbox(
            value=True, 
            label="Fix Faces", 
            info="Automatically enhance faces if detected"
        )
        remove_artifacts = gr.Checkbox(
            value=True, 
            label="Remove Artifacts", 
            info="Clean up artifacts from generation"
        )
        use_controlnet = gr.Checkbox(
            value=True, 
            label="Use ControlNet", 
            info="Use ControlNet for better guidance (disable for lower VRAM usage)"
        )
    
    # Add refiner controls
    with gr.Row():
        use_refiner = gr.Checkbox(
            value=AppConfig.USE_REFINER, 
            label="Use SDXL Refiner", 
            info="Apply SDXL refiner for better quality (requires more VRAM)"
        )
        refiner_strength = gr.Slider(
            minimum=0.1,
            maximum=0.8,
            value=AppConfig.REFINER_STRENGTH,
            step=0.05,
            label="Refiner Strength",
            info="How strongly to apply refinement (higher = more changes)",
            visible=AppConfig.USE_REFINER
        )
    
    # Make refiner_strength visible only when use_refiner is enabled
    use_refiner.change(
        fn=lambda x: {"visible": x},
        inputs=[use_refiner],
        outputs=[refiner_strength]
    )
    
    return [num_steps, guidance, enhance_details, fix_faces, remove_artifacts, 
            use_controlnet, use_refiner, refiner_strength]