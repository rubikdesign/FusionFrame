#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Componente pentru interfața Gradio în FusionFrame 2.0 (Master Branch)
"""

import gradio as gr
from typing import List, Tuple, Any

# Importăm AppConfig pentru a prelua valorile default
try:
    from config.app_config import AppConfig
except ImportError:
    print("WARNING: components.py - Could not import AppConfig. Using default values.")
    class AppConfig: # Define simple defaults if import fails
        DEFAULT_STEPS = 50
        DEFAULT_GUIDANCE_SCALE = 7.5
        # No refiner defaults needed for master branch components

def create_examples() -> List[List[Any]]:
    """
    Creează lista de exemple pentru interfață.
    (Păstrată din versiunea originală)
    """
    examples = [
        ["remove the car from the street", 0.8],
        ["change hair color to bright pink", 0.7],
        ["replace background with futuristic city", 0.85],
        ["remove watermark from top right corner", 0.75],
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
    Creează panoul de setări avansate relevant pentru branch-ul master.
    
    Returns:
        Lista de 6 componente Gradio:
        [num_steps, guidance, enhance_details, fix_faces, 
         remove_artifacts, use_controlnet]
    """
    with gr.Column(): 
        with gr.Row():
            num_steps = gr.Slider(
                minimum=10, 
                maximum=150, 
                value=AppConfig.DEFAULT_STEPS, 
                step=1, 
                label="Inference Steps",
                info="More steps can improve quality but take longer."
            )
            guidance = gr.Slider(
                minimum=1.0, 
                maximum=20.0, 
                value=AppConfig.DEFAULT_GUIDANCE_SCALE, 
                step=0.5, 
                label="Guidance Scale (CFG)",
                info="Higher values follow prompt more strictly."
            )
        
        with gr.Row():
            # Checkbox-uri pentru post-procesare (pot fi lăsate, dar funcționalitatea depinde de implementare)
            enhance_details = gr.Checkbox(
                value=True, label="Enhance Details", 
                info="Attempt post-processing detail enhancement.",
                visible=False # Ascuns dacă nu e implementat pe master
            )
            fix_faces = gr.Checkbox(
                value=True, label="Fix Faces", 
                info="Attempt post-processing face correction.",
                visible=False # Ascuns dacă nu e implementat pe master
            )
            remove_artifacts = gr.Checkbox(
                value=True, label="Remove Artifacts", 
                info="Attempt post-processing artifact removal.",
                visible=False # Ascuns dacă nu e implementat pe master
            )
        
        with gr.Row():
            use_controlnet = gr.Checkbox(
                value=True, 
                label="Use ControlNet", 
                info="Enable ControlNet guidance if available and applicable."
            )
            # NU adăugăm controalele pentru Refiner aici pentru branch-ul master
    
    # Returnăm doar cele 6 componente relevante
    return [
        num_steps, guidance, enhance_details, fix_faces, 
        remove_artifacts, use_controlnet
    ]

