#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Interfața Gradio pentru FusionFrame 2.0
"""

import os
import sys
import logging
import gradio as gr
import numpy as np
import time
from typing import Dict, Any, List, Optional, Union, Tuple
from PIL import Image

from config.app_config import AppConfig
from core.model_manager import ModelManager
from core.pipeline_manager import PipelineManager
from processing.analyzer import OperationAnalyzer
from interface.components import create_examples, create_advanced_settings_panel
from interface.styles import CSS_STYLES

# Setăm logger-ul
logger = logging.getLogger(__name__)

class FusionFrameUI:
    """
    Interfața utilizator pentru FusionFrame 2.0
    
    Implementează interfața grafică bazată pe Gradio pentru
    interacțiunea cu funcționalitățile aplicației.
    """
    
    def __init__(self):
        """Inițializează interfața utilizator"""
        self.config = AppConfig
        self.model_manager = ModelManager()
        self.pipeline_manager = PipelineManager()
        self.analyzer = OperationAnalyzer()
        
        # Încărcăm modelele necesare
        self.load_models()
        
        # Creăm interfața
        self.app = self.create_interface()
    
    def load_models(self):
        """Încarcă modelele necesare pentru funcționarea aplicației"""
        logger.info("Loading core models...")
        try:
            # Încărcăm doar modelele esențiale la pornire, restul se vor încărca la cerere
            self.model_manager.load_main_model()
            logger.info("Core models loaded successfully")
        except Exception as e:
            logger.error(f"Error loading core models: {str(e)}")
            logger.warning("Some features may be limited due to model loading issues")
    
    def create_interface(self) -> gr.Blocks:
        """
        Creează interfața Gradio
        
        Returns:
            Obiectul Gradio Blocks pentru interfață
        """
        # Creăm interfața Gradio
        with gr.Blocks(theme=gr.themes.Soft(), css=CSS_STYLES) as app:
            # Titlu și descriere
            gr.Markdown(
                f"# 🚀 FusionFrame {self.config.VERSION} - Advanced AI Image Editor"
            )
            
            # Panouri principale
            with gr.Row(equal_height=True):
                # Panou stânga (input)
                with gr.Column():
                    image_input = gr.Image(
                        type="pil", 
                        label="Upload Image",
                        elem_classes="image-preview"
                    )
                    
                    with gr.Row(elem_classes="controls"):
                        prompt = gr.Textbox(
                            label="Edit Instructions", 
                            placeholder="E.g., 'Remove the car', 'Change hair color to blonde'",
                            elem_id="prompt-input"
                        )
                        strength = gr.Slider(
                            0.1, 1.0, 0.75, 
                            label="Edit Strength",
                            info="Higher values create more dramatic changes"
                        )
                    
                    run_btn = gr.Button(
                        "Generate Edit", 
                        variant="primary", 
                        elem_id="generate-btn"
                    )
                    
                    status_area = gr.Textbox(
                        label="Status", 
                        value="Ready", 
                        elem_classes="progress-area",
                        interactive=False
                    )
                    
                # Panou dreapta (output)
                with gr.Column():
                    image_output = gr.Image(
                        label="Edited Result", 
                        elem_classes="image-preview"
                    )
                    mask_output = gr.Image(
                        label="Generated Mask"
                    )
                    
                    with gr.Accordion("Operation Details", open=False):
                        info = gr.JSON(label="Operation Analysis")
            
            # Secțiune exemple
            with gr.Row():
                gr.Markdown("## Example Prompts")
                
            # Creăm butoanele pentru exemple
            example_prompts = [
                # Operații de bază
                ["remove the car from the street", 0.8],
                ["change hair color to bright pink", 0.7],
                ["replace background with futuristic city", 0.85],
                ["remove watermark from top right corner", 0.75],
                # Exemple avansate adiționale
                ["remove person from the image", 0.9],
                ["change the shirt color to blue", 0.65],
                ["replace the sky with sunset colors", 0.8],
                ["erase all text from the image", 0.85],
                ["make the eyes green", 0.6],
                ["replace glasses with sunglasses", 0.75],
                ["add glasses", 0.65]
            ]
            
            # Divizăm exemplele în rânduri pentru o afișare mai organizată
            example_rows = [example_prompts[i:i+4] for i in range(0, len(example_prompts), 4)]
            
            for row_examples in example_rows:
                with gr.Row():
                    for ex_prompt, ex_strength in row_examples:
                        ex_btn = gr.Button(ex_prompt, elem_classes="example-btn")
                        # Definim o funcție separată pentru fiecare buton
                        ex_btn.click(
                            fn=lambda p=ex_prompt, s=ex_strength: [p, s],
                            inputs=None,
                            outputs=[prompt, strength]
                        )
            
            # Panou de setări avansate
            with gr.Accordion("Advanced Settings", open=False):
                advanced_settings = create_advanced_settings_panel()
            
            # Panou de informații
            with gr.Accordion("Tips & Info", open=False):
                gr.Markdown("""
                ### Tips for better results:
                - Be specific in your instructions (e.g., "remove the red car on the left" instead of just "remove car")
                - For replacing objects, specify what to replace them with
                - For color changes, specify the exact color (e.g., "bright pink", "deep blue")
                - Adjust strength slider for more or less dramatic changes
                - Check the generated mask to see what area will be edited
                
                ### Common operations:
                - **Remove**: "remove [object]"
                - **Replace**: "replace [object] with [new object]"
                - **Color Change**: "change color of [object] to [color]"
                - **Background Change**: "change background to [scene]"
                - **Add**: "add [object]" (e.g., "add glasses")
                """)
            
            # Funcționalitate pentru butonul de generare
            run_btn.click(
                fn=self.process_image,
                inputs=[image_input, prompt, strength, *advanced_settings],
                outputs=[image_output, mask_output, info, status_area]
            )
        
        return app
    
    def process_image(self, 
                     image, 
                     prompt, 
                     strength,
                     num_inference_steps=50,
                     guidance_scale=7.5,
                     enhance_details=True,
                     fix_faces=True,
                     remove_artifacts=True,
                     use_controlnet=True):
        """
        Procesează imaginea conform promptului utilizatorului
        
        Args:
            image: Imaginea de procesat
            prompt: Promptul pentru editare
            strength: Intensitatea editării
            num_inference_steps: Numărul de pași de inferență
            guidance_scale: Factorul de ghidare
            enhance_details: Dacă se aplică îmbunătățirea detaliilor
            fix_faces: Dacă se aplică corectarea fețelor
            remove_artifacts: Dacă se aplică eliminarea artefactelor
            use_controlnet: Dacă se folosește ControlNet
            
        Returns:
            Tuple cu imaginea rezultată, masca și informațiile de operație
        """
        if image is None:
            return None, None, {"error": "No image provided"}, "Processing cannot start without an image"
        
        try:
            # Cronometrăm procesarea
            start_time = time.time()
            
            # Analizăm promptul pentru a obține detalii despre operație
            operation = self.analyzer.analyze_operation(prompt)
            
            # Funcție pentru raportarea progresului
            def progress_callback(progress, desc=None):
                status = f"Processing: {int(progress * 100)}% - {desc or ''}"
                print(status)  # Pentru log-uri
                return status
            
            # Procesăm imaginea cu pipeline-ul potrivit
            result = self.pipeline_manager.process_image(
                image=image,
                prompt=prompt,
                strength=strength,
                operation_type=operation.get('type'),
                target=operation.get('target'),
                progress_callback=progress_callback,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                use_controlnet=use_controlnet
            )
            
            # Calculăm timpul de procesare
            processing_time = time.time() - start_time
            
            # Adăugăm informații despre timp la rezultat
            if isinstance(result, dict):
                if 'operation' in result:
                    result['operation']['processing_time'] = f"{processing_time:.2f} seconds"
                else:
                    result['operation'] = {'processing_time': f"{processing_time:.2f} seconds"}
            
            # Returnăm rezultatele
            return (
                result.get('result'),
                result.get('mask'),
                result.get('operation', {}),
                f"Completed in {processing_time:.2f}s: {result.get('message', 'Processing completed')}"
            )
            
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            return image, None, {"error": str(e)}, f"Error: {str(e)}"
    
    def launch(self, **kwargs):
        """
        Lansează interfața Gradio
        
        Args:
            **kwargs: Argumentele pentru lansarea Gradio
        """
        self.app.launch(**kwargs)


def main():
    """Funcția principală pentru rularea aplicației"""
    # Configurăm logging-ul
    AppConfig.setup_logging()
    
    # Creăm interfața
    ui = FusionFrameUI()
    
    # Lansăm aplicația
    ui.launch(share=True, server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    main()