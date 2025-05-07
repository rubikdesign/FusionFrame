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

# Asigurați-vă că aceste importuri sunt corecte în funcție de structura proiectului dvs.
# Este posibil ca sys.path.append să fie necesar dacă rulați acest script direct
# și directoarele config, core, processing, interface nu sunt în PYTHONPATH
try:
    from config.app_config import AppConfig
    from core.model_manager import ModelManager
    from core.pipeline_manager import PipelineManager
    from processing.analyzer import OperationAnalyzer
    from interface.components import create_examples, create_advanced_settings_panel # Presupunând că acest fișier există
    from interface.styles import CSS_STYLES # Presupunând că acest fișier există
except ImportError as e:
    # Adăugăm calea către directorul părinte dacă modulele nu sunt găsite
    # Acest lucru este util dacă rulați scriptul direct din subdirectorul 'interface'
    # sau dacă structura proiectului necesită ajustarea căilor.
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir) # Presupunând că 'interface' este un subdirector al rădăcinii proiectului
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    # Reîncercăm importurile după ajustarea căii
    from config.app_config import AppConfig
    from core.model_manager import ModelManager
    from core.pipeline_manager import PipelineManager
    from processing.analyzer import OperationAnalyzer
    from interface.components import create_examples, create_advanced_settings_panel
    from interface.styles import CSS_STYLES

# Setăm logger-ul
logger = logging.getLogger(__name__)
# Configurare de bază a logger-ului dacă nu este deja configurat de AppConfig la import
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


class FusionFrameUI:
    """
    Interfața utilizator pentru FusionFrame 2.0
    
    Implementează interfața grafică bazată pe Gradio pentru
    interacțiunea cu funcționalitățile aplicației.
    """
    
    def __init__(self):
        """Inițializează interfața utilizator"""
        logger.info("Initializing FusionFrameUI...")
        self.config = AppConfig
        self.model_manager = ModelManager()
        self.pipeline_manager = PipelineManager()
        self.analyzer = OperationAnalyzer()
        
        # Încărcăm modelele necesare
        self.load_models()
        
        # Creăm interfața
        self.app = self.create_interface()
        logger.info("FusionFrameUI initialized successfully.")
    
    def load_models(self):
        """Încarcă modelele necesare pentru funcționarea aplicației"""
        logger.info("Loading core models...")
        try:
            # Încărcăm doar modelele esențiale la pornire, restul se vor încărca la cerere
            # CORECȚIE: Apelăm load_main_model fără argumente.
            # HiDreamModel (instanțiat în interiorul load_main_model) 
            # va prelua AppConfig.USE_REFINER din __init__-ul său.
            self.model_manager.load_main_model() 
            logger.info("Core models loading process initiated.") 
            # Verificăm dacă modelul principal s-a încărcat efectiv
            if self.model_manager.get_model('main') and self.model_manager.get_model('main').is_loaded:
                logger.info("Main model confirmed loaded.")
            else:
                logger.warning("Main model might not have loaded successfully after load_main_model() call. Check previous logs.")

        except Exception as e:
            logger.error(f"Error during core models loading sequence: {str(e)}", exc_info=True)
            logger.warning("Some features may be limited due to model loading issues.")
    
    def create_interface(self) -> gr.Blocks:
        """
        Creează interfața Gradio
        
        Returns:
            Obiectul Gradio Blocks pentru interfață
        """
        logger.info("Creating Gradio interface...")
        # Creăm interfața Gradio
        with gr.Blocks(theme=gr.themes.Soft(), css=CSS_STYLES) as app:
            # Titlu și descriere
            gr.Markdown(
                f"# 🚀 FusionFrame {self.config.VERSION} - Advanced AI Image Editor"
            )
            
            # Panouri principale
            with gr.Row(equal_height=True):
                # Panou stânga (input)
                with gr.Column(scale=1): # Adăugat scale pentru layout
                    image_input = gr.Image(
                        type="pil", 
                        label="Upload Image",
                        elem_classes="image-preview",
                        # value=os.path.join(os.path.dirname(__file__), "assets", "placeholder.png") # Imagine placeholder opțională
                    )
                    
                    with gr.Row(elem_classes="controls"):
                        prompt = gr.Textbox(
                            label="Edit Instructions", 
                            placeholder="E.g., 'Remove the car', 'Change hair color to blonde'",
                            elem_id="prompt-input",
                            scale=3 # Mai mult spațiu pentru prompt
                        )
                        strength = gr.Slider(
                            minimum=0.1, maximum=1.0, value=0.75, step=0.05, # Adăugat step
                            label="Edit Strength",
                            info="Higher values create more dramatic changes",
                            scale=1 # Mai puțin spațiu relativ la prompt
                        )
                    
                    run_btn = gr.Button(
                        "Generate Edit", 
                        variant="primary", 
                        elem_id="generate-btn"
                    )
                    
                    status_area = gr.Textbox(
                        label="Status", 
                        value="Ready. Upload an image and provide instructions.", 
                        elem_classes="status-area", # Schimbat din progress-area pentru claritate
                        interactive=False,
                        lines=2 # Permite afișarea a mai mult text
                    )
                
                # Panou dreapta (output)
                with gr.Column(scale=1): # Adăugat scale
                    image_output = gr.Image(
                        label="Edited Result", 
                        elem_classes="image-preview"
                    )
                    mask_output = gr.Image(
                        label="Generated Mask (Debug)" # Clarificat scopul
                    )
                    
                    with gr.Accordion("Operation Details & Logs", open=False): # Titlu schimbat
                        info_json = gr.JSON(label="Operation Analysis") # Redenumit din info
            
            # Secțiune exemple
            # create_examples este o funcție importată, presupunem că gestionează crearea exemplelor
            # Dacă nu, o putem implementa aici.
            # Pentru moment, o las așa cum era în codul original.
            example_input_components = [prompt, strength] # Componentele pe care le actualizează exemplele
            if callable(create_examples):
                 create_examples(example_input_components)
            else:
                logger.warning("create_examples function not found or not callable. Skipping example creation.")
                # Implementare fallback simplă pentru exemple dacă create_examples nu e definită
                with gr.Accordion("Example Prompts", open=False):
                    example_prompts_list = [
                        ["remove the car from the street", 0.8],
                        ["change hair color to bright pink", 0.7],
                        ["replace background with futuristic city", 0.85],
                        ["add sunglasses to the person", 0.7],
                    ]
                    gr.Examples(
                        examples=example_prompts_list,
                        inputs=[prompt, strength], # Asigură-te că acestea sunt componentele corecte
                        label="Click an example to try"
                    )

            
            # Panou de setări avansate
            # create_advanced_settings_panel este o funcție importată
            advanced_settings_accordion = gr.Accordion("Advanced Settings", open=False)
            with advanced_settings_accordion:
                if callable(create_advanced_settings_panel):
                    advanced_settings_inputs = create_advanced_settings_panel()
                else:
                    logger.warning("create_advanced_settings_panel function not found or not callable. Advanced settings will be basic.")
                    # Implementare fallback pentru setări avansate
                    advanced_settings_inputs = [
                        gr.Slider(minimum=10, maximum=150, value=self.config.DEFAULT_STEPS, step=1, label="Inference Steps", info="More steps can improve quality but take longer."),
                        gr.Slider(minimum=1.0, maximum=20.0, value=self.config.DEFAULT_GUIDANCE_SCALE, step=0.5, label="Guidance Scale", info="Higher values follow prompt more strictly."),
                        # Adăugăm valori default pentru ceilalți parametri așteptați de process_image
                        gr.Checkbox(label="Enhance Details (Placeholder)", value=True, visible=False), # Ascuns dacă nu e implementat
                        gr.Checkbox(label="Fix Faces (Placeholder)", value=True, visible=False),
                        gr.Checkbox(label="Remove Artifacts (Placeholder)", value=True, visible=False),
                        gr.Checkbox(label="Use ControlNet (If Available)", value=True),
                        gr.Checkbox(label="Use Refiner (If Available)", value=self.config.USE_REFINER),
                        gr.Slider(minimum=0.0, maximum=1.0, value=self.config.REFINER_STRENGTH, step=0.05, label="Refiner Strength")
                    ]

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
            # Asigurăm că lista de input-uri corespunde cu parametrii funcției process_image
            # Parametrii pentru process_image sunt: image, prompt, strength, și apoi cei din advanced_settings
            # advanced_settings_inputs trebuie să fie o listă de componente Gradio
            
            # Verificăm dacă advanced_settings_inputs este o listă (cum ar trebui să fie)
            if not isinstance(advanced_settings_inputs, list):
                logger.error(f"advanced_settings_inputs is not a list, but {type(advanced_settings_inputs)}. This will cause an error in run_btn.click().")
                # Fallback la o listă goală pentru a evita eroare la pornire, dar funcționalitatea va fi afectată
                active_advanced_settings = []
            else:
                active_advanced_settings = advanced_settings_inputs

            run_btn.click(
                fn=self.process_image_gradio_wrapper, # Folosim un wrapper pentru a gestiona progresul
                inputs=[image_input, prompt, strength] + active_advanced_settings,
                outputs=[image_output, mask_output, info_json, status_area]
            )
        
        logger.info("Gradio interface created.")
        return app
    
    def process_image_gradio_wrapper(self, *args):
        """
        Wrapper pentru self.process_image pentru a gestiona actualizările de status în Gradio.
        *args va conține [image_input, prompt, strength] + lista de valori din advanced_settings.
        """
        # Extragem argumentele principale
        image = args[0]
        prompt_text = args[1]
        strength_value = args[2]
        
        # Extragem argumentele din setările avansate
        # Trebuie să corespundă cu ordinea din create_advanced_settings_panel()
        # sau cu implementarea fallback.
        # Valorile default sunt definite în semnătura funcției process_image.
        
        # Presupunem următoarea ordine pentru advanced_settings_inputs (din fallback-ul meu):
        # num_inference_steps, guidance_scale, enhance_details, fix_faces, 
        # remove_artifacts, use_controlnet, use_refiner, refiner_strength
        
        # Mapăm argumentele corect
        kwargs_for_processing = {}
        advanced_args = args[3:] # Restul argumentelor sunt setări avansate

        # Numele parametrilor așa cum sunt așteptați de process_image
        param_names = [
            "num_inference_steps", "guidance_scale", "enhance_details", 
            "fix_faces", "remove_artifacts", "use_controlnet", 
            "use_refiner", "refiner_strength"
        ]

        # Atribuim valorile din advanced_args la kwargs_for_processing
        for i, name in enumerate(param_names):
            if i < len(advanced_args):
                kwargs_for_processing[name] = advanced_args[i]
            # else: valorile default din process_image vor fi folosite

        # Actualizăm statusul inițial
        yield None, None, None, "Processing: Starting..."

        # Definim callback-ul pentru progres care va face yield
        # Acest yield este specific pentru funcțiile generator din Gradio
        # Pentru a actualiza status_area în timp real, process_image ar trebui să fie un generator
        # sau să folosim gr.Progress() în interiorul process_image_gradio_wrapper.
        # Pentru simplitate, vom actualiza statusul la început și la sfârșit aici.
        # O implementare mai avansată ar necesita ca self.pipeline_manager.process_image
        # să accepte un progress_callback care poate face `yield` sau să folosim `gr.Progress`.

        # --- Abordare simplificată pentru status ---
        # yield None, None, None, "Processing: Analyzing prompt..." # Exemplu de actualizare
        # --- Sfârșit abordare simplificată ---

        # Apelăm funcția de procesare principală
        try:
            # Definim un progress_callback simplu pentru logging intern
            # Nu putem face yield direct din acest callback în process_image
            # decât dacă process_image este un generator.
            
            # Pentru actualizări de status în Gradio, vom folosi gr.Progress()
            # sau vom face ca process_image să fie un generator.
            # Aici, pentru a menține structura, vom actualiza statusul doar la final.
            
            logger.info(f"Calling process_image with: image_present={image is not None}, prompt='{prompt_text}', strength={strength_value}, advanced_kwargs={kwargs_for_processing}")

            # Asigurăm că modelul principal este încărcat înainte de procesare
            if not (self.model_manager.get_model('main') and self.model_manager.get_model('main').is_loaded):
                logger.warning("Main model not loaded. Attempting to load it now before processing.")
                self.model_manager.load_main_model()
                if not (self.model_manager.get_model('main') and self.model_manager.get_model('main').is_loaded):
                    error_msg = "Critical: Main model failed to load. Cannot process image."
                    logger.error(error_msg)
                    yield image, None, {"error": error_msg}, error_msg
                    return # Ieșim din funcție


            # Apelăm funcția de procesare
            result_img, mask_img, op_info, status_msg = self.process_image(
                image, 
                prompt_text, 
                strength_value,
                **kwargs_for_processing # Pasează argumentele avansate
            )
            yield result_img, mask_img, op_info, status_msg
            
        except Exception as e:
            error_message = f"Critical error in UI wrapper: {str(e)}"
            logger.error(error_message, exc_info=True)
            # Returnăm imaginea originală în caz de eroare neașteptată
            yield image, None, {"error": error_message, "traceback": str(e)}, error_message


    def process_image(self, 
                      image: Optional[Image.Image], 
                      prompt: str, 
                      strength: float,
                      num_inference_steps: int = 50,
                      guidance_scale: float = 7.5,
                      enhance_details: bool = True, # Valori default
                      fix_faces: bool = True,
                      remove_artifacts: bool = True,
                      use_controlnet: bool = True, # Default la True, va fi verificat dacă e disponibil
                      use_refiner: Optional[bool] = None, # Va prelua din AppConfig dacă None
                      refiner_strength: Optional[float] = None # Va prelua din AppConfig dacă None
                      ) -> Tuple[Optional[Image.Image], Optional[Image.Image], Dict[str, Any], str]:
        """
        Procesează imaginea conform promptului utilizatorului.
        """
        if image is None:
            logger.warning("process_image called with no image.")
            return None, None, {"error": "No image provided for processing."}, "Error: No image provided."
        
        if not prompt or not prompt.strip():
            logger.warning("process_image called with an empty prompt.")
            return image, None, {"warning": "Empty prompt. No operation performed."}, "Warning: Prompt is empty. Original image returned."

        # Verificăm dacă modelul principal este încărcat
        main_model = self.model_manager.get_model('main')
        if not main_model or not main_model.is_loaded:
            error_msg = "Main processing model is not available. Cannot proceed."
            logger.error(error_msg)
            return image, None, {"error": error_msg}, error_msg

        logger.info(f"Processing image with prompt: '{prompt}', strength: {strength}, steps: {num_inference_steps}, guidance: {guidance_scale}")
        
        # Determină dacă se folosește refiner-ul pe baza inputului sau AppConfig
        actual_use_refiner = use_refiner if use_refiner is not None else self.config.USE_REFINER
        actual_refiner_strength = refiner_strength if refiner_strength is not None else self.config.REFINER_STRENGTH

        try:
            start_time = time.time()
            operation_details = self.analyzer.analyze_operation(prompt)
            
            # Funcție callback simplă pentru logare internă (nu pentru update Gradio UI direct)
            def log_progress_callback(progress_pct, desc_text=None):
                status_log = f"Internal Progress: {int(progress_pct * 100)}% - {desc_text or ''}"
                logger.debug(status_log) # Folosim debug pentru a nu umple log-urile INFO
            
            # Construim argumentele pentru pipeline_manager
            pipeline_kwargs = {
                "image": image,
                "prompt": prompt,
                "strength": strength,
                "operation_type": operation_details.get('type'),
                "target_object": operation_details.get('target'), # Numele parametrului așteptat de pipeline
                "progress_callback": log_progress_callback, # Pentru logare internă
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "use_controlnet_if_available": use_controlnet, # Pipeline-ul va verifica dacă e disponibil
                "use_refiner_if_available": actual_use_refiner,
                "refiner_strength": actual_refiner_strength,
                # Adăugăm și ceilalți parametri de "enhancement" dacă pipeline-ul îi suportă
                "enhance_details": enhance_details,
                "fix_faces": fix_faces,
                "remove_artifacts": remove_artifacts,
            }
            
            logger.debug(f"Calling pipeline_manager.process_image with kwargs: { {k:v for k,v in pipeline_kwargs.items() if k != 'image'} }")
            result_data = self.pipeline_manager.process_image(**pipeline_kwargs)
            
            processing_time = time.time() - start_time
            
            # Asigurăm că operation_details este actualizat cu timpul de procesare
            final_operation_info = result_data.get('operation', operation_details) # Preferăm ce returnează pipeline-ul
            final_operation_info['processing_time'] = f"{processing_time:.2f} seconds"
            final_operation_info['original_prompt_analysis'] = operation_details # Păstrăm și analiza inițială

            status_message = f"Completed in {processing_time:.2f}s: {result_data.get('message', 'Processing finished.')}"
            logger.info(status_message)
            
            return (
                result_data.get('result_image'), # Asigurăm că cheia e corectă
                result_data.get('mask_image'),   # Asigurăm că cheia e corectă
                final_operation_info,
                status_message
            )
            
        except Exception as e:
            error_message = f"Error during image processing: {str(e)}"
            logger.error(error_message, exc_info=True)
            return image, None, {"error": error_message, "traceback": str(e)}, error_message
    
    def launch(self, **kwargs):
        """
        Lansează interfața Gradio
        
        Args:
            **kwargs: Argumentele pentru lansarea Gradio (ex: server_name, server_port, share)
        """
        logger.info(f"Launching Gradio interface with options: {kwargs}")
        self.app.launch(**kwargs)

def main():
    """Funcția principală pentru rularea aplicației"""
    # Configurăm logging-ul la nivel DEBUG pentru a prinde mai multe detalii
    # Puteți schimba la logging.INFO pentru producție
    AppConfig.setup_logging(level=logging.DEBUG) 
    
    logger.info("Starting FusionFrame Application...")
    ui = FusionFrameUI()
    
    # Lansăm aplicația
    # Puteți schimba share=False dacă nu doriți un link public
    ui.launch(server_name="0.0.0.0", server_port=7860, share=False) 

if __name__ == "__main__":
    # Adăugăm o verificare pentru a ne asigura că scriptul este rulat ca modul principal
    # și nu doar importat, pentru a evita rularea `main()` la import.
    main()