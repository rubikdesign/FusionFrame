#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Interfața Gradio pentru FusionFrame 2.0 (Master Branch Corectat)
"""

import os
import sys
import logging
import gradio as gr
import numpy as np
import time
from typing import Dict, Any, List, Optional, Union, Tuple
from PIL import Image

# Asigurăm importurile corecte
try:
    from config.app_config import AppConfig
    from core.model_manager import ModelManager
    from core.pipeline_manager import PipelineManager
    from processing.analyzer import OperationAnalyzer
    # Importăm componentele actualizate (versiunea master - 6 componente)
    from interface.components import create_examples, create_advanced_settings_panel 
    from interface.styles import CSS_STYLES 
except ImportError as e:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir) 
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    print(f"Added {project_root} to sys.path to resolve imports.")
    
    try:
        from config.app_config import AppConfig
        from core.model_manager import ModelManager
        from core.pipeline_manager import PipelineManager
        from processing.analyzer import OperationAnalyzer
        from interface.components import create_examples, create_advanced_settings_panel
        from interface.styles import CSS_STYLES
    except ImportError as inner_e:
         print(f"ERROR: Failed to import modules even after path adjustment: {inner_e}")
         # Este recomandat să se oprească execuția dacă modulele core nu pot fi încărcate
         sys.exit(f"Critical import error: {inner_e}")

# Setăm logger-ul
logger = logging.getLogger(__name__)
# Configurare de bază a logger-ului dacă nu este deja configurat
if not logger.hasHandlers() or not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
    _console_handler = logging.StreamHandler()
    _formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    _console_handler.setFormatter(_formatter)
    logger.addHandler(_console_handler)
    if logger.level == logging.NOTSET: 
        logger.setLevel(logging.INFO) # Setăm INFO ca default, poate fi suprascris de setup_logging

class FusionFrameUI:
    """Interfața utilizator pentru FusionFrame 2.0"""
    
    def __init__(self):
        logger.info("Initializing FusionFrameUI...")
        self.config = AppConfig
        self.model_manager = ModelManager()
        self.pipeline_manager = PipelineManager()
        self.analyzer = OperationAnalyzer()
        self.load_models() # Încărcăm modelele la inițializare
        self.app = self.create_interface()
        logger.info("FusionFrameUI initialized successfully.")
    
    def load_models(self):
        """Încarcă modelele core necesare."""
        logger.info("Loading core models...")
        try:
            # Apel corect, fără argumente (conform ModelManager refactorizat)
            self.model_manager.load_main_model() 
            logger.info("Core models loading process initiated.") 
            # Verificăm starea după apel (get_model va aștepta/verifica încărcarea dacă e leneșă)
            main_model = self.model_manager.get_model('main') 
            if main_model and getattr(main_model, 'is_loaded', False):
                logger.info("Main model confirmed loaded.")
            else:
                # Mesajul de avertizare este suficient, eroarea va apărea la procesare dacă e cazul
                logger.warning("Main model might not have loaded successfully after load_main_model() call. Check previous logs.")
        except Exception as e:
            # Logăm eroarea completă pentru depanare
            logger.error(f"Error during core models loading sequence: {str(e)}", exc_info=True)
            logger.warning("Some features may be limited due to model loading issues.")
    
    def create_interface(self) -> gr.Blocks:
        """Creează interfața Gradio."""
        logger.info("Creating Gradio interface...")
        with gr.Blocks(theme=gr.themes.Soft(), css=CSS_STYLES) as app:
            gr.Markdown(f"# 🚀 FusionFrame {self.config.VERSION} - Advanced AI Image Editor")
            
            with gr.Row(equal_height=False): 
                with gr.Column(scale=2): 
                    with gr.Tabs():
                         with gr.TabItem("Image Input"):
                              image_input = gr.Image(type="pil", label="Upload Image", elem_classes="image-preview")
                    with gr.Row(elem_classes="controls"):
                        prompt = gr.Textbox(label="Edit Instructions", placeholder="E.g., 'Remove the car', 'Change hair color to blonde'", elem_id="prompt-input", scale=3, lines=2)
                        strength = gr.Slider(minimum=0.0, maximum=1.0, value=0.75, step=0.05, label="Edit Strength", info="For Inpainting/Img2Img (0=original, 1=full change)", scale=1)
                    run_btn = gr.Button("✨ Generate Edit", variant="primary", elem_id="generate-btn")
                    status_area = gr.Textbox(label="Status", value="Ready. Upload an image and provide instructions.", elem_classes="status-area", interactive=False, lines=2)
                
                with gr.Column(scale=1): 
                    with gr.Tabs():
                        with gr.TabItem("Edited Result"):
                             image_output = gr.Image(label="Result", elem_classes="image-preview", interactive=False)
                        with gr.TabItem("Generated Mask"):
                             mask_output = gr.Image(label="Mask Used (Debug)", interactive=False)
                        with gr.TabItem("Operation Info"):
                             info_json = gr.JSON(label="Operation Analysis & Timings")
            
            # --- Secțiune Exemple (Corectată pentru NameError) ---
            with gr.Accordion("Example Prompts", open=False):
                # Presupunem că create_examples() returnează lista corectă
                try:
                    example_prompts_list = create_examples() 
                    if not isinstance(example_prompts_list, list):
                        raise TypeError("create_examples did not return a list")
                except Exception as e_ex:
                    logger.error(f"Failed to get examples from components.py: {e_ex}. Using fallback examples.", exc_info=True)
                    example_prompts_list = [ ["remove the car from the street", 0.8], ["change hair color to bright pink", 0.7] ]

                # Funcție helper pentru a crea closure corect
                def create_click_fn(p_val, s_val):
                    def click_action(): return [p_val, s_val]
                    return click_action

                example_rows = [example_prompts_list[i:i+4] for i in range(0, len(example_prompts_list), 4)]
                for row_examples in example_rows:
                    with gr.Row():
                        for ex_prompt_text, ex_strength_val in row_examples:
                            # Validăm tipurile înainte de a crea butonul
                            if isinstance(ex_prompt_text, str) and isinstance(ex_strength_val, (int, float)):
                                ex_btn = gr.Button(ex_prompt_text, elem_classes="example-btn", size="sm")
                                click_func = create_click_fn(ex_prompt_text, ex_strength_val)
                                ex_btn.click(fn=click_func, inputs=None, outputs=[prompt, strength])
                            else:
                                logger.warning(f"Skipping invalid example format: [{ex_prompt_text}, {ex_strength_val}]")
            
            # --- Panou Setări Avansate (Folosind components.py actualizat pentru Master - 6 setări) ---
            advanced_settings_accordion = gr.Accordion("Advanced Settings", open=False)
            with advanced_settings_accordion:
                try:
                    # create_advanced_settings_panel trebuie să returneze 6 componente pentru master
                    advanced_settings_inputs = create_advanced_settings_panel() 
                    if not isinstance(advanced_settings_inputs, list) or len(advanced_settings_inputs) != 6: 
                         logger.error(f"create_advanced_settings_panel returned {len(advanced_settings_inputs)} items, expected 6 for master branch. Check components.py. Using fallback.")
                         raise TypeError("Incorrect number/type of components returned")
                except Exception as e_comp:
                    logger.error(f"Failed to create advanced settings panel: {e_comp}. Using fallback.", exc_info=True)
                    # Fallback explicit cu 6 componente
                    advanced_settings_inputs = [
                        gr.Slider(minimum=10, maximum=150, value=AppConfig.DEFAULT_STEPS, step=1, label="Fallback: Inference Steps"),
                        gr.Slider(minimum=1.0, maximum=20.0, value=AppConfig.DEFAULT_GUIDANCE_SCALE, step=0.5, label="Fallback: Guidance Scale"),
                        gr.Checkbox(label="Fallback: Enhance Details", value=True, visible=False), 
                        gr.Checkbox(label="Fallback: Fix Faces", value=True, visible=False),
                        gr.Checkbox(label="Fallback: Remove Artifacts", value=True, visible=False),
                        gr.Checkbox(label="Fallback: Use ControlNet", value=True)
                    ]

            # Panou de informații
            with gr.Accordion("Tips & Info", open=False):
                gr.Markdown("""
                ### Tips for better results:
                - Be specific: "remove the *red* car on the *left*"
                - For replacements, specify the new object/scene.
                - For colors, be descriptive (e.g., "bright pink", "deep blue").
                - Adjust strength slider for intensity.
                - Check the generated mask (if visible) to understand the edit area.
                
                ### Common operations:
                - **Remove**: "remove [object]"
                - **Replace**: "replace [object] with [new object/scene]"
                - **Color Change**: "change color of [object] to [color]"
                - **Add**: "add [object] to [context]" (e.g., "add sunglasses to the woman")
                """)
            
            # --- Buton Generare Click Handler ---
            # Asigurăm că lista de input-uri este corectă
            active_advanced_settings = advanced_settings_inputs if isinstance(advanced_settings_inputs, list) and len(advanced_settings_inputs) == 6 else []
            if len(active_advanced_settings) != 6:
                 logger.error("Advanced settings list is not correctly formed. Generate button might malfunction.")
                 # Poate dezactivăm butonul sau afișăm o eroare permanentă
                 run_btn.interactive = False # Dezactivăm butonul dacă setările sunt greșite

            run_btn.click(
                fn=self.process_image_gradio_wrapper, 
                inputs=[image_input, prompt, strength] + active_advanced_settings, # Pasează cele 6 setări
                outputs=[image_output, mask_output, info_json, status_area]
            )
        
        logger.info("Gradio interface created.")
        return app
    
    def process_image_gradio_wrapper(self, *args):
        """Wrapper Gradio pentru self.process_image, gestionează statusul."""
        # Extragem argumentele: imagine, prompt, strength + lista de 6 setări avansate
        image, prompt_text, strength_value, *advanced_args = args
        
        # Mapăm argumentele avansate la kwargs pentru process_image
        kwargs_for_processing = {}
        # Ordinea trebuie să corespundă cu cea din create_advanced_settings_panel (master version - 6 params)
        param_names = [ 
            "num_inference_steps", "guidance_scale", "enhance_details", 
            "fix_faces", "remove_artifacts", "use_controlnet" 
        ]
        
        if len(advanced_args) != len(param_names):
             error_msg = f"Argument mismatch in advanced settings. Expected {len(param_names)}, got {len(advanced_args)}."
             logger.error(error_msg)
             # Folosim yield pentru a returna eroarea în UI
             yield image, None, {"error": error_msg}, f"Error: {error_msg}"
             return # Oprim execuția

        for i, name in enumerate(param_names):
            kwargs_for_processing[name] = advanced_args[i]

        # Actualizăm statusul inițial și curățăm output-urile vechi
        yield None, None, None, "Processing: Starting..." 

        try:
            logger.info(f"Calling process_image with: prompt='{prompt_text}', strength={strength_value}, advanced_kwargs={kwargs_for_processing}")

            main_model = self.model_manager.get_model('main')
            # Verificare mai robustă a stării modelului
            if not (main_model and getattr(main_model, 'is_loaded', False)):
                error_msg = "Critical: Main model failed to load or is not available."
                logger.error(error_msg)
                # Returnăm imaginea originală și eroarea
                yield image, None, {"error": error_msg}, f"Error: {error_msg}" 
                return # Oprim execuția

            # Apelăm funcția de procesare principală (versiunea master)
            result_img, mask_img, op_info, status_msg = self.process_image(
                image, prompt_text, strength_value, **kwargs_for_processing # Pasează doar cele 6 kwargs
            )
            yield result_img, mask_img, op_info, status_msg
            
        except Exception as e:
            error_message = f"Critical error in UI processing wrapper: {str(e)}"
            logger.error(error_message, exc_info=True)
            # Returnăm imaginea originală și eroarea detaliată
            yield image, None, {"error": error_message, "traceback": str(e)}, f"Error: {error_message}"

    # Semnătura process_image pentru MASTER branch (fără refiner)
    def process_image(self, 
                      image: Optional[Image.Image], prompt: str, strength: float,
                      # Doar cei 6 parametri din components.py (master version)
                      num_inference_steps: int = 50, guidance_scale: float = 7.5,
                      enhance_details: bool = True, fix_faces: bool = True,
                      remove_artifacts: bool = True, use_controlnet: bool = True
                      ) -> Tuple[Optional[Image.Image], Optional[Image.Image], Dict[str, Any], str]:
        """Procesează imaginea (versiunea pentru Master Branch)."""
        
        if image is None: return None, None, {"error": "No image provided."}, "Error: No image provided."
        if not prompt or not prompt.strip(): return image, None, {"warning": "Empty prompt."}, "Warning: Prompt is empty."

        main_model = self.model_manager.get_model('main')
        # Verificăm din nou, deși wrapper-ul a făcut-o deja (double check)
        if not main_model or not getattr(main_model, 'is_loaded', False):
            error_msg = "Main processing model unavailable at processing time."
            logger.error(error_msg); return image, None, {"error": error_msg}, error_msg

        logger.info(f"Processing: '{prompt}', str: {strength}, steps: {num_inference_steps}, cfg: {guidance_scale}, ctrlnet: {use_controlnet}")
        
        try:
            start_time = time.time()
            # Analiza operației e utilă pentru PipelineManager și logging
            operation_details = self.analyzer.analyze_operation(prompt) 
            
            def log_progress_callback(progress_pct, desc_text=None):
                # Acest callback e doar pentru logare internă, nu actualizează UI-ul direct
                logger.debug(f"Internal Progress: {int(progress_pct * 100)}% - {desc_text or ''}")
            
            # Construim kwargs pentru pipeline_manager, FĂRĂ refiner
            pipeline_kwargs = {
                "image": image, "prompt": prompt, "strength": strength,
                "operation_details": operation_details, # Pasăm detaliile analizate
                "progress_callback": log_progress_callback, 
                "num_inference_steps": num_inference_steps, 
                "guidance_scale": guidance_scale,
                # Pasăm use_controlnet, pipeline-ul va decide dacă îl folosește
                "use_controlnet_if_available": use_controlnet, 
                # Parametrii de enhancement pot fi pasați, pipeline-ul îi va folosi dacă știe
                "enhance_details": enhance_details, 
                "fix_faces": fix_faces, 
                "remove_artifacts": remove_artifacts,
                # Adăugăm seed pentru reproductibilitate (poate fi făcut configurabil)
                "seed": int(time.time() % (2**32)) # Seed bazat pe timp
            }
            
            logger.debug(f"Calling pipeline_manager.process_image with kwargs: { {k:v for k,v in pipeline_kwargs.items() if k not in ['image','operation_details']} }")
            
            # PipelineManager alege și rulează pipeline-ul specific
            result_data = self.pipeline_manager.process_image(**pipeline_kwargs)
            
            processing_time = time.time() - start_time
            
            # Construim informațiile finale
            final_operation_info = result_data.get('operation', operation_details) 
            final_operation_info['processing_time'] = f"{processing_time:.2f} seconds"
            # Logăm parametrii efectiv folosiți (fără refiner)
            final_operation_info['params_used'] = { 
                 "strength": strength, "steps": num_inference_steps, "guidance": guidance_scale,
                 "use_ctrl": use_controlnet 
            }

            status_message = f"Completed in {processing_time:.2f}s: {result_data.get('message', 'Processing finished.')}"
            logger.info(status_message)
            
            # Returnăm rezultatele așteptate de wrapper
            return (
                result_data.get('result_image', result_data.get('result')), 
                result_data.get('mask_image', result_data.get('mask')),     
                final_operation_info,
                status_message
            )
        except Exception as e:
            error_message = f"Error during image processing logic: {str(e)}"
            logger.error(error_message, exc_info=True)
            # Returnăm imaginea originală și eroarea
            return image, None, {"error": error_message, "traceback": str(e)}, f"Error: {error_message}"
    
    def launch(self, **kwargs):
        """Lansează interfața Gradio."""
        # Adăugăm debug=False explicit dacă nu e specificat
        launch_kwargs = {'server_name': "0.0.0.0", 'server_port': 7860, 'share': False, 'debug': False}
        launch_kwargs.update(kwargs) # Permitem suprascrierea din exterior
        logger.info(f"Launching Gradio interface with options: {launch_kwargs}")
        self.app.launch(**launch_kwargs)

def main():
    """Funcția principală pentru rularea aplicației"""
    # Setăm nivelul de logging (DEBUG pentru dezvoltare, INFO pentru producție)
    log_level = logging.DEBUG if os.environ.get("FUSION_DEBUG") else logging.INFO
    AppConfig.setup_logging(level=log_level) 
    
    logger.info("Starting FusionFrame Application (Master Branch Version)...")
    try:
        ui = FusionFrameUI()
        # Setăm share=False ca default
        ui.launch(share=os.environ.get("GRADIO_SHARE", "false").lower() == "true") 
    except Exception as e:
         logger.critical(f"Failed to initialize or launch the UI: {e}", exc_info=True)

if __name__ == "__main__":
    main()
