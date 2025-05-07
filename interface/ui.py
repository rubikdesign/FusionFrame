#!/usr/bin/env python
# -*- coding: utf-8 -*-\

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
import cv2 # Adăugat pentru normalizarea depth map

# Asigurăm căile corecte pentru importuri
try:
    # Încercăm importurile directe (structură standard)
    from config.app_config import AppConfig
    from core.model_manager import ModelManager
    from core.pipeline_manager import PipelineManager
    from processing.analyzer import OperationAnalyzer, ImageAnalyzer
    # Presupunem că acestea există în directorul 'interface' sau sunt în PYTHONPATH
    # Verificăm existența fișierelor opționale
    components_path = os.path.join(os.path.dirname(__file__), 'components.py')
    styles_path = os.path.join(os.path.dirname(__file__), 'styles.py')

    if os.path.exists(components_path):
         from interface.components import create_examples, create_advanced_settings_panel
    else:
         create_examples, create_advanced_settings_panel = None, None
         logging.warning(f"Optional file not found: {components_path}. Using fallbacks.")

    if os.path.exists(styles_path):
         from interface.styles import CSS_STYLES
    else:
         CSS_STYLES = ""
         logging.warning(f"Optional file not found: {styles_path}. Using fallbacks.")

except ImportError:
    # Fallback: Adăugăm directorul rădăcină al proiectului în PYTHONPATH
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
        print(f"Added project root to sys.path: {project_root}") # Debug print

    # Reîncercăm importurile
    try:
        from config.app_config import AppConfig
        from core.model_manager import ModelManager
        from core.pipeline_manager import PipelineManager
        from processing.analyzer import OperationAnalyzer, ImageAnalyzer

        components_path = os.path.join(os.path.dirname(__file__), 'components.py')
        styles_path = os.path.join(os.path.dirname(__file__), 'styles.py')

        if os.path.exists(components_path):
             from interface.components import create_examples, create_advanced_settings_panel
        else:
             create_examples, create_advanced_settings_panel = None, None
             logging.warning(f"Optional file not found: {components_path}. Using fallbacks.")

        if os.path.exists(styles_path):
             from interface.styles import CSS_STYLES
        else:
             CSS_STYLES = ""
             logging.warning(f"Optional file not found: {styles_path}. Using fallbacks.")

    except ImportError as e_retry:
         print(f"FATAL: Could not import necessary modules even after adjusting sys.path. Error: {e_retry}")
         print(f"Current sys.path: {sys.path}")
         print("Please ensure the project structure is correct and all modules are accessible.")
         sys.exit(1)


# Setăm logger-ul principal
logger = logging.getLogger(__name__)
if not logger.hasHandlers():
     logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


class FusionFrameUI:
    """Interfața utilizator pentru FusionFrame 2.0"""
    def __init__(self):
        logger.info("Initializing FusionFrameUI...")
        self.config = AppConfig
        self.model_manager = ModelManager()
        self.pipeline_manager = PipelineManager()
        self.op_analyzer = OperationAnalyzer()
        self.img_analyzer = ImageAnalyzer()

        self.load_models()
        self.app = self.create_interface()
        logger.info("FusionFrameUI initialized successfully.")

    def load_models(self):
        """Încarcă modelele esențiale la pornire (sau verifică dacă sunt încărcate)."""
        logger.info("Loading/Checking essential models...")
        try:
            # Verificăm/încărcăm modelul principal
            main_model = self.model_manager.get_model('main')
            if main_model and getattr(main_model, 'is_loaded', False):
                 logger.info("Main model confirmed loaded.")
            else:
                 logger.warning("Main model check/load initiated. Check logs for status.")
            # Putem adăuga verificări similare și pentru alte modele esențiale dacă dorim
            # self.model_manager.get_model('sam')
            # self.model_manager.get_model('clipseg')
        except Exception as e:
            logger.error(f"Error during essential models loading sequence: {str(e)}", exc_info=True)
            logger.warning("Some features might be limited due to model loading issues.")

    def create_interface(self) -> gr.Blocks:
        """Creează interfața Gradio."""
        logger.info("Creating Gradio interface...")
        # Folosim valorile CSS importate sau default ""
        global CSS_STYLES
        css_to_use = CSS_STYLES if 'CSS_STYLES' in globals() else ""

        with gr.Blocks(theme=gr.themes.Soft(), css=css_to_use) as app:
            gr.Markdown(
                f"# 🚀 FusionFrame {self.config.VERSION} - Advanced AI Image Editor"
            )

            with gr.Row(equal_height=False):
                with gr.Column(scale=1):
                    image_input = gr.Image(type="pil", label="Upload Image", elem_id="image_input", height=400)
                    with gr.Row():
                        prompt = gr.Textbox(label="Edit Instructions", placeholder="E.g., 'Remove the car'", elem_id="prompt-input", scale=3)
                        strength = gr.Slider(minimum=0.1, maximum=1.0, value=0.75, step=0.05, label="Edit Strength", info="Higher = more dramatic", scale=1)
                    run_btn = gr.Button("Generate Edit", variant="primary", elem_id="generate-btn")
                    status_area = gr.Textbox(label="Status", value="Ready.", elem_id="status-area", interactive=False, lines=2)

                with gr.Column(scale=1):
                    image_output = gr.Image(label="Edited Result", elem_id="image_output", height=400)
                    with gr.Row():
                        mask_output = gr.Image(label="Generated Mask (Debug)", elem_id="mask_output", height=200)
                        depth_output = gr.Image(label="Depth Map (Debug)", elem_id="depth_output", height=200) # Păstrăm vizualizarea hărții de adâncime
                    with gr.Accordion("Operation Details & Analysis", open=False): # Redenumit
                        info_json = gr.JSON(label="Operation & Context Info") # Redenumit

            # Secțiune exemple
            global create_examples
            if create_examples:
                # Presupunem că create_examples returnează lista direct
                example_list = create_examples() if callable(create_examples) else []
                if example_list:
                     gr.Examples(examples=example_list, inputs=[prompt, strength], label="Example Prompts")
            else:
                logger.warning("create_examples function not available. Skipping example UI.")

            # Panou de setări avansate
            with gr.Accordion("Advanced Settings", open=False):
                global create_advanced_settings_panel
                if create_advanced_settings_panel and callable(create_advanced_settings_panel):
                    advanced_settings_inputs = create_advanced_settings_panel()
                    if not isinstance(advanced_settings_inputs, list):
                         logger.error("create_advanced_settings_panel did not return a list. Fallback.")
                         advanced_settings_inputs = self._create_fallback_advanced_settings()
                else:
                    logger.warning("create_advanced_settings_panel function not available. Using fallback.")
                    advanced_settings_inputs = self._create_fallback_advanced_settings()

                # Asigurăm logica de vizibilitate pentru refiner strength, chiar și în fallback
                use_refiner_component = None
                refiner_strength_component = None
                # Găsim componentele relevante (presupunând o ordine sau folosind label-ul)
                for component in advanced_settings_inputs:
                    if isinstance(component, (gr.Checkbox)) and "Refiner" in getattr(component, 'label', ''):
                        use_refiner_component = component
                    if isinstance(component, (gr.Slider)) and "Refiner Strength" in getattr(component, 'label', ''):
                        refiner_strength_component = component

                if use_refiner_component and refiner_strength_component:
                    use_refiner_component.change(
                         fn=lambda x: gr.update(visible=x),
                         inputs=[use_refiner_component],
                         outputs=[refiner_strength_component]
                    )
                else:
                     logger.warning("Could not find Use Refiner or Refiner Strength components to link visibility.")


            # Panou de informații
            with gr.Accordion("Tips & Info", open=False):
                 gr.Markdown("""
                 ### Tips for better results:
                 - Be specific: "remove the *red* car on the *left*" vs "remove car".
                 - For replacements, specify what to replace with.
                 - Adjust strength slider for more/less dramatic changes.
                 - Check the generated mask/depth map (debug view).

                 ### Common operations:
                 - **Remove**: "remove [object]"
                 - **Replace**: "replace [object] with [new object]"
                 - **Color**: "change color of [object] to [color]"
                 - **Background**: "change background to [scene]"
                 - **Add**: "add [object]"
                 """)

            # Funcționalitate buton
            active_advanced_settings = advanced_settings_inputs if isinstance(advanced_settings_inputs, list) else []
            run_btn.click(
                fn=self.process_image_gradio_wrapper,
                inputs=[image_input, prompt, strength] + active_advanced_settings,
                outputs=[image_output, mask_output, depth_output, info_json, status_area]
            )

        logger.info("Gradio interface created.")
        return app

    def _create_fallback_advanced_settings(self) -> List[gr.components.Component]:
         """Creează o listă de componente default pentru setări avansate."""
         # Preluăm valorile default din AppConfig sau setăm unele generice
         default_steps = getattr(self.config, 'DEFAULT_STEPS', 50)
         default_guidance = getattr(self.config, 'DEFAULT_GUIDANCE_SCALE', 7.5)
         default_use_refiner = getattr(self.config, 'USE_REFINER', True)
         default_refiner_strength = getattr(self.config, 'REFINER_STRENGTH', 0.3)

         return [
             gr.Slider(minimum=10, maximum=150, value=default_steps, step=1, label="Inference Steps"),
             gr.Slider(minimum=1.0, maximum=20.0, value=default_guidance, step=0.5, label="Guidance Scale"),
             gr.Checkbox(label="Enhance Details", value=True), # Lăsăm checkbourile, chiar dacă post-proc nu e fully integrat
             gr.Checkbox(label="Fix Faces", value=True),
             gr.Checkbox(label="Remove Artifacts", value=True),
             gr.Checkbox(label="Use ControlNet", value=True),
             gr.Checkbox(label="Use Refiner", value=default_use_refiner),
             gr.Slider(minimum=0.0, maximum=1.0, value=default_refiner_strength, step=0.05, label="Refiner Strength", visible=default_use_refiner)
         ]

    def process_image_gradio_wrapper(self, *args):
        """Wrapper pentru Gradio pentru a gestiona actualizări de status și output-uri multiple."""
        start_time_wrapper = time.time()
        image_pil, prompt_text, strength_value, *advanced_args_tuple = args
        advanced_args = list(advanced_args_tuple) # Convertim tuplul în listă

        # Mapare argumente avansate
        param_names = [
            "num_inference_steps", "guidance_scale", "enhance_details",
            "fix_faces", "remove_artifacts", "use_controlnet",
            "use_refiner", "refiner_strength"
        ]
        kwargs_for_processing = {}
        # Asigurăm că avem suficiente valori default dacă advanced_args e mai scurtă
        num_expected_advanced = len(param_names)
        advanced_args.extend([None] * (num_expected_advanced - len(advanced_args)))

        for i, name in enumerate(param_names):
             # Prioritizăm valoarea din UI dacă nu e None, altfel None (process_image va folosi default)
            kwargs_for_processing[name] = advanced_args[i]

        logger.info(f"Processing request: Prompt='{prompt_text}', Strength={strength_value}, AdvancedKwargs={kwargs_for_processing}")

        # Status inițial
        yield None, None, None, {}, "Status: Starting..."

        # Validări input
        if image_pil is None:
            yield None, None, None, {"error": "No image provided"}, "Error: Please upload an image."
            return
        if not prompt_text or not prompt_text.strip():
            yield image_pil, None, None, {"warning": "Empty prompt"}, "Warning: Prompt is empty."
            return

        # Asigurare model principal
        main_model = self.model_manager.get_model('main')
        if not main_model or not getattr(main_model, 'is_loaded', False): # Verificare mai robustă
            error_msg = "Critical: Main processing model failed to load or is not available."
            logger.error(error_msg)
            yield image_pil, None, None, {"error": error_msg}, error_msg
            return

        # Procesare efectivă
        try:
            result_dict = self.process_image(image_pil, prompt_text, strength_value, **kwargs_for_processing)

            result_img = result_dict.get('result_image')
            mask_img = result_dict.get('mask_image')
            op_info = result_dict.get('operation_info', {})
            context_info = result_dict.get('context_info', {})
            status_msg = result_dict.get('status_message', "Finished.")
            success = result_dict.get('success', False)

            # Combinăm info pentru JSON
            final_info_json = {
                 "Operation Analysis": op_info,
                 "Image Context Analysis": context_info, # Includem tot contextul
                 "Processing Status": "Success" if success else "Failed"
            }
            if not success and "error" not in final_info_json["Operation Analysis"]:
                 final_info_json["Operation Analysis"]["error"] = status_msg # Adăugăm mesajul de eroare dacă lipsește

            # Pregătim depth map pentru afișare
            depth_map_display = None
            if context_info and context_info.get('spatial_info', {}).get('depth_map_available'):
                 depth_map_np = context_info['spatial_info']['depth_map']
                 if depth_map_np is not None:
                      # Normalizăm explicit la 0-255 uint8 pentru afișare
                      depth_map_normalized = cv2.normalize(depth_map_np, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                      # Aplicăm o colormapă pentru vizualizare mai bună (opțional)
                      # depth_map_color = cv2.applyColorMap(depth_map_normalized, cv2.COLORMAP_VIRIDIS)
                      # depth_map_display = Image.fromarray(cv2.cvtColor(depth_map_color, cv2.COLOR_BGR2RGB))
                      # Sau afișăm grayscale direct
                      depth_map_display = Image.fromarray(depth_map_normalized)


            processing_time = time.time() - start_time_wrapper
            status_msg += f" (Total UI time: {processing_time:.2f}s)"
            logger.info(f"Request finished. Status: {'Success' if success else 'Failed'}. {status_msg}")

            # Actualizăm UI
            yield result_img, mask_img, depth_map_display, final_info_json, status_msg

        except Exception as e:
            error_message = f"Critical error in UI processing wrapper: {str(e)}"
            logger.error(error_message, exc_info=True)
            import traceback
            # Încercăm să returnăm imaginea originală și eroarea
            yield image_pil, None, None, {"error": error_message, "traceback": traceback.format_exc()}, error_message


    def process_image(self,
                      image: Image.Image,
                      prompt: str,
                      strength: float,
                      **kwargs) -> Dict[str, Any]:
        """Procesează imaginea și returnează un dicționar cu toate rezultatele."""
        start_time_process = time.time()
        logger.info(f"Starting process_image: Prompt='{prompt}', Strength={strength}, Kwargs={kwargs}")

        results = {
            'result_image': image, 'mask_image': None, 'operation_info': {},
            'context_info': {}, 'status_message': "Processing started.", 'success': False
        }

        try:
            # --- 1. Analiza Operației ---
            operation_details = self.op_analyzer.analyze_operation(prompt)
            results['operation_info'] = operation_details
            logger.debug(f"Operation analysis: {operation_details}")

            # --- 2. Analiza Contextului Imaginii ---
            logger.debug("Starting image context analysis...")
            image_context = self.img_analyzer.analyze_image_context(image)
            results['context_info'] = image_context

            # --- LOGURI ADAUGATE ---
            logger.info("--- Image Context Analysis Results ---")
            logger.info(f"  Analysis Time: {image_context.get('analysis_time_sec', 'N/A')}s")
            # Scenă
            scene_info = image_context.get('scene_info', {})
            logger.info(f"  Scene Tag (ML): {scene_info.get('primary_scene_tag_ml', 'N/A')}")
            logger.info(f"  Secondary Tags (ML): {scene_info.get('secondary_scene_tags_ml', [])}")
            # Obiecte Detectate
            detected_objects = scene_info.get('detected_objects', [])
            if detected_objects:
                 logger.info(f"  Detected Objects ({len(detected_objects)}):")
                 for obj in detected_objects[:3]: # Afișăm primele 3
                      logger.info(f"    - {obj.get('label', '?')} (Conf: {obj.get('confidence', 0):.2f})")
                 if len(detected_objects) > 3: logger.info("    ...")
            else:
                 logger.info("  Detected Objects: None")
            # Iluminare
            lighting_info = image_context.get('lighting_conditions', {})
            logger.info(f"  Lighting:")
            logger.info(f"    - Brightness: {lighting_info.get('brightness_heuristic', 'N/A')}")
            logger.info(f"    - Contrast: {lighting_info.get('contrast_heuristic', 'N/A')}")
            logger.info(f"    - Temperature: {lighting_info.get('temperature_heuristic', 'N/A')}")
            logger.info(f"    - Highlights: {lighting_info.get('highlights_pct', 0.0):.1f}%")
            logger.info(f"    - Shadows: {lighting_info.get('shadows_pct', 0.0):.1f}%")
            # Stil
            style_info = image_context.get('style_and_quality', {})
            logger.info(f"  Style (Heuristic): {style_info.get('visual_style_heuristic', 'N/A')}")
            # Adâncime
            depth_info = image_context.get('spatial_info', {})
            logger.info(f"  Depth Map Available: {depth_info.get('depth_map_available', False)}")
            if depth_info.get('depth_map_available'):
                 logger.info(f"  Depth Characteristics: {depth_info.get('depth_characteristics', 'N/A')}")
            logger.info(f"  Full Desc (Heuristic): {image_context.get('full_description_heuristic', 'N/A')}")
            logger.info("--- End Image Context Analysis ---")
            # --- END LOGURI ---

            if "error" in image_context:
                 logger.error(f"Error during image context analysis: {image_context['error']}")
                 results['status_message'] = f"Error in context analysis: {image_context['error']}"
                 # return results # Oprim? Depinde de cât de critică e analiza

            # --- 3. Selectare Pipeline ---
            operation_type = operation_details.get('type', 'general')
            target_object = operation_details.get('target_object', '')
            pipeline = self.pipeline_manager.get_pipeline_for_operation(operation_type, target_object)

            if not pipeline:
                error_msg = f"No pipeline for op '{operation_type}' (target: '{target_object}')."
                logger.error(error_msg)
                results['status_message'] = error_msg
                return results

            logger.info(f"Selected pipeline: {pipeline.__class__.__name__}")

            # --- 4. Executare Pipeline ---
            pipeline_kwargs = {
                "image": image, "prompt": prompt, "strength": strength,
                "operation": operation_details, "image_context": image_context,
                "progress_callback": lambda p, desc: logger.debug(f"Pipeline Progress: {p*100:.0f}% - {desc}"),
                **kwargs # Pasează toți parametrii avansati din UI
            }
            pipeline_result = pipeline.process(**pipeline_kwargs)

            # --- 5. Procesare Rezultat Pipeline ---
            if isinstance(pipeline_result, dict):
                 results['result_image'] = pipeline_result.get('result_image', results['result_image'])
                 results['mask_image'] = pipeline_result.get('mask_image')
                 results['operation_info'] = pipeline_result.get('operation', results['operation_info'])
                 results['status_message'] = pipeline_result.get('message', "Pipeline finished.")
                 # Verificăm explicit succesul din pipeline, altfel presupunem eșec dacă lipsește
                 results['success'] = pipeline_result.get('success', False)
            else:
                 # Tratăm cazul neașteptat în care pipeline-ul nu returnează dicționar
                 logger.warning(f"Pipeline {pipeline.__class__.__name__} returned unexpected type: {type(pipeline_result)}")
                 results['status_message'] = "Pipeline finished with unexpected return type."
                 # Încercăm să vedem dacă e o imagine PIL
                 if isinstance(pipeline_result, Image.Image):
                      results['result_image'] = pipeline_result
                      results['success'] = True # Asumăm succes în acest caz
                 else:
                      results['success'] = False


            # --- 6. (TODO) Post-Procesare ---
            # ... (logica de post-procesare va veni aici) ...

            processing_time_process = time.time() - start_time_process
            results['status_message'] += f" (Processing time: {processing_time_process:.2f}s)"
            logger.info(f"process_image finished in {processing_time_process:.2f}s. Success: {results['success']}")

        except Exception as e:
            error_message = f"Error during image processing pipeline execution: {str(e)}"
            logger.error(error_message, exc_info=True)
            results['status_message'] = f"Error: {e}"
            results['success'] = False
            results['operation_info']['error'] = str(e)
            import traceback
            results['operation_info']['traceback'] = traceback.format_exc()

        return results


    def launch(self, **kwargs):
        """Lansează interfața Gradio."""
        launch_kwargs = { "server_name": "0.0.0.0", "server_port": 7860, "share": False, "debug": False, **kwargs }
        logger.info(f"Launching Gradio interface with options: {launch_kwargs}")
        if hasattr(self, 'app') and self.app:
            try:
                self.app.launch(**launch_kwargs)
            except Exception as e:
                 logger.critical(f"Gradio launch failed: {e}", exc_info=True)
                 # Încercăm să oferim o sugestie dacă e o problemă comună de port
                 if "address already in use" in str(e).lower():
                      logger.info("Port may already be in use. Try stopping other services or using a different port with --port.")
                 sys.exit(1)
        else:
            logger.error("Gradio app object ('self.app') not found. Cannot launch.")


def main():
    """Funcția principală pentru rularea aplicației."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    # Argumente linie de comandă
    import argparse
    parser = argparse.ArgumentParser(description="FusionFrame 2.0 UI")
    parser.add_argument("--port", type=int, default=7860, help="Port")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host")
    parser.add_argument("--share", action="store_true", help="Enable Gradio share")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--low-vram", action="store_true", help="Enable low VRAM mode")
    args = parser.parse_args()

    # Configurare finală logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    AppConfig.setup_logging(level=log_level)

    # Setare mod low VRAM
    if args.low_vram:
         logger.info("Low VRAM mode requested via command line.")
         AppConfig.LOW_VRAM_MODE = True

    logger.info("Starting FusionFrame UI Application...")
    try:
        # Asigură directoarele necesare (mutat aici pentru a rula înainte de UI)
        AppConfig.ensure_dirs()
        ui = FusionFrameUI()
        ui.launch(
            server_name=args.host,
            server_port=args.port,
            share=args.share,
            debug=args.debug
        )
    except Exception as e:
        logger.critical(f"Failed to initialize or launch FusionFrameUI: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()