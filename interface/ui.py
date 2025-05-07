#!/usr/bin/env python
# -*- coding: utf-8 -*-\

"""
Interfața Gradio pentru FusionFrame 2.0
(Actualizat pentru a include apelul la PostProcessor)
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
import traceback # Pentru afișare erori

# Asigurăm căile corecte pentru importuri
try:
    from config.app_config import AppConfig
    from core.model_manager import ModelManager
    from core.pipeline_manager import PipelineManager
    from processing.analyzer import OperationAnalyzer, ImageAnalyzer
    # NOU: Importăm PostProcessor
    from processing.post_processor import PostProcessor

    # Verificăm existența fișierelor opționale de interfață
    components_path = os.path.join(os.path.dirname(__file__), 'components.py')
    styles_path = os.path.join(os.path.dirname(__file__), 'styles.py')

    if os.path.exists(components_path): from interface.components import create_examples, create_advanced_settings_panel
    else: create_examples, create_advanced_settings_panel = None, None; logging.warning(f"Optional UI file not found: {components_path}. Using fallbacks.")

    if os.path.exists(styles_path): from interface.styles import CSS_STYLES
    else: CSS_STYLES = ""; logging.warning(f"Optional UI file not found: {styles_path}. Using fallbacks.")

except ImportError as e:
    # Fallback: Adăugăm directorul rădăcină în PYTHONPATH
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    if project_root not in sys.path: sys.path.insert(0, project_root); print(f"Added project root to sys.path: {project_root}")
    # Reîncercăm importurile
    try:
        from config.app_config import AppConfig
        from core.model_manager import ModelManager
        from core.pipeline_manager import PipelineManager
        from processing.analyzer import OperationAnalyzer, ImageAnalyzer
        from processing.post_processor import PostProcessor # Reîncercăm și PostProcessor

        if os.path.exists(components_path): from interface.components import create_examples, create_advanced_settings_panel
        else: create_examples, create_advanced_settings_panel = None, None; logging.warning(f"Optional UI file not found: {components_path}. Using fallbacks.")
        if os.path.exists(styles_path): from interface.styles import CSS_STYLES
        else: CSS_STYLES = ""; logging.warning(f"Optional UI file not found: {styles_path}. Using fallbacks.")

    except ImportError as e_retry:
         print(f"FATAL: Could not import necessary modules. Error: {e_retry}"); print(f"sys.path: {sys.path}"); sys.exit(1)

# Setăm logger-ul principal
logger = logging.getLogger(__name__)
if not logger.hasHandlers(): logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


class FusionFrameUI:
    """Interfața utilizator pentru FusionFrame 2.0"""
    def __init__(self):
        logger.info("Initializing FusionFrameUI...")
        self.config = AppConfig
        self.model_manager = ModelManager()
        self.pipeline_manager = PipelineManager()
        self.op_analyzer = OperationAnalyzer()
        self.img_analyzer = ImageAnalyzer()
        # NOU: Instanțiem PostProcessor aici, presupunând că nu are stare complexă
        # Alternativ, poate fi instanțiat în process_image_gradio_wrapper la nevoie
        self.post_processor = PostProcessor()

        self.load_models()
        self.app = self.create_interface()
        logger.info("FusionFrameUI initialized successfully.")

    def load_models(self):
        """Încarcă/verifică modelele esențiale."""
        logger.info("Loading/Checking essential models...")
        try:
            main_model = self.model_manager.get_model('main')
            if main_model and getattr(main_model, 'is_loaded', False): logger.info("Main model confirmed loaded.")
            else: logger.warning("Main model check/load initiated or failed. Check logs.")
            # Verificăm/încărcăm și modelele necesare pentru PostProcessor (dacă e cazul)
            # self.model_manager.get_model('esrgan') # Exemplu, încărcarea se face lazy în PostProcessor
            # self.model_manager.get_model('gpen')
        except Exception as e:
            logger.error(f"Error during models loading: {str(e)}", exc_info=True)

    def create_interface(self) -> gr.Blocks:
        """Creează interfața Gradio."""
        logger.info("Creating Gradio interface...")
        global CSS_STYLES; css_to_use = CSS_STYLES if 'CSS_STYLES' in globals() else ""

        with gr.Blocks(theme=gr.themes.Soft(), css=css_to_use) as app:
            gr.Markdown(f"# 🚀 FusionFrame {self.config.VERSION} - Advanced AI Image Editor")

            with gr.Row(equal_height=False):
                with gr.Column(scale=1): # Coloana Input
                    image_input = gr.Image(type="pil", label="Upload Image", elem_id="image_input", height=400)
                    with gr.Row():
                        prompt = gr.Textbox(label="Edit Instructions", placeholder="E.g., 'Remove the car'", elem_id="prompt-input", scale=3)
                        strength = gr.Slider(minimum=0.1, maximum=1.0, value=0.75, step=0.05, label="Edit Strength", info="Higher = more dramatic", scale=1)
                    run_btn = gr.Button("Generate Edit", variant="primary", elem_id="generate-btn")
                    status_area = gr.Textbox(label="Status", value="Ready.", elem_id="status-area", interactive=False, lines=2)

                with gr.Column(scale=1): # Coloana Output
                    image_output = gr.Image(label="Edited Result", elem_id="image_output", height=400)
                    with gr.Row():
                        mask_output = gr.Image(label="Generated Mask (Debug)", elem_id="mask_output", height=200)
                        depth_output = gr.Image(label="Depth Map (Debug)", elem_id="depth_output", height=200)
                    with gr.Accordion("Operation Details & Analysis", open=False):
                        info_json = gr.JSON(label="Operation & Context Info")

            # Secțiune exemple
            global create_examples
            if create_examples and callable(create_examples):
                example_list = create_examples(); gr.Examples(examples=example_list, inputs=[prompt, strength], label="Example Prompts")
            else: logger.warning("create_examples function not available.")

            # Panou de setări avansate
            with gr.Accordion("Advanced Settings & Post-Processing", open=False): # Redenumit
                global create_advanced_settings_panel
                # Obținem lista de componente din create_advanced_settings_panel
                if create_advanced_settings_panel and callable(create_advanced_settings_panel):
                    advanced_settings_list = create_advanced_settings_panel()
                    if not isinstance(advanced_settings_list, list):
                         logger.error("create_advanced_settings_panel fallback needed."); advanced_settings_list = self._create_fallback_advanced_settings()
                else:
                    logger.warning("create_advanced_settings_panel not available. Using fallback."); advanced_settings_list = self._create_fallback_advanced_settings()

                # NOU: Adăugăm controale specifice pentru Post-Processing dacă nu există deja
                # Verificăm dacă există deja controale pentru blending/harmonization
                has_blend_ctrl = any("Blend" in getattr(c, 'label', '') for c in advanced_settings_list if isinstance(c, gr.Checkbox))
                has_harmonize_ctrl = any("Harmoniz" in getattr(c, 'label', '') for c in advanced_settings_list if isinstance(c, gr.Checkbox))

                # Creăm controalele de post-procesare dacă lipsesc
                post_proc_controls = []
                if not has_blend_ctrl: post_proc_controls.append(gr.Checkbox(label="Seamless Blending", value=True, info="Smooth edges between edited and original areas."))
                if not has_harmonize_ctrl: post_proc_controls.append(gr.Checkbox(label="Color Harmonization", value=True, info="Adjust colors in edited area to match the original."))

                # Adăugăm controalele noi la listă dacă am creat vreunul
                if post_proc_controls:
                     with gr.Row(): # Le punem într-un rând nou
                         for ctrl in post_proc_controls: ctrl.render() # Render explicit dacă le adăugăm dinamic
                     advanced_settings_list.extend(post_proc_controls) # Adăugăm la lista de inputuri

                # Linkăm vizibilitatea refiner strength (presupunând ordinea sau label-ul)
                self._link_refiner_visibility(advanced_settings_list)


            # Panou de informații
            with gr.Accordion("Tips & Info", open=False): gr.Markdown(self._get_tips_markdown())

            # Funcționalitate buton
            active_advanced_settings = advanced_settings_list if isinstance(advanced_settings_list, list) else []
            run_btn.click(
                fn=self.process_image_gradio_wrapper,
                inputs=[image_input, prompt, strength] + active_advanced_settings,
                outputs=[image_output, mask_output, depth_output, info_json, status_area]
            )

        logger.info("Gradio interface created.")
        return app

    def _link_refiner_visibility(self, components_list):
         """Leagă vizibilitatea slider-ului de refiner strength de checkbox-ul use_refiner."""
         use_refiner_comp = None; strength_comp = None
         for c in components_list:
             label = getattr(c, 'label', '').lower()
             if isinstance(c, gr.Checkbox) and 'refiner' in label: use_refiner_comp = c
             if isinstance(c, gr.Slider) and 'refiner strength' in label: strength_comp = c
         if use_refiner_comp and strength_comp:
              use_refiner_comp.change(lambda x: gr.update(visible=x), inputs=[use_refiner_comp], outputs=[strength_comp])
         else: logger.warning("Could not link refiner visibility.")

    def _get_tips_markdown(self):
         """Returnează textul Markdown pentru secțiunea Tips."""
         return """
         ### Tips for better results:
         - Be specific: "remove the *red* car on the *left*" vs "remove car".
         - For replacements, specify what to replace with.
         - Adjust strength slider for more/less dramatic changes.
         - Check the generated mask/depth map (debug view).
         - Enable Post-Processing options in Advanced Settings for better integration.

         ### Common operations:
         - **Remove**: "remove [object]"
         - **Replace**: "replace [object] with [new object]"
         - **Color**: "change color of [object] to [color]"
         - **Background**: "change background to [scene]"
         - **Add**: "add [object]"
         """

    def _create_fallback_advanced_settings(self) -> List[gr.components.Component]:
         """Creează controale default pentru setări avansate și post-procesare."""
         # Valori Default
         cfg = self.config
         steps = getattr(cfg, 'DEFAULT_STEPS', 50); guidance = getattr(cfg, 'DEFAULT_GUIDANCE_SCALE', 7.5)
         use_refiner = getattr(cfg, 'USE_REFINER', True); ref_strength = getattr(cfg, 'REFINER_STRENGTH', 0.3)

         # Componente Gradio
         comps = []
         with gr.Row():
             comps.append(gr.Slider(minimum=10, maximum=150, value=steps, step=1, label="Inference Steps"))
             comps.append(gr.Slider(minimum=1.0, maximum=20.0, value=guidance, step=0.5, label="Guidance Scale"))
         with gr.Row():
             comps.append(gr.Checkbox(label="Enhance Details", value=True))
             comps.append(gr.Checkbox(label="Fix Faces", value=True))
             comps.append(gr.Checkbox(label="Remove Artifacts", value=True))
         with gr.Row():
             comps.append(gr.Checkbox(label="Use ControlNet", value=True))
             comps.append(gr.Checkbox(label="Use Refiner", value=use_refiner))
             comps.append(gr.Slider(minimum=0.0, maximum=1.0, value=ref_strength, step=0.05, label="Refiner Strength", visible=use_refiner))
         # NOU: Adăugăm controale Post-Processing
         with gr.Row():
             comps.append(gr.Checkbox(label="Seamless Blending", value=True, info="Smooth edges."))
             comps.append(gr.Checkbox(label="Color Harmonization", value=True, info="Adjust colors."))

         return comps


    def process_image_gradio_wrapper(self, *args):
        """Wrapper Gradio: gestionează input/output și apelează procesarea + post-procesarea."""
        start_time_wrapper = time.time()
        image_pil, prompt_text, strength_value, *advanced_args_tuple = args
        advanced_args = list(advanced_args_tuple)

        # --- Mapare Argumente Avansate ---
        # Asigurăm că numele corespund celor din _create_fallback_advanced_settings / components.py
        # Ordinea este importantă!
        param_names = [
            "num_inference_steps", "guidance_scale", "enhance_details", "fix_faces",
            "remove_artifacts", "use_controlnet", "use_refiner", "refiner_strength",
            "seamless_blending", "color_harmonization" # NOU: Parametri Post-Processing
        ]
        kwargs_for_processing = {}
        num_expected_advanced = len(param_names)
        advanced_args.extend([None] * (num_expected_advanced - len(advanced_args))) # Padding cu None

        for i, name in enumerate(param_names):
            kwargs_for_processing[name] = advanced_args[i] # None va lăsa default-ul din funcție

        logger.info(f"Processing request: Prompt='{prompt_text}', Strength={strength_value}")
        logger.debug(f"Advanced Kwargs received: {kwargs_for_processing}")

        # --- Status Inițial & Validări ---
        yield None, None, None, {}, "Status: Starting..."
        if image_pil is None: yield None, None, None, {"error": "No image"}, "Error: Upload image."; return
        if not prompt_text or not prompt_text.strip(): yield image_pil, None, None, {"warning": "Empty prompt"}, "Warning: Prompt empty."; return

        # --- Asigurare Model Principal ---
        if not (self.model_manager.get_model('main') and getattr(self.model_manager.get_model('main'), 'is_loaded', False)):
            error_msg = "Critical: Main model not loaded."; logger.error(error_msg)
            yield image_pil, None, None, {"error": error_msg}, error_msg; return

        # --- Procesare Principală (Pipeline) ---
        final_result_img = image_pil # Imaginea de returnat (inițial originală)
        mask_img_pipeline = None
        depth_map_display = None
        op_info = {}
        context_info = {}
        status_msg = "Starting pipeline..."
        success_pipeline = False
        post_proc_applied = False
        pipeline_result_dict = None

        try:
            # Pasează kwargs direct la process_image (care le pasează la pipeline)
            pipeline_result_dict = self.process_image(image_pil, prompt_text, strength_value, **kwargs_for_processing)

            # Extragem rezultatele din pipeline
            final_result_img = pipeline_result_dict.get('result_image', image_pil) # Folosim ce returnează pipeline-ul
            mask_img_pipeline = pipeline_result_dict.get('mask_image') # Poate fi None
            op_info = pipeline_result_dict.get('operation_info', {})
            context_info = pipeline_result_dict.get('context_info', {})
            status_msg = pipeline_result_dict.get('status_message', "Pipeline finished.")
            success_pipeline = pipeline_result_dict.get('success', False)

            # --- NOU: Apel Post-Processor Condiționat ---
            if success_pipeline and final_result_img:
                # Verificăm flag-urile de post-procesare din kwargs_for_processing
                should_post_process = any(kwargs_for_processing.get(flag) for flag in
                                          ["enhance_details", "fix_faces", "remove_artifacts",
                                           "seamless_blending", "color_harmonization"])
                                           
                # Asigurăm valori default True pentru blend/harmonize dacă nu sunt specificate altfel
                # (dacă nu există checkbox-uri, vor fi None, deci folosim True)
                blend_flag = kwargs_for_processing.get('seamless_blending') if kwargs_for_processing.get('seamless_blending') is not None else True
                harmonize_flag = kwargs_for_processing.get('color_harmonization') if kwargs_for_processing.get('color_harmonization') is not None else True
                
                if should_post_process or blend_flag or harmonize_flag:
                     logger.info("Applying post-processing steps...")
                     status_msg += " Applying post-processing..."
                     # Actualizăm statusul în UI (yield-ul intermediar e o opțiune aici)
                     yield final_result_img, mask_img_pipeline, depth_map_display, {"status": "Post-processing..."}, status_msg

                     post_proc_kwargs = {
                         "image": final_result_img, # Imaginea de la pipeline
                         "original_image": image_pil, # Originalul
                         "mask": mask_img_pipeline, # Masca de la pipeline
                         "operation_type": op_info.get('type'),
                         "enhance_details": kwargs_for_processing.get('enhance_details', False),
                         "fix_faces": kwargs_for_processing.get('fix_faces', False),
                         "remove_artifacts": kwargs_for_processing.get('remove_artifacts', False),
                         "seamless_blending": blend_flag,
                         "color_harmonization": harmonize_flag,
                         # "progress_callback": ?? # Necesită integrare cu gr.Progress
                     }

                     try:
                         post_result_dict = self.post_processor.process(**post_proc_kwargs)
                         if post_result_dict.get('success'):
                             final_result_img = post_result_dict.get('result_image', final_result_img) # Actualizăm cu rezultatul PP
                             status_msg = status_msg.replace("Applying post-processing...", post_result_dict.get('message', 'Post-processing applied.'))
                             post_proc_applied = True
                             logger.info("Post-processing applied successfully.")
                         else:
                             status_msg += f" Post-processing failed: {post_result_dict.get('message')}"
                             logger.warning(f"Post-processing failed: {post_result_dict.get('message')}")
                     except Exception as e_pp:
                          status_msg += f" Post-processing error: {e_pp}"
                          logger.error(f"Error during PostProcessor call: {e_pp}", exc_info=True)
            # --- Sfârșit Apel Post-Processor ---

            # Pregătim afișare depth map (ca înainte)
            if context_info and context_info.get('spatial_info', {}).get('depth_map_available'):
                 depth_map_np = context_info['spatial_info']['depth_map']
                 if depth_map_np is not None:
                      depth_map_norm = cv2.normalize(depth_map_np, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                      depth_map_display = Image.fromarray(depth_map_norm)

            # JSON final
            final_info_json = { "Operation Analysis": op_info, "Image Context Analysis": context_info,
                                "Processing Status": "Success" if success_pipeline else "Failed",
                                "PostProcessing Applied": post_proc_applied }
            if not success_pipeline and "error" not in final_info_json["Operation Analysis"]:
                 final_info_json["Operation Analysis"]["error"] = status_msg

            processing_time = time.time() - start_time_wrapper
            status_msg += f" (Total UI time: {processing_time:.2f}s)"
            logger.info(f"Request finished. Pipeline Success: {success_pipeline}, PostProc Applied: {post_proc_applied}. {status_msg}")

            # Actualizăm UI cu imaginea finală (posibil post-procesată)
            yield final_result_img, mask_img_pipeline, depth_map_display, final_info_json, status_msg

        except Exception as e:
            error_message = f"Critical error in UI processing wrapper: {str(e)}"
            logger.error(error_message, exc_info=True)
            # Returnăm imaginea originală și eroarea
            yield image_pil, None, None, {"error": error_message, "traceback": traceback.format_exc()}, error_message


    def process_image(self,
                      image: Image.Image,
                      prompt: str,
                      strength: float,
                      **kwargs) -> Dict[str, Any]:
        """Orchestrează analiza și apelarea pipeline-ului corespunzător."""
        start_time_process = time.time()
        logger.info(f"Process_image started. Kwargs: {kwargs}")

        results = { # Dicționarul returnat de pipeline-uri ar trebui să aibă această structură
            'result_image': image, 'mask_image': None, 'operation_info': {},
            'context_info': {}, 'status_message': "Processing started.", 'success': False
        }

        try:
            # 1. Analiza Operației (din prompt)
            operation_details = self.op_analyzer.analyze_operation(prompt)
            results['operation_info'] = operation_details

            # 2. Analiza Contextului Imaginii
            image_context = self.img_analyzer.analyze_image_context(image)
            results['context_info'] = image_context
            # Logarea contextului (ca înainte)
            self._log_image_context(image_context)
            if "error" in image_context: logger.error(f"Context analysis error: {image_context['error']}")

            # 3. Selectare Pipeline
            op_type = operation_details.get('type', 'general')
            target_obj = operation_details.get('target_object', '')
            pipeline = self.pipeline_manager.get_pipeline_for_operation(op_type, target_obj)
            if not pipeline:
                 msg = f"No pipeline found for op '{op_type}' (target: '{target_obj}')."; logger.error(msg)
                 results['message'] = msg; return results

            logger.info(f"Selected pipeline: {pipeline.__class__.__name__}")

            # 4. Executare Pipeline (pasează toți kwargs primiți)
            pipeline_kwargs = {
                "image": image, "prompt": prompt, "strength": strength,
                "operation": operation_details, "image_context": image_context,
                "progress_callback": lambda p, desc: logger.debug(f"Pipeline: {p*100:.0f}% - {desc}"),
                **kwargs # Pasează num_steps, guidance, use_refiner, etc.
            }
            pipeline_result_dict = pipeline.process(**pipeline_kwargs)

            # Actualizăm dicționarul results cu ce a returnat pipeline-ul
            results.update(pipeline_result_dict)
            # Asigurăm că imaginea rezultată e PIL
            if results.get('result_image') and isinstance(results['result_image'], np.ndarray):
                 results['result_image'] = self._convert_cv2_to_pil(results['result_image']) # Helper de conversie

            processing_time_process = time.time() - start_time_process
            results['status_message'] = results.get('message', "Pipeline finished.") + f" (Pipeline time: {processing_time_process:.2f}s)"
            logger.info(f"process_image finished. Success: {results['success']}. {results['status_message']}")

        except Exception as e:
            msg = f"Error during process_image orchestration: {str(e)}"; logger.error(msg, exc_info=True)
            results['status_message'] = msg; results['success'] = False
            results['operation_info']['error'] = str(e); results['operation_info']['traceback'] = traceback.format_exc()

        return results # Returnăm dicționarul complet

    def _log_image_context(self, image_context):
         """Funcție helper pentru logarea contextului."""
         logger.info("--- Image Context Analysis Results ---")
         logger.info(f"  Analysis Time: {image_context.get('analysis_time_sec', 'N/A')}s")
         scene_info = image_context.get('scene_info', {})
         logger.info(f"  Scene Tag (ML): {scene_info.get('primary_scene_tag_ml', 'N/A')}")
         detected_objects = scene_info.get('detected_objects', [])
         if detected_objects: logger.info(f"  Detected Objects ({len(detected_objects)}): {[obj['label'] for obj in detected_objects[:3]]}...")
         else: logger.info("  Detected Objects: None")
         lighting_info = image_context.get('lighting_conditions', {})
         logger.info(f"  Lighting: B={lighting_info.get('brightness_heuristic','?')}, C={lighting_info.get('contrast_heuristic','?')}, T={lighting_info.get('temperature_heuristic','?')}, H={lighting_info.get('highlights_pct','?')}%, S={lighting_info.get('shadows_pct','?')}%.")
         style_info = image_context.get('style_and_quality', {})
         logger.info(f"  Style (Heuristic): {style_info.get('visual_style_heuristic', 'N/A')}")
         depth_info = image_context.get('spatial_info', {})
         logger.info(f"  Depth Map: {'Available' if depth_info.get('depth_map_available') else 'No'}, Characteristics: {depth_info.get('depth_characteristics', 'N/A')}")
         logger.info(f"  Full Desc (Heuristic): {image_context.get('full_description_heuristic', 'N/A')}")
         logger.info("--- End Image Context Analysis ---")

    def _convert_cv2_to_pil(self, image_np):
         """Convertește NumPy BGR în PIL RGB."""
         if image_np is None: return None
         try: return Image.fromarray(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))
         except Exception as e: logger.error(f"CV2 to PIL conversion failed: {e}"); return None


    def launch(self, **kwargs):
        """Lansează interfața Gradio."""
        launch_kwargs = { "server_name": "0.0.0.0", "server_port": 7860, "share": False, "debug": False, **kwargs }
        logger.info(f"Launching Gradio interface with options: {launch_kwargs}")
        if hasattr(self, 'app') and self.app:
            try: self.app.launch(**launch_kwargs)
            except Exception as e:
                 logger.critical(f"Gradio launch failed: {e}", exc_info=True)
                 if "address already in use" in str(e).lower(): logger.info("Port may be in use. Try stopping other services or using --port.")
                 sys.exit(1)
        else: logger.error("Gradio app object not found. Cannot launch.")


def main():
    """Funcția principală pentru rularea aplicației."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    import argparse
    parser = argparse.ArgumentParser(description="FusionFrame 2.0 UI")
    parser.add_argument("--port", type=int, default=7860, help="Port")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host")
    parser.add_argument("--share", action="store_true", help="Enable Gradio share")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--low-vram", action="store_true", help="Enable low VRAM mode")
    args = parser.parse_args()

    log_level = logging.DEBUG if args.debug else logging.INFO
    AppConfig.setup_logging(level=log_level) # Setup logging final

    if args.low_vram: logger.info("Low VRAM mode requested."); AppConfig.LOW_VRAM_MODE = True

    logger.info("Starting FusionFrame UI Application...")
    try:
        AppConfig.ensure_dirs() # Asigură directoare
        ui = FusionFrameUI()
        ui.launch(server_name=args.host, server_port=args.port, share=args.share, debug=args.debug)
    except Exception as e:
        logger.critical(f"Failed to initialize or launch FusionFrameUI: {e}", exc_info=True); sys.exit(1)

if __name__ == "__main__":
    main()