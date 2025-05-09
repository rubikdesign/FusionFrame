#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Gradio interface for FusionFrame 2.0
Versiune simplificată și optimizată pentru Gradio 3.50.2
"""

import os
import sys
import logging
import gradio as gr
import numpy as np
import time
from typing import Dict, Any, List, Optional, Union, Tuple, Callable
from PIL import Image
import cv2
import traceback

# Ensure correct import paths
try:
    from config.app_config import AppConfig
    from core.model_manager import ModelManager
    from core.pipeline_manager import PipelineManager
    from processing.analyzer import OperationAnalyzer, ImageAnalyzer
    from processing.post_processor import PostProcessor
    from models.base_model import BaseModel

    # Check for optional interface files
    components_path = os.path.join(os.path.dirname(__file__), 'components.py')
    styles_path = os.path.join(os.path.dirname(__file__), 'styles.py')

    if os.path.exists(components_path): 
        from interface.components import create_examples, create_advanced_settings_panel
    else: 
        create_examples, create_advanced_settings_panel = None, None 
        logging.warning(f"Optional UI file not found: {components_path}. Using fallbacks.")

    if os.path.exists(styles_path): 
        from interface.styles import CSS_STYLES
    else: 
        CSS_STYLES = "" 
        logging.warning(f"Optional UI file not found: {styles_path}. Using fallbacks.")

except ImportError as e:
    # Fallback: Add root directory to PYTHONPATH
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    if project_root not in sys.path: 
        sys.path.insert(0, project_root)
        print(f"Added project root to sys.path: {project_root}")
    
    # Retry imports
    try:
        from config.app_config import AppConfig
        from core.model_manager import ModelManager
        from core.pipeline_manager import PipelineManager
        from processing.analyzer import OperationAnalyzer, ImageAnalyzer
        from processing.post_processor import PostProcessor
        from models.base_model import BaseModel

        if os.path.exists(components_path): 
            from interface.components import create_examples, create_advanced_settings_panel
        else: 
            create_examples, create_advanced_settings_panel = None, None
            logging.warning(f"UI file missing: {components_path}")
            
        if os.path.exists(styles_path): 
            from interface.styles import CSS_STYLES
        else: 
            CSS_STYLES = ""
            logging.warning(f"UI file missing: {styles_path}")

    except ImportError as e_retry: 
        print(f"FATAL: Could not import necessary modules. Error: {e_retry}")
        print(f"sys.path: {sys.path}")
        sys.exit(1)

# Set up main logger
logger = logging.getLogger(__name__)
if not logger.hasHandlers(): 
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(name)s:%(message)s")

class FusionFrameUI:
    """User interface for FusionFrame 2.0"""
    def __init__(self):
        logger.info("Initializing FusionFrameUI...")
        self.config = AppConfig
        self.model_manager = ModelManager()
        self.pipeline_manager = PipelineManager()
        self.op_analyzer = OperationAnalyzer()
        self.img_analyzer = ImageAnalyzer()
        self.post_processor = PostProcessor()

        self.load_models() # Check main model at initialization
        self.app = self.create_interface()
        logger.info("FusionFrameUI initialized.")

    def load_models(self):
        """Load/check essential models."""
        logger.info("Loading/Checking essential models...")
        try:
            main_model = self.model_manager.get_model('main')
            status = "LOADED" if main_model and getattr(main_model, 'is_loaded', False) else "NOT LOADED/FAILED"
            logger.info(f"Main model status: {status}")
        except Exception as e: 
            logger.error(f"Error loading/checking models: {e}", exc_info=True)

    def create_interface(self) -> gr.Blocks:
        """Create the Gradio interface."""
        logger.info("Creating Gradio interface...")
        global CSS_STYLES
        css_to_use = CSS_STYLES if 'CSS_STYLES' in globals() else ""
        adv_settings_list = []

        with gr.Blocks(css=css_to_use) as app:
            gr.Markdown(f"# 🚀 FusionFrame {self.config.VERSION} - Advanced AI Image Editor")
            with gr.Row(equal_height=False):
                with gr.Column(scale=1): # Input
                    image_input = gr.Image(type="pil", label="Upload Image", elem_id="image_input", height=400)
                    with gr.Row():
                        prompt = gr.Textbox(label="Edit Instructions", placeholder="E.g., 'Remove the car'", elem_id="prompt-input", scale=3)
                        strength = gr.Slider(minimum=0.1, maximum=1.0, value=0.75, step=0.05, label="Edit Strength", info="Higher = more dramatic", scale=1)
                    run_btn = gr.Button("Generate Edit", variant="primary", elem_id="generate-btn")
                    status_area = gr.Textbox(label="Status", value="Ready.", elem_id="status-area", interactive=False, lines=2)
                with gr.Column(scale=1): # Output
                    image_output = gr.Image(label="Edited Result", elem_id="image_output", height=400)
                    with gr.Row():
                        mask_output = gr.Image(label="Mask (Debug)", elem_id="mask_output", height=200)
                        depth_output = gr.Image(label="Depth Map (Debug)", elem_id="depth_output", height=200)
                    with gr.Accordion("Operation Details & Analysis", open=False):
                        info_json = gr.JSON(label="Operation & Context Info")

            # Examples
            global create_examples
            if create_examples and callable(create_examples):
                example_list = create_examples()
                gr.Examples(examples=example_list, inputs=[prompt, strength], label="Example Prompts")
            else: 
                logger.debug("create_examples fn missing or not callable.")

            # Advanced Settings & Post-Processing
            with gr.Accordion("Advanced Settings & Post-Processing", open=False):
                global create_advanced_settings_panel
                # Get list of components
                if create_advanced_settings_panel and callable(create_advanced_settings_panel):
                    adv_settings_list = create_advanced_settings_panel()
                    if not isinstance(adv_settings_list, list): 
                        adv_settings_list = self._create_fallback_advanced_settings()
                else: 
                    adv_settings_list = self._create_fallback_advanced_settings()

            # Tips
            with gr.Accordion("Tips & Info", open=False): 
                gr.Markdown(self._get_tips_markdown())

            # Run Button
            run_btn.click(
                fn=self.process_image_wrapper,
                inputs=[image_input, prompt, strength] + adv_settings_list,
                outputs=[image_output, mask_output, depth_output, info_json, status_area]
            )
        
        logger.info("Gradio interface created.")
        return app

    def _get_tips_markdown(self):
         """Return Markdown text for Tips section."""
         return """
         ### Tips for better results:
         - Be specific: "remove the *red* car on the *left*" vs "remove car".
         - Enable relevant Post-Processing options in Advanced Settings for better integration (Blending, Harmonization are often useful).
         - Adjust strength slider. Check the generated mask/depth map.
         ### Common operations:
         - **Remove**: "remove [object]"
         - **Replace**: "replace [object] with [new object]"
         - **Color**: "change color of [object] to [color]"
         - **Background**: "change background to [scene]"
         - **Add**: "add [object]"
         """

    def _create_fallback_advanced_settings(self) -> List[gr.components.Component]:
        """Fallback for advanced UI settings (including post-processing)."""
        cfg = self.config; comps = []
        with gr.Row(): # Row 1: Steps and Guidance
            comps.extend([
                gr.Slider(minimum=10, maximum=150, value=getattr(cfg,'DEFAULT_STEPS',50), step=1, label="Inference Steps"),
                gr.Slider(minimum=1.0, maximum=20.0, value=getattr(cfg,'DEFAULT_GUIDANCE_SCALE',7.5), step=0.5, label="Guidance Scale")
            ])
        with gr.Row(): # Row 2: ControlNet and Refiner
            use_refiner_default = getattr(cfg,'USE_REFINER',True)
            comps.extend([
                gr.Checkbox(label="Use ControlNet", value=True),
                gr.Checkbox(label="Use Refiner", value=use_refiner_default),
                gr.Slider(minimum=0.0, maximum=1.0, value=getattr(cfg,'REFINER_STRENGTH',0.3), step=0.05, label="Refiner Strength", visible=use_refiner_default)
            ])
        with gr.Row(): # Row 3: Post-Processing (part 1)
            comps.extend([
                gr.Checkbox(label="Enhance Details", value=False, info="Use ESRGAN/Sharpening"), # Default False
                gr.Checkbox(label="Fix Faces", value=True, info="Use GPEN/CodeFormer"), # Default True
                gr.Checkbox(label="Remove Artifacts", value=False, info="Apply light smoothing") # Default False
            ])
        with gr.Row(): # Row 4: Post-Processing (part 2)
            comps.extend([
                gr.Checkbox(label="Seamless Blending", value=True, info="Smooth mask edges"), # Default True
                gr.Checkbox(label="Color Harmonization", value=True, info="Adjust edited colors") # Default True
            ])
            
        # Connect refiner visibility
        use_refiner = comps[1]  # Based on index in the list
        refiner_strength = comps[2]  # Based on index in the list
        
        # Update visibility
        def update_visibility(x):
            return gr.update(visible=x)
            
        use_refiner.change(
            fn=update_visibility,
            inputs=[use_refiner],
            outputs=[refiner_strength]
        )
        
        return comps

    def process_image_wrapper(self, *args):
        """Procesează imaginea - wrapper simplificat care NU folosește yield"""
        start_time = time.time()
        logger.info("Received processing request via Gradio.")
        
        # Extrage argumentele
        image_pil, prompt_text, strength_value, *advanced_args_tuple = args
        advanced_args = list(advanced_args_tuple)

        # Redimensionează imaginea pentru a economisi memorie
        if image_pil is not None and max(image_pil.width, image_pil.height) > 512:
            ratio = 512 / max(image_pil.width, image_pil.height)
            new_size = (int(image_pil.width * ratio), int(image_pil.height * ratio))
            image_pil = image_pil.resize(new_size, Image.LANCZOS)
            logger.info(f"Image resized to {new_size}")

        # --- Advanced Arguments Mapping ---
        param_names = [
            "num_inference_steps", "guidance_scale",
            "use_controlnet", "use_refiner", "refiner_strength",
            "enhance_details", "fix_faces", "remove_artifacts",
            "seamless_blending", "color_harmonization"
        ]
        kwargs_for_processing = {}
        num_expected = len(param_names)
        advanced_args.extend([None] * (num_expected - len(advanced_args)))
        for i, name in enumerate(param_names): 
            kwargs_for_processing[name] = advanced_args[i]
        
        # --- Initial Status & Validations ---
        if image_pil is None:
            return image_pil, None, None, {"error": "No image"}, "Error: Upload image."
        if not prompt_text or not prompt_text.strip():
            return image_pil, None, None, {"warning": "Empty prompt"}, "Warning: Prompt empty."
        if not (self.model_manager.get_model('main') and getattr(self.model_manager.get_model('main'), 'is_loaded', False)):
            msg = "Critical: Main model not loaded."
            logger.error(msg)
            return image_pil, None, None, {"error": msg}, msg

        # --- Main Processing (Pipeline) ---
        try:
            # Pass all kwargs (including post-proc flags) to internal process_image
            pipeline_result_dict = self.process_image(image_pil, prompt_text, strength_value, **kwargs_for_processing)

            success = pipeline_result_dict.get('success', False)
            status_msg = pipeline_result_dict.get('status_message', "Pipeline finished.")
            # Initial result (may be updated by post-processing)
            final_result_img = pipeline_result_dict.get('result_image', image_pil)
            mask_img_pipeline = pipeline_result_dict.get('mask_image')
            op_info = pipeline_result_dict.get('operation_info', {})
            context_info = pipeline_result_dict.get('context_info', {})

            # Display mask returned by pipeline in UI
            mask_img_display = mask_img_pipeline

            # Prepare depth map for display (from context)
            depth_map_display = None
            if context_info and context_info.get('spatial_info', {}).get('depth_map_available'):
                 depth_map_np = context_info['spatial_info']['depth_map']
                 if depth_map_np is not None:
                      depth_map_norm = cv2.normalize(depth_map_np, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                      depth_map_display = Image.fromarray(depth_map_norm)

            # --- Post-Processor Call ---
            post_proc_applied = False
            if success and final_result_img:
                # Check flags from kwargs_for_processing
                should_post_process = any(kwargs_for_processing.get(flag) for flag in
                                          ["enhance_details", "fix_faces", "remove_artifacts",
                                           "seamless_blending", "color_harmonization"])

                if should_post_process:
                     logger.info("Applying post-processing steps...")
                     status_msg += " + PostProcessing..."

                     post_proc_kwargs = {
                         "image": final_result_img, # Image *after* pipeline
                         "original_image": image_pil, # Original from UI
                         "mask": mask_img_pipeline, # Mask *from* pipeline (PIL 'L' or None)
                         "operation_type": op_info.get('type'),
                         # Pass flags directly
                         "enhance_details": kwargs_for_processing.get('enhance_details', False),
                         "fix_faces": kwargs_for_processing.get('fix_faces', False),
                         "remove_artifacts": kwargs_for_processing.get('remove_artifacts', False),
                         "seamless_blending": kwargs_for_processing.get('seamless_blending', True), # Default True
                         "color_harmonization": kwargs_for_processing.get('color_harmonization', True) # Default True
                     }
                     try:
                         post_result_dict = self.post_processor.process(**post_proc_kwargs)
                         if post_result_dict.get('success'):
                             # Update final image with post-processing result
                             final_result_img = post_result_dict.get('result_image', final_result_img)
                             status_msg = status_msg.replace("PostProcessing...", post_result_dict.get('message', 'PP OK.'))
                             post_proc_applied = True
                             logger.info("Post-processing successful.")
                         else:
                             status_msg += f" PP Failed: {post_result_dict.get('message')}"
                             logger.warning(f"Post-processing failed: {post_result_dict.get('message')}")
                     except Exception as e_pp:
                          status_msg += f" PP Error: {e_pp}"
                          logger.error(f"PostProcessor error: {e_pp}", exc_info=True)
            # --- End Post-Processor ---

            # Final JSON for UI
            final_info_json = { 
                "Operation": op_info, 
                "Context": context_info,
                "Status": "Success" if success else "Failed", 
                "PostProcessing": post_proc_applied 
            }
            
            if not success and "error" not in final_info_json["Operation"]: 
                final_info_json["Operation"]["error"] = status_msg

            total_time = time.time() - start_time
            status_msg += f" (Total: {total_time:.2f}s)"
            logger.info(f"Request finished. Success: {success}, PP: {post_proc_applied}. {status_msg}")

            # Salvează imaginea rezultat pentru debugging
            if success and final_result_img:
                try:
                    output_path = os.path.join(self.config.OUTPUT_DIR, f"result_{int(time.time())}.jpg")
                    final_result_img.save(output_path)
                    logger.info(f"Saved result to {output_path}")
                except Exception as e_save:
                    logger.error(f"Could not save result: {e_save}")

            # Return final result
            return final_result_img, mask_img_display, depth_map_display, final_info_json, status_msg

        except Exception as e:
            error_message = f"Critical UI wrapper error: {str(e)}"
            logger.error(error_message, exc_info=True)
            return image_pil, None, None, {"error": error_message, "traceback": traceback.format_exc()}, error_message

    def process_image(self, image: Image.Image, prompt: str, strength: float, **kwargs) -> Dict[str, Any]:
        """Orchestrate analysis and pipeline execution."""
        start_time = time.time()
        logger.info(f"process_image started.")
        # Initialize with received values, in case pipeline fails early
        results = { 
            'result_image': image, 
            'mask_image': None, 
            'operation_info': {}, 
            'context_info': {},
            'status_message': "Processing started.", 
            'success': False 
        }
        try:
            # Analiză operație și context imagine
            operation_details = self.op_analyzer.analyze_operation(prompt)
            results['operation_info'] = operation_details
            
            image_context = self.img_analyzer.analyze_image_context(image)
            results['context_info'] = image_context
            
            self._log_image_context(image_context)
            
            if "error" in image_context: 
                logger.error(f"Context analysis error: {image_context['error']}")

            # Determină pipeline-ul potrivit
            op_type = operation_details.get('type', 'general')
            target_obj = operation_details.get('target_object', '')
            pipeline = self.pipeline_manager.get_pipeline_for_operation(op_type, target_obj)
            
            if not pipeline:
                 msg = f"No pipeline for op '{op_type}'"
                 logger.error(msg)
                 results['status_message'] = msg
                 return results

            # Execută pipeline-ul
            logger.info(f"Using pipeline: {pipeline.__class__.__name__}")
            pipeline_kwargs = { 
                "image": image, 
                "prompt": prompt, 
                "strength": strength,
                "operation": operation_details, 
                "image_context": image_context,
                "progress_callback": lambda p, desc: logger.debug(f"Pipeline: {p*100:.0f}% - {desc}"), 
                **kwargs 
            }
            
            # Apel pipeline și procesare rezultat
            pipeline_result_dict = pipeline.process(**pipeline_kwargs)

            if isinstance(pipeline_result_dict, dict):
                results.update(pipeline_result_dict)
            else:
                 logger.error(f"Pipeline returned non-dict: {type(pipeline_result_dict)}")
                 results['message'] = "Pipeline returned unexpected data."
                 results['success'] = False
                 if isinstance(pipeline_result_dict, Image.Image): 
                     results['result_image'] = pipeline_result_dict

            # Standardize output formats for UI (PIL)
            if results.get('result_image') and isinstance(results['result_image'], np.ndarray):
                 results['result_image'] = self._convert_cv2_to_pil(results['result_image'])
            if results.get('mask_image') and isinstance(results['mask_image'], np.ndarray):
                 results['mask_image'] = self._ensure_pil_mask_ui(results['mask_image'])

            # Finalizare rezultat
            proc_time = time.time() - start_time
            results['status_message'] = results.get('message', "Pipeline finished.") + f" (Pipeline: {proc_time:.2f}s)"
            results['success'] = results.get('success', False)
            logger.info(f"process_image finished. Success: {results['success']}. {results['status_message']}")

        except Exception as e:
            msg = f"Orchestration error in process_image: {str(e)}"
            logger.error(msg, exc_info=True)
            results['status_message'] = msg
            results['success'] = False
            results['operation_info']['error'] = str(e)
            results['operation_info']['traceback'] = traceback.format_exc()
        
        return results

    def _log_image_context(self, ctx):
        """Helper for context logging (compact format)."""
        if not isinstance(ctx, dict): 
            logger.info(f"Image Context: {ctx}")
            return
            
        try:
             logger.info("--- Image Context ---")
             logger.info(f"  Time: {ctx.get('analysis_time_sec','?')}s")
             si=ctx.get('scene_info',{}); logger.info(f"  Scene: {si.get('primary_scene_tag_ml','?')} ({len(si.get('detected_objects',[]))} obj)")
             li=ctx.get('lighting_conditions',{}); logger.info(f"  Light: B={li.get('brightness_heuristic','?')}, C={li.get('contrast_heuristic','?')}, T={li.get('temperature_heuristic','?')}, H={li.get('highlights_pct','?')}%, S={li.get('shadows_pct','?')}%.")
             di=ctx.get('spatial_info',{}); logger.info(f"  Depth: {'Yes' if di.get('depth_map_available') else 'No'}, Char: {di.get('depth_characteristics','?')}")
             logger.info(f"  Desc: {ctx.get('full_description_heuristic','?')}")
             logger.info("--- End Context ---")
        except Exception as e_log: 
            logger.warning(f"Error logging context: {e_log}")

    def _convert_cv2_to_pil(self, img_np):
         """Convert NumPy BGR to PIL RGB."""
         if img_np is None: 
             return None
         try: 
             return Image.fromarray(cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB))
         except Exception as e: 
             logger.error(f"CV2 to PIL conversion failed: {e}")
             return None

    def _ensure_pil_mask_ui(self, mask):
         """Ensure PIL 'L' mask for UI."""
         if mask is None: 
             return None
         try: # Conversion/validation logic from BasePipeline._ensure_pil_mask
            if isinstance(mask, Image.Image): 
                return mask.convert("L") if mask.mode != "L" else mask
            elif isinstance(mask, np.ndarray):
                 if mask.ndim == 3 and mask.shape[2]==1: 
                     mask = mask.squeeze(axis=-1)
                 if mask.ndim != 2: 
                     return None
                 if mask.dtype != np.uint8: 
                     mask = np.clip(mask*255 if mask.max()<=1.0 else mask, 0, 255).astype(np.uint8)
                 return Image.fromarray(mask, 'L')
            else: 
                return None
         except Exception as e: 
             logger.error(f"Ensure PIL mask UI error: {e}")
             return None

    def launch(self, **kwargs):
        """Launch the Gradio interface."""
        launch_kwargs = { 
            "server_name": "0.0.0.0", 
            "server_port": 7860, 
            "share": False, 
            "debug": False, 
            **kwargs 
        }
        logger.info(f"Launching Gradio interface: {launch_kwargs}")
        if hasattr(self, 'app') and self.app:
            try:
                self.app.queue()  # Activare coadă pentru procesare
                self.app.launch(**launch_kwargs)
            except Exception as e: 
                logger.critical(f"Gradio launch failed: {e}", exc_info=True)
                sys.exit(1)
        else: 
            logger.error("Gradio app object not found.")


def main():
    """Main function."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(name)s:%(message)s")
    import argparse
    parser = argparse.ArgumentParser(description="FusionFrame 2.0 UI")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--low-vram", action="store_true")
    args = parser.parse_args()
    
    log_level = logging.DEBUG if args.debug else logging.INFO
    AppConfig.setup_logging(level=log_level)
    
    if args.low_vram: 
        logger.info("Low VRAM mode requested.")
        AppConfig.LOW_VRAM_MODE = True
    
    logger.info("Starting FusionFrame UI...")
    AppConfig.ensure_dirs()
    
    try: 
        ui = FusionFrameUI()
        ui.launch(server_name=args.host, server_port=args.port, share=args.share, debug=args.debug)
    except Exception as e: 
        logger.critical(f"Init/Launch failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__": 
    main()