"""
UI module for FusionFrame application.

This module provides a Gradio interface for the FusionFrame application,
with improved organization, error handling, and user feedback.
"""

import os
import logging
import gradio as gr
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
import numpy as np
from PIL import Image

# Import core functionality
from core import FusionFrame
import config
from utils import io_utils

# Set up logging
logger = logging.getLogger(__name__)

def build_gradio_interface():
    """
    Build the Gradio interface for FusionFrame application.
    
    Returns:
        gr.Blocks: Gradio Blocks interface
    """
    logger.info("Building Gradio interface")
    
    # Initialize FusionFrame
    fusion_frame = FusionFrame()
    
    # Function to collect active LoRA settings
    def collect_lora_settings(
        lora1_name, lora1_active, lora1_weight,
        lora2_name, lora2_active, lora2_weight,
        lora3_name, lora3_active, lora3_weight,
        lora4_name, lora4_active, lora4_weight,
        lora5_name, lora5_active, lora5_weight=0.75  # Adaugă valoare implicită
    ):
        """Collect LoRA settings from UI elements into a list of tuples."""
        active_loras = []
        
        if lora1_name and lora1_name != "None":
            active_loras.append((lora1_name, lora1_active, lora1_weight))
        if lora2_name and lora2_name != "None":
            active_loras.append((lora2_name, lora2_active, lora2_weight))
        if lora3_name and lora3_name != "None":
            active_loras.append((lora3_name, lora3_active, lora3_weight))
        if lora4_name and lora4_name != "None":
            active_loras.append((lora4_name, lora4_active, lora4_weight))
        if lora5_name and lora5_name != "None":
            active_loras.append((lora5_name, lora5_active, lora5_weight))
            
        return active_loras
        
    # Function to rescan LoRAs directory
    def rescan_loras():
        """Rescan LoRAs directory and update dropdowns."""
        fusion_frame.available_loras = fusion_frame.rescan_loras()
        lora_names = list(fusion_frame.available_loras.keys())
        lora_choices = ["None"] + lora_names
        return [gr.Dropdown.update(choices=lora_choices)] * 5
    
    # Function to get model ID from dropdown
    def get_model_id(model_name):
        """Get model ID from display name and update current model."""
        model_id = fusion_frame.get_model_id_from_name(model_name)
        logger.info(f"Changing model to: {model_id}")
        
        # Reset pipeline to force reloading
        fusion_frame.pipe = None
        fusion_frame.current_model_id = model_id
        
        return f"Model changed to: {model_name}"
    
    # Function to download model
    def download_selected_model(model_name):
        """Download and load a specific model."""
        model_id = fusion_frame.get_model_id_from_name(model_name)
        
        try:
            result = fusion_frame.download_model(model_id)
            
            # Pre-load the model to check functionality
            fusion_frame.load_model(model_id)
            
            return f"Model downloaded and loaded successfully: {model_name}"
        except Exception as e:
            error_msg = f"Error downloading model: {e}"
            logger.error(error_msg)
            return error_msg
    
    # Function to open outputs folder
    def open_outputs_folder():
        """Open the outputs folder in system file explorer."""
        return io_utils.open_folder(fusion_frame.outputs_dir)
    
    # Function to validate inputs before generation
    def validate_inputs(reference_image, pose_image):
        """Validate input images before generation."""
        if reference_image is None:
            return False, "Reference image is required"
        if pose_image is None:
            return False, "Pose image is required"
        return True, "Inputs validated"
    
    # Build the interface
    with gr.Blocks(title="FusionFrame App", css=".container{min-height: 75vh;}") as app:
        gr.Markdown("# FusionFrame App")
        gr.Markdown("Upload a reference image (person) and a pose image (position/scene) to generate a composite.")
        
        # Main error display
        error_box = gr.Textbox(label="Status", visible=False)
        
        with gr.Tabs() as tabs:
            # Basic tab - essential controls
            with gr.TabItem("Basic"):
                with gr.Row():
                    with gr.Column():
                        reference_image = gr.Image(label="Reference Image (Person)", type="numpy")
                        prompt = gr.Textbox(label="Additional Prompt (Optional)", 
                                          placeholder="Optional: Add details to guide the generation")
                    
                    with gr.Column():
                        pose_image = gr.Image(label="Pose Image (Position/Scene)", type="numpy")
                        negative_prompt = gr.Textbox(label="Negative Prompt (Optional)", 
                                               placeholder="Optional: Features to avoid in the generated image")
                
                with gr.Row():
                    num_images = gr.Slider(minimum=1, maximum=10, value=1, step=1, 
                                         label="Number of Images", 
                                         info="Generate multiple variations with different seeds")
                    seed = gr.Number(-1, label="Seed (-1 for random)", 
                                   info="Set a specific seed for reproducible results")
                
                with gr.Row():
                    generate_button = gr.Button("Generate Composite Image", variant="primary", size="lg")
            
            # Model settings tab
            with gr.TabItem("Model Settings"):
                with gr.Row():
                    model_dropdown = gr.Dropdown(
                        choices=fusion_frame.get_available_models(),
                        value="SDXL Refiner 1.0 (Default)",
                        label="Model"
                    )
                    
                    sampler_dropdown = gr.Dropdown(
                        choices=list(fusion_frame.available_samplers.keys()),
                        value="DPM++ 2M Karras",
                        label="Sampler"
                    )
                    
                    download_model_button = gr.Button("Download Selected Model")
                
                download_status = gr.Textbox(label="Download Status", visible=True)
                
                with gr.Row():
                    steps = gr.Slider(10, 150, 30, step=1, label="Inference Steps", 
                                    info="More steps = higher quality but slower")
                    guidance_scale = gr.Slider(1.0, 15.0, 7.5, step=0.5, label="Guidance Scale", 
                                            info="Higher = follow prompt more closely")
                    strength = gr.Slider(0.1, 1.0, 0.75, step=0.05, label="Strength", 
                                       info="How much to transform the input image (lower = preserve more)")
            
            # Face & Blend settings tab  
            with gr.TabItem("Face & Blend"):
                with gr.Row():
                    face_enhance = gr.Checkbox(label="Enable Face Enhancement", value=True, 
                                             info="Detect and preserve facial features")
                    face_strength = gr.Slider(0.1, 1.0, 0.8, step=0.05, label="Face Preservation Strength", 
                                            info="How strongly to preserve facial features")
                
                with gr.Row():
                    enable_selective_face = gr.Checkbox(label="Enable Selective Face Transfer", value=True, 
                                                      info="Transfer specific facial features (eyes, nose, mouth)")
                    face_transfer_blend = gr.Slider(0.1, 1.0, 0.85, step=0.05, label="Face Transfer Blend", 
                                                  info="Blending strength for transferred features")
                
                # Size controls
                with gr.Row():
                    keep_original_size = gr.Checkbox(label="Keep Original Size", value=True, 
                                                   info="Use original image dimensions")
                    width = gr.Slider(512, 2048, 1024, step=64, label="Width", 
                                    interactive=False, info="Output width if not keeping original size")
                    height = gr.Slider(512, 2048, 1024, step=64, label="Height", 
                                     interactive=False, info="Output height if not keeping original size")
                
                # Customization options
                with gr.Row():
                    attire_customization = gr.Textbox(label="Attire Customization", 
                                                   placeholder="E.g., wearing a red dress, formal suit")
                    decor_customization = gr.Textbox(label="Scene/Décor Customization", 
                                                  placeholder="E.g., beach background, sunny day")
            
            # Advanced generation tab
            with gr.TabItem("ControlNet & Refiner"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### ControlNet Settings")
                        enable_cn_pose = gr.Checkbox(label="Enable ControlNet Pose", value=False, 
                                                  info="Use ControlNet for better pose guidance")
                        cn_strength = gr.Slider(
                            minimum=0.0, maximum=2.0, value=1.0, step=0.05,
                            label="ControlNet Strength",
                            info="How much the pose controls the generation"
                        )
                    
                    with gr.Column():
                        gr.Markdown("### Refiner Settings")
                        two_stage_chk = gr.Checkbox(label="Enable 2-stage Refiner", value=False, 
                                                 info="Apply a second refinement pass")
                        refiner_dropdown = gr.Dropdown(
                            choices=fusion_frame.get_available_models(),
                            value="SDXL Refiner 1.0 (Default)",
                            label="Refiner Model"
                        )
                        refiner_strength = gr.Slider(
                            minimum=0.1, maximum=0.7, value=0.3, step=0.05,
                            label="Refiner Strength",
                            info="Lower values preserve more details"
                        )
            
            # LoRA tab
            with gr.TabItem("LoRA Settings"):
                gr.Markdown("Add up to 5 LoRAs to modify the generation results")
                
                # LoRA directory settings
                with gr.Row():
                    lora_dir = gr.Textbox(label="LoRAs Directory", value=fusion_frame.loras_dir)
                    rescan_loras_button = gr.Button("Rescan LoRAs")
                
                # Get available LoRAs
                lora_names = list(fusion_frame.available_loras.keys())
                lora_choices = ["None"] + lora_names
                
                # LoRA 1
                with gr.Row():
                    lora1_active = gr.Checkbox(label="Active", value=False)
                    lora1_name = gr.Dropdown(choices=lora_choices, value="None", label="LoRA 1")
                    lora1_weight = gr.Slider(0.1, 1.0, 0.75, step=0.05, label="Weight")
                
                # LoRA 2
                with gr.Row():
                    lora2_active = gr.Checkbox(label="Active", value=False)
                    lora2_name = gr.Dropdown(choices=lora_choices, value="None", label="LoRA 2")
                    lora2_weight = gr.Slider(0.1, 1.0, 0.75, step=0.05, label="Weight")
                
                # LoRA 3
                with gr.Row():
                    lora3_active = gr.Checkbox(label="Active", value=False)
                    lora3_name = gr.Dropdown(choices=lora_choices, value="None", label="LoRA 3")
                    lora3_weight = gr.Slider(0.1, 1.0, 0.75, step=0.05, label="Weight")
                
                # LoRA 4
                with gr.Row():
                    lora4_active = gr.Checkbox(label="Active", value=False)
                    lora4_name = gr.Dropdown(choices=lora_choices, value="None", label="LoRA 4")
                    lora4_weight = gr.Slider(0.1, 1.0, 0.75, step=0.05, label="Weight")
                
                # LoRA 5
                with gr.Row():
                    lora5_active = gr.Checkbox(label="Active", value=False)
                    lora5_name = gr.Dropdown(choices=lora_choices, value="None", label="LoRA 5")
                    lora5_weight = gr.Slider(0.1, 1.0, 0.75, step=0.05, label="Weight")
            
            # Save settings tab
            with gr.TabItem("Save Settings"):
                with gr.Row():
                    auto_save_checkbox = gr.Checkbox(label="Auto Save Images", value=True)
                    save_format = gr.Dropdown(
                        choices=["png", "jpg", "webp"],
                        value="png",
                        label="Save Format"
                    )
                
                with gr.Row():
                    open_folder_button = gr.Button("Open Outputs Folder")
                    save_status = gr.Textbox(label="Save Status", value=f"Images will be saved to: {fusion_frame.outputs_dir}")
        
        # Output section
        with gr.Row():
            with gr.Column():
                output_image = gr.Image(label="Generated Result")
                output_seed = gr.Textbox(label="Seeds Used (for reproducibility)")
                saved_path = gr.Textbox(label="Saved Image Paths", visible=True)
                
                # Gallery for multiple images (shown only when multiple images are generated)
                output_gallery = gr.Gallery(label="All Generated Images", 
                                          visible=False, columns=2, rows=2)
        
        # Set up events
        # Main generation function
        def generate_with_error_handling(*args):
            try:
                # Validate inputs
                valid, message = validate_inputs(args[0], args[1])
                if not valid:
                    return gr.update(visible=True, value=message), None, None, None
                
                # Run generation
                first_image, seeds, save_paths = fusion_frame.generate_image(
                    *args[:2],  # reference_image, pose_image
                    prompt=args[2], 
                    negative_prompt=args[3],
                    strength=args[5],
                    guidance_scale=args[6],
                    num_inference_steps=args[7],
                    seed=args[8],
                    width=args[9] if not args[12] else None,
                    height=args[10] if not args[12] else None,
                    keep_original_size=args[12],
                    num_images=args[4],
                    attire_customization=args[13],
                    decor_customization=args[14],
                    face_enhancement=args[15],
                    enable_two_stage=args[18],
                    refiner_model_name=args[19],
                    refiner_strength=args[20],
                    enable_cn_pose=args[16],
                    cn_strength=args[17],
                    enable_selective_face=args[21],
                    active_loras=collect_lora_settings(*args[22:]),
                    progress_callback=lambda p: gr.Progress(p)
                )
                
                # Format results
                seeds_str = ", ".join(map(str, seeds))
                paths_str = "\n".join(save_paths) if save_paths else ""
                
                # Hide error box on success
                return gr.update(visible=False), first_image, seeds_str, paths_str
            except Exception as e:
                error_message = f"Error during generation: {str(e)}"
                logger.error(error_message)
                return gr.update(visible=True, value=error_message), None, None, None
                
        # Connect the generate button to our handler with all inputs
        generate_button.click(
            fn=generate_with_error_handling,
            inputs=[
                reference_image, pose_image,
                prompt, negative_prompt, num_images,
                strength, guidance_scale, steps, seed, 
                width, height, keep_original_size,
                attire_customization, decor_customization,
                face_enhance, enable_cn_pose, cn_strength,
                two_stage_chk, refiner_dropdown, refiner_strength,
                enable_selective_face,
                # LoRA settings
                lora1_name, lora1_active, lora1_weight,
                lora2_name, lora2_active, lora2_weight,
                lora3_name, lora3_active, lora3_weight,
                lora4_name, lora4_active, lora4_weight,
                lora5_name, lora5_active, lora5_weight,
            ],
            outputs=[error_box, output_image, output_seed, saved_path],
        )
        
        # Update visibility of gallery based on num_images
        num_images.change(
            fn=lambda x: gr.update(visible=(x > 1)),
            inputs=[num_images],
            outputs=[output_gallery],
        )
        
        # Change face enhancement settings
        face_enhance.change(
            fn=fusion_frame.set_face_enhancement,
            inputs=[face_enhance],
            outputs=[],
        )
        
        face_strength.change(
            fn=fusion_frame.set_preserve_face_strength,
            inputs=[face_strength],
            outputs=[],
        )

        # Make ControlNet strength visible only when ControlNet is enabled
        enable_cn_pose.change(
            fn=lambda flag: gr.update(visible=flag),
            inputs=enable_cn_pose,
            outputs=cn_strength,
            queue=False
        )

        # Make refiner controls visible only when two-stage is enabled
        two_stage_chk.change(
            fn=lambda x: [gr.update(visible=x), gr.update(visible=x)],
            inputs=[two_stage_chk],
            outputs=[refiner_dropdown, refiner_strength],
        )

        # Update refiner strength when slider changes
        refiner_strength.change(
            fn=fusion_frame.set_refiner_strength,
            inputs=[refiner_strength],
            outputs=[],
        )

        # Set model when dropdown changes
        model_dropdown.change(
            fn=get_model_id,
            inputs=[model_dropdown],
            outputs=[download_status],
        )
        
        # Change sampler when dropdown changes
        sampler_dropdown.change(
            fn=lambda x: setattr(fusion_frame, "current_sampler", x) or x,
            inputs=[sampler_dropdown],
            outputs=[sampler_dropdown],  # Output to itself to indicate change
        )
        
        # Download model button
        download_model_button.click(
            fn=download_selected_model,
            inputs=[model_dropdown],
            outputs=[download_status],
        )
        
        # Update LoRA directory
        lora_dir.change(
            fn=lambda x: setattr(fusion_frame, "loras_dir", x) or x,
            inputs=[lora_dir],
            outputs=[lora_dir],
        )
        
        # Rescan LoRAs button
        rescan_loras_button.click(
            fn=rescan_loras,
            inputs=[],
            outputs=[lora1_name, lora2_name, lora3_name, lora4_name, lora5_name],
        )
        
        # Setup auto-save controls
        auto_save_checkbox.change(
            fn=fusion_frame.toggle_auto_save,
            inputs=[auto_save_checkbox],
            outputs=[save_status],
        )
        
        # Change save format
        save_format.change(
            fn=fusion_frame.set_save_format,
            inputs=[save_format],
            outputs=[save_status],
        )
        
        # Open outputs folder button
        open_folder_button.click(
            fn=open_outputs_folder,
            inputs=[],
            outputs=[save_status],
        )
        
        # Enable/disable width/height controls based on keep_original_size
        keep_original_size.change(
            fn=lambda x: [gr.update(interactive=not x), gr.update(interactive=not x)],
            inputs=[keep_original_size],
            outputs=[width, height],
        )
        
    return app

# Main entry point
if __name__ == "__main__":
    import os
    import sys
    
    # Add the current directory to the path so imports work correctly
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.append(current_dir)
    
    # Create the Gradio app
    app = build_gradio_interface()
    
    # Launch the app
    app.launch(share=True)