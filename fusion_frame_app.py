import os
import torch
import gradio as gr
from pathlib import Path
from tqdm import tqdm
from diffusers import AutoPipelineForImage2Image, DPMSolverMultistepScheduler
from diffusers.utils import load_image
from huggingface_hub import snapshot_download
import PIL
from PIL import Image
import numpy as np

class FusionFrame:
    def __init__(self):
        self.model_id = "RunDiffusion/Juggernaut-XL-v9"
        self.cache_dir = os.path.join(os.path.expanduser("~"), ".fusionframe")
        self.available_models = {
            "RunDiffusion/Juggernaut-XL-v9": "Juggernaut XL v9 (Default)",
            "stabilityai/stable-diffusion-xl-base-1.0": "Stable Diffusion XL",
            "prompthero/openjourney-v4": "OpenJourney v4",
        }
        self.available_samplers = {
            "DPM++ 2M Karras": DPMSolverMultistepScheduler.from_config,
            "Euler": None,  # Will be implemented as needed
            "Euler a": None,  # Will be implemented as needed
        }
        
        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Initialize pipeline to None, will load on demand
        self.pipe = None
        self.current_model = self.model_id
        self.current_sampler = "DPM++ 2M Karras"

    def load_model(self, model_id=None):
        """Load the selected model on demand"""
        if model_id:
            self.current_model = model_id
        
        print(f"Loading model: {self.current_model}")
        
        # Download and load the pipeline
        self.pipe = AutoPipelineForImage2Image.from_pretrained(
            self.current_model,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
            cache_dir=self.cache_dir,
        )
        
        # Set the scheduler based on the current_sampler
        if self.current_sampler == "DPM++ 2M Karras":
            self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                self.pipe.scheduler.config, 
                algorithm_type="dpmsolver++", 
                use_karras_sigmas=True
            )
            
        # Move to GPU if available
        if torch.cuda.is_available():
            self.pipe = self.pipe.to("cuda")
        
        return self.pipe

    def preprocess_images(self, reference_image, pose_image, target_size=None):
        """Preprocess the input images"""
        if reference_image is None or pose_image is None:
            raise ValueError("Both reference and pose images are required")
            
        # Convert to PIL if needed
        if not isinstance(reference_image, PIL.Image.Image):
            reference_image = Image.fromarray(reference_image)
        if not isinstance(pose_image, PIL.Image.Image):
            pose_image = Image.fromarray(pose_image)
            
        # Resize if target_size is provided
        if target_size:
            width, height = target_size
            reference_image = reference_image.resize((width, height), Image.LANCZOS)
            pose_image = pose_image.resize((width, height), Image.LANCZOS)
            
        return reference_image, pose_image

    def generate_image(
        self,
        reference_image,
        pose_image,
        prompt="",
        negative_prompt="",
        strength=0.75,
        guidance_scale=7.5,
        num_inference_steps=30,
        seed=-1,
        width=None,
        height=None,
        attire_customization="",
        decor_customization="",
        progress=gr.Progress(track_tqdm=True),
    ):
        """Generate a composite image based on reference and pose images"""
        # Set a random seed if not provided
        if seed == -1:
            seed = torch.randint(0, 2**32 - 1, (1,)).item()
        generator = torch.Generator("cuda").manual_seed(seed)
        
        # Load the model if not already loaded or if it has changed
        if self.pipe is None or self.current_model != self.model_id:
            self.load_model()
        
        # Create target size if specified
        target_size = None
        if width and height:
            target_size = (width, height)
        
        # Preprocess images
        ref_img, pose_img = self.preprocess_images(reference_image, pose_image, target_size)
        
        # Combine customization options with prompt
        full_prompt = prompt
        if attire_customization:
            full_prompt += f", {attire_customization}"
        if decor_customization:
            full_prompt += f", {decor_customization}"
            
        if not full_prompt:
            # Default prompt that preserves the reference image's characteristics in the pose
            full_prompt = "person from the reference image in the position of the second image, highly detailed, photorealistic"
        
        # Prepare default negative prompt if not provided
        if not negative_prompt:
            negative_prompt = "deformed, ugly, bad proportions, bad anatomy, disfigured, poorly drawn, blurry, low quality"
        
        # Generate the image with progress tracking
        with tqdm(total=num_inference_steps, desc="Generating image") as progress_bar:
            def callback(step, timestep, latents):
                progress_bar.update(1)
                return
            
            result = self.pipe(
                image=pose_img,
                prompt=full_prompt,
                negative_prompt=negative_prompt,
                strength=strength,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                generator=generator,
                callback=callback,
                callback_steps=1,
                cross_attention_kwargs={"reference_image": ref_img},
            ).images[0]
            
        return result, seed

    def list_available_models(self):
        """Return a list of available models"""
        return list(self.available_models.values())
    
    def get_model_id_from_name(self, model_name):
        """Get the model ID from the display name"""
        for model_id, name in self.available_models.items():
            if name == model_name:
                return model_id
        return self.model_id  # Return default if not found
    
    def download_model(self, model_id):
        """Download a specific model to the cache directory"""
        snapshot_download(repo_id=model_id, cache_dir=self.cache_dir)
        return f"Downloaded model: {model_id}"


def build_gradio_interface():
    fusion_frame = FusionFrame()
    
    with gr.Blocks(title="FusionFrame App") as app:
        gr.Markdown("# FusionFrame App")
        gr.Markdown("Upload a reference image and a pose image to generate a composite.")
        
        with gr.Row():
            with gr.Column():
                reference_image = gr.Image(label="Reference Image (Person)", type="numpy")
                pose_image = gr.Image(label="Pose Image (Position/Scene)", type="numpy")
                
                with gr.Accordion("Advanced Settings", open=False):
                    width = gr.Slider(512, 1024, 768, step=64, label="Width")
                    height = gr.Slider(512, 1024, 768, step=64, label="Height")
                    
                    prompt = gr.Textbox(label="Additional Prompt", placeholder="Optional: Add details to guide the generation")
                    negative_prompt = gr.Textbox(label="Negative Prompt", placeholder="Optional: Features to avoid in generated image")
                    
                    model_dropdown = gr.Dropdown(
                        choices=fusion_frame.list_available_models(),
                        value="Juggernaut XL v9 (Default)",
                        label="Model"
                    )
                    
                    sampler_dropdown = gr.Dropdown(
                        choices=list(fusion_frame.available_samplers.keys()),
                        value="DPM++ 2M Karras",
                        label="Sampler"
                    )
                    
                    with gr.Row():
                        strength = gr.Slider(0.1, 1.0, 0.75, step=0.05, label="Strength (how much to preserve pose)")
                        guidance_scale = gr.Slider(1.0, 15.0, 7.5, step=0.5, label="Guidance Scale")
                    
                    with gr.Row():
                        steps = gr.Slider(10, 100, 30, step=1, label="Inference Steps")
                        seed = gr.Number(-1, label="Seed (-1 for random)")
                    
                    with gr.Accordion("Customization Options", open=False):
                        attire_customization = gr.Textbox(label="Attire Customization", placeholder="E.g., wearing a red dress, formal suit")
                        decor_customization = gr.Textbox(label="Scene/Décor Customization", placeholder="E.g., beach background, sunny day")
                    
                    download_model_button = gr.Button("Download Selected Model")
                
                generate_button = gr.Button("Generate Composite Image", variant="primary")
            
            with gr.Column():
                output_image = gr.Image(label="Generated Result")
                output_seed = gr.Number(label="Seed Used (for reproducibility)")
        
        # Set up events
        generate_button.click(
            fn=fusion_frame.generate_image,
            inputs=[
                reference_image,
                pose_image,
                prompt,
                negative_prompt,
                strength,
                guidance_scale,
                steps,
                seed,
                width,
                height,
                attire_customization,
                decor_customization,
            ],
            outputs=[output_image, output_seed],
        )
        
        model_dropdown.change(
            fn=lambda x: fusion_frame.get_model_id_from_name(x),
            inputs=[model_dropdown],
            outputs=[],  # No direct output, just updates the fusion_frame.current_model
        )
        
        sampler_dropdown.change(
            fn=lambda x: setattr(fusion_frame, "current_sampler", x) or x,
            inputs=[sampler_dropdown],
            outputs=[sampler_dropdown],  # Output to itself to indicate change
        )
        
        download_model_button.click(
            fn=fusion_frame.download_model,
            inputs=[lambda: fusion_frame.get_model_id_from_name(model_dropdown.value)],
            outputs=[gr.Textbox(label="Download Status")],
        )
        
    return app

if __name__ == "__main__":
    app = build_gradio_interface()
    app.launch()
