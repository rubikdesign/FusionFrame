import os
import torch
import gradio as gr
from pathlib import Path
from tqdm import tqdm
from diffusers import StableDiffusionXLImg2ImgPipeline, AutoencoderKL, DPMSolverMultistepScheduler
from diffusers.utils import load_image
from huggingface_hub import snapshot_download
import PIL
from PIL import Image
import numpy as np
import glob
from safetensors.torch import load_file

class FusionFrame:
    def __init__(self):
        self.model_id = "stabilityai/stable-diffusion-xl-refiner-1.0"  # Un model foarte bun pentru realism
        self.cache_dir = os.path.join(os.path.expanduser("~"), ".fusionframe")
        self.loras_dir = os.path.join(os.getcwd(), "Loras")  # Default LoRAs directory in root
        self.available_models = {
            "stabilityai/stable-diffusion-xl-refiner-1.0": "SDXL Refiner 1.0 (Default)",
            "runwayml/stable-diffusion-v1-5": "Stable Diffusion 1.5",
            "SG161222/Realistic_Vision_V5.1_noVAE": "Realistic Vision 5.1",
            "emilianJR/epiCRealism": "EpicRealism",
            "Lykon/dreamshaper-xl-1-0": "DreamShaper XL",
            "segmind/SSD-1B": "Segmind SSD-1B",
        }
        self.available_samplers = {
            "DPM++ 2M Karras": DPMSolverMultistepScheduler.from_config,
            "Euler a": None,  # Will be implemented as needed
            "DDIM": None,     # Will be implemented as needed
        }
        
        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Create LoRAs directory if it doesn't exist
        os.makedirs(self.loras_dir, exist_ok=True)
        
        # Initialize pipeline to None, will load on demand
        self.pipe = None
        self.current_model = self.model_id
        self.current_sampler = "DPM++ 2M Karras"
        
        # LoRA related variables
        self.available_loras = self.scan_loras()
        self.active_loras = []
        
        # Activăm automatic xformers pentru optimizare memorie dacă e disponibil
        self.use_xformers = True

    def scan_loras(self):
        """Scan the LoRAs directory for available .safetensors files"""
        lora_files = glob.glob(os.path.join(self.loras_dir, "*.safetensors"))
        loras = {}
        
        for lora_file in lora_files:
            lora_name = os.path.basename(lora_file).replace(".safetensors", "")
            loras[lora_name] = lora_file
            
        return loras
    
    def load_model(self, model_id=None):
        """Load the selected model on demand"""
        if model_id:
            self.current_model = model_id
        
        print(f"Loading model: {self.current_model}")
        
        # Download and load the pipeline
        try:
            # Verificăm dacă modelul este SDXL sau SD standard
            if "xl" in self.current_model.lower():
                # Modelele SDXL
                self.pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                    self.current_model,
                    torch_dtype=torch.float16,
                    use_safetensors=True,
                    variant="fp16",
                    cache_dir=self.cache_dir,
                )
            else:
                # Modelele SD standard
                from diffusers import StableDiffusionImg2ImgPipeline
                self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                    self.current_model,
                    torch_dtype=torch.float16,
                    use_safetensors=True,
                    variant="fp16",
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
                print("Utilizăm GPU pentru generare")
                
                # Activăm xformers pentru optimizare memorie dacă e disponibil
                if self.use_xformers:
                    try:
                        self.pipe.enable_xformers_memory_efficient_attention()
                        print("xFormers activat pentru optimizare memorie")
                    except Exception as e:
                        print(f"xFormers nu a putut fi activat: {e}")
            else:
                print("Nu există GPU disponibil, utilizăm CPU (va fi mult mai lent)")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            
            # Fallback la un model cunoscut dacă cel specificat nu poate fi încărcat
            if self.current_model != "runwayml/stable-diffusion-v1-5":
                print("Încercăm un model alternativ: Stable Diffusion 1.5")
                self.current_model = "runwayml/stable-diffusion-v1-5"
                return self.load_model()
            else:
                raise e
        
        return self.pipe
        
    def load_lora(self, lora_path, scale=0.7):
        """Load a LoRA into the pipeline"""
        if self.pipe is None:
            self.load_model()
            
        try:
            # Simplificăm metoda de încărcare a LoRA folosind metoda integrată în pipeline
            if hasattr(self.pipe, "load_lora_weights"):
                # Metoda oficială și recomandată
                self.pipe.load_lora_weights(lora_path, adapter_name=os.path.basename(lora_path))
                # Setăm weight-ul pentru LoRA
                if hasattr(self.pipe, "set_adapters_weights"):
                    self.pipe.set_adapters_weights({os.path.basename(lora_path): scale})
                print(f"LoRA loaded successfully: {os.path.basename(lora_path)} with scale {scale}")
                return True
            else:
                print(f"The current pipeline doesn't support loading LoRAs directly")
                return False
        except Exception as e:
            print(f"Error loading LoRA: {e}")
            return False

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
        else:
            # Asigurăm-ne că dimensiunile sunt multipli de 8 pentru modele SD
            width, height = pose_image.size
            new_width = (width // 8) * 8
            new_height = (height // 8) * 8
            if new_width != width or new_height != height:
                pose_image = pose_image.resize((new_width, new_height), Image.LANCZOS)
                
            width, height = reference_image.size
            new_width = (width // 8) * 8
            new_height = (height // 8) * 8
            if new_width != width or new_height != height:
                reference_image = reference_image.resize((new_width, new_height), Image.LANCZOS)
            
        return reference_image, pose_image

    def apply_active_loras(self):
        """Apply all active LoRAs to the model"""
        if not self.active_loras:
            return
            
        # Make sure model is loaded
        if self.pipe is None:
            self.load_model()
            
        # Apply each active LoRA
        for lora_name, is_active, weight in self.active_loras:
            if is_active and lora_name in self.available_loras and lora_name != "None":
                lora_path = self.available_loras[lora_name]
                self.load_lora(lora_path, scale=weight)
        
        return True
    
    def set_active_loras(self, lora_settings):
        """Update the list of active LoRAs"""
        self.active_loras = lora_settings
        return self.active_loras

    def blend_images(self, ref_img, pose_img, weight=0.3):
        """
        Blend two images together to help guide the generation process
        """
        # Ensure both images are the same size
        if ref_img.size != pose_img.size:
            ref_img = ref_img.resize(pose_img.size, Image.LANCZOS)

        # Convert to numpy arrays
        ref_array = np.array(ref_img).astype(float)
        pose_array = np.array(pose_img).astype(float)
        
        # Blend images
        blended_array = (1 - weight) * pose_array + weight * ref_array
        blended_array = np.clip(blended_array, 0, 255).astype(np.uint8)
        
        # Convert back to PIL image
        blended_img = Image.fromarray(blended_array)
        return blended_img
            
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
        active_loras=None,  # List of (lora_name, is_active, weight) tuples
        progress=gr.Progress(track_tqdm=True),
    ):
        """Generate a composite image based on reference and pose images"""
        # Set a random seed if not provided
        if seed == -1:
            seed = torch.randint(0, 2**32 - 1, (1,)).item()
        generator = torch.Generator("cuda" if torch.cuda.is_available() else "cpu").manual_seed(seed)
        
        # Load the model if not already loaded or if it has changed
        if self.pipe is None or self.current_model != self.model_id:
            self.load_model()
        
        # Update and apply active LoRAs if provided
        if active_loras:
            self.set_active_loras(active_loras)
            self.apply_active_loras()
        
        # Create target size if specified
        target_size = None
        if width and height:
            target_size = (width, height)
        
        # Preprocess images
        ref_img, pose_img = self.preprocess_images(reference_image, pose_image, target_size)
        
        # Blend the reference image and pose image as a startup image
        # Acest lucru ajută modelul să transfere caracteristicile persoanei din imaginea de referință
        blended_img = self.blend_images(ref_img, pose_img, weight=0.2)
        
        # Combine customization options with prompt
        full_prompt = prompt
        if attire_customization:
            full_prompt += f", {attire_customization}"
        if decor_customization:
            full_prompt += f", {decor_customization}"
            
        if not full_prompt:
            # Default prompt that preserves the reference image's characteristics in the pose
            full_prompt = "photo of the same person from the reference image, in this exact position, highly detailed, photorealistic"
        
        # Enhancing the prompt to better guide the model
        reference_description = "same face, same hair, same identity"
        enhanced_prompt = f"{full_prompt}, {reference_description}, 8k uhd, professional photo, detailed, high quality"
        
        # Prepare default negative prompt if not provided
        if not negative_prompt:
            negative_prompt = "deformed, ugly, bad proportions, bad anatomy, disfigured, poorly drawn, blurry, low quality, cartoon, anime, illustration, different person, wrong face"
        
        # Generate the image with progress tracking
        with tqdm(total=num_inference_steps, desc="Generating image") as progress_bar:
            def callback(step, timestep, latents):
                progress_bar.update(1)
                return
            
            # Pentru SDXL, folosim IP-Adapter pentru a îmbunătăți transferul de caracteristici
            # în cazul în care e disponibil
            use_ip_adapter = False
            
            if "xl" in self.current_model.lower() and hasattr(self.pipe, "set_ip_adapter_scale"):
                try:
                    # Încercăm să utilizăm IP-Adapter pentru a îmbunătăți transferul de caracteristici
                    from diffusers.models import IPAdapterModel
                    
                    ip_model = IPAdapterModel.from_pretrained(
                        "h94/IP-Adapter", 
                        subfolder="sdxl_models", 
                        torch_dtype=torch.float16,
                        cache_dir=self.cache_dir
                    )
                    
                    self.pipe.set_ip_adapter(ip_model)
                    self.pipe.set_ip_adapter_scale(0.5)
                    use_ip_adapter = True
                    print("IP-Adapter activat pentru un transfer mai bun de caracteristici")
                except Exception as e:
                    print(f"Nu s-a putut activa IP-Adapter: {e}")
            
            # Parametri diferiți în funcție de tipul de model
            kwargs = {
                "image": blended_img,  # Folosim imaginea combinată ca bază
                "prompt": enhanced_prompt,
                "negative_prompt": negative_prompt,
                "strength": strength,
                "guidance_scale": guidance_scale,
                "num_inference_steps": num_inference_steps,
                "generator": generator,
                "callback": callback,
                "callback_steps": 1,
            }
            
            # Pentru modelele XL adăugăm și parametrii specifici
            if "xl" in self.current_model.lower():
                if use_ip_adapter:
                    kwargs["ip_adapter_image"] = ref_img
                
            # Generăm imaginea finală
            result = self.pipe(**kwargs).images[0]
            
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
    
    # Function to collect active LoRA settings
    def collect_lora_settings(
        lora1_name, lora1_active, lora1_weight,
        lora2_name, lora2_active, lora2_weight,
        lora3_name, lora3_active, lora3_weight,
        lora4_name, lora4_active, lora4_weight,
        lora5_name, lora5_active, lora5_weight
    ):
        active_loras = []
        
        if lora1_name:
            active_loras.append((lora1_name, lora1_active, lora1_weight))
        if lora2_name:
            active_loras.append((lora2_name, lora2_active, lora2_weight))
        if lora3_name:
            active_loras.append((lora3_name, lora3_active, lora3_weight))
        if lora4_name:
            active_loras.append((lora4_name, lora4_active, lora4_weight))
        if lora5_name:
            active_loras.append((lora5_name, lora5_active, lora5_weight))
            
        return active_loras
    
    # Function to rescan LoRAs directory
    def rescan_loras():
        fusion_frame.available_loras = fusion_frame.scan_loras()
        lora_names = list(fusion_frame.available_loras.keys())
        lora_choices = ["None"] + lora_names
        return [gr.Dropdown.update(choices=lora_choices)] * 5
    
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
                        value="SDXL Refiner 1.0 (Default)",
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
                    
                    # LoRA Settings
                    with gr.Accordion("LoRA Settings", open=False):
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
                    
                    download_model_button = gr.Button("Download Selected Model")
                
                generate_button = gr.Button("Generate Composite Image", variant="primary")
            
            with gr.Column():
                output_image = gr.Image(label="Generated Result")
                output_seed = gr.Number(label="Seed Used (for reproducibility)")
        
        # Set up events
        generate_button.click(
            fn=lambda *args: fusion_frame.generate_image(
                *args[:-15],  # Reference image through decor_customization
                collect_lora_settings(*args[-15:]),  # LoRA settings
            ),
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
                # LoRA settings
                lora1_name, lora1_active, lora1_weight,
                lora2_name, lora2_active, lora2_weight,
                lora3_name, lora3_active, lora3_weight,
                lora4_name, lora4_active, lora4_weight,
                lora5_name, lora5_active, lora5_weight,
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
        
        # Setup LoRA directory update and rescan
        lora_dir.change(
            fn=lambda x: setattr(fusion_frame, "loras_dir", x) or x,
            inputs=[lora_dir],
            outputs=[lora_dir],
        )
        
        rescan_loras_button.click(
            fn=rescan_loras,
            inputs=[],
            outputs=[lora1_name, lora2_name, lora3_name, lora4_name, lora5_name],
        )
        
    return app

if __name__ == "__main__":
    app = build_gradio_interface()
    app.launch()
