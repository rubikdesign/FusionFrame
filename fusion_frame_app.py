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
import cv2
from safetensors.torch import load_file
import datetime
import uuid
import time
try:
    import face_recognition  # Pentru detecția facială avansată
except ImportError:
    print("face_recognition not installed. Face alignment will not be available.")
    print("You can install it with: pip install face_recognition")
    face_recognition = None

class FusionFrame:
    def __init__(self):
        self.model_id = "stabilityai/stable-diffusion-xl-refiner-1.0"  # Un model foarte bun pentru realism
        self.cache_dir = os.path.join(os.path.expanduser("~"), ".fusionframe")
        self.loras_dir = os.path.join(os.getcwd(), "Loras")  # Default LoRAs directory in root
        self.outputs_dir = os.path.join(os.getcwd(), "outputs")  # Director pentru salvarea imaginilor generate
        
        self.available_models = {
            "stabilityai/stable-diffusion-xl-refiner-1.0": "SDXL Refiner 1.0 (Default)",
            "runwayml/stable-diffusion-v1-5": "Stable Diffusion 1.5",
            "SG161222/Realistic_Vision_V5.1_noVAE": "Realistic Vision 5.1",
            "emilianJR/epiCRealism": "EpicRealism",
            "Lykon/dreamshaper-xl-1-0": "DreamShaper XL",
            "segmind/SSD-1B": "Segmind SSD-1B",
            "gsdf/Counterfeit-V2.5": "Counterfeit V2.5 (Realist)",
            "digiplay/AbsoluteReality_v1.8.1": "Absolute Reality 1.8.1",
        }
        self.available_samplers = {
            "DPM++ 2M Karras": DPMSolverMultistepScheduler.from_config,
            "Euler a": None,  # Will be implemented as needed
            "DDIM": None,     # Will be implemented as needed
        }
        
        # Create required directories if they don't exist
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(self.loras_dir, exist_ok=True)
        os.makedirs(self.outputs_dir, exist_ok=True)
        
        # Initialize pipeline to None, will load on demand
        self.pipe = None
        self.current_model = self.model_id
        self.current_sampler = "DPM++ 2M Karras"
        
        # LoRA related variables
        self.available_loras = self.scan_loras()
        self.active_loras = []
        
        # Activăm automatic xformers pentru optimizare memorie dacă e disponibil
        self.use_xformers = True
        
        # Setări avansate
        self.auto_save = True
        self.save_format = "png"  # Format de salvare (png, jpg)
        self.face_enhancement = True  # Îmbunătățire facială
        self.face_alignment_weight = 0.8  # Pondere pentru alinierea facială (0-1)
        self.blend_method = "adaptive"  # Metoda de blending (simple, adaptive, mask)
        self.preserve_face_strength = 0.8  # Cât de mult să păstrăm din fața de referință

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

    def detect_faces(self, image):
        """Detect faces in an image using face_recognition"""
        if face_recognition is None:
            return None
        
        # Convert PIL image to numpy array for face_recognition
        if isinstance(image, PIL.Image.Image):
            image_np = np.array(image)
        else:
            image_np = image
        
        # Find all face locations
        face_locations = face_recognition.face_locations(image_np, model="hog")
        
        if not face_locations:
            return None
            
        # Get face encodings
        face_encodings = face_recognition.face_encodings(image_np, face_locations)
        
        return {
            "locations": face_locations,
            "encodings": face_encodings
        }

    def find_best_face_match(self, reference_faces, pose_faces):
        """Find the best match between reference and pose faces"""
        if not reference_faces or not pose_faces:
            return None, None
        
        best_match_ref = 0  # Default to first face
        best_match_pose = 0
        best_match_score = 0
        
        for i, ref_encoding in enumerate(reference_faces["encodings"]):
            for j, pose_encoding in enumerate(pose_faces["encodings"]):
                # Compare face encodings
                match_score = 1 - face_recognition.face_distance([ref_encoding], pose_encoding)[0]
                
                if match_score > best_match_score:
                    best_match_score = match_score
                    best_match_ref = i
                    best_match_pose = j
        
        return reference_faces["locations"][best_match_ref], pose_faces["locations"][best_match_pose]

    def create_face_mask(self, image, face_location, expand_ratio=1.5):
        """Create a mask highlighting the face area"""
        if face_location is None:
            return None
            
        height, width = image.shape[:2] if isinstance(image, np.ndarray) else (image.height, image.width)
        mask = np.zeros((height, width), dtype=np.uint8)
        
        # Expand face location
        top, right, bottom, left = face_location
        center_y, center_x = (top + bottom) // 2, (left + right) // 2
        face_height = bottom - top
        face_width = right - left
        
        # Calculate expanded dimensions
        expanded_height = int(face_height * expand_ratio)
        expanded_width = int(face_width * expand_ratio)
        
        # Calculate new boundaries
        new_top = max(0, center_y - expanded_height // 2)
        new_bottom = min(height, center_y + expanded_height // 2)
        new_left = max(0, center_x - expanded_width // 2)
        new_right = min(width, center_x + expanded_width // 2)
        
        # Create elliptical mask instead of rectangular for more natural blending
        cv2.ellipse(
            mask,
            center=(center_x, center_y),
            axes=(expanded_width // 2, expanded_height // 2),
            angle=0, startAngle=0, endAngle=360,
            color=255, thickness=-1
        )
        
        # Add feathering for smoother edges
        mask = cv2.GaussianBlur(mask, (51, 51), 0)
        
        return mask

    def adaptive_blend_images(self, ref_img, pose_img, weight=0.3, face_mask=None):
        """
        Blend images with advanced face preservation
        """
        # Ensure both images are the same size
        if ref_img.size != pose_img.size:
            ref_img = ref_img.resize(pose_img.size, Image.LANCZOS)

        # Convert to numpy arrays
        ref_array = np.array(ref_img).astype(float)
        pose_array = np.array(pose_img).astype(float)
        
        # Prepare the output array
        result_array = pose_array.copy()
        
        if face_mask is not None and self.face_enhancement:
            # Convert mask to 3-channel float
            face_mask_float = face_mask.astype(float) / 255.0
            face_mask_3ch = np.stack([face_mask_float] * 3, axis=2)
            
            # High weight for face area, lower for the rest
            face_weight = self.preserve_face_strength
            body_weight = weight
            
            # Apply adaptive blending
            result_array = (1 - face_mask_3ch) * ((1 - body_weight) * pose_array + body_weight * ref_array) + \
                           face_mask_3ch * ((1 - face_weight) * pose_array + face_weight * ref_array)
        else:
            # Simple blending if no face mask
            result_array = (1 - weight) * pose_array + weight * ref_array
        
        # Ensure valid pixel values
        result_array = np.clip(result_array, 0, 255).astype(np.uint8)
        
        # Convert back to PIL image
        blended_img = Image.fromarray(result_array)
        return blended_img

    def preprocess_images(self, reference_image, pose_image, target_size=None, keep_original_size=True):
        """Preprocess the input images with improved handling"""
        if reference_image is None or pose_image is None:
            raise ValueError("Both reference and pose images are required")
            
        # Convert to PIL if needed
        if not isinstance(reference_image, PIL.Image.Image):
            reference_image = Image.fromarray(reference_image)
        if not isinstance(pose_image, PIL.Image.Image):
            pose_image = Image.fromarray(pose_image)
        
        # Handle sizing based on parameters
        if target_size and not keep_original_size:
            width, height = target_size
            reference_image = reference_image.resize((width, height), Image.LANCZOS)
            pose_image = pose_image.resize((width, height), Image.LANCZOS)
        elif keep_original_size:
            # Keep original size, but ensure dimensions are multiples of 8
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
    
    def save_image(self, image, seed, model_name=None, index=0, batch_count=1):
        """Save the generated image to the outputs directory"""
        if not self.auto_save:
            return None
            
        # Create a filename with timestamp and seed
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_short = model_name.split("/")[-1].replace("-", "_") if model_name else "model"
        
        # Add batch index if generating multiple images
        batch_suffix = f"_{index+1}of{batch_count}" if batch_count > 1 else ""
        
        filename = f"fusion_{timestamp}{batch_suffix}_seed{seed}_{model_short}.{self.save_format}"
        filepath = os.path.join(self.outputs_dir, filename)
        
        # Save the image
        image.save(filepath)
        print(f"Image saved: {filepath}")
        return filepath
    
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
        keep_original_size=True,
        num_images=1,  # Numărul de imagini de generat
        attire_customization="",
        decor_customization="",
        face_enhancement=None,  # Override pentru face enhancement
        active_loras=None,  # List of (lora_name, is_active, weight) tuples
        progress=gr.Progress(track_tqdm=True),
    ):
        """Generate a composite image based on reference and pose images"""
        # Update face enhancement if provided
        if face_enhancement is not None:
            self.face_enhancement = face_enhancement
        
        # Rezultate pentru multiple imagini
        results = []
        seeds = []
        save_paths = []
        
        # Setăm seed-ul inițial
        if seed == -1:
            seed = torch.randint(0, 2**32 - 1, (1,)).item()
        
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
        
        # Preprocess images - keeping original size if requested
        ref_img, pose_img = self.preprocess_images(reference_image, pose_image, target_size, keep_original_size)
        
        # Pentru fiecare imagine din batch
        for i in range(num_images):
            # Incrementăm seed-ul pentru fiecare imagine din batch pentru varietate
            current_seed = seed + i if i > 0 else seed
            generator = torch.Generator("cuda" if torch.cuda.is_available() else "cpu").manual_seed(current_seed)
            
            # Detectăm fețele pentru îmbunătățirea calității
            ref_faces = None
            pose_faces = None
            face_mask = None
            
            if face_recognition is not None and self.face_enhancement:
                try:
                    # Convertim la numpy arrays
                    ref_np = np.array(ref_img)
                    pose_np = np.array(pose_img)
                    
                    # Detectăm fețele
                    ref_faces = self.detect_faces(ref_np)
                    pose_faces = self.detect_faces(pose_np)
                    
                    # Găsim cea mai bună potrivire între fețe
                    if ref_faces and pose_faces:
                        ref_face_loc, pose_face_loc = self.find_best_face_match(ref_faces, pose_faces)
                        
                        # Creăm o mască pentru zona feței
                        face_mask = self.create_face_mask(pose_np, pose_face_loc, expand_ratio=1.8)
                except Exception as e:
                    print(f"Face detection error: {e}")
            
            # Blend images with adaptive face preservation if available
            if face_mask is not None:
                blended_img = self.adaptive_blend_images(ref_img, pose_img, weight=0.3, face_mask=face_mask)
            else:
                # Fallback la blending simplu
                blended_img = self.adaptive_blend_images(ref_img, pose_img, weight=0.3)
            
            # Combine customization options with prompt
            full_prompt = prompt
            if attire_customization:
                full_prompt += f", {attire_customization}"
            if decor_customization:
                full_prompt += f", {decor_customization}"
                
            if not full_prompt:
                # Îmbunătățim promptul implicit pentru transferul caracteristicilor faciale
                full_prompt = "same person as reference image, exact same face, same identity, in the pose shown"
            
            # Enhancing the prompt to better guide the model
            reference_description = "same face structure, same facial features, same identity, exact same person"
            enhanced_prompt = f"{full_prompt}, {reference_description}, photorealistic, highly detailed, 8k professional photo"
            
            # Prepare default negative prompt if not provided
            if not negative_prompt:
                negative_prompt = "deformed, ugly, bad proportions, bad anatomy, disfigured, mutations, poorly drawn, blurry, low quality, cartoon, anime, illustration, painting, drawing, different person, wrong face, two faces, multiple faces"
            
            # Generate the image with progress tracking
            with tqdm(total=num_inference_steps, desc=f"Generating image {i+1}/{num_images}") as progress_bar:
                def callback(step, timestep, latents):
                    progress_bar.update(1)
                    progress((i * num_inference_steps + step) / (num_images * num_inference_steps))
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
                
                # Adăugăm imaginea generată la rezultate
                results.append(result)
                seeds.append(current_seed)
                
                # Salvăm imaginea generată
                save_path = self.save_image(result, current_seed, self.current_model, i, num_images)
                save_paths.append(save_path)
        
        # Formatarea rezultatelor pentru returnare
        seeds_str = ", ".join(map(str, seeds))
        paths_str = ", ".join(save_paths) if save_paths else ""
        
        # Returnăm prima imagine și stringurile formatate pentru celelalte rezultate
        return results[0], seeds_str, paths_str

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
        print(f"Downloading model: {model_id}")
        snapshot_download(repo_id=model_id, cache_dir=self.cache_dir)
        return f"Downloaded model: {model_id}"
        
    def toggle_auto_save(self, value):
        """Toggle auto-save functionality"""
        self.auto_save = value
        return f"Auto-save {'enabled' if value else 'disabled'}"
    
    def set_save_format(self, format):
        """Set the save format for images"""
        if format in ['png', 'jpg', 'jpeg', 'webp']:
            self.save_format = format
            return f"Save format set to {format}"
        return f"Unsupported format: {format}. Using {self.save_format}."
    
    def set_face_enhancement(self, value):
        """Toggle face enhancement"""
        self.face_enhancement = value
        return f"Face enhancement {'enabled' if value else 'disabled'}"
    
    def set_preserve_face_strength(self, value):
        """Set the strength of face preservation"""
        self.preserve_face_strength = value
        return f"Face preservation strength set to {value}"


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
    
    # Function to get model ID from dropdown
    def get_model_id(model_name):
        return fusion_frame.get_model_id_from_name(model_name)
    
    # Function to download model
    def download_selected_model(model_name):
        model_id = fusion_frame.get_model_id_from_name(model_name)
        return fusion_frame.download_model(model_id)
    
    # Function to open outputs folder
    def open_outputs_folder():
        import subprocess
        import platform
        
        path = os.path.abspath(fusion_frame.outputs_dir)
        
        try:
            if platform.system() == "Windows":
                os.startfile(path)
            elif platform.system() == "Darwin":  # macOS
                subprocess.Popen(["open", path])
            else:  # Linux
                subprocess.Popen(["xdg-open", path])
            return f"Opened outputs folder: {path}"
        except Exception as e:
            return f"Error opening folder: {e}"
    
    # Process multiple image generation results
    def process_generation_results(results):
        if not results:
            return None, "No images generated", ""
        
        images, seeds, paths = results
        
        # Format seed list as string
        seeds_str = ", ".join([str(s) for s in seeds])
        
        # Format paths list as string
        paths_str = "\n".join(paths) if paths else ""
        
        return images, seeds_str, paths_str
    
    with gr.Blocks(title="FusionFrame App") as app:
        gr.Markdown("# FusionFrame App")
        gr.Markdown("Upload a reference image and a pose image to generate a composite.")
        
        with gr.Row():
            with gr.Column():
                reference_image = gr.Image(label="Reference Image (Person)", type="numpy")
                pose_image = gr.Image(label="Pose Image (Position/Scene)", type="numpy")
                
                with gr.Accordion("Advanced Settings", open=False):
                    with gr.Row():
                        keep_original_size = gr.Checkbox(label="Keep Original Size", value=True)
                    
                    with gr.Row():
                        width = gr.Slider(512, 2048, 1024, step=64, label="Width")
                        height = gr.Slider(512, 2048, 1024, step=64, label="Height")
                    
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
                        steps = gr.Slider(10, 150, 30, step=1, label="Inference Steps")
                        seed = gr.Number(-1, label="Seed (-1 for random)")
                    
                    with gr.Row():
                        num_images = gr.Slider(minimum=1, maximum=10, value=1, step=1, label="Number of Images")
                    
                    with gr.Accordion("Face Enhancement", open=False):
                        face_enhance = gr.Checkbox(label="Enable Face Enhancement", value=True)
                        face_strength = gr.Slider(0.1, 1.0, 0.8, step=0.05, label="Face Preservation Strength")
                    
                    with gr.Accordion("Customization Options", open=False):
                        attire_customization = gr.Textbox(label="Attire Customization", placeholder="E.g., wearing a red dress, formal suit")
                        decor_customization = gr.Textbox(label="Scene/Décor Customization", placeholder="E.g., beach background, sunny day")
                    
                    # Save Settings
                    with gr.Accordion("Save Settings", open=False):
                        with gr.Row():
                            auto_save_checkbox = gr.Checkbox(label="Auto Save Images", value=True)
                            save_format = gr.Dropdown(
                                choices=["png", "jpg", "webp"],
                                value="png",
                                label="Save Format"
                            )
                        
                        open_folder_button = gr.Button("Open Outputs Folder")
                        save_status = gr.Textbox(label="Save Status", value=f"Images will be saved to: {fusion_frame.outputs_dir}")
                    
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
                    download_status = gr.Textbox(label="Download Status", visible=True)
                
                generate_button = gr.Button("Generate Composite Image", variant="primary")
            
            with gr.Column():
                output_image = gr.Image(label="Generated Result")
                output_seed = gr.Textbox(label="Seeds Used (for reproducibility)")
                saved_path = gr.Textbox(label="Saved Image Paths", visible=True)
                
                # Gallery for multiple images
                output_gallery = gr.Gallery(label="All Generated Images", visible=False, columns=2, rows=2)
        
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
                keep_original_size,
                num_images,
                attire_customization,
                decor_customization,
                face_enhance,
                # LoRA settings
                lora1_name, lora1_active, lora1_weight,
                lora2_name, lora2_active, lora2_weight,
                lora3_name, lora3_active, lora3_weight,
                lora4_name, lora4_active, lora4_weight,
                lora5_name, lora5_active, lora5_weight,
            ],
            outputs=[output_image, output_seed, saved_path],
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
        
        model_dropdown.change(
            fn=get_model_id,
            inputs=[model_dropdown],
            outputs=[],  # No direct output, just updates the fusion_frame.current_model
        )
        
        sampler_dropdown.change(
            fn=lambda x: setattr(fusion_frame, "current_sampler", x) or x,
            inputs=[sampler_dropdown],
            outputs=[sampler_dropdown],  # Output to itself to indicate change
        )
        
        download_model_button.click(
            fn=download_selected_model,
            inputs=[model_dropdown],
            outputs=[download_status],
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
        
        # Setup auto-save controls
        auto_save_checkbox.change(
            fn=fusion_frame.toggle_auto_save,
            inputs=[auto_save_checkbox],
            outputs=[save_status],
        )
        
        save_format.change(
            fn=fusion_frame.set_save_format,
            inputs=[save_format],
            outputs=[save_status],
        )
        
        open_folder_button.click(
            fn=open_outputs_folder,
            inputs=[],
            outputs=[save_status],
        )
        
        # Hide/show width/height controls based on keep_original_size
        keep_original_size.change(
            fn=lambda x: [gr.update(interactive=not x), gr.update(interactive=not x)],
            inputs=[keep_original_size],
            outputs=[width, height],
        )
        
    return app

if __name__ == "__main__":
    app = build_gradio_interface()
    app.launch(share=True)