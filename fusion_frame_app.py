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
    import face_recognition  # For advanced facial detection
except ImportError:
    print("face_recognition not installed. Face alignment will not be available.")
    print("You can install it with: pip install face_recognition")
    face_recognition = None

class FusionFrame:

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_id = "stabilityai/stable-diffusion-xl-refiner-1.0"  # A highly realistic model
        self.cache_dir = os.path.join(os.path.expanduser("~"), ".fusionframe")
        self.loras_dir = os.path.join(os.getcwd(), "Loras")  # Default LoRAs directory in root
        self.outputs_dir = os.path.join(os.getcwd(), "outputs")  # Directory for saving generated images
        self.default_refiner_name = "SDXL Refiner 1.0 (Default)"
        self.refiner_pipe = None
        self.refiner_model_id = None
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
        
        # Automatically activate xformers for memory optimization if available
        try:
            import xformers
            self.use_xformers = True
            print("xFormers found and enabled for memory optimization")

        except (ImportError, RuntimeError) as e:
            self.use_xformers = False
            print(f"xFormers not available, memory optimization disabled: {e}")
            print("This won't affect functionality, but might use more VRAM")
                
        # Advanced settings
        self.auto_save = True
        self.save_format = "png"  # Save format (png, jpg)
        self.face_enhancement = True  # Facial enhancement
        self.face_alignment_weight = 0.8  # Weight for facial alignment (0-1)
        self.blend_method = "adaptive"  # Blending method (simple, adaptive, mask)
        self.preserve_face_strength = 0.8  # How much to preserve of the reference face

        # -- Facial selective transfer --
        self.enable_selective_face = True   # toggled from UI (future feature)
        self.face_transfer_parts = ("eyes", "nose", "mouth")
        self.face_transfer_blend = 0.85
        
        # -- ControlNet Pose --
        self.enable_cn_pose = False
        self.cn_strength_default = 1.0
        self.openpose_cn = None     # lazy-load
        self.cn_pipe = None
        
        # -- Refiner --
        self.enable_two_stage = False
        self.refiner_strength = 0.3  # Default refiner strength

    def scan_loras(self):
        """Scan the LoRAs directory for available .safetensors files"""
        lora_files = glob.glob(os.path.join(self.loras_dir, "*.safetensors"))
        loras = {}
        
        for lora_file in lora_files:
            lora_name = os.path.basename(lora_file).replace(".safetensors", "")
            loras[lora_name] = lora_file
            
        return loras

    # ========= ControlNet helpers =========
    def load_controlnet_pose(self):
        """Load ControlNet OpenPose SDXL only on first use."""
        if self.cn_pipe is not None:
            return True

        try:
            from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline
            print("📦 Loading ControlNet-OpenPose...")
            
            # Try alternative model sources if the primary one fails
            model_sources = [

                "thibaud/controlnet-openpose-sdxl-1.0",  # This is the correct one based on HF
                "thibaud/controlnet-openpose-sdxl",
                "diffusers/controlnet-openpose-sdxl",
                "diffusers/controlnet-openpose-sdxl-1.0" 
            ]
            
            success = False
            for source in model_sources:
                try:
                    print(f"Trying to load ControlNet from: {source}")
                    self.openpose_cn = ControlNetModel.from_pretrained(
                        source,
                        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                        cache_dir=self.cache_dir,
                        use_auth_token=False,  # Don't require login
                        local_files_only=False,
                        resume_download=True
                    ).to(self.device)
                    success = True
                    print(f"Successfully loaded ControlNet from: {source}")
                    break
                except Exception as e:
                    print(f"Failed to load from {source}: {e}")
            
            if not success:
                print("Failed to load ControlNet from any source.")
                return False
                
            # Make sure self.pipe (SDXL) has been already loaded
            if self.pipe is None:
                self.load_model()
                
            self.cn_pipe = StableDiffusionXLControlNetPipeline(
                vae=self.pipe.vae,
                text_encoder=self.pipe.text_encoder,
                tokenizer=self.pipe.tokenizer,
                unet=self.pipe.unet,
                controlnet=self.openpose_cn,
                scheduler=self.pipe.scheduler,
                text_encoder_2=self.pipe.text_encoder_2 if hasattr(self.pipe, "text_encoder_2") else None,
                tokenizer_2=self.pipe.tokenizer_2 if hasattr(self.pipe, "tokenizer_2") else None,
            ).to(self.device)
            
            # xformers (optional)
            if self.use_xformers and torch.cuda.is_available():
                try:
                    self.cn_pipe.enable_xformers_memory_efficient_attention()
                    print("xFormers activated for ControlNet")
                except Exception as e:
                    print(f"xFormers could not be activated for ControlNet: {e}")
            
            return True
        except Exception as e:
            print(f"Error initializing ControlNet: {e}")













            return False





    def setup_controlnet_pipeline(self):
        """Set up a compatible ControlNet pipeline for the current model"""
        if self.pipe is None:
            self.load_model()
            
        try:
            from diffusers import ControlNetModel, StableDiffusionXLControlNetImg2ImgPipeline
            from controlnet_aux import OpenposeDetector
            
            # Load the ControlNet model with specific configuration
            controlnet = ControlNetModel.from_pretrained(
                "thibaud/controlnet-openpose-sdxl-1.0",
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                use_safetensors=True
            ).to(self.device)
            
            # Create a completely new pipeline from scratch
            # The key here is to specify target_size explicitly
            cn_pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
                self.current_model,
                controlnet=controlnet,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                cache_dir=self.cache_dir,
                use_safetensors=True,
                variant="fp16"
            ).to(self.device)
            
            # Force disable aesthetics score
            cn_pipe.config.requires_aesthetics_score = False
            
            # Set a fixed target size to avoid dimension conflicts
            cn_pipe.config.target_size = (1024, 1024)
            
            # Ensure scheduler compatibility
            cn_pipe.scheduler = self.pipe.scheduler
            
            if self.use_xformers:
                try:
                    cn_pipe.enable_xformers_memory_efficient_attention()
                    print("xFormers enabled for ControlNet")
                except Exception as e:
                    print(f"xFormers couldn't be enabled for ControlNet: {e}")
            
            # Set up the pose detector
            detector = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
            
            return cn_pipe, detector
        except Exception as e:
            print(f"Error setting up ControlNet: {e}")
            return None, None

    
    def load_model(self, model_id=None):
        """Load the selected model on demand with improved error handling"""
        if model_id:
            self.current_model = model_id
        
        print(f"Loading model: {self.current_model}")


        
        # Clear CUDA cache if available to prevent OOM issues
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        
        # Ensure the model is fully downloaded before trying to load it
        try:
            model_path = snapshot_download(
                repo_id=self.current_model, 
                cache_dir=self.cache_dir,
                resume_download=True,
                local_files_only=False
            )
            print(f"Model downloaded successfully to: {model_path}")
        except Exception as e:
            print(f"Warning during model download: {e}")
            # Continue with loading, maybe the model is already partially downloaded
        
        # Download and load the pipeline
        try:
            # Check if the model is SDXL or standard SD
            if "xl" in self.current_model.lower():
                # SDXL models
                self.pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                    self.current_model,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    use_safetensors=True,
                    variant="fp16",
                    cache_dir=self.cache_dir,
                    local_files_only=False,  # Important to allow download
                )
            else:
                # Standard SD models
                from diffusers import StableDiffusionImg2ImgPipeline
                self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                    self.current_model,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    use_safetensors=True,
                    variant="fp16",
                    cache_dir=self.cache_dir,
                    local_files_only=False,
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
                print("Using GPU for generation")
                
                # Activate xformers for memory optimization if available
                if self.use_xformers:
                    try:
                        self.pipe.enable_xformers_memory_efficient_attention()
                        print("xFormers activated for memory optimization")
                    except Exception as e:
                        print(f"xFormers could not be activated: {e}")
                        print("Install xformers with: pip install xformers")
            else:
                print("No GPU available, using CPU (will be much slower)")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            
            # Fallback to a known model if the specified one cannot be loaded
            if self.current_model != "runwayml/stable-diffusion-v1-5":
                print("Trying alternative model: Stable Diffusion 1.5")
                self.current_model = "runwayml/stable-diffusion-v1-5"
                return self.load_model()
            else:
                raise e
        
        return self.pipe

    def load_refiner_model(self, model_id):
        """Load a refiner model for two-stage generation"""
        if self.refiner_pipe and self.refiner_model_id == model_id:
            return self.refiner_pipe

        self.refiner_model_id = model_id
        print(f"Loading 2-stage refiner: {model_id}")

        from diffusers import StableDiffusionXLImg2ImgPipeline
        self.refiner_pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            variant="fp16",
            cache_dir=self.cache_dir,
        ).to("cuda" if torch.cuda.is_available() else "cpu")

        # Use the same scheduler as the first stage for consistency
        if isinstance(self.pipe.scheduler, type(self.refiner_pipe.scheduler)):
            self.refiner_pipe.scheduler = self.pipe.scheduler

        # Apply xformers if available
        if self.use_xformers and torch.cuda.is_available():
            try:
                self.refiner_pipe.enable_xformers_memory_efficient_attention()
                print("xFormers activated for refiner")
            except Exception as e:
                print(f"xFormers could not be activated for refiner: {e}")

        return self.refiner_pipe
        
    def load_lora(self, lora_path, scale=0.7):
        """Load a LoRA into the pipeline"""
        if self.pipe is None:
            self.load_model()
            
        try:
            # Simplify LoRA loading using the built-in pipeline method
            if hasattr(self.pipe, "load_lora_weights"):
                # Official and recommended method
                self.pipe.load_lora_weights(lora_path, adapter_name=os.path.basename(lora_path))
                # Set weight for LoRA
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

    def create_face_mask(self, image, face_location, expand_ratio=1.8):
        """Create a mask highlighting the face area with improved gradient"""
        if face_location is None:
            return None
            
        height, width = image.shape[:2] if isinstance(image, np.ndarray) else (image.height, image.width)
        mask = np.zeros((height, width), dtype=np.uint8)
        
        # Expand face location
        top, right, bottom, left = face_location
        center_y, center_x = (top + bottom) // 2, (left + right) // 2
        face_height = bottom - top
        face_width = right - left
        
        # Calculate expanded dimensions with greater emphasis on the face
        expanded_height = int(face_height * expand_ratio)
        expanded_width = int(face_width * expand_ratio)
        
        # Create elliptical mask instead of rectangular for more natural blending
        cv2.ellipse(
            mask,
            center=(center_x, center_y),
            axes=(expanded_width // 2, expanded_height // 2),
            angle=0, startAngle=0, endAngle=360,
            color=255, thickness=-1
        )
        
        # Add feathering for smoother edges - use a larger blur for smoother transitions
        mask = cv2.GaussianBlur(mask, (111, 111), 30)
        
        # Intensify the center of the face for more emphasis
        face_core = np.zeros((height, width), dtype=np.uint8)
        cv2.ellipse(
            face_core,
            center=(center_x, center_y),
            axes=(face_width // 2, face_height // 2),
            angle=0, startAngle=0, endAngle=360,
            color=255, thickness=-1
        )
        face_core = cv2.GaussianBlur(face_core, (31, 31), 10)
        
        # Combine mask with emphasis on the center of the face
        mask = cv2.addWeighted(mask, 0.7, face_core, 0.3, 0)
        
        return mask

    def adaptive_blend_images(self, ref_img, pose_img, weight=0.3, face_mask=None):
        """
        Blend images with advanced face preservation and feature transfer
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
            face_weight = min(self.preserve_face_strength * 1.2, 1.0)  # Boost face weight but cap at 1.0
            body_weight = weight
            
            # Apply adaptive blending with enhanced focus on facial features
            result_array = (1 - face_mask_3ch) * ((1 - body_weight) * pose_array + body_weight * ref_array) + \
                           face_mask_3ch * ((1 - face_weight) * pose_array + face_weight * ref_array)
                           
            # Optional: Transfer texture details for face only
            if self.preserve_face_strength > 0.7:
                # Extract high frequency details from reference image
                ref_gray = cv2.cvtColor(ref_array.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(float)
                ref_blur = cv2.GaussianBlur(ref_gray, (21, 21), 5)
                ref_details = ref_gray - ref_blur
                
                # Apply details only to face area with proper weighting
                details_strength = 0.4  # Control strength of detail transfer
                for c in range(3):
                    result_array[:,:,c] += face_mask_float * ref_details * details_strength
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
    
    def enhance_prompt(self, prompt, attire_customization="", decor_customization=""):
        """Enhance the prompt to get better results"""
        base_prompt = prompt
        
        # Add customizations for attire and decor
        if attire_customization:
            base_prompt += f", {attire_customization}"
        if decor_customization:
            base_prompt += f", {decor_customization}"
            
        if not base_prompt:
            # Improved default prompt for facial feature transfer
            base_prompt = "same person as reference image, exact same face, same identity, in the pose shown"
        
        # Enhance the prompt with specific instructions for faces
        face_details = "same exact face structure, same facial features, same identity, exactly the same person with identical features"
        quality_details = "photorealistic, highly detailed, 8k professional photo, perfect lighting, award-winning portrait"
        
        # Combine into final prompt
        enhanced_prompt = f"{base_prompt}, {face_details}, {quality_details}"
        
        return enhanced_prompt
    
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
        num_images=1,  # Number of images to generate
        attire_customization="",
        decor_customization="",
        face_enhancement=None,  # Override for face enhancement
        enable_two_stage=False,  # Two-stage refiner
        refiner_model_name="SDXL Refiner 1.0 (Default)",  # Refiner model
        refiner_strength=0.3,  # Refiner strength
        enable_cn_pose=False,  # ControlNet pose
        cn_strength=1.0,  # ControlNet strength
        enable_selective_face=True,  # Selective face transfer
        active_loras=None,  # List of (lora_name, is_active, weight) tuples
        progress=gr.Progress(track_tqdm=True),
    ):
        """Generate a composite image based on reference and pose images"""
        # Update face enhancement if provided
        if face_enhancement is not None:
            self.face_enhancement = face_enhancement
        
        # Results for multiple images
        results = []
        seeds = []
        save_paths = []

        self.enable_two_stage = enable_two_stage
        self.refiner_strength = refiner_strength
        
        # Set the initial seed
        if seed == -1:
            seed = torch.randint(0, 2**32 - 1, (1,)).item()
        
        # Load the model if not already loaded or if it has changed
        if self.pipe is None:
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
        
        # For each image in batch
        for i in range(num_images):
            # Increment seed for each image in batch for variety
            current_seed = seed + i if i > 0 else seed
            generator = torch.Generator("cuda" if torch.cuda.is_available() else "cpu").manual_seed(current_seed)
            
            # Detect faces for quality improvement
            ref_faces = None
            pose_faces = None
            face_mask = None
            
            if face_recognition is not None and self.face_enhancement:
                try:
                    # Convert to numpy arrays
                    ref_np = np.array(ref_img)
                    pose_np = np.array(pose_img)
                    
                    # Detect faces
                    ref_faces = self.detect_faces(ref_np)
                    pose_faces = self.detect_faces(pose_np)
                    
                    # Find the best match between faces
                    if ref_faces and pose_faces:
                        ref_face_loc, pose_face_loc = self.find_best_face_match(ref_faces, pose_faces)
                        
                        # Create a mask for the face area
                        face_mask = self.create_face_mask(pose_np, pose_face_loc, expand_ratio=1.8)
                except Exception as e:
                    print(f"Face detection error: {e}")
            
            # Blend images with adaptive face preservation if available
            if face_mask is not None:
                blended_img = self.adaptive_blend_images(ref_img, pose_img, weight=0.3, face_mask=face_mask)
            else:
                # Fallback to simple blending
                blended_img = self.adaptive_blend_images(ref_img, pose_img, weight=0.3)

            # --- (Optional) selective feature transfer ----
            if enable_selective_face and face_mask is not None:
                try:
                    import face_utils
                    src_np = np.array(ref_img)
                    dst_np = np.array(blended_img)
                    transferred = face_utils.selective_clone(
                        src_np, dst_np,
                        parts=self.face_transfer_parts,
                        alpha=self.face_transfer_blend
                    )
                    blended_img = Image.fromarray(transferred)
                except Exception as e:
                    print(f"[SelectiveClone] {e}")
            
            # Use the enhanced prompt method
            enhanced_prompt = self.enhance_prompt(
                prompt, 
                attire_customization=attire_customization, 
                decor_customization=decor_customization
            )
            
            # Prepare default negative prompt if not provided
            if not negative_prompt:
                negative_prompt = "deformed face, ugly, bad proportions, bad anatomy, disfigured, mutations, poorly drawn, blurry, low quality, cartoon, anime, illustration, painting, drawing, different person, wrong face, two faces, multiple faces, mutation, deformed iris, deformed pupils, morbid, mutilated, extra fingers, extra limbs, disfigured"
            
            # Generate the image with progress tracking
            with tqdm(total=num_inference_steps, desc=f"Generating image {i+1}/{num_images}") as progress_bar:
                def callback(step, timestep, latents):
                    progress_bar.update(1)
                    progress((i * num_inference_steps + step) / (num_images * num_inference_steps))
                    return
                
                # For SDXL, try to use IP-Adapter to improve feature transfer
                use_ip_adapter = False

                if "xl" in self.current_model.lower():
                    try:
                        from IPAdapter import IPAdapterXL
                        
                        ip_adapter = IPAdapterXL(
                            self.pipe,
                            "h94/IP-Adapter",
                            subfolder="sdxl_models",  # Make sure this is included
                            cache_dir=self.cache_dir
                        )
                        
                        self.pipe = ip_adapter.pipe
                        use_ip_adapter = "manual"
                        print("✅ Manual IP-Adapter activated for improved facial transfer")
                    except ImportError:
                        print("❌ IP-Adapter not available. Run: pip install git+https://github.com/tencent-ailab/IP-Adapter.git")
                    except Exception as e:
                        print(f"⚠️ Failed to activate IP-Adapter: {e}")

                # Different parameters based on model type
                kwargs = {
                    "image": blended_img,  # Use blended image as base
                    "prompt": enhanced_prompt,
                    "negative_prompt": negative_prompt,
                    "strength": strength,
                    "guidance_scale": guidance_scale,
                    "num_inference_steps": num_inference_steps,
                    "generator": generator,
                    "callback": callback,
                    "callback_steps": 1,
                }
                
                # For XL models add specific parameters
                if "xl" in self.current_model.lower():
                    if use_ip_adapter == True:
                        kwargs["ip_adapter_image"] = ref_img
                    elif use_ip_adapter == "manual":
                        # Manual IP-Adapter handling for older implementations
                        ip_image_embeds = ip_adapter.encode_image(ref_img)
                        kwargs["ip_adapter_image_embeds"] = ip_image_embeds

                # If user wants ControlNet Pose and we're on an XL model
                result = None
                if enable_cn_pose and "xl" in self.current_model.lower():
                    try:
                        cn_pipe, detector = self.setup_controlnet_pipeline()
                        if cn_pipe and detector:
                            pose_cond = detector(pose_img)
                            
                            cn_result = cn_pipe(
                                prompt=enhanced_prompt,
                                negative_prompt=negative_prompt,
                                image=blended_img,
                                control_image=pose_cond,
                                controlnet_conditioning_scale=float(cn_strength),
                                guidance_scale=guidance_scale,
                                num_inference_steps=num_inference_steps,
                                generator=generator,
                            ).images[0]
                            
                            result = cn_result
                            print("Successfully generated image with ControlNet Pose")
                        else:
                            print("Failed to set up ControlNet, falling back to normal pipeline")
                            result = self.pipe(**kwargs).images[0]
                    except Exception as e:
                        print(f"ControlNet error: {e}. Falling back to normal pipeline.")
                        result = self.pipe(**kwargs).images[0]
                else:
                    # Normal pipeline
                    result = self.pipe(**kwargs).images[0]
                
                # -------------------- TWO-STAGE REFINEMENT --------------------
                if enable_two_stage:
                    try:
                        refine_id = self.get_model_id_from_name(refiner_model_name)
                        refiner = self.load_refiner_model(refine_id)

                        # Setup conservative values (low strength => fine detail)
                        refine_kwargs = dict(
                            image=result,
                            prompt=enhanced_prompt,
                            negative_prompt=negative_prompt,
                            strength=self.refiner_strength,  # Use the refiner strength parameter
                            guidance_scale=max(5.0, guidance_scale - 2),
                            num_inference_steps=int(num_inference_steps * 0.6),
                            generator=generator,
                        )

                        # Callback for refiner progress tracking
                        def refiner_callback(step, timestep, latents):
                            total_steps = num_images * num_inference_steps
                            refiner_total = int(num_inference_steps * 0.6)
                            progress_value = ((i * num_inference_steps) + num_inference_steps + step) / (total_steps + refiner_total)
                            progress(min(progress_value, 1.0))
                            return

                        refine_kwargs["callback"] = refiner_callback
                        refine_kwargs["callback_steps"] = 1

                        result = refiner(**refine_kwargs).images[0]
                        print(f"Refinement complete with strength {self.refiner_strength}")
                    except Exception as e:
                        print(f"Refiner error: {e}. Using first-stage result.")
                # --------------------------------------------------------------
                
                # Add generated image to results
                results.append(result)
                seeds.append(current_seed)
                
                # Save the generated image
                save_path = self.save_image(result, current_seed, self.current_model, i, num_images)
                save_paths.append(save_path)
        
        # Format results for return
        seeds_str = ", ".join(map(str, seeds))
        paths_str = ", ".join(save_paths) if save_paths else ""
        
        # Return the first image and formatted strings for other results
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
    
    def set_refiner_strength(self, value):
        """Set the strength of the refiner"""
        self.refiner_strength = value
        return f"Refiner strength set to {value}"


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
        """Get the model ID from the display name and update the current model"""
        model_id = fusion_frame.get_model_id_from_name(model_name)
        print(f"Changing model to: {model_id}")
        
        # Reset pipeline to force reloading
        fusion_frame.pipe = None
        fusion_frame.current_model = model_id
        
        return f"Model changed to: {model_name}"
    
    # Function to download model
    def download_selected_model(model_name):
        """Download a specific model to the cache directory and load it"""
        model_id = fusion_frame.get_model_id_from_name(model_name)
        
        try:
            print(f"Downloading model: {model_id}")
            snapshot_download(
                repo_id=model_id, 
                cache_dir=fusion_frame.cache_dir,
                resume_download=True
            )
            
            # Reset pipeline to force reloading the model
            fusion_frame.pipe = None
            fusion_frame.current_model = model_id
            
            # Pre-load the model to check functionality
            fusion_frame.load_model()
            
            return f"Model downloaded and loaded successfully: {model_name}"
        except Exception as e:
            return f"Error downloading model: {e}"
    
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
                        two_stage_chk = gr.Checkbox(label="Enable 2-stage Refiner", value=False)
                        refiner_dropdown = gr.Dropdown(
                            choices=fusion_frame.list_available_models(),
                            value="SDXL Refiner 1.0 (Default)",
                            label="Refiner Model",
                            interactive=True,
                            visible=False
                        )
                        refiner_strength = gr.Slider(
                            minimum=0.1, maximum=0.7, value=0.3, step=0.05,
                            label="Refiner Strength",
                            info="Lower values preserve more details",
                            visible=False
                        )
                        
                    with gr.Row():
                        strength = gr.Slider(0.1, 1.0, 0.75, step=0.05, label="Strength (how much to preserve pose)")
                        guidance_scale = gr.Slider(1.0, 15.0, 7.5, step=0.5, label="Guidance Scale")
                    
                    with gr.Row():
                        steps = gr.Slider(10, 150, 30, step=1, label="Inference Steps")
                        seed = gr.Number(-1, label="Seed (-1 for random)")

                    # === ControlNet Pose UI ===
                    with gr.Row():
                        enable_cn_pose = gr.Checkbox(label="Enable ControlNet Pose", value=False)
                        cn_strength = gr.Slider(
                            minimum=0.0, maximum=2.0, value=1.0, step=0.05,
                            label="ControlNet Strength",
                            info="How much the pose controls the generation",
                            interactive=True,
                            visible=False
                        )
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
                *args[:15],  # First 15 existing inputs
                enable_two_stage=args[15],
                refiner_model_name=args[16],
                refiner_strength=args[17],
                enable_cn_pose=args[18],
                cn_strength=args[19],
                enable_selective_face=True,  # Always enabled for now
                active_loras=collect_lora_settings(*args[20:])
            ),
            inputs=[
                reference_image, pose_image,
                prompt, negative_prompt, strength, guidance_scale,
                steps, seed, width, height, keep_original_size,
                num_images, attire_customization, decor_customization,
                face_enhance,
                two_stage_chk, refiner_dropdown, refiner_strength,
                enable_cn_pose, cn_strength,
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

        model_dropdown.change(
            fn=get_model_id,
            inputs=[model_dropdown],
            outputs=[download_status],
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