"""
Core module for FusionFrame application.

This module contains the main FusionFrame class which handles model loading,
image preprocessing, and generation. It's been refactored to use plugins and
configuration modules for better organization.
"""

import os
import torch
import logging
import numpy as np
from PIL import Image
import datetime
import cv2
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
from huggingface_hub import snapshot_download
from diffusers import StableDiffusionXLImg2ImgPipeline, DPMSolverMultistepScheduler
from diffusers.utils import load_image

# Import configuration
import config

# Import utilities
import utils.face_utils as face_utils
import utils.io_utils as io_utils

# Import plugins
from plugins.ip_adapter import IPAdapterPlugin
from plugins.controlnet import ControlNetPlugin

# Set up logging
logging.basicConfig(
    level=getattr(logging, config.LOGGING_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(config.LOG_FILE) if config.LOG_TO_FILE else logging.NullHandler()
    ]
)

logger = logging.getLogger(__name__)

class FusionFrame:
    """
    Main class for face fusion and image generation.
    
    This class handles:
    - Model loading and management
    - Image preprocessing
    - Face detection and enhancement
    - Integration with Stable Diffusion, LoRA, IP-Adapter, and ControlNet
    """
    
    def __init__(self):
        """Initialize the FusionFrame application."""
        logger.info("Initializing FusionFrame")
        
        # System configuration
        self.device = config.DEVICE
        self.cache_dir = config.CACHE_DIR
        self.loras_dir = config.LORAS_DIR
        self.outputs_dir = config.OUTPUTS_DIR
        
        # Check if using cuda
        self.use_gpu = torch.cuda.is_available()
        if self.use_gpu:
            logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            logger.info("GPU not available, using CPU (will be much slower)")
        
        # Try to enable xFormers for memory optimization
        self.use_xformers = False
        if self.use_gpu:
            try:
                import xformers
                self.use_xformers = True
                logger.info("xFormers found and enabled for memory optimization")
            except (ImportError, RuntimeError) as e:
                logger.warning(f"xFormers not available, memory optimization disabled: {e}")
        
        # Model management - use a dictionary to cache loaded models
        self.model_cache = {}
        self.current_model_id = config.DEFAULT_MODEL_ID
        self.pipe = None  # Active pipeline
        
        # Sampler management
        self.available_samplers = {
            "DPM++ 2M Karras": DPMSolverMultistepScheduler.from_config,
            "Euler a": None,  # Will be implemented as needed
            "DDIM": None,     # Will be implemented as needed
        }
        self.current_sampler = config.DEFAULT_SAMPLER
        
        # LoRA management
        self.available_loras = self._scan_loras()
        self.active_loras = []
        
        # Initialize plugins
        self.ip_adapter = IPAdapterPlugin(self.cache_dir)
        self.controlnet = ControlNetPlugin(self.cache_dir)
        
        # Advanced settings with defaults from config
        self.auto_save = config.AUTO_SAVE_DEFAULT
        self.save_format = config.SAVE_FORMAT_DEFAULT
        self.face_enhancement = config.FACE_ENHANCEMENT_DEFAULT
        self.face_alignment_weight = config.FACE_PRESERVATION_STRENGTH
        self.face_transfer_parts = config.FACE_TRANSFER_PARTS
        self.face_transfer_blend = config.FACE_TRANSFER_BLEND
        
        # Refiner settings
        self.enable_two_stage = config.REFINER_ENABLED_DEFAULT
        self.refiner_strength = config.REFINER_STRENGTH_DEFAULT
        self.refiner_pipe = None
        self.refiner_model_id = None
        
        logger.info("FusionFrame initialized successfully")
    
    def _scan_loras(self) -> Dict[str, str]:
        """Scan for available LoRA models in the loras directory."""
        logger.info(f"Scanning for LoRAs in {self.loras_dir}")
        return io_utils.scan_directory(self.loras_dir, "*.safetensors")
    
    def _detect_faces(self, image: Union[np.ndarray, Image.Image]) -> Optional[Dict]:
        """
        Detect faces in an image using face_recognition.
        
        Args:
            image: Input image (numpy array or PIL Image)
            
        Returns:
            Dict or None: Dictionary with face locations and encodings, or None if no faces found
        """
        try:
            import face_recognition
            
            # Convert PIL image to numpy array if needed
            if isinstance(image, Image.Image):
                image_np = np.array(image)
            else:
                image_np = image
            
            # Find all face locations
            face_locations = face_recognition.face_locations(image_np, model="hog")
            
            if not face_locations:
                logger.warning("No faces detected in the image")
                return None
                
            # Get face encodings
            face_encodings = face_recognition.face_encodings(image_np, face_locations)
            
            logger.info(f"Detected {len(face_locations)} faces")
            return {
                "locations": face_locations,
                "encodings": face_encodings
            }
        except ImportError:
            logger.warning("face_recognition module not installed. Face detection disabled.")
            return None
        except Exception as e:
            logger.error(f"Error during face detection: {e}")
            return None
    
    def _find_best_face_match(self, reference_faces: Dict, pose_faces: Dict) -> Tuple[Optional[Tuple], Optional[Tuple]]:
        """
        Find the best match between reference and pose faces.
        
        Args:
            reference_faces: Dictionary with face locations and encodings from reference image
            pose_faces: Dictionary with face locations and encodings from pose image
            
        Returns:
            Tuple of (reference_face_location, pose_face_location) or (None, None) if no match
        """
        if not reference_faces or not pose_faces:
            return None, None
            
        try:
            import face_recognition
            
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
            
            logger.debug(f"Best face match score: {best_match_score:.2f}")
            return reference_faces["locations"][best_match_ref], pose_faces["locations"][best_match_pose]
        except Exception as e:
            logger.error(f"Error finding face match: {e}")
            return None, None
    
    def _create_face_mask(self, image: np.ndarray, face_location: Tuple, expand_ratio: float = 1.8) -> Optional[np.ndarray]:
        """
        Create a mask highlighting the face area with improved gradient.
        
        Args:
            image: Input image
            face_location: Tuple of (top, right, bottom, left) face coordinates
            expand_ratio: How much to expand the face area
            
        Returns:
            numpy.ndarray: Face mask or None if creation failed
        """
        if face_location is None:
            return None
            
        try:
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
        except Exception as e:
            logger.error(f"Error creating face mask: {e}")
            return None
    
    def _adaptive_blend_images(
        self, 
        ref_img: Image.Image, 
        pose_img: Image.Image, 
        weight: float = 0.3,
        face_mask: Optional[np.ndarray] = None
    ) -> Image.Image:
        """
        Blend images with advanced face preservation.
        
        Args:
            ref_img: Reference image with the face to preserve
            pose_img: Pose image with the target pose
            weight: Blending weight (0-1)
            face_mask: Optional mask for the face area
            
        Returns:
            PIL.Image: Blended image
        """
        try:
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
                face_weight = min(self.face_alignment_weight * 1.2, 1.0)  # Boost face weight but cap at 1.0
                body_weight = weight
                
                # Apply adaptive blending with enhanced focus on facial features
                result_array = (1 - face_mask_3ch) * ((1 - body_weight) * pose_array + body_weight * ref_array) + \
                               face_mask_3ch * ((1 - face_weight) * pose_array + face_weight * ref_array)
                               
                # Optional: Transfer texture details for face only
                if self.face_alignment_weight > 0.7:
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
        except Exception as e:
            logger.error(f"Error blending images: {e}")
            return pose_img
    
    def preprocess_images(
        self, 
        reference_image: Union[np.ndarray, Image.Image], 
        pose_image: Union[np.ndarray, Image.Image], 
        target_size: Optional[Tuple[int, int]] = None, 
        keep_original_size: bool = True
    ) -> Tuple[Image.Image, Image.Image]:
        """
        Preprocess the input images for generation.
        
        Args:
            reference_image: Reference image with the face to preserve
            pose_image: Pose image with the target pose
            target_size: Optional target size as (width, height)
            keep_original_size: Whether to keep original image size
            
        Returns:
            Tuple of (reference_image, pose_image) as PIL Images
        """
        logger.info("Preprocessing images")
        
        if reference_image is None or pose_image is None:
            raise ValueError("Both reference and pose images are required")
            
        # Convert to PIL if needed
        if not isinstance(reference_image, Image.Image):
            reference_image = Image.fromarray(reference_image)
        if not isinstance(pose_image, Image.Image):
            pose_image = Image.fromarray(pose_image)
        
        # Handle sizing based on parameters
        if target_size and not keep_original_size:
            width, height = target_size
            reference_image = reference_image.resize((width, height), Image.LANCZOS)
            pose_image = pose_image.resize((width, height), Image.LANCZOS)
        elif keep_original_size:
            # Keep original size, but ensure dimensions are multiples of 8 for stability
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
    
    def load_model(self, model_id: Optional[str] = None) -> Any:
        """
        Load a Stable Diffusion model.
        
        Args:
            model_id: The model identifier from Hugging Face
            
        Returns:
            The loaded pipeline
        """
        # Use provided model_id or current model
        model_id = model_id or self.current_model_id
        logger.info(f"Loading model: {model_id}")
        
        # Check if model is already loaded in cache
        if model_id in self.model_cache:
            logger.info(f"Using cached model: {model_id}")
            self.pipe = self.model_cache[model_id]
            self.current_model_id = model_id
            return self.pipe
        
        # Clear CUDA cache if available to prevent OOM issues
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Ensure the model is fully downloaded before trying to load it
        try:
            model_path = snapshot_download(
                repo_id=model_id, 
                cache_dir=self.cache_dir,
                resume_download=True,
                local_files_only=False
            )
            logger.info(f"Model downloaded successfully to: {model_path}")
        except Exception as e:
            logger.warning(f"Warning during model download: {e}")
            # Continue with loading, maybe the model is already partially downloaded
        
        # Download and load the pipeline
        try:
            # Check if the model is SDXL or standard SD
            if "xl" in model_id.lower():
                # SDXL models
                pipeline = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                    model_id,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    use_safetensors=True,
                    variant="fp16",
                    cache_dir=self.cache_dir,
                    local_files_only=False,  # Important to allow download
                )
            else:
                # Standard SD models
                from diffusers import StableDiffusionImg2ImgPipeline
                pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(
                    model_id,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    use_safetensors=True,
                    variant="fp16",
                    cache_dir=self.cache_dir,
                    local_files_only=False,
                )
            
            # Set the scheduler based on the current_sampler
            if self.current_sampler == "DPM++ 2M Karras" and self.available_samplers[self.current_sampler]:
                pipeline.scheduler = self.available_samplers[self.current_sampler](
                    pipeline.scheduler.config, 
                    algorithm_type="dpmsolver++", 
                    use_karras_sigmas=True
                )
                
            # Move to GPU if available
            if torch.cuda.is_available():
                pipeline = pipeline.to("cuda")
                logger.info("Model moved to GPU")
                
                # Activate xformers for memory optimization if available
                if self.use_xformers:
                    try:
                        pipeline.enable_xformers_memory_efficient_attention()
                        logger.info("xFormers activated for memory optimization")
                    except Exception as e:
                        logger.warning(f"xFormers could not be activated: {e}")
            
            # Store in cache and set as current
            self.model_cache[model_id] = pipeline
            self.pipe = pipeline
            self.current_model_id = model_id
            
            logger.info(f"Model {model_id} loaded successfully")
            return pipeline
            
        except Exception as e:
            logger.error(f"Error loading model {model_id}: {e}")
            
            # Fallback to a known model if the specified one cannot be loaded
            if model_id != "runwayml/stable-diffusion-v1-5":
                logger.info("Trying alternative model: Stable Diffusion 1.5")
                return self.load_model("runwayml/stable-diffusion-v1-5")
            else:
                raise e
    
    def load_refiner_model(self, model_id: str) -> Any:
        """
        Load a refiner model for two-stage generation.
        
        Args:
            model_id: The model identifier from Hugging Face
            
        Returns:
            The loaded refiner pipeline
        """
        if self.refiner_pipe and self.refiner_model_id == model_id:
            logger.info(f"Using cached refiner: {model_id}")
            return self.refiner_pipe

        logger.info(f"Loading 2-stage refiner: {model_id}")
        self.refiner_model_id = model_id

        try:
            from diffusers import StableDiffusionXLImg2ImgPipeline
            refiner_pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                variant="fp16",
                cache_dir=self.cache_dir,
            ).to("cuda" if torch.cuda.is_available() else "cpu")

            # Use the same scheduler as the first stage for consistency
            if isinstance(self.pipe.scheduler, type(refiner_pipe.scheduler)):
                refiner_pipe.scheduler = self.pipe.scheduler

            # Apply xformers if available
            if self.use_xformers and torch.cuda.is_available():
                try:
                    refiner_pipe.enable_xformers_memory_efficient_attention()
                    logger.info("xFormers activated for refiner")
                except Exception as e:
                    logger.warning(f"xFormers could not be activated for refiner: {e}")

            self.refiner_pipe = refiner_pipe
            return refiner_pipe
        except Exception as e:
            logger.error(f"Error loading refiner model: {e}")
            return None
    
    def load_lora(self, lora_path: str, scale: float = 0.7) -> bool:
        """
        Load a LoRA into the pipeline.
        
        Args:
            lora_path: Path to the LoRA file
            scale: LoRA influence scale (0-1)
            
        Returns:
            bool: True if successful, False otherwise
        """
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
                logger.info(f"LoRA loaded successfully: {os.path.basename(lora_path)} with scale {scale}")
                return True
            else:
                logger.warning(f"The current pipeline doesn't support loading LoRAs directly")
                return False
        except Exception as e:
            logger.error(f"Error loading LoRA: {e}")
            return False
    
    def apply_active_loras(self) -> bool:
        """
        Apply all active LoRAs to the model.
        
        Returns:
            bool: True if any LoRAs were applied, False otherwise
        """
        if not self.active_loras:
            return False
            
        # Make sure model is loaded
        if self.pipe is None:
            self.load_model()
            
        # Apply each active LoRA
        applied = False
        for lora_name, is_active, weight in self.active_loras:
            if is_active and lora_name in self.available_loras and lora_name != "None":
                lora_path = self.available_loras[lora_name]
                success = self.load_lora(lora_path, scale=weight)
                applied = applied or success
        
        return applied
    
    def set_active_loras(self, lora_settings: List[Tuple[str, bool, float]]) -> List[Tuple[str, bool, float]]:
        """
        Update the list of active LoRAs.
        
        Args:
            lora_settings: List of (lora_name, is_active, weight) tuples
            
        Returns:
            List of updated active LoRAs
        """
        self.active_loras = lora_settings
        return self.active_loras
    
    def enhance_prompt(
        self, 
        prompt: str, 
        attire_customization: str = "", 
        decor_customization: str = ""
    ) -> str:
        """
        Enhance the prompt to get better results.
        
        Args:
            prompt: User's input prompt
            attire_customization: Customization for clothes/attire
            decor_customization: Customization for background/scene
            
        Returns:
            Enhanced prompt
        """
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
        reference_image: Union[np.ndarray, Image.Image],
        pose_image: Union[np.ndarray, Image.Image],
        prompt: str = "",
        negative_prompt: str = "",
        strength: float = 0.75,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 30,
        seed: int = -1,
        width: Optional[int] = None,
        height: Optional[int] = None,
        keep_original_size: bool = True,
        num_images: int = 1,
        attire_customization: str = "",
        decor_customization: str = "",
        face_enhancement: Optional[bool] = None,
        enable_two_stage: Optional[bool] = None,
        refiner_model_name: str = "SDXL Refiner 1.0 (Default)",
        refiner_strength: Optional[float] = None,
        enable_cn_pose: bool = False,
        cn_strength: float = 1.0,
        enable_selective_face: bool = True,
        active_loras: Optional[List[Tuple[str, bool, float]]] = None,
        progress_callback = None,
    ) -> Tuple[Image.Image, List[int], List[str]]:
        """
        Generate a composite image based on reference and pose images.
        
        Args:
            reference_image: Reference image with the face to preserve
            pose_image: Pose image with the target pose
            prompt: Text prompt for generation
            negative_prompt: Negative text prompt
            strength: Transformation strength (0-1)
            guidance_scale: Classifier-free guidance scale
            num_inference_steps: Number of denoising steps
            seed: Random seed (-1 for random)
            width: Output width
            height: Output height
            keep_original_size: Whether to keep original image size
            num_images: Number of images to generate
            attire_customization: Customization for clothes/attire
            decor_customization: Customization for background/scene
            face_enhancement: Whether to enhance the face
            enable_two_stage: Whether to use two-stage generation
            refiner_model_name: Name of the refiner model
            refiner_strength: Strength of refinement
            enable_cn_pose: Whether to use ControlNet for pose guidance
            cn_strength: ControlNet influence strength
            enable_selective_face: Whether to enable selective face transfer
            active_loras: List of active LoRAs
            progress_callback: Callback function for progress reporting
            
        Returns:
            Tuple of (first_image, seeds, save_paths)
        """
        logger.info("Starting image generation")
        
        # Update face enhancement if provided
        if face_enhancement is not None:
            self.face_enhancement = face_enhancement
        
        # Update refiner settings if provided
        if enable_two_stage is not None:
            self.enable_two_stage = enable_two_stage
        if refiner_strength is not None:
            self.refiner_strength = refiner_strength
        
        # Results for multiple images
        results = []
        seeds = []
        save_paths = []
        
        # Set the initial seed
        if seed == -1:
            seed = torch.randint(0, 2**32 - 1, (1,)).item()
            logger.info(f"Generated random seed: {seed}")
        
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
            # Report progress
            if progress_callback:
                progress_callback((i / num_images) * 0.1)  # Initial 10% for preprocessing
            
            # Increment seed for each image in batch for variety
            current_seed = seed + i if i > 0 else seed
            generator = torch.Generator(self.device).manual_seed(current_seed)
            
            # Detect faces for quality improvement
            ref_faces = None
            pose_faces = None
            face_mask = None
            
            try:
                import face_recognition
                if self.face_enhancement:
                    # Convert to numpy arrays for face detection
                    ref_np = np.array(ref_img)
                    pose_np = np.array(pose_img)
                    
                    # Detect faces
                    ref_faces = self._detect_faces(ref_np)
                    pose_faces = self._detect_faces(pose_np)
                    
                    # Find the best match between faces
                    if ref_faces and pose_faces:
                        ref_face_loc, pose_face_loc = self._find_best_face_match(ref_faces, pose_faces)
                        
                        # Create a mask for the face area
                        face_mask = self._create_face_mask(pose_np, pose_face_loc, expand_ratio=1.8)
                        
                        logger.info("Face detection and matching successful")
            except ImportError:
                logger.warning("face_recognition not available, continuing without face detection")
            except Exception as e:
                logger.error(f"Face detection error: {e}")
            
            # Blend images with adaptive face preservation if available
            if face_mask is not None:
                blended_img = self._adaptive_blend_images(ref_img, pose_img, weight=0.3, face_mask=face_mask)
                logger.info("Applied adaptive blending with face mask")
            else:
                # Fallback to simple blending
                blended_img = self._adaptive_blend_images(ref_img, pose_img, weight=0.3)
                logger.info("Applied simple blending (no face mask)")

            # --- (Optional) selective feature transfer ----
            if enable_selective_face and face_mask is not None:
                try:
                    src_np = np.array(ref_img)
                    dst_np = np.array(blended_img)
                    transferred = face_utils.selective_clone(
                        src_np, dst_np,
                        parts=self.face_transfer_parts,
                        alpha=self.face_transfer_blend
                    )
                    blended_img = Image.fromarray(transferred)
                    logger.info("Applied selective face feature transfer")
                except Exception as e:
                    logger.error(f"Error in selective face transfer: {e}")
            
            # Use the enhanced prompt method
            enhanced_prompt = self.enhance_prompt(
                prompt, 
                attire_customization=attire_customization, 
                decor_customization=decor_customization
            )
            
            # Prepare default negative prompt if not provided
            if not negative_prompt:
                negative_prompt = config.DEFAULT_NEGATIVE_PROMPT
            
            # Report progress
            if progress_callback:
                progress_callback(0.1 + (i / num_images) * 0.1)  # 10% - 20% for preprocessing
            
            # Different parameters based on model type
            kwargs = {
                "image": blended_img,  # Use blended image as base
                "prompt": enhanced_prompt,
                "negative_prompt": negative_prompt,
                "strength": strength,
                "guidance_scale": guidance_scale,
                "num_inference_steps": num_inference_steps,
                "generator": generator,
            }
            
            # Add callback if provided
            if progress_callback:
                def callback_wrapper(step, timestep, latents):
                    progress = 0.2 + (i / num_images) * 0.6 * (step / num_inference_steps)
                    progress_callback(min(progress, 0.8))  # 20% - 80%
                    return
                
                kwargs["callback"] = callback_wrapper
                kwargs["callback_steps"] = 1
            
            # IP-Adapter setup for XL models
            # use_ip_adapter = False
            # if "xl" in self.current_model_id.lower():
            #     # Try to use IP-Adapter
            #     if self.ip_adapter.is_available:
            #         # Setup IP-Adapter
            #         setup_success = self.ip_adapter.setup(self.pipe, self.device)
            #         if setup_success:
            #             ip_image_embeds = self.ip_adapter.ip_adapter.encode_image(ref_img)
            #             kwargs["ip_adapter_image_embeds"] = ip_image_embeds
            #             use_ip_adapter = True
            #             logger.info("IP-Adapter activated for improved facial transfer")
            
            # Generate the image
            result = None
            
            # If user wants ControlNet Pose
            if enable_cn_pose:
                try:
                    # Initialize ControlNet if needed
                    if not hasattr(self.controlnet, 'pipe') or self.controlnet.pipe is None:
                        cn_setup_success = self.controlnet.setup(self.pipe, self.device)
                        if not cn_setup_success:
                            raise ValueError("Failed to set up ControlNet")
                    
                    # Detect pose in pose image
                    pose_detection = self.controlnet.detect_pose(pose_img)
                    if pose_detection is None:
                        raise ValueError("Failed to detect pose in image")
                    
                    # Generate with ControlNet
                    result = self.controlnet.generate(
                        prompt=enhanced_prompt,
                        negative_prompt=negative_prompt,
                        input_image=blended_img,
                        pose_image=pose_detection,
                        strength=strength,
                        guidance_scale=guidance_scale,
                        controlnet_conditioning_scale=float(cn_strength),
                        num_inference_steps=num_inference_steps,
                        generator=generator,
                        callback=kwargs.get("callback")
                    )
                    
                    logger.info("Successfully generated image with ControlNet Pose")
                except Exception as e:
                    logger.error(f"ControlNet error: {e}. Falling back to normal pipeline.")
                    result = self.pipe(**kwargs).images[0]
            else:
                # Normal pipeline
                result = self.pipe(**kwargs).images[0]
                logger.info("Successfully generated base image")
            
            # Report progress
            if progress_callback:
                progress_callback(0.8 + (i / num_images) * 0.1)  # 80% - 90%
            
            # -------------------- TWO-STAGE REFINEMENT --------------------
            if self.enable_two_stage:
                try:
                    from config import AVAILABLE_MODELS
                    # Find model ID from display name
                    refine_id = None
                    for model_id, name in AVAILABLE_MODELS.items():
                        if name == refiner_model_name:
                            refine_id = model_id
                            break
                    
                    if isinstance(refiner_model_name, (int, float)):
                        logger.warning(f"Refiner model name should be a string, not a number: {refiner_model_name}")
                        refiner_model_name = "SDXL Refiner 1.0 (Default)"

                    if not refine_id:
                        logger.warning(f"Refiner model '{refiner_model_name}' not found. Using default.")
                        refine_id = config.DEFAULT_REFINER_ID
                    
                    # Load refiner
                    refiner = self.load_refiner_model(refine_id)
                    if refiner is None:
                        raise ValueError(f"Failed to load refiner model: {refine_id}")

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
                    if progress_callback:
                        def refiner_callback(step, timestep, latents):
                            refiner_steps = int(num_inference_steps * 0.6)
                            progress = 0.9 + (i / num_images) * 0.1 * (step / refiner_steps)
                            progress_callback(min(progress, 1.0))  # 90% - 100%
                            return

                        refine_kwargs["callback"] = refiner_callback
                        refine_kwargs["callback_steps"] = 1

                    result = refiner(**refine_kwargs).images[0]
                    logger.info(f"Refinement complete with strength {self.refiner_strength}")
                except Exception as e:
                    logger.error(f"Refiner error: {e}. Using first-stage result.")
            # --------------------------------------------------------------
            
            # Add generated image to results
            results.append(result)
            seeds.append(current_seed)
            
            # Save the generated image
            if self.auto_save:
                save_path = io_utils.save_image(
                    result, 
                    current_seed, 
                    self.current_model_id,
                    self.outputs_dir,
                    self.save_format,
                    i, 
                    num_images
                )
                save_paths.append(save_path)
        
        # Ensure we have at least one result
        if not results:
            raise ValueError("No images were generated")
            
        # Report completion
        if progress_callback:
            progress_callback(1.0)  # 100%
        
        # Format results for return
        return results[0], seeds, save_paths
    
    def get_model_id_from_name(self, model_name: str) -> str:
        """
        Get the model ID from the display name.
        
        Args:
            model_name: Display name of the model
            
        Returns:
            Model ID or default model ID if not found
        """
        for model_id, name in config.AVAILABLE_MODELS.items():
            if name == model_name:
                return model_id
        return config.DEFAULT_MODEL_ID  # Return default if not found
    
    def download_model(self, model_id: str) -> str:
        """
        Download a specific model to the cache directory.
        
        Args:
            model_id: The model identifier from Hugging Face
            
        Returns:
            Status message
        """
        try:
            logger.info(f"Downloading model: {model_id}")
            snapshot_download(repo_id=model_id, cache_dir=self.cache_dir)
            return f"Downloaded model: {model_id}"
        except Exception as e:
            error_msg = f"Error downloading model {model_id}: {e}"
            logger.error(error_msg)
            return error_msg
    
    def toggle_auto_save(self, value: bool) -> str:
        """
        Toggle auto-save functionality.
        
        Args:
            value: Whether to enable auto-save
            
        Returns:
            Status message
        """
        self.auto_save = value
        logger.info(f"Auto-save {'enabled' if value else 'disabled'}")
        return f"Auto-save {'enabled' if value else 'disabled'}"
    
    def set_save_format(self, format: str) -> str:
        """
        Set the save format for images.
        
        Args:
            format: Image format (png, jpg, webp)
            
        Returns:
            Status message
        """
        if format in ['png', 'jpg', 'jpeg', 'webp']:
            self.save_format = format
            logger.info(f"Save format set to {format}")
            return f"Save format set to {format}"
        
        logger.warning(f"Unsupported format: {format}. Using {self.save_format}.")
        return f"Unsupported format: {format}. Using {self.save_format}."
    
    def set_face_enhancement(self, value: bool) -> str:
        """
        Toggle face enhancement.
        
        Args:
            value: Whether to enable face enhancement
            
        Returns:
            Status message
        """
        self.face_enhancement = value
        logger.info(f"Face enhancement {'enabled' if value else 'disabled'}")
        return f"Face enhancement {'enabled' if value else 'disabled'}"
    
    def set_preserve_face_strength(self, value: float) -> str:
        """
        Set the strength of face preservation.
        
        Args:
            value: Face preservation strength (0-1)
            
        Returns:
            Status message
        """
        self.face_alignment_weight = value
        logger.info(f"Face preservation strength set to {value}")
        return f"Face preservation strength set to {value}"
    
    def set_refiner_strength(self, value: float) -> str:
        """
        Set the strength of the refiner.
        
        Args:
            value: Refiner strength (0-1)
            
        Returns:
            Status message
        """
        self.refiner_strength = value
        logger.info(f"Refiner strength set to {value}")
        return f"Refiner strength set to {value}"
    
    def get_available_models(self) -> List[str]:
        """
        Get a list of available model display names.
        
        Returns:
            List of model display names
        """
        return list(config.AVAILABLE_MODELS.values())
    
    def get_available_loras(self) -> List[str]:
        """
        Get a list of available LoRA names.
        
        Returns:
            List of LoRA names
        """
        return list(self.available_loras.keys())
    
    def rescan_loras(self) -> Dict[str, str]:
        """
        Rescan the LoRAs directory.
        
        Returns:
            Dictionary of LoRA names to paths
        """
        self.available_loras = self._scan_loras()
        return self.available_loras