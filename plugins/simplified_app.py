import torch
import gradio as gr
from diffusers import StableDiffusionXLInpaintPipeline, StableDiffusionInpaintPipeline
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation, AutoModelForCausalLM, AutoTokenizer
from PIL import Image, ImageOps, ImageFilter, ImageDraw, ImageFont
import numpy as np
import cv2
import re
import json
import os
import base64
from io import BytesIO
from datetime import datetime
import uuid
import traceback

# Initialize device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Dictionary of available models
inpainting_models = {
    "SDXL Base": {
        "model_id": "stabilityai/stable-diffusion-xl-base-1.0",
        "is_xl": True,
        "variant": "fp16"
    },
    "SD 1.5": {
        "model_id": "runwayml/stable-diffusion-inpainting",
        "is_xl": False,
        "variant": None
    },
    "Realistic Vision": {
        "model_id": "SG161222/Realistic_Vision_V5.1_noVAE",
        "is_xl": False,
        "variant": None
    }
}

# Initialize model registry
model_registry = {}

# ======= DETECTION MODELS =======

# Initialize Segment Anything Model (SAM)
try:
    from segment_anything import sam_model_registry, SamPredictor
    
    # Check if CUDA is available for SAM
    sam_device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load SAM model
    sam_checkpoint = "sam_vit_h_4b8939.pth"  # Path to the SAM checkpoint
    model_type = "vit_h"
    
    if os.path.exists(sam_checkpoint):
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=sam_device)
        sam_predictor = SamPredictor(sam)
        sam_available = True
        print("SAM model loaded successfully")
    else:
        print(f"SAM checkpoint not found at {sam_checkpoint}")
        sam_available = False
except ImportError:
    print("SAM not available: segment_anything package not installed")
    sam_available = False
except Exception as e:
    print(f"Error loading SAM: {e}")
    sam_available = False

# Initialize YOLO for object detection
try:
    import ultralytics
    from ultralytics import YOLO
    
    # Load YOLOv8 model
    yolo_model = YOLO("yolov8n.pt")  # Using the nano model for speed
    yolo_available = True
    print("YOLO model loaded successfully")
except ImportError:
    print("YOLO not available: ultralytics package not installed")
    yolo_available = False
except Exception as e:
    print(f"Error loading YOLO: {e}")
    yolo_available = False

# Initialize MediaPipe for face and person detection
try:
    import mediapipe as mp
    mp_face_detection = mp.solutions.face_detection
    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils
    mp_selfie_segmentation = mp.solutions.selfie_segmentation
    mp_pose = mp.solutions.pose
    
    # Initialize models
    face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)
    selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)
    pose_detection = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)
    
    mediapipe_available = True
    print("MediaPipe models loaded successfully")
except ImportError:
    print("MediaPipe not available")
    mediapipe_available = False
except Exception as e:
    print(f"Error loading MediaPipe: {e}")
    mediapipe_available = False

# Initialize CLIPSeg for semantic segmentation
try:
    clipseg_processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
    clipseg_model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")
    if torch.cuda.is_available():
        clipseg_model = clipseg_model.to(device)
    clipseg_available = True
    print("CLIPSeg model loaded successfully")
except Exception as e:
    print(f"Error loading CLIPSeg: {e}")
    clipseg_available = False

# Load Mistral for advanced understanding
print("Loading Mistral for prompt understanding...")
mistral_available = False
try:
    mistral_tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
    mistral_model = AutoModelForCausalLM.from_pretrained(
        "mistralai/Mistral-7B-Instruct-v0.1", 
        torch_dtype=torch.float16,
        load_in_8bit=True
    )
    if torch.cuda.is_available():
        mistral_model = mistral_model.to(device)
    mistral_available = True
    print("Mistral model loaded successfully")
except Exception as e:
    print(f"Error loading Mistral: {e}")
    traceback.print_exc()
    print("Using fallback classification methods")

# ======= OPERATION CLASSIFICATION =======

# Enhanced operation classification using Mistral and pattern matching
def classify_operation(prompt):
    """Classify the editing operation using Mistral if available or rules-based approach as fallback"""
    if mistral_available:
        try:
            mistral_result = classify_with_mistral(prompt)
            if mistral_result:
                print(f"Mistral classification: {mistral_result}")
                return mistral_result
        except Exception as e:
            print(f"Mistral classification failed: {e}")
            traceback.print_exc()
    
    # Fall back to pattern-based classification
    return pattern_based_classify(prompt)

# Mistral-based classification
def classify_with_mistral(prompt):
    """Use Mistral to understand the editing operation in the prompt"""
    # Define the system prompt for Mistral
    system_prompt = """
    I need you to analyze an image editing instruction and classify it into the appropriate operation type.
    Return your response in this exact JSON format:
    {
        "type": "operation type (one of: color, remove, background, text, style, replace)",
        "target": "what is being edited (e.g., hair, background, watermark)",
        "attribute": "the new property or what it's being changed to (e.g., red, beach, cat)",
        "confidence": a number between 0.0 and 1.0 representing your confidence
    }
    
    Some examples:
    - "change hair color to red" -> {"type": "color", "target": "hair", "attribute": "red", "confidence": 0.9}
    - "remove watermark" -> {"type": "remove", "target": "watermark", "attribute": null, "confidence": 0.95}
    - "change background to beach" -> {"type": "background", "target": "background", "attribute": "beach", "confidence": 0.9}
    - "add text Hello World" -> {"type": "text", "target": "image", "attribute": "Hello World", "confidence": 0.9}
    - "convert to watercolor painting style" -> {"type": "style", "target": "image", "attribute": "watercolor", "confidence": 0.8}
    - "replace person with cat" -> {"type": "replace", "target": "person", "attribute": "cat", "confidence": 0.9}
    
    Be very precise about the target and attribute, especially for replacement operations.
    """
    
    # Create the prompt for Mistral
    full_prompt = f"<s>[INST] {system_prompt}\n\nEdit instruction: {prompt}\n\nRespond only with the JSON. [/INST]"
    
    # Generate response from Mistral
    inputs = mistral_tokenizer(full_prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = mistral_model.generate(
            inputs["input_ids"],
            max_new_tokens=200,
            temperature=0.1,
            top_p=0.9,
            repetition_penalty=1.1
        )
    
    response_text = mistral_tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "[/INST]" in response_text:
        response_text = response_text.split("[/INST]")[-1].strip()
    
    # Extract JSON from the response
    if "{" in response_text and "}" in response_text:
        json_start = response_text.find("{")
        json_end = response_text.rfind("}") + 1
        json_str = response_text[json_start:json_end]
        
        try:
            result = json.loads(json_str)
            # Validate result has required fields
            required_fields = ["type", "target", "confidence"]
            if all(field in result for field in required_fields):
                # Ensure attribute is never undefined
                if "attribute" not in result:
                    result["attribute"] = None
                return result
        except json.JSONDecodeError:
            print(f"Failed to parse Mistral JSON: {json_str}")
    
    # Return None to indicate failure, will fall back to pattern matching
    return None

# Pattern-based classification
def pattern_based_classify(prompt):
    """Classify the editing operation using pattern matching"""
    prompt = prompt.lower().strip()
    
    # Pattern mapping for operations
    patterns = [
        # Watermark removal
        (r"remove\s+(?:the\s+)?watermark", {
            "type": "remove",
            "target": "watermark",
            "attribute": None,
            "confidence": 0.95
        }),
        
        # Text removal
        (r"remove\s+(?:the\s+)?text", {
            "type": "remove",
            "target": "text",
            "attribute": None,
            "confidence": 0.95
        }),
        
        # Hair color change
        (r"(?:change|make|turn|dye|color)\s+(?:the\s+)?(?:my\s+)?(?:her\s+)?hair\s+(?:color\s+)?(?:to\s+)?(\w+)", {
            "type": "color",
            "target": "hair",
            "attribute": lambda match: match.group(1),
            "confidence": 0.9
        }),
        
        # Background change
        (r"(?:change|make|turn)\s+(?:the\s+)?background\s+(?:to\s+)?(.+)", {
            "type": "background",
            "target": "background",
            "attribute": lambda match: match.group(1),
            "confidence": 0.85
        }),
        
        # Style transfer
        (r"(?:change|make|turn|convert)(?:\s+image|\s+it|\s+photo)?\s+(?:to\s+)?(?:a\s+)?(\w+)(?:\s+style|\s+painting)", {
            "type": "style",
            "target": "image",
            "attribute": lambda match: match.group(1),
            "confidence": 0.85
        }),
        
        # Add text
        (r"add\s+text\s+(.+)", {
            "type": "text",
            "target": "image",
            "attribute": lambda match: match.group(1),
            "confidence": 0.85
        }),
        
        # Replace person/woman/man with something
        (r"replace\s+(?:the\s+)?(?:person|woman|man|girl|boy)\s+with\s+(?:a\s+)?(\w+)", {
            "type": "replace",
            "target": "person",
            "attribute": lambda match: match.group(1),
            "confidence": 0.9
        }),
        
        # General replacement
        (r"replace\s+(?:the\s+)?(\w+(?:\s+\w+)?)\s+with\s+(?:a\s+)?(\w+(?:\s+\w+)?)", {
            "type": "replace",
            "target": lambda match: match.group(1),
            "attribute": lambda match: match.group(2),
            "confidence": 0.8
        }),
        
        # General removal
        (r"remove\s+(?:the\s+)?(\w+(?:\s+\w+)?)(?:\s+from.+)?", {
            "type": "remove",
            "target": lambda match: match.group(1),
            "attribute": None,
            "confidence": 0.8
        }),
        
        # General color change
        (r"(?:change|make|turn)\s+(?:the\s+)?(\w+(?:\s+\w+)?)\s+(?:color\s+)?(?:to\s+)?(\w+)", {
            "type": "color",
            "target": lambda match: match.group(1),
            "attribute": lambda match: match.group(2),
            "confidence": 0.7
        })
    ]
    
    # Check each pattern
    for pattern, template in patterns:
        match = re.search(pattern, prompt)
        if match:
            result = template.copy()
            
            # Process dynamic attributes/targets
            for key in ["target", "attribute"]:
                if key in result and callable(result[key]):
                    try:
                        result[key] = result[key](match)
                    except:
                        result[key] = None
            
            return result
    
    # Check for special cases using keyword matching
    if "hair" in prompt and any(color in prompt for color in ["red", "blue", "green", "blonde", "black", "purple", "pink", "orange", "brown", "white", "gray", "grey"]):
        # Find the color word
        colors = ["red", "blue", "green", "blonde", "black", "purple", "pink", "orange", "brown", "white", "gray", "grey"]
        attribute = next((color for color in colors if color in prompt), "blue")
        return {"type": "color", "target": "hair", "attribute": attribute, "confidence": 0.8}
    
    # Default to general operation
    return {
        "type": "general",
        "target": "image",
        "attribute": prompt,
        "confidence": 0.5
    }

# ======= ADVANCED MASK GENERATION =======

# Detect objects using YOLO
def detect_objects_with_yolo(image, target_class=None):
    """Detect objects in an image using YOLOv8"""
    if not yolo_available:
        return None
    
    try:
        # Convert PIL image to numpy array if needed
        if isinstance(image, Image.Image):
            image_np = np.array(image)
        else:
            image_np = image
            
        # Run YOLO detection
        results = yolo_model(image_np)
        
        # Process results
        height, width = image_np.shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)
        
        # Set default classes to detect
        target_classes = []
        if target_class:
            if target_class.lower() in ["person", "human", "man", "woman", "girl", "boy"]:
                target_classes = ["person"]
            elif target_class.lower() in ["cat", "kitten"]:
                target_classes = ["cat"]
            elif target_class.lower() in ["dog", "puppy"]:
                target_classes = ["dog"]
            elif target_class.lower() in ["car", "vehicle", "auto"]:
                target_classes = ["car"]
            # Add more mappings as needed
        
        # Process detected objects
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Get class ID and name
                class_id = int(box.cls.item())
                class_name = r.names[class_id]
                
                # If target classes specified, filter
                if target_classes and class_name.lower() not in target_classes:
                    continue
                
                # Get bounding box
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Draw filled rectangle on mask
                cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
                
        return mask
    except Exception as e:
        print(f"Error in YOLO detection: {e}")
        traceback.print_exc()
        return None

# Generate semantic segmentation mask using CLIPSeg
def generate_semantic_mask(image, text_prompt):
    """Generate a mask based on text prompt using CLIPSeg"""
    if not clipseg_available:
        return None
    
    try:
        # Create different variations of the prompt for better results
        variations = [
            text_prompt,
            f"the {text_prompt}",
            f"a {text_prompt}",
            f"{text_prompt} in the image"
        ]
        
        # Try each variation
        best_mask = None
        best_score = 0
        
        for prompt in variations:
            # Process with CLIPSeg
            inputs = clipseg_processor(text=prompt, images=[image], return_tensors="pt").to(device)
            
            with torch.no_grad():
                outputs = clipseg_model(**inputs)
                preds = torch.sigmoid(outputs.logits)
                
                # Check shape
                if preds.shape[0] > 0 and preds.shape[1] > 0:
                    pred = preds[0][0]
                else:
                    continue
            
            # Set threshold based on prompt
            threshold = 0.3  # Default threshold
            
            # Convert to mask
            mask_array = (pred > threshold).cpu().numpy().astype(np.uint8) * 255
            
            # Calculate quality score (basic heuristic)
            pixel_sum = np.sum(mask_array) / 255
            total_pixels = mask_array.shape[0] * mask_array.shape[1]
            
            # Skip if mask is too small or too large
            if pixel_sum < 100 or pixel_sum > (total_pixels * 0.9):
                continue
                
            score = pixel_sum
            
            if score > best_score:
                best_score = score
                best_mask = mask_array
        
        return best_mask
    except Exception as e:
        print(f"Error in CLIPSeg segmentation: {e}")
        traceback.print_exc()
        return None

# Segment person using MediaPipe
def segment_person_with_mediapipe(image):
    """Segment person in the image using MediaPipe"""
    if not mediapipe_available:
        return None
    
    try:
        # Convert PIL image to numpy array if needed
        if isinstance(image, Image.Image):
            image_np = np.array(image)
        else:
            image_np = image
            
        # Convert to RGB for MediaPipe
        if len(image_np.shape) == 2:  # Grayscale
            image_rgb = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
        elif image_np.shape[2] == 4:  # RGBA
            image_rgb = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)
        else:
            image_rgb = image_np
            
        # Process with selfie segmentation
        results = selfie_segmentation.process(image_rgb)
        
        # Get segmentation mask
        segmentation_mask = results.segmentation_mask
        
        # Convert to binary mask
        height, width = segmentation_mask.shape
        binary_mask = np.zeros((height, width), dtype=np.uint8)
        binary_mask[segmentation_mask > 0.5] = 255
        
        return binary_mask
    except Exception as e:
        print(f"Error in MediaPipe segmentation: {e}")
        traceback.print_exc()
        return None

# Detect faces and body parts with MediaPipe
def detect_face_and_body(image):
    """Detect face and body parts in the image"""
    if not mediapipe_available:
        return {}
    
    try:
        # Convert PIL image to numpy array if needed
        if isinstance(image, Image.Image):
            image_np = np.array(image)
        else:
            image_np = image
            
        # Convert to RGB for MediaPipe
        if len(image_np.shape) == 2:  # Grayscale
            image_rgb = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
        elif image_np.shape[2] == 4:  # RGBA
            image_rgb = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)
        else:
            image_rgb = image_np
            
        height, width = image_rgb.shape[:2]
        
        # Initialize results
        results = {
            "face": np.zeros((height, width), dtype=np.uint8),
            "hair": np.zeros((height, width), dtype=np.uint8),
            "body": np.zeros((height, width), dtype=np.uint8),
            "face_landmarks": None
        }
        
        # Face detection
        face_results = face_detection.process(image_rgb)
        if face_results.detections:
            for detection in face_results.detections:
                bbox = detection.location_data.relative_bounding_box
                
                # Convert relative coordinates to absolute
                x = int(bbox.xmin * width)
                y = int(bbox.ymin * height)
                w = int(bbox.width * width)
                h = int(bbox.height * height)
                
                # Create face mask
                face_mask = np.zeros((height, width), dtype=np.uint8)
                cv2.rectangle(face_mask, (x, y), (x + w, y + h), 255, -1)
                
                # Hair mask (top portion of face)
                hair_mask = np.zeros((height, width), dtype=np.uint8)
                hair_y = max(0, y - int(h * 0.5))
                hair_h = int(h * 0.7)
                cv2.rectangle(hair_mask, (x, hair_y), (x + w, y + int(h * 0.2)), 255, -1)
                
                results["face"] = face_mask
                results["hair"] = hair_mask
                break  # Process only the first face
        
        # Body detection using pose
        pose_results = pose_detection.process(image_rgb)
        if pose_results.pose_landmarks:
            body_mask = np.zeros((height, width), dtype=np.uint8)
            landmarks = pose_results.pose_landmarks.landmark
            
            # Get all points
            points = []
            for landmark in landmarks:
                x, y = int(landmark.x * width), int(landmark.y * height)
                points.append((x, y))
            
            # Create convex hull
            if points:
                points = np.array(points)
                hull = cv2.convexHull(points)
                cv2.fillConvexPoly(body_mask, hull, 255)
            
            results["body"] = body_mask
            
        # Face mesh for detailed face parts
        mesh_results = face_mesh.process(image_rgb)
        if mesh_results.multi_face_landmarks:
            results["face_landmarks"] = mesh_results.multi_face_landmarks[0]
        
        return results
    except Exception as e:
        print(f"Error in face and body detection: {e}")
        traceback.print_exc()
        return {}

# Function to create specialized masks based on advanced detection
def create_advanced_mask(image, operation):
    """Create mask using advanced detection models based on operation type"""
    operation_type = operation["type"]
    target = operation["target"]
    attribute = operation["attribute"]
    
    # Convert PIL to numpy if needed
    if isinstance(image, Image.Image):
        image_np = np.array(image)
    else:
        image_np = image.copy()
        image = Image.fromarray(image_np)
    
    height, width = image_np.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    
    # PERSON REPLACEMENT
    if operation_type == "replace" and target and any(word in str(target).lower() for word in ["person", "woman", "man", "girl", "boy"]):
        # Try YOLO detection first
        yolo_mask = detect_objects_with_yolo(image, "person")
        if yolo_mask is not None and np.sum(yolo_mask) > 1000:
            mask = yolo_mask
        else:
            # Try MediaPipe person segmentation
            mp_mask = segment_person_with_mediapipe(image)
            if mp_mask is not None and np.sum(mp_mask) > 1000:
                mask = mp_mask
            else:
                # Fallback to detecting face and body
                parts = detect_face_and_body(image)
                if "body" in parts and np.sum(parts["body"]) > 1000:
                    mask = parts["body"]
                elif "face" in parts and np.sum(parts["face"]) > 1000:
                    # Expand face mask to estimate person
                    face_mask = parts["face"]
                    x, y, w, h = cv2.boundingRect(face_mask)
                    # Create a larger box for the person
                    person_w = int(w * 2.5)
                    person_h = int(h * 4)
                    person_x = max(0, x - (person_w - w) // 2)
                    person_y = max(0, y - int(h * 0.2))
                    cv2.rectangle(mask, (person_x, person_y), 
                                 (min(width, person_x + person_w), min(height, person_y + person_h)), 
                                 255, -1)
                else:
                    # Last resort: center of the image
                    center_x, center_y = width // 2, height // 2
                    person_w = int(width * 0.6)
                    person_h = int(height * 0.8)
                    cv2.rectangle(mask, 
                                 (center_x - person_w // 2, center_y - person_h // 2), 
                                 (center_x + person_w // 2, center_y + person_h // 2), 
                                 255, -1)
    
    # BACKGROUND CHANGE
    elif operation_type == "background" or (target and "background" in str(target).lower()):
        # Try to segment foreground person
        mp_mask = segment_person_with_mediapipe(image)
        if mp_mask is not None and np.sum(mp_mask) > 1000:
            # Invert to get background
            mask = 255 - mp_mask
        else:
            # Try YOLO to detect any objects
            yolo_mask = detect_objects_with_yolo(image)
            if yolo_mask is not None and np.sum(yolo_mask) > 1000:
                # Invert to get background
                mask = 255 - yolo_mask
            else:
                # Try CLIPSeg to segment based on prompt
                clip_mask = generate_semantic_mask(image, "background")
                if clip_mask is not None and np.sum(clip_mask) > 1000:
                    mask = clip_mask
                else:
                    # Fallback: assume foreground in center
                    center_x, center_y = width // 2, height // 2
                    foreground_mask = np.zeros((height, width), dtype=np.uint8)
                    axes = (int(width * 0.3), int(height * 0.6))
                    cv2.ellipse(foreground_mask, (center_x, center_y), axes, 0, 0, 360, 255, -1)
                    mask = 255 - foreground_mask
    
    # HAIR COLOR CHANGE
    elif operation_type == "color" and target and "hair" in str(target).lower():
        # Try CLIPSeg for hair segmentation
        clip_mask = generate_semantic_mask(image, "hair")
        if clip_mask is not None and np.sum(clip_mask) > 1000:
            mask = clip_mask
        else:
            # Try to get hair region from face detection
            parts = detect_face_and_body(image)
            if "hair" in parts and np.sum(parts["hair"]) > 1000:
                mask = parts["hair"]
            else:
                # Fallback to top of the head
                parts = detect_face_and_body(image)
                if "face" in parts and np.sum(parts["face"]) > 1000:
                    face_mask = parts["face"]
                    x, y, w, h = cv2.boundingRect(face_mask)
                    # Create a hair region above the face
                    hair_y = max(0, y - int(h * 0.5))
                    hair_h = int(h * 0.6)
                    hair_w = int(w * 1.2)
                    hair_x = max(0, x - (hair_w - w) // 2)
                    cv2.ellipse(mask, 
                               (x + w // 2, hair_y + hair_h // 2),
                               (hair_w // 2, hair_h // 2),
                               0, 0, 360, 255, -1)
                else:
                    # Last resort: top of the image
                    hair_height = int(height * 0.3)
                    hair_width = int(width * 0.5)
                    cv2.ellipse(mask, 
                               (width // 2, int(height * 0.15)),
                               (hair_width // 2, hair_height // 2),
                               0, 0, 360, 255, -1)
    
    # WATERMARK REMOVAL
    elif operation_type == "remove" and target and "watermark" in str(target).lower():
        # Try CLIPSeg first
        clip_mask = generate_semantic_mask(image, "watermark")
        if clip_mask is not None and np.sum(clip_mask) > 1000:
            mask = clip_mask
        else:
            # Fallback to bottom of image
            watermark_height = int(height * 0.18)
            mask[height - watermark_height:, :] = 255
            # Add corners for logo watermarks
            corner_size = int(min(width, height) * 0.25)
            mask[height - corner_size:, :corner_size] = 255
            mask[height - corner_size:, width - corner_size:] = 255
    
    # TEXT REMOVAL
    elif operation_type == "remove" and target and "text" in str(target).lower():
        # Try CLIPSeg first
        clip_mask = generate_semantic_mask(image, "text")
        if clip_mask is not None and np.sum(clip_mask) > 1000:
            mask = clip_mask
        else:
            # Fallback to bottom where text often appears
            text_height = int(height * 0.18)
            mask[height - text_height:, :] = 255
    
    # STYLE TRANSFER
    elif operation_type == "style":
        # For style transfer, use the entire image
        mask.fill(255)
    
    # OTHER COLOR CHANGES
    elif operation_type == "color":
        if target:
            # Try to segment the target object using CLIPSeg
            clip_mask = generate_semantic_mask(image, str(target))
            if clip_mask is not None and np.sum(clip_mask) > 1000:
                mask = clip_mask
            else:
                # Fallback based on target
                if any(word in str(target).lower() for word in ["eye", "eyes"]):
                    parts = detect_face_and_body(image)
                    if "face" in parts and np.sum(parts["face"]) > 1000:
                        face_mask = parts["face"]
                        x, y, w, h = cv2.boundingRect(face_mask)
                        eye_y = y + int(h * 0.25)
                        eye_h = int(h * 0.15)
                        cv2.rectangle(mask, (x + int(w*0.1), eye_y), 
                                     (x + int(w*0.9), eye_y + eye_h), 255, -1)
                    else:
                        # Fallback for eyes
                        eye_y = int(height * 0.3)
                        eye_x_offset = int(width * 0.15)
                        eye_radius = int(width * 0.07)
                        cv2.circle(mask, (width//2 - eye_x_offset, eye_y), eye_radius, 255, -1)
                        cv2.circle(mask, (width//2 + eye_x_offset, eye_y), eye_radius, 255, -1)
                elif any(word in str(target).lower() for word in ["lip", "lips", "mouth"]):
                    parts = detect_face_and_body(image)
                    if "face" in parts and np.sum(parts["face"]) > 1000:
                        face_mask = parts["face"]
                        x, y, w, h = cv2.boundingRect(face_mask)
                        lip_y = y + int(h * 0.7)
                        lip_h = int(h * 0.15)
                        lip_w = int(w * 0.7)
                        cv2.ellipse(mask, (x + w//2, lip_y), (lip_w//2, lip_h//2), 0, 0, 360, 255, -1)
                    else:
                        # Fallback for lips
                        lip_y = int(height * 0.6)
                        lip_w = int(width * 0.25)
                        lip_h = int(height * 0.08)
                        cv2.ellipse(mask, (width//2, lip_y), (lip_w//2, lip_h//2), 0, 0, 360, 255, -1)
                else:
                    # Default to center of image
                    center_x, center_y = width // 2, height // 2
                    radius = min(width, height) // 4
                    cv2.circle(mask, (center_x, center_y), radius, 255, -1)
        else:
            # Default mask for center of image
            center_x, center_y = width // 2, height // 2
            radius = min(width, height) // 4
            cv2.circle(mask, (center_x, center_y), radius, 255, -1)
    
    # GENERAL OBJECT REMOVAL
    elif operation_type == "remove":
        # Try to use YOLO to detect object if it's a common class
        yolo_mask = None
        if target:
            yolo_mask = detect_objects_with_yolo(image, str(target))
        
        if yolo_mask is not None and np.sum(yolo_mask) > 1000:
            mask = yolo_mask
        else:
            # Try CLIPSeg
            if target:
                clip_mask = generate_semantic_mask(image, str(target))
                if clip_mask is not None and np.sum(clip_mask) > 1000:
                    mask = clip_mask
                else:
                    # Default to center of image
                    center_x, center_y = width // 2, height // 2
                    radius = min(width, height) // 4
                    cv2.circle(mask, (center_x, center_y), radius, 255, -1)
            else:
                # No target specified - use center
                center_x, center_y = width // 2, height // 2
                radius = min(width, height) // 4
                cv2.circle(mask, (center_x, center_y), radius, 255, -1)
    
    # Default for unrecognized operations
    else:
        # Use the center of the image
        center_x, center_y = width // 2, height // 2
        radius = min(width, height) // 4
        cv2.circle(mask, (center_x, center_y), radius, 255, -1)
    
    # Apply smoothing and dilation to improve the mask
    kernel_size = 15
    if operation_type == "replace" and target and any(word in str(target).lower() for word in ["person", "woman", "man", "girl", "boy"]):
        # For person replacement, use a larger kernel and stronger dilation
        kernel_size = 25
    
    # Create kernel and apply morphological operations
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)
    mask = cv2.GaussianBlur(mask, (21, 21), 0)
    
    # Convert to PIL Image
    mask_image = Image.fromarray(mask)
    return mask_image

# Function to add text to an image
def add_text_to_image(image, text, position="bottom", color="white"):
    """Add text overlay to an image"""
    try:
        img = image.copy()
        draw = ImageDraw.Draw(img)
        
        # Set font size based on image
        size = max(24, min(img.width, img.height) // 12)
        
        # Default font
        try:
            font = ImageFont.load_default()
        except:
            font = None
        
        # Color mapping
        color_map = {
            "white": (255, 255, 255),
            "black": (0, 0, 0),
            "red": (255, 0, 0),
            "blue": (0, 0, 255),
            "green": (0, 255, 0),
            "yellow": (255, 255, 0)
        }
        text_color = color_map.get(color.lower(), (255, 255, 255))
        
        # Calculate position
        text_width = size * len(text) * 0.6
        text_height = size * 1.2
        
        if position == "top":
            text_position = ((img.width - text_width) // 2, img.height * 0.05)
        elif position == "center":
            text_position = ((img.width - text_width) // 2, (img.height - text_height) // 2)
        else:  # Default to bottom
            text_position = ((img.width - text_width) // 2, img.height * 0.85)
        
        # Add background for readability
        rect_padding = size // 2
        rect_coords = (
            text_position[0] - rect_padding,
            text_position[1] - rect_padding,
            text_position[0] + text_width + rect_padding,
            text_position[1] + text_height + rect_padding
        )
        draw.rectangle(rect_coords, fill=(0, 0, 0, 128))
        
        # Draw text
        draw.text(text_position, text, font=font, fill=text_color)
        
        return img
    except Exception as e:
        print(f"Error adding text: {e}")
        return image

# ======= PROMPT GENERATION =======

# Generate enhanced prompts for Stable Diffusion
def generate_enhanced_prompt(operation):
    """Create a detailed prompt for Stable Diffusion based on the operation"""
    operation_type = operation["type"]
    target = operation["target"] or ""
    attribute = operation["attribute"] or ""
    
    if isinstance(target, str):
        target = target.lower()
    if isinstance(attribute, str):
        attribute = attribute.lower()
    
    # WATERMARK REMOVAL
    if operation_type == "remove" and "watermark" in str(target):
        return "clean image without watermark, seamless continuation of the image, consistent background and texture, detailed, no text"
    
    # TEXT REMOVAL
    elif operation_type == "remove" and "text" in str(target):
        return "clean image without text, seamless continuation of the image, consistent background and texture, detailed"
    
    # PERSON REPLACEMENT
    elif operation_type == "replace" and any(word in str(target) for word in ["person", "woman", "man", "girl", "boy"]):
        if attribute and "cat" in str(attribute):
            return "a highly detailed photorealistic cat, striped tabby cat with orange and brown fur, whiskers, realistic cat eyes, feline features, cat nose, detailed fur texture, natural feline pose, lifelike domestic cat, professional photography of cat, 4k, ultra realistic"
        elif attribute:
            return f"a highly detailed photorealistic {attribute}, ultra detailed, professional photography of {attribute}, natural {attribute}, realistic {attribute}, detailed texture, 4k quality, cinematic lighting"
        else:
            return "a detailed photorealistic animal, high resolution, detailed texture"
    
    # HAIR COLOR CHANGE
    elif operation_type == "color" and "hair" in str(target):
        return f"detailed {attribute} colored hair, natural {attribute} hair, realistic {attribute} hair texture, shiny {attribute} hair, vibrant {attribute} hair, professionally styled {attribute} hair"
    
    # BACKGROUND CHANGE
    elif operation_type == "background" or "background" in str(target):
        return f"high quality detailed {attribute} background, photorealistic {attribute} scene, detailed {attribute} environment, professional photography of {attribute}, beautiful {attribute} setting, 4k resolution {attribute} background, ultra realistic"
    
    # STYLE TRANSFER
    elif operation_type == "style":
        return f"artwork in {attribute} style, professional {attribute} artwork, detailed {attribute} style, high quality {attribute} art, artistic {attribute} rendering, beautiful {attribute} treatment"
    
    # OBJECT REMOVAL
    elif operation_type == "remove":
        return f"clean image without {target}, seamless continuation, consistent background and texture, detailed, natural"
    
    # COLOR CHANGE
    elif operation_type == "color":
        return f"detailed {attribute} colored {target}, natural {attribute} {target}, realistic {attribute} color, vibrant {attribute} tone"
    
    # GENERAL REPLACEMENT
    elif operation_type == "replace":
        return f"highly detailed photorealistic {attribute}, natural {attribute}, professional photography of {attribute}, detailed texture, high resolution"
    
    # DEFAULT
    else:
        if attribute:
            return f"photorealistic {attribute}, detailed, high quality, professional photography"
        else:
            return "photorealistic, detailed, high quality image, professional photography"

# ======= MODEL LOADING =======

# Load Stable Diffusion model
def load_model(model_name):
    global model_registry
    
    if model_name in model_registry:
        return model_registry[model_name]
    
    print(f"Loading model: {model_name}")
    model_info = inpainting_models[model_name]
    
    try:
        if model_info["is_xl"]:
            model = StableDiffusionXLInpaintPipeline.from_pretrained(
                model_info["model_id"],
                torch_dtype=torch.float16,
                variant=model_info["variant"]
            ).to(device)
        else:
            model = StableDiffusionInpaintPipeline.from_pretrained(
                model_info["model_id"],
                torch_dtype=torch.float16
            ).to(device)
        
        model_registry[model_name] = model
        return model
    except Exception as e:
        print(f"Error loading model {model_name}: {e}")
        traceback.print_exc()
        # Fallback to existing model or create a new one
        if "SDXL Base" in model_registry:
            return model_registry["SDXL Base"]
        elif "SD 1.5" in model_registry:
            return model_registry["SD 1.5"]
        else:
            # Last resort - try to load SDXL
            model = StableDiffusionXLInpaintPipeline.from_pretrained(
                "stabilityai/stable-diffusion-xl-base-1.0",
                torch_dtype=torch.float16
            ).to(device)
            model_registry["SDXL Base"] = model
            return model

# ======= MAIN EDIT FUNCTION =======

# Enhanced image editing function
def edit_image(image, prompt, model_name="SDXL Base", steps=30, guidance_scale=7.5, 
              strength=0.7, negative_prompt=""):
    """Edit an image based on the natural language prompt"""
    try:
        if not isinstance(image, Image.Image):
            raise ValueError("Invalid image")
            
        # Load model
        pipe = load_model(model_name)
        
        # Prepare image for processing
        width, height = image.size
        new_width = width - (width % 8)
        new_height = height - (height % 8)
        
        # Keep within model limits
        new_width = max(64, min(new_width, 1024))
        new_height = max(64, min(new_height, 1024))
        
        # Resize for processing
        original_size = image.size
        resized_image = image.resize((new_width, new_height))
        
        # Classify the operation
        operation = classify_operation(prompt)
        print(f"Classified operation: {operation}")
        
        # Handle text overlay specially (doesn't need SD model)
        if operation["type"] == "text":
            text = operation["attribute"] or "Sample Text"
            position = "bottom"
            color = "white"
            
            # Check for position info in prompt
            if "top" in prompt.lower():
                position = "top"
            elif "center" in prompt.lower():
                position = "center"
                
            # Check for color info in prompt
            for color_name in ["white", "black", "red", "blue", "green", "yellow"]:
                if color_name in prompt.lower():
                    color = color_name
                    break
                    
            # Add text and return
            result = add_text_to_image(resized_image, text, position, color)
            
            # Resize back to original if needed
            if original_size != result.size:
                result = result.resize(original_size, Image.LANCZOS)
                
            return result, None
        
        # Generate advanced mask using advanced detection
        mask = create_advanced_mask(resized_image, operation)
        
        # Generate enhanced prompt for Stable Diffusion
        sd_prompt = generate_enhanced_prompt(operation)
        print(f"SD prompt: {sd_prompt}")
        
        # Special handling for person replacement - increase strength
        if operation["type"] == "replace" and any(word in str(operation.get("target", "")).lower() for word in ["person", "woman", "man", "girl", "boy"]):
            # Increase strength for better replacement
            strength = min(0.85, strength + 0.15)
        
        # Set default negative prompt if not provided
        if not negative_prompt or negative_prompt.strip() == "":
            negative_prompt = "deformed, distorted, disfigured, poor quality, bad anatomy, watermark, signature, blurry, unrealistic, low resolution, bad proportions, duplicate"
        
        # For better control, adjust parameters based on operation type
        if operation["type"] == "replace":
            # For replacement, we need stronger guidance
            guidance_scale = max(guidance_scale, 8.0)
        elif operation["type"] in ["remove", "background"]:
            # For removal/background, we need better consistency
            negative_prompt += ", inconsistent background, seams, edges"
        
        # Two-pass approach for better results for complex operations
        if operation["type"] in ["replace", "background"] or (
            operation["type"] == "remove" and any(word in str(operation.get("target", "")).lower() for word in ["watermark", "text"])):
            
            # First pass - standard inpainting
            first_result = pipe(
                prompt=sd_prompt,
                negative_prompt=negative_prompt,
                image=resized_image,
                mask_image=mask,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                strength=strength
            ).images[0]
            
            # Second pass - refine edges
            try:
                # Create a boundary mask for refining edges
                mask_np = np.array(mask)
                if mask_np.max() > 1:  # Scale to 0-1 if needed
                    mask_np = mask_np / 255.0
                
                kernel = np.ones((11, 11), np.uint8)
                dilated = cv2.dilate((mask_np * 255).astype(np.uint8), kernel, iterations=2)
                eroded = cv2.erode((mask_np * 255).astype(np.uint8), kernel, iterations=2)
                boundary_mask = dilated - eroded
                
                # Add feathering
                boundary_mask = cv2.GaussianBlur(boundary_mask, (15, 15), 0)
                boundary_mask = Image.fromarray(boundary_mask)
                
                # Second pass with lower strength for refinement
                result = pipe(
                    prompt=sd_prompt + ", seamless integration, consistent lighting and texture",
                    negative_prompt=negative_prompt + ", seam, boundary, inconsistent edges",
                    image=first_result,
                    mask_image=boundary_mask,
                    num_inference_steps=min(steps, 20),  # Lower steps for refinement
                    guidance_scale=guidance_scale * 0.9,  # Slightly lower guidance
                    strength=strength * 0.4  # Much lower strength for subtle refinement
                ).images[0]
            except Exception as e:
                print(f"Error in second pass: {e}")
                result = first_result  # Fallback to first result
        else:
            # Standard single-pass inpainting for other operations
            result = pipe(
                prompt=sd_prompt,
                negative_prompt=negative_prompt,
                image=resized_image,
                mask_image=mask,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                strength=strength
            ).images[0]
        
        # Resize back to original if needed
        if original_size != result.size:
            result = result.resize(original_size, Image.LANCZOS)
            
        return result, mask
    
    except Exception as e:
        print(f"Error editing image: {e}")
        traceback.print_exc()
        # Return original image if error
        return image, None

# ======= UI FUNCTIONS =======

# Function to handle the edit button click
def process_edit(image, prompt, model_name, steps, guidance_scale, strength, negative_prompt=""):
    """Process the edit request and return results"""
    if image is None:
        return None, "Please upload an image first"
    
    try:
        if not prompt or prompt.strip() == "":
            return image, "Please enter an edit instruction"
        
        # Process the edit
        result, mask = edit_image(
            image, prompt, model_name, steps, guidance_scale, strength, negative_prompt
        )
        
        # Classify for debug info
        operation = classify_operation(prompt)
        
        # Generate debug info
        debug_info = f"Edit type: {operation['type']}\n"
        debug_info += f"Target: {operation['target']}\n"
        debug_info += f"Attribute: {operation['attribute']}\n"
        debug_info += f"Model: {model_name}\n"
        debug_info += f"Steps: {steps}, Guidance: {guidance_scale}, Strength: {strength}"
        
        return result, debug_info
        
    except Exception as e:
        error_msg = f"Error processing edit: {e}"
        print(error_msg)
        traceback.print_exc()
        return image, error_msg

# ======= MAIN APP =======

# Load initial models
try:
    pipe = load_model("SDXL Base")
except Exception as e:
    print(f"Error initializing models: {e}")

# Create the Gradio interface
with gr.Blocks(css="""
    .container { max-width: 1200px; margin: auto; }
    .edit-panel { padding: 10px; }
    .advanced-panel { margin-top: 10px; }
    .image-display { min-height: 400px; }
    .title { text-align: center; margin-bottom: 20px; }
    .footer { text-align: center; margin-top: 30px; font-size: 0.8em; color: #666; }
""") as demo:
    gr.Markdown("# ICEdit Pro - AI-Powered Image Editing", elem_classes="title")
    
    with gr.Row(elem_classes="container"):
        with gr.Column(scale=1, elem_classes="edit-panel"):
            input_image = gr.Image(label="Input Image", type="pil", elem_classes="image-display")
            
            edit_prompt = gr.Textbox(
                label="Edit Instruction",
                placeholder="Describe what you want to change (e.g., change hair color to blue, add text Hello, make background beach)"
            )
            
            with gr.Accordion("Advanced Settings", open=False, elem_classes="advanced-panel"):
                model_choice = gr.Dropdown(
                    choices=list(inpainting_models.keys()),
                    value="SDXL Base",
                    label="Model"
                )
                
                with gr.Row():
                    steps = gr.Slider(10, 80, value=30, step=1, label="Inference Steps")
                    guidance = gr.Slider(1.0, 20.0, value=7.5, step=0.1, label="Guidance Scale")
                
                strength = gr.Slider(0.2, 0.9, value=0.7, step=0.05, label="Edit Strength (lower preserves more details)")
                negative_prompt = gr.Textbox(label="Negative Prompt", placeholder="What you don't want in the image")
            
            generate_btn = gr.Button("Generate Edit", variant="primary")
        
        with gr.Column(scale=1):
            output_image = gr.Image(label="Edited Image", elem_classes="image-display")
            edit_details = gr.Textbox(label="Edit Details", interactive=False)
    
    # Connect the button to the processing function
    generate_btn.click(
        fn=process_edit,
        inputs=[
            input_image, edit_prompt, model_choice, steps, guidance, strength, negative_prompt
        ],
        outputs=[output_image, edit_details]
    )
    
    # Add footer with usage tips (instead of examples table)
    gr.Markdown("""
    ## Tips for Best Results
    
    - For hair edits, use "change hair color to [color]"
    - For face edits, try "change skin tone to [tone]"
    - For background changes, use "change background to [scene]"
    - For removal, use "remove [object] from image"
    - For styling, try "convert to [style] painting style"
    - For watermarks, use "remove watermark" or "remove text"
    - For replacing objects, use "replace [object] with [new object]"
    - For replacing people, specifically use "replace person with [animal]"
    
    Processing may take 10-30 seconds depending on the edit complexity.
    """, elem_classes="footer")

# Launch the app
demo.launch(share=True)