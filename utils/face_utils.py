"""
Face utility functions for the FusionFrame application.

This module provides functions for:
- Detecting facial landmarks
- Creating masks for facial features
- Selective cloning of facial features
"""

import cv2
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Union

logger = logging.getLogger(__name__)

# Conditional import for face_recognition
try:
    import face_recognition
    FACE_RECOGNITION_AVAILABLE = True
except ImportError:
    FACE_RECOGNITION_AVAILABLE = False
    logger.warning("face_recognition module not installed. Face alignment will not be available.")
    logger.info("You can install it with: pip install face_recognition")

def get_face_masks(img: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Create binary masks for facial features (eyes, nose, mouth) from an image.
    
    Args:
        img (np.ndarray): Input image with a face
        
    Returns:
        Dict[str, np.ndarray]: Dictionary with masks for different facial zones
            - keys: 'left_eye', 'eyes', 'nose', 'mouth'
            - values: binary masks as numpy arrays
    
    Raises:
        ValueError: If no faces detected
        ImportError: If face_recognition not available
    """
    if not FACE_RECOGNITION_AVAILABLE:
        raise ImportError("face_recognition module is required for this function")
    
    # Detect facial landmarks
    landmarks_list = face_recognition.face_landmarks(img)
    if not landmarks_list:
        raise ValueError("No faces detected in the image")
    
    landmarks = landmarks_list[0]  # Use first face
    
    # Define facial parts to extract
    zones = {}
    parts = {
        "left_eye":  ('left_eye',),
        "right_eye": ('right_eye',),
        "eyes":      ('left_eye', 'right_eye'),
        "nose":      ('nose_tip',),
        "mouth":     ('top_lip', 'bottom_lip'),
    }
    
    # Create mask for each part
    for zone, pts_keys in parts.items():
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        for k in pts_keys:
            if k in landmarks:
                poly = np.array(landmarks[k], np.int32)
                cv2.fillPoly(mask, [poly], 255)
            else:
                logger.warning(f"Landmark '{k}' not found for zone '{zone}'")
        zones[zone] = mask
    
    logger.debug(f"Created masks for {len(zones)} facial zones")
    return zones

def selective_clone(
    src: np.ndarray, 
    dst: np.ndarray, 
    parts: Tuple[str, ...] = ('eyes', 'nose', 'mouth'), 
    alpha: float = 0.8
) -> np.ndarray:
    """
    Copy specific facial features from source to destination image using seamless cloning.
    
    Args:
        src (np.ndarray): Source image (with the face features to extract)
        dst (np.ndarray): Destination image (where to paste the features)
        parts (Tuple[str, ...]): Facial parts to transfer ('eyes', 'nose', 'mouth')
        alpha (float): Blending factor after cloning (0-1)
        
    Returns:
        np.ndarray: Result image with facial features transferred
    """
    try:
        # Check if shapes match
        if src.shape != dst.shape:
            logger.warning("Source and destination images have different shapes. Resizing source.")
            src = cv2.resize(src, (dst.shape[1], dst.shape[0]))
        
        # Extract facial masks
        masks = get_face_masks(src)
        result = dst.copy()

        for part in parts:
            if part not in masks:
                logger.warning(f"Part '{part}' not found in masks")
                continue
                
            m = masks[part]
            if m.sum() == 0:  # if zone doesn't exist
                logger.debug(f"Mask for '{part}' is empty")
                continue
                
            # Get bounding rectangle and validate boundaries
            x, y, w, h = cv2.boundingRect(m)
            
            # Safety checks to ensure center point is within image boundaries
            center_x = min(max(x+w//2, 0), dst.shape[1]-1)
            center_y = min(max(y+h//2, 0), dst.shape[0]-1)
            center = (center_x, center_y)
            
            # Ensure source and destination have same dimensions
            if src.shape != dst.shape:
                src_resized = cv2.resize(src, (dst.shape[1], dst.shape[0]))
            else:
                src_resized = src
                
            # Additional check if mask has valid dimensions
            if m.shape[:2] != dst.shape[:2]:
                m_resized = cv2.resize(m, (dst.shape[1], dst.shape[0]))
            else:
                m_resized = m
                
            # Perform seamless cloning with error handling
            try:
                result = cv2.seamlessClone(src_resized, result, m_resized, center, cv2.NORMAL_CLONE)
                logger.debug(f"Successfully cloned part: {part}")
            except cv2.error as e:
                logger.error(f"CV2 error during seamless cloning of {part}: {e}")
                continue
                
        # Apply final blending
        result = cv2.addWeighted(result, alpha, dst, 1-alpha, 0)
        return result
    except Exception as e:
        logger.error(f"Error in selective_clone: {e}")
        logger.debug("Returning original destination image as fallback")
        return dst