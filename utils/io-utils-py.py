"""
I/O utilities for the FusionFrame application.

This module provides functions for:
- Saving generated images
- Opening folders in the system's file explorer
- Scanning directories for files (like LoRAs)
"""

import os
import glob
import datetime
import subprocess
import platform
import logging
from PIL import Image
from pathlib import Path

logger = logging.getLogger(__name__)

def save_image(image, seed, model_name, output_dir, save_format="png", index=0, batch_count=1):
    """
    Save the generated image to the specified output directory.
    
    Args:
        image (PIL.Image): The image to save
        seed (int): The seed used for generation
        model_name (str): Name of the model used
        output_dir (str): Directory to save the image
        save_format (str): Image format (png, jpg, webp)
        index (int): Index in batch (if multiple images)
        batch_count (int): Total number of images in batch
        
    Returns:
        str: Path to the saved image or None if not saved
    """
    try:
        # Create a filename with timestamp and seed
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_short = model_name.split("/")[-1].replace("-", "_") if model_name else "model"
        
        # Add batch index if generating multiple images
        batch_suffix = f"_{index+1}of{batch_count}" if batch_count > 1 else ""
        
        filename = f"fusion_{timestamp}{batch_suffix}_seed{seed}_{model_short}.{save_format}"
        filepath = os.path.join(output_dir, filename)
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Save the image
        image.save(filepath)
        logger.info(f"Image saved: {filepath}")
        return filepath
    except Exception as e:
        logger.error(f"Error saving image: {e}")
        return None

def open_folder(folder_path):
    """
    Open the specified folder in the system's file explorer.
    
    Args:
        folder_path (str): Path to the folder to open
        
    Returns:
        str: Status message
    """
    path = os.path.abspath(folder_path)
    
    try:
        if platform.system() == "Windows":
            os.startfile(path)
        elif platform.system() == "Darwin":  # macOS
            subprocess.Popen(["open", path])
        else:  # Linux
            subprocess.Popen(["xdg-open", path])
        logger.info(f"Opened folder: {path}")
        return f"Opened folder: {path}"
    except Exception as e:
        error_msg = f"Error opening folder: {e}"
        logger.error(error_msg)
        return error_msg

def scan_directory(directory, file_extension="*.safetensors"):
    """
    Scan a directory for files with specified extension and return a dictionary.
    
    Args:
        directory (str): Directory to scan
        file_extension (str): File extension pattern to match
        
    Returns:
        dict: Dictionary mapping file names (without extension) to full paths
    """
    try:
        files = glob.glob(os.path.join(directory, file_extension))
        result = {}
        
        for file_path in files:
            file_name = os.path.basename(file_path).replace(file_extension.replace("*", ""), "")
            result[file_name] = file_path
            
        logger.info(f"Found {len(result)} files in {directory}")
        return result
    except Exception as e:
        logger.error(f"Error scanning directory {directory}: {e}")
        return {}
