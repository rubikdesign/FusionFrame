#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Utilitare pentru încărcarea și salvarea imaginilor în FusionFrame 2.0
"""

import os
import logging
import numpy as np
from typing import Optional, List, Union, Tuple, Dict, Any
from PIL import Image
from datetime import datetime

# Setăm logger-ul
logger = logging.getLogger(__name__)

class ImageIO:
    """
    Utilitar pentru încărcarea și salvarea imaginilor
    
    Responsabil pentru operațiile de I/O pentru imagini și
    gestionarea fișierelor de imagini.
    """
    
    @staticmethod
    def load_image(image_path: str) -> Optional[Image.Image]:
        """
        Încarcă o imagine din fișier
        
        Args:
            image_path: Calea către fișierul imagine
            
        Returns:
            Imaginea încărcată sau None în caz de eroare
        """
        try:
            logger.info(f"Loading image from {image_path}")
            image = Image.open(image_path)
            
            # Convertim la RGB dacă are alt mod
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            return image
            
        except Exception as e:
            logger.error(f"Error loading image: {str(e)}")
            return None
    
    @staticmethod
    def save_image(image: Union[np.ndarray, Image.Image], 
                 output_path: str,
                 quality: int = 95,
                 create_dirs: bool = True) -> bool:
        """
        Salvează o imagine în fișier
        
        Args:
            image: Imaginea de salvat
            output_path: Calea către fișierul de ieșire
            quality: Calitatea de salvare pentru JPEG (0-100)
            create_dirs: Dacă se creează directoarele dacă nu există
            
        Returns:
            True dacă salvarea a reușit, False altfel
        """
        try:
            # Creăm directoarele dacă nu există
            if create_dirs:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Convertim la PIL dacă este numpy array
            if isinstance(image, np.ndarray):
                image_pil = Image.fromarray(image)
            else:
                image_pil = image
            
            # Determinăm extensia
            _, ext = os.path.splitext(output_path)
            ext = ext.lower()
            
            # Salvăm imaginea cu parametrii potriviți pentru format
            if ext == '.jpg' or ext == '.jpeg':
                image_pil.save(output_path, quality=quality, optimize=True)
            elif ext == '.png':
                image_pil.save(output_path, optimize=True)
            elif ext == '.webp':
                image_pil.save(output_path, quality=quality, method=6)
            else:
                # Format implicit
                image_pil.save(output_path)
            
            logger.info(f"Image saved to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving image: {str(e)}")
            return False
    
    @staticmethod
    def generate_output_path(input_path: Optional[str] = None, 
                           output_dir: Optional[str] = None,
                           suffix: str = "_edited",
                           extension: str = ".png") -> str:
        """
        Generează o cale de ieșire pentru imagine
        
        Args:
            input_path: Calea de intrare (opțional)
            output_dir: Directorul de ieșire (opțional)
            suffix: Sufixul pentru numele fișierului
            extension: Extensia fișierului
            
        Returns:
            Calea de ieșire generată
        """
        # Generăm un timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if input_path:
            # Extragem numele fișierului fără extensie
            basename = os.path.basename(input_path)
            filename, _ = os.path.splitext(basename)
            
            # Construim calea de ieșire
            if output_dir:
                return os.path.join(output_dir, f"{filename}{suffix}{extension}")
            else:
                # Folosim același director cu intrarea
                return os.path.join(os.path.dirname(input_path), f"{filename}{suffix}{extension}")
        else:
            # Generăm un nume bazat pe timestamp
            if output_dir:
                return os.path.join(output_dir, f"fusionframe_{timestamp}{extension}")
            else:
                # Folosim directorul curent
                return f"fusionframe_{timestamp}{extension}"
    
    @staticmethod
    def get_image_info(image_path: str) -> Dict[str, Any]:
        """
        Obține informații despre o imagine
        
        Args:
            image_path: Calea către fișierul imagine
            
        Returns:
            Dicționar cu informații despre imagine
        """
        try:
            # Deschidem imaginea pentru a obține informații
            with Image.open(image_path) as img:
                info = {
                    'path': image_path,
                    'filename': os.path.basename(image_path),
                    'size': img.size,
                    'width': img.width,
                    'height': img.height,
                    'format': img.format,
                    'mode': img.mode,
                    'file_size': os.path.getsize(image_path),
                    'modified_time': os.path.getmtime(image_path)
                }
                
                # Adăugăm informații EXIF dacă există
                if hasattr(img, '_getexif') and img._getexif():
                    exif = img._getexif()
                    exif_data = {}
                    
                    # Maparea tagurilor EXIF comune
                    exif_tags = {
                        271: 'Make',
                        272: 'Model',
                        306: 'DateTime',
                        36867: 'DateTimeOriginal',
                        33434: 'ExposureTime',
                        33437: 'FNumber',
                        34855: 'ISOSpeedRatings',
                        37386: 'FocalLength'
                    }
                    
                    for tag_id, tag_name in exif_tags.items():
                        if tag_id in exif:
                            exif_data[tag_name] = exif[tag_id]
                    
                    info['exif'] = exif_data
                
                return info
                
        except Exception as e:
            logger.error(f"Error getting image info: {str(e)}")
            return {'path': image_path, 'error': str(e)}
    
    @staticmethod
    def list_image_files(directory: str) -> List[str]:
        """
        Listează toate fișierele imagine dintr-un director
        
        Args:
            directory: Calea către director
            
        Returns:
            Lista cu căile către fișierele imagine
        """
        image_extensions = ['.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff', '.tif']
        image_files = []
        
        try:
            for root, _, files in os.walk(directory):
                for file in files:
                    # Verificăm dacă este un fișier imagine
                    if any(file.lower().endswith(ext) for ext in image_extensions):
                        image_files.append(os.path.join(root, file))
            
            return image_files
            
        except Exception as e:
            logger.error(f"Error listing image files: {str(e)}")
            return []