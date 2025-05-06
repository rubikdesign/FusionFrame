#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Utilitare pentru procesarea imaginilor în FusionFrame 2.0
"""

import cv2
import numpy as np
import logging
from typing import Optional, Tuple, Union, List, Dict, Any
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

# Setăm logger-ul
logger = logging.getLogger(__name__)

class ImageUtils:
    """
    Utilitare pentru procesarea și manipularea imaginilor
    
    Oferă funcții utile pentru procesarea imaginilor care pot fi
    folosite în diverse componente ale aplicației.
    """
    
    @staticmethod
    def resize_image(image: Union[np.ndarray, Image.Image], 
                    size: Tuple[int, int], 
                    keep_aspect_ratio: bool = True) -> Union[np.ndarray, Image.Image]:
        """
        Redimensionează o imagine cu opțiunea de a păstra aspect ratio
        
        Args:
            image: Imaginea de redimensionat (numpy array sau PIL Image)
            size: Dimensiunea țintă (width, height)
            keep_aspect_ratio: Dacă se păstrează raportul de aspect
            
        Returns:
            Imaginea redimensionată
        """
        # Convertim la format potrivit
        is_pil = isinstance(image, Image.Image)
        if is_pil:
            img = image
        else:
            img = Image.fromarray(image)
        
        if keep_aspect_ratio:
            # Păstrăm raportul de aspect
            img.thumbnail(size, Image.LANCZOS)
            # Creăm o imagine goală cu dimensiunea target
            new_img = Image.new("RGB", size, (0, 0, 0))
            # Lipim imaginea redimensionată în centru
            new_img.paste(img, ((size[0] - img.width) // 2, (size[1] - img.height) // 2))
            img = new_img
        else:
            # Redimensionăm direct la dimensiunea target
            img = img.resize(size, Image.LANCZOS)
        
        # Returnăm în formatul original
        if is_pil:
            return img
        else:
            return np.array(img)
    
    @staticmethod
    def apply_mask(image: Union[np.ndarray, Image.Image], 
                  mask: Union[np.ndarray, Image.Image],
                  blur_radius: int = 5) -> Union[np.ndarray, Image.Image]:
        """
        Aplică o mască pe o imagine cu efect de blur la margini
        
        Args:
            image: Imaginea originală
            mask: Masca (alb-negru)
            blur_radius: Raza efectului de blur la margini
            
        Returns:
            Imaginea mascată
        """
        # Convertim la format potrivit
        is_pil = isinstance(image, Image.Image)
        if is_pil:
            img = np.array(image)
            mask_img = np.array(mask) if isinstance(mask, Image.Image) else mask
        else:
            img = image
            mask_img = mask
        
        # Ne asigurăm că masca este în formatul potrivit
        if mask_img.max() <= 1.0:
            mask_img = (mask_img * 255).astype(np.uint8)
        
        # Creăm o mască pentru blend
        soft_mask = cv2.GaussianBlur(mask_img, (blur_radius*2+1, blur_radius*2+1), 0)
        soft_mask = soft_mask.astype(np.float32) / 255.0
        
        # Adaptăm dimensiunile măștii dacă e necesar
        if soft_mask.shape[:2] != img.shape[:2]:
            soft_mask = cv2.resize(soft_mask, (img.shape[1], img.shape[0]))
        
        # Extinde masca la 3 canale dacă e necesar
        if len(soft_mask.shape) == 2 and len(img.shape) == 3:
            soft_mask = np.repeat(soft_mask[:, :, np.newaxis], 3, axis=2)
        
        # Convertim masca la zero pentru a crea o imagine neagră de aceeași dimensiune
        masked_img = img * soft_mask
        
        # Returnăm în formatul original
        if is_pil:
            return Image.fromarray(masked_img.astype(np.uint8))
        else:
            return masked_img
    
    @staticmethod
    def blend_images(image1: Union[np.ndarray, Image.Image],
                    image2: Union[np.ndarray, Image.Image],
                    mask: Optional[Union[np.ndarray, Image.Image]] = None,
                    alpha: float = 0.5) -> Union[np.ndarray, Image.Image]:
        """
        Combină două imagini folosind o mască sau un alpha global
        
        Args:
            image1: Prima imagine
            image2: A doua imagine
            mask: Masca pentru blend (opțional)
            alpha: Factorul de blend global (0.0-1.0)
            
        Returns:
            Imaginea combinată
        """
        # Convertim la format potrivit
        is_pil = isinstance(image1, Image.Image)
        if is_pil:
            img1 = np.array(image1)
            img2 = np.array(image2) if isinstance(image2, Image.Image) else image2
            mask_img = np.array(mask) if isinstance(mask, Image.Image) and mask is not None else mask
        else:
            img1 = image1
            img2 = image2
            mask_img = mask
        
        # Verificăm dimensiunile
        if img1.shape != img2.shape:
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
        
        # Blend cu mască sau global
        if mask_img is not None:
            # Ne asigurăm că masca este în formatul potrivit
            if mask_img.max() <= 1.0:
                mask_img = (mask_img * 255).astype(np.uint8)
            
            # Normalizăm masca
            blend_mask = mask_img.astype(np.float32) / 255.0
            
            # Adaptăm dimensiunile măștii dacă e necesar
            if blend_mask.shape[:2] != img1.shape[:2]:
                blend_mask = cv2.resize(blend_mask, (img1.shape[1], img1.shape[0]))
            
            # Extinde masca la 3 canale dacă e necesar
            if len(blend_mask.shape) == 2 and len(img1.shape) == 3:
                blend_mask = np.repeat(blend_mask[:, :, np.newaxis], 3, axis=2)
            
            # Combinăm imaginile
            result = img1 * (1 - blend_mask) + img2 * blend_mask
        else:
            # Blend global
            result = cv2.addWeighted(img1, 1-alpha, img2, alpha, 0)
        
        # Returnăm în formatul original
        if is_pil:
            return Image.fromarray(result.astype(np.uint8))
        else:
            return result
    
    @staticmethod
    def enhance_image(image: Union[np.ndarray, Image.Image], 
                     brightness: float = 1.0,
                     contrast: float = 1.0,
                     saturation: float = 1.0,
                     sharpness: float = 1.0) -> Union[np.ndarray, Image.Image]:
        """
        Îmbunătățește o imagine ajustând parametrii
        
        Args:
            image: Imaginea de îmbunătățit
            brightness: Factorul de luminozitate (default: 1.0)
            contrast: Factorul de contrast (default: 1.0)
            saturation: Factorul de saturație (default: 1.0)
            sharpness: Factorul de claritate (default: 1.0)
            
        Returns:
            Imaginea îmbunătățită
        """
        # Convertim la format potrivit
        is_pil = isinstance(image, Image.Image)
        if is_pil:
            img = image
        else:
            img = Image.fromarray(image)
        
        # Aplicăm îmbunătățirile
        if brightness != 1.0:
            img = ImageEnhance.Brightness(img).enhance(brightness)
        if contrast != 1.0:
            img = ImageEnhance.Contrast(img).enhance(contrast)
        if saturation != 1.0:
            img = ImageEnhance.Color(img).enhance(saturation)
        if sharpness != 1.0:
            img = ImageEnhance.Sharpness(img).enhance(sharpness)
        
        # Returnăm în formatul original
        if is_pil:
            return img
        else:
            return np.array(img)
    
    @staticmethod
    def create_tile_grid(images: List[Union[np.ndarray, Image.Image]], 
                        grid_size: Optional[Tuple[int, int]] = None,
                        tile_size: Optional[Tuple[int, int]] = None) -> Union[np.ndarray, Image.Image]:
        """
        Creează o grilă de imagini
        
        Args:
            images: Lista de imagini
            grid_size: Dimensiunea grilei (rows, cols) (opțional)
            tile_size: Dimensiunea unei casete (width, height) (opțional)
            
        Returns:
            Imaginea grilă
        """
        n_images = len(images)
        if n_images == 0:
            return None
        
        # Determinăm dimensiunea grilei
        if grid_size is None:
            cols = int(np.ceil(np.sqrt(n_images)))
            rows = int(np.ceil(n_images / cols))
        else:
            rows, cols = grid_size
        
        # Determinăm formatul imaginilor
        is_pil = isinstance(images[0], Image.Image)
        
        # Convertim toate imaginile la PIL pentru procesare
        pil_images = []
        for img in images:
            if isinstance(img, Image.Image):
                pil_images.append(img)
            else:
                pil_images.append(Image.fromarray(img))
        
        # Determinăm dimensiunea unei casete
        if tile_size is None:
            # Folosim dimensiunea primei imagini
            tile_size = pil_images[0].size
        
        # Redimensionăm toate imaginile la dimensiunea casetei
        for i in range(len(pil_images)):
            if pil_images[i].size != tile_size:
                pil_images[i] = pil_images[i].resize(tile_size, Image.LANCZOS)
        
        # Creăm imaginea grilă
        grid_width = cols * tile_size[0]
        grid_height = rows * tile_size[1]
        grid_img = Image.new('RGB', (grid_width, grid_height), (255, 255, 255))
        
        # Plasăm imaginile în grilă
        for i, img in enumerate(pil_images):
            if i >= rows * cols:
                break
                
            row = i // cols
            col = i % cols
            x = col * tile_size[0]
            y = row * tile_size[1]
            
            grid_img.paste(img, (x, y))
        
        # Returnăm în formatul original
        if not is_pil:
            return np.array(grid_img)
        else:
            return grid_img
    
    @staticmethod
    def create_comparison(image1: Union[np.ndarray, Image.Image],
                         image2: Union[np.ndarray, Image.Image],
                         vertical: bool = False,
                         add_separator: bool = True) -> Union[np.ndarray, Image.Image]:
        """
        Creează o imagine de comparație (înainte/după)
        
        Args:
            image1: Prima imagine
            image2: A doua imagine
            vertical: Dacă imaginile sunt aranjate vertical sau orizontal
            add_separator: Dacă se adaugă o linie separatoare
            
        Returns:
            Imaginea de comparație
        """
        # Convertim la format potrivit
        is_pil = isinstance(image1, Image.Image)
        if is_pil:
            img1 = image1
            img2 = image2 if isinstance(image2, Image.Image) else Image.fromarray(image2)
        else:
            img1 = Image.fromarray(image1)
            img2 = Image.fromarray(image2) if not isinstance(image2, Image.Image) else image2
        
        # Calculăm dimensiunile
        width1, height1 = img1.size
        width2, height2 = img2.size
        
        # Redimensionăm a doua imagine pentru a se potrivi cu prima
        img2 = img2.resize((width1, height1), Image.LANCZOS)
        
        # Creăm imaginea de comparație
        if vertical:
            comp_width = width1
            comp_height = height1 * 2 + (10 if add_separator else 0)
            comp_img = Image.new('RGB', (comp_width, comp_height), (255, 255, 255))
            comp_img.paste(img1, (0, 0))
            comp_img.paste(img2, (0, height1 + (10 if add_separator else 0)))
            
            # Adăugăm separator
            if add_separator:
                draw = ImageDraw.Draw(comp_img)
                draw.line([(0, height1 + 5), (width1, height1 + 5)], fill=(0, 0, 0), width=2)
        else:
            comp_width = width1 * 2 + (10 if add_separator else 0)
            comp_height = height1
            comp_img = Image.new('RGB', (comp_width, comp_height), (255, 255, 255))
            comp_img.paste(img1, (0, 0))
            comp_img.paste(img2, (width1 + (10 if add_separator else 0), 0))
            
            # Adăugăm separator
            if add_separator:
                draw = ImageDraw.Draw(comp_img)
                draw.line([(width1 + 5, 0), (width1 + 5, height1)], fill=(0, 0, 0), width=2)
        
        # Returnăm în formatul original
        if not is_pil:
            return np.array(comp_img)
        else:
            return comp_img