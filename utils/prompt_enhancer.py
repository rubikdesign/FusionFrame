#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
PromptEnhancer pentru îmbunătățirea prompturilor în FusionFrame 2.0
"""

import logging
import re
from typing import Dict, Any, Optional

# Setăm logger-ul
logger = logging.getLogger(__name__)

class PromptEnhancer:
    """Îmbunătățește prompturile bazate pe context și operație."""
    
    def __init__(self):
        """Inițializează enhancerul de prompturi."""
        # Fraze de bază care pot fi adăugate la prompturi pentru diferite operații
        self.operation_enhancements = {
            "remove": ", perfect seamless inpainting, realistic texture, coherent background, no artifacts",
            "add": ", seamless integration, natural lighting, proper perspective, realistic details",
            "color": ", realistic color, natural texture, proper lighting, detailed texture",
            "background": ", seamless background, natural lighting, proper perspective, ambient environment",
            "replace": ", seamless replacement, consistent lighting, proper perspective, realistic details",
            "general": ", high quality, detailed, realistic"
        }
        
        # Cuvinte de evitat în negative
        self.negative_base = "ugly, deformed, noisy, blurry, low contrast, cartoon, drawing, illustration, painting, sketch, poor quality, jpeg artifacts, compression artifacts, amateur, poorly rendered"
        
        # Îmbunătățiri bazate pe tipuri de scene (detectate în context)
        self.scene_enhancements = {
            "portrait": ", sharp face details, proper skin texture, clean facial features",
            "outdoor": ", natural lighting, clear sky details, realistic environment, proper depth",
            "indoor": ", proper interior lighting, realistic indoor atmosphere"
        }
    
    def enhance_prompt(self, prompt: str, operation_type: Optional[str] = None, 
                      image_context: Optional[Dict[str, Any]] = None) -> str:
        """
        Îmbunătățește promptul pozitiv bazat pe context și operație.
        
        Args:
            prompt: Promptul original
            operation_type: Tipul operației (remove, add, color, etc.)
            image_context: Contextul imaginii analizate
            
        Returns:
            Promptul îmbunătățit
        """
        if not prompt:
            return "photorealistic, detailed"
            
        enhanced = prompt
        
        # Adăugăm îmbunătățiri bazate pe operație
        if operation_type and operation_type in self.operation_enhancements:
            enhanced += self.operation_enhancements[operation_type]
        else:
            enhanced += self.operation_enhancements["general"]
        
        # Adăugăm îmbunătățiri bazate pe contextul imaginii
        if image_context:
            # Verifică tipul scenei
            scene_type = None
            if "scene_info" in image_context and "legacy_scene_type_heuristic" in image_context["scene_info"]:
                scene_type = image_context["scene_info"]["legacy_scene_type_heuristic"]
            
            if scene_type:
                # Simplifică textul pentru a se potrivi cu cheile noastre
                simple_type = "portrait" if "portrait" in scene_type else \
                              "outdoor" if "outdoor" in scene_type else \
                              "indoor" if "indoor" in scene_type else None
                
                if simple_type and simple_type in self.scene_enhancements:
                    enhanced += self.scene_enhancements[simple_type]
            
            # Adăugăm informații despre iluminare din context
            if "lighting_conditions" in image_context:
                lighting = image_context["lighting_conditions"]
                brightness = lighting.get("brightness_heuristic")
                temperature = lighting.get("temperature_heuristic")
                
                if brightness and brightness not in ["unknown", "balanced"]:
                    enhanced += f", {brightness} lighting"
                
                if temperature and temperature != "neutral":
                    enhanced += f", {temperature} tones"
        
        # Eliminăm duplicate și curățăm prompt-ul
        enhanced = re.sub(r',\s*,', ',', enhanced)  # Elimină virgule duble
        enhanced = re.sub(r'\s{2,}', ' ', enhanced)  # Elimină spații multiple
        enhanced = enhanced.strip(', ')  # Elimină virgule și spații de la început/sfârșit
        
        # Adaugă întotdeauna photorealistic dacă nu există deja
        if "photorealistic" not in enhanced.lower():
            enhanced += ", photorealistic"
            
        logger.debug(f"Prompt îmbunătățit: '{enhanced}'")
        return enhanced
    
    def generate_negative_prompt(self, prompt: str, operation_type: Optional[str] = None,
                                image_context: Optional[Dict[str, Any]] = None) -> str:
        """
        Generează un prompt negativ bazat pe context și operație.
        
        Args:
            prompt: Promptul original pozitiv
            operation_type: Tipul operației
            image_context: Contextul imaginii
            
        Returns:
            Promptul negativ generat
        """
        # Începem cu negativele de bază
        negative = self.negative_base
        
        # Adăugăm negative specific operației
        if operation_type == "remove":
            negative += ", visible object edges, remaining object parts, object shadow, ghosting, obvious patching"
        elif operation_type == "add":
            negative += ", floating objects, improper shadows, misplaced items, unrealistic proportions"
        elif operation_type == "color":
            negative += ", incorrect color blending, color bleeding, unnatural color, patchy color, wrong hue"
        elif operation_type == "background":
            negative += ", foreground artifacts, subject contamination, unrealistic background, perspective errors"
        
        # Adăugăm negative bazate pe context
        # Exemplu: dacă e portret, evităm distorsiuni faciale
        if image_context and "scene_info" in image_context:
            scene_type = image_context["scene_info"].get("legacy_scene_type_heuristic")
            if scene_type and "portrait" in scene_type:
                negative += ", distorted face, deformed features, mutated face, uncanny valley"
        
        # Adăugăm întotdeauna watermark și text
        if "watermark" not in negative:
            negative += ", watermark, text, signature, logo"
        
        logger.debug(f"Prompt negativ generat: '{negative}'")
        return negative