#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Utilitar pentru îmbunătățirea prompt-urilor în FusionFrame 2.0
"""

import re
import json
import logging
import random
from typing import Dict, Any, List, Optional, Union, Tuple

# Setăm logger-ul
logger = logging.getLogger(__name__)

class PromptEnhancer:
    """
    Utilitar pentru îmbunătățirea și optimizarea prompt-urilor
    
    Responsabil pentru optimizarea și îmbunătățirea prompt-urilor
    pentru a obține rezultate mai bune de la modelele AI.
    """
    
    def __init__(self, templates_path: Optional[str] = None):
        """
        Inițializează utilitar pentru îmbunătățirea prompt-urilor
        
        Args:
            templates_path: Calea către fișierul JSON cu template-uri (opțional)
        """
        self.templates = self._load_templates(templates_path)
        self.quality_terms = [
            "high quality", "detailed", "sharp", "professional", "realistic", 
            "high resolution", "clear", "crisp", "fine details"
        ]
        self.negative_terms = [
            "blurry", "distorted", "deformed", "pixelated", "low quality", 
            "artifact", "low resolution", "unclear", "noisy", "grainy"
        ]
    
    def _load_templates(self, templates_path: Optional[str]) -> Dict[str, Any]:
        """
        Încarcă template-urile dintr-un fișier JSON
        
        Args:
            templates_path: Calea către fișierul JSON cu template-uri
            
        Returns:
            Dicționarul cu template-uri
        """
        default_templates = {
            "remove": [
                "{prompt}, {quality}, clean background, natural lighting",
                "scene without {object}, {quality}, pristine environment",
                "empty space where {object} was, {quality}, seamless"
            ],
            "color": [
                "{prompt}, {quality}, natural colors, realistic textures",
                "{object} with {color} color, {quality}, realistic, detailed",
                "{color} {object}, {quality}, natural appearance"
            ],
            "background": [
                "{prompt}, {quality}, natural lighting, professional photography",
                "{object} in front of {background}, {quality}, perfect lighting",
                "subject on {background} backdrop, {quality}, realistic"
            ],
            "add": [
                "{prompt}, {quality}, realistic integration, proper scale",
                "{object} with {addition}, {quality}, natural appearance",
                "subject wearing {addition}, {quality}, realistic"
            ],
            "generic": [
                "{prompt}, {quality}, professional result",
                "{prompt}, {quality}, realistic, detailed",
                "{prompt}, {quality}, natural appearance"
            ]
        }
        
        if templates_path:
            try:
                with open(templates_path, 'r', encoding='utf-8') as f:
                    custom_templates = json.load(f)
                    # Combinăm template-urile custom cu cele implicite
                    for key, templates in custom_templates.items():
                        if key in default_templates:
                            default_templates[key].extend(templates)
                        else:
                            default_templates[key] = templates
            except Exception as e:
                logger.error(f"Error loading templates: {str(e)}")
        
        return default_templates
    
    def enhance_prompt(self, prompt: str, operation_type: Optional[str] = None) -> str:
        """
        Îmbunătățește un prompt pentru a obține rezultate mai bune
        
        Args:
            prompt: Prompt-ul original
            operation_type: Tipul operației (opțional)
            
        Returns:
            Prompt-ul îmbunătățit
        """
        # Analizăm prompt-ul pentru a determina tipul operației dacă nu este specificat
        if not operation_type:
            operation_type = self._detect_operation_type(prompt)
        
        # Selectăm template-ul potrivit
        templates = self.templates.get(operation_type, self.templates['generic'])
        selected_template = random.choice(templates)
        
        # Extragem informații din prompt
        info = self._extract_info_from_prompt(prompt, operation_type)
        
        # Selectăm termeni de calitate
        quality_terms = random.sample(self.quality_terms, 2)
        quality_string = ", ".join(quality_terms)
        
        # Formatăm template-ul
        enhanced_prompt = selected_template.format(
            prompt=prompt,
            quality=quality_string,
            object=info.get('object', ''),
            color=info.get('color', ''),
            background=info.get('background', ''),
            addition=info.get('addition', '')
        )
        
        return enhanced_prompt
    
    def generate_negative_prompt(self, operation_type: Optional[str] = None) -> str:
        """
        Generează un prompt negativ optimizat
        
        Args:
            operation_type: Tipul operației (opțional)
            
        Returns:
            Prompt-ul negativ generat
        """
        # Selectăm termeni negativi de bază
        base_negative = random.sample(self.negative_terms, 4)
        
        # Adăugăm termeni specifici pentru tipul operației
        if operation_type == 'remove':
            specific_terms = ["visible object", "incomplete removal", "object shadow", "residual traces"]
            base_negative.extend(random.sample(specific_terms, 2))
        elif operation_type == 'color':
            specific_terms = ["wrong color", "unnatural color", "color bleeding", "discoloration"]
            base_negative.extend(random.sample(specific_terms, 2))
        elif operation_type == 'background':
            specific_terms = ["bad background", "mismatched lighting", "perspective error", "inconsistent shadows"]
            base_negative.extend(random.sample(specific_terms, 2))
        elif operation_type == 'add':
            specific_terms = ["floating object", "improper size", "misplaced", "unrealistic placement"]
            base_negative.extend(random.sample(specific_terms, 2))
        
        # Adăugăm termeni generali de calitate slabă
        general_bad = ["poor composition", "amateur", "low quality", "unnatural"]
        base_negative.extend(random.sample(general_bad, 2))
        
        return ", ".join(base_negative)
    
    def _detect_operation_type(self, prompt: str) -> str:
        """
        Detectează tipul operației din prompt
        
        Args:
            prompt: Prompt-ul de analizat
            
        Returns:
            Tipul operației detectat
        """
        prompt_lower = prompt.lower()
        
        # Verificăm categoriile prin cuvinte cheie
        if any(term in prompt_lower for term in ['remove', 'delete', 'erase', 'eliminate']):
            return 'remove'
        elif any(term in prompt_lower for term in ['color', 'recolor', 'change color', 'make it']):
            return 'color'
        elif any(term in prompt_lower for term in ['background', 'scene', 'backdrop', 'behind']):
            return 'background'
        elif any(term in prompt_lower for term in ['add', 'insert', 'place', 'put', 'include']):
            return 'add'
        else:
            return 'generic'
    
    def _extract_info_from_prompt(self, prompt: str, operation_type: str) -> Dict[str, str]:
        """
        Extrage informații relevante din prompt
        
        Args:
            prompt: Prompt-ul de analizat
            operation_type: Tipul operației
            
        Returns:
            Dicționar cu informații extrase
        """
        info = {}
        prompt_lower = prompt.lower()
        
        if operation_type == 'remove':
            # Încercăm să extragem obiectul care trebuie eliminat
            match = re.search(r'remove\s+(?:the\s+)?([a-z\s]+)', prompt_lower)
            if match:
                info['object'] = match.group(1).strip()
        
        elif operation_type == 'color':
            # Încercăm să extragem obiectul și culoarea
            match = re.search(r'(?:color|change|make)\s+(?:the\s+)?([a-z\s]+)\s+(?:to|into|as)\s+([a-z\s]+)', prompt_lower)
            if match:
                info['object'] = match.group(1).strip()
                info['color'] = match.group(2).strip()
        
        elif operation_type == 'background':
            # Încercăm să extragem fundalul
            match = re.search(r'background\s+(?:to|into|with|as)\s+([a-z\s]+)', prompt_lower)
            if match:
                info['background'] = match.group(1).strip()
        
        elif operation_type == 'add':
            # Încercăm să extragem obiectul de adăugat
            match = re.search(r'add\s+(?:a\s+)?([a-z\s]+)', prompt_lower)
            if match:
                info['addition'] = match.group(1).strip()
        
        return info