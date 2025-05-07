#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Utilitar pentru îmbunătățirea prompt-urilor în FusionFrame 2.0
(Versiune actualizată pentru a folosi contextul imaginii)
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
    Utilitar pentru îmbunătățirea și optimizarea prompt-urilor folosind contextul imaginii.
    """

    def __init__(self, templates_path: Optional[str] = None):
        """
        Inițializează utilitar pentru îmbunătățirea prompt-urilor.
        Templates sunt păstrate momentan, dar logica principală nu le mai folosește.

        Args:
            templates_path: Calea către fișierul JSON cu template-uri (opțional, mai puțin relevant acum).
        """
        # self.templates = self._load_templates(templates_path) # Nu mai folosim templates direct
        self.quality_terms = [
            "photorealistic", "ultra detailed", "high resolution", "4k", "8k", "sharp focus",
            "professional photography", "cinematic lighting", "realistic rendering",
            "masterpiece", "physically based rendering", "intricate details", "hyperrealistic"
        ]
        self.negative_base_terms = [ # Extins și grupat
            # Calitate slabă
            "ugly", "blurry", "distorted", "deformed", "pixelated", "low quality", "low resolution",
            "artifact", "noisy", "grainy", "jpeg artifacts", "compression artifacts", "worst quality", "poorly drawn",
            # Anatomie/Structură greșită
            "bad anatomy", "bad proportions", "extra limbs", "extra fingers", "mutated hands", "poorly drawn hands",
            "poorly drawn face", "mutation", "disgusting", "malformed limbs", "missing arms", "missing legs",
            "extra arms", "extra legs", "fused fingers", "too many fingers", "long neck", "cloned face", "duplicate",
            # Stil/Artefacte nedorite
            "illustration", "painting", "drawing", "sketch", "cartoon", "anime", "render", "3d", "crosseyed",
            "watermark", "signature", "text", "label", "username", "artist name", "logo",
            "cropped", "out of frame", "off center", "tilting"
        ]
        # Termeni negativi specifici operațiilor (vor fi folosiți în viitor)
        self.negative_specifics = {
            'remove': ["visible object remains", "incomplete removal", "object shadow", "ghosting", "residual traces", "artifact in removed area"],
            'color': ["wrong color", "unnatural color", "color bleeding", "discoloration", "patchy color", "incorrect shade"],
            'background': ["bad background", "mismatched lighting", "perspective error", "inconsistent shadows", "subject floating", "seam visible"],
            'add': ["floating object", "improper size", "misplaced", "unrealistic placement", "mismatched scale", "wrong perspective", "object clipping"]
        }
        logger.info("PromptEnhancer initialized.")

    # _load_templates nu mai este necesar dacă nu folosim formatarea prin template
    # def _load_templates(self, templates_path: Optional[str]) -> Dict[str, Any]: ...

    def enhance_prompt(self,
                       prompt: str,
                       operation_type: Optional[str] = None,
                       image_context: Optional[Dict[str, Any]] = None) -> str:
        """
        Îmbunătățește un prompt folosind contextul imaginii (dacă este disponibil).

        Args:
            prompt: Prompt-ul original al utilizatorului.
            operation_type: Tipul operației detectat (opțional).
            image_context: Dicționarul cu analiza imaginii de la ImageAnalyzer (opțional).

        Returns:
            Prompt-ul îmbunătățit pentru modelul AI.
        """
        if not prompt or not prompt.strip():
             return "" # Returnăm string gol dacă promptul e gol

        base_prompt = prompt.strip()

        # Selectăm termeni de calitate aleatori
        num_quality_terms = random.randint(2, 4) # Alegem între 2 și 4 termeni
        quality_string = ", ".join(random.sample(self.quality_terms, min(num_quality_terms, len(self.quality_terms))))

        context_terms = []
        if image_context:
            logger.debug("Enhancing prompt using image context.")
            scene_info = image_context.get('scene_info', {})
            lighting_info = image_context.get('lighting_conditions', {})
            style_info = image_context.get('style_and_quality', {})
            spatial_info = image_context.get('spatial_info', {})

            # 1. Stil (Prioritar dacă nu e 'natural' sau 'photorealistic' implicit)
            style = style_info.get('visual_style_heuristic')
            is_photorealistic = "photo" in style if style else True # Asumăm foto dacă nu se detectează altceva
            if style and style not in ['natural', 'photorealistic', 'realistic']:
                 context_terms.append(f"in a {style} style")
                 # Dacă e un stil artistic, ajustăm termenii de calitate
                 quality_string = quality_string.replace("photorealistic", style).replace("realistic rendering", style) # Înlocuim termeni nepotriviți
            elif is_photorealistic:
                 # Adăugăm termeni specifici pentru fotorealism dacă nu sunt deja
                 if "photorealistic" not in quality_string and "realistic" not in quality_string:
                      quality_string += ", photorealistic"

            # 2. Scenă ML
            scene_tag = scene_info.get('primary_scene_tag_ml')
            if scene_tag: context_terms.append(f"{scene_tag.replace('_', ' ')} scene")

            # 3. Obiecte Detectate (adaugă specificitate)
            detected_objects = scene_info.get('detected_objects', [])
            if detected_objects:
                # Filtrăm obiectele comune/nespecifice dacă lista e lungă? Sau luăm primele?
                top_objects = [obj['label'] for obj in detected_objects[:2] if obj['confidence'] > 0.5] # Peste 50% conf, max 2
                if top_objects: context_terms.append(f"featuring {', '.join(top_objects)}")

            # 4. Iluminare (adăugăm detalii subtile)
            brightness = lighting_info.get('brightness_heuristic')
            temperature = lighting_info.get('temperature_heuristic')
            contrast = lighting_info.get('contrast_heuristic')
            lighting_desc = []
            if brightness and brightness not in ['balanced', 'unknown']: lighting_desc.append(brightness)
            if temperature and temperature != 'neutral': lighting_desc.append(f"{temperature} tones")
            if contrast and contrast not in ['medium']: lighting_desc.append(f"{contrast} contrast")
            if lighting_desc: context_terms.append(f"({' '.join(lighting_desc)} lighting)")

            # 5. Adâncime (poate influența compoziția)
            depth_char = spatial_info.get('depth_characteristics')
            if depth_char == "dominant_foreground": context_terms.append("close-up perspective")
            elif depth_char == "good_fg_bg_separation": context_terms.append("clear foreground background separation")

        else:
            logger.debug("No image context provided for prompt enhancement.")

        # Combinăm promptul original cu termenii contextuali și de calitate
        # Încercăm să punem contextul înainte de promptul utilizatorului pentru ghidare mai bună? Testăm.
        final_prompt_parts = []
        if context_terms: final_prompt_parts.append(", ".join(context_terms))
        final_prompt_parts.append(base_prompt)
        if quality_string: final_prompt_parts.append(quality_string)

        enhanced_prompt = ", ".join(filter(None, final_prompt_parts))

        # Curățare finală (eliminare spații duble etc.)
        enhanced_prompt = re.sub(r'\s{2,}', ' ', enhanced_prompt).strip()
        enhanced_prompt = re.sub(r'(,\s*)+', ', ', enhanced_prompt).strip(', ') # Elimină virgule multiple sau la capete

        logger.debug(f"Original prompt: '{prompt}'")
        logger.info(f"Enhanced prompt: '{enhanced_prompt}'") # Schimbat la INFO pentru vizibilitate testare
        return enhanced_prompt

    def generate_negative_prompt(self,
                                 prompt: Optional[str] = None, # Promptul pozitiv original poate ajuta
                                 operation_type: Optional[str] = None,
                                 image_context: Optional[Dict[str, Any]] = None) -> str:
        """
        Generează un prompt negativ optimizat, potențial contextual.
        (Versiunea curentă este similară cu cea veche, dar extinde termenii)

        Args:
            prompt: Promptul pozitiv original (opțional).
            operation_type: Tipul operației (opțional).
            image_context: Contextul imaginii (opțional, neutilizat încă extensiv aici).

        Returns:
            Prompt-ul negativ generat.
        """
        # Selectăm un număr variabil de termeni negativi de bază
        num_base_terms = random.randint(5, 8)
        negative_prompt_parts = random.sample(self.negative_base_terms, min(num_base_terms, len(self.negative_base_terms)))

        # Adăugăm termeni specifici operației, dacă este cunoscută
        if operation_type and operation_type in self.negative_specifics:
             num_specific_terms = random.randint(1, 2)
             specific_terms = self.negative_specifics[operation_type]
             negative_prompt_parts.extend(random.sample(specific_terms, min(num_specific_terms, len(specific_terms))))

        # TODO (Viitor): Adăugare termeni negativi bazați pe context
        # if image_context:
        #     lighting_info = image_context.get('lighting_conditions', {})
        #     if lighting_info.get('brightness_heuristic') == 'bright':
        #         negative_prompt_parts.extend(["dark", "underexposed"])
        #     elif lighting_info.get('brightness_heuristic') == 'dim':
        #          negative_prompt_parts.extend(["overexposed", "too bright"])
        #     # ... etc pentru alte condiții ...

        # Asigurăm unicitatea și combinăm
        final_negative_prompt = ", ".join(list(dict.fromkeys(negative_prompt_parts))) # Elimină duplicatele păstrând ordinea
        logger.debug(f"Generated negative prompt: {final_negative_prompt}")
        return final_negative_prompt

    # --- Metode Utilitare Vechi (Păstrate pentru referință/fallback) ---
    def _detect_operation_type(self, prompt: str) -> str:
        """Detectează euristic tipul operației din prompt (simplificat)."""
        prompt_lower = prompt.lower()
        if any(term in prompt_lower for term in ['remove', 'delete', 'erase', 'eliminate']): return 'remove'
        elif any(term in prompt_lower for term in ['color', 'recolor', 'change color', 'make it']): return 'color'
        elif any(term in prompt_lower for term in ['background', 'scene', 'backdrop', 'behind']): return 'background'
        elif any(term in prompt_lower for term in ['add', 'insert', 'place', 'put', 'include', 'wear']): return 'add' # Extins
        elif any(term in prompt_lower for term in ['replace', 'swap', 'substitute', 'change with']): return 'replace' # Adăugat 'replace'
        else: return 'generic'

    def _extract_info_from_prompt(self, prompt: str, operation_type: str) -> Dict[str, str]:
        """Extrage euristic informații simple din prompt (mai puțin folosit acum)."""
        info = {}; prompt_lower = prompt.lower()
        if operation_type == 'remove': match = re.search(r'(?:remove|delete|erase|eliminate)\s+(?:the\s+|an?\s+)?(.+)', prompt_lower)
        elif operation_type == 'color': match = re.search(r'(?:color|change|make)\s+(?:the\s+|an?\s+)?(.+?)\s+(?:to|into|as)\s+([a-z\s-]+)', prompt_lower)
        elif operation_type == 'background': match = re.search(r'background\s+(?:to|into|with|as)\s+(?:an?\s+)?(.+)', prompt_lower)
        elif operation_type == 'add': match = re.search(r'(?:add|insert|place|put|wear|wearing)\s+(?:an?\s+)?(.+)', prompt_lower)
        elif operation_type == 'replace': match = re.search(r'(?:replace|swap|substitute)\s+(?:the\s+|an?\s+)?(.+?)\s+(?:with|for)\s+(?:an?\s+)?(.+)', prompt_lower)
        else: match = None

        if match:
            if operation_type == 'remove': info['object'] = match.group(1).strip()
            elif operation_type == 'color' and len(match.groups()) >= 2: info['object'] = match.group(1).strip(); info['color'] = match.group(2).strip()
            elif operation_type == 'background': info['background'] = match.group(1).strip()
            elif operation_type == 'add': info['addition'] = match.group(1).strip()
            elif operation_type == 'replace' and len(match.groups()) >= 2: info['object'] = match.group(1).strip(); info['replacement'] = match.group(2).strip()

        return info