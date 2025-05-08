#!/bin/bash
# Script de reparare finală pentru FusionFrame

echo "=== Reparare finală FusionFrame ==="

# Activăm mediul virtual
source ./fusionframe_env/bin/activate

# 1. Revenim la versiuni compatibile 
echo "1. Instalăm versiuni compatibile pentru huggingface_hub și diffusers..."
pip uninstall -y diffusers huggingface_hub
pip install huggingface_hub==0.15.1
pip install diffusers==0.18.2

# 2. Reparăm setarea share=True în interfață
echo "2. Modificăm direct metoda de lansare în interface/ui.py..."

# Verifică dacă fișierul există
if [ ! -f "interface/ui.py" ]; then
    echo "EROARE: Fișierul interface/ui.py nu există!"
    exit 1
fi

# Salvăm o copie de backup
cp interface/ui.py interface/ui.py.backup

# Modificăm direct metoda launch pentru a seta share=True
cat > interface/ui.py.modified << 'EOF'
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Interfața utilizator pentru FusionFrame 2.0
"""

import os
import sys
import time
import logging
import textwrap
import gradio as gr
import pandas as pd
from pathlib import Path
from typing import Dict, List, Union, Optional, Any, Tuple

# Asigură-te că directorul rădăcină este în PYTHONPATH
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.app_config import AppConfig
from core.model_manager import ModelManager
from core.pipeline_manager import PipelineManager
from processing.analyzer import ImageAnalyzer, OperationAnalyzer
from processing.post_processor import PostProcessor

# Configurare logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

# Alte importuri și cod anterior...

class FusionFrameUI:
    """
    Interfața utilizator pentru FusionFrame 2.0
    """
    
    def __init__(self):
        """
        Inițializează interfața utilizator
        """
        logger.info("Initializing FusionFrameUI...")
        
        self.config = AppConfig
        self.model_manager = ModelManager()
        self.pipeline_manager = PipelineManager()
        self.image_analyzer = ImageAnalyzer()
        self.post_processor = PostProcessor()
        
        # Verifică disponibilitatea modelelor
        logger.info("Loading/Checking essential models...")
        self._check_models()
        
        # Creează interfața Gradio
        logger.info("Creating Gradio interface...")
        self.app = self.create_interface()
        logger.info("Gradio interface created.")
        logger.info("FusionFrameUI initialized.")
        
    # Cod existent...

    def launch(self, server_name: str = "0.0.0.0", server_port: int = 7860, debug: bool = False):
        """
        Lansează interfața web
        
        Args:
            server_name: Adresa IP a serverului
            server_port: Portul serverului
            debug: Activează modul debug
        """
        logger.info(f"Launching Gradio interface: {{'server_name': '{server_name}', 'server_port': {server_port}, 'share': True, 'debug': {debug}}}")
        launch_kwargs = {
            "server_name": server_name,
            "server_port": server_port,
            "share": True,  # IMPORTANT: Setăm share=True pentru a permite accesul extern
            "debug": debug
        }
        
        try: 
            self.app.launch(**launch_kwargs)
        except Exception as e:
            logger.critical(f"Gradio launch failed: {e}")
            raise

# Restul codului...

def main():
    logger.info("Starting FusionFrame UI...")
    ui = FusionFrameUI()
    
    # Obține portul din variabila de mediu sau folosește valoarea implicită
    port = int(os.environ.get("PORT", 7860))
    
    # Lansează interfața
    logger.info(f"Launching interface on 0.0.0.0:{port}")
    ui.launch(server_name="0.0.0.0", server_port=port)

if __name__ == "__main__":
    main()
EOF

# Extracem doar metoda launch din fișierul modificat
LAUNCH_METHOD=$(grep -n -A 30 "def launch" interface/ui.py.modified | tail -n +2)

# Găsim începutul metodei launch în fișierul original
LAUNCH_LINE=$(grep -n "def launch" interface/ui.py | head -1 | cut -d':' -f1)

# Dacă am găsit metoda launch, o înlocuim
if [ -n "$LAUNCH_LINE" ]; then
    # Găsim sfârșitul metodei launch (următoarea definiție de funcție sau sfârșitul fișierului)
    NEXT_DEF=$(tail -n +$((LAUNCH_LINE+1)) interface/ui.py | grep -n "^    def " | head -1 | cut -d':' -f1)
    
    if [ -n "$NEXT_DEF" ]; then
        # Calculăm linia de final a metodei launch
        END_LINE=$((LAUNCH_LINE + NEXT_DEF - 1))
        
        # Înlocuim metoda launch cu versiunea nouă
        sed -i "${LAUNCH_LINE},${END_LINE}c\\    def launch(self, server_name: str = \"0.0.0.0\", server_port: int = 7860, debug: bool = False):\\n        \"\"\"\\n        Lansează interfața web\\n        \\n        Args:\\n            server_name: Adresa IP a serverului\\n            server_port: Portul serverului\\n            debug: Activează modul debug\\n        \"\"\"\\n        logger.info(f\"Launching Gradio interface: {{'server_name': '{server_name}', 'server_port': {server_port}, 'share': True, 'debug': {debug}}}\")\\n        launch_kwargs = {\\n            \"server_name\": server_name,\\n            \"server_port\": server_port,\\n            \"share\": True,  # IMPORTANT: Setăm share=True pentru a permite accesul extern\\n            \"debug\": debug\\n        }\\n        \\n        try: \\n            self.app.launch(**launch_kwargs)\\n        except Exception as e:\\n            logger.critical(f\"Gradio launch failed: {e}\")\\n            raise" interface/ui.py
        
        echo "   Metoda launch modificată cu succes în interface/ui.py."
    else
        echo "   Nu am putut determina sfârșitul metodei launch. Înlocuim direct fișierul."
        cp interface/ui.py.modified interface/ui.py
    fi
else
    echo "   Nu am putut găsi metoda launch în interface/ui.py. Înlocuim direct fișierul."
    cp interface/ui.py.modified interface/ui.py
fi

# 3. Creăm un pipeline HiDream minimal
echo "3. Creăm un pipeline HiDream minim funcțional..."

# Creăm structura de directoare
mkdir -p models/custom_pipelines/hidream
touch models/custom_pipelines/hidream/__init__.py

# Creăm un stub pentru pipeline-ul HiDream
cat > models/custom_pipelines/hidream/pipeline.py << 'EOF'
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
HiDream Pipeline Stub - înlocuiește importul lipsă
"""

from diffusers import StableDiffusionXLPipeline, DiffusionPipeline
import torch
import logging

logger = logging.getLogger(__name__)

class HiDreamImagePipeline(StableDiffusionXLPipeline):
    """
    O versiune minimală a HiDreamImagePipeline bazată pe StableDiffusionXLPipeline
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        logger.info("HiDreamImagePipeline inițializat (versiune stub)")
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        """
        Încarcă modelul preantrenat și creează o instanță a pipeline-ului.
        """
        logger.info(f"Încărcare model HiDream: {pretrained_model_name_or_path}")
        
        # Folosim StableDiffusionXLPipeline pentru a încărca modelul
        pipeline = StableDiffusionXLPipeline.from_pretrained(
            pretrained_model_name_or_path,
            **kwargs
        )
        
        # Convertim pipeline-ul la HiDreamImagePipeline
        pipeline.__class__ = cls
        
        return pipeline
EOF

# 4. Modificăm hidream_model.py pentru a folosi custom pipeline
echo "4. Modificăm importurile în hidream_model.py..."

# Facem backup dacă nu există
if [ ! -f "models/hidream_model.py.backup" ]; then
    cp models/hidream_model.py models/hidream_model.py.backup
fi

# Citim fișierul pentru a găsi importul problematic
DIFFUSERS_IMPORT_LINE=$(grep -n "from diffusers import" models/hidream_model.py | head -1 | cut -d':' -f1)
HIDREAM_IMPORT_LINE=$(grep -n "from diffusers.pipelines.hidream_image" models/hidream_model.py | head -1 | cut -d':' -f1)

# Dacă am găsit importul problematic, îl înlocuim
if [ -n "$HIDREAM_IMPORT_LINE" ]; then
    # Înlocuim importul pipeline-ului HiDream
    sed -i "${HIDREAM_IMPORT_LINE}s/from diffusers.pipelines.hidream_image/# Import local pentru HiDreamImagePipeline\nfrom models.custom_pipelines.hidream.pipeline/" models/hidream_model.py
    echo "   Import HiDreamImagePipeline înlocuit în models/hidream_model.py."
else
    # Dacă nu găsim importul direct, căutăm orice referință la HiDreamImagePipeline
    HIDREAM_REF=$(grep -n "HiDreamImagePipeline" models/hidream_model.py | head -1 | cut -d':' -f1)
    
    if [ -n "$HIDREAM_REF" ] && [ -n "$DIFFUSERS_IMPORT_LINE" ]; then
        # Adăugăm importul după importul diffusers
        sed -i "${DIFFUSERS_IMPORT_LINE}a from models.custom_pipelines.hidream.pipeline import HiDreamImagePipeline" models/hidream_model.py
        echo "   Import HiDreamImagePipeline adăugat în models/hidream_model.py."
    else
        echo "   Nu am putut localiza importurile HiDreamImagePipeline. Verificați manual."
    fi
fi

echo "=== Reparare finală completă! ==="
echo "Rulați din nou ./run_fusionframe.sh pentru a porni aplicația."

# Dezactivăm mediul virtual
deactivate
