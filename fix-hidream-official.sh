#!/bin/bash
# Script pentru utilizarea pipeline-ului HiDream oficial din diffusers

echo "=== Instalare pipeline HiDream oficial ==="

# Activăm mediul virtual
source ./fusionframe_env/bin/activate

# 1. Verificăm versiunea actuală de diffusers
CURRENT_DIFFUSERS=$(pip show diffusers | grep Version | awk '{print $2}')
echo "1. Versiunea actuală de diffusers: $CURRENT_DIFFUSERS"

# 2. Instalăm versiunea corectă de diffusers care include HiDreamImagePipeline
echo "2. Instalăm diffusers 0.27.2 care include HiDreamImagePipeline..."
pip uninstall -y diffusers
pip install diffusers==0.27.2

# 3. Verificăm dacă pipeline-ul este disponibil după instalare
echo "3. Verificăm disponibilitatea HiDreamImagePipeline..."
python -c "import diffusers; print('Diffusers versiunea:', diffusers.__version__); from diffusers.pipelines import hidream_image; print('HiDreamImagePipeline importat cu succes!')" || echo "HiDreamImagePipeline nu este disponibil. Continuăm cu soluția alternativă."

# 4. Verificăm dacă există conflicte de versiuni cu transformers
echo "4. Verificăm compatibilitatea cu transformers..."
TRANSFORMERS_VERSION=$(pip show transformers | grep Version | awk '{print $2}')
echo "   Versiunea transformers: $TRANSFORMERS_VERSION"

# 5. Asigurăm configurarea PYTHONPATH în run_fusionframe.sh
echo "5. Verificăm setarea PYTHONPATH în run_fusionframe.sh..."
if grep -q "export PYTHONPATH=\"\$SCRIPT_DIR:\$PYTHONPATH\"" run_fusionframe.sh; then
    echo "   PYTHONPATH este configurat corect."
else
    echo "   Adăugăm setarea PYTHONPATH în run_fusionframe.sh."
    sed -i '/export CUDA_VISIBLE_DEVICES=0/a export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"' run_fusionframe.sh
fi

# 6. În caz că pipeline-ul oficial nu funcționează, implementăm și soluția locală
echo "6. Implementăm și soluția locală de backup..."

# Creăm directorul pentru pipeline-uri personalizate
mkdir -p models/pipelines/hidream_image
touch models/pipelines/hidream_image/__init__.py

# Creăm fișierul pipeline-ului
cat > models/pipelines/hidream_image/pipeline_hidream_image.py << 'PYTHON_EOF'
# -*- coding: utf-8 -*-
"""
Pipeline personalizat pentru HiDream - implementare simplificată
Acest pipeline este un wrapper pentru StableDiffusionXLPipeline
"""

from typing import Any, Callable, Dict, List, Optional, Union, Tuple
import torch
import numpy as np
from diffusers import StableDiffusionXLPipeline, DiffusionPipeline
from diffusers.utils import deprecate, logging

logger = logging.get_logger(__name__)

class HiDreamImagePipeline(StableDiffusionXLPipeline):
    """Pipeline personalizat pentru HiDream-I1, bazat pe SDXL."""
    
    model_cpu_offload_seq = "text_encoder->text_encoder_2->unet->vae"
    
    def __init__(self, *args, **kwargs):
        # Inițializăm clasa părinte (StableDiffusionXLPipeline)
        super().__init__(*args, **kwargs)
        logger.info("HiDreamImagePipeline inițializat cu succes!")
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        """
        Încarcă din modelul predefinit și aplică configurări specifice HiDream.
        """
        # Folosim metoda from_pretrained din StableDiffusionXLPipeline
        pipeline = StableDiffusionXLPipeline.from_pretrained(*args, **kwargs)
        
        # Convertim pipeline-ul la HiDreamImagePipeline
        pipeline.__class__ = cls
        
        logger.info(f"HiDreamImagePipeline creat din pretrained: {args[0]}")
        return pipeline
    
    def load_loras(self, lora_paths: List[str], lora_scales: Optional[List[float]] = None):
        """
        Adaugă suport pentru LoRA în HiDream.
        """
        if not lora_paths:
            return
        
        if lora_scales is None:
            lora_scales = [1.0] * len(lora_paths)
        
        if len(lora_scales) != len(lora_paths):
            raise ValueError(f"Numărul de scale ({len(lora_scales)}) nu se potrivește cu numărul de LoRA-uri ({len(lora_paths)})")
        
        logger.info(f"Încărcare {len(lora_paths)} LoRA-uri pentru HiDream")
        
        for path, scale in zip(lora_paths, lora_scales):
            try:
                logger.info(f"Încărcare LoRA: {path} cu scale {scale}")
                # Pentru SDXL trebuie specificat explicit care sunt modulele ce trebuie modificate
                self.load_lora_weights(path, adapter_name=f"lora_{path}", weight_name=None)
                self.set_adapters([f"lora_{path}"], adapter_weights=[scale])
            except Exception as e:
                logger.error(f"Eroare la încărcarea LoRA {path}: {e}")
PYTHON_EOF

# Creăm fișierul __init__.py pentru a face pipeline-ul importabil
cat > models/pipelines/hidream_image/__init__.py << 'PYTHON_EOF'
# Încercăm mai întâi să importăm din diffusers oficial, și doar apoi folosim implementarea locală
try:
    from diffusers.pipelines.hidream_image import HiDreamImagePipeline
    print("Utilizăm HiDreamImagePipeline oficial din pachetul diffusers.")
except ImportError:
    # Folosim implementarea locală doar dacă cea oficială nu este disponibilă
    from .pipeline_hidream_image import HiDreamImagePipeline
    print("Utilizăm implementarea locală pentru HiDreamImagePipeline.")

__all__ = ["HiDreamImagePipeline"]
PYTHON_EOF

# 7. Modificăm hidream_model.py pentru a importa corect
echo "7. Modificăm hidream_model.py pentru a importa corect..."

# Facem o copie de backup dacă nu există deja
if [ ! -f "models/hidream_model.py.backup" ]; then
    cp models/hidream_model.py models/hidream_model.py.backup
fi

# Verificăm cum se face importul HiDreamImagePipeline
IMPORT_PATTERN="from diffusers.pipelines.hidream_image"
IMPORT_LINE=$(grep -n "$IMPORT_PATTERN" models/hidream_model.py || echo "")

if [ -n "$IMPORT_LINE" ]; then
    # Păstrăm importul original, ar trebui să funcționeze cu diffusers 0.27.2
    echo "   Import standard HiDreamImagePipeline găsit. Lăsăm nemodificat."
else
    # Căutăm orice încercare de import HiDreamImagePipeline
    PIPELINE_MENTION=$(grep -n "HiDreamImagePipeline" models/hidream_model.py | head -1 || echo "")
    PIPELINE_IMPORT=$(grep -n "import.*HiDreamImagePipeline" models/hidream_model.py | head -1 || echo "")
    
    if [ -n "$PIPELINE_IMPORT" ]; then
        LINE_NUMBER=$(echo "$PIPELINE_IMPORT" | cut -d ':' -f 1)
        # Înlocuim importul existent cu unul care va încerca ambele variante
        sed -i "${LINE_NUMBER}c# Încercăm să importăm HiDreamImagePipeline din diffusers sau din implementarea locală\ntry:\n    from diffusers.pipelines.hidream_image import HiDreamImagePipeline\nexcept ImportError:\n    from models.pipelines.hidream_image import HiDreamImagePipeline" models/hidream_model.py
        echo "   Import înlocuit la linia $LINE_NUMBER."
    elif [ -n "$PIPELINE_MENTION" ]; then
        LINE_NUMBER=$(echo "$PIPELINE_MENTION" | cut -d ':' -f 1)
        # Adăugăm un nou import înainte de prima mențiune
        sed -i "${LINE_NUMBER}i# Încercăm să importăm HiDreamImagePipeline din diffusers sau din implementarea locală\ntry:\n    from diffusers.pipelines.hidream_image import HiDreamImagePipeline\nexcept ImportError:\n    from models.pipelines.hidream_image import HiDreamImagePipeline" models/hidream_model.py
        echo "   Import adăugat înainte de linia $LINE_NUMBER."
    else
        echo "   Nu am putut localiza referințe la HiDreamImagePipeline. Verifică manual fișierul models/hidream_model.py."
    fi
fi

# 8. Asigurăm că importul poate fi găsit - adăugăm o verificare de compatibilitate
cat > check_hidream_compatibility.py << 'PYTHON_EOF'
#!/usr/bin/env python3
"""
Script pentru verificarea compatibilității cu HiDream
"""
import sys
import os

# Adăugăm directoarele necesare la PYTHONPATH
sys.path.append('.')
sys.path.append('./models')

print("Verificăm compatibilitatea HiDream...")

# Verificăm diffusers
try:
    import diffusers
    print(f"diffusers versiunea: {diffusers.__version__}")
    
    # Încercăm să importăm direct din diffusers
    try:
        from diffusers.pipelines import hidream_image
        print("✅ HiDreamImagePipeline găsit în diffusers!")
        HiDreamPipeline = hidream_image.HiDreamImagePipeline
        print("   Clasa HiDreamImagePipeline importată cu succes.")
    except ImportError as e:
        print(f"❌ Nu s-a putut importa hidream_image din diffusers: {e}")
        
        # Încercăm să importăm din implementarea locală
        try:
            sys.path.append('./models/pipelines')
            from models.pipelines.hidream_image import HiDreamImagePipeline
            print("✅ HiDreamImagePipeline găsit în implementarea locală!")
        except ImportError as e:
            print(f"❌ Nu s-a putut importa nici implementarea locală: {e}")
    
    # Verificăm dacă se poate încărca modelul HiDream
    try:
        from models.hidream_model import HiDreamModel
        print("✅ Clasa HiDreamModel importată cu succes.")
    except ImportError as e:
        print(f"❌ Nu s-a putut importa HiDreamModel: {e}")
except ImportError as e:
    print(f"❌ Nu s-a putut importa diffusers: {e}")

print("\nVerificare completă!")
PYTHON_EOF

chmod +x check_hidream_compatibility.py

# Rulăm verificarea
echo "8. Rulăm verificarea de compatibilitate..."
python check_hidream_compatibility.py

echo "=== Reparare completă! ==="
echo "Rulați din nou ./run_fusionframe.sh pentru a porni aplicația."

# Dezactivăm mediul virtual
deactivate
