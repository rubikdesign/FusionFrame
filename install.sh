#!/bin/bash
# Script pentru crearea unui mediu virtual cu dependențe compatibile pentru FusionFrame 2.0

echo "===== Configurare mediu pentru FusionFrame 2.0 ====="
echo "Creez mediul virtual..."

# Crearea mediului virtual
python -m venv fusionframe_env
source fusionframe_env/bin/activate

# Upgrade pip
pip install --upgrade pip

echo "Instalez dependențele..."
pip install -r requirements.txt

# Modifică fișierul diffusers/utils/dynamic_modules_utils.py
sed -i 's/from huggingface_hub import cached_download, hf_hub_download, model_info/from huggingface_hub import hf_hub_download, model_info\ntry:\n    from huggingface_hub import cached_download\nexcept ImportError:\n    cached_download = hf_hub_download/' /workspace/FusionFrame/fusionframe_env/lib/python3.10/site-packages/diffusers/utils/dynamic_modules_utils.py

# Modifică fișierele transformers
sed -i 's/from huggingface_hub import cached_download, hf_hub_url/from huggingface_hub import hf_hub_url\ntry:\n    from huggingface_hub import cached_download\nexcept ImportError:\n    def cached_download(*args, **kwargs):\n        return hf_hub_url(*args, **kwargs)/' /workspace/FusionFrame/fusionframe_env/lib/python3.10/site-packages/transformers/models/cvt/convert_cvt_original_pytorch_checkpoint_to_pytorch.py

sed -i 's/from huggingface_hub import cached_download, hf_hub_url/from huggingface_hub import hf_hub_url\ntry:\n    from huggingface_hub import cached_download\nexcept ImportError:\n    def cached_download(*args, **kwargs):\n        return hf_hub_url(*args, **kwargs)/' /workspace/FusionFrame/fusionframe_env/lib/python3.10/site-packages/transformers/models/deformable_detr/convert_deformable_detr_to_pytorch.py

sed -i 's/from huggingface_hub import cached_download, hf_hub_download/from huggingface_hub import hf_hub_download\ntry:\n    from huggingface_hub import cached_download\nexcept ImportError:\n    cached_download = hf_hub_download/' /workspace/FusionFrame/fusionframe_env/lib/python3.10/site-packages/transformers/models/deprecated/van/convert_van_to_pytorch.py

sed -i 's/from huggingface_hub import cached_download, hf_hub_download, hf_hub_url/from huggingface_hub import hf_hub_download, hf_hub_url\ntry:\n    from huggingface_hub import cached_download\nexcept ImportError:\n    cached_download = hf_hub_download/' /workspace/FusionFrame/fusionframe_env/lib/python3.10/site-packages/transformers/models/deta/convert_deta_resnet_to_pytorch.py

sed -i 's/from huggingface_hub import cached_download, hf_hub_download, hf_hub_url/from huggingface_hub import hf_hub_download, hf_hub_url\ntry:\n    from huggingface_hub import cached_download\nexcept ImportError:\n    cached_download = hf_hub_download/' /workspace/FusionFrame/fusionframe_env/lib/python3.10/site-packages/transformers/models/deta/convert_deta_swin_to_pytorch.py

sed -i 's/from huggingface_hub import cached_download, hf_hub_url/from huggingface_hub import hf_hub_url\ntry:\n    from huggingface_hub import cached_download\nexcept ImportError:\n    def cached_download(*args, **kwargs):\n        return hf_hub_url(*args, **kwargs)/' /workspace/FusionFrame/fusionframe_env/lib/python3.10/site-packages/transformers/models/dpt/convert_dpt_hybrid_to_pytorch.py

sed -i 's/from huggingface_hub import cached_download, hf_hub_url/from huggingface_hub import hf_hub_url\ntry:\n    from huggingface_hub import cached_download\nexcept ImportError:\n    def cached_download(*args, **kwargs):\n        return hf_hub_url(*args, **kwargs)/' /workspace/FusionFrame/fusionframe_env/lib/python3.10/site-packages/transformers/models/dpt/convert_dpt_to_pytorch.py

sed -i 's/from huggingface_hub import cached_download, hf_hub_url/from huggingface_hub import hf_hub_url\ntry:\n    from huggingface_hub import cached_download\nexcept ImportError:\n    def cached_download(*args, **kwargs):\n        return hf_hub_url(*args, **kwargs)/' /workspace/FusionFrame/fusionframe_env/lib/python3.10/site-packages/transformers/models/regnet/convert_regnet_seer_10b_to_pytorch.py

sed -i 's/from huggingface_hub import cached_download, hf_hub_url/from huggingface_hub import hf_hub_url\ntry:\n    from huggingface_hub import cached_download\nexcept ImportError:\n    def cached_download(*args, **kwargs):\n        return hf_hub_url(*args, **kwargs)/' /workspace/FusionFrame/fusionframe_env/lib/python3.10/site-packages/transformers/models/regnet/convert_regnet_to_pytorch.py

echo "Creez script de execuție..."
# Creăm scriptul de execuție
cat > run_fusionframe.sh << 'EOL'
#!/bin/bash
source fusionframe_env/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)
python app.py "$@"
EOL

chmod +x run_fusionframe.sh

echo "===== Instalare completă! ====="
echo "Pentru a rula aplicația, folosește comanda: ./run_fusionframe.sh"
echo "NOTĂ: La prima rulare, modelele vor fi descărcate automat (poate dura câteva minute)."