# Patch file to add cached_download compatibility to huggingface_hub
import os
import sys
import huggingface_hub
from huggingface_hub.file_download import hf_hub_download

# Add a compatibility function for cached_download
def cached_download(*args, **kwargs):
    """Compatibility wrapper for old cached_download function"""
    print("Warning: cached_download is deprecated, using hf_hub_download instead")
    return hf_hub_download(*args, **kwargs)

# Add the function to the huggingface_hub module
if not hasattr(huggingface_hub, 'cached_download'):
    setattr(huggingface_hub, 'cached_download', cached_download)
    print("Added cached_download compatibility function to huggingface_hub")