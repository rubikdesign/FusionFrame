"""
FusionFrame: Advanced Face Fusion Application

FusionFrame is a tool for generating composite images that seamlessly
blend a person's face from a reference image into a target pose while
preserving identity and facial features.

Core Features:
- Face detection and feature extraction
- Selective facial feature transfer (eyes, nose, mouth)
- High-quality image generation with Stable Diffusion
- Support for LoRA adapters, ControlNet, and IP-Adapter

This package is organized into several modules:
- core: Main functionality for model loading and image generation
- ui: Gradio user interface
- utils: Utility functions for face processing and file operations
- plugins: Extensions for IP-Adapter and ControlNet integration
- config: Configuration settings
"""

import logging
import os
from pathlib import Path

__version__ = "0.1.0"

# Set up package-level logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)

# Make core components available at package level
from .core import FusionFrame
from . import utils
from . import plugins
from . import config

# Function to launch the UI
def launch_ui(share=False):
    """
    Launch the FusionFrame UI.
    
    Args:
        share (bool): Whether to create a public URL for sharing
        
    Returns:
        The launched Gradio application
    """
    from .ui import build_gradio_interface
    app = build_gradio_interface()
    app.launch(share=share)
    return app

# Create a convenient command-line entry point
def main():
    """Command-line entry point."""
    launch_ui(share=True)

if __name__ == "__main__":
    main()
