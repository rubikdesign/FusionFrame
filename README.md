# FusionFrame 2.0

![FusionFrame Logo](https://via.placeholder.com/800x200?text=FusionFrame+2.0)

## Overview

FusionFrame 2.0 is an advanced AI-powered image editing application that transforms natural language instructions into precise image manipulations. Built with state-of-the-art AI models and a comprehensive pipeline architecture, FusionFrame offers a seamless interface for complex image editing tasks.

## Key Features

- **Natural Language Editing**: Edit images using simple text instructions like "remove the car" or "change hair color to blonde"
- **Multiple Editing Operations**:
  - Object Removal: Seamlessly remove objects or people with intelligent background reconstruction
  - Color Transformation: Change the color of objects, hair, clothing, etc.
  - Background Replacement: Replace or modify image backgrounds
  - Object Addition: Add objects like glasses, hats, or other elements naturally
- **Advanced AI Models**:
  - Primary model: HiDream-E1-Full for instruction-based editing
  - Backup model: FLUX.1-dev for specialized cases
  - Auxiliary models: SAM for segmentation, ControlNet for guided generation
- **Hybrid Mask Generation**: Combines multiple models for precise segmentation
- **LoRA Support**: Load up to 3 LoRAs simultaneously for custom editing styles
- **Post-Processing Enhancements**: Automatic detail enhancement and artifact removal

## System Requirements

- **Hardware**:
  - CUDA-compatible GPU with 6GB+ VRAM (recommended)
  - 16GB+ RAM
  - 20GB+ free disk space for models
- **Software**:
  - Python 3.8 or higher
  - PyTorch 2.0+
  - CUDA Toolkit 11.7+ (for GPU acceleration)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/rubikdesign/FusionFrame.git
cd FusionFrame
```

2. Run the install script to create a virtual environment and install dependencies:
```bash
chmod +x install.sh
./install.sh
```

3. Start the application:
```bash
./run_fusionframe.sh
```

For low VRAM systems, use:
```bash
./run_fusionframe.sh --low-vram
```

## Usage

1. Upload an image using the interface
2. Enter your editing instruction in natural language (e.g., "remove the person from the background")
3. Adjust the edit strength slider if needed
4. Click "Generate Edit" and wait for the result
5. View the generated mask to see which area was affected

### Example Commands

- "Remove the car from the street"
- "Change hair color to bright pink"
- "Replace background with mountain landscape"
- "Add stylish glasses"
- "Make the shirt blue"
- "Remove watermark from top right corner"

## Advanced Settings

Access advanced settings through the interface:
- **Inference Steps**: Controls generation quality (higher = better but slower)
- **Guidance Scale**: Controls adherence to prompt (higher = more faithful)
- **Detail Enhancement**: Improves details in the final result
- **Face Fixing**: Automatically enhances faces when detected
- **ControlNet**: Toggle for better guidance (disable for lower VRAM usage)

## Project Structure

```
fusionframe/
├── config/                 # Configuration settings
├── core/                   # Core application components
├── models/                 # AI model implementations
├── processing/             # Image processing pipelines
│   ├── pipelines/          # Specialized processing pipelines
├── interface/              # UI components
├── utils/                  # Utility functions
├── loras/                  # Directory for LoRA files
├── logs/                   # Application logs
├── models_cache/           # Downloaded model storage
├── app.py                  # Main application entry point
```

## Extending FusionFrame

FusionFrame is designed to be modular and extensible:

- **Add new models**: Implement a new class inheriting from `BaseModel`
- **Create custom pipelines**: Inherit from `BasePipeline` to add specialized processing
- **Add LoRAs**: Place LoRA files in the `loras/` directory for automatic detection

## Acknowledgments

FusionFrame builds upon several open-source projects:
- [Diffusers](https://github.com/huggingface/diffusers) for Stable Diffusion models
- [Segment Anything](https://github.com/facebookresearch/segment-anything) for segmentation
- [ControlNet](https://github.com/lllyasviel/ControlNet) for guided image generation
- [Gradio](https://github.com/gradio-app/gradio) for the web interface

## License

[MIT License](LICENSE)

## Contact

For issues, feature requests, or contributions, please open an issue on the [GitHub repository](https://github.com/rubikdesign/FusionFrame).
