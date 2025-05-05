# FusionFrame

## Advanced Face Fusion Application

FusionFrame is a powerful tool for generating composite images that seamlessly blend a person's face from a reference image into a target pose while preserving identity and facial features.

![FusionFrame Example](https://user-images.githubusercontent.com/example.png)

## Features

- **Identity Preservation**: Advanced face detection and feature transfer to maintain the person's identity
- **Selective Feature Transfer**: Precise control over which facial features (eyes, nose, mouth) are transferred
- **AI-Powered Generation**: Uses state-of-the-art Stable Diffusion models for high-quality results
- **Advanced Control**: Support for LoRA adapters, ControlNet pose guidance, and IP-Adapter face preservation
- **User-Friendly Interface**: Intuitive Gradio UI with real-time progress updates

## Installation

### Prerequisites

- Python 3.8+ 
- CUDA-capable GPU (optional but recommended)

### Quick Install

1. Clone the repository:
```bash
git clone https://github.com/rubikdesign/FusionFrame.git
cd fusionframe
```

2. Run the setup script:
```bash
chmod +x setup.sh
./setup.sh
```

### Manual Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Install IP-Adapter:
```bash
pip install git+https://github.com/tencent-ailab/IP-Adapter.git
```

3. Download models:
```bash
mkdir -p ~/.fusionframe/IP-Adapter/models
git clone https://huggingface.co/h94/IP-Adapter ~/.fusionframe/IP-Adapter/models
```

## Usage

### Starting the UI

```bash
python -m fusionframe
```

### Basic Usage

1. Upload a **Reference Image** containing the person's face you want to preserve
2. Upload a **Pose Image** with the target pose/scene
3. Adjust settings as needed (or use defaults)
4. Click **Generate Composite Image**

### Advanced Options

- **Face Enhancement**: Controls how strongly facial features are preserved
- **ControlNet Pose**: Better maintains the exact pose from the target image
- **Two-Stage Refiner**: Applies a second refinement pass for higher quality
- **LoRA Integration**: Apply style adaptations using LoRA models

## Directory Structure

```
fusionframe/
├── core.py         # Core functionality
├── ui.py           # Gradio user interface
├── config.py       # Configuration settings
├── utils/          # Utility functions
│   ├── face_utils.py  # Face processing utilities
│   └── io_utils.py    # File operations
└── plugins/        # Plugin extensions
    ├── ip_adapter.py  # IP-Adapter integration
    └── controlnet.py  # ControlNet integration
```

## Model Sources

FusionFrame uses models from the following sources:

- [Stable Diffusion XL](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) 
- [IP-Adapter](https://github.com/tencent-ailab/IP-Adapter)
- [ControlNet](https://github.com/lllyasviel/ControlNet)

## Acknowledgments

- Face recognition using [face_recognition](https://github.com/ageitgey/face_recognition)
- UI powered by [Gradio](https://gradio.app/)
- Diffusion models from [Hugging Face](https://huggingface.co/) 

## License

This project is licensed under the MIT License - see the LICENSE file for details.
