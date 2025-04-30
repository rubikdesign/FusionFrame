# FusionFrame - Advanced Image Generator: Virtual Try-On + Background Integration

## Introduction

FusionFrame is a powerful AI-powered platform that enables the generation of high-quality images by combining people, clothing items, and backgrounds using advanced AI models such as Stable Diffusion XL, IP-Adapter, and ControlNet. The application offers detailed control over the generation process with numerous customization options, making it ideal for virtual try-on and creative image generation.

![Platform Interface](https://i.imgur.com/placeholder.jpg)

## System Requirements

- Python 3.8+
- CUDA 11.8+ (for GPU acceleration)
- GPU with minimum 8GB VRAM (16GB+ recommended for SDXL models)
- Minimum 16GB RAM
- Storage: 20GB+ for models and dependencies

## Installation

### Local Installation

1. Clone the repository or download the application files:
```bash
git clone https://github.com/rubikdesign/FusionFrame.git
cd FusionFrame
```

2. Run the installation script to set up the environment:
```bash
python install_runpod.py
```

3. Launch the application:
```bash
python app.py
```

### RunPod Deployment

For deployment on RunPod or similar cloud GPU services:

1. Clone the repository:
```bash
git clone https://github.com/rubikdesign/FusionFrame.git
cd FusionFrame
```

2. Run the RunPod-specific installer:
```bash
python install_runpod.py
```

3. Launch the application with default model download:
```bash
python app.py --download-default
```

4. Access the web interface at: `http://localhost:7860`

## Command Line Arguments

The application supports several command line arguments:
- `--port <PORT>`: Specify a custom port (default: 7860)
- `--share`: Make the application publicly accessible
- `--download-default`: Download default models at startup
- `--test-installation`: Run a test to verify proper installation

## User Guide

### 1. Upload Images

- **Person Image**: Upload an image of a person
- **Clothing Image (Optional)**: Upload an image of clothing items
- **Background Image (Optional)**: Upload an image of a desired background

### 2. Prompt Settings

- **Positive Prompt**: Enter descriptions for what you want to appear in the image. The application will automatically enhance the prompt based on uploaded images.
- **Negative Prompt**: Specify what you do not want to appear in the generated image (default values cover common issues).

### 3. Model Selection

- **Model**: Choose the generation model
  - Options include: RealisticVision XL, SDXL 1.0, Juggernaut XL, DreamShaper XL, SD 2.1, SD 1.5
- **VAE**: Choose a VAE for improved visual quality
  - Options include: SDXL VAE, SD 1.5 VAE, Default
- **Scheduler**: Select the diffusion scheduler algorithm
  - Options include: DPM++ 2M Karras, DPM++ SDE Karras, Euler A, DDIM
- **Seed**: Set a specific seed for reproducibility (-1 for random)

### 4. Advanced Settings

#### ControlNet
Controls the pose, structure, and composition of generated images:
- **ControlNet Type**: Select from multiple control types
  - Options include: Depth (SDXL), Pose (SDXL), Canny Edge (SDXL), Lineart, Soft Edge
- **Conditioning Scale**: Adjust how strongly the control influences the generation (0.6 is default)

#### IP-Adapter
Transfers visual style from reference images:
- **IP-Adapter Model**: Choose the appropriate adapter
  - Options include: IP-Adapter Plus (SDXL), IP-Adapter Plus Face (SDXL)
- **IP-Adapter Scale**: Adjust the intensity of reference image influence (0.6 is default)

#### LoRA
Fine-tune the style and content of generated images:
- **LoRA Upload**: Upload custom .safetensors LoRA files
- **LoRA Selection**: Configure up to 5 LoRAs simultaneously
- **Weight**: Adjust the intensity of each LoRA (0.7 is default)

### 5. Generation Settings

- **Denoising Strength**: Controls preservation of original images (0.75 is default)
- **Inference Steps**: Number of diffusion steps (30 is default)
- **Guidance Scale**: Controls adherence to prompt (7.5 is default)
- **Number of Images**: How many images to generate in one batch (1-4)
- **Image Dimensions**: Width and height of output images (768×768 is default)

### 6. Generation and Results

- Click **"Generate Images"** to start the generation process
- Generated images will appear in the gallery
- The used seed is displayed for future reproduction
- Use **"Reset"** to clear all settings

## Default Models

FusionFrame comes with pre-configured default settings for optimal results:

- **Default Model**: RealisticVision XL
- **Default VAE**: SDXL VAE
- **Default ControlNet**: Depth (SDXL)
- **Default IP-Adapter**: IP-Adapter Plus (SDXL)

These default models will be downloaded automatically when using the `--download-default` parameter.

## Tips for Optimal Results

1. **Working with Person Images**:
   - Use clear, well-lit images with the person in a neutral pose
   - For best results, use images with a simple background
   - Front-facing images work best with ControlNet

2. **Effective Prompts**:
   - Include specific details about the desired clothing and style
   - Use descriptive terms like "high quality, photorealistic, masterpiece, detailed"
   - Specify lighting conditions like "studio lighting" or "natural light"

3. **Model and ControlNet Selection**:
   - For best clothing try-on results, use RealisticVision XL with Depth ControlNet
   - For creative interpretations, try Juggernaut XL with Pose ControlNet
   - When working with detailed clothing, increase the ControlNet conditioning scale

4. **Memory Optimization**:
   - Use 768×768 resolution for most use cases (increase only if needed)
   - If experiencing memory issues, try SD 1.5 instead of XL models
   - Reduce the number of inference steps to 20-25 for faster generation

## Troubleshooting

1. **Model Loading Errors**:
   - Check available VRAM on your GPU
   - Try a smaller model (SD 1.5 instead of SDXL)
   - Run with `--test-installation` to verify your environment

2. **Image Quality Issues**:
   - Increase inference steps (40+)
   - Try a different VAE
   - Add more quality terms to your prompt
   - Adjust guidance scale (7-9 for balanced results)

3. **Memory-Related Crashes**:
   - Reduce image dimensions
   - Disable unused features (ControlNet or IP-Adapter)
   - Close other GPU-intensive applications
   - Consider upgrading your GPU or using a cloud GPU service

4. **HuggingFace Errors**:
   - If you encounter "cached_download not found" errors, reinstall huggingface_hub:
     ```bash
     pip install huggingface_hub>=0.19.4
     ```

## Technical Information

FusionFrame integrates several key technologies:

- **Diffusers Library**: For running Stable Diffusion models
- **ControlNet**: For structural and pose control
- **IP-Adapter**: For style transfer from reference images
- **LoRA**: For customizing model behavior with less computational overhead
- **Gradio**: For the web interface

The application architecture is designed with memory efficiency in mind, using:
- Attention slicing
- Model CPU offloading
- Optimized model loading

## Additional Resources

- [HuggingFace Diffusers Documentation](https://huggingface.co/docs/diffusers/index)
- [Control-Net Paper](https://arxiv.org/abs/2302.05543)
- [IP-Adapter GitHub Repository](https://github.com/tencent-ailab/IP-Adapter)
- [Stable Diffusion Models](https://huggingface.co/models?other=stable-diffusion)

## License

This project is licensed under the MIT License.

## Acknowledgments

- Hugging Face for the Diffusers library
- Stability AI for Stable Diffusion models
- Tencent AI Lab for IP-Adapter
- RunPod for cloud GPU infrastructure
- The open-source AI community for LoRAs and other resources
