## FusionFrame -  Advanced Image Generation Platform: Virtual Try-On + Background Integration

## Introduction

This platform enables the generation of high-quality images by combining a model (person), clothing items, and backgrounds using advanced AI models such as Stable Diffusion XL, IP-Adapter, and ControlNet. The application offers detailed control over the generation process with numerous customization options.

![Platform Interface](https://i.imgur.com/placeholder.jpg)

## System Requirements

- Python 3.8+
- CUDA 11.8+ (for GPU acceleration)
- GPU with minimum 8GB VRAM (12GB+ recommended for SDXL)
- Minimum 16GB RAM
- Storage: 20GB+ for models and dependencies

## Installation

1. Clone the repository or download the application files.
2. Execute the installation script to configure the environment and install all dependencies:

```bash
chmod +x install.sh
./install.sh
```

For Windows:
```bash
install.bat
```

3. After installation, activate the virtual environment:
```bash
source clothing_generator_env/bin/activate
```

For Windows:
```bash
clothing_generator_env\Scripts\activate
```

4. Run the application:
```bash
python app.py
```

5. Access the web interface at: `http://localhost:7860`

## User Guide

### 1. Image Upload

- **Model Image**: Upload an image of a woman/person
- **Clothing Image**: (Optional) Upload an image of a dress, lingerie, or other clothing items
- **Background Image**: (Optional) Upload an image of the desired location/background

### 2. Prompt Configuration

- **Positive Prompt**: Enter descriptions for what you want to appear in the image. The application will automatically add context based on the uploaded images. Use quality keywords (photorealistic, high quality, detailed) for better results.
- **Negative Prompt**: Specify what you do not want to appear in the generated image. Common issues are pre-filled here to be avoided.

### 3. AI Model Selection

- **Model**: Choose the generation model (SDXL 1.0, SD 1.5, Juggernaut, etc.)
- **VAE**: (Optional) Choose a VAE for improved visual quality
- **Scheduler**: The scheduler choice influences the quality and style of generation
- **Seed**: Set a specific seed for result reproducibility (-1 for random)

### 4. Advanced Settings

#### ControlNet
Allows control of the model's position and posture:
- **ControlNet Type**: Pose (for positioning), Canny Edge (for contours), Depth (for spatial relationships), etc.
- **Conditioning Scale**: Adjusts the intensity of the ControlNet effect (0.5 is balanced)

#### IP-Adapter
Transfers visual style and details from reference images:
- **IP-Adapter Model**: Choose the appropriate model for the desired generation
- **IP-Adapter Scale**: Adjust the intensity of the reference image influence

#### LoRA
Customizes the style and content of generated images:
- **LoRA Upload**: Upload .safetensors files containing LoRAs
- **LoRA Selection**: You can activate and configure up to 5 LoRAs simultaneously
- **Weight**: Adjust the intensity of each LoRA (0.7 is the standard value)

### 5. Generation Settings

- **Denoising Strength**: Controls how much of the original images is preserved (lower values preserve more details)
- **Inference Steps**: Number of steps in generation (30-40 is sufficient for good quality)
- **Guidance Scale**: How strictly the model follows the prompt (7.5 is balanced)
- **Dimensions**: Width and height of the generated image
- **Number of Images**: How many images to generate in a single processing

### 6. Generation and Results

- Press the **"Generate Images"** button to start the process
- Generated images will appear in the gallery
- The seed used will be displayed for reproduction
- Use the **"Reset"** button to clear all settings

## Tips for Optimal Results

1. **Combining Models and Control**:
   - To "dress" the model, use ControlNet Pose type with a reference image
   - Use IP-Adapter with the clothing image to preserve the exact style

2. **Effective Prompts**:
   - Include specific details about clothing: "red dress with lace", "silk blouse"
   - Add stylistic elements: "studio lighting", "professional photography"
   - Specify position: "full body", "standing pose", "sitting"

3. **LoRA Optimization**:
   - Use style LoRAs (realistic, portrait, fashion) with weights of ~0.6-0.7
   - Use specific clothing LoRAs with weights of ~0.5-0.6

4. **Adjusting Generation Parameters**:
   - Inference steps: 30 for fast generation, 50+ for superior details
   - Guidance scale: 7-9 for realistic results, 10-15 for strict adherence to prompt

## Troubleshooting Common Issues

1. **Model Loading Errors**:
   - Check available VRAM space
   - Try a smaller model (SD 1.5 instead of SDXL)
   - Enable memory offloading in code

2. **Quality Issues**:
   - Use more inference steps (40+)
   - Adjust the prompt to include quality terms
   - Try a different VAE

3. **Unrealistic Clothing**:
   - Use ControlNet to better define body position
   - Increase IP-Adapter scale if using a clothing reference
   - Include more detailed descriptions of the material in the prompt

4. **Memory Crashes**:
   - Reduce image size (768x768 instead of 1024x1024)
   - Disable controlnet or ip-adapter if not strictly necessary
   - Reduce the number of active LoRAs simultaneously

## Key Features

- Multiple AI model support (SDXL, SD 1.5, Juggernaut XL)
- Integration with IP-Adapter for style transfer
- ControlNet for pose and composition control
- Support for up to 5 simultaneous LoRAs
- Adjustable VAEs for improved visual quality
- Multiple schedulers for different generation styles
- Advanced memory management optimizations
- Detailed logging and error handling

## Additional Resources

- [Compatible Models List](https://huggingface.co/models?other=stable-diffusion)
- [Prompt Engineering Guide](https://prompthero.com/stable-diffusion-prompt-guide)
- [LoRA Resources](https://civitai.com/models)
- [ControlNet Documentation](https://huggingface.co/docs/diffusers/main/en/api/pipelines/controlnet)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Hugging Face for the Diffusers library
- Stability AI for Stable Diffusion models
- Tencent AI Lab for IP-Adapter
- The open-source AI community for LoRAs and other resources
