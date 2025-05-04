# FusionFrame App

FusionFrame is a Gradio application that allows users to generate composite images by blending a reference image of a person with a pose/scene from another image. The application uses advanced AI image generation models to create realistic results.

## Features

- Upload a reference image (e.g., portrait of a person) and a pose image (e.g., person in a specific scenario)
- Generate realistic composite images where the reference person is placed in the pose/scenario of the second image
- Customize attire, décor, and other details through prompt engineering
- Support for up to 5 LoRAs with configurable weights and activation toggles
- Adjust image dimensions (width/height) or retain original sizes
- Configure inference parameters (steps, strength, guidance scale, etc.)
- Select from multiple advanced AI models, including SDXL Refiner and Realistic Vision models
- Choose different sampling methods for generation
- Track generation progress in real-time
- Save seeds for reproducible results
- GPU optimizations with xFormers support

## Installation

1. Clone this repository or download the files
2. Run the environment setup script to install all necessary dependencies:

```bash
bash setup_environment.sh
```

3. Run the application:

```bash
python fusion_frame_app.py
```

## Usage

1. **Upload Images**:
   - Upload a reference image of the person you want to include in the composite
   - Upload a pose image showing the position or scenario you want to replicate

2. **Configure Settings** (optional):
   - Adjust dimensions (width/height)
   - Add custom prompts to guide the generation
   - Customize attire, décor, and other details
   - Select different models or samplers
   - Configure technical parameters (steps, strength, guidance, seed)
   - Add LoRAs to enhance the generation (up to 5)

3. **Generate**:
   - Click the "Generate Composite Image" button
   - Wait for the progress bar to complete
   - View the generated result

4. **Iterate**:
   - Adjust settings as needed
   - Save the seed number to reproduce good results
   - Try different customization prompts

## Advanced Options

- **Models**: The app comes with "SDXL Refiner 1.0" as the default model, but you can select others from the dropdown menu
- **Samplers**: Different sampling methods affect the generation quality and speed
- **Strength**: Controls how much of the pose image to preserve (higher values allow more creative freedom)
- **Guidance Scale**: Determines how closely the generation follows the prompt (higher values = more prompt adherence)
- **Inference Steps**: More steps typically yield better quality but take longer to generate
- **LoRAs**: Up to 5 LoRAs can be used simultaneously with individual weight controls
  - Each LoRA can be toggled on/off with a checkbox
  - Weight sliders control how strongly each LoRA affects the generation
  - LoRAs are automatically detected from the "Loras" folder in the application directory
  - You can rescan the LoRAs folder if you add new files while the app is running

## Technical Details

- Uses the Diffusers library's StableDiffusionXLImg2ImgPipeline pipeline
- Implements image blending techniques to transfer reference image features
- Optionally uses IP-Adapter for improved identity preservation in SDXL models
- Provides several sampling methods for different quality/speed tradeoffs
- Caches downloaded models in a user directory for faster subsequent runs
- Uses xFormers for memory-efficient attention when available

## Troubleshooting

- **Compatibility**: The app requires PyTorch 2.0+ and compatible versions of diffusers
- **GPU Required**: For optimal performance, a CUDA-compatible GPU is recommended
- **Memory Issues**: If you encounter CUDA out-of-memory errors, try reducing the image dimensions
- **First-Run Delay**: The first generation may take longer as the model is downloaded and loaded
- **LoRA Issues**: If LoRAs don't load correctly, ensure they're in .safetensors format and placed in the Loras directory
