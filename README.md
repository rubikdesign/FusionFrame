# FusionFrame

![FusionFrame Logo](https://via.placeholder.com/800x200?text=FusionFrame)

## Advanced AI-Powered Image Editing Through Text

FusionFrame is a powerful and intuitive application that leverages state-of-the-art AI models to edit images through simple text instructions. Built for both casual users and professionals, FusionFrame enables you to perform complex image manipulations with natural language commands like "remove the person" or "change hair color to blue."

## 🚀 Features

- **Text-Based Image Editing** - Edit images using natural language instructions
- **Advanced Segmentation** - Hybrid segmentation system combines multiple AI models for precise masks
- **Specialized Pipelines** - Optimized processing for different types of edits:
  - Person Removal
  - Object Removal
  - Background Replacement
  - Color Changing (including specific hair color changes)
  - Object Addition
- **Context-Aware Processing** - Analyzes image lighting, scene type, and style for improved results
- **Robust Error Handling** - Multiple fallback mechanisms to ensure successful edits
- **User-Friendly Interface** - Clean, simple Gradio-based UI with helpful examples and tips

## 💻 Technical Capabilities

FusionFrame integrates multiple AI technologies for best-in-class results:

- **Stable Diffusion XL** - For high-quality image generation and inpainting
- **Segment Anything Model (SAM)** - For precise object segmentation
- **YOLO** - For object detection and classification
- **CLIPSeg** - For text-based segmentation
- **MediaPipe** - For face and person detection
- **LaMa** - For high-quality large mask inpainting
- **REMBG** - For background removal and extraction
- **Face Enhancement** - GPEN and CodeFormer for face restoration

## 📋 Requirements

- Python 3.8+
- CUDA-capable GPU (8GB+ VRAM recommended)
- Dependencies listed in `requirements.txt`

## 🛠️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/rubikdesign/FusionFrame.git
   cd fusionframe
   ```

2. Run the installation script:
   ```bash
   chmod +x install.sh
   ./install.sh
   ```

   This script will:
   - Create a virtual environment
   - Install all required dependencies
   - Set up an execution script

3. Launch the application:
   ```bash
   ./run_icedit.sh
   ```

4. Access the web interface at `http://localhost:7860`

## 🖌️ Usage Examples

Here are some examples of what you can do with FusionFrame:

### Object Removal
```
"remove the car from the street"
"erase all text from the image"
"remove person from the image"
```

### Color Changes
```
"change hair color to bright pink"
"make the shirt blue"
"change the color of the building to white"
```

### Background Replacement
```
"replace background with beach"
"change background to futuristic city"
"replace background with forest"
```

### Object Addition
```
"add glasses"
"add a hat"
"add a necklace"
```

## 🧠 How It Works

1. **Operation Analysis** - The system analyzes the text prompt to determine the type of operation
2. **Hybrid Mask Generation** - Multiple segmentation models work together to create precise masks
3. **Mask Refinement** - Edge-aware processing enhances the mask quality
4. **Initial Inpainting** - LaMa inpainting provides a solid foundation for edits
5. **Context Analysis** - Scene, lighting, and style are analyzed for better prompt engineering
6. **Final Generation** - SDXL with ControlNet performs the final render with optimized parameters

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📞 Contact

For questions, support, or feedback, please open an issue on GitHub or contact us at [your-email@example.com].

---

*FusionFrame - Transform your images with the power of words.*
