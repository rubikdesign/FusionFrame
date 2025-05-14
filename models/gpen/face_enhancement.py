import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import os

class FaceEnhancement:
    def __init__(self, model_path, size=512, channel_multiplier=2, device='cuda'):
        self.size = size
        self.device = device
        self.channel_multiplier = channel_multiplier
        
        # Import to avoid circular dependency
        from styleGAN2.model import Generator
        
        self.generator = Generator(self.size, 512, 8, channel_multiplier=self.channel_multiplier).to(device)
        checkpoint = torch.load(model_path, map_location=device)
        self.generator.load_state_dict(checkpoint['g_ema'])
        self.generator.eval()
        
        from torchvision import transforms
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    
    def process(self, img, face, landmarks):
        # Convert to RGB if needed
        if img.ndim == 3 and img.shape[2] == 3:
            if img.dtype == np.uint8:
                img_rgb = img
            else:
                img_rgb = (img * 255).astype(np.uint8)
        else:
            raise ValueError("Input image must be RGB")
        
        # Crop and align face
        x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
        w, h = x2 - x1, y2 - y1
        
        # Add margin
        margin = int(max(w, h) * 0.3)
        x1 = max(0, x1 - margin)
        y1 = max(0, y1 - margin)
        x2 = min(img_rgb.shape[1], x2 + margin)
        y2 = min(img_rgb.shape[0], y2 + margin)
        
        # Crop face
        face_img = img_rgb[y1:y2, x1:x2]
        
        # Resize to model input size
        face_img = cv2.resize(face_img, (self.size, self.size))
        
        # Transform for model
        face_tensor = self.transform(face_img).unsqueeze(0).to(self.device)
        
        # Generate enhanced face
        with torch.no_grad():
            output = self.generator(face_tensor)[0]
            
        # Convert output tensor to image
        output = output.cpu().numpy().transpose(1, 2, 0)
        output = ((output + 1) / 2 * 255).astype(np.uint8)
        
        return output