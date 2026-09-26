import os
import argparse
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T
import cv2
import matplotlib.pyplot as plt

from models import UNetGenerator

def restore_and_colorize(image_path, generator_path, output_path=None, image_size=256, device=None):
    """
    Takes an input degraded image (grayscale, old photo, or RGB),
    preprocesses it, runs it through the trained UNet Generator,
    and returns the restored & colorized RGB image.
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    # Load trained generator
    netG = UNetGenerator(in_channels=1, out_channels=3)
    checkpoint = torch.load(generator_path, map_location=device)
    if 'netG' in checkpoint:
        netG.load_state_dict(checkpoint['netG'])
    else:
        netG.load_state_dict(checkpoint)
    netG.to(device)
    netG.eval()
    
    # Load and preprocess input image
    raw_img = Image.open(image_path)
    
    # If image is RGB, extract luminance / grayscale
    if raw_img.mode != 'L':
        rgb_np = np.array(raw_img.convert('RGB'))
        lab = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2LAB)
        L = lab[:, :, 0]
        input_pil = Image.fromarray(L, mode='L')
    else:
        input_pil = raw_img
        
    orig_w, orig_h = input_pil.size
    
    # Resize for generator
    transform = T.Compose([
        T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize((0.5,), (0.5,))
    ])
    
    input_tensor = transform(input_pil).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output_tensor = netG(input_tensor)
        
    # Denormalize output tensor to [0, 255] uint8 RGB
    out_np = (output_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5).clip(0, 1)
    out_img = Image.fromarray((out_np * 255).astype(np.uint8))
    
    # Optionally resize back to original dimensions
    out_img_resized = out_img.resize((orig_w, orig_h), Image.Resampling.LANCZOS)
    
    if output_path:
        out_img_resized.save(output_path)
        print(f"Restored and colorized image saved to: {output_path}")
        
    return input_pil, out_img_resized


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Restore and Colorize an Old Photo using trained GAN")
    parser.add_argument('--image_path', type=str, required=True, help="Path to input photo")
    parser.add_argument('--generator_path', type=str, default='checkpoints/best_generator.pth', help="Path to generator checkpoint")
    parser.add_argument('--output_path', type=str, default='restored_output.png', help="Path to save output image")
    parser.add_argument('--image_size', type=int, default=256, help="Model resolution (256)")
    
    args = parser.parse_args()
    restore_and_colorize(args.image_path, args.generator_path, args.output_path, args.image_size)
