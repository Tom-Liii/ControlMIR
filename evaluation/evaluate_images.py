import torch
from PIL import Image
import torchvision.transforms as transforms
from evaluation_metric import compute_measure
import numpy as np
import os

def load_and_preprocess_image(image_path, target_size=None):
    # Load image
    img = Image.open(image_path).convert('L')  # Convert to grayscale
    
    # Resize if target size is provided
    if target_size:
        img = img.resize(target_size, Image.Resampling.LANCZOS)
    
    # Convert to tensor
    transform = transforms.ToTensor()
    img_tensor = transform(img)
    return img_tensor

def main():
    # Base directories
    hq_dir = '/home/heming/research/medical/ControlNetPlus/medical_data/HQ'
    sr_dir = '/home/heming/research/medical/ControlNetPlus/medical_data/SR_SSD1B'
    
    # Get all PNG files in HQ directory
    hq_files = sorted([f for f in os.listdir(hq_dir) if f.endswith('.png')])
    
    # Initialize lists to store metrics
    all_psnr = []
    all_ssim = []
    all_rmse = []
    
    # Process each pair of images
    for filename in hq_files:
        hq_path = os.path.join(hq_dir, filename)
        sr_path = os.path.join(sr_dir, filename)
        
        # Skip if SR file doesn't exist
        if not os.path.exists(sr_path):
            print(f"Warning: No matching SR file for {filename}")
            continue
        
        # Get image dimensions
        sr_img_size = Image.open(sr_path).size
        hq_img_size = Image.open(hq_path).size
        
        # Load and process images
        # hq_img = load_and_preprocess_image(hq_path, target_size=sr_img_size)
        # sr_img = load_and_preprocess_image(sr_path)
        hq_img = load_and_preprocess_image(hq_path)
        sr_img = load_and_preprocess_image(sr_path, target_size=hq_img_size)
        
        # Calculate metrics
        psnr, ssim, rmse = compute_measure(hq_img, sr_img, data_range=1.0)
        
        # Store metrics
        all_psnr.append(psnr)
        all_ssim.append(ssim)
        all_rmse.append(rmse)
        
        # Print individual results
        print(f"\nResults for {filename}:")
        print(f"PSNR: {psnr:.4f} dB")
        print(f"SSIM: {ssim:.4f}")
        print(f"RMSE: {rmse:.4f}")
    
    # Print average metrics
    if all_psnr:
        print("\nAverage metrics across all images:")
        print(f"Average PSNR: {np.mean(all_psnr):.4f} ± {np.std(all_psnr):.4f} dB")
        print(f"Average SSIM: {np.mean(all_ssim):.4f} ± {np.std(all_ssim):.4f}")
        print(f"Average RMSE: {np.mean(all_rmse):.4f} ± {np.std(all_rmse):.4f}")

if __name__ == "__main__":
    main()
