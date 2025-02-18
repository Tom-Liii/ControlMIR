LQ_path = '/home/heming/research/medical/ControlNetPlus/medical_data/LQ'
HQ_path = '/home/heming/research/medical/ControlNetPlus/medical_data/HQ'
SR_path = '/home/heming/research/medical/ControlNetPlus/medical_data/SR'
import os
from PIL import Image

def get_image_info(img_path):
    img = Image.open(img_path)
    width, height = img.size
    pixels = width * height
    return {
        'size': f"{width}x{height}",
        'pixels': pixels,
        'megapixels': pixels/1_000_000
    }

def analyze_directory(dir_path):
    print(f"\nAnalyzing {dir_path}:")
    print("-" * 50)
    
    png_files = [f for f in os.listdir(dir_path) if f.endswith('.png')]
    
    if not png_files:
        print("No PNG files found")
        return
        
    print(f"Found {len(png_files)} PNG files")
    
    total_pixels = 0
    for png_file in png_files:
        file_path = os.path.join(dir_path, png_file)
        info = get_image_info(file_path)
        print(f"\n{png_file}:")
        print(f"Resolution: {info['size']}")
        print(f"Pixels: {info['pixels']:,}")
        print(f"Megapixels: {info['megapixels']:.2f}MP")
        total_pixels += info['pixels']
        
    avg_pixels = total_pixels / len(png_files)
    print(f"\nAverage pixels per image: {avg_pixels:,.0f}")
    print(f"Average megapixels: {avg_pixels/1_000_000:.2f}MP")

# Analyze each directory
for path in [LQ_path, HQ_path, SR_path]:
    if os.path.exists(path):
        analyze_directory(path)
    else:
        print(f"\nDirectory not found: {path}")
