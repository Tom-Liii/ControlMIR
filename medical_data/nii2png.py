nii_dirs = [
    '/home/heming/research/medical/ControlNetPlus/medical_data/HQ',
    '/home/heming/research/medical/ControlNetPlus/medical_data/LQ'
]
import os
import nibabel as nib
import numpy as np
from PIL import Image

def convert_nii_to_png(nii_path, output_dir):
    # Load nii file
    nii_img = nib.load(nii_path)
    nii_data = nii_img.get_fdata()
    
    # Get basic info
    print(f"\nProcessing {nii_path}")
    print(f"Shape: {nii_data.shape}")
    print(f"Data type: {nii_data.dtype}")
    print(f"Value range: [{np.min(nii_data)}, {np.max(nii_data)}]")
    
    # Convert to uint8 for PNG
    nii_data = ((nii_data - np.min(nii_data)) / (np.max(nii_data) - np.min(nii_data)) * 255).astype(np.uint8)
    
    # Save middle slice as PNG
    middle_slice = nii_data[:, :, nii_data.shape[2]//2]
    img = Image.fromarray(middle_slice)
    
    # Create output filename
    base_name = os.path.splitext(os.path.basename(nii_path))[0]
    if base_name.endswith('.nii'):
        base_name = base_name[:-4]
    out_path = os.path.join(output_dir, f"{base_name}.png")
    
    img.save(out_path)
    print(f"Saved to {out_path}")

# Process each directory
for nii_dir in nii_dirs:
    print(f"\nProcessing directory: {nii_dir}")
    
    # Create output directory if needed
    if not os.path.exists(nii_dir):
        os.makedirs(nii_dir)
        
    # Find all .nii.gz files
    for file in os.listdir(nii_dir):
        if file.endswith('.nii.gz'):
            nii_path = os.path.join(nii_dir, file)
            convert_nii_to_png(nii_path, nii_dir)
