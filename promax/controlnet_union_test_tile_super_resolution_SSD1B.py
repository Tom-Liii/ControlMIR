# diffusers测试ControlNet
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import sys
sys.path.append('..')
import cv2
import time
import torch
import random
import numpy as np
from PIL import Image 
from diffusers.utils import load_image
from diffusers import EulerAncestralDiscreteScheduler, AutoencoderKL
from models.controlnet_union import ControlNetModel_Union
from pipeline.pipeline_controlnet_union_sd_xl_img2img import StableDiffusionXLControlNetUnionImg2ImgPipeline


device=torch.device('cuda:3')

eulera_scheduler = EulerAncestralDiscreteScheduler.from_pretrained(
    "segmind/SSD-1B", 
    subfolder="scheduler",
    resume_download=True,
    max_retries=5)
vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
# Note you should set the model and the config to the promax version manually, default is not the promax version. 
# from huggingface_hub import snapshot_download
# snapshot_download(repo_id="xinsir/controlnet-union-sdxl-1.0", local_dir='controlnet-union-sdxl-1.0')
# you should make a new dir controlnet-union-sdxl-1.0-promax and mv the promax config and promax model into it and rename the promax config and the promax model.
controlnet_model = ControlNetModel_Union.from_pretrained("/home/heming/research/medical/ControlNetPlus/controlnet-union-sdxl-1.0-promax", torch_dtype=torch.float16, use_safetensors=True)


pipe = StableDiffusionXLControlNetUnionImg2ImgPipeline.from_pretrained(
    "segmind/SSD-1B", controlnet=controlnet_model, 
    vae=vae,
    torch_dtype=torch.float16,
    # scheduler=ddim_scheduler,
    scheduler=eulera_scheduler,
)

pipe = pipe.to(device)


prompt = "a MRI image, brain, high resolution, anatomy, high quality"
negative_prompt = 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality'

seed = random.randint(0, 2147483647)
generator = torch.Generator('cuda').manual_seed(seed)


input_dir = "/home/heming/research/medical/ControlNetPlus/medical_data/LQ"
output_dir = "/home/heming/research/medical/ControlNetPlus/medical_data/SR_SSD1B"

# Create output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Get list of PNG files in input directory
png_files = [f for f in os.listdir(input_dir) if f.endswith('.png')]

for png_file in png_files:
    input_path = os.path.join(input_dir, png_file)
    controlnet_img = cv2.imread(input_path)
    
    height, width, _  = controlnet_img.shape
    ratio = np.sqrt(1024. * 1024. / (width * height))
    # 3 * 3 upscale correspond to 16 * 3 multiply, 2 * 2 correspond to 16 * 2 multiply and so on.
    W, H = int(width * ratio) // 48 * 48, int(height * ratio) // 48 * 48
    controlnet_img = cv2.resize(controlnet_img, (W, H))

    controlnet_img = cv2.cvtColor(controlnet_img, cv2.COLOR_BGR2RGB)
    controlnet_img = Image.fromarray(controlnet_img)

    # 计算每个块的目标大小
    target_width = W // 3
    target_height = H // 3

    # 创建一个列表用于存储子图
    images = []

    # 分割图像
    crops_coords_list = [(0, 0), (0, width // 2), (height // 2, 0), (width // 2, height // 2), 0, 0, 0, 0, 0]
    for i in range(3):  # 三行
        for j in range(3):  # 三列
            left = j * target_width
            top = i * target_height
            right = left + target_width
            bottom = top + target_height

            # 根据计算的边界裁剪图像
            cropped_image = controlnet_img.crop((left, top, right, bottom))
            cropped_image = cropped_image.resize((W, H))
            images.append(cropped_image)


    # 0 -- openpose
    # 1 -- depth
    # 2 -- hed/pidi/scribble/ted
    # 3 -- canny/lineart/anime_lineart/mlsd
    # 4 -- normal
    # 5 -- segment
    # 6 -- tile
    # 7 -- repaint
    result_images = []
    for sub_img, crops_coords in zip(images, crops_coords_list):
        new_width, new_height = W, H
        out = pipe(prompt=[prompt]*1,
                    image=sub_img, 
                    control_image_list=[0, 0, 0, 0, 0, 0, sub_img, 0],
                    negative_prompt=[negative_prompt]*1,
                    generator=generator,
                    width=new_width, 
                    height=new_height,
                    num_inference_steps=30,
                    crops_coords_top_left=(W, H),
                    target_size=(W, H),
                    original_size=(W * 2, H * 2),
                    union_control=True,
                    union_control_type=torch.Tensor([0, 0, 0, 0, 0, 0, 1, 0]),
                )
        result_images.append(out.images[0])

    new_im = Image.new('RGB', (new_width*3, new_height*3))
    # 拼接图片到新的图像上
    new_im.paste(result_images[0], (0, 0))  
    new_im.paste(result_images[1], (new_width, 0))
    new_im.paste(result_images[2], (new_width * 2, 0))
    new_im.paste(result_images[3], (0, new_height))
    new_im.paste(result_images[4], (new_width, new_height))  
    new_im.paste(result_images[5], (new_width * 2, new_height))
    new_im.paste(result_images[6], (0, new_height * 2))
    new_im.paste(result_images[7], (new_width, new_height * 2))
    new_im.paste(result_images[8], (new_width * 2, new_height * 2))  
    # Print input image information
    input_width, input_height = controlnet_img.size
    print(f"\nInput image size: {input_width} x {input_height}")
    print(f"Input image resolution: {input_width * input_height / 1000000:.2f}M pixels")

    # Print output image information 
    output_width, output_height = new_im.size
    print(f"\nOutput image size: {output_width} x {output_height}")
    print(f"Output image resolution: {output_width * output_height / 1000000:.2f}M pixels")
    print(f"Resolution increased by {(output_width * output_height)/(input_width * input_height):.1f}x")


    output_path = os.path.join(output_dir, png_file)
    new_im.save(output_path)