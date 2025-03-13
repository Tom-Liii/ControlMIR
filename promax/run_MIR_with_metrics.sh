# export MODEL_DIR="/home/heming/research/medical/Segmind-Vega"
export OUTPUT_DIR="output/$(date +%Y-%m-%d_%H-%M-%S)"
export DATA_DIR="/hpc2hdd/home/sfei285/datasets/heming/All-in-One"
export HF_ENDPOINT=https://hf-mirror.com
export VAE_DIR="madebyollin/sdxl-vae-fp16-fix"

# CUDA_LAUNCH_BLOCKING=1 accelerate launch --config_file /home/heming/research/medical/ControlNetPlus/accelarate_config/med.yaml /home/heming/research/medical/ControlNetPlus/promax/controlnet_union_train_medical_image_restoration.py \
CUDA_LAUNCH_BLOCKING=1 accelerate launch --config_file /home/heming/research/medical/ControlNetPlus/accelarate_config/med2.yaml /home/heming/research/medical/ControlNetPlus/promax/controlnet_union_train_with_metrics.py \
 --pretrained_model_name_or_path "/home/heming/research/medical/ControlNetPlus/SSD-1B/models--segmind--SSD-1B/snapshots/60987f37e94cd59c36b1cba832b9f97b57395a10" \
 --output_dir=$OUTPUT_DIR \
 --train_data_dir=$DATA_DIR \
 --mixed_precision="fp16" \
 --resolution=16 \
 --learning_rate=1e-5 \
 --max_train_steps=15000 \
 --validation_image "/hpc2hdd/home/sfei285/Project/heming/ControlMIR/medical_data/LQ/IXI002-Guys-0828-T2_0.png" "/hpc2hdd/home/sfei285/Project/heming/ControlMIR/medical_data/LQ/IXI002-Guys-0828-T2_1.png" \
 --validation_image "/hpc2hdd/home/sfei285/Project/heming/ControlMIR/medical_data/HQ/IXI002-Guys-0828-T2_0.png" "/hpc2hdd/home/sfei285/Project/heming/ControlMIR/medical_data/HQ/IXI002-Guys-0828-T2_1.png" \
 --validation_prompt "MRI" "MRI" \
 --validation_steps=5 \
 --train_batch_size=2 \
 --gradient_accumulation_steps=8 \
 --gradient_checkpointing \
 --use_8bit_adam \
 --report_to="tensorboard" \
 --seed=42 \
 --dataloader_num_workers=4 \
 --pretrained_vae_model_name_or_path=$VAE_DIR \
#  --controlnet_model_name_or_path "/hpc2hdd/home/sfei285/Project/heming/ControlMIR/controlnet-union-sdxl-1.0-promax" \