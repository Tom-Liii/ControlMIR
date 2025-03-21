# export MODEL_DIR="/home/heming/research/medical/Segmind-Vega"
export OUTPUT_DIR="output/$(date +%Y-%m-%d_%H-%M-%S)"
export DATA_DIR="/hpc2hdd/home/sfei285/datasets/heming/All-in-One"
export HF_ENDPOINT=https://hf-mirror.com
export VAE_DIR="madebyollin/sdxl-vae-fp16-fix"

# CUDA_LAUNCH_BLOCKING=1 accelerate launch --config_file /hpc2hdd/home/sfei285/Project/heming/ControlMIR/accelarate_config/med.yaml /hpc2hdd/home/sfei285/Project/heming/ControlMIR/promax/controlnet_union_train_with_metrics.py \
#  --pretrained_model_name_or_path "/hpc2hdd/home/sfei285/Project/heming/controlnet_training/SSD1B/snapshots/60987f37e94cd59c36b1cba832b9f97b57395a10" \
#  --output_dir=$OUTPUT_DIR \
#  --train_data_dir=$DATA_DIR \
#  --mixed_precision="fp16" \
#  --resolution=1024 \
#  --learning_rate=1e-5 \
#  --max_train_steps=5000 \
#  --validation_image "/hpc2hdd/home/sfei285/Project/heming/ControlMIR/medical_data/LQ/IXI002-Guys-0828-T2_0.png" "/hpc2hdd/home/sfei285/Project/heming/ControlMIR/medical_data/LQ/IXI002-Guys-0828-T2_1.png" \
#  --validation_image "/hpc2hdd/home/sfei285/Project/heming/ControlMIR/medical_data/HQ/IXI002-Guys-0828-T2_0.png" "/hpc2hdd/home/sfei285/Project/heming/ControlMIR/medical_data/HQ/IXI002-Guys-0828-T2_1.png" \
#  --validation_prompt "MRI" "MRI" \
#  --validation_steps=100 \
#  --train_batch_size=2 \
#  --gradient_accumulation_steps=8 \
#  --gradient_checkpointing \
#  --use_8bit_adam \
#  --report_to="tensorboard" \
#  --seed=42 \
#  --dataloader_num_workers=4 \
#  --pretrained_vae_model_name_or_path=$VAE_DIR \

# CUDA_LAUNCH_BLOCKING=1 accelerate launch --config_file /hpc2hdd/home/sfei285/Project/heming/ControlMIR/accelarate_config/med.yaml /hpc2hdd/home/sfei285/Project/heming/ControlMIR/promax/controlnet_union_train_medical_image_restoration.py \
CUDA_LAUNCH_BLOCKING=1 accelerate launch --config_file /hpc2hdd/home/sfei285/Project/heming/ControlMIR/accelarate_config/med.yaml /hpc2hdd/home/sfei285/Project/heming/ControlMIR/promax/controlnet_union_train_with_metrics.py \
 --pretrained_model_name_or_path "/hpc2hdd/home/sfei285/Project/heming/controlnet_training/SSD1B/snapshots/60987f37e94cd59c36b1cba832b9f97b57395a10" \
 --output_dir=$OUTPUT_DIR \
 --train_data_dir=$DATA_DIR \
 --mixed_precision="fp16" \
 --resolution=1024 \
 --learning_rate=1e-4 \
 --max_train_steps=10000000 \
 --validation_image "/hpc2hdd/home/sfei285/Project/heming/ControlMIR/medical_data/LQ/IXI002-Guys-0828-T2_0.png" "/hpc2hdd/home/sfei285/Project/heming/ControlMIR/medical_data/LQ/IXI002-Guys-0828-T2_1.png" \
 --validation_image "/hpc2hdd/home/sfei285/Project/heming/ControlMIR/medical_data/HQ/IXI002-Guys-0828-T2_0.png" "/hpc2hdd/home/sfei285/Project/heming/ControlMIR/medical_data/HQ/IXI002-Guys-0828-T2_1.png" \
 --validation_prompt "MRI" "MRI" \
 --validation_steps=100 \
 --train_batch_size=2 \
 --gradient_accumulation_steps=4 \
 --gradient_checkpointing \
 --use_8bit_adam \
 --report_to="tensorboard" \
 --seed=42 \
 --num_train_epochs=10 \
 --dataloader_num_workers=4 \
 --pretrained_vae_model_name_or_path=$VAE_DIR \
