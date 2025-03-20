# export MODEL_DIR="/home/heming/research/medical/Segmind-Vega"
export OUTPUT_DIR="weather_output/$(date +%Y-%m-%d_%H-%M-%S)"
export DATA_DIR="/hpc2hdd/home/sfei285/datasets/real_rain/RealRain-1k/RealRain-1k/RealRain-1k-H"
export HF_ENDPOINT=https://hf-mirror.com
export VAE_DIR="madebyollin/sdxl-vae-fp16-fix"

# CUDA_LAUNCH_BLOCKING=1 accelerate launch --config_file /home/heming/research/medical/ControlNetPlus/accelarate_config/med.yaml /home/heming/research/medical/ControlNetPlus/promax/controlnet_union_train_medical_image_restoration.py \
CUDA_LAUNCH_BLOCKING=1 accelerate launch --config_file /hpc2hdd/home/sfei285/Project/heming/ControlMIR/accelarate_config/med.yaml /hpc2hdd/home/sfei285/Project/heming/ControlMIR/promax/controlnet_union_train_weather_val.py \
 --pretrained_model_name_or_path "/hpc2hdd/home/sfei285/Project/heming/controlnet_training/SSD1B/snapshots/60987f37e94cd59c36b1cba832b9f97b57395a10" \
 --output_dir=$OUTPUT_DIR \
 --train_data_dir=$DATA_DIR \
 --mixed_precision="fp16" \
 --resolution=512 \
 --learning_rate=1e-4 \
 --max_train_steps=15000 \
 --validation_image "/hpc2hdd/home/sfei285/datasets/real_rain/RealRain-1k/RealRain-1k/RealRain-1k-H/validation/input/3.png" \
 --validation_prompt "Remove Rain" \
 --validation_steps=500 \
 --train_batch_size=2 \
 --gradient_accumulation_steps=8 \
 --gradient_checkpointing \
 --use_8bit_adam \
 --report_to="tensorboard" \
 --seed=42 \
 --dataloader_num_workers=4 \
 --pretrained_vae_model_name_or_path=$VAE_DIR \
 --checkpointing_steps=1000 \
#  --controlnet_model_name_or_path "/hpc2hdd/home/sfei285/Project/heming/ControlMIR/controlnet-union-sdxl-1.0-promax" \