import pyiqa
# from torchmetrics.multimodal import CLIPImageQualityAssessment
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import os
import random
from tqdm import tqdm
from torchvision import transforms
import logging
from torchvision.utils import save_image

def prepare_control_image(
    image,
    width,
    height,
    batch_size,
    num_images_per_prompt,
    device,
    dtype,
    do_classifier_free_guidance=False,
    guess_mode=False,
    vae_image_processor=None,
):
    # image = vae_image_processor.preprocess(image, height=height, width=width).to(dtype=torch.float32)


    image_batch_size = image.shape[0]

    if image_batch_size == 1:
        repeat_by = batch_size
    else:
        # image batch size is the same as prompt batch size
        repeat_by = num_images_per_prompt
    # import pdb; pdb.set_trace()
    image = image.repeat_interleave(repeat_by, dim=0)

    image = image.to(device=device, dtype=dtype)
    
    if do_classifier_free_guidance and not guess_mode:
        image = torch.cat([image] * 2)
    return image

def validate_pipeline(
    vae_enc, 
    vae_dec, 
    unet, 
    controlnet, 
    noise_scheduler,
    skip_conv_out,
    val_dataset, 
    writer, 
    step, 
    image_dir, 
    args,
    accelerator,
    weight_dtype=torch.float32
):
    """
    Validate the pipeline using the provided components and dataset
    """
    # Set all models to evaluation mode
    vae_enc.eval()
    vae_dec.eval()
    controlnet.eval()
    if skip_conv_out is not None:
        skip_conv_out.eval()
    
    # Initialize metrics
    val_nima, val_musiq, val_clip_iqa, val_liqe, val_brisque, val_niqe, val_maniqa = 0, 0, 0, 0, 0, 0, 0
    val_l1, val_mse = 0, 0
    all_image_names = []
    resize_transform = transforms.Resize((512, 512))
    device = accelerator.device
    
    # Number of samples to visualize
    # num_samples = min(args.val_num, len(val_dataset))
    # indices = random.sample(range(len(val_dataset)), num_samples)
    combined_images = []
    
    # Initialize quality metrics
    nima_metric = pyiqa.create_metric('nima').to(device)
    musiq_metric = pyiqa.create_metric('musiq').to(device)
    liqe_metric = pyiqa.create_metric('liqe').to(device)
    # metric = CLIPImageQualityAssessment().to(device)
    brisque_metric = pyiqa.create_metric('brisque').to(device)
    niqe_metric = pyiqa.create_metric('niqe').to(device)
    maniqa_metric = pyiqa.create_metric('maniqa').to(device)
    
    # Control type for super resolution
    union_control_type = torch.Tensor([0, 0, 0, 0, 0, 0, 1, 0]).to(device)
    
    logging.info("Starting validation...")
    
    with torch.no_grad():  # This ensures no gradients are computed
        for i, batch in tqdm(enumerate(val_dataset), total=len(val_dataset)):
            if i >= len(val_dataset):  # Hard stop if i overflows
                break
            control_image_list = [0, 0, 0, 0, 0, 0, batch["conditioning_pixel_values"], 0]
            # logging.info(f'step {i} of {len(val_dataset)}')
            for idx in range(len(control_image_list)):
                    # TODO: need to conduct some check before prepare control image                            
                    # check_image(image=batch_imgs[idx], prompt=batch["caption"], prompt_embeds=batch["prompt_embeds"])
                    if type(control_image_list[idx]) is int and control_image_list[idx] == 0:
                        continue
                    else:
                        # import pdb; pdb.set_trace()
                        control_image = prepare_control_image(
                                            image=control_image_list[idx],
                                            width=args.resolution,
                                            height=args.resolution,
                                            batch_size=args.train_batch_size * 1,
                                            num_images_per_prompt=1,
                                            device=accelerator.device,
                                            dtype=torch.float32,
                                            do_classifier_free_guidance=False,
                                            guess_mode=False,
                                        )
                        height, width = control_image.shape[-2:]
                        control_image_list[idx] = control_image
            pixel_values = batch["pixel_values"].to(accelerator.device, dtype=torch.float32)
            lq_values = batch["conditioning_pixel_values"].to(accelerator.device, dtype=torch.float32)
            latents = vae_enc(lq_values, direction="a2b")
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
            timesteps = timesteps.long()
            latent_model_input = latents # dont know why need to multiply by 2! [resolved]
            noisy_latents = noise_scheduler.add_noise(latent_model_input.float(), noise.float(), timesteps).to(
                dtype=torch.float32
            )
            controlnet_image = batch["conditioning_pixel_values"].to(device=accelerator.device, dtype=torch.float32)
            
            union_control_type=torch.Tensor([0, 0, 0, 0, 0, 0, 1, 0]) # control type for super resolution
            text_embeds = batch["unet_added_conditions"]["text_embeds"]
            time_ids = batch["unet_added_conditions"]["time_ids"]
            # control_type = union_control_type.reshape(1, -1).to(accelerator.device, dtype=batch["prompt_embeds"].dtype).repeat(args.train_batch_size * 1 * 2, 1)
            # control_type = union_control_type.reshape(1, -1).to(accelerator.device, dtype=torch.float32).repeat(args.train_batch_size, 1)
            control_type = union_control_type.reshape(1, -1).to(accelerator.device, dtype=torch.float32).repeat(args.train_batch_size, 1)
            # control_type = control_type.unsqueeze(0).repeat(args.train_batch_size, 1, 1)
            
            controlnet_added_cond_kwargs = {
                "text_embeds": text_embeds.squeeze(1).to(torch.float32),
                "time_ids": time_ids.squeeze(1).to(torch.float32),
                "control_type": control_type
            }
            # import pdb; pdb.set_trace()
            hidden_states = batch["prompt_embeds"].to(accelerator.device).repeat(1, 1, 1, 1).squeeze(1).float()
            # ! BUG: if cfg, repeat the controlnet_cond_list's image [resolved]
            # Check dtype of ControlNet's parameters    
            # import pdb; pdb.set_trace()  
            # ! Now testing done, can pass controlnet
            down_block_res_samples, mid_block_res_sample = controlnet(
                noisy_latents,
                timesteps.squeeze(0).float(),
                encoder_hidden_states=hidden_states.to(accelerator.device),
                added_cond_kwargs=controlnet_added_cond_kwargs,
                controlnet_cond_list=control_image_list, # ! assuming batch_size=1 
                return_dict=False,
            )
            added_cond_kwargs = {
                "text_embeds": text_embeds.squeeze(1).to(accelerator.device, dtype=weight_dtype),
                "time_ids": time_ids.squeeze(1).to(accelerator.device, dtype=weight_dtype),
                "control_type": control_type
            }
            model_pred = unet(
                noisy_latents.to(accelerator.device, dtype=torch.float16),
                timesteps[0].to(accelerator.device, dtype=torch.float16),
                encoder_hidden_states=hidden_states.to(accelerator.device, dtype=torch.float16),
                added_cond_kwargs=added_cond_kwargs,
                down_block_additional_residuals=[
                    res.to(accelerator.device, dtype=torch.float16)
                    for res in down_block_res_samples
                ],
                mid_block_additional_residual=mid_block_res_sample.to(
                    accelerator.device, dtype=torch.float16
                ),
                return_dict=False,
            )[0].to(dtype=torch.float32)
            x_denoised = noise_scheduler.step(model_pred.to(accelerator.device, dtype=torch.float32), timesteps[0].to(accelerator.device), noisy_latents.to(accelerator.device, dtype=torch.float32), return_dict=True).prev_sample
            output_image = vae_dec(x_denoised.to(accelerator.device, dtype=torch.float32), direction="a2b") + skip_conv_out(batch["conditioning_pixel_values"].to(accelerator.device, dtype=torch.float32))
            # # normalize output image to [0, 1], using min-max normalization
            # output_image = (output_image - output_image.min()) / (output_image.max() - output_image.min())
            # # clamp output image to [0, 1]
            # output_image = output_image.clamp(0, 1)
            # Calculate losses and metrics
            # TODO: handle normalization
            # output_image = TF.to_tensor(output_image)
            
            output_image = TF.normalize(output_image, mean=[0.5], std=[0.5])
            output_image = output_image.clamp(-1, 1)
            output_image_0_1 = (output_image*0.5) + 0.5
            output_image = output_image_0_1
            
            
            loss_l1 = F.l1_loss(output_image.float(), pixel_values.float(), reduction="mean")
            loss_mse = F.mse_loss(output_image.float(), pixel_values.float(), reduction="mean")
            val_l1 += loss_l1.item()
            val_mse += loss_mse.item()
            
            # Calculate quality metrics
            val_nima += nima_metric(output_image).mean().item()
            val_musiq += musiq_metric(output_image).mean().item()
            # val_clip_iqa += metric(output_image).mean().item()
            val_liqe += liqe_metric(output_image).mean().item()
            val_brisque += brisque_metric(output_image).mean().item()
            val_niqe += niqe_metric(output_image).mean().item()
            val_maniqa += maniqa_metric(output_image).mean().item()
            
            # Save sample images
            if i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
                image_name = batch["name"][0] if isinstance(batch["name"], list) else batch["name"]
                all_image_names.append(image_name)
                
                # Prepare images for visualization
                if output_image.shape[-1] > 1024 or output_image.shape[-2] > 1024:
                    resized_output = resize_transform(output_image)
                    resized_gt = resize_transform(pixel_values)
                    resized_src = resize_transform(lq_values)
                else:
                    resized_src = lq_values
                    resized_output = output_image
                    resized_gt = pixel_values
                
                # Combine and save
                combined_image = torch.cat([resized_src, resized_output, resized_gt], dim=3)
                combined_images.append(combined_image)
                    
            
    
    # Calculate average metrics
    num_batches = len(val_dataset)
    val_l1 /= num_batches
    val_mse /= num_batches
    val_nima /= num_batches
    val_musiq /= num_batches
    # val_clip_iqa /= num_batches
    val_liqe /= num_batches
    val_brisque /= num_batches
    val_niqe /= num_batches
    val_maniqa /= num_batches
    
    # Log metrics
    logging.info(
        f"Validation results - L1: {val_l1:.4f}, MSE: {val_mse:.4f}, "
        f"NIMA: {val_nima:.2f}, MUSIQ: {val_musiq:.2f}, "
        # f"CLIP-IQA: {val_clip_iqa:.2f}, LIQE: {val_liqe:.2f}, "
        f"BRISQUE: {val_brisque:.2f}, NIQE: {val_niqe:.2f}, "
        f"MANIQA: {val_maniqa:.2f}"
    )
    
    # Write to tensorboard
    if accelerator.is_main_process:
        writer.add_scalar('Validation/L1', val_l1, step)
        writer.add_scalar('Validation/MSE', val_mse, step)
        writer.add_scalar('Validation/NIMA', val_nima, step)
        writer.add_scalar('Validation/MUSIQ', val_musiq, step)
        # writer.add_scalar('Validation/CLIP-IQA', val_clip_iqa, step)
        writer.add_scalar('Validation/LIQE', val_liqe, step)
        writer.add_scalar('Validation/BRISQUE', val_brisque, step)
        writer.add_scalar('Validation/NIQE', val_niqe, step)
        writer.add_scalar('Validation/MANIQA', val_maniqa, step)
        
        # Save sample images
        for i, combined_image in enumerate(combined_images):
            formatted_metrics = (
                f"NIMA_{val_nima:.2f}↑",
                f"MUSIQ_{val_musiq:.2f}↑",
                # f"CLIP-IQA_{val_clip_iqa:.2f}↑",
                f"LIQE_{val_liqe:.2f}↑",
                f"BRISQUE_{val_brisque:.2f}↓",
                f"NIQE_{val_niqe:.2f}↓",
                f"MANIQA_{val_maniqa:.2f}↑"
            )
            metrics_str = "_".join(formatted_metrics)
            save_path = os.path.join(image_dir, f"step_{step}", f"val_pred_{step}_{all_image_names[i]}_{metrics_str}.png")
            os.makedirs(os.path.join(image_dir, f"step_{step}"), exist_ok=True)
            
            # Normalize and save
            combined_image = (combined_image + 1) / 2  # Scale from [-1,1] to [0,1]

            save_image(combined_image, save_path)
            logging.info(f"Saved validation image to {save_path}")
    vae_enc.train()
    vae_dec.train()
    controlnet.train()
    if skip_conv_out is not None:
        skip_conv_out.train()
    return {
        "l1": val_l1,
        "mse": val_mse,
        "nima": val_nima,
        "musiq": val_musiq,
        # "clip_iqa": val_clip_iqa,
        "liqe": val_liqe,
        "brisque": val_brisque,
        "niqe": val_niqe,
        "maniqa": val_maniqa
    }
