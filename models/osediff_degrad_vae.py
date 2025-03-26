import os
import sys
sys.path.append(os.getcwd())
import yaml
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, CLIPTextModel
from diffusers import DDPMScheduler
from diffusers import models
from peft import LoraConfig, get_peft_model

from diffusers import AutoencoderKL
from diffusers import UNet2DConditionModel
# from network.MoCo import MoCo, Encoder, CrossAttention
# from network.ciconv2d import PriorConv2d, Degrader_projection
from utils.vaehook import VAEHook, perfcount
# from models.autoencoder_kl import AutoencoderKL
# from models.unet_2d_condition import UNet2DConditionModel
# from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
import time

class VAE_encode(nn.Module):
    def __init__(self, vae, vae_b2a=None):
        super(VAE_encode, self).__init__()
        self.vae = vae
        self.vae_b2a = vae_b2a

    def forward(self, x, direction):
        assert direction in ["a2b", "b2a"]
        if direction == "a2b":
            _vae = self.vae
        else:
            _vae = self.vae_b2a
        return _vae.encode(x).latent_dist.sample() * _vae.config.scaling_factor


class VAE_decode(nn.Module):
    def __init__(self, vae, vae_b2a=None):
        super(VAE_decode, self).__init__()
        self.vae = vae
        self.vae_b2a = vae_b2a

    def forward(self, x, direction):
        assert direction in ["a2b", "b2a"]
        if direction == "a2b":
            _vae = self.vae
        else:
            _vae = self.vae_b2a
        assert _vae.encoder.current_down_blocks is not None
        _vae.decoder.incoming_skip_acts = _vae.encoder.current_down_blocks
        x_decoded = (_vae.decode((x / _vae.config.scaling_factor)).sample).clamp(-1, 1)
        return x_decoded

def my_vae_encoder_fwd(self, sample):
    sample = self.conv_in(sample)
    l_blocks = []
    # down
    # print("num down_blocks", len(self.down_blocks))
    # um down_blocks 4
    # sample.shape torch.Size([8, 128, 256, 256])
    # sample.shape torch.Size([8, 128, 128, 128])
    # sample.shape torch.Size([8, 256, 64, 64])
    # sample.shape torch.Size([8, 512, 32, 32])
    for down_block in self.down_blocks:
        # print("sample.shape", sample.shape)
        l_blocks.append(sample)
        sample = down_block(sample)
    # middle
    sample = self.mid_block(sample)
    sample = self.conv_norm_out(sample)
    sample = self.conv_act(sample)
    sample = self.conv_out(sample)
    self.current_down_blocks = l_blocks
    return sample

def my_vae_decoder_fwd(self, sample, latent_embeds=None):
    #sample.shape torch.Size([1, 4, 32, 32])
    sample = self.conv_in(sample)
    # sample.shape torch.Size([1, 512, 32, 32])
    upscale_dtype = next(iter(self.up_blocks.parameters())).dtype
    # middle
    sample = self.mid_block(sample, latent_embeds)
    # sample.shape torch.Size([1, 512, 32, 32])
    sample = sample.to(upscale_dtype)
    if not self.ignore_skip:
        skip_convs = [self.skip_conv_1, self.skip_conv_2, self.skip_conv_3, self.skip_conv_4]
        # up
        # print("num up_blocks", len(self.up_blocks))
        
        # num up_blocks 4
        # skip_in.size torch.Size([4, 512, 32, 32])
        # sample.size torch.Size([4, 512, 32, 32])
        # skip_in.size torch.Size([4, 512, 64, 64])
        # sample.size torch.Size([4, 512, 64, 64])
        # skip_in.size torch.Size([4, 512, 128, 128])
        # sample.size torch.Size([4, 512, 128, 128])
        # skip_in.size torch.Size([4, 256, 256, 256])
        # sample.size torch.Size([4, 256, 256, 256])
        for idx, up_block in enumerate(self.up_blocks):
            skip_in = skip_convs[idx](self.incoming_skip_acts[::-1][idx] * self.gamma)
            # add skip
            _,_,H,W = skip_in.size()
            # print("skip_in.size",skip_in.size())
            # print("sample.size",sample.size())
            sample = torch.nn.functional.interpolate(
                        sample,
                        size=(H,W),
                        mode="bicubic",
                        align_corners=True,
                    )
            sample = sample + skip_in
            sample = up_block(sample, latent_embeds)
            # sample.shape torch.Size([1, 512, 64, 64])
            # sample.shape torch.Size([1, 512, 128, 128])
            # sample.shape torch.Size([1, 256, 256, 256])
            # sample.shape torch.Size([1, 128, 256, 256])
    else:
        for idx, up_block in enumerate(self.up_blocks):
            sample = up_block(sample, latent_embeds)
    # post-process
    if latent_embeds is None:
        sample = self.conv_norm_out(sample)
        # ample.shape torch.Size([1, 128, 256, 256])
    else:
        sample = self.conv_norm_out(sample, latent_embeds)
    sample = self.conv_act(sample)
    # sample.shape torch.Size([1, 128, 256, 256])
    sample = self.conv_out(sample) 
    # sample.shape torch.Size([1, 3, 256, 256])
    return sample

def initialize_vae(args):
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")

    sd = torch.load('/hpc2hdd/home/sfei285/Project/real-derain/weights/day2night.pkl')
    vae.encoder.forward = my_vae_encoder_fwd.__get__(vae.encoder, vae.encoder.__class__)
    vae.decoder.forward = my_vae_decoder_fwd.__get__(vae.decoder, vae.decoder.__class__)
    vae.decoder.skip_conv_1 = torch.nn.Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
    vae.decoder.skip_conv_2 = torch.nn.Conv2d(256, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
    vae.decoder.skip_conv_3 = torch.nn.Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
    vae.decoder.skip_conv_4 = torch.nn.Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
    vae.decoder.ignore_skip = False
    # vae_lora_config = LoraConfig(r=args.lora_rank, init_lora_weights="gaussian", target_modules=sd["vae_lora_target_modules"])
    vae_lora_config = LoraConfig(r=2, init_lora_weights="gaussian", target_modules=sd["vae_lora_target_modules"])
    vae = get_peft_model(vae, vae_lora_config)
    vae.add_adapter(adapter_name="vae_skip", peft_config=vae_lora_config)
    vae.decoder.gamma = 1
    vae_b2a = copy.deepcopy(vae)
    vae_enc = VAE_encode(vae, vae_b2a=vae_b2a)
    vae_dec = VAE_decode(vae, vae_b2a=vae_b2a)
    return vae_enc, vae_dec


def initialize_unet(args, return_lora_module_names=False, pretrained_model_name_or_path=None):
    unet = UNet2DConditionModel().from_pretrained(args.pretrained_model_name_or_path, subfolder="unet",  low_cpu_mem_usage=False, ignore_mismatched_sizes=True)
    unet.requires_grad_(False)
    unet.train()

    l_target_modules_encoder, l_target_modules_decoder, l_modules_others = [], [], []
    l_grep = ["to_k", "to_q", "to_v", "to_out.0", "conv", "conv1", "conv2", "conv_in", "conv_shortcut", "conv_out", "proj_out", "proj_in", "ff.net.2", "ff.net.0.proj"]
    for n, p in unet.named_parameters():
        if "attn" in n:
            print("attn", n)
            continue
        if "bias" in n or "norm" in n or "attn2" in n:
            continue
        for pattern in l_grep:
            if pattern in n and ("down_blocks" in n or "conv_in" in n):
                l_target_modules_encoder.append(n.replace(".weight",""))
                break
            elif pattern in n and ("up_blocks" in n or "conv_out" in n):
                l_target_modules_decoder.append(n.replace(".weight",""))
                break
            elif pattern in n:
                l_modules_others.append(n.replace(".weight",""))
                break
    # unet.forward = my_unet_fwd.__get__(unet, unet.__class__)
    # unet = unet.to("cuda")
    # input = torch.randn(1, 4, 256, 256).to("cuda")
    # timesteps = torch.tensor([999], device="cuda").long()
    # context = torch.randn(1, 77, 1024).to("cuda")
    # output = unet(input, timesteps, context)
    # print("output", output.shape)
    # exit()
    lora_conf_encoder = LoraConfig(r=args.lora_rank, init_lora_weights="gaussian",target_modules=l_target_modules_encoder)
    lora_conf_decoder = LoraConfig(r=args.lora_rank, init_lora_weights="gaussian",target_modules=l_target_modules_decoder)
    lora_conf_others = LoraConfig(r=args.lora_rank, init_lora_weights="gaussian",target_modules=l_modules_others)
    unet.add_adapter(lora_conf_encoder, adapter_name="default_encoder")
    unet.add_adapter(lora_conf_decoder, adapter_name="default_decoder")
    unet.add_adapter(lora_conf_others, adapter_name="default_others")
    unet.set_adapter(["default_encoder", "default_decoder", "default_others"])
    return unet, l_target_modules_encoder, l_target_modules_decoder, l_modules_others

def initialize_degrader(args):
    degrader = MoCo(base_encoder=Encoder).to('cuda')
    # degrader = PriorConv2d('W', k=3, scale=0.0).to('cuda')
    if args.degrader_path is not None:
        state_dict = torch.load(args.degrader_path)
        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = k.replace("module.", "")
            new_state_dict[new_key] = v
        degrader.load_state_dict(new_state_dict)
        print("degrader loaded")
    # 提取并冻结degrader中的所有参数
    for param in degrader.parameters():
        param.requires_grad = False
    return degrader

class OSEDiff_gen(torch.nn.Module):
    def __init__(self, args):
        super().__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder").cuda()
        self.noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
        self.noise_scheduler.set_timesteps(1, device="cuda")
        self.noise_scheduler.alphas_cumprod = self.noise_scheduler.alphas_cumprod.cuda()
        self.args = args

        self.degrader = initialize_degrader(self.args)
        self.vae_enc, self.vae_dec = initialize_vae(self.args)
        self.unet, self.lora_unet_modules_encoder, self.lora_unet_modules_decoder, self.lora_unet_others = initialize_unet(self.args)

        self.lora_rank_unet = self.args.lora_rank
        self.lora_rank_vae = self.args.lora_rank

        self.skip_conv_out = nn.Sequential(
            nn.Conv2d(3,3,kernel_size=3,stride=1,padding=3//2),
            nn.ReLU(),
            nn.Conv2d(3,3,kernel_size=3,stride=1,padding=3//2)
        ).to("cuda")
        
        self.mlp = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU()
        ).to("cuda")

        self.unet.to("cuda")
        self.vae_dec.to("cuda")
        self.vae_enc.to("cuda")
        self.timesteps = torch.tensor([999], device="cuda").long()
        self.text_encoder.requires_grad_(False)
        self.degrader_projection = Degrader_projection().to("cuda")
        # self.tokenizer.model_max_length = 128

    def set_train(self):
        self.unet.train()
        self.vae_dec.train()
        self.vae_enc.train()
        
        self.mlp.requires_grad_(True)
        self.degrader.requires_grad_(True)
        for name, param in self.named_parameters():
            if 'lora' in name or 'adapter' in name or 'skip_conv' in name or 'conv_out' in name:
                param.requires_grad = True
                print("Trainable:", name)
            else:
                param.requires_grad = False
                
            if 'unet.conv_in' in name:
                param.requires_grad = True
                print("Trainable:", name)  
                
        for name, param in self.unet.named_parameters():
            if 'attn2' in name:
                param.requires_grad = True
                print("Trainable:", name)
                
    def encode_prompt(self, prompt_batch):
        prompt_embeds_list = []
        with torch.no_grad():
            for caption in prompt_batch:
                text_input_ids = self.tokenizer(
                    caption, max_length=self.tokenizer.model_max_length,
                    padding="max_length", truncation=True, return_tensors="pt"
                ).input_ids
                prompt_embeds = self.text_encoder(
                    text_input_ids.to(self.text_encoder.device),
                )[0]
                prompt_embeds_list.append(prompt_embeds)
        prompt_embeds = torch.concat(prompt_embeds_list, dim=0)
        return prompt_embeds

    def forward(self, c_t, batch=None, direction="a2b", args=None):
        
        degraded_feature = self.degrader(c_t, c_t)
            
        if isinstance(degraded_feature, tuple):
            degraded_feature = degraded_feature[0]  # 提取元组中的第一个元素

        encoded_control = self.vae_enc(c_t, direction=direction)
        prompt_embeds = self.encode_prompt(batch["prompt"])
        degraded_feature = degraded_feature.to(self.mlp[0].weight.dtype)
        degraded_feature = self.mlp(degraded_feature).unsqueeze(1)
        prompt_degraded_embeds = torch.cat((prompt_embeds, degraded_feature), dim=1)
        model_pred = self.unet(encoded_control, self.timesteps, encoder_hidden_states=prompt_degraded_embeds.to(torch.float32),).sample
        x_denoised = self.noise_scheduler.step(model_pred, self.timesteps, encoded_control, return_dict=True).prev_sample
        output_image = self.vae_dec(x_denoised, direction=direction) + self.skip_conv_out(c_t)
    
        return output_image

    def save_model(self, outf):
        sd_pre = torch.load('./weights/day2night.pkl')
        print("sd_pre", sd_pre.keys())
        sd = {}
        sd["vae_lora_target_modules"] = sd_pre["vae_lora_target_modules"]
        sd["l_target_modules_encoder"], sd["l_target_modules_encoder"], sd["l_modules_others"] =\
            self.lora_unet_modules_encoder, self.lora_unet_modules_decoder, self.lora_unet_others
        sd["rank_unet"] = self.args.lora_rank
        sd["rank_vae"] = self.args.lora_rank
        # sd["state_dict_unet"] = {k: v for k, v in self.unet.state_dict().items() if "lora" in k or "conv_in" in k or "attn2" in k}
        # sd["state_dict_vae"] = {k: v for k, v in self.vae.state_dict().items() if "lora" in k}
        # 保存可训练参数
        sd["state_dict_trainable_params"] = {name: param for name, param in self.named_parameters() if param.requires_grad}
        sd["state_dict_degrader"] = {k: v for k, v in self.degrader.state_dict().items()}
        sd["state_dict_degrader_projection"] = {k: v for k, v in self.degrader_projection.state_dict().items()}
        sd["state_dict_mlp"] = {k: v for k, v in self.mlp.state_dict().items()}
        torch.save(sd, outf)
        
    def load_ckpt(self, model_path):        
        print("begin load ckpt")
        model = torch.load(model_path)
        for name, param in model["state_dict_trainable_params"].items():
            if name in self.state_dict():
                self.state_dict()[name].copy_(param)
                print("loaded", name)
        
        self.degrader.load_state_dict(model["state_dict_degrader"])
        self.degrader_projection.load_state_dict(model["state_dict_degrader_projection"])
        self.mlp.load_state_dict(model["state_dict_mlp"])
        print("model loaded")

class OSEDiff_test(torch.nn.Module):
    def __init__(self, args):
        super().__init__()

        self.args = args
        self.device =  torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.pretrained_model_name_or_path, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(self.args.pretrained_model_name_or_path, subfolder="text_encoder")
        self.noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
        self.noise_scheduler.set_timesteps(1, device="cuda")
        self.vae_enc, self.vae_dec = initialize_vae(self.args)
        self.unet = UNet2DConditionModel.from_pretrained(self.args.pretrained_model_name_or_path, subfolder="unet")

        
        self.weight_dtype = torch.float16 if args.mixed_precision == "fp16" else torch.float32
        print("weight_dtype", self.weight_dtype)
        self.skip_conv_out = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=3//2),
            nn.ReLU(),
            nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=3//2)
        ).to(self.device, dtype=self.weight_dtype)

        self.degrader = initialize_degrader(self.args).to('cuda', dtype=self.weight_dtype)
        self.degrader_projection = Degrader_projection().to('cuda', dtype=self.weight_dtype)
        self.mlp = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU()
        ).to("cuda")

        osediff = torch.load(args.osediff_path)
        self.load_ckpt(osediff)

        # merge lora
        if self.args.merge_and_unload_lora:
            print(f'===> MERGE LORA <===')
            # self.vae = self.vae.merge_and_unload()
            self.unet = self.unet.merge_and_unload()

        self.unet.to("cuda", dtype=self.weight_dtype)
        self.vae_enc.to("cuda", dtype=self.weight_dtype)
        self.vae_dec.to("cuda", dtype=self.weight_dtype)
        self.text_encoder.to("cuda", dtype=self.weight_dtype)
        self.timesteps = torch.tensor([999], device="cuda").long() 
        self.noise_scheduler.alphas_cumprod = self.noise_scheduler.alphas_cumprod.cuda()

    def load_ckpt(self, model):        
        print("begin load ckpt")
        # load unet lora
        lora_conf_encoder = LoraConfig(r=model["rank_unet"], init_lora_weights="gaussian", target_modules=model["l_target_modules_encoder"])
        lora_conf_decoder = LoraConfig(r=model["rank_unet"], init_lora_weights="gaussian", target_modules=model["l_target_modules_encoder"])
        lora_conf_others = LoraConfig(r=model["rank_unet"], init_lora_weights="gaussian", target_modules=model["l_modules_others"])
        self.unet.add_adapter(lora_conf_encoder, adapter_name="default_encoder")
        self.unet.add_adapter(lora_conf_decoder, adapter_name="default_decoder")
        self.unet.add_adapter(lora_conf_others, adapter_name="default_others")
        self.unet.set_adapter(["default_encoder", "default_decoder", "default_others"])
        
        #load vae lora
        vae = AutoencoderKL.from_pretrained(self.args.pretrained_model_name_or_path, subfolder="vae")

        vae.encoder.forward = my_vae_encoder_fwd.__get__(vae.encoder, vae.encoder.__class__)
        vae.decoder.forward = my_vae_decoder_fwd.__get__(vae.decoder, vae.decoder.__class__)
        vae.decoder.skip_conv_1 = torch.nn.Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        vae.decoder.skip_conv_2 = torch.nn.Conv2d(256, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        vae.decoder.skip_conv_3 = torch.nn.Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        vae.decoder.skip_conv_4 = torch.nn.Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        vae.decoder.ignore_skip = False
        vae_lora_conf_encoder = LoraConfig(r=model["rank_vae"], init_lora_weights="gaussian", target_modules=model["vae_lora_target_modules"])
        vae = get_peft_model(vae, vae_lora_conf_encoder)
        vae.add_adapter(adapter_name="vae_skip", peft_config=vae_lora_conf_encoder)
        vae.decoder.gamma = 1
        vae_b2a = copy.deepcopy(vae)
        self.vae_enc = VAE_encode(vae, vae_b2a=vae_b2a).to("cuda")
        self.vae_dec = VAE_decode(vae, vae_b2a=vae_b2a).to("cuda")
        
        for name, param in model["state_dict_trainable_params"].items():
            if name in self.state_dict():
                self.state_dict()[name].copy_(param)
                print("loaded", name)
        
        self.degrader.load_state_dict(model["state_dict_degrader"])
        self.degrader_projection.load_state_dict(model["state_dict_degrader_projection"])
        self.mlp.load_state_dict(model["state_dict_mlp"])
        print("model loaded")
    def encode_prompt(self, prompt_batch):
        prompt_embeds_list = []
        with torch.no_grad():
            for caption in prompt_batch:
                text_input_ids = self.tokenizer(
                    caption, max_length=self.tokenizer.model_max_length,
                    padding="max_length", truncation=True, return_tensors="pt"
                ).input_ids
                prompt_embeds = self.text_encoder(
                    text_input_ids.to(self.text_encoder.device),
                )[0]
                prompt_embeds_list.append(prompt_embeds)
        prompt_embeds = torch.concat(prompt_embeds_list, dim=0)
        return prompt_embeds

    @torch.no_grad()
    def forward(self, lq, prompt, direction="a2b"):
        lq = lq.to(self.weight_dtype)

        degraded_feature = self.degrader(lq, lq)
        if isinstance(degraded_feature, tuple):
            degraded_feature = degraded_feature[0]

        degraded_feature = degraded_feature.to(self.mlp[0].weight.dtype)

        prompt_embeds = self.encode_prompt([prompt])

        lq_latent = self.vae_enc(lq, direction=direction)

        degraded_feature = self.mlp(degraded_feature).unsqueeze(1)
        prompt_degraded_embeds = torch.cat((prompt_embeds, degraded_feature), dim=1)
        prompt_degraded_embeds = prompt_degraded_embeds.to(self.weight_dtype)
        model_pred = self.unet(lq_latent, self.timesteps, encoder_hidden_states=prompt_degraded_embeds).sample

        x_denoised = self.noise_scheduler.step(model_pred, self.timesteps, lq_latent, return_dict=True).prev_sample

        output_image = self.vae_dec(x_denoised.to(self.weight_dtype), direction=direction) + self.skip_conv_out(lq)

        return output_image

    def _init_tiled_vae(self,
            encoder_tile_size = 256,
            decoder_tile_size = 256,
            fast_decoder = False,
            fast_encoder = False,
            color_fix = False,
            vae_to_gpu = True):
        # save original forward (only once)
        if not hasattr(self.vae_enc, 'original_forward'):
            setattr(self.vae_enc, 'original_forward', self.vae_enc.forward)
        if not hasattr(self.vae_dec, 'original_forward'):
            setattr(self.vae_dec, 'original_forward', self.vae_dec.forward)

        encoder = self.vae_enc
        decoder = self.vae_dec

        self.vae_enc.forward = VAEHook(
            encoder, encoder_tile_size, is_decoder=False, fast_decoder=fast_decoder, fast_encoder=fast_encoder, color_fix=color_fix, to_gpu=vae_to_gpu)
        self.vae_dec.forward = VAEHook(
            decoder, decoder_tile_size, is_decoder=True, fast_decoder=fast_decoder, fast_encoder=fast_encoder, color_fix=color_fix, to_gpu=vae_to_gpu)

    def _gaussian_weights(self, tile_width, tile_height, nbatches):
        """Generates a gaussian mask of weights for tile contributions"""
        from numpy import pi, exp, sqrt
        import numpy as np

        latent_width = tile_width
        latent_height = tile_height

        var = 0.01
        midpoint = (latent_width - 1) / 2  # -1 because index goes from 0 to latent_width - 1
        x_probs = [exp(-(x-midpoint)*(x-midpoint)/(latent_width*latent_width)/(2*var)) / sqrt(2*pi*var) for x in range(latent_width)]
        midpoint = latent_height / 2
        y_probs = [exp(-(y-midpoint)*(y-midpoint)/(latent_height*latent_height)/(2*var)) / sqrt(2*pi*var) for y in range(latent_height)]

        weights = np.outer(y_probs, x_probs)
        return torch.tile(torch.tensor(weights, device=self.device), (nbatches, self.unet.config.in_channels, 1, 1))


class OSEDiff_inference_time(torch.nn.Module):
    def __init__(self, args):
        super().__init__()

        self.args = args
        self.device =  torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.pretrained_model_name_or_path, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(self.args.pretrained_model_name_or_path, subfolder="text_encoder")
        self.noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
        self.noise_scheduler.set_timesteps(1, device="cuda")
        self.vae_enc, self.vae_dec = initialize_vae(self.args)
        self.unet = UNet2DConditionModel.from_pretrained(self.args.pretrained_model_name_or_path, subfolder="unet")

        
        self.weight_dtype = torch.float16 if args.mixed_precision == "fp16" else torch.float32
        print("weight_dtype", self.weight_dtype)
        self.skip_conv_out = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=3//2),
            nn.ReLU(),
            nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=3//2)
        ).to(self.device, dtype=self.weight_dtype)

        self.degrader = initialize_degrader(self.args).to('cuda', dtype=self.weight_dtype)
        self.degrader_projection = Degrader_projection().to('cuda', dtype=self.weight_dtype)
        self.mlp = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU()
        ).to("cuda")

        osediff = torch.load(args.osediff_path)
        self.load_ckpt(osediff)

        # merge lora
        if self.args.merge_and_unload_lora:
            print(f'===> MERGE LORA <===')
            # self.vae = self.vae.merge_and_unload()
            self.unet = self.unet.merge_and_unload()

        self.unet.to("cuda", dtype=self.weight_dtype)
        self.vae_enc.to("cuda", dtype=self.weight_dtype)
        self.vae_dec.to("cuda", dtype=self.weight_dtype)
        self.text_encoder.to("cuda", dtype=self.weight_dtype)
        self.timesteps = torch.tensor([999], device="cuda").long() 
        self.noise_scheduler.alphas_cumprod = self.noise_scheduler.alphas_cumprod.cuda()

        

    def load_ckpt(self, model):
        print("begin load ckpt")
        # load unet lora
        lora_conf_encoder = LoraConfig(r=model["rank_unet"], init_lora_weights="gaussian", target_modules=model["l_target_modules_encoder"])
        lora_conf_decoder = LoraConfig(r=model["rank_unet"], init_lora_weights="gaussian", target_modules=model["l_target_modules_encoder"])
        lora_conf_others = LoraConfig(r=model["rank_unet"], init_lora_weights="gaussian", target_modules=model["l_modules_others"])
        self.unet.add_adapter(lora_conf_encoder, adapter_name="default_encoder")
        self.unet.add_adapter(lora_conf_decoder, adapter_name="default_decoder")
        self.unet.add_adapter(lora_conf_others, adapter_name="default_others")
        self.unet.set_adapter(["default_encoder", "default_decoder", "default_others"])
        
        #load vae lora
        vae = AutoencoderKL.from_pretrained(self.args.pretrained_model_name_or_path, subfolder="vae")

        vae.encoder.forward = my_vae_encoder_fwd.__get__(vae.encoder, vae.encoder.__class__)
        vae.decoder.forward = my_vae_decoder_fwd.__get__(vae.decoder, vae.decoder.__class__)
        vae.decoder.skip_conv_1 = torch.nn.Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        vae.decoder.skip_conv_2 = torch.nn.Conv2d(256, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        vae.decoder.skip_conv_3 = torch.nn.Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        vae.decoder.skip_conv_4 = torch.nn.Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        vae.decoder.ignore_skip = False
        vae_lora_conf_encoder = LoraConfig(r=model["rank_vae"], init_lora_weights="gaussian", target_modules=model["vae_lora_target_modules"])
        vae = get_peft_model(vae, vae_lora_conf_encoder)
        vae.add_adapter(adapter_name="vae_skip", peft_config=vae_lora_conf_encoder)
        vae.decoder.gamma = 1
        vae_b2a = copy.deepcopy(vae)
        self.vae_enc = VAE_encode(vae, vae_b2a=vae_b2a).to("cuda")
        self.vae_dec = VAE_decode(vae, vae_b2a=vae_b2a).to("cuda")
        
        for name, param in model["state_dict_trainable_params"].items():
            if name in self.state_dict():
                self.state_dict()[name].copy_(param)
                print("loaded", name)
        
        self.degrader.load_state_dict(model["state_dict_degrader"])
        self.degrader_projection.load_state_dict(model["state_dict_degrader_projection"])
        self.mlp.load_state_dict(model["state_dict_mlp"])
        print("model loaded")

    def encode_prompt(self, prompt_batch):
        prompt_embeds_list = []
        with torch.no_grad():
            for caption in prompt_batch:
                text_input_ids = self.tokenizer(
                    caption, max_length=self.tokenizer.model_max_length,
                    padding="max_length", truncation=True, return_tensors="pt"
                ).input_ids
                prompt_embeds = self.text_encoder(
                    text_input_ids.to(self.text_encoder.device),
                )[0]
                prompt_embeds_list.append(prompt_embeds)
        prompt_embeds = torch.concat(prompt_embeds_list, dim=0)
        return prompt_embeds

    @torch.no_grad()
    def forward(self, lq, prompt, direction="a2b"):
        
        lq = lq.to(self.weight_dtype)
        degraded_feature = self.degrader(lq, lq)
        if isinstance(degraded_feature, tuple):
            degraded_feature = degraded_feature[0]
        degraded_feature = degraded_feature.to(self.mlp[0].weight.dtype)

        prompt_embeds = self.encode_prompt([prompt])

        lq_latent = self.vae_enc(lq, direction=direction)

        degraded_feature = self.mlp(degraded_feature).unsqueeze(1)
        prompt_degraded_embeds = torch.cat((prompt_embeds, degraded_feature), dim=1)
        prompt_degraded_embeds = prompt_degraded_embeds.to(self.weight_dtype)
        model_pred = self.unet(lq_latent, self.timesteps, encoder_hidden_states=prompt_degraded_embeds).sample

        x_denoised = self.noise_scheduler.step(model_pred, self.timesteps, lq_latent, return_dict=True).prev_sample

        output_image = self.vae_dec(x_denoised.to(self.weight_dtype), direction=direction) + self.skip_conv_out(lq)

        return output_image
    
def generate_mixed_noise(latents, poisson_lambda=1.0):
    # 生成高斯噪声
    gaussian_noise = torch.randn_like(latents)

    # 生成泊松噪声
    # 使用泊松分布的参数 lambda
    poisson_noise = torch.poisson(torch.full(latents.shape, poisson_lambda, device=latents.device))

    # 混合高斯噪声和泊松噪声
    mixed_noise = gaussian_noise + poisson_noise.float()  # 确保类型一致

    return mixed_noise

