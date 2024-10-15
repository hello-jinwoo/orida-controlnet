#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

import json
import argparse
import contextlib
import gc
import logging
import math
import os
import random
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from datasets import load_dataset
from huggingface_hub import create_repo, upload_folder
from packaging import version
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig

import diffusers
from diffusers import (
    AutoencoderKL,
    # ControlNetModel,
    DDPMScheduler,
    # DDIMScheduler,
    # StableDiffusionControlNetPipeline,
    # StableDiffusionControlNetImg2ImgPipeline,
    AutoPipelineForInpainting,
    StableDiffusionInpaintPipeline,
    UNet2DConditionModel,
    UniPCMultistepScheduler,
)
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module
from diffusers.image_processor import PipelineImageInput, VaeImageProcessor 

from diffusers.image_processor import IPAdapterMaskProcessor

##### Customized Part #####
from custom_pipeline_stable_diffusion_inpaint import CustomStableDiffusionInpaintPipeline

if is_wandb_available():
    import wandb

import os
from PIL import Image
from datasets import Dataset
import cv2
# import pyvips 

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.31.0.dev0")

def get_masked_img(img, mask):
    img_np = np.array(img)
    mask_np = np.array(mask)
    if mask_np.ndim == 2:
        mask_np = mask_np[..., None]
    masked_img_np = np.where(mask_np > 1e-2, img_np, 0)
    masked_img = Image.fromarray(masked_img_np)
    return masked_img

def get_resized_masked_img(img, mask, bbox, img_len):
    masked_img = get_masked_img(img, mask)
    x_min_, y_min_, x_max_, y_max_ = map(float, bbox.split(','))
    masked_w = x_max_ - x_min_
    masked_h = y_max_ - y_min_
    if masked_w > masked_h:
        masked_reshaped_w = 1.0
        masked_reshaped_h = masked_h / masked_w
    else:
        masked_reshaped_w = masked_w / masked_h
        masked_reshaped_h = 1.0
    reshaped_x_min_ = max(0.01, 0.5-(masked_reshaped_w/2))
    reshaped_y_min_ = max(0.01, 0.5-(masked_reshaped_h/2))
    reshaped_x_max_ = min(0.99, 0.5+(masked_reshaped_w/2))
    reshaped_y_max_ = min(0.99, 0.5+(masked_reshaped_h/2))
    masked_reshaped_bbox = f"{reshaped_x_min_}, {reshaped_y_min_}, {reshaped_x_max_}, {reshaped_y_max_}"
    masked_img = reshape_image_to_tgt_pos(masked_img, bbox, masked_reshaped_bbox, img_len)
    return masked_img

def reshape_image_to_tgt_pos(src_img, src_obj_bbox, tgt_pos_bbox, img_size=1024, margin=0):
    # Parse bounding boxes (assuming comma-separated format)
    src_x_min, src_y_min, src_x_max, src_y_max = map(float, src_obj_bbox.split(','))
    tgt_x_min, tgt_y_min, tgt_x_max, tgt_y_max = map(float, tgt_pos_bbox.split(','))

    src_x_min, src_y_min, src_x_max, src_y_max = src_x_min*img_size, src_y_min*img_size, src_x_max*img_size, src_y_max*img_size
    tgt_x_min, tgt_y_min, tgt_x_max, tgt_y_max = tgt_x_min*img_size, tgt_y_min*img_size, tgt_x_max*img_size, tgt_y_max*img_size
    src_x_min, src_y_min, src_x_max, src_y_max = int(src_x_min), int(src_y_min), int(src_x_max), int(src_y_max)
    tgt_x_min, tgt_y_min, tgt_x_max, tgt_y_max = int(tgt_x_min), int(tgt_y_min), int(tgt_x_max), int(tgt_y_max)

    # Calculate original, source and target center
    x, y = src_img.size
    x_center, y_center = x // 2, y // 2
    tgt_x_center = (tgt_x_min + tgt_x_max) // 2
    tgt_y_center = (tgt_y_min + tgt_y_max) // 2

    # Calculate target translation
    tgt_x_t, tgt_y_t = tgt_x_center - x_center, tgt_y_center - y_center
    center_to_tgt_matrix = (1, 0, -tgt_x_t, 0, 1, -tgt_y_t)
    # Calculate target bounding box dimensions
    src_w = src_x_max - src_x_min
    src_h = src_y_max - src_y_min
    tgt_w = tgt_x_max - tgt_x_min
    tgt_h = tgt_y_max - tgt_y_min
    scale_w = tgt_w / src_w
    scale_h = tgt_h / src_h

    # Apply margin to make new length values
    src_x_min -= margin
    src_y_min -= margin
    src_x_max += margin
    src_x_max += margin
    src_w = src_x_max - src_x_min
    src_h = src_y_max - src_y_min
    tgt_x_min -= int(margin * scale_w)
    tgt_y_min -= int(margin * scale_h)
    tgt_x_max += int(margin * scale_w)
    tgt_y_max += int(margin * scale_h)
    tgt_w = tgt_x_max - tgt_x_min
    tgt_h = tgt_y_max - tgt_y_min

    # 1. Crop object mask
    cropped_img = src_img.crop((src_x_min, src_y_min, src_x_max, src_y_max))
    # 2. Scale the cropped mask
    scaled_img = cropped_img.resize((tgt_w, tgt_h))
    # 3. Resized image 
    x_, y_ = scaled_img.size
    left_padding, top_padding, left_crop, top_crop = 0, 0, 0, 0
    if x > x_ and y > y_:
        # padding (x,y)
        left_padding = (x - x_) // 2 
        top_padding = (y - y_) // 2
        resized_img = Image.new("RGB", (x, y), (0, 0, 0))
        resized_img.paste(scaled_img, (left_padding, top_padding))
    elif x > x_ and y <= y_:
        # padding (x)
        left_padding = (x - x_) // 2 
        resized_img = Image.new("RGB", (x, y_), (0, 0, 0))
        resized_img.paste(scaled_img, (left_padding, 0))
        # crop (y)
        top_crop = (y_ - y) // 2
        bottom_crop = top_crop + y
        resized_img = resized_img.crop((0, top_crop, 0, bottom_crop))
    elif x <= x_ and y > y_:
        # padding (y)
        top_padding = (y - y_) // 2
        resized_img = Image.new("RGB", (x_, y), (0, 0, 0))
        resized_img.paste(scaled_img, (0, top_padding))
        # crop (x)
        left_crop = (x_ - x) // 2 
        right_crop = left_crop + x
        resized_img = resized_img.crop((left_crop, 0, right_crop, 0))
    else:
        # crop (x,y)
        left_crop = (x_ - x) // 2 
        top_crop = (y_ - y) // 2
        right_crop = left_crop + x
        bottom_crop = top_crop + y
        resized_img = scaled_img.crop((left_crop, top_crop, right_crop, bottom_crop))

    # 4. Translate centered mask to tgt pos
    final_mask = resized_img.transform(
        resized_img.size,  # keep the same size
        Image.AFFINE,  # specify affine transformation
        center_to_tgt_matrix  # provide the matrix
    )

    return final_mask

def get_concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst

def get_concat_v(im1, im2):
    dst = Image.new('RGB', (im1.width, im1.height + im2.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    return dst

def validation(
    vae, text_encoder, tokenizer, unet, args, weight_dtype
):
  
    pipeline = CustomStableDiffusionInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting", 
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        safety_checker=None,
        torch_dtype=weight_dtype,
    )
    # pipeline.enable_model_cpu_offload()
    # # remove following line if xFormers is not installed or you have PyTorch 2.0 or higher installed
    # pipeline.enable_xformers_memory_efficient_attention()
    # Load IP Adapter
    pipeline.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter_sd15.bin") # TODO : path
    pipeline.set_ip_adapter_scale(args.ip_adapter_scale) 

    # pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
    # pipeline.scheduler = DDPMScheduler.from_config(pipeline.scheduler.config) # TODO: [Validation] DDIM? DDPM? UniPC?
    pipeline = pipeline.to("cuda")
    pipeline.set_progress_bar_config(disable=True)

    if args.enable_xformers_memory_efficient_attention:
        pipeline.enable_xformers_memory_efficient_attention()

    if args.seed is None:
        generator = None
    else:
        generator = torch.Generator(device="cuda").manual_seed(args.seed)

    validation_src_dir = args.validation_src_dir
    validation_tgt_dir = args.validation_tgt_dir
    validation_prompts = args.validation_prompt
    image_logs = []
    inference_ctx = torch.autocast("cuda")

    validation_src_images = sorted([f for f in os.listdir(validation_src_dir) if ".jpg" in f  and "mask" not in f and f[0] != "."])
    validation_tgt_images = sorted([f for f in os.listdir(validation_tgt_dir) if ".jpg" in f and "_0.jpg" not in f and f[0] != "."])
    assert len(validation_src_images) == len(validation_tgt_images)

    validation_prompt_dict = None
    if args.validation_prompt_dict_dir != "":
        with open(args.validation_prompt_dict_dir, 'r') as f:
            validation_prompt_dict = json.load(f)

    for validation_src_image, validation_tgt_image in zip(validation_src_images, validation_tgt_images):
        if validation_prompt_dict == None:
            validation_prompt = ""
        else:
            validation_prompt = validation_prompt_dict[validation_tgt_image.split("_")[0]]

        validation_bg_image = f"{validation_tgt_dir}/{validation_tgt_image.split('.jpg')[0][:-2]}_0.jpg"
        validation_src_mask = f"{validation_src_dir}/{validation_src_image.split('.jpg')[0]}_mask.jpg"
        validation_src_bbox = f"{validation_src_dir}/{validation_src_image.split('.jpg')[0]}_bbox.txt"
        validation_tgt_bbox = f"{validation_tgt_dir}/{validation_tgt_image.split('.jpg')[0]}_bbox.txt"
        validation_src_image = f"{validation_src_dir}/{validation_src_image}"
        validation_tgt_image = f"{validation_tgt_dir}/{validation_tgt_image}"
        _, src_scene_id, src_pos_idx = validation_src_image.split("/")[1].split("_")
        obj_idx, tgt_scene_id, tgt_pos_idx = validation_tgt_image.split("/")[1].split("_")
        vis_name = f"{obj_idx}__from__{src_scene_id}_{src_pos_idx}__to_{tgt_scene_id}__{tgt_pos_idx}.jpg"
        
        validation_src_image = Image.open(validation_src_image).convert("RGB").resize((args.resolution, args.resolution), resample=Image.BILINEAR)
        validation_tgt_image = Image.open(validation_tgt_image).convert("RGB").resize((args.resolution, args.resolution), resample=Image.BILINEAR)
        validation_bg_image = Image.open(validation_bg_image).convert("RGB").resize((args.resolution, args.resolution), resample=Image.BILINEAR)
        validation_src_mask = Image.open(validation_src_mask).convert("RGB").resize((args.resolution, args.resolution))

        with open(validation_src_bbox, 'r') as f:
            validation_src_bbox = f.read().strip()
        with open(validation_tgt_bbox, 'r') as f:
            validation_tgt_bbox = f.read().strip()
        validation_tgt_mask = reshape_image_to_tgt_pos(validation_src_mask, validation_src_bbox, validation_tgt_bbox, args.resolution)

        validation_bg_image_np = np.array(validation_bg_image)
        validation_src_image_reshaped_np = np.array(reshape_image_to_tgt_pos(validation_src_image, validation_src_bbox, validation_tgt_bbox, args.resolution))
        validation_tgt_mask_np = np.array(validation_tgt_mask)
        validation_input_image_np = np.where(validation_tgt_mask_np > 1, validation_src_image_reshaped_np, validation_bg_image_np)
        validation_input_image = Image.fromarray(validation_input_image_np)

        # ip adapter input image and mask
        validation_srb_obj_image = get_resized_masked_img(validation_src_image, validation_src_mask, validation_src_bbox, args.resolution)
        processor = IPAdapterMaskProcessor()
        ip_adapter_masks = processor.preprocess(Image.fromarray(validation_tgt_mask_np), height=args.resolution, width=args.resolution) # TODO: list? one item?
        ip_adapter_masks = ip_adapter_masks.reshape(1, ip_adapter_masks.shape[0], ip_adapter_masks.shape[2], ip_adapter_masks.shape[3])

        
        vis_image_inputs = get_concat_h(validation_src_image, validation_bg_image)
        vis_image_inputs = get_concat_h(vis_image_inputs, validation_input_image)
        vis_image_inputs = get_concat_h(vis_image_inputs, validation_tgt_image)
        vis_image_inputs = get_concat_h(vis_image_inputs, validation_srb_obj_image)

        # for i in range(args.num_validation_images):
        for i in range(5): # to fit the visualize results, set num_validation_images == 4
            with inference_ctx:
                result_image = pipeline(
                    num_inference_steps=args.validation_num_inference_steps,
                    prompt=validation_prompt, 
                    image=validation_input_image, # v0.1~v0.6, v0.7??
                    # image=validation_tgt_image, # v0.7??
                    mask_image=validation_tgt_mask,
                    masked_image_latents=vae.encode(VaeImageProcessor().preprocess(validation_input_image).to(vae.device)).latent_dist.sample() * vae.config.scaling_factor, 
                    ip_adapter_image=validation_srb_obj_image,
                    cross_attention_kwargs={"ip_adapter_masks": ip_adapter_masks},
                    generator=generator
                ).images[0]
                if i == 0:
                    vis_image_outputs1 = result_image.copy()
                else:
                    vis_image_outputs1 = get_concat_h(vis_image_outputs1, result_image)
        # for i in range(5): # to fit the visualize results, set num_validation_images == 4
        #     with inference_ctx:
        #         result_image = pipeline(
        #             custom_unet=unet,
        #             custom_unet_init_timestep=args.denoising_init_timestep,
        #             custom_unet_end_timestep=args.denoising_end_timestep,
        #             num_inference_steps=args.validation_num_inference_steps,
        #             prompt=validation_prompt, 
        #             image=validation_input_image, # v0.1~v0.6, v0.7??
        #             # image=validation_tgt_image, # v0.7??
        #             mask_image=validation_tgt_mask,
        #             masked_image_latents=vae.encode(VaeImageProcessor().preprocess(validation_input_image).to(vae.device)).latent_dist.sample() * vae.config.scaling_factor, 
        #             ip_adapter_image=validation_input_image, # TODO : change ?
        #             generator=generator
        #         ).images[0]
        #         if i == 0:
        #             vis_image_outputs2 = result_image.copy()
        #         else:
        #             vis_image_outputs2 = get_concat_h(vis_image_outputs2, result_image)

        vis_image_final = get_concat_v(vis_image_inputs, vis_image_outputs1)
        # vis_image_final = get_concat_v(vis_image_final, vis_image_outputs2)
        os.makedirs(args.output_dir, exist_ok=True)
        vis_image_final.save(f"{args.output_dir}/{vis_name}")

    del pipeline
    gc.collect()
    torch.cuda.empty_cache()

def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "RobertaSeriesModelWithTransformation":
        from diffusers.pipelines.alt_diffusion.modeling_roberta_series import RobertaSeriesModelWithTransformation

        return RobertaSeriesModelWithTransformation
    else:
        raise ValueError(f"{model_class} is not supported.")


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="ORIDa training using sd-inpaint")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--custom_pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--image_column", type=str, default="image", help="The column of the dataset containing the target image."
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default="text",
        help="The column of the dataset containing a caption or a list of captions.",
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
        nargs="+",
        help=(
            "A set of prompts evaluated every `--validation_steps` and logged to `--report_to`."
            " Provide either a matching number of `--validation_image`s, a single `--validation_image`"
            " to be used with all prompts, or a single prompt that will be used with all `--validation_image`s."
        ),
    )
    parser.add_argument(
        "--validation_prompt_dict_dir", # TODO: edit the description
        type=str,
        default="",
        help=(
            ""
        ),
    )
    parser.add_argument(
        "--validation_src_dir", # TODO: edit the description
        type=str,
        default=None,
        # nargs="+",
        help=(
            ""
        ),
    )
    parser.add_argument(
        "--validation_tgt_dir", # TODO: edit the description
        type=str,
        default=None,
        # nargs="+",
        help=(
            ""
        ),
    )
    parser.add_argument(
        "--validation_num_inference_steps", # TODO: edit the description
        type=int,
        default=20,
        help=(
            ""
        ),
    )
    parser.add_argument(
        "--validation_init_timesteps", # TODO: edit the description
        type=int,
        nargs="+",
        default=None,
        help=(
            ""
        ),
    )
    parser.add_argument(
        "--denoising_init_timestep", # TODO: edit the description
        type=int,
        default=1000,
        help=(
            "out of 1000 denoising steps"
        ),
    )
    parser.add_argument(
        "--denoising_end_timestep", # TODO: edit the description
        type=int,
        default=0,
        help=(
            "out of 1000 denoising steps"
        ),
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=1,
        help="Number of images to be generated for each `--validation_image`, `--validation_prompt` pair",
    )

    parser.add_argument(
        "--ip_adapter_scale",
        type=float,
        default=0.0,
        help="",
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    if args.resolution % 8 != 0:
        raise ValueError(
            # "`--resolution` must be divisible by 8 for consistently sized encoded images between the VAE and the controlnet encoder."
            "`--resolution` must be divisible by 8."
        )

    return args

def main(args):

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        use_fast=False,
    )

    # import correct text encoder class
    text_encoder_cls = import_model_class_from_model_name_or_path(args.pretrained_model_name_or_path)

    # Load models
    text_encoder = text_encoder_cls.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(args.custom_pretrained_model_name_or_path, subfolder="unet")

    vae.requires_grad_(False)
    unet.requires_grad_(False) # [change]
    text_encoder.requires_grad_(False)
    unet.eval()
    
    # Move vae, unet and text_encoder to device and cast to weight_dtype
    weight_dtype = torch.float32
    vae.to("cuda", dtype=weight_dtype)
    unet.to("cuda", dtype=weight_dtype)
    text_encoder.to("cuda", dtype=weight_dtype)

    validation(vae=vae,
               text_encoder=text_encoder,
               tokenizer=tokenizer,
               unet=unet,
               args=args,
               weight_dtype=weight_dtype)

if __name__ == "__main__":
    args = parse_args()
    main(args)
