from diffusers import AutoPipelineForInpainting
from diffusers.utils import load_image
import torch

import torch
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
    use_safetensors=True,
)

pipe.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter_sd15.bin")
pipe = pipe.to('cuda')
pipe.set_ip_adapter_scale(0.2)

img = load_image("asdf.png")
# pipe.enable_attention_slicing()

image = pipe(prompt="", ip_adapter_image=img).images[0]

image.save("no_prompt.png")

image = pipe(
    prompt="a pear-shaped ceramic is on the table",
    ip_adapter_image=img,
).images[0]

image.save("yes_prompt.png")