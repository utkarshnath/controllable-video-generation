import torch
import os
from huggingface_hub import HfApi
from pathlib import Path
from diffusers.utils import load_image
from PIL import Image
import numpy as np
from transformers import pipeline


from diffusers import (
    ControlNetModel,
    StableDiffusionControlNetPipeline,
    UniPCMultistepScheduler,
)

checkpoint = "lllyasviel/control_v11f1p_sd15_depth"
image = load_image(
    "https://huggingface.co/lllyasviel/control_v11p_sd15_depth/resolve/main/images/input.png"
)

prompt = "Stormtrooper's lecture in beautiful lecture hall"

depth_estimator = pipeline('depth-estimation')
image = depth_estimator(image)['depth']
image = np.array(image)
print(image.shape, image.dtype, image.min(), image.max())
control_image = Image.fromarray(image)
control_image.save("./images/control_2d.png")

image = image[:, :, None]
image = np.concatenate([image, image, image], axis=2)
print(2, image.shape)
control_image = Image.fromarray(image)

control_image.save("./images/control.png")

controlnet = ControlNetModel.from_pretrained(checkpoint, torch_dtype=torch.float16)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16
)

pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()

generator = torch.manual_seed(0)
image = pipe(prompt, num_inference_steps=30, generator=generator, image=control_image).images[0]

image.save('images/image_out.png')

