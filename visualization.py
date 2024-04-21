from diffusers.utils import load_image
from diffusers import StableDiffusionControlNetPipeline, StableDiffusionControlNetImg2ImgPipeline, ControlNetModel, UniPCMultistepScheduler
from transformers import pipeline
import torch
from PIL import Image
import cv2
import numpy as np

from datasets import load_dataset
import numpy as np
import matplotlib.pyplot as plt


cmap = plt.cm.viridis

ds = load_dataset("sayakpaul/nyu_depth_v2", streaming=True)
shuffled_ds = ds.shuffle(seed=367)

class NormalizeImage(object):
    """Normlize image by given mean and std."""

    def __init__(self, mean, std):
        self.__mean = mean
        self.__std = std

    def __call__(self, sample):
        sample["image"] = (sample["image"] - self.__mean) / self.__std

        return sample

def colored_depthmap(depth, d_min=None, d_max=None):
    if d_min is None:
        d_min = np.min(depth)
    if d_max is None:
        d_max = np.max(depth)
    depth_relative = (depth - d_min) / (d_max - d_min)
    return 255 * cmap(depth_relative)[:,:,:3] # H, W, C


def merge_into_row(input, depth_target):
    input = np.array(input)
    depth_target = np.squeeze(np.array(depth_target))

    d_min = np.min(depth_target)
    d_max = np.max(depth_target)
    depth_target_col = colored_depthmap(depth_target, d_min, d_max)
    img_merge = np.hstack([input, depth_target_col])
    
    return img_merge

def preprocess_nyu_depth(image):
    image = np.squeeze(np.array(image))
    
    image = (image.min() - image) / (image.max() - image.min())
    image += 1
    #image = (image).astype(np.uint8)
    #control_image = Image.fromarray(image)

    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    image = (255 * image).astype(np.uint8)
    control_image = Image.fromarray(image)
    return control_image

def get_depth_map(image):
    #image = np.array(image)
    image = np.squeeze(np.array(image))
    print(image.min(), image.max())
    control_image = Image.fromarray(image)
    control_image.save("./cvg_images/control_2d.png") 
    print(image.shape)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    #detected_map = torch.from_numpy(image).float() / 255.0
    #depth_map = detected_map.permute(2, 0, 1)
    
    control_image = Image.fromarray(image)
    return control_image

def getPredictedDepthmap(image):
    #depth_estimator = pipeline('depth-estimation')
    depth_estimator = pipeline(task="depth-estimation", model="LiheYoung/depth-anything-small-hf")
    image = depth_estimator(image)['depth']
    image = np.array(image)
    
    image = image[:, :, None]
    print(image.min(), image.max())
    image = np.concatenate([image, image, image], axis=2)
    control_image = Image.fromarray(image)
    return control_image


dirPath = './cvg_images/'
'''
normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
transform = Compose(
        [
            Resize(
                net_w,
                net_h,
                resize_target=None,
                keep_aspect_ratio=True,
                ensure_multiple_of=32,
                resize_method="minimal",
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            normalization,
            PrepareForNet(),
        ]
    )
'''

for i, example in enumerate(shuffled_ds["train"]):
    print(i)

    depth_map = preprocess_nyu_depth(example["depth_map"])
    #depth_map = getPredictedDepthmap(example["image"])
    #depth_map = get_depth_map(example["depth_map"]).unsqueeze(0).half().to("cuda")

    #controlnet = ControlNetModel.from_pretrained("lllyasviel/control_v11f1p_sd15_depth", torch_dtype=torch.float16, use_safetensors=True)
    #pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
    #"runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16, use_safetensors=True
    #)

    controlnet = ControlNetModel.from_pretrained("lllyasviel/control_v11f1p_sd15_depth", torch_dtype=torch.float16)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16, use_safetensors=True
    )

    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()

    generator = torch.manual_seed(0)
    output = pipe("", num_inference_steps=30, generator=generator, image=depth_map, guess_mode=True).images[0]


    #pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    #pipe.enable_model_cpu_offload()

    #output = pipe("", image=example["image"], control_image=depth_map, guess_mode=True,).images[0]

    output.save(dirPath + str(i) + 'output.png')
    #depth = depth_map[0,0].cpu().numpy()
    #print(depth.shape)
    #depth = Image.fromarray(depth, mode='L')
    depth_map.save(dirPath + str(i) + 'depth.png')
    example["image"].save(dirPath + str(i) + 'image.png')
    
    if i>10:
        print('break ' + str(i))
        break

