import torch
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline
from PIL import Image
import numpy as np
from scipy import ndimage


class StableDiffusionModel:

    def __init__(
        self,
        controlnet_model_name,
        sd_model_name,
        device,
    ):
        self.device = device
        self.controlnet = ControlNetModel.from_pretrained(controlnet_model_name).to(device, torch.float16)
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            sd_model_name,
            controlnet=self.controlnet,
            torch_dtype=torch.float16
        ).to(device)

    def generate_painting(self, scribble_path, text_prompt: str, wall_image: Image.Image) -> Image.Image:
        wall_width, wall_height = wall_image.size

        if isinstance(scribble_path, Image.Image):
            scribble_img = scribble_path.convert("L")
        else:
            scribble_img = Image.open(scribble_path).convert("L")

        scribble_img = scribble_img.resize((wall_width, wall_height), Image.LANCZOS)

        result = self.pipe(
            prompt=text_prompt,
            image=[scribble_img],
            controlnet_conditioning_scale=1.0,
            num_inference_steps=400,
            width=wall_width,
            height=wall_height
        )

        return result.images[0]

