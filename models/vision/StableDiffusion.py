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
        torch.cuda.empty_cache()
        
        self.controlnet = ControlNetModel.from_pretrained(
            controlnet_model_name,
            torch_dtype=torch.float16,
            use_safetensors=True
        ).to("cpu").to(device)
        torch.cuda.empty_cache()
        
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            sd_model_name,
            controlnet=self.controlnet, 
            torch_dtype=torch.float16,
            use_safetensors=True
        ).to("cpu").to(device)
        torch.cuda.empty_cache()

        self.pipe.safety_checker = None
        self.pipe.requires_safety_checking = False

        lora_path = r"C:\Users\013\Desktop\ml_image_server\saekdam_style"
        lora_weights = [
            "animals-10.safetensors",
            "nature-10.safetensors", 
            "play-10.safetensors",
            "saekdam_ocean-10.safetensors",
            "urban-10.safetensors",
            "saekdam_space-10.safetensors",
            "saekdam-10.safetensors",
            "gru-10.safetensors"
        ]

        for weight in lora_weights:
            self.pipe.load_lora_weights(lora_path, weight_name=weight)
            
        torch.cuda.empty_cache()




    def generate_painting(self, scribble_path, text_prompt: str, wall_image: Image.Image) -> Image.Image:
        wall_width, wall_height = wall_image.size
        
        if isinstance(scribble_path, str):
            scribble_img = Image.open(scribble_path).convert("L")
        elif isinstance(scribble_path, np.ndarray):
            scribble_img = Image.fromarray(scribble_path).convert("L")
        else:
            scribble_img = scribble_path.convert("L")

        scribble_img = scribble_img.resize((wall_width, wall_height), Image.LANCZOS)

        result = self.pipe(
            prompt=text_prompt,
            image=[scribble_img],
            controlnet_conditioning_scale=1.0,
            num_inference_steps=200,
            width=wall_width,
            height=wall_height
        )

        return result.images[0]

