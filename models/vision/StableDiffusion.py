from diffusers import StableDiffusionPipeline
import torch
from safetensors.torch import load_file

from accelerate import init_empty_weights


class StableDiffusionModel:
    def __init__(self, model_name, device):
        self.device = device
        
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            local_files_only=True,
            use_safetensors=True
        ).to(device)

        # Disable NSFW filter
        self.pipe.safety_checker = None 
        self.pipe.requires_safety_checking = False

        # Load LoRA weights
        lora_path = r"C:\Users\013\Desktop\ml_image_server\checkpoints"

        self.pipe.load_lora_weights(lora_path, weight_name="saekdam-10.safetensors")
        self.pipe.load_lora_weights(lora_path, weight_name="gru-10.safetensors")
        

        torch.cuda.empty_cache()

    def generate_painting(self, prompt):
        image = self.pipe(prompt).images[0]
        torch.cuda.empty_cache()
        return image