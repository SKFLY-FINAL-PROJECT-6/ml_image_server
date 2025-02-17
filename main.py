from PIL import Image
import cv2
import numpy as np

import torch
from models.StableDiffusion import StableDiffusionModel
from models.Segmentation import SegmentationModel
from utils.image_utils import apply_canny, cut_image_with_mask, reduce_image_size



# Check if CUDA is available. If not, raise an exception.
if not torch.cuda.is_available():
    raise Exception("CUDA is not available. Please use a GPU-enabled machine.")

cuda = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_seg_model_checkpoint(cuda: bool = False):
    print("Loading segmentation model checkpoint...")
    seg_model = SegmentationModel("nvidia/segformer-b4-finetuned-ade-512-512", cuda)

    return seg_model

def load_diffuser_model_checkpoint(cuda: bool = False):
    print("Loading diffuser model checkpoint...")
    diffuser_model = StableDiffusionModel(
        controlnet_model_name="lllyasviel/control_v11p_sd15_scribble", 
        sd_model_name="runwayml/stable-diffusion-v1-5",
        device=cuda
        )
    return diffuser_model


def process_wall_painting(
        segmentation_model, 
        image_genrator_model, 
        image_path, 
        scribble_path, 
        text_prompt
    ):

    image_resized = reduce_image_size(image_path, 50)

    segmentation = seg_model.segment(image_resized)

    binary_mask = seg_model.get_target_binary_mask(segmentation, target_class_name='wall')

    painting = diffuser_model.generate_painting(scribble_path, text_prompt, image_resized)

    result = cut_image_with_mask(painting, binary_mask, image_resized)

    return result 



if __name__ == "__main__":

    
    seg_model = load_seg_model_checkpoint(cuda=cuda)
    diffuser_model = load_diffuser_model_checkpoint(cuda=cuda)


    image_path = r"C:\Users\013\Desktop\test\test_image_002.jpg"
    wall_image_path = r"C:\Users\013\Desktop\test\test_image_002.jpg"
    scribble_path = r"C:\Users\013\Desktop\test\test_image_002.jpg"
    text_prompt = "A beautiful painting of a sunset on a beach"


    result = process_wall_painting(segmentation_model=seg_model, 
                                   image_genrator_model=diffuser_model, 
                                   image_path=image_path, 
                                   scribble_path=scribble_path, 
                                   text_prompt=text_prompt)

    result.show()
