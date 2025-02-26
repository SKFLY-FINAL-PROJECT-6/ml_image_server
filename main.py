from PIL import Image
import cv2
import numpy as np
import config
import torch
import redis
import json
import boto3
import io
import time 
import requests
from io import BytesIO
from enum import Enum
from models.vision.StableDiffusionControlNet import StableDiffusionControlNetModel
from models.vision.StableDiffusion import StableDiffusionModel
from models.vision.Segmentation import SegmentationModel
from models.llm.llm_model import LLMModel
from utils.image_utils import apply_canny, paste_largest_segment, reduce_image_size_to_MEDIUM_SIZE, cut_image_by_user_input, merge_cropped_image_back

# Check if CUDA is available. If not, raise an exception.
if not torch.cuda.is_available():
    raise Exception("CUDA is not available. Please use a GPU-enabled machine.")

cuda = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_seg_model_checkpoint(cuda: bool = False):
    print("Loading segmentation model checkpoint...")
    seg_model = SegmentationModel(
        "nvidia/segformer-b4-finetuned-ade-512-512", 
        device=cuda,
        use_fast=True,
        )

    return seg_model

def load_diffuser_controlnet_model_checkpoint(cuda: bool = False, theme: str = "custom"):
    print("Loading diffuser model checkpoint...")
    diffuser_controlnet_model = StableDiffusionControlNetModel(
        controlnet_model_name="lllyasviel/control_v11p_sd15_scribble", 
        sd_model_name="runwayml/stable-diffusion-v1-5",
        device=cuda,
        theme=theme
        )
    return diffuser_controlnet_model


def load_diffuser_model_checkpoint(cuda: bool = False):
    print("Loading diffuser model checkpoint...")
    diffuser_model = StableDiffusionModel(
        model_name="runwayml/stable-diffusion-v1-5",
        device=cuda,
    )
    return diffuser_model

def process_wall_painting(
        segmentation_model,
        image_genrator_model,
        image,
        text_prompt,
        x, y, w, h
    ):

    is_segmentation_required = x == 0 and y == 0 and w == 1 and h == 1
    
    if is_segmentation_required:
        # Segmentation-based processing
        segmentation = segmentation_model.segment(image)
        binary_mask_255 = segmentation_model.get_target_binary_mask(segmentation, target_class_name='wall')
        binary_mask_to_canny_edge = apply_canny(binary_mask_255, 100, 200)
    image_resized = reduce_image_size(image_path, SCALE_PERCENTAGE)

    segmentation = seg_model.segment(image_resized)

        # Generate painting using masked input
        painting = image_genrator_model.generate_painting(binary_mask_to_canny_edge, text_prompt, image)

        result = paste_largest_segment(painting, binary_mask_255, image)

    else:
        cropped_img = cut_image_by_user_input(image, x, y, w, h)

        processed_cropped = image_genrator_model.generate_painting(cropped_img, text_prompt, cropped_img)
        result = merge_cropped_image_back(image, processed_cropped, x, y, w, h)

    return result


if __name__ == "__main__":
    class TaskProgress(Enum):
        WAITING = "WAITING"
        IN_PROGRESS = "IN_PROGRESS"
        COMPLETED = "COMPLETED"
        FAILED = "FAILED"
    
    redis_client = redis.Redis(host=config.redis_host, port=config.redis_port)
    segmentation_model = load_seg_model_checkpoint(cuda=cuda)
    #diffuser_model = load_diffuser_model_checkpoint(cuda=cuda)

    llm_model = LLMModel()


    # image_path = r"C:\Users\013\Desktop\ml_image_server\test_files\pngtree-graphic-depiction-of-an-indoor-space-featuring-a-textured-concrete-wall-image_13805502.png"
    # wall_image_path = r"C:\Users\013\Desktop\test\test_image_002.jpg"
    # scribble_path = r"C:\Users\013\Desktop\test\test_image_002.jpg"
    # text_prompt = "draw painting of multiculture"
    #image_path = r"C:\Users\013\Desktop\ml_image_server\original_image.png"


    #themes = ["Custom", "Animals", "Nature", "Play", "Ocean", "Urban", "Space", "Saekdam", "Gru"]
    themes = ["Animals", "Ocean", "Nature", "Play", "Urban", "Space"]

    # Load models once
    diffuser_controlnet_models = {
        theme: load_diffuser_controlnet_model_checkpoint(cuda=cuda, theme=theme)
        for theme in themes
    }

    theme_model_mapping = {key: diffuser_controlnet_models[key] for key in themes}


    while True:
         _,message = redis_client.blpop(config.queue_name)
         status = TaskProgress.IN_PROGRESS.value
         
         if message:
             response = json.loads(message)
             image_id = response['id']
             theme = response['theme']
             prompt = response['requirement']
             x = response['x']
             y = response['y']
             w = response['w']
             h = response['h']

             transfer = {
             'taskId':image_id,
             'status':status,
            }
             image_url = config.s3_client.generate_presigned_url(
             'get_object',
             Params={'Bucket': config.s3_bucket, 'Key': image_id},
             ExpiresIn=3600
            )


             time.sleep(2)  # Add a 1-second delay to slow down the thread
             
             download_image = requests.get(image_url)
             print(image_url)
             print(download_image)

             temp_path = "temp.jpeg"
             url_img = Image.open(BytesIO(download_image.content))
             url_img.save(temp_path)
             prompt_enhanced = llm_model.process_prompt(prompt, 'enhancing')

             medium_image = reduce_image_size_to_MEDIUM_SIZE(temp_path)

             prompt_final = prompt_enhanced +", flat design"  
             print(prompt_final)


             selected_model = theme_model_mapping.get(theme, diffuser_controlnet_models[theme])
             redis_client.publish(config.channel_name, json.dumps(transfer))

             result_img = process_wall_painting(
                segmentation_model=segmentation_model,
                image_genrator_model=selected_model,
                image=medium_image,
                text_prompt=prompt_final,
                x=x, y=y, w=w, h=h,
            )

             print("completed")
             result_img.save("result.png")
             
             put_image_url = config.s3_client.generate_presigned_url(
                'put_object',
                Params={'Bucket': config.s3_bucket, 'Key': image_id},
                ExpiresIn=3600
            )
             
             buffer = io.BytesIO()
             result_img = result_img.convert('RGB')
             result_img.save(buffer, format='JPEG')

             buffer.seek(0)
             requests.put(put_image_url, data=buffer.getvalue() ,headers={'Content-Type': 'image/jpeg'})
             status = TaskProgress.COMPLETED
             transfer['status'] = status.value
             redis_client.publish(config.channel_name, json.dumps(transfer))

             url_img.close()
             buffer.close()




