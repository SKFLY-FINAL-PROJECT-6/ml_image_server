from PIL import Image
import cv2
import numpy as np
import config
import torch
import redis
import json
import boto3

import requests
from io import BytesIO

from models.vision.StableDiffusionControlNet import StableDiffusionControlNetModel
from models.vision.StableDiffusion import StableDiffusionModel
from models.vision.Segmentation import SegmentationModel
from utils.image_utils import apply_canny, paste_largest_segment, reduce_image_size

SCALE_PERCENTAGE = 50


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
        image_path, 
        text_prompt,
    ):

    image_resized = reduce_image_size(image_path, SCALE_PERCENTAGE)
    #image_resized = reduce_image_size(image_path, 50)

    image = Image.open(image_path)

    image = reduce_image_size(image_path, 50)

    segmentation = segmentation_model.segment(image)

    binary_mask_255 = segmentation_model.get_target_binary_mask(segmentation, target_class_name='wall')

    binary_mask_to_canny_edge = apply_canny(binary_mask_255, 100, 200)

    #지우지 마시오 
    #painting = diffuser_model.generate_painting(text_prompt)

    painting = image_genrator_model.generate_painting(binary_mask_to_canny_edge, text_prompt, image)

    result = paste_largest_segment(painting, binary_mask_255, image)

    return result 

def get_presigned_url(bucket_name: str, uuid: str, expiration: int = 3600):
    """
    S3에서 특정 UUID에 해당하는 Presigned URL을 가져오는 함수.

    :param bucket_name: S3 버킷 이름
    :param uuid: 이미지의 UUID (S3 키)
    :param expiration: Presigned URL 만료 시간 (초 단위, 기본 3600초)
    :return: Presigned URL (문자열)
    """
    s3_client = boto3.client('s3')

    try:
        # Presigned URL 생성
        presigned_url = s3_client.generate_presigned_url(
            'get_object',
            Params={'Bucket': bucket_name, 'Key': uuid},
            ExpiresIn=expiration
        )
        return presigned_url

    except Exception as e:
        print(f"Error generating presigned URL: {e}")
        return None

if __name__ == "__main__":

    redis_client = redis.Redis(host=config.redis_host, port=config.redis_port)


    segmentation_model = load_seg_model_checkpoint(cuda=cuda)
    diffuser_model = load_diffuser_model_checkpoint(cuda=cuda)


    image_path = r"C:\Users\013\Desktop\ml_image_server\test_files\pngtree-graphic-depiction-of-an-indoor-space-featuring-a-textured-concrete-wall-image_13805502.png"
    wall_image_path = r"C:\Users\013\Desktop\test\test_image_002.jpg"
    scribble_path = r"C:\Users\013\Desktop\test\test_image_002.jpg"
    text_prompt = "draw painting of multiculture"
    
    #diffuser_controlnet_model_custom = load_diffuser_controlnet_model_checkpoint(cuda=cuda, theme="custom")
    #diffuser_controlnet_model_animals = load_diffuser_controlnet_model_checkpoint(cuda=cuda, theme="animals")
    #diffuser_controlnet_model_nature = load_diffuser_controlnet_model_checkpoint(cuda=cuda, theme="nature")
    #diffuser_controlnet_model_play = load_diffuser_controlnet_model_checkpoint(cuda=cuda, theme="play")
    diffuser_controlnet_model_ocean = load_diffuser_controlnet_model_checkpoint(cuda=cuda, theme="ocean")
    #diffuser_controlnet_model_urban = load_diffuser_controlnet_model_checkpoint(cuda=cuda, theme="urban")
    #diffuser_controlnet_model_space = load_diffuser_controlnet_model_checkpoint(cuda=cuda, theme="space")
    #diffuser_controlnet_model_saekdam = load_diffuser_controlnet_model_checkpoint(cuda=cuda, theme="saekdam")
    #diffuser_controlnet_model_gru = load_diffuser_controlnet_model_checkpoint(cuda=cuda, theme="gru")


    image_path = r"C:\Users\013\Desktop\ml_image_server\original_image.png"
    # wall_image_path = r"C:\Users\013\Desktop\test\test_image_002.jpg"
    # scribble_path = r"C:\Users\013\Desktop\test\test_image_002.jpg"
    print('load completed')

    #text_prompt = "give the best painting, saekdam style, flat design, wall painting"

    while True:
        _,message = redis_client.blpop(config.queue_name)
        if message:
            response = json.loads(message)
            image_id = response['id']
            prompt = response['requirement']
            theme = response['theme']
            

            image_url = config.s3_client.generate_presigned_url(
                'get_object',
                Params={'Bucket': config.s3_bucket, 'Key': image_id},
                ExpiresIn=3600
            )
            download_image = requests.get(image_url)
            img = Image.open(BytesIO(download_image.content))
            temp_path = "temp.jpg"
            img.save(temp_path)
            

            result_img = process_wall_painting(segmentation_model=segmentation_model, 
                                   image_genrator_model=diffuser_controlnet_model_ocean, 
                                   image_path=temp_path, 
                                   text_prompt=prompt)

            result_img.show()


