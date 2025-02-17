import cv2
from PIL import Image
import numpy as np
from scipy import ndimage


def apply_canny(binary_mask, threshold1=100, threshold2=200):
    canny_edges = cv2.Canny(binary_mask, threshold1, threshold2)
    return canny_edges


def cut_image_with_mask(image_input, mask_input, original_input=None) -> Image.Image:
    image = image_input

    mask_array = mask_input
    binary_mask = (mask_array > 0).astype(np.uint8)

    if original_input is not None:
        if isinstance(original_input, str):
            original = Image.open(original_input)
        else:
            original = original_input
    else:
        original = None

    if original is not None:
        original = original.resize(image.size)

    labeled_array, num_features = ndimage.label(binary_mask)
    component_sizes = [(labeled_array == i).sum() for i in range(1, num_features + 1)]

    if component_sizes:
        largest_component = np.argmax(component_sizes) + 1
        new_mask = (labeled_array == largest_component).astype(np.uint8) * 255
        mask = Image.fromarray(new_mask)
    else:
        mask = Image.fromarray(binary_mask * 255)

    image = image.convert("RGBA")
    mask = mask.convert("L")

    if original is not None:
        original = original.convert("RGBA")
        original.paste(image, mask=mask)
        return original
    else:
        result = Image.new("RGBA", image.size, (255, 255, 255, 255))
        result.paste(image, mask=mask)
        return result


def reduce_image_size(image_path, scale_percentage):
    img = Image.open(image_path)
    width, height = img.size
    
    new_width = int(width * (scale_percentage/100))
    new_height = int(height * (scale_percentage/100))
    
    return img.resize((new_width, new_height))