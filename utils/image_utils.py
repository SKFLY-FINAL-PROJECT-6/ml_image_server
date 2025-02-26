import cv2
from PIL import Image
import numpy as np
from scipy import ndimage


def apply_canny(binary_mask, threshold1=100, threshold2=200):
    canny_edges = cv2.Canny(binary_mask, threshold1, threshold2)
    return canny_edges


def paste_largest_segment(image_input, mask_input, original_input=None) -> Image.Image:
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
    mask = mask.resize(image.size)

    if original is not None:
        original = original.convert("RGBA")
        original.paste(image, mask=mask)
        return original
    else:
        result = Image.new("RGBA", image.size, (255, 255, 255, 255))
        result.paste(image, mask=mask)
        return result


def reduce_image_size_to_LARGE_SIZE(image_path):
    
    img = Image.open(image_path)
    width, height = img.size

    new_width = int(width * 0.75)
    new_height = int(height * 0.75)

    resized_img = img.resize((new_width, new_height), Image.LANCZOS)

    return resized_img 


def reduce_image_size_to_MEDIUM_SIZE(image_path):
    
    img = Image.open(image_path)
    width, height = img.size

    new_width = int(width * 0.3)
    new_height = int(height * 0.3)

    resized_img = img.resize((new_width, new_height), Image.LANCZOS)

    print(resized_img.size)

    return resized_img 


def reduce_image_size_to_SMALL_SIZE(image_path):
    
    img = Image.open(image_path)
    width, height = img.size

    new_width = int(width * 0.25)
    new_height = int(height * 0.25)

    resized_img = img.resize((new_width, new_height), Image.LANCZOS)

    return resized_img 


def cut_image_by_user_input(img, x, y, w, h):
    """
    Crops the given image based on relative (normalized) coordinates.
    
    x, y, w, h are normalized values (0-1), so they must be converted to pixel values.
    """
    width, height = img.size
    x_px = int(x * width)
    y_px = int(y * height)
    w_px = int(w * width)
    h_px = int(h * height)

    # Crop image
    cropped_img = img.crop((x_px, y_px, x_px + w_px, y_px + h_px))
    
    return cropped_img


def draw_bounding_box_on_image(img, x, y, w, h):
    """
    Draws a bounding box on the image at the specified normalized coordinates.
    Returns the full image with bounding box drawn.
    
    x, y, w, h are normalized values (0-1), so they must be converted to pixel values.
    """
    width, height = img.size
    x_px = int(x * width)
    y_px = int(y * height)
    w_px = int(w * width)
    h_px = int(h * height)

    # Convert PIL image to OpenCV format
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    
    # Draw bounding box
    cv2.rectangle(img_cv, (x_px, y_px), (x_px + w_px, y_px + h_px), (0, 255, 0), 2)
    
    # Convert back to PIL
    img_with_bbox = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))

    return img_with_bbox



def merge_cropped_image_back(original_img, processed_cropped, x, y, w, h):
    """
    Merges the processed cropped image back into the original image.
    Ensures that the processed cropped image has the exact dimensions as the original cropped area.
    """
    width, height = original_img.size
    x_px = int(x * width)
    y_px = int(y * height)
    w_px = int(w * width)
    h_px = int(h * height)

    # Resize processed_cropped to ensure it matches the expected cropped size
    processed_cropped_resized = processed_cropped.resize((w_px, h_px), Image.LANCZOS)

    # Convert images to NumPy arrays
    original_array = np.array(original_img)
    processed_array = np.array(processed_cropped_resized)

    # Check if dimensions match before merging
    if processed_array.shape[:2] != (h_px, w_px):
        raise ValueError(f"Shape mismatch after resizing: expected ({h_px},{w_px}), got {processed_array.shape[:2]}")

    # Merge processed region back into the original image
    original_array[y_px:y_px+h_px, x_px:x_px+w_px] = processed_array

    # Convert back to PIL Image
    merged_img = Image.fromarray(original_array)

    return merged_img 