import torch
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation
import numpy as np
import cv2

class SegmentationModel:
    def __init__(self, model_name, device, use_fast=False):
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = SegformerForSemanticSegmentation.from_pretrained(model_name)
        self.device = device
        self.model.to(device)
        self.model.eval()
        self.use_fast = use_fast
   
    def segment(self, image):
        if image.mode != 'RGB':
            image = image.convert('RGB')

        inputs = self.processor(images=image, return_tensors="pt")
        # Move inputs to same device as model
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        segmentation = self.processor.post_process_semantic_segmentation(
            outputs=outputs,
            target_sizes=[image.size[::-1]]
        )[0]
        # Move segmentation tensor back to CPU before returning
        return segmentation.cpu()

    def get_class_counts(self, segmentation):
        class_counts = {}
        for class_id in segmentation.unique():
            class_id_int = int(class_id)
            label = self.model.config.id2label.get(str(class_id_int), self.model.config.id2label.get(class_id_int))
            count = (segmentation == class_id).sum().item()
            class_counts[label] = count
        return class_counts

    def get_target_binary_mask(self, segmentation, target_class_name):
        target_class = None
        for k, v in self.model.config.id2label.items():
            if v.lower() == target_class_name.lower():
                target_class = int(k)
                break

        if target_class is None:
            raise ValueError(f"Target class '{target_class_name}' not found in the id2label mapping.")

        binary_mask = (segmentation == target_class).numpy().astype(np.uint8)
        binary_mask_255 = binary_mask * 255
        
        return binary_mask_255
    

    def get_chosen_target_binary_mask(self, segmentation, target_class_names):
        # Convert single string to list for consistent handling
        if isinstance(target_class_names, str):
            target_class_names = [target_class_names]
        
        # Initialize combined mask with zeros
        combined_mask = np.zeros_like(segmentation.numpy(), dtype=np.uint8)
        
        for target_class_name in target_class_names:
            target_class = None
            for k, v in self.model.config.id2label.items():
                if v.lower() == target_class_name.lower():
                    target_class = int(k)
                    break

            if target_class is None:
                raise ValueError(f"Target class '{target_class_name}' not found in the id2label mapping.")

            # Create mask for current class and combine with existing mask using OR operation
            class_mask = (segmentation == target_class).numpy().astype(np.uint8)
            combined_mask = cv2.bitwise_or(combined_mask, class_mask)
        
        # Convert to 255 scale
        binary_mask_255 = combined_mask * 255
        
        return binary_mask_255

  