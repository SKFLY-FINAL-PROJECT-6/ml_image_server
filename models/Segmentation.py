import torch
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation
import numpy as np
import cv2

class SegmentationModel:
    def __init__(self, model_name, device):
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = SegformerForSemanticSegmentation.from_pretrained(model_name)
        self.model.eval()
        
    def segment(self, image):
        inputs = self.processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
        segmentation = self.processor.post_process_semantic_segmentation(
            outputs=outputs,
            target_sizes=[image.size[::-1]]
        )[0]
        return segmentation

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
        return binary_mask, binary_mask_255

  