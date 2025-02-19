import streamlit as st
from PIL import Image
import torch
from models.vision.StableDiffusion import StableDiffusionModel
from models.vision.Segmentation import SegmentationModel
from utils.image_utils import apply_canny, paste_largest_segment, reduce_image_size


import warnings
warnings.filterwarnings("ignore")

# Check CUDA availability
cuda = 'cuda' if torch.cuda.is_available() else 'cpu'

def main():
    st.title("Wall Painting Generator")
    
    # File uploaders
    image_file = st.file_uploader("Upload Room Image", type=['png', 'jpg', 'jpeg'])
    scribble_file = st.file_uploader("Upload Scribble Image (Optional)", type=['png', 'jpg', 'jpeg'])
    
    # Text prompt
    text_prompt = st.text_input("Enter your painting description", 
                               "A beautiful painting of a sunset on a beach")
    
    if image_file and text_prompt:
        # Load models
        with st.spinner('Loading models...'):
            seg_model = SegmentationModel("nvidia/segformer-b4-finetuned-ade-512-512", cuda)
            diffuser_model = StableDiffusionModel(
                controlnet_model_name="lllyasviel/control_v11p_sd15_scribble",
                sd_model_name="runwayml/stable-diffusion-v1-5",
                device=cuda
            )
        
        # Process images
        image = Image.open(image_file)
        image = reduce_image_size(image_file, 50)  # Reduce image size by 50%
        scribble = Image.open(scribble_file) if scribble_file else None
        if scribble:
            scribble = reduce_image_size(scribble_file, 50)  # Reduce scribble size by 50%
        
        # Display original images
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Wall Image")
            st.image(image)
        if scribble:
            with col2:
                st.subheader("Scribble")
                st.image(scribble)
        
        if st.button("Generate Painting"):
            with st.spinner('Generating wall segmentation...'):
                # Get wall segmentation
                segmentation = seg_model.segment(image)
                binary_mask_255 = seg_model.get_target_binary_mask(segmentation, target_class_name='wall')
                binary_mask_to_canny_edge = apply_canny(binary_mask_255, 100, 200)
            # Display segmentation and edge detection results
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Wall Segmentation")
                st.image(binary_mask_255, caption="Wall Mask")
            with col2:
                st.subheader("Edge Detection")
                st.image(binary_mask_to_canny_edge, caption="Edge Detection Result")
            
            with st.spinner('Generating painting from your prompt...'):
                # Generate painting
                # Add secret prompt to enhance wall painting generation
                hidden_prompt = f"flat design, bright and friendly mural-style illustration, children's artwork style, {text_prompt}"
                text_prompt = hidden_prompt

                painting = diffuser_model.generate_painting(binary_mask_to_canny_edge, text_prompt, image)
                st.subheader("Generated Raw Painting")
                st.image(painting)
                # Cut and paste the painting onto the wall using the mask
                result = paste_largest_segment(painting, binary_mask_255, image)
            
            # Display result
            st.subheader("Generated Painting")
            st.image(result)

if __name__ == "__main__":
    main()
