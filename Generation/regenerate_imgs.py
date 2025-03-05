
import torch
import clip
import open_clip
from torch.nn import functional as F
from tqdm import tqdm
from PIL import Image
import numpy as np
import random
import pandas as pd
import os, argparse

from custom_pipeline_phil import Generator4Embeds
from plotly.subplots import make_subplots
import plotly.graph_objects as go

os.environ['HF_HOME'] = '/work3/s184984/repos/EEG_Image_decode/Generation/huggingface/cache/hub'

def set_seed(seed):
    # Set the seed for Python's built-in random module
    random.seed(seed)
    
    # Set the seed for numpy (used in many data science libraries)
    np.random.seed(seed)
    
    # Set the seed for PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # If you use CUDA, this ensures all GPU seeds are set

from PIL import Image

def load_and_correct_image(image_path):
    """
    Loads an image, corrects its orientation using EXIF data, converts to RGB format, 
    and center crops it to the largest possible square.
    
    Args:
        image_path (str): Path to the image file.
    
    Returns:
        PIL.Image.Image: A corrected and square-cropped PIL Image object.
    """
    with Image.open(image_path) as image:
        # Correct orientation using EXIF metadata if available
        if hasattr(image, "_getexif") and image._getexif():
            exif = image._getexif()
            orientation_key = 274  # Key for orientation tag
            if exif and orientation_key in exif:
                orientation = exif[orientation_key]
                # Apply transformations based on orientation value
                if orientation == 3:
                    image = image.rotate(180, expand=True)
                elif orientation == 6:
                    image = image.rotate(270, expand=True)
                elif orientation == 8:
                    image = image.rotate(90, expand=True)
        
        # Convert to RGB to ensure consistent processing
        image = image.convert("RGB")
        
        # Perform center square cropping
        width, height = image.size
        min_dim = min(width, height)
        left = (width - min_dim) // 2
        top = (height - min_dim) // 2
        right = (width + min_dim) // 2
        bottom = (height + min_dim) // 2
        image = image.crop((left, top, right, bottom))
    
    return image



# Defining CLIP image and text encoders for use
def Textencoder(text):   
    # Initialize tqdm progress bar
    text_inputs = torch.cat([clip.tokenize(t) for t in tqdm(text, desc="Encoding Text")]).to(device)

    with torch.no_grad():
        text_features = vlmodel.encode_text(text_inputs)
    
    # text_features = F.normalize(text_features, dim=-1).detach()
    
    return text_features

def ImageEncoder(images):
    batch_size = 1
    image_features_list = []
    
    # Initialize tqdm progress bar
    for i in tqdm(range(0, len(images), batch_size), desc="Encoding Images"):
        batch_images = images[i:i + batch_size]
        image_inputs = torch.stack([preprocess_train(load_and_correct_image(img)) for img in batch_images]).to(device)

        with torch.no_grad():
            batch_image_features = vlmodel.encode_image(image_inputs)
            # batch_image_features /= batch_image_features.norm(dim=-1, keepdim=True)   #remove code when reconstruction 

        image_features_list.append(batch_image_features)

    image_features = torch.cat(image_features_list, dim=0)

    return image_features


# Initialization

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='EEG Transformer Training Script')
    parser.add_argument('--seed', type=int, default=42, help='Number of epochs')
    args = parser.parse_args()

    # Example usage
    set_seed(args.seed)
    # Loading CLIP image encoder
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # vlmodel, preprocess = clip.load("ViT-B/32", device=device)
    model_type = 'ViT-H-14'
    vlmodel, preprocess_train, feature_extractor = open_clip.create_model_and_transforms(
        model_type, pretrained='laion2b_s32b_b79k', precision='fp32', device=device)
    
    generator = Generator4Embeds(num_inference_steps=4, device=device, force_download=True)

    # Start generation
    originals_folder = "/work3/s184984/repos/EEG_Image_decode/Generation/manual_CLIP_generation/originals2"
    for filename in os.listdir(originals_folder):
        image_path = os.path.join(originals_folder, filename)

        img_emb = ImageEncoder([image_path])

        # Initialize Model
        generated_image = generator.generate(image_embeds = img_emb, text_prompt="")

        try:
            original_img = load_and_correct_image(image_path)  # Convert to RGB for consistency
            img_np = np.array(original_img)  # Convert to numpy array for matplotlib
            print(f"Original image loaded with shape: {img_np.shape}")
        except Exception as e:
            print(f"Error loading original image from path {image_path}: {e}")
            continue  # Skip this batch if image loading fails

        # Convert generated image to numpy if it's a PIL Image
        try:
            if isinstance(generated_image, Image.Image):
                gen_img_np = np.array(generated_image)
                print(f"Generated image loaded with shape: {gen_img_np.shape}")
            else:
                raise TypeError("Generated image is not a PIL Image.")
        except Exception as e:
            print(f"Error processing generated image: {e}")
            continue   

        # Create a 1x3 subplot grid
        fig = make_subplots(rows=1, cols=2, column_widths=[0.5, 0.5], specs=[[{"type": "image"}, {"type": "image"}]])

        # Original Image in the first column
        fig.add_trace(go.Image(z=img_np), row=1, col=1)
        fig.update_xaxes(visible=False, showticklabels=False, row=1, col=1)
        fig.update_yaxes(visible=False, showticklabels=False, row=1, col=1)

        # Generated Image in the second column
        fig.add_trace(go.Image(z=gen_img_np), row=1, col=2)
        fig.update_xaxes(visible=False, showticklabels=False, row=1, col=2)
        fig.update_yaxes(visible=False, showticklabels=False, row=1, col=2)

        # Update layout for spacing, centering, and lowering the title
        fig.update_layout(
            title_text=f"CLIP Regeneration",
            title_x=0.4,          # Adjust to center above the first two images
            title_y=0.85,         # Lower title closer to images
            title_xanchor="center",
            font=dict(
                family="Editorial New Italic",  # Consistent font family
                size=14,
            ),
            width=1000,
            height=500,
            margin=dict(t=80, b=50, l=50, r=50)  # Adjust top margin for closer title
        )                

        # Save the figure as HTML or static image
        os.makedirs(f"/work3/s184984/repos/EEG_Image_decode/Generation/manual_CLIP_generation/generated/seed{args.seed}", exist_ok=True)
        fig.write_image(f"/work3/s184984/repos/EEG_Image_decode/Generation/manual_CLIP_generation/generated/seed{args.seed}/{filename}.jpg")          


