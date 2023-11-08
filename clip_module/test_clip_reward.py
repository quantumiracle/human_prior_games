import os
import torch
from PIL import Image
import open_clip
import argparse
import numpy as np


def project_c_onto_a_minus_b(a, b, c):  # for 1-d vectors
    # Calculate the difference vector a - b
    d = a - b
    
    # Calculate the dot product of c with d
    dot_product = torch.dot(c, d)
    
    # Calculate the magnitude squared of d
    mag_squared = torch.dot(d, d)
    
    # Calculate the scalar projection multiplier
    scalar_proj = dot_product / mag_squared
    
    # Calculate the projection vector
    projection = scalar_proj * d
    
    return projection

def project_c_onto_a_minus_b_batch(a, b, c):
    # Calculate the difference vectors a - b
    d = a - b
    
    # Reshape c and d for batched matrix multiplication
    c_reshaped = c.view(c.size(0), -1, 1) # Shape (N, d, 1)
    d_reshaped = d.view(d.size(0), 1, -1) # Shape (N, 1, d)
    
    # Calculate the dot products of c with d for each pair in the batch (N, 1, 1)
    dot_products = torch.bmm(d_reshaped, c_reshaped).view(-1, 1)
    
    # Calculate the magnitude squared of d (N, 1)
    mag_squared = torch.sum(d * d, dim=1, keepdim=True)
    
    # Calculate the scalar projection multipliers (N, 1)
    scalar_proj = dot_products / mag_squared
    
    # Calculate the projection vectors (N, d)
    projection = scalar_proj * d
    
    return projection

def test_clip_reward(env,  model_name, baseline_reg=False, alpha=1.0):
    # https://huggingface.co/models?library=open_clip
    pretrained_data_dict = {
        'ViT-B-32': 'laion2b_s34b_b79k',
        'ViT-B-16': 'laion2b_s34b_b88k',
        'ViT-H-14': 'laion2b_s32b_b79k',
        'ViT-L-14': 'laion2b_s13b_b90k',
        'ViT-bigG-14': 'laion2b_s39b_b160k',
    }
    pretrained_data = pretrained_data_dict[model_name]

    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained_data)
    tokenizer = open_clip.get_tokenizer(model_name)

    # Set the device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Text question
    question_dict = {
        'CartPole-v1': 'pole vertically upright on top of the cart',
        'Pendulum-v1': 'pendulum in the upright position',
        'MountainCar-v0': 'a car at the peak of the mountain, next to the yellow flag',
    }
    baseline_dict = {
        'CartPole-v1': 'pole and cart',
        'Pendulum-v1': 'pendulum',
        'MountainCar-v0': 'a car in the mountain',
    }
    question = question_dict[env]    
    text_inputs = tokenizer([question]).to(device)
    if baseline_reg:
        baseline = baseline_dict[env]    
        baseline_inputs = tokenizer([baseline]).to(device)

    # Folder with images
    folder_path = f'data/gym/{env}'
    image_files = sorted(
        [f for f in os.listdir(folder_path) if f.lower().endswith('.png')],
        key=lambda x: int(os.path.splitext(x)[0])
    )

    # Iterate over images and calculate probabilities
    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        image_input = preprocess(Image.open(image_path)).unsqueeze(0).to(device)

        with torch.no_grad(), torch.cuda.amp.autocast():
            # Encode image and text
            image_features = model.encode_image(image_input)  # s
            text_features = model.encode_text(text_inputs) # g
            
            # Normalize features
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            if baseline_reg:
                baseline_features = model.encode_text(baseline_inputs)  # b
                baseline_features /= baseline_features.norm(dim=-1, keepdim=True)

                proj_image = project_c_onto_a_minus_b_batch(text_features, baseline_features, image_features)
                reg_image = alpha * proj_image + (1 - alpha) * image_features
                reward = 1 - 0.5 * torch.norm(reg_image - text_features, dim=-1) ** 2
                # print(image_features.shape, text_features.shape, baseline_features.shape, proj_image.shape, reg_image.shape, reward.shape)
            else:
                # Calculate text probabilities
                # text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                reward = image_features @ text_features.T
        
        print(f"Reward for {image_file}:", reward.cpu().numpy())  # Move to CPU for numpy conversion

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default='CartPole-v1', required=True)
    parser.add_argument('--model', choices=['ViT-B-32', 'ViT-B-16', 'ViT-H-14', 'ViT-L-14', 'ViT-bigG-14'], default='ViT-B-32', help='Your move in the game.')
    parser.add_argument("--baseline-reg", action='store_true', required=False)
    args = parser.parse_args()

    test_clip_reward(args.env, args.model, args.baseline_reg)