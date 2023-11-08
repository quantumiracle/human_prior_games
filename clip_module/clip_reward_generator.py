import torch
from PIL import Image
import open_clip
import numpy as np

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


class ClipReward:
    def __init__(self, model_name='ViT-B-32'):
        """
        Initializes the CLIP model with the specified parameters.
        """
        # https://huggingface.co/models?library=open_clip
        pretrained_dict = {
            'ViT-B-32': 'laion2b_s34b_b79k',
            'ViT-B-16': 'laion2b_s34b_b88k',
            'ViT-H-14': 'laion2b_s32b_b79k',
            'ViT-L-14': 'laion2b_s13b_b90k',
            'ViT-bigG-14': 'laion2b_s39b_b160k',
        }
        pretrained_weights = pretrained_dict[model_name]

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained_weights
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model.to(self.device)

    def get_reward(self, image, question, baseline=None, alpha=1.0):
        """
        Calculate the reward for a given image and a question.

        Parameters:
        image (str/img): The file path to the image.
        question (str): The text question.

        Returns:
        numpy.ndarray: The reward.
        """
        # Preprocess the image
        if isinstance(image, str):
            image_input = self.preprocess(Image.open(image)).unsqueeze(0).to(self.device)
        elif isinstance(image, np.ndarray):
            image_input = self.preprocess(Image.fromarray(image)).unsqueeze(0).to(self.device)
        else:
            image_input = self.preprocess(image).unsqueeze(0).to(self.device)

        # Tokenize the text
        text_inputs = self.tokenizer([question]).to(self.device)

        with torch.no_grad(), torch.cuda.amp.autocast():
            # Encode image and text
            image_features = self.model.encode_image(image_input)
            text_features = self.model.encode_text(text_inputs)

            # Normalize features
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            # Calculate reward
            if baseline is not None:
                baseline_inputs = self.tokenizer([baseline]).to(self.device)
                baseline_features = self.model.encode_text(baseline_inputs)  # b
                baseline_features /= baseline_features.norm(dim=-1, keepdim=True)

                proj_image = project_c_onto_a_minus_b_batch(text_features, baseline_features, image_features)
                reg_image = alpha * proj_image + (1 - alpha) * image_features
                reward = 1 - 0.5 * torch.norm(reg_image - text_features, dim=-1) ** 2
                # print(image_features.shape, text_features.shape, baseline_features.shape, proj_image.shape, reg_image.shape, reward.shape)
            else:
                reward = image_features @ text_features.T

        # Return the probabilities as a numpy array
        return reward.cpu().numpy()

if __name__ == "__main__":
    clip_reward = ClipReward()
    image_path = "data/gym/1.png"
    question = "What is in this image?"
    probs = clip_reward.get_reward(image_path, question)
    print("reward:", probs)