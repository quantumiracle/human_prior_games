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

class ClipBase:
    def __init__(self, model_name='ViT-B-32'):
        """
        Initializes the CLIP model with the specified parameters.
        """
        # https://huggingface.co/models?library=open_clip
        pretrained_dict = {
            'RN50': 'openai', # size?
            # 'ViT-B-32': 'openai',  # 256x256
            # 'ViT-B-16': 'openai',  # 224x224
            # 'ViT-L-14': 'openai',  # 224x224
            'ViT-L-14-336': 'openai', # 336x336

            'ViT-B-32': 'laion2b_s34b_b79k',  # 224x224
            'ViT-B-16': 'laion2b_s34b_b88k',  # 224x224
            'ViT-H-14': 'laion2b_s32b_b79k',  # 224x224
            'ViT-L-14': 'laion2b_s32b_b82k',  # 224x224
            'ViT-bigG-14': 'laion2b_s39b_b160k', # 224x224
        }
        # a full list, see: https://colab.research.google.com/github/mlfoundations/open_clip/blob/master/docs/Interacting_with_open_clip.ipynb#scrollTo=uLFS29hnhlY4
        # pretrained_dict = {key: value for (key, value) in open_clip.list_pretrained()}

        pretrained_weights = pretrained_dict[model_name]

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained_weights
        )

        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model.to(self.device)

        if model_name == 'ViT-L-14-336':
            self.desired_width = 336
            self.desired_height = 336
        else:
            self.desired_width = 224
            self.desired_height = 224

        # https://github.com/mlfoundations/open_clip/blob/main/docs/model_profile.csv
        if model_name in ['ViT-B-32', 'ViT-B-16']:
            self.embed_dim = 512
        elif model_name in ['ViT-L-14', 'ViT-L-14-336']:
            self.embed_dim = 768
        elif model_name in ['ViT-H-14', 'RN50']:
            self.embed_dim = 1024
        elif model_name in ['ViT-bigG-14']:
            self.embed_dim = 1280
        

    def _reshape_image(self, image):
        resized_image = image.resize((self.desired_width, self.desired_height))
        return resized_image

class ClipReward(ClipBase):
    def __init__(self, model_name='ViT-B-32'):
        super(ClipReward, self).__init__(model_name)

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
        if isinstance(image, list):   # batch input
            image_inputs = [self.preprocess(self._reshape_image(Image.open(im) if isinstance(im, str) else Image.fromarray(im))).unsqueeze(0) for im in image]
            image_input = torch.cat(image_inputs, dim=0).to(self.device)
        elif isinstance(image, str):
            image = Image.open(image)
            image_input = self.preprocess(self._reshape_image(image)).unsqueeze(0).to(self.device)
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
            image_input = self.preprocess(self._reshape_image(image)).unsqueeze(0).to(self.device)

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


class ClipEncoder(ClipBase):
    def __init__(self, model_name='ViT-B-32'):
        super(ClipEncoder, self).__init__(model_name)

    def encode(self, image):
        """
        Encode an image to a vector.

        Parameters:
        image (str/img): The file path to the image.

        Returns:
        numpy.ndarray: The encoded image.
        """
        # Preprocess the image
        if isinstance(image, list):  # batch input
            image_inputs = [self.preprocess(self._reshape_image(Image.open(im) if isinstance(im, str) else Image.fromarray(im))).unsqueeze(0) for im in image]
            image_input = torch.cat(image_inputs, dim=0).to(self.device)
        if isinstance(image, str):
            image = Image.open(image)
            image_input = self.preprocess(self._reshape_image(image)).unsqueeze(0).to(self.device)
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
            image_input = self.preprocess(self._reshape_image(image)).unsqueeze(0).to(self.device)

        with torch.no_grad(), torch.cuda.amp.autocast():
            # Encode image
            image_features = self.model.encode_image(image_input)

            # Normalize features
            image_features /= image_features.norm(dim=-1, keepdim=True)

        # Return the encoded image as a numpy array
        return image_features.cpu().numpy()


if __name__ == "__main__":
    clip_reward = ClipReward()
    image_path = "data/gym/1.png"
    question = "What is in this image?"
    probs = clip_reward.get_reward(image_path, question)
    print("reward:", probs)