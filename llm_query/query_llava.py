import argparse
import torch
import os
import io

# from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
# from llava.conversation import conv_templates, SeparatorStyle
# from llava.model.builder import load_pretrained_model
# from llava.utils import disable_torch_init
# from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
# from transformers import TextStreamer

from PIL import Image

import requests
from PIL import Image
from io import BytesIO
from llm_query.common import input_fifo_name, output_fifo_name, split_token


def load_image(image_file):
    if image_file.startswith('http') or image_file.startswith('https'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image



def image2str(image):
    byte_io = io.BytesIO()
    image.save(byte_io, format='JPEG')  # Or use another format if you prefer
    img_str = byte_io.getvalue()

    return img_str

def open_fifo():
    if not os.path.exists(input_fifo_name):
        os.mkfifo(input_fifo_name)
    if not os.path.exists(output_fifo_name):
        os.mkfifo(output_fifo_name)

def query_llm(image, question="", separator = split_token):  # single '#' may appear in image encoding
    open_fifo()
    if isinstance(image, list):
        input_str = question.encode('utf-8')
        for img in image:
            img_str = image2str(img)
            input_str += separator.encode('utf-8') + img_str   # text # image 1 # image 2 ...

    else:    
        img_str = image2str(image)
        input_str = (question + separator).encode('utf-8') + img_str

    # Write to pipe
    with open(input_fifo_name, 'wb') as f:
        f.write(input_str)

    # Read from pipe
    with open(output_fifo_name, 'r') as f:
        output_str = f.read()

    return output_str


if __name__ == "__main__":
    # Ensure the pipe exists before reading from it
    if not os.path.exists(input_fifo_name):
        os.mkfifo(input_fifo_name)
    if not os.path.exists(output_fifo_name):
        os.mkfifo(output_fifo_name)

    parser = argparse.ArgumentParser()
    parser.add_argument("--image-file", type=str, required=True)
    parser.add_argument("--question", type=str, default='', required=False)
    args = parser.parse_args()

    image = load_image(args.image_file)
    query_llm(image, args.question)
