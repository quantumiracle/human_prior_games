import argparse
import torch
import os
import io

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from PIL import Image

import requests
from PIL import Image
from io import BytesIO
from transformers import TextStreamer
from common import input_fifo_name, output_fifo_name, split_token
from collections import deque
from load_prompt import load_prompt


def add_prompt(image_list, query, prompt_data, image_processor):
    """
    prompt: [(image, text), (image, text), ...]
    """
    prompt_images = []
    prompt_queries = ""
    for image, text in prompt_data:
        image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'].half().cuda()
        prompt_images.append(image_tensor)
        prompt_queries += text

    image_list = prompt_images + image_list
    query = prompt_queries + query

    return image_list, query


def main(args):
    # Model
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)

    def initialize_conv():
        if 'llama-2' in model_name.lower():
            conv_mode = "llava_llama_2"
        elif "v1" in model_name.lower():
            conv_mode = "llava_v1"
        elif "mpt" in model_name.lower():
            conv_mode = "mpt"
        else:
            conv_mode = "llava_v0"

        if args.conv_mode is not None and conv_mode != args.conv_mode:
            print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(conv_mode, args.conv_mode, args.conv_mode))
        else:
            args.conv_mode = conv_mode

        conv = conv_templates[args.conv_mode].copy()
        if "mpt" in model_name.lower():
            roles = ('user', 'assistant')
        else:
            roles = conv.roles

        return conv, roles

    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, args.load_8bit, args.load_4bit)

    # Ensure the pipe exists before reading from it
    if not os.path.exists(input_fifo_name):
        os.mkfifo(input_fifo_name)
    if not os.path.exists(output_fifo_name):
        os.mkfifo(output_fifo_name)

    max_conv_history = 2 # length of conversation history in numbers of QA pairs
    add_prompt_flag = True  # add valid examples
    image_queue = deque(maxlen=max_conv_history)  # including current

    if add_prompt_flag:
        folder = 'llm_query/prompts'
        prompt_data = load_prompt(1,2, image_dir=folder, text_dir=folder)
        # print('prompt_data: ', prompt_data)

    # initialize the conversation
    conv, roles = initialize_conv()

    while True:    
        # Read from pipe
        with open(input_fifo_name, 'rb') as f:
            message = f.read()
            # Separate the string and image bytes
            splitted_msg = message.split(split_token.encode('utf-8')) 
            question_str = splitted_msg[0].decode('utf-8')
            img_str = splitted_msg[1]
            byte_io = io.BytesIO(img_str)
            image = Image.open(byte_io)
            # image.show()

        image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'].half().cuda()
        image_queue.append(image_tensor)
        
        # while True:    
        for num_query in range(1):
            if len(question_str) > 0:
                inp = question_str
            else:
                try:
                    inp = input(f"{roles[0]}: ")
                except EOFError:
                    inp = ""
                if not inp:
                    print("exit...")
                    break

            print(f"{roles[1]}: ", end="")

            if image is not None:
                # first message
                if model.config.mm_use_im_start_end:
                    inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
                else:
                    inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
                
                image = None  # no new image for more than one query

            conv.append_message(conv.roles[0], inp)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_images = list(image_queue)

            if add_prompt_flag:
                split_idx = prompt.find('USER:')
                prefix_prompt = prompt[:split_idx]  # remain unchanged
                suffix_prompt = prompt[split_idx:]  # add given prompt examples
                input_images, suffix_prompt = add_prompt(input_images, suffix_prompt, prompt_data, image_processor)
                prompt = prefix_prompt + suffix_prompt

            print('\n************\n conv: ', conv)
            # print('\n************\n prompt: ', prompt)
            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
            streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
            # print('input: ', input_ids.shape)
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=input_images,
                    do_sample=True,
                    temperature=0.2,
                    max_new_tokens=1024,
                    streamer=streamer,
                    use_cache=True,
                    stopping_criteria=[stopping_criteria])
            # print('output: ', output_ids.shape)

            outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
            conv.messages[-1][-1] = outputs  # fill assistent's reply in messages: [['User', 'xxx'], ['Assistant', 'xxx'], ...]

            if len(image_queue) >= max_conv_history:
                conv.messages.pop(0)   # user
                conv.messages.pop(0)   # assistant
                image_queue.pop()

            # Write to pipe
            with open(output_fifo_name, 'w') as f:
                f.write(outputs)

            if args.debug:
                print("\n", {"prompt": prompt, "outputs": outputs}, "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    # parser.add_argument("--image-file", type=str, required=True)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    main(args)
