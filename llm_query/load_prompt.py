from PIL import Image

def load_prompt(start_idx, end_idx, image_dir='.', text_dir='.'):
    data = []

    for i in range(start_idx, end_idx + 1):
        image_filename = f"{image_dir}/maze_image{i}.png"
        text_filename = f"{text_dir}/maze_prompt{i}.txt"

        try:
            image = Image.open(image_filename).convert('RGB')
        except FileNotFoundError:
            print(f"Image file {image_filename} not found.")
            image = None

        try:
            with open(text_filename, 'r') as file:
                text = file.read()
        except FileNotFoundError:
            print(f"Text file {text_filename} not found.")
            text = None

        data.append((image, text))
    print('Loaded prompt examples: ', len(data))

    return data

# loaded_data = load_prompt(1, 2, image_dir='prompts', text_dir='prompts')
# import pdb; pdb.set_trace()