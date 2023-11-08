# import functions from parent llm_query
from llm_query.query_llava import load_image, query_llm

# load string from cartpole_goal.txt
with open('gym/cartpole_goal.txt', 'r') as f:
    question = f.read()

# iteratively query llm for png images under folder data/gym
image_dir = 'data/gym'
image_list = []
for i in range(1, 5):
    image_file = f"{image_dir}/{i}.png"
    image = load_image(image_file)
    image_list.append(image)
    output_str = query_llm(image, question)
    print(f"output_str: {output_str}")