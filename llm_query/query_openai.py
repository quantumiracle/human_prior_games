import base64
import requests
from openai_key import api_key

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Path to your image
image_path = "test.png"

# Getting the base64 string
base64_image = encode_image(image_path)

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
}

question = "Context: After taking action from last step. This is the current frame of the video game. The agent to control is a white and gray character, \
    its action choice: 0 for left, 1 for right, 2 for jump, 3 for climb up. \
    Goal: the goal is to reach the princess character without death. \
    Condition: a. If the agent touches the fire it will die. \
    b. If the agent touches the purple enemy it will die.\
    What is the location of the agent, and how should it move now? \n\
    Reply in this format: action value, only the number. Then describe the current scenen and give reason for the chosen action,\
        based on the current location of the agent and its relative location to the princess. \n\
    For example: \
    2: The agent should jump because [To Fill]. The location of the agent is [To Fill]. It's getting [Choose one: closer/farther] from the goal.\
    3: The agent should climb up because [To Fill]. The location of the agent is [To Fill]. It's getting [Choose one: closer/farther] from the goal. \
    0: The agent should move left because [To Fill]. The location of the agent is [To Fill]. It's getting [Choose one: closer/farther] from the goal.\
    1: The agent should move right because [To Fill]. The location of the agent is [To Fill]. It's getting [Choose one: closer/farther] from the goal."     

payload = {
    "model": "gpt-4-vision-preview",
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": "What’s in this image?"
          },
          {
            "type": "image_url",
            "image_url": {
              "url": f"data:image/jpeg;base64,{base64_image}"
            }
          }
        ]
      }
    ],
    "max_tokens": 300
}

response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

print(response.json())