import base64
import requests
from private import openai_api_key

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Path to your image
image_path = "../data/maze_image1.png"

# Getting the base64 string
base64_image = encode_image(image_path)

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {openai_api_key}",
    # "OpenAI-Organization": "org-ASVyDkhQ8rJjEQlABoHSMg07",  # personal
}

question = "Context: After taking action from last step. This is the current frame of the video game. The agent to control is the one wearing white outfit with a helmet, \
    its action choice: 0 for left, 1 for right, 2 for jump, 3 for climb up. \
    Goal: the goal is to reach the princess character without death. \
    Condition: a. If the agent touches the fire it will die. \
    b. If the agent touches the purple enemy it will die.\
    What is the location of the agent, and how should it move now? \n\
    Reply in this format: action value, only the number. Then describe the current scenen and give reason for the chosen action,\
        based on the current location of the agent and its relative location to the princess. \n\
    For example: \
    0: The agent should move left because [To Fill]. The location of the agent is [To Fill]. It's getting [Choose one: closer/farther] from the goal.\
    1: The agent should move right because [To Fill]. The location of the agent is [To Fill]. It's getting [Choose one: closer/farther] from the goal. \
    2: The agent should jump because [To Fill]. The location of the agent is [To Fill]. It's getting [Choose one: closer/farther] from the goal.\
    3: The agent should climb up because [To Fill]. The location of the agent is [To Fill]. It's getting [Choose one: closer/farther] from the goal."     


question = "This is the current frame of the video game. Describe in details about the location of all the characters and potentially useful objects in the game. Do not guess the controllable character and be careful of the number of objects."

image_description = "You're looking at a screenshot from a classic-style platform video game. Here's the detailed description of the scene:\n\n- In the top left corner, there is a character standing on a brick platform. This character has orange hair and is wearing a blue outfit.\n- Towards the middle right of the image, there are two ladders stretching vertically. The ladders connect the lower level and the upper level.\n- On the upper level, where the first character is located, there is also a box with a magenta and black question mark on it. This box typically indicates that it could contain an item or a power-up. It's positioned on the platform to the right.\n- On the lower level, directly below the two ladders, there is another character. This character appears blue and is facing to the left, standing on the ground level.\n- At the bottom right, there's a trio of flames closely grouped together, which could represent an obstacle or an enemy. These flames are orange with a touch of red, and they are stationary in this frame.\n\nAll characters and objects are enclosed within the brick-wall boundaries of the game level, which has openings on the upper platforms possibly allowing movement to other areas of the level. The visual style is reminiscent of early platform games from the 80s or early 90s, with pixel art graphics."
frames_description = "These images depict a sequence of frames from a 2D platform-style video game that has a distinctly retro, pixel-art aesthetic. The game scene appears to be set inside what could be a virtual castle or dungeon with a dark background.\
In each frame, you have:\
1. The character at the top left: A character with orange hair, wearing a blue outfit, is stationed on a block at the upper-left corner of the scene throughout all the frames. This character seems to be a static element in the scenario or possibly an objective for the player to reach.\
2. A movable character: There is a character with a white helmet and gray suit, likely the player-controlled avatar, moving across the platforms. The position of this character changes throughout the frames, indicating motion, as the character climbs up and down ladders and navigates through the platform level.\
3. Hazardous obstacles: At the bottom of the scene, there are fires burning, which serve as dangerous obstacles that the player must avoid. These fires remain in the same location in each frame, directly beneath the movable character's starting position.\
4. A ladder system: Two large ladders connect the different levels of platforms. The movable character interacts with these ladders, showcasing the climbing mechanics provided within the game for vertical movement.\
5. A purple block: This block, which is purple with a face, could represent either an in-game enemy, obstacle, or a special item. It does not move in the series of frames provided and seems to be another static element of the level.\
The goal of the player in this game might be to navigate through the level, potentially avoiding or overcoming obstacles and enemies to reach the character at the top left. The sequence of images does not provide enough information to understand whether there are any additional mechanics or objectives at play."

question = f"Context: After taking action from last step. This is the current frame of the video game, with a detailed description: {frames_description}. The agent to control is the one wearing white outfit with a helmet, \
    its action choice: 0 for left, 1 for right, 2 for jump, 3 for climb up. \
    Goal: the goal is to reach the princess character without death. \
    Please break the goal into several sub-goals for the agent in a time sequence succinctly. "

payload = {
    "model": "gpt-4-vision-preview",
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            # "text": "What’s in this image?"
            "text": question
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
    "max_tokens": 400
}

response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

print(response.json())

