import cv2
import base64
import os
import requests
import json

# Make sure to set your OpenAI API key in your environment variables
# or replace os.getenv('OPENAI_API_KEY') with the string of your key.
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Read the video file and convert frames to base64
video = cv2.VideoCapture("../data/smallGame.mp4")

base64Frames = []
while video.isOpened():
    success, frame = video.read()
    if not success:
        break
    _, buffer = cv2.imencode(".jpg", frame)
    base64Frames.append(base64.b64encode(buffer).decode("utf-8"))

video.release()
print(f"{len(base64Frames)} frames read.")

# Create the prompt message
PROMPT_MESSAGES = [
    {
        "role": "user",
        "content": [
            # "These are frames from a video that I want to upload. Generate a compelling description that I can upload along with the video.",
            "These are frames from a video game. Generate a compelling description about the location of all the characters and potentially useful objects in the game.",
            *map(lambda x: {"image": x, "resize": 768}, base64Frames[0::30]),
        ],
    },
]

# Set up the headers and payload for the POST request
headers = {
    'Authorization': f'Bearer {OPENAI_API_KEY}',
    'Content-Type': 'application/json',
    'Openai-Version': '2020-11-07'
}

payload = {
    "model": "gpt-4-vision-preview",
    "messages": PROMPT_MESSAGES,
    "max_tokens": 400,
}

# Make the POST request to the OpenAI API
response = requests.post('https://api.openai.com/v1/chat/completions',
                         headers=headers,
                         data=json.dumps(payload))

# Check if the request was successful and print the result
if response.status_code == 200:
    result = response.json()
    print(result['choices'][0]['message']['content'])
else:
    print(f'Error: {response.status_code}')
    print(response.text)
