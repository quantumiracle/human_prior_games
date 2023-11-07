#!/usr/bin/python


"""

This tests that all the PLE games launch, except for doom; we
explicitly check that it isn't defined.


"""
import numpy as np
import matplotlib.pyplot as plt
from ple import originalGame, nosemantics, nosimilarity, noobject, noaffordance, continualgame, smallGame
from ple import PLE
from humanRL_gym_ple.ple_env import PLEEnv
from PIL import Image
from llm_query.query import query_llm
from llm_query.common import *

import os

general_strategy_gpt4 = "Some general strategies: Start by observing the entire game frame to identify the locations of the fire, the purple enemy, ladders, and the princess. Strategize an initial path that seems the safest, keeping in mind that it may need to be adjusted based on the dynamic elements of the game. Begin by moving towards the first ladder in the agent's path, ensuring that no immediate threats (fire or purple enemy) are in the way. Use the 'climb up' action to ascend the ladder. If there's a ladder leading downwards and it's safer, consider using it. As the agent moves and climbs, constantly be aware of the fire's location. If the fire blocks the path, wait for a safe gap or look for an alternative route. If the fire moves, adjust the agent's strategy to ensure it maintains a safe distance. After avoiding the fire, or while doing so, be cautious of the purple enemy. Observe its pattern of movement. If it's stationary, plan a path around it. If it moves, time the agent's movement to bypass it when it's safest. Once the immediate threats have been navigated and the agent is in a safe zone, move towards the princess. Continuously adjust the agent's path as it approaches the princess, ensuring that it doesn't inadvertently come into contact with the fire or the purple enemy. Throughout this sequence, the agent needs to be adaptable. Even if it has a planned route, the dynamic nature of the game might necessitate adjustments. Constant observation and reaction to the game environment will be key in chaining these sub-goals effectively to achieve the primary goal."

def get_llm_feedback(image, ):
    image = Image.fromarray(image)
    # question = 'Describe the image contents in details.'
    # question = "Context: After taking action from last step. This is the current frame of the video game. The agent to control is a white and gray character, \
    #     its action choice: 0 for left, 1 for right, 2 for jump, 3 for climb up. \
    #     Goal: the goal is to reach the princess character without death. \
    #     Condition: a. If the agent touches the fire it will die. \
    #     b. If the agent touches the purple enemy it will die.\
    #     How should it move now? \n\
    #     Reply in this format: action value, only the number. Then describe the current scenen and give reason for the chosen action,\
    #         based on the current location of the agent and its relative location to the princess. \n\
    #     For example: \
    #     2: The agent should jump because xxx. \
    #     3: The agent should climb up because xxx. \
    #     0: The agent should move left because xxx. \
    #     1: The agent should move right because xxx." 
    
    # more information
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

    # chatgpt suggested
    # question = "Context: The current frame of the video game after the last action. \
    #     The agent is the white and gray character. Actions: 0 = left, 1 = right, 2 = jump, 3 = climb up. \
    #     Goal: Reach the princess without dying. Hazards: a. Fire = death, b. Purple enemy = death.\n\
    #     How should the agent move? Reply with the action number and rationale.\n\
    #     For example: \
    #     3: Climb up along the ladder to get closer to the goal.\
    #     2: Jump to avoid an obstacle."

    answer = query_llm(image, question)
    print(f'Answer: {answer}')
    return answer

def run(render=False):
    # game = originalGame()
    # game = smallGame()
    # game = nosimilarity()
    # game = continualgame()

    # p = PLE(game, fps=30, display_screen=True, force_fps=False)
    # p.init()

    # NUM_STEPS=150

    # class NaiveAgent():
    #     def __init__(self, actions):
    #         self.actions = actions
    #     def pickAction(self, reward, obs):
    #         return self.actions[np.random.randint(0, len(self.actions))]

    # myAgent = NaiveAgent(p.getActionSet())

    # nb_frames = 1000
    # reward = 0.0

    # for f in range(nb_frames):
    #     if p.game_over(): #check if the game is over
    #         p.reset_game()

    #     obs = p.getScreenRGB()
    #     action = myAgent.pickAction(reward, obs)
    #     reward = p.act(action)

    # env = PLEEnv('smallGame', display_screen=True, fps=30, downsample_size=(14, 14))
    game_name = 'smallGame'
    env = PLEEnv(game_name, display_screen=True, fps=30, downsample_size=None)
    query_freq = 3
    log_dir = f'./data/{game_name}/'
    os.makedirs(log_dir, exist_ok=True)
    fifo_stat = os.stat(input_fifo_name)
    print(fifo_stat)
    # Print some status information
    print(f"Size: {fifo_stat.st_size} bytes")
    print(f"Permissions: {oct(fifo_stat.st_mode & 0o777)}")
    print(f"Last modified: {fifo_stat.st_mtime}")


    for _ in range(1):
        env.reset()
        llm_action = None
        for i in range(1000):
            if (i-1) % query_freq == 0:  # skip first black frame
                frame = env.render(mode='rgb_array')
                answer = get_llm_feedback(frame)
                try:
                    llm_action = int(answer[0])
                except:
                    print('Invalid action: ', llm_action)
                    llm_action = None
                # check if the action is valid: integer from 0 to 3
                if llm_action not in range(4):
                    print('Invalid action')
                    llm_action = None
                    # continue

                with open(f'{log_dir}/llm_feedbacks.txt', 'a') as f:
                    f.write(f'{i}: {answer}\n')
                # save image
                image = Image.fromarray(frame)
                image.save(f'{log_dir}/frame{i}.png', )
                
                if render:
                    plt.imshow(frame)
                    plt.axis('off')  # Hide axes for better visualization
                    plt.show()
                    # print(frame.shape)
                print("====================")
            if llm_action is None:
                action = env.action_space.sample()  # random action
            else:
                action = llm_action

            print('action: ', action)
            obs, rew, done, _ = env.step(action)
            # print(obs.shape, rew, done)
            print('reward:', rew)
            if done:
                print('episode done')
                break

    env.close()

if __name__ == "__main__":
    run()
