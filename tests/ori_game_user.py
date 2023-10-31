import numpy as np
# from ple.games.originalGame import originalGame
from ple import originalGame, nosemantics, nosimilarity, noobject, noaffordance, continualgame, smallGame
from ple import PLE
import pygame  # Importing pygame for keyboard inputs

def run():
    # game = originalGame()
    game = smallGame()
    # game = nosimilarity()
    # game = continualgame()

    p = PLE(game, fps=30, display_screen=True, force_fps=False)
    p.init()

    class NaiveAgent():
        def __init__(self, actions):
            self.actions = actions

        def pickAction(self, reward, obs):
            # Capture keyboard input and return action
            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]:
                return self.actions[2]  # Assuming action 2 corresponds to JUMP
            elif keys[pygame.K_DOWN]:
                return self.actions[3]  # Assuming action 3 corresponds to UP
            elif keys[pygame.K_LEFT]:
                return self.actions[0]  # Assuming action 0 corresponds to LEFT
            elif keys[pygame.K_RIGHT]:
                return self.actions[1]  # Assuming action 1 corresponds to RIGHT
            # else:
            #     return self.actions[np.random.randint(0, len(self.actions))]  # Random action if no key is pressed

    myAgent = NaiveAgent(p.getActionSet())

    nb_frames = 1000
    reward = 0.0

    for f in range(nb_frames):
        if p.game_over():  # check if the game is over
            p.reset_game()

        obs = p.getScreenRGB()
        action = myAgent.pickAction(reward, obs)
        reward = p.act(action)

if __name__ == "__main__":
    run()
