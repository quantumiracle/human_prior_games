#!/usr/bin/python


"""

This tests that all the PLE games launch, except for doom; we
explicitly check that it isn't defined.


"""
import numpy as np
# from ple.games.originalGame import originalGame
from ple import originalGame, nosemantics, nosimilarity, noobject, noaffordance, continualgame, smallGame
from ple import PLE

def run():
    # game = originalGame()
    game = smallGame()
    # game = nosimilarity()
    # game = continualgame()

    p = PLE(game, fps=30, display_screen=True, force_fps=False)
    p.init()

    NUM_STEPS=150

    class NaiveAgent():
        def __init__(self, actions):
            self.actions = actions
        def pickAction(self, reward, obs):
            return self.actions[np.random.randint(0, len(self.actions))]

    myAgent = NaiveAgent(p.getActionSet())

    nb_frames = 1000
    reward = 0.0

    for f in range(nb_frames):
        if p.game_over(): #check if the game is over
            p.reset_game()

        obs = p.getScreenRGB()
        action = myAgent.pickAction(reward, obs)
        reward = p.act(action)

if __name__ == "__main__":
    run()
