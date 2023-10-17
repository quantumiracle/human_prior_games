import gym
from gym import spaces
from ple import PLE
import numpy as np
import os

def downsample_image(image_matrix, output_size):
    # get the original size of the image matrix
    original_size = image_matrix.shape[:2]  # (H, W, C)
    
    # calculate the ratio to scale the image down to the desired size
    ratio = [float(output_size[i])/original_size[i] for i in (0, 1)]
    
    # calculate the new size of the image matrix
    new_size = tuple(int(original_size[i] * ratio[i]) for i in (0, 1))
    
    # create an empty output image matrix with the desired size and number of channels
    output_matrix = np.zeros(output_size + (image_matrix.shape[2],), dtype=image_matrix.dtype)
    
    # downsample the image matrix by averaging over each block of pixels
    for i in range(output_size[0]):
        for j in range(output_size[1]):
            row_start = int(i/ratio[0])
            row_end = int((i+1)/ratio[0])
            col_start = int(j/ratio[1])
            col_end = int((j+1)/ratio[1])
            block = image_matrix[row_start:row_end, col_start:col_end, :]
            output_matrix[i, j, :] = np.mean(block, axis=(0, 1))
    
    return output_matrix

class PLEEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, game_name, display_screen=True, downsample_size=None):
        # set headless mode
        os.environ['SDL_VIDEODRIVER'] = 'dummy'
        # open up a game state to communicate with emulator
        import importlib
        game_module_name = ('ple.games.%s' % game_name).lower()
        game_module = importlib.import_module(game_module_name)
        game = getattr(game_module, game_name)()
        self.game_state = PLE(game, fps=30, display_screen=display_screen)
        self.game_state.init()
        self._action_set = self.game_state.getActionSet()
        self.action_space = spaces.Discrete(len(self._action_set))
        self.screen_width, self.screen_height = self.game_state.getScreenDims()
        if downsample_size is not None:
            self.observation_space = spaces.Box(low=0, high=255, shape=(downsample_size[0], downsample_size[1], 3))
        else:
            self.observation_space = spaces.Box(low=0, high=255, shape=(self.screen_width, self.screen_height, 3))
        self.viewer = None
        self.count = 0
        self.downsample_size = downsample_size

    def step(self, a):
        reward = self.game_state.act(self._action_set[a])
        state = self._get_image()
        #import scipy.misc
        #scipy.misc.imsave('outfile'+str(self.count)+'.jpg', state)
        #self.count = self.count+1
        terminal = self.game_state.game_over()
        if self.downsample_size is not None:
            state = downsample_image(state, output_size=self.downsample_size)
        return state, reward, terminal, {}

    def _get_image(self):
        #image_rotated = self.game_state.getScreenRGB()
        image_rotated = np.fliplr(np.rot90(self.game_state.getScreenRGB(),3)) # Hack to fix the rotated image returned by ple
        return image_rotated

    @property
    def _n_actions(self):
        return len(self._action_set)

    # return: (states, observations)
    def reset(self):
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.screen_width, self.screen_height, 3))
        self.game_state.reset_game()
        state = self._get_image()
        if self.downsample_size is not None:
            state = downsample_image(state, output_size=self.downsample_size)
        return state

    def render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return
        img = self._get_image()
        if self.downsample_size is not None:
            # downsample the image to the desired size
            img = downsample_image(img, output_size=self.downsample_size)

        if mode == 'rgb_array':
            return img
        elif mode == 'human':
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer(maxwidth=500)

            self.viewer.imshow(img)

    def seed(self, seed):
        rng = np.random.RandomState(seed)
        self.game_state.rng = rng
        self.game_state.game.rng = self.game_state.rng

        self.game_state.init()


if __name__ == "__main__":
    env = PLEEnv('nosemantics', downsample_size=(14, 14))
    for _ in range(3):
        env.reset()
        for i in range(1000):
            env.render()
            action = env.action_space.sample()
            print(action)
            obs, rew, done, _ = env.step(action)
            print(obs.shape, rew, done)
            if done:
                break

    env.close()
