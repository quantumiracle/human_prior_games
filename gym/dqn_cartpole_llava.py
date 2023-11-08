import os
import gym
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
import matplotlib.pyplot as plt

class RetrieveImageCallback(BaseCallback):
    """
    Callback for saving a snapshot of the render every step
    """
    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super(RetrieveImageCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.log_dir is not None:
            os.makedirs(self.log_dir, exist_ok=True)

    def _on_step(self) -> bool:
        # Retrieve training environment
        env = self.training_env.envs[0]

        # Render the image
        image = env.render()

        # Save the image
        img_name = f"{self.num_timesteps}.png"
        plt.imsave(os.path.join(self.log_dir, img_name), image)

        return True

class CustomRewardWrapper(gym.RewardWrapper):
    def reward(self, reward):
        # Modify the reward here
        modified_reward = 1
        return modified_reward

# Create log dir
log_dir = "data/gym/"
os.makedirs(log_dir, exist_ok=True)

# Create the environment
env = gym.make('CartPole-v1', render_mode='rgb_array')
# env = CustomRewardWrapper(env)
# env = make_vec_env('CartPole-v1', n_envs=1)

# Define the model using the DQN class
model = DQN('MlpPolicy', env, verbose=1, tensorboard_log=log_dir)

# Initialize the callback
callback = RetrieveImageCallback(check_freq=1, log_dir=log_dir)

# Train the model with the callback
model.learn(total_timesteps=100, callback=callback)

# Save the trained model (optional)
model.save("dqn_cartpole")

# Close the environment
env.close()
