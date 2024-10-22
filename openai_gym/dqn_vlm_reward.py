import os
import gym
# import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from clip_module.clip_reward_generator import ClipReward, ClipEncoder
import argparse
import time
from plot_eval_results import plot_eval
import torch.nn as nn
import imageio
from stable_baselines3 import DQN, SAC
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.env_util import make_vec_env

# Assuming that ClipReward expects an image and returns a scalar reward
class CustomRewardWrapper(gym.Wrapper):
    def __init__(self, env, reward_generator, baseline_reg=False):
        super(CustomRewardWrapper, self).__init__(env)
        self.reward_generator = reward_generator
        self.baseline_reg = baseline_reg

        self.question_dict = {
            'CartPole-v1': 'pole vertically upright on top of the cart',
            'Pendulum-v1': 'pendulum in the upright position',
            'MountainCar-v0': 'a car at the peak of the mountain, next to the yellow flag',
            'MountainCarContinuous-v0': 'a car at the peak of the mountain, next to the yellow flag',
        }
        try:
            env_id = env.spec.id
        except:  # vecenv
            env_id = env.envs[0].spec.id
        self.question = self.question_dict[env_id]

        if self.baseline_reg:
            self.baseline_dict = {
                'CartPole-v1': 'pole and cart',
                'Pendulum-v1': 'pendulum',
                'MountainCar-v0': 'a car in the mountain',
                'MountainCarContinuous-v0': 'a car at the peak of the mountain, next to the yellow flag',
            }
            self.baseline = self.baseline_dict[env_id]    
        else:
            self.baseline = None

    def step(self, action):
        # https://stackoverflow.com/questions/52950547/getting-error-valueerror-too-many-values-to-unpack-expected-5
        observation, reward, done, truncated, info = self.env.step(action)
        image = self.env.render()

        custom_reward = self.reward_generator.get_reward(image, self.question, self.baseline, alpha=1.0)  # Modify this line to match the method of your reward generator
        custom_reward = custom_reward[0][0]
        # print('reward: ', reward, custom_reward)
        return observation, custom_reward, done, truncated, info

# an observation embedding wrapper
class ObservationEmbeddingWrapper(gym.Wrapper):
    def __init__(self, env, obs_encoder):
        super(ObservationEmbeddingWrapper, self).__init__(env)
        self.observation_space = gym.spaces.Box(low=-1e10, high=1e10, shape=(obs_encoder.embed_dim,))  # obs encoding shape by clip
        self.observation_encoder = obs_encoder

    def step(self, action):
        observation, reward, done, truncated, info = self.env.step(action)
        image = self.env.render()
        observation_embed = self.observation_encoder.encode(image)[0]  # first dim is batch size
        return observation_embed, reward, done, truncated, info

    def reset(self, *args, **kwargs):
        observation, info = self.env.reset(*args, **kwargs)
        image = self.env.render()
        observation_embed = self.observation_encoder.encode(image)[0]  # first dim is batch size
        return observation_embed, info

class RetrieveImageCallback(BaseCallback):
    def __init__(self, check_freq: int, log_dir: str, verbose=0):
        super(RetrieveImageCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir

    def _init_callback(self) -> None:
        if self.log_dir is not None:
            os.makedirs(self.log_dir, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            image = self.training_env.envs[0].render()
            img_name = f"{self.num_timesteps}.png"
            plt.imsave(os.path.join(self.log_dir, img_name), image)
        return True

# Create a custom callback by extending EvalCallback
class CustomEvalCallback(EvalCallback):
    def __init__(self, eval_env, eval_freq=10000, num_eval_episodes=1, log_path=None):
        super(CustomEvalCallback, self).__init__(eval_env, best_model_save_path=None, log_path=log_path,
                                                 eval_freq=eval_freq, deterministic=True,
                                                )
        self.num_eval_episodes = num_eval_episodes
        self.mean_reward = 0.0
        self.std_reward = 0.0
        self.log_path = log_path

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            self.logger.record("eval/episodes", self.num_eval_episodes, exclude="tensorboard")
            self.logger.record("eval/mean_reward", self.mean_reward, exclude="tensorboard")
            self.logger.record("eval/std_reward", self.std_reward, exclude="tensorboard")

            # Generate and save a GIF of the environment rendering
            gif_path = os.path.join(self.log_path, f"render_{self.n_calls}.gif")
            self.generate_gif(gif_path)

        # return super()._on_step()

    def generate_gif(self, gif_path):
        frames = []
        obs = self.training_env.reset()
        done = False
        while not done:
            action, _ = self.model.predict(obs, deterministic=True)
            obs, _, done, _ = self.training_env.step(action)
            frame = self.training_env.render(mode='rgb_array')
            frames.append(frame)

        imageio.mimsave(gif_path, frames)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default='CartPole-v1', required=True)
    parser.add_argument("--save-image", action='store_true', required=False)
    parser.add_argument("--render-eval", action='store_true', required=False)
    parser.add_argument("--clip-reward", action='store_true', required=False)
    parser.add_argument("--obs-embedding", action='store_true', required=False)
    parser.add_argument('--clip-model', choices=['RN50', 'ViT-B-32', 'ViT-B-16', 'ViT-H-14', 'ViT-L-14', 'ViT-L-14-336', 'ViT-bigG-14'], default='ViT-B-32', help='Your move in the game.')
    parser.add_argument("--baseline-reg", action='store_true', required=False)
    args = parser.parse_args()

    # Create log dir
    # game = ['CartPole-v1', 'Pendulum-v1', 'MountainCar-v0'][2]
    game = args.env
    model_dir = f"data/gym/{game}"
    os.makedirs(model_dir, exist_ok=True)

    # Create the environment
    env = gym.make(game, render_mode='rgb_array')

    if args.clip_reward:
        # Instantiate the reward generator
        rewarder = ClipReward(args.clip_model)
        env = CustomRewardWrapper(env, reward_generator=rewarder)
    if args.obs_embedding:
        # Instantiate the observation encoder
        obs_encoder = ClipEncoder(args.clip_model)
        env = ObservationEmbeddingWrapper(env, obs_encoder)

    # Create the original environment for evaluation
    if args.render_eval:
        eval_env = gym.make(game, render_mode='human')  # use true reward for evaluation
    else:
        eval_env = gym.make(game)

    if args.obs_embedding:
        eval_env = gym.make(game, render_mode='rgb_array')
        eval_env = ObservationEmbeddingWrapper(eval_env, obs_encoder)

    if 'Continuous' in game:
        algorithm = 'sac'
    else:
        algorithm = 'dqn'
    class CustomNetwork(BaseFeaturesExtractor):
        def __init__(self, observation_space: gym.spaces.Space, features_dim: int = 256):
            super(CustomNetwork, self).__init__(observation_space, features_dim)

            # Customize these layers and their sizes according to your requirements
            self.network = nn.Sequential(
                nn.Linear(observation_space.shape[0], 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, features_dim),
                nn.ReLU()
            )

        def forward(self, observations):
            return self.network(observations)

    policy_kwargs = dict(
        features_extractor_class=CustomNetwork,
        features_extractor_kwargs=dict(features_dim=256)  # Adjust the features_dim if needed
    )
    # model = DQN('MlpPolicy', env, policy_kwargs=policy_kwargs, verbose=1, tensorboard_log=model_dir)

    # Define the model using the DQN class
    if algorithm == 'dqn':
        custom_architecture = [256, 256]  # Define the number of hidden units for each layer
        model = DQN('MlpPolicy', env, learning_rate=2e-3, learning_starts=75000, train_freq=200, gradient_steps=200, verbose=1, policy_kwargs={"net_arch": custom_architecture}, tensorboard_log=model_dir)
    elif algorithm == 'sac':
        custom_architecture = [64, 64]
        model = SAC("MlpPolicy", env, learning_rate=3e-4, gamma=0.9999, tau=0.01, verbose=1, policy_kwargs={"net_arch": custom_architecture}, tensorboard_log=model_dir)
    else:
        raise ValueError("Invalid algorithm. Choose either 'dqn' or 'sac'.")
        

    # Use the CustomEvalCallback
    time_stamp = time.strftime('%Y%m%d-%H%M%S', time.localtime())
    log_dir = os.path.join(model_dir, f"logs/{time_stamp}/")
    os.makedirs(log_dir, exist_ok=True)

    eval_callback = EvalCallback(
        eval_env=eval_env,
        eval_freq=1000,  # evaluate every 5000 steps
        deterministic=True,
        log_path=log_dir,
        best_model_save_path=log_dir,
        render=False
    )
    custom_callback = CustomEvalCallback(env, log_path=log_dir)

    # Initialize the callback
    if args.save_image:
        callback = RetrieveImageCallback(check_freq=1, log_dir=log_dir)
        callbacks = [callback, eval_callback, custom_callback]
    else:
        callbacks = [eval_callback, custom_callback]

    # Train the model with the callback
    model.learn(total_timesteps=100000, callback=callbacks)

    plot_eval(log_dir)

    # Save the trained model (optional)
    model.save(f"{algorithm}_cartpole")

    # Close the environment
    env.close()
