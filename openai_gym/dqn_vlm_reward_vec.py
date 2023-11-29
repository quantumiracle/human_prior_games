import os
# import gym
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from clip_module.clip_reward_generator import ClipReward, ClipEncoder
import argparse
import time
import wandb
from PIL import Image
from plot_eval_results import plot_eval
import torch.nn as nn
import imageio
from stable_baselines3 import DQN, SAC
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnvWrapper
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor

class CustomRewardWrapper(VecEnvWrapper):
    def __init__(self, venv, reward_generator, baseline_reg=False):
        super(CustomRewardWrapper, self).__init__(venv)
        self.reward_generator = reward_generator
        self.baseline_reg = baseline_reg

        self.question_dict = {
            'CartPole-v1': 'pole vertically upright on top of the cart',
            'Pendulum-v1': 'pendulum in the upright position',
            'MountainCar-v0': 'a car at the peak of the mountain, next to the yellow flag',
            'MountainCarContinuous-v0': 'a car at the peak of the mountain, next to the yellow flag',
        }
        
        # Try to get the env_id from the first environment in the vectorized env
        try:
            env_id = venv.envs[0].spec.id
        except AttributeError:
            raise ValueError("The provided environment does not have a 'spec.id' attribute.")

        self.question = self.question_dict.get(env_id, "Default question if env_id not found")

        if self.baseline_reg:
            self.baseline_dict = {
                'CartPole-v1': 'pole and cart',
                'Pendulum-v1': 'pendulum',
                'MountainCar-v0': 'a car in the mountain',
                'MountainCarContinuous-v0': 'a car at the peak of the mountain, next to the yellow flag',
            }
            self.baseline = self.baseline_dict.get(env_id, "Default baseline if env_id not found")
        else:
            self.baseline = None

    def reset(self):
        # Custom logic on reset
        return self.venv.reset()

    def step_async(self, actions):
        # Custom logic before environment steps
        self.venv.step_async(actions)

    def step_wait(self):
        # Custom logic after environment steps
        observations, rewards, dones, infos = self.venv.step_wait()
        # return observations, rewards, dones, infos

        # image = self.venv.get_images()[i]  # get one image
        # Save the image for debugging
        # image_arr = Image.fromarray(image)
        # output_path = os.path.join('test', f'saved_image{i}.png')
        # image_arr.save(output_path)

        image = self.venv.get_images()
        custom_reward = self.reward_generator.get_reward(image, self.question, self.baseline, alpha=1.0)
        new_rewards = custom_reward.reshape(-1)  # reshape to (num_envs, 1) to (num_envs,)
        return observations, new_rewards, dones, infos



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

class ObservationEmbeddingVecWrapper(VecEnvWrapper):
    def __init__(self, venv, obs_encoder):
        super(ObservationEmbeddingVecWrapper, self).__init__(venv)
        # Update the observation space to match the output of the encoder
        self.observation_space = gym.spaces.Box(low=-1e10, high=1e10, shape=(obs_encoder.embed_dim,), dtype=np.float32)
        self.observation_encoder = obs_encoder

    def step_wait(self):
        observations, rewards, dones, infos = self.venv.step_wait()
        encoded_observations = np.array([self.observation_encoder.encode(obs)[0] for obs in observations])  # TODO: batch inference
        # Ensure correct shape and add terminal_observation if needed
        for i in range(len(dones)):
            if dones[i]:
                infos[i]['terminal_observation'] = encoded_observations[i]
            assert encoded_observations[i].shape == self.observation_space.shape, "Mismatch in observation shape"

        return encoded_observations, rewards, dones, infos

    def reset(self):
        observations = self.venv.reset()
        encoded_observations = np.array([self.observation_encoder.encode(obs)[0] for obs in observations])
        return encoded_observations


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
            # Retrieve an image from the first environment instance
            image = self.training_env.envs[0].env.render()
            img_name = f"{self.num_timesteps}.png"
            plt.imsave(os.path.join(self.log_dir, img_name), image)
        return True

class CustomEvalCallback(EvalCallback):
    # "https://stable-baselines3.readthedocs.io/en/master/_modules/stable_baselines3/common/callbacks.html#EvalCallback"
    def __init__(self, eval_env, wandb_log=False, eval_freq=10000, num_eval_episodes=1, log_path=None):
        super(CustomEvalCallback, self).__init__(eval_env, best_model_save_path=None, log_path=log_path,
                                                 eval_freq=eval_freq, deterministic=True,
                                                 n_eval_episodes=num_eval_episodes)
        self.wandb_log = wandb_log
        self.log_path = log_path

    def _on_step(self) -> bool:
        if self.n_calls % 100 == 0:
            # Compute mean and standard deviation of rewards
            if len(self.evaluations_results) > 0 and len(self.evaluations_length) > 0:   
                episode_reawrds = np.array(self.evaluations_results[-1])
                episode_lengths = np.array(self.evaluations_length[-1])
                mean_reward = np.mean(episode_reawrds)
                std_reward = np.std(episode_reawrds) 
                mean_ep_length = np.mean(episode_lengths)
            else:
                mean_reward = 0.0
                std_reward = 0.0
                mean_ep_length = 0.0

            # Log results
            if self.wandb_log:
                wandb.log({'eval/mean_reward': mean_reward, 'eval/std_reward': std_reward, 'eval/mean_ep_length': mean_ep_length}, step=self.n_calls)


        if self.n_calls % self.eval_freq == 0:
            # Optional: Save a GIF of the environment rendering, if log_path is set
            if self.log_path:
                gif_path = os.path.join(self.log_path, f"render_{self.n_calls}.gif")
                # Ensure that the generate_gif method is implemented if you wish to use this feature
                self.generate_gif(gif_path)

        return super()._on_step()

    def generate_gif(self, gif_path):
        frames = []
        obs = self.training_env.reset()
        done = [False]  # Initialize 'done' as a list with one False element

        while not done[0]:  # Check only the first element
            action, _ = self.model.predict(obs, deterministic=True)
            obs, _, dones, _ = self.training_env.step(action)
            frame = self.training_env.envs[0].env.render()
            frames.append(frame)
            done[0] = dones[0]  # Update the done status for the first environment

        imageio.mimsave(gif_path, frames)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default='CartPole-v1', required=True)
    parser.add_argument("--save-image", action='store_true', required=False)
    parser.add_argument("--render-eval", action='store_true', required=False)
    parser.add_argument("--clip-reward", action='store_true', required=False)
    parser.add_argument("--obs-embedding", action='store_true', required=False)
    parser.add_argument('--clip-model', choices=['RN50', 'ViT-B-32', 'ViT-B-16', 'ViT-H-14', 'ViT-L-14', 'ViT-L-14-336', 'ViT-bigG-14'], default='ViT-L-14-336', help='Your move in the game.')
    parser.add_argument("--baseline-reg", action='store_true', required=False)
    parser.add_argument("--init-wandb", action='store_true', required=False)
    args = parser.parse_args()

  
    # Create log dir
    # game = ['CartPole-v1', 'Pendulum-v1', 'MountainCar-v0'][2]
    game = args.env
    model_dir = f"data/gym/{game}"
    os.makedirs(model_dir, exist_ok=True)

    # Create the environment
    # env = gym.make(game, render_mode='rgb_array')

    num_envs = 10  # Number of parallel environments
    num_eval_envs = 1
    # Create a vectorized environment
    # env = make_vec_env(game, n_envs=num_envs, seed=0)
    env = DummyVecEnv([lambda: gym.make(game, render_mode='rgb_array') for _ in range(4)])

    if args.clip_reward:
        # Instantiate the reward generator
        rewarder = ClipReward(args.clip_model)
        env = CustomRewardWrapper(env, reward_generator=rewarder)
    if args.obs_embedding:
        # Instantiate the observation encoder
        obs_encoder = ClipEncoder(args.clip_model)
        env = ObservationEmbeddingVecWrapper(env, obs_encoder)

    # Create the original environment for evaluation
    if args.render_eval:
        eval_env = gym.make(game, render_mode='human')  # use true reward for evaluation
    else:
        eval_env = gym.make(game, render_mode='rgb_array')

    if args.obs_embedding:
        eval_env = gym.make(game, render_mode='rgb_array')
        eval_env = ObservationEmbeddingWrapper(eval_env, obs_encoder)

    eval_env = Monitor(eval_env)

    if 'Continuous' in game:
        algorithm = 'sac'
    else:
        algorithm = 'dqn'

    if args.init_wandb:
        wandb.init(project='human_prior_rl', name=f"{game}_{algorithm}_ObsEmbed_{args.obs_embedding}_ClipReward-{args.clip_reward}_{args.clip_model}", \
            config={"algorithm": algorithm, "env": game, "clip_reward": args.clip_reward, "obs_embedding": args.obs_embedding, "clip_model": args.clip_model, "baseline_reg": args.baseline_reg, "render_eval": args.render_eval, "save_image": args.save_image, "init_wandb": args.init_wandb})

        wandb.config.update({
            "algorithm": algorithm,
            "env": game,
            # Add other relevant configurations
        })

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
        eval_freq=1000 // num_envs,  # evaluate every x steps
        deterministic=True,
        log_path=log_dir,
        best_model_save_path=log_dir,
        render=False
    )

    eval_freq = 10000
    eval_freq = eval_freq // num_envs  # call for each env
    custom_callback = CustomEvalCallback(eval_env, args.init_wandb, log_path=log_dir, eval_freq=eval_freq)  # for gif

    # Initialize the callback
    if args.save_image:
        callback = RetrieveImageCallback(check_freq=1, log_dir=log_dir)
        callbacks = [callback, eval_callback, custom_callback]
    else:
        callbacks = [eval_callback, custom_callback]

    # Train the model with the callback
    model.learn(total_timesteps=1000000, callback=callbacks)

    # Save the trained model (optional)
    model.save(f"{algorithm}_{game}")

    # Close the environment
    env.close()
    del model
    # wandb.finish()  

    plot_eval(log_dir)  # cannot plot when vec env
