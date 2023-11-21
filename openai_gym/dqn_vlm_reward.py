import os
import gym
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
import matplotlib.pyplot as plt
from clip_module.clip_reward_generator import ClipReward, ClipEncoder
from stable_baselines3.common.evaluation import evaluate_policy
import argparse
import time
from plot_eval_results import plot_eval

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
        }
        self.question = self.question_dict[env.spec.id]

        if self.baseline_reg:
            self.baseline_dict = {
                'CartPole-v1': 'pole and cart',
                'Pendulum-v1': 'pendulum',
                'MountainCar-v0': 'a car in the mountain',
            }
            self.baseline = self.baseline_dict[env.spec.id]    
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

    # Define the model using the DQN class
    model = DQN('MlpPolicy', env, verbose=1, tensorboard_log=model_dir)

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

    # Initialize the callback
    if args.save_image:
        callback = RetrieveImageCallback(check_freq=1, log_dir=log_dir)
        callbacks = [callback, eval_callback]
    else:
        callbacks = [eval_callback]

    # Train the model with the callback
    model.learn(total_timesteps=100000, callback=callbacks)

    plot_eval(log_dir)

    # Save the trained model (optional)
    model.save("dqn_cartpole")

    # Close the environment
    env.close()
