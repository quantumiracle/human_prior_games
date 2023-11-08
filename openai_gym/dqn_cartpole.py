import gym
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy

# Step 3: Create the environment
# env = make_vec_env('CartPole-v1', n_envs=1)
env = gym.make('CartPole-v1', render_mode='human')

# Step 4: Define the model using the DQN class
model = DQN('MlpPolicy', env, verbose=1)

# Step 5: Train the model
model.learn(total_timesteps=10000)

# Step 6: Save the trained model (optional)
model.save("dqn_cartpole")

# If you want to load the model later, you can use:
# model = DQN.load("dqn_cartpole", env=env)

# Step 7: Test the trained model (optional)
# Evaluate the policy
mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)

# Enjoy trained agent
obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = env.step(action)
    env.render()
    if dones:
        obs = env.reset()

env.close()