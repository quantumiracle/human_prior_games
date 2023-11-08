import gym
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
import argparse

def eval_model(env, render):
    # Load the saved best model
    best_model_path = f"data/gym/{env}/logs/best_model"  # Adjust the path if necessary
    model = DQN.load(best_model_path)

    # Create the evaluation environment with 'human' rendering mode
    eval_env = gym.make(env, render_mode='human')

    # Evaluate the trained agent
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10, render=render)
    print(f"Mean reward: {mean_reward} +/- {std_reward}")

    # Optionally, you can manually loop through episodes to see them rendering
    # for episode in range(10):  # Change 10 to the number of episodes you want to see
    #     obs = eval_env.reset()
    #     done = False
    #     while not done:
    #         action, _states = model.predict(obs, deterministic=True)
    #         obs, reward, done, _ = eval_env.step(action)
    #         if render:
    #             eval_env.render()  # This will display the environment on the screen

    # # Close the evaluation environment
    # eval_env.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default='CartPole-v1', required=True)
    parser.add_argument("--render-eval", action='store_true', required=False)
    args = parser.parse_args()

    eval_model(args.env, args.render_eval)