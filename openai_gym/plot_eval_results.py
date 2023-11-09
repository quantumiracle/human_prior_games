import numpy as np
import os
import matplotlib.pyplot as plt
import argparse


def plot(env_name):
    # env_name = 'CartPole-v1'
    result_path = f'data/gym/{env_name}/logs/'

    # Load the npz file
    with np.load(os.path.join(result_path, 'evaluations.npz')) as data:
        ep_lengths = data['ep_lengths']
        results = data['results']
        timesteps = data['timesteps']

    # Calculate the mean and standard deviation for ep_lengths and results
    ep_lengths_mean = np.mean(ep_lengths, axis=1)
    ep_lengths_std = np.std(ep_lengths, axis=1)

    results_mean = np.mean(results, axis=1)
    results_std = np.std(results, axis=1)

    # Time step for each episode
    episodes = np.arange(len(ep_lengths_mean))

    # Create subplots
    fig, axs = plt.subplots(3, 1, figsize=(12, 18))  # 3 Rows, 1 Column

    # Plot Episode Lengths Mean with Shaded Standard Deviation
    axs[0].plot(episodes, ep_lengths_mean, label='Episode Lengths Mean')
    axs[0].fill_between(episodes, ep_lengths_mean - ep_lengths_std, ep_lengths_mean + ep_lengths_std, alpha=0.2)
    axs[0].set_title('Episode Lengths over Time')
    axs[0].set_xlabel('Steps (x1000)')
    axs[0].set_ylabel('Length')
    axs[0].legend()

    # Plot Results Mean with Shaded Standard Deviation
    axs[1].plot(episodes, results_mean, label='Results Mean')
    axs[1].fill_between(episodes, results_mean - results_std, results_mean + results_std, alpha=0.2)
    axs[1].set_title('Results over Time')
    axs[1].set_xlabel('Steps (x1000)')
    axs[1].set_ylabel('Reward')
    axs[1].legend()

    # Plot Timesteps (assuming it is a 1D array representing cumulative timesteps)
    axs[2].plot(timesteps, label='Timesteps')
    axs[2].set_title('Timesteps over Time')
    axs[2].set_xlabel('Time Step')
    axs[2].set_ylabel('Cumulative Timesteps')
    axs[2].legend()

    # Adjust layout and display the plot
    plt.tight_layout()
    plt.savefig(os.path.join(result_path, 'evaluations.png'))
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default='CartPole-v1', required=True)
    args = parser.parse_args()

    plot(args.env)
