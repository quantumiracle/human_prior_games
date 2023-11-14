# import gym
import gymnasium as gym
import numpy as np
from PIL import Image
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
import io, pickle
from clip_module.clip_reward_generator import ClipReward
import argparse


# Function to discretize the state space and render images for MountainCar-v0
def discretize_and_render_states_mountaincar(env, num_states_per_dimension=5):
    position_range = np.linspace(env.min_position, env.max_position, num_states_per_dimension)
    # velocity_range = np.linspace(-env.max_speed, env.max_speed, num_states_per_dimension)
    print(env.min_position, env.max_position, position_range)
    env = env.unwrapped  # important to get the unwrapped env
    rendered_images = []
    rewards = []

    for position in position_range:
        env.reset()
        # for velocity in velocity_range:
        env.state = np.array([position, 0])
        # obs, _, _, _, _ = env.step(1)
        # print(obs)
        img = env.render()
        # rendered_images.append(Image.fromarray(img))
        rendered_images.append(img)

        # For MountainCar, reward is always -1 unless the goal is reached
        reward = -1
        rewards.append(reward)

    return rendered_images, rewards, position_range

def discretize_and_render_states_cartpole(env, num_states_per_dimension=5):
    # Define the ranges for each state variable
    # cart_position_range = np.linspace(-2.4, 2.4, num_states_per_dimension)
    # cart_velocity_range = np.linspace(-1, 1, num_states_per_dimension)  # Arbitrarily chosen range
    pole_angle_range = np.linspace(-0.20943951, 0.20943951, num_states_per_dimension)  # ~12 degrees
    # pole_velocity_range = np.linspace(-1, 1, num_states_per_dimension)  # Arbitrarily chosen range

    # Store the rendered images and rewards
    rendered_images = []
    rewards = []
    env = env.unwrapped  # important to get the unwrapped env

    # Iterate through the discretized state space
    # for cp in cart_position_range:
    #     for cv in cart_velocity_range:
    for pa in pole_angle_range:
                # for pv in pole_velocity_range:
        # Set the state in the environment
        state = np.array([0, 0, pa, 0])
        # env.env.state = state
        env.state = state

        # Render the image for this state
        img = env.render()
        rendered_images.append(Image.fromarray(img))

        # Compute the reward for this state
        x, x_dot, theta, theta_dot = state
        r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
        r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
        reward = r1 + r2
        rewards.append(reward)

    return rendered_images, rewards, pole_angle_range

# Function to save images and rewards in a PDF
def save_images_rewards_pdf(images, rewards, filename):
    c = canvas.Canvas(filename, pagesize=letter)
    width, height = letter

    for i, (img, reward) in enumerate(zip(images, rewards)):
        img_buffer = io.BytesIO()
        img.save(img_buffer, format='PNG')
        img_buffer.seek(0)
        img_rl = ImageReader(img_buffer)

        img_y_position = height - 100 - (i % 3) * 200
        text_y_position = img_y_position - 20

        c.drawImage(img_rl, 50, img_y_position, width=400, height=200, preserveAspectRatio=True)
        c.drawString(50, text_y_position, f'Reward: {reward}')

        if (i + 1) % 3 == 0:
            c.showPage()

    c.save()

def get_reward_from_image(reward_generator, image, question, baseline, alpha=1.0):
    custom_reward = reward_generator.get_reward(image, question, baseline, alpha)  # Modify this line to match the method of your reward generator
    custom_reward = np.array(custom_reward).reshape(-1)[0]
    # custom_reward = custom_reward[0][0]
    return custom_reward

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default='MountainCar-v0', required=False)
    parser.add_argument('--clip-model', choices=['ViT-B-32', 'ViT-B-16', 'ViT-H-14', 'ViT-L-14', 'ViT-bigG-14'], default='ViT-B-32', help='Your move in the game.')
    parser.add_argument("--disable-baseline-reg", action='store_true', required=False)  # by default baseline regularization is enabled
    parser.add_argument("--alpha", type=int, default=1.0, required=False)
    parser.add_argument("--samples", type=int, default=50, required=False)
    args = parser.parse_args()

    rewarder = ClipReward(args.clip_model)
    # Example usage
    env = gym.make(f'{args.env}', render_mode='rgb_array')
    env.reset()
    if args.env == 'CartPole-v1':
        rendered_images, rewards, vars = discretize_and_render_states_cartpole(env, num_states_per_dimension=args.samples)
    elif args.env == 'Pendulum-v1':
        pass
        # rendered_images, rewards = discretize_and_render_states_pendulum(env, num_states_per_dimension=args.samples)
    elif args.env == 'MountainCar-v0':
        rendered_images, rewards, vars = discretize_and_render_states_mountaincar(env, num_states_per_dimension=args.samples)
    env.close()
    result_dict = {'rendered_images': rendered_images, 'rewards': rewards, 'vars': vars}
    # save_images_rewards_pdf(rendered_images, rewards, 'mountaincar_states.pdf')

    # get rewards from images
    question_dict = {
        'CartPole-v1': 'pole vertically upright on top of the cart',
        'Pendulum-v1': 'pendulum in the upright position',
        'MountainCar-v0': 'a car at the peak of the mountain, next to the yellow flag',
    }
    question = question_dict[env.spec.id]

    if not args.disable_baseline_reg:
        baseline_dict = {
            'CartPole-v1': 'pole and cart',
            'Pendulum-v1': 'pendulum',
            'MountainCar-v0': 'a car in the mountain',
        }
        baseline = baseline_dict[env.spec.id]    
    else:
        baseline = None
    vlm_rewards = []
    for image in rendered_images:
        vlm_reward = get_reward_from_image(rewarder, image, question, baseline, alpha=args.alpha)
        vlm_rewards.append(vlm_reward)
        print(vlm_reward)

    result_dict['vlm_rewards'] = vlm_rewards

    # Writing pkl data
    if args.disable_baseline_reg:
        alpha = 0.0
    else:
        alpha = args.alpha

    with open(f'{args.env}_{args.clip_model}_alpha{alpha}_results.pkl', 'wb') as file:
        pickle.dump(result_dict, file)