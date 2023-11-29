
## Usage

### CLIP reward
Test:
```
# save image for test
python openai_gym/dqn_vlm_reward.py --env CartPole-v1 --save-image
# test reward for saved images
python clip_module/test_clip_reward.py --env CartPole-v1
# test reward with baseline regularization for saved images
python clip_module/test_clip_reward.py --env CartPole-v1 --baseline-reg
```
Render image with VLM reward:
```
python vlm_reward_test.py --env=CartPole-v1 --samples=50 --clip-model=ViT-B-32
```

Train with CLIP reward:
```
# without baseline regularization
python openai_gym/dqn_vlm_reward.py --env CartPole-v1 --clip-reward
python openai_gym/dqn_vlm_reward.py --env Pendulum-v1 --clip-reward
python openai_gym/dqn_vlm_reward.py --env MountainCar-v0 --clip-reward
# with baseline regularization
python openai_gym/dqn_vlm_reward.py --env CartPole-v1 --clip-reward --baseline-reg
# with vectorized env for training, WandB for tracking
python openai_gym/dqn_vlm_reward_vec.py --env CartPole-v1 --clip-reward --init-wandb
```

Train with observation embedding (using CLIP):
```
python openai_gym/dqn_vlm_reward.py --env CartPole-v1 --obs-embedding
```

Eval/render trained models:
```
python openai_gym/dqn_eval.py --env CartPole-v1 --render-eval
```
Plot eval (in training) curves:
```
python openai_gym/plot_eval_results.py --env CartPole-v1
```

### Llava Query
```
# launch a llava model
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 python llm_query/launch.py --model-path liuhaotian/llava-v1.5-7b --load-4bit

# open another terminal
# single image single round query
python llm_query/query_llava.py --image-file xxx --question xxx

# feedback in maze game:
python tests/test_feedback.py

# multiple images in multiple round query
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 python llm_query/launch_multi_round.py --model-path liuhaotian/llava-v1.5-7b --load-4bit
# feedback in maze game:
python tests/test_feedback.py


# multiple images in one round query
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 python llm_query/launch_multi_images.py --model-path liuhaotian/llava-v1.5-7b --load-4bit
# feedback in maze game:
python tests/test_feedback_multi_images.py
```

remote query from another server:
```
echo "your message" | sshpass -p zihanding ssh zihan@128.112.50.79 "cat > /data/zihan/research/human_prior_games/query_pipe"
```

## Investigating Human Priors for Playing Video Games ##
#### In ICML 2018 [[Project Website]](https://rach0012.github.io/humanRL_website/) 
[Rachit Dubey](http://cocosci.berkeley.edu/rachit/), [Pulkit Agrawal](https://people.eecs.berkeley.edu/~pulkitag/), [Deepak Pathak](https://people.eecs.berkeley.edu/~pathak/), [Thomas L. Griffiths](http://cocosci.berkeley.edu/tom/tom.php), [Alexei A. Efros](https://people.eecs.berkeley.edu/~efros/)<br/>
University of California, Berkeley<br/>

![Games](screens.png?raw=True "Games!")

This contains code for our suit of custom games built to test performance of RL agents for our paper 'Investigating Human Priors for Playing Video Games' published in ICML 2018.

The 'original game' is a simple platformer game. The 'no semantics' is the game which removes semantic information prior. The 'no object' is the game which removes concept of object prior. The 'no affordance' is the game which removes affordance prior. The 'no similarity' is the game which masks similarity prior. 

Refer to our paper available here for more details - https://arxiv.org/abs/1802.10217

We used the PyGame-Learning-Environment to build these games - https://github.com/ntasfi/PyGame-Learning-Environment. 
All the games are based on the codes from the game 'MonsterKong' from PLE. 

To train your RL agent on the games, you would also need gym-ple, a package that allows to use PLE as a gym environment. Our version of gym-ple which includes our games is available here - https://github.com/rach0012/humanRL_gym_ple/

For the experiments in our paper, we used A3C-ICM, a curiosity augmented RL algorithm which is available here - https://github.com/pathak22/noreward-rl. 

If you find this work useful in your research, please cite:

    @inproceedings{dubeyICMl18humanRL,
        Author = {Dubey, Rachit and Agrawal, Pulkit and Pathak, Deepak and Griffiths, Thomas L.
                 and Efros, Alexei A.},
        Title = {Investigating Human Priors for Playing Video Games},
        Booktitle = {International Conference on Machine Learning ({ICML})},
        Year = {2018}
    }


## Getting started

A `PLE` instance requires a game exposing a set of control methods. To see the required methods look at `ple/games/base.py`. 

Here's an example of importing the original game from the games library within PLE:

```python
from ple.games.originalGame import originalGame

game = originalGame()
```

Next we configure and initialize PLE:

```python
from ple import PLE

p = PLE(game, fps=30, display_screen=True, force_fps=False)
p.init()
```

The options above instruct PLE to display the game screen, with `display_screen`, while allowing PyGame to select the appropriate delay timing between frames to ensure 30fps with `force_fps`.

You are free to use any agent with the PLE. Below we create a fictional agent and grab the valid actions:

```python
myAgent = MyAgent(p.getActionSet())
```

We can now have our agent, with the help of PLE, interact with the game over a certain number of frames:

```python

nb_frames = 1000
reward = 0.0

for f in range(nb_frames):
	if p.game_over(): #check if the game is over
		p.reset_game()

	obs = p.getScreenRGB()
	action = myAgent.pickAction(reward, obs)
	reward = p.act(action)

```

Just like that we have our agent interacting with our game environment.

## Installation

PLE requires the following dependencies:
* numpy
* pygame
* pillow

Clone the repo and install with pip.

```bash
git clone https://github.com/ntasfi/PyGame-Learning-Environment.git
cd PyGame-Learning-Environment/
pip install -e .
``` 

## Headless Usage

Set the following in your code before usage:
```python
os.putenv('SDL_VIDEODRIVER', 'fbcon')
os.environ["SDL_VIDEODRIVER"] = "dummy"
```
