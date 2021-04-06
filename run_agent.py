# Author: @hany606
# About: Running an agent with a specific policy

# TODO:
# - Create cli argument to know which policy to run and which agent to choose from the trained agents
 

import gym
from DQN import VanillaDQN
from torch import load
from time import time
import matplotlib.pyplot as plt
from matplotlib import animation


# Source: http://www.pinchofintelligence.com/getting-started-openai-gym/
def display_frames_as_gif(frames, filename_gif = None):
    """
    Displays a list of frames as a gif, with controls
    """
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi = 72)
    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
    if filename_gif: 
        anim.save(filename_gif, writer = 'imagemagick', fps=20)
    # display(display_animation(anim, default_mode='loop'))



def get_action(state, policy=None):
    if policy is None:
        return env.action_space.sample()
    else:
        return policy.sample_action(state)

# https://github.com/openai/gym/wiki/MountainCar-v0
env = gym.make('MountainCar-v0')
dqn_agent = VanillaDQN(env, restore="dqn_trained_agents/agent2/best_model_dqn")


observation = env.reset()

done = False
total_reward = 0
frames = []
while not done:
    frames.append(env.render(mode = 'rgb_array'))
    action = get_action(observation, dqn_agent)
    observation, reward, done, _ = env.step(action)
    total_reward += reward
display_frames_as_gif(frames, "agent2.gif")
print(f"Total reward: {total_reward}, Done flag: {done}")
# env.close()
    