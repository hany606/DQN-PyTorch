# Author: @hany606
# About: Running an agent with a specific policy

# TODO:
# - Create cli argument to know which policy to run and which agent to choose from the trained agents
 

import gym
from DQN import VanillaDQN

def get_action(state, policy=None):
    if policy is None:
        return env.action_space.sample()
    else:
        policy.sample_action(state)

# https://github.com/openai/gym/wiki/MountainCar-v0
env = gym.make('MountainCar-v0')
dqn_agent = VanillaDQN(env)


observation = env.reset()

done = False
while not done:
    env.render()
    action = get_action(observation)
    observation, reward, done, _ = env.step(action)

env.close()
    