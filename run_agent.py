# Author: @hany606
# About: Running an agent with a specific policy

# TODO:
# - Create cli argument to know which policy to run and which agent to choose from the trained agents
 

import gym

env = gym.make('MountainCar-v0')

def get_action(policy=None):
    if policy is None:
        return env.action_space.sample()
    else:
        pass


observation = env.reset()

done = False
while not done:
    env.render()
    action = get_action()
    observation, reward, done, _ = env.step(action)

env.close()
    