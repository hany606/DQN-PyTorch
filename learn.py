# Author: @hany606
# About: Learn the agent using specific policy

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
dqn_agent = VanillaDQN(env, hidden_dim=64)

dqn_agent.train(num_epochs=500, batch_size=128, target_update_freq=5000, render=True, eps_prob=0.1, learning_rate=0.003, num_steps=200)
# dqn_agent.train(num_epochs=500, batch_size=128, target_update_freq=1000, render=True, eps_prob=0.5, learning_rate=0.1, num_steps=1000)

# for i in range(num_epochs):
#     reward = dqn_agent.forward()
#     print(f"Epoch {i} (reward): {reward}")
# observation = env.reset()

# done = False
# while not done:
#     env.render()
#     action = get_action(observation)
#     observation, reward, done, _ = env.step(action)

# env.close()
    