# References:
# https://jonathan-hui.medium.com/rl-dqn-deep-q-network-e207751f7ae4
# https://www.analyticsvidhya.com/blog/2019/04/introduction-deep-q-learning-python/#:~:text=Deep%20Q%2DNetworks,is%20generated%20as%20the%20output.
# http://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-8.pdf
# https://github.com/openai/gym/wiki/MountainCar-v0

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import numpy as np
from copy import deepcopy
import random
from math import tanh

np.random.seed(0)
random.seed(0)

class Net(nn.Module):
    def __init__(self, observations_dim, actions_dim, hidden_dim=500):
        super(Net, self).__init__()
        self._input_layer = nn.Linear(observations_dim, hidden_dim)
        self._hidden1 = nn.Linear(hidden_dim, hidden_dim)
        # self._hidden2 = nn.Linear(64, 32)
        # self.hid3 = nn.Linear(100, 50)
        self._output_layer = nn.Linear(hidden_dim,actions_dim)

    def forward(self, x):
        x = F.relu(self._input_layer(x))
        x = F.relu(self._hidden1(x))
        # x = F.relu(self._hidden2(x))
        x = self._output_layer(x)
        return x

    def save(self, path, name):
        pass

class ReplayMemory:
    def __init__(self, observation_size, action_size, replay_size=1000):
        self.replay = deque(maxlen=replay_size)
        self.observation_size = observation_size
        self.action_size = action_size

    def sample(self, num_samples=100):
        sample = np.array(random.sample(list(self.replay), num_samples))#np.random.choice(self.replay, num_samples)#min(len(self.replay),num_samples))
        st  = sample[:,:self.observation_size] 
        at  = sample[:,self.observation_size:self.observation_size+self.action_size]
        rt  = sample[:,self.observation_size+self.action_size:self.observation_size+self.action_size+1]
        st1 = sample[:,self.observation_size+self.action_size+1:]
        return (st, at, rt, st1)#, len(sample)

    # Sample is (s_i, a_i, r_i, s_{i+1})
    def add(self, s, a, r, s1):
        self.replay.append([*s, *a, r, *s1])

    def len(self):
        return len(self.replay)

class VanillaDQN:
    def __init__(self, env, hidden_dim=500, replay_size=1000):
        self.env = env
        self.observations_dim = self.env.observation_space.shape[0] # For later it can be changed to be ok with other shapes
        self.actions_dim = self.env.action_space.n
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net = Net(self.observations_dim, self.actions_dim, hidden_dim).to(self.device)
        self.target_net = deepcopy(self.net)
        self.replay = ReplayMemory(self.observations_dim, 1, replay_size)
        self.loss = nn.MSELoss()


    def sample_action(self, observation):
        if(isinstance(observation, np.ndarray)):
            observation = torch.from_numpy(observation).float().unsqueeze(0).to(self.device)
        q_values = self.net.forward(observation)
        return torch.argmax(q_values).int().sum().item()

    def _compute_loss(self, batch, gamma):
        st, at, rt, st1 = batch# rt + gamma*
        # print(st)
        # print(at)
        # print(rt)
        # print(st1)
        st1_torch = torch.from_numpy(st1).float().unsqueeze(0).to(self.device)
        rt_torch = torch.from_numpy(rt).float().unsqueeze(0).view(len(rt),1).to(self.device)#
        st_torch = torch.from_numpy(st).float().unsqueeze(0).to(self.device)
        at_torch = torch.from_numpy(at).long().unsqueeze(0).to(self.device)
        # print(rt_torch)
        # print(self.target_net.forward(st1_torch))
        # print(torch.max(self.target_net.forward(st1_torch)))
        # print(torch.max(self.target_net.forward(st1_torch),2))
        # print(torch.max(self.target_net.forward(st1_torch),2)[0])
        # print(gamma*(torch.max(self.target_net.forward(st1_torch),2)[0]).view(len(rt),-1))
        # print("-----------")
        target = rt_torch + gamma*(torch.max(self.target_net.forward(st1_torch),2)[0]).view(len(rt),1)
        # print(at_torch)
        # print(self.net.forward(st_torch))
        # print(self.net.forward(st_torch).gather(2, at_torch))
        predicted = self.net.forward(st_torch).gather(2, at_torch)[0]
        # print(rt_torch)
        # print(target)
        # print(predicted)
        return self.loss(target, predicted)

    def train(self, render=False, num_epochs=100, num_steps=1000, eps_prob=0.5, target_update_freq=500, batch_size=100, gamma=0.99, learning_rate=1e-3):
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=learning_rate)
        total_num_steps = 0
        for i in range(num_epochs):
            # self.target_net = deepcopy(self.net)
            obs = self.env.reset()
            epoch_reward = 0
            for t in range(num_steps):
                if(render):
                    self.env.render()
                total_num_steps += 1
                obs_torch = torch.from_numpy(obs).float().unsqueeze(0).to(self.device) # test without -> This should be the correct to change from numpy to torch and add it on cuda
                # States(Observation) -> DQN -> Q-values for all the actions
                q_vals = self.net.forward(obs_torch)
                # print("Got Q Values")
                # Select an action based on epsilon-greedy policy with the current Q-Network
                action = int(self.sample_action(obs))
                if(np.random.rand() < eps_prob):
                    action = self.env.action_space.sample()
                # print("Got the action")
                # Perform the action
                observation, r, done, _ = self.env.step(int(action))
                # print(observation[0])
                # When only the position is in the reward it make it only try to go up not by going right and left but just go right
                reward = r + abs(observation[1])*10-abs(observation[0]-0.5)#r + abs(observation[1]-2)*5 - abs(observation[0]-0.5)*10 #r - abs(observation[0]-0.5)*5 + (observation[1]-2)*5#r*abs(observation[0] - 0.5) #r+(observation[0] - 0.5) # r*tanh(observation[0] - 0.5) 
                # print(reward, abs(observation[0]-0.5), abs(observation[1])*2)
                # print("Env step is done")
                epoch_reward += r
                # Store the transition (s_t, a_t, r_t, s_{t+1})
                self.replay.add(list(deepcopy(obs)), list([action]), reward, list(observation))
                # print("Experience is added")
                obs = deepcopy(observation)
                if(self.replay.len() >= batch_size):
                    # print("Going to update the policy")
                    # Sample random batch
                    batch = self.replay.sample(batch_size)
                    # print("Sample Batch is done")
                    # print(batch)
                    # Compute loss
                    loss = self._compute_loss(batch, gamma)
                    # print(loss)
                    # print("Compute loss is finished")
                    # Optimize the network
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    # print("Optimizer updated the policy")
                    # exit()

                if(total_num_steps % target_update_freq == 0):
                    self.target_net = deepcopy(self.net)
                    # print("Target is shifted")
                    # exit()

                if(observation[0] > 0.48):
                    print(observation[0])
                    print("Done with 0.01 difference between the goal")
                    print(f"Epoch {i+1} (reward): {epoch_reward}")
                    # self.save("./")
                    observation = self.env.reset()
                    done = False
                    total_reward = 0
                    while not done:
                        # self.env.render()
                        action = int(self.sample_action(observation))
                        observation, reward, done, _ = self.env.step(action)
                        total_reward += reward
                    print(f"Epoch {i+1} (Test reward): {total_reward}")
                    break         
                
                if(t == num_steps-1 or reward == 0):
                    print(r, done)
                    # print(abs(observation[0]-0.5))
                    print(f"Epoch {i+1} (reward): {epoch_reward}")
                    break         

    def save(self, path, name):
        self.net.save(path, name)


if __name__ == '__main__':
    pass
    