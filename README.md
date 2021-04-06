# Internship-Task
This repository is for solving a task for a summer 2021 research internship  "Implement DQN, policy gradient or actor-critic RL algorith to solve Mountain-Car gym environment"

## Details and Explanation

I have implemented (Simple-Vanilla) Deep Q-Network (DQN) algorithm with experience replay buffer and frequent change for the target network inside "DQN.py".

![Trained Agent](https://github.com/hany606/Internship-Task/gif/agent2.gif)

After testing with the original reward of the environment, nothing was improved in the training. So, I have changed the reward function, to test different behavior and see some improvements.

Multiple reward functions have been tested to conform with the desired behaviors:
- Move with fast right and left -> Correlated with velocity [2nd observation]

- Move closer to the goal -> Correlated with the position [1st observation]

I have noticed some observations:

- When only the position is in the reward (or the position dominated) it makes it only try to go up not by going right and left but just go right

- When only the velocity is in the reward (or the velocity dominated) it makes it only to move fast right and left and don't care about the real goal (position)

## How to use?

- start training script:
    ```bash
        python3 learn.py
    ```

- start trained agent:
    ```bash
        python3 run_agent.py
    ```

## Plot for rewards during the training

Reward training plot for 500 epochs:

![Reward plot](https://github.com/hany606/Internship-Task/dqn_trained_agnets/agent2/best_model_dqn.png)

Reward testing plot for 100 epochs:

![Reward plot](https://github.com/hany606/Internship-Task/dqn_trained_agnets/agent2/best_model_dqn_testing.png)



## References:

* [DQN paper](https://arxiv.org/abs/1312.5602v1)
* [Good online tutorial for theory](https://www.analyticsvidhya.com/blog/2019/04/introduction-deep-q-learning-python/#:~:text=Deep%20Q%2DNetworks,is%20generated%20as%20the%20output.)
* [Good course slides (CS285)](http://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-8.pdf)
* [Gym environment for MountainCar](https://github.com/openai/gym/wiki/MountainCar-v0)

## TODO (Later):

- [x] Implement Vanilla DQN for value-based RL algorithm

- [ ] Implement REINFORCE for Policy Gradient

- [ ] Implement simple Actor-Critic algorithm

- [x] Write good README.md with cool gifs

- [ ] Add more plots with more experiments with different seeds

- [ ] Perform different experiments

- [ ] Add trained agents and videos directories

- [ ] Add plots for different results with different algorithms

- [ ] Use RLlib to show the difference between the implementd and the library's implementation and provide plots

- [ ] Create a report with references and papers
