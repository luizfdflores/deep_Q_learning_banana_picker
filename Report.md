# Project 1 - Deep Reinforcement Learning
# Deep Q Learning - Banana Picker
# REPORT

### Learning Algorithm

The report clearly describes the learning algorithm

###### Hyperparameters
The chosen hyperparameters values were:
- eps_start = 1.0 - starting value of epsilon, for epsilon-greedy action selection
- eps_end = 0.01 - minimum value of epsilon
- eps_decay = 0.995 - epsilon decay factor
- buffer_size = int(1e5) - replay buffer size
- batch_size = 64 - minibatch size
- gamma = 0.99 - discount factor
- tau = 1e-3 - for soft update of target parameters
- lr = 5e-4 - learning rate
- update_every = 4 - how often to update the network
- n_fc_layers = 5 - Number of fully connected layers in the model

###### Neural network Model
Only fully connected layers were used in the model for this agent. I ran the agent for 200 episodes, using different numbers of hidden layers in the network. In the final version, I used 4 hidden layers in the neural network, which worked great.

### Training

![Rewards Plot](images/img.png?raw=true "Title")

The agent was trained for 2000 episodes. The goal, an average score over 100 episodes of +13 was achieved after 429 episodes. Until the end of the 2000 episodes, every time the goal was hit, it was increased by 0.5. At the end of the training, the highest average score, over 100 episodes that was achieved was 17.0 after 931 episodes.

### Ideas for Future Work

- Optimize hyperparameters for this environment, like the learning rate, number of fully connected layers in the deep learning model, batch_size, gamma, tau and update_every;

- Implement a double DQN, a dueling DQN, and/or prioritized experience replay, for enhanced learning.

# load the weights from file
agent.qnetwork_local.load_state_dict(torch.load('checkpoint.pth'))
eps = hyperparameters['eps_start']
env_info = env.reset(train_mode=True)[brain_name] # reset the environment
state = env_info.vector_observations[0]            # get the current state
score = 0                                          # initialize the score
for t in range(1000):
    action = agent.act(state, eps)                 # select an action
    env_info = env.step(action)[brain_name]        # send the action to the environment
    next_state = env_info.vector_observations[0]   # get the next state
    reward = env_info.rewards[0]                   # get the reward
    done = env_info.local_done[0]                  # see if episode has finished
    score += reward                                # update the score
    state = next_state                             # roll over the state to next time step
    eps = max(hyperparameters['eps_end'],eps*hyperparameters['eps_decay'])
    if done:                                       # exit loop if episode finished
        break

print("Score: {}".format(score))
