# Project 1 - Deep Reinforcement Learning
# Deep Q Learning - Banana Picker
# Report

[//]: # (Image References)

[image1]: https://github.com/luizfdflores/deep_Q_learning_banana_picker/blob/master/images/agent_unity.gif?raw=true "Trained Agent"   

##### Please check out README.md for additional information regarding the environment!

### Trained agent playing the game
Below you can check of the video of the trained agent playing the game. This agent was trained for 931 episodes and got an average score of +17 over 100 episodes. The weights for this model are avaiable on the checkpoint_17.0_931.pth file.

<center>

![Trained Agent][image1]

https://youtu.be/l16UTAzm0h0

</center>

### The Learning Algorithm
For learning how to play the game, the Agent uses a Deep Q-Learning Network (DQN), which approximates the Action-value function with a deep neural network. This means that, given the state as input, the Neural Network (in this case, a simple multi fully-connected layer network) returns the agent's action.

The learning Algorithm uses two techniques, for a stable and optimized training, which are Experience Replay and Fixed Q-Targets and will be discussed below.

#### Experience Replay
When training the agent, the observations that an agent recieves are related to the previous observation. This correlation affects the network convergence, since this correlation would be encoded into the network. To break this, the agent learns from a batch of observations that are independant and uniformly distributed. By using a experience buffer that holds the last 100000 observations, which contains the current state, action, reward and next state, the agent learning from a batch of random selected observations from this buffer, while, at every step, the current observation is added to the buffer.

#### Fixed Q-Targets
When the agent begins training, the neural network is started with random weights, which leads to high oscilations in the values, during the first episodes. Since, in this episodes, the agent hasn't learned enough information to find the next best action, convergence can be troublesome. To mitigate this, we use not one, but two different neural networks, with the same architecute, to train the agent, synchronising both periodically. When the agent chooses the next action, it uses the active network, while the action is evaluated using the  fixed network.

### Hyperparameters
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

### Neural network Model
Only fully connected layers were used in the model for this agent. I ran the agent for 200 episodes, using different numbers of hidden layers in the network. In the final version of the layers can be viewed below.

<center>

| Layer | Input Size | Output Size |
| ----- | ---------- | ----------- |  
|Fully Connected Layer 1 + ReLU | 37 | 128|
|Fully Connected Layer 2 + ReLU | 64 | 32 |
|Fully Connected Layer 3 + ReLU | 32 | 16 |
|Fully Connected Layer 4 + ReLU | 16 | 8 |
|Fully Connected Layer 5 | 8 | 4 |

</center>

### Training

<center>

![Rewards Plot](images/img_training_score.png?raw=true "Title")

</center>

The agent was trained for 2000 episodes, which took almost 49 minutes. The goal, an average score over 100 episodes of +13 was achieved after 429 episodes. Until the end of the 2000 episodes, every time the goal was hit, it was increased by 0.5. At the end of the training, the highest goal that was achieved was 17.0 after 931 episodes.

### Ideas for Future Work

- Optimize hyperparameters for this environment, like the learning rate, number of fully connected layers in the deep learning model, batch_size, gamma, tau and update_every;

- Implement a double DQN, a dueling DQN, and prioritized experience replay, for enhanced learning.
