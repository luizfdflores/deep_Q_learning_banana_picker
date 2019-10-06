# Project 1 - Deep Reinforcement Learning
# Deep Q Learning - Banana Picker

[//]: # (Image References)

[image1]: https://github.com/luizfdflores/deep_Q_learning_banana_picker/blob/master/images/agent_unity.gif?raw=true "Trained Agent"

### Introduction

This project was developed in October/2019 as the first project for the Deep Reinforcement Learning Nanodegree from Udacity.

The objective of this project was to train an agent to play a banana picking game, using Deep Q Learning.

### Project Details

In this game, the agent have to pick as many yellow bananas as posible, while avoiding blue bananas.

![Trained Agent][image1]
https://youtu.be/l16UTAzm0h0

To do so, it can pick one of four discrete actions:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The information given to the agent to learn how to navigate this environment is the state space, which has 37 dimensions, containing the agent's velocity and ray-based perception of objects around the agent's forward direction.

To win in this episodic task, the agent must get an average score of +13 over 100 consecutive episodes.

### Getting Started

1. Clone the repository

2. Install the depedencies
```
pip install -r requirements.txt
```

3. Run the file Navigation.ipynb

### Instructions

- Follow the flow in the `Navigation_Deep_Q_Learning.ipynb` notebook to get started with training your own agent!
- You can tweek with the learning by changing the values for the hyperparameters in the `hyp` dictionary, in the section `Defining Hyperparameters` in the notebook.
