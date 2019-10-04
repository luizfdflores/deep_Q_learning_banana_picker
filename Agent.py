import numpy as np
import random
from collections import namedtuple, deque
from model import Network
import torch
import torch.nn.functional as F
import torch.optim as optim

#HYPERPARAMETERS

BUFFER_SIZE = int(1e5) 
BATCH_SIZE = 64
GAMMA = 0.99
TAU = 1e-3
LR = 5e-4
UPDATE_EVERY = 4

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class agent():
    
    def __init__(self, state_size, action_size, hyperparameters, seed):
        
        self.buffer_size = hyperparameters['buffer_size']
        self.batch_size = hyperparameters['batch_size']
        self.gamma = hyperparameters['gamma']
        self.tau = hyperparameters['tau']
        self.lr = hyperparameters['lr']
        self.update_every = hyperparameters['update_every']
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        
        self.network_local = Network(state_size, action_size, hyperparameters['n_fc_layers']).to(device)
        self.network_target = Network(state_size, action_size, hyperparameters['n_fc_layers']).to(device)
        
        self.optimizer = optim.Adam(self.network_local.parameters(), lr=self.lr)
        self.memory = ReplayBuffer(action_size, self.buffer_size, self.batch_size, seed)
        self.t_step = 0
    
    def step(self, state, action, reward, next_state, done):
        #Adds experience to the memory
        
        self.memory.add(state, action, reward, next_state, done)
        
        self.t_step += 1
        
        if self.t_step % self.update_every == 0:
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample()
                self.learn(experiences, self.gamma)
    
    def act(self, state, eps):
        # Returns action given the state, according to the current policy
        
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.network_local.eval()
        with torch.no_grad():
            action_values = self.network_local(state)
        self.network_local.train()
        
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))
    
    def learn(self, experiences, gamma):
        # Update the network with batch from experiences
        
        states, actions, rewards, next_states, dones = experiences
        
        Q_targets_next = self.network_target(next_states).detach().max(1)[0].unsqueeze(1)
        Q_targets = rewards + (gamma * Q_targets_next * (1-dones))
        
        Q_expected = self.network_local(states).gather(1, actions)
        
        loss = F.mse_loss(Q_expected, Q_targets)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.soft_update(self.network_local, self.network_target, self.tau)
        
    def soft_update(self, local_model, target_model, tau):
        
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
    
class ReplayBuffer():
    
    def __init__(self, action_size, buffer_size, batch_size, seed):
        
        self.action_size = action_size
        self.batch_size = batch_size
        self.seed = random.seed(seed)
        
        self.memory = deque(maxlen=buffer_size)
        self.experience = namedtuple('Experience', field_names=['state','action','reward','next_state','done'])
    
    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)
        
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.memory)