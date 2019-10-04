import torch
import torch.nn as nn
import torch.nn.functional as F

class Network(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, n_fc_layers):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
        """
        super(Network, self).__init__()
        modules = []
        for i in range(1, n_fc_layers):
            in_features = state_size if i == 1 else action_size*(2**(n_fc_layers-i+1))
            out_features = action_size*(2**(n_fc_layers-i))
            modules.append(nn.Linear(in_features, out_features))
            modules.append(nn.ReLU())
        
        modules.append(nn.Linear(action_size*2, action_size))
          
        self.fc_layers = nn.Sequential(*modules)
            
        #self.fc1 = nn.Linear(state_size, action_size*16)
        #self.fc2 = nn.Linear(action_size*16, action_size*8)
        #self.fc3 = nn.Linear(action_size*8, action_size*4)
        #self.fc4 = nn.Linear(action_size*4, action_size*2)
        #self.fc5 = nn.Linear(action_size*2, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        #x = F.relu(self.fc1(state))
        #x = F.relu(self.fc2(x))
        #x = F.relu(self.fc3(x))
        #x = F.relu(self.fc4(x))
        #return self.fc5(x)
        return self.fc_layers(state)