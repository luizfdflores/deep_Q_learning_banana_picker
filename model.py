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

    def forward(self, state):
        """Build a network that maps state -> action values."""
        return self.fc_layers(state)
