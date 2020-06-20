import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


class Sin(nn.Module):
    def __init__(self):
        super(Sin, self).__init__()
    def forward(self, x):
        return torch.sin(x)

class Model(nn.Module):
    """
    Class creates a model for the dynamics system
    a model of the reward function is also created
    TODO: make this a MDN so that the model can generalize better
    ### Words of matt; machine teaching
    """
    def __init__(self, num_states, num_actions,
                 def_layers=[200, 200], std=0.):

        super(Model, self).__init__()
        self.num_states  = num_states
        self.num_actions = num_actions

        self.mu = nn.Sequential(
            nn.Linear(num_states+num_actions, def_layers[0]), Sin()
            nn.Linear(def_layers[0], def_layers[0]), Sin(),
            nn.Linear(def_layers[0], num_states)
        )

        self.log_std = nn.Parameter(torch.randn(1, num_states) * std)

    def forward(self, s, a):
        """
        dx, rew = forward(s, a)
        dx is the change in the state
        """
        x   = torch.cat([s, a], axis=1)

        x = self.mu(x)
        # in case I want to update the way the var looks like
        std = torch.clamp(self.log_std, -10., 2).exp().expand_as(x)

        return x+s, std

    def step(self, x, u):
        mean, std = self.forward(x, u)
        return mean
