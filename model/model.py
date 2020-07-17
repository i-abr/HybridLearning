import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

class Model(nn.Module):
    def __init__(self, num_states, num_actions,
                 def_layers=[200, 200], std=0.):

        super(Model, self).__init__()
        self.num_states  = num_states
        self.num_actions = num_actions

        self.mu = nn.Sequential(
            nn.Linear(num_states+num_actions, def_layers[0]),
            nn.ReLU(),
            nn.Linear(def_layers[0], def_layers[0]),
            nn.ReLU(),
            nn.Linear(def_layers[0], num_states)
        )

        self.reward_fun = nn.Sequential(
            nn.Linear(num_states+num_actions, def_layers[0]),
            nn.ReLU(),
            nn.Linear(def_layers[0], def_layers[0]),
            nn.ReLU(),
            nn.Linear(def_layers[0], 1)
        )

        self.log_std = nn.Parameter(torch.randn(1, num_states) * std)

    def forward(self, s, a):
        """
        dx, rew = forward(s, a)
        dx is the change in the state
        """
        _in   = torch.cat([s, a], axis=1)

        x = self.mu(_in)
        # in case I want to update the way the var looks like
        std = torch.clamp(self.log_std, -10., 2).exp().expand_as(x)

        return x+s, std, self.reward_fun(_in)

    def step(self, x, u):
        mean, std, rew = self.forward(x, u)
        return mean, rew
