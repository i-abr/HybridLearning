import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

class _AF(nn.Module):
    def __init__(self):
        super(_AF, self).__init__()
    def forward(self, x):
        return torch.sin(x)


class Model(nn.Module):
    def __init__(self, num_states, num_actions,
                 def_layers=[200, 200], std=0., AF=None):

        super(Model, self).__init__()
        self.num_states  = num_states
        self.num_actions = num_actions

        '''
        Model representation of dynamics is single layer network with sin(x) nonlinearity.
        For locomotion tasks, we use the rectifying linear unit (RELU) nonlinearity.
        '''
        if AF == 'sin':
            self.mu = nn.Sequential(
                nn.Linear(num_states+num_actions, def_layers[0]),
                _AF(),
                # nn.Linear(def_layers[0], def_layers[0]), _AF(),
                nn.Linear(def_layers[0], num_states)
            )
        else:
            self.mu = nn.Sequential(
                nn.Linear(num_states+num_actions, def_layers[0]),
                nn.ReLU(),
                # nn.Linear(def_layers[0], def_layers[0]), nn.ReLU(),
                nn.Linear(def_layers[0], num_states)
            )

        '''
        The reward function is modeled as a two layer network and
        RELU activation function.
        '''
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
