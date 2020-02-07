import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

class Model(nn.Module):

    def __init__(self, num_states, num_actions,
                 layers=[200, 128], eps=3e-3):

        super(Model, self).__init__()
        self.num_states  = num_states
        self.num_actions = num_actions

        self.mean = nn.Sequential(
            nn.Linear(num_states+num_actions, layers[0]),
            nn.ReLU(),
            nn.Linear(layers[0], layers[1]),
            nn.ReLU(),
            nn.Linear(layers[1], num_states)
        )

        self.reward_fun = nn.Sequential(
            nn.Linear(num_states+num_actions, layers[0]),
            nn.ReLU(),
            nn.Linear(layers[0], layers[1]),
            nn.ReLU(),
            nn.Linear(layers[1], 1)
        )

        self.log_std = nn.Parameter(torch.randn(1, num_states) * eps)


    def forward(self, s, a):
        x       = torch.cat([s, a], axis=1)
        dx      = self.mean(x)
        std     = torch.clamp(self.log_std, -20., 2).exp().expand_as(s)
        return s+dx, std, self.reward_fun(torch.cat([s, a], axis=1))

    def step(self, x, u):
        mean, std, rew = self.forward(x, u)
        return mean, rew
