import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

class Model(nn.Module):
    """
    Class creates a model for the dynamics system
    a model of the reward function is also created
    TODO: make this a MDN so that the model can generalize better
    ### Words of matt; machine teaching
    """
    def __init__(self, num_states, num_actions,
                 hidden_size=128, std=0.):

        super(Model, self).__init__()
        self.num_states  = num_states
        self.num_actions = num_actions

        self.mean = nn.Sequential(
            nn.Linear(num_states+num_actions, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_states)
        )

        self.reward_fun = nn.Sequential(
            nn.Linear(num_states+num_actions, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

        self.log_std = nn.Parameter(torch.ones(1, num_states) * std)


    def forward(self, s, a):
        x   = torch.cat([s, a], axis=1)
        dx = self.mean(x)
        std = torch.clamp(self.log_std, -20., 2).exp().expand_as(s)
        return s+dx, std, self.reward_fun(x)

    def step(self, x, u):
        mean, std, rew = self.forward(x, u)
        return mean, rew
