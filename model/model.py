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
                 hidden_dim=256, std=0., init_w=3e-3):

        super(Model, self).__init__()
        self.num_states  = num_states
        self.num_actions = num_actions

        self.linear1 = nn.Linear(num_states + num_actions, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean_linear = nn.Linear(hidden_dim, num_states)

        self.mean_linear.weight.data.uniform_(-init_w, init_w)
        self.mean_linear.bias.data.uniform_(-init_w, init_w)


        self.reward_fun = nn.Sequential(
            nn.Linear(num_states+num_actions, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )


        self.log_std = nn.Parameter(torch.randn(1, num_states) * 3e-3)


    def forward(self, s, a):
        """
        dx, rew = forward(s, a)
        dx is the change in the state
        """
        x   = torch.cat([s, a], axis=1)

        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.mean_linear(x)

        # in case I want to update the way the var looks like
        std = torch.clamp(self.log_std, -10., 2.).exp().expand_as(s)

        return s + x, std, self.reward_fun(torch.cat([s, a], axis=1))

    def step(self, x, u):
        mean, std, rew = self.forward(x, u)
        return mean, rew
