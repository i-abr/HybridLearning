import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal

class PolicyNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size=[200,128],
                        init_w=3e-3, log_std_min=-20, log_std_max=2):
        super(PolicyNetwork, self).__init__()

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.linear1 = nn.Linear(num_inputs, hidden_size[0])
        self.linear2 = nn.Linear(hidden_size[0], hidden_size[1])

        self.var_linear1 = nn.Linear(num_inputs, hidden_size[0])
        self.var_linear2 = nn.Linear(hidden_size[0], hidden_size[1])

        self.mean_linear = nn.Linear(hidden_size[1], num_actions)

        # self.mean_linear.weight.data.uniform_(-init_w, init_w)
        # self.mean_linear.bias.data.uniform_(-init_w, init_w)

        self.log_std_linear = nn.Linear(hidden_size[1], num_actions)
        # self.log_std_linear.weight.data.uniform_(-init_w, init_w)
        # self.log_std_linear.bias.data.uniform_(-init_w, init_w)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        # x = torch.sin(self.linear1(state))
        var = F.relu(self.var_linear1(state))
        var = F.relu(self.var_linear2(var))

        mean    = torch.tanh(self.mean_linear(x))
        log_std = self.log_std_linear(var)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        return mean, log_std

    # def evaluate(self, state, epsilon=1e-6):
    #     mean, log_std = self.forward(state)
    #     std = log_std.exp()
    #
    #     normal = Normal(mean, std)
    #     z = normal.sample()
    #     action = torch.tanh(z)
    #
    #     log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + epsilon)
    #     log_prob = log_prob.sum(-1, keepdim=True)
    #
    #     return action, log_prob, z, mean, log_std


    # def get_action(self, state):
    #     state = torch.FloatTensor(state).unsqueeze(0)
    #     mean, log_std = self.forward(state)
    #     std = log_std.exp()
    #
    #     normal = Normal(mean, std)
    #     action = normal.sample()
    #
    #     action = action.detach().cpu().numpy()
    #     return action[0]
