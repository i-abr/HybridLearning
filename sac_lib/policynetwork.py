import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal


_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# class _AF(nn.Module):
#     def __init__(self):
#         super(_AF, self).__init__()
#     def forward(self, x):
#         return torch.sin(x)
class PolicyNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, init_w=3e-3, log_std_min=-5, log_std_max=2, AF=None):
        super(PolicyNetwork, self).__init__()
        self.a_dim = num_actions
        '''
        For the policy, we parameterize a normal distribution with a mean
        function defined as a single layer network with sin(x) nonlinearity with 128 nodes
        (similar to the dynamics model used). The diagonal of the variance is specified
        using a single layer with 128 nodes and rectifying linear unit activation function.
        '''
        if AF == 'sin':
            self.sin = True
            self.linear1 = nn.Linear(num_inputs, hidden_size)
            self.mu_linear2 = nn.Linear(hidden_size, num_actions)
            self.log_std_linear2 = nn.Linear(hidden_size, num_actions)
        else:
            self.sin = False
            self.mu = nn.Sequential(
                nn.Linear(num_inputs, hidden_size), nn.ReLU(), #_AF(),
                # nn.Linear(hidden_size, hidden_size), nn.ReLU(),#_AF(),
                nn.Linear(hidden_size, num_actions*2)
            )

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        #
        # self.linear1 = nn.Linear(num_inputs, hidden_size)
        # self.linear2 = nn.Linear(hidden_size, hidden_size)
        #
        # self.mean_linear = nn.Linear(hidden_size, num_actions)
        # self.mean_linear.weight.data.uniform_(-init_w, init_w)
        # self.mean_linear.bias.data.uniform_(-init_w, init_w)
        #
        # self.log_std_linear1 = nn.Linear(hidden_size, hidden_size)
        # self.log_std_linear2 = nn.Linear(hidden_size, num_actions)
        #
        # self.log_std_linear2.weight.data.uniform_(-init_w, init_w)
        # self.log_std_linear2.bias.data.uniform_(-init_w, init_w)
        # self.log_std_linear.weight.data.zero_()
        # self.log_std_linear.bias.data.zero_()

    def forward(self, state):
        if self.sin:
            x = self.linear1(state)
            mu = self.mu_linear2(torch.sin(x))
            log_std = self.log_std_linear2(F.relu(x))
        else:
            out = self.mu(state)
            mu, log_std = torch.split(out, [self.a_dim, self.a_dim], dim=1)
        # x = F.relu(self.linear1(state))
        # x = F.relu(self.linear2(x))
        # mean    = self.mean_linear(x)
        # log_std = self.log_std_linear2(F.relu(self.log_std_linear1(x)))
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        return mu, log_std

    def evaluate(self, state, epsilon=1e-6):
        mean, log_std = self.forward(state)
        std = log_std.exp()

        normal = Normal(mean, std)
        z = normal.rsample()
        action = torch.tanh(z)

        log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + epsilon)
        log_prob = log_prob.sum(-1, keepdim=True)

        return action, log_prob, z, mean, log_std


    def get_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(_device)
        mean, log_std = self.forward(state)
        std = log_std.exp()

        normal = Normal(mean, std)
        z      = normal.sample()
        action = torch.tanh(z)

        action = action.detach().cpu().numpy()
        return action[0]
