import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal


_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class _AF(nn.Module):
    def __init__(self):
        super(_AF, self).__init__()
    def forward(self, x):
        return torch.sin(x)
class PolicyNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, init_w=3e-3, log_std_min=-5, log_std_max=2, AF=None):
        super(PolicyNetwork, self).__init__()
        self.a_dim = num_actions
        '''
        For the policy, we parameterize a normal distribution with a mean
        function defined as a single layer network with sin(x) nonlayerity with 128 nodes
        (similar to the dynamics model used). The diagonal of the variance is specified
        using a single layer with 128 nodes and rectifying layer unit activation function.
        '''
        if AF == 'sin':
            self.sin = True
            self.mu = nn.Sequential(
                nn.Linear(num_inputs, hidden_size), _AF(),
                # nn.Linear(hidden_size, hidden_size),_AF(),
                nn.Linear(hidden_size, num_actions)
            )
            self.log_std = nn.Sequential(
                nn.Linear(num_inputs, hidden_size), nn.ReLU(),
                # nn.Linear(hidden_size, hidden_size), nn.ReLU(),
                nn.Linear(hidden_size, num_actions)
            )            # initialize weights
            # initialize weights
            self.mu[-1].weight.data.uniform_(-init_w, init_w)
            self.mu[-1].bias.data.uniform_(-init_w, init_w)
            self.log_std[-1].weight.data.uniform_(-init_w, init_w)
            self.log_std[-1].bias.data.uniform_(-init_w, init_w)
        else:
            self.sin = False
            self.mu = nn.Sequential(
                nn.Linear(num_inputs, hidden_size), nn.ReLU(), #_AF(),
                # nn.Linear(hidden_size, hidden_size), nn.ReLU(),#_AF(),
                nn.Linear(hidden_size, num_actions*2)
            )
            # initialize weights
            self.mu[-1].weight.data.uniform_(-init_w, init_w)
            self.mu[-1].bias.data.uniform_(-init_w, init_w)

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        #
        # self.layer1 = nn.Linear(num_inputs, hidden_size)
        # self.layer2 = nn.Linear(hidden_size, hidden_size)
        #
        # self.mean_layer = nn.Linear(hidden_size, num_actions)
        # self.mean_layer.weight.data.uniform_(-init_w, init_w)
        # self.mean_layer.bias.data.uniform_(-init_w, init_w)
        #
        # self.log_std_layer1 = nn.Linear(hidden_size, hidden_size)
        # self.log_std_layer2 = nn.Linear(hidden_size, num_actions)
        #
        # self.log_std_layer2.weight.data.uniform_(-init_w, init_w)
        # self.log_std_layer2.bias.data.uniform_(-init_w, init_w)
        # self.log_std_layer.weight.data.zero_()
        # self.log_std_layer.bias.data.zero_()

    def forward(self, state):
        if self.sin:
            mu = self.mu(state)
            log_std = self.log_std(state)
        else:
            out = self.mu(state)
            mu, log_std = torch.split(out, [self.a_dim, self.a_dim], dim=1)
        # x = F.relu(self.layer1(state))
        # x = F.relu(self.layer2(x))
        # mean    = self.mean_layer(x)
        # log_std = self.log_std_layer2(F.relu(self.log_std_layer1(x)))
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
