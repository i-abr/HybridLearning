import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.autograd import Variable

class Policy(nn.Module):

    def __init__(self, num_states, num_actions, init_w=3e-3,
                 layers=[128, 64], std=0.0):

        super(Policy, self).__init__()

        # self.mean = nn.Parameter(torch.randn(1, num_states))
        # self.std = nn.Parameter(torch.randn(1, num_states))


        self.ml1 = nn.Linear(num_states, layers[0])
        self.ml2 = nn.Linear(layers[0], layers[1])
        self.ml3 = nn.Linear(layers[1], num_actions)

        self.ml3.weight.data.uniform_(-init_w, init_w)
        self.ml3.bias.data.uniform_(-init_w, init_w)


        self.vl1 = nn.Linear(num_states, layers[0])
        self.vl2 = nn.Linear(layers[0], layers[1])
        self.vl3 = nn.Linear(layers[1], num_actions)

        self.vl3.weight.data.uniform_(-init_w, init_w)
        self.vl3.bias.data.uniform_(-init_w, init_w)



    def get_action(self, state):
        s = torch.FloatTensor(state).unsqueeze(0)
        mu, log_std = self.forward(s)
        pi = Normal(mu, log_std.exp())
        return pi.sample().numpy()[0]
        # return mu.detach().numpy()[0]

    # def forward(self, s, fisher_inf=False):
    #     # a = (s - self.mean.expand_as(s)).div(self.std.expand_as(s))
    #     a = s
    #     a = torch.tanh(self.ml1(a))
    #     a = torch.tanh(self.ml2(a))
    #     a = self.ml3(a)
    #
    #     # log_std = (s - self.mean.expand_as(s)).div(self.std.expand_as(s))
    #     log_std = s
    #     log_std = torch.tanh(self.vl1(log_std))
    #     log_std = torch.tanh(self.vl2(log_std))
    #     log_std = self.vl3(log_std)
    #
    #     # log_std = torch.clamp(self.log_std.expand_as(a), -20.,2.)
    #     log_std = torch.clamp(log_std, -5.,2.)
    #
    #     return a, log_std
    def forward(self, s):
        a = s
        a = F.relu(self.ml1(a))
        a = F.relu(self.ml2(a))
        a = self.ml3(a)

        log_std = s
        log_std = F.relu(self.vl1(log_std))
        log_std = F.relu(self.vl2(log_std))
        log_std = self.vl3(log_std)

        # log_std = torch.clamp(self.log_std.expand_as(a), -20.,2.)
        log_std = torch.clamp(log_std, -4.,2.)

        return a, log_std
