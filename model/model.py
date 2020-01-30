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
                 def_layers=[200, 200], std=0., init_w=3e-3):

        super(Model, self).__init__()
        self.num_states  = num_states
        self.num_actions = num_actions
        layers = [num_states + num_actions] + def_layers + [num_states]

        # self.n_params = []
        # for i, (insize, outsize) in enumerate(zip(layers[:-1], layers[1:])):
        #     var = 'layer' + str(i)
        #     setattr(self, var, nn.Linear(insize, outsize))
        #     self.n_params.append(i)
        # modules = []
        # for i, (insize, outsize) in enumerate(zip(layers[:-1], layers[1:])):
        #     print(insize, outsize)
        #     modules.append(nn.Linear(insize, outsize))
        #     modules.append(nn.ReLU())
        # self.mean = nn.Sequential(*modules)
        self.linear1 = nn.Linear(num_states + num_actions, def_layers[0])
        # self.linear2 = nn.Linear(hidden_size, hidden_size)

        self.mean_linear = nn.Linear(def_layers[0], num_states)
        self.mean_linear.weight.data.uniform_(-init_w, init_w)
        self.mean_linear.bias.data.uniform_(-init_w, init_w)


        self.reward_fun = nn.Sequential(
            nn.Linear(num_states+num_actions, def_layers[0]),
            nn.ReLU(),
            nn.Linear(def_layers[0], def_layers[0]),
            nn.ReLU(),
            nn.Linear(def_layers[0], 1)
        )


        # layers = [num_states + num_actions] + def_layers + [1]
        # for i, (insize, outsize) in enumerate(zip(layers[:-1], layers[1:])):
        #     var = 'rew_layer' + str(i)
        #     setattr(self, var, nn.Linear(insize, outsize))

        self.log_std = nn.Parameter(torch.randn(1, num_states) * 3e-3)

        # self.log_std = nn.Sequential(
        #     nn.Linear(def_layers[-1], def_layers[-1]),
        #     nn.ReLU(),
        #     nn.Linear(def_layers[-1], num_states)
        # )


    def forward(self, s, a):
        """
        dx, rew = forward(s, a)
        dx is the change in the state
        """
        x   = torch.cat([s, a], axis=1)

        x = torch.sin(self.linear1(x))
        # self.linear2 = nn.Linear(hidden_size, hidden_size)

        x = self.mean_linear(x)


        # for i in self.n_params[:-1]:
        #     w = getattr(self, 'layer' + str(i))
        #     x = w(x)
        #     # x = F.relu(x)
        #     x = torch.sin(x)
        # std = torch.clamp(self.log_std(x), -20., 2.).exp()

        # w = getattr(self, 'layer' + str(self.n_params[-1]))
        # x = w(x)
        # dx = self.mean(x)

        # in case I want to update the way the var looks like
        std = torch.clamp(self.log_std, -20., 2).exp().expand_as(s)

        return s+x, std, self.reward_fun(torch.cat([s, a], axis=1))

    def step(self, x, u):
        mean, std, rew = self.forward(x, u)
        return mean, rew
