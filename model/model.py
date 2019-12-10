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
                 def_layers=[200, 200], std=0.):

        super(Model, self).__init__()
        self.num_states  = num_states
        self.num_actions = num_actions
        layers = [num_states + num_actions] + def_layers + [num_states]

        self.n_params = []
        for i, (insize, outsize) in enumerate(zip(layers[:-1], layers[1:])):
            var = 'layer' + str(i)
            setattr(self, var, nn.Linear(insize, outsize))
            self.n_params.append(i)



        layers = [num_states + num_actions] + def_layers + [1]
        for i, (insize, outsize) in enumerate(zip(layers[:-1], layers[1:])):
            var = 'rew_layer' + str(i)
            setattr(self, var, nn.Linear(insize, outsize))

        #self.log_std = nn.Parameter(torch.ones(1, num_states) * std)

        self.log_std = nn.Sequential(
            nn.Linear(def_layers[-1], def_layers[-1]),
            nn.ReLU(),
            nn.Linear(def_layers[-1], num_states)
        )


    def forward(self, s, a):
        """
        dx, rew = forward(s, a)
        dx is the change in the state
        """
        x   = torch.cat([s, a], axis=1)
        rew = torch.cat([s, a], axis=1)
        for i in self.n_params[:-1]:
            w = getattr(self, 'layer' + str(i))
            x = w(x)
            x = F.relu(x)
            #x = torch.sin(x)

            w = getattr(self, 'rew_layer' + str(i))
            rew = w(rew)
            rew = F.relu(rew)
            #rew = torch.sin(rew)

        std = torch.clamp(self.log_std(x), -20., 2.).exp()

        w = getattr(self, 'layer' + str(self.n_params[-1]))
        x = w(x)

        w = getattr(self, 'rew_layer' + str(self.n_params[-1]))
        rew = w(rew)

        #std = torch.clamp(self.log_std, -20., 2).exp().expand_as(x)

        dist = Normal(s+x, std)

        return dist, rew

    def step(self, x, u):
        dist, rew = self.forward(x, u)
        return dist.mean, rew
