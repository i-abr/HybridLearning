import torch # package for building functions with learnable parameters
import torch.nn as nn # prebuilt functions specific to neural networks
import torch.nn.functional as F
from torch.distributions import Normal

import numpy as np

def gumbel_sample(x, axis=1):
    z = np.random.gumbel(loc=0, scale=1, size=x.shape)
    return (np.log(x) + z).argmax(axis=axis)

class MDNModel(nn.Module):
    def __init__(self, num_features, num_inputs, num_outputs, default_layers=[128], scale=1.0):
        super(MDNModel, self).__init__()

        self.num_features   = num_features
        self.num_inputs     = num_inputs
        self.num_outputs    = num_outputs

        self.mu         = nn.Linear(num_inputs, num_outputs * num_features, bias=True)
        self.log_std    = nn.Parameter(torch.ones(1, num_features * num_outputs))
        self.pi         = nn.Sequential(
                            nn.Linear(num_inputs, default_layers[0]),
                            nn.ReLU(),
                            nn.Linear(default_layers[0], num_features)
                        )

    def forward(self, s, predict=False):

        # TODO: get rid of this later
        # _input = torch.cat([s, a], axis=1)
        _input = s
        pi = F.softmax(self.pi(_input), -1)

        if predict is True:
            k = gumbel_sample(pi.data.numpy() + 1e-5)
            idx = (np.arange(k.shape[0]), k)
            mu  = self.mu(_input).view(-1, self.num_features, self.num_outputs)
            return mu[idx]

        mu  = self.mu(_input)
        mu  = mu.view(-1, self.num_features, self.num_outputs)
        std = torch.exp(self.log_std)
        std  = std.view(-1, self.num_features, self.num_outputs)

        return pi, mu, std
