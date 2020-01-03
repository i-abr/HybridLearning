import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal

class Policy(nn.Module):
    def __init__(self, num_states, num_actions, hidden_size=64,
                    init_w=3e-3, log_std_min=-20, log_std_max=2):
        super(Policy, self).__init__()

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.mean = nn.Sequential(
            nn.Linear(num_states, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_actions)
        )
        self.log_std = nn.Sequential(
            nn.Linear(num_states, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_actions)
        )

    def forward(self, state):
        mean = self.mean(state)
        std = torch.clamp(self.log_std(state), self.log_std_min, self.log_std_max).exp()

        return mean, std

    def get_action(self, state):
        mean, std = self.forward(torch.FloatTensor(state).unsqueeze(0))
        return Normal(mean, std).sample().detach().numpy()[0]
