import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class PolicyNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, init_w=3e-3):
        super(PolicyNetwork, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_size)
        # self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, num_actions)

        # self.linear3.weight.data.uniform_(-init_w, init_w)
        # self.linear3.bias.data.uniform_(-init_w, init_w)

    def forward(self, state):
        # x = F.relu(self.linear1(state))
        # x = F.relu(self.linear2(x))
        x = torch.sin(self.linear1(state))
        #x = torch.sin(self.linear2(x))
        x = torch.tanh(self.linear3(x))
        return x

    def get_action(self, state):
        state  = torch.FloatTensor(state).unsqueeze(0)
        action = self.forward(state)
        return action.detach().cpu().numpy()[0]
