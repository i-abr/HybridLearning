import numpy as np
from copy import deepcopy

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from valuenetwork import ValueNetwork



class DDPG(object):

    def __init__(self, policy, state_dim, action_dim, replay_buffer,
                    hidden_dim=256,
                    value_lr=1e-3,
                    policy_lr=1e-4
                ):

        self.value_net  = ValueNetwork(state_dim, action_dim, hidden_dim)
        self.policy_net = policy

        self.target_value_net  = ValueNetwork(state_dim, action_dim, hidden_dim)
        self.target_policy_net = deepcopy(policy)

        for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_(param.data)

        for target_param, param in zip(self.target_policy_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(param.data)

        self.value_optimizer  = optim.Adam(self.value_net.parameters(),  lr=value_lr)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=policy_lr)

        self.value_criterion = nn.MSELoss()

        self.replay_buffer = replay_buffer

        self.log = {
            'policy_loss' : [],
            'value_loss' : []
        }


    def update(self, batch_size,
                           gamma = 0.99,
                           min_value=-np.inf,
                           max_value=np.inf,
                           soft_tau=1e-2
                           ):

        state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)

        state      = torch.FloatTensor(state)
        next_state = torch.FloatTensor(next_state)
        action     = torch.FloatTensor(action)
        reward     = torch.FloatTensor(reward).unsqueeze(1)
        done       = torch.FloatTensor(np.float32(done)).unsqueeze(1)

        policy_loss = self.value_net(state, self.policy_net(state))
        policy_loss = -policy_loss.mean()

        next_action    = self.target_policy_net(next_state)
        target_value   = self.target_value_net(next_state, next_action.detach())
        expected_value = reward + (1.0 - done) * gamma * target_value
        expected_value = torch.clamp(expected_value, min_value, max_value)

        value = self.value_net(state, action)
        value_loss = self.value_criterion(value, expected_value.detach())


        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
                target_param.data.copy_(
                    target_param.data * (1.0 - soft_tau) + param.data * soft_tau
                )

        for target_param, param in zip(self.target_policy_net.parameters(), self.policy_net.parameters()):
                target_param.data.copy_(
                    target_param.data * (1.0 - soft_tau) + param.data * soft_tau
                )

        self.log['policy_loss'].append(policy_loss.item())
        self.log['value_loss'].append(value_loss.item())
