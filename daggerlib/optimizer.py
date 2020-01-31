import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal

import random
from itertools import compress

class DAggerOptim(object):

    def __init__(self, policy, replay_buffer, lr=0.01):

        # reference the model and buffer
        self.policy         = policy
        self.replay_buffer  = replay_buffer

        # set the model optimizer
        self.optimizer  = optim.Adam(self.policy.parameters(), lr=lr)

        # logger
        self.log = {'loss' : []}

    def update_policy(self, batch_size, epochs=10):

        for k in range(epochs):
            for i in range(0, len(self.replay_buffer), batch_size):
                batch = self.replay_buffer.buffer[i:i+batch_size]
                state, action, nextstate, expert_action = map(np.stack, zip(*batch))

                state  = torch.FloatTensor(state)
                expert_action = torch.FloatTensor(expert_action)

                mu, log_std = self.policy(state)
                pi = Normal(mu, log_std.exp())
                clone_loss = -torch.mean(pi.log_prob(expert_action)) #- entropy_eps * torch.mean(pred_action_dist.entropy())

                self.optimizer.zero_grad()
                clone_loss.backward()
                self.optimizer.step()

            self.log['loss'].append(clone_loss.item())

class DAggerOnlineOptim(object):

    def __init__(self, policy, replay_buffer, lr=0.01):

        # reference the model and buffer
        self.policy         = policy
        self.replay_buffer  = replay_buffer

        # set the model optimizer
        self.optimizer  = optim.Adam(self.policy.parameters(), lr=lr)

        # logger
        self.log = {'loss' : []}

    def update_policy(self, batch_size, epochs=1):
        if batch_size > len(self.replay_buffer):
            batch_size = len(self.replay_buffer)
        for k in range(epochs):
            state, action, next_state, expert_action = self.replay_buffer.sample(batch_size)

            state  = torch.FloatTensor(state)
            expert_action = torch.FloatTensor(expert_action)

            mu, log_std = self.policy(state)
            pi = Normal(mu, log_std.exp())
            clone_loss = -torch.mean(pi.log_prob(expert_action)) #- entropy_eps * torch.mean(pred_action_dist.entropy())

            self.optimizer.zero_grad()
            clone_loss.backward()
            self.optimizer.step()

            self.log['loss'].append(clone_loss.item())
