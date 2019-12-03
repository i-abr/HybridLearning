import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal


class ModelOptimizer(object):

    def __init__(self, model, replay_buffer, lr=1e-2):

        # reference the model and buffer
        self.model          = model
        self.replay_buffer  = replay_buffer

        # set the model optimizer
        self.model_optimizer  = optim.Adam(self.model.parameters(), lr=lr)

        # logger
        self.log = {'loss' : []}

    def update_model(self, batch_size, mini_iter=1):

        for k in range(mini_iter):
            states, actions, rewards, next_states, done = self.replay_buffer.sample(batch_size)

            states = torch.FloatTensor(states)
            next_states = torch.FloatTensor(next_states)
            actions = torch.FloatTensor(actions)
            rewards = torch.FloatTensor(rewards).unsqueeze(1)
            done    = torch.FloatTensor(np.float32(done)).unsqueeze(1)

            pred_next_state_dist, pred_rewards = self.model(states, actions)

            rew_loss = torch.mean(torch.pow(pred_rewards - rewards,2))
            model_loss = -torch.mean(pred_next_state_dist.log_prob(next_states)) - 1e-3*pred_next_state_dist.entropy().mean()

            loss = 0.5 * rew_loss + model_loss

            self.model_optimizer.zero_grad()
            loss.backward()
            self.model_optimizer.step()

        self.log['loss'].append(loss.item())

class MDNModelOptimizer(object):

    def __init__(self, model, replay_buffer, lr=1e-2):

        # reference the model and buffer
        self.model          = model
        self.replay_buffer  = replay_buffer

        # set the model optimizer
        self.model_optimizer  = optim.Adam(self.model.parameters(), lr=lr)

        # logger
        self.log = {'loss' : []}

    def update_model(self, batch_size, mini_iter=1):

        for k in range(mini_iter):
            states, actions, rewards, next_states, done = self.replay_buffer.sample(batch_size)

            states = torch.FloatTensor(states)
            next_states = torch.FloatTensor(next_states)
            actions = torch.FloatTensor(actions)
            rewards = torch.FloatTensor(rewards).unsqueeze(1)
            done    = torch.FloatTensor(np.float32(done)).unsqueeze(1)

            log_probs, pred_rewards = self.model(states, actions, next_states)

            rew_loss = torch.mean(torch.pow(pred_rewards - rewards,2))
            model_loss = -torch.mean(log_probs)

            loss = 0.5 * rew_loss + model_loss

            self.model_optimizer.zero_grad()
            loss.backward()
            self.model_optimizer.step()

        self.log['loss'].append(loss.item())
