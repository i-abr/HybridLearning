import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import random
from .jacobian import jacobian


class ModelOptimizer(object):

    def __init__(self, model, replay_buffer, lr=1e-2, eps=1e-1, lam=0.2, expert_data=None):

        # reference the model and buffer
        self.model          = model
        self.replay_buffer  = replay_buffer
        # set the model optimizer
        self.model_optimizer  = optim.Adam(self.model.parameters(), lr=lr)
        # logger
        self._eps = eps
        self._lam = lam
        self.log = {'loss' : [], 'rew_loss': [], 'model_loss' : []}

        self.expert_data = None

        if expert_data is not None:
            self.expert_data = expert_data


    def update_model(self, batch_size, mini_iter=1, verbose=False):
        if batch_size > len(self.replay_buffer):
            batch_size = len(self.replay_buffer)
        for k in range(mini_iter):
            states, actions, rewards, next_states, next_actions, done = self.replay_buffer.sample(batch_size)

            # if self.expert_data is not None:
            #     batch = random.sample(self.expert_data, batch_size)
            #     e_states, e_actions, e_rewards, e_next_states, e_next_actions, e_done \
            #                 =  map(np.stack, zip(*batch))
            #
            #     states = np.concatenate([states, e_states], axis=0)
            #     actions = np.concatenate([actions, e_actions], axis=0)
            #     rewards = np.concatenate([rewards, e_rewards], axis=0)
            #     next_states = np.concatenate([next_states, e_next_states], axis=0)
            #     next_actions = np.concatenate([next_actions, e_next_actions], axis=0)
            #     done = np.concatenate([done, e_done], axis=0)

            states = torch.FloatTensor(states)
            # states.requires_grad = True
            next_states = torch.FloatTensor(next_states)
            actions = torch.FloatTensor(actions)
            next_action = torch.FloatTensor(next_actions)
            rewards = torch.FloatTensor(rewards).unsqueeze(1)
            done    = torch.FloatTensor(np.float32(done)).unsqueeze(1)

            pred_mean, pred_std, pred_rew = self.model(states, actions)

            state_dist = Normal(pred_mean, pred_std)

            # df = jacobian(pred_mean, states)

            next_vals = self.model.reward_fun(torch.cat([next_states, next_action], axis=1))

            rew_loss = torch.mean(torch.pow((rewards+self._lam*(1-done)*next_vals).detach() - pred_rew,2))
            # rew_loss = torch.mean(torch.pow(rewards - pred_rew,2))

            model_loss = -torch.mean(state_dist.log_prob(next_states))# + 1e-2 * torch.norm(df, dim=[1,2]).mean()
            # - 1e-3*pred_next_state_dist.entropy().mean()

            loss = 0.5 * rew_loss + model_loss

            self.model_optimizer.zero_grad()
            loss.backward()
            self.model_optimizer.step()
            self.log['loss'].append(loss.item())
            self.log['model_loss'].append(model_loss.item())
            self.log['rew_loss'].append(rew_loss.item())
            if verbose:
                if k % 10 == 0:
                    print('epoch', k, 'loss {}, model loss {}, rew loss {}'.format(
                        self.log['loss'][-1], self.log['model_loss'][-1], self.log['rew_loss'][-1]
                    ))

class MDNModelOptimizer(object):

    def __init__(self, model, replay_buffer, lr=1e-2):

        # reference the model and buffer
        self.model          = model
        self.replay_buffer  = replay_buffer

        # set the model optimizer
        self.model_optimizer  = optim.Adam(self.model.parameters(), lr=lr)

        # logger
        self.log = {'loss' : [], 'rew_loss': []}

    def update_model(self, batch_size, mini_iter=1):

        for k in range(mini_iter):
            states, actions, rewards, next_states, done = self.replay_buffer.sample(batch_size)

            states = torch.FloatTensor(states)
            states.requires_grad = True

            next_states = torch.FloatTensor(next_states)
            actions = torch.FloatTensor(actions)
            rewards = torch.FloatTensor(rewards).unsqueeze(1)
            done    = torch.FloatTensor(np.float32(done)).unsqueeze(1)

            log_probs, pred_rewards = self.model(states, actions, next_states)

            df = jacobian(log_probs, states)


            next_value = self.model.predict_reward(next_states)

            #rew_loss = torch.mean(torch.pow(pred_rewards - rewards,2))
            rew_loss = torch.mean(torch.pow((rewards+(1-done)*0.99*next_value).detach()-pred_rewards,2))
            model_loss = -torch.mean(log_probs) + 1e-2 * torch.norm(df, dim=[1,2]).mean()

            loss = 0.5 * rew_loss + model_loss

            self.model_optimizer.zero_grad()
            loss.backward()
            self.model_optimizer.step()

        self.log['loss'].append(loss.item())
        self.log['rew_loss'].append(rew_loss.item())
