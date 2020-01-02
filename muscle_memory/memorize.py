import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from .jacobian import jacobian

class Memorize(object):


    def __init__(self, model, policy, replay_buffer,
                        policy_lr=3e-4,
                        model_lr=3e-4
                    ):

        self.model_net      = model
        self.policy_net     = policy
        self.replay_buffer  = replay_buffer
        self.log            = {'policy_loss' : [], 'model_loss' : []}

        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=policy_lr)

        self.model_optimizer = optim.Adam(self.model_net.parameters(), lr=model_lr)

    def update(self, batch_size, entropy_eps = 1e-3, mini_iter=1):

        for k in range(mini_iter):
            state, action, reward, next_state, next_action, done = self.replay_buffer.sample(batch_size)

            state      = torch.FloatTensor(state)
            state.requires_grad = True
            next_state = torch.FloatTensor(next_state)
            action     = torch.FloatTensor(action)
            next_action = torch.FloatTensor(next_action)

            reward     = torch.FloatTensor(reward).unsqueeze(1)
            done       = torch.FloatTensor(np.float32(done)).unsqueeze(1)

            ns, nstd, val = self.model_net(state, action)

            nval = self.model_net.reward_fun(torch.cat([next_state, next_action], axis=1))

            a, log_std = self.policy_net(state.detach())

            beta = Normal(ns, nstd)
            pi = Normal(a, log_std.exp())


            df = jacobian(ns, state)

            # print((pi.log_prob(action)*val).shape)

            # print(torch.abs(val).max(), torch.abs(pi.entropy()).max(), torch.abs(pi.log_prob(action)).max() )

            # pval = self.model_net.reward_fun(torch.cat([state, pi.sample()],axis=1)).detach()

            # pval = torch.clamp(self.model_net(state.detach(), pi.sample())[2].detach(), -50,50)

            policy_loss = -torch.mean(pi.log_prob(action)) - 3e-3*pi.entropy().mean()

            rew_loss = torch.mean(torch.pow((reward+0.95*(1-done)*nval).detach()-val,2))
            model_loss = -torch.mean(beta.log_prob(next_state)) \
                            + 1e-3 * torch.norm(df, dim=[1,2]).mean() \
                            + 0.5*rew_loss# - 1e-3 * beta.entropy().mean()

            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()


            self.model_optimizer.zero_grad()
            model_loss.backward()
            self.model_optimizer.step()


        self.log['policy_loss'].append(policy_loss.item())
        self.log['model_loss'].append(model_loss.item())
