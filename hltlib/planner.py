import torch
from torch.distributions import Normal
import time
import numpy as np

class PathIntegral(object):

    def __init__(self, model, policy, samples=10, t_H=10, lam=0.1):


        self.model          = model
        self.policy         = policy
        self.num_actions    = model.num_actions
        self.t_H            = t_H
        self.lam            = lam
        self.samples         = samples

        self.a = torch.zeros(t_H, self.num_actions)

        self.eps = Normal(torch.zeros(self.samples, self.num_actions), torch.ones(self.samples, self.num_actions)*0.2)

    def reset(self):
        self.a.zero_()

    def __call__(self, state):
        # return np.random.normal(0,0.1,size=(2,))
        with torch.no_grad():
            self.a[:-1] = self.a[1:].clone()
            self.a[-1].zero_()

            s0 = torch.FloatTensor(state.copy()).unsqueeze(0)
            s = s0.repeat(self.samples, 1)
            mu, log_std = self.policy(s)

            sk = []
            da = []
            log_prob = []
            for t in range(self.t_H):
                # pi = Normal(mu, log_std.exp())
                # v = pi.sample()
                # log_prob.append(pi.log_prob(self.a[t].expand_as(v)).sum(1))
                # log_prob.append(pi.log_prob(v).sum(1))
                # da.append(v - self.a[t].expand_as(v))
                # da.append(v)
                eps = self.eps.sample()
                da.append(eps)
                v = self.a[t].expand_as(eps) + eps
                s, rew = self.model.step(s, v)
                mu, log_std = self.policy(s)
                sk.append(rew.squeeze())
            sk = torch.stack(sk)
            sk = torch.cumsum(sk.flip(0), 0).flip(0)


            sk = sk - torch.max(sk, dim=1, keepdim=True)[0]
            # sk /= torch.norm(sk, dim=1, keepdim=True)
            # log_prob = torch.stack(log_prob)
            # log_prob -= torch.max(log_prob, dim=1, keepdim=True)[0]
            # log_prob /= (torch.norm(log_prob,dim=1, keepdim=True) + 1e-4)

            w = torch.exp(sk.div(self.lam))+1e-5 # + log_prob) + 1e-5
            w.div_(torch.sum(w, dim=1, keepdim=True))

            for t in range(self.t_H):
                self.a[t] = self.a[t] + torch.mv(da[t].T, w[t])
                # self.a[t] = torch.mv(da[t].T, w[t])

            return self.a[0].clone().numpy()
