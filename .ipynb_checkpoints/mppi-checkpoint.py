import torch
from torch.distributions import Normal

class MPPI(object):


    def __init__(self, model, policy,
                    samples = 10, t_H=10, lam=0.1, noise=0.1):

        self.model          = model
        self.policy         = policy
        self.num_actions    = model.num_actions
        self.t_H            = t_H
        self.lam            = lam
        self.samples         = samples
        self.a = [torch.zeros(1, self.num_actions)
                        for _ in range(t_H)]
        self.sk = torch.zeros((self.samples, self.t_H))

    def reset(self):
        self.a = [torch.zeros(1, self.num_actions)
                        for _ in range(self.t_H)]
    def __call__(self, state):
        with torch.no_grad():
            self.a[:-1] = self.a[1:]
            self.a[-1]  = torch.zeros(1, self.num_actions)

            s0 = torch.FloatTensor(state).unsqueeze(0)
            s = s0.repeat(self.samples, 1)
            mu, log_std = self.policy(s)

            da = []
            log_prob = []
            for t in range(self.t_H):
                pi = Normal(mu, log_std.exp()) 
                v = pi.sample()
                log_prob.append(pi.log_prob(self.a[t]).sum(1))
                da.append(v - self.a[t])
                s_dist, rew = self.model(s, v)
                s = s_dist.mean
                mu, log_std = self.policy(s)
                self.sk[:,t] = -rew.squeeze() #+ torch.pow(da[-1], 2).sum(1)

            self.sk = torch.cumsum(self.sk.flip(1), 1).flip(1)
            # self.sk -= torch.min(self.sk, 1)[0].unsqueeze(1)
            for t in range(self.t_H):
                sk = self.sk[:, t]
                sk -= torch.min(sk)
                w = torch.exp(-sk.div(self.lam) + log_prob[t]) + 1e-5
                w.div_(w.sum(0))
                self.a[t] = self.a[t] + w.unsqueeze(1).t().mm(da[t])
            return self.a[0].data.numpy()[0]
