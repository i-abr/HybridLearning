import torch
from torch.distributions import Normal

class PathIntegral(object):

    def __init__(self, model, samples=10, t_H=10, lam=0.1, eps=0.3):


        self.model          = model
        self.num_actions    = model.num_actions
        self.t_H            = t_H
        self.lam            = lam
        self.samples         = samples

        self.a = torch.zeros(t_H, self.num_actions)
        self.eps = Normal(torch.zeros(self.samples, self.num_actions),
                            torch.ones(self.samples, self.num_actions) * eps)

    def reset(self):
        self.a.zero_()

    def __call__(self, state):

        with torch.no_grad():
            self.a[:-1] = self.a[1:].clone()
            self.a[-1].zero_()

            s0 = torch.FloatTensor(state).unsqueeze(0)
            s = s0.repeat(self.samples, 1)

            sk = []
            da = []
            log_prob = []
            for t in range(self.t_H):
                eps = self.eps.sample()
                log_prob.append(self.eps.log_prob(eps).sum(1))
                da.append(eps)
                v = self.a[t].expand_as(eps) + eps
                s, rew = self.model.step(s, v)
                sk.append(-rew.squeeze())

            sk = torch.stack(sk)
            sk = torch.cumsum(sk.flip(0), 0).flip(0)
            sk = sk - torch.min(sk, dim=1, keepdim=True)[0]
            log_prob = torch.stack(log_prob)
            log_prob -= torch.max(log_prob, dim=1, keepdim=True)[0]
            w = torch.exp(-sk.div(self.lam) + log_prob) + 1e-5
            w.div_(torch.sum(w, dim=1, keepdim=True))
            for t in range(self.t_H):
                self.a[t] = self.a[t] + torch.mv(da[t].T, w[t])

            return self.a[0].clone().numpy()
