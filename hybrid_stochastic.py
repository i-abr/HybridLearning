import torch
from torch.distributions import Normal

class PathIntegral(object):


    def __init__(self, model, policy,
                    samples=10, t_H=10, lam=0.1):

        self.model          = model
        self.policy         = policy
        self.num_actions    = model.num_actions
        self.t_H            = t_H
        self.lam            = lam
        self.samples         = samples
        self.a = torch.zeros(t_H, self.num_actions)
        self.sk = torch.zeros((self.samples, self.t_H))

    def reset(self):
        self.a.zero_()

    def __call__(self, state):
        with torch.no_grad():
            self.a[:-1] = self.a[1:].clone()
            self.a[-1].zero_()

            s0 = torch.FloatTensor(state).unsqueeze(0)
            s = s0.repeat(self.samples, 1)
            mu, log_std = self.policy(s)
            
            sk = []
            da = []
            log_prob = []
            for t in range(self.t_H):
                pi = Normal(mu, log_std.exp())
                v = pi.sample()
                log_prob.append(pi.log_prob(self.a[t].expand_as(v)).sum(1,keepdim=True).T )
                da.append(v - self.a[t].expand_as(v))
                s, rew = self.model.step(s, v)
                mu, log_std = self.policy(s)
                sk.append(-rew.T)
                #self.sk[:,t] = -rew.squeeze()#+ torch.pow(da[-1], 2).sum(1)

            #self.sk = torch.cumsum(self.sk.flip(1), 1).flip(1)
            # self.sk -= torch.min(self.sk, 1)[0].unsqueeze(1)

            sk = torch.cat(sk, dim=0)
            sk = torch.cumsum(sk.flip(0), 0).flip(0)
            sk = sk - torch.min(sk, dim=1, keepdim=True)[0]

            
            log_prob = torch.cat(log_prob, dim=0)
            log_prob = log_prob - torch.max(log_prob, dim=1, keepdim=True)[0]

            w = torch.exp(-sk.div(self.lam) + log_prob) + 1e-5
            w.div_(torch.sum(w, dim=1, keepdim=True))
            
            for t in range(self.t_H):
                #sk = self.sk[:, t]
                #sk -= torch.min(sk)
                #log_prob_t = log_prob[t]
                #log_prob_t -= torch.max(log_prob[t])
                #w = torch.exp(-sk.div(self.lam) + log_prob_t) + 1e-5
                #w.div_(w.sum(0))
                #self.a[t] = self.a[t] + w.unsqueeze(1).t().mm(da[t])
                self.a[t] = self.a[t] + torch.mv(da[t].T, w[t])
            return self.a[0].clone().numpy()



