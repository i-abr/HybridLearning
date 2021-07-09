import torch
from torch.distributions import Normal

class StochPolicyWrapper(object):

    def __init__(self, model, policy, samples=10, t_H=10, frame_skip=5, lam=0.1):


        self.model          = model
        self.policy         = policy
        self.num_actions    = model.num_actions
        self.t_H            = t_H
        self.lam            = lam
        self.samples        = samples

        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = 'cuda'

        self.a = torch.zeros(t_H, self.num_actions)
        self.a = self.a.to(self.device)

    def reset(self):
        with torch.no_grad():
            self.a.zero_()

    def __call__(self, state):

        with torch.no_grad():
            self.a[:-1] = self.a[1:].clone()
            self.a[-1].zero_()

            s0 = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            s = s0.repeat(self.samples, 1)

            mu, log_std = self.policy(s)
#             sk, da, log_prob = [], [], []
            sk = torch.zeros(self.t_H,self.samples).to(self.device)
            da = torch.zeros(self.t_H,self.samples,self.num_actions).to(self.device)
            log_prob = torch.zeros(self.t_H,self.samples).to(self.device)
            for t in range(self.t_H):
                pi = Normal(mu, log_std.exp())
                v = torch.tanh(pi.sample())
                da_t = v - self.a[t].expand_as(v)
                log_prob[t] = pi.log_prob(da_t).sum(1)
#                 log_prob.append(pi.log_prob(v).sum(1))
                # log_prob.append(pi.log_prob(self.a[t].expand_as(v)).sum(1))  # should this be da? alp
                # log_prob.append(pi.log_prob(v).sum(1))
                da[t] = da_t
                # da.append(v)
                s, rew = self.model.step(s, v)
                mu, log_std = self.policy(s)
                sk[t] = rew.squeeze()

#             sk = torch.stack(sk)
            sk = torch.cumsum(sk.flip(0), 0).flip(0)
#             log_prob = torch.stack(log_prob)

            sk = sk + self.lam*log_prob
            # sk = sk - torch.min(sk, dim=1, keepdim=True)[0]
            sk = sk - torch.max(sk, dim=1, keepdim=True)[0]

            #log_prob -= torch.max(log_prob, dim=1, keepdim=True)[0]

            #w = torch.exp(sk.div(self.lam) + log_prob) + 1e-5
            w = torch.exp(sk.div(self.lam)) + 1e-5
            w.div_(torch.sum(w, dim=1, keepdim=True))
            for t in range(self.t_H):
                self.a[t] = self.a[t] + torch.mv(da[t].T, w[t])
                # self.a[t] = torch.mv(da[t].T, w[t])
            return self.a[0].cpu().clone().numpy(), da[0].cpu().clone().numpy()
