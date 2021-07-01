import torch
from torch.distributions import Normal

class PathIntegral(object):

    def __init__(self, model, samples=10, t_H=10, lam=0.1, eps=0.3):


        self.model           = model
        self.num_actions     = model.num_actions
        self.t_H             = t_H
        self.lam             = lam
        self.samples         = samples

        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = 'cuda'

        self.a = torch.zeros(t_H, self.num_actions).to(self.device)
        self.eps = Normal(torch.zeros(self.samples, self.num_actions).to(self.device),
                            torch.ones(self.samples, self.num_actions).to(self.device) * eps)

    def reset(self):
        with torch.no_grad():
            self.a.zero_()

    def __call__(self, state):

        with torch.no_grad():
            self.a[:-1] = self.a[1:].clone()
            self.a[-1].zero_()

            s0 = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            s = s0.repeat(self.samples, 1)

#             sk, da, log_prob = [], [], []
            sk = torch.zeros(self.t_H,self.samples).to(self.device)
            da = torch.zeros(self.t_H,self.samples,self.num_actions).to(self.device)
            log_prob = torch.zeros(self.t_H,self.samples).to(self.device)
            eta = torch.zeros(1).to(self.device)
            for t in range(self.t_H):
                eps = self.eps.sample()
                eta = 0.5 * eta + (1-0.5) * eps
                log_prob[t] = self.eps.log_prob(eta).sum(1)
                da[t] = eta
                v = self.a[t].expand_as(eta) + eta
                s, rew = self.model.step(s, v)
                sk[t] = rew.squeeze()

#             sk = torch.stack(sk)
            sk = torch.cumsum(sk.flip(0), 0).flip(0)
#             log_prob = torch.stack(log_prob)

            # sk = sk.div(self.lam) + log_prob # doing this twice? (alp commented out 5/10)
            sk = sk + self.lam*log_prob # copied from stoch_wrapper.py (alp added 5/10)
            sk = sk - torch.max(sk, dim=1, keepdim=True)[0]

            # log_prob = torch.stack(log_prob)
            # log_prob -= torch.max(log_prob, dim=1, keepdim=True)[0]
            # w = torch.exp(-sk.div(self.lam) + log_prob) + 1e-5
            # w = torch.exp(sk.div(self.lam) + log_prob) + 1e-5  (alp commented out 5/10)
            w = torch.exp(sk.div(self.lam)) + 1e-5 # copied from stoch_wrapper.py (alp added 5/10)
            w.div_(torch.sum(w, dim=1, keepdim=True))
            for t in range(self.t_H):
                self.a[t] = self.a[t] + torch.mv(da[t].T, w[t])

            return self.a[0].cpu().clone().numpy(), None
