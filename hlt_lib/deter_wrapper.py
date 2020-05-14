import torch
import torch.autograd as autograd
from torch.autograd import Variable
from torch.autograd.gradcheck import zero_gradients
import torch.nn.functional as F
from torch.distributions import Normal
import torch.optim as optim

class DetPolicyWrapper(object):

    def __init__(self, model, policy, T=10, lr=0.1, eps=1e-1, reg=1.0):
        self.model = model
        self.policy = policy
        self.T = T
        self.eps = eps
        self.reg = reg

        self.state_dim = model.num_states
        self.action_dim = model.num_actions

        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = 'cuda:0'
        self.lr = lr
        self.u = torch.zeros(T, self.action_dim).to(self.device)
        self.u.requires_grad = True
        self.optim = optim.SGD([self.u], lr=lr)

    def reset(self):
        with torch.no_grad():
            self.u.zero_()

    def __call__(self, state, epochs=1):

        for epoch in range(epochs):
            s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            cost = 0.
            t = 0
            for u in self.u:
                u_t, log_std_t = self.policy(s)
                pi = Normal(torch.tanh(u_t), log_std_t.exp())
                u_app = torch.tanh(u_t+u.unsqueeze(0)) #+ torch.randn_like(u_t) * torch.exp(log_std_t)
                s, r = self.model.step(s, u_app)
                cost = cost - (r + pi.log_prob(u_app).mean())
                t += 1
            self.optim.zero_grad()
            cost.backward()
            self.optim.step()

        with torch.no_grad():
            u_t, log_std_t = self.policy(torch.FloatTensor(state).unsqueeze(0).to(self.device))
            v = u_t + torch.randn_like(log_std_t) * torch.exp(log_std_t)
            # u = torch.tanh((v.squeeze() + self.u[0]).cpu().clone()).numpy()
            u = torch.tanh(v.squeeze() + self.u[0]).cpu().clone().numpy()
            self.u[:-1] = self.u[1:].clone()
            self.u[-1].zero_()
            return u
