import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.autograd import Variable

class MDNPolicy(nn.Module):
    def __init__(self, num_inputs, num_outputs,
                    n_hidden=200, n_gaussians=10):
        super(MDNPolicy, self).__init__()

        # self.linear1 = nn.Linear(num_inputs, n_hidden)

        self.z_h = nn.Sequential(
            nn.Linear(num_inputs, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, int(n_hidden/2))
        )

        self.z_pi = nn.Linear(int(n_hidden/2), n_gaussians)
        self.z_sigma = nn.Linear(int(n_hidden/2), n_gaussians * num_outputs)
        self.z_mu = nn.Linear(int(n_hidden/2), n_gaussians * num_outputs)

        self.num_outputs = num_outputs
        self.num_gauss   = n_gaussians
        self.num_inputs  = num_inputs

    def get_action(self, state):
        x = torch.FloatTensor(state).unsqueeze(0)
        pi, sigma, mu = self.forward(x)
        pi_picked = torch.multinomial(pi, 1)
        # pi_picked = torch.argmax(pi, dim=1, keepdim=True)
        res = []
        for i, r in enumerate(pi_picked):
            res.append(
                torch.normal(mu[i, r], sigma[i, r])
            )

        return torch.cat(res).numpy()[0]

    def logits(self, x, y):
        pi, sigma, mu = self.forward(x)
        y_expand = y.unsqueeze(1).expand_as(mu)

        log_pi = torch.log(pi)
        log_pdf = -torch.log(sigma).sum(2)-0.5*((y_expand - mu) * torch.reciprocal(sigma)).pow(2).sum(2)
        return torch.logsumexp(log_pdf + log_pi, dim=1, keepdim=True)

    def forward(self, x):
        z_h = self.z_h(x)
        # z_h = torch.sin(self.linear1(torch.cat([x, u], dim=1)))
        pi = nn.functional.softmax(self.z_pi(z_h) + 1e-3, -1)
        sigma = torch.clamp(self.z_sigma(z_h), -4, 4).exp()

        mu = self.z_mu(z_h).view(-1, self.num_gauss, self.num_outputs)
        return pi, sigma.view(-1, self.num_gauss, self.num_outputs), mu
