import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from compute_jacobian import compute_jacobian

class Model(nn.Module):

    def __init__(self, nx, nu, H=32):

        self.nx = nx
        self.nu = nu

        self.g = nn.Sequential(
            nn.Linear(nx, H),
            nn.Tanh(),
            nn.Linear(H, H)
            nn.Tanh(),
            nn.Linear(H, nx)
        )

        self.h = nn.Sequential(
            nn.Linear(nx, H),
            nn.Tanh(),
            nn.Linear(H, H),
            nn.Tanh(),
            nn.Linear(H, nx*nu)
        )
    

    @property
    def A(self):
        return self._A.clone()

    @property
    def B(self):
        return self._B.clone()

    def forward(self, x, u, get_jacobian=False):
        if get_jacobian:
            x = Variable(x, requires_grad=True)
            u = Variable(u, requires_grad=True)
        gx = self.g(x)
        h  = self.h(x).view(-1, self.nx, self.nu)
        hu = h.bmm(u.view(-1,self.nu,1)).squeeze(-1)
        out = x + gx + hu
        if get_jacobian:
            fdx = compute_jacobian(x, out)
            self._A = fdx.data
            self._B = h.data
        return out
