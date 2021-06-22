import torch
import torch.autograd as autograd
from torch.autograd import Variable
from torch.autograd.gradcheck import zero_gradients
import torch.nn.functional as F
from torch.distributions import Normal

def compute_jacobian(inputs, output, create_graph=False):
    """
    :param inputs: Batch X Size (e.g. Depth X Width X Height)
    :param output: Batch X Classes
    :return: jacobian: Batch X Classes X Size
    """
    assert inputs.requires_grad

    num_classes = output.size()[1]

    jacobian = torch.zeros(num_classes, *inputs.size())
    grad_output = torch.zeros(*output.size())
    if inputs.is_cuda:
        grad_output = grad_output.cuda()
        jacobian = jacobian.cuda()

    for i in range(num_classes):
        zero_gradients(inputs)
        grad_output.zero_()
        grad_output[:, i] = 1
        output.backward(grad_output, retain_graph=True, create_graph=create_graph)
        jacobian[i] = inputs.grad.clone()

    return torch.transpose(jacobian, dim0=0, dim1=1)

class DetPolicyWrapper(object):

    def __init__(self, model, policy, T=10, lr=0.1,  eps=1e-1, reg=1.0):
        self.model = model
        self.policy = policy
        self.T = T
        self.eps = eps
        self.reg = reg

        self.num_states = model.num_states
        self.num_actions = model.num_actions

        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = 'cuda:0'
        self.u = torch.zeros(T, self.num_actions).to(self.device)
        self.u.requires_grad = True

    def reset(self):
        with torch.no_grad():
            self.u.zero_()

    def __call__(self, state):

        with torch.no_grad():
            self.u[:-1] = self.u[1:].clone()
            self.u[-1].zero_()
            x_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
#             x = []
            # u = []
            for t in range(self.T):
                x.append(x_t.clone())
                u_t, log_std_t = self.policy(x_t)
                x_t, r_t = self.model.step(x_t, self.u[t].unsqueeze(0)+u_t)

        # compute those derivatives
        x = torch.cat(x)
        x.requires_grad = True

        u_p, log_std_p = self.policy(x)

        pred_state, pred_rew = self.model.step(x, self.u+u_p)

        # loss = pred_rew.mean()
        dfdx = compute_jacobian(x, pred_state)
        dfdu = compute_jacobian(self.u, pred_state)
        dldx = compute_jacobian(x, pred_rew)

        with torch.no_grad():
            rho = torch.zeros(1, self.num_states).to(self.device)
            for t in reversed(range(self.T)):
                rho = (0.2**(t)) * dldx[t] + rho.mm(dfdx[t])
                # TODO: add the gamma as part of a parameter
                # rho =  dldx[t] + rho.mm(dfdx[t])

                # self.u[t] = self.u[t] + (dldu[t] + rho.mm(dfdu[t])) * self.eps
                # self.u[t] = 2.0* log_std_p[t].exp() * rho.mm(dfdu[t])
            # _u = torch.pow(log_std_p[0].exp(),2) * (rho.mm(dfdu[0]))
            rho = torch.clamp(rho, -1,+1)
            sig = torch.pow(log_std_p[0].exp().unsqueeze(0),2)
            # u_t, log_std_t = self.policy(torch.FloatTensor(state).unsqueeze(0).to(self.device))
            _u = sig * rho.mm(dfdu[0]) #+ torch.randn_like(log_std_t) * torch.exp(log_std_t)

            #f1,_ = self.model.step(x[0].unsqueeze(0), torch.clamp(torch.tanh(u_p[0].unsqueeze(0)),-1,+1) )
            #f2,_ = self.model.step(x[0].unsqueeze(0), torch.clamp(_u[0]+torch.tanh(u_p[0].unsqueeze(0)),-1,+1))
            self.u.grad.zero_()
            return torch.clamp(_u[0]+u_p[0], -1, +1).cpu().clone().numpy()
            #, rho.mm((f2-f1).T).detach().clone().numpy().squeeze()
            # return torch.tanh(self.u[0] + u_p[0]).detach().clone().numpy()
