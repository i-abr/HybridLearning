import torch
import torch.autograd as autograd
from torch.autograd import Variable
from torch.autograd.gradcheck import zero_gradients
import torch.nn.functional as F

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
        jacobian[i] = inputs.grad.data

    return torch.transpose(jacobian, dim0=0, dim1=1)

class DeterministicCtrl(object):

    def __init__(self, model, policy, T=10, eps=1e-1):
        self.model = model
        self.policy = policy
        self.T = T
        self.eps = eps

        self.num_states = model.num_states
        self.num_actions = model.num_actions

        self.u = torch.cat(
            [torch.zeros(1, self.num_actions) for t in range(self.T-1)], dim=0)
        self.u.requires_grad = True

    def reset(self):
        self.u = torch.cat(
            [torch.zeros(1, self.num_actions) for t in range(self.T-1)], dim=0)
        self.u.requires_grad = True


    def __call__(self, state):
        with torch.no_grad():
            self.u[:-1] = self.u[1:].clone()
            self.u[-1].zero_()
            x_t = torch.FloatTensor(state).unsqueeze(0)
            x = []
            for t in range(self.T-1):
                x.append(x_t.clone())
                u_t = self.policy(x_t)
                x_t, r_t = self.model.step(x_t, torch.tanh(self.u[t].unsqueeze(0) + u_t))

        # compute those derivatives
        x = torch.cat(x)
        x.requires_grad = True
        u_p = self.policy(x)

        pred_state, pred_rew = self.model.step(x, torch.tanh(self.u+u_p))
        pred_rew = pred_rew - torch.sum(torch.pow(self.u,2), dim=1, keepdim=True)
        dfdx = compute_jacobian(x, pred_state)
        dfdu = compute_jacobian(self.u, pred_state)
        dldx = compute_jacobian(x, pred_rew)
        dldu = compute_jacobian(self.u, pred_rew)


        with torch.no_grad():
            rho = torch.zeros(1, self.num_states)
            for t in reversed(range(self.T-1)):
                rho = dldx[t] + rho.mm(dfdx[t])
                # rho_norm = torch.norm(rho)
                # if rho_norm > 1.0:
                #     rho = rho/torch.norm(rho)
                self.u[t] = self.u[t] + self.eps*(dldu[t] + rho.mm(dfdu[t]))
#                 self.u[t] = self.u[t] + self.eps * (dldu[t] + rho.mm(dfdu[t]))

        return torch.tanh(self.u[0] + u_p[0]).data.numpy()
