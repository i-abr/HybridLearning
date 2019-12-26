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

    def __init__(self, model, policy, T=10, eps=1e-1, reg=1.0):
        self.model = model
        self.policy = policy
        self.T = T
        self.eps = eps
        self.reg = reg

        self.num_states = model.num_states
        self.num_actions = model.num_actions

        self.u = torch.cat(
            [torch.zeros(1, self.num_actions) for t in range(self.T)], dim=0)
        self.u.requires_grad = True

    def reset(self):
        self.u = torch.cat(
            [torch.zeros(1, self.num_actions) for t in range(self.T)], dim=0)
        self.u.requires_grad = True


    def __call__(self, state):
        with torch.no_grad():
            self.u[:-1] = self.u[1:].clone()
            self.u[-1].zero_()
            x_t = torch.FloatTensor(state).unsqueeze(0)
            x = []
            for t in range(self.T):
                x.append(x_t.clone())
                u_t = self.policy(x_t)
                x_t, r_t = self.model.step(x_t, torch.tanh(self.u[t].unsqueeze(0) + u_t))

        # compute those derivatives
        x = torch.cat(x)
        x.requires_grad = True
        u_p = self.policy(x)

        pred_state, pred_rew = self.model.step(x, torch.tanh(self.u+u_p))
        # pred_rew = pred_rew - torch.max(pred_rew)
        pred_rew = pred_rew - self.reg*torch.sum(torch.pow(self.u,2), dim=1, keepdim=True)

        # loss = pred_rew.mean()
        dfdx = compute_jacobian(x, pred_state)
        dfdu = compute_jacobian(self.u, pred_state)
        dldx = compute_jacobian(x, pred_rew)
        dldu = compute_jacobian(self.u, pred_rew)

        # loss.backward()

        # with torch.no_grad():
            # self.u += 1e-3 * self.u.grad

        # self.u.grad.zero_()

        # normalize
        # _dldx = torch.norm(dldx, dim=1, keepdim=True)
        # _dldu = torch.norm(dldu, dim=1, keepdim=True)

        # dldx = dldx / _dldx
        # dldu = dldu / _dldu

        # if not torch.isfinite(dldx).any() or not torch.isfinite(dldu).any(): print('yuppp')

        with torch.no_grad():
            rho = torch.zeros(1, self.num_states)
            for t in reversed(range(self.T)):
                # dldx_norm = torch.norm(dldx[t]) + 1e-5
                # dldu_norm = torch.norm(dldu[t]) + 1e-5
                # if torch.abs(dldx_norm) < 1e-8:
                #     dldx_norm = 1.0
                # if torch.abs(dldu_norm) < 1e-8:
                #     dldu_norm = 1.0
                rho = dldx[t] + rho.mm(dfdx[t])
                rho_norm = torch.norm(rho)
                if rho_norm > 1.0:
                    rho = rho/rho_norm
                # self.u[t] = self.u[t] + self.eps*(dldu[t] + rho.mm(dfdu[t]))
                self.u[t] = self.u[t] + (dldu[t] + rho.mm(dfdu[t])) * 0.01
                # self.u[t] = rho.mm(dfdu[t])# + self.u[t]
                # if not torch.isfinite(self.u[t]).any(): print('dsdfsfgsdgsdf',rho, dldx[t], dldu[t], _dldx[t], _dldu[t])
        return torch.tanh(self.u[0] + u_p[0]).detach().clone().numpy()
