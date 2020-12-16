import numpy as np
import torch

from torch.optim.optimizer import Optimizer

from zo.models import device


def make_e_vector(n):
    e = np.random.normal(size=n)
    e = e / np.linalg.norm(e)
    e = torch.Tensor(e).to(device)
    return e


def GradientEstimate_dicrs(netD, data, target, loss, tau=0.000001):
    model_params = netD.get_params()
    n = len(model_params)
    e = make_e_vector(n)
    updated_model_params_1 = model_params + tau * e
    updated_model_params_2 = model_params - tau * e

    netD.set_params(updated_model_params_1)
    output = netD(data)
    loss1 = loss(output, target)

    netD.set_params(updated_model_params_2)
    output = netD(data)
    loss2 = loss(output, target)

    grads = (loss1 - loss2) / (len(output) * tau) * n * e
    netD.set_params(model_params)
    return grads


def GradientEstimate(netG, netD, noise, target, loss, tau=0.000001):
    netG_params = netG.get_params()
    n = len(netG_params)
    e = make_e_vector(n)
    updated_model_params_1 = netG_params + tau * e
    updated_model_params_2 = netG_params - tau * e

    netG.set_params(updated_model_params_1)
    fake = netG(noise)
    output = netD(fake).view(-1)
    loss1 = loss(output, target)

    netG.set_params(updated_model_params_2)
    fake = netG(noise)
    output = netD(fake).view(-1)
    loss2 = loss(output, target)

    grads = (loss1 - loss2) / (len(output) * tau) * n * e
    netG.set_params(netG_params)
    return grads


class zoVIA(Optimizer):
    def __init__(self, model, lr=2e-3, q=10):
        defaults = dict(lr=lr, q=q)
        # super(zoVIA, self).__init__(model, defaults)
        self.lr = lr

    def step_update(self, model, grads):
        loss = None

        flat_params = model.get_params()
        flat_params = flat_params - grads * self.lr
        model.set_params(flat_params)

        return loss


class zoESVIA(Optimizer):
    # TO-DO
    def __init__(self, model, lr=2e-3, q=10):
        defaults = dict(lr=lr, q=q)
        # super(zoVIA, self).__init__(model, defaults)
        self.lr = lr

    def step_update(self, model, grads):
        loss = None

        flat_params = model.get_params()
        flat_params = flat_params - grads * self.lr
        model.set_params(flat_params)

        return loss


class zoscESVIA(Optimizer):
    # TO-DO
    def __init__(self, model, lr=2e-3, q=10):
        defaults = dict(lr=lr, q=q)
        # super(zoVIA, self).__init__(model, defaults)
        self.lr = lr

    def step_update(self, model, grads):
        loss = None

        flat_params = model.get_params()
        flat_params = flat_params - grads * self.lr
        model.set_params(flat_params)

        return loss
