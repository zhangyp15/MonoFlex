import torch
import torch.nn.functional as F
import numpy as np
import math
import torchvision

class Uncertainty_Reg_Loss(torch.nn.Module):
    def __init__(self, reg_loss_fnc):
        super(Uncertainty_Reg_Loss, self).__init__()
        self.reg_loss_fnc = reg_loss_fnc

    def forward(self, pred, target, uncertainty):
        reg_loss = self.reg_loss_fnc(pred, target)
        reg_loss = reg_loss * torch.exp(- uncertainty) + 0.5 * uncertainty

        return loss

class Laplace_Loss(torch.nn.Module):
    def __init__(self):
        super(Laplace_Loss, self).__init__()

    def forward(self, pred, target, reduction='none'):
        # pred/target: K x ...
        loss = (1 - pred / target).abs() 
        return loss

def clip_gradient(model, clip_norm):
    """Computes a gradient clipping coefficient based on gradient norm."""
    totalnorm = 0
    for p in model.parameters():
        if p.requires_grad:
            modulenorm = (p.grad ** 2).sum()
            totalnorm += modulenorm
    
    totalnorm = torch.sqrt(totalnorm)

    norm = clip_norm / torch.max(totalnorm, clip_norm)
    for p in model.parameters():
        if p.requires_grad:
            p.grad.mul_(norm)

def cat(tensors, dim=0):
    """
    Efficient version of torch.cat that avoids a copy if there is only a single element in a list
    """
    assert isinstance(tensors, (list, tuple))
    if len(tensors) == 1:
        return tensors[0]
    return torch.cat(tensors, dim)

class Wing_Loss(torch.nn.Module):
    def __init__(self, w=10, eps=2):
        super(Wing_Loss, self).__init__()
        self.w = w
        self.eps = eps
        self.C = w - w * np.log(1 + w / eps)

    def forward(self, prediction, target):
        differ = (prediction - target).abs()
        log_idxs = (differ < self.w).nonzero()
        l1_idxs = (differ >= self.w).nonzero()

        loss = prediction.new_zeros(prediction.shape[0])
        loss[log_idxs] = self.w * torch.log(differ[log_idxs] / self.eps + 1)
        loss[l1_idxs] = differ[l1_idxs] - self.C

        return loss


if __name__ == '__main__':
    num = 10000
    a = torch.zeros(num)
    b = (torch.arange(num).float() - (num / 2)) / (num / 20)

    wing_loss_fnc = Wing_Loss()
    loss = wing_loss_fnc(a, b)

    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(b, loss)
    plt.show()

