import torch
import pdb
from torch import nn

class Vanilla_FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super(Vanilla_FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, prediction, target):
        positive_index = target.eq(1).float()
        negative_index = (target.lt(1) & target.ge(0)).float()

        loss = 0.
        positive_loss = torch.log(prediction) \
                        * torch.pow(target - prediction, self.gamma) * positive_index
                        
        negative_loss = torch.log(1 - prediction) \
                        * torch.pow(prediction, self.gamma) * negative_index

        positive_loss = positive_loss.sum() * self.alpha
        negative_loss = negative_loss.sum() * (1 - self.alpha)

        loss = - negative_loss - positive_loss

        return loss

class FocalLoss(nn.Module):
    def __init__(self, alpha=2, beta=4):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, prediction, target):
        positive_index = target.eq(1).float()
        negative_index = (target.lt(1) & target.ge(0)).float()
        ignore_index = target.eq(-1).float() # ignored pixels

        negative_weights = torch.pow(1 - target, self.beta)
        loss = 0.

        positive_loss = torch.log(prediction) \
                        * torch.pow(1 - prediction, self.alpha) * positive_index
                        
        negative_loss = torch.log(1 - prediction) \
                        * torch.pow(prediction, self.alpha) * negative_weights * negative_index

        num_positive = positive_index.float().sum()
        positive_loss = positive_loss.sum()
        negative_loss = negative_loss.sum()

        loss = - negative_loss - positive_loss

        return loss, num_positive

if __name__ == '__main__':
    focal_1 = Vanilla_FocalLoss(alpha=0.5)
    focal_2 = FocalLoss()

    pred = torch.rand(20)
    target = torch.randint(low=0, high=2, size=(20, 1)).view(-1)

    loss1 = focal_1(pred, target) * 2
    loss2 = focal_2(pred, target)

    print(loss1, loss2)






