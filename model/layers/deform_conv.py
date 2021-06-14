import torch.nn.functional as F
from torch import nn

from .dcn_v2 import DCN

class DeformConv(nn.Module):
    def __init__(self,
                 in_channel,
                 out_channel,
                 norm_func,
                 kernel_size=3,
                 stride=1,
                 padding=0):
        super(DeformConv, self).__init__()

        self.norm = norm_func(out_channel)
        self.relu = nn.ReLU(inplace=True)
        self.deform_conv = DCN(in_channels=in_channel,
                               out_channels=out_channel,
                               kernel_size=(kernel_size, kernel_size),
                               stride=stride,
                               padding=padding,
                               dilation=1,
                               deformable_groups=1)

    def forward(self, x):
        x = self.deform_conv(x)
        x = self.norm(x)
        x = self.relu(x)

        return x
