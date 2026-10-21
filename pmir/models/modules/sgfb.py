import torch
import torch.nn as nn
import torch.nn.functional as F
from pmir.models.modules.basic_block import BasicBlock
from pmir.models.modules.sep_conv import SepConv


class SGFB(nn.Module):

    def __init__(self, channels, use_hin=True):
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(1))
        self.block1 = BasicBlock(channels, channels, use_hin=use_hin)
        self.block2 = BasicBlock(channels, channels, use_hin=use_hin)
        self.grad_gate = nn.Sequential(SepConv(channels, channels, 3, bias=False), nn.Sigmoid())
        kx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]])
        ky = kx.t().contiguous()
        self.register_buffer("kx", kx.view(1, 1, 3, 3).repeat(channels, 1, 1, 1))
        self.register_buffer("ky", ky.view(1, 1, 3, 3).repeat(channels, 1, 1, 1))

    def forward(self, x):
        xp = F.pad(x, (1, 1, 1, 1), mode="reflect")
        gx = F.conv2d(xp, self.kx, groups=x.shape[1])
        gy = F.conv2d(xp, self.ky, groups=x.shape[1])
        gate = self.grad_gate(torch.sqrt(gx.square() + gy.square() + 1e-6))
        y = self.block1(x)
        a = torch.sigmoid(self.alpha)
        y = (1.0 - a) * y + a * gate * y
        return self.block2(y)
