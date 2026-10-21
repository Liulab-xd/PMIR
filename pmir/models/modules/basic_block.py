import torch
import torch.nn as nn
from pmir.models.modules.sep_conv import SepConv


class BasicBlock(nn.Module):
    def __init__(self, in_size, out_size, use_hin=True):
        super().__init__()
        self.use_hin = use_hin
        self.identity = nn.Conv2d(in_size, out_size, 1)
        self.conv1 = SepConv(in_size, out_size, 3)
        self.conv2 = SepConv(out_size, out_size, 3)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        if use_hin:
            self.norm = nn.InstanceNorm2d(out_size // 2, affine=True)

    def forward(self, x):
        y = self.conv1(x)
        if self.use_hin:
            y1, y2 = torch.chunk(y, 2, dim=1)
            y = torch.cat([self.norm(y1), y2], dim=1)
        y = self.act(y)
        y = self.act(self.conv2(y))
        return y + self.identity(x)
