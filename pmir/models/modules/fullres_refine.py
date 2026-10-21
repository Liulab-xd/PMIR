import torch
import torch.nn as nn
from pmir.models.modules.sep_conv import SepConv


class FullResRefine(nn.Module):
    def __init__(self, channels, n_blocks=4):
        super().__init__()
        self.inject = nn.Conv2d(channels + 1, channels, 3, padding=1)
        self.blocks = nn.ModuleList([
            nn.Sequential(
                SepConv(channels, channels, 3),
                nn.LeakyReLU(0.1, inplace=True),
                SepConv(channels, channels, 3),
            ) for _ in range(n_blocks)
        ])

    def forward(self, x, dop):
        x = self.inject(torch.cat([x, dop], dim=1))
        for block in self.blocks:
            x = x + block(x)
        return x
