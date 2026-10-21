import torch.nn as nn
from pmir.models.modules.ops import pixel_unshuffle


class Downsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels // 2, 3, padding=1, bias=False)

    def forward(self, x):
        return pixel_unshuffle(self.conv(x), 2)
