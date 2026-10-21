import torch
import torch.nn as nn
from pmir.models.modules.sgfb import SGFB
from pmir.models.modules.wavelet_enhance import WaveletEnhanceBlock


class BasicLayer(nn.Module):
    def __init__(self, channels, n_blocks=1, use_hin=True):
        super().__init__()
        self.wave = nn.ModuleList([WaveletEnhanceBlock(channels) for _ in range(n_blocks)])
        self.grad = nn.ModuleList([SGFB(channels, use_hin=use_hin) for _ in range(n_blocks)])
        self.res_gate = nn.Parameter(torch.zeros(n_blocks))

    def forward(self, x):
        for i, (wave, grad) in enumerate(zip(self.wave, self.grad)):
            identity = x
            x = grad(x + wave(x))
            g = torch.sigmoid(self.res_gate[i])
            x = g * x + (1.0 - g) * identity
        return x
