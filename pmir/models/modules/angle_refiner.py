import torch.nn as nn
from pmir.models.modules.wavelet_detail import WaveletDetailEnhancer


class AngleWaveletRefiner(nn.Module):

    def __init__(self, feature_channels):
        super().__init__()
        self.enhancer = WaveletDetailEnhancer(3, feature_channels)

    def forward(self, angles, observed, feature, dop):
        b, _, h, w = angles.shape
        images = angles.view(b, 4, 3, h, w).reshape(b * 4, 3, h, w)
        observed4 = observed.view(b, 4, 3, h, w).reshape(b * 4, 3, h, w)
        feature4 = feature[:, None].expand(-1, 4, -1, -1, -1).reshape(
            b * 4, feature.shape[1], h, w
        )
        dop4 = dop[:, None].expand(-1, 4, -1, -1, -1).reshape(b * 4, 1, h, w)
        images = images + self.enhancer(images, observed4, feature4, dop4)
        return images.view(b, 4, 3, h, w).reshape(b, 12, h, w)
