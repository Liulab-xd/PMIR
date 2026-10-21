import torch
import numpy as np


def compute_dop(i0, i45, i90, i135):

    s0 = 0.5 * (i0 + i45 + i90 + i135)
    s1 = i0 - i90
    s2 = i45 - i135
    magnitude = np.sqrt(s1 ** 2 + s2 ** 2 + 1e-12)
    dop_per_channel = np.clip(magnitude / (s0 + 1e-6), 0.0, 1.0)
    dop = dop_per_channel.mean(axis=2, keepdims=True)
    return torch.from_numpy(dop.transpose(2, 0, 1)).float()
