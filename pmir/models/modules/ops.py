import torch
import torch.nn.functional as F


def soft_clip01(x, beta=40.0):

    positive = F.relu(x)
    return positive / (1.0 + F.softplus(positive - 1.0, beta=beta))


def make_haar_kernel(channels):

    ll = torch.tensor([[0.5, 0.5], [0.5, 0.5]], dtype=torch.float32)
    lh = torch.tensor([[-0.5, -0.5], [0.5, 0.5]], dtype=torch.float32)
    hl = torch.tensor([[-0.5, 0.5], [-0.5, 0.5]], dtype=torch.float32)
    hh = torch.tensor([[0.5, -0.5], [-0.5, 0.5]], dtype=torch.float32)
    return torch.stack([ll, lh, hl, hh], dim=0).unsqueeze(1).repeat(channels, 1, 1, 1)


def haar_split(x, kernel):
    b, c, h, w = x.shape
    if h % 2 or w % 2:
        x = F.pad(x, (0, w % 2, 0, h % 2), mode="reflect")
        h, w = x.shape[-2:]
    coeff = F.conv2d(x, kernel, stride=2, groups=c)
    return coeff.view(b, c, 4, h // 2, w // 2)


def pixel_unshuffle(x, downscale_factor=2):
    b, c, h, w = x.shape
    r = downscale_factor
    x = x.view(b, c, h // r, r, w // r, r)
    x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
    return x.view(b, c * r * r, h // r, w // r)
