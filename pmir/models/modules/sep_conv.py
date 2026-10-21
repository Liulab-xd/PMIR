import torch.nn as nn


class SepConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1,
                 bias=True, padding_mode="zeros"):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channel, in_channel, kernel_size, stride=stride,
            padding=kernel_size // 2, groups=in_channel, bias=bias,
            padding_mode=padding_mode,
        )
        self.conv2 = nn.Conv2d(in_channel, out_channel, 1, bias=bias)

    def forward(self, x):
        return self.conv2(self.conv1(x))
