import torch.nn as nn
import torch

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


class Reshape(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.reshape(-1, *self.shape)


class LinerInDim(nn.Module):
    def __init__(self, in_size, out_size, dim=-1):
        super().__init__()
        self.dim = dim
        self.linear = nn.Linear(in_size, out_size)

    def forward(self, x):
        x_dims = list(range(len(x.shape)))
        x_dims[-1], x_dims[self.dim] = x_dims[self.dim], x_dims[-1]
        x = x.permute(*x_dims)
        x = self.linear(x)
        x = x.permute(*x_dims)

        return x


class Lambda(nn.Module):
    def __init__(self, handler):
        super().__init__()
        self.handler = handler

    def forward(self, x):
        return self.handler(x)


class Permute(Lambda):
    def __init__(self, permutation):
        super().__init__(lambda x: x.permute(permutation))


def conv11(in_channels, out_channels, bias):
    return nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=1,
        padding=0,
        stride=1,
        bias=True,
    )


def conv_same(in_channels, out_channels, ks, bias=False):
    return nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=ks,
        padding=ks // 2,
        stride=1,
        bias=bias,
    )


BOSCO_INIT = torch.tensor(
    [
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
        [0, 1, 1, 0, 0, 1, 1, 1, 1, 0],
        [1, 1, 0, 0, 0, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 1, 1, 1, 1],
        [1, 1, 0, 0, 0, 0, 1, 1, 1, 1],
        [1, 1, 1, 0, 1, 1, 1, 1, 1, 1],
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
        [0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
    ],
    dtype=float,
)


def bosco_update(U):
    """Bosco's rule, b1..b2 is birth range, s1..s2 is stable range (outside s1..s2 is shrink range)"""
    b1, b2, s1, s2 = 34, 45, 34, 58
    return ((U >= b1) & (U <= b2)).to(torch.float32) - ((U < s1) | (U > s2)).to(
        torch.float32
    )
