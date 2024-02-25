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
