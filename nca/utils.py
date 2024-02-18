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


def to_np(t: torch.Tensor):
    t = t.detach().cpu().numpy()
    return t


def zoom(img, scale=4):
    img = np.repeat(img, scale, 0)
    img = np.repeat(img, scale, 1)
    return img


def imshow(t):
    if t.ndim == 2:
        t = plt.cm.viridis(t)[:, :, :3]
        t = (t * 255).astype(np.uint8)
        return Image.fromarray(t)

    raise NotImplementedError("Implement for more than two dims")
