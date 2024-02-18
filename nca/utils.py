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
