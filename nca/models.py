import torch.nn as nn
import torch
from nca.nn_utils import Lambda, LinerInDim, Permute, conv_same

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from matplotlib import cm
import matplotlib.pyplot as plt
import nca.external_utils as external_utils
from tqdm import tqdm
from nca.nn_utils import LinerInDim


class NCA(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=1,
                kernel_size=21,
                stride=1,
                padding=10,
                padding_mode="circular",
                bias=False,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=1,
                out_channels=10,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=10,
                out_channels=1,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
        )
        for p in self.parameters():
            nn.init.normal_(p, 0, 0.8)

    def forward(self, x, steps):
        seq = [x]
        for _ in range(steps):
            x = torch.tanh(x + self.net(x))
            seq.append(x)

        return torch.stack(seq, dim=1)


class NCASim(nn.Module):
    def __init__(self, in_size) -> None:
        super().__init__()
        self.ca = NCA()
        self.decoder = nn.Sequential(
            nn.Linear(64 * 64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.Sigmoid(),
        )

    def seed(self, bs, msg_size):
        torch.rand()


class BasicNCA(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.kernel = nn.Sequential(
            conv_same(1, 1, ks=11),
        )
        for p in self.kernel.parameters():
            nn.init.uniform_(p)

        self.rule = nn.Sequential(
            Lambda(lambda x: x),
            nn.Linear(1, 10),
            nn.ReLU(),
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, 1),
        )

        self.conv_rule = nn.Sequential(
            Permute([0, 3, 2, 1]),
            self.rule,
            Permute([0, 3, 2, 1]),
        )

    def forward(self, x, steps):
        seq = [x]
        for i in range(steps):
            out = self.kernel(x)
            out = self.conv_rule(out)
            out = torch.sigmoid(out) * 2 - 1
            x = torch.clip(x + out, 0, 1)
            seq.append(x)

        return seq


class MsgEncoder(nn.Module):
    def __init__(self, msg_size, im_size):
        super().__init__()
        self.net = nn.Sequential(Lambda(), nn.Conv2d(1))
