import torch.nn as nn
import torch
from nca.utils import LinerInDim, Permute


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


def growth(U):
    """Bosco's rule, b1..b2 is birth range, s1..s2 is stable range (outside s1..s2 is shrink range)"""
    b1, b2, s1, s2 = 34, 45, 34, 58
    return ((U >= b1) & (U <= b2)).to(torch.float32) - ((U < s1) | (U > s2)).to(
        torch.float32
    )


class BasicNCA(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.kernel = nn.Sequential(
            conv_same(1, 1, ks=11),
        )
        # for p in self.kernel.parameters():
        #     nn.init.ones_(p)

        self.rule = nn.Sequential(
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

            x = torch.clip(x + out, 0, 1)

            seq.append(x)

        return seq
