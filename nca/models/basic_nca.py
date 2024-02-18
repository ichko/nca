import torch.nn as nn
import torch


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


class BasicNCA(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.kernel = nn.Sequential(
            conv_same(1, 1, ks=5),
        )
        for p in self.kernel.parameters():
            nn.init.uniform_(p)

        self.rule = nn.Sequential(
            conv11(1, 10, bias=False),
            nn.Tanh(),
            conv11(10, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x, steps):
        seq = [x]
        for i in range(steps):
            out = self.kernel(x)
            out = torch.exp(-((out - 2) ** 2) * 2) * 2 - 1
            # out = self.rule(out)

            x = torch.clip(x + out, 0, 1)

            seq.append(x)

        return seq
