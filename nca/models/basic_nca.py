import torch.nn as nn
import torch


class BasicNCA(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        kernel_size = 3

        self.kernel = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=1,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                stride=1,
                bias=False,
            ),
        )
        for p in self.kernel.parameters():
            nn.init.ones_(p)

        self.rule = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=10,
                kernel_size=1,
                padding=0,
                stride=1,
                bias=True,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=10,
                out_channels=1,
                kernel_size=1,
                padding=0,
                stride=1,
                bias=False,
            ),
            nn.Sigmoid(),
        )

    def forward(self, x, steps):
        seq = [x]
        for i in range(steps):
            out = self.kernel(x)
            out = torch.exp(-((out - 1) ** 2))
            out = self.rule(out)
            x = x + (out - 0.5)

            seq.append(x)

        return seq
