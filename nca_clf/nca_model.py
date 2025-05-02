import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


# SRC: https://github.com/chenmingxiang110/Growing-Neural-Cellular-Automata/blob/master/lib/CAModel.py
class NCAModel(nn.Module):
    def __init__(self, channel_n, device, hidden_size=128):
        super().__init__()

        self.device = device
        self.channel_n = channel_n

        self.fc0 = nn.Linear(channel_n * 3, hidden_size)
        self.fc1 = nn.Linear(hidden_size, channel_n, bias=False)
        init.zeros_(self.fc1.weight)

        self.to(self.device)

    def alive(self, x):
        return F.max_pool2d(x[:, :1, :, :], kernel_size=3, stride=1, padding=0) > 0.1

    def perceive(self, x, angle):
        def _perceive_with(x, weight):
            conv_weights = torch.tensor(weight, dtype=x.dtype).to(self.device)
            conv_weights = conv_weights.view(1, 1, 3, 3).repeat(self.channel_n, 1, 1, 1)
            return F.conv2d(x, conv_weights, padding=0, groups=self.channel_n)

        dx = np.outer([1, 2, 1], [-1, 0, 1]) / 8.0  # Sobel filter
        dy = dx.T
        c = np.cos(angle * np.pi / 180)
        s = np.sin(angle * np.pi / 180)
        w1 = c * dx - s * dy
        w2 = s * dx + c * dy

        y1 = _perceive_with(x, w1)
        y2 = _perceive_with(x, w2)
        return y1, y2

    def update(self, x):
        x_padded = F.pad(x, (1, 1, 1, 1), "circular")
        pre_life_mask = self.alive(x_padded)
        y1, y2 = self.perceive(x_padded, 0)
        dx = torch.cat([x, y1, y2], dim=1)

        dx = dx.permute(0, 2, 3, 1)
        dx = self.fc0(dx)
        dx = F.relu(dx)
        dx = self.fc1(dx)
        dx = dx.permute(0, 3, 1, 2)

        x = x + dx

        post_life_mask = self.alive(F.pad(x, (1, 1, 1, 1), "circular"))
        life_mask = (pre_life_mask & post_life_mask).to(x.dtype)
        # x = x * life_mask
        return x

    def forward(self, x, steps=1):
        seq = [x]
        for step in range(steps):
            x = self.update(x)
            seq.append(x)
        return seq


class SimpleNCA(nn.Module):
    def __init__(self, channs, hid=128) -> None:
        super().__init__()
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]) / 8
        sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]) / 8
        identity = torch.tensor([[0, 0, 0], [0, 1, 0], [0, 0, 0]])

        all_filters = torch.stack((identity, sobel_x, sobel_y))
        all_filters_batch = all_filters.repeat(channs, 1, 1).unsqueeze(1)
        all_filters_batch = nn.Parameter(all_filters_batch, requires_grad=False)

        self.channs = channs
        self.all_filters_batch = all_filters_batch
        self.rule = nn.Sequential(
            nn.Conv2d(3 * channs, hid, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(hid, channs, kernel_size=1, bias=False),
        )

        nn.init.zeros_(self.rule[-1].weight)

    def forward(self, x, steps):
        seq = [x]
        for _ in range(steps):
            delta = F.conv2d(
                F.pad(x, (1, 1, 1, 1), "circular"),
                # F.pad(x, (1, 1, 1, 1), "constant", 0),
                self.all_filters_batch,
                groups=self.channs,
            )
            delta = self.rule(delta)
            x = x + delta
            seq.append(x)

        seq = torch.stack(seq)
        return seq
