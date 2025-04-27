import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


# SRC: https://github.com/chenmingxiang110/Growing-Neural-Cellular-Automata/blob/master/lib/CAModel.py
class NCAModel(nn.Module):
    def __init__(self, channel_n, fire_rate, device, hidden_size=128):
        super().__init__()

        self.device = device
        self.channel_n = channel_n

        self.fc0 = nn.Linear(channel_n * 3, hidden_size)
        self.fc1 = nn.Linear(hidden_size, channel_n, bias=False)
        init.zeros_(self.fc1.weight)

        self.fire_rate = fire_rate
        self.to(self.device)

    def alive(self, x):
        return F.max_pool2d(x[:, :1, :, :], kernel_size=3, stride=1, padding=0) > 0.1

    def perceive(self, x, angle):
        def _perceive_with(x, weight):
            conv_weights = torch.from_numpy(weight.astype(np.float32)).to(self.device)
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

    def update(self, x, fire_rate, angle):
        x_padded = F.pad(x, (1, 1, 1, 1), "circular")
        pre_life_mask = self.alive(x_padded)
        y1, y2 = self.perceive(x_padded, angle)
        dx = torch.cat([x, y1, y2], dim=1)

        dx = dx.permute(0, 2, 3, 1)
        dx = self.fc0(dx)
        dx = F.relu(dx)
        dx = self.fc1(dx)
        dx = dx.permute(0, 3, 1, 2)

        if fire_rate is None:
            fire_rate = self.fire_rate
        stochastic = torch.rand([dx.size(0), dx.size(1), dx.size(2), 1]) > fire_rate
        stochastic = stochastic.float().to(self.device)
        dx = dx * stochastic

        x = x + dx

        post_life_mask = self.alive(F.pad(x, (1, 1, 1, 1), "circular"))
        life_mask = (pre_life_mask & post_life_mask).float()
        x = x * life_mask
        return x

    def forward(self, x, steps=1, fire_rate=None, angle=0.0):
        seq = [x]
        for step in range(steps):
            x = self.update(x, fire_rate, angle)
            seq.append(x)
        return seq
