import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from matplotlib import cm
import matplotlib.pyplot as plt
import utils


class NCA(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=64,
                out_channels=1,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.Tanh(),
        )

    def forward(self, x, steps):
        seq = [x]
        for _ in range(steps):
            x = x + self.net(x)
            seq.append(x)

        return torch.stack(seq, dim=1)


if __name__ == "__main__":
    model = NCA()

    inp = torch.rand(16, 1, 32, 32)
    seq = model.forward(inp, steps=60)
    seq = seq.detach().cpu().numpy()
    seq = seq[:, :, 0]
    cmap = cm.get_cmap("viridis")
    seq = cmap(seq)[..., :3]
    bs, seq_len, h, w, c = seq.shape
    seq = torch.tensor(seq).permute(0, 1, 4, 2, 3)

    grid_seq = []

    with utils.VideoWriter(filename="vid.ignore.mp4", fps=10) as vid:
        for i in range(seq_len):
            frame_batch = seq[:, i]
            frame_batch = torchvision.utils.make_grid(frame_batch, padding=2, nrow=4)
            frame_batch = frame_batch.permute(1, 2, 0)
            grid_seq.append(frame_batch)

            vid.add(utils.zoom(frame_batch, scale=5))
