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
                out_channels=1,
                kernel_size=15,
                stride=1,
                padding=7,
                bias=False,
            ),
            nn.ReLU6(),
            nn.Conv2d(
                in_channels=1,
                out_channels=5,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.ReLU6(),
            nn.Conv2d(
                in_channels=5,
                out_channels=2,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
        )
        nn.init.uniform_(self.net[0].weight)
        nn.init.zeros_(self.net[0].weight[:, :, 5:10, 5:10])

    def forward(self, x, steps):
        seq = [x]
        for _ in range(steps):
            out = self.net(x)
            new_x = x + F.tanh(out[:, :1])
            lerp = torch.sigmoid(out[:, 1:])
            x = lerp * new_x + (1 - lerp) * x
            seq.append(x)

        return torch.stack(seq, dim=1)


if __name__ == "__main__":
    seq_len = 60

    multi_seqs = []
    for i in range(49):
        model = NCA()

        inp = torch.rand(1, 1, 64, 64)
        seq = model.forward(inp, steps=seq_len)
        seq = seq.detach().cpu().numpy()
        seq = seq[:, :, 0]
        cmap = cm.get_cmap("viridis")
        seq = cmap(seq)[..., :3]
        bs, _seq_len, h, w, c = seq.shape
        seq = torch.tensor(seq).permute(0, 1, 4, 2, 3)

        multi_seqs.append(seq[0])
    multi_seqs = torch.stack(multi_seqs, dim=0)

    with utils.VideoWriter(filename="vid.ignore.mp4", fps=5) as vid:
        for i in range(seq_len):
            frame_batch = multi_seqs[:, i]
            frame_batch = torchvision.utils.make_grid(frame_batch, padding=2, nrow=7)
            frame_batch = frame_batch.permute(1, 2, 0)

            vid.add(utils.zoom(frame_batch, scale=3))
