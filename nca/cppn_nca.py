import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from matplotlib import cm
import matplotlib.pyplot as plt
import utils
from tqdm import tqdm


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


if __name__ == "__main__":
    seq_len = 60

    multi_seqs = []
    inp = torch.zeros(1, 1, 64, 64)
    nn.init.uniform_(inp[0, 0, 20:40, 20:40])
    inp = (inp > 0.9).float()
    cmap = cm.get_cmap("viridis")

    for i in tqdm(range(49)):
        model = NCA()

        seq = model.forward(inp, steps=seq_len)
        seq = seq.detach().cpu().numpy()
        mi, ma = seq.min(), seq.max()
        seq = (seq[:, :, 0] - mi) / (ma - mi)
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
