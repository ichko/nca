from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from kornia import augmentation
from tqdm.auto import tqdm
from nca.utils import Lambda, Permute, conv_same
import numpy as np


class BaseModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_history = []

    def fit(self, its, *args, **kwargs):
        pbar = tqdm(range(its))
        for it in pbar:
            optim_res = self.optim_step(*args, **kwargs)
            loss, info = optim_res
            pbar.set_description(f"Loss: {loss:0.5f}")
            self.loss_history.append(loss)

    def plot_loss(self):
        fig, ax = plt.subplots(dpi=120, figsize=(8, 4))
        ax.plot(range(len(self.loss_history)), self.loss_history)
        ax.set_yscale("log")

        plt.close()
        return fig

    @property
    def device(self):
        return next(self.parameters()).device


class BaselineNCA(BaseModule):
    def __init__(self, hidden_n=6, zero_w2=True, device="cuda"):
        super().__init__()
        self.filters = torch.stack(
            [
                torch.tensor([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]),
                torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]),
                torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]).T,
                torch.tensor([[1.0, 2.0, 1.0], [2.0, -12, 2.0], [1.0, 2.0, 1.0]]),
            ]
        ).to(device)
        # self.filters = torch.randn_like(self.filters)
        self.chn = 4

        self.w1 = nn.Conv2d(4 * 4, hidden_n, 1).to(device)
        self.relu = nn.ReLU()
        self.w2 = nn.Conv2d(hidden_n, 4, 1, bias=True).to(device)
        self.w3 = nn.Conv2d(hidden_n, 4, 1).to(device)

        if zero_w2:
            self.w2.weight.data.zero_()

    def perchannel_conv(self, x, filters):
        b, ch, h, w = x.shape
        y = x.reshape(b * ch, 1, h, w)
        y = torch.nn.functional.pad(y, [1, 1, 1, 1], "circular")
        y = torch.nn.functional.conv2d(y, filters[:, None])
        return y.reshape(b, -1, h, w)

    def forward(self, x, update_rate=0.5):
        y = self.perchannel_conv(x, self.filters)
        hid = self.relu(self.w1(y))
        y = self.w2(hid)
        y = torch.tanh(y)
        l = torch.sigmoid(self.w3(hid))

        return x * l + (1 - l) * y

    def forward_many(self, x, steps):
        seq = [x]
        for _ in range(steps):
            x = self.forward(x)
            seq.append(x)

        return seq


class FCInvAE(BaseModule):
    """Fully connected Inverted auto-encoder"""

    def __init__(self, msg_size, frame_size) -> None:
        super().__init__()
        self.msg_size = msg_size
        self.frame_size = frame_size

        self.encoder = nn.Sequential(
            nn.Linear(msg_size, 100),
            nn.BatchNorm1d(100),
            nn.LeakyReLU(),
            nn.Linear(100, 100),
            nn.BatchNorm1d(100),
            nn.LeakyReLU(),
            nn.Linear(100, frame_size * frame_size),
            nn.Sigmoid(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(frame_size * frame_size, 100),
            nn.BatchNorm1d(100),
            nn.LeakyReLU(),
            nn.Linear(100, 100),
            nn.BatchNorm1d(100),
            nn.LeakyReLU(),
            nn.Linear(100, msg_size),
            nn.Sigmoid(),
        )

        noise_size = 1
        self.noiser = nn.Sequential(
            augmentation.RandomAffine(
                degrees=(-10, 10), translate=(0.1, 0.1), scale=(0.9, 1.1), p=noise_size
            ),
            # augmentation.RandomBoxBlur(kernel_size=(5, 5), p=noise_size),
            # augmentation.RandomErasing((0.1, 0.2), (0.3, 1 / 0.3), p=noise_size),
            # augmentation.RandomJigsaw(grid=(4, 4), p=noise_size),
            augmentation.RandomGaussianNoise(0.5, 1, same_on_batch=False, p=noise_size),
        )

    def noise(self, frame, noise_size):
        return self.noiser(frame)

    def encode(self, msg):
        x = self.encoder(msg)
        return x.reshape(-1, 1, self.frame_size, self.frame_size)

    def decode(self, frame):
        bs = frame.shape[0]
        frame = frame.reshape(bs, -1)
        x = self.decoder(frame)
        return x

    def forward_msg(self, msg, noise_size):
        generated_image = self.encode(msg)
        noised_image = self.noise(generated_image, noise_size)
        decoded_msg = self.decode(noised_image)
        return {
            "msg": msg,
            "image": generated_image,
            "noised_image": noised_image,
            "decoded_msg": decoded_msg,
        }

    def forward(self, bs, noise_size):
        msg = self.sample_msg(bs)
        return self.forward_msg(msg, noise_size)

    def optim_step(self, bs, noise_size, lr):
        optim = torch.optim.Adam(self.parameters(), lr=lr)
        out = self.forward(bs, noise_size)
        loss = F.mse_loss(out["decoded_msg"], out["msg"])

        optim.zero_grad()
        loss.backward()
        optim.step()

        return loss.item(), out

    def sample_msg(self, bs):
        return torch.rand(bs, self.msg_size).to(self.device)

    def render_out(self, out, size=3):
        out = {k: v.detach().cpu().numpy() for k, v in out.items()}
        bs = out["msg"].shape[0]

        rows = bs
        cols = 5
        fig, axs = plt.subplots(rows, cols, dpi=100, figsize=(cols * size, rows * size))
        plt.tight_layout()
        if rows == 1:
            axs = [axs]

        for ax in [ax[-1] for ax in axs]:
            ax.set_ylim(0, 1)

        for b in range(bs):
            if b == 0:
                axs[b][0].set_title("input msg")
                axs[b][1].set_title("generated image")
                axs[b][2].set_title("noised image")
                axs[b][3].set_title("decoded msg")
                axs[b][4].set_title("msg difference")

            # axs[b][0].axis("off")
            axs[b][1].axis("off")
            axs[b][2].axis("off")
            # axs[b][3].axis("off")

            axs[b][0].bar(range(out["msg"].shape[1]), out["msg"][b])
            axs[b][1].imshow(out["image"][b][0])
            axs[b][2].imshow(out["noised_image"][b][0])
            axs[b][3].bar(range(out["msg"].shape[1]), out["decoded_msg"][b])
            diff = out["msg"][b] - out["decoded_msg"][b]
            axs[b][4].bar(
                range(len(diff)),
                abs(diff),
                color=np.array(["red", "blue"])[(diff > 0).astype(int)],
            )

        plt.close()
        return fig
