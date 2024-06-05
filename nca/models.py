from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from kornia import augmentation
from tqdm.auto import tqdm
from nca.utils import Lambda, Permute, conv_same


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
    def __init__(self, lr) -> None:
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
        self.lr = lr

    def forward(self, x, steps):
        seq = [x]
        for i in range(steps):
            out = self.kernel(x)
            out = self.conv_rule(out)
            out = torch.sigmoid(out) * 2 - 1
            x = torch.clip(x + out, 0, 1)
            seq.append(x)

        return seq

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def optim_step(self, batch):
        seed, steps = batch["seed"], batch["steps"]
        self.zero_grad()
        out = self.out(batch["seed"])
        loss = self.criterion()


class FCInvAE(BaseModule):
    """Fully connected Inverted auto-encoder"""

    def __init__(self, msg_size, frame_size) -> None:
        super().__init__()
        self.msg_size = msg_size
        self.frame_size = frame_size

        self.encoder = nn.Sequential(
            nn.Linear(msg_size, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Linear(100, frame_size * frame_size),
            nn.Sigmoid(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(frame_size * frame_size, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Linear(100, msg_size),
            nn.Sigmoid(),
        )

    def noise(self, frame, noise_size):
        noiser = nn.Sequential(
            augmentation.RandomGaussianNoise(0, noise_size, same_on_batch=False, p=1),
            augmentation.RandomAffine(
                degrees=(-10, 10), translate=(0.1, 0.1), scale=(0.9, 1.1), p=noise_size
            ),
        )
        return noiser(frame)

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
            axs[b][4].bar(
                range(out["msg"].shape[1]), out["msg"][b] - out["decoded_msg"][b]
            )

        plt.close()
        return fig
