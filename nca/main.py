import kornia
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm


class Decoder(nn.Module):
    def __init__(self, from_shape, latent_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(np.prod(from_shape), 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, latent_size),
        )

    def forward(self, x):
        return self.net(x)


class NCA(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(
                in_channels=num_channels,
                out_channels=32,
                kernel_size=(3, 3),
                padding=1,
                dilation=1,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=(1, 1),
                padding=0,
                dilation=1,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=32,
                out_channels=num_channels,
                kernel_size=(1, 1),
                padding=0,
                dilation=1,
            ),
            nn.Tanh(),
        )

    def forward(self, x, steps):
        for _ in range(steps):
            x = x + self.net(x)
        return x

    def forward_frames(self, x, steps):
        frames = [x]
        for _ in range(steps):
            x = x + self.net(x)
            frames.append(x)
        return frames

    @staticmethod
    def sanity_check():
        x = torch.rand(10, 3, 32, 32)
        nca = NCA(num_channels=3)
        y = nca(x, steps=10)
        assert y.shape == x.shape


def generate_seed(bs, shape, seed_size):
    c, h, w = shape
    screen = torch.zeros(bs, *shape)
    seed = torch.rand(bs, c, seed_size, seed_size)
    screen[
        :,
        :,
        w // 2 - seed_size // 2 : w // 2 + seed_size // 2,
        h // 2 - seed_size // 2 : h // 2 + seed_size // 2,
    ] = seed

    seed_msg = seed.view(bs, -1)
    return screen, seed_msg


def main():
    torch.autograd.set_detect_anomaly(True)

    NCA.sanity_check()

    noise = nn.Sequential(
        kornia.augmentation.RandomAffine(
            degrees=30,
            translate=[0.15, 0.15],
            scale=[0.85, 1.16],
            shear=[-10, 10],
        ),
        # kornia.augmentation.RandomPerspective(0.6, p=0.5),
        kornia.augmentation.RandomGaussianNoise(mean=0, std=1, p=1),
    )

    steps = 32
    epochs = 1000
    DEVICE = "cuda"
    nca = NCA(num_channels=1).to(DEVICE)
    decoder = Decoder(from_shape=[1, 32, 32], latent_size=16)
    optim = torch.optim.Adam([*nca.parameters(), *decoder.parameters()])

    pbar = tqdm(range(epochs))
    for i in pbar:
        seed, seed_msg = generate_seed(64, shape=[1, 32, 32], seed_size=4)
        seed = seed.to(DEVICE)
        seed_msg = seed_msg.to(DEVICE)
        generated_image = nca(seed, steps=steps)
        pred_image = noise(generated_image)
        pred_msg = decoder(pred_image)
        loss = F.mse_loss(pred_msg, seed_msg)

        optim.zero_grad()
        loss.backward()
        optim.step()

        pbar.set_description(f"Loss: {loss.item():0.5f}")

    fig = plt.figure()

    seed, _ = generate_seed(64, shape=[1, 32, 32], seed_size=6)
    generated_frames = nca.forward_frames(seed, steps=steps)
    frames = [f.detach().cpu().numpy() for f in generated_frames]
    im = plt.imshow(frames[0][0, 0], animated=True)

    i = 0
    b = 0

    def updatefig(*args):
        nonlocal b, i
        i += 1
        if i % 9 == 0:
            i = 0
            b += 1
        f = frames[i][b, 0]
        im.set_array(f)
        return (im,)

    ani = animation.FuncAnimation(fig, updatefig, interval=1000 // 5, blit=True)
    plt.show()


if __name__ == "__main__":
    main()
