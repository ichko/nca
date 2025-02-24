import cv2
import numpy as np
import torch
import torchsummary
from aesthetic_tensor import monkey_patch_torch
from torch.utils.data import DataLoader
from torchdata.datapipes.iter import IterableWrapper
from torchvision.datasets.mnist import MNIST
from torchvision.transforms import ToTensor

from nca.models import NCAAE

monkey_patch_torch()

W, H = 128, 64


def get_divider_screen():
    screen = np.zeros((H, W))
    pad = 10

    cv2.line(screen, [W // 2, 0], [W // 2, H // 2 - pad], color=1, thickness=2)
    cv2.line(screen, [W // 2, H // 2 + pad], [W // 2, H], color=1, thickness=2)

    divider_screen = torch.tensor(screen, dtype=torch.float32)
    divider_screen.ae.zoom(2).img
    return divider_screen


def get_data_generator(bs):
    num_classes = 10
    ds = MNIST(root="./.data", download=True, transform=ToTensor(), train=True)

    dl = DataLoader(
        IterableWrapper(ds).filter(lambda x: x[1] < num_classes),
        batch_size=bs,
        shuffle=True,
    )

    def map_input(dl):
        it = iter(dl)
        for x, y in it:
            bs, _, h, w = x.shape
            c = nca.chans
            inp_screen = torch.zeros(bs, c, H, W)
            out_screen = torch.zeros(bs, 1, H, W)
            pad = 30
            inp_screen[
                :,
                :1,
                H // 2 - h // 2 : H // 2 + h // 2,
                pad - w // 2 : pad + w // 2,
            ] = x
            out_screen[
                :,
                :1,
                H // 2 - h // 2 : H // 2 + h // 2,
                W - pad - w // 2 : W - pad + w // 2,
            ] = x
            yield inp_screen, out_screen

    gen = map_input(dl)
    batch = next(gen)
    x, y = batch

    torch.cat([x[:8, :1], y[:8, :1]]).ae.cmap(dim=2).grid()[0, :3].img

    return map_input


def sanity_check(nca):
    inp1 = torch.rand(2, nca.chans, H, W) * 2 - 1
    inp2 = torch.zeros(1, nca.chans, H, W)
    inp2[0, 0, 32, 32] = 1

    inp3 = torch.ones(1, nca.chans, H, W)
    inp3[0, 0, 32, 32] = 0

    inp = torch.cat([inp1, inp2, inp3])
    out = nca(inp, steps=20)

    torchsummary.summary(nca.to("cuda"), input_size=inp1.size()[1:])


if __name__ == "__main__":
    divider_screen = get_divider_screen()
    nca = NCAAE(divider_screen)

    sanity_check(nca)

    # out.ae.zoom(3).grid(ncols=4, pad=4)[:, 0].gif(fps=24).save()
