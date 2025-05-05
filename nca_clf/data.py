from dataclasses import dataclass
import cv2
import numpy as np
import torch
from torchvision.datasets.mnist import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset, DataLoader
import mediapy as mpy

MNIST_ROOT = "~/.cache"


def visualize_mnist_classes():
    mnist = MNIST(MNIST_ROOT, download=True, transform=ToTensor(), train=True)

    groups = {}
    for i in range(200):
        t, l = mnist[i]
        groups.setdefault(l, []).append([t[0], i])

    for i in range(10):
        ims = [im for im, i in groups[i]]
        idxs = [str(i) for im, i in groups[i]]
        mpy.show_images(ims, idxs, width=120, cmap="viridis")


def generate_canonical_mnist_digits():
    mnist = MNIST(MNIST_ROOT, download=True, transform=ToTensor(), train=True)
    canon_ids = [1, 6, 5, 7, 64, 175, 126, 38, 97, 162]
    ims = [mnist[i][0][0] for i in canon_ids]
    return torch.stack(ims)


class MappedDataset(Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, index):
        data = self.dataset[index]
        return self.transform(data)

    def __len__(self):
        return len(self.dataset)


class ShuffleDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.index_map = np.random.permutation(len(dataset))

    def __getitem__(self, idx):
        real_idx = self.index_map[idx]
        return self.dataset[real_idx]

    def __len__(self):
        return len(self.index_map)


def iterate_forever(iterable):
    it = iter(iterable)
    while True:
        try:
            yield next(it)
        except StopIteration:
            it = iter(iterable)


def generate_radial_circles_pattern(size, num_classes, sr=8, r=27):
    pattern = np.zeros((num_classes, size, size))
    for i, tau in enumerate(np.linspace(0, np.pi * 2, num_classes, endpoint=False)):
        x = size // 2 + int(np.cos(tau) * r)
        y = size // 2 + int(np.sin(tau) * r)
        cv2.circle(pattern[i], [x, y], sr, 1, thickness=-1)
    return torch.tensor(pattern)


def _to_nca_example(ds_item, channs, pattern):
    _, H, W = pattern.shape
    assert H == W, "we assume W and H are equal"
    size = H

    x, y = ds_item
    xs = x.shape[-1]
    inp = torch.zeros(channs, size, size)
    f = size // 2 - xs // 2
    inp[0, f : f + xs, f : f + xs] = x[0]
    out = (pattern[y]).float()

    return {"inp": inp, "out": out, "label": y}


@dataclass(frozen=True)
class _Sample:
    batch: dict
    index: torch.tensor


class MNISTPatternGenerator:
    def __init__(self, is_train, channs, bs, pattern, loop_forever=True, shuffle=True):
        self.pattern = pattern
        mnist = MNIST(MNIST_ROOT, download=True, transform=ToTensor(), train=is_train)
        self.ds = MappedDataset(
            mnist, lambda item: _to_nca_example(item, channs, self.pattern)
        )
        self.channs = channs
        dl = DataLoader(self.ds, batch_size=bs, shuffle=shuffle)
        if loop_forever:
            self.it = iterate_forever(dl)
        else:
            self.it = iter(dl)

    def __len__(self):
        return len(self.ds)

    def __iter__(self):
        return self

    def __next__(self):
        return next(self.it)


class MNISTPatternPool:
    def __init__(self, is_train, channs, size, pool_size, pattern, replacement):
        self.pattern = pattern
        self.replacement = replacement
        mnist = MNIST(MNIST_ROOT, download=True, transform=ToTensor(), train=is_train)
        ds = MappedDataset(
            mnist, lambda item: _to_nca_example(item, channs, self.pattern)
        )
        if is_train:
            ds = ShuffleDataset(ds)
        self.gen = iterate_forever(ds)

        self.pools = (
            torch.zeros(pool_size, channs, size, size),
            torch.zeros(pool_size, size, size),
            torch.zeros(pool_size, dtype=torch.int64),
        )

        for i in range(pool_size):
            batch = next(self.gen)
            inps, outs, labels = self.pools
            inps[i] = batch["inp"]
            outs[i] = batch["out"]
            labels[i] = batch["label"]

    def sample(self, bs):
        inps, outs, labels = self.pools
        index = np.random.choice(len(inps), bs)
        return _Sample(
            batch={"inp": inps[index], "out": outs[index], "label": labels[index]},
            index=index,
        )

    def update(self, sample: _Sample, out_preds, losses):
        inps, outs, labels = self.pools
        inps[sample.index] = out_preds.detach().cpu()

        replacement_size = int(len(losses) * self.replacement)
        rand_indices = torch.randint(0, len(losses), size=(replacement_size,))
        # worst_loss_indices = losses.argsort()[-replacement_size:].cpu()
        worst_loss_pool_indices = sample.index[rand_indices]
        for index in worst_loss_pool_indices:
            batch = next(self.gen)
            inps[index] = batch["inp"]
            outs[index] = batch["out"]
            labels[index] = batch["label"]
