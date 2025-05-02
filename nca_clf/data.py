from dataclasses import dataclass
import cv2
import numpy as np
import torch
from torchvision.datasets.mnist import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset, DataLoader


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


def _generate_radial_circles_pattern(size, num_classes):
    pattern = np.zeros((size, size))
    for i, tau in enumerate(np.linspace(0, np.pi * 2, num_classes, endpoint=False)):
        sr = 8
        r = 27
        x = size // 2 + int(np.cos(tau) * r)
        y = size // 2 + int(np.sin(tau) * r)
        cv2.circle(pattern, [x, y], sr, i + 1, thickness=-1)

    return torch.tensor(pattern)


def _to_nca_example(ds_item, channs, pattern):
    H, W = pattern.shape
    assert H == W, "we assume W and H are equal"
    size = H

    x, y = ds_item
    xs = x.shape[-1]
    inp = torch.zeros(channs, size, size)
    f = size // 2 - xs // 2
    inp[0, f : f + xs, f : f + xs] = x[0]
    out = (pattern == y + 1).float()

    return inp, out


@dataclass(frozen=True)
class _Sample:
    batch: torch.Tensor
    index: torch.tensor


class MNISTRadialCirclesGenerator:
    def __init__(self, is_train, num_classes, channs, size, bs):
        self.pattern = _generate_radial_circles_pattern(size, num_classes)
        mnist = MNIST("~/.cache", download=True, transform=ToTensor(), train=is_train)
        ds = MappedDataset(
            mnist, lambda item: _to_nca_example(item, channs, self.pattern)
        )
        dl = DataLoader(ds, batch_size=bs, shuffle=is_train)
        self.dl = iterate_forever(dl)

    def __next__(self):
        return next(self.dl)


class MNISTRadialCirclesPool:
    def __init__(self, is_train, num_classes, channs, size, pool_size, replacement=0.1):
        self.pattern = _generate_radial_circles_pattern(size, num_classes)
        self.replacement = replacement
        mnist = MNIST("~/.cache", download=True, transform=ToTensor(), train=is_train)
        ds = MappedDataset(
            mnist, lambda item: _to_nca_example(item, channs, self.pattern)
        )
        if is_train:
            ds = ShuffleDataset(ds)
        self.gen = iterate_forever(ds)

        self.pools = (
            torch.zeros(pool_size, channs, size, size),
            torch.zeros(pool_size, size, size),
        )

        for i in range(pool_size):
            inp, out = next(self.gen)
            inps, outs = self.pools
            inps[i] = inp
            outs[i] = out

    def sample(self, bs):
        inps, outs = self.pools
        index = np.random.choice(len(inps), bs)
        return _Sample(batch=[inps[index], outs[index]], index=index)

    def update(self, sample: _Sample, out_preds, losses):
        inps, outs = self.pools
        inps[sample.index] = out_preds.detach().cpu()

        replacement_size = int(len(losses) * self.replacement)
        worst_loss_indices = losses.argsort()[-replacement_size:].cpu()
        worst_loss_pool_indices = sample.index[worst_loss_indices]

        for index in worst_loss_pool_indices:
            inp, out = next(self.gen)
            inps[index] = inp
            outs[index] = out
