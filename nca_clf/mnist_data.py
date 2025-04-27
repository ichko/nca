from torchvision.datasets.mnist import MNIST
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import ToTensor


class MNISTGenerator:
    def __init__(self, bs, is_train=True, num_classes=10) -> None:
        ds = MNIST(root="~/.cache", download=True, transform=ToTensor(), train=is_train)
        if num_classes == 10:
            subset = ds
        else:
            indices = [i for i, (img, label) in enumerate(ds) if label < num_classes]
            subset = Subset(ds, indices)

        dl = DataLoader(subset, batch_size=bs, shuffle=is_train)

        self.ds = ds
        self.dl = dl
        self.it = iter(self.dl)

    def __next__(self):
        try:
            return next(self.it)
        except StopIteration:
            self.it = iter(self.dl)
            return next(self.it)
