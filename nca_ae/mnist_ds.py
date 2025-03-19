import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
from torchvision.datasets import MNIST


def _draw_image(screen, pos, image):
    x, y = pos
    h, w = image.shape[:2]
    screen[y : y + w, x : x + w] = image


class MNISTDataset(Dataset):
    def __init__(self, W, H, train=True):
        super().__init__()
        self.W, self.H = W, H
        self.ds = MNIST(root="./.data", train=train, download=True)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        pil, target = self.ds[index]
        np_im = np.array(pil)
        # plt.ion()
        # plt.imshow(np_im)
        # plt.savefig("res.png")
        inp_screen = np.zeros((self.H, self.W))
        _draw_image(inp_screen, (10, 15), np_im)  # TODO: Hard coded

        return {"inp_screen": inp_screen}


if __name__ == "__main__":
    ds = MNISTDataset(128, 64)
    item = ds[0]
    plt.imshow(item["inp_screen"])
    plt.savefig("res.png")
    print("done!")
