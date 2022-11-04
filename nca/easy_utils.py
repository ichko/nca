import ipywidgets as w
from IPython.display import display, clear_output
from tqdm.auto import tqdm
import torch
import torch.nn.functional as F
import torchvision
from PIL import Image

import matplotlib.pyplot as plt
import matplotlib as mpl
import threading
import time


class ThreadDraw:
    def __init__(self):
        self.running = True
        self.t = threading.Thread(target=self.run)
        self.out = w.Output()
        self.draw_handler = None
        self.t.start()

    def stop(self):
        self.running = False

    def draw(self, draw_handler):
        self.draw_handler = draw_handler

    def run(self):
        i = 0
        while self.running:
            i += 1
            time.sleep(0.1)
            with self.out:
                if self.draw_handler is not None:
                    clear_output(wait=True)
                    self.draw_handler()
                    self.draw_handler = None

    def __del__(self):
        self.stop()


class LazyUI:
    td = None

    def __init__(self, handler):
        if LazyUI.td is None:
            LazyUI.td = ThreadDraw()

        LazyUI.td.draw(lambda: handler(EasyUI))

    def show(self):
        clear_output(wait=True)
        display(LazyUI.td.out)


class EasyUI:
    def __init__(self, matrix):
        self.container = w.VBox(
            [w.HBox([EasyUI.show(el) for el in row]) for row in matrix]
        )

    @staticmethod
    def plot(*args, **kwargs):
        out = w.Output()
        with out:
            plt.plot(*args, **kwargs)
            plt.show()
        return out

    @staticmethod
    def print(*args, **kwargs):
        out = w.Output()
        with out:
            print(*args, **kwargs)
        return out

    @staticmethod
    def show(*args, **kwargs):
        out = w.Output()
        with out:
            display(*args, **kwargs)
        return out

    def display(self):
        display(self.container)


class EasyTensor:
    def __init__(self, target):
        self.target = target

    def __getitem__(self, key):
        return EasyTensor(self.target.__getitem__(key))

    def dim_shift(self, size):
        shape = self.target.shape
        ndim = len(shape) - 1  # no batch
        shift = [((d - size) % ndim) + 1 for d in range(ndim)]
        return EasyTensor(self.target.permute(0, *shift))

    def cmap(self, cm="viridis"):
        cmap = mpl.cm.get_cmap(cm)
        t = torch.tensor(cmap(self.normal.np))
        return EasyTensor(t).dim_shift(1).uint8

    @property
    def info(self):
        mi = self.target.min()
        ma = self.target.max()
        return f"min: {mi}, max: {ma}, shape: {self.target.shape}"

    @property
    def np(self):
        return self.target.detach().cpu().numpy()

    def grid(self, nrow=8, pad=2):
        out = torchvision.utils.make_grid(
            self.target, nrow=nrow, padding=pad
        ).unsqueeze(0)
        return EasyTensor(out)

    @property
    def uint8(self):
        return EasyTensor((self.target * 255).to(torch.uint8))

    @property
    def pil(self):
        return [Image.fromarray(im) for im in self.dim_shift(-1).np]

    def zoom(self, scale=1):
        return EasyTensor(F.interpolate(self.target, scale_factor=scale))

    @property
    def normal(self) -> "EasyTensor":
        t = self.target
        shape = t.shape
        bs = shape[0]
        t_arr = t.reshape(bs, -1)
        mi = t_arr.min(dim=1).values.view(bs, *([1] * (len(shape) - 1)))
        ma = t_arr.max(dim=1).values.view(bs, *([1] * (len(shape) - 1)))
        return EasyTensor((t - mi) / (ma - mi))

    @property
    def raw(self):
        return self.target


def make_torch_tensor_easy():
    torch.Tensor.et = property(lambda self: EasyTensor(self))
    print("done")
