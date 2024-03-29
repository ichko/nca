# @title Imports and Notebook Utilities
import base64
import glob
import io
import json
import os
import zipfile

import matplotlib.pylab as pl
import numpy as np
import PIL.Image
import PIL.ImageDraw
import requests
from IPython.display import HTML, Image, Markdown, clear_output, display
from tqdm import tnrange, tqdm_notebook


def imread(url, max_size=None, mode=None):
    if url.startswith(("http:", "https:")):
        # wikimedia requires a user agent
        headers = {
            "User-Agent": "Requests in Colab/0.0 (https://colab.research.google.com/; no-reply@google.com) requests/0.0"
        }
        r = requests.get(url, headers=headers)
        f = io.BytesIO(r.content)
    else:
        f = url
    img = PIL.Image.open(f)
    if max_size is not None:
        img.thumbnail((max_size, max_size), PIL.Image.ANTIALIAS)
    if mode is not None:
        img = img.convert(mode)
    img = np.float32(img) / 255.0
    return img


def np2pil(a):
    if a.dtype in [np.float32, np.float64]:
        a = np.uint8(np.clip(a, 0, 1) * 255)
    return PIL.Image.fromarray(a)


def imwrite(f, a, fmt=None):
    a = np.asarray(a)
    if isinstance(f, str):
        fmt = f.rsplit(".", 1)[-1].lower()
        if fmt == "jpg":
            fmt = "jpeg"
        f = open(f, "wb")
    np2pil(a).save(f, fmt, quality=95)


def imencode(a, fmt="jpeg"):
    a = np.asarray(a)
    if len(a.shape) == 3 and a.shape[-1] == 4:
        fmt = "png"
    f = io.BytesIO()
    imwrite(f, a, fmt)
    return f.getvalue()


def im2url(a, fmt="jpeg"):
    encoded = imencode(a, fmt)
    base64_byte_string = base64.b64encode(encoded).decode("ascii")
    return "data:image/" + fmt.upper() + ";base64," + base64_byte_string


def imshow(a, fmt="jpeg", id=None):
    return display(Image(data=imencode(a, fmt)), display_id=id)


def grab_plot(close=True):
    """Return the current Matplotlib figure as an image"""
    fig = pl.gcf()
    fig.canvas.draw()
    img = np.array(fig.canvas.renderer._renderer)
    a = np.float32(img[..., 3:] / 255.0)
    img = np.uint8(255 * (1.0 - a) + img[..., :3] * a)  # alpha
    if close:
        pl.close()
    return img


def tile2d(a, w=None):
    a = np.asarray(a)
    if w is None:
        w = int(np.ceil(np.sqrt(len(a))))
    th, tw = a.shape[1:3]
    pad = (w - len(a)) % w
    a = np.pad(a, [(0, pad)] + [(0, 0)] * (a.ndim - 1), "constant")
    h = len(a) // w
    a = a.reshape([h, w] + list(a.shape[1:]))
    a = np.rollaxis(a, 2, 1).reshape([th * h, tw * w] + list(a.shape[4:]))
    return a


class VideoWriter:
    def __init__(self, filename="_autoplay.mp4", fps=30.0, **kw):
        self.writer = None
        self.params = dict(filename=filename, fps=fps, **kw)

    def add(self, img):
        img = np.asarray(img)
        if self.writer is None:
            h, w = img.shape[:2]
            self.writer = FFMPEG_VideoWriter(size=(w, h), **self.params)
        if img.dtype in [np.float32, np.float64]:
            img = np.uint8(img.clip(0, 1) * 255)
        if len(img.shape) == 2:
            img = np.repeat(img[..., None], 3, -1)
        self.writer.write_frame(img)

    def close(self):
        if self.writer:
            self.writer.close()

    def __enter__(self):
        return self

    def __exit__(self, *kw):
        self.close()
        if self.params["filename"] == "_autoplay.mp4":
            self.show()

    def show(self, **kw):
        self.close()
        fn = self.params["filename"]
        display(mvp.ipython_display(fn, **kw))

class LoopWriter(VideoWriter):
    def __init__(self, *a, cross_len=1.0, **kw):
        super().__init__(*a, **kw)
        self._intro = []
        self._outro = []
        self.cross_len = int(cross_len * self.params["fps"])

    def add(self, img):
        if len(self._intro) < self.cross_len:
            self._intro.append(img)
            return
        self._outro.append(img)
        if len(self._outro) > self.cross_len:
            super().add(self._outro.pop(0))

    def close(self):
        for t in np.linspace(0, 1, len(self._intro)):
            img = self._intro.pop(0) * t + self._outro.pop(0) * (1.0 - t)
            super().add(img)
        super().close()
