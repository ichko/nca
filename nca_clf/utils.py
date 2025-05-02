import os
import mediapy as mpy
import numpy as np
import torch
from datetime import datetime


def nca_cmap(seq, vmin=0, vmax=1, cmap="viridis"):
    # out.shape == [seq, batch, channs, H, W]
    seq = seq.swapaxes(0, 1)
    seq = seq[:, :, 0]
    seq = seq.detach().cpu().numpy()
    seq = mpy.to_rgb(seq, vmin=vmin, vmax=vmax, cmap=cmap)
    seq = seq.transpose(0, 1, 4, 2, 3)
    seq = (seq * 255).astype(np.uint8)
    return seq


def tonp(t):
    return t.detach().cpu().numpy()


def nca_out_to_vids(out, height=150, columns=16, fps=20):
    rgb_out = mpy.to_rgb(
        tonp(out.transpose(0, 1)[:, :, 0]), vmin=0, vmax=1, cmap="viridis"
    )

    return mpy.show_videos(
        rgb_out, height=height, fps=fps, codec="gif", border=True, columns=columns
    )


def save_model(model, path):
    dir, file_name = os.path.split(path)
    os.makedirs(dir, exist_ok=True)

    now = datetime.now()
    now = now.strftime("[%Y-%m-%d-%H-%M-%S]")
    file_name = file_name.format(now=now)

    with open(os.path.join(dir, file_name), "wb+") as fp:
        torch.save(model, fp)
        return fp.name


def load_latest_model(dir):
    paths = [os.path.abspath(os.path.join(dir, f)) for f in os.listdir(dir)]
    paths = [p for p in paths if os.path.isfile(p)]
    paths = sorted(paths, key=os.path.getmtime)
    latest_file = paths[-1]

    with open(latest_file, "rb") as fp:
        return torch.load(fp)
