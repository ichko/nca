import mediapy as mpy
import torch


def tonp(t):
    return t.detach().cpu().numpy()


def nca_out_to_vids(out, height=150):
    rgb_out = mpy.to_rgb(
        tonp(torch.stack(out).transpose(0, 1)[:, :, 0]), vmin=0, vmax=1, cmap="viridis"
    )

    return mpy.show_videos(
        rgb_out, height=height, fps=20, codec="gif", border=True, columns=16
    )
