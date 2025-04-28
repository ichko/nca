import cv2
import numpy as np
import torch


def generate_start_pattern(S, num_classes):
    screen = np.zeros((S, S))

    for i, tau in enumerate(np.linspace(0, np.pi * 2, num_classes, endpoint=False)):
        sr = 10
        r = 25

        x1 = S // 2 + int(np.cos(tau) * sr)
        y1 = S // 2 + int(np.sin(tau) * sr)
        x2 = S // 2 + int(np.cos(tau) * r)
        y2 = S // 2 + int(np.sin(tau) * r)
        cv2.line(screen, [x1, y1], [x2, y2], color=1, thickness=2)

    return torch.tensor(screen)


class RadialCirclesProcessor:
    def __init__(self, num_classes, S) -> None:
        self.S = S
        screen = np.zeros((S, S))

        for i, tau in enumerate(np.linspace(0, np.pi * 2, num_classes, endpoint=False)):
            sr = 8
            r = 27
            x = S // 2 + int(np.cos(tau) * r)
            y = S // 2 + int(np.sin(tau) * r)
            cv2.circle(screen, [x, y], sr, i + 1, thickness=-1)

            # x1 = S // 2 + int(np.cos(tau) * sr)
            # y1 = S // 2 + int(np.sin(tau) * sr)
            # x2 = S // 2 + int(np.cos(tau) * r)
            # y2 = S // 2 + int(np.sin(tau) * r)

            # cv2.line(screen, [x1, y1], [x2, y2], color=i + 1, thickness=2)
            # cv2.circle(screen, [x,  y], sr * 3 - i * 3, i + 1, thickness=-1)
            # cv2.polylines(screen, [[x, y], [x]])

        self.screen = torch.tensor(screen)

    def map_batch(self, batch, chans):
        x, y = batch
        bs, _, H, W = x.shape
        s = x.shape[-1]
        inp = torch.zeros(bs, chans, self.S, self.S)
        f = self.S // 2 - s // 2
        inp[:, :3, f : f + s, f : f + s] = x

        out = (self.screen.unsqueeze(0) == y.unsqueeze(1).unsqueeze(1) + 1).float()

        return inp, out
