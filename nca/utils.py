import numpy as np
import lightning as L
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.profilers import AdvancedProfiler
from lightning.pytorch.callbacks import DeviceStatsMonitor
import lightning as L
import torch

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


class Reshape(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.reshape(-1, *self.shape)


class LinerInDim(nn.Module):
    """A linear layer applied at a specific dimension"""

    def __init__(self, in_size, out_size, dim=-1):
        super().__init__()
        self.dim = dim
        self.linear = nn.Linear(in_size, out_size)

    def forward(self, x):
        x_dims = list(range(len(x.shape)))
        x_dims[-1], x_dims[self.dim] = x_dims[self.dim], x_dims[-1]
        x = x.permute(*x_dims)
        x = self.linear(x)
        x = x.permute(*x_dims)

        return x


class Lambda(nn.Module):
    def __init__(self, handler):
        super().__init__()
        self.handler = handler

    def forward(self, x):
        return self.handler(x)


class Permute(Lambda):
    def __init__(self, permutation):
        super().__init__(lambda x: x.permute(permutation))


def conv11(in_channels, out_channels, bias):
    return nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=1,
        padding=0,
        stride=1,
        bias=True,
    )


def conv_same(in_channels, out_channels, ks, bias=False):
    return nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=ks,
        padding=ks // 2,
        stride=1,
        bias=bias,
        padding_mode="circular",
    )



def get_lightning_trainer(model_name, max_epochs, device="cpu"):
    # optim_metric = "metric_val_mean_F1"
    # optim_metric_mode = "max"
    optim_metric = "val_loss"
    optim_metric_mode = "min"

    logger = TensorBoardLogger("logs/", name=model_name)
    early_stop = EarlyStopping(
        monitor=optim_metric, mode=optim_metric_mode, patience=50
    )
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/",
        save_top_k=5,
        monitor=optim_metric,
        mode=optim_metric_mode,
    )
    profiler = AdvancedProfiler(dirpath=".", filename="perf_logs")
    # device_monitor = DeviceStatsMonitor(cpu_stats=False)

    trainer = L.Trainer(
        max_epochs=max_epochs,
        logger=logger,
        callbacks=[early_stop, checkpoint_callback],
        profiler=profiler,
        gradient_clip_val=0.1,
        accelerator=device,
    )

    return trainer
