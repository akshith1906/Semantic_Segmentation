from abc import abstractmethod

import torch.nn as nn


class UNet(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(UNet, self).__init__()

    @abstractmethod
    def forward(self, x):
        raise NotImplementedError("forward pass not implemented.")
