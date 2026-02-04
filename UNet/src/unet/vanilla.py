"""
references:
- https://arxiv.org/abs/1505.04597
- https://www.youtube.com/watch?v=u1loyDCoGbE
"""

import torch
import torch.nn as nn

from src.unet.unet import UNet


class DoubleConv(nn.Module):
    """
    convolutional block

    - 'each convolutional block consists of two convolutional layers'
    - the layers seem to preserve spatial dimensions
    - kernel size of 3 is picked from the original paper
    - padding of 1 is used as a consequence of the above two
    """

    def __init__(self, in_channels: int, out_channels: int):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        """
        (B, in_channels, H, W) -> (B, out_channels, H, W, out_channels)
        """
        return self.conv(x)


class Down(nn.Module):
    """
    downsampling block

    - maxpool dimensions picked from the original paper
    """

    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()

        self.pool_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2), DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        """
        (B, H, W, in_channels) -> (B, H/2, W/2, out_channels)
        """
        return self.pool_conv(x)


def _crop_tensor(x: torch.Tensor, target_tensor: torch.Tensor) -> torch.Tensor:
    """
    (B, C, H, W) -> (B, C, target_H, target_W)

    - H, W from x
    - target_H, target_W from target_tensor
    """

    _, _, H, W = x.shape
    _, _, target_H, target_W = target_tensor.shape
    assert H == W and target_H == target_W, "only square tensors are supported"
    assert H >= target_H, "target tensor must be smaller than the input tensor"

    delta = (H - target_H) // 2

    start, end = delta, delta + target_H

    cropped_x = x[:, :, start:end, start:end]
    return cropped_x


class Up(nn.Module):
    """
    upsampling block

    - feature resolution restored using ConvTranspose2d
    - convolutional blocks used to restore the number of channels
    """

    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        """
        x1: (B, in_channels, H, W) -> (B, out_channels, 2*H, 2*W)
        x2: (B, in_channels, H, W) -> (B, in_channels, H of x1, W of x1)

        output: (B, out_channels, H of x1, W of x1)
        """

        # x1 is the decoder feature map, x2 is the encoder feature map (skip connection)
        x1 = self.up(x1)

        # NOTE:
        # if necessary, crop x2 to match x1's spatial dimensions
        # this does not seem necessary for the given implementation, but
        # was implemented in the original U-Net paper, and is done for
        # generality
        x2 = _crop_tensor(x2, x1.detach().clone())

        # Concatenate along the channels dimension
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class VanillaUNet(UNet):
    def __init__(self, in_channels: int, out_channels: int):
        super(VanillaUNet, self).__init__(in_channels, out_channels)

        # encoder
        self.inc = DoubleConv(in_channels, 64)  # initial double conv
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)

        # Bottleneck (4th "down" block or deeper layer)
        self.down4 = Down(512, 1024)

        # Decoder
        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)

        # Final 1x1 convolution
        self.outc = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        x0 = self.inc(x)  # Level 0
        x1 = self.down1(x0)  # Level 1
        x2 = self.down2(x1)  # Level 2
        x3 = self.down3(x2)  # Level 3

        # Bottleneck
        x4 = self.down4(x3)  # Level 4

        # Decoder
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.up4(x, x0)

        # Output
        logits = self.outc(x)
        return logits
