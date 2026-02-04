"""
altered from VanillaUnet by changing the Up block and the forward pass
to remove all skip connections
"""

import torch.nn as nn

from src.unet.unet import UNet


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.pool_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2), DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.pool_conv(x)


class Up(nn.Module):
    """
    Upsampling block without skip connections:

    - Restores spatial resolution using ConvTranspose2d.
    - Applies a DoubleConv on the upsampled feature map.
    """

    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()
        # Upsample the feature map
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        # After upsampling, use a double convolution.
        self.conv = DoubleConv(out_channels, out_channels)

    def forward(self, x):
        x = self.up(x)
        return self.conv(x)


class NoSkipUNet(UNet):
    def __init__(self, in_channels: int, out_channels: int):
        super(NoSkipUNet, self).__init__(in_channels, out_channels)

        # Encoder
        self.inc = DoubleConv(in_channels, 64)  # initial double conv
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)

        # Bottleneck
        self.down4 = Down(512, 1024)

        # Decoder (now without skip connections)
        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)

        # Final 1x1 convolution
        self.outc = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        x0 = self.inc(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)

        # Bottleneck
        x4 = self.down4(x3)

        # Decoder (without using encoder skip connections)
        x = self.up1(x4)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)

        # Output
        logits = self.outc(x)
        return logits
