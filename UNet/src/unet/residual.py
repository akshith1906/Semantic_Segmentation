import torch
import torch.nn as nn

from src.unet.unet import UNet


class ResidualBlock(nn.Module):
    """
    Residual convolutional block.

    Consists of two 3x3 convolutions (with padding=1 to preserve dimensions)
    and a skip connection that adds the input to the output.
    If the number of input and output channels differ, a 1x1 convolution is applied
    on the skip connection to match the dimensions.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.skip_conv = None
        if in_channels != out_channels:
            self.skip_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        identity = x if self.skip_conv is None else self.skip_conv(x)
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out += identity
        out = self.relu(out)
        return out


class Down(nn.Module):
    """
    Downsampling block.

    Applies a 2x2 max pool to reduce the spatial dimensions,
    then uses a ResidualBlock to process the features.
    """

    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.pool_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            ResidualBlock(in_channels, out_channels),
        )

    def forward(self, x):
        return self.pool_conv(x)


def _crop_tensor(x: torch.Tensor, target_tensor: torch.Tensor) -> torch.Tensor:
    _, _, H, W = x.shape
    _, _, target_H, target_W = target_tensor.shape
    assert H == W and target_H == target_W, "only square tensors are supported"
    assert H >= target_H, "target tensor must be smaller than the input tensor"

    delta = (H - target_H) // 2
    cropped_x = x[:, :, delta : delta + target_H, delta : delta + target_W]
    return cropped_x


class Up(nn.Module):
    """
    Upsampling block with skip connections.

    Uses ConvTranspose2d for upsampling, then concatenates the upsampled decoder features
    with the corresponding encoder features. A ResidualBlock is then applied to the concatenated tensor.
    """

    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = ResidualBlock(out_channels * 2, out_channels)

    def forward(self, x1, x2):
        # x1: decoder feature map, x2: corresponding encoder feature map (skip connection)
        x1 = self.up(x1)
        x2 = _crop_tensor(x2, x1.detach().clone())
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class ResidualUNet(UNet):
    def __init__(self, in_channels: int, out_channels: int):
        super(ResidualUNet, self).__init__(in_channels, out_channels)

        # Encoder
        self.inc = ResidualBlock(in_channels, 64)  # initial block
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)

        # Bottleneck
        self.down4 = Down(512, 1024)

        # Decoder with skip connections
        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)

        # Final 1x1 convolution for segmentation map
        self.outc = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        x0 = self.inc(x)  # Level 0
        x1 = self.down1(x0)  # Level 1
        x2 = self.down2(x1)  # Level 2
        x3 = self.down3(x2)  # Level 3

        # Bottleneck
        x4 = self.down4(x3)  # Level 4

        # Decoder with skip connections from corresponding encoder levels
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.up4(x, x0)

        # Output layer
        logits = self.outc(x)
        return logits
