"""
modified from Vanilla UNet
"""

import torch
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


def _crop_tensor(x: torch.Tensor, target_tensor: torch.Tensor) -> torch.Tensor:
    _, _, H, W = x.shape
    _, _, target_H, target_W = target_tensor.shape
    assert H == W and target_H == target_W, "only square tensors are supported"
    assert H >= target_H, "target tensor must be smaller than the input tensor"

    delta = (H - target_H) // 2
    start, end = delta, delta + target_H
    return x[:, :, start:end, start:end]


# --- Attention Gate Module ---
# ref: https://github.com/LeeJunHyun/Image_Segmentation/blob/5e9da9395c52b119d55dfc6532c34ac0e88f446e/network.py#L108


class AttentionGate(nn.Module):
    """
    Additive attention gate as described in 'Attention U-Net: Where to look for the Pancreas'
    It computes an attention coefficient for the encoder feature map (x) using the gating signal (g)
    from the decoder.

    Both x and g are first mapped to an intermediate number of channels F_int (here F_int = F_l // 2)
        based on the above reference.
    """

    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()

        # params selected to retain spatial dims as in the image
        self.W_g = nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True)
        self.W_x = nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True)

        # 1x1x1 as in the image
        self.psi = nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, g):
        """
        x: encoder feature map (skip connection) with F_l channels.
        g: gating signal from decoder with F_g channels.
        Returns the encoder feature map x weighted by the attention map.
        """
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        f = self.relu(g1 + x1)
        psi = self.sigmoid(self.psi(f))
        # not multiplying by anything as alpha is fixed to 1
        # alpha is just for scaling
        return x * psi


# --- Modified Up block using Attention Gate in skip connection ---


class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.att_gate = AttentionGate(
            F_g=out_channels, F_l=out_channels, F_int=out_channels // 2
        )
        # After concatenation, the channel count is doubled (skip + upsampled decoder features)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        """
        x1: feature map from decoder (to be upsampled; gating signal).
        x2: feature map from encoder (skip connection).
        """
        x1 = self.up(x1)
        x2 = _crop_tensor(x2, x1.detach().clone())
        # Apply attention gate on the encoder feature map using x1 as gating signal
        x2 = self.att_gate(x2, x1)
        # Concatenate along the channel dimension and apply DoubleConv
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class GatedAttentionUNet(UNet):
    def __init__(self, in_channels: int, out_channels: int):
        super(GatedAttentionUNet, self).__init__(in_channels, out_channels)

        # Encoder
        self.inc = DoubleConv(in_channels, 64)  # Level 0
        self.down1 = Down(64, 128)  # Level 1
        self.down2 = Down(128, 256)  # Level 2
        self.down3 = Down(256, 512)  # Level 3

        # Bottleneck
        self.down4 = Down(512, 1024)  # Level 4

        # Decoder with attention in skip connections:
        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)

        self.outc = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder pathway
        x0 = self.inc(x)  # Level 0
        x1 = self.down1(x0)  # Level 1
        x2 = self.down2(x1)  # Level 2
        x3 = self.down3(x2)  # Level 3

        # Bottleneck
        x4 = self.down4(x3)  # Level 4

        # Decoder pathway with attention on skip connections
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.up4(x, x0)

        # Final output
        logits = self.outc(x)
        return logits
