from enum import Enum

import torch.nn as nn
from torchvision.models import vgg16, VGG16_Weights


class FCNVariant(Enum):
    FCN8S = "fcn8s"
    FCN16S = "fcn16s"
    FCN32S = "fcn32s"


class FCN(nn.Module):
    """
    see ../docs/ref.png
    """

    def __init__(self, variant: FCNVariant, num_classes: int):
        super().__init__()
        self.variant = variant

        # load a VGG16 backbone with pretrained weights.
        vgg = vgg16(weights=VGG16_Weights.DEFAULT)
        # for name, module in vgg.features.named_children():
        #     print(f"{name}: {module}")

        # we only extract the feature extractors, we convolutionise the classifier
        features = list(vgg.features.children())

        # define feature extractors:
        # pool3: up to layer 16 (inclusive) produces features at 1/8 resolution.
        self.pool3 = nn.Sequential(*features[:17])
        # pool4: layers 17 to 23 produce features at 1/16 resolution.
        self.pool4 = nn.Sequential(*features[17:24])
        # pool5: layers 24 to 30 produce features at 1/32 resolution.
        self.pool5 = nn.Sequential(*features[24:31])

        # NOTE:
        # this is what the paper calls "convolutionalising" the classifier
        # score layers: 1x1 convolutions to map features to the desired number of classes.
        if self.variant == FCNVariant.FCN8S:
            self.score_pool3 = nn.Conv2d(256, num_classes, kernel_size=1)
        if self.variant in [FCNVariant.FCN8S, FCNVariant.FCN16S]:
            self.score_pool4 = nn.Conv2d(512, num_classes, kernel_size=1)
        self.score_pool5 = nn.Conv2d(512, num_classes, kernel_size=1)

        # NOTE:
        # values of kernel size and stride from: https://github.com/wkentaro/pytorch-fcn/tree/main/torchfcn/models
        self.upscore2 = nn.ConvTranspose2d(
            num_classes, num_classes, kernel_size=4, stride=2, padding=1
        )

        # final upsampling layer depends on the variant.
        match self.variant:
            case FCNVariant.FCN32S:
                self.final_upsampler = nn.ConvTranspose2d(
                    num_classes, num_classes, kernel_size=64, stride=32, padding=16
                )
            case FCNVariant.FCN16S:
                self.final_upsampler = nn.ConvTranspose2d(
                    num_classes, num_classes, kernel_size=32, stride=16, padding=8
                )
            case FCNVariant.FCN8S:
                self.final_upsampler = nn.ConvTranspose2d(
                    num_classes, num_classes, kernel_size=16, stride=8, padding=4
                )

    def freeze_backbone(self):
        for layer in [self.pool3, self.pool4, self.pool5]:
            for param in layer.parameters():
                param.requires_grad = False

    def forward(self, x):
        pool3 = self.pool3(x)  # (B, 256, H/8, W/8)
        pool4 = self.pool4(pool3)  # (B, 512, H/16, W/16)
        pool5 = self.pool5(pool4)  # (B, 512, H/32, W/32)

        # initial score map
        score5 = self.score_pool5(pool5)  # (B, num_classes, H/32, W/32)

        if self.variant == FCNVariant.FCN32S:
            return self.final_upsampler(score5)

        # For FCN-16s: fuse with pool4.
        score4 = self.score_pool4(pool4)  # (B, num_classes, H/16, W/16)
        upscore2 = self.upscore2(score5)  # upsample score5 to H/16, W/16
        fused = score4 + upscore2

        if self.variant == FCNVariant.FCN16S:
            return self.final_upsampler(fused)

        # For FCN-8s: fuse with pool3.
        score3 = self.score_pool3(pool3)  # (B, num_classes, H/8, W/8)
        upscore2_2 = self.upscore2(fused)  # upsample fused score to H/8, W/8
        fused_final = score3 + upscore2_2
        return self.final_upsampler(fused_final)


def get_project_name(variant: FCNVariant, freeze_backbone: bool):
    return f"road-segmentation-{variant.value}-{'freeze' if freeze_backbone else 'unfreeze'}"
