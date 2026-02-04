from enum import Enum


from src.unet.unet import UNet
from src.unet.vanilla import VanillaUNet
from src.unet.noskip import NoSkipUNet
from src.unet.residual import ResidualUNet
from src.unet.gated_attention import GatedAttentionUNet


class Variant(Enum):
    Vanilla = "vanilla"
    NoSkip = "noskip"
    Residual = "residual"
    GatedAttention = "gated_attention"


def fetch_unet(variant: Variant) -> type[UNet]:
    match variant:
        case Variant.Vanilla:
            return VanillaUNet
        case Variant.NoSkip:
            return NoSkipUNet
        case Variant.Residual:
            return ResidualUNet
        case Variant.GatedAttention:
            return GatedAttentionUNet


def get_project_name(variant: str):
    return f"road-segmentation-{variant}-unet"
