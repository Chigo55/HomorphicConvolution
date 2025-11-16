import torch
import torch.nn as nn
from torch import Tensor

from model.blocks.homomorphic import ImageComposition, ImageDecomposition
from model.blocks.illuminationenhancer import IlluminationEnhancer
from model.blocks.reflectanceguidedfusion import ReflectanceGuidedFusion


class LowLightEnhancer(nn.Module):
    def __init__(
        self,
        hidden_channels: int,
        num_resolution: int,
        kernel_size,
        sigma,
    ) -> None:
        super().__init__()

        self.decomposition: ImageDecomposition = ImageDecomposition(
            kernel_size=kernel_size,
            sigma=sigma,
        )

        self.illumination_enhancer: IlluminationEnhancer = IlluminationEnhancer(
            in_channels=1,
            out_channels=1,
            hidden_channels=hidden_channels,
            num_resolution=num_resolution,
        )

        self.refiner: ReflectanceGuidedFusion = ReflectanceGuidedFusion(
            hidden_channels=hidden_channels,
        )

        self.composition: ImageComposition = ImageComposition()

    def forward(self, low: Tensor) -> dict[str, Tensor]:
        y, cr, cb, il, re = self.decomposition(low)

        il_enh = self.illumination_enhancer(il)

        y_enh, alpha = self.refiner(il_enh, re)

        img_enh = self.composition(cr, cb, y_enh)
        img_enh = torch.clamp(input=img_enh, min=0.0, max=1.0)

        outputs: dict[str, Tensor] = {
            "low_rgb": low,
            "low_luminance": y,
            "low_chroma_red": cr,
            "low_chroma_blue": cb,
            "low_illuminance": il,
            "low_reflectance": re,
            "alpha_component": alpha,
            "enh_illuminance": il_enh,
            "enh_luminance": y_enh,
            "enh_rgb": img_enh,
        }
        return outputs
