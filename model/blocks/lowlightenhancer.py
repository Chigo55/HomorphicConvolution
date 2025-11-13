import torch
import torch.nn as nn
from torch import Tensor

from model.blocks.homomorphic import ImageComposition, ImageDecomposition
from model.blocks.illuminationenhancer import IlluminationEnhancer
from model.blocks.iterablerefine import IterableRefine


class LowLightEnhancer(nn.Module):
    def __init__(
        self,
        hidden_channels: int,
        num_resolution: int,
        cutoff: float,
        offset: float,
    ) -> None:
        super().__init__()

        self.decomposition: ImageDecomposition = ImageDecomposition(
            offset=offset,
            cutoff=cutoff,
        )

        self.illumination_enhancer: IlluminationEnhancer = IlluminationEnhancer(
            in_channels=1,
            out_channels=1,
            hidden_channels=hidden_channels,
            num_resolution=num_resolution,
        )

        self.Iterable_refine: IterableRefine = IterableRefine(
            hidden_channels=hidden_channels,
        )

        self.composition: ImageComposition = ImageComposition(
            offset=offset,
        )

    def forward(self, low: Tensor) -> dict[str, Tensor]:
        y, cr, cb, il, re = self.decomposition(low)

        il_enh = self.illumination_enhancer(il)

        il_ref = self.Iterable_refine(il_enh)

        img_enh, y_enh = self.composition(
            cr,
            cb,
            il_ref,
            re,
        )
        img_enh = torch.clamp(input=img_enh, min=0.0, max=1.0)

        outputs: dict[str, Tensor] = {
            "low_luminance": y,
            "low_chroma_red": cr,
            "low_chroma_blue": cb,
            "low_illuminance": il,
            "low_reflectance": re,
            "low_rgb": low,
            "enh_illuminance": il_enh,
            "ref_illuminance": il_ref,
            "enh_luminance": y_enh,
            "enh_rgb": img_enh,
        }
        return outputs
