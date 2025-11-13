import math
from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor


class RGB2YCrCbBlock(nn.Module):
    def __init__(
        self,
        offset: float,
    ) -> None:
        super().__init__()
        transform = torch.tensor(
            data=[
                [0.299, 0.587, 0.114],
                [0.5, -0.418688, -0.081312],
                [-0.168736, -0.331264, 0.5],
            ],
            dtype=torch.float32,
        )
        bias = torch.tensor(data=[0.0, offset, offset], dtype=torch.float32).view(
            1, 3, 1, 1
        )
        self.register_buffer(name="transform_matrix", tensor=transform)
        self.register_buffer(name="chrominance_bias", tensor=bias)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        transform = self.transform_matrix.to(dtype=x.dtype, device=x.device)
        bias = self.chrominance_bias.to(dtype=x.dtype, device=x.device)
        ycrcb: Tensor = torch.einsum("bchw,dc->bdhw", x, transform) + bias
        y, cr, cb = torch.chunk(input=ycrcb, chunks=3, dim=1)
        return y, cr, cb


class YCrCb2RGBBlock(nn.Module):
    def __init__(
        self,
        offset: float,
    ) -> None:
        super().__init__()
        transform = torch.tensor(
            data=[
                [1.0, 1.403, 0.0],
                [1.0, -0.714, -0.344],
                [1.0, 0.0, 1.773],
            ],
            dtype=torch.float32,
        )
        bias = torch.tensor(data=[0.0, offset, offset], dtype=torch.float32).view(
            1, 3, 1, 1
        )
        self.register_buffer(name="transform_matrix", tensor=transform)
        self.register_buffer(name="chrominance_bias", tensor=bias)

    def forward(
        self,
        y: Tensor,
        cr: Tensor,
        cb: Tensor,
    ) -> Tensor:
        ycrcb: Tensor = torch.cat(tensors=[y, cr, cb], dim=1)
        transform = self.transform_matrix.to(dtype=ycrcb.dtype, device=ycrcb.device)
        bias = self.chrominance_bias.to(dtype=ycrcb.dtype, device=ycrcb.device)
        centered: Tensor = ycrcb - bias
        rgb: Tensor = torch.einsum("bchw,dc->bdhw", centered, transform)
        return rgb


class HomomorphicSeparationBlock(nn.Module):
    def __init__(
        self,
        cutoff: float,
    ) -> None:
        super().__init__()
        cutoff_tensor = torch.tensor(data=float(cutoff), dtype=torch.float32)
        self.register_buffer(name="cutoff", tensor=cutoff_tensor)
        self._sigma_denom: float = math.sqrt(math.log(2.0))
        self._filter_cache: dict[
            tuple[int, int, torch.device, torch.dtype], Tensor
        ] = {}

    def _gaussian_lpf(
        self,
        size: tuple[int, int],
        reference: Tensor,
    ) -> Tensor:
        key = (size[0], size[1], reference.device, reference.dtype)
        cached = self._filter_cache.get(key)
        if cached is not None:
            return cached

        height, width = size
        device = reference.device
        dtype = reference.dtype

        fy: Tensor = torch.fft.fftfreq(height, d=1.0, device=device, dtype=dtype)
        fx: Tensor = torch.fft.fftfreq(width, d=1.0, device=device, dtype=dtype)
        fy = torch.fft.fftshift(fy)
        fx = torch.fft.fftshift(fx)

        y, x = torch.meshgrid(fy, fx, indexing="ij")
        radius: Tensor = torch.hypot(input=x, other=y)

        cutoff = self.cutoff.to(device=device, dtype=dtype)
        sigma = cutoff / reference.new_tensor(data=self._sigma_denom)
        denominator = 2.0 * sigma * sigma
        h: Tensor = torch.exp(input=-(radius * radius) / denominator)
        h = h.unsqueeze(dim=0).unsqueeze(dim=0)
        self._filter_cache[key] = h
        return h

    def forward(
        self,
        x: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        height, width = x.shape[-2:]

        x_log: Tensor = torch.log(input=torch.clamp(input=x, min=1e-5))

        x_fft: Tensor = torch.fft.fft2(x_log, norm="ortho")
        x_fft = torch.fft.fftshift(x_fft)

        h: Tensor = self._gaussian_lpf(size=(height, width), reference=x)
        h = h.to(dtype=x_fft.dtype)
        low_fft: Tensor = x_fft * h

        low_log: torch.Tensor = torch.fft.ifft2(
            torch.fft.ifftshift(low_fft), norm="ortho"
        ).real
        high_log: torch.Tensor = x_log - low_log

        il: Tensor = torch.exp(input=low_log)
        re: Tensor = torch.exp(input=high_log)
        return il, re


class ImageDecomposition(nn.Module):
    def __init__(
        self,
        offset: float,
        cutoff: float,
    ) -> None:
        super().__init__()
        self.rgb2ycrcb: RGB2YCrCbBlock = RGB2YCrCbBlock(
            offset=offset,
        )
        self.homomorphic: HomomorphicSeparationBlock = HomomorphicSeparationBlock(
            cutoff=cutoff,
        )
        self.ycrcb2rgb: YCrCb2RGBBlock = YCrCb2RGBBlock(
            offset=offset,
        )

    def forward(
        self,
        x: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        y, cr, cb = self.rgb2ycrcb(x)
        il, re = self.homomorphic(y)
        return y, cr, cb, il, re


class ImageComposition(nn.Module):
    def __init__(
        self,
        offset: float,
    ) -> None:
        super().__init__()
        self.ycrcb2rgb: YCrCb2RGBBlock = YCrCb2RGBBlock(
            offset=offset,
        )

    def forward(
        self,
        cr: Tensor,
        cb: Tensor,
        il: Tensor,
        re: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        y_enh: Tensor = il * re
        img_enh: Tensor = self.ycrcb2rgb(y_enh, cr, cb)
        return img_enh, y_enh
