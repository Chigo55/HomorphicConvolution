import math

import torch
import torch.nn as nn
from torch import Tensor


class RGB2YCrCbBlock(nn.Module):
    def __init__(
        self,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, bias=False)
        transform = torch.tensor(
            data=[
                [0.299, 0.587, 0.114],
                [0.5, -0.418688, -0.081312],
                [-0.168736, -0.331264, 0.5],
            ],
            dtype=torch.float32,
        ).view(3, 3, 1, 1)
        self.conv.weight = nn.Parameter(data=transform, requires_grad=False)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        ycrcb: Tensor = self.conv(x)
        y, cr, cb = torch.chunk(input=ycrcb, chunks=3, dim=1)
        return y, cr, cb


class YCrCb2RGBBlock(nn.Module):
    def __init__(
        self,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, bias=False)
        transform = torch.tensor(
            data=[
                [1.0, 1.403, 0.0],
                [1.0, -0.714, -0.344],
                [1.0, 0.0, 1.773],
            ],
            dtype=torch.float32,
        ).view(3, 3, 1, 1)
        self.conv.weight = nn.Parameter(data=transform, requires_grad=False)

    def forward(self, ycrcb: Tensor) -> Tensor:
        rgb: Tensor = self.conv(ycrcb)
        return rgb


class HomomorphicSeparationBlock(nn.Module):
    def __init__(
        self,
        kernel_size: int = 21,
        sigma: float = 5.0,
    ) -> None:
        super().__init__()
        self.gaussian_blur = self._build_gaussian_blur(
            kernel_size=kernel_size, sigma=sigma
        )

    def _build_gaussian_blur(self, kernel_size: int, sigma: float) -> nn.Module:
        x_cord = torch.arange(end=kernel_size)
        x_grid = x_cord.repeat(kernel_size).view(kernel_size, kernel_size)
        y_grid = x_grid.t()
        xy_grid = torch.stack(tensors=[x_grid, y_grid], dim=-1).float()
        mean = (kernel_size - 1) / 2.0
        variance = sigma**2.0
        gaussian_kernel = (1.0 / (2.0 * math.pi * variance)) * torch.exp(
            input=-torch.sum(input=(xy_grid - mean) ** 2.0, dim=-1) / (2 * variance)
        )
        gaussian_kernel = gaussian_kernel / torch.sum(input=gaussian_kernel)
        gaussian_weights = gaussian_kernel.view(1, 1, kernel_size, kernel_size)

        gaussian_blur = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=kernel_size,
            padding=(kernel_size // 2),
            bias=False,
        )
        gaussian_blur.weight = nn.Parameter(data=gaussian_weights, requires_grad=False)
        return gaussian_blur

    def forward(
        self,
        y: Tensor,
    ) -> tuple[Tensor, Tensor]:
        original_dtype = y.dtype

        y_f32: Tensor = y.float()
        y_clamped_f32: Tensor = torch.clamp(input=y_f32, min=1e-6)
        y_log_f32: Tensor = torch.log(input=y_clamped_f32)

        y_log_original: Tensor = y_log_f32.to(dtype=original_dtype)
        low_log_original: Tensor = self.gaussian_blur(y_log_original)

        low_log_f32: Tensor = low_log_original.float()
        LOG_CLAMP_MAX = 80.0
        low_log_f32 = torch.clamp(input=low_log_f32, max=LOG_CLAMP_MAX)
        il_f32: Tensor = torch.exp(input=low_log_f32)

        il_f32_safe = torch.clamp(input=il_f32, min=1e-6)
        re_f32: Tensor = y_clamped_f32 / il_f32_safe
        re_f32 = torch.clamp(input=re_f32, min=0.0, max=1.0)
        return il_f32.to(dtype=original_dtype), re_f32.to(dtype=original_dtype)


class ImageDecomposition(nn.Module):
    def __init__(
        self,
        kernel_size,
        sigma,
    ) -> None:
        super().__init__()

        self.rgb2ycrcb: RGB2YCrCbBlock = RGB2YCrCbBlock()
        self.homomorphic: HomomorphicSeparationBlock = HomomorphicSeparationBlock(
            kernel_size=kernel_size,
            sigma=sigma,
        )

    def forward(
        self,
        x: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        y, cr, cb = self.rgb2ycrcb(x)
        il, re = self.homomorphic(y)
        return y, cr, cb, il, re


class ImageComposition(nn.Module):
    def __init__(
        self,
    ) -> None:
        super().__init__()

        self.ycrcb2rgb: YCrCb2RGBBlock = YCrCb2RGBBlock()

    def forward(
        self,
        cr: Tensor,
        cb: Tensor,
        y: Tensor,
    ) -> Tensor:
        ycrcb: Tensor = torch.cat(tensors=[y, cr, cb], dim=1)
        img_enh: Tensor = self.ycrcb2rgb(ycrcb)
        return img_enh
