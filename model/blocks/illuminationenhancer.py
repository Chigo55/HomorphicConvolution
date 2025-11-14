import torch
import torch.nn as nn
from torch import Tensor


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__()
        self.conv1: nn.Conv2d = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn1: nn.BatchNorm2d = nn.BatchNorm2d(num_features=out_channels)
        self.act1: nn.ReLU = nn.ReLU()

        self.conv2: nn.Conv2d = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2: nn.BatchNorm2d = nn.BatchNorm2d(num_features=out_channels)
        self.act2: nn.ReLU = nn.ReLU()

        if in_channels != out_channels:
            self.skip_proj: nn.Module = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
            )
        else:
            self.skip_proj = nn.Identity()

    def forward(
        self,
        x: Tensor,
    ) -> Tensor:
        x1: Tensor = self.act1(self.bn1(self.conv1(x)))
        x2: Tensor = self.act2(self.bn2(self.conv2(x1)))

        residual: Tensor = self.skip_proj(x) + x2
        return residual


class DoubleConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__()
        self.conv1: ResidualBlock = ResidualBlock(
            in_channels=in_channels,
            out_channels=hidden_channels,
        )
        self.conv2: ResidualBlock = ResidualBlock(
            in_channels=hidden_channels,
            out_channels=out_channels,
        )

    def forward(
        self,
        x: Tensor,
    ) -> Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class Downsampling(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__()
        self.conv: nn.Conv2d = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=2,
            stride=2,
            padding=0,
            bias=False,
        )

    def forward(
        self,
        x: Tensor,
    ) -> Tensor:
        return self.conv(x)


class Upsampling(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__()
        self.conv: nn.ConvTranspose2d = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=2,
            stride=2,
            padding=0,
            bias=False,
        )

    def forward(
        self,
        x: Tensor,
    ) -> Tensor:
        return self.conv(x)


class IlluminationEnhancer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        num_resolution: int,
    ) -> None:
        super().__init__()

        self.in_conv: nn.Conv2d = nn.Conv2d(
            in_channels=in_channels,
            out_channels=hidden_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.in_act: nn.Sigmoid = nn.Sigmoid()

        hidden_level = hidden_channels
        down: list[nn.Module] = []
        for level in range(num_resolution):
            down.append(
                DoubleConv(
                    in_channels=hidden_level,
                    hidden_channels=hidden_level,
                    out_channels=hidden_level,
                )
            )
            down.append(
                Downsampling(
                    in_channels=hidden_level,
                    out_channels=hidden_level * 2,
                )
            )
            hidden_level *= 2
        self.down: nn.ModuleList = nn.ModuleList(modules=down)

        mid: list[nn.Module] = []
        for _ in range(num_resolution // 2):
            mid.append(
                DoubleConv(
                    in_channels=hidden_level,
                    hidden_channels=hidden_level,
                    out_channels=hidden_level,
                )
            )
        self.mid: nn.ModuleList = nn.ModuleList(modules=mid)

        up: list[nn.Module] = []
        for level in range(num_resolution):
            up.append(
                Upsampling(
                    in_channels=hidden_level,
                    out_channels=hidden_level // 2,
                )
            )
            up.append(
                DoubleConv(
                    in_channels=hidden_level,
                    hidden_channels=hidden_level,
                    out_channels=hidden_level // 2,
                )
            )
            hidden_level //= 2

        self.up: nn.ModuleList = nn.ModuleList(modules=up)

        self.out_conv: nn.Conv2d = nn.Conv2d(
            in_channels=hidden_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.out_act: nn.Sigmoid = nn.Sigmoid()

    def forward(
        self,
        x: Tensor,
    ) -> Tensor:
        x = self.in_conv(x)
        x = self.in_act(x)

        residuals: list[Tensor] = []
        for module in self.down:
            if isinstance(module, Downsampling):
                residuals.append(x)
                x = module(x)
            else:
                x = module(x)

        for module in self.mid:
            x = module(x)

        for module in self.up:
            if isinstance(module, Upsampling):
                x = module(x)
                x = torch.cat(tensors=[x, residuals.pop()], dim=1)
            else:
                x = module(x)

        x = self.out_conv(x)
        x = self.out_act(x)
        return x
