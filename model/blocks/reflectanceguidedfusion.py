import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class ReflectanceGuidedFusion(nn.Module):
    def __init__(self, hidden_channels: int):
        super().__init__()

        self.re_refiner = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=hidden_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=hidden_channels,
                out_channels=1,
                kernel_size=3,
                padding=1,
            ),
            nn.Sigmoid(),
        )

        self.param_net = nn.Sequential(
            nn.Conv2d(
                in_channels=2,
                out_channels=hidden_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=hidden_channels,
                out_channels=1,
                kernel_size=3,
                padding=1,
            ),
            nn.Tanh(),
        )

        nn.init.constant_(tensor=self.param_net[-2].weight, val=0.0)
        nn.init.constant_(tensor=self.param_net[-2].bias, val=0.0)

        nn.init.xavier_normal_(tensor=self.re_refiner[0].weight)

    def forward(self, il_enh: Tensor, re: Tensor) -> tuple[Tensor, Tensor]:
        re_ref = self.re_refiner(re)

        y_base = il_enh * re_ref

        concat_input = torch.cat(tensors=[il_enh, re], dim=1)
        curve_map = self.param_net(concat_input)

        y_enh = y_base + curve_map * y_base * (1 - y_base)

        y_enh = torch.clamp(input=y_enh, min=0.0, max=1.0)
        return y_enh, re_ref
