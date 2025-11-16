import torch
import torch.nn as nn
from torch import Tensor


class ReflectanceGuidedFusion(nn.Module):
    def __init__(self, hidden_channels: int):
        super().__init__()
        self.param_net = nn.Sequential(
            nn.Conv2d(
                in_channels=2,
                out_channels=hidden_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=hidden_channels,
                out_channels=1,
                kernel_size=3,
                padding=1,
            ),
            nn.Tanh(),
        )

    def forward(self, il_enh: Tensor, re: Tensor) -> tuple[Tensor, Tensor]:
        y_base = il_enh * re

        concat_input = torch.cat(tensors=[il_enh, re], dim=1)
        alpha = self.param_net(concat_input)

        y_enh = y_base + alpha * y_base * (1 - y_base)

        y_enh = torch.clamp(input=y_enh, min=0.0, max=1.0)
        return y_enh, alpha
