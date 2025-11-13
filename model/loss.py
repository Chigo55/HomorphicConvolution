from typing import Any

import pyiqa
import torch
import torch.nn as nn
from torch import Tensor


class MeanAbsoluteError(nn.L1Loss):
    def __init__(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)

    def forward(
        self,
        input: Tensor,
        target: Tensor,
    ) -> Tensor:
        return super().forward(input=input, target=target)


class MeanSquaredError(nn.MSELoss):
    def __init__(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)

    def forward(
        self,
        input: Tensor,
        target: Tensor,
    ) -> Tensor:
        return super().forward(input=input, target=target)


class StructuralSimilarity(nn.Module):
    def __init__(
        self,
        device: str = "cuda",
    ) -> None:
        super().__init__()
        self.ssim = pyiqa.create_metric(
            metric_name="ssimc",
            device=device,
            as_loss=True,
        )

    def forward(
        self,
        input: Tensor,
        targets: Tensor,
    ) -> Tensor:
        return 1 - torch.mean(input=self.ssim(input, targets))
