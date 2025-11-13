from typing import Any

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


class TotalLoss(nn.Module):
    def __init__(
        self,
        lambda_mae: float = 1.0,
        lambda_mse: float = 1.0,
    ) -> None:
        super().__init__()

        self.lambda_mae = lambda_mae
        self.lambda_mse = lambda_mse

        self.loss_mae = MeanAbsoluteError()
        self.loss_mse = MeanSquaredError()

    def forward(
        self,
        input: Tensor,
        target: Tensor,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        l_mae = self.loss_mae(input, target)
        l_mse = self.loss_mse(input, target)

        total_loss = self.lambda_mae * l_mae + self.lambda_mse * l_mse

        loss_dict = {
            "loss_total": total_loss.detach(),
            "loss_mae": self.lambda_mae * l_mae.detach(),
            "loss_mse": self.lambda_mse * l_mse.detach(),
        }

        return total_loss, loss_dict
