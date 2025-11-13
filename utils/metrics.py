import pyiqa
import torch.nn as nn
from torch import Tensor


class ImageQualityMetrics(nn.Module):
    def __init__(self, device: str = "cuda") -> None:
        super().__init__()
        self.device_type: str = device

        self.psnr: nn.Module = pyiqa.create_metric(
            metric_name="psnr",
            device=device,
        )
        self.ssim: nn.Module = pyiqa.create_metric(
            metric_name="ssim",
            device=device,
        )
        self.lpips: nn.Module = pyiqa.create_metric(
            metric_name="lpips",
            device=device,
        )
        self.brisque: nn.Module = pyiqa.create_metric(
            metric_name="brisque",
            device=device,
        )
        self.niqe: nn.Module = pyiqa.create_metric(
            metric_name="niqe",
            device=device,
        )

    def forward(self, preds: Tensor, targets: Tensor) -> dict[str, float]:
        preds = preds.to(device=self.device_type)
        targets = targets.to(device=self.device_type)

        return {
            "PSNR": self.psnr(preds, targets).mean().item(),
            "SSIM": self.ssim(preds, targets).mean().item(),
            "LPIPS": self.lpips(preds, targets).mean().item(),
        }

    def no_ref(self, preds: Tensor) -> dict[str, float]:
        preds = preds.to(device=self.device_type)

        return {
            "BRISQUE": self.brisque(preds).mean().item(),
            "NIQE": self.niqe(preds).mean().item(),
        }

    def full(self, preds: Tensor, targets: Tensor) -> dict[str, float]:
        ref_metrics = self.forward(preds=preds, targets=targets)
        no_ref_metrics = self.no_ref(preds=preds)
        return {**ref_metrics, **no_ref_metrics}
