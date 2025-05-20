import torch
import torch.nn as nn
import pyiqa


class ImageQualityMetrics(nn.Module):
    def __init__(self, device="cuda"):
        super().__init__()
        self.device_type = device

        # pyiqa 기반의 메트릭 생성
        self.psnr = pyiqa.create_metric(
            metric_name='psnr',
            device=device
        )
        self.ssim = pyiqa.create_metric(
            metric_name='ssim',
            device=device
        )
        self.lpips = pyiqa.create_metric(
            metric_name='lpips',
            device=device
        )
        self.brisque = pyiqa.create_metric(
            metric_name='brisque',
            device=device
        )
        self.niqe = pyiqa.create_metric(
            metric_name='niqe',
            device=device
        )

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> dict:
        preds = preds.to(device=self.device_type)
        targets = targets.to(device=self.device_type)

        return {
            "PSNR": self.psnr(preds, targets).mean().item(),
            "SSIM": self.ssim(preds, targets).mean().item(),
            "LPIPS": self.lpips(preds, targets).mean().item(),
        }

    def no_ref(self, preds: torch.Tensor) -> dict:
        preds = preds.to(device=self.device_type)

        return {
            "BRISQUE": self.brisque(preds).mean().item(),
            "NIQE": self.niqe(preds).mean().item(),
        }

    def full(self, preds: torch.Tensor, targets: torch.Tensor) -> dict:
        ref_metrics = self.forward(preds=preds, targets=targets)
        no_ref_metrics = self.no_ref(preds=preds)
        return {**ref_metrics, **no_ref_metrics}
