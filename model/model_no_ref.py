from typing import Any, Literal

import lightning as L
import torch
from torch import Tensor
from torch.optim.adam import Adam
from torch.optim.optimizer import Optimizer

from data.utils import LowLightSample
from model.blocks.lowlightenhancer import LowLightEnhancer
from model.loss_no_ref import TotalLoss
from utils.metrics import ImageQualityMetrics


class LowLightEnhancerLightning(L.LightningModule):
    def __init__(self, hparams: dict[str, Any]) -> None:
        super().__init__()
        torch.autograd.set_detect_anomaly(mode=True)
        self.save_hyperparameters(hparams)

        self.model: LowLightEnhancer = LowLightEnhancer(
            hidden_channels=self.hparams.get("hidden_channels", 32),
            num_resolution=self.hparams.get("num_resolution", 2),
            kernel_size=self.hparams.get("kernel_size", 15),
            sigma=self.hparams.get("sigma", 5.0),
        )

        self.loss: TotalLoss = TotalLoss(
            lambda_spa=self.hparams.get("lambda_spa", 10.0),
            lambda_exp=self.hparams.get("lambda_exp", 100.0),
            lambda_col=self.hparams.get("lambda_col", 100.0),
            lambda_illum=self.hparams.get("lambda_illum", 100.0),
            exp_patch_size=self.hparams.get("exp_patch_size", 16),
            exp_mean_val=self.hparams.get("exp_mean_val", 0.6),
        ).eval()

        self.metric = ImageQualityMetrics().eval()

    def forward(self, low: Tensor) -> dict[str, Tensor]:
        return self.model(low)

    def _calculate_loss(
        self,
        outputs: dict[str, Tensor],
    ) -> tuple[Tensor, dict[str, Tensor]]:
        low_lum = outputs["low_luminance"]
        enh_lum = outputs["enh_luminance"]
        enh_illum = outputs["enh_illuminance"]
        enh_rgb = outputs["enh_rgb"]

        total_loss, loss_dict = self.loss(
            low_luminance=low_lum,
            enh_luminance=enh_lum,
            enh_illuminance=enh_illum,
            enh_rgb=enh_rgb,
        )
        return total_loss, loss_dict

    def _shared_step(
        self,
        batch: LowLightSample,
    ) -> tuple[dict[str, Tensor], Tensor, dict[str, Tensor]]:
        low_img, _ = batch
        outputs = self.forward(low=low_img)
        total_loss, loss_dict = self._calculate_loss(outputs=outputs)
        return outputs, total_loss, loss_dict

    def _logging(
        self,
        stage: Literal["train", "valid"],
        outputs: dict[str, Tensor],
        loss_dict: dict[str, Tensor],
        batch_idx: int,
    ) -> None:
        if batch_idx % 50 != 0:
            return

        logger = self.logger.experiment

        image_keys = [
            "low_luminance",
            "low_chroma_red",
            "low_chroma_blue",
            "low_illuminance",
            "low_reflectance",
            "low_rgb",
            "enh_illuminance",
            "enh_luminance",
            "enh_rgb",
        ]

        for i, key in enumerate(iterable=image_keys):
            if key in outputs:
                logger.add_images(
                    f"{stage}/{i + 1}_{key}", outputs[key], self.global_step
                )

        log_dict = {f"{stage}/{k}": v for k, v in loss_dict.items()}
        self.log_dict(dictionary=log_dict, prog_bar=True)

    def training_step(
        self,
        batch: LowLightSample,
        batch_idx: int,
    ) -> Tensor:
        outputs, total_loss, loss_dict = self._shared_step(batch=batch)
        self._logging(
            stage="train", outputs=outputs, loss_dict=loss_dict, batch_idx=batch_idx
        )
        return total_loss

    def validation_step(
        self,
        batch: LowLightSample,
        batch_idx: int,
    ) -> Tensor:
        outputs, total_loss, loss_dict = self._shared_step(batch=batch)
        self._logging(
            stage="valid", outputs=outputs, loss_dict=loss_dict, batch_idx=batch_idx
        )
        return total_loss

    def test_step(
        self,
        batch: LowLightSample,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        low_img, high_img = batch
        outputs = self.forward(low=low_img)

        metrics = self.metric.full(preds=outputs["enh_rgb"], targets=high_img)

        self.log_dict(
            dictionary={
                "test/PSNR": metrics["PSNR"],
                "test/SSIM": metrics["SSIM"],
                "test/LPIPS": metrics["LPIPS"],
                "test/NIQE": metrics["NIQE"],
                "test/BRISQUE": metrics["BRISQUE"],
            },
            prog_bar=True,
        )

    def predict_step(
        self,
        batch: LowLightSample,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> list[Tensor]:
        low_img, _ = batch
        results = self.forward(low=low_img)
        return [results["enh_rgb"]]

    def configure_optimizers(self) -> list[Optimizer]:
        lr = float(self.hparams.get("lr", 1e-3))

        optimizer = Adam(
            params=self.parameters(),
            lr=lr,
            betas=self.hparams.get("betas", (0.9, 0.999)),
            eps=self.hparams.get("eps", 1e-8),
            weight_decay=self.hparams.get("weight_decay", 0.0),
        )
        return [optimizer]
