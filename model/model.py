from typing import Any, Literal

import lightning as L
from torch import Tensor
from torch.optim.adam import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.optim.optimizer import Optimizer

from data.utils import LowLightSample
from model.blocks.lowlightenhancer import LowLightEnhancer
from model.loss import MeanAbsoluteError, MeanSquaredError, StructuralSimilarity
from utils.metrics import ImageQualityMetrics


class LowLightEnhancerLightning(L.LightningModule):
    def __init__(self, hparams: dict[str, Any]) -> None:
        super().__init__()
        self.save_hyperparameters(hparams)

        self.model: LowLightEnhancer = LowLightEnhancer(
            hidden_channels=self.hparams.get("hidden_channels", 32),
            num_resolution=self.hparams.get("num_resolution", 4),
            offset=self.hparams.get("offset", 0.5),
            cutoff=self.hparams.get("cutoff", 0.1),
        )

        self.mae_loss: MeanAbsoluteError = MeanAbsoluteError().eval()
        self.mse_loss: MeanSquaredError = MeanSquaredError().eval()
        self.ssim_loss: StructuralSimilarity = StructuralSimilarity().eval()

        self.metric = ImageQualityMetrics().eval()

    def forward(self, low: Tensor) -> dict[str, Tensor]:
        return self.model(low)

    def _calculate_loss(
        self,
        outputs: dict[str, Tensor],
        target: Tensor,
    ) -> dict[str, Tensor]:
        pred: Tensor = outputs["enh_rgb"]

        loss_mae: Tensor = self.mae_loss(pred, target)
        loss_mse: Tensor = self.mse_loss(pred, target)
        loss_ssim: Tensor = self.ssim_loss(pred, target)
        loss_total: Tensor = loss_mae + loss_mse + loss_ssim

        loss_dict: dict[str, Tensor] = {
            "mae": loss_mae,
            "mse": loss_mse,
            "ssim": loss_ssim,
            "total": loss_total,
        }
        return loss_dict

    def _shared_step(
        self,
        batch: LowLightSample,
    ) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        low_img, high_img = batch
        outputs = self.forward(low=low_img)
        loss_dict = self._calculate_loss(outputs=outputs, target=high_img)
        return outputs, loss_dict

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
        outputs, loss_dict = self._shared_step(batch=batch)
        self._logging(
            stage="train", outputs=outputs, loss_dict=loss_dict, batch_idx=batch_idx
        )
        return loss_dict["total"]

    def validation_step(
        self,
        batch: LowLightSample,
        batch_idx: int,
    ) -> Tensor:
        outputs, loss_dict = self._shared_step(batch=batch)
        self._logging(
            stage="valid", outputs=outputs, loss_dict=loss_dict, batch_idx=batch_idx
        )
        return loss_dict["total"]

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

    def configure_optimizers(self) -> tuple[list[Optimizer], list[dict[str, Any]]]:
        lr = float(self.hparams.get("lr", 1e-6))

        optimizer = Adam(
            params=self.parameters(),
            lr=lr,
            betas=self.hparams.get("betas", (0.9, 0.999)),
            eps=self.hparams.get("eps", 1e-8),
            weight_decay=self.hparams.get("weight_decay", 0.0),
        )

        total_epochs = int(self.hparams.get("max_epochs", 100))
        warmup_epochs = max(1, int(0.05 * total_epochs))

        warmup = LinearLR(
            optimizer=optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_epochs,
        )
        cosine = CosineAnnealingLR(
            optimizer=optimizer,
            T_max=total_epochs - warmup_epochs,
            eta_min=lr * 0.01,
        )
        scheduler = SequentialLR(
            optimizer=optimizer,
            schedulers=[warmup, cosine],
            milestones=[warmup_epochs],
        )

        sched_cfg = {
            "scheduler": scheduler,
            "interval": "epoch",
            "frequency": 1,
        }
        return [optimizer], [sched_cfg]
