import os
import torch
from typing import Optional

from lightning import Trainer, LightningModule
from lightning.pytorch.callbacks import (
    Callback, ModelCheckpoint, EarlyStopping, LearningRateMonitor
)
from lightning.pytorch.loggers import TensorBoardLogger

from engine.trainer import LightningTrainer
from engine.validater import LightningValidater
from engine.benchmarker import LightningBenchmarker
from engine.inferencer import LightningInferencer


class DetectNanCallback(Callback):
    def __init__(self, save_dir="./nan_checkpoints"):
        super().__init__()
        self.save_dir = save_dir
        os.makedirs(name=self.save_dir, exist_ok=True)
        self.last_ckpt_path = None

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        for name, param in pl_module.named_parameters():
            if torch.isnan(input=param.data).any() or (param.grad is not None and torch.isnan(input=param.grad).any()):
                log_dir = trainer.logger.log_dir
                ckpt_path = os.path.join(log_dir, "nan_ckpt.ckpt")
                trainer.save_checkpoint(filepath=ckpt_path)
                print(f"[DetectNanCallback] NaN 발생 → 모델 저장 완료: {ckpt_path}")
                self.last_ckpt_path = ckpt_path
                raise RuntimeError(f"[DetectNanCallback] NaN in {name}")


class LightningEngine:
    def __init__(self, model: LightningModule, hparams: dict, ckpt: Optional[str] = None):
        self.model = model
        self.hparams = hparams
        self.ckpt = ckpt
        self.nan_callback = DetectNanCallback()

        # --- 로깅 설정
        self.logger = self._build_logger()

        # --- 콜백 정의
        self.callbacks = self._build_callbacks()

        # --- Trainer 정의
        self.trainer = Trainer(
            max_epochs=hparams["epochs"],
            accelerator="gpu",
            devices=1,
            precision="16-mixed",
            logger=self.logger,
            callbacks=self.callbacks,
            log_every_n_steps=5,
            gradient_clip_val=0.5,
        )

    def _build_logger(self):
        return TensorBoardLogger(
            save_dir=self.hparams["log_dir"],
            name=self.hparams["experiment_name"]
        )

    def _build_callbacks(self):
        return [
            ModelCheckpoint(
                monitor="valid/5_tot",
                save_top_k=1,
                mode="min",
                filename="best-{epoch:02d}",
            ),
            ModelCheckpoint(
                every_n_epochs=1,
                save_top_k=-1,
                filename="epoch-{epoch:02d}",
            ),
            EarlyStopping(
                monitor="valid/5_tot",
                patience=self.hparams["patience"],
                mode="min",
                verbose=True,
            ),
            LearningRateMonitor(logging_interval="step"),
            DetectNanCallback(),
        ]

    def train(self):
        LightningTrainer(
            model=self.model,
            trainer=self.trainer,
            hparams=self.hparams,
            ckpt=self.ckpt
        ).run()

    def valid(self):
        LightningValidater(
            model=self.model,
            trainer=self.trainer,
            hparams=self.hparams,
            ckpt=self.ckpt
        ).run()

    def bench(self):
        LightningBenchmarker(
            model=self.model,
            trainer=self.trainer,
            hparams=self.hparams,
            ckpt=self.ckpt
        ).run()

    def infer(self):
        LightningInferencer(
            model=self.model,
            trainer=self.trainer,
            hparams=self.hparams,
            ckpt=self.ckpt
        ).run()

    def update_ckpt_from_nan(self):
        """NaN 콜백에서 저장된 ckpt 경로가 있다면 그것으로 업데이트"""
        if self.nan_callback.last_ckpt_path is not None:
            self.ckpt = self.nan_callback.last_ckpt_path
            print(f"[Engine] NaN 발생 후 체크포인트 로드 경로 갱신: {self.ckpt}")
