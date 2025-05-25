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
    def __init__(self):
        super().__init__()
        self.last_ckpt_path = None

    def on_after_backward(self, trainer, pl_module):
        """
        backward가 끝난 직후에 호출됩니다.
        grad에 NaN이 있는지 체크하고, 있을 경우 업데이트 이전의 파라미터 상태를 저장합니다.
        """
        for name, param in pl_module.named_parameters():
            if param.grad is not None and torch.isnan(input=param.grad).any():
                log_dir = trainer.log_dir
                ckpt_path = os.path.join(log_dir, "pre_nan_ckpt.ckpt")
                # 이 시점에는 파라미터가 아직 업데이트되지 않았으므로, NaN 이전 모델이 저장됩니다.
                trainer.save_checkpoint(filepath=ckpt_path)
                print(
                    f"[DetectNanCallback] NaN 발생한 grad 감지 → 이전 파라미터로 체크포인트 저장: {ckpt_path}")
                self.last_ckpt_path = ckpt_path
                # 그 즉시 학습을 멈추고 에러를 띄웁니다.
                raise RuntimeError(
                    f"[DetectNanCallback] NaN in gradient of {name}")


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
