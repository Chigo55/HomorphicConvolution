import os
from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import *
from lightning.pytorch.loggers import TensorBoardLogger

from data.dataloader import CustomDataModule
from data.utils import DataTransform


class LightningTrainer:
    def __init__(self, model, hparams: dict, transform=None):
        self.hparams = hparams
        self.transform = transform if transform else DataTransform()
        seed_everything(seed=hparams["seed"], workers=True)

        # --- 모델 정의
        self.model = model

        # --- DataModule 정의
        self.datamodule = self._build_datamodule()

        # --- 로깅 설정
        self.logger = self._build_logger()

        # --- 콜백 정의
        self.callbacks = self._build_callbacks()

        # --- Lightning Trainer 정의
        self.trainer = Trainer(
            max_epochs=hparams["epochs"],
            accelerator="gpu",
            devices=1,
            precision="16",
            logger=self.logger,
            callbacks=self.callbacks,
            log_every_n_steps=5,
        )

    def _build_logger(self):
        return TensorBoardLogger(
            save_dir=self.hparams["log_dir"],
            name=self.hparams["experiment_name"]
        )

    def _build_datamodule(self):
        return CustomDataModule(
            train_dir=self.hparams["train_data_path"],
            valid_dir=self.hparams["valid_data_path"],
            infer_dir=self.hparams["infer_data_path"],
            bench_dir=self.hparams["bench_data_path"],
            transform=DataTransform(image_size=self.hparams["image_size"]),
            batch_size=self.hparams["batch_size"],
            num_workers=int(os.cpu_count() * 0.7),
        )

    def _build_callbacks(self):
        return [
            ModelCheckpoint(
                monitor="valid/total",
                save_top_k=1,
                mode="min",
                filename="best-{epoch:02d}",
            ),
            ModelCheckpoint(
                every_n_epochs=10,
                save_top_k=-1,  # 모두 저장
                filename="epoch-{epoch:02d}",
            ),
            EarlyStopping(
                monitor="valid/total",
                patience=10,
                mode="min",
                verbose=True,
            ),
            LearningRateMonitor(logging_interval="step"),
            RichProgressBar(),
        ]

    def run(self):
        print("[INFO] Start training...")
        self.trainer.fit(
            model=self.model,
            datamodule=self.datamodule
        )
        print("[INFO] Training completed!")
