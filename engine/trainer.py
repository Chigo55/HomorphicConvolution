import os

from lightning import Trainer, LightningModule, seed_everything
from typing import Optional

from data.dataloader import CustomDataModule
from data.utils import DataTransform


class LightningTrainer:
    def __init__(self, model: LightningModule, trainer: Trainer, hparams: dict, ckpt: Optional[str] = None):
        self.trainer = trainer
        self.hparams = hparams

        if ckpt:
            self.model = model.load_from_checkpoint(
                checkpoint_path=ckpt,
                map_location="cpu",
            )
            self.ckpt = ckpt
        else:
            self.model = model(hparams=hparams)

        # --- DataModule 정의
        self.datamodule = self._build_datamodule()

        seed_everything(seed=self.hparams["seed"], workers=True)

    def _build_datamodule(self):
        return CustomDataModule(
            train_dir=self.hparams["train_data_path"],
            valid_dir=self.hparams["valid_data_path"],
            infer_dir=self.hparams["infer_data_path"],
            bench_dir=self.hparams["bench_data_path"],
            transform=DataTransform(image_size=self.hparams["image_size"]),
            batch_size=self.hparams["batch_size"],
            num_workers=int(os.cpu_count() * 0.9),
        )

    def run(self):
        print("[INFO] Start Training...")
        self.trainer.fit(
            model=self.model,
            datamodule=self.datamodule
        )
        print("[INFO] Training Completed.")
