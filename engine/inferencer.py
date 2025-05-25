import os

from lightning import Trainer, LightningModule
from pathlib import Path
from typing import Optional

from data.dataloader import CustomDataModule
from data.utils import DataTransform
from utils.utils import save_images


class LightningInferencer:
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
            self.ckpt = "best"

        # --- DataModule 정의
        self.datamodule = self._build_datamodule()

        self.save_dir = Path(
            self.trainer.log_dir,
            self.hparams["inference"]
        )
        print(f"Save Directory: {self.save_dir}")

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
        print("[INFO] Start Inferencing...")
        results = self.trainer.predict(
            model=self.model,
            datamodule=self.datamodule,
        )
        save_images(results=results, save_dir=self.save_dir)
        print("[INFO] Inference Completed.")
