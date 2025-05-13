import os
import yaml

from lightning import Trainer
from pathlib import Path

from data.dataloader import CustomDataModule
from data.utils import DataTransform
from utils.utils import save_images


class LightningInferencer:
    def __init__(self, model, trainer: Trainer, ckpt: Path, hparams: Path):
        with open(hparams) as f:
            hparams = yaml.load(f, Loader=yaml.FullLoader)
        self.hparams = hparams

        if ckpt:
            self.model = model.load_from_checkpoint(
                checkpoint_path=str(ckpt),
                map_location="cuda",
            )
            self.ckpt = ckpt
        else:
            self.model = model
            self.ckpt = "best"

        # --- Lightning Trainer 정의
        self.trainer = trainer

        # --- DataModule 정의
        self.datamodule = self._build_datamodule()

        self.save_dir = ckpt.parents[1] / hparams["inference"]
        print(f"save_dir: {self.save_dir}")

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

    def run(self):
        print("[INFO] Start training...")
        results = self.trainer.predict(
            model=self.model,
            datamodule=self.datamodule,
            ckpt_path=self.ckpt
        )
        save_images(results=results, save_dir=self.save_dir)
        print("[INFO] Inference completed.")
