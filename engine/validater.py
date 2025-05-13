import os

from lightning import Trainer
from pathlib import Path
from tqdm.auto import tqdm

from data.dataloader import CustomDataModule
from data.utils import DataTransform


class LightningValidater:
    def __init__(self, model, trainer: Trainer, ckpt: Path, hparams: dict):
        self.hparams = hparams

        # --- 모델 정의
        if ckpt:
            self.model = model.load_from_checkpoint(
                checkpoint_path=ckpt,
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
        print("[INFO] Start validating...")
        results = self.trainer.validate(
            model=self.model,
            datamodule=self.datamodule,
            ckpt_path=self.ckpt
        )
        print("[VALIDATION RESULT]")
        for res in tqdm(results):
            print(res)
