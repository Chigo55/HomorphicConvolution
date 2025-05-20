import os

from lightning import Trainer, LightningModule
from typing import Optional

from data.dataloader import CustomDataModule
from data.utils import DataTransform
from utils.metrics import ImageQualityMetrics


class LightningBenchmarker:
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

        # --- 평가 메트릭 정의
        self.metric = ImageQualityMetrics(device="cuda")
        self.metric.eval()

    def _build_datamodule(self):
        datamodule = CustomDataModule(
            train_dir=self.hparams["train_data_path"],
            valid_dir=self.hparams["valid_data_path"],
            infer_dir=self.hparams["infer_data_path"],
            bench_dir=self.hparams["bench_data_path"],
            transform=DataTransform(image_size=self.hparams["image_size"]),
            batch_size=self.hparams["batch_size"],
            num_workers=int(os.cpu_count() * 0.9),
        )
        datamodule.setup()  # 벤치마크 데이터셋 사용 위해 미리 세팅
        return datamodule

    def run(self):
        print("[INFO] Start Benchmarking")
        results = self.trainer.test(
            model=self.model,
            datamodule=self.datamodule,
        )
        print("[INFO] Benchmark Completed.")
