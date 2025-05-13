import os

from lightning import Trainer
from pathlib import Path
from tqdm.auto import tqdm

from data.dataloader import CustomDataModule
from data.utils import DataTransform
from utils.metrics import ImageQualityMetrics


class LightningBenchmarker:
    def __init__(self, model, trainer: Trainer, ckpt: Path, hparams: dict):
        self.hparams = hparams

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

        # --- 평가 메트릭 정의
        self.metric = ImageQualityMetrics(device="cuda")
        self.metric.eval()

    def _build_datamodule(self):
        datamodule = CustomDataModule(
            train_dir=self.hparams["train_data_path"],
            valid_dir=self.hparams["valid_data_path"],
            infer_dir=self.hparams["infer_data_path"],
            bench_dir=self.hparams["bench_data_path"],
            transform=DataTransform(self.hparams["image_size"]),
            batch_size=self.hparams["batch_size"],
            num_workers=int(os.cpu_count() * 0.7),
        )
        datamodule.setup()  # 벤치마크 데이터셋 사용 위해 미리 세팅
        return datamodule

    def run(self):
        print("[INFO] Start benchmarking image quality metrics...")

        outputs = self.trainer.test(
            model=self.model,
            datamodule=self.datamodule,
            ckpt_path=self.ckpt
        )
        print("outputs", outputs)
        print("\n[FINAL BENCHMARK RESULT]")
        for k, v in tqdm(outputs.items()):
            print(f"{k}: {v:.4f}")
