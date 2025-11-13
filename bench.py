import random
from typing import Any

from engine.engine import LightningEngine
from model.model import LowLightEnhancerLightning


def get_hparams() -> dict[str, Any]:
    hparams: dict[str, Any] = {
        # Engine
        "seed": 42,
        "max_epochs": 100,
        "accelerator": "gpu",
        "devices": 1,
        "precision": "16-mixed",
        "log_every_n_steps": 5,
        "log_dir": "runs/",
        "experiment_name": "bench/",
        "patience": 100,
        # Runner
        "inference": "inference/",
        "train_data_path": "data/1_train",
        "valid_data_path": "data/2_valid",
        "bench_data_path": "data/3_bench",
        "infer_data_path": "data/4_infer",
        "image_size": 256,
        "batch_size": 8,
        "num_workers": 10,
        # Model
        "hidden_channels": 32,
        "num_resolution": 2,
        "kernel_size": 15,
        "sigma": 5,
        # Loss
        "lambda_mae": 1.0,
        "lambda_mse": 1.0,
    }
    return hparams


def main() -> None:
    hparams: dict[str, Any] = get_hparams()
    seed: int = random.randint(0, 1000)
    hparams["seed"] = seed

    engine: LightningEngine = LightningEngine(
        model_class=LowLightEnhancerLightning,
        hparams=hparams,
        checkpoint_path=r"runs\ref\version_0\checkpoints\best.ckpt",
    )
    engine.bench()


if __name__ == "__main__":
    main()
