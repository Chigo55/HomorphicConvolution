import random
from typing import Any

from engine.engine import LightningEngine
from model.model_no_ref import LowLightEnhancerLightning


def get_hparams() -> dict[str, Any]:
    hparams: dict[str, Any] = {
        # Engine
        "seed": 42,
        "max_epochs": 100,
        "accelerator": "gpu",
        "devices": 1,
        "precision": "16-mixed",
        # "precision": 32,
        "log_every_n_steps": 5,
        "log_dir": "runs/",
        "experiment_name": " no_ref/",
        "patience": 100,
        # Runner
        "inference": "inference/",
        "train_data_path": "data/1_train",
        "valid_data_path": "data/2_valid",
        "bench_data_path": "data/3_bench",
        "infer_data_path": "data/4_infer",
        "image_size": 256,
        "batch_size": 32,
        "num_workers": 10,
        # Model
        "hidden_channels": 32,
        "num_resolution": 2,
        "kernel_size": 17,
        "sigma": 5,
        # Loss
        "lambda_spa": 5.0,
        "lambda_exp": 5.0,
        # "lambda_col": 10.0,
        "lambda_parm": 5.0,
        "lambda_illum": 1.0,
        "exp_patch_size": 8,
        "exp_mean_val": 0.7,
    }
    return hparams


def main() -> None:
    hparams: dict[str, Any] = get_hparams()
    seed: int = random.randint(0, 1000)
    hparams["seed"] = seed

    engine: LightningEngine = LightningEngine(
        model_class=LowLightEnhancerLightning,
        hparams=hparams,
    )
    engine.train()
    engine.bench()


if __name__ == "__main__":
    main()
