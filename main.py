import os
import random
from typing import Any

import numpy as np

from engine.engine import LightningEngine
from model.model import LowLightEnhancerLightning

os.environ["CUDA_VISIBLE_DEVICES"] = "1, 0"


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
        "experiment_name": "ref/",
        "patience": 100,
        # Runner
        "inference": "inference/",
        "train_data_path": "data/1_train",
        "valid_data_path": "data/2_valid",
        "bench_data_path": "data/3_bench",
        "infer_data_path": "data/4_infer",
        "image_size": 256,
        "batch_size": 24,
        "num_workers": 10,
        # Model
        "hidden_channels": 32,
        "num_resolution": 2,
        "offset": 0.5,
        "cutoff": 0.1,
    }
    return hparams


def main() -> None:
    hparams: dict[str, Any] = get_hparams()
    seed: int = random.randint(0, 1000)
    hparams["seed"] = seed

    for i in np.arange(0.05, 0.5, 0.05):
        hparams["cutoff"] = i
        engine: LightningEngine = LightningEngine(
            model_class=LowLightEnhancerLightning,
            hparams=hparams,
        )
        engine.train()
        engine.bench()


if __name__ == "__main__":
    main()
