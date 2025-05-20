import random

from engine import LightningEngine
from model.model import *


def get_hparams():
    hparams = {
        # 모델 구조
        "image_size": 512,
        "offset": 0.5,
        "cutoff": 0.3,

        # 손실 함수 가중치 (losses.py 기준)
        "lambda_col": 100.0,
        "lambda_exp": 1.0,
        "lambda_spa": 10.0,
        "lambda_tva": 1.0,

        # 최적화 및 학습 설정
        "optim": "sgd",
        "lr": 1e-7,
        "decay": 1e-8,
        "epochs": 100,
        "patience": 30,
        "batch_size": 16,
        "seed": 7077,

        # 데이터 경로
        "train_data_path": "data/1_train",
        "valid_data_path": "data/2_valid",
        "bench_data_path": "data/3_bench",
        "infer_data_path": "data/4_infer",

        # 로깅 설정
        "log_dir": "./runs/HomomorphicUnet/optims/ReduceActz",
        "experiment_name": "test",
        "inference": "inference",
    }
    return hparams


# def main():
#     hparams = get_hparams()
#     engin = LightningEngine(
#         model=HomomorphicUnetLightning,
#         hparams=hparams,
#         ckpt="runs/HomomorphicUnet/add_tva_loss/version_14/checkpoints/step-step=2300.ckpt"
#     )

#     print("[RUNNING] Trainer...")
#     engin.train()

#     print("[RUNNING] Validater...")
#     engin.valid()

#     print("[RUNNING] Benchmarker...")
#     engin.bench()

#     print("[RUNNING] Inferencer...")
#     engin.infer()


def main():
    hparams = get_hparams()
    opts = ["sgd", "asgd", "rmsprop", "rprop",
            "adam", "adamw", "adamax", "adadelta"]

    for opt in opts:
        print(f"\n[STARTING] Optimizer: {opt}")
        hparams["optim"] = opt
        hparams["experiment_name"] = opt

        try:
            engin = LightningEngine(
                model=HomomorphicUnetLightning,
                hparams=hparams
            )
            engin.train()

        except RuntimeError as e:
            print(f"[NaN DETECTED] Optimizer '{opt}' 에서 학습 중단됨: {e}")
            engin.update_ckpt_from_nan()

        # 학습 성공 또는 NaN 발생 후 복구된 상태로 평가
        engin.valid()
        engin.bench()


if __name__ == "__main__":
    main()
