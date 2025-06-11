import torch.nn as nn

from pathlib import Path
from torchvision.utils import save_image
from torchinfo import summary


def make_dirs(path: str | Path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def print_metrics(metrics: dict, prefix: str = ""):
    for k, v in metrics.items():
        print(f"{prefix}{k}: {v:.4f}")


def save_images(results, save_dir, prefix="infer", ext="png"):
    for i, datasets in enumerate(results):
        save_path = make_dirs(path=f"{save_dir}/batch{i+1}")
        for ii, batch in enumerate(datasets):
            save_image(
                tensor=batch,
                fp=save_path / f"{prefix}_{ii:04d}.{ext}",
                nrow=8,
                padding=2,
                normalize=True,
                value_range=(0, 1)
            )


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def summarize_model(model, input_size):
    return summary(model=model, input_size=input_size, depth=3, col_names=["input_size", "output_size", "num_params"])


def weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.xavier_normal_(tensor=m.weight)  # 또는 xavier_uniform_
        if m.bias is not None:
            nn.init.constant_(tensor=m.bias, val=0.0)

    elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
        nn.init.constant_(tensor=m.weight, val=1.0)
        nn.init.constant_(tensor=m.bias, val=0.0)
