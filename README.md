# HomomorphicDiT: Low-Light Image Enhancement

This repository implements a deep learning pipeline for low-light image enhancement using a homomorphic separation and transformer-based architecture. The project is organized for training, validation, benchmarking, and inference of the HomomorphicDiT model.

## Project Structure

```
.
├── main.py
├── requirements.txt
├── data_split.ipynb
├── difusion.ipynb
├── image.ipynb
├── times.ttf
├── data/
│   ├── dataloader.py
│   ├── utils.py
│   ├── 1_train/
│   ├── 2_valid/
│   ├── 3_bench/
│   ├── 4_infer/
│   └── database/
├── engine/
│   ├── trainer.py
│   ├── validater.py
│   ├── inferencer.py
│   └── benchmarker.py
├── model/
│   ├── model.py
│   ├── block.py
│   └── losses.py
├── utils/
│   ├── utils.py
│   ├── metrics.py
│   └── hook.py
├── runs/
│   └── HomomorphicUnet/
└── zoo/
```

## Main Components

- **Model**: The core model is implemented in [`model/model.py`](model/model.py), with transformer blocks in [`model/block.py`](model/block.py) and loss functions in [`model/losses.py`](model/losses.py).
- **Data**: Data loading and augmentation utilities are in [`data/dataloader.py`](data/dataloader.py) and [`data/utils.py`](data/utils.py).
- **Engine**: Training, validation, inference, and benchmarking logic are in [`engine/trainer.py`](engine/trainer.py), [`engine/validater.py`](engine/validater.py), [`engine/inferencer.py`](engine/inferencer.py), and [`engine/benchmarker.py`](engine/benchmarker.py).
- **Utilities**: General utilities and metrics are in [`utils/utils.py`](utils/utils.py) and [`utils/metrics.py`](utils/metrics.py).

## Data Preparation

- Place your training, validation, benchmarking, and inference images in the respective folders under `data/`.
- Use [`data_split.ipynb`](data_split.ipynb) to split and organize your dataset if needed.

## Training

To train the model, run:

```sh
python main.py
```

Or use the training logic in [`difusion.ipynb`](difusion.ipynb) for interactive experimentation.

Hyperparameters are defined in the `get_hparams()` function (see [`difusion.ipynb`](difusion.ipynb)), including model size, loss weights, learning rate, and data paths.

## Validation & Benchmarking

- Validation and benchmarking can be performed using the scripts in the `engine/` directory or interactively in [`difusion.ipynb`](difusion.ipynb).
- Image quality metrics such as PSNR, SSIM, LPIPS, NIQE, and BRISQUE are computed using [`utils/metrics.py`](utils/metrics.py).

## Inference

To run inference on new images, use the inference logic in [`engine/inferencer.py`](engine/inferencer.py) or the corresponding cells in [`difusion.ipynb`](difusion.ipynb).

## Visualization

[`image.ipynb`](image.ipynb) provides visualization utilities for inspecting input, intermediate, and enhanced images.

## Requirements

Install dependencies with:

```sh
pip install -r requirements.txt
```

Key dependencies include:
- PyTorch
- PyTorch Lightning
- torchvision
- opencv-contrib-python
- numpy
- matplotlib

## Utilities

- Model parameter counting and summaries: [`utils/utils.py`](utils/utils.py)
- Image saving and metric printing: [`utils/utils.py`](utils/utils.py)

## Checkpoints & Logs

- Training logs and checkpoints are saved under `runs/` and `runs2/` directories.
- Best checkpoints are saved based on validation metrics.

## Citation

If you use this codebase, please cite the original paper (add citation here if available).

---

**Note:** This repository is designed for research and educational purposes. For production use, further testing and optimization are