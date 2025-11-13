# HomomorphicDiT — Low-Light Image Enhancement

A research codebase for low-light image enhancement using homomorphic separation and transformer-based modules. The project contains training, validation, benchmarking and inference components, plus utilities for metrics (including BRISQUE/NIQE) and data handling.

## Repository layout (important files & folders)

- model/
  - model.py, model_no_ref.py — model entry points
  - loss.py, loss_no_ref.py — loss definitions
  - blocks/ — homomorphic / illumination / lowlight blocks (homomorphic.py, lowlightenhancer.py, and no-ref variants)
- engine/
  - engine.py — engine utilities
  - runner.py — top-level runner (training / validation / inference orchestration)
- data/
  - dataloader.py — dataset & loader logic
  - utils.py — dataset helpers
  - 1_train, 2_valid, 3_bench, 4_infer, database — expected dataset layout (high/low subfolders)
- utils/
  - utils.py — general utilities (saving, helpers)
  - metrics.py — PSNR/SSIM/LPIPS/NIQE/BRISQUE helpers
  - files/ — BRISQUE model & range YAMLs used by metrics
- __init__.py files included for package imports

## Quick start

1. Prepare Python environment (example):
   - python >= 3.8, PyTorch and common packages (torch, torchvision, numpy, opencv-python, PyYAML)
   - Create virtualenv and install dependencies used in your environment.

2. Arrange data:
   - Put paired or unpaired datasets under data/ following the existing folders:
     - data/1_train/.../high and /low
     - data/2_valid/.../high and /low
     - data/3_bench and data/4_infer as needed

3. Run the runner
   - The project uses engine/runner.py as the orchestration entry. Inspect runner.py for available CLI/flags and usage.
   - Example:
     - python -m engine.runner  (or) python engine/runner.py --help

4. Training / Validation / Inference
   - Use engine.runner to start training and validation; engine.engine contains lower-level utilities used by the runner.
   - Model variants:
     - model.py / loss.py — reference-based training
     - model_no_ref.py / loss_no_ref.py — no-reference training flows

## Metrics & evaluation

- metrics.py implements common quality metrics (PSNR, SSIM, LPIPS) and wrappers for no-reference metrics (NIQE / BRISQUE). BRISQUE uses YAML files in utils/files/.
- Use the provided utilities to compute metrics during validation and benchmarking.

## Development notes

- The codebase separates "ref" and "no-ref" flows; check filenames with `_no_ref` to find the alternative implementations.
- Blocks/ contains the homomorphic and illumination-specific components; these are the building blocks for the model variants.

## Contributing

- Open issues or PRs for bug fixes, small improvements, or additions.
- Keep experiments and new checkpoints/logs outside the repository (e.g., in runs/).

## License & citation

- Add appropriate license and citation to any paper or repository this code is derived from. If used for publications, cite the correct work.

For details about configuration options, hyperparameters and exact CLI usage, inspect engine/runner.py, model/*.py and data/dataloader.py.