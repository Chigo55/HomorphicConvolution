import torch
import torch.nn as nn


def add_nan_hooks(module: torch.nn.Module):
    for name, p in module.named_parameters():
        if not p.requires_grad:          # ★ 학습 대상이 아니면 건너뛰기
            continue

        # -------- 그래디언트 검사 --------
        def _grad_check(grad, n=name):
            if torch.isnan(input=grad).any() or torch.isinf(input=grad).any():
                print(f"[GRAD NaN] {n}")

        # -------- 가중치 검사 (옵티머 step 직후) --------
        def _weight_check(grad, n=name):
            if torch.isnan(input=p.data).any() or torch.isinf(input=p.data).any():
                print(f"[WEIGHT NaN] {n}")

        p.register_hook(hook=_grad_check)
        p.register_hook(hook=_weight_check)


def register_full_nan_inf_hooks(model):
    def forward_hook(module, input, output):
        name = module.__class__.__name__
        if isinstance(output, torch.Tensor):
            if torch.isnan(input=output).any():
                print(f"[NaN DETECTED - FORWARD] in {name}")
            if torch.isinf(input=output).any():
                print(f"[Inf DETECTED - FORWARD] in {name}")

    def backward_hook(module, grad_input, grad_output):
        name = module.__class__.__name__
        for idx, g in enumerate(grad_output):
            if g is not None:
                if torch.isnan(input=g).any():
                    print(
                        f"[NaN DETECTED - BACKWARD] in {name} grad_output[{idx}]")
                if torch.isinf(input=g).any():
                    print(
                        f"[Inf DETECTED - BACKWARD] in {name} grad_output[{idx}]")

    for module in model.modules():  # or named_modules() + 조건
        module.register_forward_hook(forward_hook)
        module.register_full_backward_hook(backward_hook)
