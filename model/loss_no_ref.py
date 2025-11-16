import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class ColorConstancy(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        input: Tensor,
    ) -> Tensor:
        if input.shape[1] == 1:
            return torch.tensor(data=0.0, device=input.device, dtype=input.dtype)

        mean_rgb = torch.mean(input=input, dim=[2, 3], keepdim=True)
        mr, mg, mb = torch.split(tensor=mean_rgb, split_size_or_sections=1, dim=1)
        Drg = (mr - mg) ** 2
        Drb = (mr - mb) ** 2
        Dgb = (mb - mg) ** 2

        loss = ((Drg**2) + (Drb**2) + (Dgb**2)) ** 0.5
        loss = torch.mean(input=loss)
        return loss


class SpatialConsistency(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        kernel_l = (
            torch.tensor(
                data=[[0, 0, 0], [-1, 1, 0], [0, 0, 0]],
                dtype=torch.float32,
            )
            .unsqueeze(dim=0)
            .unsqueeze(dim=0)
        )
        kernel_r = (
            torch.tensor(
                data=[[0, 0, 0], [0, 1, -1], [0, 0, 0]],
                dtype=torch.float32,
            )
            .unsqueeze(dim=0)
            .unsqueeze(dim=0)
        )
        kernel_u = (
            torch.tensor(
                data=[[0, -1, 0], [0, 1, 0], [0, 0, 0]],
                dtype=torch.float32,
            )
            .unsqueeze(dim=0)
            .unsqueeze(dim=0)
        )
        kernel_d = (
            torch.tensor(
                data=[[0, 0, 0], [0, 1, 0], [0, -1, 0]],
                dtype=torch.float32,
            )
            .unsqueeze(dim=0)
            .unsqueeze(dim=0)
        )

        self.register_buffer(name="weight_l", tensor=kernel_l)
        self.register_buffer(name="weight_r", tensor=kernel_r)
        self.register_buffer(name="weight_u", tensor=kernel_u)
        self.register_buffer(name="weight_d", tensor=kernel_d)

        self.pool = nn.AvgPool2d(kernel_size=4)

    def forward(
        self,
        input: Tensor,
        target: Tensor,
    ) -> Tensor:
        if input.shape[1] > 1:
            org_mean = torch.mean(input=input, dim=1, keepdim=True)
            enh_mean = torch.mean(input=target, dim=1, keepdim=True)

        else:
            org_mean = input
            enh_mean = target

        org_pool = self.pool(org_mean)
        enh_pool = self.pool(enh_mean)

        D_org_l = F.conv2d(input=org_pool, weight=self.weight_l, padding=1)
        D_org_r = F.conv2d(input=org_pool, weight=self.weight_r, padding=1)
        D_org_u = F.conv2d(input=org_pool, weight=self.weight_u, padding=1)
        D_org_d = F.conv2d(input=org_pool, weight=self.weight_d, padding=1)

        D_enh_l = F.conv2d(input=enh_pool, weight=self.weight_l, padding=1)
        D_enh_r = F.conv2d(input=enh_pool, weight=self.weight_r, padding=1)
        D_enh_u = F.conv2d(input=enh_pool, weight=self.weight_u, padding=1)
        D_enh_d = F.conv2d(input=enh_pool, weight=self.weight_d, padding=1)

        loss = (
            (D_org_l - D_enh_l) ** 2
            + (D_org_r - D_enh_r) ** 2
            + (D_org_u - D_enh_u) ** 2
            + (D_org_d - D_enh_d) ** 2
        )
        loss = torch.mean(input=loss)

        return loss


class Exposurecontrol(nn.Module):
    def __init__(
        self,
        patch_size: int = 16,
        mean_val: float = 0.8,
    ) -> None:
        super().__init__()
        self.pool = nn.AvgPool2d(kernel_size=patch_size)
        self.register_buffer(
            name="mean_val_tensor",
            tensor=torch.tensor(
                data=[mean_val],
                dtype=torch.float32,
            ),
        )

    def forward(
        self,
        input: Tensor,
    ) -> Tensor:
        if input.shape[1] > 1:
            x = torch.mean(input=input, dim=1, keepdim=True)

        else:
            x = input

        x_pool = self.pool(x)
        loss = torch.mean(input=(x_pool - self.mean_val_tensor) ** 2)
        return loss


class IlluminanceSmoothness(nn.Module):
    def __init__(
        self,
    ) -> None:
        super().__init__()

    def forward(
        self,
        input: Tensor,
    ) -> Tensor:
        batch = input.size(dim=0)
        h = input.size(dim=2)
        w = input.size(dim=3)

        if h == 1 or w == 1:
            return torch.tensor(data=0.0, device=input.device, dtype=input.dtype)

        count_h = (h - 1) * w
        count_w = h * (w - 1)

        h_tv = ((input[:, :, 1:, :] - input[:, :, : h - 1, :]) ** 2).sum()
        w_tv = ((input[:, :, :, 1:] - input[:, :, :, : w - 1]) ** 2).sum()

        loss = 2 * (h_tv / count_h + w_tv / count_w) / batch
        loss = torch.mean(input=loss)

        return loss


class ParameterSmoothness(nn.Module):
    def __init__(
        self,
    ) -> None:
        super().__init__()

    def forward(
        self,
        input: Tensor,
    ) -> Tensor:
        batch = input.size(dim=0)
        h = input.size(dim=2)
        w = input.size(dim=3)

        if h == 1 or w == 1:
            return torch.tensor(data=0.0, device=input.device, dtype=input.dtype)

        count_h = (h - 1) * w
        count_w = h * (w - 1)

        h_tv = ((input[:, :, 1:, :] - input[:, :, : h - 1, :]) ** 2).sum()
        w_tv = ((input[:, :, :, 1:] - input[:, :, :, : w - 1]) ** 2).sum()

        loss = 2 * (h_tv / count_h + w_tv / count_w) / batch
        loss = torch.mean(input=loss)

        return loss


class TotalLoss(nn.Module):
    def __init__(
        self,
        lambda_spa: float = 1.0,
        lambda_exp: float = 1.0,
        # lambda_col: float = 1.0,
        lambda_alpha: float = 1.0,
        lambda_illum: float = 1.0,
        exp_patch_size: int = 16,
        exp_mean_val: float = 0.8,
    ) -> None:
        super().__init__()

        self.lambda_spa = lambda_spa
        self.lambda_exp = lambda_exp
        # self.lambda_col = lambda_col
        self.lambda_alpha = lambda_alpha
        self.lambda_illum = lambda_illum

        self.loss_spa = SpatialConsistency()
        self.loss_exp = Exposurecontrol(
            patch_size=exp_patch_size,
            mean_val=exp_mean_val,
        )
        # self.loss_col = ColorConstancy()
        self.loss_alpha = ParameterSmoothness()
        self.loss_illum = IlluminanceSmoothness()

    def forward(
        self,
        low_luminance: Tensor,
        enh_luminance: Tensor,
        # enh_rgb: Tensor,
        alpha_component: Tensor,
        enh_illuminance: Tensor,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        l_spa = self.loss_spa(low_luminance, enh_luminance)
        l_exp = self.loss_exp(enh_illuminance)
        # l_col = self.loss_col(enh_rgb)
        l_alpha = self.loss_alpha(alpha_component)
        l_illum = self.loss_illum(enh_illuminance)

        total_loss = (
            self.lambda_spa * l_spa
            + self.lambda_exp * l_exp
            # + self.lambda_col * l_col
            + self.lambda_alpha * l_alpha
            + self.lambda_illum * l_illum
        )

        loss_dict = {
            "loss_total": total_loss.detach(),
            "loss_spa": self.lambda_spa * l_spa.detach(),
            "loss_exp": self.lambda_exp * l_exp.detach(),
            # "loss_col": self.lambda_col * l_col.detach(),
            "loss_alpha": self.lambda_alpha * l_alpha.detach(),
            "loss_illum": self.lambda_illum * l_illum.detach(),
        }

        return total_loss, loss_dict
