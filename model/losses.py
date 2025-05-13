import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16


class L_color(nn.Module):
    def forward(self, x):
        mean_rgb = torch.mean(input=x, dim=[2, 3], keepdim=True)
        mean_r, mean_g, mean_b = torch.split(
            tensor=mean_rgb,
            split_size_or_sections=1,
            dim=1
        )

        diff_rg = torch.pow(input=(mean_r-mean_g), exponent=2)
        diff_rb = torch.pow(input=(mean_r-mean_b), exponent=2)
        diff_gb = torch.pow(input=(mean_g-mean_b), exponent=2)

        loss = torch.pow(
            input=(
                torch.pow(input=diff_rg, exponent=2) +
                torch.pow(input=diff_rb, exponent=2) +
                torch.pow(input=diff_gb, exponent=2)
            ), exponent=0.5
        )
        return loss


class L_spa(nn.Module):
    def __init__(self,):
        super().__init__()
        kernel_l = torch.FloatTensor([
            [0, 0, 0],
            [-1, 1, 0],
            [0, 0, 0]
        ]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_r = torch.FloatTensor([
            [0, 0, 0],
            [0, 1, -1],
            [0, 0, 0]
        ]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_u = torch.FloatTensor([
            [0, -1, 0],
            [0, 1, 0],
            [0, 0, 0]
        ]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_d = torch.FloatTensor([
            [0, 0, 0],
            [0, 1, 0],
            [0, -1, 0]
        ]).cuda().unsqueeze(0).unsqueeze(0)

        # Register kernels as non-trainable parameters
        self.weight_l = nn.Parameter(data=kernel_l, requires_grad=False)
        self.weight_r = nn.Parameter(data=kernel_r, requires_grad=False)
        self.weight_u = nn.Parameter(data=kernel_u, requires_grad=False)
        self.weight_d = nn.Parameter(data=kernel_d, requires_grad=False)
        self.avg_pool = nn.AvgPool2d(kernel_size=4)

    def forward(self, original_image, enhanced_image):
        original_mean = torch.mean(input=original_image, dim=1, keepdim=True)
        enhanced_mean = torch.mean(input=enhanced_image, dim=1, keepdim=True)

        original_pooled = self.avg_pool(original_mean)
        enhanced_pooled = self.avg_pool(enhanced_mean)

        diff_l = (
            F.conv2d(input=original_pooled, weight=self.weight_l, padding=1) -
            F.conv2d(input=enhanced_pooled, weight=self.weight_l, padding=1)
        )
        diff_r = (
            F.conv2d(input=original_pooled, weight=self.weight_r, padding=1) -
            F.conv2d(input=enhanced_pooled, weight=self.weight_r, padding=1)
        )
        diff_u = (
            F.conv2d(input=original_pooled, weight=self.weight_u, padding=1) -
            F.conv2d(input=enhanced_pooled, weight=self.weight_u, padding=1)
        )
        diff_d = (
            F.conv2d(input=original_pooled, weight=self.weight_d, padding=1) -
            F.conv2d(input=enhanced_pooled, weight=self.weight_d, padding=1)
        )

        loss = (
            torch.pow(input=diff_l, exponent=2) +
            torch.pow(input=diff_r, exponent=2) +
            torch.pow(input=diff_u, exponent=2) +
            torch.pow(input=diff_d, exponent=2)
        )
        return loss


class L_exp(nn.Module):
    def __init__(self, patch_size=16, mean_val=0.9):
        super().__init__()
        self.pool = nn.AvgPool2d(kernel_size=patch_size)
        self.mean_val = mean_val

    def forward(self, x):
        mean = self.pool(
            torch.mean(
                input=x,
                dim=1,
                keepdim=True
            )
        )

        loss = torch.mean(
            input=torch.pow(
                # input=mean-self.mean_val,
                input=mean-torch.FloatTensor([self.mean_val]).cuda(),
                exponent=2
            )
        )
        return loss


class L_TV(nn.Module):
    def __init__(self, weight=1.0):
        super().__init__()
        self.weight = weight

    def forward(self, x):
        B, C, H, W = x.size()
        c_h = (H - 1) * W
        c_w = H * (W - 1)
        h_tv = torch.pow(
            input=(x[:, :, 1:, :]-x[:, :, :H-1, :]),
            exponent=2
        ).sum()
        w_tv = torch.pow(
            input=(x[:, :, :, 1:]-x[:, :, :, :W-1]),
            exponent=2
        ).sum()

        return self.weight*2*(h_tv/c_h + w_tv/c_w)/B


class Sa_Loss(nn.Module):
    def forward(self, x):
        r, g, b = x.chunk(3, dim=1)
        mean_rgb = torch.mean(x, dim=[2, 3], keepdim=True)
        mr, mg, mb = mean_rgb.chunk(3, dim=1)
        Dr, Dg, Db = r - mr, g - mg, b - mb
        return torch.sqrt(Dr ** 2 + Dg ** 2 + Db ** 2).mean()


class PerceptualLoss(nn.Module):
    def __init__(self, layer="relu4_3"):
        super().__init__()
        vgg = vgg16(pretrained=True).features
        self.layers = {
            "relu1_2": vgg[:4],
            "relu2_2": vgg[:9],
            "relu3_3": vgg[:16],
            "relu4_3": vgg[:23],
        }
        self.feature_extractor = self.layers[layer].eval()
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

    def forward(self, x, y):
        f1 = self.feature_extractor(x)
        f2 = self.feature_extractor(y)
        return F.l1_loss(f1, f2)
