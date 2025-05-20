import torch
import torch.nn as nn
import torch.nn.functional as F


class L_col(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        mean_rgb = torch.mean(
            input=x,
            dim=[2, 3],
            keepdim=True
        )
        mr, mg, mb = torch.split(
            tensor=mean_rgb,
            split_size_or_sections=1,
            dim=1
        )

        Drg = torch.pow(input=mr - mg, exponent=2)
        Drb = torch.pow(input=mr - mb, exponent=2)
        Dgb = torch.pow(input=mb - mg, exponent=2)

        c = torch.pow(
            input=(
                torch.pow(input=Drg, exponent=2) +
                torch.pow(input=Drb, exponent=2) +
                torch.pow(input=Dgb, exponent=2)
            ),
            exponent=0.5
        )
        return c


class L_spa(nn.Module):
    def __init__(self):
        super().__init__()
        kernel_l = torch.FloatTensor([
            [0, 0, 0],
            [-1, 1, 0],
            [0, 0, 0]
        ]).cuda().unsqueeze(dim=0).unsqueeze(dim=0)
        kernel_r = torch.FloatTensor([
            [0, 0, 0],
            [0, 1, -1],
            [0, 0, 0]
        ]).cuda().unsqueeze(dim=0).unsqueeze(dim=0)
        kernel_u = torch.FloatTensor([
            [0, -1, 0],
            [0, 1, 0],
            [0, 0, 0]
        ]).cuda().unsqueeze(dim=0).unsqueeze(dim=0)
        kernel_d = torch.FloatTensor([
            [0, 0, 0],
            [0, 1, 0],
            [0, -1, 0]
        ]).cuda().unsqueeze(dim=0).unsqueeze(dim=0)

        self.weight_l = nn.Parameter(data=kernel_l, requires_grad=False)
        self.weight_r = nn.Parameter(data=kernel_r, requires_grad=False)
        self.weight_u = nn.Parameter(data=kernel_u, requires_grad=False)
        self.weight_d = nn.Parameter(data=kernel_d, requires_grad=False)
        self.pool = nn.AvgPool2d(kernel_size=4)

    def forward(self, org, enh):
        org_mean = torch.mean(input=org, dim=1, keepdim=True)
        enh_mean = torch.mean(input=enh, dim=1, keepdim=True)

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

        D_l = torch.pow(input=D_org_l - D_enh_l, exponent=2)
        D_r = torch.pow(input=D_org_r - D_enh_r, exponent=2)
        D_u = torch.pow(input=D_org_u - D_enh_u, exponent=2)
        D_d = torch.pow(input=D_org_d - D_enh_d, exponent=2)

        s = (D_l + D_r + D_u + D_d)
        return s


class L_exp(nn.Module):
    def __init__(self, patch_size=16, mean_val=0.8):
        super().__init__()
        self.pool = nn.AvgPool2d(kernel_size=patch_size)
        self.mean_val = mean_val

    def forward(self, x):
        x = torch.mean(input=x, dim=1, keepdim=True)
        mean = self.pool(x)

        e = torch.mean(
            input=torch.pow(
                input=mean - torch.FloatTensor(
                    [self.mean_val]
                ).cuda(),
                exponent=2
            )
        )
        return e


class L_tva(nn.Module):
    def __init__(self, TVLoss_weight=1):
        super().__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = (x.size()[2]-1) * x.size()[3]
        count_w = x.size()[2] * (x.size()[3] - 1)
        h_tv = torch.pow(
            input=(x[:, :, 1:, :]-x[:, :, :h_x-1, :]),
            exponent=2
        ).sum()
        w_tv = torch.pow(
            input=(x[:, :, :, 1:]-x[:, :, :, :w_x-1]),
            exponent=2
        ).sum()
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size
