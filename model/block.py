import torch
import torch.nn as nn
import torch.nn.functional as F


class RGB2YCrCb(nn.Module):
    def __init__(self, offset=0.5):
        super().__init__()
        self.offset = float(offset)

    def forward(self, x):
        R = x[:, 0:1, :, :].clone()
        G = x[:, 1:2, :, :].clone()
        B = x[:, 2:3, :, :].clone()

        offset = x.new_tensor(self.offset)

        Y = 0.299 * R + 0.587 * G + 0.114 * B
        Cr = (R - Y) * 0.713 + offset
        Cb = (B - Y) * 0.564 + offset

        Y = torch.clamp(input=Y, min=0.0, max=1.0)
        Cr = torch.clamp(input=Cr, min=0.0, max=1.0)
        Cb = torch.clamp(input=Cb, min=0.0, max=1.0)

        return Y, Cr, Cb


class YCrCb2RGB(nn.Module):
    def __init__(self, offset=0.5):
        super().__init__()
        self.offset = float(offset)

    def forward(self, Y, Cr, Cb):
        offset = Y.new_tensor(self.offset)

        Cr = torch.nan_to_num(input=Cr - offset, nan=0.0,
                              posinf=0.0, neginf=0.0)
        Cb = torch.nan_to_num(input=Cb - offset, nan=0.0,
                              posinf=0.0, neginf=0.0)

        R = Y + 1.403 * Cr
        G = Y - 0.344 * Cb - 0.714 * Cr
        B = Y + 1.773 * Cb

        RGB = torch.cat(tensors=[R, G, B], dim=1)
        RGB = torch.clamp(input=RGB, min=0.0, max=1.0)
        RGB = torch.nan_to_num(input=RGB, nan=0.0, posinf=1.0, neginf=0.0)

        return RGB


class HomomorphicSeparation(nn.Module):
    def __init__(self, size=512, cutoff=0.1, trainable=False, eps=1e-6):
        super().__init__()
        self.size = size
        self.eps = float(eps)

        # cutoff → logit 변환 후 파라미터화
        p = torch.tensor(data=max(cutoff, 0.1), dtype=torch.float32)
        logit = torch.log(input=p / (1.0 - p))
        self.raw_cutoff = nn.Parameter(data=logit, requires_grad=trainable)

        # 고정된 거리 맵 (원형 필터 마스크 용도)
        coord = torch.linspace(start=-1.0, end=1.0, steps=size)
        y, x = torch.meshgrid(coord, coord, indexing="ij")
        self.radius = torch.sqrt(input=x**2 + y**2)  # (H, W) float64 on CPU

    def _gaussian_lpf(self, dtype, device):
        cutoff = torch.sigmoid(input=self.raw_cutoff).to(
            dtype=dtype, device=device)
        cutoff = torch.clamp(input=cutoff, min=self.eps, max=1.0 - self.eps)

        radius = self.radius.to(dtype=dtype, device=device)
        h = torch.exp(input=-radius**2 / (2.0 * cutoff**2))  # (H, W)
        return h

    def forward(self, x):
        dtype, device = x.dtype, x.device

        # 1. 로그 도메인 변환
        x = torch.clamp(input=x, min=self.eps)
        x_log = torch.log(input=x + self.eps)

        # 2. FFT
        x_fft = torch.fft.fft2(x_log, norm='ortho')
        x_fft = torch.fft.fftshift(x_fft)

        # 3. 필터링
        H = self._gaussian_lpf(
            dtype=dtype, device=device).unsqueeze(dim=0).unsqueeze(dim=0)
        low_fft = x_fft * H
        high_fft = x_fft * (1 - H)

        # 8. iFFT
        low_ifft = torch.fft.ifft2(
            torch.fft.ifftshift(low_fft), norm='ortho').real
        high_ifft = torch.fft.ifft2(
            torch.fft.ifftshift(high_fft), norm='ortho').real
        low_ifft = torch.clamp(input=low_ifft, min=-10.0, max=10.0)
        high_ifft = torch.clamp(input=high_ifft, min=-10.0, max=10.0)

        # 5. exp 복원 + 안정화
        low = torch.exp(input=low_ifft).clamp(max=1e3) - self.eps
        high = torch.exp(input=high_ifft).clamp(max=1e3) - self.eps

        low = torch.nan_to_num(input=low, nan=0.0, posinf=1.0, neginf=0.0)
        high = torch.nan_to_num(input=high, nan=0.0, posinf=1.0, neginf=0.0)

        # 6. 출력 정규화
        low = torch.clamp(input=low, min=0.0, max=1.0)
        high = torch.clamp(input=high, min=0.0, max=1.0)
        return low, high


class SigmoidXSym(nn.Sigmoid):
    def __init__(self, *arg, **kwargs):
        super().__init__(*arg, **kwargs)

    def forward(self, x):
        x = -super().forward(input=x)
        return x


class SigmoidYSym(nn.Sigmoid):
    def __init__(self, *arg, **kwargs):
        super().__init__(*arg, **kwargs)

    def forward(self, x):
        x = super().forward(input=-x)
        return x


class SigmoidXYSym(nn.Sigmoid):
    def __init__(self, *arg, **kwargs):
        super().__init__(*arg, **kwargs)

    def forward(self, x):
        x = super().forward(input=-x)
        x = torch.abs(input=x)
        return x


class TanhSym(nn.Tanh):
    def __init__(self, *arg, **kwargs):
        super().__init__(*arg, **kwargs)

    def forward(self, x):
        x = super().forward(input=-x)
        return x


class TanhAbs(nn.Tanh):
    def __init__(self, *arg, **kwargs):
        super().__init__(*arg, **kwargs)

    def forward(self, x):
        x = torch.abs(input=super().forward(input=x))
        return x


class TanhSymAbs(nn.Tanh):
    def __init__(self, *arg, **kwargs):
        super().__init__(*arg, **kwargs)

    def forward(self, x):
        x = super().forward(input=-x)
        x = torch.abs(input=x)
        return x


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        hidden_channels = (in_channels + out_channels)/2
        hidden_channels = int(hidden_channels)

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=hidden_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(num_features=hidden_channels),
            nn.Conv2d(
                in_channels=hidden_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(num_features=out_channels),
        )

    def forward(self, x):
        return self.block(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            ),
            DoubleConv(
                in_channels=in_channels,
                out_channels=out_channels,
            ),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.maxpool_conv(x)
        return x


class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=2,
            stride=2
        )

        self.conv_block = nn.Sequential(
            DoubleConv(
                in_channels=in_channels,
                out_channels=out_channels
            ),
            nn.ReLU()
        )

    def forward(self, x1, x2):
        x1_up = self.up(x1)
        diffY = x2.size()[2] - x1_up.size()[2]
        diffX = x2.size()[3] - x1_up.size()[3]

        x1_pad = F.pad(
            input=x1_up,
            pad=[
                diffX // 2, diffX - diffX // 2,
                diffY // 2, diffY - diffY // 2
            ]
        )
        x = torch.cat(tensors=[x2, x1_pad], dim=1)
        x = self.conv_block(x)
        return x


class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.inc = nn.Sequential(
            DoubleConv(
                in_channels=1,
                out_channels=16,
            ),
            nn.ReLU()
        )
        self.down1 = Down(in_channels=16, out_channels=32)
        self.down2 = Down(in_channels=32, out_channels=64)
        self.down3 = Down(in_channels=64, out_channels=128)
        self.down4 = Down(in_channels=128, out_channels=256)
        self.down5 = Down(in_channels=256, out_channels=512)
        self.up5 = Up(in_channels=512, out_channels=256)
        self.up4 = Up(in_channels=256, out_channels=128)
        self.up3 = Up(in_channels=128, out_channels=64)
        self.up2 = Up(in_channels=64, out_channels=32)
        self.up1 = Up(in_channels=32, out_channels=16)
        self.outc = nn.Sequential(
            DoubleConv(
                in_channels=16,
                out_channels=1,
            ),
            nn.ReLU()
        )

    def forward(self, x):
        x_i = self.inc(x)
        d_1 = self.down1(x_i)
        d_2 = self.down2(d_1)
        d_3 = self.down3(d_2)
        d_4 = self.down4(d_3)
        d_5 = self.down5(d_4)
        u_5 = self.up5(d_5, d_4)
        u_4 = self.up4(u_5, d_3)
        u_3 = self.up3(u_4, d_2)
        u_2 = self.up2(u_3, d_1)
        u_1 = self.up1(u_2, x_i)
        x_o = self.outc(u_1)
        return x_o


class IterableRefine(nn.Module):
    def __init__(self):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=8,
                kernel_size=3,
                padding=1,
            ),
            TanhSymAbs()
        )

    def forward(self, x, r):
        r1, r2, r3, r4, r5, r6, r7, r8 = torch.split(
            tensor=self.block(r),
            split_size_or_sections=1,
            dim=1
        )
        x1 = x - r1*(torch.pow(input=x, exponent=2)-x)
        x2 = x1 - r2*(torch.pow(input=x1, exponent=2)-x1)
        x3 = x2 - r3*(torch.pow(input=x2, exponent=2)-x2)
        x4 = x3 - r4*(torch.pow(input=x3, exponent=2)-x3)
        x5 = x4 - r5*(torch.pow(input=x4, exponent=2)-x4)
        x6 = x5 - r6*(torch.pow(input=x5, exponent=2)-x5)
        x7 = x6 - r7*(torch.pow(input=x6, exponent=2)-x6)
        x8 = x7 - r8*(torch.pow(input=x7, exponent=2)-x7)
        x8 = torch.clamp(input=x8, min=0, max=1)
        return x8
