import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from functools import partial


class RGB2YCrCb(nn.Module):
    def __init__(self, offset=0.5,):
        super().__init__()
        self.offset = offset

    def forward(self, x):
        R = x[:, 0:1, :, :]
        G = x[:, 1:2, :, :]
        B = x[:, 2:3, :, :]

        Y = 0.299 * R + 0.587 * G + 0.114 * B
        Cr = (R - Y) * 0.713 + self.offset
        Cb = (B - Y) * 0.564 + self.offset
        return Y, Cr, Cb


class YCrCb2RGB(nn.Module):
    def __init__(self, offset=0.5,):
        super().__init__()
        self.offset = offset

    def forward(self, ycrcb):
        Y = ycrcb[:, 0:1, :, :]
        Cr = ycrcb[:, 1:2, :, :] - self.offset
        Cb = ycrcb[:, 2:3, :, :] - self.offset

        R = Y + 1.403 * Cr
        G = Y - 0.344 * Cb - 0.714 * Cr
        B = Y + 1.773 * Cb

        rgb = torch.cat([R, G, B], dim=1)
        return rgb


class HomomorphicSeparation(nn.Module):
    def __init__(self, cutoff=0.1, eps=1e-6,):
        super().__init__()
        self.eps = eps
        self.cutoff = cutoff

    def _gaussian_low_pass_filter(self, shape, cutoff, device):
        H, W = shape
        y = torch.linspace(-1, 1, H, device=device).reshape(-1, 1)
        x = torch.linspace(-1, 1, W, device=device).reshape(1, -1)
        d = torch.sqrt(x ** 2 + y ** 2)  # 거리

        # Gaussian LPF
        filter_mask = torch.exp(-(d ** 2) / (2 * (cutoff ** 2)))
        return filter_mask  # (H, W)

    def forward(self, x):
        B, C, H, W = x.shape

        # 1. log 변환
        x_log = torch.log(x + self.eps)

        # 2. FFT
        x_fft = torch.fft.fft2(x_log.squeeze(1))  # (B, H, W)

        # 3. 필터 마스크 생성
        filter_mask = self._gaussian_low_pass_filter(
            shape=(H, W), cutoff=self.cutoff, device=x.device)
        filter_mask = filter_mask[None, :, :].expand(B, -1, -1)  # (B, H, W)

        # 4. 분리
        low_fft = x_fft * filter_mask
        high_fft = x_fft * (1 - filter_mask)

        # 5. IFFT → Real
        low_spatial = torch.real(torch.fft.ifft2(low_fft)).unsqueeze(1)
        high_spatial = torch.real(torch.fft.ifft2(high_fft)).unsqueeze(1)

        # 6. exp 복원
        illumination = torch.exp(low_spatial)
        detail = torch.exp(high_spatial)

        return illumination, detail


class PatchEmbedding(nn.Module):
    def __init__(self, image_size=512, in_channels=1, embed_dim=768, patch_size=4, bias=True):
        super().__init__()
        self.image_size = (image_size, image_size)
        self.grid_size_h = self.image_size[0] // patch_size
        self.grid_size_w = self.image_size[1] // patch_size
        self.num_patches = self.grid_size_h * self.grid_size_w
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.bias = bias

        self.proj = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=self.bias
        )

    def forward(self, x):
        # (B, embed_dim, image_size/patch_size, image_size/patch_size) -> (B, 768, 128, 128)
        x = self.proj(x)
        # (B, embed_dim, (image_size/patch_size)^2) -> (B, 768, 16384)
        x = x.flatten(2)
        # (B, (image_size/patch_size)^2, embed_dim) -> (B, 16384, 768)
        x = x.transpose(1, 2)
        return x


class PositionalEmbedding(nn.Module):
    def __init__(self, embed_dim=768, size_h=128, size_w=128):
        super().__init__()
        self.embed_dim = embed_dim
        self.size_h = size_h
        self.size_w = size_w

        self.register_buffer("pos_embed", self._build_sincos_embedding())

    def _build_sincos_embedding(self):
        grid = self._create_2d_grid(self.size_h, self.size_w)
        pos_embed = self._get_2d_sincos_pos_embed_from_grid(
            self.embed_dim, grid)

        return torch.from_numpy(pos_embed).float().unsqueeze(0)  # (1, N, D)

    def _create_2d_grid(self, H, W):
        grid_h = np.arange(H, dtype=np.float32)
        grid_w = np.arange(W, dtype=np.float32)
        grid = np.meshgrid(grid_w, grid_h)  # (2, H, W)
        grid = np.stack(grid, axis=0).reshape(2, 1, H, W)
        return grid

    def _get_1d_sincos_pos_embed_from_grid(self, embed_dim, pos):
        omega = np.arange(embed_dim // 2, dtype=np.float64)
        omega /= embed_dim / 2.
        omega = 1. / 10000**omega

        pos = pos.reshape(-1)
        out = np.einsum('m,d->md', pos, omega)

        emb_sin = np.sin(out)
        emb_cos = np.cos(out)

        return np.concatenate([emb_sin, emb_cos], axis=1)

    def _get_2d_sincos_pos_embed_from_grid(self, embed_dim, grid):
        emb_h = self._get_1d_sincos_pos_embed_from_grid(
            embed_dim // 2, grid[0])
        emb_w = self._get_1d_sincos_pos_embed_from_grid(
            embed_dim // 2, grid[1])
        return np.concatenate([emb_h, emb_w], axis=1)

    def forward(self):
        # (1, 16384, 768)
        return self.pos_embed


class TimeEmbedding(nn.Module):
    def __init__(self, hidden_size=768, embed_dim=256, bias=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.embed_dim = embed_dim
        self.bias = bias

        self.mlp = nn.Sequential(
            nn.Linear(
                in_features=self.embed_dim,
                out_features=self.hidden_size,
                bias=self.bias
            ),
            nn.SiLU(),
            nn.Linear(
                in_features=self.hidden_size,
                out_features=self.hidden_size,
                bias=self.bias
            )
        )

    @staticmethod
    def timestep_embedding(t, embed_dim, max_period=10000):
        half_dim = embed_dim // 2
        emb = torch.exp(math.log(max_period) * torch.arange(end=half_dim,
                        dtype=t.dtype, device=t.device) / half_dim)
        emb = t[:, None].float() * emb[None, :]
        emb = torch.cat([torch.cos(emb), torch.sin(emb)], dim=1)
        return emb

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.embed_dim)
        t_emb = self.mlp(t_freq)
        # (B, 768)
        return t_emb


class MultiLayerPerceptron(nn.Module):
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            bias=True,
            drop=0.,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(
            in_features=in_features,
            out_features=hidden_features,
            bias=bias
        )
        self.act = act_layer()
        self.drop1 = nn.Dropout(p=drop)
        self.fc2 = nn.Linear(
            in_features=hidden_features,
            out_features=out_features,
            bias=bias
        )
        self.drop2 = nn.Dropout(p=drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(
            in_features=dim,
            out_features=dim * 3,
            bias=qkv_bias
        )
        self.q_norm = nn.LayerNorm(normalized_shape=self.head_dim)
        self.k_norm = nn.LayerNorm(normalized_shape=self.head_dim)
        self.attn_drop = nn.Dropout(p=attn_drop)
        self.proj = nn.Linear(in_features=dim, out_features=dim)
        self.proj_drop = nn.Dropout(p=proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class DiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, use_detail_embedding=False,):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.use_detail_embedding = use_detail_embedding
        mlp_hidden_dim = int(hidden_size * self.mlp_ratio)
        approx_gelu = partial(nn.GELU, approximate="tanh")

        self.norm1 = nn.LayerNorm(
            normalized_shape=self.hidden_size,
            elementwise_affine=False,
            eps=1e-6
        )
        self.attn = Attention(
            dim=self.hidden_size,
            num_heads=self.num_heads,
            qkv_bias=True
        )
        self.norm2 = nn.LayerNorm(
            normalized_shape=self.hidden_size,
            elementwise_affine=False,
            eps=1e-6
        )
        self.mlp = MultiLayerPerceptron(
            in_features=self.hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=approx_gelu,
            drop=0
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                in_features=self.hidden_size,
                out_features=6 * self.hidden_size,
                bias=True
            )
        )

    def forward(self, x, t):
        if self.use_detail_embedding:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(
                t).chunk(6, dim=2)
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(
                t).chunk(6, dim=1)
            shift_msa = shift_msa.unsqueeze(1)
            scale_msa = scale_msa.unsqueeze(1)
            gate_msa = gate_msa.unsqueeze(1)
            shift_mlp = shift_mlp.unsqueeze(1)
            scale_mlp = scale_mlp.unsqueeze(1)
            gate_mlp = gate_mlp.unsqueeze(1)

        x = self.norm1(x) * (1 + scale_msa) + shift_msa
        x = x + gate_msa * self.attn(x)
        x = self.norm2(x) * (1 + scale_mlp) + shift_mlp
        x = x + gate_mlp * self.mlp(x)
        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels, use_detail_embedding=False,):
        super().__init__()
        self.use_detail_embedding = use_detail_embedding
        self.norm_final = nn.LayerNorm(
            normalized_shape=hidden_size,
            elementwise_affine=False,
            eps=1e-6
        )
        self.linear = nn.Linear(
            in_features=hidden_size,
            out_features=patch_size * patch_size * out_channels,
            bias=True
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                in_features=hidden_size,
                out_features=2 * hidden_size,
                bias=True
            )
        )

    def forward(self, x, t):
        if self.use_detail_embedding:
            t = t.mean(dim=1)
        shift, scale = self.adaLN_modulation(t).chunk(2, dim=1)
        x = self.norm_final(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        x = self.linear(x)
        return x


class DCE(nn.Module):
    def __init__(self, in_channels=3, hidden=32, out_channel=24):
        super().__init__()

        self.conv_input = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=hidden,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True
            ),
            nn.SiLU()
        )
        self.conv_hidden_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=hidden,
                out_channels=hidden,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True
            ),
            nn.SiLU()
        )
        self.conv_hidden_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=hidden*2,
                out_channels=hidden,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True
            ),
            nn.SiLU()
        )
        self.conv_tanh = nn.Sequential(
            nn.Conv2d(
                in_channels=hidden*2,
                out_channels=out_channel,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True
            ),
            nn.Tanh()
        )

    def forward(self, x, rgb):
        rgb_1 = self.conv_input(rgb)
        rgb_2 = self.conv_hidden_1(rgb_1)
        rgb_3 = self.conv_hidden_1(rgb_2)
        rgb_4 = self.conv_hidden_1(rgb_3)
        rgb_5 = self.conv_hidden_2(
            torch.cat(
                tensors=[rgb_3, rgb_4],
                dim=1
            )
        )
        rgb_6 = self.conv_hidden_2(
            torch.cat(
                tensors=[rgb_2, rgb_5],
                dim=1
            )
        )
        rgb_r = self.conv_tanh(
            torch.cat(
                tensors=[rgb_1, rgb_6],
                dim=1
            )
        )
        r1, r2, r3, r4, r5, r6, r7, r8 = torch.split(
            tensor=rgb_r, split_size_or_sections=3, dim=1)

        x = x + r1 * (torch.pow(x, 2)-x)
        x = x + r2 * (torch.pow(x, 2)-x)
        x = x + r3 * (torch.pow(x, 2)-x)
        x = x + r4 * (torch.pow(x, 2)-x)
        x = x + r5 * (torch.pow(x, 2)-x)
        x = x + r6 * (torch.pow(x, 2)-x)
        x = x + r7 * (torch.pow(x, 2)-x)
        r = torch.cat([r1, r2, r3, r4, r5, r6, r7, r8], 1)
        enh_img = x + r8 * (torch.pow(x, 2)-x)
        return enh_img, r
