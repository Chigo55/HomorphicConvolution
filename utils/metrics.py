import cv2
import math
import numpy as np
import torch
import torch.nn as nn

from scipy.ndimage import convolve
from scipy.special import gamma
from torchmetrics.image import (
    PeakSignalNoiseRatio,
    StructuralSimilarityIndexMeasure,
    LearnedPerceptualImagePatchSimilarity,
)


class ImageQualityMetrics(nn.Module):
    def __init__(self, device="cuda", data_range=1.0):
        super().__init__()

        self.device_type = device

        # reference-based metrics
        self.psnr = PeakSignalNoiseRatio(data_range=data_range).to(device)
        self.ssim = StructuralSimilarityIndexMeasure(
            data_range=data_range).to(device)
        self.lpips = LearnedPerceptualImagePatchSimilarity(
            net_type='squeeze').to(device)

        # for NIQE (dummy pristine dist)
        self.niqe_stats = np.load(
            "utils/files/niqe_params.npz", allow_pickle=True)
        self.mu_pris_param = self.niqe_stats['mu_pris_param']
        self.cov_pris_param = self.niqe_stats['cov_pris_param']
        self.gaussian_window = self.niqe_stats['gaussian_window']

    def forward(self, preds: torch.Tensor, targets: torch.Tensor):
        preds = preds.to(self.device_type)
        targets = targets.to(self.device_type)

        return {
            "PSNR": self.psnr(preds, targets).item(),
            "SSIM": self.ssim(preds, targets).item(),
            "LPIPS": self.lpips(preds, targets).squeeze().mean().item(),
        }

    def no_ref(self, preds: torch.Tensor):
        preds = preds.to(self.device_type)
        preds_np = preds.detach().cpu().numpy()
        preds_np = np.clip(preds_np, 0, 1)

        niqe_list = []
        brisque_list = []

        for img in preds_np:
            img_np = np.transpose(img, (1, 2, 0))  # (C, H, W) → (H, W, C)
            img_np_uint8 = (img_np * 255).astype(np.uint8)

            niqe = self._compute_niqe(img_np)
            brisuqe = self._compute_brisque(img_np_uint8)

            niqe_list.append(niqe)
            brisque_list.append(brisuqe)

        return {
            "NIQE": float(np.mean(niqe_list)),
            "BRISQUE": float(np.mean(brisque_list)),
        }

    def full(self, preds, targets):
        ref_metrics = self.forward(preds, targets)
        no_ref_metrics = self.no_ref(preds)
        return {**ref_metrics, **no_ref_metrics}

    # --------------------------
    # Custom no-reference metrics
    # --------------------------

    def _compute_niqe(self, img_np):
        img = img_np.astype(np.float32)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray = gray.round()
        return self._niqe(gray)

    def _compute_brisque(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        brisque_score = cv2.quality.QualityBRISQUE_compute(
            img, "utils/files/brisque_model.yaml", "utils/files/brisque_range.yaml")
        return brisque_score

    def _niqe(self, img, block_size_h=96, block_size_w=96):
        assert img.ndim == 2
        h, w = img.shape
        num_block_h = math.floor(h / block_size_h)
        num_block_w = math.floor(w / block_size_w)
        img = img[0:num_block_h * block_size_h, 0:num_block_w * block_size_w]

        distparam = []
        for scale in (1, 2):
            mu = convolve(img, self.gaussian_window, mode='nearest')
            sigma = np.sqrt(np.abs(convolve(
                np.square(img), self.gaussian_window, mode='nearest') - np.square(mu)))
            img_nomalized = (img - mu) / (sigma + 1)

            feat = []
            for idx_w in range(num_block_w):
                for idx_h in range(num_block_h):
                    block = img_nomalized[
                        idx_h * block_size_h // scale:(idx_h + 1) * block_size_h // scale,
                        idx_w * block_size_w // scale:(idx_w + 1) * block_size_w // scale
                    ]
                    feat.append(self._compute_feature(block))

            distparam.append(np.array(feat))

            if scale == 1:
                img = cv2.resize(img / 255., dsize=(0, 0), fx=0.5,
                                 fy=0.5, interpolation=cv2.INTER_CUBIC) * 255.

        distparam = np.concatenate(distparam, axis=1)
        mu_distparam = np.nanmean(distparam, axis=0)
        distparam_no_nan = distparam[~np.isnan(distparam).any(axis=1)]
        cov_distparam = np.cov(distparam_no_nan, rowvar=False)

        invcov_param = np.linalg.pinv(
            (self.cov_pris_param + cov_distparam) / 2)
        quality = np.matmul(
            np.matmul((self.mu_pris_param - mu_distparam), invcov_param),
            np.transpose((self.mu_pris_param - mu_distparam))
        )
        return float(np.sqrt(quality))

    def _compute_feature(self, block):
        def estimate_aggd_param(block):
            block = block.flatten()
            gam = np.arange(0.2, 10.001, 0.001)
            gam_reciprocal = np.reciprocal(gam)
            r_gam = np.square(gamma(gam_reciprocal * 2)) / (
                gamma(gam_reciprocal) * gamma(gam_reciprocal * 3))

            left_std = np.sqrt(np.mean(block[block < 0]**2))
            right_std = np.sqrt(np.mean(block[block > 0]**2))
            gammahat = left_std / right_std
            rhat = (np.mean(np.abs(block)))**2 / np.mean(block**2)
            rhatnorm = (rhat * (gammahat**3 + 1) *
                        (gammahat + 1)) / ((gammahat**2 + 1)**2)
            array_position = np.argmin((r_gam - rhatnorm)**2)
            alpha = gam[array_position]
            beta_l = left_std * np.sqrt(gamma(1 / alpha) / gamma(3 / alpha))
            beta_r = right_std * np.sqrt(gamma(1 / alpha) / gamma(3 / alpha))
            return (alpha, beta_l, beta_r)

        feat = []
        alpha, beta_l, beta_r = estimate_aggd_param(block)
        feat.extend([alpha, (beta_l + beta_r) / 2])
        shifts = [[0, 1], [1, 0], [1, 1], [1, -1]]
        for shift in shifts:
            shifted = np.roll(block, shift, axis=(0, 1))
            alpha, beta_l, beta_r = estimate_aggd_param(block * shifted)
            mean = (beta_r - beta_l) * (gamma(2 / alpha) / gamma(1 / alpha))
            feat.extend([alpha, mean, beta_l, beta_r])
        return feat
