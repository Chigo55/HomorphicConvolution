import torch
import torch.nn as nn
import lightning as L

from torch.optim import SGD, Adam, AdamW, RMSprop, Adagrad, Adadelta, Rprop, LBFGS, ASGD, Adamax

from model.losses import *
from model.block import *
from utils.metrics import *
from utils.hook import register_full_nan_inf_hooks
from utils.utils import weights_init


class HomomorphicUnet(nn.Module):
    def __init__(self, image_size, offset, cutoff):
        super().__init__()

        self.rgb2ycrcb = RGB2YCrCb(
            offset=offset
        )
        self.homo_separate = HomomorphicSeparation(
            size=image_size,
            cutoff=cutoff,
            trainable=False
        )
        self.unet = UNet(
        )
        self.refine = IterableRefine(
        )
        self.ycrcb2rgb = YCrCb2RGB(
            offset=offset
        )

    def forward(self, x):
        self.Y, self.Cr, self.Cb = self.rgb2ycrcb(x)
        self.x_i, self.x_d = self.homo_separate(self.Y)
        self.o_i = self.unet(self.x_i)
        self.n_i = self.refine(self.x_i, self.o_i)
        self.n_Y = self.n_i * self.x_d
        self.enh_img = self.ycrcb2rgb(self.n_Y, self.Cr, self.Cb)
        return self.Y, self.Cr, self.Cb, self.x_i, self.x_d, self.o_i, self.n_i, self.n_Y, self.enh_img


class HomomorphicUnetLightning(L.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)

        self.model = HomomorphicUnet(
            image_size=hparams['image_size'],
            offset=hparams['offset'],
            cutoff=hparams["cutoff"],
        )
        self.model.apply(fn=weights_init)

        self.spa_loss = L_spa().eval()
        self.col_loss = L_col().eval()
        self.exp_loss = L_exp().eval()
        self.tva_loss = L_tva().eval()

        self.lambda_spa = hparams["lambda_spa"]
        self.lambda_col = hparams["lambda_col"]
        self.lambda_exp = hparams["lambda_exp"]
        self.lambda_tva = hparams["lambda_tva"]

        self.metric = ImageQualityMetrics(device="cuda")
        self.metric.eval()

    def forward(self, x):
        return self.model(x)

    def on_fit_start(self):
        register_full_nan_inf_hooks(model=self.model)

    def training_step(self, batch, batch_idx):
        x = batch.to(self.device)
        Y, Cr, Cb, x_i, x_d, o_i, n_i, n_Y, enh_img = self(x)

        loss_spa = self.lambda_spa * torch.mean(
            input=self.spa_loss(enh_img, x)
        )
        loss_col = self.lambda_col * torch.mean(
            input=self.col_loss(enh_img)
        )
        loss_exp = self.lambda_exp * torch.mean(
            input=self.exp_loss(enh_img)
        )
        loss_tva = self.lambda_tva * torch.mean(
            input=self.tva_loss(o_i)
        )

        loss_tot = (
            loss_spa +
            loss_col +
            loss_exp +
            loss_tva
        )

        self.log_dict(dictionary={
            "train/1_spa": loss_spa,
            "train/2_col": loss_col,
            "train/3_exp": loss_exp,
            "train/4_tva": loss_tva,
            "train/5_tot": loss_tot,
            # "train/4_tot": loss_tot,
        }, prog_bar=True)

        if torch.isnan(input=loss_spa):
            print("LOSS SPA IS NAN!")
        if torch.isnan(input=loss_col):
            print("LOSS COL IS NAN!")
        if torch.isnan(input=loss_exp):
            print("LOSS EXP IS NAN!")
        if torch.isnan(input=loss_tva):
            print("LOSS TVA IS NAN!")
        if torch.isnan(input=loss_tot):
            print("LOSS TOT IS NAN!")

        if batch_idx % 250 == 0:
            self.logger.experiment.add_images(
                "train/1_input",
                x,
                self.global_step
            )
            self.logger.experiment.add_images(
                "train/2_Y",
                Y,
                self.global_step
            )
            self.logger.experiment.add_images(
                "train/3_Cr",
                Cr,
                self.global_step
            )
            self.logger.experiment.add_images(
                "train/4_Cb",
                Cb,
                self.global_step
            )
            self.logger.experiment.add_images(
                "train/5_x_i",
                x_i,
                self.global_step
            )
            self.logger.experiment.add_images(
                "train/6_x_d",
                x_d,
                self.global_step
            )
            self.logger.experiment.add_images(
                "train/7_o_i",
                o_i,
                self.global_step
            )
            self.logger.experiment.add_images(
                "train/8_n_i",
                n_i,
                self.global_step
            )
            self.logger.experiment.add_images(
                "train/9_n_Y",
                n_Y,
                self.global_step
            )
            self.logger.experiment.add_images(
                "train/0_enh_img",
                enh_img,
                self.global_step
            )
        return loss_tot

    def validation_step(self, batch, batch_idx):
        x = batch.to(self.device)
        Y, Cr, Cb, x_i, x_d, o_i, n_i, n_Y, enh_img = self(x)

        loss_spa = self.lambda_spa * torch.mean(
            input=self.spa_loss(enh_img, x)
        )
        loss_col = self.lambda_col * torch.mean(
            input=self.col_loss(enh_img)
        )
        loss_exp = self.lambda_exp * torch.mean(
            input=self.exp_loss(enh_img)
        )
        loss_tva = self.lambda_tva * torch.mean(
            input=self.tva_loss(o_i)
        )

        loss_tot = (
            loss_spa +
            loss_col +
            loss_exp +
            loss_tva
        )

        self.log_dict(dictionary={
            "valid/1_spa": loss_spa,
            "valid/2_col": loss_col,
            "valid/3_exp": loss_exp,
            "valid/4_tva": loss_tva,
            "valid/5_tot": loss_tot,
        }, prog_bar=True)
        return loss_tot

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        x = batch.to(self.device)
        Y, Cr, Cb, x_i, x_d, o_i, n_i, n_Y, enh_img = self(x)

        metrics = self.metric.full(preds=enh_img, targets=x)

        self.log_dict(dictionary={
            "bench/1_PSNR": metrics["PSNR"],
            "bench/2_SSIM": metrics["SSIM"],
            "bench/3_LPIPS": metrics["LPIPS"],
            "bench/4_NIQE": metrics["NIQE"],
            "bench/5_BRISQUE": metrics["BRISQUE"],
        }, prog_bar=True)
        return metrics

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x = batch.to(self.device)
        Y, Cr, Cb, x_i, x_d, o_i, n_i, n_Y, enh_img = self(x)
        return enh_img

    def configure_optimizers(self):
        optim_name = self.hparams['optim'].lower()
        lr = self.hparams.get('lr', 1e-4)
        weight_decay = self.hparams.get('decay', 1e-5)
        eps = self.hparams.get('eps', 1e-8)

        if optim_name == "sgd":
            return SGD(
                params=self.parameters(),
                lr=lr,
                momentum=self.hparams.get('momentum', 0.9),
                weight_decay=weight_decay
            )
        elif optim_name == "adam":
            return Adam(
                params=self.parameters(),
                lr=lr,
                betas=self.hparams.get('betas', (0.9, 0.999)),
                eps=eps,
                weight_decay=weight_decay
            )
        elif optim_name == "adamw":
            return AdamW(
                params=self.parameters(),
                lr=lr,
                betas=self.hparams.get('betas', (0.9, 0.999)),
                eps=eps,
                weight_decay=weight_decay
            )
        elif optim_name == "rmsprop":
            return RMSprop(
                params=self.parameters(),
                lr=lr,
                alpha=self.hparams.get('alpha', 0.99),
                eps=eps,
                momentum=self.hparams.get('momentum', 0.9),
                weight_decay=weight_decay
            )
        elif optim_name == "adagrad":
            return Adagrad(
                params=self.parameters(),
                lr=lr,
                lr_decay=self.hparams.get('lr_decay', 1e-9),
                weight_decay=weight_decay,
                eps=eps
            )
        elif optim_name == "adadelta":
            return Adadelta(
                params=self.parameters(),
                lr=lr,
                rho=self.hparams.get('rho', 0.9),
                eps=eps,
                weight_decay=weight_decay
            )
        elif optim_name == "rprop":
            return Rprop(
                params=self.parameters(),
                lr=lr,
                etas=self.hparams.get('etas', (0.5, 1.2)),
                step_sizes=self.hparams.get('step_sizes', (1e-6, 50))
            )
        elif optim_name == "lbfgs":
            return LBFGS(
                params=self.parameters(),
                lr=lr,
                max_iter=self.hparams.get('max_iter', 20),
                max_eval=self.hparams.get('max_eval', None),
                tolerance_grad=self.hparams.get('tolerance_grad', 1e-7),
                tolerance_change=self.hparams.get('tolerance_change', 1e-9),
                history_size=self.hparams.get('history_size', 100),
                line_search_fn=self.hparams.get('line_search_fn', None)
            )
        elif optim_name == "asgd":
            return ASGD(
                params=self.parameters(),
                lr=lr,
                lambd=self.hparams.get('lambd', 1e-4),
                alpha=self.hparams.get('alpha', 0.75),
                t0=self.hparams.get('t0', 1e6),
                weight_decay=weight_decay
            )
        elif optim_name == "adamax":
            return Adamax(
                params=self.parameters(),
                lr=lr,
                betas=self.hparams.get('betas', (0.9, 0.999)),
                eps=eps,
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optim_name}")
