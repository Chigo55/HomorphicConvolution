import torch
import lightning as L

from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers.optimization import get_cosine_schedule_with_warmup

from model.block2 import *
from model.losses2 import *
from utils.metrics import ImageQualityMetrics


class HomomorphicDit(nn.Module):
    def __init__(
        self,
        image_size=512,
        hidden_size=768,
        patch_size=4,
        depth=12,
        num_heads=12,
        in_channels=3,
        out_channels=1,
        use_detail_embedding=False
    ):
        super().__init__()
        self.image_size = image_size
        self.hidden_size = hidden_size
        self.patch_size = patch_size
        self.depth = depth
        self.num_heads = num_heads
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_detail_embedding = use_detail_embedding

        self.rgb2ycrcb = RGB2YCrCb()
        self.homo_separate = HomomorphicSeparation()
        self.illum_embedding = PatchEmbedding(
            image_size=self.image_size,
            embed_dim=self.hidden_size,
            patch_size=self.patch_size,
            bias=True
        )
        self.detail_embedding = PatchEmbedding(
            image_size=self.image_size,
            embed_dim=self.hidden_size,
            patch_size=self.patch_size,
            bias=True
        )
        self.pos_embedding = PositionalEmbedding(
            embed_dim=self.hidden_size,
            size_h=self.illum_embedding.grid_size_h,
            size_w=self.illum_embedding.grid_size_w
        )
        self.t_embedding = TimeEmbedding(
            hidden_size=self.hidden_size,
            bias=True
        )
        self.blocks = nn.ModuleList([
            DiTBlock(
                hidden_size=self.hidden_size,
                num_heads=self.num_heads,
                use_detail_embedding=self.use_detail_embedding,
            )
            for _ in range(self.depth)
        ])
        self.final_layer = FinalLayer(
            hidden_size=self.hidden_size,
            patch_size=self.patch_size,
            out_channels=self.out_channels,
            use_detail_embedding=self.use_detail_embedding,
        )
        self.ycrcb2rgb = YCrCb2RGB()
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w_i = self.illum_embedding.proj.weight.data
        nn.init.xavier_uniform_(w_i.view([w_i.shape[0], -1]))
        nn.init.constant_(self.illum_embedding.proj.bias, 0)
        w_d = self.detail_embedding.proj.weight.data
        nn.init.xavier_uniform_(w_d.view([w_d.shape[0], -1]))
        nn.init.constant_(self.detail_embedding.proj.bias, 0)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedding.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedding.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            # nn.init.xavier_uniform_(block.adaLN_modulation[-1].weight)
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        # nn.init.xavier_uniform_(self.final_layer.adaLN_modulation[-1].weight)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        c = self.out_channels
        p = self.illum_embedding.patch_size
        h = w = int(x.shape[1] ** 0.5)

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        x = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return x

    def forward(self, x, t):
        Y, Cr, Cb = self.rgb2ycrcb(x)
        i_x, d_x = self.homo_separate(Y)
        i_emb = self.illum_embedding(i_x) + self.pos_embedding()

        if self.use_detail_embedding:
            d_emb = self.detail_embedding(d_x) + self.pos_embedding()
            t_emb = self.t_embedding(t).unsqueeze(
                1).expand(-1, d_emb.shape[1], -1)
            cond = t_emb + d_emb
        else:
            t_emb = self.t_embedding(t)
            cond = t_emb

        for block in self.blocks:
            i_emb = block(i_emb, cond)

        i_emb = self.final_layer(i_emb, cond)
        i_x = self.unpatchify(x=i_emb)
        n_Y = i_x * d_x
        YCrCb = torch.cat(tensors=[n_Y, Cr, Cb], dim=1)
        dit_enh_img = self.ycrcb2rgb(YCrCb)
        return dit_enh_img, n_Y, Y


class HomomorphicDitDCE(HomomorphicDit):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.dce = DCE(
            in_channels=self.in_channels
        )

    def forward(self, x, t):
        Y, Cr, Cb = self.rgb2ycrcb(x)
        i_x, d_x = self.homo_separate(Y)
        i_emb = self.illum_embedding(i_x) + self.pos_embedding()

        if self.use_detail_embedding:
            d_emb = self.detail_embedding(d_x) + self.pos_embedding()
            t_emb = self.t_embedding(t).unsqueeze(
                1).expand(-1, d_emb.shape[1], -1)
            cond = t_emb + d_emb
        else:
            t_emb = self.t_embedding(t)
            cond = t_emb

        for block in self.blocks:
            i_emb = block(i_emb, cond)

        i_emb = self.final_layer(i_emb, cond)
        i_x = self.unpatchify(i_emb)
        n_Y = i_x * d_x
        YCrCb = torch.cat([n_Y, Cr, Cb], dim=1)
        dit_enh_img = self.ycrcb2rgb(YCrCb)
        dce_enh_img, r = self.dce(x, dit_enh_img)
        return dce_enh_img, r, n_Y, Y


class HomomorphicDiTLightning(L.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)

        self.model = HomomorphicDit(
            image_size=hparams['image_size'],
            hidden_size=hparams['hidden_size'],
            patch_size=hparams['patch_size'],
            depth=hparams['depth'],
            num_heads=hparams['num_heads'],
            in_channels=hparams["in_channels"],
            out_channels=hparams["out_channels"],
            use_detail_embedding=hparams['use_detail_embedding'],
        )

        self.spa_loss = L_spa()
        self.col_loss = L_col()
        self.exp_loss = L_exp()
        self.con_loss = L_con()

        self.lambda_spa = hparams["lambda_spa"]
        self.lambda_col = hparams["lambda_col"]
        self.lambda_exp = hparams["lambda_exp"]
        self.lambda_con = hparams["lambda_con"]

        self.timestep_range = hparams['timestep_range']

        self.metric = ImageQualityMetrics(device="cuda")
        self.metric.eval()

    def forward(self, x, t):
        return self.model(x, t)

    def training_step(self, batch, batch_idx):
        x = batch.to(self.device)
        t = torch.zeros(
            size=(x.size(0),),
            dtype=torch.long,
            device=self.device
        )
        enh_img, n_Y, Y = self(x, t)

        loss_spa = self.lambda_spa * torch.mean(self.spa_loss(n_Y, Y))
        loss_col = self.lambda_col * torch.mean(self.col_loss(enh_img))
        loss_exp = self.lambda_exp * torch.mean(self.exp_loss(n_Y))
        loss_con = self.lambda_con * torch.mean(self.con_loss(n_Y, t))

        loss_tot = (loss_spa + loss_col + loss_exp + loss_con)

        self.log_dict(dictionary={
            "train/spa": loss_spa,
            "train/col": loss_col,
            "train/exp": loss_exp,
            "train/con": loss_con,
            "train/tot": loss_tot,
        }, prog_bar=True)
        return loss_tot

    def validation_step(self, batch, batch_idx):
        x = batch.to(self.device)
        t = torch.zeros(
            size=(x.size(0),),
            dtype=torch.long,
            device=self.device
        )
        enh_img, n_Y, Y = self(x, t)

        loss_spa = self.lambda_spa * torch.mean(self.spa_loss(n_Y, Y))
        loss_col = self.lambda_col * torch.mean(self.col_loss(enh_img))
        loss_exp = self.lambda_exp * torch.mean(self.exp_loss(n_Y))
        loss_con = self.lambda_con * torch.mean(self.con_loss(n_Y, t))

        loss_tot = (loss_spa + loss_col + loss_exp + loss_con)

        self.log_dict(dictionary={
            "valid/spa": loss_spa,
            "valid/col": loss_col,
            "valid/exp": loss_exp,
            "valid/con": loss_con,
            "valid/tot": loss_tot,
        }, prog_bar=True)
        return loss_tot

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        x = batch.to(self.device)
        t = torch.zeros(
            size=(x.size(0),),
            dtype=torch.long,
            device=self.device
        )
        enh_img, n_Y, Y = self(x, t)

        metrics = self.metric.full(enh_img, x)

        self.log_dict(dictionary={
            "bench/PSNR": metrics["PSNR"],
            "bench/SSIM": metrics["SSIM"],
            "bench/LPIPS": metrics["LPIPS"],
            "bench/NIQE": metrics["NIQE"],
            "bench/BRISQUE": metrics["BRISQUE"],
        }, prog_bar=True)
        return metrics

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x = batch.to(self.device)
        t = torch.zeros(
            size=(x.size(0),),
            dtype=torch.long,
            device=self.device
        )
        enh_img, n_Y, Y = self(x, t)
        return enh_img

    def configure_optimizers(self):
        total_steps = self.trainer.estimated_stepping_batches

        optimizer = torch.optim.Adam(
            params=self.parameters(),
            lr=self.hparams['lr']
        )

        scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=2600,  # 1~2 epoch 분량
            num_training_steps=total_steps,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            }
        }


class HomomorphicDiTDCELightning(L.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)

        self.model = HomomorphicDitDCE(
            image_size=hparams['image_size'],
            hidden_size=hparams['hidden_size'],
            patch_size=hparams['patch_size'],
            depth=hparams['depth'],
            num_heads=hparams['num_heads'],
            in_channels=hparams["in_channels"],
            out_channels=hparams["out_channels"],
            use_detail_embedding=hparams['use_detail_embedding'],
        )

        self.spa_loss = L_spa()
        self.col_loss = L_col()
        self.exp_loss = L_exp()
        self.con_loss = L_con()
        self.tva_loss = L_tva()

        self.lambda_spa = hparams["lambda_spa"]
        self.lambda_col = hparams["lambda_col"]
        self.lambda_exp = hparams["lambda_exp"]
        self.lambda_con = hparams["lambda_con"]
        self.lambda_tva = hparams["lambda_tva"]

        self.timestep_range = hparams['timestep_range']

        self.metric = ImageQualityMetrics(device="cuda")
        self.metric.eval()

    def forward(self, x, t):
        return self.model(x, t)

    def training_step(self, batch, batch_idx):
        x = batch.to(self.device)
        t = torch.zeros(
            size=(x.size(0),),
            dtype=torch.long,
            device=self.device
        )
        dce_enh_img, r, n_Y, Y = self(x, t)

        loss_spa = self.lambda_spa * torch.mean(self.spa_loss(n_Y, Y))
        loss_col = self.lambda_col * torch.mean(self.col_loss(dce_enh_img))
        loss_exp = self.lambda_exp * torch.mean(self.exp_loss(n_Y))
        loss_con = self.lambda_con * torch.mean(self.con_loss(n_Y, t))
        loss_tva = self.lambda_tva * torch.mean(self.tva_loss(r))

        loss_tot = (loss_spa + loss_col + loss_exp + loss_con + loss_tva)

        self.log_dict(dictionary={
            "train/spa": loss_spa,
            "train/col": loss_col,
            "train/exp": loss_exp,
            "train/con": loss_con,
            "train/tva": loss_tva,
            "train/tot": loss_tot,
        }, prog_bar=True)

        return loss_tot

    def validation_step(self, batch, batch_idx):
        x = batch.to(self.device)
        t = torch.zeros(
            size=(x.size(0),),
            dtype=torch.long,
            device=self.device
        )
        dce_enh_img, r, n_Y, Y = self(x, t)

        loss_spa = self.lambda_spa * torch.mean(self.spa_loss(x, dce_enh_img))
        loss_col = self.lambda_col * torch.mean(self.col_loss(dce_enh_img))
        loss_exp = self.lambda_exp * torch.mean(self.exp_loss(dce_enh_img))
        loss_con = self.lambda_con * torch.mean(self.con_loss(dce_enh_img, t))
        loss_tva = self.lambda_tva * torch.mean(self.tva_loss(r))

        loss_tot = (loss_spa + loss_col + loss_exp + loss_con + loss_tva)

        self.log_dict(dictionary={
            "valid/spa": loss_spa,
            "valid/col": loss_col,
            "valid/exp": loss_exp,
            "valid/con": loss_con,
            "valid/tva": loss_tva,
            "valid/tot": loss_tot,
        }, prog_bar=True)

        return loss_tot

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        x = batch.to(self.device)
        t = torch.zeros(
            size=(x.size(0),),
            dtype=torch.long,
            device=self.device
        )
        dce_enh_img, r, n_Y, Y = self(x, t)

        metrics = self.metric.full(dce_enh_img, x)
        return metrics

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x = batch.to(self.device)
        t = torch.zeros(
            size=(x.size(0),),
            dtype=torch.long,
            device=self.device
        )
        dce_enh_img, r, n_Y, Y = self(x, t)
        return dce_enh_img

    def configure_optimizers(self):
        total_steps = self.trainer.estimated_stepping_batches

        optimizer = torch.optim.Adam(
            params=self.parameters(),
            lr=self.hparams['lr']
        )

        scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=1600,  # 1~2 epoch 분량
            num_training_steps=total_steps,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            }
        }
