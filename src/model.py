"""
Model definitions: a segmentation-models-pytorch U-Net wrapper and the
PyTorch Lightning training module that ties model + loss + metrics together.
"""

import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from torchmetrics.classification import MulticlassJaccardIndex

from .losses import DiceCELoss, dice_score

NUM_CLASSES = 4  # background, LV cavity, RV cavity, myocardium


class SMPUNet(nn.Module):
    """Thin wrapper around segmentation_models_pytorch's U-Net so the encoder
    backbone is swappable via a single string argument."""

    def __init__(self, encoder_name="resnet18", in_channels=1, out_channels=NUM_CLASSES):
        super().__init__()
        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights="imagenet",
            in_channels=in_channels,
            classes=out_channels,
        )

    def forward(self, x):
        return self.model(x)


class LitUNet2D(pl.LightningModule):
    """LightningModule for 2D cardiac MRI segmentation with a swappable
    SMP U-Net encoder backbone.

    Logs `{stage}_loss`, `{stage}_dice`, `{stage}_miou` for stage in
    {train, val, test}.
    """

    def __init__(self, lr=1e-3, encoder_name="resnet18"):
        super().__init__()
        self.save_hyperparameters()
        self.model = SMPUNet(encoder_name=encoder_name)
        self.out_channels = NUM_CLASSES
        self.loss_fn = DiceCELoss(num_classes=self.out_channels)
        self.miou_metric = MulticlassJaccardIndex(num_classes=self.out_channels, average="macro")

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)
        return {"optimizer": opt, "lr_scheduler": sch}

    def base_step(self, batch, batch_idx, stage: str):
        x, y = batch
        y = y.long()

        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log(f"{stage}_loss", loss, on_epoch=True, prog_bar=True)

        with torch.no_grad():
            preds = torch.argmax(logits, dim=1)
            dice = dice_score(preds, y, num_classes=self.out_channels)
            miou = self.miou_metric(preds, y)
            self.log(f"{stage}_dice", dice, on_epoch=True, prog_bar=True)
            self.log(f"{stage}_miou", miou, on_epoch=True, prog_bar=True)

        return loss

    def training_step(self, batch, batch_idx):
        return self.base_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self.base_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self.base_step(batch, batch_idx, "test")
