"""
Segmentation loss and metric helpers shared by training and evaluation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def dice_score(pred, target, num_classes):
    """Macro Dice score over foreground classes only (background, class 0, excluded).

    Args:
        pred: (B, H, W) integer tensor of predicted class indices.
        target: (B, H, W) integer tensor of ground-truth class indices.
        num_classes: total number of classes including background.

    Returns:
        Scalar tensor: mean Dice over classes 1..num_classes-1.
    """
    pred_oh = F.one_hot(pred, num_classes=num_classes).permute(0, 3, 1, 2).float()
    target_oh = F.one_hot(target, num_classes=num_classes).permute(0, 3, 1, 2).float()

    dims = (0, 2, 3)
    intersection = torch.sum(pred_oh * target_oh, dims)
    cardinality = torch.sum(pred_oh + target_oh, dims)

    dice = (2.0 * intersection / (cardinality + 1e-6))[1:].mean()
    return dice


class DiceCELoss(nn.Module):
    """Combined Dice + cross-entropy loss for multi-class segmentation."""

    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.ce = nn.CrossEntropyLoss()

    def forward(self, logits, target):
        ce_loss = self.ce(logits, target)

        num_classes = logits.shape[1]
        target_oh = F.one_hot(target, num_classes=num_classes).permute(0, 3, 1, 2).float()
        probs = F.softmax(logits, dim=1)

        dims = (0, 2, 3)
        intersection = torch.sum(probs * target_oh, dims)
        cardinality = torch.sum(probs + target_oh, dims)
        dice_loss = 1 - (2.0 * intersection / (cardinality + 1e-6)).mean()

        return ce_loss + dice_loss


def per_class_dice(preds, targets, class_names):
    """Per-class Dice score computed on stacked prediction/target tensors.

    Args:
        preds: (N, H, W) integer tensor of predicted class indices.
        targets: (N, H, W) integer tensor of ground-truth class indices.
        class_names: dict mapping class index (foreground only, e.g. {1: "LV", ...}).

    Returns:
        dict mapping class name -> Dice score (float, rounded to 4 dp).
    """
    scores = {}
    for cls, name in class_names.items():
        pred_bin = (preds == cls).float()
        tgt_bin = (targets == cls).float()
        intersection = (pred_bin * tgt_bin).sum()
        d = (2 * intersection) / (pred_bin.sum() + tgt_bin.sum() + 1e-5)
        scores[name] = round(d.item(), 4)
    return scores
