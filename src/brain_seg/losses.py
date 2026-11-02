"""Loss functions for the 4-class softmax output.

All losses take ``logits`` of shape (B, K, X, Y, Z) and ``target`` of shape
(B, 1, X, Y, Z) holding canonical class indices; one-hot encoding and softmax
are applied inside the loss so they always match the output representation.

Class imbalance (tumour classes are often < 1% of brain voxels) is handled in
three places:

* the Dice term is computed per class and averaged, so small classes count as
  much as large ones; ``include_background=False`` drops the dominant class
  from the Dice average;
* ``batch=True`` pools Dice statistics over the whole batch, which keeps the
  loss stable when a class is absent from an individual patch;
* label-aware patch sampling (``train.label_sampling_ratios``) over-represents
  tumour voxels in the training patches.
"""

from __future__ import annotations

import torch
from monai.losses import DiceCELoss, DiceFocalLoss, DiceLoss
from torch import nn

from .config import LossConfig
from .labels import region_targets


class CrossEntropy3D(nn.Module):
    """``nn.CrossEntropyLoss`` that accepts the (B, 1, ...) target layout used by MONAI."""

    def __init__(self, weight: torch.Tensor | None = None) -> None:
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=weight)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if target.ndim == logits.ndim:
            target = target[:, 0]
        return self.ce(logits, target.long())


class _DiceWithCEWeights(nn.Module):
    """Dice (+ weighted CE) composed explicitly so CE class weights never touch the Dice term."""

    def __init__(self, dice: nn.Module, ce: nn.Module, lambda_dice: float, lambda_ce: float) -> None:
        super().__init__()
        self.dice, self.ce = dice, ce
        self.lambda_dice, self.lambda_ce = lambda_dice, lambda_ce

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.lambda_dice * self.dice(logits, target) + self.lambda_ce * self.ce(logits, target)


class RegionDiceBCELoss(nn.Module):
    """Dice + binary cross-entropy on the three nested regions (ET, TC, WT) with sigmoid outputs.

    ``logits`` are (B, 3, ...) and ``target`` holds canonical class labels (B, 1, ...); the region masks
    are derived from the labels inside the loss, so training optimises the quantities that are evaluated.
    Dice is computed per region over the whole batch (``batch=True``), keeping it stable when a region is
    absent from a patch; ``lambda_ce`` weights the BCE term (``lambda_ce=0`` gives pure Dice).
    """

    def __init__(self, lambda_dice: float = 1.0, lambda_bce: float = 1.0) -> None:
        super().__init__()
        self.dice = DiceLoss(include_background=True, to_onehot_y=False, sigmoid=True, batch=True)
        self.bce = nn.BCEWithLogitsLoss()
        self.lambda_dice, self.lambda_bce = lambda_dice, lambda_bce

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        regions = region_targets(target)
        loss = self.lambda_dice * self.dice(logits, regions)
        if self.lambda_bce:
            loss = loss + self.lambda_bce * self.bce(logits, regions)
        return loss


def build_loss(cfg: LossConfig, num_classes: int, output: str = "softmax") -> nn.Module:
    if output == "regions":
        if cfg.name not in {"dice_ce", "dice"}:
            raise ValueError("model.output=regions supports loss.name dice_ce (Dice + BCE) or dice only")
        return RegionDiceBCELoss(cfg.lambda_dice, cfg.lambda_ce if cfg.name == "dice_ce" else 0.0)
    weights = torch.tensor(cfg.class_weights, dtype=torch.float32) if cfg.class_weights else None
    if weights is not None and len(weights) != num_classes:
        raise ValueError(f"loss.class_weights needs {num_classes} values, got {len(weights)}")
    dice_kwargs = dict(
        include_background=cfg.include_background, to_onehot_y=True, softmax=True, batch=True
    )
    if cfg.name == "dice":
        return DiceLoss(**dice_kwargs)
    if cfg.name == "ce":
        return CrossEntropy3D(weights)
    if cfg.name == "dice_ce":
        if weights is not None:
            return _DiceWithCEWeights(DiceLoss(**dice_kwargs), CrossEntropy3D(weights), cfg.lambda_dice, cfg.lambda_ce)
        return DiceCELoss(**dice_kwargs, lambda_dice=cfg.lambda_dice, lambda_ce=cfg.lambda_ce)
    if cfg.name == "dice_focal":
        if weights is not None:
            raise ValueError("loss.class_weights is not supported with dice_focal; use focal_gamma instead")
        return DiceFocalLoss(
            **dice_kwargs, gamma=cfg.focal_gamma, lambda_dice=cfg.lambda_dice, lambda_focal=cfg.lambda_ce
        )
    raise ValueError(f"Unknown loss {cfg.name!r}")
