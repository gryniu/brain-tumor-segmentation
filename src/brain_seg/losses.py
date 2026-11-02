"""Losses on logits (B, K, ...) and canonical label targets (B, 1, ...).

Dice is computed per class over the batch (``batch=True``) and, by default, without background.
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
    """Dice + class-weighted CE (the weights apply to CE only)."""

    def __init__(self, dice: nn.Module, ce: nn.Module, lambda_dice: float, lambda_ce: float) -> None:
        super().__init__()
        self.dice, self.ce = dice, ce
        self.lambda_dice, self.lambda_ce = lambda_dice, lambda_ce

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.lambda_dice * self.dice(logits, target) + self.lambda_ce * self.ce(logits, target)


class RegionDiceBCELoss(nn.Module):
    """Dice + BCE on sigmoid ET/TC/WT outputs (B, 3, ...); region targets are derived from the labels."""

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
