"""MONAI transform pipelines for training, validation and inference.

Shapes (C = number of modalities, K = 4 canonical classes):

* after loading:      image (C, X, Y, Z) float32, label (1, X, Y, Z) canonical IDs 0-3
* training samples:   image (C, *patch), label (1, *patch)
* after collation:    image (B * samples_per_volume, C, *patch)

Deterministic preprocessing (identical for train/val/test/inference):

1. Load each modality as one channel (fixed canonical order) + mask.
2. Convert raw label IDs to canonical classes (raises on unknown labels).
3. Reorient to RAS so that axes have a consistent anatomical meaning.
4. Resample to ``data.target_spacing`` (bilinear for images, nearest for masks).
5. Crop to the brain bounding box (non-zero voxels in any modality).
6. Z-score each modality using only brain voxels (background stays 0).

Steps 3-5 are invertible; :func:`brain_seg.inference.predict_case` uses MONAI's
``Invertd`` to map predictions back to the original voxel grid.
"""

from __future__ import annotations

import math
from collections.abc import Hashable, Mapping
from typing import Any

import torch
from monai.transforms import (
    Compose,
    CropForegroundd,
    EnsureTyped,
    LoadImaged,
    MapTransform,
    NormalizeIntensityd,
    Orientationd,
    RandAffined,
    RandCropByLabelClassesd,
    RandFlipd,
    RandGaussianNoised,
    RandScaleIntensityd,
    RandShiftIntensityd,
    Spacingd,
    SpatialPadd,
)

from .config import Config
from .data import DatasetProfile
from .labels import NUM_CLASSES, remap_labels_tensor


class ConvertToCanonicalLabelsd(MapTransform):
    """Map dataset-specific label IDs to canonical classes; unknown IDs raise an error."""

    def __init__(self, keys: Any, label_map: Mapping[int, int], allow_missing_keys: bool = False) -> None:
        super().__init__(keys, allow_missing_keys)
        self.label_map = dict(label_map)

    def __call__(self, data: Mapping[Hashable, Any]) -> dict[Hashable, Any]:
        d = dict(data)
        for key in self.key_iterator(d):
            source = getattr(d[key], "meta", {}).get("filename_or_obj", str(key))
            d[key] = remap_labels_tensor(d[key], self.label_map, source=str(source))
        return d


def preprocessing_transforms(cfg: Config, profile: DatasetProfile, with_label: bool) -> list:
    keys = ["image", "label"] if with_label else ["image"]
    modes = ("bilinear", "nearest") if with_label else ("bilinear",)
    transforms: list = [
        # A list of modality paths is stacked as channels: (C, H, W, D).
        LoadImaged(keys=keys, reader="NibabelReader", image_only=True, ensure_channel_first=True),
        EnsureTyped(keys="image", dtype="float32", track_meta=True),
    ]
    if with_label:
        transforms.append(ConvertToCanonicalLabelsd(keys="label", label_map=profile.label_map))
    transforms.append(Orientationd(keys=keys, axcodes="RAS", labels=None))
    if cfg.data.target_spacing is not None:
        transforms.append(Spacingd(keys=keys, pixdim=cfg.data.target_spacing, mode=modes))
    if cfg.data.crop_foreground:
        transforms.append(CropForegroundd(keys=keys, source_key="image", allow_smaller=True))
    transforms.append(NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True))
    return transforms


def augmentation_transforms(cfg: Config) -> list:
    aug = cfg.augmentation
    if not aug.enabled:
        return []
    keys = ["image", "label"]
    transforms: list = [RandFlipd(keys=keys, prob=aug.flip_prob, spatial_axis=axis) for axis in range(3)]
    if aug.affine_prob > 0:
        angle = math.radians(aug.rotate_range_deg)
        transforms.append(
            RandAffined(
                keys=keys,
                prob=aug.affine_prob,
                rotate_range=(angle, angle, angle),
                scale_range=(aug.scale_range,) * 3,
                mode=("bilinear", "nearest"),
                padding_mode="zeros",
            )
        )
    transforms += [
        RandScaleIntensityd(keys="image", factors=aug.intensity_scale_factor, prob=aug.intensity_scale_prob),
        RandShiftIntensityd(keys="image", offsets=aug.intensity_shift_offset, prob=aug.intensity_shift_prob),
        RandGaussianNoised(keys="image", prob=aug.noise_prob, std=aug.noise_std),
    ]
    return transforms


def sampling_and_augmentation_transforms(cfg: Config) -> list:
    """Random part of the training pipeline: pad, class-balanced patch crop, augmentation.

    Runs after :func:`preprocessing_transforms`. Everything before the crop may be cached because it
    is deterministic; everything from the crop on must run on every access.
    """
    keys = ["image", "label"]
    t = cfg.train
    return [
        SpatialPadd(keys=keys, spatial_size=t.patch_size),
        # Patch centres are drawn per class with the given ratios (BG, NCR, ED, ET), so small
        # tumour classes are seen far more often than their voxel share (~0.1-1%) suggests.
        # Background centres are restricted to brain voxels (image > 0). Classes absent from a
        # case are skipped by MONAI.
        RandCropByLabelClassesd(
            keys=keys,
            label_key="label",
            spatial_size=t.patch_size,
            ratios=list(t.label_sampling_ratios),
            num_classes=NUM_CLASSES,
            num_samples=t.samples_per_volume,
            image_key="image",
            image_threshold=0.0,
            warn=False,
        ),
        # Only the small patch is converted to float32 (a no-op for float32 input); the compact
        # disk cache stores float16 images, and augmentation must not run in half precision.
        EnsureTyped(keys="image", dtype=torch.float32),
        *augmentation_transforms(cfg),
        EnsureTyped(keys=keys, track_meta=False),
    ]


def build_train_transforms(cfg: Config, profile: DatasetProfile) -> Compose:
    """Preprocessing + class-balanced patch sampling + augmentation."""
    return Compose(
        [
            *preprocessing_transforms(cfg, profile, with_label=True),
            *sampling_and_augmentation_transforms(cfg),
        ]
    )


def build_train_parts(cfg: Config, profile: DatasetProfile) -> tuple[Compose, Compose]:
    """The training pipeline split into (deterministic preprocessing, random sampling/augmentation)."""
    return (
        Compose(preprocessing_transforms(cfg, profile, with_label=True)),
        Compose(sampling_and_augmentation_transforms(cfg)),
    )


def build_eval_transforms(cfg: Config, profile: DatasetProfile, with_label: bool = True) -> Compose:
    """Deterministic preprocessing for full-volume validation, evaluation and inference."""
    return Compose(preprocessing_transforms(cfg, profile, with_label=with_label))
