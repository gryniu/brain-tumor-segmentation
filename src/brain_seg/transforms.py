"""MONAI transform pipelines for training, validation and inference.

Deterministic preprocessing (shared by all pipelines): load modalities as channels, map labels to canonical
classes, reorient to RAS, resample, crop to the brain, z-score over brain voxels. Reorientation, resampling
and cropping are inverted after inference (:func:`brain_seg.inference.predict_case`).
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
    """Random part of the training pipeline (runs after preprocessing): pad, class-balanced crop, augment."""
    keys = ["image", "label"]
    t = cfg.train
    return [
        SpatialPadd(keys=keys, spatial_size=t.patch_size),
        # patch centres drawn per class (BG, NCR, ED, ET); background centres only inside the brain
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
        # the npy cache stores float16; augment in float32
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
