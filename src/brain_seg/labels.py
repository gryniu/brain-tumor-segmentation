"""Canonical classes and BraTS evaluation regions.

Canonical IDs: 0 BG, 1 NCR (necrosis / non-enhancing core), 2 ED (oedema), 3 ET (enhancing tumour).
Datasets use their own IDs (ET is 4 in BraTS 2018-2021, 3 in 2023) and are mapped on load.
Regions are nested unions: ET = {ET}, TC = {NCR, ET}, WT = {NCR, ED, ET}.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import torch

NUM_CLASSES = 4
CLASS_NAMES: dict[int, str] = {0: "BG", 1: "NCR", 2: "ED", 3: "ET"}
REGIONS: tuple[str, ...] = ("ET", "TC", "WT")
REGION_CLASSES: dict[str, tuple[int, ...]] = {
    "ET": (3,),
    "TC": (1, 3),
    "WT": (1, 2, 3),
}


class LabelConventionError(ValueError):
    """Raised when a mask contains label values that the dataset profile does not define."""


def labels_to_regions(label_map: np.ndarray) -> dict[str, np.ndarray]:
    """Convert a canonical label map (values 0-3) into binary ET/TC/WT masks."""
    label_map = np.asarray(label_map)
    return {region: np.isin(label_map, classes) for region, classes in REGION_CLASSES.items()}


def _check_integral(values: np.ndarray, source: str) -> np.ndarray:
    rounded = np.rint(values)
    if not np.allclose(values, rounded, atol=1e-3):
        raise LabelConventionError(
            f"{source}: segmentation contains non-integer values; is this really a label mask?"
        )
    return rounded.astype(np.int64)


def remap_labels_numpy(
    label_map: np.ndarray, raw_to_canonical: Mapping[int, int], source: str = "mask"
) -> np.ndarray:
    """Map dataset-specific label IDs to canonical IDs, refusing unknown values."""
    labels = _check_integral(np.asarray(label_map), source)
    present = set(np.unique(labels).tolist())
    unknown = present - set(raw_to_canonical)
    if unknown:
        raise LabelConventionError(
            f"{source}: found label value(s) {sorted(unknown)} that are not defined by the dataset "
            f"profile (expected a subset of {sorted(raw_to_canonical)}). The dataset may use a "
            "different label convention - run scripts/inspect_dataset.py and check data.profile."
        )
    out = np.zeros(labels.shape, dtype=np.uint8)
    for raw, canonical in raw_to_canonical.items():
        out[labels == raw] = canonical
    return out


def remap_labels_tensor(
    label: torch.Tensor, raw_to_canonical: Mapping[int, int], source: str = "mask"
) -> torch.Tensor:
    """Torch equivalent of :func:`remap_labels_numpy` that preserves MONAI metadata."""
    rounded = torch.round(label)
    if not torch.allclose(label.float(), rounded.float(), atol=1e-3):
        raise LabelConventionError(
            f"{source}: segmentation contains non-integer values; is this really a label mask?"
        )
    present = set(torch.unique(rounded).long().tolist())
    unknown = present - set(raw_to_canonical)
    if unknown:
        raise LabelConventionError(
            f"{source}: found label value(s) {sorted(unknown)} that are not defined by the dataset "
            f"profile (expected a subset of {sorted(raw_to_canonical)}). The dataset may use a "
            "different label convention - run scripts/inspect_dataset.py and check data.profile."
        )
    out = torch.zeros_like(label)  # keeps MetaTensor affine/meta
    for raw, canonical in raw_to_canonical.items():
        out[rounded == raw] = canonical
    return out


def invert_label_map(raw_to_canonical: Mapping[int, int]) -> dict[int, int]:
    """Build the canonical->raw mapping used when saving predictions in the dataset convention."""
    inverse: dict[int, int] = {}
    for raw, canonical in raw_to_canonical.items():
        if canonical in inverse:
            raise ValueError(
                f"Label mapping is not invertible: canonical class {canonical} has several raw IDs "
                f"({inverse[canonical]} and {raw}). Save predictions with --label-convention canonical."
            )
        inverse[canonical] = raw
    missing = set(range(NUM_CLASSES)) - set(inverse)
    if missing:
        raise ValueError(f"Label mapping does not cover canonical classes {sorted(missing)}")
    return inverse


def region_targets(label: torch.Tensor) -> torch.Tensor:
    """Canonical labels (B, 1, ...) -> float ET/TC/WT region masks (B, 3, ...), channel order ``REGIONS``."""
    labels = label.long()
    channels = [
        torch.isin(labels, torch.tensor(REGION_CLASSES[region], device=labels.device)) for region in REGIONS
    ]
    return torch.cat(channels, dim=1).float()


def labels_from_region_masks(masks: torch.Tensor, dim: int = 1) -> torch.Tensor:
    """Boolean ET/TC/WT masks -> canonical labels, enforcing nesting (TC |= ET, WT |= TC)."""
    et, tc, wt = (m.bool() for m in masks.unbind(dim))
    tc = tc | et
    wt = wt | tc
    out = torch.zeros(et.shape, dtype=torch.long, device=et.device)
    out[wt] = 2
    out[tc] = 1
    out[et] = 3
    return out
