"""Canonical label definitions and tumour sub-region derivation.

Different BraTS releases encode the same tissue classes with different integer
IDs (e.g. enhancing tumour is ``4`` in BraTS 2018-2021 but ``3`` in BraTS 2023).
Everything inside this project works with a *canonical* encoding and only
converts back to the dataset's own convention when writing predictions to disk:

====  ==========  ============================================================
ID    Short name  Meaning
====  ==========  ============================================================
0     BG          Background (healthy tissue and everything outside the brain)
1     NCR         Necrotic / non-enhancing tumour core (NETC in BraTS 2023)
2     ED          Peritumoural oedema / non-enhancing FLAIR hyperintensity (SNFH)
3     ET          Gadolinium-enhancing tumour
====  ==========  ============================================================

The evaluation regions used by the BraTS challenge are *nested unions* of these
classes, not individual labels:

* ET (enhancing tumour)  = {ET}
* TC (tumour core)       = {NCR, ET}
* WT (whole tumour)      = {NCR, ED, ET}
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
REGION_DESCRIPTIONS: dict[str, str] = {
    "ET": "Enhancing tumour",
    "TC": "Tumour core (necrosis + enhancing tumour)",
    "WT": "Whole tumour (necrosis + oedema + enhancing tumour)",
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
