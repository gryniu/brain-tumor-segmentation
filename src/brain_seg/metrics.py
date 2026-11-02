"""Region-based segmentation metrics (ET / TC / WT) with explicit empty-mask conventions.

Metrics are always computed on *binary region masks* derived from canonical
label maps (see :mod:`brain_seg.labels`), never on raw label IDs.

Empty-mask conventions (documented in docs/methodology.md):

==========================  ======  ======  ===========  =========  ========
Case                        Dice    IoU     Sensitivity  Precision  HD95
==========================  ======  ======  ===========  =========  ========
reference & prediction = 0  1.0     1.0     NaN          NaN        0.0
reference = 0, pred > 0     0.0     0.0     NaN          0.0        NaN
reference > 0, pred = 0     0.0     0.0     0.0          NaN        NaN
==========================  ======  ======  ===========  =========  ========

Dice/IoU follow the BraTS challenge convention (a correctly empty prediction is
a perfect score). Sensitivity and precision are undefined (NaN) when their
denominator is zero; HD95 is undefined when exactly one mask is empty. NaNs are
excluded from aggregate statistics and the number of valid values is reported.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch

from .labels import REGIONS, labels_to_regions

NAN = float("nan")


def binary_metrics(pred: np.ndarray, ref: np.ndarray) -> dict[str, float]:
    """Overlap metrics for one pair of boolean masks."""
    pred = np.asarray(pred, dtype=bool)
    ref = np.asarray(ref, dtype=bool)
    if pred.shape != ref.shape:
        raise ValueError(f"Prediction shape {pred.shape} != reference shape {ref.shape}")
    tp = int(np.count_nonzero(pred & ref))
    fp = int(np.count_nonzero(pred & ~ref))
    fn = int(np.count_nonzero(~pred & ref))
    ref_n, pred_n = tp + fn, tp + fp
    if ref_n == 0 and pred_n == 0:
        dice = iou = 1.0
    else:
        dice = 2 * tp / (2 * tp + fp + fn)
        iou = tp / (tp + fp + fn)
    return {
        "dice": dice,
        "iou": iou,
        "sensitivity": tp / ref_n if ref_n else NAN,
        "precision": tp / pred_n if pred_n else NAN,
        "ref_voxels": float(ref_n),
        "pred_voxels": float(pred_n),
    }


def hd95(pred: np.ndarray, ref: np.ndarray, spacing: Sequence[float] | None = None) -> float:
    """95th-percentile symmetric Hausdorff distance in mm (voxel units if no spacing)."""
    pred = np.asarray(pred, dtype=bool)
    ref = np.asarray(ref, dtype=bool)
    pred_empty, ref_empty = not pred.any(), not ref.any()
    if pred_empty and ref_empty:
        return 0.0
    if pred_empty or ref_empty:
        return NAN
    from monai.metrics import compute_hausdorff_distance

    value = compute_hausdorff_distance(
        torch.from_numpy(pred)[None, None],
        torch.from_numpy(ref)[None, None],
        include_background=True,
        percentile=95,
        spacing=list(spacing) if spacing is not None else None,
    )
    return float(value.item())


def compute_case_metrics(
    pred_labels: np.ndarray,
    ref_labels: np.ndarray,
    spacing: Sequence[float] | None = None,
    include_hd95: bool = False,
) -> dict[str, float]:
    """Flat ``{metric_region: value}`` dict for one case, e.g. ``dice_ET``, ``hd95_WT``."""
    pred_labels = np.asarray(pred_labels)
    ref_labels = np.asarray(ref_labels)
    if pred_labels.shape != ref_labels.shape:
        raise ValueError(
            f"Prediction {pred_labels.shape} and reference {ref_labels.shape} are not in the same "
            "voxel grid; predictions must be inverted to the original space first."
        )
    pred_regions = labels_to_regions(pred_labels)
    ref_regions = labels_to_regions(ref_labels)
    result: dict[str, float] = {}
    for region in REGIONS:
        for name, value in binary_metrics(pred_regions[region], ref_regions[region]).items():
            result[f"{name}_{region}"] = value
        if include_hd95:
            result[f"hd95_{region}"] = hd95(pred_regions[region], ref_regions[region], spacing)
    result["dice_mean"] = float(np.mean([result[f"dice_{r}"] for r in REGIONS]))
    return result


def aggregate(values: Sequence[float]) -> dict[str, float]:
    """NaN-aware summary statistics; ``n`` counts the finite values used."""
    arr = np.asarray([v for v in values if v is not None], dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return {"mean": NAN, "median": NAN, "std": NAN, "min": NAN, "max": NAN, "n": 0}
    return {
        "mean": float(finite.mean()),
        "median": float(np.median(finite)),
        "std": float(finite.std(ddof=1)) if finite.size > 1 else 0.0,
        "min": float(finite.min()),
        "max": float(finite.max()),
        "n": int(finite.size),
    }


def aggregate_case_metrics(rows: list[dict[str, Any]], metric_names: Sequence[str]) -> dict[str, dict[str, float]]:
    return {name: aggregate([row.get(name, NAN) for row in rows]) for name in metric_names}


def json_safe(value: Any) -> Any:
    """Replace NaN/inf with None so JSON output stays standard-compliant."""
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


# --------------------------------------------------------------------------- uncertainty


def _cluster_sums(values: Sequence[float], groups: Sequence[Any]) -> tuple[np.ndarray, np.ndarray]:
    """Per-group sum and count of the finite values (groups with no finite value are dropped)."""
    values = np.asarray(values, dtype=float)
    groups = np.asarray(groups)
    keep = np.isfinite(values)
    values, groups = values[keep], groups[keep]
    _, inverse = np.unique(groups, return_inverse=True)
    sums = np.bincount(inverse, weights=values)
    counts = np.bincount(inverse).astype(float)
    return sums, counts


def cluster_bootstrap_ci(
    values: Sequence[float], groups: Sequence[Any], n_boot: int = 2000, seed: int = 0, alpha: float = 0.05
) -> dict[str, float]:
    """Mean with a percentile bootstrap confidence interval, resampling whole *patients*.

    BraTS patients can have several scans (time points), which are correlated; resampling cases
    independently would give intervals that are too narrow. Non-finite values are ignored; ``n`` is the
    number of finite cases and ``n_groups`` the number of patients they come from.
    """
    sums, counts = _cluster_sums(values, groups)
    if len(sums) == 0:
        return {"mean": NAN, "ci_low": NAN, "ci_high": NAN, "n": 0, "n_groups": 0}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(sums), size=(n_boot, len(sums)))
    means = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
    low, high = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return {
        "mean": float(sums.sum() / counts.sum()), "ci_low": float(low), "ci_high": float(high),
        "n": int(counts.sum()), "n_groups": int(len(sums)),
    }


def paired_cluster_bootstrap(
    a: Sequence[float], b: Sequence[float], groups: Sequence[Any], n_boot: int = 5000, seed: int = 0,
    alpha: float = 0.05,
) -> dict[str, float]:
    """Mean paired difference ``a - b`` per case with a patient-level bootstrap interval.

    Cases are paired by position (same cases, same order). Pairs where either value is not finite are
    dropped. ``p_a_better`` is the fraction of bootstrap resamples in which the mean difference is > 0.
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("Paired comparison needs the same cases in the same order")
    diff = a - b
    sums, counts = _cluster_sums(diff, groups)
    if len(sums) == 0:
        return {"mean_diff": NAN, "ci_low": NAN, "ci_high": NAN, "p_a_better": NAN, "n": 0}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(sums), size=(n_boot, len(sums)))
    means = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
    low, high = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return {
        "mean_diff": float(sums.sum() / counts.sum()), "ci_low": float(low), "ci_high": float(high),
        "p_a_better": float((means > 0).mean()), "n": int(counts.sum()),
    }


def confidence_intervals(frame: Any, columns: Sequence[str], n_boot: int = 2000) -> dict[str, dict[str, float]]:
    """Patient-level bootstrap CIs for metric columns of a per-case table with a ``patient_id`` column."""
    return {
        c: cluster_bootstrap_ci(frame[c].to_numpy(dtype=float), frame["patient_id"].to_numpy(), n_boot)
        for c in columns if c in frame
    }
