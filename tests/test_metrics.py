"""Dice/IoU/sensitivity on known masks, empty-mask conventions, region derivation, labels."""

from __future__ import annotations

import math

import numpy as np
import pytest

from brain_seg.labels import LabelConventionError, invert_label_map, labels_to_regions, remap_labels_numpy
from brain_seg.metrics import aggregate, binary_metrics, compute_case_metrics, hd95


def test_perfect_and_known_overlap():
    a = np.zeros((10, 10, 10), bool)
    a[2:6, 2:6, 2:6] = True  # 64 voxels
    assert binary_metrics(a, a)["dice"] == 1.0
    b = np.zeros_like(a)
    b[4:8, 2:6, 2:6] = True  # overlaps a in 32 voxels
    m = binary_metrics(b, a)
    assert m["dice"] == pytest.approx(2 * 32 / (64 + 64))
    assert m["iou"] == pytest.approx(32 / 96)
    assert m["sensitivity"] == pytest.approx(0.5)
    assert m["precision"] == pytest.approx(0.5)


def test_empty_mask_conventions():
    empty = np.zeros((5, 5, 5), bool)
    full = np.zeros_like(empty)
    full[1:3, 1:3, 1:3] = True
    both = binary_metrics(empty, empty)
    assert both["dice"] == 1.0 and both["iou"] == 1.0
    assert math.isnan(both["sensitivity"]) and math.isnan(both["precision"])
    fp = binary_metrics(full, empty)  # prediction where reference is empty
    assert fp["dice"] == 0.0 and math.isnan(fp["sensitivity"]) and fp["precision"] == 0.0
    fn = binary_metrics(empty, full)  # missed lesion
    assert fn["dice"] == 0.0 and fn["sensitivity"] == 0.0 and math.isnan(fn["precision"])
    assert hd95(empty, empty) == 0.0
    assert math.isnan(hd95(full, empty))


def test_hd95_respects_spacing():
    a = np.zeros((20, 20, 20), bool)
    a[5:10, 5:10, 5:10] = True
    b = np.roll(a, 2, axis=0)
    assert hd95(b, a, spacing=(1, 1, 1)) == pytest.approx(2.0)
    assert hd95(b, a, spacing=(2, 1, 1)) == pytest.approx(4.0)


def test_regions_are_nested_unions_of_canonical_classes():
    labels = np.array([0, 1, 2, 3])
    regions = labels_to_regions(labels)
    assert regions["ET"].tolist() == [False, False, False, True]
    assert regions["TC"].tolist() == [False, True, False, True]
    assert regions["WT"].tolist() == [False, True, True, True]


def test_case_metrics_use_regions_not_raw_labels():
    ref = np.zeros((8, 8, 8), np.uint8)
    ref[2:6, 2:6, 2:6] = 2  # edema
    ref[3:5, 3:5, 3:5] = 3  # enhancing core
    pred = ref.copy()
    pred[pred == 3] = 1  # ET mislabelled as necrosis: TC and WT still perfect, ET missed
    m = compute_case_metrics(pred, ref)
    assert m["dice_WT"] == 1.0 and m["dice_TC"] == 1.0 and m["dice_ET"] == 0.0
    assert m["dice_mean"] == pytest.approx(2 / 3)
    with pytest.raises(ValueError, match="same voxel grid"):
        compute_case_metrics(pred[:-1], ref)


def test_label_conversion_brats2021_and_2023():
    raw2021 = np.array([0, 1, 2, 4])
    assert remap_labels_numpy(raw2021, {0: 0, 1: 1, 2: 2, 4: 3}).tolist() == [0, 1, 2, 3]
    with pytest.raises(LabelConventionError, match=r"\[3\]"):
        remap_labels_numpy(np.array([0, 3]), {0: 0, 1: 1, 2: 2, 4: 3})  # 2023 mask with 2021 profile
    with pytest.raises(LabelConventionError, match="non-integer"):
        remap_labels_numpy(np.array([0.0, 0.5]), {0: 0})
    assert invert_label_map({0: 0, 1: 1, 2: 2, 4: 3}) == {0: 0, 1: 1, 2: 2, 3: 4}


def test_aggregate_ignores_nan_and_reports_n():
    stats = aggregate([1.0, 0.5, float("nan"), 0.0])
    assert stats["n"] == 3
    assert stats["mean"] == pytest.approx(0.5)
    assert stats["median"] == pytest.approx(0.5)
    assert stats["std"] == pytest.approx(0.5)
    assert aggregate([float("nan")])["n"] == 0
