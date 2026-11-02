"""Region outputs, TTA, post-processing, bootstrap intervals and failure analysis."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import torch

from brain_seg.analysis import analyse, classify_failure, write_report
from brain_seg.config import ConfigError, LossConfig
from brain_seg.evaluate import evaluate
from brain_seg.inference import (
    labels_from_probabilities, load_checkpoint, probabilities_from_logits, run_inference,
    sliding_window_logits, tta_flip_sets,
)
from brain_seg.labels import NUM_CLASSES, REGIONS, labels_from_region_masks, region_targets
from brain_seg.losses import RegionDiceBCELoss, build_loss
from brain_seg.metrics import cluster_bootstrap_ci, compute_case_metrics, paired_cluster_bootstrap
from brain_seg.model import build_model
from brain_seg.postprocess import postprocess_labels, relabel_small_et, remove_small_components
from brain_seg.train import train
from brain_seg.tuning import crop_to_union, score_case, summarize
from conftest import tiny_config


# ------------------------------------------------------------------ regions


def test_region_targets_and_decoding_round_trip():
    labels = torch.randint(0, NUM_CLASSES, (2, 1, 6, 6, 6))
    targets = region_targets(labels)
    assert targets.shape == (2, 3, 6, 6, 6)
    assert torch.equal(labels_from_region_masks(targets > 0.5)[:, None], labels)


def test_region_decoding_enforces_nesting():
    # inconsistent regions still decode to nested labels
    masks = torch.zeros(1, 3, 1, 1, 3, dtype=torch.bool)
    masks[0, 0, 0, 0, 0] = True          # ET only
    masks[0, 1, 0, 0, 1] = True          # TC only
    masks[0, 2, 0, 0, 2] = True          # WT only
    assert labels_from_region_masks(masks).flatten().tolist() == [3, 1, 2]


def test_region_loss_prefers_correct_prediction():
    loss_fn = build_loss(LossConfig(), NUM_CLASSES, output="regions")
    assert isinstance(loss_fn, RegionDiceBCELoss)
    target = torch.randint(0, NUM_CLASSES, (2, 1, 8, 8, 8))
    perfect = (region_targets(target) * 2 - 1) * 10
    assert loss_fn(perfect, target) < loss_fn(torch.randn(2, 3, 8, 8, 8), target)
    assert torch.isfinite(loss_fn(torch.randn(2, 3, 8, 8, 8), target))


def test_regions_model_has_three_outputs_and_invalid_combinations_fail():
    cfg = tiny_config(None)
    cfg.model.output = "regions"
    assert build_model(cfg.model, 4, NUM_CLASSES)(torch.randn(1, 4, 16, 16, 16)).shape[1] == len(REGIONS)
    cfg.loss.name = "ce"
    with pytest.raises(ConfigError, match="regions"):
        cfg.validate()
    cfg.loss.name = "dice_ce"
    cfg.model.output = "nonsense"
    with pytest.raises(ConfigError, match="softmax|regions"):
        cfg.validate()


def test_decoding_helpers_for_both_outputs():
    logits = torch.randn(1, 4, 5, 5, 5)
    probs = probabilities_from_logits(logits, "softmax")
    assert torch.allclose(probs.sum(dim=1), torch.ones(1, 5, 5, 5), atol=1e-5)
    assert torch.equal(labels_from_probabilities(probs, "softmax"), logits.argmax(dim=1))
    region_logits = torch.randn(1, 3, 5, 5, 5)
    labels = labels_from_probabilities(probabilities_from_logits(region_logits, "regions"), "regions")
    assert labels.shape == (1, 5, 5, 5) and int(labels.max()) <= 3


# ------------------------------------------------------------------ TTA


def test_tta_flip_sets():
    assert tta_flip_sets("none") == [()]
    assert len(tta_flip_sets("flips3")) == 4 and len(tta_flip_sets("flips8")) == 8
    assert len(set(tta_flip_sets("flips8"))) == 8
    with pytest.raises(ValueError):
        tta_flip_sets("bogus")


@pytest.mark.parametrize("mode", ["flips3", "flips8"])
def test_tta_is_identity_for_a_flip_equivariant_model(mode):
    """A per-voxel (1x1x1) model is flip-equivariant, so averaging flipped passes must change nothing."""
    cfg = tiny_config(None)
    model = torch.nn.Conv3d(4, 4, kernel_size=1).eval()
    image = torch.randn(1, 4, 16, 16, 16)
    device = torch.device("cpu")
    plain = sliding_window_logits(model, image, cfg, device, use_amp=False)
    averaged = sliding_window_logits(model, image, cfg, device, use_amp=False, tta=mode)
    assert torch.allclose(plain, averaged, atol=1e-5)


# ------------------------------------------------------------------ post-processing


def _volume():
    labels = np.zeros((20, 20, 20), dtype=np.uint8)
    labels[2:10, 2:10, 2:10] = 2         # big tumour (oedema)
    labels[4:8, 4:8, 4:8] = 1            # core
    labels[5:7, 5:7, 5:7] = 3            # 8 voxels of ET
    labels[16, 16, 16] = 2               # isolated speck
    return labels


def test_remove_small_components_keeps_the_main_tumour():
    out = remove_small_components(_volume(), 5)
    assert out[16, 16, 16] == 0 and (out[2:10, 2:10, 2:10] > 0).all()
    assert np.array_equal(remove_small_components(_volume(), 0), _volume())  # disabled


def test_relabel_small_et_only_below_threshold():
    labels = _volume()
    assert (relabel_small_et(labels, 8) == 3).sum() == 8       # exactly at threshold: kept
    assert (relabel_small_et(labels, 9) == 3).sum() == 0       # below threshold: relabelled
    assert (relabel_small_et(labels, 9) == 1).sum() == (labels == 1).sum() + 8
    assert postprocess_labels(labels, 9, 5).dtype == np.uint8


def test_scoring_on_the_cropped_box_equals_scoring_the_full_volume():
    ref = _volume()
    pred = ref.copy()
    pred[3, 3, 3] = 0
    scores = score_case(pred, ref, et_grid=(0,), component_grid=(0,))
    full = compute_case_metrics(pred, ref)
    assert scores[(0, 0)] == pytest.approx((full["dice_ET"], full["dice_TC"], full["dice_WT"]))
    cropped_pred, cropped_ref = crop_to_union(pred, ref)
    assert cropped_pred.shape == cropped_ref.shape and cropped_pred.size < pred.size


def test_tuning_prefers_removing_a_false_positive_speck():
    ref = _volume()
    ref[16, 16, 16] = 0
    pred = _volume()                       # contains the speck as a false positive
    summary = summarize([score_case(pred, ref)])
    assert summary["best"]["min_component_voxels"] > 0
    assert summary["best"]["dice_mean"] >= summary["baseline"]["dice_mean"]


# ------------------------------------------------------------------ uncertainty and failure analysis


def test_cluster_bootstrap_ci_properties():
    rng = np.random.default_rng(1)
    values = rng.normal(0.8, 0.1, 60)
    groups = np.repeat(np.arange(30), 2)
    ci = cluster_bootstrap_ci(values, groups, n_boot=500)
    assert ci["ci_low"] < ci["mean"] < ci["ci_high"] and ci["n"] == 60 and ci["n_groups"] == 30
    assert ci == cluster_bootstrap_ci(values, groups, n_boot=500)             # seeded: reproducible
    assert np.isnan(cluster_bootstrap_ci([np.nan], [0])["mean"])               # no finite values
    # resampling patients widens the interval for correlated scans
    correlated = np.repeat(rng.normal(0.8, 0.1, 30), 2)
    wide = cluster_bootstrap_ci(correlated, groups, n_boot=500)
    narrow = cluster_bootstrap_ci(correlated, np.arange(60), n_boot=500)
    assert wide["ci_high"] - wide["ci_low"] > narrow["ci_high"] - narrow["ci_low"]


def test_paired_bootstrap_detects_a_real_difference_and_a_null_one():
    rng = np.random.default_rng(2)
    base = rng.normal(0.8, 0.1, 80)
    groups = np.arange(80)
    same = paired_cluster_bootstrap(base, base, groups, n_boot=300)
    assert same["mean_diff"] == 0 and same["ci_low"] == 0 == same["ci_high"]
    better = paired_cluster_bootstrap(base + 0.05, base, groups, n_boot=300)
    assert better["ci_low"] > 0 and better["p_a_better"] == 1.0
    with pytest.raises(ValueError):
        paired_cluster_bootstrap([1.0], [1.0, 2.0], [0])


def test_failure_classification():
    assert classify_failure(1000, 900, 0.8) == "ok"
    assert classify_failure(0, 50, 0.0).startswith("false positive")
    assert classify_failure(1000, 0, 0.0).startswith("missed")
    assert classify_failure(1000, 100, 0.2).startswith("under-segmented")
    assert classify_failure(100, 900, 0.2).startswith("over-segmented")
    assert classify_failure(1000, 1000, 0.1).startswith("misplaced")


def test_failure_analysis_report(tmp_path):
    rng = np.random.default_rng(3)
    n = 40
    frame = pd.DataFrame({"case_id": [f"c{i}" for i in range(n)], "patient_id": [f"p{i}" for i in range(n)]})
    for region in REGIONS:
        ref = rng.integers(1, 5000, n)
        dice = np.clip(rng.normal(0.8, 0.2, n), 0, 1)
        frame[f"ref_voxels_{region}"] = ref
        frame[f"pred_voxels_{region}"] = (ref * rng.uniform(0.2, 1.5, n)).astype(int)
        frame[f"dice_{region}"] = dice
    frame["dice_mean"] = frame[[f"dice_{r}" for r in REGIONS]].mean(axis=1)
    analysis = analyse(frame)
    assert set(analysis["regions"]) == set(REGIONS) and len(analysis["worst_cases"]) == 10
    write_report(analysis, tmp_path / "report.md")
    assert "Ten worst cases" in (tmp_path / "report.md").read_text()


# ------------------------------------------------------------------ end to end


def test_regions_training_evaluation_tta_postprocessing_and_prediction(brats2021_root, tmp_path):
    cfg = tiny_config(brats2021_root, output_dir=tmp_path / "out")
    cfg.run_name, cfg.model.output, cfg.train.epochs = "regions_run", "regions", 2
    run_dir = train(cfg)
    ckpt = load_checkpoint(run_dir / "checkpoints" / "best.pt")
    assert ckpt["config"]["model"]["output"] == "regions"
    summary = evaluate(
        run_dir / "checkpoints" / "best.pt", subset="test", device_name="cpu", num_workers=0,
        overrides=["inference.tta=flips3", "inference.postprocess_et_min_voxels=5",
                   "inference.postprocess_min_component_voxels=3"],
    )
    assert summary["inference"]["tta"] == "flips3" and summary["inference"]["output"] == "regions"
    assert summary["inference"]["postprocess_et_min_voxels"] == 5
    ci = summary["confidence_intervals"]["dice_mean"]
    assert ci["n_groups"] == summary["n_patients"] and ci["ci_low"] <= ci["mean"] <= ci["ci_high"]
    md = (run_dir / "eval_test" / "summary.md").read_text()
    assert "confidence intervals" in md and "TTA `flips3`" in md
    split = json.loads((run_dir / "split.json").read_text())
    case_id = split["test"][0]
    result = run_inference(run_dir / "checkpoints" / "best.pt", brats2021_root / case_id, tmp_path / "pred",
                           device_name="cpu", overrides=["inference.tta=flips3"])
    assert set(result["regions"]) == set(REGIONS)


def test_softmax_training_still_works_with_overrides(brats2021_root, tmp_path):
    cfg = tiny_config(brats2021_root, output_dir=tmp_path / "out")
    cfg.run_name, cfg.train.epochs = "softmax_run", 1
    run_dir = train(cfg)
    summary = evaluate(run_dir / "checkpoints" / "best.pt", subset="val", device_name="cpu", num_workers=0,
                       overrides=["inference.tta=flips8"])
    assert summary["inference"]["output"] == "softmax" and summary["n_cases"] > 0
