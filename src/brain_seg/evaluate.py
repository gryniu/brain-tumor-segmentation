"""Held-out evaluation of a trained checkpoint on full volumes in the original voxel space.

For each case of the chosen subset (default: ``test``) of the run's own split:

1. preprocess the modalities exactly as during validation (image only);
2. sliding-window inference -> softmax probabilities in the preprocessed space;
3. invert crop/resampling/reorientation back to the original voxel grid;
4. argmax -> canonical label map; load the reference mask *from disk* in its
   original grid and convert it to canonical classes;
5. derive binary ET/TC/WT masks from both and compute Dice, IoU, sensitivity,
   precision and (optionally) HD95.

Outputs in ``<run_dir>/eval_<subset>/``: ``per_case_metrics.csv``,
``summary.json``, ``summary.md``, ``predictions/*.nii.gz`` and ``figures/``.

The test subset must only be evaluated once a model has been chosen with the
validation set; it is never used during training or model selection.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from monai.data import DataLoader, Dataset, decollate_batch

from .data import load_canonical_label, load_split, prepare_cases, select_cases
from .inference import DISCLAIMER, load_trained_model, predict_case, save_label_nifti
from .labels import REGIONS, invert_label_map
from .metrics import aggregate_case_metrics, compute_case_metrics, json_safe
from .transforms import build_eval_transforms
from .utils import get_logger, resolve_amp, select_device, setup_logging

CONVENTIONS = {
    "regions": "ET = label ET; TC = NCR + ET; WT = NCR + ED + ET (canonical classes)",
    "empty_masks": (
        "Dice/IoU = 1 if reference and prediction are both empty, 0 if exactly one is empty. "
        "Sensitivity is NaN if the reference is empty; precision is NaN if the prediction is empty. "
        "HD95 = 0 if both are empty and NaN if exactly one is empty. NaNs are excluded from "
        "aggregates; 'n' reports how many values were used."
    ),
    "space": "Metrics computed in the original voxel grid of each case (predictions inverted).",
    "std": "Sample standard deviation (ddof=1).",
}


def metric_columns(include_hd95: bool) -> list[str]:
    names = ["dice", "iou", "sensitivity", "precision"] + (["hd95"] if include_hd95 else [])
    return [f"{m}_{r}" for m in names for r in REGIONS] + ["dice_mean"]


def write_markdown_summary(summary: dict[str, Any], path: Path) -> None:
    agg = summary["metrics"]
    lines = [
        f"# Evaluation: {summary['subset']} subset ({summary['n_cases']} cases)",
        "",
        f"Checkpoint: `{summary['checkpoint']}` (epoch {summary['checkpoint_epoch']})",
        "",
        "| Metric | Region | Mean | Median | Std | Min | Max | n |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for column, stats in agg.items():
        metric, _, region = column.partition("_")
        if stats["n"] == 0:
            continue
        lines.append(
            f"| {metric} | {region or 'mean'} | {stats['mean']:.4f} | {stats['median']:.4f} | "
            f"{stats['std']:.4f} | {stats['min']:.4f} | {stats['max']:.4f} | {stats['n']} |"
        )
    lines += [
        "",
        "Empty reference masks per region: "
        + ", ".join(f"{r}={summary['empty_reference_cases'][r]}" for r in REGIONS),
        "",
        f"Conventions: {CONVENTIONS['empty_masks']}",
        "",
        f"_{DISCLAIMER}_",
    ]
    path.write_text("\n".join(lines) + "\n")


def evaluate(
    checkpoint: str | Path,
    subset: str = "test",
    data_root: str | None = None,
    output_dir: str | Path | None = None,
    device_name: str = "auto",
    include_hd95: bool = True,
    save_predictions: bool = True,
    limit: int | None = None,
    num_workers: int | None = None,
) -> dict[str, Any]:
    logger = get_logger()
    checkpoint = Path(checkpoint)
    run_dir = checkpoint.resolve().parent.parent
    split_path = run_dir / "split.json"
    if not split_path.exists():
        raise FileNotFoundError(f"{split_path} not found; evaluation must use the split stored with the run")
    out_dir = Path(output_dir) if output_dir else run_dir / f"eval_{subset}"
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    setup_logging(out_dir / "evaluate.log")

    device = select_device(device_name)
    model, cfg, profile, ckpt = load_trained_model(checkpoint, device)
    use_amp = resolve_amp(cfg.train.amp, cfg.train.amp_dtype, device)
    if data_root:
        cfg.data.root = data_root
    _, cases = prepare_cases(cfg)
    split = load_split(split_path)
    if subset not in split:
        raise ValueError(f"Unknown subset {subset!r}")
    case_ids = split[subset][:limit] if limit else split[subset]
    if not case_ids:
        raise ValueError(f"Subset '{subset}' of {split_path} is empty")
    if subset == "test":
        logger.info("Evaluating the held-out TEST set. Do not use these numbers for model selection.")
    selected = select_cases(cases, case_ids)

    transforms = build_eval_transforms(cfg, profile, with_label=False)
    items = [c.to_monai(cfg.data.modalities) for c in selected]
    workers = cfg.data.num_workers if num_workers is None else num_workers
    loader = DataLoader(Dataset(items, transform=transforms), batch_size=1, num_workers=workers)
    canonical_to_raw = invert_label_map(profile.label_map)
    by_id = {c.case_id: c for c in selected}

    rows: list[dict[str, Any]] = []
    for index, batch in enumerate(loader):
        data = decollate_batch(batch)[0]
        case = by_id[data["case_id"]]
        pred = predict_case(model, data, transforms, cfg, device, use_amp)
        ref = load_canonical_label(case.label, profile)
        if pred.shape != ref.shape:
            raise RuntimeError(f"{case.case_id}: prediction {pred.shape} vs reference {ref.shape} after inversion")
        spacing = tuple(float(s) for s in nib.load(str(case.label)).header.get_zooms()[:3])
        metrics = compute_case_metrics(pred, ref, spacing, include_hd95)
        row = {"case_id": case.case_id, "patient_id": case.patient_id, **metrics}
        if save_predictions:
            path = out_dir / "predictions" / f"{case.case_id}_pred.nii.gz"
            save_label_nifti(pred, case.images[cfg.data.modalities[0]], path, canonical_to_raw)
            row["prediction_file"] = str(path)
        rows.append(row)
        logger.info(
            "[%d/%d] %s  Dice ET %.3f  TC %.3f  WT %.3f", index + 1, len(selected), case.case_id,
            metrics["dice_ET"], metrics["dice_TC"], metrics["dice_WT"],
        )

    columns = metric_columns(include_hd95)
    frame = pd.DataFrame(rows)
    frame.to_csv(out_dir / "per_case_metrics.csv", index=False)
    summary = {
        "checkpoint": str(checkpoint),
        "checkpoint_epoch": ckpt.get("epoch"),
        "best_validation_metric": ckpt.get("best_metric"),
        "subset": subset,
        "n_cases": len(rows),
        "data_root": str(cfg.data.root),
        "dataset_profile": profile.name,
        "modalities": cfg.data.modalities,
        "inference": {"overlap": cfg.inference.overlap, "patch_size": cfg.train.patch_size,
                      "blend_mode": cfg.inference.blend_mode},
        "conventions": CONVENTIONS,
        "empty_reference_cases": {r: int((frame[f"ref_voxels_{r}"] == 0).sum()) for r in REGIONS},
        "metrics": aggregate_case_metrics(rows, columns),
    }
    (out_dir / "summary.json").write_text(json.dumps(json_safe(summary), indent=2))
    write_markdown_summary(summary, out_dir / "summary.md")
    logger.info("Wrote %s", out_dir / "per_case_metrics.csv")

    from .visualize import plot_evaluation_figures

    plot_evaluation_figures(out_dir, cfg, profile, cases)
    for region in REGIONS:
        stats = summary["metrics"][f"dice_{region}"]
        logger.info("Dice %s: mean %.4f  median %.4f  std %.4f  (n=%d)", region,
                    stats["mean"], stats["median"], stats["std"], stats["n"])
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint on its run's held-out split.")
    parser.add_argument("--checkpoint", required=True, help="Usually outputs/runs/<run>/checkpoints/best.pt")
    parser.add_argument("--subset", default="test", choices=["test", "val", "train"])
    parser.add_argument("--data-root", default=None, help="Dataset directory if it moved since training")
    parser.add_argument("--output-dir", default=None, help="Default: <run_dir>/eval_<subset>")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-hd95", action="store_true", help="Skip the (slower) HD95 computation")
    parser.add_argument("--no-save-predictions", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first N cases (debugging)")
    parser.add_argument("--num-workers", type=int, default=None)
    args = parser.parse_args(argv)
    setup_logging()
    evaluate(
        args.checkpoint, args.subset, args.data_root, args.output_dir, args.device,
        not args.no_hd95, not args.no_save_predictions, args.limit, args.num_workers,
    )


if __name__ == "__main__":
    main()
