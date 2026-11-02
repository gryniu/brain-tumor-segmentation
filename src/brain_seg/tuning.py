"""Tune post-processing thresholds (and optionally TTA) on the VALIDATION set.

For every validation case the model is run once (without post-processing); then every combination of
``(et_min_voxels, min_component_voxels)`` from the grids is scored against the reference mask. The setting
with the highest mean Dice (mean of ET, TC, WT) is reported next to the unprocessed baseline. The test set
is rejected on purpose: thresholds chosen on the test set would make the final result optimistic.

Metrics are computed on the bounding box of prediction and reference (plus a margin), which does not
change Dice or connected components because everything outside it is background in both.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np

from .metrics import compute_case_metrics
from .postprocess import relabel_small_et, remove_small_components

ET_GRID = (0, 50, 100, 200, 500, 1000)
COMPONENT_GRID = (0, 50, 200, 500, 1000)


def crop_to_union(pred: np.ndarray, ref: np.ndarray, margin: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Crop both label maps to the bounding box of all non-background voxels (plus ``margin``)."""
    union = (pred > 0) | (ref > 0)
    if not union.any():
        return pred[:1, :1, :1], ref[:1, :1, :1]
    slices = tuple(
        slice(max(int(idx.min()) - margin, 0), int(idx.max()) + 1 + margin)
        for idx in np.nonzero(union)
    )
    return pred[slices], ref[slices]


def score_case(
    pred: np.ndarray, ref: np.ndarray, et_grid=ET_GRID, component_grid=COMPONENT_GRID
) -> dict[tuple[int, int], tuple[float, float, float]]:
    """Dice (ET, TC, WT) of one case for every post-processing setting ``(et_min, component_min)``."""
    pred, ref = crop_to_union(np.asarray(pred), np.asarray(ref))
    scores: dict[tuple[int, int], tuple[float, float, float]] = {}
    for component_min in component_grid:
        cleaned = remove_small_components(pred, component_min)
        for et_min in et_grid:
            labels = relabel_small_et(cleaned, et_min)
            m = compute_case_metrics(labels, ref)
            scores[(et_min, component_min)] = (m["dice_ET"], m["dice_TC"], m["dice_WT"])
    return scores


def summarize(all_scores: list[dict[tuple[int, int], tuple[float, float, float]]]) -> dict[str, Any]:
    """Mean Dice per setting over cases, the baseline (0, 0) and the best setting."""
    table = []
    for key in all_scores[0]:
        per_case = np.array([s[key] for s in all_scores])  # (cases, 3)
        et, tc, wt = per_case.mean(axis=0)
        table.append({
            "et_min_voxels": key[0], "min_component_voxels": key[1],
            "dice_ET": float(et), "dice_TC": float(tc), "dice_WT": float(wt),
            "dice_mean": float(per_case.mean()),
        })
    baseline = next(r for r in table if r["et_min_voxels"] == 0 and r["min_component_voxels"] == 0)
    best = max(table, key=lambda r: r["dice_mean"])
    return {"n_cases": len(all_scores), "baseline": baseline, "best": best, "table": table,
            "gain_of_best_over_baseline": best["dice_mean"] - baseline["dice_mean"]}


def write_report(summary: dict[str, Any], path: Path, subset: str, tta: str) -> None:
    b, best = summary["baseline"], summary["best"]
    lines = [
        f"# Post-processing tuning on the {subset} set ({summary['n_cases']} cases, TTA `{tta}`)", "",
        f"Baseline (no post-processing): mean Dice {b['dice_mean']:.4f} "
        f"(ET {b['dice_ET']:.4f}, TC {b['dice_TC']:.4f}, WT {b['dice_WT']:.4f}).",
        f"Best setting: ET<{best['et_min_voxels']} voxels -> NCR, components<{best['min_component_voxels']} voxels removed: "
        f"mean Dice {best['dice_mean']:.4f} (ET {best['dice_ET']:.4f}, TC {best['dice_TC']:.4f}, WT {best['dice_WT']:.4f}); "
        f"gain {summary['gain_of_best_over_baseline']:+.4f}.", "",
        "A gain of a few thousandths on a small validation set may be noise; prefer a setting that is clearly "
        "better than the baseline, and check it with scripts/compare_eval.py before using it.", "",
        "| ET min voxels | Component min voxels | Dice ET | Dice TC | Dice WT | Dice mean |", "|---|---|---|---|---|---|",
    ]
    for r in sorted(summary["table"], key=lambda r: -r["dice_mean"]):
        lines.append(f"| {r['et_min_voxels']} | {r['min_component_voxels']} | {r['dice_ET']:.4f} | {r['dice_TC']:.4f} | "
                     f"{r['dice_WT']:.4f} | {r['dice_mean']:.4f} |")
    lines += ["", "Use the chosen values with e.g. `--set inference.postprocess_et_min_voxels=N "
              "inference.postprocess_min_component_voxels=M` in evaluate.py / predict.py."]
    path.write_text("\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> None:
    from monai.data import DataLoader, Dataset, decollate_batch

    from .config import apply_overrides
    from .data import load_canonical_label, load_split, prepare_cases, select_cases
    from .inference import load_trained_model, predict_case
    from .transforms import build_eval_transforms
    from .utils import get_logger, resolve_amp, select_device, setup_logging

    parser = argparse.ArgumentParser(description="Tune post-processing thresholds on the validation set.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--subset", default="val", choices=["val", "train"], help="The test set is not allowed")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit", type=int, default=None, help="Use only the first N cases")
    parser.add_argument("--tta", default=None, choices=["none", "flips3", "flips8"], help="Override inference.tta")
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args(argv)
    setup_logging()
    logger = get_logger()

    checkpoint = Path(args.checkpoint)
    run_dir = checkpoint.resolve().parent.parent
    device = select_device(args.device)
    model, cfg, profile, _ = load_trained_model(checkpoint, device)
    overrides = ["inference.postprocess_et_min_voxels=0", "inference.postprocess_min_component_voxels=0"]
    if args.tta:
        overrides.append(f"inference.tta={args.tta}")
    cfg = apply_overrides(cfg, overrides)
    if args.data_root:
        cfg.data.root = args.data_root
    _, cases = prepare_cases(cfg)
    ids = load_split(run_dir / "split.json")[args.subset]
    selected = select_cases(cases, ids[: args.limit] if args.limit else ids)
    transforms = build_eval_transforms(cfg, profile, with_label=False)
    loader = DataLoader(
        Dataset([c.to_monai(cfg.data.modalities) for c in selected], transform=transforms),
        batch_size=1, num_workers=args.num_workers,
    )
    by_id = {c.case_id: c for c in selected}
    use_amp = resolve_amp(cfg.train.amp, cfg.train.amp_dtype, device)

    all_scores = []
    for index, batch in enumerate(loader):
        data = decollate_batch(batch)[0]
        case = by_id[data["case_id"]]
        pred = predict_case(model, data, transforms, cfg, device, use_amp)
        all_scores.append(score_case(pred, load_canonical_label(case.label, profile)))
        logger.info("[%d/%d] %s scored", index + 1, len(selected), case.case_id)
    summary = summarize(all_scores)
    stem = f"postprocess_tuning_{args.subset}_tta-{cfg.inference.tta}"
    (run_dir / f"{stem}.json").write_text(json.dumps(summary, indent=2))
    write_report(summary, run_dir / f"{stem}.md", args.subset, cfg.inference.tta)
    print((run_dir / f"{stem}.md").read_text())


if __name__ == "__main__":
    main()
