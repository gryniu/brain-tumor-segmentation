"""Failure analysis of an evaluation run from ``per_case_metrics.csv`` (numbers only, no patient images).

For each region (ET, TC, WT) this answers: which cases fail, how do they fail (the model found nothing,
found far too much, or found something in the wrong place), and does the Dice depend on the size of the
reference region? It works purely from per-case voxel counts and metrics, so it can be shared safely.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .labels import REGIONS  # noqa: E402

FAIL_DICE = 0.5  # a region with Dice below this counts as a failure


def classify_failure(ref_voxels: float, pred_voxels: float, dice: float) -> str:
    """Failure mode of one region of one case (``ok`` if Dice >= FAIL_DICE)."""
    if dice >= FAIL_DICE:
        return "ok"
    if ref_voxels == 0 and pred_voxels > 0:
        return "false positive (reference empty)"
    if pred_voxels == 0:
        return "missed (nothing predicted)"
    if pred_voxels < 0.5 * ref_voxels:
        return "under-segmented (<50% of reference volume)"
    if pred_voxels > 2.0 * ref_voxels:
        return "over-segmented (>200% of reference volume)"
    return "misplaced (similar volume, little overlap)"


def analyse(frame: pd.DataFrame) -> dict:
    """Failure table, volume dependence and worst cases from a per-case metrics table."""
    from scipy.stats import spearmanr

    result: dict = {"n_cases": len(frame), "regions": {}}
    for region in REGIONS:
        ref, pred, dice = (frame[f"{c}_{region}"].to_numpy(dtype=float) for c in ("ref_voxels", "pred_voxels", "dice"))
        modes = [classify_failure(r, p, d) for r, p, d in zip(ref, pred, dice)]
        counts = pd.Series(modes).value_counts().to_dict()
        has_ref = ref > 0
        rho = np.nan
        if has_ref.sum() > 3:
            rho = float(spearmanr(np.log10(ref[has_ref]), dice[has_ref])[0])
        bins = {}
        if has_ref.sum() >= 8:
            quartile = pd.qcut(ref[has_ref], 4, labels=["smallest 25%", "25-50%", "50-75%", "largest 25%"])
            bins = pd.Series(dice[has_ref]).groupby(quartile, observed=True).agg(["mean", "count"]).to_dict("index")
        result["regions"][region] = {
            "failure_modes": counts, "n_failures": int(sum(v for k, v in counts.items() if k != "ok")),
            "spearman_dice_vs_log_ref_volume": rho, "dice_by_reference_volume_quartile": bins,
        }
    worst = frame.sort_values("dice_mean").head(10)
    result["worst_cases"] = worst[["case_id", "patient_id", "dice_mean"]
                                  + [f"dice_{r}" for r in REGIONS]
                                  + [f"ref_voxels_{r}" for r in REGIONS]
                                  + [f"pred_voxels_{r}" for r in REGIONS]].to_dict("records")
    return result


def write_report(analysis: dict, path: Path) -> None:
    lines = [f"# Failure analysis ({analysis['n_cases']} cases)", "",
             f"A region counts as a failure when its Dice is below {FAIL_DICE}. Volumes are in voxels (1 mm³ for BraTS).", ""]
    for region, info in analysis["regions"].items():
        lines += [f"## {region}", "",
                  f"Failures: **{info['n_failures']}** of {analysis['n_cases']} cases.", ""]
        for mode, count in sorted(info["failure_modes"].items(), key=lambda kv: -kv[1]):
            if mode != "ok":
                lines.append(f"* {mode}: {count}")
        rho = info["spearman_dice_vs_log_ref_volume"]
        lines += ["", "Dice vs. size of the reference region (cases with a non-empty reference): Spearman rho = "
                  + ("n/a" if np.isnan(rho) else f"{rho:.2f}") + " (positive: small regions are harder)."]
        if info["dice_by_reference_volume_quartile"]:
            lines += ["", "| Reference volume | Mean Dice | Cases |", "|---|---|---|"]
            for name, row in info["dice_by_reference_volume_quartile"].items():
                lines.append(f"| {name} | {row['mean']:.3f} | {int(row['count'])} |")
        lines.append("")
    lines += ["## Ten worst cases (by mean Dice)", "",
              "| Case | Dice mean | Dice ET / TC / WT | Reference voxels ET / TC / WT | Predicted voxels ET / TC / WT |",
              "|---|---|---|---|---|"]
    for row in analysis["worst_cases"]:
        fmt = lambda prefix, f="{:.2f}": " / ".join(f.format(row[f"{prefix}_{r}"]) for r in REGIONS)  # noqa: E731
        lines.append(f"| {row['case_id']} | {row['dice_mean']:.3f} | {fmt('dice')} | {fmt('ref_voxels', '{:.0f}')} | "
                     f"{fmt('pred_voxels', '{:.0f}')} |")
    path.write_text("\n".join(lines) + "\n")


def plot_dice_vs_volume(frame: pd.DataFrame, path: Path) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)
    for ax, region in zip(axes, REGIONS):
        ref = frame[f"ref_voxels_{region}"].to_numpy(dtype=float)
        dice = frame[f"dice_{region}"].to_numpy(dtype=float)
        keep = ref > 0
        ax.scatter(ref[keep], dice[keep], s=14, alpha=0.7)
        ax.set_xscale("log")
        ax.set_title(f"{region} ({int((~keep).sum())} empty references not shown)")
        ax.set_xlabel("reference volume (voxels, log scale)")
        ax.axhline(FAIL_DICE, color="grey", linewidth=0.8, linestyle="--")
    axes[0].set_ylabel("Dice")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Failure analysis of an evaluation directory.")
    parser.add_argument("--eval-dir", required=True, help="Directory containing per_case_metrics.csv")
    args = parser.parse_args(argv)
    eval_dir = Path(args.eval_dir)
    frame = pd.read_csv(eval_dir / "per_case_metrics.csv")
    analysis = analyse(frame)
    report = eval_dir / "failure_analysis.md"
    write_report(analysis, report)
    (eval_dir / "figures").mkdir(exist_ok=True)
    figure = plot_dice_vs_volume(frame, eval_dir / "figures" / "dice_vs_reference_volume.png")
    print(report.read_text())
    print(f"Saved {report} and {figure}")


if __name__ == "__main__":
    main()
