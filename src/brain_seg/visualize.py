"""Figures for data inspection, training curves and qualitative/quantitative evaluation.

All figures are generated from real files on disk (MRI volumes, reference
masks, saved predictions, history/metrics files) - never from made-up data.

Display conventions:

* volumes are reoriented to RAS before slicing, so every figure has the same
  anatomical orientation regardless of how the NIfTI was stored;
* axial/coronal slices are shown in *neurological* convention (patient left on
  the image left), superior/anterior up;
* one colour per meaning everywhere: ET = blue, NCR/TC = orange, ED/WT = aqua
  (colour-vision-deficiency-safe trio), always with a legend.

Command line (``scripts/visualize.py``)::

    case        modalities + reference overlay for a dataset case
    history     loss and Dice curves of a training run
    evaluation  Dice distributions + best/median/worst test cases of an evaluation
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless: figures are written to files
import matplotlib.pyplot as plt  # noqa: E402
import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from nibabel.orientations import apply_orientation, io_orientation  # noqa: E402

from .labels import CLASS_NAMES, REGIONS, remap_labels_numpy  # noqa: E402

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
VIOLET, YELLOW = "#6250d6", "#eda100"  # error map only: false positive / false negative
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
CLASS_COLORS = {1: ORANGE, 2: AQUA, 3: BLUE}  # NCR, ED, ET
REGION_COLORS = {"ET": BLUE, "TC": ORANGE, "WT": AQUA}
PLANES = ("axial", "coronal", "sagittal")
OVERLAY_ALPHA = 0.55

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 150, "savefig.bbox": "tight",
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "axes.titlesize": 10,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "legend.frameon": False,
    "font.size": 9,
})


# --------------------------------------------------------------------------- slicing helpers


def reorient_to_ras(volume: np.ndarray, affine: np.ndarray) -> np.ndarray:
    """Reorder/flip voxel axes so that array axes point to Right, Anterior, Superior."""
    return apply_orientation(volume, io_orientation(affine))


def load_ras(path: str | Path) -> np.ndarray:
    img = nib.load(str(path))
    data = np.asanyarray(img.dataobj).astype(np.float32)
    if data.ndim == 4 and data.shape[-1] == 1:
        data = data[..., 0]
    return reorient_to_ras(data, img.affine)


def get_slice(volume: np.ndarray, plane: str, index: int) -> np.ndarray:
    """2D slice of a RAS volume, transposed for ``imshow(origin='lower')``."""
    if plane == "axial":
        return volume[:, :, index].T
    if plane == "coronal":
        return volume[:, index, :].T
    if plane == "sagittal":
        return volume[index, :, :].T
    raise ValueError(plane)


def choose_slices(mask: np.ndarray | None, shape: tuple[int, ...]) -> dict[str, int]:
    """Slices through the largest tumour cross-section per plane (volume centre if no tumour)."""
    if mask is None or not mask.any():
        return {p: shape[a] // 2 for p, a in zip(PLANES, (2, 1, 0))}
    return {
        "axial": int(mask.sum(axis=(0, 1)).argmax()),
        "coronal": int(mask.sum(axis=(0, 2)).argmax()),
        "sagittal": int(mask.sum(axis=(1, 2)).argmax()),
    }


def _label_cmap() -> ListedColormap:
    return ListedColormap(["#00000000", CLASS_COLORS[1], CLASS_COLORS[2], CLASS_COLORS[3]])


def _show_background(ax, image_slice: np.ndarray) -> None:
    nonzero = image_slice[image_slice > 0]
    vmin, vmax = (np.percentile(nonzero, [1, 99]) if nonzero.size else (0, 1))
    ax.imshow(image_slice, cmap="gray", vmin=vmin, vmax=vmax, origin="lower", interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)


def _show_labels(ax, label_slice: np.ndarray) -> None:
    masked = np.ma.masked_where(label_slice == 0, label_slice)
    ax.imshow(masked, cmap=_label_cmap(), vmin=0, vmax=3, origin="lower", alpha=OVERLAY_ALPHA,
              interpolation="nearest")


def _class_legend(fig) -> None:
    handles = [Patch(color=CLASS_COLORS[c], label=f"{CLASS_NAMES[c]}") for c in (1, 2, 3)]
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.02))


def _save(fig, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- MRI figures


def plot_case_overview(
    images: dict[str, np.ndarray], label: np.ndarray | None, out_path: str | Path, title: str
) -> Path:
    """Rows: each modality (+ reference overlay on FLAIR); columns: axial/coronal/sagittal."""
    some = next(iter(images.values()))
    slices = choose_slices(label > 0 if label is not None else None, some.shape)
    rows = list(images) + (["overlay"] if label is not None else [])
    fig, axes = plt.subplots(len(rows), 3, figsize=(8, 2.7 * len(rows)), squeeze=False)
    overlay_bg = images.get("flair", some)
    for r, row in enumerate(rows):
        for c, plane in enumerate(PLANES):
            ax = axes[r, c]
            vol = overlay_bg if row == "overlay" else images[row]
            _show_background(ax, get_slice(vol, plane, slices[plane]))
            if row == "overlay":
                _show_labels(ax, get_slice(label, plane, slices[plane]))
            if r == 0:
                ax.set_title(f"{plane} (slice {slices[plane]})")
            if c == 0:
                ax.set_ylabel("reference on FLAIR" if row == "overlay" else row.upper(), color=INK)
    fig.suptitle(title, color=INK)
    if label is not None:
        _class_legend(fig)
    return _save(fig, out_path)


def plot_comparison(
    background: np.ndarray, reference: np.ndarray, prediction: np.ndarray,
    out_path: str | Path, title: str,
) -> Path:
    """Rows: planes. Columns: MRI, reference, prediction, whole-tumour error map (TP/FP/FN)."""
    union = (reference > 0) | (prediction > 0)
    slices = choose_slices(union, background.shape)
    fig, axes = plt.subplots(3, 4, figsize=(11, 8.4), squeeze=False)
    error_cmap = ListedColormap(["#00000000", "#d9d9d9", VIOLET, YELLOW])
    for r, plane in enumerate(PLANES):
        bg = get_slice(background, plane, slices[plane])
        ref = get_slice(reference, plane, slices[plane])
        pred = get_slice(prediction, plane, slices[plane])
        errors = np.zeros_like(ref, dtype=np.uint8)
        errors[(ref > 0) & (pred > 0)] = 1
        errors[(ref == 0) & (pred > 0)] = 2
        errors[(ref > 0) & (pred == 0)] = 3
        for c, (name, overlay) in enumerate([("MRI", None), ("Reference", ref), ("Prediction", pred),
                                             ("WT errors", errors)]):
            ax = axes[r, c]
            _show_background(ax, bg)
            if name in ("Reference", "Prediction"):
                _show_labels(ax, overlay)
            elif name == "WT errors":
                ax.imshow(np.ma.masked_where(errors == 0, errors), cmap=error_cmap, vmin=0, vmax=3,
                          origin="lower", alpha=0.75, interpolation="nearest")
            if r == 0:
                ax.set_title(name)
            if c == 0:
                ax.set_ylabel(f"{plane} ({slices[plane]})", color=INK)
    handles = [Patch(color=CLASS_COLORS[c], label=CLASS_NAMES[c]) for c in (1, 2, 3)]
    handles += [Patch(color="#d9d9d9", label="WT true positive"), Patch(color=VIOLET, label="WT false positive"),
                Patch(color=YELLOW, label="WT false negative")]
    fig.legend(handles=handles, loc="lower center", ncol=6, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(title, color=INK)
    return _save(fig, out_path)


def plot_prediction(
    background_path: str | Path, label_map: np.ndarray, affine: np.ndarray, out_path: str | Path, title: str
) -> Path:
    """Predicted segmentation (canonical labels, original grid) over an MRI in three planes."""
    background = load_ras(background_path)
    labels = reorient_to_ras(label_map, affine)
    slices = choose_slices(labels > 0, labels.shape)
    fig, axes = plt.subplots(2, 3, figsize=(9, 6.4), squeeze=False)
    for c, plane in enumerate(PLANES):
        for r in range(2):
            ax = axes[r, c]
            _show_background(ax, get_slice(background, plane, slices[plane]))
            if r == 1:
                _show_labels(ax, get_slice(labels, plane, slices[plane]))
        axes[0, c].set_title(f"{plane} (slice {slices[plane]})")
    axes[0, 0].set_ylabel("MRI", color=INK)
    axes[1, 0].set_ylabel("prediction", color=INK)
    _class_legend(fig)
    fig.suptitle(title + "\nResearch output - not for clinical use", color=INK)
    return _save(fig, out_path)


# --------------------------------------------------------------------------- curves and distributions


def plot_training_history(history: list[dict[str, Any]], out_dir: str | Path) -> list[Path]:
    """Loss curves and validation Dice curves from ``history.json`` records."""
    out_dir = Path(out_dir)
    paths = []
    epochs = [h["epoch"] for h in history]
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.plot(epochs, [h["train_loss"] for h in history], color=BLUE, lw=2, label="train loss (patches)")
    val = [(h["epoch"], h["val_loss"]) for h in history if "val_loss" in h]
    if val:
        ax.plot(*zip(*val), color=ORANGE, lw=2, marker="o", ms=4, label="val loss (full volumes)")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title("Training and validation loss")
    ax.legend()
    paths.append(_save(fig, out_dir / "loss_curves.png"))

    val_rows = [h for h in history if "val_dice_mean" in h]
    if val_rows:
        fig, ax = plt.subplots(figsize=(6.4, 3.8))
        x = [h["epoch"] for h in val_rows]
        for region in REGIONS:
            ax.plot(x, [h[f"val_dice_{region}"] for h in val_rows], color=REGION_COLORS[region], lw=2,
                    marker="o", ms=4, label=f"val {region}")
        ax.plot(x, [h["val_dice_mean"] for h in val_rows], color=INK, lw=2, ls="--", label="val mean")
        if all("train_eval_dice_mean" in h for h in val_rows):
            ax.plot(x, [h["train_eval_dice_mean"] for h in val_rows], color=MUTED, lw=2, ls=":",
                    label="train-subset mean")
        best = max(val_rows, key=lambda h: h["val_dice_mean"])
        ax.axvline(best["epoch"], color=MUTED, lw=1)
        ax.annotate(f"best epoch {best['epoch']}", (best["epoch"], 0.02), color=MUTED, fontsize=8,
                    xytext=(4, 0), textcoords="offset points")
        ax.set_ylim(0, 1)
        ax.set_xlabel("epoch")
        ax.set_ylabel("Dice")
        ax.set_title("Validation Dice (full volumes)")
        ax.legend(ncol=2, loc="lower right")
        paths.append(_save(fig, out_dir / "val_dice.png"))
    return paths


def plot_dice_distribution(frame, out_path: str | Path, title: str) -> Path:
    """Box plot + individual cases for per-case Dice of ET, TC and WT."""
    fig, ax = plt.subplots(figsize=(5.6, 4))
    data = [frame[f"dice_{r}"].dropna().to_numpy() for r in REGIONS]
    box = ax.boxplot(data, widths=0.45, patch_artist=True, showfliers=False, medianprops={"color": INK})
    rng = np.random.default_rng(0)
    for i, (region, values) in enumerate(zip(REGIONS, data), start=1):
        box["boxes"][i - 1].set(facecolor=REGION_COLORS[region] + "33", edgecolor=REGION_COLORS[region])
        ax.scatter(i + rng.uniform(-0.12, 0.12, len(values)), values, s=14, color=REGION_COLORS[region],
                   edgecolor="white", linewidth=0.5, zorder=3)
        if len(values):
            ax.annotate(f"median {np.median(values):.3f}", (i, 1.02), ha="center", fontsize=8, color=MUTED)
    ax.set_xticks(range(1, len(REGIONS) + 1), [f"{r}\n(n={len(d)})" for r, d in zip(REGIONS, data)])
    ax.set_ylim(-0.02, 1.08)
    ax.set_ylabel("Dice per case")
    ax.set_title(title)
    return _save(fig, out_path)


def plot_evaluation_figures(eval_dir: str | Path, cfg, profile, cases) -> list[Path]:
    """Dice distribution + best/median/worst case comparisons from saved evaluation outputs."""
    import pandas as pd

    from .data import load_canonical_label

    eval_dir = Path(eval_dir)
    frame = pd.read_csv(eval_dir / "per_case_metrics.csv")
    summary = json.loads((eval_dir / "summary.json").read_text())
    paths = [plot_dice_distribution(frame, eval_dir / "figures" / "dice_distribution.png",
                                    f"Per-case Dice, {summary['subset']} set (n={len(frame)})")]
    if "prediction_file" not in frame or frame.empty:
        return paths
    ranked = frame.sort_values("dice_mean", ascending=False).reset_index(drop=True)
    picks = {"best": 0, "median": len(ranked) // 2, "worst": len(ranked) - 1}
    by_id = {c.case_id: c for c in cases}
    background_key = "flair" if "flair" in cfg.data.modalities else cfg.data.modalities[0]
    for tag, index in picks.items():
        row = ranked.iloc[index]
        case = by_id[row["case_id"]]
        affine = nib.load(str(case.label)).affine
        reference = reorient_to_ras(load_canonical_label(case.label, profile), affine)
        pred_img = nib.load(row["prediction_file"])
        prediction = reorient_to_ras(
            remap_labels_numpy(np.asanyarray(pred_img.dataobj), profile.label_map), pred_img.affine
        )
        title = (f"{tag.capitalize()} case by mean Dice: {case.case_id} - Dice ET {row['dice_ET']:.3f}, "
                 f"TC {row['dice_TC']:.3f}, WT {row['dice_WT']:.3f}")
        paths.append(plot_comparison(load_ras(case.images[background_key]), reference, prediction,
                                     eval_dir / "figures" / f"{tag}_case_{case.case_id}.png", title))
    return paths


# --------------------------------------------------------------------------- CLI


def main(argv: list[str] | None = None) -> None:
    from .config import config_from_dict, load_config
    from .data import load_canonical_label, prepare_cases
    from .inference import load_checkpoint
    from .utils import setup_logging

    parser = argparse.ArgumentParser(description="Generate figures from real data and run outputs.")
    sub = parser.add_subparsers(dest="command", required=True)
    p_case = sub.add_parser("case", help="Modalities and reference overlay for one dataset case")
    p_case.add_argument("--config", default="configs/default.yaml")
    p_case.add_argument("--data-root", default=None)
    p_case.add_argument("--case-id", default=None, help="Default: first case found")
    p_case.add_argument("--output", default="outputs/figures")
    p_hist = sub.add_parser("history", help="Loss and Dice curves of a training run")
    p_hist.add_argument("--run-dir", required=True)
    p_eval = sub.add_parser("evaluation", help="Figures from an evaluation directory")
    p_eval.add_argument("--eval-dir", required=True)
    p_eval.add_argument("--data-root", default=None, help="Dataset directory if it moved")
    args = parser.parse_args(argv)
    setup_logging()

    if args.command == "case":
        cfg = load_config(args.config)
        profile, cases = prepare_cases(cfg, data_root=args.data_root)
        case = next((c for c in cases if c.case_id == args.case_id), None) if args.case_id else cases[0]
        if case is None:
            raise SystemExit(f"Case {args.case_id!r} not found")
        affine = nib.load(str(case.label)).affine
        images = {m: load_ras(case.images[m]) for m in cfg.data.modalities}
        label = reorient_to_ras(load_canonical_label(case.label, profile), affine)
        path = plot_case_overview(images, label, Path(args.output) / f"{case.case_id}_overview.png",
                                  f"{case.case_id} - modalities and reference segmentation")
        print(f"Saved {path}")
    elif args.command == "history":
        run_dir = Path(args.run_dir)
        history = json.loads((run_dir / "history.json").read_text())
        for path in plot_training_history(history, run_dir / "figures"):
            print(f"Saved {path}")
    else:
        eval_dir = Path(args.eval_dir)
        summary = json.loads((eval_dir / "summary.json").read_text())
        cfg = config_from_dict(load_checkpoint(summary["checkpoint"])["config"])
        profile, cases = prepare_cases(cfg, data_root=args.data_root or summary["data_root"])
        for path in plot_evaluation_figures(eval_dir, cfg, profile, cases):
            print(f"Saved {path}")


if __name__ == "__main__":
    main()
