"""Draw the schematic diagrams used in the README (no patient data, no model output).

    python scripts/make_diagrams.py            # writes docs/images/pipeline.png and docs/images/regions.png

Both figures are hand-drawn schematics of the method. They are NOT MRI scans and NOT results.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch  # noqa: E402

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"  # ET, NCR, ED: same colours as brain_seg.visualize
INK, MUTED, PANEL, EDGE = "#0b0b0b", "#52514e", "#f3f2ee", "#c9c7c0"
OUT = Path(__file__).resolve().parents[1] / "docs" / "images"


def _box(ax, x, y, w, h, title, sub="", face=PANEL, edge=EDGE, title_color=INK):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                facecolor=face, edgecolor=edge, linewidth=1.4))
    ax.text(x + w / 2, y + h / 2 + (0.14 if sub else 0), title, ha="center", va="center",
            fontsize=10.5, fontweight="bold", color=title_color)
    if sub:
        ax.text(x + w / 2, y + h / 2 - 0.2, sub, ha="center", va="center", fontsize=8.2, color=MUTED)


def _arrow(ax, x0, y0, x1, y1, color=MUTED):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=14,
                                 color=color, linewidth=1.5, shrinkA=0, shrinkB=0))


def pipeline(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(13, 5.6))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 5.6)
    ax.axis("off")

    ax.text(0.1, 5.25, "TRAINING", fontsize=11, fontweight="bold", color=BLUE)
    train = [
        ("4 MRI sequences", "T1 · T1ce · T2 · FLAIR\n+ expert mask"),
        ("Preprocess", "RAS · 1 mm · brain crop\nz-score per modality"),
        ("Patch sampling", "96³ patches, centres\nbalanced over classes"),
        ("Augment", "flips · rotation · scale\nintensity · noise"),
        ("3D U-Net", "4 channels in\n4 class logits out"),
        ("Dice + CE loss", "AdamW · mixed precision\nbest checkpoint on val"),
    ]
    w, gap, y = 1.8, 0.33, 3.45
    for i, (title, sub) in enumerate(train):
        x = 0.1 + i * (w + gap)
        _box(ax, x, y, w, 1.45, title, sub, face="#e8f0fb" if i == 4 else PANEL,
             edge=BLUE if i == 4 else EDGE)
        if i:
            _arrow(ax, x - gap + 0.03, y + 0.72, x - 0.03, y + 0.72)

    ax.text(0.1, 2.75, "INFERENCE ON A NEW PATIENT", fontsize=11, fontweight="bold", color=ORANGE)
    infer = [
        ("4 MRI sequences", "one patient, any\nBraTS-style folder"),
        ("Preprocess", "same deterministic\nsteps as training"),
        ("Sliding window", "overlapping 96³ windows\nGaussian blending"),
        ("Softmax", "4 class probabilities\nper voxel"),
        ("Undo preprocessing", "back to the original grid\nthen argmax"),
        ("Mask + report", "NIfTI mask · ET/TC/WT\nvolumes (mL) · overlay"),
    ]
    y2 = 0.85
    for i, (title, sub) in enumerate(infer):
        x = 0.1 + i * (w + gap)
        _box(ax, x, y2, w, 1.45, title, sub, face="#fdf0ea" if i == 5 else PANEL,
             edge=ORANGE if i == 5 else EDGE)
        if i:
            _arrow(ax, x - gap + 0.03, y2 + 0.72, x - 0.03, y2 + 0.72)

    ax.text(0.1, 0.3, "Evaluation (once, on the held-out test set): Dice and HD95 for ET, TC and WT in the "
            "original voxel space.", fontsize=9, color=MUTED)
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def regions(path: Path) -> None:
    fig, (ax, tx) = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw={"width_ratios": [1, 1.25]})
    ax.set_xlim(-1.4, 1.4)
    ax.set_ylim(-1.6, 1.4)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.add_patch(Circle((0, 0), 1.3, facecolor="#dcdad3", edgecolor=EDGE, linewidth=1.2))
    ax.add_patch(Circle((0.05, 0.02), 0.95, facecolor=AQUA, alpha=0.75, edgecolor="white", linewidth=1.5))
    ax.add_patch(Circle((0.08, 0.04), 0.6, facecolor=ORANGE, alpha=0.9, edgecolor="white", linewidth=1.5))
    ax.add_patch(Circle((0.1, 0.06), 0.3, facecolor=BLUE, alpha=0.95, edgecolor="white", linewidth=1.5))
    ax.text(0, -1.5, "schematic, not a scan", ha="center", fontsize=8, color=MUTED, style="italic")

    tx.axis("off")
    tx.set_xlim(0, 10)
    tx.set_ylim(0, 10)
    rows = [
        (BLUE, "ET", "enhancing tumour", "label ET"),
        (ORANGE, "NCR", "necrotic / non-enhancing core", "label NCR"),
        (AQUA, "ED", "peritumoural oedema (FLAIR)", "label ED"),
    ]
    tx.text(0, 9.3, "The model predicts 4 classes per voxel", fontsize=12, fontweight="bold", color=INK)
    for i, (color, short, long, _) in enumerate(rows):
        yy = 7.9 - i * 1.15
        tx.add_patch(Circle((0.3, yy), 0.28, facecolor=color, edgecolor="none"))
        tx.text(0.85, yy, f"{short}  ", fontsize=11, fontweight="bold", va="center", color=INK)
        tx.text(2.1, yy, long, fontsize=10, va="center", color=MUTED)
    tx.text(0, 4.2, "Scored on three nested regions (BraTS protocol)", fontsize=12, fontweight="bold", color=INK)
    for i, (name, rule) in enumerate([("ET", "ET"), ("TC", "NCR ∪ ET"), ("WT", "NCR ∪ ED ∪ ET")]):
        yy = 3.0 - i * 1.0
        tx.text(0.1, yy, name, fontsize=11, fontweight="bold", va="center", color=INK)
        tx.text(1.6, yy, rule, fontsize=10, va="center", color=MUTED)
        tx.text(5.2, yy, {"ET": "bright rim after contrast", "TC": "what is typically resected",
                          "WT": "full extent incl. oedema"}[name], fontsize=9, va="center", color=MUTED)
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pipeline(OUT / "pipeline.png")
    regions(OUT / "regions.png")
    print(f"Saved {OUT / 'pipeline.png'} and {OUT / 'regions.png'}")


if __name__ == "__main__":
    main()
