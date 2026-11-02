"""Tabulate training runs and show which config keys differ from a baseline run.

Controlled comparisons should change one thing at a time; this script makes
that visible by listing every config key that differs from the baseline.
It reports *validation* metrics only - test metrics are for the final model.

Example:
    python scripts/compare_runs.py outputs/runs/baseline outputs/runs/patch_128 outputs/runs/lr_1e-3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

IGNORED_KEYS = {"run_name", "output_dir", "data.root", "data.num_workers", "train.device"}


def flatten(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in d.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            out.update(flatten(value, name + "."))
        else:
            out[name] = value
    return out


def summarize(run_dir: Path) -> dict[str, Any]:
    history = json.loads((run_dir / "history.json").read_text())
    val_rows = [h for h in history if "val_dice_mean" in h]
    best = max(val_rows, key=lambda h: h["val_dice_mean"]) if val_rows else {}
    return {
        "run": run_dir.name,
        "epochs_done": history[-1]["epoch"] if history else 0,
        "best_epoch": best.get("epoch"),
        "val_dice_mean": best.get("val_dice_mean"),
        "val_dice_ET": best.get("val_dice_ET"),
        "val_dice_TC": best.get("val_dice_TC"),
        "val_dice_WT": best.get("val_dice_WT"),
        "train_eval_dice_mean": best.get("train_eval_dice_mean"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", help="Run directories; the first one is the baseline")
    parser.add_argument("--output", default=None, help="Optional Markdown file to write")
    args = parser.parse_args()
    runs = [Path(r) for r in args.runs]
    configs = [flatten(yaml.safe_load((r / "config.yaml").read_text())) for r in runs]
    baseline = configs[0]

    lines = ["| Run | Changed vs baseline | Epochs | Best epoch | Val Dice mean | ET | TC | WT | Train-subset Dice |",
             "|---|---|---|---|---|---|---|---|---|"]
    fmt = lambda v: f"{v:.4f}" if isinstance(v, float) else ("-" if v is None else str(v))  # noqa: E731
    for run, cfg in zip(runs, configs):
        changed = {k: v for k, v in cfg.items() if k not in IGNORED_KEYS and baseline.get(k) != v}
        s = summarize(run)
        changes = ", ".join(f"`{k}={v}`" for k, v in changed.items()) or "(baseline)"
        if len(changed) > 1:
            changes += " **(not a single-variable comparison)**"
        lines.append(f"| {s['run']} | {changes} | {s['epochs_done']} | {fmt(s['best_epoch'])} | "
                     f"{fmt(s['val_dice_mean'])} | {fmt(s['val_dice_ET'])} | {fmt(s['val_dice_TC'])} | "
                     f"{fmt(s['val_dice_WT'])} | {fmt(s['train_eval_dice_mean'])} |")
    table = "\n".join(lines)
    print(table)
    print("\nDifferences of a few Dice points between single runs can be seed noise; "
          "repeat promising settings with other seeds before drawing conclusions.")
    if args.output:
        Path(args.output).write_text(table + "\n")


if __name__ == "__main__":
    main()
