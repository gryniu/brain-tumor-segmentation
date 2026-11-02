"""Paired comparison of two evaluations on the same cases (patient-level bootstrap).

    python scripts/compare_eval.py outputs/runs/A/eval_val outputs/runs/B/eval_val
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from brain_seg.metrics import paired_cluster_bootstrap

DICE = ["dice_ET", "dice_TC", "dice_WT", "dice_mean"]
HD95 = ["hd95_ET", "hd95_TC", "hd95_WT"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("a", help="Evaluation directory A (contains per_case_metrics.csv)")
    parser.add_argument("b", help="Evaluation directory B")
    parser.add_argument("--n-boot", type=int, default=5000)
    parser.add_argument("--output", default=None, help="Optional Markdown file to write")
    args = parser.parse_args()
    fa = pd.read_csv(Path(args.a) / "per_case_metrics.csv").set_index("case_id")
    fb = pd.read_csv(Path(args.b) / "per_case_metrics.csv").set_index("case_id")
    common = fa.index.intersection(fb.index)
    if len(common) == 0:
        raise SystemExit("The two evaluations share no cases")
    if len(common) < max(len(fa), len(fb)):
        print(f"Note: comparing on the {len(common)} cases both evaluations contain.")
    fa, fb = fa.loc[common], fb.loc[common]
    groups = fa["patient_id"].to_numpy()
    lines = [f"A = `{args.a}`", f"B = `{args.b}`", "",
             "| Metric | Mean A - B | 95% CI | P(A better) | Cases |", "|---|---|---|---|---|"]
    for column in DICE + HD95:
        if column not in fa or column not in fb:
            continue
        sign = -1.0 if column.startswith("hd95") else 1.0  # lower HD95 is better
        r = paired_cluster_bootstrap(sign * fa[column].to_numpy(), sign * fb[column].to_numpy(), groups, args.n_boot)
        lines.append(f"| {column} | {sign * r['mean_diff']:+.4f} | "
                     f"[{min(sign * r['ci_low'], sign * r['ci_high']):+.4f}, "
                     f"{max(sign * r['ci_low'], sign * r['ci_high']):+.4f}] | {r['p_a_better']:.2f} | {r['n']} |")
    lines += ["", "For HD95 the difference is in mm (negative = A has the smaller distance). 'P(A better)' is the share of "
              "bootstrap resamples (patients resampled with replacement) in which A beats B; values near 0.5 mean no clear difference."]
    text = "\n".join(lines)
    print(text)
    if args.output:
        Path(args.output).write_text(text + "\n")


if __name__ == "__main__":
    main()
