# Failure analysis: `colab_full`, test set

Output of `scripts/analyze_failures.py` on `eval_test/per_case_metrics.csv` of the 30-epoch full-data run (194 test
cases, checkpoint of epoch 25, no TTA, no post-processing). Case IDs are public BraTS identifiers; no image data is
included. The report is generated from the metrics table only: it says *what* failed (how many voxels were predicted
against the reference), not *why*, and no scans were inspected.

A region counts as a failure when its Dice is below 0.5. Volumes are in voxels (1 mm³ for BraTS).

## Summary

| Region | Failures (Dice < 0.5) | Breakdown |
|---|---|---|
| ET | 12 of 194 | over-segmented 5, under-segmented 4, false positive on an empty reference 3 |
| TC | 7 of 194 | over-segmented 3, misplaced 2, under-segmented 2 |
| WT | 1 of 194 | under-segmented 1 |

What the numbers show:

* **Failures sit in the inner classes.** Whole tumour fails once in 194 cases; enhancing tumour 12 times. Locating the
  tumour is reliable, separating its sub-regions is where the errors are.
* **Tiny or empty ET references drive the worst cases.** Four of the ten worst cases have an enhancing reference of at
  most 22 voxels (three of them are empty). Three of the five empty ET references in the test set received a false
  positive, which scores Dice 0 for that region.
* **One case finds the tumour but not its core.** `BraTS-GLI-01483-000`: the reference tumour core has 59,424 voxels,
  the predicted one 63, while the whole-tumour volumes agree (107,864 reference, 98,065 predicted; Dice 0.87). The
  model delineated the lesion but assigned almost none of it to the core.
* **Small regions are somewhat harder, but the effect is modest.** Spearman rho between Dice and log reference volume is
  0.27 (ET), 0.28 (TC) and 0.30 (WT). Only the smallest quartile of ET stands out (mean Dice 0.794 against about 0.90
  for the other three quartiles).

The thresholds in the classification (50% and 200% of the reference volume) are heuristic, and each failure count
rests on a handful of cases, so the breakdown is descriptive rather than statistically tested.

## Full report

### ET

Failures: **12** of 194 cases.

* over-segmented (>200% of reference volume): 5
* under-segmented (<50% of reference volume): 4
* false positive (reference empty): 3

Dice vs. size of the reference region (cases with a non-empty reference): Spearman rho = 0.27.

| Reference volume | Mean Dice | Cases |
|---|---|---|
| smallest 25% | 0.794 | 48 |
| 25-50% | 0.899 | 47 |
| 50-75% | 0.899 | 47 |
| largest 25% | 0.906 | 47 |

### TC

Failures: **7** of 194 cases.

* over-segmented (>200% of reference volume): 3
* misplaced (similar volume, little overlap): 2
* under-segmented (<50% of reference volume): 2

Dice vs. size of the reference region (cases with a non-empty reference): Spearman rho = 0.28.

| Reference volume | Mean Dice | Cases |
|---|---|---|
| smallest 25% | 0.875 | 49 |
| 25-50% | 0.911 | 48 |
| 50-75% | 0.939 | 48 |
| largest 25% | 0.924 | 48 |

### WT

Failures: **1** of 194 cases.

* under-segmented (<50% of reference volume): 1

Dice vs. size of the reference region (cases with a non-empty reference): Spearman rho = 0.30.

| Reference volume | Mean Dice | Cases |
|---|---|---|
| smallest 25% | 0.895 | 49 |
| 25-50% | 0.924 | 48 |
| 50-75% | 0.936 | 48 |
| largest 25% | 0.937 | 49 |

### Ten worst cases (by mean Dice)

| Case | Dice mean | Dice ET / TC / WT | Reference voxels ET / TC / WT | Predicted voxels ET / TC / WT |
|---|---|---|---|---|
| BraTS-GLI-01483-000 | 0.290 | 0.00 / 0.00 / 0.87 | 0 / 59424 / 107864 | 63 / 63 / 98065 |
| BraTS-GLI-00412-000 | 0.331 | 0.01 / 0.01 / 0.97 | 22 / 22 / 50946 | 2355 / 3820 / 53698 |
| BraTS-GLI-01434-000 | 0.396 | 0.00 / 0.34 / 0.85 | 0 / 4487 / 30440 | 2068 / 11918 / 39852 |
| BraTS-GLI-01535-000 | 0.481 | 0.00 / 0.48 / 0.96 | 0 / 32724 / 53625 | 277 / 10561 / 52620 |
| BraTS-GLI-01428-000 | 0.513 | 0.22 / 0.48 / 0.84 | 19677 / 65516 / 139254 | 2541 / 44773 / 117453 |
| BraTS-GLI-00610-001 | 0.519 | 0.32 / 0.48 / 0.76 | 7178 / 12110 / 73176 | 23458 / 30336 / 94759 |
| BraTS-GLI-00155-000 | 0.622 | 0.49 / 0.46 / 0.91 | 1136 / 1586 / 23396 | 2603 / 2812 / 25586 |
| BraTS-GLI-01110-000 | 0.648 | 0.37 / 0.77 / 0.81 | 3345 / 5493 / 42216 | 930 / 4414 / 34696 |
| BraTS-GLI-01527-000 | 0.697 | 0.22 / 0.91 / 0.96 | 119 / 62120 / 87527 | 306 / 54887 / 89867 |
| BraTS-GLI-01314-000 | 0.707 | 0.38 / 0.79 / 0.96 | 3003 / 4866 / 70433 | 877 / 6001 / 72899 |
