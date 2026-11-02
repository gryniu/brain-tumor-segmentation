# Methodology and engineering decisions

This document explains *why* the pipeline is built the way it is. [reference.md](reference.md) covers *how* to run it.

## 1. Data handling

### 1.1 Label conventions
BraTS releases encode the same tissue classes differently:

| Tissue | BraTS 2018–2021 | BraTS 2023 | Canonical (this code) |
|---|---|---|---|
| background | 0 | 0 | 0 |
| necrotic / non-enhancing core (NCR, NETC) | 1 | 1 | 1 |
| oedema / non-enhancing FLAIR hyperintensity (ED, SNFH) | 2 | 2 | 2 |
| enhancing tumour (ET) | **4** | **3** | 3 |

A pipeline that hard-codes `ET == 4` silently scores ET = 0 on BraTS 2023 (or, worse, trains on a
label that never appears). The code therefore (1) detects the convention from mask values,
(2) converts to canonical IDs immediately after loading, and (3) raises an error for any unexpected
label value. The detection evidence is logged and written to the dataset report.

Regions are derived from canonical classes, never from raw IDs:
`ET = {3}`, `TC = {1, 3}`, `WT = {1, 2, 3}`.

### 1.2 Validation before training
Header-only checks run on every case: all files present and unique, 3D volumes, identical shapes and
affines across modalities and mask. Training on a mis-registered or mis-assembled case would
teach the model to segment the wrong voxels without any error message, so these checks fail
loudly.

### 1.3 Spatial pipeline and why it is inverted
`RAS reorientation → 1 mm resampling → brain crop → z-score`. Reorientation makes the array axes
mean the same thing for every case (important for flips and for visualisation). Resampling makes
patch sizes correspond to physical size. Cropping removes the empty background around the head, which saves
compute and memory.

Metrics must compare voxels that refer to the same anatomical location. Predictions are therefore
inverted (crop → resampling → orientation, in reverse) to the original grid before comparison with
the mask read from disk. The inversion is tested with a deliberately awkward LPS, anisotropic
(1.0 × 1.2 × 1.5 mm) affine, and `predict_case` checks that the inverted shape equals the original
shape. During development, MONAI's `Invertd` silently skipped a non-`MetaTensor` input (with only a
warning); this guard and a regression test came out of that incident.

### 1.4 Patient-level split
Patches from the same patient are highly correlated, and BraTS 2023 also contains several timepoints of
the same patient. Splitting at patch, slice or even case level would leak information and inflate
validation scores. Patient IDs are extracted with a per-profile regex; scikit-learn's
`train_test_split` runs over the sorted patient list with a fixed seed. `verify_split` refuses any
split with overlapping cases or patients.

## 2. Model and output representation
A plain 3D U-Net (no attention, no residuals) is a strong, well-understood baseline for volumetric
segmentation: nnU-Net's results show a well-configured U-Net is competitive on BraTS. The encoder
uses max-pooling and the decoder transposed convolutions, as in the original 3D U-Net. Instance
norm and LeakyReLU are used because the batch holds only 1–2 patches.

The four canonical classes are mutually exclusive, so a 4-way softmax is the natural output. The
argmax label map is a valid BraTS segmentation by construction (every voxel gets exactly one class).
The alternative, three sigmoid outputs for the nested ET/TC/WT regions, optimises the evaluation
regions directly but needs a post-processing rule to turn overlapping regions back into labels. It
is listed as a future controlled experiment.

## 3. Training

### 3.1 Loss and class imbalance
Tumour classes are typically a few percent of brain voxels, and ET can be absent. Measures:

1. **Soft Dice** per foreground class (scale-invariant), averaged across classes.
2. **Cross-entropy** for stable per-voxel gradients and background calibration.
3. **Batch Dice** (`batch=True`): sums intersections/unions over the whole batch, so a patch without
   ET doesn't create a degenerate per-sample Dice term.
4. **Class-balanced patch sampling**: patch centres drawn equally from BG, NCR, ED and ET voxels.
   This changes the data distribution, not the loss. Full-volume validation measures the real distribution.

### 3.2 Optimisation
AdamW at 2·10⁻⁴ is robust for small batches and needs little tuning. Linear warm-up avoids large
early updates while instance-norm statistics settle. Cosine decay ends with small steps, which
usually helps the final Dice. Gradient clipping at norm 12 only acts on rare spikes.

### 3.3 Hardware strategy
| Concern | CUDA | Apple MPS | CPU |
|---|---|---|---|
| Precision | float16 autocast + GradScaler, or bfloat16 (Ampere+) | float32 | float32 |
| Memory saving | AMP, smaller patch, `stitch_on_cpu` | smaller patch / features | n/a |
| Data loading | `pin_memory`, 2–8 workers | 2 spawn workers | 0–2 workers |
| Determinism | optional (`train.deterministic`) | not guaranteed | yes |

All device-specific decisions are made in `utils.select_device` and `utils.resolve_amp`; the
training loop itself is device-agnostic. Patch-based training keeps memory independent of volume
size, and sliding-window inference keeps it bounded at test time.

### 3.4 Model selection and early stopping
Selection metric: mean over validation cases of (Dice ET + Dice TC + Dice WT) / 3, on full volumes.
The test set is never touched during training. Early stopping counts *validation rounds* without
an improvement of at least `early_stopping_min_delta`.

## 4. Evaluation protocol
* Evaluate `best.pt` once on the test split stored in the run directory.
* Report mean, median and standard deviation per region. Dice distributions in BraTS are skewed
  (a few near-zero failures), so the median is often more representative and the mean exposes
  failures. Report both.
* Empty-mask conventions (table in reference.md) follow BraTS for Dice. Undefined metrics are NaN, not 0
  or 1, and the number of valid values is reported per metric.
* HD95 is reported in millimetres using voxel spacing from the NIfTI header.

## 5. Experimental protocol

### 5.1 Baseline first
Train and validate `default.yaml` (or `cloud_cuda.yaml`) unchanged before changing anything. Every
later run is compared against that baseline, with the same split, seed and number of epochs.

### 5.2 One change at a time
Suggested order, from most to least likely to matter:
1. patch size (96³ vs 128³): context vs. memory;
2. learning rate (1·10⁻⁴, 2·10⁻⁴, 5·10⁻⁴);
3. model width (`features`);
4. loss (Dice+CE vs Dice+focal);
5. augmentation strength (on/off, then individual probabilities).

`scripts/compare_runs.py` prints the config keys that differ from the baseline and flags
comparisons that change more than one thing.

### 5.3 Noise
Two runs that differ only in seed can differ by 1–2 Dice points, mostly on ET. Claim that a change
helps only if the gain is larger than the seed-to-seed spread (run ≥ 2 seeds), or confirm it with a
paired comparison over the same validation cases.

<a id="detecting-overfitting"></a>
### 5.4 Detecting overfitting and reading validation metrics
* `train_eval_dice_mean` evaluates a fixed subset of *training* cases exactly like validation
  (full volumes, no augmentation). A widening gap between it and `val_dice_mean` is the clearest
  overfitting signal. Training loss alone isn't one, because it is computed on augmented,
  tumour-rich patches.
* Validation loss and training loss are on different data distributions (full volumes vs. sampled
  patches). Compare their trends, not their values.
* Validation Dice that is still rising at the end of training means the model is under-trained, not
  converged. A decline after a peak means overfitting: keep `best.pt`, consider stronger
  augmentation or earlier stopping.
* ET Dice fluctuates the most because of small lesions and empty-reference cases. Look at TC/WT
  trends for stability and inspect per-case ET results before drawing conclusions.

## 6. Reproducibility
Each run stores the resolved config (including seeds), the split, dataset profile, library versions
and hardware, per-epoch history, and checkpoints with optimizer/scheduler/scaler state. Checkpoints
are written atomically and loaded with `torch.load(weights_only=True)`. Seeds are re-applied at the
start of every epoch, so a resumed run follows the same shuffling order as an uninterrupted one. Augmentation
random streams in persistent DataLoader workers are seeded once per process, so resumed runs are
close to, but not bitwise identical with, uninterrupted runs.
