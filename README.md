# 3D Brain Tumour Segmentation from Multi-parametric MRI

A reproducible **PyTorch + MONAI** pipeline that trains a **3D U-Net** to segment glioma sub-regions in
multi-parametric brain MRI ([BraTS 2023](https://www.synapse.org/Synapse:syn51156910/wiki/621282) data).
Four MRI sequences go in, a voxel-wise tumour mask comes out: location, shape and volume of the
enhancing tumour, the tumour core and the whole tumour.

![Python](https://img.shields.io/badge/python-3.10%2B-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c) ![MONAI](https://img.shields.io/badge/MONAI-1.x-5b8def)

> **Not a medical device.** This is a research and educational project. It is not clinically validated and
> must not be used for diagnosis, treatment planning or any clinical decision.

## Status

| | |
|---|---|
| Code | Complete. **64 unit tests** pass (on synthetic phantoms that live only in `tests/`; the library refuses phantom data elsewhere). |
| Real data | BraTS 2023 Adult Glioma training set downloaded and validated: **1251 complete cases, 1133 patients**, 4 modalities + mask each, no geometry errors. |
| Pilot | A short real-data pilot on a Colab T4 ran end to end: the loss decreased and validation Dice rose. A pilot is a sanity check, **not a result**. |
| Full training | Not finished yet. **This README therefore reports no Dice scores**; the [Results](#results) section is filled only from a real test-set evaluation. |

## How it works

![Training and inference pipeline](docs/images/pipeline.png)

1. **Preprocess** (deterministic, identical for training and inference): reorient to RAS, resample to 1 mm,
   crop to the brain, z-score every modality over brain voxels only.
2. **Sample patches**: the volume is too large for a GPU, so the network trains on 96³ patches. Patch
   centres are drawn per class (background / necrosis / oedema / enhancing), so the tiny tumour classes
   are seen far more often than their voxel share (about 0.1 to 1 percent) would suggest.
3. **Segment with a 3D U-Net**: an encoder-decoder with skip connections outputs 4 class logits per voxel.
   Loss: Dice (tumour classes) + cross-entropy (all classes).
4. **Infer on whole volumes**: overlapping windows with Gaussian blending, then the preprocessing is
   **inverted**, so masks align voxel for voxel with the original scan.

```mermaid
flowchart LR
    I["4 × 96³ input"] --> E["Encoder<br/>16 → 32 → 64 → 128"]
    E --> B["Bottleneck<br/>256 ch, 6³"]
    B --> D["Decoder<br/>128 → 64 → 32 → 16"]
    E -. skip connections .-> D
    D --> H["1×1×1 conv<br/>4 logits"]
```

The default model has **5.65 M parameters** (instance norm, LeakyReLU). Details and the reasoning behind
each choice: [docs/methodology.md](docs/methodology.md).

### What is being segmented

![BraTS tumour classes and the three scored regions](docs/images/regions.png)

| Sequence | Shows best |
|---|---|
| T1 | anatomy; necrosis is dark |
| T1ce (after gadolinium) | **enhancing tumour** (bright rim) |
| T2 | fluid; tumour core and oedema are bright |
| FLAIR | **oedema** (bright, CSF suppressed) |

## Figures from your own data

No MRI scans or other BraTS files are included in this repository: the BraTS data-use terms forbid
redistributing them. After you download the data yourself, the project generates its figures locally:
an overview of one case (all four modalities with the expert mask), training curves, per-case Dice plots and
prediction overlays.

```bash
python scripts/visualize.py case --config configs/default.yaml --data-root "$DATA" --case-id BraTS-GLI-00000-000
```

## What is in the box

* **Verified, not assumed:** label conventions are detected from the masks, unknown label values raise an
  error, and file geometry (shape, spacing, orientation) is checked for every case before training.
* **No patient leakage:** the 70/15/15 split is made at the *patient* level (118 patients have more than one
  scan) and verified; the test set is evaluated once.
* **Clinically meaningful metrics:** Dice and HD95 for ET / TC / WT in the *original* voxel space, with explicit,
  documented conventions for empty masks.
* **Fits small hardware:** mixed precision on CUDA, patch-based training, sliding-window inference, and a compact
  **float16/uint8 disk cache** (`data.cache: npy`, about 3-4x smaller than MONAI's cache on full-size phantoms)
  for 2-CPU machines such as free Colab.
* **Budget-aware and resumable:** `train.max_hours` never starts an epoch that will not fit, validates once more
  and writes a checkpoint you can resume from.
* **Reproducible:** typed YAML configs with inheritance, every run stores its config, split, environment and
  history next to the checkpoints.
* **Tested:** unit tests for labels, metrics, data discovery, splits, caches, the model, losses and the whole
  train / evaluate / predict loop.

## Quick start

```bash
git clone <this-repo-url> && cd brain-tumor-segmentation
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"          # on a CUDA machine install PyTorch first (see docs/reference.md)
pytest -q                        # ~10 s, synthetic phantoms only
```

Get the data (needs your own Synapse account and acceptance of the terms, see
[Getting the data](docs/reference.md#getting-the-data)), then:

```bash
export DATA=/path/to/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData
python scripts/inspect_dataset.py --data-root "$DATA"                                  # validate the dataset
python scripts/smoke_test.py --config configs/local_mps.yaml --data-root "$DATA"       # one step, memory, speed
python scripts/train.py --config configs/pilot.yaml --data-root "$DATA" --run-name pilot   # short sanity run
```

| Where | Config | Notes |
|---|---|---|
| Apple Silicon Mac | `configs/local_mps.yaml` | development and pilots |
| NVIDIA GPU (rented) | `configs/cloud_cuda.yaml` | full training, see [docs/cloud_gpu.md](docs/cloud_gpu.md) |
| Free Colab T4, about 1 hour | `configs/colab_t4_1h.yaml` | subset of the data, compact cache, time cap |

Evaluate the best checkpoint on the held-out test set (once) and segment a new patient:

```bash
python scripts/evaluate.py --checkpoint outputs/runs/<run>/checkpoints/best.pt --data-root "$DATA"
python scripts/predict.py  --checkpoint outputs/runs/<run>/checkpoints/best.pt --input-dir /path/to/patient --output-dir outputs/predictions
```

## Results

> **No final results yet.** A full training run and a single test-set evaluation have not been completed, so this
> repository reports no Dice scores. This section is filled from `outputs/runs/<run>/eval_test/summary.md`
> after a real run. Numbers from phantoms, from pilots, or from other papers are never copied here.

| Run | Dice ET | Dice TC | Dice WT | HD95 ET / TC / WT (mm) |
|---|---|---|---|---|
| _to be filled from the test evaluation_ | | | | |

## Documentation

| Document | Contents |
|---|---|
| [docs/reference.md](docs/reference.md) | Full reference: dataset, method, installation, workflow, hardware settings, experiments, limitations |
| [docs/methodology.md](docs/methodology.md) | Why the pipeline is built this way; evaluation protocol |
| [docs/cloud_gpu.md](docs/cloud_gpu.md) | Renting a GPU, persistence, exact cloud commands |
| [docs/data_report.md](docs/data_report.md) | Measured facts about the downloaded BraTS data and the T4 benchmark |
| [docs/improvements_report.md](docs/improvements_report.md) | What was changed in the code review and why |

## Repository layout

```text
configs/      YAML configs (default, Mac, cloud GPU, Colab T4, pilot, experiments)
src/brain_seg/  config, labels, data, transforms, model, losses, metrics, train, evaluate, inference, visualize
scripts/      command-line entry points (inspect_dataset, smoke_test, train, evaluate, predict, visualize, ...)
tests/        pytest suite (synthetic phantoms live only here)
docs/         documentation and figures
```

## Limitations and ethics

* Trained on skull-stripped, co-registered, 1 mm BraTS data. Other scans are out of distribution and give
  unreliable masks. A single model, no ensembling or post-processing.
* A high mean Dice can hide clinically important failures (small enhancing lesions), so inspect the worst cases.
* BraTS images are de-identified patient data: respect the data-use agreement, do not redistribute the files and
  do not commit them (`.gitignore` excludes `*.nii*`, checkpoints and `outputs/`).
* Full list: [Limitations](docs/reference.md#limitations) and [Ethical considerations](docs/reference.md#ethical-considerations).

## Data attribution and citations

*"Data used in this publication were obtained as part of the Brain Tumor Segmentation (BraTS) Challenge project
through Synapse ID: syn51156910."* The data licence is CC-BY-NC 4.0 (non-commercial). Full citation list:
[docs/reference.md](docs/reference.md#citations-and-attribution). The code in this repository is by its author; choose and
add a `LICENSE` file before publishing.
