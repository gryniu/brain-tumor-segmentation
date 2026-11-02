# Renting an NVIDIA GPU: recommendation and exact commands

Goal: run the full baseline training on a rented CUDA GPU, keep every artifact after the machine is
stopped, and spend as little as possible. Do the free steps on your Mac first.

## 1. Which GPU?

**Recommendation: any 24 GB card (RTX 4090, RTX 3090, RTX A5000).** 3D U-Net training is limited by
GPU memory, not by double-precision speed, so a consumer/prosumer 24 GB card gives the best
price/performance. 16 GB cards (e.g. RTX 4080, T4) also work with the lighter `default.yaml`
model; below 12 GB use `local_mps.yaml` settings with smaller patches.

Price snapshot (retrieved Nov 2026, **prices change daily; check the live page before renting**):

| Provider | RTX 4090 (24 GB) | Notes |
|---|---|---|
| [RunPod](https://www.runpod.io/product/cloud-gpus) | roughly $0.34-0.89/h depending on Community vs Secure Cloud | easiest persistence: network volumes |
| [Vast.ai](https://gpuperhour.com/rent/rtx-4090) | roughly $0.34-0.72/h | marketplace of hosts; reliability varies; prefer verified hosts |

Sources: [Cloud GPU pricing comparison](https://www.spheron.network/blog/gpu-cloud-pricing-comparison-runpod-vs-vastai-2026/),
[gpuperhour.com](https://gpuperhour.com/). These are third-party snapshots, not rate cards.

How much memory the configuration needs is **measured, not assumed**: `scripts/smoke_test.py`
prints peak GPU memory for one training step. As a reference, `cloud_cuda.yaml`'s model and patch
need about 7 GB in float32 on an Apple M4 (unified memory); with mixed precision on CUDA expect less.

## 2. Persistence: what survives when the machine stops

| Storage | Survives pod stop? | Cost (RunPod docs, Nov 2026) |
|---|---|---|
| Container disk | No (erased) | free while stopped |
| Volume disk | until the pod is deleted | $0.10/GB/month running, $0.20/GB/month stopped |
| **Network volume** | **Yes, independent of pods** | **$0.07/GB/month** (first TB) |

Source: [RunPod network volumes](https://docs.runpod.io/storage/network-volumes), [pricing](https://docs.runpod.io/pods/pricing).
A network volume is deleted if your account balance cannot cover its storage, so keep a small balance.

Plan: attach a **60 GB network volume** at `/workspace` (about $4/month), and keep **everything**
there: the data and the whole `outputs/` directory (checkpoints, logs, configs, splits, metrics,
figures, predictions). The exact dataset size is printed by `download_brats.py` before it starts
downloading; it stops if the disk is too small. On providers without network volumes (e.g. Vast.ai),
copy results off the machine continuously:

```bash
bash scripts/sync_outputs.sh user@my-server:/backups/brats --loop 600      # any ssh/rsync target
bash scripts/sync_outputs.sh rclone:gdrive:brats_outputs --loop 600        # after `rclone config`
```

Checkpoints are written atomically, so a sync never copies a half-written file. A `last.pt` of the
16.6 M-parameter model (weights + optimizer state) is a few hundred MB.

## 3. Free steps first (on the Mac)

1. Download the dataset ([Reference: Getting the data](reference.md#getting-the-data)).
2. `python scripts/inspect_dataset.py --data-root "$DATA"`: must report 1251 complete cases, label values 0-3, no geometry errors.
3. `python scripts/smoke_test.py --config configs/local_mps.yaml --data-root "$DATA"`.
4. `python scripts/train.py --config configs/pilot.yaml --data-root "$DATA" --run-name pilot_mps`: loss must decrease and validation Dice must be above 0 and rising. Fix anything odd here, where it costs nothing.

## 4. On the rented GPU

Deploy a pod with: one 24 GB NVIDIA GPU, a recent **PyTorch** template/image (CUDA build of torch
preinstalled), the 60 GB network volume mounted at `/workspace`. Open a terminal:

```bash
cd /workspace
```
```bash
git clone <your-repo-url> brain-tumor-segmentation && cd brain-tumor-segmentation
```
```bash
bash scripts/cloud_setup.sh
```
Download the data **onto the volume**, using your own token (see reference.md; never paste it into chat or commit it):
```bash
export SYNAPSE_AUTH_TOKEN=<your personal access token>
```
```bash
python scripts/download_brats.py --output-dir /workspace/data
```
```bash
export DATA=/workspace/data     # the folder containing the BraTS-GLI-xxxxx-000 case folders (inspect_dataset tells you)
```
```bash
python scripts/inspect_dataset.py --data-root "$DATA" --report /workspace/outputs/dataset_report.json
```
**Measure before you commit money.** This prints peak GPU memory and measured s/step including data loading:
```bash
python scripts/smoke_test.py --config configs/cloud_cuda.yaml --data-root "$DATA" --set output_dir=/workspace/outputs
```
If peak memory is close to the card's capacity, or the output warns that the GPU waits for data,
adjust as described in reference.md "Choosing settings for your hardware", then rerun the smoke test.
Compute a budget from the measured numbers: `epochs x projected-minutes-per-epoch / 60 x hourly price`
(validation rounds add time; measure with the pilot run). Only then start the real run, inside `tmux`
so a dropped connection does not kill it, with a hard wall-clock cap that you choose:

```bash
tmux new -s train
```
```bash
python scripts/train.py --config configs/cloud_cuda.yaml --data-root "$DATA" --run-name cuda_baseline --set output_dir=/workspace/outputs train.max_hours=8
```
`train.max_hours` stops cleanly after the epoch in which the cap is reached and prints the resume command.
Detach with `Ctrl-b d`, reattach with `tmux attach -t train`, watch with `tensorboard --logdir /workspace/outputs/runs --bind_all`.

### Resume (after a stop, preemption, or the cost cap)
```bash
python scripts/train.py --resume /workspace/outputs/runs/cuda_baseline/checkpoints/last.pt --data-root "$DATA" --set train.max_hours=8
```
(`--set` after `--resume` applies on top of the config stored in the checkpoint; raise `train.epochs` to extend.)

### Evaluate the best checkpoint on the held-out test set, exactly once
```bash
python scripts/evaluate.py --checkpoint /workspace/outputs/runs/cuda_baseline/checkpoints/best.pt --data-root "$DATA"
```
Results land in `/workspace/outputs/runs/cuda_baseline/eval_test/` (`per_case_metrics.csv`, `summary.json`, `summary.md`, `predictions/`, `figures/`).

### Bring results home and stop paying
```bash
tar --exclude='*/predictions' -czf cuda_baseline.tar.gz -C /workspace/outputs/runs cuda_baseline
```
Download the tarball (or `rsync`/`scp` it), then **stop or delete the pod**. The GPU bills per
second while the pod runs, even if idle. The network volume keeps your data and outputs for
$0.07/GB/month until you delete it.

Back on the Mac, `scripts/evaluate.py` and `scripts/predict.py` work with the copied run directory
(it contains `split.json` and `config.yaml`); `scripts/visualize.py` regenerates all figures.
