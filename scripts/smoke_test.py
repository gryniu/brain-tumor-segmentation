"""Data-loading and model smoke test on the real dataset (no training run is started).

Loads a few training batches through the exact training pipeline, checks tensor
shapes, dtypes and label values, times the loader, runs one forward/backward
pass on the configured device (reporting peak GPU memory on CUDA), one
full-volume sliding-window prediction on a validation case, and (--bench-steps)
measures real training throughput so you can budget a run from measurements on
the GPU you actually rented.

Example:
    python scripts/smoke_test.py --config configs/default.yaml --data-root /data/BraTS2021 --num-batches 3
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch
from monai.data import DataLoader

from brain_seg.config import load_config
from brain_seg.data import build_dataset, make_patient_split, prepare_cases, select_cases
from brain_seg.inference import sliding_window_logits
from brain_seg.labels import CLASS_NAMES, NUM_CLASSES
from brain_seg.losses import build_loss
from brain_seg.model import build_model, count_parameters
from brain_seg.train import build_optimizer
from brain_seg.transforms import build_eval_transforms, build_train_transforms
from brain_seg.utils import amp_context, resolve_amp, seed_everything, select_device, setup_logging


def benchmark(model, loader, loss_fn, cfg, device, use_amp, steps: int, n_train_cases: int, workers: int) -> None:
    """Time real optimizer steps (data loading included) and project one epoch from the measurement."""
    t = cfg.train
    model.train()
    optimizer = build_optimizer(model, cfg)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and t.amp_dtype == "float16")
    sync = {"cuda": torch.cuda.synchronize, "mps": torch.mps.synchronize}.get(device.type, lambda: None)
    warmup, times, load_times = 3, [], []
    iterator = iter(loader)
    for step in range(steps):
        t0 = time.time()
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        load = time.time() - t0
        image, label = batch["image"].to(device), batch["label"].to(device)
        optimizer.zero_grad(set_to_none=True)
        with amp_context(device, use_amp, t.amp_dtype):
            logits = model(image)
        loss = loss_fn(logits.float(), label)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        sync()
        total = time.time() - t0
        if step >= warmup:
            times.append(total)
            load_times.append(load)
    if not times:
        print("\nBenchmark skipped: --bench-steps must be larger than 3 (warm-up)")
        return
    per_step, per_load = float(np.mean(times)), float(np.mean(load_times))
    steps_per_epoch = n_train_cases / t.batch_size  # one loaded volume per sample in the loader
    print(f"\nMeasured on {device} with {workers} loader worker(s), {len(times)} steps after warm-up:")
    print(f"  {per_step:.2f} s/step (of which waiting for data: {per_load:.2f} s) | "
          f"{t.batch_size * t.samples_per_volume} patch(es)/step")
    if per_load > 0.5 * per_step:
        print("  WARNING: the GPU is waiting for data (> 50% of the step). Raise data.num_workers, "
              "use samples_per_volume=2, or put the dataset on a local SSD.")
    print(f"  Projected epoch over {n_train_cases} training cases: ~{per_step * steps_per_epoch / 60:.1f} min "
          f"(excludes validation; a projection from {len(times)} steps, not a guarantee)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--num-batches", type=int, default=3)
    parser.add_argument("--num-workers", type=int, default=None, help="Default: data.num_workers from the config")
    parser.add_argument("--bench-steps", type=int, default=20,
                        help="Optimizer steps timed on real data (0 = skip); the first 3 are warm-up")
    parser.add_argument("--device", default=None)
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()
    setup_logging()

    overrides = list(args.set) + ([f"train.device={args.device}"] if args.device else [])
    cfg = load_config(args.config, overrides)
    seed_everything(cfg.train.seed)
    profile, cases = prepare_cases(cfg, data_root=args.data_root)
    # A throwaway split just to pick cases; the real split is created by train.py.
    split = make_patient_split(cases, cfg.data.val_fraction, cfg.data.test_fraction, cfg.data.split_seed)
    t = cfg.train
    workers = cfg.data.num_workers if args.num_workers is None else args.num_workers
    n_cases = max(1, max(args.num_batches, args.bench_steps) * t.batch_size)
    train_items = [c.to_monai(cfg.data.modalities) for c in select_cases(cases, split["train"][:n_cases])]
    loader = DataLoader(build_dataset(train_items, build_train_transforms(cfg, profile), "none"),
                        batch_size=t.batch_size, shuffle=False, num_workers=workers)

    print(f"\nProfile {profile.name} | modalities {cfg.data.modalities} | patch {t.patch_size}")
    start = time.time()
    batch = None
    for i, batch in enumerate(loader):
        image, label = batch["image"], batch["label"]
        values = torch.unique(label).long().tolist()
        fractions = {CLASS_NAMES[v]: f"{(label == v).float().mean().item():.3f}" for v in values}
        print(f"batch {i}: image {tuple(image.shape)} {image.dtype} | label {tuple(label.shape)} | "
              f"classes {fractions} | image mean {image.mean():.2f} std {image.std():.2f} | "
              f"{time.time() - start:.1f}s")
        expected = (t.batch_size * t.samples_per_volume, len(cfg.data.modalities), *t.patch_size)
        if tuple(image.shape) != expected and i < len(loader) - 1:
            raise AssertionError(f"Unexpected image batch shape {tuple(image.shape)}, expected {expected}")
        if not set(values) <= set(range(NUM_CLASSES)):
            raise AssertionError(f"Label values {values} are not canonical classes 0-3")
        if not torch.isfinite(image).all():
            raise AssertionError("Non-finite values in the image batch")
        start = time.time()
        if i + 1 >= args.num_batches:
            break
    assert batch is not None, "Loader produced no batches"

    device = select_device(t.device)
    model = build_model(cfg.model, len(cfg.data.modalities), NUM_CLASSES).to(device)
    loss_fn = build_loss(cfg.loss, NUM_CLASSES).to(device)
    use_amp = resolve_amp(t.amp, t.amp_dtype, device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    with amp_context(device, use_amp, t.amp_dtype):
        logits = model(batch["image"].to(device))
    loss = loss_fn(logits.float(), batch["label"].to(device))
    loss.backward()
    print(f"\nModel {cfg.model.name} ({count_parameters(model) / 1e6:.2f}M params) on {device}: "
          f"logits {tuple(logits.shape)}, loss {loss.item():.4f}")
    if device.type == "cuda":
        print(f"Peak GPU memory for one training step: {torch.cuda.max_memory_allocated(device) / 1024**3:.2f} GB")

    model.eval()
    val_case = select_cases(cases, split["val"][:1])[0]
    data = build_eval_transforms(cfg, profile, with_label=True)(val_case.to_monai(cfg.data.modalities))
    start = time.time()
    out = sliding_window_logits(model, data["image"].unsqueeze(0), cfg, device, use_amp)
    print(f"Full-volume inference on {val_case.case_id}: input {tuple(data['image'].shape)} -> "
          f"logits {tuple(out.shape)} in {time.time() - start:.1f}s")
    assert out.shape[2:] == data["image"].shape[1:], "Sliding-window output does not match input size"
    print(f"Predicted classes (untrained model): {np.unique(out.argmax(1).cpu().numpy()).tolist()}")
    if args.bench_steps > 0:
        benchmark(model, loader, loss_fn, cfg, device, use_amp, args.bench_steps, len(split["train"]), workers)
    print("\nSmoke test passed.")


if __name__ == "__main__":
    main()
