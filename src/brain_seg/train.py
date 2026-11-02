"""Training loop with patch-based sampling, mixed precision, checkpointing and early stopping.

Model selection uses ``val_dice_mean``: the mean of ET, TC and WT Dice computed
per validation case on full volumes (sliding-window inference, BraTS empty-mask
convention) and averaged over cases. Validation runs in the preprocessed space
(RAS, target spacing, brain crop); the final test evaluation in
:mod:`brain_seg.evaluate` additionally inverts predictions to the original grid.

Run directory layout (``<output_dir>/runs/<run_name>/``)::

    config.yaml           resolved configuration of this run
    split.json            exact patient-level split used
    dataset_profile.json  label convention used
    environment.json      library versions and hardware
    train.log             full log
    history.json/.csv     per-epoch metrics
    checkpoints/last.pt   end of the latest completed epoch (resumable)
    checkpoints/best.pt   best validation Dice so far
    figures/              loss and Dice curves
    tensorboard/          TensorBoard event files
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from monai.data import DataLoader
from torch import nn

from .config import Config, apply_overrides, config_from_dict, config_to_dict, load_config, save_config
from .data import (
    CaseRecord,
    DatasetProfile,
    build_dataset,
    dataset_fingerprint,
    default_split_path,
    get_or_create_split,
    prepare_cases,
    select_cases,
)
from .inference import (
    labels_from_probabilities, load_checkpoint, probabilities_from_logits, sliding_window_logits,
)
from .labels import NUM_CLASSES, REGIONS
from .losses import build_loss
from .metrics import compute_case_metrics
from .model import build_model, count_parameters
from .transforms import build_eval_transforms, build_train_parts, build_train_transforms
from .utils import (
    amp_context,
    atomic_torch_save,
    environment_info,
    get_logger,
    resolve_amp,
    seed_everything,
    select_device,
    setup_logging,
)

SELECTION_METRIC = "val_dice_mean"


# --------------------------------------------------------------------------- building blocks


def build_optimizer(model: nn.Module, cfg: Config) -> torch.optim.Optimizer:
    t = cfg.train
    if t.optimizer == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=t.lr, weight_decay=t.weight_decay)
    return torch.optim.SGD(model.parameters(), lr=t.lr, momentum=t.momentum, nesterov=True,
                           weight_decay=t.weight_decay)


def build_scheduler(optimizer: torch.optim.Optimizer, cfg: Config) -> torch.optim.lr_scheduler.LambdaLR:
    """Per-epoch LR factor: linear warm-up, then cosine / polynomial decay (or constant)."""
    t = cfg.train
    min_ratio = min(1.0, t.min_lr / t.lr)
    warmup = max(0, min(t.warmup_epochs, t.epochs - 1))

    def factor(epoch: int) -> float:
        if epoch < warmup:
            return (epoch + 1) / (warmup + 1)
        progress = min(1.0, (epoch - warmup) / max(1, t.epochs - warmup))
        if t.scheduler == "cosine":
            return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))
        if t.scheduler == "poly":
            return max(min_ratio, (1 - progress) ** 0.9)
        return 1.0

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def preprocessing_hash(cfg: Config, profile: DatasetProfile) -> str:
    """Identify cached preprocessing results; any change in preprocessing gives a new cache dir."""
    key = json.dumps(
        {
            "modalities": cfg.data.modalities,
            "spacing": cfg.data.target_spacing,
            "crop": cfg.data.crop_foreground,
            "profile": profile.to_dict(),
        },
        sort_keys=True,
    )
    return hashlib.sha1(key.encode()).hexdigest()[:12]


def build_loaders(
    cfg: Config, profile: DatasetProfile, cases: list[CaseRecord], split: dict[str, list[str]]
) -> dict[str, DataLoader]:
    t, d = cfg.train, cfg.data
    train_ids = split["train"][: t.limit_train_cases] if t.limit_train_cases else split["train"]
    val_ids = split["val"][: t.limit_val_cases] if t.limit_val_cases else split["val"]
    if not train_ids or not val_ids:
        raise ValueError(f"Empty training ({len(train_ids)}) or validation ({len(val_ids)}) subset")
    rng = np.random.default_rng(t.seed)
    train_eval_ids = sorted(rng.permutation(train_ids)[: t.train_eval_cases].tolist())

    def items(ids: list[str]) -> list[dict[str, Any]]:
        return [c.to_monai(d.modalities) for c in select_cases(cases, ids)]

    cache_dir = Path(d.cache_dir or Path(cfg.output_dir) / "cache") / preprocessing_hash(cfg, profile)
    train_tf = build_train_transforms(cfg, profile)
    eval_tf = build_eval_transforms(cfg, profile, with_label=True)
    train_parts = build_train_parts(cfg, profile)
    eval_parts = (eval_tf, None)  # the compact cache returns full volumes for validation
    pin = torch.cuda.is_available()
    persistent = d.num_workers > 0
    loaders = {
        "train": DataLoader(
            build_dataset(items(train_ids), train_tf, d.cache, d.cache_rate, d.num_workers, cache_dir, train_parts),
            batch_size=t.batch_size, shuffle=True, num_workers=d.num_workers, pin_memory=pin,
            persistent_workers=persistent, drop_last=len(train_ids) > t.batch_size,
        ),
        "val": DataLoader(
            build_dataset(items(val_ids), eval_tf, d.cache, d.cache_rate, d.num_workers, cache_dir, eval_parts),
            batch_size=1, shuffle=False, num_workers=d.num_workers, pin_memory=pin,
            persistent_workers=persistent,
        ),
    }
    if train_eval_ids:
        loaders["train_eval"] = DataLoader(
            build_dataset(items(train_eval_ids), eval_tf, d.cache, d.cache_rate, d.num_workers, cache_dir, eval_parts),
            batch_size=1, shuffle=False, num_workers=0,
        )
    return loaders


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    cfg: Config,
    device: torch.device,
    use_amp: bool,
    epoch: int,
) -> float:
    logger = get_logger()
    t = cfg.train
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total, steps = 0.0, 0
    num_batches = len(loader)
    for step, batch in enumerate(loader):
        image = batch["image"].to(device, non_blocking=True)  # (B*S, C, *patch)
        label = batch["label"].to(device, non_blocking=True)  # (B*S, 1, *patch)
        with amp_context(device, use_amp, t.amp_dtype):
            logits = model(image)
        loss = loss_fn(logits.float(), label)
        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"Non-finite training loss ({loss.item()}) at epoch {epoch + 1}, step {step + 1}. "
                "Try a lower learning rate or disabling AMP (train.amp=false)."
            )
        scaler.scale(loss / t.accumulation_steps).backward()
        if (step + 1) % t.accumulation_steps == 0 or step + 1 == num_batches:
            if t.grad_clip_norm is not None:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), t.grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        total += loss.item()
        steps += 1
        if t.log_interval and (step + 1) % t.log_interval == 0:
            logger.info("  epoch %d step %d/%d loss %.4f", epoch + 1, step + 1, num_batches, total / steps)
    if steps == 0:
        raise RuntimeError("The training loader produced no batches")
    return total / steps


@torch.no_grad()
def validate(
    model: nn.Module, loader: DataLoader, loss_fn: nn.Module, cfg: Config, device: torch.device, use_amp: bool
) -> dict[str, float]:
    """Full-volume sliding-window validation: mean loss and per-case region Dice averaged over cases."""
    model.eval()
    losses, rows = [], []
    for batch in loader:
        logits = sliding_window_logits(model, batch["image"], cfg, device, use_amp)
        label = torch.as_tensor(batch["label"]).to(logits.device)
        losses.append(loss_fn(logits, label).item())
        probabilities = probabilities_from_logits(logits, cfg.model.output)
        pred = labels_from_probabilities(probabilities, cfg.model.output)[0].cpu().numpy()
        rows.append(compute_case_metrics(pred, label[0, 0].cpu().numpy()))
    if not rows:
        raise RuntimeError("Validation loader is empty")
    if device.type == "mps":
        torch.mps.empty_cache()  # hand full-volume buffers back to unified memory
    result = {"loss": float(np.mean(losses)), "n_cases": len(rows)}
    for key in [f"dice_{r}" for r in REGIONS] + ["dice_mean"]:
        result[key] = float(np.mean([row[key] for row in rows]))
    return result


def build_checkpoint(
    model: nn.Module, optimizer, scheduler, scaler, cfg: Config, profile: DatasetProfile,
    epoch: int, best_metric: float, best_epoch: int, history: list[dict], patience_counter: int,
) -> dict[str, Any]:
    return {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state": scaler.state_dict(),
        "epoch": epoch,  # number of completed epochs
        "best_metric": best_metric,
        "best_epoch": best_epoch,
        "selection_metric": SELECTION_METRIC,
        "patience_counter": patience_counter,
        "history": history,
        "config": config_to_dict(cfg),
        "dataset_profile": profile.to_dict(),
        "in_channels": len(cfg.data.modalities),
        "num_classes": NUM_CLASSES,
        "torch_version": str(torch.__version__),
    }


def write_history(history: list[dict], run_dir: Path) -> None:
    (run_dir / "history.json").write_text(json.dumps(history, indent=2))
    keys: list[str] = []
    for row in history:
        keys += [k for k in row if k not in keys]
    with (run_dir / "history.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        writer.writerows(history)


# --------------------------------------------------------------------------- main loop


def _new_run_dir(cfg: Config, config_path: str | Path | None) -> Path:
    name = cfg.run_name or f"{Path(config_path).stem if config_path else 'run'}_{datetime.now():%Y%m%d-%H%M%S}"
    run_dir = Path(cfg.output_dir) / "runs" / name
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(
            f"Run directory {run_dir} already exists. Choose another run_name, delete it, or resume "
            f"with --resume {run_dir / 'checkpoints' / 'last.pt'}"
        )
    return run_dir


def train(
    cfg: Config,
    data_root: str | None = None,
    resume: str | Path | None = None,
    config_path: str | Path | None = None,
) -> Path:
    """Train a model and return the run directory."""
    if data_root:
        cfg.data.root = data_root
    device = select_device(cfg.train.device)  # fail before creating any run files
    ckpt: dict[str, Any] | None = None
    if resume is not None:
        ckpt = load_checkpoint(resume)
        run_dir = Path(resume).resolve().parent.parent
    else:
        run_dir = _new_run_dir(cfg, config_path)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "figures").mkdir(exist_ok=True)
    logger = setup_logging(run_dir / "train.log")

    use_amp = resolve_amp(cfg.train.amp, cfg.train.amp_dtype, device)
    if cfg.train.amp and not use_amp:
        logger.info("Mixed precision is only used on CUDA; training in float32 on %s", device)
    seed_everything(cfg.train.seed, cfg.train.deterministic)

    profile, cases = prepare_cases(cfg)
    if resume is not None:
        split_path = run_dir / "split.json"
        from .data import load_split, verify_split

        split = load_split(split_path)
        verify_split(split, cases)
    else:
        split = get_or_create_split(cfg, cases, profile)
        shutil.copy(default_split_path(cfg), run_dir / "split.json")
    save_config(cfg, run_dir / "config.yaml")
    (run_dir / "dataset_profile.json").write_text(json.dumps(profile.to_dict(), indent=2))
    (run_dir / "dataset_info.json").write_text(json.dumps({
        "data_root": str(Path(cfg.data.root).resolve()),
        "profile": profile.name,
        "n_cases": len(cases),
        "n_patients": len({c.patient_id for c in cases}),
        "case_id_fingerprint": dataset_fingerprint(cases),
        "split_sizes": {k: len(v) for k, v in split.items()},
        "modalities": cfg.data.modalities,
    }, indent=2))

    loaders = build_loaders(cfg, profile, cases, split)
    model = build_model(cfg.model, len(cfg.data.modalities), NUM_CLASSES).to(device)
    loss_fn = build_loss(cfg.loss, NUM_CLASSES, cfg.model.output).to(device)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and cfg.train.amp_dtype == "float16")
    env = environment_info(device) | {"model_parameters": count_parameters(model)}
    (run_dir / "environment.json").write_text(json.dumps(env, indent=2))

    start_epoch, best_metric, best_epoch, patience_counter = 0, -math.inf, 0, 0
    history: list[dict] = []
    if ckpt is not None:
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        scaler.load_state_dict(ckpt["scaler_state"])
        start_epoch, best_metric = ckpt["epoch"], ckpt["best_metric"]
        best_epoch, patience_counter, history = ckpt["best_epoch"], ckpt["patience_counter"], ckpt["history"]
        logger.info("Resumed from %s after epoch %d (best %s=%.4f)", resume, start_epoch, SELECTION_METRIC, best_metric)
    if start_epoch >= cfg.train.epochs:
        logger.info("Nothing to do: %d epochs already completed (train.epochs=%d)", start_epoch, cfg.train.epochs)
        return run_dir

    t = cfg.train
    logger.info(
        "Model %s: %.2fM parameters | device %s | AMP %s | patch %s | %d patch(es)/step x %d accumulation",
        cfg.model.name, env["model_parameters"] / 1e6, device, t.amp_dtype if use_amp else "off", t.patch_size,
        t.batch_size * t.samples_per_volume, t.accumulation_steps,
    )
    from torch.utils.tensorboard import SummaryWriter

    from .visualize import plot_training_history

    writer = SummaryWriter(log_dir=str(run_dir / "tensorboard"))
    epoch = start_epoch
    started_at = time.time()
    try:
        for epoch in range(start_epoch, t.epochs):
            # Re-seeding per epoch makes the shuffling order of a resumed run match an uninterrupted
            # one. Augmentation RNGs in persistent DataLoader workers are seeded once per process,
            # so resumed runs are close to, but not bitwise identical with, uninterrupted runs.
            seed_everything(t.seed + epoch, t.deterministic)
            start = time.time()
            lr = optimizer.param_groups[0]["lr"]
            train_loss = train_one_epoch(model, loaders["train"], loss_fn, optimizer, scaler, cfg, device, use_amp, epoch)
            scheduler.step()
            train_seconds = time.time() - start
            # Time cap: do not start an epoch that would overrun it (an epoch can last 30+ min on a
            # small GPU). The last epoch is then always validated so best.pt reflects the final state.
            out_of_time = (
                t.max_hours is not None and epoch + 1 < t.epochs
                and time.time() - started_at + train_seconds >= t.max_hours * 3600
            )
            record: dict[str, Any] = {"epoch": epoch + 1, "lr": lr, "train_loss": train_loss}
            writer.add_scalar("train/loss", train_loss, epoch + 1)
            writer.add_scalar("train/lr", lr, epoch + 1)

            is_val_epoch = (epoch + 1) % t.val_interval == 0 or epoch + 1 == t.epochs or out_of_time
            improved = False
            if is_val_epoch:
                val = validate(model, loaders["val"], loss_fn, cfg, device, use_amp)
                record.update({f"val_{k}": v for k, v in val.items() if k != "n_cases"})
                if "train_eval" in loaders:
                    tr = validate(model, loaders["train_eval"], loss_fn, cfg, device, use_amp)
                    record["train_eval_dice_mean"] = tr["dice_mean"]
                    writer.add_scalar("train_eval/dice_mean", tr["dice_mean"], epoch + 1)
                for key, value in record.items():
                    if key.startswith("val_"):
                        writer.add_scalar(f"val/{key[4:]}", value, epoch + 1)
                current = record[SELECTION_METRIC]
                if current > best_metric + t.early_stopping_min_delta:
                    best_metric, best_epoch, patience_counter, improved = current, epoch + 1, 0, True
                else:
                    patience_counter += 1
            record["epoch_time_s"] = round(time.time() - start, 1)
            history.append(record)

            state = build_checkpoint(model, optimizer, scheduler, scaler, cfg, profile, epoch + 1,
                                     best_metric, best_epoch, history, patience_counter)
            atomic_torch_save(state, run_dir / "checkpoints" / "last.pt")
            if improved:
                atomic_torch_save(state, run_dir / "checkpoints" / "best.pt")
            write_history(history, run_dir)

            message = f"Epoch {epoch + 1}/{t.epochs} | lr {lr:.2e} | train loss {train_loss:.4f}"
            if is_val_epoch:
                message += (
                    f" | val loss {record['val_loss']:.4f} | Dice ET {record['val_dice_ET']:.3f} "
                    f"TC {record['val_dice_TC']:.3f} WT {record['val_dice_WT']:.3f} "
                    f"mean {record['val_dice_mean']:.3f}" + (" *best*" if improved else "")
                )
                plot_training_history(history, run_dir / "figures")
            logger.info("%s | %.0fs", message, record["epoch_time_s"])

            if (is_val_epoch and t.early_stopping_patience is not None
                    and patience_counter >= t.early_stopping_patience):
                logger.info("Early stopping: no improvement for %d validation rounds", patience_counter)
                break
            cap_reached = t.max_hours is not None and (time.time() - started_at) / 3600 >= t.max_hours
            if (out_of_time or cap_reached) and epoch + 1 < t.epochs:
                logger.warning(
                    "Stopping after epoch %d: the next epoch (~%.0fs) would exceed train.max_hours=%.2f. "
                    "Resume with:\n  python scripts/train.py --resume %s",
                    epoch + 1, train_seconds, t.max_hours, run_dir / "checkpoints" / "last.pt")
                break
    except KeyboardInterrupt:
        last = run_dir / "checkpoints" / "last.pt"
        if last.exists():
            logger.warning("Interrupted during epoch %d. Resume from the last completed epoch with:\n"
                           "  python scripts/train.py --resume %s", epoch + 1, last)
        else:
            logger.warning("Interrupted before the first epoch finished; no checkpoint was written.")
        raise
    finally:
        writer.close()
        if history:
            write_history(history, run_dir)
            plot_training_history(history, run_dir / "figures")

    if best_epoch:
        logger.info("Best %s = %.4f at epoch %d -> %s", SELECTION_METRIC, best_metric, best_epoch,
                    run_dir / "checkpoints" / "best.pt")
    else:
        logger.warning("No validation round improved on -inf; best.pt was not written")
    return run_dir


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train a 3D U-Net for BraTS brain tumour segmentation.")
    parser.add_argument("--config", default="configs/default.yaml", help="YAML config file")
    parser.add_argument("--data-root", default=None, help="Dataset directory (overrides data.root)")
    parser.add_argument("--run-name", default=None, help="Run directory name (default: <config>_<time>)")
    parser.add_argument("--resume", default=None, help="Resume from a checkpoint (uses its stored config)")
    parser.add_argument("--device", default=None, help="Override train.device (auto|cuda|mps|cpu)")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                        help="Config overrides, e.g. --set train.lr=1e-3 model.features=[8,16,32,64]")
    args = parser.parse_args(argv)

    overrides = list(args.set)
    if args.run_name:
        overrides.append(f"run_name={args.run_name}")
    if args.device:
        overrides.append(f"train.device={args.device}")
    if args.resume:
        cfg = apply_overrides(config_from_dict(load_checkpoint(args.resume)["config"]), overrides)
    else:
        cfg = load_config(args.config, overrides)
    try:
        train(cfg, data_root=args.data_root, resume=args.resume, config_path=args.config)
    except KeyboardInterrupt:
        raise SystemExit(130)


if __name__ == "__main__":
    main()
