"""Compact ``npy`` disk cache and the wall-time budget of the training loop (synthetic phantoms, tests only)."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch
from monai.data import DataLoader

from brain_seg.data import CompactCacheDataset, build_dataset, prepare_cases
from brain_seg.inference import load_checkpoint
from brain_seg.train import train
from brain_seg.transforms import build_eval_transforms, build_train_parts
from conftest import tiny_config


def _items(root, n=3):
    cfg = tiny_config(root)
    profile, cases = prepare_cases(cfg)
    return cfg, profile, [c.to_monai(cfg.data.modalities) for c in cases[:n]]


def test_npy_cache_matches_uncached_preprocessing(brats2021_root, tmp_path):
    cfg, profile, items = _items(brats2021_root)
    eval_tf = build_eval_transforms(cfg, profile)
    dataset = CompactCacheDataset(items, eval_tf, None, tmp_path / "cache")
    for index, item in enumerate(items):
        reference = eval_tf(dict(item))
        cached = dataset[index]
        assert cached["image"].dtype == torch.float32 and cached["label"].dtype == torch.float32
        assert tuple(cached["image"].shape) == tuple(reference["image"].shape)
        assert torch.equal(cached["label"], torch.as_tensor(reference["label"]))  # uint8 storage is lossless
        # float16 has a relative error of ~5e-4: far below anything that matters for z-scored MRI
        assert torch.allclose(cached["image"], torch.as_tensor(reference["image"]), rtol=2e-3, atol=2e-3)
        assert cached["case_id"] == item["case_id"]
    assert len(list((tmp_path / "cache").glob("*_image.npy"))) == len(items)


def test_npy_cache_is_built_once_and_is_smaller(brats2021_root, tmp_path):
    cfg, profile, items = _items(brats2021_root, n=2)
    calls = {"n": 0}

    class Counting:
        def __init__(self, inner):
            self.inner = inner

        def __call__(self, data):
            calls["n"] += 1
            return self.inner(data)

    dataset = CompactCacheDataset(items, Counting(build_eval_transforms(cfg, profile)), None, tmp_path / "cache")
    for _ in range(3):
        for index in range(len(items)):
            dataset[index]
    assert calls["n"] == len(items)  # preprocessing ran once per case, never again
    image = np.load(next((tmp_path / "cache").glob("*_image.npy")), mmap_mode="r")
    label = np.load(next((tmp_path / "cache").glob("*_label.npy")), mmap_mode="r")
    assert image.dtype == np.float16 and label.dtype == np.uint8
    assert image.nbytes + label.nbytes < 0.5 * (image.size + label.size) * 4  # vs float32 image + float32 mask


def test_npy_cache_rebuilds_corrupt_entries(brats2021_root, tmp_path):
    cfg, profile, items = _items(brats2021_root, n=1)
    dataset = CompactCacheDataset(items, build_eval_transforms(cfg, profile), None, tmp_path / "cache")
    first = dataset[0]["image"].clone()
    image_file = next((tmp_path / "cache").glob("*_image.npy"))
    image_file.write_bytes(image_file.read_bytes()[:50])  # simulate an interrupted write / full disk
    assert torch.equal(dataset[0]["image"], first)
    assert not list((tmp_path / "cache").glob("*.tmp"))


def test_npy_training_samples_match_uncached_pipeline_shapes(brats2021_root, tmp_path):
    cfg, profile, items = _items(brats2021_root)
    cfg.train.samples_per_volume = 2
    dataset = build_dataset(items, None, "npy", cache_dir=tmp_path / "c", split_transforms=build_train_parts(cfg, profile))
    batch = next(iter(DataLoader(dataset, batch_size=2, num_workers=0)))
    assert tuple(batch["image"].shape) == (4, 4, 16, 16, 16)  # (batch * samples_per_volume, C, *patch)
    assert batch["image"].dtype == torch.float32 and batch["label"].dtype == torch.float32
    assert set(batch["label"].unique().tolist()) <= {0.0, 1.0, 2.0, 3.0}


def test_npy_cache_requires_split_transforms(tmp_path):
    from brain_seg.config import ConfigError

    with pytest.raises(ConfigError, match="npy"):
        build_dataset([], None, "npy", cache_dir=tmp_path)


def test_training_end_to_end_with_npy_cache(brats2021_root, tmp_path):
    cfg = tiny_config(brats2021_root, output_dir=tmp_path / "out")
    cfg.run_name, cfg.data.cache, cfg.train.epochs = "npy_run", "npy", 2
    cfg.data.cache_dir = str(tmp_path / "fast_local_cache")
    run_dir = train(cfg)
    history = json.loads((run_dir / "history.json").read_text())
    assert [h["epoch"] for h in history] == [1, 2] and all(np.isfinite(h["train_loss"]) for h in history)
    assert (run_dir / "checkpoints" / "best.pt").exists()
    assert list((tmp_path / "fast_local_cache").rglob("*_image.npy"))  # cache went to data.cache_dir ...
    assert not (tmp_path / "out" / "cache").exists()  # ... and not next to the outputs


def test_time_cap_never_starts_an_epoch_that_does_not_fit_and_validates_last_epoch(brats2021_root, tmp_path):
    cfg = tiny_config(brats2021_root, output_dir=tmp_path / "out")
    cfg.run_name, cfg.train.epochs, cfg.train.max_hours = "budget", 6, 1e-9
    cfg.train.val_interval = 100  # periodic validation would never run: the cap must trigger the final one
    run_dir = train(cfg)
    history = json.loads((run_dir / "history.json").read_text())
    assert len(history) == 1
    assert "val_dice_mean" in history[0]  # validated because training stopped for time
    assert (run_dir / "checkpoints" / "best.pt").exists()
    assert load_checkpoint(run_dir / "checkpoints" / "last.pt")["epoch"] == 1


def test_time_cap_does_not_interfere_when_budget_is_ample(brats2021_root, tmp_path):
    cfg = tiny_config(brats2021_root, output_dir=tmp_path / "out")
    cfg.run_name, cfg.train.epochs, cfg.train.max_hours = "ample", 3, 10.0
    run_dir = train(cfg)
    assert load_checkpoint(run_dir / "checkpoints" / "last.pt")["epoch"] == 3
