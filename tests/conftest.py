"""Shared fixtures.

IMPORTANT: every test uses small *synthetic phantoms* (tests/synthetic.py), never BraTS data.
Synthetic data lives only in the test suite; the library refuses phantom datasets unless the
BRAIN_SEG_ALLOW_SYNTHETIC environment variable is set, which only this file does. Passing tests show the code is internally consistent; they say
nothing about segmentation quality on real MRI.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from brain_seg.config import Config
from synthetic import write_synthetic_dataset

os.environ["BRAIN_SEG_ALLOW_SYNTHETIC"] = "1"  # tests only; see brain_seg.data.reject_synthetic_data

SHAPE = (32, 32, 28)


def available_devices() -> list[str]:
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    if torch.backends.mps.is_available():
        devices.append("mps")
    return devices


@pytest.fixture(scope="session")
def brats2021_root(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("brats2021")
    write_synthetic_dataset(root, n_cases=8, convention="brats2021", shape=SHAPE)
    return root


@pytest.fixture(scope="session")
def brats2023_root(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("brats2023")
    write_synthetic_dataset(root, n_cases=4, convention="brats2023", shape=SHAPE)
    return root


def tiny_config(data_root: Path | None, output_dir: Path | None = None) -> Config:
    """A configuration small enough to train for a few steps on a CPU in seconds."""
    cfg = Config()
    cfg.data.root = str(data_root) if data_root else None
    cfg.data.num_workers = 0
    cfg.data.val_fraction = 0.25
    cfg.data.test_fraction = 0.25
    cfg.model.features = [4, 8, 16]
    cfg.train.patch_size = [16, 16, 16]
    cfg.train.batch_size = 2
    cfg.train.epochs = 2
    cfg.train.val_interval = 1
    cfg.train.warmup_epochs = 0
    cfg.train.train_eval_cases = 1
    cfg.train.device = "cpu"
    cfg.train.log_interval = 0
    cfg.augmentation.affine_prob = 0.0
    if output_dir is not None:
        cfg.output_dir = str(output_dir)
    cfg.validate()
    return cfg
