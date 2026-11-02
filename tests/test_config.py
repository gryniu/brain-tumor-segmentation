"""Configuration loading, inheritance, overrides and validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from brain_seg.config import ConfigError, apply_overrides, load_config

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


@pytest.mark.parametrize("path", sorted(CONFIG_DIR.rglob("*.yaml")), ids=lambda p: p.name)
def test_shipped_configs_are_valid(path):
    load_config(path)


def test_profiles_differ_where_expected():
    default = load_config(CONFIG_DIR / "default.yaml")
    mps = load_config(CONFIG_DIR / "local_mps.yaml")
    cuda = load_config(CONFIG_DIR / "cloud_cuda.yaml")
    assert default.train.batch_size == 1 and default.train.device == "auto"
    assert mps.model.features == default.model.features  # inherited, not repeated
    assert cuda.train.device == "cuda" and cuda.train.patch_size == [128, 128, 128]


def test_experiment_configs_change_one_thing():
    base = load_config(CONFIG_DIR / "default.yaml")
    exp = load_config(CONFIG_DIR / "experiments" / "lr_1e-3.yaml")
    assert exp.train.lr == 1e-3 and exp.train.patch_size == base.train.patch_size


def test_overrides_and_type_checks():
    cfg = load_config(CONFIG_DIR / "default.yaml", ["train.lr=1e-3", "model.features=[8,16,32]",
                                                    "train.patch_size=[32,32,32]"])
    assert cfg.train.lr == 1e-3 and cfg.model.features == [8, 16, 32]
    assert apply_overrides(cfg, ["train.epochs=7"]).train.epochs == 7
    with pytest.raises(ConfigError, match="Unknown key"):
        load_config(CONFIG_DIR / "default.yaml", ["train.learning_rate=1e-3"])
    with pytest.raises(ConfigError, match="divisible by 16"):
        load_config(CONFIG_DIR / "default.yaml", ["train.patch_size=[100,96,96]"])
    with pytest.raises(ConfigError, match="type int"):
        load_config(CONFIG_DIR / "default.yaml", ["train.epochs=ten"])
