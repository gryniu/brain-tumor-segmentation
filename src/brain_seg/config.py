"""Typed experiment configuration.

Configurations are YAML files that map onto the dataclasses below. Two small
features keep experiments controlled and reproducible:

* ``_base_: other.yaml`` - a config may inherit from another file and override
  only the keys it changes (one change per experiment file).
* ``--set section.key=value`` - command-line overrides, parsed as YAML values.

Unknown keys are rejected so that a typo never silently falls back to a default.
"""

from __future__ import annotations

import copy
import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, get_args, get_origin, get_type_hints

import yaml


class ConfigError(ValueError):
    """Raised for invalid or inconsistent configuration values."""


@dataclass
class DataConfig:
    root: str | None = None  # dataset directory; may also be given with --data-root
    profile: str = "auto"  # auto | brats2021 | brats2023 | custom
    custom_profile: dict[str, Any] | None = None  # used when profile == "custom"
    modalities: list[str] = field(default_factory=lambda: ["t1", "t1ce", "t2", "flair"])
    split_file: str | None = None  # default: <output_dir>/splits/split_seed<seed>.json
    val_fraction: float = 0.15
    test_fraction: float = 0.15
    split_seed: int = 42
    validate_headers: bool = True
    skip_invalid_cases: bool = False  # if true, invalid cases are excluded and reported
    target_spacing: list[float] | None = field(default_factory=lambda: [1.0, 1.0, 1.0])
    crop_foreground: bool = True
    cache: str = "none"  # none | memory | disk | npy (compact float16/uint8 disk cache, ~2.2x smaller than disk)
    cache_rate: float = 1.0
    cache_dir: str | None = None  # where disk/npy caches go (default: <output_dir>/cache); keep it on a fast local disk
    num_workers: int = 2  # DataLoader worker processes (0 = load in the main process)


@dataclass
class ModelConfig:
    name: str = "unet3d"  # unet3d (this repo) | monai_unet (MONAI residual U-Net)
    features: list[int] = field(default_factory=lambda: [16, 32, 64, 128, 256])
    norm: str = "instance"  # instance | batch | group
    dropout: float = 0.0


@dataclass
class LossConfig:
    name: str = "dice_ce"  # dice_ce | dice_focal | dice | ce
    lambda_dice: float = 1.0
    lambda_ce: float = 1.0  # weight of the CE (or focal) term
    include_background: bool = False  # only affects the Dice term
    focal_gamma: float = 2.0
    class_weights: list[float] | None = None  # per-class CE weights (length 4)


@dataclass
class AugmentationConfig:
    enabled: bool = True
    flip_prob: float = 0.5  # per spatial axis
    affine_prob: float = 0.2
    rotate_range_deg: float = 15.0
    scale_range: float = 0.1
    intensity_scale_prob: float = 0.5
    intensity_scale_factor: float = 0.1
    intensity_shift_prob: float = 0.5
    intensity_shift_offset: float = 0.1
    noise_prob: float = 0.15
    noise_std: float = 0.1


@dataclass
class TrainConfig:
    device: str = "auto"  # auto | cuda | cuda:N | mps | cpu
    seed: int = 42
    deterministic: bool = False  # slower; forces deterministic cuDNN algorithms
    epochs: int = 300
    batch_size: int = 1
    patch_size: list[int] = field(default_factory=lambda: [96, 96, 96])
    samples_per_volume: int = 1  # patches per loaded volume; patches/step = batch_size * this
    label_sampling_ratios: list[float] = field(default_factory=lambda: [1.0, 1.0, 1.0, 1.0])
    accumulation_steps: int = 1
    optimizer: str = "adamw"  # adamw | sgd
    lr: float = 2e-4
    weight_decay: float = 1e-5
    momentum: float = 0.99  # SGD only
    scheduler: str = "cosine"  # cosine | poly | none
    warmup_epochs: int = 5
    min_lr: float = 1e-6
    grad_clip_norm: float | None = 12.0
    amp: bool = True  # mixed precision; only applied on CUDA (MPS/CPU always train in float32)
    amp_dtype: str = "float16"  # float16 (any CUDA GPU, uses GradScaler) | bfloat16 (Ampere or newer)
    val_interval: int = 5
    early_stopping_patience: int | None = 10  # in validation rounds; null disables
    early_stopping_min_delta: float = 1e-4
    train_eval_cases: int = 4  # training cases re-evaluated like validation (overfitting check)
    max_hours: float | None = None  # wall-time cap (resumable): never starts an epoch that would not fit, validates once more before stopping
    limit_train_cases: int | None = None  # debugging only
    limit_val_cases: int | None = None  # debugging only
    log_interval: int = 20


@dataclass
class InferenceConfig:
    overlap: float = 0.5
    sw_batch_size: int = 1
    blend_mode: str = "gaussian"  # gaussian | constant
    stitch_on_cpu: bool = False  # assemble full-volume logits in CPU RAM to save GPU memory


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    output_dir: str = "outputs"
    run_name: str | None = None

    def validate(self) -> None:
        """Check cross-field consistency; raises ConfigError with an actionable message."""
        d, m, t, lo, inf = self.data, self.model, self.train, self.loss, self.inference
        if d.profile not in {"auto", "brats2021", "brats2023", "custom"}:
            raise ConfigError(f"data.profile must be auto|brats2021|brats2023|custom, got {d.profile!r}")
        if d.profile == "custom" and not d.custom_profile:
            raise ConfigError("data.profile is 'custom' but data.custom_profile is empty")
        if not d.modalities or len(set(d.modalities)) != len(d.modalities):
            raise ConfigError(f"data.modalities must be a non-empty list without duplicates: {d.modalities}")
        if not (0 <= d.val_fraction < 1 and 0 <= d.test_fraction < 1 and d.val_fraction + d.test_fraction < 1):
            raise ConfigError("data.val_fraction and data.test_fraction must be in [0, 1) and sum to < 1")
        if d.val_fraction == 0:
            raise ConfigError("data.val_fraction must be > 0: the validation set is used for model selection")
        if d.cache not in {"none", "memory", "disk", "npy"}:
            raise ConfigError(f"data.cache must be none|memory|disk|npy, got {d.cache!r}")
        if d.target_spacing is not None and (len(d.target_spacing) != 3 or min(d.target_spacing) <= 0):
            raise ConfigError(f"data.target_spacing must be null or 3 positive numbers: {d.target_spacing}")
        if m.name not in {"unet3d", "monai_unet"}:
            raise ConfigError(f"model.name must be unet3d|monai_unet, got {m.name!r}")
        if len(m.features) < 2 or any(f <= 0 for f in m.features):
            raise ConfigError(f"model.features needs >= 2 positive channel counts: {m.features}")
        if m.norm not in {"instance", "batch", "group"}:
            raise ConfigError(f"model.norm must be instance|batch|group, got {m.norm!r}")
        if len(t.patch_size) != 3:
            raise ConfigError(f"train.patch_size must have 3 values: {t.patch_size}")
        factor = 2 ** (len(m.features) - 1)
        if any(p % factor for p in t.patch_size):
            raise ConfigError(
                f"train.patch_size {t.patch_size} must be divisible by {factor} "
                f"(2^(number of downsampling steps) for {len(m.features)} feature levels)"
            )
        if len(t.label_sampling_ratios) != 4 or min(t.label_sampling_ratios) < 0:
            raise ConfigError("train.label_sampling_ratios needs 4 non-negative values (BG, NCR, ED, ET)")
        for name in ("epochs", "batch_size", "samples_per_volume", "accumulation_steps", "val_interval"):
            if getattr(t, name) < 1:
                raise ConfigError(f"train.{name} must be >= 1")
        if t.optimizer not in {"adamw", "sgd"}:
            raise ConfigError(f"train.optimizer must be adamw|sgd, got {t.optimizer!r}")
        if t.scheduler not in {"cosine", "poly", "none"}:
            raise ConfigError(f"train.scheduler must be cosine|poly|none, got {t.scheduler!r}")
        if t.amp_dtype not in {"float16", "bfloat16"}:
            raise ConfigError(f"train.amp_dtype must be float16|bfloat16, got {t.amp_dtype!r}")
        if t.max_hours is not None and t.max_hours <= 0:
            raise ConfigError("train.max_hours must be positive or null")
        if t.lr <= 0:
            raise ConfigError("train.lr must be positive")
        if lo.name not in {"dice_ce", "dice_focal", "dice", "ce"}:
            raise ConfigError(f"loss.name must be dice_ce|dice_focal|dice|ce, got {lo.name!r}")
        if lo.class_weights is not None and len(lo.class_weights) != 4:
            raise ConfigError("loss.class_weights must have 4 values (BG, NCR, ED, ET)")
        if not 0 <= inf.overlap < 1:
            raise ConfigError("inference.overlap must be in [0, 1)")
        if inf.blend_mode not in {"gaussian", "constant"}:
            raise ConfigError("inference.blend_mode must be gaussian|constant")


def _deep_merge(base: dict, override: dict) -> dict:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _read_yaml_with_bases(path: Path, _seen: tuple[Path, ...] = ()) -> dict:
    path = path.resolve()
    if path in _seen:
        raise ConfigError(f"Circular _base_ reference involving {path}")
    if not path.is_file():
        raise ConfigError(f"Config file not found: {path}")
    with path.open() as fh:
        raw = yaml.safe_load(fh) or {}
    if not isinstance(raw, dict):
        raise ConfigError(f"{path} must contain a YAML mapping at the top level")
    base_ref = raw.pop("_base_", None)
    if base_ref is None:
        return raw
    base = _read_yaml_with_bases((path.parent / base_ref), _seen + (path,))
    return _deep_merge(base, raw)


def _coerce(value: Any, hint: Any, where: str) -> Any:
    """Light type coercion/validation for YAML values (int->float, list checks)."""
    if value is None:
        if type(None) in get_args(hint) or hint is Any:
            return None
        raise ConfigError(f"{where} may not be null")
    if dataclasses.is_dataclass(hint):
        if not isinstance(value, dict):
            raise ConfigError(f"{where} must be a mapping")
        return _build_dataclass(hint, value, where)
    origin = get_origin(hint)
    args = [a for a in get_args(hint) if a is not type(None)]
    if origin is not None and type(None) in get_args(hint) and len(args) == 1:
        return _coerce(value, args[0], where)  # Optional[X]
    if hint is float and isinstance(value, int) and not isinstance(value, bool):
        return float(value)
    if hint is float and isinstance(value, str):
        # PyYAML follows YAML 1.1, which reads "1e-3" (no decimal point) as a string.
        try:
            return float(value)
        except ValueError:
            raise ConfigError(f"{where} must be a number, got {value!r}") from None
    if hint in (int, float, str, bool):
        if not isinstance(value, hint) or (hint is int and isinstance(value, bool)):
            raise ConfigError(f"{where} must be of type {hint.__name__}, got {value!r}")
        return value
    if origin is list:
        if not isinstance(value, (list, tuple)):
            raise ConfigError(f"{where} must be a list, got {value!r}")
        (item_hint,) = get_args(hint) or (Any,)
        return [_coerce(v, item_hint, f"{where}[{i}]") for i, v in enumerate(value)]
    return value


def _build_dataclass(cls: type, values: dict, where: str = "config") -> Any:
    hints = get_type_hints(cls)
    known = {f.name for f in dataclasses.fields(cls)}
    unknown = set(values) - known
    if unknown:
        raise ConfigError(f"Unknown key(s) in {where}: {sorted(unknown)}. Valid keys: {sorted(known)}")
    kwargs = {k: _coerce(v, hints[k], f"{where}.{k}") for k, v in values.items()}
    return cls(**kwargs)


def parse_overrides(overrides: list[str] | None) -> dict:
    """Turn ``["train.lr=1e-3", "model.features=[8,16,32]"]`` into a nested dict."""
    result: dict = {}
    for item in overrides or []:
        if "=" not in item:
            raise ConfigError(f"Override {item!r} must look like section.key=value")
        dotted, raw_value = item.split("=", 1)
        value = yaml.safe_load(raw_value)
        node = result
        parts = dotted.strip().split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    return result


def config_from_dict(values: dict) -> Config:
    cfg = _build_dataclass(Config, values)
    cfg.validate()
    return cfg


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> Config:
    """Load a YAML config (with ``_base_`` inheritance) and apply ``--set`` overrides."""
    values = _read_yaml_with_bases(Path(path)) if path is not None else {}
    values = _deep_merge(values, parse_overrides(overrides))
    return config_from_dict(values)


def apply_overrides(cfg: Config, overrides: list[str] | None) -> Config:
    """Return a new config with ``--set`` style overrides applied."""
    return config_from_dict(_deep_merge(config_to_dict(cfg), parse_overrides(overrides)))


def config_to_dict(cfg: Config) -> dict:
    return dataclasses.asdict(cfg)


def save_config(cfg: Config, path: str | Path) -> None:
    with Path(path).open("w") as fh:
        yaml.safe_dump(config_to_dict(cfg), fh, sort_keys=False)
