"""Small shared helpers: device selection, seeding, logging and environment capture."""

from __future__ import annotations

import contextlib
import logging
import os
import platform
import sys
from pathlib import Path
from typing import Any

import torch

LOGGER_NAME = "brain_seg"


def get_logger() -> logging.Logger:
    return logging.getLogger(LOGGER_NAME)


def setup_logging(log_file: str | Path | None = None, level: int = logging.INFO) -> logging.Logger:
    """Log to stderr and, optionally, to a file inside the run directory."""
    logger = get_logger()
    logger.setLevel(level)
    logger.propagate = False
    formatter = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%H:%M:%S")
    if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
               for h in logger.handlers):
        stream = logging.StreamHandler()
        stream.setFormatter(formatter)
        logger.addHandler(stream)
    if log_file is not None:
        log_file = Path(log_file).resolve()
        for handler in list(logger.handlers):
            if isinstance(handler, logging.FileHandler):
                logger.removeHandler(handler)
                handler.close()
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger


def select_device(requested: str = "auto") -> torch.device:
    """Pick CUDA, then Apple MPS, then CPU for ``auto``; validate explicit requests."""
    requested = requested.lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but is not available. Check that you installed a CUDA build of "
            "PyTorch (see docs/reference.md) or use --device cpu."
        )
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available on this machine.")
    return device


def resolve_amp(enabled: bool, dtype: str, device: torch.device) -> bool:
    """Decide whether mixed precision is actually used on ``device``.

    AMP is applied only on CUDA. On Apple MPS, float16 autocast for 3D convolutions
    is less mature and gives little speed-up for training, and on CPU it is
    slower than float32, so both always train in float32.
    """
    if not enabled or device.type != "cuda":
        return False
    if dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError(
            "train.amp_dtype=bfloat16 is not supported by this GPU (needs Ampere or newer); "
            "use train.amp_dtype=float16"
        )
    return True


def amp_context(device: torch.device, enabled: bool, dtype: str = "float16"):
    """Autocast context for CUDA mixed precision, or a no-op elsewhere."""
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=getattr(torch, dtype))
    return contextlib.nullcontext()


def seed_everything(seed: int, deterministic: bool = False) -> None:
    """Seed Python, NumPy, PyTorch and MONAI's random transforms."""
    from monai.utils import set_determinism

    set_determinism(seed=seed, use_deterministic_algorithms=deterministic)
    if not deterministic and torch.cuda.is_available():
        # set_determinism(seed=...) disables cuDNN autotuning; re-enable it for speed when
        # bitwise reproducibility was not requested.
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def environment_info(device: torch.device | None = None) -> dict[str, Any]:
    """Versions and hardware details stored next to every run for reproducibility."""
    import monai
    import nibabel
    import numpy

    info: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "monai": monai.__version__,
        "numpy": numpy.__version__,
        "nibabel": nibabel.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "device": str(device) if device is not None else None,
    }
    if device is not None and device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        info["gpu_name"] = props.name
        info["gpu_memory_gb"] = round(props.total_memory / 1024**3, 2)
    return info


def atomic_torch_save(obj: Any, path: str | Path) -> None:
    """Write a checkpoint via a temporary file so an interruption never leaves a corrupt file."""
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)
