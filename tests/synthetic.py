"""Synthetic BraTS-*like* volumes for UNIT TESTS ONLY.

These are geometric phantoms (ellipsoidal "brain" with a nested lesion), NOT MRI data. They exist
only to test software components (shapes, label conversion, I/O, resume logic) without BraTS.
Never train, validate or report metrics with them: every dataset written here carries a marker
file that makes brain_seg refuse it unless BRAIN_SEG_ALLOW_SYNTHETIC=1 (set by tests/conftest.py).
"""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np

from brain_seg.data import SYNTHETIC_MARKER

# Filename tokens and raw label IDs for each naming convention.
_CONVENTIONS = {
    "brats2021": {
        "sep": "_",
        "modalities": {"t1": "t1", "t1ce": "t1ce", "t2": "t2", "flair": "flair"},
        "labels": {"NCR": 1, "ED": 2, "ET": 4},
        "case_id": "BraTS2021_{index:05d}",
    },
    "brats2023": {
        "sep": "-",
        "modalities": {"t1": "t1n", "t1ce": "t1c", "t2": "t2w", "flair": "t2f"},
        "labels": {"NCR": 1, "ED": 2, "ET": 3},
        "case_id": "BraTS-GLI-{index:05d}-000",
    },
}


def _ellipsoid(shape: tuple[int, int, int], center: np.ndarray, radii: np.ndarray) -> np.ndarray:
    grid = np.stack(np.meshgrid(*[np.arange(s) for s in shape], indexing="ij"), axis=-1)
    return (((grid - center) / radii) ** 2).sum(-1) <= 1.0


def make_synthetic_case(
    shape: tuple[int, int, int] = (48, 48, 40), seed: int = 0, include_et: bool = True
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Return ``(images, masks)``; masks are boolean NCR/ED/ET arrays."""
    rng = np.random.default_rng(seed)
    shape_arr = np.array(shape, dtype=float)
    brain = _ellipsoid(shape, shape_arr / 2, shape_arr * 0.42)
    centre = shape_arr / 2 + rng.uniform(-0.12, 0.12, 3) * shape_arr
    r = rng.uniform(0.10, 0.16) * shape_arr.min()
    edema = _ellipsoid(shape, centre, np.array([r * 1.7, r * 1.5, r * 1.4])) & brain
    core = _ellipsoid(shape, centre, np.array([r, r * 0.9, r * 0.85])) & brain
    necrosis = _ellipsoid(shape, centre, np.array([r * 0.55, r * 0.5, r * 0.5])) & brain
    et = (core & ~necrosis) if include_et else np.zeros(shape, bool)
    ncr = necrosis if include_et else core
    ed = edema & ~core
    masks = {"NCR": ncr, "ED": ed, "ET": et}

    # Rough tissue contrasts: ED bright on FLAIR/T2, ET bright on T1ce, NCR dark on T1.
    contrast = {
        "t1": {"brain": 0.6, "NCR": 0.25, "ED": 0.5, "ET": 0.55},
        "t1ce": {"brain": 0.6, "NCR": 0.3, "ED": 0.55, "ET": 1.4},
        "t2": {"brain": 0.5, "NCR": 1.2, "ED": 1.0, "ET": 0.8},
        "flair": {"brain": 0.5, "NCR": 0.7, "ED": 1.3, "ET": 0.9},
    }
    images = {}
    for modality, values in contrast.items():
        vol = np.where(brain, values["brain"], 0.0)
        for region in ("ED", "NCR", "ET"):
            vol = np.where(masks[region], values[region], vol)
        vol = vol + brain * rng.normal(0, 0.08, shape)
        images[modality] = (np.clip(vol, 0, None) * 1000 * rng.uniform(0.8, 1.2)).astype(np.float32)
    return images, masks


def write_synthetic_dataset(
    root: str | Path,
    n_cases: int = 8,
    convention: str = "brats2021",
    shape: tuple[int, int, int] = (48, 48, 40),
    affine: np.ndarray | None = None,
    seed: int = 0,
) -> list[str]:
    """Write ``n_cases`` phantom cases in the directory/naming layout of a BraTS release."""
    spec = _CONVENTIONS[convention]
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    (root / SYNTHETIC_MARKER).write_text("Geometric phantoms for unit tests. Not MRI. Never use for training or evaluation.\n")
    affine = np.diag([1.0, 1.0, 1.0, 1.0]) if affine is None else affine
    case_ids = []
    for index in range(n_cases):
        case_id = spec["case_id"].format(index=index)
        case_dir = root / case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        # Every 4th case has no enhancing tumour, as happens in real BraTS data.
        images, masks = make_synthetic_case(shape, seed=seed + index, include_et=index % 4 != 3)
        for modality, token in spec["modalities"].items():
            nib.save(nib.Nifti1Image(images[modality], affine), case_dir / f"{case_id}{spec['sep']}{token}.nii.gz")
        seg = np.zeros(shape, dtype=np.uint8)
        for region in ("ED", "NCR", "ET"):
            seg[masks[region]] = spec["labels"][region]
        nib.save(nib.Nifti1Image(seg, affine), case_dir / f"{case_id}{spec['sep']}seg.nii.gz")
        case_ids.append(case_id)
    return case_ids
