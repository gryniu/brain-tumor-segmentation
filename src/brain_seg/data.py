"""Dataset discovery, validation, label-convention profiles and patient-level splits.

The BraTS releases differ in directory layout, file naming and label IDs, so
nothing here assumes a fixed structure:

* Cases are discovered by scanning ``data.root`` recursively for NIfTI files and
  grouping them by filename prefix. The *last* ``_``/``-``-separated token of a
  filename identifies its role (``flair``, ``t1c``, ``seg``, ...). Nested
  (``root/<case>/<case>_t1.nii.gz``) and flat layouts both work.
* Modality aliases map every release's naming to four canonical channels
  (``t1``, ``t1ce``, ``t2``, ``flair``), so the channel order is always fixed.
* A *profile* maps the release's raw label IDs to the canonical classes defined
  in :mod:`brain_seg.labels`. ``profile: auto`` reads label values from real
  masks before choosing; unknown label values always raise an error.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np

from .config import Config, ConfigError
from .labels import NUM_CLASSES, remap_labels_numpy
from .utils import get_logger

NIFTI_SUFFIXES = (".nii.gz", ".nii")
SYNTHETIC_MARKER = "SYNTHETIC_PHANTOM_DATA.txt"  # written by tests/synthetic.py
ALLOW_SYNTHETIC_ENV = "BRAIN_SEG_ALLOW_SYNTHETIC"
DEFAULT_MODALITY_ALIASES: dict[str, tuple[str, ...]] = {
    "t1": ("t1", "t1n"),  # native T1-weighted
    "t1ce": ("t1ce", "t1c", "t1gd"),  # post-contrast (gadolinium) T1-weighted
    "t2": ("t2", "t2w"),  # T2-weighted
    "flair": ("flair", "t2f"),  # T2 fluid-attenuated inversion recovery
}
DEFAULT_LABEL_ALIASES: tuple[str, ...] = ("seg",)


class DatasetValidationError(ValueError):
    """Raised when dataset files are missing, duplicated or spatially inconsistent."""


@dataclass(frozen=True)
class DatasetProfile:
    """Label convention and naming rules of one BraTS release."""

    name: str
    label_map: dict[int, int]  # raw label ID -> canonical class
    label_descriptions: dict[int, str]  # raw label ID -> meaning
    patient_id_pattern: str | None = None  # regex; group(1) is the patient ID
    modality_aliases: dict[str, tuple[str, ...]] = field(
        default_factory=lambda: dict(DEFAULT_MODALITY_ALIASES)
    )
    label_aliases: tuple[str, ...] = DEFAULT_LABEL_ALIASES

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["modality_aliases"] = {k: list(v) for k, v in self.modality_aliases.items()}
        d["label_aliases"] = list(self.label_aliases)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> DatasetProfile:
        required = {"name", "label_map"}
        if missing := required - set(d):
            raise ConfigError(f"Dataset profile is missing keys {sorted(missing)}")
        label_map = {int(k): int(v) for k, v in d["label_map"].items()}
        if bad := {v for v in label_map.values() if not 0 <= v < NUM_CLASSES}:
            raise ConfigError(f"Profile label_map targets must be canonical classes 0-3, got {sorted(bad)}")
        if label_map.get(0) != 0:
            raise ConfigError("Profile label_map must map raw label 0 to background (0)")
        aliases = d.get("modality_aliases") or DEFAULT_MODALITY_ALIASES
        return cls(
            name=str(d["name"]),
            label_map=label_map,
            label_descriptions={int(k): str(v) for k, v in (d.get("label_descriptions") or {}).items()},
            patient_id_pattern=d.get("patient_id_pattern"),
            modality_aliases={k: tuple(v) for k, v in aliases.items()},
            label_aliases=tuple(d.get("label_aliases") or DEFAULT_LABEL_ALIASES),
        )


BUILTIN_PROFILES: dict[str, DatasetProfile] = {
    # BraTS 2018-2021 (incl. RSNA-ASNR-MICCAI BraTS 2021): label 3 is unused, ET = 4.
    "brats2021": DatasetProfile(
        name="brats2021",
        label_map={0: 0, 1: 1, 2: 2, 4: 3},
        label_descriptions={
            0: "background",
            1: "necrotic and non-enhancing tumour core (NCR/NET)",
            2: "peritumoral edema (ED)",
            4: "GD-enhancing tumour (ET)",
        },
        patient_id_pattern=None,  # one case per patient
    ),
    # BraTS 2023 (GLI, MEN, MET, PED, SSA): ET = 3. Case IDs end with a timepoint
    # suffix (BraTS-GLI-00001-000, -001, ...) and several timepoints can share a patient.
    "brats2023": DatasetProfile(
        name="brats2023",
        label_map={0: 0, 1: 1, 2: 2, 3: 3},
        label_descriptions={
            0: "background",
            1: "non-enhancing tumour core / necrosis (NETC)",
            2: "surrounding non-enhancing FLAIR hyperintensity (SNFH, edema)",
            3: "enhancing tumour (ET)",
        },
        patient_id_pattern=r"^(BraTS-[A-Za-z]+-\d+)-\d+$",
    ),
}


@dataclass
class CaseRecord:
    """All files belonging to one MRI examination."""

    case_id: str
    images: dict[str, Path]  # canonical modality -> file
    label: Path | None = None
    patient_id: str = ""

    def to_monai(self, modalities: list[str]) -> dict[str, Any]:
        item: dict[str, Any] = {
            "image": [str(self.images[m]) for m in modalities],
            "case_id": self.case_id,
            "patient_id": self.patient_id,
        }
        if self.label is not None:
            item["label"] = str(self.label)
        return item


@dataclass
class DiscoveryResult:
    cases: list[CaseRecord]
    invalid: dict[str, list[str]]  # case ID -> problems
    unrecognized_files: list[Path]


def reject_synthetic_data(root: Path) -> None:
    """Refuse phantom datasets so synthetic data can never enter training or evaluation."""
    if (root / SYNTHETIC_MARKER).exists() and os.environ.get(ALLOW_SYNTHETIC_ENV) != "1":
        raise DatasetValidationError(
            f"{root} contains {SYNTHETIC_MARKER}: it is a synthetic phantom dataset used only by the "
            "unit tests, not real MRI. Point --data-root at a real BraTS download."
        )


def _strip_nifti_suffix(name: str) -> str | None:
    lower = name.lower()
    for suffix in NIFTI_SUFFIXES:
        if lower.endswith(suffix):
            return name[: -len(suffix)]
    return None


def parse_case_filename(path: Path) -> tuple[str, str] | None:
    """Split ``BraTS2021_00000_t1ce.nii.gz`` into ``("BraTS2021_00000", "t1ce")``."""
    stem = _strip_nifti_suffix(path.name)
    if stem is None:
        return None
    match = re.match(r"^(.+)[_-]([^_-]+)$", stem)
    if match is None:
        return None
    return match.group(1), match.group(2).lower()


def _role_lookup(
    modality_aliases: dict[str, tuple[str, ...]], label_aliases: tuple[str, ...]
) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for role, aliases in [*modality_aliases.items(), ("label", label_aliases)]:
        for alias in aliases:
            alias = alias.lower()
            if alias in lookup and lookup[alias] != role:
                raise ConfigError(f"Filename alias {alias!r} is assigned to both {lookup[alias]} and {role}")
            lookup[alias] = role
    return lookup


def discover_cases(
    root: str | Path,
    modalities: list[str],
    require_label: bool = True,
    modality_aliases: dict[str, tuple[str, ...]] | None = None,
    label_aliases: tuple[str, ...] = DEFAULT_LABEL_ALIASES,
) -> DiscoveryResult:
    """Find every case under ``root`` and report (but do not hide) incomplete ones."""
    root = Path(root).expanduser()
    if not root.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Dataset path is not a directory: {root}")
    reject_synthetic_data(root)
    modality_aliases = modality_aliases or DEFAULT_MODALITY_ALIASES
    unknown_modalities = set(modalities) - set(modality_aliases)
    if unknown_modalities:
        raise ConfigError(
            f"Unknown modalities {sorted(unknown_modalities)}; known: {sorted(modality_aliases)}"
        )
    lookup = _role_lookup(modality_aliases, label_aliases)

    groups: dict[str, dict[str, list[Path]]] = defaultdict(lambda: defaultdict(list))
    unrecognized: list[Path] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name.startswith("."):  # skips macOS "._" resource forks
            continue
        if any(part.startswith(".") for part in path.relative_to(root).parts[:-1]):
            continue
        parsed = parse_case_filename(path)
        if parsed is None:
            if _strip_nifti_suffix(path.name) is not None:
                unrecognized.append(path)
            continue
        prefix, token = parsed
        role = lookup.get(token)
        if role is None:
            unrecognized.append(path)
            continue
        groups[prefix][role].append(path)

    cases: list[CaseRecord] = []
    invalid: dict[str, list[str]] = {}
    for case_id in sorted(groups):
        files = groups[case_id]
        problems = [
            f"duplicate {role} files: {', '.join(str(p) for p in paths)}"
            for role, paths in files.items()
            if len(paths) > 1
        ]
        missing = [m for m in modalities if m not in files]
        if missing:
            problems.append(f"missing modalities {missing} (found {sorted(r for r in files if r != 'label')})")
        if require_label and "label" not in files:
            problems.append("missing segmentation mask (*_seg.nii.gz)")
        if problems:
            invalid[case_id] = problems
            continue
        cases.append(
            CaseRecord(
                case_id=case_id,
                images={m: files[m][0] for m in modalities},
                label=files["label"][0] if "label" in files else None,
            )
        )
    return DiscoveryResult(cases=cases, invalid=invalid, unrecognized_files=unrecognized)


def check_case_geometry(
    case: CaseRecord, modalities: list[str], atol: float = 1e-3
) -> tuple[tuple[int, ...], tuple[float, ...]]:
    """Verify that all modalities and the mask share shape and affine (header-only read).

    Returns the shared shape and voxel spacing. BraTS volumes are co-registered,
    so a mismatch means a corrupted or wrongly assembled case.
    """
    paths = [(m, case.images[m]) for m in modalities]
    if case.label is not None:
        paths.append(("label", case.label))
    ref_shape: tuple[int, ...] | None = None
    ref_affine: np.ndarray | None = None
    ref_name = ""
    for role, path in paths:
        try:
            img = nib.load(str(path))
        except Exception as exc:  # nibabel raises several exception types for corrupt files
            raise DatasetValidationError(f"{case.case_id}: cannot read {role} file {path}: {exc}") from exc
        shape = tuple(int(s) for s in img.shape)
        if len(shape) == 4 and shape[-1] == 1:
            shape = shape[:3]
        if len(shape) != 3:
            raise DatasetValidationError(
                f"{case.case_id}: {role} file {path.name} has shape {shape}; expected a 3D volume"
            )
        if ref_shape is None:
            ref_shape, ref_affine, ref_name = shape, img.affine, role
            continue
        if shape != ref_shape:
            raise DatasetValidationError(
                f"{case.case_id}: {role} shape {shape} differs from {ref_name} shape {ref_shape}"
            )
        if not np.allclose(img.affine, ref_affine, atol=atol):
            raise DatasetValidationError(
                f"{case.case_id}: {role} affine differs from {ref_name} affine - the volumes are "
                f"not in the same voxel space.\n{ref_name}:\n{ref_affine}\n{role}:\n{img.affine}"
            )
    assert ref_affine is not None and ref_shape is not None
    spacing = tuple(float(s) for s in np.sqrt((ref_affine[:3, :3] ** 2).sum(axis=0)))
    return ref_shape, spacing


def read_label_values(path: str | Path) -> dict[int, int]:
    """Return ``{label value: voxel count}`` for a mask, rejecting non-integer data."""
    data = np.asanyarray(nib.load(str(path)).dataobj)
    rounded = np.rint(data)
    if not np.allclose(data, rounded, atol=1e-3):
        raise DatasetValidationError(f"{path}: mask contains non-integer values")
    values, counts = np.unique(rounded.astype(np.int64), return_counts=True)
    return {int(v): int(c) for v, c in zip(values, counts)}


def detect_profile(cases: list[CaseRecord], max_cases: int = 5) -> tuple[DatasetProfile, str]:
    """Choose a built-in profile from label values actually present in the masks."""
    labelled = [c for c in cases if c.label is not None][:max_cases]
    if not labelled:
        raise DatasetValidationError("Cannot auto-detect the label convention: no masks found")
    values: set[int] = set()
    for case in labelled:
        values |= set(read_label_values(case.label))  # type: ignore[arg-type]
    evidence = f"label values {sorted(values)} in {len(labelled)} mask(s)"
    if 4 in values and 3 in values:
        raise DatasetValidationError(
            f"Masks contain both labels 3 and 4 ({evidence}). This matches neither BraTS 2021 nor "
            "BraTS 2023 (BraTS 2024 post-treatment adds label 4 = resection cavity). Define "
            "data.profile: custom with an explicit label_map."
        )
    if unexpected := values - {0, 1, 2, 3, 4}:
        raise DatasetValidationError(f"Unexpected label values {sorted(unexpected)} ({evidence})")
    if 4 in values:
        return BUILTIN_PROFILES["brats2021"], evidence
    if 3 in values:
        return BUILTIN_PROFILES["brats2023"], evidence
    # Only 0/1/2 seen: fall back to the naming scheme, but say so.
    name = "brats2023" if labelled[0].case_id.startswith("BraTS-") else "brats2021"
    return BUILTIN_PROFILES[name], f"{evidence}; ET label absent, chose {name} from file naming"


def resolve_profile(cfg: Config, cases: list[CaseRecord]) -> DatasetProfile:
    data_cfg = cfg.data
    logger = get_logger()
    if data_cfg.profile == "custom":
        return DatasetProfile.from_dict(data_cfg.custom_profile or {})
    if data_cfg.profile in BUILTIN_PROFILES:
        return BUILTIN_PROFILES[data_cfg.profile]
    profile, evidence = detect_profile(cases)
    logger.info("Auto-detected dataset profile '%s' (%s)", profile.name, evidence)
    return profile


def assign_patient_ids(cases: list[CaseRecord], profile: DatasetProfile) -> None:
    """Fill ``patient_id`` so that splits never separate timepoints of one patient."""
    pattern = re.compile(profile.patient_id_pattern) if profile.patient_id_pattern else None
    unmatched = []
    for case in cases:
        match = pattern.match(case.case_id) if pattern else None
        if pattern and match is None:
            unmatched.append(case.case_id)
        case.patient_id = match.group(1) if match else case.case_id
    if unmatched:
        get_logger().warning(
            "%d case ID(s) did not match patient_id_pattern %r and are treated as separate "
            "patients (e.g. %s)", len(unmatched), profile.patient_id_pattern, unmatched[:3]
        )


def prepare_cases(
    cfg: Config, data_root: str | Path | None = None, require_label: bool = True
) -> tuple[DatasetProfile, list[CaseRecord]]:
    """Discover, validate and annotate all cases described by the config."""
    logger = get_logger()
    root = data_root or cfg.data.root
    if not root:
        raise ConfigError("No dataset directory given: set data.root in the config or pass --data-root")
    custom_aliases = None
    if cfg.data.profile == "custom" and cfg.data.custom_profile:
        custom_aliases = DatasetProfile.from_dict(cfg.data.custom_profile).modality_aliases
    result = discover_cases(root, cfg.data.modalities, require_label, custom_aliases)
    if result.invalid:
        report = "\n".join(f"  - {cid}: {'; '.join(p)}" for cid, p in result.invalid.items())
        if not cfg.data.skip_invalid_cases:
            raise DatasetValidationError(
                f"{len(result.invalid)} incomplete case(s) under {root}:\n{report}\n"
                "Fix or remove them, or set data.skip_invalid_cases: true to exclude them explicitly."
            )
        logger.warning("Excluding %d invalid case(s):\n%s", len(result.invalid), report)
    if not result.cases:
        hint = ", ".join(p.name for p in result.unrecognized_files[:5]) or "no NIfTI files"
        raise DatasetValidationError(
            f"No complete cases found under {root} (found: {hint}). Expected one NIfTI file per "
            f"modality {cfg.data.modalities} plus *_seg.nii.gz per case, named <case>_<modality>.nii.gz "
            "or <case>-<modality>.nii.gz. See docs/reference.md 'Dataset' section."
        )
    profile = resolve_profile(cfg, result.cases)
    assign_patient_ids(result.cases, profile)
    if cfg.data.validate_headers:
        for case in result.cases:
            check_case_geometry(case, cfg.data.modalities)
    logger.info(
        "Found %d valid case(s) from %d patient(s) under %s",
        len(result.cases), len({c.patient_id for c in result.cases}), root,
    )
    return profile, result.cases


# --------------------------------------------------------------------------- splits


def make_patient_split(
    cases: list[CaseRecord], val_fraction: float, test_fraction: float, seed: int
) -> dict[str, list[str]]:
    """Split by *patient* into train/val/test case-ID lists (no patient appears twice)."""
    from sklearn.model_selection import train_test_split

    patients = sorted({c.patient_id for c in cases})
    n = len(patients)
    n_test = max(1, round(n * test_fraction)) if test_fraction > 0 else 0
    n_val = max(1, round(n * val_fraction)) if val_fraction > 0 else 0
    if n - n_test - n_val < 1:
        raise DatasetValidationError(
            f"Only {n} patient(s): not enough for train/val/test fractions "
            f"{1 - val_fraction - test_fraction:.2f}/{val_fraction:.2f}/{test_fraction:.2f}"
        )
    rest, test_patients = (patients, []) if n_test == 0 else train_test_split(
        patients, test_size=n_test, random_state=seed, shuffle=True
    )
    train_patients, val_patients = train_test_split(rest, test_size=n_val, random_state=seed, shuffle=True)
    subset_of = {p: "train" for p in train_patients}
    subset_of.update({p: "val" for p in val_patients})
    subset_of.update({p: "test" for p in test_patients})
    split: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    for case in sorted(cases, key=lambda c: c.case_id):
        split[subset_of[case.patient_id]].append(case.case_id)
    return split


def verify_split(split: dict[str, list[str]], cases: list[CaseRecord]) -> None:
    """Fail loudly on overlapping cases/patients or IDs missing from the dataset."""
    by_id = {c.case_id: c for c in cases}
    seen: dict[str, str] = {}
    for subset in ("train", "val", "test"):
        for case_id in split.get(subset, []):
            if case_id in seen:
                raise DatasetValidationError(f"Case {case_id} is in both {seen[case_id]} and {subset}")
            seen[case_id] = subset
    missing = [cid for cid in seen if cid not in by_id]
    if missing:
        raise DatasetValidationError(
            f"{len(missing)} case(s) in the split file are not present in the dataset "
            f"(e.g. {missing[:5]}). Is data.root pointing at the same dataset version?"
        )
    patient_subsets: dict[str, set[str]] = defaultdict(set)
    for case_id, subset in seen.items():
        patient_subsets[by_id[case_id].patient_id].add(subset)
    leaked = {p: s for p, s in patient_subsets.items() if len(s) > 1}
    if leaked:
        raise DatasetValidationError(f"Patient-level leakage between subsets: {dict(list(leaked.items())[:5])}")


def default_split_path(cfg: Config) -> Path:
    if cfg.data.split_file:
        return Path(cfg.data.split_file)
    return Path(cfg.output_dir) / "splits" / f"split_seed{cfg.data.split_seed}.json"


def save_split(split: dict[str, list[str]], path: str | Path, metadata: dict[str, Any] | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"metadata": metadata or {}, **{k: split[k] for k in ("train", "val", "test")}}
    path.write_text(json.dumps(payload, indent=2))


def load_split(path: str | Path) -> dict[str, list[str]]:
    payload = json.loads(Path(path).read_text())
    for key in ("train", "val", "test"):
        if key not in payload or not isinstance(payload[key], list):
            raise DatasetValidationError(f"Split file {path} has no '{key}' list")
    return {k: payload[k] for k in ("train", "val", "test")}


def get_or_create_split(cfg: Config, cases: list[CaseRecord], profile: DatasetProfile) -> dict[str, list[str]]:
    """Reuse an existing split file (reproducibility) or create a new patient-level split."""
    logger = get_logger()
    path = default_split_path(cfg)
    if path.exists():
        split = load_split(path)
        logger.info("Using existing split %s", path)
        unassigned = {c.case_id for c in cases} - {cid for ids in split.values() for cid in ids}
        if unassigned:
            logger.warning(
                "%d case(s) in the dataset are not in the split file and will be ignored "
                "(e.g. %s). Delete the split file to re-split.", len(unassigned), sorted(unassigned)[:3]
            )
    else:
        split = make_patient_split(cases, cfg.data.val_fraction, cfg.data.test_fraction, cfg.data.split_seed)
        save_split(
            split,
            path,
            {
                "seed": cfg.data.split_seed,
                "val_fraction": cfg.data.val_fraction,
                "test_fraction": cfg.data.test_fraction,
                "profile": profile.name,
                "grouping": "patient",
                "n_cases": len(cases),
            },
        )
        logger.info("Created new patient-level split %s", path)
    verify_split(split, cases)
    logger.info("Split sizes: train=%d val=%d test=%d", *(len(split[k]) for k in ("train", "val", "test")))
    return split


def dataset_fingerprint(cases: list[CaseRecord]) -> str:
    """Short hash of the sorted case IDs, to tell exactly which cases a run used."""
    return hashlib.sha1("\n".join(sorted(c.case_id for c in cases)).encode()).hexdigest()[:16]


def select_cases(cases: list[CaseRecord], case_ids: list[str]) -> list[CaseRecord]:
    by_id = {c.case_id: c for c in cases}
    return [by_id[cid] for cid in case_ids]


def load_canonical_label(path: str | Path, profile: DatasetProfile) -> np.ndarray:
    """Load a mask from disk in its original voxel grid and convert it to canonical classes."""
    data = np.asanyarray(nib.load(str(path)).dataobj)
    if data.ndim == 4 and data.shape[-1] == 1:
        data = data[..., 0]
    return remap_labels_numpy(data, profile.label_map, source=str(path))


# --------------------------------------------------------------------------- MONAI datasets


class CompactCacheDataset:
    """Disk cache of the deterministic preprocessing in a compact, fast-to-read format.

    The first access to a case runs ``pre_transform`` (decompress NIfTI, reorient, resample, crop,
    z-score) and stores the result as two ``.npy`` files: the image as **float16** and the mask as
    **uint8** (canonical classes 0-3, so the mask is lossless). Later accesses memory-map those files,
    so no decompression or resampling is repeated and, with about 9 bytes per voxel instead of the
    20 bytes of a float32 image plus float32 mask, the cache is roughly 2.2x smaller than ``disk``.

    float16 keeps ~3 significant digits, ample for z-scored intensities (|value| < ~10); the
    quantisation is far below the intensity augmentation noise (std 0.1) applied during training.
    ``post_transform`` (random patch sampling and augmentation) runs on every access. Without a
    ``post_transform`` the full volume is returned as float32 tensors (validation).

    Files are written atomically and keyed by the case's file paths, so several DataLoader workers
    or processes can share a cache directory, and a truncated or corrupt file is rebuilt.
    """

    def __init__(self, items: list[dict[str, Any]], pre_transform: Any, post_transform: Any, cache_dir: str | Path):
        self.items = items
        self.pre_transform = pre_transform
        self.post_transform = post_transform
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def __len__(self) -> int:
        return len(self.items)

    def _paths(self, item: dict[str, Any]) -> tuple[Path, Path]:
        key = hashlib.sha1(json.dumps([item["image"], item.get("label")], sort_keys=True).encode()).hexdigest()[:20]
        return self.cache_dir / f"{key}_image.npy", self.cache_dir / f"{key}_label.npy"

    @staticmethod
    def _atomic_save(array: np.ndarray, path: Path) -> None:
        tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        with open(tmp, "wb") as fh:
            np.save(fh, array)
        os.replace(tmp, path)

    def _build(self, item: dict[str, Any], image_path: Path, label_path: Path) -> None:
        import torch

        data = self.pre_transform(dict(item))
        image = torch.as_tensor(data["image"]).float().numpy()
        if not np.isfinite(image).all() or np.abs(image).max() > 6e4:
            raise ValueError(f"{item.get('case_id')}: preprocessed image has non-finite or float16-overflowing values")
        label = torch.as_tensor(data["label"]).numpy()
        if label.min() < 0 or label.max() >= NUM_CLASSES or not np.array_equal(label, np.round(label)):
            raise ValueError(f"{item.get('case_id')}: preprocessed mask is not a canonical class map")
        self._atomic_save(label.astype(np.uint8), label_path)
        self._atomic_save(image.astype(np.float16), image_path)  # written last: its presence marks a complete entry

    def _load(self, item: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        image_path, label_path = self._paths(item)
        for attempt in range(2):
            try:
                image = np.load(image_path, mmap_mode="r")
                label = np.load(label_path, mmap_mode="r")
                if image.shape[1:] != label.shape[1:]:
                    raise ValueError("image and mask shapes differ")
                return image, label
            except (FileNotFoundError, ValueError, EOFError, OSError):
                if attempt:
                    raise
                self._build(item, image_path, label_path)
        raise AssertionError("unreachable")

    def __getitem__(self, index: int) -> Any:
        import torch

        item = self.items[index]
        image, label = self._load(item)
        sample = {k: v for k, v in item.items() if k not in ("image", "label")}
        if self.post_transform is None:
            sample["image"] = torch.from_numpy(np.array(image, dtype=np.float32))
            sample["label"] = torch.from_numpy(np.array(label, dtype=np.float32))
            return sample
        sample["image"] = torch.from_numpy(np.array(image))  # float16; cast to float32 after the crop
        sample["label"] = torch.from_numpy(np.array(label, dtype=np.float32))
        return self.post_transform(sample)


def build_dataset(
    items: list[dict[str, Any]],
    transform: Any,
    cache: str,
    cache_rate: float = 1.0,
    num_workers: int = 0,
    cache_dir: str | Path | None = None,
    split_transforms: tuple[Any, Any | None] | None = None,
):
    """Wrap case dictionaries in a MONAI dataset with the requested caching strategy.

    * ``none``   - load and preprocess every time (lowest memory, slowest).
    * ``memory`` - cache deterministic preprocessing in RAM (fast, ~70 MB per BraTS case).
    * ``disk``   - cache deterministic preprocessing as .pt files (fast after the first epoch).
    * ``npy``    - compact float16/uint8 disk cache (:class:`CompactCacheDataset`); needs
      ``split_transforms=(preprocessing, random_part_or_None)`` instead of ``transform``.
    """
    from monai.data import CacheDataset, Dataset, PersistentDataset

    if cache == "none":
        return Dataset(items, transform=transform)
    if cache == "npy":
        if cache_dir is None or split_transforms is None:
            raise ConfigError("cache='npy' requires a cache directory and split (pre, post) transforms")
        return CompactCacheDataset(items, split_transforms[0], split_transforms[1], cache_dir)
    if cache == "memory":
        return CacheDataset(items, transform=transform, cache_rate=cache_rate, num_workers=num_workers or None)
    if cache == "disk":
        if cache_dir is None:
            raise ConfigError("cache='disk' requires a cache directory")
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        return PersistentDataset(items, transform=transform, cache_dir=cache_dir)
    raise ConfigError(f"Unknown cache mode {cache!r}")
