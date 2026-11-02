"""Dataset discovery/validation, profiles, patient-level splits, transforms and spatial inversion."""

from __future__ import annotations

import shutil

import nibabel as nib
import numpy as np
import pytest
import torch
from monai.transforms import Invertd

from brain_seg.config import Config, ConfigError
from brain_seg.data import (
    CaseRecord,
    DatasetValidationError,
    assign_patient_ids,
    BUILTIN_PROFILES,
    check_case_geometry,
    detect_profile,
    discover_cases,
    load_canonical_label,
    make_patient_split,
    prepare_cases,
    verify_split,
)
from brain_seg.labels import LabelConventionError
from synthetic import write_synthetic_dataset
from brain_seg.transforms import build_eval_transforms, build_train_transforms
from conftest import SHAPE, tiny_config

MODALITIES = ["t1", "t1ce", "t2", "flair"]


def test_discovery_brats2021_layout(brats2021_root):
    result = discover_cases(brats2021_root, MODALITIES)
    assert len(result.cases) == 8 and not result.invalid
    case = result.cases[0]
    assert case.case_id == "BraTS2021_00000"
    assert case.images["t1ce"].name == "BraTS2021_00000_t1ce.nii.gz"  # t1ce is not confused with t1
    assert case.images["t1"].name == "BraTS2021_00000_t1.nii.gz"


def test_discovery_brats2023_aliases_and_profile(brats2023_root):
    result = discover_cases(brats2023_root, MODALITIES)
    assert len(result.cases) == 4
    assert result.cases[0].images["flair"].name.endswith("-t2f.nii.gz")
    profile, evidence = detect_profile(result.cases)
    assert profile.name == "brats2023" and "3" in evidence


def test_profile_detection_from_label_values(brats2021_root):
    profile, _ = detect_profile(discover_cases(brats2021_root, MODALITIES).cases)
    assert profile.name == "brats2021"


def test_missing_modality_and_mask_are_reported(tmp_path):
    write_synthetic_dataset(tmp_path, n_cases=3, shape=SHAPE)
    (tmp_path / "BraTS2021_00001" / "BraTS2021_00001_t2.nii.gz").unlink()
    (tmp_path / "BraTS2021_00002" / "BraTS2021_00002_seg.nii.gz").unlink()
    (tmp_path / "BraTS2021_00000" / "._BraTS2021_00000_t1.nii.gz").write_bytes(b"junk")  # macOS fork
    result = discover_cases(tmp_path, MODALITIES)
    assert [c.case_id for c in result.cases] == ["BraTS2021_00000"]
    assert "missing modalities ['t2']" in result.invalid["BraTS2021_00001"][0]
    assert "segmentation" in result.invalid["BraTS2021_00002"][0]

    cfg = tiny_config(tmp_path)
    with pytest.raises(DatasetValidationError, match="2 incomplete case"):
        prepare_cases(cfg)


def test_duplicate_modality_is_invalid(tmp_path):
    write_synthetic_dataset(tmp_path, n_cases=1, shape=SHAPE)
    case_dir = tmp_path / "BraTS2021_00000"
    shutil.copy(case_dir / "BraTS2021_00000_t1.nii.gz", tmp_path / "BraTS2021_00000_t1.nii")
    result = discover_cases(tmp_path, MODALITIES)
    assert "duplicate t1" in result.invalid["BraTS2021_00000"][0]


def test_geometry_mismatch_is_rejected(tmp_path):
    write_synthetic_dataset(tmp_path, n_cases=1, shape=SHAPE)
    case = discover_cases(tmp_path, MODALITIES).cases[0]
    check_case_geometry(case, MODALITIES)  # consistent case passes
    nib.save(nib.Nifti1Image(np.zeros((30, 32, 28), np.float32), np.eye(4)), case.images["t2"])
    with pytest.raises(DatasetValidationError, match="shape"):
        check_case_geometry(case, MODALITIES)
    shifted = np.eye(4)
    shifted[0, 3] = 5.0
    nib.save(nib.Nifti1Image(np.zeros(SHAPE, np.float32), shifted), case.images["t2"])
    with pytest.raises(DatasetValidationError, match="affine"):
        check_case_geometry(case, MODALITIES)


def test_missing_root_and_empty_root(tmp_path):
    with pytest.raises(FileNotFoundError):
        discover_cases(tmp_path / "nope", MODALITIES)
    with pytest.raises(ConfigError, match="data.root"):
        prepare_cases(Config())
    with pytest.raises(DatasetValidationError, match="No complete cases"):
        prepare_cases(tiny_config(tmp_path))


def test_patient_level_split_keeps_timepoints_together():
    cases = [CaseRecord(f"BraTS-GLI-{p:05d}-{t:03d}", {}) for p in range(20) for t in range(2)]
    assign_patient_ids(cases, BUILTIN_PROFILES["brats2023"])
    assert cases[1].patient_id == "BraTS-GLI-00000"
    split = make_patient_split(cases, 0.2, 0.2, seed=1)
    assert sum(len(v) for v in split.values()) == 40
    verify_split(split, cases)  # raises on leakage
    patients = {k: {c[:15] for c in v} for k, v in split.items()}
    assert not (patients["train"] & patients["val"]) and not (patients["train"] & patients["test"])
    assert len(patients["test"]) == 4 and len(patients["val"]) == 4
    assert split == make_patient_split(cases, 0.2, 0.2, seed=1)  # reproducible
    leaky = {"train": split["train"] + split["test"][:1], "val": split["val"], "test": split["test"][1:]}
    with pytest.raises(DatasetValidationError, match="leakage"):
        verify_split(leaky, cases)


def test_train_transforms_shapes_and_canonical_labels(brats2021_root):
    cfg = tiny_config(brats2021_root)
    profile, cases = prepare_cases(cfg)
    samples = build_train_transforms(cfg, profile)(cases[0].to_monai(cfg.data.modalities))
    assert len(samples) == cfg.train.samples_per_volume
    image, label = samples[0]["image"], samples[0]["label"]
    assert image.shape == (4, 16, 16, 16) and image.dtype == torch.float32
    assert label.shape == (1, 16, 16, 16)
    assert set(torch.unique(label).long().tolist()) <= {0, 1, 2, 3}  # raw ET=4 became 3


def test_eval_transforms_normalize_and_crop(brats2021_root):
    cfg = tiny_config(brats2021_root)
    profile, cases = prepare_cases(cfg)
    data = build_eval_transforms(cfg, profile)(cases[0].to_monai(cfg.data.modalities))
    image, label = data["image"], data["label"]
    assert image.shape[0] == 4 and image.shape[1:] == label.shape[1:]
    assert all(s <= o for s, o in zip(image.shape[1:], SHAPE))  # cropped to the brain
    brain = image[0] != 0
    assert abs(float(image[0][brain].mean())) < 1e-3 and abs(float(image[0][brain].std()) - 1) < 1e-2
    assert 3 in torch.unique(label).tolist()


def test_unknown_label_in_mask_raises(tmp_path):
    write_synthetic_dataset(tmp_path, n_cases=1, convention="brats2023", shape=SHAPE)
    cfg = tiny_config(tmp_path)
    cfg.data.profile = "brats2021"  # wrong convention on purpose: label 3 is undefined there
    profile, cases = prepare_cases(cfg)
    with pytest.raises(RuntimeError) as info:  # MONAI wraps transform errors; ours is the cause
        build_eval_transforms(cfg, profile)(cases[0].to_monai(cfg.data.modalities))
    cause = info.value
    while cause is not None and not isinstance(cause, LabelConventionError):
        cause = cause.__cause__
    assert cause is not None and "[3]" in str(cause)


def test_inverted_prediction_matches_original_grid(tmp_path):
    """Reorientation + resampling + cropping must be undone exactly before computing metrics."""
    affine = np.array([[-1.0, 0, 0, 30], [0, -1.2, 0, 40], [0, 0, 1.5, -10], [0, 0, 0, 1]])  # LPS, anisotropic
    write_synthetic_dataset(tmp_path, n_cases=1, shape=SHAPE, affine=affine)
    cfg = tiny_config(tmp_path)
    profile, cases = prepare_cases(cfg)
    transforms = build_eval_transforms(cfg, profile, with_label=True)
    data = transforms(cases[0].to_monai(cfg.data.modalities))
    assert data["image"].shape[1:] != SHAPE  # resampled to 1 mm, so the grid really changed
    perfect = torch.nn.functional.one_hot(data["label"][0].long(), 4).permute(3, 0, 1, 2).float()
    inverted = Invertd(keys="pred", transform=transforms, orig_keys="image", nearest_interp=False)(
        {**data, "pred": perfect})["pred"]
    restored = torch.as_tensor(inverted).argmax(0).numpy()
    original = load_canonical_label(cases[0].label, profile)
    assert restored.shape == original.shape == SHAPE
    assert (restored == original).mean() > 0.999


def _load_download_script():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "scripts" / "download_brats.py"
    spec = importlib.util.spec_from_file_location("download_brats", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_download_script_unpack_rejects_path_traversal_and_corruption(tmp_path):
    import zipfile

    module = _load_download_script()
    good = tmp_path / "good.zip"
    with zipfile.ZipFile(good, "w") as zf:
        zf.writestr("TrainingData/case/case-seg.nii.gz", b"x")
    module.unpack(good, tmp_path / "out")
    assert (tmp_path / "out" / "TrainingData" / "case" / "case-seg.nii.gz").exists()

    evil = tmp_path / "evil.zip"
    with zipfile.ZipFile(evil, "w") as zf:
        zf.writestr("../escape.txt", b"x")
    with pytest.raises(SystemExit):
        module.unpack(evil, tmp_path / "out2")
    assert not (tmp_path / "escape.txt").exists()
