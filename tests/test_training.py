"""Loss functions, a minimal training step per device, and the full pipeline on synthetic phantoms (tests only).

These are SOFTWARE tests. The phantoms are geometric shapes, so any Dice value
seen here is meaningless for brain tumour segmentation on real MRI.
"""

from __future__ import annotations

import json

import nibabel as nib
import numpy as np
import pytest
import torch
from monai.data import DataLoader, Dataset

from brain_seg.config import LossConfig
from brain_seg.data import prepare_cases
from brain_seg.evaluate import evaluate
from brain_seg.inference import load_checkpoint, run_inference
from brain_seg.labels import NUM_CLASSES
from brain_seg.losses import build_loss
from brain_seg.model import build_model
from brain_seg.train import build_optimizer, train, train_one_epoch
from brain_seg.transforms import build_train_transforms
from conftest import available_devices, tiny_config


@pytest.mark.parametrize("name", ["dice_ce", "dice_focal", "dice", "ce"])
def test_losses_match_output_representation(name):
    loss_fn = build_loss(LossConfig(name=name), NUM_CLASSES)
    target = torch.randint(0, NUM_CLASSES, (2, 1, 8, 8, 8))
    perfect = torch.nn.functional.one_hot(target[:, 0], NUM_CLASSES).permute(0, 4, 1, 2, 3).float() * 20
    random = torch.randn(2, NUM_CLASSES, 8, 8, 8)
    assert loss_fn(perfect, target) < loss_fn(random, target)
    assert torch.isfinite(loss_fn(random, target))


def test_weighted_dice_ce():
    loss_fn = build_loss(LossConfig(name="dice_ce", class_weights=[0.5, 1, 1, 2]), NUM_CLASSES)
    assert torch.isfinite(loss_fn(torch.randn(1, 4, 8, 8, 8), torch.randint(0, 4, (1, 1, 8, 8, 8))))


@pytest.mark.parametrize("device_name", available_devices())
def test_minimal_training_step_reduces_loss(brats2021_root, device_name):
    """Real data pipeline + optimizer steps on each available device (CPU / MPS / CUDA)."""
    cfg = tiny_config(brats2021_root)
    cfg.train.lr = 1e-2
    cfg.train.batch_size = 4
    cfg.augmentation.enabled = False
    profile, cases = prepare_cases(cfg)
    items = [c.to_monai(cfg.data.modalities) for c in cases[:4]]
    loader = DataLoader(Dataset(items, build_train_transforms(cfg, profile)), batch_size=4)
    device = torch.device(device_name)
    torch.manual_seed(0)
    model = build_model(cfg.model, 4, NUM_CLASSES).to(device)
    loss_fn = build_loss(cfg.loss, NUM_CLASSES)
    optimizer = build_optimizer(model, cfg)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    fixed = [next(iter(loader))]  # same batch every step -> loss must go down
    losses = [train_one_epoch(model, fixed, loss_fn, optimizer, scaler, cfg, device, False, epoch)
              for epoch in range(15)]
    assert all(np.isfinite(losses))
    assert losses[-1] < losses[0]


@pytest.mark.slow
def test_full_pipeline_train_resume_evaluate_predict(brats2021_root, tmp_path):
    cfg = tiny_config(brats2021_root, output_dir=tmp_path / "outputs")
    cfg.run_name = "synthetic_pipeline_test"
    run_dir = train(cfg)

    for name in ["config.yaml", "split.json", "environment.json", "dataset_profile.json", "history.json",
                 "history.csv", "train.log", "checkpoints/last.pt", "checkpoints/best.pt",
                 "figures/loss_curves.png", "figures/val_dice.png"]:
        assert (run_dir / name).exists(), name
    history = json.loads((run_dir / "history.json").read_text())
    assert [h["epoch"] for h in history] == [1, 2]
    assert all(0 <= h["val_dice_mean"] <= 1 for h in history)

    # Resume: extend the same run by one epoch from last.pt.
    resumed_cfg = tiny_config(brats2021_root, output_dir=tmp_path / "outputs")
    resumed_cfg.run_name, resumed_cfg.train.epochs = "synthetic_pipeline_test", 3
    train(resumed_cfg, resume=run_dir / "checkpoints" / "last.pt")
    assert [h["epoch"] for h in json.loads((run_dir / "history.json").read_text())] == [1, 2, 3]
    assert load_checkpoint(run_dir / "checkpoints" / "last.pt")["epoch"] == 3

    split = json.loads((run_dir / "split.json").read_text())
    summary = evaluate(run_dir / "checkpoints" / "best.pt", subset="test", device_name="cpu", num_workers=0)
    assert summary["n_cases"] == len(split["test"])
    eval_dir = run_dir / "eval_test"
    assert (eval_dir / "per_case_metrics.csv").exists() and (eval_dir / "summary.md").exists()
    assert (eval_dir / "figures" / "dice_distribution.png").exists()
    pred_file = next((eval_dir / "predictions").glob("*.nii.gz"))
    pred_img = nib.load(pred_file)
    case_id = pred_file.name.replace("_pred.nii.gz", "")
    ref_img = nib.load(brats2021_root / case_id / f"{case_id}_t1.nii.gz")
    assert pred_img.shape == ref_img.shape and np.allclose(pred_img.affine, ref_img.affine)
    assert set(np.unique(np.asanyarray(pred_img.dataobj))) <= {0, 1, 2, 4}  # dataset convention

    result = run_inference(run_dir / "checkpoints" / "best.pt", brats2021_root / case_id,
                           tmp_path / "pred", device_name="cpu")
    assert set(result["regions"]) == {"ET", "TC", "WT"}
    assert (tmp_path / "pred" / f"{case_id}_pred.nii.gz").exists()
    assert (tmp_path / "pred" / f"{case_id}_overlay.png").exists()
    profile, _ = prepare_cases(cfg)
    assert profile.name == "brats2021"


@pytest.mark.parametrize("device_name", available_devices())
def test_predict_case_returns_original_grid(tmp_path, device_name):
    """Regression: predictions must be inverted to the original voxel grid on every device.

    MONAI's Invertd silently skips plain (non-Meta) tensors, which once left predictions in the
    cropped/resampled grid. predict_case must return the shape of the file on disk.
    """
    from brain_seg.inference import predict_case
    from synthetic import write_synthetic_dataset
    from brain_seg.transforms import build_eval_transforms

    affine = np.array([[-1.0, 0, 0, 30], [0, -1.2, 0, 40], [0, 0, 1.5, -10], [0, 0, 0, 1]])
    write_synthetic_dataset(tmp_path, n_cases=1, shape=(32, 32, 28), affine=affine)
    cfg = tiny_config(tmp_path)
    profile, cases = prepare_cases(cfg)
    transforms = build_eval_transforms(cfg, profile, with_label=False)
    data = transforms(cases[0].to_monai(cfg.data.modalities))
    device = torch.device(device_name)
    model = build_model(cfg.model, 4, NUM_CLASSES).to(device).eval()
    pred = predict_case(model, data, transforms, cfg, device, use_amp=False)
    assert pred.shape == (32, 32, 28) and pred.dtype == np.uint8


def test_real_pipeline_refuses_synthetic_dataset(tmp_path, monkeypatch):
    """Phantoms must never reach training/evaluation: the library rejects them without the test-only flag."""
    from synthetic import write_synthetic_dataset

    from brain_seg.data import DatasetValidationError

    write_synthetic_dataset(tmp_path, n_cases=2, shape=(32, 32, 28))
    monkeypatch.delenv("BRAIN_SEG_ALLOW_SYNTHETIC")
    with pytest.raises(DatasetValidationError, match="synthetic phantom"):
        prepare_cases(tiny_config(tmp_path))


def test_max_hours_stops_cleanly_and_is_resumable(brats2021_root, tmp_path):
    cfg = tiny_config(brats2021_root, output_dir=tmp_path / "out")
    cfg.run_name, cfg.train.epochs, cfg.train.max_hours = "capped", 5, 1e-9
    run_dir = train(cfg)
    assert load_checkpoint(run_dir / "checkpoints" / "last.pt")["epoch"] == 1  # stopped after 1 epoch
    info = json.loads((run_dir / "dataset_info.json").read_text())
    assert info["n_cases"] == 8 and info["split_sizes"] == {"train": 4, "val": 2, "test": 2}
    cfg2 = tiny_config(brats2021_root, output_dir=tmp_path / "out")
    cfg2.run_name, cfg2.train.epochs = "capped", 3
    train(cfg2, resume=run_dir / "checkpoints" / "last.pt")
    assert load_checkpoint(run_dir / "checkpoints" / "last.pt")["epoch"] == 3
