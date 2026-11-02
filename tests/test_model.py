"""Model shapes, forward/backward passes, device handling and checkpoint round trip."""

from __future__ import annotations

import pytest
import torch

from brain_seg.config import ModelConfig
from brain_seg.inference import load_trained_model
from brain_seg.labels import NUM_CLASSES
from brain_seg.model import UNet3D, build_model, count_parameters
from brain_seg.utils import atomic_torch_save, resolve_amp, select_device
from conftest import available_devices, tiny_config


@pytest.mark.parametrize("in_channels", [1, 4])
def test_unet3d_output_shape(in_channels):
    model = UNet3D(in_channels, NUM_CLASSES, features=[4, 8, 16, 32]).eval()
    x = torch.randn(2, in_channels, 32, 24, 16)  # non-cubic, divisible by 2**3
    with torch.no_grad():
        y = model(x)
    assert y.shape == (2, NUM_CLASSES, 32, 24, 16)
    assert torch.isfinite(y).all()


def test_unet3d_is_volumetric_and_has_skip_connections():
    model = UNet3D(4, 4, features=[4, 8, 16])
    convs = [m for m in model.modules() if isinstance(m, torch.nn.Conv3d)]
    assert convs and all(c.kernel_size in [(3, 3, 3), (1, 1, 1)] for c in convs)
    # Decoder blocks receive concatenated skip + upsampled features (2x channels).
    assert [d[0].in_channels for d in model.decoders] == [16, 8]
    assert len(model.encoders) == 2 and model.bottleneck[0].out_channels == 16


def test_unet3d_rejects_bad_shapes():
    model = UNet3D(4, 4, features=[4, 8, 16])
    with pytest.raises(ValueError, match="divisible by 4"):
        model(torch.randn(1, 4, 18, 16, 16))
    with pytest.raises(ValueError, match="Expected input"):
        model(torch.randn(1, 3, 16, 16, 16))


@pytest.mark.parametrize("name", ["unet3d", "monai_unet"])
def test_build_model_variants(name):
    model = build_model(ModelConfig(name=name, features=[4, 8, 16]), in_channels=4, out_channels=4)
    y = model(torch.randn(1, 4, 16, 16, 16))
    assert y.shape == (1, 4, 16, 16, 16)
    assert count_parameters(model) > 0


@pytest.mark.parametrize("device_name", available_devices())
def test_forward_backward_on_available_devices(device_name):
    device = torch.device(device_name)
    model = UNet3D(4, 4, features=[4, 8, 16]).to(device)
    x = torch.randn(1, 4, 16, 16, 16, device=device)
    model(x).mean().backward()
    grads = [p.grad for p in model.parameters()]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)


def test_select_device():
    assert select_device("cpu").type == "cpu"
    expected = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    assert select_device("auto").type == expected
    if not torch.cuda.is_available():
        with pytest.raises(RuntimeError, match="CUDA was requested"):
            select_device("cuda")
    if not torch.backends.mps.is_available():
        with pytest.raises(RuntimeError, match="MPS was requested"):
            select_device("mps")


def test_amp_only_on_cuda():
    assert resolve_amp(True, "float16", torch.device("cpu")) is False
    assert resolve_amp(True, "float16", torch.device("mps")) is False
    assert resolve_amp(False, "float16", torch.device("cuda")) is False


def test_checkpoint_save_and_load(tmp_path):
    from brain_seg.data import BUILTIN_PROFILES
    from brain_seg.losses import build_loss
    from brain_seg.train import build_checkpoint, build_optimizer, build_scheduler

    cfg = tiny_config(None)
    model = build_model(cfg.model, 4, NUM_CLASSES)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    # one real optimisation step so optimizer state is non-trivial
    loss = build_loss(cfg.loss, NUM_CLASSES)(model(torch.randn(1, 4, 16, 16, 16)),
                                            torch.randint(0, 4, (1, 1, 16, 16, 16)))
    loss.backward()
    optimizer.step()
    state = build_checkpoint(model, optimizer, scheduler, scaler, cfg, BUILTIN_PROFILES["brats2021"],
                             epoch=3, best_metric=0.5, best_epoch=2, history=[{"epoch": 1, "train_loss": 1.0}],
                             patience_counter=1)
    path = tmp_path / "checkpoints" / "last.pt"
    path.parent.mkdir()
    atomic_torch_save(state, path)
    assert not path.with_suffix(".pt.tmp").exists()

    loaded, loaded_cfg, profile, ckpt = load_trained_model(path, torch.device("cpu"))  # weights_only=True
    assert ckpt["epoch"] == 3 and ckpt["best_epoch"] == 2
    assert profile.label_map == {0: 0, 1: 1, 2: 2, 4: 3}
    assert loaded_cfg.model.features == cfg.model.features
    for (k, a), (_, b) in zip(model.state_dict().items(), loaded.state_dict().items()):
        assert torch.equal(a, b), k
    new_opt = build_optimizer(loaded, loaded_cfg)
    new_opt.load_state_dict(ckpt["optimizer_state"])
    assert new_opt.state_dict()["state"].keys() == optimizer.state_dict()["state"].keys()
