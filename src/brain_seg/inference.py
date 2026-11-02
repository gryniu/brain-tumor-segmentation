"""Full-volume inference: checkpoint loading, sliding-window prediction, NIfTI export.

The network is trained on patches, but every MRI volume is larger than a patch.
At inference time the preprocessed volume is tiled with overlapping windows of
the training patch size (``inference.overlap``); window logits are blended with
a Gaussian weight that down-weights window borders, where context is missing.

The resulting softmax probabilities live in the preprocessed space (RAS, target
spacing, cropped). They are mapped back to the original voxel grid with MONAI
``Invertd`` (undo crop -> resampling -> reorientation) and only then converted
to labels with argmax, so saved masks align voxel-for-voxel with the input
images and can be compared directly against reference masks.

Run as a script::

    python scripts/predict.py --checkpoint outputs/runs/<run>/checkpoints/best.pt \
        --input-dir /path/to/case_dir --output-dir outputs/predictions
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import torch
from monai.data import MetaTensor
from monai.inferers import sliding_window_inference
from monai.transforms import Compose, Invertd
from torch import nn

from .config import Config, config_from_dict
from .data import DatasetProfile, DatasetValidationError, check_case_geometry, discover_cases
from .labels import CLASS_NAMES, NUM_CLASSES, REGION_CLASSES, REGIONS, invert_label_map
from .model import build_model
from .transforms import build_eval_transforms
from .utils import amp_context, get_logger, resolve_amp, select_device, setup_logging

DISCLAIMER = (
    "Research/educational model output - not a medical device, not clinically validated, "
    "and not to be used for diagnosis or treatment decisions."
)


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    # weights_only=True: checkpoints only contain tensors and plain Python containers,
    # so loading never executes arbitrary pickled code.
    return torch.load(path, map_location=map_location, weights_only=True)


def load_trained_model(
    checkpoint_path: str | Path, device: torch.device
) -> tuple[nn.Module, Config, DatasetProfile, dict[str, Any]]:
    """Rebuild the network from the config stored in the checkpoint and load its weights."""
    ckpt = load_checkpoint(checkpoint_path)
    for key in ("model_state", "config", "dataset_profile"):
        if key not in ckpt:
            raise ValueError(f"{checkpoint_path} is not a brain_seg checkpoint (missing '{key}')")
    cfg = config_from_dict(ckpt["config"])
    profile = DatasetProfile.from_dict(ckpt["dataset_profile"])
    model = build_model(cfg.model, len(cfg.data.modalities), NUM_CLASSES)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return model, cfg, profile, ckpt


@torch.no_grad()
def sliding_window_logits(
    model: nn.Module, image: torch.Tensor, cfg: Config, device: torch.device, use_amp: bool
) -> torch.Tensor:
    """Logits (B, K, *spatial) for a (B, C, *spatial) volume of arbitrary size."""
    stitch_device = torch.device("cpu") if cfg.inference.stitch_on_cpu else device
    with amp_context(device, use_amp, cfg.train.amp_dtype):
        logits = sliding_window_inference(
            inputs=torch.as_tensor(image).to(device),
            roi_size=cfg.train.patch_size,
            sw_batch_size=cfg.inference.sw_batch_size,
            predictor=model,
            overlap=cfg.inference.overlap,
            mode=cfg.inference.blend_mode,
            sw_device=device,
            device=stitch_device,
        )
    if isinstance(logits, MetaTensor):
        logits = logits.as_tensor()  # plain tensor: metadata is tracked on the input image
    return logits.float()


def predict_case(
    model: nn.Module,
    data: dict[str, Any],
    transforms: Compose,
    cfg: Config,
    device: torch.device,
    use_amp: bool,
) -> np.ndarray:
    """Predict a canonical label map (uint8) in the *original* voxel grid of the input files.

    ``data`` is the output of ``transforms`` for a single case (un-batched), so
    ``data["image"]`` is a MetaTensor carrying the applied spatial operations.
    """
    image = data["image"]
    if not isinstance(image, MetaTensor) or "spatial_shape" not in image.meta:
        raise TypeError("predict_case expects the MetaTensor produced by build_eval_transforms")
    logits = sliding_window_logits(model, image.unsqueeze(0), cfg, device, use_amp)
    # Must be a MetaTensor: for plain tensors Invertd only warns and skips the inversion.
    probabilities = MetaTensor(torch.softmax(logits, dim=1)[0].cpu())
    inverter = Invertd(keys="pred", transform=transforms, orig_keys="image", nearest_interp=False)
    inverted = torch.as_tensor(inverter({**data, "pred": probabilities})["pred"])
    original_shape = tuple(int(s) for s in image.meta["spatial_shape"])
    if tuple(inverted.shape[1:]) != original_shape:
        raise RuntimeError(
            f"Inverting preprocessing gave shape {tuple(inverted.shape[1:])}, expected the original "
            f"{original_shape}; predictions would not align with the input images"
        )
    return inverted.argmax(dim=0).numpy().astype(np.uint8)


def save_label_nifti(
    label_map: np.ndarray, reference_path: str | Path, out_path: str | Path,
    canonical_to_raw: dict[int, int] | None = None,
) -> Path:
    """Save a label map with the affine and header of ``reference_path``."""
    reference = nib.load(str(reference_path))
    ref_shape = tuple(reference.shape[:3])
    if tuple(label_map.shape) != ref_shape:
        raise ValueError(f"Label map shape {label_map.shape} != reference image shape {ref_shape}")
    out = label_map.astype(np.uint8)
    if canonical_to_raw is not None:
        lut = np.zeros(max(NUM_CLASSES, max(canonical_to_raw) + 1), dtype=np.uint8)
        for canonical, raw in canonical_to_raw.items():
            lut[canonical] = raw
        out = lut[out]
    header = reference.header.copy()
    header.set_data_dtype(np.uint8)
    image = nib.Nifti1Image(out, reference.affine, header)
    image.header.set_slope_inter(1, 0)
    sform, sform_code = reference.header.get_sform(coded=True)
    qform, qform_code = reference.header.get_qform(coded=True)
    if sform is not None:
        image.set_sform(sform, int(sform_code))
    if qform is not None:
        image.set_qform(qform, int(qform_code))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(image, str(out_path))
    return out_path


def summarize_label_map(label_map: np.ndarray, spacing: tuple[float, ...]) -> dict[str, Any]:
    """Voxel counts and volumes (mL) per canonical class and per BraTS region."""
    voxel_ml = float(np.prod(spacing)) / 1000.0
    counts = np.bincount(label_map.ravel(), minlength=NUM_CLASSES)
    classes = {
        CLASS_NAMES[c]: {"voxels": int(counts[c]), "volume_ml": round(float(counts[c]) * voxel_ml, 3)}
        for c in range(1, NUM_CLASSES)
    }
    regions = {}
    for region in REGIONS:
        voxels = int(sum(counts[c] for c in REGION_CLASSES[region]))
        regions[region] = {"voxels": voxels, "volume_ml": round(voxels * voxel_ml, 3)}
    return {"voxel_spacing_mm": list(spacing), "classes": classes, "regions": regions}


def run_inference(
    checkpoint: str | Path,
    input_dir: str | Path,
    output_dir: str | Path,
    device_name: str = "auto",
    case_id: str | None = None,
    label_convention: str = "dataset",
    make_figure: bool = True,
) -> dict[str, Any]:
    """Segment one patient folder and write NIfTI mask, JSON summary and overlay figure."""
    logger = get_logger()
    device = select_device(device_name)
    model, cfg, profile, ckpt = load_trained_model(checkpoint, device)
    discovery = discover_cases(
        input_dir, cfg.data.modalities, require_label=False, modality_aliases=profile.modality_aliases
    )
    if case_id is not None:
        cases = [c for c in discovery.cases if c.case_id == case_id]
        if not cases and case_id in discovery.invalid:
            raise DatasetValidationError(f"{case_id}: {'; '.join(discovery.invalid[case_id])}")
    else:
        cases = discovery.cases
    if not cases:
        problems = "; ".join(f"{k}: {', '.join(v)}" for k, v in discovery.invalid.items()) or "no NIfTI files"
        raise DatasetValidationError(
            f"No complete case in {input_dir}. Required modalities: {cfg.data.modalities} "
            f"(filename suffixes {profile.modality_aliases}). Problems: {problems}"
        )
    if len(cases) > 1:
        raise DatasetValidationError(
            f"{input_dir} contains {len(cases)} cases ({[c.case_id for c in cases][:5]}...); "
            "pass --case-id to choose one"
        )
    case = cases[0]
    case.label = None  # never feed a reference mask into inference
    _, spacing = check_case_geometry(case, cfg.data.modalities)

    transforms = build_eval_transforms(cfg, profile, with_label=False)
    data = transforms(case.to_monai(cfg.data.modalities))
    logger.info("Running sliding-window inference on %s (%s)", case.case_id, device)
    use_amp = resolve_amp(cfg.train.amp, cfg.train.amp_dtype, device)
    label_map = predict_case(model, data, transforms, cfg, device, use_amp)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reference_path = case.images[cfg.data.modalities[0]]
    if label_convention == "dataset":
        mapping, convention = invert_label_map(profile.label_map), f"{profile.name} label IDs"
    elif label_convention == "canonical":
        mapping, convention = None, "canonical IDs (0 BG, 1 NCR, 2 ED, 3 ET)"
    else:
        raise ValueError("label_convention must be 'dataset' or 'canonical'")
    nifti_path = save_label_nifti(label_map, reference_path, output_dir / f"{case.case_id}_pred.nii.gz", mapping)

    summary = {
        "case_id": case.case_id,
        "checkpoint": str(checkpoint),
        "checkpoint_epoch": ckpt.get("epoch"),
        "prediction_file": str(nifti_path),
        "label_convention": convention,
        "label_ids": mapping if mapping is not None else {c: c for c in range(NUM_CLASSES)},
        **summarize_label_map(label_map, spacing),
        "disclaimer": DISCLAIMER,
    }
    if make_figure:
        from .visualize import plot_prediction

        background = cfg.data.modalities[-1] if "flair" not in cfg.data.modalities else "flair"
        figure = plot_prediction(
            case.images[background], label_map, nib.load(str(reference_path)).affine,
            output_dir / f"{case.case_id}_overlay.png",
            title=f"{case.case_id} - predicted segmentation ({background.upper()} background)",
        )
        summary["figure"] = str(figure)
    summary_path = output_dir / f"{case.case_id}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    logger.info("Saved %s and %s", nifti_path, summary_path)
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Segment one patient's MRI folder with a trained checkpoint.")
    parser.add_argument("--checkpoint", required=True, help="Path to best.pt / last.pt")
    parser.add_argument("--input-dir", required=True, help="Folder with the patient's modality NIfTI files")
    parser.add_argument("--output-dir", default="outputs/predictions")
    parser.add_argument("--case-id", default=None, help="Case to use if the folder holds several")
    parser.add_argument("--device", default="auto", help="auto | cuda | cuda:N | mps | cpu")
    parser.add_argument(
        "--label-convention", choices=["dataset", "canonical"], default="dataset",
        help="'dataset' writes the training dataset's label IDs (e.g. ET=4 for BraTS 2021)",
    )
    parser.add_argument("--no-figure", action="store_true")
    args = parser.parse_args(argv)
    setup_logging()
    summary = run_inference(
        args.checkpoint, args.input_dir, args.output_dir, args.device, args.case_id,
        args.label_convention, not args.no_figure,
    )
    print(f"\nCase {summary['case_id']}  (voxel spacing {summary['voxel_spacing_mm']} mm)")
    for region, stats in summary["regions"].items():
        print(f"  {region}: {stats['volume_ml']:9.2f} mL  ({stats['voxels']} voxels)")
    print(f"Mask: {summary['prediction_file']}")
    print(DISCLAIMER)


if __name__ == "__main__":
    main()
