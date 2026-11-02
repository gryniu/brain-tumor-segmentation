"""Inspect a BraTS-style dataset before training.

Reports discovered/incomplete cases, unrecognised files, the detected label
convention (from real mask values), geometry (shapes, spacings, orientations),
patient grouping and label voxel statistics. Nothing is modified.

Examples:
    python scripts/inspect_dataset.py --data-root /data/BraTS2021_Training_Data
    python scripts/inspect_dataset.py --config configs/default.yaml --max-label-cases 0   # all masks
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import nibabel as nib
import numpy as np

from brain_seg.config import load_config
from brain_seg.data import (
    DatasetProfile,
    DatasetValidationError,
    assign_patient_ids,
    check_case_geometry,
    detect_profile,
    discover_cases,
    read_label_values,
    resolve_profile,
)
from brain_seg.labels import CLASS_NAMES
from brain_seg.utils import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--data-root", default=None, help="Overrides data.root")
    parser.add_argument("--profile", default=None, help="Force auto|brats2021|brats2023|custom")
    parser.add_argument("--max-label-cases", type=int, default=25,
                        help="Masks to read for label statistics (0 = all; reading all takes minutes)")
    parser.add_argument("--report", default="outputs/dataset_report.json")
    args = parser.parse_args()
    setup_logging()

    overrides = [f"data.profile={args.profile}"] if args.profile else []
    cfg = load_config(args.config, overrides)
    root = args.data_root or cfg.data.root
    if not root:
        raise SystemExit("Pass --data-root or set data.root in the config")
    custom_aliases = (DatasetProfile.from_dict(cfg.data.custom_profile).modality_aliases
                      if cfg.data.profile == "custom" else None)
    result = discover_cases(root, cfg.data.modalities, require_label=True, modality_aliases=custom_aliases)
    print(f"\nDataset root: {Path(root).resolve()}")
    print(f"Complete cases: {len(result.cases)} | incomplete: {len(result.invalid)} | "
          f"unrecognised NIfTI files: {len(result.unrecognized_files)}")
    for case_id, problems in list(result.invalid.items())[:20]:
        print(f"  INCOMPLETE {case_id}: {'; '.join(problems)}")
    for path in result.unrecognized_files[:10]:
        print(f"  UNRECOGNISED {path}")
    if not result.cases:
        raise SystemExit("No complete cases found - check the directory structure (docs/reference.md 'Dataset').")

    detected, evidence = detect_profile(result.cases)
    profile = resolve_profile(cfg, result.cases)
    print(f"\nLabel convention detected from masks: {detected.name} ({evidence})")
    if profile.name != detected.name:
        print(f"  WARNING: config uses profile '{profile.name}' but masks suggest '{detected.name}'")
    print(f"Profile in use: {profile.name}")
    for raw, canonical in sorted(profile.label_map.items()):
        print(f"  raw {raw} -> {CLASS_NAMES[canonical]:3s}  {profile.label_descriptions.get(raw, '')}")
    assign_patient_ids(result.cases, profile)
    patients = Counter(c.patient_id for c in result.cases)
    multi = sum(1 for n in patients.values() if n > 1)
    print(f"\nPatients: {len(patients)} ({multi} with more than one case/timepoint - kept together in splits)")

    shapes: Counter = Counter()
    spacings: Counter = Counter()
    orientations: Counter = Counter()
    geometry_errors = []
    for case in result.cases:
        try:
            shape, spacing = check_case_geometry(case, cfg.data.modalities)
        except DatasetValidationError as exc:
            geometry_errors.append(str(exc))
            continue
        shapes[shape] += 1
        spacings[tuple(round(s, 3) for s in spacing)] += 1
        orientations["".join(nib.aff2axcodes(nib.load(str(case.label)).affine))] += 1
    print(f"\nGeometry check: {len(result.cases) - len(geometry_errors)} consistent, {len(geometry_errors)} errors")
    for err in geometry_errors[:10]:
        print(f"  ERROR {err.splitlines()[0]}")
    print(f"  shapes: {dict(shapes.most_common(5))}")
    print(f"  voxel spacing (mm): {dict(spacings.most_common(5))}")
    print(f"  orientation codes: {dict(orientations)}")

    sample = result.cases if args.max_label_cases == 0 else result.cases[: args.max_label_cases]
    totals: Counter = Counter()
    presence: Counter = Counter()
    unknown_cases = []
    for case in sample:
        values = read_label_values(case.label)
        totals.update(values)
        presence.update(v for v, n in values.items() if n > 0)
        if set(values) - set(profile.label_map):
            unknown_cases.append((case.case_id, sorted(set(values) - set(profile.label_map))))
    tumour = sum(n for v, n in totals.items() if v != 0)
    print(f"\nLabel statistics over {len(sample)} mask(s):")
    for value in sorted(totals):
        share = totals[value] / max(1, tumour) if value else float("nan")
        name = CLASS_NAMES.get(profile.label_map.get(value, -1), "UNKNOWN")
        print(f"  label {value} ({name}): present in {presence[value]}/{len(sample)} cases"
              + (f", {100 * share:.1f}% of tumour voxels" if value else ""))
    if unknown_cases:
        print(f"  ERROR: label values outside the profile in {len(unknown_cases)} case(s): {unknown_cases[:5]}")

    first = result.cases[0]
    stats = {}
    for modality in cfg.data.modalities:
        data = np.asanyarray(nib.load(str(first.images[modality])).dataobj).astype(np.float32)
        brain = data[data > 0]
        stats[modality] = {"dtype": str(data.dtype), "brain_voxels": int(brain.size),
                           "p1": float(np.percentile(brain, 1)) if brain.size else None,
                           "p99": float(np.percentile(brain, 99)) if brain.size else None}
    print(f"\nRaw intensity range of {first.case_id} (brain voxels, before z-scoring):")
    for modality, s in stats.items():
        print(f"  {modality:6s} p1={s['p1']:.1f} p99={s['p99']:.1f}")

    report = {
        "root": str(Path(root).resolve()),
        "n_complete_cases": len(result.cases),
        "incomplete_cases": result.invalid,
        "unrecognized_files": [str(p) for p in result.unrecognized_files],
        "detected_profile": detected.name,
        "detection_evidence": evidence,
        "profile_in_use": profile.name,
        "n_patients": len(patients),
        "geometry_errors": geometry_errors,
        "shapes": {str(k): v for k, v in shapes.items()},
        "spacings": {str(k): v for k, v in spacings.items()},
        "orientations": dict(orientations),
        "label_voxels": {str(k): v for k, v in totals.items()},
        "label_presence": {str(k): v for k, v in presence.items()},
        "label_cases_read": len(sample),
        "unknown_label_cases": unknown_cases,
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2))
    print(f"\nReport written to {args.report}")
    if geometry_errors or unknown_cases or result.invalid:
        print("Problems found - fix them (or set data.skip_invalid_cases: true for incomplete cases) before training.")


if __name__ == "__main__":
    main()
