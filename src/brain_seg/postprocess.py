"""Post-processing of predicted label maps.

Two simple, standard clean-ups of a canonical label map (0 BG, 1 NCR, 2 ED, 3 ET):

* **small components**: connected groups of tumour voxels (26-connectivity) smaller than
  ``min_component_voxels`` are removed. Isolated specks are rarely real tumour but cost precision and HD95.
* **small enhancing tumour**: if the predicted ET has fewer than ``et_min_voxels`` voxels it is relabelled
  as necrosis/non-enhancing core (NCR). A tiny enhancing prediction is usually noise, and relabelling
  turns a false-positive ET into an (almost always) harmless TC voxel set. This follows a trick used by
  BraTS challenge participants.

Both thresholds default to 0 (disabled). They are tuned on the VALIDATION set with
``scripts/tune_postprocessing.py``; they must never be tuned on the test set.
"""

from __future__ import annotations

import numpy as np

ET, NCR = 3, 1


def remove_small_components(label_map: np.ndarray, min_voxels: int) -> np.ndarray:
    """Set whole-tumour connected components with fewer than ``min_voxels`` voxels to background."""
    from scipy import ndimage

    out = label_map.copy()
    if min_voxels <= 0 or not out.any():
        return out
    components, n = ndimage.label(out > 0, structure=np.ones((3, 3, 3), dtype=bool))
    if n == 0:
        return out
    sizes = np.bincount(components.ravel())
    small = sizes < min_voxels
    small[0] = False  # background is not a component
    out[small[components]] = 0
    return out


def relabel_small_et(label_map: np.ndarray, min_voxels: int) -> np.ndarray:
    """Relabel ET as NCR when the predicted ET volume is below ``min_voxels`` (0 < ET < min_voxels)."""
    out = label_map.copy()
    if min_voxels <= 0:
        return out
    et = out == ET
    count = int(et.sum())
    if 0 < count < min_voxels:
        out[et] = NCR
    return out


def postprocess_labels(
    label_map: np.ndarray, et_min_voxels: int = 0, min_component_voxels: int = 0
) -> np.ndarray:
    """Apply component removal, then the small-ET rule; returns a new uint8 array."""
    out = remove_small_components(np.asarray(label_map), min_component_voxels)
    out = relabel_small_et(out, et_min_voxels)
    return out.astype(np.uint8)
