#!/usr/bin/env bash
# One-time setup on a rented Linux + NVIDIA machine (RunPod / Vast.ai / Lambda ...).
# Usage:  bash scripts/cloud_setup.sh
# Assumes a PyTorch image (CUDA build of torch preinstalled). It does NOT download any data.
set -euo pipefail

command -v nvidia-smi >/dev/null || { echo "ERROR: nvidia-smi not found: this machine has no NVIDIA driver/GPU."; exit 1; }
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

python - <<'PY'
import sys, torch
print("python", sys.version.split()[0], "| torch", torch.__version__, "| CUDA build", torch.version.cuda)
if not torch.cuda.is_available():
    sys.exit("ERROR: torch cannot see the GPU. Use an image with a CUDA build of PyTorch, or install one from pytorch.org.")
print("GPU:", torch.cuda.get_device_name(0), f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB",
      "| bf16 supported:", torch.cuda.is_bf16_supported())
PY

pip install -q -e ".[dev,download]"
pytest -q -x
echo "Setup OK. Next: download the data (scripts/download_brats.py), inspect it, run scripts/smoke_test.py."
