# Dataset report: BraTS 2023 Adult Glioma training set

Measured with `scripts/inspect_dataset.py` and `scripts/smoke_test.py` on Colab (Tesla T4).

## Dataset

| Item | Value |
|---|---|
| Source | Synapse syn51514132 (training archive), CC-BY-NC 4.0 |
| Complete cases | 1251 (0 incomplete) |
| Patients | 1133; 118 with more than one time point (kept together in splits) |
| Files per case | `t1n`, `t1c`, `t2w`, `t2f`, `seg` |
| Shape / spacing / orientation | (240, 240, 155), 1 mm isotropic, LPS, for all cases |
| Split (patient level, seed 42) | train 874, validation 183, test 194 |

## Labels (`brats2023` profile)

| Value | Class |
|---|---|
| 0 | background |
| 1 | NCR: non-enhancing tumour core / necrosis |
| 2 | ED: surrounding FLAIR hyperintensity (oedema) |
| 3 | ET: enhancing tumour |

On a sample of 25 masks: NCR in 22, ED in 25 and ET in 24 cases; NCR / ED / ET make up 16.5% / 64.6% / 18.9% of
tumour voxels. In sampled training patches background is about 91-95% of voxels.

## Tesla T4 benchmark (`cloud_cuda.yaml`, 16.6M parameters, 128³ patches)

| Item | Value |
|---|---|
| Peak GPU memory, one training step | 4.78 GB |
| Time per step (2 patches) | 2.19 s, of which 1.51 s waiting for data |
| `disk` cache size (pilot run) | ~160 MB per case (8.4 GB for ~50 cases) |

With 2 vCPUs the GPU waits for data; the `npy` cache addresses this. `colab_t4_1h.yaml` (5.65M parameters, 96³)
measured 490 s for the first epoch (cache build) and ~78 s for later epochs over 250 cases.
