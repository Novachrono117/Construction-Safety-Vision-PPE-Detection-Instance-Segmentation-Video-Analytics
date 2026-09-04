# Detection Runtime and Baseline Readiness - Phase 6A

Phase: 6A · Commit: `33430b10835fd3383e14de8a37a5f6052035cc35` · Runtime: **GPU_READY**

This report separates two things that reproducibility discussions often merge. The **software environment** is pinned and reproducible: exact package versions, locked. The **execution hardware** is recorded but not pinned - a driver version and a GPU model are facts about where a run happened, not dependencies someone else must match.

## 1. Software environment

| Component | Version |
| --- | --- |
| Python | 3.12.13 |
| torch | 2.11.0+cu128 |
| torchvision | 0.26.0+cu128 |
| ultralytics | 8.4.138 |
| CUDA runtime (torch) | 12.8 |
| cuDNN | 91900 |

`FACT` These are locked in `uv.lock`. torch and torchvision resolve from the CUDA 12.8
build index rather than PyPI, declared in `pyproject.toml`.

## 2. Execution hardware

| Property | Value |
| --- | --- |
| GPU | NVIDIA GeForce RTX 5070 Laptop GPU |
| Compute capability | 12.0 |
| VRAM | 7.96 GiB |
| Streaming multiprocessors | 36 |
| NVIDIA driver | 610.88 |
| Devices visible | 1 |

`FACT` Hardware is recorded, never made a package dependency. Reproducing the *software* environment does not require this GPU; reproducing the *timings* does.

## 3. Is the GPU actually usable?

`COMPUTED RESULT` `torch.cuda.is_available()` is a weak claim, so it is not the evidence here. This GPU is **sm_120** (Blackwell), and only CUDA 12.8 builds carry kernels for it; an older build would see the device and fail at the first matmul.

| Check | Result |
| --- | --- |
| `torch.cuda.is_available()` | True |
| compiled architectures | `sm_75, sm_80, sm_86, sm_90, sm_100, sm_120` |
| kernels present for sm_120 | True |
| real fp32 matmul executed | OK |
| max abs error vs CPU | 9.918212890625e-05 |
| AMP autocast + backward | OK |

`COMPUTED RESULT` The device executes real kernels and agrees with the CPU to float32 precision, and mixed precision works. The D0 protocol's `device_requirement: CUDA_GPU_REQUIRED` is satisfiable, and no CPU fallback was used or needed.

## 4. Pretrained weights

| Property | Value |
| --- | --- |
| Identifier | `yolo11n.pt` |
| Source | ULTRALYTICS_ASSET_DOWNLOAD (ultralytics 8.4.138) |
| Stored at | `artifacts/weights/yolo11n.pt` (git-ignored) |
| SHA-256 | `0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1` |
| Size | 5613764 bytes |

`FACT` Fingerprinted rather than trusted by name: two files called `yolo11n.pt` are not necessarily the same bytes, and a baseline that cannot name its starting weights cannot be reproduced. The binary itself is not committed.

## 5. Smoke test

`NON_EXPERIMENTAL` `DO_NOT_REPORT_AS_MODEL_RESULT` The smoke test exists only to prove the training stack executes: weights load, the dataset parses, forward and backward run on the GPU, the validation loader works, and checkpoints can be written.

| Property | Value |
| --- | --- |
| Result | **OK** |
| Epochs | 1 |
| Image size | 640 |
| Batch | 16 |
| Splits used | train, validation |
| Device | NVIDIA GeForce RTX 5070 Laptop GPU |
| Wall clock | 62.0 s |
| Peak GPU memory allocated | 2394.9 MiB |
| Peak GPU memory reserved | 2952.0 MiB |
| Checkpoint written | True |

**No metric from this run is recorded anywhere.** No AP, no loss, no precision or recall. It is not compared to anything, nothing was tuned from it, and its checkpoints are temporary engineering artifacts that are not committed. Reporting a one-epoch number as a model result would be inventing a finding.

## 6. What this phase did not do

`FACT` The full D0 baseline was **not run**. No hyperparameter was tuned against validation. No model result exists. The protected holdout was not materialised, adapted, read or measured, and `CSVISION_ALLOW_TEST_SPLIT` was not set; the Ultralytics dataset descriptor carries no `test` key at all.
