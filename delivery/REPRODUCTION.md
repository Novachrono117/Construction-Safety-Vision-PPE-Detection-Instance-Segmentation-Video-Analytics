# Quickstart and research reproduction

## A. Public quickstart

### Evidence notebook (Mode A) and optional inference (Mode B)

The [Colab notebook](../notebooks/construction_safety_vision_demo.ipynb) defaults
to displaying committed metrics, architecture, validation FP/FN and video figures
without weights, a GPU, or project dependency installation. Its optional Mode B
installs the locked runtime and accepts only uploaded D2/S1 bytes matching their
recorded hashes before running an external-video demo.

See [Colab setup](COLAB.md) and the [Phase 14A validation report](../reports/academic_colab_delivery.md).
Local implementation checks and both human-executed Colab modes passed, with
cloud evidence tied to `8ce5d0375903e3e3760873e0a75ddce37cc1149b`.
The canonical notebook targets `main`; its final main publication is pending.

GAP-004 (executable Colab) is RESOLVED under the human-approved two-mode
delivery scope; assignment requirement R18 is COMPLETE. Mode A reads existing
metrics and Mode B runs an external-video demo with uploaded frozen checkpoints.
Neither recomputes validation metrics or proves full clean-room research
reproduction. GAP-014 concerns that separate reproduction audit and remains OPEN.

### Repository tooling and existing CLI

**Phase 13B real-video demo complete locally; public distribution pending.** The
[video CLI](VIDEO.md) requires the exact local frozen checkpoints. Public
checkpoint distribution still needs license review, so a clean-clone inference
path is not yet self-contained. The following commands validate repository
tooling and delivery metadata.

Validated interpreter: **Python 3.12** (local runtime 3.12.14).
`.python-version` selects 3.12. The `>=3.11` requirement in `pyproject.toml` and
`uv.lock` is a historical resolver floor, not evidence of tested 3.11 support.
That dependency constraint and Ruff's older syntax target remain unchanged;
support for other Python versions is not claimed.

PowerShell or Bash, from the repository root after cloning:

```text
uv sync --locked
uv run python scripts/validate_public_delivery.py
uv run ruff check .
uv run ruff format --check .
uv run pytest --metadata-only
```

No dataset credential or `.env` is needed for metadata validation. Keep
`CSVISION_ALLOW_TEST_SPLIT` **unset**. Do not execute historical holdout commands.
The existing environment-inspection script reaches the split metadata accessor
and imports optional runtimes, so it is not part of the metadata-only quickstart.

### Hardware and test scope

The scientific training/inference path was validated on an **NVIDIA RTX 5070
Laptop GPU**, with PyTorch **2.11.0+cu128** and the **CUDA 12.8** build. The
project's Blackwell device (`sm_120`) required CUDA >=12.8-compatible kernels.
This is the tested environment, not a claim that this exact laptop is required.
See the [runtime report](../reports/detection_runtime_report.md) and
[controlled benchmark](../reports/detector_segmenter_latency_report.md).

The lockfile installs CUDA-flavoured wheels. Installing them and running lint,
metadata validation and synthetic unit tests do not require GPU execution, but
the dependencies still have a substantial download/disk cost. A GPU-free clean
installation has not been demonstrated. Some legacy integration tests depend on
ignored local artifacts; they do not constitute a clean-clone test guarantee.
The `--metadata-only` mode explicitly skips real-checkpoint integration tests
and blocks unexpected reads of ignored experiment/data files before opening
them. It leaves synthetic fixtures and committed text evidence available. The
Phase 12B run sets `PYTEST_ADDOPTS=--metadata-only` and executes `uv run pytest`.
Phase 13A verified CUDA and a one-frame CPU video integration smoke test.
CPU is explicitly requested; CUDA never silently falls back. This is functional
support, not equivalent speed or a broad hardware guarantee. See the
[video runtime report](../reports/video_runtime_foundation.md).

Phase 14A additionally observed Tesla T4 inference in Colab, using Python 3.12.3,
PyTorch 2.11.0+cu128 / CUDA 12.8, Ultralytics 8.4.138 and OpenCV 5.0.0.
That cloud validation covers `compare` on the recorded 125-frame external clip;
it is not a new benchmark, a CPU cloud validation or a hardware compatibility survey.

## B. Full research reproduction path

**Reproducible by design, not yet demonstrated from a clean clone.** This is a
route through the recorded pipeline, with executable stages and restrictions
distinguished. It is not permission to reopen the closed research experiment.
The final holdout evaluation is excluded from reproduction.

| Stage | Existing implementation / evidence | Provider / GPU | Current restriction |
| --- | --- | --- | --- |
| Environment | [pyproject.toml](../pyproject.toml), [uv.lock](../uv.lock), commands above | No provider; no GPU for metadata checks | Python 3.12 tested; clean-room run pending |
| Acquisition | [download_dataset.py](../scripts/download_dataset.py), [fetch_source_inventory.py](../scripts/fetch_source_inventory.py), [download_source_images.py](../scripts/download_source_images.py), [recover_source_geometry.py](../scripts/recover_source_geometry.py), [provenance](../reports/dataset_provenance.md) | Provider/network and author-supplied credential; CPU | Current live source may differ from the frozen snapshot; verify recorded hashes, never silently replace it |
| Canonicalization | [canonical decision](../reports/canonical_annotation_decision.md), [population](../reports/canonical_modeling_population_report.md), [build_modeling_population.py](../scripts/build_modeling_population.py) | Offline, CPU | Requires the exact ignored source inputs; preserve historical outputs |
| Split | [freeze report](../reports/split_freeze_report.md) | Offline, CPU | Reuse committed freeze through its guarded API; never rerun search or refreeze |
| Task materialization | [materialize_task_datasets.py](../scripts/materialize_task_datasets.py), [report](../reports/task_materialization_report.md) | Offline, CPU | Defaults to train/validation only; exact source images and modeling annotation inputs required |
| Detection adapter | [build_detection_adapter.py](../scripts/build_detection_adapter.py), [report](../reports/detection_adapter_report.md) | Offline, CPU | Derived from canonical development annotations; no holdout |
| Segmentation adapter | [fidelity audit](../reports/segmentation_adapter_fidelity_report.md), [approval](../reports/segmentation_adapter_approval.json) | Offline, CPU | Restore exact approved bytes; do not regenerate or rewrite the historical audit |
| Historical training | [D0 script](../scripts/train_detection_baseline.py), [D1/D2 script](../scripts/train_detection_experiment.py), [S0 script](../scripts/train_segmentation_baseline.py), [S1 script](../scripts/train_segmentation_comparison.py) | NVIDIA CUDA | CLOSED: archival implementations/configs, not a supported rerun or checkpoint retrieval route |
| Frozen checkpoints | [Public metadata](checkpoints.json), [distribution policy](LICENSING.md) | Verifier uses CPU, no model loading | Retrieval blocked pending review and publication; never substitute or retrain |
| Validation-only comparison | [comparison script](../scripts/compare_detector_segmenter.py), [frozen protocol](../reports/detector_segmenter_comparison_protocol.md), [result](../reports/detector_segmenter_validation_comparison.md) | NVIDIA CUDA | Future clean-room execution must isolate outputs and use frozen settings; not executed in 12B |
| Cost benchmark | [benchmark script](../scripts/benchmark_detector_segmenter.py), [report](../reports/detector_segmenter_latency_report.md) | NVIDIA CUDA, hardware-specific | Historical measurements remain immutable; another machine's measurement is separate evidence |
| Scientific synthesis | [Final synthesis](../reports/detector_segmenter_scientific_synthesis.md) | Read-only committed evidence | Do not reinterpret claims or rewrite historical metrics |

The materializer's existing development-only verification command is
`uv run python scripts/materialize_task_datasets.py --verify-only`; it requires
the documented ignored inputs and is not a clean-clone demo. Running the writer
also writes historical report locations, so it must not be used casually in this
closed evidence repository. Future clean-room work must first establish output
isolation, source retrieval and checkpoint retrieval, then document executed
commands and failures. GAP-009 remains partially resolved and GAP-014 remains open.

### Holdout publication boundary

Frozen holdout membership is deliberately committed for protocol auditability.
Phase 11 result/report artifacts omit test IDs; bulk holdout imagery and
predictions are not committed. Membership records are not a data-access route.
The final test is observed and spent; no second attempt or holdout re-access is
part of this path. Read the [published final report](../reports/final_test_evaluation.md)
and [execution accounting](../reports/final_test_execution_accounting.md).

### Historical optimizer capture (GAP-019, LOW)

`scripts/train_detection_baseline.py` retains its old optimizer log regex, which
does not strip the ANSI escape sequences. The shared
[detection_run.py](../src/construction_safety_vision/detection_run.py) capture
handles them. D0 is closed, and its optimizer determination is explicitly
recorded as inferred in its committed manifest. The historical runner is not
an active delivery utility. Phase 12B documents the defect and leaves its code
and all historical results unchanged; convergence remains a separate cleanup
action, not a reason to rerun D0. GAP-019 is partially resolved by classification.
