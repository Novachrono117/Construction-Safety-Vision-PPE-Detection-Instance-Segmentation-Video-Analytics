# notebooks/

The delivery notebook is
[`construction_safety_vision_demo.ipynb`](construction_safety_vision_demo.ipynb).
It offers two modes:

- **Mode A (default):** architecture, published final metrics, validation FP/FN
  figures and recorded video frames. No checkpoints or project dependency install.
- **Mode B (opt-in):** locked environment installation, exact D2/S1 checkpoint
  uploads and SHA-256 verification, then inference on an uploaded external video.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Novachrono117/Construction-Safety-Vision-PPE-Detection-Instance-Segmentation-Video-Analytics/blob/main/notebooks/construction_safety_vision_demo.ipynb)

See [Colab setup and boundaries](../delivery/COLAB.md) and the
[Phase 14A report](../reports/academic_colab_delivery.md). Local implementation
checks, real Mode A execution and real Mode B inference are separate evidence
categories; both cloud modes passed at revision
`8ce5d0375903e3e3760873e0a75ddce37cc1149b`.

The canonical notebook is unchanged and clones `main`. The badge targets its
production path and becomes usable after the final Phase 14A commit is pushed
to `main`. The temporary validation branch is not the final badge target.
The notebook's prepublication warning is preserved historical wording from the
validated execution surface. Full MP4 publication remains pending.

## Rules

- Notebooks are a thin execution surface, not where logic lives. Substantive
  code belongs in `src/construction_safety_vision/` and is imported here, so the
  same code path runs locally and in Colab and can be tested.
- The default notebook must run top to bottom on a fresh Colab runtime, starting
  from a clone of this repository. Only optional inference installs dependencies;
  Mode A requires no project packages, checkpoints, credentials or private files.
- Configuration comes from `configs/`, not from values typed into cells.
- Notebooks must not read the `test` split. The single final evaluation is a
  script run, recorded and reviewable.
- Clear outputs that embed large binaries before committing; keep diffs readable.

## Historical notebook plan (scientific lifecycle now closed)

The following table records the original plan. It does not authorize training,
evaluation or data access now, and these historical notebooks were not delivered.

| Notebook | Phase | Purpose |
| --- | --- | --- |
| Dataset audit and EDA | 4 | Class distribution, annotation integrity, duplicate and leakage checks. |
| Detection training | 6-7 | Fine-tuning runs on Colab GPU. |
| Segmentation training | 8-9 | Instance segmentation runs on the same splits. |
| Evaluation and error analysis | 10-12 | Metrics, confusion matrix, qualitative FP/FN review. |
| Video inference | 13 | End-to-end inference over the demonstration video. |
