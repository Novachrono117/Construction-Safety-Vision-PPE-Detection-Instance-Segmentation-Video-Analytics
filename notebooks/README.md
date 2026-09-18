# notebooks/

The delivery notebook is
[`construction_safety_vision_demo.ipynb`](construction_safety_vision_demo.ipynb).
It runs in an academic sequence - overview, dataset and classes, **TREINO**,
**AVALIACAO**, qualitative evidence, video evidence, **INFERENCIA**, limitations -
and offers two modes:

- **Mode A (default):** architecture, dataset and class inventory, the recorded
  training recipes and committed training curves, the committed final-test
  results with both confusion matrices, validation FP/FN figures and recorded
  video frames. No checkpoints, GPU or project dependency install.
- **Mode B (opt-in):** locked environment installation, exact D2/S1 checkpoint
  uploads and SHA-256 verification, then inference on an uploaded external video.

The training and evaluation sections are executable and render visible output,
but they **present recorded evidence**: the notebook does not retrain the models
and does not rerun the spent holdout evaluation, and offers no executable path
that would.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Novachrono117/Construction-Safety-Vision-PPE-Detection-Instance-Segmentation-Video-Analytics/blob/main/notebooks/construction_safety_vision_demo.ipynb)

See [Colab setup and boundaries](../delivery/COLAB.md) and the
[Phase 15C report](../reports/academic_colab_delivery.md). Local implementation
checks, real Mode A execution and real Mode B inference are separate evidence
categories. Mode A passed on a fresh CPU runtime at revision
`958eefaa58b785ff2f9eb4dd2751efc26dd9b09e`; Mode B passed at revision
`8ce5d0375903e3e3760873e0a75ddce37cc1149b` and was not re-executed, because every
Mode B execution input is byte-identical between the validated revisions and
phase 15C changed no code cell.

The canonical notebook clones `main`. The badge targets its production path and
becomes usable after the final commit is pushed to `main`. The temporary
validation branches are not the final badge target. Open the notebook from GitHub
rather than from a saved Drive copy: a Phase 14A session copy carries an
uncommitted GPU diagnostic cell that fails on a CPU runtime.
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
