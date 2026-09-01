# notebooks/

Colab-executable notebooks. None exist yet: notebooks are written by the phase
that has something real to run, not created empty to fill the directory.

## Rules

- Notebooks are a thin execution surface, not where logic lives. Substantive
  code belongs in `src/construction_safety_vision/` and is imported here, so the
  same code path runs locally and in Colab and can be tested.
- Each notebook must run top to bottom on a fresh Colab runtime, starting from a
  clone of this repository plus a dependency install cell.
- Configuration comes from `configs/`, not from values typed into cells.
- Notebooks must not read the `test` split. The single final evaluation is a
  script run, recorded and reviewable.
- Clear outputs that embed large binaries before committing; keep diffs readable.

## Planned notebooks

| Notebook | Phase | Purpose |
| --- | --- | --- |
| Dataset audit and EDA | 4 | Class distribution, annotation integrity, duplicate and leakage checks. |
| Detection training | 6-7 | Fine-tuning runs on Colab GPU. |
| Segmentation training | 8-9 | Instance segmentation runs on the same splits. |
| Evaluation and error analysis | 10-12 | Metrics, confusion matrix, qualitative FP/FN review. |
| Video inference | 13 | End-to-end inference over the demonstration video. |
