# configs/

Versioned experiment configuration. Everything that changes an experimental
outcome lives here, never in notebook cells or ad-hoc command-line flags.

| File | Purpose |
| --- | --- |
| `project.yaml` | Project-level settings: seed, split ratios, declared dataset. |

## Rules

- One file per experiment. Configuration files are append-only in spirit: to run
  a variant, add a new file instead of editing a file whose results are already
  reported.
- Parsing is strict (`construction_safety_vision.config`). An unknown or
  misspelled key raises instead of being silently ignored.
- Every run embeds its configuration snapshot in a provenance record, so a
  reported number can always be traced back to the exact settings that produced it.
- Hyperparameter files for detection and segmentation are added by phases 6-9.
