# configs/

Versioned experiment configuration. Everything that changes an experimental
outcome lives here, never in notebook cells or ad-hoc command-line flags.

| File | Purpose |
| --- | --- |
| `project.yaml` | Project-level settings: seed, split ratios, declared dataset. |
| `split_search.yaml` | Phase 5C.1 split-search protocol (see below). |
| `split_freeze.yaml` | Phase 5C.2 split-freeze protocol. |
| `task_materialization.yaml` | Phase 5D canonical COCO task-dataset materialisation. |
| `detection_adapter.yaml` | Phase 6A YOLO detection adapter. |
| `detection_baseline.yaml` | Phase 6A D0 protocol: model, every hyperparameter, metric hierarchy. |
| `detection_experiments.yaml` | Phase 7A comparison protocol: D0/D1/D2, support rule, margin (see below). |

## Rules

- One file per experiment. Configuration files are append-only in spirit: to run
  a variant, add a new file instead of editing a file whose results are already
  reported.
- Parsing is strict (`construction_safety_vision.config`). An unknown or
  misspelled key raises instead of being silently ignored.
- Every run embeds its configuration snapshot in a provenance record, so a
  reported number can always be traced back to the exact settings that produced it.
- Hyperparameter files for detection and segmentation are added by phases 6-9.

## `split_search.yaml`

The phase 5C.1 split-search protocol: target ratios and integer image counts,
the hard constraints a candidate must satisfy, the rare class to protect and its
per-split floors, the allocation families to search, the objective weights and
the search parameters. Parsed strictly - an unknown key raises rather than being
ignored, so a typo cannot silently change the protocol.

It defines how candidates are *searched for and scored*. It does not select one
and it freezes nothing. The provider's split is absent by design.

## `detection_experiments.yaml`

The phase 7A controlled-comparison protocol, frozen while D1 and D2 did not yet
exist: the class-support rule that decides which classes may order two models,
the primary selection metric, the mandatory all-class metric, the
practical-equivalence margin, the batch/memory policy and the two authorised
experiments.

Its shape is the point. D1 and D2 carry **no protocol of their own**: they name
`inherits: D0` plus a set of dotted-path `overrides`, and D0's protocol is read
from `detection_baseline.yaml`. So every shared hyperparameter is stated once
and cannot drift between experiments, and "only one thing differs" is checkable
rather than asserted - the parser rejects a candidate whose `overrides` do not
exactly match its declared `intentional_fields` plus `consequential_fields`.

Parsing is strict in two extra ways beyond unknown keys: the support thresholds
and the margin must equal the frozen constants in
`construction_safety_vision.detection_comparison`, and the metric names must be
the frozen ones. Relaxing a threshold or renaming the deciding metric after a
result exists is exactly the failure the file is meant to prevent, so it raises
rather than loads.
