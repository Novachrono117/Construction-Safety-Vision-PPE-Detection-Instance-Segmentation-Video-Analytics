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

## `split_search.yaml`

The phase 5C.1 split-search protocol: target ratios and integer image counts,
the hard constraints a candidate must satisfy, the rare class to protect and its
per-split floors, the allocation families to search, the objective weights and
the search parameters. Parsed strictly - an unknown key raises rather than being
ignored, so a typo cannot silently change the protocol.

It defines how candidates are *searched for and scored*. It does not select one
and it freezes nothing. The provider's split is absent by design.
