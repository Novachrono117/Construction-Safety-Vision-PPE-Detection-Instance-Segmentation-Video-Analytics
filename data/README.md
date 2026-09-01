# data/

Datasets are **never committed**. Only this documentation, `.gitkeep` markers and
small text provenance/manifest files are tracked (see `.gitignore`).

## Layout

| Directory | Contents | Mutability |
| --- | --- | --- |
| `raw/` | Exactly what was downloaded from the source, untouched. | Immutable. Never edited, never re-encoded, never manually fixed. |
| `interim/` | Intermediate artifacts of the preprocessing pipeline (parsed annotations, audit tables, duplicate reports). | Regenerable. |
| `processed/` | Task-ready datasets consumed by training: detection and segmentation views of the same frozen splits. | Regenerable. |
| `external/` | Assets that are not the primary dataset, e.g. the construction-site video used for inference. | Immutable. |

## Rules

1. `raw/` is read-only. Any correction to an annotation is expressed as code in
   the preprocessing pipeline, so it is reproducible and reviewable. Manual,
   undocumented edits are forbidden.
2. Everything under `processed/` must be re-derivable from `raw/` by running the
   pipeline. If it is not, the pipeline is incomplete.
3. Detection and segmentation datasets are two *views* of one canonical source.
   They share image IDs split by split; boxes are derived from polygons rather
   than annotated separately.
4. The `test` split is a locked holdout. See `CLAUDE.md` and
   `construction_safety_vision.splits`.
5. Every acquisition and every derivation writes a `*.provenance.json` record
   (source URL, version, license, file hashes, code commit, configuration).

## Current status

Empty. No dataset has been downloaded (foundation phase).
