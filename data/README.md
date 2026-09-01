# data/

Datasets are **never committed**. Only this documentation, `.gitkeep` markers and
small text provenance/manifest files are tracked (see `.gitignore`).

## Layout and semantics

| Directory | Contents | Mutability |
| --- | --- | --- |
| `external/` | The immutable artifact **as received from the external provider** - the downloaded export archive, plus its `*.provenance.json` record. | Immutable. Never edited, never re-compressed. |
| `raw/` | The **extracted canonical dataset representation**, exactly as it came out of the archive. Input to every later step. | Immutable. Never edited, never manually fixed. |
| `interim/` | Derived and audited representations produced by the audit phase (parsed annotations, audit tables, duplicate reports). | Regenerable. |
| `processed/` | Model-ready datasets: the detection and segmentation views of the frozen splits. | Regenerable. |

`interim/` and `processed/` are **reserved and currently empty**. Phase 3 does not
write to them, and no derived detection labels exist yet.

## Rules

1. `external/` and `raw/` are read-only. Any correction to an annotation is
   expressed as code in the preprocessing pipeline, so it is reproducible and
   reviewable. Manual, undocumented edits are forbidden.
2. Everything under `interim/` and `processed/` must be re-derivable from `raw/`
   by running the pipeline. If it is not, the pipeline is incomplete.
3. Detection and segmentation datasets are two *views* of one canonical source.
   They share image IDs split by split; boxes are derived from polygons rather
   than annotated separately.
4. The `test` split is a locked holdout. See `CLAUDE.md` and
   `construction_safety_vision.splits`.
5. Every acquisition and every derivation writes a `*.provenance.json` record
   (source URL, version, license, file hashes, code commit, configuration).

## Current status - dataset acquired (phase 3)

The canonical COCO instance-segmentation export has been acquired. Full evidence:
[`reports/dataset_provenance.md`](../reports/dataset_provenance.md).

| | |
| --- | --- |
| Source | Roboflow Universe, `agis-workspace-8gs52/construction-ppe-compliance-detection`, version 4 |
| Format | `coco-segmentation` |
| License | CC BY 4.0 (attribution required) |
| Archive | `construction-ppe-compliance-detection-v4-coco-segmentation.zip`, 51 867 677 bytes |
| Archive SHA-256 | `5c0c35f79be251af349f289ab300a8c706f260f9a5b52f66facfbd23466538f6` |
| Extracted to | `raw/construction-ppe-compliance-detection-v4-coco-segmentation/` |

> **The version has 742 images, but only 436 independent source images.** The
> train split is offline-augmented x2 (306 source -> 612); validation and test are
> unchanged. Never treat the 742 as independent samples.

## How to obtain the data

The dataset is not redistributed here. To reproduce this directory:

```bash
export ROBOFLOW_API_KEY=...      # never commit this; see .env.example
uv run python scripts/download_dataset.py
uv run python scripts/inspect_dataset.py
```

The download is idempotent: an archive whose digest matches the recorded one is
reused rather than fetched again, and an archive with a different digest is never
silently overwritten. If the recorded SHA-256 above does not match what you
download, the provider changed the export - stop and investigate before using it.
