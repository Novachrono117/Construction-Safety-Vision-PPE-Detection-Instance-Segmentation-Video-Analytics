# data/

Datasets are **never committed**. Only this documentation, `.gitkeep` markers and
small text provenance/manifest files are tracked (see `.gitignore`).

## Layout and semantics

| Directory | Contents | Mutability |
| --- | --- | --- |
| `external/` | Artifacts **as received from the external provider** - the downloaded export archive with its `*.provenance.json` record, and `source_images/`, the 436 source originals at full resolution. | Immutable. Never edited, never re-compressed. |
| `raw/` | The **extracted canonical dataset representation**, exactly as it came out of the archive. Input to every later step. | Immutable. Never edited, never manually fixed. |
| `interim/` | Derived and audited representations produced by the audit phase (parsed annotations, audit tables, duplicate reports). | Regenerable. |
| `processed/` | Model-ready datasets: the canonical COCO detection and instance-segmentation views of the frozen development splits. Images are byte-identical copies of the canonical originals. | Regenerable. |

`interim/` currently holds `source_image_stats.jsonl` and `source_annotations.jsonl`
(phase 4A measurements) and `source_geometry.jsonl` (phase 5A recovered annotation
geometry, in original image coordinates) and `modeling_annotations.jsonl` (phase
5B eligible annotations with their canonical class index). All are bulk data: git-ignored and
re-derivable by running the scripts named in `scripts/README.md`. What is
committed instead is the counts-only summary under `reports/`, together with the
file hashes recorded in `reports/canonical_annotation_manifest.json`.

`processed/` holds the canonical task datasets built by phase 5D:

```text
processed/canonical/
├── images/
│   ├── train/         303 source originals, byte-identical copies
│   └── validation/     65 source originals, byte-identical copies
└── annotations/
    ├── detection_train.coco.json
    ├── detection_validation.coco.json
    ├── segmentation_train.coco.json
    └── segmentation_validation.coco.json
```

**`train` and `validation` only.** There is no `images/test/` and no
`*_test.coco.json`: the holdout is not materialised while the models are
unfrozen. The final-evaluation phase materialises it through the same code path,
which requires both holdout opt-ins.

Images here are **binary copies** of the canonical originals in `external/` - no
resize, crop, re-encode, EXIF rotation or colour conversion - and both sides are
hashed after the copy, so byte-identity is measured rather than assumed. The COCO
documents preserve the canonical geometry unchanged: a polygon stays a polygon
and a compressed RLE stays a compressed RLE. Detection boxes are **derived from
the segmentation**, never copied from the provider.

Phase 6A adds a derived model-specific view alongside it:

```text
processed/adapters/yolo_detection/
├── images/{train,val}/     byte-identical copies of the canonical images
├── labels/{train,val}/     YOLO detection labels, empty file = valid negative
└── dataset.yaml            Ultralytics descriptor, resolved paths, no `test` key
```

That adapter is **derived, never canonical**: if it and the COCO file disagree,
the COCO file is right. It carries detection boxes only - no segmentation labels
exist in any model format - and there is no `test` directory in it either.

All of it is git-ignored and re-derivable:

```bash
uv run python scripts/materialize_task_datasets.py   # canonical COCO datasets
uv run python scripts/build_detection_adapter.py     # derived YOLO detection view
```

What is committed instead is `reports/task_dataset_manifest.json` and
`reports/task_materialization_report.md`, which carry the counts, the
fingerprints and the validation results. Membership always comes from
`reports/split_manifest.json`, read through
`construction_safety_vision.data.split_freeze.load_frozen_splits`, never
re-derived.

## Rules

1. `external/` and `raw/` are read-only. Any correction to an annotation is
   expressed as code in the preprocessing pipeline, so it is reproducible and
   reviewable. Manual, undocumented edits are forbidden.
2. Everything under `interim/` and `processed/` must be re-derivable from `raw/`
   by running the pipeline. If it is not, the pipeline is incomplete.
3. Detection and segmentation datasets are two *views* of one canonical source.
   They share image IDs split by split; boxes are derived from polygons rather
   than annotated separately.
4. The `test` split is a locked holdout, frozen by name in
   `reports/split_manifest.json` since phase 5C.2. Reading it requires
   `allow_test=True` in code **and** `CSVISION_ALLOW_TEST_SPLIT=1` in the
   environment; neither alone is enough, and reading the manifest's `test`
   section directly to avoid the guard defeats the point of having one. See
   `CLAUDE.md`, `construction_safety_vision.splits` and
   `construction_safety_vision.data.split_freeze`.
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
