# Canonical Annotation Snapshot Decision

Generated: 2026-09-04T12:36:12+00:00 · Phase: 5A · Commit: `10ff78972f857c97a1c56fe10249d95578f447e6`

## Decision: `CURRENT_COMPLETE_GEOMETRY`

Phase 5A classification: **COMPLETE_CURRENT**.

**Question.** Which reproducible annotation state becomes the source of truth every later phase derives from? This decides nothing about images, splits or labels - only which annotations are authoritative.

## The two candidates

| | Version-4 export | Live source project |
| --- | --- | --- |
| Source images | 436 | 436 |
| Annotations | 1961 | 2031 |
| Coordinate frame | 640x640, stretch-resized | original image pixels |
| Geometry | polygon + compressed RLE | polygon + compressed RLE |
| Frozen | yes, hash-verified archive | no, provider may edit again |
| Augmented copies present | yes, 306 in train | not applicable |

## What was established

### The live state can be recovered read-only, with complete geometry (FACT)

Phase 4A recorded 1,022 live annotations as carrying a bounding box and no usable geometry, which would have ruled the live state out. That was a consumption gap, not a provider limitation: a `mask`-type annotation carries its geometry inline as base64-wrapped, zlib-compressed COCO run-length encoding in a field the earlier walk did not read. Decoded against the full original canvas, it is complete segmentation geometry.

No mutating call was made. The recovery uses one documented read-only endpoint.

| Geometry classification | Annotations |
| --- | --- |
| `BITMASK_OR_MASK_ASSET` | 0 |
| `OTHER_SUPPORTED` | 0 |
| `POLYGON` | 1007 |
| `RLE` | 1022 |
| `UNSUPPORTED` | 2 |
| **total** | **2031** |

### The decode is verified, not assumed (FACT)

The encoding is undocumented, so every recovered instance was re-measured with the reference COCO implementation and checked against the provider's own declared area and bounding box. Agreement across the whole population:

| Representation | Boxes agreeing within 1 px | Max deviation |
| --- | --- | --- |
| `POLYGON` | 1007/1007 | 0.0 px |
| `RLE` | 1022/1022 | 0.0 px |

Exact agreement on every one of the 2,029 annotations that carry geometry is what makes the decode trustworthy. A wrong decode would not reproduce the provider's own areas and boxes to the pixel.

### Version 4 remains fully usable as a snapshot (FACT)

All 436 source images map to exactly one non-augmented version-4 representation (436 of 436), 435 by exact filename and one after Unicode normalisation. The export README states that augmentation produced two versions of each source image; measurement contradicts that wording - in every train pair one record reproduces the deterministic preprocessing re-render closely and the other does not. So option A was viable, and was not rejected for being unavailable.

### What the drift actually is (FACT, corrected in phase 5B)

The live state holds +70 annotations relative to the snapshot, which conceals 76 additions and 6 removals across 21 images with a positive delta. Every one of the 76 additions lies at least 80% inside an annotation of its own class that the snapshot already had, and is under half its area. `vest_loose`, the rare class the split design turns on, is identical in both states.

**This report originally concluded from that measurement that the additions add no coverage and are fragments. That conclusion was withdrawn in phase 5B.** The measurement is correct; the inference was not. The additions include legitimate instance splits - in one image a single oversized `person` box covering two people, replaced by one box per person - alongside genuinely degenerate slivers. A coarse over-merged parent contains its own corrections by definition, so containment could not distinguish the two. No automatic filter was adopted and all annotations are retained; see `reports/fragment_rule_report.md`.

See `reports/annotation_drift_report.md` and `reports/figures/review_n_added_annotations.jpg`.

### The two geometry-less records are resolved (FACT)

Phase 4A found two live records carrying a class and a box but no geometry, and could not say what they were. They are resolved here rather than dropped.

| Annotation | Class | Box (px) | v4 counterpart | v4 geometry | Classification |
| --- | --- | --- | --- | --- | --- |
| `OQJwjQoY` id `I` | `helmet_on_head` | 7.9x6.0 at (921.3, 0.3) | yes, IoU 1.0 | polygon, 5 vertices (a rectangle derived from the box) | `VALID_BUT_UNSUPPORTED_GEOMETRY` |
| `OQJwjQoY` id `J` | `vest_on_body` | 13.3x15.4 at (909.3, 6.0) | yes, IoU 0.9999 | polygon, 5 vertices (a rectangle derived from the box) | `VALID_BUT_UNSUPPORTED_GEOMETRY` |

Both fall on the same real, distant, partially occluded worker at the top edge of the image - a helmet and a high-visibility vest - which `reports/figures/review_m_microannotations.jpg` shows directly. Neither box is degenerate and neither leaves the canvas, so neither is malformed.

The version-4 counterpart matters for what it is *not*: its polygon is a four-corner ring whose area equals its own bounding box exactly. It is a rectangle the exporter generated from the box, not a mask anyone drew. Adopting the live state therefore loses no mask information for these two records - there was never any to lose. Phase 5B can materialise the same rectangle, and must do so explicitly.

## Why the live state was chosen

* the live source state was recovered read-only for all 436 source images and all 2031 annotations, of which 2029 carry complete instance-segmentation geometry in original image coordinates
* 2 record(s) carry no segmentation geometry; they are counted and classified, never dropped

The decisive argument is coordinate fidelity, not recency. The live state is expressed in original image pixels. The version-4 geometry is expressed after a stretch resize to 640x640, so adopting it would require inverting that resize for **all** 1,961 annotations - and its RLE masks were rasterised at 640x640, so the inversion cannot recover what rasterisation discarded. Choosing the live state avoids a systematic geometric degradation across the entire dataset.

The live state is *not* chosen for having more annotations. It has 76 more, and the measurement above shows those 76 add nothing: they subdivide objects that were already labelled. That is a defect the snapshot does not have. It is accepted here because it is identifiable by a reproducible rule and can be removed by an explicit, recorded pipeline step in phase 5B, whereas a stretch resize applied to every annotation cannot be undone.

Newer is not treated as more correct. Nothing in this phase measures either state against a ground truth, and the report says so.

## Limitations carried into phase 5B

* Two live-source records (image OQJwjQoYsf1KUgr9G0V8, annotations I and J) carry a class and a bounding box but no segmentation geometry. Visual review shows both fall on a real distant worker, so they are valid objects with unsupported geometry. They are counted in the 2031 total and must be handled explicitly by phase 5B, not silently dropped.
* All 76 annotations added since version 4 lie at least 80% inside an annotation of their own class that the snapshot already had, and are under half its area (median 0.44%). RESOLVED IN PHASE 5B: this was originally read as evidence that they add no coverage and are fragments, and that reading was withdrawn. They are a mixture of degenerate slivers and legitimate corrections - instance splits, coarse annotations replaced by several precise ones, and geometry refinements - which no geometric rule separates. Automatic filtering was rejected (fragment_rule_status = REJECTED_FOR_AUTOMATIC_FILTERING) and all 2031 annotations are retained.
* The canonical state is the provider's live project, which can change again. Its reproducibility rests on the recorded recovery method plus the committed artifact hashes, not on the provider freezing anything.
* The provider's stored bounding boxes agree with the recovered geometry to 0.0 px in the live state, but phase 4B measured them disagreeing by up to 123.5 px inside the version-4 export. Detection boxes must still be derived from segmentation, as the constitution requires.
* Annotation correctness itself was not assessed against any independent ground truth. No such reference exists for this dataset.

## Reproducing this

```bash
uv run python scripts/recover_source_geometry.py
uv run python scripts/map_v4_sources.py
uv run python scripts/analyze_annotation_drift.py
uv run python scripts/build_drift_figures.py
uv run python scripts/resolve_canonical_snapshot.py
```

Artifact fingerprints are recorded in `reports/canonical_annotation_manifest.json`.

## What this decision does NOT do

It does not create a split, exclude an image, group duplicates, freeze a holdout, generate model-ready labels, or convert any geometry between coordinate frames. Those belong to phases 5B onward.

