# Annotation Drift: live source project vs version-4 snapshot

Generated: 2026-09-04T12:36:12+00:00 · Phase: 5A · Commit: `10ff78972f857c97a1c56fe10249d95578f447e6`

**Question.** The live source project holds more annotations than the frozen version-4 export. Is that difference a set of pure additions, does it conceal removals, and does it change what the dataset can teach?

## Method (COMPUTED)

Each of the 436 source images is compared against **one** export record: the non-augmented representation resolved in `reports/v4_source_mapping.csv`. The augmented train copies are excluded - they are not a second observation of the annotation state.

Boxes are compared in a normalised frame, divided by each side's own image dimensions. The export's preprocessing is a pure stretch resize to 640x640, so a normalised box is invariant under it and no coordinate inversion is performed.

Export boxes are **derived from segmentation**, not read from the stored `bbox` field, which `reports/bbox_consistency_audit.md` measured disagreeing with its own geometry by up to 123.5 px. Live boxes are the provider's stored values, which the recovery step measured to agree with the recovered geometry to 0.0 px.

Annotations are paired greedily by overlap at IoU >= 0.5, same-class pairs first so that a relabel is not masked. Export annotations whose segmentation could not be converted and fell back to the stored box: **0**.

**Caveat on the shape signal (FACT).** The two sides are rasterised at different resolutions - the live geometry in full original pixels, the snapshot's at 640x640. One pixel of quantisation is a large fraction of a small object and negligible for a large one, so a fixed overlap threshold partly measures the export's resolution rather than editing. The size breakdown below is reported so this is visible, and the reshaped count must be read as an upper bound, not as an edit count.

## Result

| | Value |
| --- | --- |
| Source images compared | 436 |
| Live annotations | 2031 |
| Version-4 snapshot annotations | 1961 |
| Net delta | +70 |
| Gross additions | 76 |
| Gross removals | 6 |
| Images with unchanged annotation count | 415 |
| Images with a positive delta | 21 |
| Images with a negative delta | 0 |
| Images touched in any way | 67 |
| Matched pairs whose class changed | 0 |
| Matched pairs below IoU 0.95 (upper bound on reshaping) | 220 |
| Large objects below IoU 0.80 (quantisation cannot explain) | 8 |

### Per-image outcome (COMPUTED)

| Status | Images |
| --- | --- |
| `added_and_removed` | 6 |
| `added_only` | 15 |
| `modified_in_place` | 46 |
| `unchanged` | 369 |

### Matched-pair overlap (COMPUTED)

Across 1955 matched pairs:

| Overlap below | Pairs |
| --- | --- |
| 0.99 | 694 |
| 0.95 | 220 |
| 0.90 | 75 |
| 0.80 | 16 |
| 0.70 | 6 |
| 0.50 | 0 |

Broken down by how large the object is in the snapshot's own 640x640 frame:

| Snapshot object area (px^2) | Pairs | Below IoU 0.95 | Median IoU |
| --- | --- | --- | --- |
| 0-256 | 61 | 40 (65.6%) | 0.888 |
| 256-1024 | 162 | 101 (62.3%) | 0.9376 |
| 1024-4096 | 418 | 61 (14.6%) | 0.982 |
| 4096-16384 | 406 | 2 (0.5%) | 1.0 |
| 16384+ | 908 | 16 (1.8%) | 0.9986 |

### Drift by class (COMPUTED)

| Class | Added | Removed | Net |
| --- | --- | --- | --- |
| `helmet_loose` | 1 | 0 | +1 |
| `person` | 33 | 1 | +32 |
| `vest_on_body` | 42 | 5 | +37 |
| **total** | **76** | **6** | **+70** |

Classes with **no** drift in either direction: `helmet_on_head`, `vest_loose`.

`vest_loose` is called out explicitly because phase 4B recorded it as the rare, low-diversity class whose handling the split design turns on. Its annotation count is identical in both states, so choosing between them neither helps nor harms that class.

### Do the additions cover objects that were not labelled before? (COMPUTED)

For every added annotation, how much of it lies inside the best-enclosing annotation **of its own class** that the snapshot already had, and how large it is relative to that annotation. An addition wholly inside an existing same-class annotation, at a small fraction of its size, cannot be a newly covered object.

| | Value |
| --- | --- |
| Additions examined | 76 |
| Lying >= 80% inside an existing same-class annotation | 76 |
| Covering an object the snapshot did not annotate | 0 |
| Median size relative to the annotation enclosing it | 0.44% |

| Addition smaller than this share of its container | Count |
| --- | --- |
| 1% | 40 |
| 5% | 49 |
| 10% | 55 |
| 25% | 64 |
| 50% | 76 |

Visual evidence: `reports/figures/review_n_added_annotations.jpg` shows every addition as a zoomed crop.

### Are the edited images unusually crowded? (COMPUTED)

| Group | Images | Mean snapshot annotations per image |
| --- | --- | --- |
| Touched by drift | 67 | 9.70 |
| Untouched | 369 | 3.55 |

### Drift by the provider's split (COMPUTED)

The provider's split was rejected for the final protocol in phase 4B. It is reported here only to show where in the existing organisation the edits fell.

| Split | Images | Changed | Added | Removed | Net |
| --- | --- | --- | --- | --- | --- |
| `test` | 43 | 10 | 14 | 1 | +13 |
| `train` | 306 | 43 | 60 | 5 | +55 |
| `valid` | 87 | 14 | 2 | 0 | +2 |

### Most-edited images (COMPUTED)

| Source image | Split | Live | v4 | Delta | Added | Removed |
| --- | --- | --- | --- | --- | --- | --- |
| `ljmzAWSjDGWvNtmeIhi5` | `train` | 22 | 5 | +17 | `person` 17 | - |
| `dwz8H6tUE91TOzWAqabt` | `train` | 13 | 3 | +10 | `person` 8, `vest_on_body` 2 | - |
| `qpGvhFrN8YelHx1yTsM9` | `test` | 9 | 3 | +6 | `vest_on_body` 6 | - |
| `CDbWzCeakwV73xXd2qlq` | `train` | 6 | 3 | +3 | `vest_on_body` 4 | `vest_on_body` 1 |
| `Y1meTyz9qAlTOKqcMF21` | `train` | 6 | 3 | +3 | `vest_on_body` 4 | `vest_on_body` 1 |
| `lHkj3xNmUL3G67dHG1Zr` | `test` | 8 | 3 | +5 | `vest_on_body` 5 | - |
| `EMcfZIuO2gmvi4krBUxS` | `train` | 5 | 3 | +2 | `person` 2, `vest_on_body` 1 | `person` 1 |
| `RRlPrUJXfRWGGJ6CeKIL` | `train` | 5 | 3 | +2 | `vest_on_body` 3 | `vest_on_body` 1 |
| `WvGmeXrUqULTXbaSGJgZ` | `train` | 5 | 3 | +2 | `vest_on_body` 3 | `vest_on_body` 1 |
| `5p4hDwmBcrn3EI04GmlA` | `test` | 4 | 3 | +1 | `vest_on_body` 2 | `vest_on_body` 1 |
| `MkxXQ8wmfbTbRbgDBAK1` | `train` | 6 | 3 | +3 | `person` 3 | - |
| `q2wyXKnABjAqqL0mLiG7` | `train` | 6 | 3 | +3 | `vest_on_body` 3 | - |
| `zIjMWx3EGNSbPhJ6eyDE` | `train` | 6 | 3 | +3 | `person` 3 | - |
| `oL1x933NyN6wiKXhxOob` | `train` | 5 | 3 | +2 | `vest_on_body` 2 | - |
| `ywH2LFvRJto6sV68hk5v` | `valid` | 5 | 3 | +2 | `vest_on_body` 2 | - |

## Reading (interpretation, labelled)

**FACT.** The two states differ by +70 annotations, but the net figure is not the amount of change: 76 annotations were added and 6 removed across 67 of 436 images.

**FACT.** Every source image resolves to exactly one non-augmented version-4 representation, so both annotation states describe the same 436 images.

**LIKELY.** Most of the 220 matched pairs below IoU 0.95 reflect the export's 640x640 rasterisation rather than editing. The size breakdown is the evidence: overlap degrades sharply as objects get smaller, which is what quantisation does and what editing has no reason to do. Only 8 large objects fall below IoU 0.80, where quantisation cannot be the explanation. This is not a controlled test, so the attribution stays LIKELY rather than FACT.

**FACT.** Every one of the 76 additions lies at least 80% inside an annotation of its own class that the snapshot already had, and every one is under half that annotation's area (median 0.44%).

**CORRECTION (phase 5B).** An earlier version of this report inferred from the measurement above that the additions therefore add no class coverage and are fragments. **That inference was wrong and is withdrawn.** The measurement stands; what does not follow from it is the conclusion. Phase 5B inspected the affected images and found the additions include legitimate corrections:

* a single oversized `person` box covering **two** people, replaced by one box per person - which is new instance coverage;
* a coarse `vest_on_body` polygon replaced by several tighter ones covering the parts of the vest actually visible;
* geometry refinements.

Containment inside an older same-class annotation looked like evidence of redundancy, but a coarse over-merged parent contains its own corrections by definition. See `reports/fragment_rule_report.md`.

**FACT.** Some additions are genuinely degenerate - a few dozen pixels on a bracelet, on glove lettering, on a hard hat. Both kinds are present, and no geometric rule separates them, which is why no automatic filter was adopted and all 2031 annotations are retained.

**UNKNOWN.** Whether the live annotations are *more correct* than the snapshot's overall. They are newer, and newer is not a measurement. Nothing here compares either state against an independent ground truth, and no such reference exists for this dataset.

