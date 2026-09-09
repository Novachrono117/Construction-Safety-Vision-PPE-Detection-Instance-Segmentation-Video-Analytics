# Phase 8A - YOLO Instance-Segmentation Adapter Fidelity Audit

Phase: 8A · Commit: `c44d17f3c84703c4749976aa5f253172f187d232` · Classification: **SEGMENTATION_ADAPTER_AUDIT_COMPLETE**

**No segmentation model was trained, evaluated or downloaded, and no architecture was selected.** This phase measures one thing: how much of the project's canonical COCO instance-mask geometry survives being expressed as Ultralytics YOLO segmentation labels. The holdout was not read, adapted, converted, counted or inspected.

Claims are labelled `FACT` (measured here), `CANONICAL_DATA_POLICY` (frozen earlier and inherited), `MODEL_FORMAT_CONSTRAINT` (a property of the installed framework), `COMPUTED_RESULT` (this audit's arithmetic), `APPROXIMATION` (information the format cannot carry), `LIMITATION` and `PENDING_HUMAN_DECISION`.

## 1. Audit objective

`PENDING_HUMAN_DECISION` The project's canonical annotations are COCO instance masks, half of them stored as compressed RLE. Before any segmentation architecture is chosen, one question has to be answered with numbers rather than assumption: **would a YOLO segmentation label still describe the same object?**

`PENDING_HUMAN_DECISION` This phase answers that and stops. It does not approve YOLO segmentation and it does not reject it. A high IoU distribution is not an approval; a low one is not a rejection. The architecture decision belongs to a reviewed human step in phase 8B, and this report exists to inform it.

## 2. Why model-format fidelity matters

`MODEL_FORMAT_CONSTRAINT` A COCO instance mask can express things a single YOLO segmentation row cannot: an object in several disconnected pieces, an object with a hole through it, a boundary at arbitrary sub-pixel precision. If those are silently flattened during conversion, every mask metric the project later reports is measured against labels that are **not** the annotations anyone reviewed - and the loss would be invisible, because the model would be trained and evaluated against the same degraded labels.

`LIMITATION` This audit measures **representation**, not model performance. It says nothing about how well any model would learn from these labels.

## 3. Canonical segmentation source

`CANONICAL_DATA_POLICY` The source of truth is `COCO_INSTANCE_SEGMENTATION`, materialised in phase 5D. It was read and never modified: no canonical COCO document was rewritten, no provider mask was reinterpreted, and no provider box or provider preprocessing was used. What this phase produced is a **derived** representation.

`FACT` Development population: **368 images** and **1726 annotations**, verified against the canonical documents rather than assumed.

| Split | Images | Annotations | Instance rows written | Negative images |
| --- | --- | --- | --- | --- |
| `train` | 303 | 1422 | 1422 | 10 |
| `validation` | 65 | 304 | 304 | 2 |

`FACT` Canonical geometry, counted from the documents:

| Canonical representation | Annotations |
| --- | --- |
| polygon | 843 |
| compressed RLE | 881 |
| synthetic rectangle | 2 |
| **total** | **1726** |

`CANONICAL_DATA_POLICY` The two synthetic rectangles are the phase 5B materialisation of the annotations that carried no geometry at all. They are stored as polygons but are not human-drawn segmentation, so they are reported as their own stratum rather than left to inflate the polygon population.

## 4. Ultralytics segmentation-format constraints

`MODEL_FORMAT_CONSTRAINT` Established by reading the installed `ultralytics 8.4.138` source on this machine (`INSTALLED_PACKAGE_SOURCE_INSPECTION`), not from documentation.

| Property | Finding |
| --- | --- |
| Row structure | class then a flat sequence of x y pairs, all normalised to the image width and height. ultralytics/data/utils.py::verify_image_label parses the tail as np.array(fields[1:], dtype=np.float32).reshape(-1, 2). |
| One row per instance | true |
| Row is a single ring | true |
| Multi-segment separator | NONE. The format has no separator between rings and no ring-role marker, so an instance made of several components cannot be written as several rings in one row. |
| Holes representable | **false** |
| Coordinate parsing | float32. Coordinates are bounds-checked into [-0.01, 1.01]; a row outside that range makes the whole image-label pair corrupt. |

`MODEL_FORMAT_CONSTRAINT` **Holes.** Two independent confirmations in the installed source. convert_segment_masks_to_yolo_seg extracts contours with cv2.RETR_EXTERNAL, which returns outer boundaries only and discards interior contours by construction. And polygon2mask fills a single ring with cv2.fillPoly, which has no even-odd subtraction, so anything the outer boundary encloses is filled.

`MODEL_FORMAT_CONSTRAINT` **Multiple components.** ultralytics/data/converter.py::convert_coco calls merge_multi_segment when a COCO annotation carries more than one polygon, bridging components along their nearest points with a zero-width connector so that one instance stays one row. This audit uses that same primitive.

`MODEL_FORMAT_CONSTRAINT` **Internal resampling.** Yes, at training time: YOLODataset.update_labels_info calls resample_segments with n=1000, raised to max_len+1 when a segment already has more points. Because resample_segments inserts the original vertices into the interpolation grid and never downsamples below the input length under that rule, it densifies a path rather than simplifying it, and introduces no additional geometric loss. It is a training-time transformation and is not part of the label format.

`MODEL_FORMAT_CONSTRAINT` **Rasterisation.** polygon2mask casts coordinates with np.asarray(..., dtype=np.int32), which truncates toward zero, then fills with cv2.fillPoly. The truncation is a real source of sub-pixel loss and this audit measures it rather than avoiding it.

`MODEL_FORMAT_CONSTRAINT` **A cardinality hazard.** After parsing, verify_image_label derives each row's bounding box with segments2boxes and removes duplicate class-plus-box rows via np.unique. Two distinct instances of the same class sharing a bounding box would silently collapse into one. The audit checks for that collision explicitly rather than assuming it cannot happen. Collisions found in this development set: **0**.

`MODEL_FORMAT_CONSTRAINT` **Negatives.** An image with an empty label file is counted as a background image; an image with no label file at all is counted as a missing label. Negatives therefore require empty files, not absent ones.

## 5. Development population

`CANONICAL_DATA_POLICY` The audit adapter uses exactly the development images the canonical segmentation documents use - none added, none dropped - and the frozen class map, by fingerprint:

- `class_map_sha256` `596dab5b5756e3d4253924f84bf46d37a87d80b6ec830da781a08a93cd823753`
- `split_assignment_sha256` `a230869ff4cb45f53654d67357a27f2a0def6d8ba79f9fa4880f0fdda2f046cc`
- placeholder category `object`: absent, and no index shifted

`FACT` Zero-instance images are retained as empty label files: 10 in train and 2 in validation. There is no holdout directory and no holdout key in the dataset descriptor.

## 6. Conversion strategy

`FACT` **One canonical annotation becomes exactly one YOLO row.** 1726 annotations produced 1726 instance rows; 0 were split, 0 merged and 0 dropped.

`CANONICAL_DATA_POLICY` That invariant is not a convenience. Splitting a disconnected mask into several rows would raise every fidelity number in this report and would quietly redefine what an instance is - changing per-image object counts and invalidating any later detection-versus-segmentation comparison.

`FACT` Images were transferred without transformation: HARDLINK, with every adapter image's SHA-256 verified equal to the canonical image's. No resize, no re-encode, no colour conversion.

## 7. Polygon handling

`FACT` A canonical polygon keeps its own coordinates. Re-tracing it from a raster would discard precision the canonical file already holds, so nothing is re-derived and no `approxPolyDP` simplification is applied - shrinking the label file is not worth boundary detail.

## 8. RLE handling

`FACT` Compressed RLE is decoded with `PYCOCOTOOLS_REFERENCE_DECODER`, the same reference decoder the canonical masks were built with in phases 5A-5D. A private decoder that disagreed with it would have made every number in this report a measurement of that disagreement.

`FACT` The decoded mask's boundary is then traced with `RETR_EXTERNAL` and `CHAIN_APPROX_SIMPLE`, epsilon simplification `NONE`. Those are the settings Ultralytics' own mask converter uses, so the audit measures the framework's behaviour rather than a private variant.

## 9. Disconnected components

`FACT` Connected components counted at connectivity 8:

| Components | Instances |
| --- | --- |
| 1 | 1419 |
| 2 | 144 |
| 3 | 57 |
| 4 | 32 |
| 5 | 23 |
| 6 | 15 |
| 7 | 11 |
| 8 | 3 |
| 9 | 7 |
| 10 | 3 |
| 13 | 1 |
| 15 | 1 |
| 16 | 3 |
| 17 | 1 |
| 19 | 2 |
| 21 | 1 |
| 29 | 1 |
| 48 | 1 |
| 78 | 1 |

`FACT` Instances with more than one component: **307**. Maximum in one instance: **78**.

`APPROXIMATION` `ULTRALYTICS_MERGE_MULTI_SEGMENT` - components are bridged into one path with a zero-width connector, classified `COMPONENT_JOIN_APPROXIMATION`. The connector adds area between the pieces. Section 15 reports what that costs on its own, separately from every other source of loss, so it is not hidden inside an aggregate mean.

## 10. Holes

`FACT` Instances whose canonical mask contains at least one interior ring: **180**. Total holes: **699**. Maximum in one instance: **117**. Total hole area: **1072133 px**. Largest single-instance hole-area fraction: **17.57%**.

`APPROXIMATION` `NOT_REPRESENTABLE_FILLED` - the format has no interior ring, so every hole is filled, classified `HOLE_FILL_APPROXIMATION`. The area above is what a conversion would add.

## 11. Round-trip methodology

`FACT` Every instance was measured against **the label as written to disk**, not against an in-memory object. The chain is: canonical mask → YOLO row → written file → re-read and parsed with the framework's own float32 semantics → denormalised → rasterised by `ULTRALYTICS_POLYGON2MASK` on the `ORIGINAL_SOURCE_IMAGE` canvas → compared.

`COMPUTED_RESULT` **pycocotools and OpenCV do not rasterise identical geometry into identical pixels.** They differ on boundary-pixel inclusion, so even a polygon carried through unchanged will not reproduce the canonical mask exactly. Reporting that gap as format loss would blame the format for a rasteriser convention, so each instance is measured at three levels and each stage is charged only for what it costs:

| Level | What it measures | Global mean |
| --- | --- | --- |
| `control_iou` | canonical vs OpenCV rasterisation of the rings **before** any conversion - the rasteriser and contour convention alone | 0.986368 |
| `merged_iou` | after components are bridged into one ring, still at float64 - adds the cost of **component joining** | 0.985525 |
| `mask_iou` | after serialisation, float32 parsing and the int32 snap - adds the cost of **serialisation and quantisation** | 0.973066 |

## 12. Global fidelity

`COMPUTED_RESULT` Over all **1726** development instances:

| Statistic | Mask IoU |
| --- | --- |
| minimum | 0.307692 |
| P01 | 0.857961 |
| P05 | 0.918176 |
| P25 | 0.966956 |
| median | 0.984576 |
| mean | 0.973066 |
| P75 | 0.992425 |
| P95 | 0.996446 |
| P99 | 0.997954 |
| maximum | 0.998987 |

`COMPUTED_RESULT` Descriptive bands. **These are not thresholds**, and phase 8A turns none of them into an approval or rejection rule:

| Band | Instances | Share |
| --- | --- | --- |
| `iou_exact` | 0 | 0.00% |
| `iou_ge_0.99` | 602 | 34.88% |
| `iou_0.95_to_0.99` | 872 | 50.52% |
| `iou_0.90_to_0.95` | 205 | 11.88% |
| `iou_lt_0.90` | 47 | 2.72% |

## 13. Fidelity by canonical representation

`COMPUTED_RESULT` The most important table in this report. An aggregate would let one representation vouch for the other:

| Stratum | Instances | Median area px | Mean IoU | Median IoU | P05 IoU | Min IoU | Mean rel. area err | P95 rel. area err |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `CANONICAL_POLYGON` | 843 | 40657 | 0.978091 | 0.989090 | 0.931939 | 0.307692 | 0.025094 | 0.069495 |
| `CANONICAL_RLE` | 881 | 21452 | 0.968538 | 0.977699 | 0.912740 | 0.743243 | 0.007974 | 0.032599 |
| `SYNTHETIC_RECTANGLE` | 2 | 129 | 0.849702 | 0.849702 | 0.770685 | 0.761905 | 0.189583 | 0.300208 |

## 14. Fidelity by class

`COMPUTED_RESULT` Reported for every class, including the rare one:

| Stratum | Instances | Median area px | Mean IoU | Median IoU | P05 IoU | Min IoU | Mean rel. area err | P95 rel. area err |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `helmet_loose` | 333 | 18970 | 0.981097 | 0.986574 | 0.943611 | 0.892553 | 0.015618 | 0.053581 |
| `helmet_on_head` | 267 | 10571 | 0.965756 | 0.979194 | 0.899357 | 0.698413 | 0.021743 | 0.074424 |
| `person` | 778 | 55914 | 0.971631 | 0.984853 | 0.911898 | 0.307692 | 0.015262 | 0.035930 |
| `vest_loose` | 38 | 70827 | 0.985071 | 0.990803 | 0.961158 | 0.877622 | 0.010336 | 0.027483 |
| `vest_on_body` | 310 | 51192 | 0.972865 | 0.985906 | 0.929144 | 0.636364 | 0.017052 | 0.057423 |

`LIMITATION` A class-level fidelity figure inherits that class's instance count. `vest_loose` carries few development instances, so its row describes those instances and should not be read as a property of the class.

## 15. Fidelity by topology

`COMPUTED_RESULT` By connected-component count:

| Stratum | Instances | Median area px | Mean IoU | Median IoU | P05 IoU | Min IoU | Mean rel. area err | P95 rel. area err |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 component(s) | 1419 | 26212 | 0.974257 | 0.985874 | 0.924840 | 0.307692 | 0.017741 | 0.052744 |
| 2 component(s) | 144 | 28847 | 0.964071 | 0.975346 | 0.908337 | 0.743243 | 0.010203 | 0.039033 |
| 3+ component(s) | 163 | 85511 | 0.970645 | 0.980858 | 0.904274 | 0.823036 | 0.011749 | 0.046362 |

`COMPUTED_RESULT` With and without interior holes:

| Stratum | Instances | Median area px | Mean IoU | Median IoU | P05 IoU | Min IoU | Mean rel. area err | P95 rel. area err |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `with_holes` | 180 | 148640 | 0.975249 | 0.985886 | 0.918736 | 0.823036 | 0.014120 | 0.058540 |
| `without_holes` | 1546 | 24690 | 0.972812 | 0.984307 | 0.917747 | 0.307692 | 0.016829 | 0.049713 |

`LIMITATION` **That comparison is confounded by size and must not be read as 'holes do not matter'.** Instances that have holes are much larger - median 148640 px against 24690 px - and section 16 shows mask area is the strongest driver of fidelity in this dataset. The two effects run in opposite directions here and the strata are not matched, so the near-equal IoU means the size advantage roughly offsets the hole-filling penalty, not that filling holes is free. The absolute cost is stated in section 10: 1072133 px added across 180 instances.

`COMPUTED_RESULT` Isolating each approximation, averaged over all instances:

- component joining costs **0.000843** mean IoU (max 0.084871)
- serialisation and quantisation cost **0.012458** mean IoU (max 0.171886)

## 16. Fidelity by mask size

`COMPUTED_RESULT` Quartiles of the development mask-area distribution: P25 5778 px, P50 31357 px, P75 135439 px.

`LIMITATION` Quartiles of the development mask-area distribution, computed from the data itself. Diagnostic bins for this audit only; they deliberately do not reuse or redefine the project's small-object EDA threshold, which is a different measurement.

| Stratum | Instances | Median area px | Mean IoU | Median IoU | P05 IoU | Min IoU | Mean rel. area err | P95 rel. area err |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `Q1_smallest` | 432 | 2264 | 0.935802 | 0.948461 | 0.885542 | 0.307692 | 0.042492 | 0.104642 |
| `Q2` | 431 | 14543 | 0.976787 | 0.979977 | 0.952552 | 0.856887 | 0.011534 | 0.027496 |
| `Q3` | 431 | 65610 | 0.987350 | 0.989650 | 0.972511 | 0.883105 | 0.007075 | 0.015231 |
| `Q4_largest` | 432 | 294643 | 0.992369 | 0.994710 | 0.983927 | 0.823036 | 0.005049 | 0.010502 |

`COMPUTED_RESULT` Thin-structure proxy (perimeter over the square root of area), reported as a covariate rather than as a definition of 'thin': median 5.090, P95 8.757, maximum 19.540.

## 17. Area error

`COMPUTED_RESULT` Absolute relative mask-area error: mean 0.016546, median 0.006224, P95 0.050637, P99 0.142804, maximum 2.250000.

| Exceeds | Instances | Share |
| --- | --- | --- |
| 1% | 610 | 35.34% |
| 2% | 282 | 16.34% |
| 5% | 87 | 5.04% |
| 10% | 29 | 1.68% |

`COMPUTED_RESULT` Total pixels added across the development set: 2035661. Total dropped: 333468. Against a canonical total of 220018740 px.

`LIMITATION` These bands are descriptive evidence, not a training-approval threshold.

## 18. Worst cases

`COMPUTED_RESULT` The 20 lowest mask IoU instances, selected deterministically. No source photograph was opened: the diagnosis is geometry-only.

| Image | Ann | Class | Geometry | Area px | Comp | Holes | IoU | Control | Reasons |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `dwz8H6tUE91TOzWAqabt` | 1118 | `person` | polygon | 4 | 1 | 0 | 0.3077 | 0.3636 | RASTER_BOUNDARY_DIFFERENCE, SMALL_MASK_SENSITIVITY |
| `dwz8H6tUE91TOzWAqabt` | 1115 | `person` | polygon | 10 | 1 | 0 | 0.4762 | 0.4737 | RASTER_BOUNDARY_DIFFERENCE, SMALL_MASK_SENSITIVITY |
| `dwz8H6tUE91TOzWAqabt` | 1112 | `person` | polygon | 8 | 1 | 0 | 0.5333 | 0.4444 | RASTER_BOUNDARY_DIFFERENCE, SMALL_MASK_SENSITIVITY |
| `dwz8H6tUE91TOzWAqabt` | 1109 | `vest_on_body` | polygon | 28 | 1 | 0 | 0.6364 | 0.6364 | RASTER_BOUNDARY_DIFFERENCE, SMALL_MASK_SENSITIVITY |
| `dwz8H6tUE91TOzWAqabt` | 1110 | `vest_on_body` | polygon | 6 | 1 | 0 | 0.6667 | 0.2143 | RASTER_BOUNDARY_DIFFERENCE, SMALL_MASK_SENSITIVITY |
| `q2wyXKnABjAqqL0mLiG7` | 1687 | `vest_on_body` | polygon | 59 | 1 | 0 | 0.6782 | 0.6588 | RASTER_BOUNDARY_DIFFERENCE, SMALL_MASK_SENSITIVITY |
| `OQJwjQoYsf1KUgr9G0V8` | 677 | `helmet_on_head` | polygon | 44 | 1 | 0 | 0.6984 | 0.7333 | RASTER_BOUNDARY_DIFFERENCE, SMALL_MASK_SENSITIVITY |
| `pE6UkMMWzcDOy86QHSoC` | 1631 | `person` | rle | 248 | 2 | 0 | 0.7432 | 1.0000 | MULTI_COMPONENT_APPROXIMATION, COMPONENT_JOIN_APPROXIMATION, SMALL_MASK_SENSITIVITY |
| `x9uzQ6SwddanWcvV1McI` | 1948 | `helmet_on_head` | polygon | 132 | 1 | 0 | 0.7500 | 0.7844 | RASTER_BOUNDARY_DIFFERENCE, SMALL_MASK_SENSITIVITY |
| `OQJwjQoYsf1KUgr9G0V8` | 678 | `helmet_on_head` | synthetic_rectangle | 48 | 1 | 0 | 0.7619 | 0.7619 | RASTER_BOUNDARY_DIFFERENCE, SMALL_MASK_SENSITIVITY |
| `ywH2LFvRJto6sV68hk5v` | 2022 | `vest_on_body` | polygon | 164 | 1 | 0 | 0.7847 | 0.8446 | RASTER_BOUNDARY_DIFFERENCE, SMALL_MASK_SENSITIVITY |
| `MkxXQ8wmfbTbRbgDBAK1` | 647 | `person` | polygon | 21 | 1 | 0 | 0.8077 | 0.6250 | RASTER_BOUNDARY_DIFFERENCE, SMALL_MASK_SENSITIVITY |
| `dwz8H6tUE91TOzWAqabt` | 1116 | `person` | polygon | 234 | 1 | 0 | 0.8107 | 0.8191 | RASTER_BOUNDARY_DIFFERENCE, SMALL_MASK_SENSITIVITY |
| `dwz8H6tUE91TOzWAqabt` | 1117 | `person` | polygon | 203 | 1 | 0 | 0.8120 | 0.7795 | RASTER_BOUNDARY_DIFFERENCE, SMALL_MASK_SENSITIVITY |
| `zs9Fbr1mSnefWqSSmWcg` | 2031 | `person` | rle | 2953647 | 3 | 3 | 0.8230 | 0.8243 | RASTER_BOUNDARY_DIFFERENCE, MULTI_COMPONENT_APPROXIMATION, HOLE_FILL_APPROXIMATION |
| `r3HWnAoRdj2C2qeAdFkr` | 1724 | `helmet_on_head` | polygon | 355 | 1 | 0 | 0.8373 | 0.8869 | RASTER_BOUNDARY_DIFFERENCE, SMALL_MASK_SENSITIVITY |
| `niY7CVigYcNQ0OdTBcor` | 1540 | `person` | rle | 353 | 1 | 0 | 0.8545 | 1.0000 | SERIALIZATION_ONLY, SMALL_MASK_SENSITIVITY |
| `A2iSTnVfTHioOl73tkQd` | 310 | `helmet_on_head` | rle | 9660 | 1 | 1 | 0.8569 | 0.8732 | RASTER_BOUNDARY_DIFFERENCE, HOLE_FILL_APPROXIMATION |
| `SAxTZz7s67P7PHgPiVQD` | 788 | `vest_on_body` | rle | 2509 | 2 | 0 | 0.8612 | 1.0000 | MULTI_COMPONENT_APPROXIMATION, COMPONENT_JOIN_APPROXIMATION |
| `x9uzQ6SwddanWcvV1McI` | 1947 | `helmet_on_head` | polygon | 491 | 1 | 0 | 0.8629 | 0.8836 | RASTER_BOUNDARY_DIFFERENCE, SMALL_MASK_SENSITIVITY |

`COMPUTED_RESULT` The 20 largest relative area errors:

| Image | Ann | Class | Geometry | Area px | Comp | Holes | Rel. area err | Reasons |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `dwz8H6tUE91TOzWAqabt` | 1118 | `person` | polygon | 4 | 1 | 0 | 2.2500 | RASTER_BOUNDARY_DIFFERENCE, SMALL_MASK_SENSITIVITY |
| `dwz8H6tUE91TOzWAqabt` | 1115 | `person` | polygon | 10 | 1 | 0 | 1.1000 | RASTER_BOUNDARY_DIFFERENCE, SMALL_MASK_SENSITIVITY |
| `dwz8H6tUE91TOzWAqabt` | 1112 | `person` | polygon | 8 | 1 | 0 | 0.8750 | RASTER_BOUNDARY_DIFFERENCE, SMALL_MASK_SENSITIVITY |
| `dwz8H6tUE91TOzWAqabt` | 1109 | `vest_on_body` | polygon | 28 | 1 | 0 | 0.5714 | RASTER_BOUNDARY_DIFFERENCE, SMALL_MASK_SENSITIVITY |
| `dwz8H6tUE91TOzWAqabt` | 1110 | `vest_on_body` | polygon | 6 | 1 | 0 | 0.5000 | RASTER_BOUNDARY_DIFFERENCE, SMALL_MASK_SENSITIVITY |
| `q2wyXKnABjAqqL0mLiG7` | 1687 | `vest_on_body` | polygon | 59 | 1 | 0 | 0.4746 | RASTER_BOUNDARY_DIFFERENCE, SMALL_MASK_SENSITIVITY |
| `OQJwjQoYsf1KUgr9G0V8` | 677 | `helmet_on_head` | polygon | 44 | 1 | 0 | 0.4318 | RASTER_BOUNDARY_DIFFERENCE, SMALL_MASK_SENSITIVITY |
| `x9uzQ6SwddanWcvV1McI` | 1948 | `helmet_on_head` | polygon | 132 | 1 | 0 | 0.3333 | RASTER_BOUNDARY_DIFFERENCE, SMALL_MASK_SENSITIVITY |
| `OQJwjQoYsf1KUgr9G0V8` | 678 | `helmet_on_head` | synthetic_rectangle | 48 | 1 | 0 | 0.3125 | RASTER_BOUNDARY_DIFFERENCE, SMALL_MASK_SENSITIVITY |
| `ywH2LFvRJto6sV68hk5v` | 2022 | `vest_on_body` | polygon | 164 | 1 | 0 | 0.2744 | RASTER_BOUNDARY_DIFFERENCE, SMALL_MASK_SENSITIVITY |
| `MkxXQ8wmfbTbRbgDBAK1` | 647 | `person` | polygon | 21 | 1 | 0 | 0.2381 | RASTER_BOUNDARY_DIFFERENCE, SMALL_MASK_SENSITIVITY |
| `dwz8H6tUE91TOzWAqabt` | 1117 | `person` | polygon | 203 | 1 | 0 | 0.2315 | RASTER_BOUNDARY_DIFFERENCE, SMALL_MASK_SENSITIVITY |
| `zs9Fbr1mSnefWqSSmWcg` | 2031 | `person` | rle | 2953647 | 3 | 3 | 0.2124 | RASTER_BOUNDARY_DIFFERENCE, MULTI_COMPONENT_APPROXIMATION, HOLE_FILL_APPROXIMATION |
| `r3HWnAoRdj2C2qeAdFkr` | 1724 | `helmet_on_head` | polygon | 355 | 1 | 0 | 0.1944 | RASTER_BOUNDARY_DIFFERENCE, SMALL_MASK_SENSITIVITY |
| `dwz8H6tUE91TOzWAqabt` | 1116 | `person` | polygon | 234 | 1 | 0 | 0.1667 | RASTER_BOUNDARY_DIFFERENCE, SMALL_MASK_SENSITIVITY |
| `x9uzQ6SwddanWcvV1McI` | 1947 | `helmet_on_head` | polygon | 491 | 1 | 0 | 0.1589 | RASTER_BOUNDARY_DIFFERENCE, SMALL_MASK_SENSITIVITY |
| `l3CKxvrKhNSJ5s7EZGYx` | 1409 | `helmet_on_head` | polygon | 317 | 1 | 0 | 0.1577 | RASTER_BOUNDARY_DIFFERENCE, SMALL_MASK_SENSITIVITY |
| `A2iSTnVfTHioOl73tkQd` | 310 | `helmet_on_head` | rle | 9660 | 1 | 1 | 0.1491 | RASTER_BOUNDARY_DIFFERENCE, HOLE_FILL_APPROXIMATION |
| `l3CKxvrKhNSJ5s7EZGYx` | 1408 | `helmet_on_head` | polygon | 379 | 1 | 0 | 0.1240 | RASTER_BOUNDARY_DIFFERENCE, SMALL_MASK_SENSITIVITY |
| `lHkj3xNmUL3G67dHG1Zr` | 1423 | `vest_on_body` | polygon | 381 | 1 | 0 | 0.1234 | RASTER_BOUNDARY_DIFFERENCE, SMALL_MASK_SENSITIVITY |

`COMPUTED_RESULT` Reason-flag census across all instances. An instance whose deviation no stage and no recorded topology explains is `UNATTRIBUTED` rather than assigned the most plausible-sounding cause:

| Flag | Instances |
| --- | --- |
| (not materially non-exact, IoU >= 0.999) | 0 |
| `COMPONENT_JOIN_APPROXIMATION` | 277 |
| `HOLE_FILL_APPROXIMATION` | 180 |
| `MULTI_COMPONENT_APPROXIMATION` | 307 |
| `RASTER_BOUNDARY_DIFFERENCE` | 1065 |
| `SERIALIZATION_ONLY` | 502 |
| `SMALL_MASK_SENSITIVITY` | 102 |

## 19. Framework-parser validation

`FACT` The generated labels were handed to the installed Ultralytics dataset scanner. Structural parsing only - no model was constructed, no weight downloaded and nothing trained.

| Split | Images discovered | Instances parsed | Segment rows | Negatives |
| --- | --- | --- | --- | --- |
| `train` | 303 | 1422 | 1422 | 10 |
| `validation` | 65 | 304 | 304 | 2 |

`FACT` Classes: 5. Corrupt labels: 0. Holdout split present in the descriptor: false. Coordinate bounds violations: 0.

`FACT` **Instance cardinality survived parsing.** The framework read back exactly the instance count that was written, in both splits. Had it not, the audit would have stopped rather than reported a fidelity distribution over a different set of objects.

## 20. Reproducibility

`FACT` Audit protocol `configs/segmentation_adapter_audit.yaml`, SHA-256 `51090981d320a259b6a5a9a62ce3ebd2ffbc2f499ea87890265ddbecd0ccee7c`. Every methodological choice that could move a number - decoder, contour modes, joining policy, serialisation precision, rasterisation, connectivity, bands - lives in that file rather than in Python.

| Artifact | SHA-256 |
| --- | --- |
| `audit_config_sha256` | `51090981d320a259b6a5a9a62ce3ebd2ffbc2f499ea87890265ddbecd0ccee7c` |
| `fidelity_rows_sha256` | `79f2157326ee898cafcdcb35f733ec42e6a924e86430690758d8ed9520241dde` |
| `image_membership_sha256` | `616701e18deb1e6d873459c94e702d33752f1d712fe0f6eebb0fa8288f126d41` |
| `labels_development_sha256` | `ff21c7822921601a5c2ac6d0d324eb9f234f1a13ce16647f834936de5d5682f2` |
| `labels_train_sha256` | `63a8145d976f7a4c9d1e2868b052c0e0d4233a15f7c6e6b3a1a0e25be6b4cd17` |
| `labels_validation_sha256` | `dae69290936641d73e576682631ef73e026bf7f33cec8c2a60a7789ac054e83f` |

`FACT` Generation is deterministic: running the audit twice produces byte-identical labels, table, manifest and report.

## 21. Holdout compliance

`CANONICAL_DATA_POLICY` Status: **`PROTECTED_NOT_ACCESSED`**. Phase 8A measured annotation-format fidelity on the development splits only. The holdout was not read, materialised, adapted, converted, counted or inspected; no holdout label, statistic or identifier exists in any artifact this phase wrote. Nothing about how COCO geometry serialises depends on which images are held out, so reading it would have bought no information and spent the one look the protocol allows.

`CANONICAL_DATA_POLICY` `CSVISION_ALLOW_TEST_SPLIT` was not set. There is no holdout directory in the audit adapter, no holdout key in the dataset descriptor, no holdout label file, and no holdout row in the fidelity table.

## 22. Limitations

`LIMITATION` This measures **representation, not performance**. Nothing here predicts how well a model would learn from these labels, or what mask AP any architecture would reach.

`LIMITATION` The reconstruction is compared on the original source-image canvas. Training letterboxes and downsamples masks (`mask_ratio`), which is a further transformation this audit deliberately excludes, because it is a model-input choice rather than a property of the label format.

`LIMITATION` The control isolates the pycocotools-versus-OpenCV rasterisation difference but does not eliminate it. Some of the reported gap is a convention difference between two libraries, not information destroyed.

`LIMITATION` Fidelity was measured on development data only. It is a property of these annotations, not a general claim about the format.

## 23. Architecture-decision evidence

`PENDING_HUMAN_DECISION` `segmentation_architecture_selection: UNSELECTED_PENDING_FIDELITY_REVIEW` · `segmentation_baseline: UNFROZEN` · `S0: NOT_DEFINED`.

`PENDING_HUMAN_DECISION` What this audit establishes, and nothing more: the conversion is **structurally possible** - every one of the 1726 development annotations became exactly one parseable YOLO row, with no instance split, merged or dropped - and the geometric cost of doing so is distributed as reported above. Whether that cost is acceptable is a judgement about this project's goals, not a fact this phase can measure.

`PENDING_HUMAN_DECISION` Recorded as a future alternative only, neither implemented nor benchmarked nor selected: if format fidelity is judged materially inadequate, a mask-native instance-segmentation architecture able to consume canonical COCO masks and RLE directly (Mask R-CNN, for example) would avoid this conversion entirely. Nothing about that option has been installed, run or costed, and naming it here is not a preference.

## 24. Next phase

Phase 8B - human review of this evidence and the segmentation architecture decision. It has not started. No segmentation model may be trained, and no S0 protocol defined, before that decision is taken and recorded.

`FACT` The frozen detector was not retrained, revalidated, run or altered by this phase; it remains `84d30d64a1e7ebfc4e4763643ef3a3b5c8a5ce3cedc74bbefdb8031d48235a6e`. No detector-versus-segmenter comparison was made, and none is authorised until a segmentation model is frozen.
