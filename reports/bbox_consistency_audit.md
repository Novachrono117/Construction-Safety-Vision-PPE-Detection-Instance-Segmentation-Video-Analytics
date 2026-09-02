# BBox / Segmentation Consistency Audit

Generated: 2026-09-01T14:43:41+00:00 · Phase: 4 · Commit: `ed9f59aed5b54335a344ec12e82b8bf1e9cd92d2`

**Question.** May phase 5 reuse the `bbox` field supplied by the export when building the detection view, or must it always recompute the box from the segmentation geometry?

**Method (COMPUTED).** For every annotation carrying segmentation geometry, the enclosing box is derived from that geometry - polygons by coordinate extrema, compressed RLE through the reference COCO implementation - and compared corner by corner against the supplied `bbox`. Primary tolerance: **1.0 px**.

**Why that tolerance (FACT).** A compressed RLE is a rasterised binary mask, so a box derived from it is quantised to whole pixels while a polygon-derived box is fractional. A one-pixel allowance is the coarsest representation's own resolution, not a threshold chosen to produce a pleasing result. Agreement at several tolerances is reported below so the reader can apply their own.

## Result

| | Value |
| --- | --- |
| Annotations checked | 3373 of 3373 |
| Polygon / RLE | 1570 / 1803 |
| Agree within 1.0 px | 2589 (76.76%) |
| Mismatches at 1.0 px | 784 |
| Maximum discrepancy | 123.5 px |
| Mean discrepancy | 3.9626 px |

### Agreement by tolerance and representation (COMPUTED)

| Tolerance (px) | All | Polygon | RLE |
| --- | --- | --- | --- |
| 0.0 | 45 (1.33%) | 23 (1.46%) | 22 (1.22%) |
| 0.5 | 1793 (53.16%) | 1570 (100.00%) | 223 (12.37%) |
| 1.0 | 2589 (76.76%) | 1570 (100.00%) | 1019 (56.52%) |
| 2.0 | 2725 (80.79%) | 1570 (100.00%) | 1155 (64.06%) |
| 5.0 | 2910 (86.27%) | 1570 (100.00%) | 1340 (74.32%) |

### Per split (COMPUTED)

| Split | Checked | Polygon | RLE | Max delta (px) | Mean delta (px) | Mismatches at 1.0 px |
| --- | --- | --- | --- | --- | --- | --- |
| test | 166 | 65 | 101 | 39.0 | 0.7079 | 7 |
| train | 2836 | 1337 | 1499 | 123.5 | 4.596 | 762 |
| valid | 371 | 168 | 203 | 28.25 | 0.5774 | 15 |

`valid` and `test` are un-augmented, so their annotations are the preprocessed source annotations. `train` additionally contains geometry transformed by offline augmentation, which is why the splits are not pooled.

### Largest discrepancies (COMPUTED)

| Split | Annotation | Image | Class | Repr. | Max delta (px) |
| --- | --- | --- | --- | --- | --- |
| test | 144 | 34 | `person` | rle | 39.0 |
| test | 119 | 31 | `vest_on_body` | rle | 1.75 |
| test | 10 | 4 | `vest_on_body` | rle | 1.5 |
| test | 16 | 4 | `person` | rle | 1.5 |
| test | 61 | 17 | `person` | rle | 1.25 |
| train | 1831 | 380 | `vest_on_body` | rle | 123.5 |
| train | 689 | 146 | `person` | rle | 121.5 |
| train | 1835 | 380 | `person` | rle | 103.0 |
| train | 11 | 1 | `person` | rle | 99.0 |
| train | 2782 | 595 | `person` | rle | 99.0 |
| valid | 319 | 76 | `vest_on_body` | rle | 28.25 |
| valid | 208 | 55 | `vest_on_body` | rle | 16.75 |
| valid | 96 | 24 | `person` | rle | 6.0 |
| valid | 91 | 23 | `vest_loose` | rle | 3.25 |
| valid | 150 | 40 | `person` | rle | 1.5 |

### Why they disagree (COMPUTED diagnostics)

| Diagnostic | Count |
| --- | --- |
| supplied box `polygon|inside|agrees` | 1570 |
| supplied box `rle|inside|agrees` | 953 |
| supplied box `rle|inside|differs` | 585 |
| supplied box `rle|spills|agrees` | 66 |
| supplied box `rle|spills|differs` | 199 |
| `polygon|derived_inside_supplied` | 1570 |
| `rle|derived_escapes` | 26 |
| `rle|derived_inside_supplied` | 1777 |

## Interpretation

- COMPUTED: polygon annotations agree essentially perfectly - 1570 of 1570 (100.00%) within 0.5 px, though only 23 match to the bit. The supplied box for a polygon is the polygon's own coordinate extrema.
- COMPUTED: the entire disagreement lives in the RLE annotations - 1019 of 1803 (56.52%) within 1.0 px. The largest single discrepancy is 123.5 px.
- COMPUTED: the mask-derived box lies inside the supplied box in 1777 of 1803 RLE cases (98.56%). The supplied box is systematically the larger of the two, not randomly offset.
- COMPUTED: 265 supplied boxes extend outside the image canvas, which a rasterised mask cannot do; only 24.91% of those agree, against 61.96% for boxes fully inside the canvas. Clipping is therefore a contributing factor but NOT the explanation: 585 RLE annotations disagree while their supplied box is entirely within the canvas.
- COMPUTED: the export's `area` field equals bbox width x height, not the mask area, so it carries no independent evidence about the mask and was not used here.
- COMPUTED: mean discrepancy is far larger on the augmented split (train 4.596 px) than on the un-augmented ones (test 0.7079 px, valid 0.5774 px).
- HYPOTHESIS (untested): the supplied box may be transformed as a box through augmentation while the mask is transformed as pixels - rotating an axis-aligned box and re-axis-aligning it always enlarges it, which would produce exactly this direction of error and exactly this train/validation gap. This is consistent with the numbers; it has NOT been demonstrated, and it does not account for the disagreements on the un-augmented splits.
- DECISION FOR PHASE 5: recompute every detection box from the segmentation geometry and do not use the supplied `bbox`. This is not merely tidier - for the RLE annotations the two disagree materially, and the segmentation is by project definition the source of truth. Whether the mask or the supplied box better reflects the real object is a visual question this audit cannot settle.

## Open questions

- Whether the same agreement holds for the ORIGINAL source annotations: this audit sees the preprocessed export, in which every image was resized to 640x640.
- Whether the annotations with the largest discrepancies are genuine annotation defects or artefacts of the polygon-to-RLE conversion. Requires visual review.
- Why the export mixes polygon and RLE at all - whether it tracks how each instance was originally drawn. Not answerable from the export alone.
