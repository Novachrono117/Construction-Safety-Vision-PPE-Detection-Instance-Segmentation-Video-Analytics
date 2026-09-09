# Canonical segmentation comparison reference

Phase 8E · timing `POST_S0_PRE_S1_CANONICAL_COMPARISON_REFERENCE_EVALUATION` · S1 `FROZEN_NOT_EXECUTED` · final segmenter `UNSELECTED`

**This is an addendum, not a correction.** Nothing in the phase 8C S0 report is withdrawn, revised or regenerated. A third measurement is added beside the two S0 already publishes, because a comparison that phase 8C did not anticipate now needs one.

Repository commit at production time: `776317f2a8375cf4fe8e8fd19d72478cb3a4ad8c`.

## 1. Why a common evaluator was introduced

Phase 8D established from the installed source that the framework builds both its mask target and its validation ground truth according to overlap_mask: polygons2masks_overlap gives contested pixels to the smaller instance, and SegmentationValidator._prepare_batch scores against that same resolved target. S0 trained with the flag set; the S1 candidate clears it. Their native mask AP would therefore be measured against different ground truth, so a common reference that no training flag can move was needed before the two could be compared at all.

The timing is recorded rather than smoothed over: the need was discovered by the phase 8D error analysis, **after S0 ran**. The protocol is nonetheless frozen **before S1 exists**, which is what keeps it a protocol rather than a description of whichever number turns out to be convenient.

## 2. Why native mask AP will not arbitrate S0 versus S1

Native mask AP is not wrong and is not withdrawn - it is a valid statement about each model against its own target. It simply cannot rank two models whose targets differ, because a difference between the two numbers would confound the model with the target it was scored on. The canonical COCO evaluator decides instead.

Native mask AP is therefore labelled `NOT_CROSS_TARGET_COMPARABLE_FOR_S0_S1_SELECTION`. It is **not** suppressed: both experiments report it in full, and for each it remains a valid statement about that model against its own target.

## 3. What remains valid from phase 8C

| S0 result | Value | Status |
| --- | --- | --- |
| Native framework mask mAP@0.50:0.95 | 0.407942 | `NATIVE_TARGET_EVALUATION` - still valid |
| Direct GT-normalised mask IoU | 0.556977 | `CANONICAL_GT_RECOVERY_DIAGNOSTIC` - still valid |
| Direct matched mask IoU mean | 0.717462 | `CANONICAL_GT_RECOVERY_DIAGNOSTIC` - still valid |

The phase 8D re-score against effective overlap targets stays `HYPOTHESIS_GENERATING_DIAGNOSTIC_ONLY` and replaces no published S0 result.

## 4. The canonical evaluator

`PREDECLARED_PROTOCOL`, fingerprint `282eb0ec125c47687340325266a2ea397bc0f85e239290a26a5c1ec8d8721e64`, frozen in `configs/segmentation_canonical_evaluation.yaml` and validated against synthetic fixtures before it was pointed at any checkpoint.

| Setting | Value |
| --- | --- |
| `agnostic_nms` | False |
| `augment` | False |
| `classes` | ALL_FIVE_FROZEN_CLASSES |
| `conf` | 0.001 |
| `half` | False |
| `imgsz` | 768 |
| `iou` | 0.7 |
| `max_det` | 300 |
| `retina_masks` | True |
| COCOeval IoU thresholds | 0.5:0.05:0.95 (10 steps) |
| COCOeval maxDets | [1, 10, 100] |
| Mask encoding | `PYCOCOTOOLS_BINARY_MASK_RLE` |
| Category mapping | `NONE_CANONICAL_IDS_USED_DIRECTLY` |

The confidence of 0.001 is **not an operating point**. Average precision integrates over the score curve and needs the low-scoring tail; the phase 8C direct-IoU diagnostic keeps its own operational 0.25, and the two protocols are never mixed. Note also that the model may propose up to 300 candidates per image while COCOeval applies its conventional cap of 100 when scoring - two different numbers, both recorded, neither constraining the other.

## 5. S0 canonical reference result

`COMPUTED_RESULT`, classified `POST_S0_PRE_S1_CANONICAL_COMPARISON_REFERENCE_EVALUATION` and executed exactly once. Validation only.

| Metric | Value |
| --- | --- |
| **CANONICAL_SUPPORTED_MACRO_MASK_MAP50_95** | **0.484643** |
| CANONICAL_ALL_CLASS_MASK_MAP50_95 | 0.388009 |
| CANONICAL_ALL_CLASS_MASK_MAP50 | 0.537536 |

| Class | canonical AP@0.50:0.95 | canonical AP@0.50 | admitted by the support rule |
| --- | --- | --- | --- |
| `helmet_loose` | 0.793946 | 0.891599 | True |
| `helmet_on_head` | 0.609345 | 0.793362 | True |
| `person` | 0.135036 | 0.309623 | True |
| `vest_loose` | 0.001474 | 0.007173 | False |
| `vest_on_body` | 0.400246 | 0.685925 | True |

The primary metric averages the 4 admitted classes (`helmet_loose`, `helmet_on_head`, `person`, `vest_on_body`) under the unchanged phase 7A rule: COMPARISON_SUPPORTED requires >= 5 positive validation source images AND >= 20 validation instances. The rule names no class; the admitted set is its output. `vest_loose` stays `DESCRIPTIVE_HIGH_UNCERTAINTY`, reported in full and deciding nothing.

**These canonical numbers are not comparable with the native mask AP above.** They are computed against different ground truth, by a different implementation, at a different confidence. Reading a difference between them as a change in the model would be a mistake; each answers its own question about the same checkpoint.

### 5.1 Where the two evaluators disagree, and why that is interesting

The two measurements are **not comparable in absolute terms** - different ground truth, a different implementation and a different confidence - so no figure below is a difference anyone should quote as a change in the model. What *is* legible is the **shape** of the disagreement across classes:

| Class | canonical AP@0.50:0.95 | native AP@0.50:0.95 |
| --- | --- | --- |
| `helmet_loose` | 0.793946 | 0.789927 |
| `helmet_on_head` | 0.609345 | 0.586523 |
| `person` | 0.135036 | 0.271182 |
| `vest_loose` | 0.001474 | 0.001782 |
| `vest_on_body` | 0.400246 | 0.390296 |

The compact classes land close together under the two evaluators. `person` does not: it is by a wide margin the class where the canonical evaluator and the native one most disagree. That is exactly the class phase 8D identified, and exactly the direction the overlap-target mechanism predicts - the canonical ground truth includes the vest-and-helmet pixels the training target assigns away from the person, and the native evaluator does not.

**This corroborates the phase 8D mechanism; it does not prove it, and it certainly does not show that S1 will be better.** It is one more observation consistent with the same explanation, produced by an evaluator built for a different purpose - which is worth more than a second look at the same numbers, and still less than an experiment.

## 6. The frozen S0-versus-S1 comparison

`PREDECLARED_PROTOCOL`, fingerprint `a74609c1c58371f91de2b1699e695fcbdf3fed6ac9a68237c190cfb1d1393f16`. The primary metric is `CANONICAL_SUPPORTED_MACRO_MASK_MAP50_95`, the margin is 0.005, and the three cases are fixed before the candidate exists:

| Case | Condition | Outcome |
| --- | --- | --- |
| `S1_IMPROVES_S0_BEYOND_MARGIN` | delta > +0.005 | S1 leads |
| `PRACTICALLY_EQUIVALENT` | within +/-0.005 | S0 preferred, because the baseline exists and no material canonical advantage was shown |
| `S1_BELOW_S0` | delta < -0.005 | S0 retained |

The margin is an engineering threshold, **not a significance test**: nothing is repeated, so run-to-run variance remains UNKNOWN. If the canonical AP and the direct IoU diagnostic move in opposite directions the outcome is `CROSS_METRIC_DIRECTION_DISAGREEMENT` - recorded, ranked by the canonical metric, and never resolved by inventing a weighted score.

## 7. The overlap-target hypothesis this tests

Phase 8D measured that 27.3% of canonical person pixels are contested by another class, while 71.0% of the pixels S0 misses on person lie in that contested region; contested fraction versus person mask IoU has Spearman -0.5466; and re-scoring the same predictions against the overlap-resolved target recovers +0.0700 GT-normalised IoU for person and under 0.002 for every other class. S1 tests whether removing the overlap resolution from training changes canonical performance. It is a hypothesis: the measurement showed a target mismatch, not that training differently helps, and even against its own target S0's person mask IoU remained far below the compact classes.

## 8. S1 status

`FROZEN_NOT_EXECUTED`. S1 has not been trained, and no S1 number exists anywhere in this repository. The final segmenter remains `UNSELECTED`.

---

S0 checkpoint `d7b512b95fafdc8658fd802459d2c87d9ad3f11f922162a774dacf8ca75b87a3` · canonical evaluation `reports/segmentation_S0_canonical_evaluation.json`.
