# Final holdout evaluation - one-shot results

**Phase 11B. `ONE_SHOT_FINAL_HOLDOUT_EVALUATION`. `FINAL_TEST_OBSERVED`.**

## 1. Holdout policy

The `test` split was a locked holdout from phase 5C.2 until this phase. It took no part in any design, selection, tuning or diagnostic decision: not in model selection, not in architecture selection, not in hyperparameter or augmentation tuning, not in threshold tuning, not in qualitative debugging, and it was not looked at. Both models were frozen long before - the detector in phase 7D, the segmenter in phase 8G - and the whole validation comparison was published in phases 10A-10D.

It has now been read **once**, under the protocol phase 11A froze before any holdout number could exist. Every rule below - which checkpoints, at which settings, judged by which evaluator, with which confusion-matrix semantics, and which qualitative examples - was written down in advance. Nothing was adapted after a number was seen.

## 2. One-shot execution status

|  | Value |
| --- | --- |
| Classification | `TEST_EVALUATION_COMPLETE` |
| Attempt | 1 |
| Attempt justification | `FIRST_AND_ONLY_AUTHORISED_ATTEMPT` |
| Ledger final state | `METRICS_COMPUTED` |
| Attempt counter resettable | no |
| Holdout reads permitted | 1 |
| Holdout reads performed | 1 |
| Models executed | 2 |
| Inference passes | 3 |
| Models trained | 0 |
| Thresholds tuned | 0 |
| Latency measurements taken | 0 |
| New spatial metrics | 0 |
| Prediction reruns | 0 |
| Protocol fingerprint | `a5a328b3a8e49b06fb8fd9e792abcf43ccdd9aac5422729814dac0dbadc1daef` |

Predictions were persisted and fingerprinted **before any metric was computed**, and every figure in this report was then derived from those persisted files. No model was invoked during metric computation or report generation. That ordering is what makes a report rebuild possible without a second inference run, and it is why an artifact-write failure and a prediction failure have separate names in the frozen policy.

## 3. Final model identities

Both were resolved **by digest**, never by path, through their existing freeze accessors. Neither was retrained, revalidated, re-thresholded or substituted.

|  | Detector | Segmenter |
| --- | --- | --- |
| Experiment | **D2** | **S1** |
| Architecture | YOLO11n | YOLO11n-seg |
| Input size | 768 | 768 |
| `mask_ratio` | n/a | 4 |
| `overlap_mask` | n/a | false |
| Checkpoint SHA-256 | `0466f872a9de22898c70d8834cf1fcfb3d77f6c9fb27e81ee6248d3d19cbc206` | `29337d671459d0f742fb713613cc41d0eafcb8a21f5846ac0da65b2ffc024f20` |
| Checkpoint bytes | 5502289 | 6041685 |
| Frozen in phase | 7D | 8G |
| Trained in this phase | no | no |
| Modified in this phase | no | no |

### Prediction fingerprints

| Pass | Model | Confidence | Masks | Predictions | Fingerprint |
| --- | --- | --- | --- | --- | --- |
| `DETECTOR_AP_PASS` | D2 | 0.001 | no | 3605 | `bfcf35762b56a761c152ab14035b0f3c3c3cb2c6faafe65de3fee493403053b6` |
| `SEGMENTER_AP_PASS` | S1 | 0.001 | yes | 4760 | `181d036c3b4e7e039b72061fc3f4e4ee7291a45b5fa7f8ead433e328314ed504` |
| `SEGMENTER_OPERATIONAL_PASS_FOR_DIRECT_IOU` | S1 | 0.25 | yes | 261 | `cadd4170c604174b6c988e39a2bdfa8eb9e4f97af0b8dbb57239fc5685bf2b6d` |

Three inference passes ran, all declared in advance: each model's AP pass at conf 0.001, and the segmenter's operational pass at conf 0.25 for the phase 8C direct-IoU diagnostic. The two confidences are never mixed, averaged or swapped: average precision integrates over the score curve and needs its low-scoring tail, which is why it is deliberately **not** an operating point.

## 4. Test population

The holdout was materialised by the **phase 5D function with the phase 5D configuration** - the same function that wrote `train` and `validation`. There is no split-specific branch that could give the protected split different treatment, and the predeclared technical validation ran before any model did.

|  | Value |
| --- | --- |
| Images | 65 |
| Annotations | 305 |
| Evaluated | all frozen holdout images |
| Sampled | no |
| Manually excluded | 0 |
| Membership fingerprint | `0052ee3bc1d73e4f7c852c3bb16b4273a0ba5ef7bb62283ff9b95557a1e7c699` |
| Integrity problems | 0 |
| Byte-identical image copies | 65/65 |
| Detection/segmentation alignment problems | 0 |
| Geometry round-trip mismatches | 0 |
| Zero-instance images | 2 |

The counts match the aggregate figures frozen in phase 5C.2, long before any model existed. Per-image membership is not published here: a holdout identifier in a committed artifact is a leak even when no pixel travels with it.

### Class support on the holdout

The **phase 7A support rule is reused unchanged** - `COMPARISON_SUPPORTED` requires at least 5 positive images and at least 20 instances. It names no class; the admitted set is its output. It was applied only after the evaluation, its thresholds were not re-parameterised in response to what the holdout turned out to contain, and on the holdout it is a descriptive caveat that decides nothing.

| Class | Images | Instances | Status |
| --- | --- | --- | --- |
| `helmet_loose` | 11 | 60 | `COMPARISON_SUPPORTED` |
| `helmet_on_head` | 27 | 47 | `COMPARISON_SUPPORTED` |
| `person` | 55 | 136 | `COMPARISON_SUPPORTED` |
| `vest_loose` | 2 | 7 | `DESCRIPTIVE_HIGH_UNCERTAINTY` |
| `vest_on_body` | 31 | 55 | `COMPARISON_SUPPORTED` |

## 5. Detector canonical bounding-box results

One external evaluator judges both models' boxes - `pycocotools.cocoeval.COCOeval` at `iouType='bbox'` against the canonical phase 5D holdout boxes, IoU 0.50:0.05:0.95, `maxDets` [1, 10, 100] - because the two models run through different framework validation paths and their native box numbers are not guaranteed to be computed identically.

| Metric | Value |
| --- | --- |
| `CANONICAL_TEST_BOX_MAP50_95` | **0.427031** |
| `CANONICAL_TEST_BOX_MAP50` | 0.565260 |
| Detections scored | 3605 |
| Inference confidence | 0.001 |

### 5.1 Detector per-class results

| Class | AP@0.50:0.95 | AP@0.50 |
| --- | --- | --- |
| `helmet_loose` | 0.596792 | 0.692540 |
| `helmet_on_head` | 0.610335 | 0.832987 |
| `person` | 0.489896 | 0.675138 |
| `vest_loose` | 0.000000 | 0.000000 |
| `vest_on_body` | 0.438132 | 0.625636 |

### 5.2 Detector precision and recall at the frozen operating point

Precision and recall are **not threshold-independent**. They are reported at the project's single operational confidence of 0.25, counted at IoU 0.5 over exactly the detections the AP figures were computed from. That operating point has been frozen since phase 10A and was not chosen with any holdout number in view, and no threshold was swept on the holdout.

|  | Precision | Recall | TP | FP | FN |
| --- | --- | --- | --- | --- | --- |
| **all classes** | 0.787500 | 0.619672 | 189 | 51 | 116 |
| `helmet_loose` | 0.764706 | 0.650000 | 39 | 12 | 21 |
| `helmet_on_head` | 0.804878 | 0.702128 | 33 | 8 | 14 |
| `person` | 0.815534 | 0.617647 | 84 | 19 | 52 |
| `vest_loose` | n/a | 0.000000 | 0 | 0 | 7 |
| `vest_on_body` | 0.733333 | 0.600000 | 33 | 12 | 22 |

Descriptive supported macro AP@0.50:0.95 over the classes the frozen rule admits: **0.533789** (`helmet_loose`, `helmet_on_head`, `person`, `vest_on_body`). This is a descriptive figure on the holdout, never a selection metric.

## 6. Segmenter canonical mask results

`pycocotools.cocoeval.COCOeval` at `iouType='segm'` against the canonical phase 5D holdout masks, on the original image canvas. **Mask and box never merge**: the box figures below are reported in their own section and no composite exists.

| Metric | Value |
| --- | --- |
| `CANONICAL_TEST_MASK_MAP50_95` | **0.410143** |
| `CANONICAL_TEST_MASK_MAP50` | 0.579074 |
| Detections scored | 4760 |
| Inference confidence | 0.001 |

### 6.1 Segmenter per-class mask results

| Class | AP@0.50:0.95 | AP@0.50 |
| --- | --- | --- |
| `helmet_loose` | 0.561808 | 0.710431 |
| `helmet_on_head` | 0.664600 | 0.900424 |
| `person` | 0.446269 | 0.635148 |
| `vest_loose` | 0.003850 | 0.005501 |
| `vest_on_body` | 0.374189 | 0.643865 |

### 6.2 Segmenter mask precision and recall at the frozen operating point

|  | Precision | Recall | TP | FP | FN |
| --- | --- | --- | --- | --- | --- |
| **all classes** | 0.773946 | 0.662295 | 202 | 59 | 103 |
| `helmet_loose` | 0.767857 | 0.716667 | 43 | 13 | 17 |
| `helmet_on_head` | 0.851064 | 0.851064 | 40 | 7 | 7 |
| `person` | 0.765766 | 0.625000 | 85 | 26 | 51 |
| `vest_loose` | n/a | 0.000000 | 0 | 0 | 7 |
| `vest_on_body` | 0.723404 | 0.618182 | 34 | 13 | 21 |

Descriptive supported macro mask AP@0.50:0.95: **0.511717**.

## 7. Segmenter canonical bounding-box results

The segmenter's boxes are **its own predicted boxes**, never re-derived from its masks: re-deriving them would improve their geometric consistency with the mask branch and would then be measuring a post-processing choice this project invented rather than the model. They go through the **same** canonical bbox evaluator the detector's boxes go through.

| Metric | Value |
| --- | --- |
| `S1_CANONICAL_TEST_BOX_MAP50_95` | **0.433764** |
| `S1_CANONICAL_TEST_BOX_MAP50` | 0.583500 |
| Detections scored | 4760 |
| Boxes derived from masks | no |

| Class | AP@0.50:0.95 | AP@0.50 |
| --- | --- | --- |
| `helmet_loose` | 0.589534 | 0.682427 |
| `helmet_on_head` | 0.687047 | 0.900424 |
| `person` | 0.476795 | 0.669169 |
| `vest_loose` | 0.003850 | 0.005501 |
| `vest_on_body` | 0.411595 | 0.659978 |

## 8. Detector versus segmenter localisation on the holdout

`DESCRIPTIVE_ONLY`. The two models do not solve the same output task - the detector emits class, confidence and a box; the segmenter emits those **plus a mask** - so this is a description of localisation and never a ranking. **No winner is declared, no composite or weighted score exists, no significance test was run, and no model selection follows.**

| Canonical box metric | D2 | S1 | Delta |
| --- | --- | --- | --- |
| mAP@0.50:0.95 | 0.427031 | 0.433764 | +0.006733 |
| mAP@0.50 | 0.565260 | 0.583500 | +0.018240 |

### 8.1 Per-class decomposition

The all-class figure is the unweighted mean of five per-class APs, so it decomposes exactly. Quoting the aggregate without this table is the error the decomposition exists to prevent - especially because `vest_loose` carries very little support.

| Class | D2 AP@0.50:0.95 | S1 AP@0.50:0.95 | Delta | Contribution to delta |
| --- | --- | --- | --- | --- |
| `helmet_loose` | 0.596792 | 0.589534 | -0.007258 | -0.001452 |
| `helmet_on_head` | 0.610335 | 0.687047 | +0.076712 | +0.015342 |
| `person` | 0.489896 | 0.476795 | -0.013101 | -0.002620 |
| `vest_loose` | 0.000000 | 0.003850 | +0.003850 | +0.000770 |
| `vest_on_body` | 0.438132 | 0.411595 | -0.026537 | -0.005307 |

Computational cost is **not** re-measured here. Phase 10C measured latency and inference memory under a frozen symmetric protocol on this machine; those are properties of the model, the runtime and the hardware, not of which split the images came from, and phase 11B ran no benchmark of any kind.

## 9. Direct instance-mask IoU diagnostic

`SECONDARY_CANONICAL_DIAGNOSTIC`. This is the **exact phase 8C protocol, reused by fingerprint and unchanged** (`b912039ca77b36959f74fcdbaed109dbf5e3bd707709556f95a19085c7f34d80`). No new matching rule was invented for the holdout, because inventing one now would let it be chosen with the final result in view. It scores against the canonical COCO masks at the diagnostic's own operational confidence of 0.25, matched one-to-one per image and per class by `scipy.optimize.linear_sum_assignment` maximising total IoU.

**It is not an average precision and it is not the primary segmenter metric.**

| Diagnostic | Value |
| --- | --- |
| `matched_mask_iou_mean` | **0.834548** |
| `gt_normalized_mask_iou` | **0.585551** |
| `gt_match_coverage` | 0.701639 |
| `gt_iou50_coverage` | 0.662295 |
| `gt_iou75_coverage` | 0.560656 |
| Canonical instances | 305 |
| Predictions | 261 |
| Matched | 214 |
| Unmatched ground truth | 91 |
| Unmatched predictions | 47 |

**The two headlines are not interchangeable.** `matched_mask_iou_mean` describes mask quality *where the model produced an overlapping same-class instance*; `gt_normalized_mask_iou` divides the same IoU sum by *every* canonical instance, so the misses lower it. Neither may be quoted as the project's IoU without saying which.

| Class | GT | Pred | Matched | matched IoU | GT-normalised | IoU>=0.50 |
| --- | --- | --- | --- | --- | --- | --- |
| `helmet_loose` | 60 | 56 | 46 | 0.846611 | 0.649068 | 0.716667 |
| `helmet_on_head` | 47 | 47 | 40 | 0.864540 | 0.735779 | 0.851064 |
| `person` | 136 | 111 | 91 | 0.837574 | 0.560436 | 0.625000 |
| `vest_loose` | 7 | 0 | 0 | n/a | 0.000000 | 0.000000 |
| `vest_on_body` | 55 | 47 | 37 | 0.779682 | 0.524513 | 0.618182 |

## 10. Confusion matrix

The semantics were **read from the installed framework** in phase 11A rather than assumed, and they are the exact values that produced every committed validation matrix in this repository: `ultralytics.utils.metrics.ConfusionMatrix`, confidence 0.25, IoU 0.45, class-agnostic matching with the class pair then recorded, a 6x6 matrix with **rows predicted and columns ground truth**, and the final row and column the unmatched background bucket. A matched pair whose classes disagree lands off-diagonal and counts as **both** a false positive and a false negative, which is the framework's own behaviour and is not modified here.

Neither threshold was altered after the results were seen.

### 10.1 Detector (D2)

| predicted \ ground truth | `helmet_loose` | `helmet_on_head` | `person` | `vest_loose` | `vest_on_body` | `background` |
| --- | --- | --- | --- | --- | --- | --- |
| `helmet_loose` | 39 | 0 | 0 | 0 | 0 | 12 |
| `helmet_on_head` | 0 | 33 | 0 | 0 | 0 | 8 |
| `person` | 0 | 0 | 82 | 1 | 1 | 19 |
| `vest_loose` | 0 | 0 | 0 | 0 | 0 | 0 |
| `vest_on_body` | 0 | 0 | 0 | 1 | 36 | 8 |
| `background` | 21 | 14 | 54 | 5 | 18 | 0 |

### 10.2 Segmenter (S1)

| predicted \ ground truth | `helmet_loose` | `helmet_on_head` | `person` | `vest_loose` | `vest_on_body` | `background` |
| --- | --- | --- | --- | --- | --- | --- |
| `helmet_loose` | 43 | 0 | 0 | 0 | 0 | 13 |
| `helmet_on_head` | 1 | 40 | 0 | 0 | 0 | 6 |
| `person` | 0 | 0 | 88 | 3 | 0 | 20 |
| `vest_loose` | 0 | 0 | 0 | 0 | 0 | 0 |
| `vest_on_body` | 0 | 0 | 1 | 0 | 36 | 10 |
| `background` | 16 | 7 | 47 | 4 | 19 | 0 |

## 11. Object-level TP / FP / FN

Frozen **separately** from the confusion matrix and not to be conflated with it: class-aware, IoU 0.5, one-to-one per image and per class, at the operational confidence 0.25, assigned greedily by descending confidence and then highest IoU. The qualitative work needs per-instance outcomes, not a matrix cell.

|  | TP | FP | FN | Ground truth | Precision | Recall |
| --- | --- | --- | --- | --- | --- | --- |
| **D2** | 189 | 51 | 116 | 305 | 0.787500 | 0.619672 |
| **S1** | 205 | 56 | 100 | 305 | 0.785441 | 0.672131 |

### 11.1 Frozen segmentation failure taxonomy

The four categories are evaluated **in the order listed, first match wins**, and they explicitly do **not** partition every instance - the remainder is `WELL_HANDLED_INSTANCE`. No category was added in response to what the holdout contained.

| Category | D2 | S1 |
| --- | --- | --- |
| `DETECTION_MISS` | 108 | 92 |
| `CLASSIFICATION_MISMATCH` | 5 | 6 |
| `LOCALIZATION_FAILURE` | 3 | 2 |
| `MASK_QUALITY_FAILURE` | 0 | 8 |
| `WELL_HANDLED_INSTANCE` | 189 | 197 |

`MASK_QUALITY_FAILURE` is structurally unavailable to the detector, which emits no mask; its zero is a property of the output type, not a performance statement.

## 12. Deterministically selected qualitative FP/FN examples

**No holdout image was browsed before selection.** The examples were chosen by the rule phase 11A froze, from the persisted predictions and the canonical annotations alone: six categories, three examples each, each ranked by a declared quantity in a declared direction, with a tie-breaking chain that ends in an identifier so the order is total on any machine. One canonical instance appears in at most one category. An underfilled category publishes what it has and records the shortfall; it is never topped up from another. Human interpretation happens **after** the ranking, never before it.

Qualitative selection fingerprint: `d47df93114b1e5e2e3c44287c7f317cb9873d7eabdb372cb1c31e7c78ac946c0`.

The selected instances and the rendered figures are deliberately **not committed**: a holdout image identifier in this repository would be a leak, and holdout imagery more so. They exist under the git-ignored run directory for human review. The frozen protocol lists qualitative figures among its permitted outputs *and* prohibits committing holdout identifiers or imagery; where those two clauses meet, the prohibition wins and the tension is recorded rather than silently resolved.

### 12.1 Detector (D2)

| Category | Qualifying | Selected | Note |
| --- | --- | --- | --- |
| `HIGHEST_CONFIDENCE_FALSE_POSITIVE` | 51 | 3 | quota met - ranked values 0.955142, 0.915201, 0.915113 |
| `HIGHEST_CONFIDENCE_CLASSIFICATION_MISMATCH` | 5 | 3 | quota met - ranked values 0.958944, 0.871160, 0.735544 |
| `LOWEST_IOU_MATCHED_INSTANCE` | 189 | 3 | quota met - ranked values 0.523421, 0.555228, 0.582777 |
| `LARGEST_MISSED_INSTANCE` | 108 | 3 | quota met - ranked values 849981.000000, 365560.000000, 307736.000000 |
| `SEGMENTATION_UNDER_COVERAGE` | n/a | n/a | not applicable to this model |
| `SEGMENTATION_OVER_COVERAGE` | n/a | n/a | not applicable to this model |

### 12.2 Segmenter (S1)

| Category | Qualifying | Selected | Note |
| --- | --- | --- | --- |
| `HIGHEST_CONFIDENCE_FALSE_POSITIVE` | 56 | 3 | quota met - ranked values 0.962725, 0.915431, 0.910890 |
| `HIGHEST_CONFIDENCE_CLASSIFICATION_MISMATCH` | 6 | 3 | quota met - ranked values 0.886892, 0.882208, 0.876826 |
| `LOWEST_IOU_MATCHED_INSTANCE` | 205 | 3 | quota met - ranked values 0.514424, 0.522205, 0.524898 |
| `LARGEST_MISSED_INSTANCE` | 92 | 3 | quota met - ranked values 849981.000000, 365560.000000, 285786.000000 |
| `SEGMENTATION_UNDER_COVERAGE` | 205 | 3 | quota met - ranked values 0.378837, 0.593778, 0.676325 |
| `SEGMENTATION_OVER_COVERAGE` | 205 | 3 | quota met - ranked values 3.207836, 2.136303, 1.951494 |

## 13. Relationship to the validation conclusions

`DESCRIPTIVE_GENERALIZATION_COMPARISON`. Only metrics that **already existed before the holdout was read**, their holdout counterparts and the absolute difference. No new metric was invented for this comparison, no subgroup was mined, and **no significance test is reported, because none was predeclared and none may be added afterwards**. Each model was trained once and evaluated once per split, so run-to-run variance is UNKNOWN and a gap of any size is an observation rather than a result.

| Metric | Validation | Test | Absolute difference |
| --- | --- | --- | --- |
| D2 canonical box mAP@0.50:0.95 | 0.485390 | 0.427031 | 0.058359 |
| D2 canonical box mAP@0.50 | 0.641107 | 0.565260 | 0.075847 |
| S1 canonical box mAP@0.50:0.95 | 0.505682 | 0.433764 | 0.071918 |
| S1 canonical box mAP@0.50 | 0.692955 | 0.583500 | 0.109455 |
| S1 canonical mask mAP@0.50:0.95 | 0.455034 | 0.410143 | 0.044891 |
| S1 canonical mask mAP@0.50 | 0.634023 | 0.579074 | 0.054949 |
| S1 direct `matched_mask_iou_mean` | 0.794849 | 0.834548 | 0.039699 |
| S1 direct `gt_normalized_mask_iou` | 0.635356 | 0.585551 | 0.049805 |
| S1 direct `gt_match_coverage` | 0.799342 | 0.701639 | 0.097703 |
| S1 direct `gt_iou50_coverage` | 0.707237 | 0.662295 | 0.044942 |
| S1 direct `gt_iou75_coverage` | 0.595395 | 0.560656 | 0.034739 |

The validation figures are quoted from their committed artifacts by field. They were produced by the same evaluators and the same inference settings as their holdout counterparts, which is what makes the pair comparable at all; where an evaluator or a confidence differs, the two numbers are not placed in the same row.

**Why a gap exists in either direction is UNKNOWN.** Nothing here tested an explanation, and no experiment may now be run to produce one for a reported holdout number.

## 14. Final limitations

- **`vest_loose` carries 7 instances over 2 holdout image(s)** and is classified `DESCRIPTIVE_HIGH_UNCERTAINTY` by the unchanged phase 7A rule. Its per-class figures are reported in full and in exactly the same detail as every other class, and they decide nothing. No support threshold was invented or relaxed after the holdout's support was seen.
- **Each model was trained once and evaluated once per split.** Run-to-run variance is UNKNOWN on this setup, so no difference reported here is an effect size and none is backed by a significance test.
- **The holdout holds 65 images.** Every figure carries the sampling uncertainty of a set that size, and per-class figures carry more of it than the aggregates.
- **Precision and recall depend on an operating point** and are reported at the project's frozen operational confidence. They are not threshold-independent properties of either model, and the framework's own F1-maximising pair is a different quantity that is never differenced against them.
- **The split is a group-aware and class-aware constrained split**, not a perfectly stratified one. Nothing establishes that two images in different splits do not share a site, a day, a camera or a worker, so the holdout estimates generalisation to held-out images from this dataset, not to a new construction site.
- **The canonical and the native framework metrics are different quantities** produced by different implementations against different ground-truth documents at different confidences. They are never differenced.
- **Computational cost was not measured on the holdout.** The committed evidence remains phase 10C's controlled local hardware benchmark, valid for that machine, that runtime and that protocol.
- **The segmentation adapter is lossy.** Phase 8A measured its round-trip at mean mask IoU 0.973066 with no instance exact; some portion of the segmenter's mask error is attributable to it, and nothing here separates the two.

## 15. No test-driven tuning

**Nothing in this repository was tuned, selected or changed in response to a holdout number.** The holdout was read once, under a protocol frozen before it could be read. Specifically, during and after this phase:

- no model was trained, fine-tuned or retrained;
- no model, architecture or checkpoint was selected;
- no hyperparameter was changed;
- no threshold was tuned or swept, and no operating point was chosen with a result in view;
- no dataset cleaning, annotation correction or class regrouping was motivated by a test outcome;
- no inference was re-run, and no second attempt exists;
- no metric was added after the results were seen, and no evaluator was swapped;
- no qualitative example was replaced for being unattractive or inconvenient;
- no winner was declared, and no composite, weighted or aggregate score was computed.

`FINAL_TEST_OBSERVED`. Model selection, hyperparameter tuning, threshold tuning and performance-motivated data cleaning are **CLOSED**. The results above may inform reporting, discussion and limitations. They may not trigger a D3, an S2, retraining, threshold optimisation, class regrouping, data filtering or a new model selection within the reported experiment.

---

### Artifact fingerprints

| Artifact | Fingerprint |
| --- | --- |
| `detector_test_prediction_sha256` | `bfcf35762b56a761c152ab14035b0f3c3c3cb2c6faafe65de3fee493403053b6` |
| `segmenter_test_prediction_sha256` | `181d036c3b4e7e039b72061fc3f4e4ee7291a45b5fa7f8ead433e328314ed504` |
| `detector_test_result_sha256` | `2e0837ca75fbdfc76ff18075788fa622fbf1812b2f2ba441f56965299004dff5` |
| `segmenter_test_result_sha256` | `f6b0baa96ae1255b987955d183a6e4f58c2eeffa49beada17a8e91393392bccf` |
| `direct_iou_test_result_sha256` | `4a30d7de4b519f65ca272530675cd0b306e3363386b0422d8feb8b45d84bb30c` |
| `qualitative_selection_sha256` | `d47df93114b1e5e2e3c44287c7f317cb9873d7eabdb372cb1c31e7c78ac946c0` |
| protocol fingerprint | `a5a328b3a8e49b06fb8fd9e792abcf43ccdd9aac5422729814dac0dbadc1daef` |

Every reported figure derives from the two persisted prediction sets those fingerprints cover. A report whose prediction fingerprint does not match the ledger's is not a rebuild; it is a second run.
