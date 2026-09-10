# Detector-versus-segmenter comparison protocol

Phase 10A · status `FROZEN_NOT_EXECUTED` · protocol fingerprint `d92a157637baf52f9a7248bf24257d5a177e8496b7727ffb5afc874ab96d1c4d`

**This document contains no results.** It is written before any comparison runs, which is the only thing that makes it a protocol. Both models were frozen first - the detector in phase 7D, the segmenter in phase 8G - so nothing here could have been chosen to flatter either of them.

Repository commit at production time: `29bffdceecbde2d39864ef66a2d63bafe79b1699`.

## 1. Scientific question

> What additional spatial and operational information does the frozen instance segmentation model provide beyond the frozen bounding-box detector, and what computational cost does that additional information introduce?

`PREDECLARED_PROTOCOL`. The detector emits a class, a confidence and a box. The segmenter emits those plus an instance mask. The comparison asks what the mask adds and what it costs; it does not rank the two, and no aggregate score is defined.

## 2. Frozen model identities

| | Detector | Segmenter |
| --- | --- | --- |
| Experiment | **D2** | **S1** |
| Role | `FINAL_OBJECT_DETECTOR` | `FINAL_INSTANCE_SEGMENTER` |
| Model | YOLO11n | YOLO11n-seg |
| imgsz | 768 | 768 |
| Checkpoint SHA-256 | `0466f872a9de22898c70d8834cf1fcfb3d77f6c9fb27e81ee6248d3d19cbc206` | `29337d671459d0f742fb713613cc41d0eafcb8a21f5846ac0da65b2ffc024f20` |
| Bytes | 5502289 | 6041685 |
| Outputs | class, confidence, bounding_box | class, confidence, bounding_box, instance_mask |

`FROZEN_MODEL`. Both were verified by digest through their freeze accessors and **neither was executed** (`executed_in_this_phase: False`). The segmenter additionally carries `overlap_mask: False` and `mask_ratio: 4`, which are part of its identity.

## 3. Comparison scope

* **A_recognition** - Does the segmenter retain comparable object localisation, measured as canonical box AP against the same ground truth by the same evaluator?
* **B_spatial_information** - Which spatial quantities do predicted masks make available, and which of them a bounding box can already approximate?
* **C_computational_cost** - What latency and memory does the mask branch add, measured under one symmetric benchmark on one machine?
* **D_operational_reasoning** - Where do mask-based and box-based person-PPE spatial associations agree, and where do they disagree?

## 4. Validation population

`validation` only: **65 images**, 304 annotations, from the frozen phase 5C.2 split (`a230869ff4cb45f53654d67357a27f2a0def6d8ba79f9fa4880f0fdda2f046cc`). Membership fingerprint `54e4ae8afd711314b472649b1564286d0a6d3d8f13e32c035d31402cbfa5f152`, committed in `reports/detector_segmenter_comparison_membership.csv`.

Both models see the same images in the same order (`identical_images_for_both_models: True`) and nothing is re-split (`resplit: False`).

## 5. Holdout policy

`HOLDOUT_POLICY` `PROTECTED_NOT_ACCESSED`. Phase 10A froze a comparison protocol. It ran no model, produced no prediction, measured no latency and read no image pixel. The holdout was not read, materialised, adapted, counted, predicted on or inspected; no holdout identifier, image, prediction or statistic exists in any artifact this phase wrote.

## 6-8. Recognition comparison and the AP protocol

`CANONICAL_EVALUATION`. Both models' predicted boxes are scored against the **same** canonical ground truth by the **same** external evaluator.

* Implementation: `pycocotools.cocoeval.COCOeval`, `iouType='bbox'`
* IoU thresholds: [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
* `maxDets`: [1, 10, 100]
* Ground truth: `CANONICAL_COCO_DETECTION_BOXES` - `data/processed/canonical/annotations/detection_validation.coco.json`
* Category ids: `CANONICAL_CATEGORY_IDS_USED_DIRECTLY_NO_REMAPPING`

The two models run through different framework validation paths, so their native box metrics are not guaranteed to be computed identically. One external evaluator applied to both removes that doubt. The canonical boxes are the phase 5D ones, derived from segmentation polygons, never the provider's stored boxes.

AP inference, for both models: imgsz 768, conf **0.001**, NMS IoU 0.7, max_det 300, augment False, TTA False, precision **FP32** (`quantize: 32`).

`LIMITATION`. Each model may propose up to 300 candidates while COCOeval scores at its conventional cap of 100. Two different numbers, both recorded.

**Box comparability.** S1 predicts boxes directly, and those are what a user of the model would get. Re-deriving them from its masks after inference would improve their geometric consistency with the mask branch and would then be measuring a post-processing choice this project invented, not the model.

## 9. Operational inference protocol

`OPERATIONAL_ANALYSIS`. imgsz 768, conf **0.25**, NMS IoU 0.7, max_det 300, augment False, precision **FP32**.

Neither threshold is tuned, for either model. 0.001 belongs to the AP protocol and 0.25 to the operational protocol; they are never mixed, averaged or swapped, and neither is selected by looking at a result.

## 10-11. Why masks add representation, and what they measure

`SPATIAL_INFORMATION`. A box asserts an axis-aligned rectangle; a mask asserts which pixels belong to the instance. The quantities below are what that difference makes computable.

| Quantity | Definition |
| --- | --- |
| `INSTANCE_AREA_PIXELS` | Count of foreground pixels in the predicted instance mask, on the original image canvas. |
| `MASK_TO_BOX_FILL_RATIO` | Mask area divided by the area of the model's own predicted box. How much of the box the object actually occupies. |
| `MASK_CENTROID` | Centroid of the mask's foreground pixels, in original image coordinates. |
| `SHAPE_EXTENT` | Foreground support relative to the tight bounding rectangle of the mask itself, describing how far the shape departs from filling a rectangle. |
| `PERSON_PPE_MASK_INTERSECTION` | Count of pixels shared between a PPE instance mask and a person instance mask. |
| `PERSON_PPE_MASK_CONTAINMENT` | Intersection divided by the PPE mask's own area: how much of the PPE lies inside the person. |
| `VISIBLE_PPE_COVERAGE_PROXY` | An overlap-derived operational proxy for how much visible PPE support a person instance has. A proxy, named as one, and not a compliance measure. |

`LIMITATION` `MASK_ENABLED_SPATIAL_MEASUREMENTS. They describe predicted geometry. They are not ground truth, not compliance labels, and not accuracy metrics.`

## 12. Box-only proxies

`BOX_PROXY`. Each mask quantity is paired with what a box-only pipeline could compute instead, or declared to have no equivalent. That pairing is the answer to the question this comparison asks.

| Mask measurement | Box proxy |
| --- | --- |
| `INSTANCE_AREA_PIXELS` | `BOX_AREA_PIXELS` |
| `MASK_TO_BOX_FILL_RATIO` | `NO_BOX_ONLY_EQUIVALENT` |
| `MASK_CENTROID` | `BOX_CENTER` |
| `SHAPE_EXTENT` | `NO_BOX_ONLY_EQUIVALENT` |
| `PERSON_PPE_MASK_INTERSECTION` | `BOX_INTERSECTION_AREA` |
| `PERSON_PPE_MASK_CONTAINMENT` | `BOX_INTERSECTION_OVER_PPE_BOX_AREA` |
| `VISIBLE_PPE_COVERAGE_PROXY` | `BOX_OVERLAP_DERIVED_COVERAGE` |

A declared proxy is not a claim that it is equivalent. It is a claim that a box-only pipeline could compute something in its place, so the comparison can measure how far the two answers diverge rather than asserting a difference.

## 13. Person-PPE association

`SPATIAL_ASSOCIATION_ANALYSIS` · `DESCRIPTIVE_OPERATIONAL_ANALYSIS_NOT_AN_ACCURACY_METRIC`. Relationships: `helmet_to_person`, `vest_to_person`, evaluated at the operational threshold with a containment floor of 0.5.

* Mask rule: A PPE instance is associated with the person instance whose mask it overlaps most, subject to a containment floor declared before any result exists.
* Box rule: The same rule computed on the models' predicted boxes: intersection over the PPE box's own area, associating with the highest-scoring person box.
* Tie-breaking: `HIGHEST_OVERLAP_THEN_LOWEST_PERSON_INSTANCE_INDEX`, `deterministic: True`

No appearance embeddings (`False`), no tracking (`False`), no learned association (`False`), and the five frozen classes are not collapsed (`classes_collapsed: False`).

`LIMITATION`. The five frozen classes stay distinct. `helmet_on_head` and `vest_on_body` already encode a provider-level worn state, so this analysis does not invent a compliance label on top of them and does not report accuracy: there is no canonical compliance ground truth in this project to be accurate against.

## 14. Association disagreement categories

* `BOX_AND_MASK_AGREE`
* `BOX_ONLY_ASSOCIATION`
* `MASK_ONLY_ASSOCIATION`
* `NEITHER_ASSOCIATION`

Each deterministic person-PPE candidate pair falls in exactly one. This is a descriptive comparison of two geometric rules, **not** a model accuracy metric: there is no association ground truth to be accurate against.

## 15-18. Latency benchmark

`CONTROLLED_HARDWARE_BENCHMARK` `CONTROLLED_LOCAL_HARDWARE_BENCHMARK`. batch 1, imgsz 768, precision **FP32** (`quantize: 32`), warmup **20** iterations discarded, **30** timed repetitions per image over **20** images.

### Timing boundaries

**`MODEL_INFERENCE_LATENCY_MS`** - The model core, as consistently as the framework allows for both.

* includes: forward pass on an already-prepared input tensor
* excludes: image decode, preprocessing, postprocessing, mask reconstruction

**`END_TO_END_MODEL_OUTPUT_LATENCY_MS`** - What a caller actually waits for. Mask reconstruction is inside this boundary for the segmenter, because excluding it would hide precisely the cost this comparison exists to quantify.

* includes: input preprocessing, forward pass, NMS and postprocessing, mask reconstruction to expose final instance masks

The distinction matters because segmentation adds postprocessing, not just a heavier forward pass. Mask reconstruction is **inside** the segmenter's end-to-end boundary (`segmenter_mask_reconstruction_included: True`); putting it outside would hide the cost this comparison exists to quantify.

### GPU synchronization

CUDA work is asynchronous. Without an explicit synchronize on both edges a wall-clock reading measures how long it took to queue the work, not to run it, and the faster-to-queue model would appear faster. Timing primitive: `time.perf_counter`, the same for both models.

### Statistics

Reported for each boundary and each model: mean, median, std, p50, p90, p95, p99, min, max.

* `absolute_latency_delta_ms` - S1 minus D2, at an identical measurement boundary
* `images_per_second_from_mean` - batch 1, derived from the mean, never from the fastest iteration
* `relative_latency_cost` - (S1 / D2) - 1
* `throughput_ratio` - S1 images per second divided by D2 images per second

## 19. Deterministic benchmark membership

`STABLE_SHA256_RANK_OF_IMAGE_ID`. Ranked by the SHA-256 of the image identifier and truncated to the first 20. Chosen without opening a single image, so the subset cannot have been picked for being easy, crowded or visually interesting. Ordered fingerprint `45059c2cdda1285b4fa6dbfdcfbeac6fedcad551e551e6810b1369af90f7f162`, committed in `reports/detector_segmenter_latency_membership.csv`.

## 20. Execution-order control

Pass one, for each benchmark image in the frozen order: time the detector, then time the segmenter. Pass two, same images and order: time the segmenter, then the detector. Both passes contribute to the reported distribution, so any residual thermal or ordering drift falls on both models equally instead of on whichever ran second.

`interleaved: True`, `symmetric: True`, `randomized: False`. The runner aborts if either model deviates from the frozen invariants (`abort_on_deviation: True`):

* same machine
* same GPU
* same driver and CUDA runtime
* same Python environment
* same process where practical
* same source image bytes
* same input resolution
* same batch size
* same precision
* same warmup count
* same benchmark images in the same order
* same timed iteration count
* same synchronization strategy
* same timing primitive

## 21. Memory measurement

batch 1, imgsz 768, peak stats reset **after** warmup (`reset_peak_stats_after_warmup: True`). Recorded: peak_memory_allocated, peak_memory_reserved. Peak stats are reset after warmup so the figure describes inference, not the allocator's warmup high-water mark. Training-time memory is a different quantity and is never substituted.

## 22. Model complexity

`DESCRIPTIVE_COMPLEXITY_ONLY`. Fields: parameters, gflops. Read from the committed experiment manifests where they already carry authoritative values. Nothing is retrained to obtain them.

## 23. Reporting plan

* Box metrics: `CANONICAL_BOX_MAP50_95`, `CANONICAL_BOX_MAP50`
* Per-class box AP: True
* Deltas reported: True
* Aggregate benefit score: **False**
* Winner declared: **False**

Deltas are reported for description, not to select. Neither frozen model changes as a result of this comparison, and no retraining follows from it.

Spatial-value comparisons frozen in advance:

* `mask_versus_box_overlap_magnitude`
* `mask_versus_box_containment_magnitude`
* `absolute_distance_between_box_center_and_mask_centroid`
* `mask_fill_ratio_distribution`
* `cases_where_box_overlap_exists_but_mask_overlap_is_negligible`
* `cases_where_boxes_overlap_strongly_while_masks_separate_the_instances`
* `cases_where_mask_association_and_box_proxy_association_disagree`

## 24. Interpretation boundaries

* **BENEFIT** - additional measurable spatial support that masks make available
* **COST** - latency, memory and computational overhead
* **RECOGNITION_TRADEOFF** - the difference in canonical box localisation
* **OPERATIONAL_VALUE** - how far mask measurement and bounding-box proxies diverge for person-PPE spatial association

`single_aggregate_score: False`, `weighted_cost_benefit_index: False`. Benefit and cost are reported side by side and never collapsed.

## 25. Limitations

`LIMITATION`.

* Validation only. The holdout has never been evaluated and takes no part in this comparison.
* The two models do not solve the same output task. The comparison describes what is gained and what it costs; it does not rank them.
* One run of each model was ever trained. Run-to-run variance is UNKNOWN for both, so a small recognition delta is not evidence of an ordering.
* The spatial quantities are derived from predictions, not from ground truth. They describe what the model asserts, not what is true in the scene.
* There is no canonical person-PPE association ground truth, so the association analysis is descriptive and reports no accuracy.
* The latency benchmark is valid for this machine, this runtime and this protocol. It is not a universal statement about either architecture.

## 26. Next phase

`PHASE_10B_CONTROLLED_VALIDATION_RECOGNITION_AND_SPATIAL_COMPARISON`. Nothing has been executed here: `models_executed_in_this_phase: 0`, `predictions_produced: 0`, `latency_measurements_taken: 0`, `images_read: 0`.

