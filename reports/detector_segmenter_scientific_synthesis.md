# Detector versus segmenter - scientific and operational synthesis

Phase **10D** · Status **`DETECTOR_SEGMENTER_SCIENTIFIC_SYNTHESIS_COMPLETE`** · `SCIENTIFIC_SYNTHESIS`

**This phase executed no model.** It trained nothing, ran no inference, recomputed no average precision, reran neither the phase 10B spatial analysis nor the phase 10C benchmark, tuned no threshold and read no holdout data. Every number below is copied from a committed artifact, named beside it.

## 1. Research question

> What additional spatial and operational information does instance segmentation provide beyond bounding boxes, and what is its performance and latency cost?

The answer is organised on exactly four axes, each separately interpretable and none combined with another:

1. `RECOGNITION_LOCALIZATION`
2. `SPATIAL_REPRESENTATION_GAIN`
3. `OPERATIONAL_ASSOCIATION_VALUE`
4. `COMPUTATIONAL_COST`

## 2. Frozen detector and segmenter

|  | Detector | Segmenter |
| --- | --- | --- |
| Experiment | **D2** | **S1** |
| Architecture | YOLO11n | YOLO11n-seg |
| Input size | 768 | 768 |
| `mask_ratio` | n/a | 4 |
| `overlap_mask` | n/a | false |
| Checkpoint SHA-256 | `0466f872a9de22898c70d8834cf1fcfb3d77f6c9fb27e81ee6248d3d19cbc206` | `29337d671459d0f742fb713613cc41d0eafcb8a21f5846ac0da65b2ffc024f20` |
| Identity fingerprint | `84d30d64a1e7ebfc4e4763643ef3a3b5c8a5ce3cedc74bbefdb8031d48235a6e` | `63ef41961ba3fedfef46658754940fb6ce075cdd93087b926d7c6620402579a7` |
| Trained in this phase | no | no |
| Modified in this phase | no | no |

> `LIMITATION` D2 emits class, confidence and a box; S1 emits those and an instance mask. They are the project's frozen models for two different output tasks, not two candidates for one. Neither is a drop-in replacement for the other, and their architectures are related but not identical.

## 3. Evidence sources

`COMMITTED_EVIDENCE` - every figure in this document resolves to one of these committed artifacts, by file digest and, where the artifact records one, by its own semantic fingerprint.

| Role | Artifact | File SHA-256 | Recorded fingerprint |
| --- | --- | --- | --- |
| `box_comparison` | `reports/detector_segmenter_box_comparison.json` | `ebeabe3f6bf87e2059446d21bba7ae61cbd717a2b38b4b6eba90fbe90ec76704` | `68c9826a314be2ea24e4f89b3798acc5bdead6739d82b50a33f1d0984eaf1b27` |
| `comparison_protocol` | `reports/detector_segmenter_comparison_protocol.json` | `cdef5fb260b87374cc506a314de1127ccad22d91914a5f9550697b40f48eb40e` | `d92a157637baf52f9a7248bf24257d5a177e8496b7727ffb5afc874ab96d1c4d` |
| `final_detector` | `reports/final_detector_manifest.json` | `97c1ecaf6fdd8c7437cd063a233a3959fc7144e9f69a2c966d5b452d4bbfc205` | `84d30d64a1e7ebfc4e4763643ef3a3b5c8a5ce3cedc74bbefdb8031d48235a6e` |
| `final_segmenter` | `reports/final_segmenter_manifest.json` | `dbdc63211c73ceb0f6ba78e493665fb8fc5432697164466e61b2c086d66acbeb` | `63ef41961ba3fedfef46658754940fb6ce075cdd93087b926d7c6620402579a7` |
| `latency_benchmark` | `reports/detector_segmenter_latency_comparison.json` | `e7b98d37e6f9018e4cdbd112bfa59b9589050c6ae7452c4d5ef69dea5ee0fc02` | `27c1705f15887690f575b91e17e892df8e0226f93b16a35e2b799b505fdc8813` |
| `memory_benchmark` | `reports/detector_segmenter_memory_comparison.json` | `e3a4e487860181ebb8deaaf638625f3159071860b0b0faabeb55ea4757b7f418` | `7f452c8a7be93b8dbdec9f89d316629522c092918dbc53420bb18e9d95132ed6` |
| `spatial_comparison` | `reports/detector_segmenter_spatial_comparison.json` | `49420d585b58e1a1ca0d5992c4b046f0b1260cf18bc566d6f397ef73b5598f69` | `3988bcf688633c63bc0c422ba515246119b94a1c6b8b508c09ff215bda74e383` |
| `taxonomy_correction` | `reports/association_taxonomy_correction.provenance.json` | `0a01051ca1adc8bffca674d325a1f0b10bf9e3e5ae6cba531641d2737693d72e` | `-` |

Comparison protocol fingerprint: `d92a157637baf52f9a7248bf24257d5a177e8496b7727ffb5afc874ab96d1c4d` (phase 10A, historical and unmodified).

## 4. Recognition and localisation

`COMMITTED_EVIDENCE` - source: `reports/detector_segmenter_box_comparison.json`.

One external evaluator judges both models: pycocotools.cocoeval.COCOeval at `iouType='bbox'` against `CANONICAL_COCO_DETECTION_BOXES`, at conf 0.001, over the frozen 65 validation images and 304 canonical annotations. S1's boxes are S1's own predicted boxes, not boxes re-derived from its masks.

| Metric | D2 | S1 | Delta |
| --- | --- | --- | --- |
| Canonical box mAP@0.50:0.95 (all classes) | **0.48539** | **0.505682** | **+0.020292** |
| Canonical box mAP@0.50 (all classes) | 0.641107 | 0.692955 | +0.051848 |
| Supported-class macro AP@0.50:0.95 (descriptive) | **0.589729** | **0.581689** | **-0.008040** |

`SCIENTIFIC_SYNTHESIS` - **S1 retains broadly similar localisation performance to D2 while adding mask output.** The positive all-class delta is **not** robust evidence that S1 is the superior localiser: it is dominated by `vest_loose`, already frozen `DESCRIPTIVE_HIGH_UNCERTAINTY`. Over the four adequately supported classes the same comparison gives -0.008040, placing the segmenter marginally below the detector.

Neither statement is a ranking. It is not claimed that S1 detects objects better than D2, nor that D2 definitively detects objects better than S1.

> `LIMITATION` The supported-class macro is `POST_HOC_DESCRIPTIVE_SUPPORT_SENSITIVITY`: descriptive, reusing the project's pre-existing support rule (`PREEXISTING_PHASE_7A_SUPPORT_RULE_REUSED_UNCHANGED`). It is **not** a frozen phase 10A metric, **not** a selection rule and **not** a significance test, and it changes no frozen number.

### Per-class localisation trade-off

| Class | D2 AP@0.50:0.95 | S1 AP@0.50:0.95 | Delta | Contribution | Status |
| --- | --- | --- | --- | --- | --- |
| `helmet_loose` | 0.767262 | 0.748577 | -0.018685 | -0.003737 | COMPARISON_REPORTED |
| `helmet_on_head` | 0.628022 | 0.629018 | +0.000996 | +0.000199 | COMPARISON_REPORTED |
| `person` | 0.485281 | 0.517231 | +0.031950 | +0.006390 | COMPARISON_REPORTED |
| `vest_loose` | 0.068034 | 0.201654 | +0.133620 | +0.026724 | DESCRIPTIVE_HIGH_UNCERTAINTY |
| `vest_on_body` | 0.47835 | 0.431929 | -0.046421 | -0.009284 | COMPARISON_REPORTED |

Some classes improved (`helmet_on_head`, `person`, `vest_loose`) and some regressed (`helmet_loose`, `vest_on_body`). **Why any individual class moved is UNKNOWN** - this comparison ran no experiment isolating a cause, and nothing was repeated, so run-to-run variance on this setup is also UNKNOWN.

> `LIMITATION` These are canonical-evaluator figures. They are NOT the native framework box metrics either model reported in its own experiment phase, and the two must never be differenced: different evaluator implementation, different ground-truth document and a different confidence. Only the D2-versus-S1 delta computed here, by one evaluator over one ground truth, is a comparison.

## 5. Rare-class sensitivity

`vest_loose` holds **one** validation image and eight instances under the frozen split, and is frozen `DESCRIPTIVE_HIGH_UNCERTAINTY`. Its canonical box AP moved +0.133620, contributing +0.026724 to the five-class unweighted mean - **more than the whole all-class delta of +0.020292**. Excluding it, the mean over the remaining four classes is -0.008040.

It is reported in full and decides nothing. Never quote the all-class delta on its own.

## 6. What masks represent beyond boxes

`COMMITTED_EVIDENCE` - source: `reports/detector_segmenter_spatial_comparison.json`, operational inference at conf 0.25, 336 segmenter instances with 336 masks reconstructed on the original canvas and 0 excluded.

`REPRESENTATION_GAIN` - **masks represent foreground support inside bounding boxes.** The median predicted instance mask occupied **0.664433** of its bounding rectangle, illustrating information that the rectangular representation does not encode.

| Quantity | Median | P25 | P75 | Mean | Box equivalent |
| --- | --- | --- | --- | --- | --- |
| `MASK_TO_BOX_FILL_RATIO` | **0.664433** | 0.555943 | 0.775478 | 0.649956 | `NO_BOX_ONLY_EQUIVALENT` |
| `SHAPE_EXTENT` | **0.672173** | 0.570469 | 0.78614 | 0.661156 | `NO_BOX_ONLY_EQUIVALENT` |

> `LIMITATION` This is **not** a 33.6 per cent background-error rate. The measure compares *predicted mask support* with *predicted box area*; no ground-truth background classification enters it.

## 7. Mask-only geometric quantities

`REPRESENTATION_GAIN` - two of the seven frozen spatial features were declared `NO_BOX_ONLY_EQUIVALENT` **before any measurement was taken**: `MASK_TO_BOX_FILL_RATIO`, `SHAPE_EXTENT`.

A bounding box cannot directly encode non-rectangular foreground support. It has no way to express how much of itself is object, or how far the object departs from a rectangle. These quantities are therefore a gain in **what is computable**, not a demonstration of improved predictive accuracy: nothing here compares either representation against ground-truth geometry.

Deciding which quantities a box could approximate *after* seeing the numbers would have been circular, which is why the pairing was frozen in phase 10A.

## 8. Box-proxy inflation and refinement

`PROXY_REFINEMENT` - for the 4 features where a mask measurement and a like-for-like box proxy both exist, the box proxy was **systematically inflated** relative to the mask measurement. Each pair below is the same summary statistic of the same population under the two representations. `MASK_CENTROID` also has a box proxy but is reported separately below, because its recorded statistic is a displacement between the two rather than a pair of like measurements.

| Quantity | Box proxy | Mask measurement (mean) | Box proxy (mean) |
| --- | --- | --- | --- |
| `INSTANCE_AREA_PIXELS` | `BOX_AREA_PIXELS` | 144562.997024 | 237205.836504 |
| `PERSON_PPE_MASK_CONTAINMENT` | `BOX_INTERSECTION_OVER_PPE_BOX_AREA` | 0.310145 | 0.338868 |
| `PERSON_PPE_MASK_INTERSECTION` | `BOX_INTERSECTION_AREA` | 26828.34384 | 47280.877637 |
| `VISIBLE_PPE_COVERAGE_PROXY` | `BOX_OVERLAP_DERIVED_COVERAGE` | 0.176234 | 0.181968 |

> `LIMITATION` Read this as `PROXY_REFINEMENT`, **never** as `BOX_ERROR`. No ground-truth geometry entered the comparison, so neither representation is shown to be right or wrong against it. Unlike statistics are never compared: mean is set against mean, median against median, over the same instances.

A further geometric disagreement is recorded as a count rather than binned: **11 of 349 candidate pairs (0.031519)** had overlapping boxes whose masks shared no pixel at all. Phase 10A declared no threshold for 'strong' or 'minimal' overlap, so none was invented.

### Centroid information

`PROXY_REFINEMENT` - displacement between the model's own box centre and the mask centroid: median **16.048565 px**, P95 **126.933417 px**, max **262.773275 px** over 336 instances.

A box centre and a foreground centroid can differ materially for irregular, partially visible or spatially imbalanced shapes, and the mask makes the second computable. That is additional geometric information.

> `LIMITATION` The mask centroid is **not** "the true object centre". It is a different geometric quantity. The mask centroid averages pixel indices while the box is in continuous coordinates, so a mask perfectly filling its box reports about 0.71 px rather than 0. The offset is constant and far below the displacements described.

## 9. Person-PPE association

`COMMITTED_EVIDENCE` - `SPATIAL_ASSOCIATION_ANALYSIS`, frozen containment floor **0.5**.

The **geometry-isolating** reading is the one that answers the question: it holds the model and its instances fixed - S1's own boxes against S1's own masks - and varies only the shape representation.

| Category | Geometry-isolating | Pipeline-level (confounded) |
| --- | --- | --- |
| `BOX_AND_MASK_AGREE` | 103 | 81 |
| `BOX_ONLY_ASSOCIATION` | 3 | 7 |
| `MASK_ONLY_ASSOCIATION` | **0** | 6 |
| `NEITHER_ASSOCIATION` | 66 | 62 |
| Taxonomy exceptions | 1 | 17 |
| Classified / total | 172 / 173 | 156 / 173 |
| Taxonomy coverage | 0.99422 | 0.901734 |

Counts are over all relationships; percentages elsewhere use `classified_relationships`.

`SCIENTIFIC_SYNTHESIS` - **at the frozen association rule and on this validation population, instance masks did NOT demonstrate a substantial advantage in discovering additional person-PPE associations.** Specifically, the geometry-isolating analysis found **zero `MASK_ONLY_ASSOCIATION`** cases: the mask changed almost no association decision the box rule had already made.

> `LIMITATION` This does not generalise beyond this containment floor and this population. The pipeline-level column varies the **model** as well as the geometry - different instances, different predictions - and its disagreements **must not be attributed solely to mask-versus-box geometry**.

## 10. Association taxonomy limitation

`OPERATIONAL_LIMITATION` - the frozen four-category taxonomy is `FROZEN_TAXONOMY_NON_EXHAUSTIVE_FOR_OBSERVED_DATA`.

Phase 10A froze four categories on the implicit assumption that a rule either associates or it does not. A state occurred that none of them describes: **box and mask rules both associated, but selected different persons** (`BOTH_RULES_ASSOCIATE_DIFFERENT_PERSON`).

It is recorded `UNCLASSIFIED_BY_FROZEN_FOUR_CATEGORY_TAXONOMY` - counted on its own, excluded from the classified denominator - and is **not** a fifth frozen category (`fifth_peer_category_added: false`). The phase 10A protocol is historical and was not modified.

> `LIMITATION` A different-person outcome is an ASSOCIATION_RULE_DISAGREEMENT, not an association error. The project holds no person-PPE association ground truth, so neither rule's answer can be called wrong.

## 11. Operational proxy limitation

`OPERATIONAL_LIMITATION` - `VISIBLE_PPE_COVERAGE_PROXY` remains `INTERPRETIVE_OPERATIONAL_PROXY` and is **not** used as a primary scientific conclusion.

**No PPE compliance accuracy, safety-violation accuracy or correct-wearing classification accuracy is claimed here, and none is claimable**: the project holds no canonical compliance ground truth, so there is nothing such a claim could be measured against. What may be said is only this: masks permit a more spatially specific coverage-like proxy than rectangular boxes.

## 12. Model inference cost

`CONTROLLED_LOCAL_HARDWARE_BENCHMARK` - source: `reports/detector_segmenter_latency_comparison.json`. Batch 1 at imgsz 768 in FP32, conf 0.25, 20 frozen validation images, 20 warmup iterations discarded and 30 timed repetitions per block, in a symmetric interleaved order over 80 blocks - 4800 timed readings.

`MODEL_INFERENCE_LATENCY_MS` - the forward pass alone.

| Statistic (ms) | D2 | S1 |
| --- | --- | --- |
| mean | 6.05574 | 7.777487 |
| median | 4.5507 | 5.4443 |
| p90 | 9.95982 | 11.43956 |
| p95 | 10.12162 | 11.59655 |
| p99 | 10.514269 | 11.911593 |
| min | 4.1543 | 4.9609 |
| max | 11.1425 | 12.2581 |
| std | 2.412568 | 2.91804 |

Mean delta **+1.721747 ms** (**+0.284317** relative).

## 13. End-to-end output cost

`END_TO_END_MODEL_OUTPUT_LATENCY_MS` - what a caller waits for: preprocessing, the forward pass, NMS and postprocessing, **including the segmenter's mask reconstruction onto the original canvas**. This is the primary operational cost statement, because it is the boundary at which both models have usable outputs.

| Statistic (ms) | D2 | S1 |
| --- | --- | --- |
| mean | 9.157766 | 11.914757 |
| median | 7.30475 | 9.96245 |
| p90 | 13.87224 | 16.60673 |
| p95 | 14.145295 | 17.240505 |
| p99 | 14.500324 | 24.721743 |
| min | 6.0559 | 7.7405 |
| max | 16.3404 | 29.4292 |
| std | 3.002305 | 4.003441 |

Mean delta **+2.756991 ms** (**+0.301055** relative, approximately 30 per cent).

> `LIMITATION` Call this `ADDITIONAL_SEGMENTATION_PIPELINE_COST`, never `PURE_MASK_RECONSTRUCTION_CAUSAL_COST`. YOLO11n and YOLO11n-seg differ in the mask branch of the network as well as in postprocessing, and this benchmark isolates neither from the other. The measured difference is the cost of the whole segmentation pipeline relative to the whole detection pipeline.

## 14. Throughput

`MEAN_DERIVED_BATCH1_THROUGHPUT` - 1000 / mean latency at batch 1.

| Boundary | D2 (images/s) | S1 (images/s) | Ratio |
| --- | --- | --- | --- |
| `MODEL_INFERENCE_LATENCY_MS` | 165.132585 | 128.576235 | 0.778624 |
| `END_TO_END_MODEL_OUTPUT_LATENCY_MS` | 109.196937 | 83.929534 | 0.768607 |

> `LIMITATION` This is a latency reciprocal, not batched throughput and not application or video FPS. It is not derived from the fastest iteration. A system that does not process frames independently under these same assumptions will not see these rates.

## 15. Inference memory

`CONTROLLED_LOCAL_HARDWARE_BENCHMARK` - `INFERENCE_MEMORY`, source: `reports/detector_segmenter_memory_comparison.json`. Each model measured in a dedicated process (`SINGLE_MODEL_RESIDENCY_IN_A_DEDICATED_PROCESS`), peak statistics reset after the frozen warmup.

| Peak | D2 | S1 | Ratio |
| --- | --- | --- | --- |
| Allocated | 0.073403 GiB (78815744 B) | 0.231621 GiB (248700928 B) | **3.155473** |
| Reserved | 0.125 GiB (134217728 B) | 0.296875 GiB (318767104 B) | **2.375** |

**Both statements hold together.** The relative overhead is substantial - over three times the allocated peak and 2.375 times the reserved peak. The absolute footprint is low: on the measured NVIDIA GeForce RTX 5070 Laptop GPU with 8546484224 bytes of device memory, both models peak well under a third of a GiB reserved. S1 is **not** memory-heavy in absolute terms.

> `LIMITATION` `INFERENCE_MEMORY` may never be compared with training memory: NOT_MEASURED_TRAINING_MEMORY_IS_A_DIFFERENT_QUANTITY.

## 16. Static model complexity

`STATIC_MODEL_COMPLEXITY` - read from the committed experiment manifests, not recomputed.

| Figure | D2 | S1 | Delta |
| --- | --- | --- | --- |
| Parameters (unfused) | 2624080 | 2843583 | +219503 |
| GFLOPs at the 640 reference input | 6.673 | 9.8 | +3.127000 |

> `LIMITATION` These are `FRAMEWORK_DEFAULT_640_NOT_THE_BENCHMARK_768`. `ultralytics.utils.torch_utils.get_flops` and `model_info` both default to `imgsz=640`, and both committed figures came from those paths. **They are not the FLOPs of the timed configuration**, no imgsz 768 value was derived, and latency was measured independently at imgsz 768. They explain none of the timings.

## 17. Benchmark distribution and DVFS limitation

`POST_HOC_HARDWARE_BEHAVIOR_DIAGNOSTIC` / `POST_HOC_DIAGNOSTIC_ONLY` - computed from all 4800 observations. **0 observations were discarded**, no outlier rejection was applied, nothing was normalised or rescaled, and the benchmark was not re-run.

The distribution is broad and multimodal, so the mean alone misleads. Mean-to-median ratios:

| Boundary | D2 mean/median | S1 mean/median | D2 block-mean range (ms) | S1 block-mean range (ms) |
| --- | --- | --- | --- | --- |
| `MODEL_INFERENCE_LATENCY_MS` | 1.330727 | 1.428556 | 4.318367 - 9.98153 | 5.118167 - 11.413863 |
| `END_TO_END_MODEL_OUTPUT_LATENCY_MS` | 1.253673 | 1.195967 | 6.33297 - 13.88759 | 8.092287 - 22.381947 |

The headline delta remains based on the frozen **mean** statistic; the median, P95 and range are reported beside it rather than replacing it.

> `LIMITATION` `causal_attribution: UNKNOWN`. The observed multimodality is *consistent with* mobile-GPU DVFS and power-state behaviour, and that stays `UNTESTED_HYPOTHESIS`: `NO_SYNCHRONIZED_PER_OBSERVATION_POWER_STATE_TELEMETRY` - no clock, P-state, utilisation, temperature or power reading accompanied the timed regions, so no observation maps to a device state and no alternative was excluded. **It is not stated that DVFS caused the distribution**, and `proportionality_across_models_demonstrated: false` - no proportional effect across the two models is claimed. The controlled symmetric order mitigates order bias; it does not prove the source of the multimodality.

## 18. Confidence-protocol-gap disclosure

`PRE_BENCHMARK_PROTOCOL_GAP_RESOLUTION`.

Phase 10A's latency subsection froze batch, resolution, precision, warmup, repetitions, membership and execution order - but **no confidence threshold**. It is therefore **false** to say that latency confidence 0.25 was explicitly frozen by phase 10A, and this synthesis does not say it.

Before any timing result existed, phase 10C resolved the benchmark to conf **0.25** (`operational_inference`), because that is the project's already frozen operational inference threshold - the protocol itself declares the AP block's 0.001 deliberately *not* an operating point. The resolution was applied equally to both models.

This does **not** invalidate the benchmark. It scopes it: `OPERATIONAL_OUTPUT_LATENCY_AT_CONF_0_25`. **No latency claim is made for conf 0.001** - a lower threshold pushes more candidates through NMS and, for the segmenter, more masks through reconstruction, and that was not measured.

## 19. Benefit-versus-cost interpretation

The four axes are reported separately and are **not** combined. There is no weighted score, no overall benefit score, no cost-benefit index and no single winner metric anywhere in this phase's artifacts, and the validator refuses one.

What the evidence supports is a conditional reading:

- **Recognition.** Broadly similar, with the positive aggregate carried by a one-image class and the supported-class reading marginally favouring the detector. Neither model is the established better localiser.
- **Spatial representation.** A real, measured gain for the segmenter, in quantities a rectangle cannot express and in the refinement of quantities it can only approximate.
- **Operational association.** No measured advantage at the frozen containment rule.
- **Computational cost.** A consistent, measured premium in latency and in inference memory on this machine.

## 20. Use-case-conditional recommendation

`USE_CASE_CONDITIONAL` - not `ONE_MODEL_UNIVERSALLY_SUPERIOR`.

**D2 is attractive when the application primarily needs:**

- object presence;
- the object class;
- a confidence score;
- bounding-box localisation;
- lower inference cost in latency and memory;

**S1 is attractive when the application additionally requires:**

- foreground support inside the box;
- non-rectangular geometry;
- mask area;
- mask-to-box fill ratio or shape extent;
- a mask centroid;
- more spatially specific overlap and containment measurements;

Both remain the project's frozen final models for their respective tasks. Neither is a drop-in replacement for the other: D2 emits class, confidence and a box; S1 emits those and an instance mask. They do not solve the same output task, and no architectural identity is claimed between them.

## 21. Central scientific answer

`SCIENTIFIC_SYNTHESIS`

> Instance segmentation did not demonstrate a robust localisation advantage over the dedicated detector on adequately supported classes - over those four classes the canonical box comparison gives -0.008040 - nor did it uncover substantial new person-PPE associations at the frozen containment rule of 0.5, where the geometry-isolating analysis found zero mask-only associations. Its demonstrated value was instead richer spatial representation: masks encode foreground support, non-rectangular shape and spatial measurements that bounding boxes cannot directly represent - the median predicted mask occupied 0.664433 of its own bounding rectangle - while reducing the systematic inflation of box-based area and intersection proxies. On the controlled local RTX 5070 Laptop FP32 benchmark, at the operational confidence of 0.25, exposing that additional mask output cost +2.756991 ms of mean end-to-end model-output latency (+0.301055 relative, approximately 30 per cent) and higher inference memory use.

`HOLDOUT_POLICY` - every figure above is a **validation** figure. `generalises_to_test: false`, `generalises_to_other_hardware: false`, `significance_tested: false`.

### Claim register

Each headline claim, with the artifact and field that support it and the limitation that travels with it. Intended for reuse by the academic report and the pitch, so that no claim is repeated without its scope.

| Claim | Evidence artifact | Evidence field | Scope |
| --- | --- | --- | --- |
| `LOCALIZATION_SIMILARITY` | `reports/detector_segmenter_box_comparison.json` | `metrics.CANONICAL_BOX_MAP50_95, supported_class_sensitivity.delta` | `VALIDATION_SPLIT_ONE_EXECUTION_PER_MODEL_CANONICAL_COCOEVAL_BBOX` |
| `MASK_REPRESENTATION_GAIN` | `reports/detector_segmenter_spatial_comparison.json` | `spatial_features.MASK_TO_BOX_FILL_RATIO.statistics, spatial_features.SHAPE_EXTENT.statistics, box_proxies.*.proxy` | `VALIDATION_SPLIT_SEGMENTER_PREDICTIONS_AT_OPERATIONAL_CONF_0_25_PREDICTED_MASK_AGAINST_PREDICTED_BOX` |
| `BOX_PROXY_INFLATION` | `reports/detector_segmenter_spatial_comparison.json` | `box_proxies.*.statistics.{mask_measurement,box_proxy}` | `VALIDATION_SPLIT_SAME_STATISTIC_COMPARED_WITH_SAME_STATISTIC_SEGMENTER_OWN_PREDICTIONS` |
| `NO_MAJOR_MASK_ONLY_ASSOCIATION_GAIN_AT_FROZEN_RULE` | `reports/detector_segmenter_spatial_comparison.json` | `association.geometry_isolating.frozen_category_counts` | `VALIDATION_SPLIT_GEOMETRY_ISOLATING_READING_AT_CONTAINMENT_FLOOR_0_50` |
| `END_TO_END_LATENCY_COST` | `reports/detector_segmenter_latency_comparison.json` | `latency.*.END_TO_END_MODEL_OUTPUT_LATENCY_MS, deltas.END_TO_END_MODEL_OUTPUT_LATENCY_MS` | `CONTROLLED_LOCAL_HARDWARE_BENCHMARK_RTX_5070_LAPTOP_FP32_BATCH_1_IMGSZ_768_OPERATIONAL_OUTPUT_LATENCY_AT_CONF_0_25` |
| `INFERENCE_MEMORY_OVERHEAD` | `reports/detector_segmenter_memory_comparison.json` | `memory.*.peak_memory_{allocated,reserved}_gib, delta.*_ratio` | `CONTROLLED_LOCAL_HARDWARE_BENCHMARK_INFERENCE_MEMORY_BATCH_1_IMGSZ_768_FP32` |
| `USE_CASE_CONDITIONAL_SELECTION` | `reports/detector_segmenter_box_comparison.json, reports/detector_segmenter_spatial_comparison.json, reports/detector_segmenter_latency_comparison.json, reports/detector_segmenter_memory_comparison.json` | `supported_class_sensitivity.delta, spatial_features.MASK_TO_BOX_FILL_RATIO, association.geometry_isolating.frozen_category_counts, deltas.END_TO_END_MODEL_OUTPUT_LATENCY_MS, delta.peak_memory_reserved_ratio` | `SYNTHESIS_OVER_THE_FOUR_AXES_NO_AGGREGATE_SCORE` |

The full claim text and the limitation attached to each claim are in `reports/detector_segmenter_scientific_synthesis.json` under `claim_register`.

## 22. Assignment coverage

`COMMITTED_EVIDENCE` - where the assignment's required metrics now live. Every entry is **validation only**.

| Requirement | Artifact | Field |
| --- | --- | --- |
| `detection_mAP50` | `reports/detection_D2_manifest.json` | `validation_metrics.mAP@0.50` |
| `detection_mAP50_95` | `reports/detection_D2_manifest.json` | `validation_metrics.mAP@0.50:0.95` |
| `detection_precision` | `reports/detection_D2_manifest.json` | `validation_metrics.precision` |
| `detection_recall` | `reports/detection_D2_manifest.json` | `validation_metrics.recall` |
| `segmentation_mask_mAP50` | `reports/segmentation_S1_result_manifest.json` | `native_metrics.mask.mAP@0.50` |
| `segmentation_mask_mAP50_95` | `reports/segmentation_S1_result_manifest.json` | `native_metrics.mask.mAP@0.50:0.95` |
| `segmentation_mask_precision` | `reports/segmentation_S1_result_manifest.json` | `native_metrics.mask.precision` |
| `segmentation_mask_recall` | `reports/segmentation_S1_result_manifest.json` | `native_metrics.mask.recall` |
| `segmentation_canonical_mask_AP` | `reports/segmentation_S1_canonical_evaluation.json` | `canonical, supported_macro` |
| `segmentation_direct_instance_mask_IoU` | `reports/segmentation_S1_mask_iou.json` | `global.matched_mask_iou_mean, global.gt_normalized_mask_iou` |
| `confusion_matrix_and_metric_figures` | `reports/figures/detection/D2/, reports/figures/segmentation_S1/` | `confusion_matrix.png, confusion_matrix_normalized.png, PR/P/R/F1 curves` |
| `detector_versus_segmenter_comparison` | `reports/detector_segmenter_box_comparison.json, reports/detector_segmenter_spatial_comparison.json, reports/detector_segmenter_latency_comparison.json, reports/detector_segmenter_memory_comparison.json` | `phases 10B and 10C, synthesised by phase 10D` |

**Still pending:**

- `qualitative_fp_fn_analysis_per_class` - PENDING, phase 12 - error analysis. Phase 8D produced a per-instance segmentation error analysis for S0 and inspected six instances. The assignment's per-class documented false positive and false negative gallery, for both tasks, has not been produced.
- `holdout_metrics_for_both_tasks` - PENDING, phase 11 - one-shot final test evaluation. The holdout has never been evaluated. No figure exists for it for any model, and none may be estimated from validation.
- `video_application` - PENDING, phase 13 - video inference and tracking. No video has been acquired, run or measured.

## 23. Limitations

- LIMITATION: every figure is validation-only and says nothing about the holdout, which has never been evaluated.
- LIMITATION: each frozen model was trained once and measured once under each protocol. Run-to-run variance is UNKNOWN and no margin here is a significance test.
- LIMITATION: the positive all-class localisation delta is carried by a class with one validation image; the support sensitivity that shows this is descriptive and decides nothing.
- LIMITATION: the association result holds at the frozen 0.50 containment floor on this population only, and the frozen four-category taxonomy proved non-exhaustive for the observed data.
- LIMITATION: the latency and memory figures are a CONTROLLED_LOCAL_HARDWARE_BENCHMARK on one laptop GPU at batch 1 in FP32 at conf 0.25, not a property of either architecture.
- LIMITATION: the latency distribution is wide and multimodal and its cause is UNKNOWN; the DVFS reading is an untested hypothesis because NO_SYNCHRONIZED_PER_OBSERVATION_POWER_STATE_TELEMETRY.
- LIMITATION: the committed parameter and GFLOPs figures are static complexity at the framework's 640 reference input, not the benchmark's 768.
- LIMITATION: no compliance, violation or correct-wearing accuracy is claimed or claimable - the project holds no compliance ground truth.

## 24. Remaining work

The repository roadmap is authoritative. Nothing below is marked complete, because no artifact supports any of it yet.

| Item | Phase | Status | Note |
| --- | --- | --- | --- |
| `FINAL_ONE_SHOT_HOLDOUT_EVALUATION` | 11 | NOT_STARTED | The test split has never been evaluated, inspected or materialised. It is read exactly once, after both models are frozen, which they now are. The immediate next phase is 11A, freezing the holdout evaluation protocol. |
| `ERROR_ANALYSIS` | 12 | NOT_STARTED | A per-class documented false-positive and false-negative analysis for both tasks. Phase 8D's S0 per-instance analysis is related evidence, not this deliverable. |
| `REAL_VIDEO_INFERENCE_AT_LEAST_30_SECONDS` | 13 | NOT_STARTED | Includes measured FPS on named hardware and temporal failure modes. |
| `TRACKING_BONUS` | 13 | OPTIONAL_NOT_STARTED | Bonus only, and only after the mandatory deliverables are complete. |
| `EXECUTABLE_COLAB_NOTEBOOK` | 14 | NOT_STARTED | No notebook exists yet; notebooks/ holds only a README. |
| `ACADEMIC_REPORT_AND_PDF` | 14 | NOT_STARTED | This synthesis and its claim register are inputs to it. |
| `PITCH_SCRIPT_AND_RECORDING` | 14 | NOT_STARTED | Every spoken number must match a committed artifact. |
| `REPRODUCIBILITY_AUDIT_FROM_A_CLEAN_CLONE` | 14 | NOT_STARTED | Walks the rubric contract item by item against real artifacts. |

## 25. Next phase

**Phase 11A - final holdout evaluation protocol freeze.**

`HOLDOUT_POLICY` - the `test` split remains **LOCKED** and has never been evaluated, inspected, materialised, adapted or plotted. Both models are now frozen, which is the precondition phase 11 requires, but phase 10D neither unlocked nor accessed the holdout and started no part of phase 11. Access needs `allow_test=True` **and** `CSVISION_ALLOW_TEST_SPLIT=1`, and neither was used here.

---

Phase 10D - synthesis fingerprint `7ddd369f8a0b1c2a65041660cdf9d6434cbeac5004fe52ffbd7171b2818ac150`. Models executed: 0. Latency measurements taken: 0. AP recomputed: false. Spatial analysis rerun: false. Test accessed: false.
