# Detector versus segmenter on validation

Phase 10B · `FROZEN_PROTOCOL` `d92a157637baf52f9a7248bf24257d5a177e8496b7727ffb5afc874ab96d1c4d` · validation only · no latency measured

**Every number here is a validation number.** The holdout has never been evaluated. Neither model was trained, modified or re-thresholded, and no latency or memory benchmark was run - that is phase 10C.

Repository commit at production time: `17a2810aa84d6531701afbfd8526bb602ada71ea`.

## 1. Scientific comparison question

> What additional spatial and operational information does the frozen instance segmentation model provide beyond the frozen bounding-box detector?

`FROZEN_PROTOCOL`. The two models do not produce the same output, so this is not a contest and no winner is declared. The cost half of the question - latency and memory - is `LATENCY_PENDING` and belongs to phase 10C.

## 2. Frozen model identities

| | Detector | Segmenter |
| --- | --- | --- |
| Experiment | **D2** | **S1** |
| Model | YOLO11n | YOLO11n-seg |
| imgsz | 768 | 768 |
| Checkpoint | `0466f872a9de22898c70d8834cf1fcfb3d77f6c9fb27e81ee6248d3d19cbc206` | `29337d671459d0f742fb713613cc41d0eafcb8a21f5846ac0da65b2ffc024f20` |

## 3-4. Frozen protocol and validation population

Protocol fingerprint `d92a157637baf52f9a7248bf24257d5a177e8496b7727ffb5afc874ab96d1c4d`, frozen in phase 10A before any of this ran. Population: **65 validation images**, 304 canonical annotations, membership `54e4ae8afd711314b472649b1564286d0a6d3d8f13e32c035d31402cbfa5f152`. Both models saw the same images (`identical_images_for_both_models: True`).

## 5. Precision verification

`EFFECTIVE_PRECISION_PARITY_VERIFIED`. Both models were probed at runtime before any comparison number existed: backend FP16 flag `False`, parameter dtypes `['torch.float32']`, input tensor dtype `torch.float32`, autocast during forward `False`, quantization config present `False`. Identical for both (`identical_across_models: True`).

One prediction per model with a forward pre-hook on the network, capturing the dtype of the tensor that actually reached it and the autocast state at that moment. Configuration values were not trusted.

## 6-9. Canonical box evaluation

`CANONICAL_BOX_EVALUATION`. Both models' **own predicted boxes** (`segmenter_boxes_derived_from_masks: False`) scored against the same canonical ground truth by one external evaluator: `pycocotools.cocoeval.COCOeval`, `iouType='bbox'`, IoU 0.5:0.95, maxDets [1, 10, 100]. AP inference at conf 0.001, imgsz 768, FP32.

| Metric | D2 | S1 | delta (S1 - D2) |
| --- | --- | --- | --- |
| **CANONICAL_BOX_MAP50_95** | 0.48539 | 0.505682 | **+0.020292** |
| **CANONICAL_BOX_MAP50** | 0.641107 | 0.692955 | **+0.051848** |

`COMPUTED_RESULT`. Per class, AP@0.50:0.95:

| Class | D2 | S1 | delta | status |
| --- | --- | --- | --- | --- |
| helmet_loose | 0.767262 | 0.748577 | -0.018685 | `COMPARISON_REPORTED` |
| helmet_on_head | 0.628022 | 0.629018 | +0.000996 | `COMPARISON_REPORTED` |
| person | 0.485281 | 0.517231 | +0.031950 | `COMPARISON_REPORTED` |
| vest_loose | 0.068034 | 0.201654 | +0.133620 | `DESCRIPTIVE_HIGH_UNCERTAINTY` |
| vest_on_body | 0.47835 | 0.431929 | -0.046421 | `COMPARISON_REPORTED` |

Descriptive. The question is how much object-localisation capability the frozen segmenter retains relative to the frozen detector while also producing masks. Neither model changes as a result, and no winner is declared.

**The aggregate delta is not an across-the-board improvement.** The all-class figure is the unweighted mean of the five per-class APs (`all_class_is_unweighted_mean_of_per_class: True`), so each class contributes its own delta divided by 5: `helmet_loose` -0.003737, `helmet_on_head` +0.000199, `person` +0.006390, `vest_loose` +0.026724, `vest_on_body` -0.009284.

`LIMITATION`. **`vest_loose` alone contributes +0.026724, which is larger than the entire aggregate delta of +0.020292** - and it is the class the project already classifies `DESCRIPTIVE_HIGH_UNCERTAINTY`, with one validation source image. Excluding it, the mean delta over the other 4 classes is **-0.008040**: the segmenter would sit *below* the detector. Classes that declined: `helmet_loose`, `vest_on_body`. Quoting the aggregate improvement without this would be exactly the metric-shopping the project's phase 7A policy forbids.

`LIMITATION`. These are canonical-evaluator figures. They are NOT the native framework box metrics either model reported in its own experiment phase, and the two must never be differenced: different evaluator implementation, different ground-truth document and a different confidence. Only the D2-versus-S1 delta computed here, by one evaluator over one ground truth, is a comparison.

## 10. Operational inference population

`OPERATIONAL_ANALYSIS` at conf 0.25, imgsz 768, FP32 - a separate pass from the AP protocol, never mixed with it.

| | D2 | S1 |
| --- | --- | --- |
| Total predictions | 269 | 336 |
| helmet_loose | 55 | 66 |
| helmet_on_head | 41 | 46 |
| person | 120 | 163 |
| vest_loose | 0 | 5 |
| vest_on_body | 53 | 56 |

S1 masks reconstructed on the original canvas: **336**; predictions excluded from mask-derived analysis: **0**. A predicted mask whose shape does not match its source image is excluded from every mask-derived measurement and counted here, rather than resized into agreement - which would make each measurement partly a measurement of the resize.

## 11. Spatial information available only from masks

`SPATIAL_INFORMATION_GAIN`. Two of the seven frozen quantities have no box-only equivalent at all - they are what a box fundamentally cannot express.

| Quantity | n | mean | median | std | P25 | P75 | P90 | P95 | min | max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `MASK_TO_BOX_FILL_RATIO` | 336 | 0.649956 | 0.664433 | 0.159737 | 0.555943 | 0.775478 | 0.828113 | 0.851913 | 0.093441 | 0.964112 |
| `SHAPE_EXTENT` | 336 | 0.661156 | 0.672173 | 0.157453 | 0.570469 | 0.78614 | 0.830013 | 0.869896 | 0.101474 | 0.99179 |

A fill ratio well below 1 means the predicted box is mostly not the object. A shape extent below 1 means the instance does not fill even its own tight rectangle. Both are `MASK_MEASUREMENT` with `BOX_PROXY` `NO_BOX_ONLY_EQUIVALENT`; the distributions are reported continuously, because phase 10A declared no bins.

Per class, fill ratio:

| Quantity | n | mean | median | std | P25 | P75 | P90 | P95 | min | max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `helmet_loose` | 66 | 0.764328 | 0.787538 | 0.085904 | 0.743047 | 0.821356 | 0.83289 | 0.851264 | 0.488169 | 0.91217 |
| `helmet_on_head` | 46 | 0.675846 | 0.654846 | 0.128586 | 0.574094 | 0.775666 | 0.823424 | 0.871102 | 0.450137 | 0.961441 |
| `person` | 163 | 0.602999 | 0.607486 | 0.159423 | 0.523847 | 0.703759 | 0.783838 | 0.842896 | 0.099862 | 0.964112 |
| `vest_loose` | 5 | 0.616042 | 0.618413 | 0.110905 | 0.604749 | 0.663334 | 0.714535 | 0.731602 | 0.445045 | 0.748669 |
| `vest_on_body` | 56 | 0.633599 | 0.672329 | 0.185044 | 0.51999 | 0.7683 | 0.837909 | 0.861176 | 0.093441 | 0.956094 |

## 12. Mask measurements versus box proxies

`MASK_MEASUREMENT` against `BOX_PROXY`, on the same instances and the same pairs.

**`INSTANCE_AREA_PIXELS`** vs `BOX_AREA_PIXELS`

| Quantity | n | mean | median | std | P25 | P75 | P90 | P95 | min | max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mask measurement | 336 | 144562.997024 | 52788.0 | 300965.615254 | 12526.75 | 148527.75 | 368353.5 | 524271.0 | 294.0 | 2712388.0 |
| box proxy | 336 | 237205.836504 | 81334.30387 | 468575.173397 | 20930.378859 | 247020.034232 | 599975.85509 | 865102.1419 | 379.628829 | 3555669.822824 |

**`PERSON_PPE_MASK_INTERSECTION`** vs `BOX_INTERSECTION_AREA`

| Quantity | n | mean | median | std | P25 | P75 | P90 | P95 | min | max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mask measurement | 349 | 26828.34384 | 0.0 | 87621.724531 | 0.0 | 7919.0 | 75573.2 | 178009.2 | 0.0 | 1151858.0 |
| box proxy | 349 | 47280.877637 | 0.0 | 144095.357177 | 0.0 | 24013.505752 | 146176.517535 | 292394.988042 | 0.0 | 1905485.851149 |

**`PERSON_PPE_MASK_CONTAINMENT`** vs `BOX_INTERSECTION_OVER_PPE_BOX_AREA`

| Quantity | n | mean | median | std | P25 | P75 | P90 | P95 | min | max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mask measurement | 349 | 0.310145 | 0.0 | 0.451641 | 0.0 | 0.96897 | 0.999068 | 1.0 | 0.0 | 1.0 |
| box proxy | 349 | 0.338868 | 0.0 | 0.454443 | 0.0 | 0.990489 | 1.0 | 1.0 | 0.0 | 1.0 |

**`VISIBLE_PPE_COVERAGE_PROXY`** vs `BOX_OVERLAP_DERIVED_COVERAGE`

| Quantity | n | mean | median | std | P25 | P75 | P90 | P95 | min | max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mask measurement | 163 | 0.176234 | 0.0 | 0.26977 | 0.0 | 0.368709 | 0.573063 | 0.793024 | 0.0 | 0.998108 |
| box proxy | 163 | 0.181968 | 0.0 | 0.276635 | 0.0 | 0.326291 | 0.666186 | 0.75796 | 0.0 | 1.022104 |

**`MASK_CENTROID`** vs `BOX_CENTER` - the displacement between them, in pixels:

| Quantity | n | mean | median | std | P25 | P75 | P90 | P95 | min | max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| centroid displacement | 336 | 33.600488 | 16.048565 | 44.302987 | 5.685858 | 40.308741 | 88.469912 | 126.933417 | 0.092067 | 262.773275 |

## 13-14. Person-PPE spatial association and disagreement

`OPERATIONAL_ANALYSIS`. The frozen deterministic rule at containment floor **0.5**, applied to two box sources and reported separately.

**Geometry-isolating (S1's own boxes)** - `SEGMENTER_OWN_BOXES_GEOMETRY_ISOLATING`, 173 candidate relationships, of which **172 are classified by the frozen taxonomy** and 1 fall outside it (coverage 0.99422).

| Frozen category | count | % of classified (172) |
| --- | --- | --- |
| `BOX_AND_MASK_AGREE` | 103 | 59.884 |
| `BOX_ONLY_ASSOCIATION` | 3 | 1.744 |
| `MASK_ONLY_ASSOCIATION` | 0 | 0.0 |
| `NEITHER_ASSOCIATION` | 66 | 38.372 |
| *(outside the taxonomy)* `BOTH_RULES_ASSOCIATE_DIFFERENT_PERSON` | 1 | *excluded from the denominator* |

**Pipeline-level (D2's boxes)** - `DETECTOR_BOXES_PIPELINE_LEVEL`, 173 candidate relationships, of which **156 are classified by the frozen taxonomy** and 17 fall outside it (coverage 0.901734).

| Frozen category | count | % of classified (156) |
| --- | --- | --- |
| `BOX_AND_MASK_AGREE` | 81 | 51.923 |
| `BOX_ONLY_ASSOCIATION` | 7 | 4.487 |
| `MASK_ONLY_ASSOCIATION` | 6 | 3.846 |
| `NEITHER_ASSOCIATION` | 62 | 39.744 |
| *(outside the taxonomy)* `BOTH_RULES_ASSOCIATE_DIFFERENT_PERSON` | 17 | *excluded from the denominator* |

Counts are over all relationships; percentages use `classified_relationships` as their denominator, stated here because mixing an undeclared state into it would quietly change what the frozen percentages mean. Taxonomy coverage is protocol bookkeeping - how much of the observed data the frozen taxonomy describes - and not a spatial-performance metric.

`LIMITATION` `FROZEN_TAXONOMY_NON_EXHAUSTIVE_FOR_OBSERVED_DATA`.

Phase 10A froze four categories on the implicit assumption that a rule either associates or it does not. Two rules can both associate and pick different people, and none of the four is true of that. It is recorded as an exception to the taxonomy's coverage - counted on its own, excluded from the classified denominator - rather than as a fifth peer category, because adding a category after seeing data is what a frozen taxonomy exists to prevent. The phase 10A protocol is historical and was not modified. This is a protocol-design limitation found during execution; it invalidates no canonical metric, no continuous spatial metric, no raw association decision and no model prediction.

A different-person outcome is an ASSOCIATION_RULE_DISAGREEMENT, not an association error. The project holds no person-PPE association ground truth, so neither rule's answer can be called wrong.

`LIMITATION`. The geometry-isolating reading holds the model and its instances fixed and varies only the shape representation, so its exceptions are attributable to mask-versus-box geometry. The pipeline-level reading compares the frozen detector's outputs with the frozen segmenter's, so its exceptions reflect BOTH representational geometry AND the fact that different models produced different instances. The pipeline-level exceptions must not be attributed solely to geometry.

Phase 10B's row-level records preserve each disagreement's image, class, score and both containment values, but not which person index each rule selected. That is enough to locate and count every exception, which is all this correction needed, and it is why no model was re-run. The runner now records both selected person indices so a future execution can reproduce an exception down to the person.

Per relationship, over the frozen categories:

| Relationship | `BOX_AND_MASK_AGREE` | `BOX_ONLY_ASSOCIATION` | `MASK_ONLY_ASSOCIATION` | `NEITHER_ASSOCIATION` | classified | exceptions | total |
| --- | --- | --- | --- | --- | --- | --- | --- |
| helmet_to_person | 48 | 3 | 0 | 61 | 112 | 0 | 112 |
| vest_to_person | 55 | 0 | 0 | 5 | 60 | 1 | 61 |

## 15. Per-class context

| Class | `BOX_AND_MASK_AGREE` | `BOX_ONLY_ASSOCIATION` | `MASK_ONLY_ASSOCIATION` | `NEITHER_ASSOCIATION` | classified | exceptions | total |
| --- | --- | --- | --- | --- | --- | --- | --- |
| helmet_loose | 6 | 1 | 0 | 59 | 66 | 0 | 66 |
| helmet_on_head | 42 | 2 | 0 | 2 | 46 | 0 | 46 |
| vest_loose | 0 | 0 | 0 | 5 | 5 | 0 | 5 |
| vest_on_body | 55 | 0 | 0 | 0 | 55 | 1 | 56 |

## 16-17. Representation gain and proxy refinement

* **REPRESENTATION GAIN** - Quantities a mask makes computable that a box cannot express at all: MASK_TO_BOX_FILL_RATIO and SHAPE_EXTENT, both frozen as NO_BOX_ONLY_EQUIVALENT.
* **PROXY REFINEMENT** - Quantities a box can already approximate, where the mask changes the value: area, centroid, intersection, containment and the coverage proxy.
* **ASSOCIATION DIFFERENCE** - Cases where the mask rule and the box rule reach different conclusions about which person a PPE instance belongs to.

`COMPUTED_RESULT`. Of 349 PPE-person candidate pairs, **11** had overlapping boxes but masks sharing no pixel at all (0.031519 of pairs). Pairs where the two boxes overlap but the two masks share no pixel at all. Reported as a count and a fraction rather than binned, because phase 10A declared no threshold for 'strong' or 'minimal' overlap.

## 18-19. Operational interpretation and the coverage proxy's limitation

`OPERATIONAL_PROXY` `INTERPRETIVE_OPERATIONAL_PROXY`. Summed PPE-mask intersection with a person mask, divided by that person's own mask area. The literal reading of the frozen sentence, which fixes the inputs and the direction but not an exact formula.

`LIMITATION`. This is the one frozen quantity whose definition is qualitative (`definition_is_qualitative_in_the_frozen_protocol: True`): the protocol fixed its inputs and its direction but not an exact formula, so the implementation is a literal reading rather than a derivation. It is **not a compliance measure** (`is_not_a_compliance_measure: True`), and No safety-compliance accuracy is claimed anywhere. The project holds no compliance ground truth, so there is nothing such a claim could be measured against. Nothing in this report should rest on this quantity alone.

## 20. Recognition and spatial trade-off

`POST_HOC_DESCRIPTIVE_SUPPORT_SENSITIVITY`. Because the all-class figure is an unweighted mean, the same comparison is shown over the classes the project's **pre-existing** support rule already admits (['helmet_loose', 'helmet_on_head', 'person', 'vest_on_body']), setting aside vest_loose:

| | D2 | S1 | delta |
| --- | --- | --- | --- |
| All-class canonical box mAP@0.50:0.95 | 0.48539 | 0.505682 | **+0.020292** |
| Supported-class macro (descriptive) | 0.589729 | 0.581689 | **-0.008040** |

`LIMITATION`. This is **not** a frozen phase 10A metric (`is_a_frozen_phase_10a_metric: False`), **not** a selection rule (`is_a_selection_rule: False`) and **not** a significance test (`is_a_significance_test: False`). It changes no frozen number (`changes_any_frozen_number: False`). The all-class canonical figure is an unweighted mean over five classes, so a class with one validation image can move it more than the aggregate itself. This shows the same comparison over the classes the project's pre-existing support rule already admits. It is descriptive and selects nothing.

`COMPUTED_RESULT`. S1 retains broadly similar localisation capability to D2 while additionally producing masks. The positive all-class delta of +0.020292 is driven by vest_loose, which is already frozen as DESCRIPTIVE_HIGH_UNCERTAINTY, and should not be read as robust evidence that S1 is the superior object localiser: over the adequately supported classes the same comparison gives -0.008040. Neither reading selects a model, and neither frozen model changes.

Alongside that, the segmenter produces instance masks, which make two quantities computable that a box cannot express at all and change the value of five more. Whether the trade is worth making also depends on cost, which this phase did not measure.

`LIMITATION`. One run of each model was ever trained, so run-to-run variance is UNKNOWN for both and a small localisation delta is not evidence of an ordering.

## 21. Holdout compliance

`HOLDOUT_POLICY` `PROTECTED_NOT_ACCESSED`. Phase 10B ran controlled inference with both frozen models on the frozen validation split only. The holdout was not read, materialised, adapted, counted, predicted on or inspected; no holdout identifier, image, prediction or statistic exists in any artifact this phase wrote.

## 22. Limitations

`LIMITATION`.

* Validation only. Nothing here says anything about test performance.
* The spatial quantities are computed from **predictions**, not ground truth. They describe what the models assert, not what is true in the scene.
* There is no person-PPE association ground truth, so the association analysis reports agreement between two geometric rules and **no accuracy**.
* The frozen four-category taxonomy does not cover both rules associating to different people; that case is counted separately rather than absorbed.
* `VISIBLE_PPE_COVERAGE_PROXY` rests on a qualitative frozen definition and is labelled an interpretive proxy throughout.
* No inferential test is reported. None was predeclared, and choosing one now would be choosing it after seeing the data.
* `vest_loose` remains `DESCRIPTIVE_HIGH_UNCERTAINTY` and decides nothing.

## 23. Pending latency comparison

`LATENCY_PENDING`. No latency, throughput or memory benchmark was executed (`latency_measured: False`). The cost half of the scientific question is phase 10C, under the benchmark protocol phase 10A already froze. Any framework speed line emitted incidentally during this phase's inference is `INCIDENTAL_NOT_10C_BENCHMARK` and was not recorded or used.

## 24. Next phase

Phase 10C runs the frozen latency and memory benchmark; phase 10D synthesises benefit against cost. No aggregate score will be produced in either.

