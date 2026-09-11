# Final holdout evaluation - frozen protocol

Phase **11A** · Status **`FROZEN_NOT_EXECUTED`** · `PREDECLARED_PROTOCOL`

**This phase evaluated nothing.** It predeclares, in full, what the single holdout evaluation will measure, at which settings, by which evaluator, how failures are handled and what may never be done afterwards - and then stops. No holdout identifier, image, annotation, prediction or metric exists in any artifact it wrote, and neither authorisation gate was activated.

## 1. Objective

> Measure, exactly once, how the two frozen models generalise to data that took no part in any design, selection, tuning or diagnostic decision, and report that measurement with the same evidential discipline every earlier phase used.

## 2. Why the holdout is still protected

Every design decision this project has made - the split, the adapters, both architectures, both resolutions, the `overlap_mask` treatment, every threshold, every evaluator and every diagnostic - was made on `train` and `validation` alone. That is what gives the holdout its value: it is the only data that has never informed a choice.

The protocol below is written **now**, while no holdout number exists and none can exist, precisely so that the answer to "why this threshold, this rule, these examples?" is always "because it was frozen before anyone could see the result". A protocol written afterwards would be indistinguishable from a set of choices that happened to flatter the outcome.

## 3. Frozen model identities

`FROZEN_MODEL` - both were frozen long before this protocol was written, and neither is retrained, revalidated, re-thresholded or substituted by the final evaluation. Each is resolved **by digest** through its existing freeze accessor, never by path.

|  | Detector | Segmenter |
| --- | --- | --- |
| Experiment | **D2** | **S1** |
| Architecture | YOLO11n | YOLO11n-seg |
| Input size | 768 | 768 |
| `mask_ratio` | n/a | 4 |
| `overlap_mask` | n/a | false |
| Checkpoint SHA-256 | `0466f872a9de22898c70d8834cf1fcfb3d77f6c9fb27e81ee6248d3d19cbc206` | `29337d671459d0f742fb713613cc41d0eafcb8a21f5846ac0da65b2ffc024f20` |
| Checkpoint bytes | 5502289 | 6041685 |
| Identity fingerprint | `84d30d64a1e7ebfc4e4763643ef3a3b5c8a5ce3cedc74bbefdb8031d48235a6e` | `63ef41961ba3fedfef46658754940fb6ce075cdd93087b926d7c6620402579a7` |
| Frozen in phase | 7D | 8G |
| Accessor | `construction_safety_vision.detection_freeze` | `construction_safety_vision.segmentation_freeze` |
| Executed in this phase | no | no |

No alternative checkpoint may be substituted in phase 11B - not `last.pt`, not D0, D1 or S0, and not a retrained copy carrying the same experiment name. A missing binary is `BLOCKED_MISSING_MODEL_ARTIFACT` and is **never** a reason to retrain: a re-run produces different bytes under the same name, which is precisely the substitution the digest accessor exists to catch.

## 4. The one-shot principle

`ONE_SHOT_EVALUATION` - `ONE_SHOT_FINAL_HOLDOUT_EVALUATION`, `reads_permitted: 1`.

The holdout is the project's only remaining unbiased estimate. Every read spends some of that, and a second read spends the rest, because the second is necessarily informed by the first.

**The test split is not:**

- development data;
- model-selection data;
- architecture-selection data;
- hyperparameter-tuning data;
- threshold-tuning data;
- augmentation-tuning data;
- error-driven retraining data;
- data for choosing which qualitative examples look best;

**Forbidden from the moment phase 11B begins** - not from the moment it finishes, because seeing a partial result is still seeing a result:

- model selection of any kind;
- architecture change;
- threshold tuning;
- hyperparameter tuning;
- retraining or fine-tuning either model;
- dataset cleaning motivated by a test outcome;
- annotation correction motivated by a test outcome;
- any post-test experiment intended to improve a reported test number;
- re-running the evaluation to obtain a different number;
- reporting the better of two holdout runs;

> A number that can be revised after it is seen is not a generalisation estimate; it is a selection metric with extra steps. The prohibitions above are what make the reported figure mean what it says.

## 5. Authorization gates

`HOLDOUT_POLICY` - `DUAL_INDEPENDENT_OPT_IN`. The project's existing dual gate is preserved exactly and is not re-implemented: the check delegates to `construction_safety_vision.splits.assert_split_allowed`, so this repository has one guard rather than two that can drift apart.

| Gate | Requirement | Sufficient alone? |
| --- | --- | --- |
| Code | `allow_test=true` | **no** |
| Environment | `CSVISION_ALLOW_TEST_SPLIT=1` | **no** |

A stray environment variable cannot unlock a development script, and a stray in-code flag cannot unlock a shell session.

Two further restrictions apply:

- **Access is restricted to the declared runner** (`scripts/evaluate_final_holdout.py`). A generic development script holding both opt-ins is still refused, because the accessor checks the declared purpose and caller (`generic_development_access_permitted: false`).
- **No code may satisfy its own precondition.** The runner may read the environment gate and refuse; it may never write it (`runner_may_write_environment: false`, `automatic_environment_unlock_permitted: false`). Only a person sets it, deliberately, immediately before phase 11B.

**Phase 11A activated neither gate** (`activated_in_this_phase: false`), and `CSVISION_ALLOW_TEST_SPLIT` was verified unset at process, user and machine scope while this protocol was written.

## 6. Test population policy

`FINAL_HOLDOUT` - aggregate facts only. These counts were frozen in phase 5C.2, long before any model existed, and are read from the committed manifest's **aggregate count fields** (`actual_image_counts, actual_annotation_counts, actual_group_counts, actual_negative_image_counts`). The membership sections were never opened, so **no holdout identifier entered this phase** (`membership_section_read: false`).

| Quantity | Value |
| --- | --- |
| Images | **65** |
| Annotations | **305** |
| Indivisible groups | 65 |
| Zero-instance images retained | 2 |
| `split_assignment_sha256` | `a230869ff4cb45f53654d67357a27f2a0def6d8ba79f9fa4880f0fdda2f046cc` |
| `holdout_sha256` | `bb7ed43b20a84644d5a3917c6d0ead688132f82a30052b06ae7ad121e4851a00` |
| `class_map_sha256` | `596dab5b5756e3d4253924f84bf46d37a87d80b6ec830da781a08a93cd823753` |

Phase 11B evaluates **`ALL_FROZEN_TEST_IMAGES`**: no exclusions, no sampling, no manual removals, no stratification.

The single exception is technical and is declared here rather than improvised later: a file that cannot be decoded at all is not evaluable. The check runs **before any model executes** (`FILE_UNREADABLE_OR_UNDECODABLE_OR_ANNOTATION_DOCUMENT_MALFORMED`), a failure is classified `TEST_DATA_INTEGRITY_FAILURE`, and `silent_removal_permitted: false` - the count and the affected identifiers are recorded in the ledger and the phase stops for human review rather than the population shrinking quietly.

## 7. Detector metrics

`CANONICAL_EVALUATION` - primary:

- `CANONICAL_TEST_BOX_MAP50_95`
- `CANONICAL_TEST_BOX_MAP50`

Also reported:

- `CANONICAL_TEST_BOX_PRECISION`
- `CANONICAL_TEST_BOX_RECALL`

Per class, for **all five** frozen classes with no collapsing: `AP@0.50:0.95`, `AP@0.50`, `precision`, `recall`.

## 8. Segmenter metrics

`CANONICAL_EVALUATION` - primary:

- `CANONICAL_TEST_MASK_MAP50_95`
- `CANONICAL_TEST_MASK_MAP50`

Also reported:

- `CANONICAL_TEST_MASK_PRECISION`
- `CANONICAL_TEST_MASK_RECALL`

Per class: `AP@0.50:0.95`, `AP@0.50`, `precision`, `recall`.

**The segmenter's own predicted boxes** are additionally scored through the *same* bbox evaluator the detector uses, so the final localisation comparison is one evaluator over one ground truth:

- `S1_CANONICAL_TEST_BOX_MAP50_95`
- `S1_CANONICAL_TEST_BOX_MAP50`

`box_metrics_derived_from_masks: false` - re-deriving them from its masks would improve their geometric consistency with the mask branch and would then measure a post-processing choice this project invented, not the model.

## 9. Canonical evaluators and inference settings

One external evaluator judges both models, because they run through different framework validation paths and their native numbers are not guaranteed to be computed identically.

|  | Bounding box | Instance mask |
| --- | --- | --- |
| Implementation | `pycocotools.cocoeval.COCOeval` | `pycocotools.cocoeval.COCOeval` |
| `iouType` | `bbox` | `segm` |
| IoU sweep | 0.50:0.05:0.95 | 0.50:0.05:0.95 |
| `maxDets` | [1, 10, 100] | [1, 10, 100] |
| Ground truth | `CANONICAL_COCO_DETECTION_BOXES` | `CANONICAL_COCO_INSTANCE_SEGMENTATION` |
| Mirrors | `PHASE_10A_RECOGNITION_EVALUATOR` | `PHASE_8E_CANONICAL_EVALUATOR` |

Inference, frozen and identical in structure for both models:

| Setting | Detector | Segmenter |
| --- | --- | --- |
| `imgsz` | 768 | 768 |
| `conf` | **0.001** | **0.001** |
| NMS `iou` | 0.7 | 0.7 |
| `max_det` | 300 | 300 |
| Precision | FP32 | FP32 |
| `augment` / TTA | false / false | false / false |
| `retina_masks` | n/a | true |

**conf 0.001 is deliberately NOT an operating point.** Average precision integrates precision over recall, so it needs the low-scoring tail that an operational threshold discards; truncating it would silently cap recall for reasons that have nothing to do with the model. No sweep is authorised and no confidence may be tuned on the holdout.

`retina_masks` puts predicted masks on the **original canvas**, where the canonical ground-truth masks live. Resizing either side to meet the other would make every IoU partly a measurement of the resize. This behaviour may not be changed.

## 11. Precision and recall semantics

Two different quantities carry these names in this project, and they are never mixed or differenced.

**Canonical** (`CANONICAL_TEST_PRECISION_RECALL_AT_FROZEN_OPERATING_POINT`): computed from the canonical accumulation at IoU 0.5, at the frozen operating confidence **0.25** (`PROJECT_FROZEN_OPERATIONAL_THRESHOLD_UNCHANGED`). It is `is_threshold_independent: false` and `tuned_on_test: false`.

**Native framework**: `REPORTED_SEPARATELY_NEVER_MERGED_WITH_CANONICAL`, with the caveat `F1_MAXIMISING_OPERATING_POINT_NOT_A_FIXED_CONFIDENCE` - Ultralytics reports one precision/recall pair at the F1-maximising point rather than at a fixed confidence, so it is not comparable with the canonical pair and the two are never differenced.

## 10. Direct mask-IoU diagnostic

`PREDECLARED_PROTOCOL` - the **exact phase 8C protocol**, reused unchanged (`protocol_fingerprint` `b912039ca77b36959f74fcdbaed109dbf5e3bd707709556f95a19085c7f34d80`, `unchanged_from_phase_8c: true`, `new_rule_invented: false`). Inventing a new matching rule now would let it be chosen with the final result in view.

Role: **`SECONDARY_CANONICAL_DIAGNOSTIC`**. It is `is_primary_segmenter_metric: false` and `is_an_average_precision: false`.

Reported at minimum: `matched_mask_iou_mean`, `gt_normalized_mask_iou`, `gt_match_coverage`, `gt_iou50_coverage`, `gt_iou75_coverage`, plus the per-class breakdown.

> This diagnostic keeps its own **operational confidence of 0.25**, exactly as in phase 8C, while the canonical AP protocols keep 0.001. The two are never mixed, averaged or swapped, and a figure produced at one may never be reported under the other's name.

> matched_mask_iou_mean describes mask quality where the model produced an overlapping same-class instance; gt_normalized_mask_iou divides the same IoU sum by every canonical instance, so misses lower it. Neither is a COCO AP and neither may be quoted as "the project's IoU" without saying which.

## 12. Confusion-matrix semantics

`PREDECLARED_PROTOCOL` - no canonical confusion-matrix protocol existed before this phase (`prior_canonical_protocol_existed: false`). One is frozen here on the basis `EXISTING_VALIDATION_CONVENTION_READ_FROM_INSTALLED_SOURCE`: the semantics already in force during validation, **read from the installed `ultralytics==8.4.138` source**. No holdout data was consulted while deciding it, and freezing it changes nothing and invents nothing - these are the exact values that produced every committed validation matrix in this repository.

| Setting | Value | Where it comes from |
| --- | --- | --- |
| Implementation | `ultralytics.utils.metrics.ConfusionMatrix` | ultralytics==8.4.138 |
| Confidence | **0.25** | `DETECTION_VALIDATOR_CONFUSION_MATRIX_CONF_DEFAULT` |
| IoU threshold | **0.45** | `CONFUSION_MATRIX_PROCESS_BATCH_SIGNATURE_DEFAULT` |
| Matching | `CLASS_AGNOSTIC_IOU_THEN_CLASS_PAIR_RECORDED` | read from source |
| Assignment | `GREEDY_DESCENDING_IOU_DEDUPLICATED_ON_BOTH_AXES` | read from source |
| Shape | 6x6 | `nc + 1` |
| Orientation | `ROWS_ARE_PREDICTED_COLUMNS_ARE_GROUND_TRUTH` | read from source |

**Background handling.** The final row and column are the unmatched bucket. matrix[predicted, 5] is a prediction with no matched ground truth (false positive); matrix[5, gt] is a ground truth with no matched prediction (false negative). A matched pair with disagreeing classes increments matrix[predicted, gt] and counts as both a false positive and a false negative, which is the framework's own documented behaviour and is not modified here.

Class ordering is the canonical class-map index order: `helmet_loose`, `helmet_on_head`, `person`, `vest_loose`, `vest_on_body`. A normalised variant is reported alongside the raw counts.

A matrix is produced for **both** models, using identical detection semantics over each one's predicted boxes, so the two are directly comparable. The segmenter's confusion matrix uses the identical detection semantics over its predicted boxes, exactly as the framework does during segmentation validation. It is a detection-level matrix for both models, so the two are directly comparable; no mask-level confusion matrix is defined, because the framework defines none and inventing one here would be a new metric.

`threshold_chosen_after_seeing_test: false`.

## 13. FP / FN semantics

`PREDECLARED_PROTOCOL` - frozen separately from the confusion matrix, because the qualitative selection needs per-instance outcomes rather than a matrix cell, and because a **class-aware** rule is the right one for "which objects did the model get wrong".

| Property | Value |
| --- | --- |
| Class-aware | true |
| IoU threshold | 0.5 |
| Assignment | `ONE_TO_ONE_PER_IMAGE_PER_CLASS` |
| Algorithm | `GREEDY_DESCENDING_CONFIDENCE_THEN_HIGHEST_IOU` |
| Confidence threshold | 0.25 |

Definitions, fixed in advance:

- **true positive** - A prediction whose class equals the ground truth's class and whose IoU with an as-yet-unmatched ground truth of that class is at least 0.50.
- **false positive** - A prediction above the confidence threshold that is not a true positive: no unmatched same-class ground truth reaches the IoU threshold with it.
- **false negative** - A ground-truth instance with no true positive assigned to it.

For the segmentation qualitative work a miss and a bad mask are different failures and are not pooled. Categories are evaluated in the order listed and the first that matches wins:

- **`DETECTION_MISS`** - No prediction of any class reaches box IoU 0.50 with the ground-truth instance. The object was not found at all.
- **`CLASSIFICATION_MISMATCH`** - A prediction reaches box IoU 0.50 with the ground-truth instance but carries a different class.
- **`LOCALIZATION_FAILURE`** - A same-class prediction exists and overlaps the ground truth, but its box IoU is below 0.50.
- **`MASK_QUALITY_FAILURE`** - A same-class prediction is matched at box IoU 0.50 or above, but its mask IoU against the canonical mask is below 0.50.

An instance matching none of them is `WELL_HANDLED_INSTANCE`; the categories describe failures and do not claim to partition every instance (`categories_partition_every_ground_truth_instance: false`).

`redefinition_after_seeing_test_permitted: false`.

## 14. Qualitative selection

`QUALITATIVE_SELECTION` - `DETERMINISTIC_RANKING_FROZEN_BEFORE_ANY_TEST_ACCESS`.

This is the point of the whole phase in miniature: the examples that appear in the academic report are chosen by a rule frozen **before anyone has seen a holdout image**. Browsing first and choosing second is exactly what this prevents.

`manual_cherry_picking_permitted: false` · `random_sampling_permitted: false` · `images_inspected_to_design_this_rule: 0`.

**3 examples per category**, ranked as follows:

| Category | Applies to | Ranked by | Order |
| --- | --- | --- | --- |
| `HIGHEST_CONFIDENCE_FALSE_POSITIVE` | detector, segmenter | `prediction_confidence` | descending |
| `HIGHEST_CONFIDENCE_CLASSIFICATION_MISMATCH` | detector, segmenter | `prediction_confidence` | descending |
| `LOWEST_IOU_MATCHED_INSTANCE` | detector, segmenter | `matched_box_iou` | ascending |
| `LARGEST_MISSED_INSTANCE` | detector, segmenter | `ground_truth_area_pixels` | descending |
| `SEGMENTATION_UNDER_COVERAGE` | segmenter | `predicted_mask_area_over_canonical_mask_area` | ascending |
| `SEGMENTATION_OVER_COVERAGE` | segmenter | `predicted_mask_area_over_canonical_mask_area` | descending |

**Tie-breaking** is a total order, so the same holdout produces the same gallery on any machine:

1. the ranked quantity itself
2. canonical_annotation_id ascending
3. canonical_prediction_index ascending
4. source_image_id lexicographic ascending

Every ranked quantity is a float, so ties are possible; the chain ends in an identifier, which is unique.

`same_image_may_occupy_multiple_categories: true` · `same_instance_may_occupy_multiple_categories: false`. One canonical instance appears in at most one category. Categories are resolved in the order listed above and an instance already selected by an earlier category is skipped by every later one, so the published set never presents the same failure twice under two names. An image may legitimately appear more than once, because one image can contain several distinct failures.

A category with fewer qualifying instances than the quota publishes what it has and records the shortfall (`REPORT_AVAILABLE_COUNT_AND_RECORD_SHORTFALL`); `topping_up_from_another_category_permitted: false`.

**Human visual interpretation may occur only after the deterministic selection has been generated.** The ranking decides which instances are looked at; a person then explains what they show.

## 15. Prediction persistence

Raw predictions go to deterministic, **git-ignored** runtime locations and are persisted **before any metric is computed** (`persisted_before_metric_computation: true`), so a later report rebuild can regenerate every number without touching a model.

| Artifact | Location |
| --- | --- |
| Detector raw predictions | `artifacts/final_test/D2/predictions/` |
| Segmenter raw predictions | `artifacts/final_test/S1/predictions/` |
| Canonical bbox predictions | `artifacts/final_test/canonical/bbox_predictions.json` |
| Canonical mask predictions | `artifacts/final_test/canonical/mask_predictions.json` |
| Direct-IoU representation | `artifacts/final_test/canonical/direct_iou_instances.json` |
| One-shot ledger | `artifacts/final_test/one_shot_ledger.json` |

Committed to the repository:

- aggregated metrics;
- small deterministic manifests;
- approved qualitative figures;
- fingerprints;
- the one-shot ledger summary;

**Not** committed: bulk prediction tensors, holdout imagery, holdout identifiers (`bulk_tensors_committed: false`, `test_imagery_committed: false`).

## 16. Prediction fingerprints

`detector_test_prediction_sha256` and `segmenter_test_prediction_sha256` cover:

- model identity and checkpoint digest;
- protocol fingerprint;
- test split reference fingerprint;
- inference settings;
- the full ordered set of predictions with class, score and geometry;

> Proves that every reported final number came from one prediction set. A report whose prediction fingerprint does not match the ledger's is not a rebuild; it is a second run.

## 17. Result fingerprints

| Result | Fingerprint field |
| --- | --- |
| Detector | `detector_test_result_sha256` |
| Segmenter | `segmenter_test_result_sha256` |
| Direct IoU | `direct_iou_test_result_sha256` |
| Qualitative selection | `qualitative_selection_sha256` |

Each covers:

- model identity;
- test split reference fingerprint;
- evaluator protocol;
- inference settings;
- metric outputs;
- the corresponding prediction fingerprint;

Each **excludes**:

- timestamp;
- username;
- hostname;
- absolute filesystem path;
- machine-specific run directory;

`deterministic_recomputation_required: true`. A report rebuild re-derives metrics from the persisted prediction artifacts and must reproduce every committed figure identically. It is never a route to a different number.

## 18. The one-shot ledger

`ONE_SHOT_EVALUATION` - append-only, at `artifacts/final_test/one_shot_ledger.json`, summarised into `reports/final_test_evaluation.provenance.json`. `attempt_counter_resettable: false` and `mutable_history: false`.

States:

1. `NOT_STARTED`
2. `AUTHORIZED`
3. `MODEL_IDENTITY_VERIFIED`
4. `TEST_LOADED`
5. `DETECTOR_PREDICTION_STARTED`
6. `DETECTOR_PREDICTION_COMPLETE`
7. `SEGMENTER_PREDICTION_STARTED`
8. `SEGMENTER_PREDICTION_COMPLETE`
9. `PREDICTIONS_PERSISTED`
10. `METRICS_COMPUTED`
11. `ARTIFACTS_WRITTEN`
12. `COMPLETE`
13. `FAILED_REQUIRES_HUMAN_REVIEW`

Recorded at each step:

- test unlock time;
- the authorising environment and code opt-in state;
- model identity verification result for both checkpoints;
- protocol fingerprint verification result;
- prediction start and end for each model;
- prediction completion state;
- prediction fingerprints;
- metric generation state;
- artifact generation state;
- every failure and every recovery action;
- attempt number and its written justification;

**Rerun policy.** `automatic_rerun_permitted: false`, `human_authorization_required: true`, `attempt_must_be_numbered: true`, `second_run_may_be_presented_as_the_first: false`.

## 19. Failure and recovery policy

`FAILURE_POLICY` - named in advance, so a failure is classified rather than improvised, and so the difference between "the report failed to write" and "the model failed to run" is settled before either can happen.

| State | Meaning |
| --- | --- |
| `TEST_EVALUATION_COMPLETE` | Predictions produced, metrics computed, artifacts written, fingerprints verified. The one and only successful outcome. |
| `BLOCKED_MISSING_MODEL_ARTIFACT` | A frozen checkpoint is absent. Never a reason to retrain: a re-run produces different bytes under the same experiment name, which is exactly the substitution the digest accessor exists to catch. |
| `MODEL_IDENTITY_MISMATCH` | A checkpoint is present but its digest is not the frozen one. Stop. |
| `TEST_UNLOCK_AUTHORIZATION_MISSING` | One or both gates absent. The runner refuses and changes nothing. It may never satisfy the gate itself. |
| `TEST_DATA_INTEGRITY_FAILURE` | A holdout file or annotation document fails the predeclared technical validation, before any model runs. Reported with counts; never silently excluded. |
| `PREDICTION_EXECUTION_FAILED_BEFORE_RESULTS` | Inference failed before a complete prediction set existed. Preserve all evidence, stop, require human review. No automatic restart, and a later authorised run is a numbered attempt, never presented as the original. |
| `EVALUATION_ARTIFACT_WRITE_FAILURE_AFTER_VALID_PREDICTIONS` | A complete, fingerprinted prediction set exists and only report writing failed. DO NOT rerun inference. Rebuild the metrics and reports from the persisted predictions. |
| `PROTOCOL_VIOLATION` | The runner detected a deviation from this file. Stop without writing. |

**The two cases that matter most.** If a complete, fingerprinted prediction set exists and only report writing failed, **do not rerun inference** - rebuild the metrics and reports from the persisted predictions. If inference crashed *before* a complete prediction set existed, preserve the evidence, stop, and require human review: `automatic_restart_permitted: false`, `present_second_run_as_original_permitted: false`.

`silent_rerun_permitted: false`.

## 20. Validation-versus-test comparison

`DESCRIPTIVE_GENERALIZATION_COMPARISON` - permitted, and bounded. Only these may be put side by side:

- the already-existing validation metric;
- the corresponding test metric;
- their absolute difference;

`metrics_must_already_exist: true` and `new_metric_invented_for_the_comparison: false`.

**No significance test** (`significance_test: false`). No significance test is predeclared and none may be added afterwards. Each model was trained once and evaluated once per split, so run-to-run variance is UNKNOWN and a gap of any size is an observation rather than a result.

Prohibited:

- retuning in response to the gap;
- creating a post-test model;
- retrospectively selecting the better configuration;
- explaining the gap as though an experiment had tested the explanation;

## 21. Prohibited post-test actions

Every one of these becomes forbidden the moment phase 11B begins:

- unlocking the holdout from any script;
- evaluating the holdout more than once;
- tuning any threshold on the holdout;
- sweeping any threshold on the holdout;
- selecting a model, architecture or checkpoint after seeing a holdout number;
- retraining or fine-tuning either frozen model;
- re-benchmarking latency or memory in phase 11B;
- introducing a new spatial metric on the holdout;
- repeating the phase 10B association analysis on the holdout;
- declaring a winner between the detector and the segmenter;
- computing a composite or weighted score;
- claiming compliance, violation or correct-wearing accuracy;
- inventing or relaxing a support threshold after seeing holdout support;
- cherry-picking qualitative examples by browsing;
- committing test identifiers, imagery, predictions or bulk tensors;
- presenting a second run as the original one-shot evaluation;

**Computational cost is not re-measured.** `rerun_on_test: false` - Latency and inference memory are properties of the model, the runtime and the machine, not of which split the images came from. Phase 10C measured them under a frozen symmetric protocol and phase 11B must not re-benchmark.

**The phase 10B exploratory spatial study is not repeated.** `repeat_full_phase_10b_exploration: false`, `new_spatial_metrics_permitted: false`. The assignment requires final generalisation metrics, not a repeat of the validation-only exploratory spatial study. Phase 10B's spatial and association findings are already published as validation findings and are not re-opened on the holdout. Repeating them would add degrees of freedom without adding a required deliverable.

## 22. Reporting plan

`PREDECLARED_PROTOCOL` - schemas only. `created_in_phase_11a: false` and `placeholder_values_permitted: false`: **no artifact below exists yet, and none may be created with invented values.**

| Artifact | Content |
| --- | --- |
| `reports/final_test_detector.json` | detector canonical results and per-class table |
| `reports/final_test_segmenter.json` | segmenter canonical mask and box results |
| `reports/final_test_direct_iou.json` | the secondary direct mask-IoU diagnostic |
| `reports/final_test_evaluation.md` | the human-readable final evaluation |
| `reports/final_test_evaluation.provenance.json` | one-shot ledger summary and fingerprints |

Required sections of the final report:

1. holdout policy
2. one-shot execution status
3. final model identities
4. test population
5. detector canonical bbox results
6. detector per-class results
7. segmenter canonical mask results
8. segmenter per-class results
9. S1 canonical box results
10. D2 versus S1 localisation comparison
11. direct mask-IoU diagnostic
12. confusion matrix
13. deterministically selected qualitative FP/FN examples
14. final limitations
15. relationship to the validation conclusions
16. explicit no-test-driven-tuning statement

**Permitted figures:**

- detector confusion matrix;
- detector normalised confusion matrix;
- segmenter confusion matrix;
- per-class AP summary;
- deterministically selected qualitative FP/FN examples;

Any holdout image shown in the academic report must come from the deterministic selection (`test_imagery_source: DETERMINISTIC_QUALITATIVE_SELECTION_ONLY`). `browsing_then_choosing_permitted: false`, `exploratory_visual_mining_permitted: false`.

### The final detector-versus-segmenter comparison

`DESCRIPTIVE_ONLY`, on the phase 10D axes: `RECOGNITION_LOCALIZATION`, `SPATIAL_REPRESENTATION`, `COMPUTATIONAL_COST`.

`winner_declared: false` · `composite_score: false` · `weighted_ranking: false` · `model_selection_follows: false`. The comparison remains what it has been since phase 10A: descriptive, not a contest.

## 23. Assignment coverage

This protocol closes the remaining evaluation requirements of criterion C4 once phase 11B executes it:

| Requirement | Covered by |
| --- | --- |
| mAP@0.50 and mAP@0.50:0.95, detection | `CANONICAL_TEST_BOX_MAP50{,_95}` |
| mAP@0.50 and mAP@0.50:0.95, segmentation | `CANONICAL_TEST_MASK_MAP50{,_95}` |
| IoU | the direct instance-mask IoU diagnostic (phase 8C protocol) |
| precision and recall | canonical, at the frozen operating point |
| confusion matrix | frozen framework semantics, both models |
| qualitative FP/FN analysis | deterministic selection, six categories |
| single holdout evaluation | the one-shot ledger |

## 24. Phase 11B execution contract

The runner executes exactly these steps, in this order:

1. `VALIDATE_AUTHORIZATION`
2. `VERIFY_FROZEN_MODEL_BINARIES`
3. `VERIFY_PROTOCOL_FINGERPRINT`
4. `CREATE_ONE_SHOT_LEDGER`
5. `LOAD_TEST_SPLIT_ONCE`
6. `RUN_DETECTOR_PREDICTIONS`
7. `PERSIST_AND_FINGERPRINT_DETECTOR_PREDICTIONS`
8. `RUN_SEGMENTER_PREDICTIONS`
9. `PERSIST_AND_FINGERPRINT_SEGMENTER_PREDICTIONS`
10. `COMPUTE_CANONICAL_METRICS`
11. `COMPUTE_DIRECT_MASK_IOU`
12. `COMPUTE_FROZEN_CONFUSION_MATRIX`
13. `SELECT_QUALITATIVE_EXAMPLES_DETERMINISTICALLY`
14. `GENERATE_REPORTS`
15. `LOCK_RESULT_PROVENANCE`

Implemented at `scripts/evaluate_final_holdout.py`, which **refuses to execute** while phase 11B is unauthorised. It reads the environment gate and never writes it.

## 25. Holdout status

`HOLDOUT_POLICY` - **`PROTECTED_NOT_ACCESSED`**.

> Phase 11A predeclared the final evaluation and executed none of it. The holdout was not unlocked, materialised, adapted, enumerated, predicted on or inspected; only the aggregate counts frozen in phase 5C.2 were read, from the manifest's aggregate count fields rather than its membership sections.

| Count | Value |
| --- | --- |
| Models executed | 0 |
| Test predictions produced | 0 |
| Test metrics computed | 0 |
| Test images read | 0 |
| Test annotations read | 0 |
| Test identifiers recorded | 0 |
| Latency measurements taken | 0 |
| Thresholds tuned | 0 |
| Figures generated | 0 |

---

Phase 11A · `FROZEN_NOT_EXECUTED` · protocol fingerprint `a5a328b3a8e49b06fb8fd9e792abcf43ccdd9aac5422729814dac0dbadc1daef` · executes in phase 11B.
