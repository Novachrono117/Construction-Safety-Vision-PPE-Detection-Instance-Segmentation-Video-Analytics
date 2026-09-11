# Phase 11B execution accounting - provenance clarification

**`PHASE_11B_EXECUTION_ACCOUNTING_CLARIFICATION`. `PROVENANCE_METADATA_ONLY`.**

This is a provenance clarification, **not** a test-result correction. No metric, ranking, prediction byte, fingerprint or frozen protocol changed; the holdout was not accessed and no model was executed to produce it. Every value below is read by field from an artifact phase 11B already committed, or from committed source.

## 1. Authorisation is closed

| | Value |
| --- | --- |
| `final_test_observed` | true |
| `environment_test_gate_present` | false |
| `effective_holdout_access_authorized` | false |
| Gate variable | `CSVISION_ALLOW_TEST_SPLIT`, verified absent in the process, user and machine scopes |
| Gate written by this clarification | no |

## 2. Authoritative inference accounting

| | Value |
| --- | --- |
| Holdout evaluation attempts | 1 |
| Unique models executed | 2 |
| Total model inference passes | 3 |
| D2 inference invocations | 1 |
| S1 inference invocations | 2 |

| Pass | Model | Confidence | Purpose | Real execution |
| --- | --- | --- | --- | --- |
| `DETECTOR_AP_PASS` | D2 | 0.001 | `AP_CURVE` | yes |
| `SEGMENTER_AP_PASS` | S1 | 0.001 | `AP_CURVE` | yes |
| `SEGMENTER_OPERATIONAL_PASS_FOR_DIRECT_IOU` | S1 | 0.25 | `DIRECT_INSTANCE_MASK_IOU_DIAGNOSTIC` | yes |

The segmenter's operational pass was a **second, real `predict()` execution** of the same frozen checkpoint, not a derived view of the AP pass. `reports/final_test_evaluation.provenance.json` records `details.models_executed: 2`; that field counts **distinct model identities**, not inference passes. It is not reinterpreted here - the unambiguous siblings `unique_models_executed`, `total_model_inference_passes`, `d2_inference_invocations`, `s1_inference_invocations` are added beside it, and the historical artifact was not edited. Its companion `details.inference_passes: 3` already agrees with this clarification.

## 3. What "one-shot" means

The sanctioned label is **`ONE_SHOT_FINAL_HOLDOUT_EVALUATION_VALID`**. The reading `ONE_SHOT_MODEL_PREDICTION_EXECUTION_VALID` is **rejected**: it reads as one prediction invocation per model, and that is false - the segmenter was invoked twice, at the two confidences the frozen protocol declares.

- exactly one human-authorised final holdout evaluation attempt
- the complete frozen test population was used
- all inference passes belonged to that single attempt
- no result-driven rerun occurred
- no prediction pass was repeated after an outcome was observed
- no tuning occurred

| | Value |
| --- | --- |
| `evaluation_attempt_count` | 1 |
| `adaptive_prediction_rerun_count` | 0 |
| `post_metric_model_invocation_count` | 0 |
| `prediction_regeneration_after_immutability_barrier` | false |
| Phase 11B classification | `TEST_EVALUATION_COMPLETE` |

## 4. Frozen-protocol context, and one mismatch

The frozen phase 11A protocol declares **three** inference blocks - `detector_inference`, `segmenter_inference`, `direct_iou.inference` - and `direct_iou.inference` reuses the phase 8C operational confidence 0.25. Three inference passes were therefore consistent with it.

| | Value |
| --- | --- |
| `FROZEN_PROTOCOL_SATISFIED` | **true** |
| `LATER_EXECUTION_INSTRUCTION_SINGLE_S1_INVOCATION_SATISFIED` | **false** |
| Mismatch disclosed | yes |
| Caused by observing a holdout outcome | no |

The frozen phase 11A protocol declares three inference blocks, so three inference passes are consistent with it. A later, informal phase 11B execution instruction expected the segmenter's operational predictions to be reused from the AP pass rather than produced by a second prediction call. The implementation ran the declared block instead. Both facts are recorded; neither is hidden.

## 5. Second-S1-pass consequence audit

**`SECOND_S1_PASS_EQUIVALENCE_TO_AP_FILTER: VERIFIED`.** `reported_metric_dependency_on_second_execution: NONE_BEYOND_IDENTICAL_REPRODUCTION`.

| Quantity | Value | Evidence |
| --- | --- | --- |
| S1 AP predictions, total | 4760 | `COMMITTED_ARTIFACT_FIELD` |
| S1 AP predictions at score >= 0.25 | 261 | `OPERATOR_DECLARED_PRIOR_SESSION_AUDIT_NOT_PERSISTED` |
| S1 operational-pass predictions | 261 | `COMMITTED_ARTIFACT_FIELD` |
| Images compared | 65 | `OPERATOR_DECLARED_PRIOR_SESSION_AUDIT_NOT_PERSISTED` |
| Images in the holdout | 65 | `COMMITTED_ARTIFACT_FIELD` |
| Count divergences | 0 | `OPERATOR_DECLARED_PRIOR_SESSION_AUDIT_NOT_PERSISTED` |
| Content divergences | 0 | `OPERATOR_DECLARED_PRIOR_SESSION_AUDIT_NOT_PERSISTED` |

Fields compared: `class`, `score`, `box`, `mask_rle`.

The committed operational prediction count and the committed image count agree with the audit's aggregates. The per-instance field comparison itself was not persisted and is not re-derived here, because re-deriving it would mean opening holdout-derived prediction files after FINAL_TEST_OBSERVED. The second pass **was real**; its output was identical to the deterministic operational subset of the AP pass, which makes the second execution redundant in hindsight - not retrospectively something other than a real execution.

## 6. Nothing scientific changed

| | Value |
| --- | --- |
| D2 canonical box mAP@0.50:0.95 | 0.427031 |
| S1 canonical box mAP@0.50:0.95 | 0.433764 |
| S1 canonical mask mAP@0.50:0.95 | 0.410143 |
| Direct `matched_mask_iou_mean` | 0.834548 |
| Direct `gt_normalized_mask_iou` | 0.585551 |
| D2 object-level TP / FP / FN | 189 / 51 / 116 |
| S1 object-level TP / FP / FN | 205 / 56 / 100 |
| Detector prediction fingerprint | `bfcf35762b56a761c152ab14035b0f3c3c3cb2c6faafe65de3fee493403053b6` |
| Segmenter prediction fingerprint | `181d036c3b4e7e039b72061fc3f4e4ee7291a45b5fa7f8ead433e328314ed504` |
| Segmenter operational prediction fingerprint | `cadd4170c604174b6c988e39a2bdfa8eb9e4f97af0b8dbb57239fc5685bf2b6d` |

Precision, recall, the confusion matrices, the qualitative ranking, the model checkpoint fingerprints, the test population and the validation-versus-test deltas are equally untouched, and every protected artifact is verified byte-identical before and after this clarification.

## 7. Protocol-gap chronology

| | Value |
| --- | --- |
| `RUNTIME_FILESYSTEM_EVIDENCE` | `CONSISTENT_WITH_RESOLVED_BEFORE_HOLDOUT_ACCESS` |
| `VCS_PRE_EXECUTION_CHECKPOINT` | `ABSENT` |
| Strongest unambiguous scientific claim | `RESOLVED_BEFORE_OUTCOME_METRICS_WERE_OBSERVED` |

The runtime evidence is that the implementation module carrying both gap resolutions has a last-modified time earlier than the persisted prediction directories, the one-shot ledger and every committed result of the run. But the implementation is absent from the parent commit, so no signed pre-execution checkpoint proves the resolutions predated holdout access - only that they predated the observation of any outcome metric. A last-modified time is not a signature, and it is not presented as one.

## 8. Post-observation metadata access, disclosed

| | Value |
| --- | --- |
| `post_observation_test_content_access` | false |
| `post_observation_test_metadata_inspection` | **true** |
| `post_observation_model_execution` | false |
| `post_observation_prediction_generation` | false |

After `FINAL_TEST_OBSERVED`, the audit behind section 5 made this filesystem contact: a directory listing of the materialised test image directory; filesystem last-modified metadata. Zero filesystem contact with the test directory is **not** claimed. It played no part in model selection, tuning or any reported metric.

## 9. What this clarification did

| | Value |
| --- | --- |
| Models executed | 0 |
| Model inference passes | 0 |
| Holdout reads | 0 |
| Test images read | 0 |
| Test annotations read | 0 |
| Test identifiers enumerated | 0 |
| Predictions regenerated | 0 |
| Metrics recomputed from test data | 0 |
| Qualitative selection rerun | no |
| Holdout accessor invoked | no |
| Frozen historical protocol altered | no |

Clarification fingerprint `806396c320400e9f76e0110e0663d5c148bb3fd0f391d6231448f391d8105332`.
