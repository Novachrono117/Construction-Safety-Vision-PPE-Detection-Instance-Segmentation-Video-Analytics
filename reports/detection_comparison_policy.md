# Phase 7 - Controlled Detection Comparison Policy

Phase: 7A · Commit: `29b52b412e08cb3257bddb9c8141f54927680ead` · Classification: **DETECTION_EXPERIMENT_PROTOCOL_FROZEN**

**D0 exists. D1 and D2 do not.** That ordering is the only thing that makes this document a protocol rather than a description, so it is stated first. Every threshold, metric and decision rule below was fixed before any comparative result existed.

Claims are labelled `PREDECLARED_PHASE7_POLICY` (fixed in this phase), `FROZEN_DATASET_FACT` (measured before modelling), `D0_REFERENCE_RESULT` (produced by the single D0 run), `SELECTION_RULE` (deterministic) and `LIMITATION`.

## 1. Why a support filter is necessary

`FROZEN_DATASET_FACT` Under the split frozen in phase 5C.2, the five classes do not carry comparable validation evidence:

| Class | Validation images | Validation instances | Classification |
| --- | --- | --- | --- |
| `helmet_loose` | 11 | 57 | `COMPARISON_SUPPORTED` |
| `helmet_on_head` | 26 | 47 | `COMPARISON_SUPPORTED` |
| `person` | 55 | 137 | `COMPARISON_SUPPORTED` |
| `vest_loose` | 1 | 8 | `DESCRIPTIVE_HIGH_UNCERTAINTY` |
| `vest_on_body` | 31 | 55 | `COMPARISON_SUPPORTED` |

`D0_REFERENCE_RESULT` The all-class `mAP@0.50:0.95` the framework reports is the unweighted mean over all five classes, which is checkable against D0's own numbers: the mean of its five committed per-class `AP@0.50:0.95` values is 0.464428, against a reported 0.464429, the residual being the rounding of the per-class values. So a class standing on one validation image carries a full fifth of the deciding number, and a model can be ranked above another because of what happened in a single scene.

`PREDECLARED_PHASE7_POLICY` The support rule removes that specific failure mode from the Phase 7 ranking. It does not make the remaining per-class estimates precise, and it does not change what the official all-class figure means.

## 2. The support rule

`PREDECLARED_PHASE7_POLICY` A class is `COMPARISON_SUPPORTED` when **both** `validation_positive_source_images >= 5` **and** `validation_instances >= 20`. Otherwise it is `DESCRIPTIVE_HIGH_UNCERTAINTY`.

Two floors, because they fail independently: a class can hold many instances inside one or two images, which leaves the instance count looking healthy while the estimate still rests on a single scene, camera and lighting condition.

The rule is written as a general threshold and applied mechanically to the frozen counts. It names no class. Excluding `vest_loose` is the rule's **output**, derived from the table above, and the code that computes the selection metric never mentions the class.

- `COMPARISON_SUPPORTED`: `helmet_loose`, `helmet_on_head`, `person`, `vest_on_body`
- `DESCRIPTIVE_HIGH_UNCERTAINTY`: `vest_loose`

## 3. Metrics

| Role | Metric |
| --- | --- |
| `PRIMARY_PHASE7_SELECTION_METRIC` | `supported_macro_map50_95` |
| `OFFICIAL_ALL_CLASS_REPORTING_METRIC` | `mAP@0.50:0.95` |
| Secondary | `mAP@0.50`, `precision`, `recall` |
| Per class | `AP@0.50`, `AP@0.50:0.95`, `precision`, `recall` |
| Diagnostic | `confusion_matrix`, `training_curves`, `validation_loss_curves`, `framework_validation_speed`, `recall_diagnostic` |

`PREDECLARED_PHASE7_POLICY` `supported_macro_map50_95` is the unweighted arithmetic mean of per-class `AP@0.50:0.95` over the `COMPARISON_SUPPORTED` classes. Unweighted deliberately: a support-weighted mean would let `person`, the most frequent class, absorb the decision, which is the opposite of what a PPE-compliance evaluation cares about.

`PREDECLARED_PHASE7_POLICY` The five-class `mAP@0.50:0.95` stays **mandatory** and is reported for every experiment together with `mAP@0.50`, precision and recall. It is not the selection metric and it is never hidden. This does not rewrite the phase 6B protocol: D0's primary metric is still `mAP@0.50:0.95` and its published numbers are unchanged.

`PREDECLARED_PHASE7_POLICY` The honest sequencing argument for adding a comparison metric after D0: the support structure it filters on was frozen in phase 5C.2 before any modelling, and `vest_loose` was recorded `HIGH_SAMPLING_UNCERTAINTY` in `configs/detection_baseline.yaml` before D0 trained. The filter is therefore not a reaction to D0's per-class results. It is still a metric declared after one result exists, which is why it decides only the Phase 7 comparison and never replaces the official figure.

## 4. D0 reference values

`D0_REFERENCE_RESULT` **Validation only. These numbers say nothing about test.**

| Metric | Value |
| --- | --- |
| `supported_macro_map50_95` (primary selection) | 0.570142 |
| `mAP@0.50:0.95` (official, all class) | 0.464429 |
| `mAP@0.50` | 0.619373 |
| `precision` | 0.85568 |
| `recall` | 0.539992 |

| Class | `AP@0.50` | `AP@0.50:0.95` | precision | recall | In selection metric |
| --- | --- | --- | --- | --- | --- |
| `helmet_loose` | 0.879592 | 0.792559 | 0.898364 | 0.807018 | yes |
| `helmet_on_head` | 0.735806 | 0.561699 | 0.887663 | 0.672636 | yes |
| `person` | 0.68902 | 0.491618 | 0.792384 | 0.583942 | yes |
| `vest_loose` | 0.131486 | 0.041575 | 1.0 | 0.0 | no |
| `vest_on_body` | 0.660959 | 0.434691 | 0.699987 | 0.636364 | yes |

`D0_REFERENCE_RESULT` The supported macro exceeds the official all-class figure by 0.105713. The gap is not a better result; it is arithmetic. Both are unweighted means over per-class AP, and the difference is exactly the effect of dropping one very low AP from the average. **The two numbers are not interchangeable and must never be compared with each other as though one improved on the other.**

| Identity | Value |
| --- | --- |
| Model | YOLO11n |
| `imgsz` | 640 |
| Parameters | not recorded |
| `experiment_sha256` | `cbd79fd2f2f70eb31ede61b813f991e973bb5d2f69c223a3826ee6aeadd0ffb3` |
| `best_checkpoint` SHA-256 | `30288fbc3abe60ce1713fdb430feb2e1a3716dcf4b52b80d26a9b2aea1ca5c34` |

`PREDECLARED_PHASE7_POLICY` D0 was run once and is frozen. It is not retuned, not re-run to improve it, and not averaged over repeated runs - a second run would measure noise, not a change.

## 5. The two authorised experiments

### D1 - MODEL_CAPACITY

`PREDECLARED_PHASE7_POLICY` **Question.** Does increasing model capacity improve detection under the same input resolution and training protocol?

**Hypothesis.** HYPOTHESIS, untested. A larger backbone may fit this 303-image training set better, or may overfit it; 303 images is small enough that neither outcome is the obvious one. No prediction is recorded as expected, and the result is not interpreted as confirming a preference either way.

| Field | Value |
| --- | --- |
| Inherits | D0 |
| Intentional variable | `MODEL_CAPACITY` |
| Override `model` (intentional) | `YOLO11s` |
| Override `weight_identifier` (consequential) | `yolo11s.pt` |
| Status | `FROZEN_NOT_EXECUTED` |

`PREDECLARED_PHASE7_POLICY` Starting weights `yolo11s.pt` from `ULTRALYTICS_STANDARD_PRETRAINED_SAME_STACK_AS_D0`, currently `NOT_YET_FINGERPRINTED`. Before training, the run must record `identifier`, `sha256`, `size_bytes`, `ultralytics_version`, `source_mechanism`. Two files with the same name are not necessarily the same bytes, and the binary is never committed.

### D2 - INPUT_RESOLUTION

`PREDECLARED_PHASE7_POLICY` **Question.** Does moderately increasing input resolution improve detection, especially for small PPE objects, while model capacity remains fixed?

**Hypothesis.** HYPOTHESIS, untested. The motivation is a measurement, not an intuition: phase 4A's `reports/eda_source.json` records, over all 2031 canonical annotations and at a small-object threshold of 0.01 relative box area, helmet_on_head 52.87% small (166/314), helmet_loose 45.29% (178/393), person 27.57% (252/914), vest_on_body 25.21% (92/365) and vest_loose 4.44% (2/45); the median relative box area over all annotations is 0.0301. Raising the network input from 640 to 768 preserves roughly 1.44x the pixels per object, which MAY help the smallest instances survive the downsampling stages - or may not, since it also changes the scale distribution the pretrained backbone sees. Whether it helps is exactly what the experiment measures, and no direction is recorded as expected. Two honest caveats on the evidence: those fractions are measured over the whole canonical population rather than per split, and the 0.01 threshold is this project's own convention, not the COCO small-object definition. Nothing about the holdout motivates this, and nothing about the holdout will evaluate it.

| Field | Value |
| --- | --- |
| Inherits | D0 |
| Intentional variable | `INPUT_RESOLUTION` |
| Override `training.imgsz` (intentional) | `768` |
| Status | `FROZEN_NOT_EXECUTED` |

## 6. One-variable discipline

`PREDECLARED_PHASE7_POLICY` The candidates carry no protocol of their own. They inherit D0's protocol from `configs/detection_baseline.yaml` and declare an override set, and the parser rejects a matrix whose override set differs from its declared variable fields. So *only one thing differs* is a structural property of the configuration rather than a promise in prose.

| Comparison | Intentional difference | Verified |
| --- | --- | --- |
| D0 vs D1 | `MODEL_CAPACITY` (`model`, `weight_identifier`) | yes |
| D0 vs D2 | `INPUT_RESOLUTION` (`training.imgsz`) | yes |

Frozen identical across D0, D1 and D2: split, labels, class map, epochs, batch, optimizer policy, patience, seed, `deterministic`, augmentation policy, checkpoint-selection rule and validation protocol.

`LIMITATION` `optimizer: auto` is the declared policy, so the framework-resolved optimizer and learning rate may legitimately differ between experiments - the rule depends on class count and iteration estimate. Each run records what it actually used. None is forced to match another after execution, and a resolved value is never quoted as though the configuration had stated it.

## 7. Batch and memory policy

`PREDECLARED_PHASE7_POLICY` The controlled batch is **16** for D0, D1 and D2. If a candidate cannot execute at that batch because of a genuine CUDA out-of-memory failure, the experiment **stops** and is classified `MEMORY_CONSTRAINT_REVIEW_REQUIRED`.

Forbidden as a silent rescue:

- `reduce_batch`
- `auto_batch`
- `gradient_accumulation`
- `change_imgsz`
- `change_model`
- `reduce_workers_to_fit_by_changing_batch`

Every listed mitigation changes the effective optimisation problem, so a run rescued by one of them is no longer a one-variable comparison against D0 - it differs in capacity or resolution *and* in batch size, and no metric difference could be attributed to either. If either candidate cannot execute at batch 16 on this GPU, that experiment stops and is classified MEMORY_CONSTRAINT_REVIEW_REQUIRED. A revised protocol may be designed and reviewed afterwards; the failure is a hardware constraint on the comparison, never evidence that the model is worse.

## 8. Selection logic

`SELECTION_RULE` Ranked by `supported_macro_map50_95`, margin `0.005`.

| Case | Condition | Outcome |
| --- | --- | --- |
| `CASE_A_RETAIN_REFERENCE` | no candidate improves D0 by more than 0.005 | retain D0, the lower-complexity baseline |
| `CASE_B_VALIDATION_PERFORMANCE_LEADER` | one candidate improves D0 by more than 0.005 **and** separates from the runner-up by more than 0.005 | that candidate is the validation-performance leader |
| `CASE_C_PERFORMANCE_TIE_REQUIRES_EFFICIENCY_COMPARISON` | a candidate improves D0 by more than 0.005 but does **not** separate from the runner-up | no winner; `PERFORMANCE_TIE_REQUIRES_EFFICIENCY_COMPARISON` |
| `CASE_D_EXECUTION_OR_PROTOCOL_FAILURE` | execution or protocol failure | return for protocol review; the failure is **not** read as model inferiority |

`SELECTION_RULE` Strictly greater than 0.005 counts as an improvement; a difference of exactly 0.005 is PRACTICALLY_EQUIVALENT_ON_THIS_VALIDATION_SET. Comparisons are made on exact decimals at the manifests' six-place precision, so the boundary is reproducible rather than dependent on binary floating point.

`SELECTION_RULE` Case C is defined by the leader's separation from the runner-up rather than by both candidates having cleared the reference. That covers the tie the protocol was written for and also the region where only one candidate clears the reference while the two candidates sit within the margin of each other. The reasoning is identical in both - the candidates cannot be ordered - and leaving that region undefined would mean deciding it after seeing the numbers. Recorded explicitly because it is a predeclared extension of the case wording, not a reinterpretation made later.

`LIMITATION` The margin is an engineering decision threshold that prevents escalating to a larger model for a trivial validation difference. It is not a hypothesis test, it carries no confidence level, and no claim of statistical significance may be derived from it.

Nothing may override these cases:

- no per-class metric may override these cases
- no DESCRIPTIVE_HIGH_UNCERTAINTY class may override these cases
- recall may explain a movement but never selects the winner
- no recall-weighted or otherwise composite metric may be introduced
- the holdout may not be consulted to break a tie

## 9. The descriptive class

`PREDECLARED_PHASE7_POLICY` The rule classified `vest_loose` as `DESCRIPTIVE_HIGH_UNCERTAINTY` from the support in section 1 - 1 positive validation image(s) and 8 instances.

A DESCRIPTIVE_HIGH_UNCERTAINTY class remains a required project class and its full per-class metrics are reported for every experiment, wherever the framework exposes them reliably. Excluding it from the deciding metric is a statement about how much validation evidence stands behind its AP, not about whether the class matters. The academic report must show the class and explain this limitation.

Forbidden:

- tuning any hyperparameter for this class
- rejecting a model because this class's metric decreased
- selecting a model because this class's metric increased
- ranking this class against the supported classes
- consulting the holdout to resolve the uncertainty

## 10. Recall

`D0_REFERENCE_RESULT` D0 reported precision 0.85568 against recall 0.539992 - it misses far more than it mislabels.

`PREDECLARED_PHASE7_POLICY` Recall is recorded as an important **diagnostic** for phase 7. It is not promoted to the selection metric and no recall-weighted composite is introduced. After D1 and D2 it may help explain why `supported_macro_map50_95` moved; it does not independently choose a winner.

## 11. No post-hoc intervention

`PREDECLARED_PHASE7_POLICY` Any experiment beyond D1 and D2 requires a new, explicitly reviewed protocol frozen before it runs. There is no D3 in Phase 7A, and none may be added by reading a Phase 7B or 7C result.

Not to be attempted automatically once a result appears:

- try YOLO11m or any other capacity
- try imgsz 896, 960 or any other resolution
- tune the optimizer or the learning rate
- change mosaic, close_mosaic or any augmentation setting
- oversample or rebalance classes
- alter loss weights or apply class weighting
- relabel or filter annotations
- add a metric, a composite or a tie-breaker after seeing results
- re-run D0, D1 or D2 to obtain a better number

## 12. D0 revalidation audit note

`LIMITATION` Classified `NON_SELECTION_REVALIDATION` (equivalently `PROTOCOL_DEVIATION_WITHOUT_SELECTION_DEGREE_OF_FREEDOM`).

The Phase 6B result-rebuild procedure re-executed validation of the same best.pt under the same frozen configuration while repairing the result provenance. The metrics were reproduced identically and the committed D0 numbers are those metrics.

**Why no selection freedom.** The checkpoint was already fixed by the predeclared ULTRALYTICS_BEST_ON_VALIDATION_FITNESS rule before the rebuild, and the rebuild varied nothing: same weights, same validation split, same image size, same confidence and IoU thresholds, same metric set. A re-execution that varies nothing and compares nothing cannot select anything.

**Reporting rule.** D0 is one experiment with one result. The rebuild must never be presented as a second run, as a replication, or as evidence about run-to-run variance, and no D0 metric may be averaged, re-rounded or restated because of it.

**Evidence basis: `MAINTAINER_DECLARED_PARTIALLY_CORROBORATED_ON_DISK`.** Two different things, kept apart. Corroborated on disk: a separate validation run directory artifacts/detection/D0_val/ exists (git-ignored), so a second validation execution demonstrably happened; six of the seven committed D0 figures under reports/figures/detection/D0/ are byte-identical to that directory's output, so the committed metric figures come from that run rather than from the training run's own validation pass; the committed D0 manifest validates against the result schema and its metrics agree with the report and the provenance record; exactly one D0 experiment fingerprint exists in the repository. **Not** independently verified here: that nothing was varied between the two executions - Ultralytics writes no args.yaml for a validation run, so the configuration that run used cannot be read back off disk. This is the maintainer's account, and the corroborated facts above do not establish it.

## 13. Holdout

`PREDECLARED_PHASE7_POLICY` Every phase 7 decision - model selection, capacity, resolution, thresholds, tie-breaking and error-driven iteration - uses the frozen validation split only. The holdout is read once, in phase 11, after both models are frozen.

During this phase: `PROTECTED_NOT_ACCESSED`, guard state `LOCKED`, `CSVISION_ALLOW_TEST_SPLIT` unset. No holdout image, label, annotation, count or adapter was read, written or produced, and no comparison artifact carries a holdout number.

## 14. Limitations

- `LIMITATION` **SMALL_VALIDATION_SPLIT.** Selection rests on 65 validation images and 304 annotations. Every metric here is an estimate from that sample, and the support rule reduces but does not remove the problem - a supported class clears a floor on interpretability, not a bar for precise estimation.
- `LIMITATION` **RUN_TO_RUN_VARIANCE_NOT_MEASURED.** deterministic: true reduces variance without eliminating it, and no experiment is repeated, so the size of the run-to-run variance on this setup is UNKNOWN. The 0.005 margin is a judgement about what difference is worth acting on, not a measurement of that variance.
- `LIMITATION` **DESCRIPTIVE_CLASS_EXCLUDED_FROM_SELECTION.** The deciding metric ignores vest_loose. A model that improved that class while leaving the others unchanged would be recorded as practically equivalent and would not be selected. That is the deliberate cost of refusing to let one validation image decide, and it is a real cost rather than a free choice.
- `LIMITATION` **TWO_EXPERIMENTS_NOT_A_SEARCH.** Two controlled experiments test two hypotheses. They do not search the hyperparameter space, and a Case A outcome means these two variations did not help by more than the margin - not that the baseline is optimal.
- `LIMITATION` **NO_INDEPENDENCE_CLAIM_BEYOND_DUPLICATE_SCREENING.** The split is group-aware and class-aware but nothing establishes that two images in different splits do not share a site, a day, a camera or a worker. Any comparison made on it inherits that limitation.

## 15. What this phase did not do

- D1 was not trained. D2 was not trained. No alternative detector was trained.
- D0 was not retrained, re-validated or modified.
- `yolo11s.pt` was not fetched or fingerprinted; that is required before D1 runs.
- No holdout data was read, materialised, adapted or measured.
- No image-level error analysis was performed.
