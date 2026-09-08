# Phase 7D - Final Controlled Detector Selection

Phase: 7D · Commit: `23ae55a4c42be0c4ecfb650ba79b1291a0d70df3` · Classification: **DETECTOR_FROZEN**

**No model was trained, validated, benchmarked or run in this phase.** Every number below was read from a committed result manifest and every comparison is arithmetic over those numbers. The holdout was not read, materialised, adapted, evaluated or inspected.

Claims are labelled `PREDECLARED_POLICY` (fixed in Phase 7A, before D1 and D2 existed), `COMPUTED_RESULT` (arithmetic over committed artifacts), `SELECTION_RULE` (deterministic), `HUMAN_REVIEW`, `FINAL_DECISION`, `LIMITATION` and `HOLDOUT_POLICY`.

## 1. Selection objective

`PREDECLARED_POLICY` Phase 7 asked one question: among the frozen controlled detection experiments, which configuration should become the project's detector? Phase 7A fixed how that question would be answered. Phases 7B and 7C produced the results. This phase applies the rule and freezes the answer - it does not extend the experiment matrix, and no further detection experiment is authorised.

`SELECTION_RULE` What is being selected is a *model identity*: architecture, training checkpoint and input resolution. Not a confidence threshold, not an IoU or NMS setting, not a deployment configuration.

## 2. Frozen comparison policy

`PREDECLARED_POLICY` Primary selection metric: `supported_macro_map50_95` - the unweighted arithmetic mean of per-class `AP@0.50:0.95` over the classes the frozen support rule admits.

`PREDECLARED_POLICY` Support rule: a class is `COMPARISON_SUPPORTED` when **both** `validation_positive_images >= 5` **and** `validation_instances >= 20`; otherwise `DESCRIPTIVE_HIGH_UNCERTAINTY`. The rule names no class. Applied to the split frozen in phase 5C.2 it admits `helmet_loose`, `helmet_on_head`, `person`, `vest_on_body` and classifies `vest_loose` as descriptive.

`PREDECLARED_POLICY` Practical-equivalence margin: **0.005** absolute AP. Strictly greater than the margin counts as an improvement; exactly the margin is practically equivalent. It is an engineering decision threshold that prevents escalating to a larger or slower model for a trivial validation difference. It is **not** a significance test and carries no confidence level.

`PREDECLARED_POLICY` The official all-class `mAP@0.50:0.95` remains mandatory and is never hidden. It does not decide the winner.

`PREDECLARED_POLICY` The four selection cases, fixed in Phase 7A:

| Case | Condition | Outcome |
| --- | --- | --- |
| A | No candidate clears D0 by more than the margin | Retain D0, the lower-complexity model |
| B | One candidate clears the reference by more than the margin **and** separates from the runner-up by more than the margin | That candidate is the validation-performance leader |
| C | A candidate clears the reference but does not separate from the runner-up | No winner; a later controlled efficiency comparison decides |
| D | Execution or protocol failure | Protocol review; never read as model inferiority |

`PREDECLARED_POLICY` Case C coverage was clarified in the frozen policy itself, not afterwards: it is defined by the leader's separation from the runner-up rather than by both candidates having cleared the reference, so it also covers the region where only one candidate clears the reference while the two sit within the margin of each other. Leaving that region undefined would have meant deciding it after seeing the numbers.

## 3. Controlled experiment matrix

`PREDECLARED_POLICY` The candidates carry no protocol of their own. Each inherits `configs/detection_baseline.yaml` wholesale and declares an override set, so *only one thing differs* is a structural property of the configuration rather than a promise in prose. This phase re-derived each contract from the committed configuration and required it to agree with the verdict the experiment recorded when it ran.

| Experiment | Role | Intentional variable | Declared override | Model | imgsz | Batch |
| --- | --- | --- | --- | --- | --- | --- |
| `D0` | `REFERENCE_BASELINE` | `NONE_REFERENCE` | - | YOLO11n | 640 | 16 |
| `D1` | `CONTROLLED_EXPERIMENT` | `MODEL_CAPACITY` | `model: YOLO11s`, `weight_identifier: yolo11s.pt` | YOLO11s | 640 | 16 |
| `D2` | `CONTROLLED_EXPERIMENT` | `INPUT_RESOLUTION` | `training.imgsz: 768` | YOLO11n | 768 | 16 |

`COMPUTED_RESULT` All three experiments share identical dataset, split, adapter and class-map fingerprints, verified field by field. Batch stayed at 16 throughout; no run was rescued by a smaller batch, auto-batch, gradient accumulation, a different image size or a different model.

`COMPUTED_RESULT` `optimizer: auto` resolved to AdamW at lr0 0.001111 for all three experiments, so the optimizer does not confound any comparison.

## 4. D0 result - baseline reference

`COMPUTED_RESULT` D0 is the reference point, not a candidate. It was run once and must not be re-run, retuned or averaged; a second run would measure noise rather than a change.

| Metric | Value |
| --- | --- |
| `supported_macro_map50_95` | **0.570141750000** |
| `mAP@0.50:0.95` (all-class, official) | 0.464429 |
| `mAP@0.50` | 0.619373 |
| precision | 0.85568 |
| recall | 0.539992 |
| best epoch | 67 / 100 |
| experiment SHA-256 | `cbd79fd2f2f70eb31ede61b813f991e973bb5d2f69c223a3826ee6aeadd0ffb3` |
| best checkpoint SHA-256 | `30288fbc3abe60ce1713fdb430feb2e1a3716dcf4b52b80d26a9b2aea1ca5c34` |

All of these are **validation** numbers and say nothing about test performance.

## 5. D1 result - capacity intervention

`COMPUTED_RESULT` D1 varied `MODEL_CAPACITY` (YOLO11s), with the pretrained checkpoint identity recorded as consequential rather than as a second variable.

| Metric | Value |
| --- | --- |
| `supported_macro_map50_95` | **0.560017000000** |
| `mAP@0.50:0.95` (all-class, official) | 0.471114 |
| `mAP@0.50` | 0.618243 |
| precision | 0.85964 |
| recall | 0.527798 |
| best epoch | 73 / 100 |
| experiment SHA-256 | `0589d4c0dabedfe0e6c72da3d248883d750347ce60b122b71aaa44e939158503` |
| best checkpoint SHA-256 | `a933196ee15bc369fa1ea682986d892f06b52ebd8086ce027ae4604856640717` |

All of these are **validation** numbers and say nothing about test performance.

`COMPUTED_RESULT` D1's two headline metrics moved in opposite directions, and the reason is arithmetic rather than a paradox: both are unweighted means over the same per-class APs and differ only in which classes they average. `vest_loose` - excluded by the support rule, standing on one validation image - gained enough to flip the sign of the five-class figure on its own. **D1's all-class improvement is not evidence that it beat D0**; quoting it that way would be the metric-shopping the Phase 7A policy exists to prevent, and suppressing it would be equally wrong.

`COMPUTED_RESULT` The substantive D1 finding is `vest_on_body`, which fell further than the other three supported classes gained. **Why is UNKNOWN**: one run cannot separate it from run-to-run variance, and diagnosing it needs image-level error analysis that belongs to a later phase.

## 6. D2 result - resolution intervention

`COMPUTED_RESULT` D2 varied `INPUT_RESOLUTION` (imgsz 768) starting from the same `yolo11n.pt` bytes D0 used, verified by digest.

| Metric | Value |
| --- | --- |
| `supported_macro_map50_95` | **0.594018000000** |
| `mAP@0.50:0.95` (all-class, official) | 0.490386 |
| `mAP@0.50` | 0.646304 |
| precision | 0.926331 |
| recall | 0.516549 |
| best epoch | 90 / 100 |
| experiment SHA-256 | `8417f64f3c01c8994291f1fb58837059a9db6a814fd5dbbb955a44dcc1979685` |
| best checkpoint SHA-256 | `0466f872a9de22898c70d8834cf1fcfb3d77f6c9fb27e81ee6248d3d19cbc206` |

All of these are **validation** numbers and say nothing about test performance.

`COMPUTED_RESULT` Unlike D1, both of D2's headline metrics move in the same direction, so nothing turns on which is read.

`LIMITATION` D2's predeclared small-object hypothesis is **not supported by the shape of the result**. Ranking the four supported classes by their frozen small-object fraction against their AP change gives a rank correlation of -0.20: the largest gain went to the *least* small-object-heavy class and the only decline was `person`. Resolution improved the selection metric beyond the margin - that is the controlled claim - but the proposed mechanism does not explain it.

## 7. Primary metric comparison

`COMPUTED_RESULT` Exact decimals, recomputed in this phase from the committed per-class metrics rather than copied from any summary:

| Experiment | `supported_macro_map50_95` (exact) | Rounded | Delta vs D0 | Margin status |
| --- | --- | --- | --- | --- |
| `D0` | 0.570141750000 | 0.570142 | - | `REFERENCE` |
| `D1` | 0.560017000000 | 0.560017 | -0.010124750000 | `BELOW_D0` |
| `D2` | 0.594018000000 | 0.594018 | 0.023876250000 | `IMPROVES_D0_BEYOND_MARGIN` |

`COMPUTED_RESULT` D2 against D1: 0.034001000000 on the selection metric.

`LIMITATION` That last figure is a **difference, not a ranking**. D1 and D2 differ from each other in two things at once, because each varies a different field relative to D0. Only each candidate's comparison with the reference is controlled.

### Overall ordering of all three experiments

`COMPUTED_RESULT` Descriptive, and reported here so that no reader infers a total ordering from the candidate-only fields the selection rule uses:

| Overall rank | Experiment | Role | `supported_macro_map50_95` (exact) |
| --- | --- | --- | --- |
| 1 | `D2` | controlled candidate | 0.594018000000 |
| 2 | `D0` | reference baseline | 0.570141750000 |
| 3 | `D1` | controlled candidate | 0.560017000000 |

`LIMITATION` **The reference outscores one of the candidates.** D0 sits above D1 on the selection metric, which is why the phrase "runner-up" is qualified everywhere in this report and in the machine-readable artifacts. This ordering did **not** enter the decision; the frozen rule compares each candidate against the reference and the candidates against each other, and never ranks the reference as a peer.

## 8. Official all-class comparison

`COMPUTED_RESULT` The official `mAP@0.50:0.95` is reported in full alongside the selection metric, as the frozen policy requires. It does not decide the winner.

| Experiment | `mAP@0.50:0.95` | `mAP@0.50` | precision | recall |
| --- | --- | --- | --- | --- |
| `D0` | 0.464429 | 0.619373 | 0.85568 | 0.539992 |
| `D1` | 0.471114 | 0.618243 | 0.85964 | 0.527798 |
| `D2` | 0.490386 | 0.646304 | 0.926331 | 0.516549 |

`LIMITATION` Precision and recall carry an operating-point caveat. Ultralytics reports one precision/recall pair at the F1-maximising point rather than at a fixed confidence, so a large move in either can partly reflect where that point landed. Read them as a hint about the precision/recall balance, never as a threshold-independent property.

### Per-class comparison

`COMPUTED_RESULT` Per-class `AP@0.50:0.95`, every class reported:

| Class | Support | `D0` | `D1` | `D2` |
| --- | --- | --- | --- | --- |
| `helmet_loose` | `COMPARISON_SUPPORTED` | 0.792559 | 0.798361 | 0.814493 |
| `helmet_on_head` | `COMPARISON_SUPPORTED` | 0.561699 | 0.5941 | 0.60234 |
| `person` | `COMPARISON_SUPPORTED` | 0.491618 | 0.496203 | 0.468534 |
| `vest_loose` | `DESCRIPTIVE_HIGH_UNCERTAINTY` | 0.041575 | 0.1155 | 0.075857 |
| `vest_on_body` | `COMPARISON_SUPPORTED` | 0.434691 | 0.351404 | 0.490705 |

## 9. Frozen selection-case application

`SELECTION_RULE` The frozen logic was applied mechanically, by the same code that implements the Phase 7A policy, over records rebuilt from the committed result manifests. The winning experiment id is that function's **output**; it is not written into the freeze script, and the freeze refuses if any other case is produced.

`SELECTION_RULE` Case B compares D1 and D2 as **controlled challengers** against D0 as the **reference baseline**. D0 is not a candidate and is never ranked as a peer of the two: it enters the rule through the separate 'clears the reference by more than the margin' test. Three distinct things follow, and this report keeps them apart because conflating them is an easy and consequential mistake:

| Concept | Value | Meaning |
| --- | --- | --- |
| **Validation performance leader** | **`D2`** | The experiment the frozen rule elects, and the frozen detector |
| **Second-highest experiment overall** | **`D0`** | Descriptive ordering by supported_macro_map50_95 across all three; played no part in the decision |
| **Non-reference candidate runner-up** | **`D1`** | The second-ranked *challenger*, the quantity the Case B separation test uses |

`LIMITATION` **`D1` being the other candidate does not make its metric the second-highest overall.** Here it is not: D0 scores 0.570141750000 against D1's 0.560017000000, so the overall ordering is D2 > D0 > D1 while the candidate ordering is D2 > D1. Both statements are true and they are about different sets.

- Candidates clearing D0 by more than 0.005: **D2**
- Validation performance leader: **D2** · non-reference candidate runner-up: **D1** · separation between those two candidates **0.034001**, more than the margin
- Derived case: **`CASE_B_VALIDATION_PERFORMANCE_LEADER`**

`SELECTION_RULE` D2 improves supported_macro_map50_95 over D0 by more than 0.005 and separates from every other candidate by more than the same margin.

## 10. Human review

`HUMAN_REVIEW` The project maintainer reviewed the complete D0/D1/D2 results, the frozen policy, each candidate's one-variable contract and the holdout compliance recorded by every experiment, and confirmed that the protocol was respected, that no disqualifying experimental violation exists, that the policy was correctly applied, and that D2 is accepted as the final detector checkpoint.

`HUMAN_REVIEW` Review was **confirmatory, not corrective**. It did not override the frozen policy, introduce a metric or tie-breaker, re-run anything, or consult the holdout. Had review disagreed with the policy's output, the correct action would have been to record the disagreement and stop - not to select a different model.

## 11. Final selected detector

`FINAL_DECISION` **D2 - YOLO11n at imgsz 768.**

| Field | Value |
| --- | --- |
| Selected experiment | `D2` |
| Selection status | `FINAL_SELECTED` |
| Selection method | `PREDECLARED_POLICY_PLUS_HUMAN_REVIEW` |
| Policy case | `CASE_B_VALIDATION_PERFORMANCE_LEADER` |
| Model | YOLO11n |
| Input resolution | 768 |
| Batch | 16 |
| Seed | 42 |
| Best epoch | 90 / 100 |
| Resolved optimizer | AdamW |
| Pretrained source | `yolo11n.pt` (SHA-256 `0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1`) |
| Experiment SHA-256 | `8417f64f3c01c8994291f1fb58837059a9db6a814fd5dbbb955a44dcc1979685` |
| Phase 7 policy SHA-256 | `ab4e44a1cd502db73a906c81095b3a0eeb9eee191b5cb5df1c2013fb40b5b330` |
| `final_detector_sha256` | `84d30d64a1e7ebfc4e4763643ef3a3b5c8a5ce3cedc74bbefdb8031d48235a6e` |

### Selection rationale

`FINAL_DECISION` `D0` is the baseline reference. `D1`, the capacity intervention, scored **below** D0 on the primary Phase 7 metric. `D2`, the resolution intervention, **improves D0 beyond the frozen margin**, by an amount materially larger than 0.005, and also exceeds D1 by more than the same margin - so no performance tie exists and no efficiency tie-break is required. D2 also improved the official five-class `mAP@0.50:0.95`, so the two headline figures agree in direction.

`LIMITATION` The rationale above rests entirely on the supported classes. `vest_loose` played no part in it and may not be cited as a reason for this decision.

## 12. Frozen checkpoint identity

`FINAL_DECISION` The final detector is one specific set of bytes, identified by digest rather than by path:

| Field | Value |
| --- | --- |
| Producing run | `artifacts/detection/D2/weights/best.pt` |
| SHA-256 | `0466f872a9de22898c70d8834cf1fcfb3d77f6c9fb27e81ee6248d3d19cbc206` |
| Size | 5502289 bytes |
| Checkpoint rule | `ULTRALYTICS_BEST_ON_VALIDATION_FITNESS` |
| Committed to git | false |
| Immutable local copy | `artifacts/frozen/detection/D2_best.pt` |
| Copy SHA-256 | `0466f872a9de22898c70d8834cf1fcfb3d77f6c9fb27e81ee6248d3d19cbc206` (verified equal) |

`SELECTION_RULE` `last.pt` is explicitly **not** the frozen detector. It is the final epoch's weights, sits in the same directory and loads without complaint, which is why the freeze accessor rejects it by name as well as by digest.

## 13. Why no efficiency tie-break was required

`SELECTION_RULE` The frozen policy requires an efficiency comparison only in Case C, where candidate performance is practically tied. D2 separates from the next-best candidate (D1) by 0.034001 on supported_macro_map50_95, more than the frozen margin of 0.005, so no tie exists among the candidates and no tie-break is required.

`LIMITATION` The FRAMEWORK_VALIDATION_SPEED values recorded in each experiment manifest are descriptive only. They were measured by the training framework under differing input resolutions and were not used in this selection.

`LIMITATION` A standardised detector-versus-segmenter latency study remains required by the project's scientific question. It is a later phase and is not authorised now.

## 14. Rare-class limitation

`LIMITATION` Classified `DESCRIPTIVE_HIGH_UNCERTAINTY` by the frozen support rule: `vest_loose`.

`LIMITATION` Every class listed here failed the frozen support rule on the validation split. Its precision, recall, AP@0.50 and AP@0.50:0.95 are reported in full for every experiment, and it remains a required class of the project. What it may not do is decide a winner: it was not tuned for, no model was preferred or rejected because it moved, it is never ranked against the supported classes, and the holdout was not consulted to resolve its uncertainty.

## 15. Small-object hypothesis limitation

`LIMITATION` Phase 7C ranked the four supported classes by their frozen small-object fraction against their AP change and found a rank correlation of -0.20: the largest gain went to the least small-object-heavy class and the only decline was person. Resolution improved the selection metric beyond the margin - that is the controlled claim - but the proposed mechanism does not explain it, and the beyond-margin gain must never be presented as confirming the hypothesis.

## 16. Single-run and statistical limitation

`LIMITATION` Each configuration was trained exactly once. deterministic: true reduces run-to-run variance but does not remove it, so every delta reported here is a difference between two single runs, not an estimate with an interval.

`LIMITATION` What the Phase 7 result supports: Under this frozen dataset, split, software stack and one-run-per-configuration controlled protocol, D2 - YOLO11n at imgsz 768 - was the best validation detector among D0, D1 and D2 on the predeclared selection metric.

`LIMITATION` What it does **not** establish:

- statistical significance: the margin is an engineering decision threshold, carries no confidence level, and nothing was repeated, so run-to-run variance on this setup is UNKNOWN
- superiority across arbitrary datasets: one dataset of 433 eligible images from one provider was used
- superiority across random seeds: every experiment ran once at seed 42
- that increased resolution helped because of small objects: the class-level diagnostic conducted in Phase 7C did not support that mechanism
- any statement about test performance: the holdout has never been evaluated

## 17. Holdout compliance

`HOLDOUT_POLICY` Status: **`PROTECTED_NOT_ACCESSED`**. Phase 7D selected a detector from committed validation results only. The holdout was not read, materialised, adapted, evaluated, counted or inspected, no artifact this phase wrote carries a holdout number, and no holdout identifier appears in any of them.

`HOLDOUT_POLICY` `CSVISION_ALLOW_TEST_SPLIT` was not set, no holdout adapter, label file, dataset entry, prediction or metric exists anywhere in the repository, and no holdout identifier appears in any artifact this phase wrote. The holdout is read once, in phase 11, after both models are frozen.

`HOLDOUT_POLICY` No threshold was tuned in this phase: Validation ran at the pinned framework defaults, identical across all three experiments. If the video demonstration later needs an operational inference threshold, that is a separate, predeclared, validation-only decision in which no holdout data may participate.

## 18. Binary distribution and reproducibility status

`LIMITATION` `binary_distribution_status: LOCAL_IGNORED_FROZEN_ARTIFACT` · `repository_contains_model_binary: false`.

`LIMITATION` A fresh clone cannot run inference immediately. The repository commits the record that identifies the frozen checkpoint - digest, size, producing experiment and complete protocol - not the binary itself. The weights must be obtained separately or reproduced by re-running the recorded protocol, and a re-run produces different bytes. How the frozen model is distributed for academic release is an open decision.

## 19. Next phase

Phase 8 - the instance-segmentation model adapter and protocol. It has not started. The RLE-to-polygon fidelity audit the segmentation adapter requires has not been done, and no claim about how lossy that conversion would be may be made until it has.

`SELECTION_RULE` No further detection experiment is authorised. Any new one needs a new, explicitly reviewed protocol frozen before it runs.
