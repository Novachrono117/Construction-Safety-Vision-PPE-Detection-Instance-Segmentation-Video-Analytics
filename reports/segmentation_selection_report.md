# Final segmenter selection

Phase 8G · status `FROZEN` · `FINAL_SELECTED` · selected **S1** · `PREDECLARED_CANONICAL_POLICY_PLUS_HUMAN_REVIEW`

**Every number here is a validation number.** The holdout has never been evaluated, and nothing below says anything about test performance. This phase trained nothing, evaluated nothing and ran no inference.

Repository commit at production time: `64f090378688f62837aa932bbfebf319402d4bfb`.

## 1. Selection objective

`FINAL_DECISION`. Choose one instance-segmentation model as the project's segmenter, from the two experiments the frozen phase 8E policy admits, using only committed evidence.

## 2. S0 baseline

`S0` - YOLO11n-seg, imgsz 768, batch 8, mask_ratio 4, `overlap_mask: true`. Canonical supported macro **0.484643**.

## 3. Phase 8D hypothesis

`LIMITATION`. Phase 8D measured that S0's `person` masks lose most of their overlap in pixels contested by a smaller class, and traced that to `overlap_mask: true`, under which the smaller instance owns shared pixels. That analysis is `POST_HOC_HYPOTHESIS_GENERATING`: it showed a target mismatch, not that training differently would help.

## 4. Phase 8E comparison policy

`PREDECLARED_POLICY`. Frozen before S1 existed, digest `6603bdbe7b2ceafdab969dc535db3890d334e632e1a999ccbc286077b54c257b`. Primary metric `CANONICAL_SUPPORTED_MACRO_MASK_MAP50_95`, practical-equivalence margin **0.005**, three cases fixed in advance. The canonical evaluator it names is `282eb0ec125c47687340325266a2ea397bc0f85e239290a26a5c1ec8d8721e64` - pycocotools `COCOeval` at `iouType='segm'` against the canonical phase 5D COCO validation masks, which no training flag can move.

## 5. S1 controlled intervention

`CONTROLLED_INTERVENTION`. `overlap_mask` true -> false. Verified here against **each experiment's own committed effective configuration**, not against a restatement of the protocol: `adapter_fingerprints`, `batch`, `checkpoint_policy`, `class_map_sha256`, `epochs`, `imgsz`, `mask_ratio`, `model`, `pretrained_weight_sha256`, `seed`, `split_assignment_sha256` are identical across the two runs.

## 6. Canonical comparison rationale

`CANONICAL_PRIMARY_METRIC`. Both models are scored against the same canonical COCO masks by the same evaluator. That is the only comparison the two experiments admit, because their native targets differ.

## 7-9. Canonical results and the primary delta

| | S0 | S1 | delta |
| --- | --- | --- | --- |
| **CANONICAL_SUPPORTED_MACRO_MASK_MAP50_95** | 0.484643 | **0.559463** | **+0.074820** |
| Canonical all-class mAP@0.50:0.95 | 0.388009 | 0.455034 | +0.067025 |
| Canonical all-class mAP@0.50 | 0.537536 | 0.634023 | |

Derived from `COMMITTED_CANONICAL_EVALUATION_ARTIFACTS_ONLY`. No prediction was recomputed: `predictions_recomputed: False`.

## 10. Practical-equivalence rule

`PREDECLARED_POLICY`. Margin 0.005 absolute AP. Classification: **`S1_IMPROVES_S0_BEYOND_MARGIN`**. The margin is an engineering threshold, **not a significance test**: nothing was repeated, so run-to-run variance on this setup remains UNKNOWN.

## 11. Direct-IoU consistency

`DIRECT_IOU_DIAGNOSTIC` · `SECONDARY_CANONICAL_DIAGNOSTIC` · `UNCHANGED_PHASE_8C_PROTOCOL`. Status: **`CROSS_METRIC_DIRECTION_CONSISTENT`**.

| Figure | S0 | S1 | delta |
| --- | --- | --- | --- |
| matched_mask_iou_mean | 0.717462 | 0.794849 | +0.077387 |
| gt_normalized_mask_iou | 0.556977 | 0.635356 | +0.078379 |
| gt_match_coverage | 0.776316 | 0.799342 | +0.023026 |
| gt_iou50_coverage | 0.588816 | 0.707237 | +0.118421 |
| gt_iou75_coverage | 0.460526 | 0.595395 | +0.134869 |

The secondary diagnostic informs the reading; it does not rank (`does_not_rank: True`) and no composite was created.

## 12. Native-metric comparability limitation

`LIMITATION` `NOT_CROSS_TARGET_COMPARABLE_FOR_S0_S1_SELECTION`. overlap_mask changes the framework's validation ground truth as well as its training target, so the two experiments' native mask AP is measured against different targets. The figures are reported in full for each model against its own target and are never differenced.

For the record, each against its own target: S0 native mask mAP@0.50:0.95 0.407942, S1 0.458206. **These two numbers must never be differenced.**

## 13. Per-class trade-offs

| Class | S0 AP@0.50:0.95 | S1 AP@0.50:0.95 | delta | support |
| --- | --- | --- | --- | --- |
| helmet_loose | 0.793946 | 0.734911 | -0.059035 | `COMPARISON_SUPPORTED` |
| helmet_on_head | 0.609345 | 0.637685 | +0.028340 | `COMPARISON_SUPPORTED` |
| person | 0.135036 | 0.452352 | +0.317316 | `COMPARISON_SUPPORTED` |
| vest_loose | 0.001474 | 0.037319 | +0.035845 | `DESCRIPTIVE_HIGH_UNCERTAINTY` |
| vest_on_body | 0.400246 | 0.412905 | +0.012659 | `COMPARISON_SUPPORTED` |

Selection does not require every class to improve. It requires the predeclared primary metric to clear the predeclared margin, which it does.

## 14. The person finding

`CONSISTENT_WITH_PHASE_8D_OVERLAP_TARGET_HYPOTHESIS`, and explicitly **not** `PROOF_OF_CAUSAL_MECHANISM`. `person` moved 0.135036 -> 0.452352 (+0.317316), the largest canonical improvement of any class.

`LIMITATION`. person showed the largest canonical improvement, which is consistent with the phase 8D overlap-target observation. Phase 8D was POST_HOC_HYPOTHESIS_GENERATING and this phase ran no experiment to test the mechanism, so the consistency is not converted into a causal claim and the hypothesis is not retroactively presented as predeclared.

## 15. The helmet_loose regression

`LIMITATION`. A supported class moved the other way: `helmet_loose` -0.059035.

Selection does not require every class to improve. The primary metric is an unweighted mean over the admitted classes and it moved beyond the margin, while these supported classes moved the other way. The trade-off is recorded here and in the selection report rather than left inside the mean.

Why it moved is **UNKNOWN**. The experiment varied one flag and measured the outcome. It tested no per-class mechanism, and one run cannot separate a per-class movement from run-to-run variance, which on this setup is UNKNOWN because nothing was repeated.

## 16. vest_loose uncertainty

`LIMITATION` `DESCRIPTIVE_HIGH_UNCERTAINTY`. `vest_loose` moved 0.001474 -> 0.037319 (+0.035845). vest_loose holds one validation source image and eight instances under the frozen split. Its movement is reported for completeness and took no part in the selection. It is not in the supported macro (`in_supported_macro: False`) and `may_decide_anything: False`.

## 17. Single-run limitation

`LIMITATION`.

* Every figure is a validation figure. Nothing here says anything about test performance, and the holdout has never been evaluated.
* One run per experiment. Run-to-run variance on this setup is UNKNOWN, and the 0.005 margin is an engineering threshold rather than a significance test.
* The comparison policy was frozen after S0 ran (POST_S0_PRE_S1_PROTOCOL_FREEZE), unlike phase 7A's, which predated its candidates. No S1 number influenced any rule.
* Both checkpoints were selected by the same native fitness rule computed against different targets, because overlap_mask reshapes the target. That is why the decision was made externally against canonical COCO masks.
* The aggregate gain is not uniform: person carries most of it and helmet_loose regressed. Why any individual class moved is UNKNOWN.
* No standardised latency or throughput benchmark has been run for either model.

## 18. Human review

`HUMAN_REVIEW` `HUMAN_REVIEWED_AND_ACCEPTED`. `confirms_policy_result: True`, `overrides_policy_result: False`. The frozen policy produced the classification mechanically from committed artifacts; human review accepted it. Had the derived classification differed from the reviewed decision, this phase would have stopped rather than record a selection the evidence does not support.

Recorded rationale:

* S1 improves S0 on the primary canonical metric by +0.074820, which clears the frozen practical-equivalence margin of 0.005.
* The improvement is roughly 15.0x the engineering margin, so the verdict does not rest on a value near the boundary. The margin remains a threshold, not a significance test.
* Canonical all-class mAP@0.50:0.95 moves in the same favourable direction (+0.067025), so the selection metric and the mandatory all-class figure do not disagree.
* The secondary canonical diagnostic moves the same way: GT-normalised direct mask IoU +0.078379, giving CROSS_METRIC_DIRECTION_CONSISTENT.
* Native framework AP was not used to arbitrate, because overlap_mask changes the native validation target and the two experiments' native numbers are therefore measured against different ground truth.
* Trade-off, recorded rather than hidden: helmet_loose regressed -0.059035. Selection does not require every class to improve, and why any individual class moved is UNKNOWN.

## 19. Final selected segmenter

| Field | Value |
| --- | --- |
| Selected experiment | **S1** |
| Model | YOLO11n-seg |
| imgsz | 768 |
| batch | 8 |
| mask_ratio | 4 |
| overlap_mask | **False** |
| Best epoch | 77 |
| Checkpoint policy | `ULTRALYTICS_NATIVE_SEGMENTATION_FITNESS` |
| `experiment_sha256` | `d14b98fbea6402691245037a296a7da4c040299535c2743d994b65a58acaa402` |
| `final_segmenter_sha256` | `63ef41961ba3fedfef46658754940fb6ce075cdd93087b926d7c6620402579a7` |

## 20. Frozen checkpoint identity

`FINAL_DECISION`. The frozen segmenter is **S1's `best.pt`**, SHA-256 `29337d671459d0f742fb713613cc41d0eafcb8a21f5846ac0da65b2ffc024f20`, 6041685 bytes, selected by `ULTRALYTICS_NATIVE_SEGMENTATION_FITNESS`. A byte-identical immutable copy lives at `artifacts/frozen/segmentation/S1_best.pt`, outside the run directory a re-run would overwrite.

Two checkpoints are rejected **by identity, not by convention**: `S0`'s `best.pt` (`d7b512b95fafdc8658fd802459d2c87d9ad3f11f922162a774dacf8ca75b87a3`) and any `last.pt`. S0 was trained with overlap_mask=True and predicts a different target. Its checkpoint is the same architecture and the same number of bytes as the selected one, so it is rejected by digest rather than by inspection.

## 21. Binary distribution

`LOCAL_IGNORED_FROZEN_ARTIFACT`. `repository_contains_model_binary: False`. The repository commits the record, not the weights: a fresh clone must obtain the checkpoint by digest, and must not retrain to produce one.

## 22. Holdout compliance

`HOLDOUT_POLICY` `PROTECTED_NOT_ACCESSED`. Phase 8G selected and froze the final segmenter from artifacts that already existed. It trained nothing, evaluated nothing and ran no inference. The holdout was not read, materialised, adapted, counted, predicted on or inspected; no holdout identifier, image, label, prediction or statistic exists in any artifact this phase wrote.

## 23. Next phase

`PHASE_9_DETECTOR_VERSUS_SEGMENTER_OPERATIONAL_COMPARISON`. A standardised detector-versus-segmenter comparison is still owed by the project's scientific question, and no latency, throughput or memory benchmark was run here. The final test evaluation remains a separate, single-shot phase against the locked holdout.

