# S1 - overlap-preserving instance-mask targets

Phase 8F · classification `S1_CONTROLLED_EXPERIMENT_COMPLETE` · experiment `S1` · reference `S0` · final segmenter `UNSELECTED_PENDING_REVIEW`

**Every number here is a validation number.** The holdout has never been evaluated, and nothing below says anything about test performance. S1 is a controlled candidate, not the project's final segmenter.

Repository commit at production time: `426669ff041887f048d471964da1dc3ed5e5be40`.

## 1. Experimental question

> Does preserving overlapping instance-mask supervision improve canonical instance-segmentation performance relative to S0?

`CONTROLLED_CANDIDATE_NOT_THE_FINAL_SEGMENTER`.

## 2. Phase 8D motivation

`PREDECLARED_S1_HYPOTHESIS`. Phase 8D measured that 27.3% of canonical person pixels are contested by another class, while 71.0% of the pixels S0 misses on person lie in that contested region; contested fraction versus person mask IoU has Spearman -0.5466; and re-scoring the same predictions against the overlap-resolved target recovers +0.0700 GT-normalised IoU for person and under 0.002 for every other class. S1 tests whether removing the overlap resolution from training changes canonical performance. It is a hypothesis: the measurement showed a target mismatch, not that training differently helps, and even against its own target S0's person mask IoU remained far below the compact classes.

## 3. Phase 8E common evaluator

`CANONICAL_PRIMARY_METRIC`. Both experiments are scored against the canonical phase 5D COCO validation masks with `pycocotools` `COCOeval` at `iouType='segm'`, IoU 0.50:0.05:0.95, `maxDets` [1, 10, 100], conf 0.001, NMS IoU 0.70, imgsz 768, `retina_masks: true`, no TTA. Evaluator fingerprint `282eb0ec125c47687340325266a2ea397bc0f85e239290a26a5c1ec8d8721e64`, unchanged from phase 8E and verified before S1 trained.

conf 0.001 is **not** an operating point: average precision integrates over the score curve and needs the low-scoring tail. The direct-IoU diagnostic keeps its own frozen operational 0.25, and the two protocols are never mixed.

## 4. One-variable contract

`ONE_VARIABLE_INTERVENTION`. S1 carries no protocol of its own. Its framework arguments are resolved from `configs/segmentation_baseline.yaml` with the single override declared in `configs/segmentation_comparison.yaml`: **`overlap_mask` true -> false**. 43 other framework arguments are inherited unchanged, and the parser refuses any second override.

| Held constant | Value |
| --- | --- |
| model | artifacts/weights/yolo11n-seg.pt |
| imgsz | 768 |
| batch | 8 |
| epochs | 100 |
| seed | 42 |
| mask_ratio | 4 |
| patience | 50 |
| pretrained weights | `yolo11n-seg.pt` SHA-256 `55ed65c56c91713d23e8402371c6c49a6fd84f257f7dce452e8d70e41dcbe152` |
| adapter label bytes | identical to S0, `True` |
| checkpoint policy | `ULTRALYTICS_NATIVE_SEGMENTATION_FITNESS` |

## 5. Training target difference

`LIMITATION`. S1 does not merely change a reporting flag. overlap_mask=false changes the framework's instance-mask target construction: the loader returns one mask plane per instance instead of a single indexed map in which the smaller instance owns contested pixels, and SegmentationValidator._prepare_batch builds its ground truth the same way. That consequence is part of the S1 treatment, not a side effect to be corrected for. Both experiments keep ULTRALYTICS_NATIVE_SEGMENTATION_FITNESS for selecting their own best.pt, so each checkpoint is chosen against its own target - which is precisely why the final S0-versus-S1 comparison is made externally, against canonical COCO masks that neither flag can move.

## 6. Runtime provenance

Adapter fingerprints re-verified before and after training against the phase 8A approved digests: 1726 instances over 303 train and 65 validation images, `regenerated: False`, `filtered: False`. Training read a hard-linked runtime view of those bytes so the framework's `.cache` files did not land inside phase 8A's evidence.

Every phase 8A-8E artifact was digested before and after this phase and is byte-identical: `True`, 26 artifacts. S0 was not retrained, not revalidated and not recomputed; its committed numbers are quoted.

## 7. Effective training configuration

| Field | Value |
| --- | --- |
| amp | True |
| auto_augment | randaugment |
| batch | 8 |
| bgr | 0.0 |
| close_mosaic | 10 |
| copy_paste | 0.0 |
| copy_paste_mode | flip |
| cos_lr | False |
| cutmix | 0.0 |
| degrees | 0.0 |
| deterministic | True |
| device |  |
| dropout | 0.0 |
| epochs | 100 |
| erasing | 0.4 |
| fliplr | 0.5 |
| flipud | 0.0 |
| hsv_h | 0.015 |
| hsv_s | 0.7 |
| hsv_v | 0.4 |
| imgsz | 768 |
| lr0 | 0.01 |
| lrf | 0.01 |
| mask_ratio | 4 |
| max_det | 300 |
| mixup | 0.0 |
| model | artifacts/weights/yolo11n-seg.pt |
| momentum | 0.937 |
| mosaic | 1.0 |
| multi_scale | 0.0 |
| optimizer | auto |
| overlap_mask | False |
| patience | 50 |
| perspective | 0.0 |
| rect | False |
| retina_masks | False |
| scale | 0.5 |
| seed | 42 |
| shear | 0.0 |
| single_cls | False |
| translate | 0.1 |
| warmup_bias_lr | 0.1 |
| warmup_epochs | 3.0 |
| warmup_momentum | 0.8 |
| weight_decay | 0.0005 |
| workers | 8 |

`DECLARED_OPTIMIZER_POLICY` `auto` resolved to `ACTUAL_RESOLVED_OPTIMIZER` **AdamW** at lr0 0.001111, momentum 0.9. `RESOLUTION_EVIDENCE` `FRAMEWORK_LOG_LINE_DIRECT_CAPTURE`. Weight decay 0.0005, warmup 3.0 epochs, scheduler `LINEAR_LAMBDA_LR0_TO_LR0_TIMES_LRF`, final LR factor 0.01.

optimizer: auto overrides the file's lr0 and momentum. The declared values are the protocol's policy inputs, never a statement of what ran.

## 8. Checkpoint-selection limitation

`LIMITATION`. S1 keeps S0's rule, `ULTRALYTICS_NATIVE_SEGMENTATION_FITNESS` (`BOX_MAP50_95_PLUS_MASK_MAP50_95`), unchanged. Because `overlap_mask` alters the native target, each experiment's fitness is computed against its own target, so the two checkpoints are selected under different fitness definitions. That is a genuine limitation of the comparison, not a detail - and it is exactly why the decision is made externally, against canonical masks. No mask-only checkpoint selector was created: `mask_only_checkpoint_created: False`.

## 9. Training execution

`COMPUTED_RESULT`. One run, 100/100 epochs, termination `ALL_EPOCHS_COMPLETED`, best epoch **77** at native composite fitness **0.97452**, verified as the argmax of the composite recomputed from `results.csv`. Engineering aborts: 0. Resumed: False.

`best.pt` SHA-256 `29337d671459d0f742fb713613cc41d0eafcb8a21f5846ac0da65b2ffc024f20`, 6041685 bytes; `last.pt` `06a9af1d6e62be0bd8625a6e42f6a126ff79f61611872f2ebd39f1ec6cdfb765`, 6041685 bytes. Neither is committed.

## 10. Native framework results

`NATIVE_TARGET_METRIC`. Validation split, one run, `overlap_mask: false` passed explicitly.

| Family | precision | recall | mAP@0.50 | mAP@0.50:0.95 |
| --- | --- | --- | --- | --- |
| mask | 0.719937 | 0.599587 | 0.663319 | 0.458206 |
| box | 0.781121 | 0.633628 | 0.727308 | 0.518779 |

| Class | native mask AP@0.50 | native mask AP@0.50:0.95 |
| --- | --- | --- |
| helmet_loose | 0.840625 | 0.70902 |
| helmet_on_head | 0.809125 | 0.606724 |
| person | 0.721904 | 0.470086 |
| vest_loose | 0.247262 | 0.093935 |
| vest_on_body | 0.697679 | 0.411264 |

Descriptive native supported macro: **0.549273** (`DESCRIPTIVE_NATIVE_TARGET_METRIC`, over ['helmet_loose', 'helmet_on_head', 'person', 'vest_on_body']). Computed for descriptive continuity with S0's reported figure. It may not select between S0 and S1: overlap_mask changes the native target, so the two experiments' native mask AP is measured against different ground truth.

## 11. Why native AP is not the primary cross-target metric

`NOT_CROSS_TARGET_COMPARABLE_FOR_S0_S1_SELECTION`. overlap_mask changes the framework's validation ground truth as well as its training target - SegmentationValidator._prepare_batch builds its masks with `masks == index` only when the flag is set - so S0's and S1's native mask AP are measured against different targets. Reported in full, never differenced against S0's, and never used to rank.

## 12. Canonical S1 results

`CANONICAL_PRIMARY_METRIC`. One evaluation, 5485 detections scored.

* CANONICAL_ALL_CLASS_MASK_MAP50_95: **0.455034**
* CANONICAL_ALL_CLASS_MASK_MAP50: **0.634023**

## 13. Canonical S0-versus-S1 comparison

| Class | S0 AP@0.50:0.95 | S1 AP@0.50:0.95 | delta | support |
| --- | --- | --- | --- | --- |
| helmet_loose | 0.793946 | 0.734911 | -0.059035 | `COMPARISON_SUPPORTED` |
| helmet_on_head | 0.609345 | 0.637685 | 0.02834 | `COMPARISON_SUPPORTED` |
| person | 0.135036 | 0.452352 | 0.317316 | `COMPARISON_SUPPORTED` |
| vest_loose | 0.001474 | 0.037319 | 0.035845 | `DESCRIPTIVE_HIGH_UNCERTAINTY` |
| vest_on_body | 0.400246 | 0.412905 | 0.012659 | `COMPARISON_SUPPORTED` |

All-class mAP@0.50:0.95: S0 0.388009 -> S1 0.455034, delta 0.067025. This is reported, not used to rank.

**The aggregate is not a uniform effect.** `person` moved +0.317316 and carries 88.6% of the total gain across admitted classes, while `helmet_loose` moved -0.059035.

A supported class **regressed**: `helmet_loose`. It is named here rather than left inside the mean. Each admitted class contributes its own delta divided by the four admitted classes: `helmet_loose` -0.014759, `helmet_on_head` +0.007085, `person` +0.079329, `vest_on_body` +0.003165.

`LIMITATION`. Arithmetic only. The aggregate moving in one direction does not mean every class did, and the class carrying most of the movement is named so the mean is not read as a uniform effect. Why any individual class moved is UNKNOWN: the experiment varied one flag and measured the outcome, it did not test a per-class mechanism.

## 14. Primary supported macro

`CANONICAL_PRIMARY_METRIC` `CANONICAL_SUPPORTED_MACRO_MASK_MAP50_95`, the unweighted mean canonical mask AP@0.50:0.95 over the classes the frozen phase 7A support rule admits (['helmet_loose', 'helmet_on_head', 'person', 'vest_on_body']). The rule names no class; the admitted set is its output.

* S0 (committed phase 8E reference): **0.484643**
* S1: **0.559463**
* delta: **0.07482**

## 15. Margin classification

`COMPUTED_RESULT`. Frozen margin 0.005 absolute AP. Classification: **`S1_IMPROVES_S0_BEYOND_MARGIN`**.

The margin is an engineering practical-equivalence threshold, **not a significance test**: nothing is repeated, so run-to-run variance on this setup remains UNKNOWN. Under the frozen rule, `PRACTICALLY_EQUIVALENT` prefers S0, decided in advance.

## 16. Person diagnostic

`PREDECLARED_PHASE_8D_DIAGNOSTIC_FOCUS`. Canonical AP@0.50:0.95 S0 0.135036 -> S1 0.452352, delta 0.317316.

| Direct-IoU figure | S0 | S1 | delta |
| --- | --- | --- | --- |
| gt_match_coverage | 0.751825 | 0.839416 | 0.087591 |
| gt_normalized_mask_iou | 0.434635 | 0.621189 | 0.186554 |
| matched_mask_iou_mean | 0.578106 | 0.740025 | 0.161919 |

`LIMITATION`. The phase 8D overlap-target analysis that motivated this focus is POST_HOC_HYPOTHESIS_GENERATING. A movement here is consistent with that mechanism; it does not confirm it, and one run cannot separate it from run-to-run variance. It may not select the final model.

## 17. Per-class canonical comparison

| Class | S0 AP@0.50 | S1 AP@0.50 | delta |
| --- | --- | --- | --- |
| helmet_loose | 0.891599 | 0.852829 | -0.03877 |
| helmet_on_head | 0.793362 | 0.804906 | 0.011544 |
| person | 0.309623 | 0.716194 | 0.406571 |
| vest_loose | 0.007173 | 0.12602 | 0.118847 |
| vest_on_body | 0.685925 | 0.670165 | -0.01576 |

## 18. Direct mask-IoU

`DIRECT_IOU_DIAGNOSTIC`. One run under the unchanged phase 8C protocol (`b912039ca77b36959f74fcdbaed109dbf5e3bd707709556f95a19085c7f34d80`): canonical COCO ground truth, conf 0.25, NMS IoU 0.70, imgsz 768, max_det 300, per-image per-class one-to-one Hungarian matching maximising total IoU. No threshold was swept.

| Figure | S0 | S1 | delta |
| --- | --- | --- | --- |
| matched_mask_iou_mean | 0.717462 | 0.794849 | 0.077387 |
| gt_normalized_mask_iou | 0.556977 | 0.635356 | 0.078379 |
| gt_match_coverage | 0.776316 | 0.799342 | 0.023026 |
| gt_iou50_coverage | 0.588816 | 0.707237 | 0.118421 |
| gt_iou75_coverage | 0.460526 | 0.595395 | 0.134869 |

`LIMITATION`. Neither figure is a COCO AP. AP is a ranking-sensitive average over IoU thresholds; these are mask overlap at one predeclared operating point.

## 19. Cross-metric direction

`COMPUTED_RESULT`. Primary delta 0.07482, secondary (GT-normalised direct mask IoU) delta 0.078379: **`CROSS_METRIC_DIRECTION_CONSISTENT`**.

Rule, frozen before S1 ran: A disagreement requires strictly opposite signs. A delta of exactly zero has no direction and does not disagree with anything. Ranked by `CANONICAL_SUPPORTED_MACRO_MASK_MAP50_95`; no composite was created (`composite_created: False`).

## 20. Training dynamics

`COMPUTED_RESULT`. 100 epochs logged. Native composite fitness first 0.1423, best 0.97452 at epoch 77, last 0.94407; 23 epochs ran after the best without exceeding it (`improved_after_best: False`).

| Loss / LR column | first epoch | last epoch |
| --- | --- | --- |
| lr/pg0 | 0.000361 | 2.2e-05 |
| lr/pg1 | 0.000361 | 2.2e-05 |
| lr/pg2 | 0.000361 | 2.2e-05 |
| train/box_loss | 0.87441 | 0.47102 |
| train/cls_loss | 3.42249 | 0.47913 |
| train/dfl_loss | 1.16492 | 0.8897 |
| train/seg_loss | 2.23655 | 0.80083 |
| train/sem_loss | 0.0 | 0.0 |
| val/box_loss | 0.80625 | 0.75609 |
| val/cls_loss | 3.73484 | 0.89359 |
| val/dfl_loss | 1.19879 | 1.16246 |
| val/seg_loss | 2.27435 | 1.79044 |
| val/sem_loss | 0.0 | 0.0 |

Read from results.csv only. No validation image was opened: image-level error analysis is a later, deliberate phase and starting it here would pre-empt it.

## 21. Resource facts

| Fact | Value |
| --- | --- |
| Training wall clock (s) | 1195.052 |
| results.csv cumulative time (s) | 1160.93 |
| GPU | NVIDIA GeForce RTX 5070 Laptop GPU (sm_120) |
| Epochs configured / completed | 100 / 100 |
| Best epoch | 77 |
| batch / imgsz | 8 / 768 |
| Peak GPU memory reserved (GiB) | 4.102 |
| Parameters / GFLOPs | 2843583 / 9.8 |
| Fused parameters / GFLOPs | 2835543 / 9.6 |

`FRAMEWORK_VALIDATION_SPEED` {'preprocess': 2.471532, 'inference': 10.508565, 'loss': 0.001155, 'postprocess': 1.600334} ms per image. Measured by the framework during validation, not a production latency benchmark: a standardised detector-versus-segmenter latency study is `NOT_RUN_IN_THIS_PHASE`.

## 22. vest_loose limitation

`LIMITATION` `DESCRIPTIVE_HIGH_UNCERTAINTY`. vest_loose occupies 1 validation source image carrying 8 instances under the frozen split, so it is classified DESCRIPTIVE_HIGH_UNCERTAINTY by the frozen support rule and its validation AP is not a usable selection signal. Therefore: do not tune hyperparameters against vest_loose; do not prefer or reject a model because vest_loose moved; do not rank it against the supported classes. It remains a required class and is reported in full - mask and box, AP@0.50 and AP@0.50:0.95, precision and recall - always with an explicit small-sample caveat. The holdout also cannot be consulted to resolve the uncertainty.

It is reported in full in every table above and decides nothing: `may_decide_anything: False`.

## 23. Holdout compliance

`HOLDOUT_POLICY` `PROTECTED_NOT_ACCESSED`. Phase 8F trained S1 on the frozen train split, validated it on the frozen validation split, and ran the canonical COCO evaluation and the direct mask-IoU diagnostic on that same validation split. The holdout was not read, materialised, adapted, counted, predicted on, inspected or plotted; no holdout identifier, label, prediction or statistic exists in any artifact this phase wrote, and the dataset descriptor the framework read carries no holdout key.

## 24. Interpretation limits

`LIMITATION`.

* Every figure is a **validation** figure. Nothing here says anything about test performance.
* One run each. Run-to-run variance on this setup is UNKNOWN, because nothing was repeated. The margin is an engineering threshold, not a significance test.
* The comparison protocol was frozen **after** S0 ran (`POST_S0_PRE_S1_PROTOCOL_FREEZE`), unlike phase 7A's, which predated its candidates. No S1 number influenced any rule, but the asymmetry is real and is recorded rather than smoothed over.
* The two experiments' checkpoints were selected by the same policy computed against **different targets**. That is part of the treatment, and it is why the decision is external.
* The phase 8D analysis that motivated S1 is `POST_HOC_HYPOTHESIS_GENERATING`. A movement consistent with it is not confirmation of the mechanism.
* Native and canonical numbers are **not** comparable and must never be differenced: different ground truth, different implementation, different confidence.

## 25. Final segmenter

`PENDING_HUMAN_REVIEW`. `final_segmenter: UNSELECTED_PENDING_REVIEW`. The frozen rule produces a classification, not a frozen segmenter. Selecting the project's final segmenter remains a reviewed human decision, exactly as phase 7D was for detection. No further segmentation experiment is authorised: no S1 re-run, no `mask_ratio` variant, no YOLO11s-seg, no resolution or batch change, no threshold tuning and no change to the canonical evaluator.

