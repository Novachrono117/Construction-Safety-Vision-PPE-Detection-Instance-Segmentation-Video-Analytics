# S1 - overlap-mask segmentation experiment protocol

Phase 8E · status `FROZEN_NOT_EXECUTED` · final segmenter `UNSELECTED`

**S1 has not been trained.** This document freezes what it will be, before it exists. No S1 number appears anywhere in this repository.

Repository commit at production time: `776317f2a8375cf4fe8e8fd19d72478cb3a4ad8c`.

## 1. Experiment identity and hypothesis

> Does preserving overlapping instance-mask supervision improve canonical instance-segmentation performance relative to S0?

Phase 8D measured that 27.3% of canonical person pixels are contested by another class, while 71.0% of the pixels S0 misses on person lie in that contested region; contested fraction versus person mask IoU has Spearman -0.5466; and re-scoring the same predictions against the overlap-resolved target recovers +0.0700 GT-normalised IoU for person and under 0.002 for every other class. S1 tests whether removing the overlap resolution from training changes canonical performance. It is a hypothesis: the measurement showed a target mismatch, not that training differently helps, and even against its own target S0's person mask IoU remained far below the compact classes.

**This is a hypothesis, not a prediction.** Phase 8D measured a mechanism aligned with S0's dominant person mask error; it did not establish that removing overlap resolution improves anything. The experiment exists to find out.

## 2. The one intentional difference

`overlap_mask`: `True` in S0, `False` in S1. **Nothing else.** 43 other fields are inherited unchanged, verified against S0's own frozen protocol rather than against a restatement of it.

## 3. Inherited protocol, in full

| Field | Value | Source |
| --- | --- | --- |
| `amp` | True | inherited from S0 |
| `auto_augment` | randaugment | inherited from S0 |
| `batch` | 8 | inherited from S0 |
| `bgr` | 0.0 | inherited from S0 |
| `close_mosaic` | 10 | inherited from S0 |
| `copy_paste` | 0.0 | inherited from S0 |
| `copy_paste_mode` | flip | inherited from S0 |
| `cos_lr` | False | inherited from S0 |
| `cutmix` | 0.0 | inherited from S0 |
| `degrees` | 0.0 | inherited from S0 |
| `deterministic` | True | inherited from S0 |
| `dropout` | 0.0 | inherited from S0 |
| `epochs` | 100 | inherited from S0 |
| `erasing` | 0.4 | inherited from S0 |
| `fliplr` | 0.5 | inherited from S0 |
| `flipud` | 0.0 | inherited from S0 |
| `hsv_h` | 0.015 | inherited from S0 |
| `hsv_s` | 0.7 | inherited from S0 |
| `hsv_v` | 0.4 | inherited from S0 |
| `imgsz` | 768 | inherited from S0 |
| `lr0` | 0.01 | inherited from S0 |
| `lrf` | 0.01 | inherited from S0 |
| `mask_ratio` | 4 | inherited from S0 |
| `max_det` | 300 | inherited from S0 |
| `mixup` | 0.0 | inherited from S0 |
| `momentum` | 0.937 | inherited from S0 |
| `mosaic` | 1.0 | inherited from S0 |
| `multi_scale` | 0.0 | inherited from S0 |
| `optimizer` | auto | inherited from S0 |
| `patience` | 50 | inherited from S0 |
| `perspective` | 0.0 | inherited from S0 |
| `pretrained` | True | inherited from S0 |
| `rect` | False | inherited from S0 |
| `retina_masks` | False | inherited from S0 |
| `scale` | 0.5 | inherited from S0 |
| `seed` | 42 | inherited from S0 |
| `shear` | 0.0 | inherited from S0 |
| `single_cls` | False | inherited from S0 |
| `translate` | 0.1 | inherited from S0 |
| `val` | True | inherited from S0 |
| `warmup_epochs` | 3.0 | inherited from S0 |
| `weight_decay` | 0.0005 | inherited from S0 |
| `workers` | 8 | inherited from S0 |
| `overlap_mask` | False | **the intervention** |

## 4. Checkpoint semantics

S1 keeps S0's checkpoint rule, ULTRALYTICS_NATIVE_SEGMENTATION_FITNESS, unchanged. Because overlap_mask alters the native target semantics, that fitness is computed against the candidate's own target rather than against S0's - so the checkpoint rule is part of the treatment environment, not a constant held across the two runs. It is kept identical in policy precisely so that no second variable is introduced, and the S0-versus-S1 decision is then made externally, against canonical COCO masks that neither flag can move. Creating a mask-only checkpoint selector for either experiment is not authorised.

## 5. What S1 must report

`PREDECLARED_PROTOCOL`. All three families, none of them optional:

- CANONICAL_SUPPORTED_MACRO_MASK_MAP50_95
- CANONICAL_ALL_CLASS_MASK_MAP50_95
- CANONICAL_ALL_CLASS_MASK_MAP50
- canonical per-class mask AP@0.50:0.95 and AP@0.50
- the phase 8C direct mask-IoU diagnostic, executed exactly once
- native framework mask mAP@0.50:0.95, mAP@0.50, precision and recall
- native framework box metrics

## 6. Adapter identity

S1 reads exactly the same approved phase 8A adapter label bytes as S0. The intervention happens inside the framework's target construction, not in the data: no conversion change, no mask filtering, no relabelling, and no change to the canonical modelling population.

| Fingerprint | Value |
| --- | --- |
| `image_membership_sha256` | `616701e18deb1e6d873459c94e702d33752f1d712fe0f6eebb0fa8288f126d41` |
| `labels_development_sha256` | `ff21c7822921601a5c2ac6d0d324eb9f234f1a13ce16647f834936de5d5682f2` |
| `labels_train_sha256` | `63a8145d976f7a4c9d1e2868b052c0e0d4233a15f7c6e6b3a1a0e25be6b4cd17` |
| `labels_validation_sha256` | `dae69290936641d73e576682631ef73e026bf7f33cec8c2a60a7789ac054e83f` |

## 7. Memory feasibility

`NON_EXPERIMENTAL` · `DO_NOT_REPORT_AS_MODEL_RESULT` · status `SUCCESS`.

One real training batch built with the candidate's own target construction, then a forward pass, the loss, and a backward pass.

| Field | Value |
| --- | --- |
| Batch | 8 |
| imgsz | 768 |
| `overlap_mask` | False |
| Mask target shape | [60, 192, 192] |
| Peak GPU memory reserved | 3.463 GiB (3718250496 bytes) |
| Out of memory | False |
| Optimizer step taken | False |
| Validation run | False |
| Checkpoint written | False |
| Accuracy metrics recorded | False |
| Runtime | 0.745 s |

**What it establishes.** That the frozen batch of 8 at imgsz 768 completes a forward and backward pass with overlap_mask disabled on this device. Nothing about accuracy, convergence or the eventual S1 result.

With overlap resolution off the loader returns one mask plane per instance rather than a single indexed map, which is the growth this check exists to size.

## 8. Holdout

`HOLDOUT_POLICY` · `PROTECTED_NOT_ACCESSED`. Phase 8E evaluated the frozen S0 checkpoint on the frozen validation split under a newly frozen canonical evaluator, and ran one non-experimental memory feasibility check. The holdout was not read, materialised, adapted, counted, predicted on or inspected; no holdout identifier, image, mask or statistic exists in any artifact this phase wrote.

## 9. Next phase

Phase 8F: run S1 exactly once under this frozen protocol, then apply the frozen comparison policy. S1 is `FROZEN_NOT_EXECUTED` and the final segmenter is `UNSELECTED`. Do not begin without an explicit instruction.
