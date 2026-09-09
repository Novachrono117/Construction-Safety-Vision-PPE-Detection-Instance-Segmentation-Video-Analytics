# S0-versus-S1 segmentation comparison policy

`POST_S0_PRE_S1_PROTOCOL_FREEZE` · reference `S0` · candidate `S1` `FROZEN_NOT_EXECUTED` · final segmenter `UNSELECTED`

Repository commit at production time: `776317f2a8375cf4fe8e8fd19d72478cb3a4ad8c`.

**Frozen after S0 ran and before S1 exists.** That is stated first because it is the protocol's main limitation: phase 7A's detection policy was written before either of its candidates existed, and this one could not be. What it still guarantees is that no S1 number influenced any rule below.

## 1. `CANONICAL_GROUND_TRUTH`

Every comparison is scored against `CANONICAL_COCO_INSTANCE_SEGMENTATION` - the canonical phase 5D COCO instance segmentation, which no training flag can reshape. Never the YOLO adapter.

## 2. `NATIVE_TARGET_METRIC`

Native mask AP is not wrong and is not withdrawn - it is a valid statement about each model against its own target. It simply cannot rank two models whose targets differ, because a difference between the two numbers would confound the model with the target it was scored on. The canonical COCO evaluator decides instead.

Status `NOT_CROSS_TARGET_COMPARABLE_FOR_S0_S1_SELECTION`. Both experiments still report native mask and box metrics in full; they are demoted, not suppressed.

## 3. `PRIMARY_SELECTION_METRIC`

`CANONICAL_SUPPORTED_MACRO_MASK_MAP50_95`, with the margin 0.005.

- **`below`** - delta < -0.005: S1_BELOW_S0.
- **`human_review_still_required`** - The rule produces a classification, not a frozen segmenter. Selecting the project's final segmenter remains a reviewed human decision, exactly as phase 7D was for detection.
- **`improves`** - delta > +0.005 on the primary canonical metric: S1_IMPROVES_S0_BEYOND_MARGIN.
- **`practically_equivalent`** - -0.005 <= delta <= +0.005: PRACTICALLY_EQUIVALENT. S0 is preferred, because the baseline already exists and no material canonical advantage was demonstrated. Preferring the incumbent on a tie is a decision rule fixed in advance, not a judgement made once the numbers are visible.

All-class canonical metrics are required reporting and never override the selection policy - the same discipline phase 7A applied when D1's all-class figure moved opposite to its selection metric.

## 4. `PREDECLARED_S1_HYPOTHESIS`

> Does preserving overlapping instance-mask supervision improve canonical instance-segmentation performance relative to S0?

| Field | Value |
| --- | --- |
| Inherits | `S0` |
| Intentional field | `overlap_mask` |
| Reference value | `True` -> candidate `False` |
| Fields inherited unchanged | 43 |
| Checkpoint policy | `ULTRALYTICS_NATIVE_SEGMENTATION_FITNESS` |
| Adapter | `SAME_APPROVED_PHASE_8A_ADAPTER_LABEL_BYTES` |

The one-variable contract is enforced by the parser and re-verified against S0's own protocol at freeze time, so it is a structural property rather than a promise.

## 5. `LIMITATION`

- **Frozen after S0 ran.** Phase 7A's detection policy predated both its candidates; this one could not, because the need for a common evaluator was discovered by phase 8D. No S1 number influenced any rule here, but the asymmetry is real and recorded.
- **The margin is an engineering threshold.** It carries no confidence level, nothing is repeated, and run-to-run variance on this setup remains UNKNOWN.
- **One dataset, one split, one seed.** 65 validation images and 304 instances; a result under this policy generalises no further than that.
- **The canonical evaluator is not the framework's.** Its absolute values are not comparable with native mask AP, and no reader should difference the two.
- **overlap_mask changes the treatment environment, not just a hyperparameter.** The native checkpoint fitness that selects each experiment's best.pt is itself computed against the flag's own target, so the two experiments select their checkpoints under different fitness definitions. That is why the comparison is external, and it is a genuine limitation rather than a detail.

## 6. `HOLDOUT_POLICY`

`PROTECTED_NOT_ACCESSED`. Phase 8E evaluated the frozen S0 checkpoint on the frozen validation split under a newly frozen canonical evaluator, and ran one non-experimental memory feasibility check. The holdout was not read, materialised, adapted, counted, predicted on or inspected; no holdout identifier, image, mask or statistic exists in any artifact this phase wrote.

---

Comparison protocol fingerprint `a74609c1c58371f91de2b1699e695fcbdf3fed6ac9a68237c190cfb1d1393f16`.
