# S0 - segmentation baseline protocol

Phase 8B · classification `S0_PROTOCOL_FROZEN` · experiment `S0` · S0 execution `NOT_EXECUTED_PROTOCOL_ONLY`

This report freezes a protocol. It contains **no S0 performance result**, because S0 has not been trained. The only model that ran in this phase was a one-epoch `NON_EXPERIMENTAL` smoke test whose metrics are `DO_NOT_REPORT_AS_MODEL_RESULT` and were not recorded.

Repository commit at production time: `7871ff69221ccb50144919218a2842ca2715d423`.

## 1. Architecture decision

`HUMAN_ARCHITECTURE_DECISION` · status `FINAL_SELECTED_FOR_S0` · selected **YOLO11n-seg** (`yolo11n-seg.pt`).

The decision is the maintainer's, taken on the phase 8A evidence. Phase 8A itself selected nothing and could not: a high IoU distribution is not an approval and a low one is not a rejection.

| Field | Value |
| --- | --- |
| Selected architecture | YOLO11n-seg |
| Weight identifier | `yolo11n-seg.pt` |
| Selection basis | `PHASE_8A_QUANTITATIVE_FIDELITY_AUDIT_PLUS_HUMAN_REVIEW` |
| Fallback | mask-native instance segmentation consuming canonical COCO masks and RLE directly, for example Mask R-CNN |
| Fallback status | `NOT_SELECTED_FALLBACK` |

## 2. Phase 8A evidence

`AUDIT_EVIDENCE`. Every figure below is read from the committed phase 8A manifest.

| Measurement | Value |
| --- | --- |
| Development instances | 1726 |
| Instance cardinality preserved | True (0 dropped, 0 merged, 0 split) |
| Framework parser corrupt labels | 0 |
| Mask IoU mean | 0.973066 |
| Mask IoU median | 0.984576 |
| Mask IoU P05 | 0.918176 |
| Mask IoU minimum | 0.307692 |
| Instances at IoU >= 0.90 | 1679 (97.2769%) |
| Instances at IoU >= 0.95 | 1474 (85.3998%) |
| Instances below IoU 0.90 | 47 |
| Exactly-converted instances | 0 |
| Control-level IoU mean (rasteriser convention alone) | 0.986368 |
| Merged-level IoU mean (adds component joining) | 0.985525 |

Mask IoU by canonical representation, never aggregated into one figure:

| Representation | Mean mask IoU |
| --- | --- |
| `CANONICAL_POLYGON` | 0.978091 |
| `CANONICAL_RLE` | 0.968538 |
| `SYNTHETIC_RECTANGLE` | 0.849702 |

Topology: 307 instances have more than one component and 180 carry holes totalling 1072133 filled pixels. Neither is the dominant cost - component joining costs 0.000843 mean IoU while serialisation and the integer snap cost 0.012458.

## 3. Why YOLO11n-seg was accepted

`HUMAN_ARCHITECTURE_DECISION`. Four reasons, stated so a reader can disagree with them:

1. **The audited representation works and its cost is quantified.** All 1726 development instances convert to exactly 1726 rows, the framework's own parser reads them with 0 corrupt labels, and the approximation is measured rather than assumed.
2. **Family alignment.** YOLO11, the same family as the frozen detector, so a later operational comparison is between comparable things rather than between two frameworks.
3. **Resolution alignment.** imgsz 768, the frozen detector's input resolution, so a later detector-versus-segmenter comparison is not also a resolution comparison.
4. **The mask-native alternative stays available.** It is recorded as `NOT_SELECTED_FALLBACK`, not discarded.

The conversion is characterised as `ACCEPTED_WITH_QUANTIFIED_APPROXIMATION`, never as lossless: **no instance round-trips exactly**, and the reason is partly the rasteriser convention rather than the format.

No mask-native architecture was installed, trained or benchmarked in this project, so nothing here says YOLO11n-seg is better than one. It is the architecture chosen under this evidence and these constraints.

## 4. Why canonical COCO remains authoritative

`MODEL_ADAPTER_LIMITATION`. Canonical ground truth is `COCO_INSTANCE_SEGMENTATION`; the YOLO labels are a `MODEL_SPECIFIC_DERIVED_REPRESENTATION`. Approving them for training does not promote them. If a YOLO label and the canonical COCO document ever disagree, the COCO document is right and the adapter is broken: regenerate it from the canonical source, never hand-edit it, and never re-derive the canonical document from a label.

## 5. Known adapter limitations

`MODEL_ADAPTER_LIMITATION`.

- **`SERIALIZATION_AND_QUANTIZATION`** - Coordinates are written at six decimal places and rasterised through an int32 truncation. This is the largest contributor: 0.012458 mean IoU.
- **`COMPONENT_JOIN_APPROXIMATION`** - A disconnected mask cannot be expressed: one row is one flat ring, so the 307 multi-component instances are bridged with zero-width connectors. Costs 0.000843 mean IoU.
- **`HOLE_FILL_APPROXIMATION`** - An interior ring cannot be expressed, so holes are filled: 180 instances carry holes totalling 1072133 filled pixels.
- **`RASTER_CONVENTION_DIFFERENCE`** - pycocotools and OpenCV do not rasterise identical geometry into identical pixels, which is why no instance round-trips exactly and why this level is never reported as YOLO format loss.
- **`SMALL_MASK_SENSITIVITY`** - All 20 worst instances are 4-59 px masks, where a single boundary pixel is a large share of the area. This is the primary fidelity-risk stratum.

## 6. Why no instance filtering occurred

All 1726 development instances remain. Excluding the 47 instances whose round-trip IoU fell below 0.90, or the tiny, multi-component, holed or synthetic-rectangle masks, would change the canonical modelling population in response to a model-format limitation - choosing the data to suit the tool, and quietly making every later metric describe an easier dataset than the project claims to have.

## 7. S0 experimental question

> What validation instance-segmentation performance does a pretrained YOLO11n-seg model achieve at the same 768-pixel input resolution as the frozen detector, using the phase 8A audited segmentation adapter? S0 is a baseline reference point, not the project's final segmenter and not an attempt at a good score.

S0 is a **baseline** (`BASELINE_NOT_THE_FINAL_SEGMENTER`). It is not the project's final segmenter, and no comparison protocol exists for it yet.

## 8. Frozen data

`PREDECLARED_PROTOCOL`. The exact bytes phase 8A audited, verified on disk.

| Field | Value |
| --- | --- |
| Dataset descriptor | `data/processed/adapters/yolo_segmentation_audit/dataset.yaml` |
| Descriptor keys | `names`, `path`, `train`, `val` |
| Train images / instances | 303 / 1422 |
| Validation images / instances | 65 / 304 |
| Total instances | 1726 |
| Negative images | 12 |
| Train label digest | `63a8145d976f7a4c9d1e2868b052c0e0d4233a15f7c6e6b3a1a0e25be6b4cd17` |
| Validation label digest | `dae69290936641d73e576682631ef73e026bf7f33cec8c2a60a7789ac054e83f` |
| Development label digest | `ff21c7822921601a5c2ac6d0d324eb9f234f1a13ce16647f834936de5d5682f2` |
| Image membership digest | `616701e18deb1e6d873459c94e702d33752f1d712fe0f6eebb0fa8288f126d41` |
| Class map digest | `596dab5b5756e3d4253924f84bf46d37a87d80b6ec830da781a08a93cd823753` |
| Split digest | `a230869ff4cb45f53654d67357a27f2a0def6d8ba79f9fa4880f0fdda2f046cc` |

The labels were **not regenerated**. A differing digest stops the phase as `ADAPTER_FINGERPRINT_MISMATCH` rather than triggering a rebuild, because a rebuild would silently replace measured bytes with unmeasured ones.

## 9. Frozen training configuration

`PREDECLARED_PROTOCOL`. Complete, not just the fields that were changed.

| Argument | Value | Against installed default |
| --- | --- | --- |
| `seed` | 42 | `PROJECT_OVERRIDE` |
| `amp` | True | `FRAMEWORK_DEFAULT` |
| `batch` | 8 | `PROJECT_OVERRIDE` |
| `close_mosaic` | 10 | `FRAMEWORK_DEFAULT` |
| `cos_lr` | False | `FRAMEWORK_DEFAULT` |
| `deterministic` | True | `FRAMEWORK_DEFAULT` |
| `epochs` | 100 | `FRAMEWORK_DEFAULT` |
| `imgsz` | 768 | `PROJECT_OVERRIDE` |
| `lr0` | 0.01 | `FRAMEWORK_DEFAULT` |
| `lrf` | 0.01 | `FRAMEWORK_DEFAULT` |
| `momentum` | 0.937 | `FRAMEWORK_DEFAULT` |
| `optimizer` | auto | `FRAMEWORK_DEFAULT` |
| `patience` | 50 | `PROJECT_OVERRIDE` |
| `pretrained` | True | `FRAMEWORK_DEFAULT` |
| `val` | True | `FRAMEWORK_DEFAULT` |
| `warmup_epochs` | 3.0 | `FRAMEWORK_DEFAULT` |
| `weight_decay` | 0.0005 | `FRAMEWORK_DEFAULT` |
| `workers` | 8 | `FRAMEWORK_DEFAULT` |
| `augmentation` | ULTRALYTICS_DEFAULT_SEGMENTATION_TRAINING_POLICY | policy |

**Batch.** PREDECLARED_EXECUTION_DECISION. Batch 8 rather than the detection phases' 16, fixed before any S0 performance number existed. The segmentation head carries an additional mask branch and the trainer materialises per-image mask targets, so the memory profile is not the detector's, and this machine has roughly 8 GiB of VRAM. No scientific question in this project depends on the segmentation batch matching the detection batch: S0 is compared against future segmentation experiments under this same protocol, never against a detector's training configuration. A genuine CUDA out-of-memory event at batch 8 stops the experiment as MEMORY_CONSTRAINT_REVIEW_REQUIRED and is never rescued by a smaller batch, auto-batch, gradient accumulation, a different imgsz or a different model.

Device requirement is `CUDA_GPU_REQUIRED`. Runtime verified: python 3.12.13, torch 2.11.0+cu128, torchvision 0.26.0+cu128, ultralytics 8.4.138, CUDA 12.8, NVIDIA GeForce RTX 5070 Laptop GPU (sm_120, 7.96 GiB).

Pretrained weights `yolo11n-seg.pt`, obtained by `ULTRALYTICS_ASSET_DOWNLOAD` under ultralytics 8.4.138, SHA-256 `55ed65c56c91713d23e8402371c6c49a6fd84f257f7dce452e8d70e41dcbe152`, 6182636 bytes. Git-ignored: the repository commits the record, not the binary.

## 10. Segmentation-specific framework arguments

`PREDECLARED_PROTOCOL`, evidence `INSTALLED_EFFECTIVE_CONFIGURATION`. Read from the installed configuration, not from online documentation, and verified equal to it at freeze time.

| Argument | Value | Why it is recorded |
| --- | --- | --- |
| `dropout` | 0.0 | regularisation, exposed by the framework |
| `mask_ratio` | 4 | the mask target is downsampled by this factor before the loss sees it |
| `max_det` | 300 | caps detections per image during validation |
| `multi_scale` | 0.0 | would vary input resolution during training if set |
| `overlap_mask` | True | overlapping instance masks are rasterised into one indexed map, so an occluded instance's target is what remains visible |
| `rect` | False | would change letterboxing if set |
| `retina_masks` | False | affects mask resolution at inference, not training targets |
| `single_cls` | False | would collapse the class map if set |

**Augmentation policy** `ULTRALYTICS_DEFAULT_SEGMENTATION_TRAINING_POLICY`. The installed version's own defaults, enumerated in full so the report does not depend on a library version's documentation:

| Argument | Value |
| --- | --- |
| `auto_augment` | randaugment |
| `bgr` | 0.0 |
| `copy_paste` | 0.0 |
| `copy_paste_mode` | flip |
| `cutmix` | 0.0 |
| `degrees` | 0.0 |
| `erasing` | 0.4 |
| `fliplr` | 0.5 |
| `flipud` | 0.0 |
| `hsv_h` | 0.015 |
| `hsv_s` | 0.7 |
| `hsv_v` | 0.4 |
| `mixup` | 0.0 |
| `mosaic` | 1.0 |
| `perspective` | 0.0 |
| `scale` | 0.5 |
| `shear` | 0.0 |
| `translate` | 0.1 |

These are **model training transformations, not canonical preprocessing**. The canonical images and masks are unchanged, and augmentation is not applied to validation. Nothing here was tuned, and tuning it before S0 exists would make S0 a tuned result rather than a baseline.

## 11. Metric hierarchy

`PREDECLARED_PROTOCOL`. Fixed before training, so the headline number cannot be chosen once results are visible.

- **Primary:** `mask_mAP@0.50:0.95`
- **Secondary mask:** `mask_mAP@0.50`, `mask_precision`, `mask_recall`
- **Box metrics from the segmenter:** `box_mAP@0.50:0.95`, `box_mAP@0.50`, `box_precision`, `box_recall`
- **Per class (mask):** `mask_AP@0.50`, `mask_AP@0.50:0.95`
- **Per class (box):** `box_AP@0.50`, `box_AP@0.50:0.95`
- **Macro over supported classes:** `supported_macro_mask_map50_95`

Mask and box families are reported side by side and never merged. A composite box-plus-mask score is refused by the configuration parser, not merely discouraged: inventing one after the fact is how a weak mask result hides behind a strong box one.

`supported_macro_mask_map50_95` is the unweighted mean of per-class mask AP@0.50:0.95 over the classes the frozen phase 7 support rule admits. **S0 reports it; it is not a winner-selection metric**, because there is nothing yet to select between and the comparison protocol that would use it has not been frozen.

## 12. The `vest_loose` limitation

`PREDECLARED_PROTOCOL`. The support rule is `FROZEN_IN_PHASE_7A_REUSED_UNCHANGED`: COMPARISON_SUPPORTED requires >= 5 positive validation source images AND >= 20 validation instances. Applied to the frozen split, `vest_loose` holds 1 validation image and 8 instances, so it is classified `DESCRIPTIVE_HIGH_UNCERTAINTY`.

vest_loose occupies 1 validation source image carrying 8 instances under the frozen split, so it is classified DESCRIPTIVE_HIGH_UNCERTAINTY by the frozen support rule and its validation AP is not a usable selection signal. Therefore: do not tune hyperparameters against vest_loose; do not prefer or reject a model because vest_loose moved; do not rank it against the supported classes. It remains a required class and is reported in full - mask and box, AP@0.50 and AP@0.50:0.95, precision and recall - always with an explicit small-sample caveat. The holdout also cannot be consulted to resolve the uncertainty.

## 13. Checkpoint-selection behaviour

`HUMAN_ARCHITECTURE_DECISION` and `PREDECLARED_PROTOCOL`, evidence `INSTALLED_PACKAGE_SOURCE_INSPECTION`. The rule is `ULTRALYTICS_BEST_ON_VALIDATION_FITNESS`. What that means for a segmentation model was established from the installed source **before** any full S0 result existed, returned for methodological review, and accepted.

### 13.1 What the framework actually optimises

- **Classification:** `COMBINED_MASK_AND_BOX_MAP50_95_UNWEIGHTED_SUM` (driven by `BOTH_MASK_AND_BOX`)
- **Definition:** SegmentMetrics.fitness returns self.seg.fitness() + DetMetrics.fitness, and each term is Metric.fitness, a weighted mean of [precision, recall, mAP@0.50, mAP@0.50:0.95] with weights [0.0, 0.0, 0.0, 1.0]. So the framework's validation fitness is the unweighted SUM of mask mAP@0.50:0.95 and box mAP@0.50:0.95, and best.pt is not selected on the mask metric alone.
- **Consequence:** The reported checkpoint can differ from the epoch that maximised the mask metric on its own. This is recorded before training rather than discovered after it, and it is NOT replaced by a private checkpoint rule: substituting one would make S0 incomparable to any experiment that used the framework's own. If a later phase judges the misalignment material, that is a methodological review, not a quiet fix.

### 13.2 The reviewed decision

**Accepted: the native Ultralytics composite fitness selects S0's checkpoint.** No custom mask-only checkpoint selector was written, and none is authorised.

| Field | Value |
| --- | --- |
| `checkpoint_selection_policy` | `ULTRALYTICS_NATIVE_SEGMENTATION_FITNESS` |
| `checkpoint_selection_review_status` | `HUMAN_REVIEWED_AND_ACCEPTED_BEFORE_S0` |
| `checkpoint_selection_semantics` | `BOX_MAP50_95_PLUS_MASK_MAP50_95` |
| `checkpoint_selection_box_component_weight` | 1.0 |
| `checkpoint_selection_mask_component_weight` | 1.0 |
| `primary_scientific_reporting_metric` | `MASK_MAP50_95` |
| `selection_metric_equals_primary_reporting_metric` | **False** - intentional |

That last row is the whole point of this section. **The checkpoint is selected on a box-plus-mask composite while the project reports mask mAP@0.50:0.95**, so the epoch S0 reports need not be the epoch that maximised the reported metric. That divergence is recorded as a known property of the protocol rather than left for a reader to discover.

### 13.3 Why the native fitness was kept

- The behaviour was established from the installed source before any full S0 result existed, so accepting it is a protocol decision rather than a rationalisation of whichever checkpoint happened to score well.
- It is the native, deterministic behaviour of the frozen Ultralytics version. Keeping it means S0 introduces no custom checkpoint-selection mechanism whose own behaviour would then need validating.
- Instance segmentation requires both instance localisation and mask prediction, so a checkpoint chosen on both terms is a defensible baseline - which is a reason it is acceptable, not evidence that it is better.
- This is NOT a claim that the composite fitness is scientifically superior to mask-only selection. Nothing in this project compares the two, and no experiment here could support such a claim.
- S0 may never be retrospectively re-read as an epoch selected only by mask mAP@0.50:0.95. The checkpoint S0 reports is the one the native fitness chose, and changing that after the fact would silently replace the experiment with a different one.

**Superiority claim:** NONE. Keeping the native composite is an explicitly accepted baseline protocol choice, not a finding that it beats mask-only selection. No experiment in this project compares the two, and none could support such a claim.

The framework fitness is a **checkpoint-selection mechanism, not a scientific headline metric**: `framework_fitness_is_a_reporting_metric: False`. No combined box-plus-mask number is introduced for reporting anywhere in this project, and the metric hierarchy in section 11 is unchanged by this decision.

### 13.4 Protocol invariant for future comparisons

`PREDECLARED_PROTOCOL`. PROTOCOL_INVARIANT. Every future segmentation experiment compared directly with S0 must select its checkpoint by ULTRALYTICS_NATIVE_SEGMENTATION_FITNESS, unless a new comparison protocol is human-reviewed and frozen BEFORE any affected experiment is run. A comparison in which one model's checkpoint was chosen on a composite and another's on the mask metric alone would be measuring the selection rule as well as the model. S0's checkpoint semantics are never altered retroactively to satisfy a later protocol; a new rule applies to the experiments frozen under it.

Two things follow and are enforced by the configuration parser: no post-hoc mask-only checkpoint selection is authorised (`custom_mask_only_selector_authorized: False`), and S0's checkpoint semantics are never reinterpreted after the fact (`retrospective_reinterpretation_allowed: False`).

## 14. Future direct mask-IoU requirement

`FUTURE_EVALUATION_REQUIREMENT` · status `REQUIRED_FUTURE_PREDECLARED_EVALUATION`.

The academic deliverable requires IoU. Ultralytics' mask AP is an averaged detection-style metric over IoU thresholds and is not a direct instance-mask IoU diagnostic, so reporting AP alone would not satisfy it.

**What is owed.** After S0's checkpoint is selected, the project must report an explicit instance-mask IoU diagnostic under a protocol that predeclares the inference settings, the prediction-to-ground-truth matching rule, how unmatched predictions and unmatched ground truth are counted, and the averaging scheme.

**What must not happen.** That protocol must not be written after looking at S0's predictions. Choosing a matching rule once the failures are visible is how a diagnostic becomes a flattering one.

Phase 8B defines the requirement only: `defined_here: False`, `executed_here: False`. It belongs to a later phase, alongside the phase 10 evaluation protocol.

## 15. Smoke-test result

`NON_EXPERIMENTAL` · `DO_NOT_REPORT_AS_MODEL_RESULT` · status `SUCCESS`.

Prove the model loads, the segmentation labels parse, CUDA forward and backward run, the validation loader works, the mask loss executes and a checkpoint reaches disk. Nothing else.

| Field | Value |
| --- | --- |
| Epochs | 1 |
| imgsz | 768 |
| Batch | 8 |
| Seed | 42 |
| Runtime | 51.275 s |
| Peak GPU memory reserved | 3.197 GiB (3433037824 bytes) |
| Out of memory | False |
| Optimizer selected | AdamW |
| Optimizer evidence | `FRAMEWORK_LOG_LINE_DIRECT_CAPTURE` |
| `best.pt` written | True |
| `last.pt` written | True |
| Run directory | `artifacts/segmentation/S0_smoke` (git-ignored) |
| Accuracy metrics recorded | False |

Deviations from the frozen S0 protocol, both deliberate:
- epochs 1 instead of 100
- plots disabled, so no dataset or prediction imagery is rendered

A one-epoch run's AP is not a model result. Recording it would create a number the project would then have to explain, and inviting a comparison against it is exactly the tuning this phase forbids.

## 16. Holdout policy

`HOLDOUT_POLICY` · status `PROTECTED_NOT_ACCESSED`.

Phase 8B selected an architecture, approved already-audited label bytes, froze the S0 protocol and ran a one-epoch engineering smoke test on the development splits. The holdout was not read, materialised, adapted, converted, counted, inspected or predicted on; no holdout label, statistic or identifier exists in any artifact this phase wrote, and no holdout adapter directory was created. Nothing about choosing an architecture or proving that a training loop executes depends on which images are held out.

The adapter descriptor carries no holdout key, no holdout adapter directory exists, and the phase runs under an environment where the unlock variable is not set. Phase 11 reads the holdout once, after both models are frozen.

## 17. Fallback status

`NOT_SELECTED_FALLBACK`. mask-native instance segmentation consuming canonical COCO masks and RLE directly, for example Mask R-CNN. Nothing about it has been implemented, installed, benchmarked or measured, so no statement in this report compares the two. It remains available if a later phase judges the approximation material.

## 18. Next phase

Phase 8C: run S0 once under this frozen protocol. S0 is currently `NOT_EXECUTED_PROTOCOL_ONLY` and this report contains no S0 metric. Do not tune, do not vary a hyperparameter to see whether the number moves, and do not begin 8C without an explicit instruction.

---

Protocol fingerprint `0c6bbcd8529be08b1ef328edd3d8cd13739f3b7eab24b627220cdb5bf8a5dd9e` · configuration `configs/segmentation_baseline.yaml` SHA-256 `c43f9dde3e512c00a6738b2210cc45538439e6940d076322ea0620c12e3e3867`.
