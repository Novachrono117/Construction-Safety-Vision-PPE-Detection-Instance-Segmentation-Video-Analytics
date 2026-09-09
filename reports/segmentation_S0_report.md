# S0 - YOLO11n-seg segmentation baseline

Phase 8C · classification `S0_SEGMENTATION_BASELINE_COMPLETE` · experiment `S0` · baseline status `S0_COMPLETE` · final segmenter `UNSELECTED_PENDING_REVIEW`

**Every number here is a validation number.** The holdout has never been evaluated, and nothing below says anything about test performance. S0 is a baseline, not the project's final segmenter.

Repository commit at production time: `3aaaae4af0ec20f6e732f94aaa206d0b28d4667d`.

## 1. Experimental question

> What validation instance-segmentation performance does a pretrained YOLO11n-seg model achieve at the same 768-pixel input resolution as the frozen detector, using the phase 8A audited segmentation adapter? S0 is a baseline reference point, not the project's final segmenter and not an attempt at a good score.

`BASELINE_NOT_THE_FINAL_SEGMENTER`. S0 is the project's only segmentation experiment. There is nothing to select between and no segmentation comparison protocol has been frozen, so no final segmenter is chosen here.

## 2. Frozen S0 protocol

`PREDECLARED_PROTOCOL`. Frozen in phase 8B before S0 trained, protocol fingerprint `0c6bbcd8529be08b1ef328edd3d8cd13739f3b7eab24b627220cdb5bf8a5dd9e`, verified against the committed phase 8B manifest before the first optimisation step.

| Field | Value |
| --- | --- |
| Architecture | YOLO11n-seg |
| imgsz | 768 |
| batch | 8 |
| seed | 42 |
| Pretrained weights | `yolo11n-seg.pt` SHA-256 `55ed65c56c91713d23e8402371c6c49a6fd84f257f7dce452e8d70e41dcbe152` |

## 3. Canonical representation versus the model adapter

`MODEL_ADAPTER_LIMITATION`. Canonical ground truth is `COCO_INSTANCE_SEGMENTATION`; the labels S0 trained on are a `MODEL_SPECIFIC_DERIVED_REPRESENTATION` approved in phase 8B. The approval attaches to bytes, and those bytes were re-verified against the approved digests before training and again after it: 1726 instances over 303 train and 65 validation images, `regenerated: False`, `modified: False`.

Training read a hard-linked runtime view of those bytes rather than the audited directory itself, so the framework's `.cache` files did not land inside phase 8A's evidence. Image membership, image bytes and label bytes are identical; no label was converted and no coordinate was rewritten.

## 4. Phase 8A fidelity context

`MODEL_ADAPTER_LIMITATION`. The YOLO segmentation format cannot express an interior hole or a disconnected mask, and phase 8A measured what that costs: mean mask IoU 0.973066, median 0.984576, P05 0.918176, and **no instance round-trips exactly**. Every metric below is therefore a measurement of a model trained on an approximated representation of the canonical masks - which is one reason the direct IoU diagnostic in section 17 scores against the canonical COCO masks rather than against the adapter.

## 5. Runtime provenance

python 3.12.13, torch 2.11.0+cu128, torchvision 0.26.0+cu128, ultralytics 8.4.138, CUDA 12.8, NVIDIA GeForce RTX 5070 Laptop GPU (sm_120, 7.96 GiB).

A pre-run provenance record was written before the first optimisation step, so the protocol demonstrably preceded the result.

## 6. Effective training configuration

`COMPUTED_RESULT`, read back from the framework's own recorded arguments.

| Argument | Effective value |
| --- | --- |
| `amp` | True |
| `auto_augment` | randaugment |
| `batch` | 8 |
| `bgr` | 0.0 |
| `close_mosaic` | 10 |
| `copy_paste` | 0.0 |
| `copy_paste_mode` | flip |
| `cos_lr` | False |
| `cutmix` | 0.0 |
| `degrees` | 0.0 |
| `deterministic` | True |
| `device` |  |
| `dropout` | 0.0 |
| `epochs` | 100 |
| `erasing` | 0.4 |
| `fliplr` | 0.5 |
| `flipud` | 0.0 |
| `hsv_h` | 0.015 |
| `hsv_s` | 0.7 |
| `hsv_v` | 0.4 |
| `imgsz` | 768 |
| `lr0` | 0.01 |
| `lrf` | 0.01 |
| `mask_ratio` | 4 |
| `max_det` | 300 |
| `mixup` | 0.0 |
| `model` | artifacts/weights/yolo11n-seg.pt |
| `momentum` | 0.937 |
| `mosaic` | 1.0 |
| `multi_scale` | 0.0 |
| `optimizer` | auto |
| `overlap_mask` | True |
| `patience` | 50 |
| `perspective` | 0.0 |
| `rect` | False |
| `retina_masks` | False |
| `scale` | 0.5 |
| `seed` | 42 |
| `shear` | 0.0 |
| `single_cls` | False |
| `translate` | 0.1 |
| `warmup_bias_lr` | 0.1 |
| `warmup_epochs` | 3.0 |
| `warmup_momentum` | 0.8 |
| `weight_decay` | 0.0005 |
| `workers` | 8 |

**Optimizer.** Declared policy `auto`, which resolved to **AdamW** at lr0 0.001111 and momentum 0.9, evidence `FRAMEWORK_LOG_LINE_DIRECT_CAPTURE`. optimizer: auto overrides the file's lr0 and momentum. The declared values are the protocol's policy inputs, never a statement of what ran. Scheduler `LINEAR_LAMBDA_LR0_TO_LR0_TIMES_LRF`, weight decay 0.0005, warmup 3.0 epochs.

## 7. Native checkpoint-selection semantics

`FRAMEWORK_CHECKPOINT_POLICY`. Policy `ULTRALYTICS_NATIVE_SEGMENTATION_FITNESS`, semantics `BOX_MAP50_95_PLUS_MASK_MAP50_95` - the unweighted sum of box and mask mAP@0.50:0.95, reviewed and accepted in phase 8B. **The checkpoint is therefore not selected on the primary reported metric**, and the epoch S0 reports need not be the epoch that maximised mask mAP@0.50:0.95 on its own.

The composite is recomputed from results.csv as metrics/mAP50-95(B) + metrics/mAP50-95(M), which is exactly SegmentMetrics.fitness for the installed version, and the recorded best epoch is its argmax.

`manual_epoch_selection: False` · `mask_only_checkpoint_created: False`

## 8. Training execution

`COMPUTED_RESULT`. Exactly one full run.

| Field | Value |
| --- | --- |
| Epochs configured | 100 |
| Epochs logged | 100 |
| Termination | ALL_EPOCHS_COMPLETED |
| Early stopping | False |
| Training time | 1106.85 s (`FRAMEWORK_RESULTS_CSV_CUMULATIVE_TIME`; the framework's own log reports 0.308 hours for 100 epochs) |
| Peak GPU memory reserved | NOT_PERSISTED_FOR_THIS_RUN |
| Engineering aborts | 0 |
| Resumed | False |
| Run directory | `artifacts/segmentation/S0` (git-ignored) |

The training process measured peak device memory but exited before the artifact could be written, and the figure is not recoverable from any file on disk. It is recorded as not persisted rather than reconstructed from terminal output, because a number a reader cannot trace to an artifact is not evidence. The frozen batch of 8 completed without an out-of-memory event, which is what the protocol required.

**One artifact-write re-execution is recorded.** The first attempt trained, validated and ran the diagnostic successfully, then refused to write its artifacts: the sensitive-content scan found machine-specific absolute paths that Ultralytics records in args.yaml and the runner was copying verbatim into the effective-configuration block. No artifact was written, so nothing was published and nothing was corrected after the fact. Absolute paths recorded by the framework are now rewritten repository-relative before they reach an artifact. No metric, threshold, hyperparameter or selection rule was touched. Re-executed: the authoritative framework validation, the direct mask-IoU diagnostic. Not re-executed: training - the single S0 run and its weights are unchanged, and the checkpoint the frozen rule selected is the same file. Validation and the diagnostic are deterministic functions of a fixed checkpoint, fixed data and fixed settings, all of which are unchanged. Re-executing them reproduces the same numbers; what changed is only whether they could be written down.

## 9. Checkpoint selection

`FRAMEWORK_CHECKPOINT_POLICY`. Best epoch **59** at native composite fitness 0.88418; next best epoch 77 at 0.87272. The recorded best epoch was confirmed to be the argmax of the recomputed composite, so the frozen rule provably chose it.

| Checkpoint | SHA-256 | Bytes |
| --- | --- | --- |
| `best.pt` | `d7b512b95fafdc8658fd802459d2c87d9ad3f11f922162a774dacf8ca75b87a3` | 6041685 |
| `last.pt` | `9b8a956fb02e0719c07b36c0c8f5278827c1208c5165e0ac87d39c812f5efa92` | 6041685 |

Neither weight file is committed; the repository records the digest, not the binary.

## 10. Primary mask result

`PRIMARY_SCIENTIFIC_RESULT`. **mask mAP@0.50:0.95 = 0.407942** on the frozen validation split.

One run, one configuration, no tuning. `deterministic: true` reduces run-to-run variance without removing it, so this is a single measurement and not an estimate with an interval.

## 11. Secondary mask metrics

| Metric | Value |
| --- | --- |
| mask mAP@0.50 | 0.57983 |
| mask precision | 0.850509 |
| mask recall | 0.528435 |

**Operating-point caveat.** Ultralytics reports one precision/recall pair at the F1-maximising point rather than at a fixed confidence, so both figures partly reflect where that point landed. Read them as a hint about the precision/recall balance, never as threshold-independent properties. No threshold was tuned.

## 12. Per-class mask metrics

| Class | precision | recall | AP@0.50 | AP@0.50:0.95 |
| --- | --- | --- | --- | --- |
| `helmet_loose` | 0.925178 | 0.789474 | 0.909395 | 0.789927 |
| `helmet_on_head` | 0.917247 | 0.707609 | 0.786113 | 0.586523 |
| `person` | 0.718326 | 0.532847 | 0.539634 | 0.271182 |
| `vest_loose` | 1.0 | 0.0 | 0.011359 | 0.001782 |
| `vest_on_body` | 0.691796 | 0.612247 | 0.652652 | 0.390296 |

## 13. Box metrics from the segmenter

`COMPUTED_RESULT`. Reported separately and never merged with the mask family. The framework's composite fitness remains checkpoint-selection metadata, not a scientific score.

| Metric | Value |
| --- | --- |
| box mAP@0.50:0.95 | 0.478156 |
| box mAP@0.50 | 0.647256 |
| box precision | 0.88616 |
| box recall | 0.552147 |

| Class | box precision | box recall | box AP@0.50 | box AP@0.50:0.95 |
| --- | --- | --- | --- | --- |
| `helmet_loose` | 0.931835 | 0.789474 | 0.906553 | 0.802332 |
| `helmet_on_head` | 0.88891 | 0.681001 | 0.784851 | 0.625702 |
| `person` | 0.861512 | 0.635715 | 0.748113 | 0.522227 |
| `vest_loose` | 1.0 | 0.0 | 0.088504 | 0.025059 |
| `vest_on_body` | 0.748545 | 0.654545 | 0.70826 | 0.415462 |

**These are not comparable to the frozen detector's numbers.** D2 is a different model trained under a different protocol, and no controlled detector-versus-segmenter comparison has been run. Reading the two side by side would be comparing two experiments that differ in more than one thing.

## 14. Supported mask macro

`COMPUTED_RESULT`. `supported_macro_mask_map50_95` = **0.509482**, the unweighted mean of per-class mask AP@0.50:0.95 over the classes the frozen support rule admits (`helmet_loose`, `helmet_on_head`, `person`, `vest_on_body`).

Rule (`FROZEN_IN_PHASE_7A_REUSED_UNCHANGED`): COMPARISON_SUPPORTED requires >= 5 positive validation source images AND >= 20 validation instances.

`is_a_selection_metric: False`. S0 is the only segmentation experiment, so there is nothing to select between. This value is descriptive and declares no winner. **No segmentation winner is declared.**

## 15. The `vest_loose` limitation

`LIMITATION` · status `DESCRIPTIVE_HIGH_UNCERTAINTY`.

vest_loose occupies 1 validation source image carrying 8 instances under the frozen split, so it is classified DESCRIPTIVE_HIGH_UNCERTAINTY by the frozen support rule and its validation AP is not a usable selection signal. Therefore: do not tune hyperparameters against vest_loose; do not prefer or reject a model because vest_loose moved; do not rank it against the supported classes. It remains a required class and is reported in full - mask and box, AP@0.50 and AP@0.50:0.95, precision and recall - always with an explicit small-sample caveat. The holdout also cannot be consulted to resolve the uncertainty.

It is reported in full above and in the diagnostic below (`reported_in_full: True`), and it decides nothing (`may_decide_anything: False`).

## 16. Direct mask-IoU protocol

`PREDECLARED_PROTOCOL`. `DIRECT_INSTANCE_MASK_IOU_DIAGNOSTIC`, frozen in `configs/segmentation_mask_iou_evaluation.yaml` **before the first optimisation step**, fingerprint `b912039ca77b36959f74fcdbaed109dbf5e3bd707709556f95a19085c7f34d80`.

Ground truth is `CANONICAL_COCO_INSTANCE_SEGMENTATION` - the canonical phase 5D COCO validation masks, **never the YOLO adapter**, whose own approximation would otherwise be folded into the model's score.

| Setting | Value |
| --- | --- |
| `agnostic_nms` | False |
| `augment` | False |
| `classes` | ALL_FIVE_FROZEN_CLASSES |
| `conf` | 0.25 |
| `half` | False |
| `imgsz` | 768 |
| `iou` | 0.7 |
| `max_det` | 300 |
| `retina_masks` | True |
| matching | `SCIPY_LINEAR_SUM_ASSIGNMENT_MAXIMIZE_MASK_IOU` |

The confidence and NMS IoU are a **predeclared operational diagnostic threshold**, fixed before any S0 performance number existed. No sweep, no second threshold, and no test-time augmentation. Matching is one-to-one within an image and a class, solved to maximise total IoU; an assigned pair sharing no pixel is discarded rather than counted as a match.

## 17. Direct mask-IoU results

`DIRECT_IOU_DIAGNOSTIC`, executed exactly once on the selected checkpoint.

| Diagnostic | Value |
| --- | --- |
| `matched_mask_iou_mean` | **0.717462** |
| `gt_normalized_mask_iou` | **0.556977** |
| `gt_match_coverage` | 0.776316 |
| `gt_iou50_coverage` | 0.588816 |
| `gt_iou75_coverage` | 0.460526 |

| Count | Value |
| --- | --- |
| Canonical GT instances | 304 |
| Predicted instances | 304 |
| Assignments with positive overlap | 236 |
| Unmatched GT | 68 |
| Unmatched predictions | 68 |

Per class:

| Class | GT | pred | matched | matched mean IoU | GT-normalised IoU | IoU>=0.50 | IoU>=0.75 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `helmet_loose` | 57 | 55 | 47 | 0.935424 | 0.771314 | 0.824561 | 0.824561 |
| `helmet_on_head` | 47 | 40 | 36 | 0.876559 | 0.671407 | 0.765957 | 0.680851 |
| `person` | 137 | 137 | 103 | 0.578106 | 0.434635 | 0.423358 | 0.233577 |
| `vest_loose` | 8 | 2 | 2 | 0.263543 | 0.065886 | 0.0 | 0.0 |
| `vest_on_body` | 55 | 70 | 48 | 0.702667 | 0.613237 | 0.690909 | 0.527273 |

`vest_loose` remains `DESCRIPTIVE_HIGH_UNCERTAINTY` here too: its direct-IoU figures stand on one validation image and must not be used to tune, rank or select anything.

## 18. How AP and direct IoU relate

`LIMITATION`. The two answer different questions and are not interchangeable.

- **`matched_mask_iou_mean`** answers, approximately: *when S0 produced an overlapping same-class instance, how similar was its mask to the canonical one?* It says nothing about the instances S0 never found.
- **`gt_normalized_mask_iou`** answers, approximately: *across every canonical instance, counting a miss as zero, how much same-class mask overlap did S0 recover at the frozen operating point?* It folds mask quality and coverage into one figure.
- **mask mAP@0.50:0.95** is neither. It averages precision over recall at a range of IoU thresholds and is sensitive to confidence ranking, which the two diagnostics above ignore entirely.

Neither figure is a COCO AP. AP is a ranking-sensitive average over IoU thresholds; these are mask overlap at one predeclared operating point. None of the three may be substituted for another, and none of them was used to change S0.

## 19. Training dynamics

`COMPUTED_RESULT`, read from `results.csv` only.

| Quantity | First epoch | Last epoch |
| --- | --- | --- |
| `lr/pg0` | 0.000361 | 2.2e-05 |
| `lr/pg1` | 0.000361 | 2.2e-05 |
| `lr/pg2` | 0.000361 | 2.2e-05 |
| `train/box_loss` | 0.89322 | 0.48091 |
| `train/cls_loss` | 3.42437 | 0.49306 |
| `train/dfl_loss` | 1.17061 | 0.8989 |
| `train/seg_loss` | 3.00528 | 0.89039 |
| `train/sem_loss` | 0.0 | 0.0 |
| `val/box_loss` | 0.95393 | 0.78484 |
| `val/cls_loss` | 3.74085 | 0.99324 |
| `val/dfl_loss` | 1.49463 | 1.16664 |
| `val/seg_loss` | 3.73355 | 2.25638 |
| `val/sem_loss` | 0.0 | 0.0 |

Native composite fitness ran 0.20135 at the first logged epoch to 0.86817 at the last, peaking at 0.88418 on epoch 59, with 41 epochs logged after it.

Read from results.csv only. No validation image was opened: image-level error analysis is a later, deliberate phase and starting it here would pre-empt it.

## 20. Confusion and metric figures

`COMPUTED_RESULT`. Metric-only figures are committed under `reports/figures/`: `BoxF1_curve.png`, `BoxPR_curve.png`, `BoxP_curve.png`, `BoxR_curve.png`, `MaskF1_curve.png`, `MaskPR_curve.png`, `MaskP_curve.png`, `MaskR_curve.png`, `confusion_matrix.png`, `confusion_matrix_normalized.png`, `results.png`.

The framework also writes `train_batch*.jpg`, `val_batch*.jpg` and `labels.jpg`, which render dataset imagery and prediction montages. Those stay in the git-ignored run directory: committing them would publish dataset images and pre-empt the deliberate error-analysis phase. **No validation image was opened to explain an individual failure**, and no image-level error analysis was performed.

## 21. Resource use

| Field | Value |
| --- | --- |
| Parameters | 2843583 |
| GFLOPs | 9.8 |
| Layers | 204 |
| Training time | 1106.85 s |
| Peak GPU memory reserved | NOT_PERSISTED_FOR_THIS_RUN |
| GPU | NVIDIA GeForce RTX 5070 Laptop GPU (sm_120) |

`FRAMEWORK_VALIDATION_SPEED` (ms per image): inference 9.336043, loss 0.003875, postprocess 1.439111, preprocess 1.860951. Measured by the framework during validation, not a production latency benchmark. No standardised latency study was run in this phase.

## 22. Holdout compliance

`HOLDOUT_POLICY` · status `PROTECTED_NOT_ACCESSED`.

Phase 8C trained S0 on the frozen train split, validated it on the frozen validation split, and ran the direct mask-IoU diagnostic on that same validation split. The holdout was not read, materialised, adapted, counted, predicted on, inspected or plotted; no holdout identifier, label, prediction or statistic exists in any artifact this phase wrote, and the dataset descriptor the framework read carries no holdout key.

## 23. Limitations

`LIMITATION`. What this experiment does and does not establish.

- **One run, one configuration.** Nothing was repeated, so run-to-run variance on this setup is UNKNOWN and every figure is a single measurement.
- **Validation only.** The holdout has never been evaluated. Nothing here predicts test performance.
- **No comparison exists.** S0 is the only segmentation experiment, so no claim about architecture, resolution or capacity is supported by it.
- **Trained on an approximated representation.** The YOLO adapter cannot express holes or disconnected masks; phase 8A measured the cost and phase 8B accepted it, but the model never saw the canonical geometry.
- **The checkpoint was not selected on the reported metric.** The native composite includes box mAP@0.50:0.95, which is a reviewed and accepted property of this baseline, not an oversight.
- **Direct IoU is reported at one operating point.** Confidence 0.25 and NMS IoU 0.70 were predeclared; a different threshold would give different coverage figures, and no sweep was run to find a flattering one.
- **`vest_loose` stands on one validation image.** Every figure for it carries high sampling uncertainty.
- **No latency benchmark.** The framework validation speed above is descriptive; the standardised detector-versus-segmenter study the project's question needs is a later phase.

## 24. Next decision

`PENDING_HUMAN_REVIEW`. `segmentation_baseline_status: S0_COMPLETE`, `final_segmenter: UNSELECTED_PENDING_REVIEW`.

S0 is **not** frozen as the final segmenter, and no S1 or alternative model, resolution or batch was trained. The next phase is human review of this result and the planning of a segmentation experiment protocol - which, like phase 7A, must be frozen before the experiments it would decide exist.

---

`S0_experiment_sha256` `1761007ab1fd3937a56a870618c11115307988f2031598980e8dfea0e218f7bc` · direct mask-IoU result `reports/segmentation_S0_mask_iou.json` SHA-256 `867fe4441a7097974ead9d2df788379c9f72a612b2a7a5852a23fa3c18a049ba`.
