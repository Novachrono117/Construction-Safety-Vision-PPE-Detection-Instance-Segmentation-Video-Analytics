# D0 - YOLO11n Detection Baseline - Phase 6B

Phase: 6B · Commit: `2a53c2bf413bf221936299a919554d28db8cccea` · Status: **D0_BASELINE_COMPLETE**

**D0 is a reference point, not the project's detector.** It answers one question - does a small pretrained YOLO11 detector learn this task under the frozen data protocol - and it was specified in full before it ran. Nothing here was tuned.

Claims are labelled `PREDECLARED PROTOCOL` (fixed before the run), `COMPUTED RESULT` (measured by this run), `OBSERVATION` (a reading of those numbers), `LIMITATION` and `FUTURE EXPERIMENT`.

## 1. Objective

`PREDECLARED PROTOCOL` What does a small pretrained YOLO11 detector achieve under this project's frozen data protocol, before any model-size, resolution or augmentation experiment? D0 is a reference point, not an attempt at a good score.

## 2. Frozen protocol

| Setting | Declared |
| --- | --- |
| `experiment_id` | D0 |
| Model | YOLO11n, pretrained `yolo11n.pt` |
| Image size | 640 |
| Epochs | 100 (max) |
| Batch | 16 |
| Seed | 42 |
| Optimizer policy | `auto` |
| Patience | 50 |
| Augmentation | `ULTRALYTICS_DEFAULT` |
| AMP / deterministic | True / True |

`PREDECLARED PROTOCOL` Baseline config SHA-256: `ebff2d461d877f327ba2e707451c24a7a3d0df432447317cf8820ef224ddc3c6`. Checkpoint rule: ULTRALYTICS_BEST_ON_VALIDATION_FITNESS - the `best.pt` Ultralytics writes according to its own predeclared validation fitness, computed on the frozen validation split only. No other checkpoint is inspected, compared or reported, and the holdout takes no part in the selection.

## 3. Dataset and split

| | train | validation | test |
| --- | --- | --- | --- |
| Images | 303 | 65 | **PROTECTED_NOT_ACCESSED** |
| Annotations | 1422 | 304 | - |

`PREDECLARED PROTOCOL` Only `train` updated model parameters. `validation` drove the framework's fitness, checkpoint selection and early stopping - which is what a validation split is for - and the single authoritative evaluation below.

| Fingerprint | Value |
| --- | --- |
| `split_assignment_sha256` | `a230869ff4cb45f53654d67357a27f2a0def6d8ba79f9fa4880f0fdda2f046cc` |
| `class_map_sha256` | `596dab5b5756e3d4253924f84bf46d37a87d80b6ec830da781a08a93cd823753` |
| `adapter_manifest_sha256` | `1300ff027402a8473c55b53cc9ce3c27297cf4ef05999dd088c2d3ad58eafefe` |
| `d0_experiment_sha256` | `cbd79fd2f2f70eb31ede61b813f991e973bb5d2f69c223a3826ee6aeadd0ffb3` |

## 4. Environment

| Component | Version |
| --- | --- |
| Python | 3.12.13 |
| torch | 2.11.0+cu128 |
| torchvision | 0.26.0+cu128 |
| ultralytics | 8.4.138 |
| CUDA runtime | 12.8 |
| GPU | NVIDIA GeForce RTX 5070 Laptop GPU (sm_120) |

## 5. Effective training configuration

`COMPUTED RESULT` The protocol declares `optimizer: auto`, so the declared policy and the settings that actually ran are **not the same thing**. Ultralytics chose the optimizer and its learning rate from the dataset size and epoch count, overriding the generic values in the configuration file. What follows is what the run actually used, read back from the framework's own resolved arguments.

| Resolved setting | Value |
| --- | --- |
| `optimizer` | AdamW |
| `lr0` | 0.001111 |
| `lrf` | 0.01 |
| `momentum` | 0.9 |
| `weight_decay` | 0.0005 |
| `warmup_epochs` | 3.0 |
| `warmup_momentum` | 0.8 |
| `warmup_bias_lr` | 0.1 |
| `cos_lr` | False |
| `epochs` | 100 |
| `batch` | 16 |
| `imgsz` | 640 |
| `amp` | True |
| `deterministic` | True |
| `seed` | 42 |
| `patience` | 50 |
| `workers` | 8 |
| `device` | 0 |
| `box` | 7.5 |
| `cls` | 0.5 |
| `dfl` | 1.5 |

`COMPUTED RESULT` **Declared** `optimizer: auto` -> **resolved** `AdamW`. Quoting the configuration file's `lr0: 0.01` as this run's learning rate would be wrong: the effective `lr0` was `0.001111`.

`COMPUTED RESULT` Augmentation values actually applied:

| Augmentation | Value |
| --- | --- |
| `hsv_h` | 0.015 |
| `hsv_s` | 0.7 |
| `hsv_v` | 0.4 |
| `degrees` | 0.0 |
| `translate` | 0.1 |
| `scale` | 0.5 |
| `shear` | 0.0 |
| `perspective` | 0.0 |
| `flipud` | 0.0 |
| `fliplr` | 0.5 |
| `bgr` | 0.0 |
| `mosaic` | 1.0 |
| `mixup` | 0.0 |
| `cutmix` | 0.0 |
| `copy_paste` | 0.0 |
| `erasing` | 0.4 |
| `auto_augment` | randaugment |
| `close_mosaic` | 10 |

`PREDECLARED PROTOCOL` These are **model training transformations**, not canonical preprocessing: the source images on disk are untouched and validation applies none of them. They were not tuned in this phase.

## 6. Training execution

| | |
| --- | --- |
| Termination | **COMPLETED_ALL_EPOCHS** |
| Epochs configured | 100 |
| Epochs completed | 100 |
| Early stopped | False (patience 50) |
| Resumed | False |
| Wall clock | 772.7 s |
| Batch / imgsz | 16 / 640 |

`PREDECLARED PROTOCOL` Exactly one training execution was performed. It was not stopped by hand, no hyperparameter was changed after seeing an epoch result, and no second run was launched to see whether the number would move.

## 7. Checkpoint selection

`PREDECLARED PROTOCOL` `ULTRALYTICS_BEST_ON_VALIDATION_FITNESS`, fixed before the run.

* Best epoch: **67** of 100 completed
* Best validation fitness: **0.481862**
* `best.pt` SHA-256 `30288fbc3abe60ce1713fdb430feb2e1a3716dcf4b52b80d26a9b2aea1ca5c34` (5486289 bytes)
* `last.pt` SHA-256 `d0d9f4f0bd80c3677f6b5a03fc3179c66c35c016edbc02f322c8557611e029bc` (5486289 bytes)

`COMPUTED RESULT` The best epoch was **recomputed independently** from `results.csv` using Ultralytics' own fitness definition (`0.1 * mAP@0.50 + 0.9 * mAP@0.50:0.95`) rather than read from the checkpoint being verified. No epoch was inspected and then chosen; only `best.pt` is the official D0 model.

`COMPUTED RESULT` Metric cross-check, so the headline number provably belongs to the
selected checkpoint rather than to the last epoch or an earlier run:

| Source | mAP@0.50:0.95 |
| --- | --- |
| Authoritative `best.pt` validation | 0.464429 |
| `results.csv` at best epoch 67 | 0.46206 |
| `results.csv` at last epoch 100 | 0.46269 |

`COMPUTED RESULT` Difference at the best epoch: **0.002369** (tolerance 0.02). The two are separate evaluations of the same weights under slightly different batching, so they are expected to agree closely rather than exactly.

## 8. Validation results

`COMPUTED RESULT` One evaluation of `best.pt` on the frozen validation split (65 images, 304 annotations) at the frozen image size. No threshold sweep, no alternative image size, no second checkpoint.

| Metric | Value |
| --- | --- |
| **mAP@0.50:0.95** (primary) | **0.464429** |
| mAP@0.50 | 0.619373 |
| precision | 0.85568 |
| recall | 0.539992 |

## 9. Per-class results

| Class | AP@0.50 | AP@0.50:0.95 | precision | recall |
| --- | --- | --- | --- | --- |
| `helmet_loose` | 0.879592 | 0.792559 | 0.898364 | 0.807018 |
| `helmet_on_head` | 0.735806 | 0.561699 | 0.887663 | 0.672636 |
| `person` | 0.68902 | 0.491618 | 0.792384 | 0.583942 |
| `vest_loose` (small sample) | 0.131486 | 0.041575 | 1.0 | 0.0 |
| `vest_on_body` | 0.660959 | 0.434691 | 0.699987 | 0.636364 |

A metric recorded as `NOT_EXPOSED_RELIABLY` was not dependably exposed by the installed framework and was **not** reconstructed by guesswork.

## 10. `vest_loose` sampling limitation

`LIMITATION` **HIGH_SAMPLING_UNCERTAINTY.** The frozen validation split contains **1 `vest_loose` image** carrying **8 instances**. Whatever number appears above for that class is one image's worth of evidence.

`LIMITATION` Do not conclude that D0 is generally good or bad at `vest_loose` from it. This limitation was predeclared in `configs/detection_baseline.yaml` **before** the run, so it is not an excuse built around an inconvenient result. Nothing was tuned against this metric, and the holdout - which holds 2 such images - was not consulted to resolve the uncertainty.

## 11. Training dynamics

`COMPUTED RESULT` The full per-epoch history is preserved unmodified in the run's `results.csv`: box, classification and DFL losses for train and validation, the metric curves and the learning-rate progression. The committed figures under `reports/figures/detection/D0/` are the framework's own metric plots - no curve was smoothed, truncated or redrawn.

## 12. Confusion matrix

`COMPUTED RESULT` Counts from the validation confusion matrix, rows **predicted** and columns **ground truth**, with `background` covering unmatched predictions and undetected objects.

| predicted \ true | `helmet_loose` | `helmet_on_head` | `person` | `vest_loose` | `vest_on_body` | `background` |
| --- | --- | --- | --- | --- | --- | --- |
| `helmet_loose` | 47 | 0 | 1 | 0 | 0 | 11 |
| `helmet_on_head` | 0 | 32 | 0 | 0 | 0 | 6 |
| `person` | 0 | 0 | 89 | 0 | 1 | 32 |
| `vest_loose` | 0 | 0 | 0 | 0 | 0 | 0 |
| `vest_on_body` | 1 | 0 | 0 | 0 | 37 | 22 |
| `background` | 9 | 15 | 47 | 8 | 17 | 0 |

`OBSERVATION` **Errors concentrate on the background row and column, not between classes.** 96 ground-truth objects were not detected and 71 predictions matched no object, against 3 class-to-class confusions in total. The model's difficulty at this operating point is finding objects, more than telling the classes apart once found - which is consistent with the global recall being well below the global precision.

`OBSERVATION` Undetected ground truth is led by `person` 47, `vest_on_body` 17, `helmet_on_head` 15.

`OBSERVATION` The largest class-to-class confusions are `helmet_loose` predicted where truth was `person` (1), `person` predicted where truth was `vest_on_body` (1), `vest_on_body` predicted where truth was `helmet_loose` (1).

`LIMITATION` These are counts on one 65-image validation split, and a single confusion of one or two instances is not a pattern. **No image-level qualitative error analysis was performed**: no validation image was opened to explain an individual failure. That is a later, deliberate stage.

## 13. Runtime and resources

| | |
| --- | --- |
| Training wall clock | 772.7 s |
| Epochs completed | 100 |
| GPU | NVIDIA GeForce RTX 5070 Laptop GPU |
| `FRAMEWORK_VALIDATION_SPEED` inference | 8.757 ms/image |
| `FRAMEWORK_VALIDATION_SPEED` loss | 0.001 ms/image |
| `FRAMEWORK_VALIDATION_SPEED` postprocess | 1.132 ms/image |
| `FRAMEWORK_VALIDATION_SPEED` preprocess | 2.844 ms/image |

`LIMITATION` These timings describe **this execution on this device** and nothing more. They are not the project's latency benchmark: a standardised operational measurement belongs to the later detector/segmenter comparison, and no tuning run was performed to improve them.

## 14. Reproducibility

`PREDECLARED PROTOCOL` seed 42, `deterministic: true`, the pinned software stack above, and every input fingerprinted. `d0_experiment_sha256` covers the protocol, the pretrained weights, the adapter dataset, the split, the critical resolved arguments and the selected checkpoint; it excludes timestamps, paths and machine identity, so the same experiment elsewhere fingerprints the same.

`LIMITATION` `deterministic: true` reduces run-to-run variance but does not eliminate it - some cuDNN and CUDA kernels remain non-deterministic. **D0 was run once, deliberately.** A second run purely to see whether the number moved was not performed: that would be measuring noise, and a reproducibility study, if wanted, is its own predeclared experiment.

## 15. Holdout compliance

`PREDECLARED PROTOCOL` `test` is `PROTECTED_NOT_ACCESSED`. It has no YOLO labels, no adapter directory and no key in the Ultralytics dataset descriptor, so it could not have been loaded even by accident. `CSVISION_ALLOW_TEST_SPLIT` was not set at any point, no test image or annotation was read, and no test metric exists. Model selection used validation only.

## 16. Baseline interpretation

`OBSERVATION` **The detector learns the task.** A primary metric of 0.464429 with mAP@0.50 of 0.619373 is far above the zero a non-learning model would produce, so the data pipeline, the adapter and the training stack are working end to end. That is what a baseline is for.

`OBSERVATION` Among the classes with meaningful validation support, `helmet_loose` scores highest (AP@0.50:0.95 0.792559) and `vest_on_body` lowest (0.434691). This is a description of one validation split of 65 images, not a claim about the classes in general.

`LIMITATION` `vest_loose` is excluded from that comparison on purpose: with one validation image it cannot be ranked against classes measured on dozens.

`LIMITATION` **D0 is not the project's detector and this is not a final result.** It is one small pretrained model, trained once, on 303 images, evaluated on 65. No comparison against another model has been run, so nothing here says whether a larger model, a different image size or different augmentation would do better.

`LIMITATION` **No claim of generalisation.** These numbers describe this dataset's validation split. They say nothing about arbitrary construction sites, other cameras, other countries' PPE, or conditions absent from 433 images.

`LIMITATION` **The holdout is still unmeasured**, by design. D0's validation performance is not an estimate of its test performance, and the gap between them is unknown until phase 11.

## 17. Questions for phase 7

`FUTURE EXPERIMENT` Hypotheses worth testing, recorded here **without acting on them**. Phase 7 defines a controlled experiment matrix before any of these run, so that each is a comparison rather than a reaction to D0's numbers:

1. **Model capacity.** Does YOLO11s change the primary metric materially, and is any change larger than run-to-run variance?
2. **Input resolution.** PPE objects are often small at distance; does a larger `imgsz` help, and at what cost in time and memory?
3. **Epoch budget.** Did D0 stop for the right reason - is there evidence of underfitting or of overfitting in the loss curves?
4. **Class imbalance.** `vest_loose` has 45 instances against `person`'s 914. Does any imbalance-aware treatment help, and can it even be evaluated given the validation support?

`PREDECLARED PROTOCOL` None of these was tried after seeing D0's results, and D0 was not modified in response to them. Phase 7 has not been started.
