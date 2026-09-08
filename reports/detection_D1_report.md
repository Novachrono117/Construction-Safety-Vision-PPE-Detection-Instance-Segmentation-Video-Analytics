# D1 - YOLO11s Detection Capacity Experiment - Phase 7B

Phase: 7B · Commit: `3a23042cfbe52047505f5082ea942342a2ec456a` · Status: **DETECTION_EXPERIMENT_COMPLETE**

**D1 is one controlled experiment, not a verdict.** It answers a single question against D0, and the Phase 7 winner cannot be declared here because D2 has not been executed. Every number below is a **validation** number.

Claims are labelled `PREDECLARED_PROTOCOL` (fixed before the run), `COMPUTED_RESULT` (measured by this run), `CONTROLLED_COMPARISON` (a difference against the reference under the frozen contract), `OBSERVATION` (a reading of those numbers), `LIMITATION` and `PENDING_EXPERIMENT`.

## 1. Experimental question

`PREDECLARED_PROTOCOL` Does increasing model capacity improve detection under the same input resolution and training protocol?

**Hypothesis, recorded before the run.** HYPOTHESIS, untested. A larger backbone may fit this 303-image training set better, or may overfit it; 303 images is small enough that neither outcome is the obvious one. No prediction is recorded as expected, and the result is not interpreted as confirming a preference either way.

## 2. Controlled-variable contract

`PREDECLARED_PROTOCOL` Intentional variable `MODEL_CAPACITY`. D1 carries no protocol of its own: it inherits `configs/detection_baseline.yaml` and applies a declared override set, and the run refuses to start if the resolved protocol differs from the reference anywhere it did not declare.

| Field | Kind | Value |
| --- | --- | --- |
| `model` | intentional | `YOLO11s` |
| `weight_identifier` | consequential | `yolo11s.pt` |

`CONTROLLED_COMPARISON` Observed differences against D0: `model`, `weight_identifier`. Contract satisfied: **yes**.

Identical to the reference by inheritance: split, labels, class map, epochs, batch, image size, optimizer policy, patience, seed, `deterministic`, augmentation policy, checkpoint-selection rule and validation protocol.

## 3. Frozen Phase 7 comparison policy

| Item | Value |
| --- | --- |
| Primary selection metric | `supported_macro_map50_95` |
| Official all-class metric | `mAP@0.50:0.95` |
| Support rule | >= 5 positive validation images **and** >= 20 instances |
| Selection classes | `helmet_loose`, `helmet_on_head`, `person`, `vest_on_body` |
| Descriptive classes | `vest_loose` |
| Practical-equivalence margin | 0.005 |
| Policy SHA-256 | `ab4e44a1cd502db73a906c81095b3a0eeb9eee191b5cb5df1c2013fb40b5b330` |
| Experiment matrix SHA-256 | `e6acad129dc6b38c3a51b3388fd7078f07790e4801f7ee33f49afd245e49a998` |

`PREDECLARED_PROTOCOL` The support rule and the margin were frozen in phase 7A, before D1 existed. Neither was touched by this phase.

## 4. Input and runtime provenance

| Fingerprint | Value |
| --- | --- |
| `split_assignment_sha256` | `a230869ff4cb45f53654d67357a27f2a0def6d8ba79f9fa4880f0fdda2f046cc` |
| `class_map_sha256` | `596dab5b5756e3d4253924f84bf46d37a87d80b6ec830da781a08a93cd823753` |
| `adapter_manifest_sha256` | `1300ff027402a8473c55b53cc9ce3c27297cf4ef05999dd088c2d3ad58eafefe` |
| `resolved_config_sha256` | `69db53b64c1342fe40240e50b7990db6490be0ae55bbf04e4cb07ba6e1fbe96c` |
| `D1_experiment_sha256` | `0589d4c0dabedfe0e6c72da3d248883d750347ce60b122b71aaa44e939158503` |

| Pretrained weights | Value |
| --- | --- |
| Identifier | `yolo11s.pt` |
| SHA-256 | `85a76fe86dd8afe384648546b56a7a78580c7cb7b404fc595f97969322d502d5` |
| Size | 19313732 bytes |
| Source | ULTRALYTICS_ASSET_DOWNLOAD (ultralytics 8.4.138) |
| Committed | False |

`PREDECLARED_PROTOCOL` Fingerprinted rather than trusted by name: two files called `yolo11s.pt` are not necessarily the same bytes. The binary is not committed.

| Component | Version |
| --- | --- |
| Python | 3.12.13 |
| torch | 2.11.0+cu128 |
| torchvision | 0.26.0+cu128 |
| ultralytics | 8.4.138 |
| CUDA runtime | 12.8 |
| GPU | NVIDIA GeForce RTX 5070 Laptop GPU (sm_120) |

## 5. Effective training configuration

`COMPUTED RESULT` The protocol declares `optimizer: auto`, so the declared policy and the settings that actually ran are **not the same thing**. What follows is what the run used.

| Setting | Declared | Effective |
| --- | --- | --- |
| Optimizer | `auto` | **AdamW** |
| `lr0` | 0.01 | 0.001111 |
| Momentum | 0.937 | 0.9 |
| `imgsz` | 640 | 640 |
| Batch | 16 | 16 |
| Epochs | 100 | 100 |
| Patience | 50 | 50 |
| Seed | 42 | 42 |
| `deterministic` | True | True |
| AMP | True | True |
| `weight_decay` | 0.0005 | 0.0005 |
| `warmup_epochs` | 3.0 | 3.0 |
| `close_mosaic` | 10 | 10 |
| `cos_lr` | False | False |
| Workers | 8 | 8 |

`COMPUTED_RESULT` **Optimizer evidence: `FRAMEWORK_LOG_LINE_DIRECT_CAPTURE`.** Read directly from the framework's own log line at the moment it built the optimizer: `optimizer: AdamW(lr=0.001111, momentum=0.9)`. Nothing was inferred.

`PREDECLARED_PROTOCOL` Effective augmentation values, as resolved by the framework:

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
| `mosaic` | 1.0 |
| `mixup` | 0.0 |
| `cutmix` | 0.0 |
| `copy_paste` | 0.0 |
| `erasing` | 0.4 |
| `auto_augment` | randaugment |

`CONTROLLED_COMPARISON` These are the frozen `ULTRALYTICS_DEFAULT` policy for the pinned version, inherited unchanged from the reference. No augmentation was tuned.

## 6. Training execution

| | |
| --- | --- |
| Termination | `COMPLETED_ALL_EPOCHS` |
| Epochs configured | 100 |
| Epochs completed | 100 |
| Early stopped | False |
| Resumed | False |
| Duration | 1187.8 s |
| Peak GPU allocated | 4359202816 bytes |
| Peak GPU reserved | 4544528384 bytes |

| Model complexity | Value |
| --- | --- |
| Parameters | 9458752 |
| GFLOPs | 21.836 |
| Source | ULTRALYTICS_TORCH_UTILS |

`OBSERVATION` Parameter count and FLOPs describe what the intentional variable actually changed. They are descriptive facts at this stage and are **not** used to choose between experiments; that would be an efficiency comparison, which is not authorised here.

## 7. Checkpoint selection

`PREDECLARED_PROTOCOL` ULTRALYTICS_BEST_ON_VALIDATION_FITNESS - the `best.pt` Ultralytics writes according to its own predeclared validation fitness, computed on the frozen validation split only. No other checkpoint is inspected, compared or reported, and the holdout takes no part in the selection.

| | |
| --- | --- |
| Best epoch | 73 |
| Best validation fitness | 0.485745 |
| `best.pt` SHA-256 | `a933196ee15bc369fa1ea682986d892f06b52ebd8086ce027ae4604856640717` |
| `best.pt` size | 19193809 bytes |
| `last.pt` SHA-256 | `792a03b16a485a0f29d9052eb11adbb94de81f92fcf58c2033c0da839ce715a1` |
| `last.pt` size | 19193809 bytes |

`COMPUTED_RESULT` The published headline metric was cross-checked against the epoch history recomputed independently from `results.csv`: authoritative 0.471114, history at the selected epoch 0.47104, delta 7.4e-05 against a tolerance of 0.02. So the reported number provably belongs to `best.pt` rather than to the last epoch.

`LIMITATION` Neither checkpoint is committed. Both are referenced by digest.

## 8. Validation metrics

`PREDECLARED_PROTOCOL` One evaluation of `best.pt` on the frozen validation split at imgsz 640, batch 16, conf 0.001, IoU 0.7 - the same validation protocol the reference used. No threshold sweep, no IoU sweep, no alternative image size, no `last.pt` comparison.

| Global metric | D1 | D0 | Delta |
| --- | --- | --- | --- |
| `mAP@0.50:0.95` | 0.471114 | 0.464429 | +0.006685 |
| `mAP@0.50` | 0.618243 | 0.619373 | -0.001130 |
| `precision` | 0.85964 | 0.85568 | +0.003960 |
| `recall` | 0.527798 | 0.539992 | -0.012194 |

## 9. Supported-class primary metric

`COMPUTED_RESULT` `supported_macro_map50_95` is the unweighted mean of per-class `AP@0.50:0.95` over the classes the frozen support rule admits: `helmet_loose`, `helmet_on_head`, `person`, `vest_on_body`.

| | Value |
| --- | --- |
| D1 | **0.560017** |
| D1 exact | 0.560017 |
| D0 | 0.570142 |
| Delta | **-0.010125** |

## 10. Comparison with the reference

`CONTROLLED_COMPARISON` Margin status: **`BELOW_D0`** at a frozen margin of 0.005.

`LIMITATION` The margin is an engineering decision threshold, not a significance test. `deterministic: true` reduces run-to-run variance without eliminating it, and no experiment is repeated, so the size of that variance on this setup is **UNKNOWN**. A difference near the margin is not an ordering.

`PENDING_EXPERIMENT` The frozen Phase 7 selection logic is **not applied here**. It runs once every declared candidate has a result, and D2 has not been executed. D1 is therefore not the Phase 7 winner, and no final detector is declared by this phase.

## 11. Official all-class reporting metric

`COMPUTED_RESULT` The five-class `mAP@0.50:0.95` is 0.471114 against the reference's 0.464429, a delta of +0.006685. It remains `OFFICIAL_ALL_CLASS_REPORTING_METRIC` and is reported whichever direction it moved.

`LIMITATION` The all-class figure includes the descriptive class, so part of any movement in it can come from a class standing on very little validation evidence. That is exactly why it is not the selection metric - and exactly why it is not suppressed either.

## 11b. When the two metrics disagree

`COMPUTED_RESULT` **The two metrics moved in opposite directions.** `supported_macro_map50_95` fell by 0.010125 while the all-class `mAP@0.50:0.95` rose by 0.006685. That is not a contradiction and neither number is wrong; they average different class sets, and this run happens to sit where the choice of set flips the sign. Since it would be easy to quote whichever one flatters the experiment, the arithmetic is set out in full.

| Class | `AP@0.50:0.95` delta | In selection metric |
| --- | --- | --- |
| `helmet_loose` | +0.005802 | yes |
| `helmet_on_head` | +0.032401 | yes |
| `person` | +0.004585 | yes |
| `vest_loose` | +0.073925 | no |
| `vest_on_body` | -0.083287 | yes |

`COMPUTED_RESULT` Both metrics are unweighted means over the same per-class values, so each class's contribution is its delta divided by the number of classes averaged. The descriptive class `vest_loose` contributes +0.014785 to the all-class mean. Remove that single contribution and the all-class delta becomes -0.008100 - the same direction as the selection metric. **The entire sign change in the official metric comes from the class the support rule set aside.**

`CONTROLLED_COMPARISON` The frozen policy governs: `supported_macro_map50_95` decides the ranking, and its verdict is `BELOW_D0`. The all-class improvement **must not** be quoted as evidence that this experiment beat the reference. That is precisely the metric-shopping the phase 7A policy was written to prevent, and the policy was frozen before this result existed - which is the only reason this paragraph is a rule being applied rather than an excuse being made.

`LIMITATION` Equally, the all-class figure is not suppressed and is not being called wrong. It is the official reporting metric and it improved. What it cannot do is order two models when its movement is driven by a class standing on one validation image.

## 12. Per-class metrics

| Class | `AP@0.50` | `AP@0.50:0.95` | precision | recall | D0 `AP@0.50:0.95` | In selection metric |
| --- | --- | --- | --- | --- | --- | --- |
| `helmet_loose` | 0.862006 | 0.798361 | 0.907613 | 0.824561 | 0.792559 | yes |
| `helmet_on_head` | 0.734993 | 0.5941 | 0.872208 | 0.659574 | 0.561699 | yes |
| `person` | 0.706444 | 0.496203 | 0.860498 | 0.630357 | 0.491618 | yes |
| `vest_loose` | 0.165 | 0.1155 | 1.0 | 0.0 | 0.041575 | no |
| `vest_on_body` | 0.622773 | 0.351404 | 0.65788 | 0.524496 | 0.434691 | yes |

## 13. The descriptive class

`LIMITATION` `vest_loose` holds 1 positive validation image(s) and 8 instances, so the frozen rule classified it `DESCRIPTIVE_HIGH_UNCERTAINTY`.

Every class listed here failed the frozen support rule on the validation split, so its AP is reported in full but excluded from the selection metric. It must not be tuned for, must not decide between models, and must be quoted with an explicit small-sample caveat. The holdout remains protected and cannot be consulted to resolve the uncertainty.

It is reported in full in section 12 and was **not** used in the selection metric, not tuned for, and not allowed to move the comparison in either direction.

## 14. Recall and precision diagnostic

`COMPUTED_RESULT` Precision 0.85964 against recall 0.527798; the reference reported 0.85568 against 0.539992. Recall delta -0.012194, precision delta +0.003960.

`PREDECLARED_PROTOCOL` Recall is a **diagnostic** here. It helps explain why the selection metric moved; it is not promoted to a selection objective and no recall-weighted composite is introduced.

`LIMITATION` These two numbers deserve more caution than the AP figures. Ultralytics reports a single precision and recall taken at the operating point that maximises F1, not at a fixed confidence threshold, so a large move in one of them can partly reflect *where that point landed* rather than a uniform change in behaviour. They are read as a hint about the precision/recall balance, never as a threshold-independent property of the model - and no threshold was tuned in either direction.

## 15. Training dynamics

`OBSERVATION` The run terminated as `COMPLETED_ALL_EPOCHS` after 100 of 100 configured epochs, with the predeclared rule selecting epoch 73.

`LIMITATION` Curves are read at the level of aggregate metrics only. No individual validation image was opened, and no qualitative source-image analysis was performed - that is a deliberate later phase.

## 16. Confusion matrix

`COMPUTED_RESULT` Rows predicted, columns ground truth.

| predicted \ truth | `helmet_loose` | `helmet_on_head` | `person` | `vest_loose` | `vest_on_body` | `background` |
| --- | --- | --- | --- | --- | --- | --- |
| `helmet_loose` | 48 | 1 | 0 | 1 | 0 | 9 |
| `helmet_on_head` | 0 | 31 | 0 | 0 | 0 | 6 |
| `person` | 0 | 0 | 95 | 0 | 2 | 26 |
| `vest_loose` | 0 | 0 | 0 | 0 | 0 | 0 |
| `vest_on_body` | 0 | 0 | 0 | 0 | 33 | 21 |
| `background` | 9 | 15 | 42 | 7 | 20 | 0 |

`OBSERVATION` Structural reading only: 62 predictions matched no ground-truth object, 93 ground-truth objects went undetected, and 4 objects were detected but given the wrong class.

`LIMITATION` These are counts. Explaining any individual error would require opening validation images, which this phase does not do.

## 17. Runtime and resource observations

| Validation stage | ms/image |
| --- | --- |
| inference | 5.181 |
| loss | 0.022 |
| postprocess | 1.391 |
| preprocess | 0.93 |

`OBSERVATION` `FRAMEWORK_VALIDATION_SPEED`, as the framework reported it during the authoritative validation. No separate benchmark was run, the numbers include this machine's incidental load, and they are **descriptive only** - runtime efficiency is not used to choose between experiments at this stage.

## 18. Holdout compliance

`PREDECLARED_PROTOCOL` `PROTECTED_NOT_ACCESSED` - the holdout was not adapted, loaded, evaluated or measured in this experiment. It has no labels, no dataset entry and no result. `CSVISION_ALLOW_TEST_SPLIT` was not set at any point, the adapter dataset descriptor carries no holdout key, and no holdout image, label, count, prediction or metric was produced. Every number in this report comes from the 65-image validation split.

## 19. Interpretation

`CONTROLLED_COMPARISON` Changing `model` 'YOLO11n' -> 'YOLO11s', `weight_identifier` 'yolo11n.pt' -> 'yolo11s.pt' and nothing else - everything else held fixed by inheritance from D0 - moved `supported_macro_map50_95` by -0.010125, which the frozen margin classifies as `BELOW_D0`.

`OBSERVATION` The aggregate hides the shape of the change. Of the 4 classes in the selection metric, 3 improved and 1 did not.

- improved: `helmet_on_head` +0.032401, `helmet_loose` +0.005802, `person` +0.004585 (total +0.042788)
- declined: `vest_on_body` -0.083287 (total -0.083287)

`OBSERVATION` So this is not a uniformly worse model. A single class, `vest_on_body`, fell by 0.083287 - more than the other improvements combined - and that one class is what carries the selection metric below the reference. Its recall moved -0.111868, from 0.636364 to 0.524496, so the class lost detections rather than localisation quality.

`LIMITATION` Why `vest_on_body` behaved that way is **UNKNOWN**. A plausible story is easy to construct and none is tested here: this is one run, the movement could be run-to-run variance, and diagnosing it would need either a repeated run or the image-level error analysis this phase deliberately does not perform. It is recorded as an open question, not explained.

`LIMITATION` This is one run of each configuration on a 65-image validation split. It establishes what these two runs scored; it does not establish that `MODEL_CAPACITY` causes the difference in general, and it cannot separate a real effect from run-to-run variance, because neither experiment was repeated.

`LIMITATION` Nothing here was tuned, and nothing may be tuned in response to it. The phase 7A policy forbids trying another capacity, another resolution, another optimizer or another augmentation setting because of what this result shows.

## 20. Final Phase 7 comparison still pending

`PENDING_EXPERIMENT` Every declared candidate now has a result, so the frozen selection logic *could* be evaluated - and this phase still does not apply it. Freezing the project's detector is a reviewed decision, so `reports/detection_experiment_results.json` records the computed case as a `policy_case_candidate` marked `advisory_only`, with `final_selected_detector: UNSELECTED_PENDING_REVIEW`.

Either way, and regardless of what the numbers above show:

- no experiment is called the Phase 7 winner, D1 included;
- D0 remains the preferred detector **by default, not by comparison**;
- no efficiency or latency benchmark is run to break any tie;
- no further experiment is authorised - not another resolution, another capacity, a combination of the two, or any tuning prompted by this result.

Committed metric-only figures: `reports/figures/detection/D1/`.
