# D2 - YOLO11s Detection Capacity Experiment - Phase 7B

Phase: 7C · Commit: `2f960ed67aa198b6102e61b5676e4c4b74a942ec` · Status: **DETECTION_EXPERIMENT_COMPLETE**

**D2 is one controlled experiment, not a verdict.** It answers a single question against D0, and the Phase 7 winner cannot be declared here because D2 has not been executed. Every number below is a **validation** number.

Claims are labelled `PREDECLARED_PROTOCOL` (fixed before the run), `COMPUTED_RESULT` (measured by this run), `CONTROLLED_COMPARISON` (a difference against the reference under the frozen contract), `OBSERVATION` (a reading of those numbers), `LIMITATION` and `PENDING_EXPERIMENT`.

## 1. Experimental question

`PREDECLARED_PROTOCOL` Does moderately increasing input resolution improve detection, especially for small PPE objects, while model capacity remains fixed?

**Hypothesis, recorded before the run.** HYPOTHESIS, untested. The motivation is a measurement, not an intuition: phase 4A's `reports/eda_source.json` records, over all 2031 canonical annotations and at a small-object threshold of 0.01 relative box area, helmet_on_head 52.87% small (166/314), helmet_loose 45.29% (178/393), person 27.57% (252/914), vest_on_body 25.21% (92/365) and vest_loose 4.44% (2/45); the median relative box area over all annotations is 0.0301. Raising the network input from 640 to 768 preserves roughly 1.44x the pixels per object, which MAY help the smallest instances survive the downsampling stages - or may not, since it also changes the scale distribution the pretrained backbone sees. Whether it helps is exactly what the experiment measures, and no direction is recorded as expected. Two honest caveats on the evidence: those fractions are measured over the whole canonical population rather than per split, and the 0.01 threshold is this project's own convention, not the COCO small-object definition. Nothing about the holdout motivates this, and nothing about the holdout will evaluate it.

## 2. Controlled-variable contract

`PREDECLARED_PROTOCOL` Intentional variable `INPUT_RESOLUTION`. D2 carries no protocol of its own: it inherits `configs/detection_baseline.yaml` and applies a declared override set, and the run refuses to start if the resolved protocol differs from the reference anywhere it did not declare.

| Field | Kind | Value |
| --- | --- | --- |
| `training.imgsz` | intentional | `768` |

`CONTROLLED_COMPARISON` Observed differences against D0: `training.imgsz`. Contract satisfied: **yes**.

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

`PREDECLARED_PROTOCOL` The support rule and the margin were frozen in phase 7A, before D2 existed. Neither was touched by this phase.

## 4. Input and runtime provenance

| Fingerprint | Value |
| --- | --- |
| `split_assignment_sha256` | `a230869ff4cb45f53654d67357a27f2a0def6d8ba79f9fa4880f0fdda2f046cc` |
| `class_map_sha256` | `596dab5b5756e3d4253924f84bf46d37a87d80b6ec830da781a08a93cd823753` |
| `adapter_manifest_sha256` | `1300ff027402a8473c55b53cc9ce3c27297cf4ef05999dd088c2d3ad58eafefe` |
| `resolved_config_sha256` | `3b75cdddd7bf2bcdc9643ba0cf1f143c016e39a4aab4b201b5eaded9ad5e2854` |
| `D2_experiment_sha256` | `8417f64f3c01c8994291f1fb58837059a9db6a814fd5dbbb955a44dcc1979685` |

| Pretrained weights | Value |
| --- | --- |
| Identifier | `yolo11n.pt` |
| SHA-256 | `0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1` |
| Size | 5613764 bytes |
| Source | ULTRALYTICS_ASSET_DOWNLOAD (ultralytics 8.4.138) |
| Committed | False |

`PREDECLARED_PROTOCOL` Fingerprinted rather than trusted by name: two files called `yolo11n.pt` are not necessarily the same bytes. The binary is not committed.

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
| `imgsz` | 768 | 768 |
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
| Duration | 911.8 s |
| Peak GPU allocated | 3649816064 bytes |
| Peak GPU reserved | 4431282176 bytes |

| Model complexity | Value |
| --- | --- |
| Parameters | 2624080 |
| GFLOPs | 6.673 |
| Source | ULTRALYTICS_TORCH_UTILS |

`OBSERVATION` Parameter count and FLOPs describe what the intentional variable actually changed. They are descriptive facts at this stage and are **not** used to choose between experiments; that would be an efficiency comparison, which is not authorised here.

## 7. Checkpoint selection

`PREDECLARED_PROTOCOL` ULTRALYTICS_BEST_ON_VALIDATION_FITNESS - the `best.pt` Ultralytics writes according to its own predeclared validation fitness, computed on the frozen validation split only. No other checkpoint is inspected, compared or reported, and the holdout takes no part in the selection.

| | |
| --- | --- |
| Best epoch | 90 |
| Best validation fitness | 0.508697 |
| `best.pt` SHA-256 | `0466f872a9de22898c70d8834cf1fcfb3d77f6c9fb27e81ee6248d3d19cbc206` |
| `best.pt` size | 5502289 bytes |
| `last.pt` SHA-256 | `8dcb7c370408eacda34b3ce9f50c9265bb66c6c1482ecf5ce0c2d8fb2cc8020c` |
| `last.pt` size | 5502289 bytes |

`COMPUTED_RESULT` The published headline metric was cross-checked against the epoch history recomputed independently from `results.csv`: authoritative 0.490386, history at the selected epoch 0.49333, delta 0.002944 against a tolerance of 0.02. So the reported number provably belongs to `best.pt` rather than to the last epoch.

`LIMITATION` Neither checkpoint is committed. Both are referenced by digest.

## 8. Validation metrics

`PREDECLARED_PROTOCOL` One evaluation of `best.pt` on the frozen validation split at imgsz 768, batch 16, conf 0.001, IoU 0.7 - the same validation protocol the reference used. No threshold sweep, no IoU sweep, no alternative image size, no `last.pt` comparison.

| Global metric | D2 | D0 | Delta |
| --- | --- | --- | --- |
| `mAP@0.50:0.95` | 0.490386 | 0.464429 | +0.025957 |
| `mAP@0.50` | 0.646304 | 0.619373 | +0.026931 |
| `precision` | 0.926331 | 0.85568 | +0.070651 |
| `recall` | 0.516549 | 0.539992 | -0.023443 |

## 9. Supported-class primary metric

`COMPUTED_RESULT` `supported_macro_map50_95` is the unweighted mean of per-class `AP@0.50:0.95` over the classes the frozen support rule admits: `helmet_loose`, `helmet_on_head`, `person`, `vest_on_body`.

| | Value |
| --- | --- |
| D2 | **0.594018** |
| D2 exact | 0.594018 |
| D0 | 0.570142 |
| Delta | **+0.023876** |

## 10. Comparison with the reference

`CONTROLLED_COMPARISON` Margin status: **`IMPROVES_D0_BEYOND_MARGIN`** at a frozen margin of 0.005.

`LIMITATION` The margin is an engineering decision threshold, not a significance test. `deterministic: true` reduces run-to-run variance without eliminating it, and no experiment is repeated, so the size of that variance on this setup is **UNKNOWN**. A difference near the margin is not an ordering.

`PENDING_EXPERIMENT` The frozen Phase 7 selection logic is **not applied here**. It runs once every declared candidate has a result, and D2 has not been executed. D2 is therefore not the Phase 7 winner, and no final detector is declared by this phase.

## 11. Official all-class reporting metric

`COMPUTED_RESULT` The five-class `mAP@0.50:0.95` is 0.490386 against the reference's 0.464429, a delta of +0.025957. It remains `OFFICIAL_ALL_CLASS_REPORTING_METRIC` and is reported whichever direction it moved.

`LIMITATION` The all-class figure includes the descriptive class, so part of any movement in it can come from a class standing on very little validation evidence. That is exactly why it is not the selection metric - and exactly why it is not suppressed either.

## 11b. When the two metrics disagree

`OBSERVATION` The selection metric and the all-class metric moved in the same direction here, so nothing turns on which of the two is read. The section is kept because that agreement is a fact about this run, not a property of the metrics.

## 11c. Comparison with the other candidate

`CONTROLLED_COMPARISON` Recorded for completeness. The frozen policy ranks candidates **against D0**, not against each other, so what follows is a difference and not a verdict - and it orders nothing.

| Against | intentional variable | `supported_macro_map50_95` delta | `mAP@0.50:0.95` delta | recall delta |
| --- | --- | --- | --- | --- |
| D1 | `MODEL_CAPACITY` | +0.034001 | +0.019272 | -0.011249 |

`LIMITATION` These two candidates differ from each other in **two** things at once - each varies a different field relative to the reference - so no difference between them is attributable to either variable. Only each candidate's comparison with the reference is a controlled one.

## 12. Per-class metrics

| Class | `AP@0.50` | `AP@0.50:0.95` | precision | recall | D0 `AP@0.50:0.95` | In selection metric |
| --- | --- | --- | --- | --- | --- | --- |
| `helmet_loose` | 0.918909 | 0.814493 | 0.967942 | 0.719298 | 0.792559 | yes |
| `helmet_on_head` | 0.767135 | 0.60234 | 0.969884 | 0.685326 | 0.561699 | yes |
| `person` | 0.716256 | 0.468534 | 0.881327 | 0.596305 | 0.491618 | yes |
| `vest_loose` | 0.133571 | 0.075857 | 1.0 | 0.0 | 0.041575 | no |
| `vest_on_body` | 0.695648 | 0.490705 | 0.812503 | 0.581818 | 0.434691 | yes |

## 13. The descriptive class

`LIMITATION` `vest_loose` holds 1 positive validation image(s) and 8 instances, so the frozen rule classified it `DESCRIPTIVE_HIGH_UNCERTAINTY`.

Every class listed here failed the frozen support rule on the validation split, so its AP is reported in full but excluded from the selection metric. It must not be tuned for, must not decide between models, and must be quoted with an explicit small-sample caveat. The holdout remains protected and cannot be consulted to resolve the uncertainty.

It is reported in full in section 12 and was **not** used in the selection metric, not tuned for, and not allowed to move the comparison in either direction.

## 14. Recall and precision diagnostic

`COMPUTED_RESULT` Precision 0.926331 against recall 0.516549; the reference reported 0.85568 against 0.539992. Recall delta -0.023443, precision delta +0.070651.

`PREDECLARED_PROTOCOL` Recall is a **diagnostic** here. It helps explain why the selection metric moved; it is not promoted to a selection objective and no recall-weighted composite is introduced.

`LIMITATION` These two numbers deserve more caution than the AP figures. Ultralytics reports a single precision and recall taken at the operating point that maximises F1, not at a fixed confidence threshold, so a large move in one of them can partly reflect *where that point landed* rather than a uniform change in behaviour. They are read as a hint about the precision/recall balance, never as a threshold-independent property of the model - and no threshold was tuned in either direction.

## 14b. The small-object hypothesis

`PREDECLARED_PROTOCOL` This experiment was motivated by a measurement already in the repository, not by intuition: phase 4A's `reports/eda_source.json` records, over all 2031 canonical annotations at a small-object threshold of 0.01 relative box area, helmet_on_head 52.87% small, helmet_loose 45.29%, person 27.57%, vest_on_body 25.21% and vest_loose 4.44%, with a median relative box area of 0.0301. The hypothesis was that raising the input from 640 to 768 preserves more spatial information per object and *may* help the smallest instances.

`LIMITATION` No new EDA was performed for this section, and none may be: the figures above are the frozen source measurement, computed over the whole canonical population rather than per split, and the 0.01 threshold is this project's own convention rather than the COCO small-object definition.

`OBSERVATION` What the result shows, ordered by each class's small-object fraction in that frozen measurement (most small first):

| Class | small fraction (frozen EDA) | `AP@0.50:0.95` delta | recall delta |
| --- | --- | --- | --- |
| `helmet_on_head` | 52.87% | +0.040641 | +0.012690 |
| `helmet_loose` | 45.29% | +0.021934 | -0.087720 |
| `person` | 27.57% | -0.023084 | +0.012363 |
| `vest_on_body` | 25.21% | +0.056014 | -0.054546 |

`OBSERVATION` **The result does not track the small-object fraction.** Ranking the 4 supported classes by that fraction and by their AP change gives a rank correlation of -0.20. The largest gain went to `vest_on_body` (25.21% small, +0.056014) - the *least* small-object-heavy of them - and the only class to decline was `person` (27.57% small, -0.023084).

`OBSERVATION` So on this evidence the outcome is **NOT consistent** with the specific story the hypothesis told - that the benefit would concentrate in the classes holding the most small objects. Resolution did improve the selection metric beyond the margin, which is the controlled claim; *why* it improved is not explained by the small-object account as stated. Both halves of that are recorded because reporting only the first would make a predeclared hypothesis look confirmed by a result that does not support its mechanism.

`LIMITATION` A rank correlation over 4 points establishes nothing on its own - it is a compact way of stating the ordering, not a test. It is reported to stop the two agreeing rows being read as confirmation while the two disagreeing rows go unmentioned.

`LIMITATION` Whether this table is consistent or inconsistent with the hypothesis is a **reading of four numbers from one run**, and it is not evidence about causation. The classes differ in more than their object-size distribution, the small-object fractions are population-level rather than per-split, and a monotonic relationship between that fraction and the AP change would be suggestive at best. Establishing that resolution helps small objects would need a size-stratified evaluation, which this phase does not perform. The controlled claim available here is narrower and is stated in section 10: what changing resolution alone did to the selection metric.

## 15. Training dynamics

`OBSERVATION` The run terminated as `COMPLETED_ALL_EPOCHS` after 100 of 100 configured epochs, with the predeclared rule selecting epoch 90.

`LIMITATION` Curves are read at the level of aggregate metrics only. No individual validation image was opened, and no qualitative source-image analysis was performed - that is a deliberate later phase.

## 16. Confusion matrix

`COMPUTED_RESULT` Rows predicted, columns ground truth.

| predicted \ truth | `helmet_loose` | `helmet_on_head` | `person` | `vest_loose` | `vest_on_body` | `background` |
| --- | --- | --- | --- | --- | --- | --- |
| `helmet_loose` | 44 | 0 | 0 | 0 | 0 | 3 |
| `helmet_on_head` | 1 | 34 | 0 | 0 | 0 | 4 |
| `person` | 0 | 0 | 93 | 0 | 1 | 32 |
| `vest_loose` | 0 | 0 | 0 | 0 | 0 | 0 |
| `vest_on_body` | 1 | 0 | 0 | 0 | 34 | 20 |
| `background` | 11 | 13 | 44 | 8 | 20 | 0 |

`OBSERVATION` Structural reading only: 59 predictions matched no ground-truth object, 96 ground-truth objects went undetected, and 3 objects were detected but given the wrong class.

`LIMITATION` These are counts. Explaining any individual error would require opening validation images, which this phase does not do.

## 17. Runtime and resource observations

| Validation stage | ms/image |
| --- | --- |
| inference | 6.34 |
| loss | 0.008 |
| postprocess | 1.88 |
| preprocess | 1.75 |

`OBSERVATION` `FRAMEWORK_VALIDATION_SPEED`, as the framework reported it during the authoritative validation. No separate benchmark was run, the numbers include this machine's incidental load, and they are **descriptive only** - runtime efficiency is not used to choose between experiments at this stage.

## 18. Holdout compliance

`PREDECLARED_PROTOCOL` `PROTECTED_NOT_ACCESSED` - the holdout was not adapted, loaded, evaluated or measured in this experiment. It has no labels, no dataset entry and no result. `CSVISION_ALLOW_TEST_SPLIT` was not set at any point, the adapter dataset descriptor carries no holdout key, and no holdout image, label, count, prediction or metric was produced. Every number in this report comes from the 65-image validation split.

## 19. Interpretation

`CONTROLLED_COMPARISON` Changing `training.imgsz` 640 -> 768 and nothing else - everything else held fixed by inheritance from D0 - moved `supported_macro_map50_95` by +0.023876, which the frozen margin classifies as `IMPROVES_D0_BEYOND_MARGIN`.

`OBSERVATION` The aggregate hides the shape of the change. Of the 4 classes in the selection metric, 3 improved and 1 did not.

- improved: `vest_on_body` +0.056014, `helmet_on_head` +0.040641, `helmet_loose` +0.021934 (total +0.118589)
- declined: `person` -0.023084 (total -0.023084)

`LIMITATION` This is one run of each configuration on a 65-image validation split. It establishes what these two runs scored; it does not establish that `INPUT_RESOLUTION` causes the difference in general, and it cannot separate a real effect from run-to-run variance, because neither experiment was repeated.

`LIMITATION` Nothing here was tuned, and nothing may be tuned in response to it. The phase 7A policy forbids trying another capacity, another resolution, another optimizer or another augmentation setting because of what this result shows.

## 20. Final Phase 7 comparison still pending

`PENDING_EXPERIMENT` Every declared candidate now has a result, so the frozen selection logic *could* be evaluated - and this phase still does not apply it. Freezing the project's detector is a reviewed decision, so `reports/detection_experiment_results.json` records the computed case as a `policy_case_candidate` marked `advisory_only`, with `final_selected_detector: UNSELECTED_PENDING_REVIEW`.

Either way, and regardless of what the numbers above show:

- no experiment is called the Phase 7 winner, D2 included;
- D0 remains the preferred detector **by default, not by comparison**;
- no efficiency or latency benchmark is run to break any tie;
- no further experiment is authorised - not another resolution, another capacity, a combination of the two, or any tuning prompted by this result.

Committed metric-only figures: `reports/figures/detection/D2/`.
