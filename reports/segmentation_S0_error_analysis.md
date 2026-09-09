# S0 - validation error analysis

Phase 8D · classification `S1_HYPOTHESIS_READY_FOR_PROTOCOL_FREEZE` · experiment `S0` · final segmenter `UNSELECTED_PENDING_REVIEW`

**No model was trained, re-validated or modified.** This phase re-ran inference on the frozen validation split under the settings phase 8C already froze, and reports the outcome one canonical instance at a time. Every number is a validation number.

Repository commit at production time: `1c7808d01b8fa949220dd8ec49ec5ccec55baca8`.

## 1. What this analysis is, and what it is not

`PREDECLARED_PROTOCOL`. The inference settings and the matching rule are read from the committed phase 8C diagnostic protocol (`b912039ca77b36959f74fcdbaed109dbf5e3bd707709556f95a19085c7f34d80`) and asserted, not restated: conf 0.25, NMS IoU 0.7, imgsz 768, max_det 300. **No threshold was introduced, swept or tuned.**

The committed phase 8C aggregates were **not regenerated**. They were re-derived from the per-instance table only to prove this analysis describes the same run, and they agree: The committed phase 8C figures were not regenerated. They were re-derived from the per-instance table purely to prove this analysis describes the same run.

It cannot establish a cause. Nothing here is a controlled experiment, so labels naming one - model capacity, insufficient training, insufficient data - are refused by the recorder rather than merely discouraged.

## 2. Outcome census

`COMPUTED_RESULT`. All 304 canonical validation instances, partitioned by outcome. The four bands are exhaustive and disjoint, so the counts sum to the total - which is what makes the census checkable.

| Band | Definition | Count | Share |
| --- | --- | --- | --- |
| `DETECTION_MISS` | no overlapping same-class prediction | 68 | 22.4% |
| `LOW_OVERLAP_MASK` | matched, mask IoU < 0.5 | 57 | 18.8% |
| `MODERATE_MASK` | matched, 0.5 <= IoU < 0.75 | 39 | 12.8% |
| `HIGH_QUALITY_MASK` | matched, IoU >= 0.75 | 140 | 46.1% |

## 3. The `person` diagnostic

`PREDECLARED_FOCUS`. `person` was named as the focus before this analysis began, because S0 showed it the largest box-to-mask AP gap of any comparison-supported class. That gap is a fact read from the committed S0 manifest; it is **not** evidence for any particular mechanism.

| Quantity | Value |
| --- | --- |
| box AP@0.50:0.95 | 0.522227 |
| mask AP@0.50:0.95 | 0.271182 |
| **box minus mask** | **0.251045** |
| GT instances | 137 |
| Matched | 103 |
| Misses | 34 |
| matched mask IoU mean | 0.578106 |
| GT-normalised mask IoU | 0.434635 |
| IoU >= 0.50 coverage | 0.423358 |
| IoU >= 0.75 coverage | 0.233577 |

By canonical mask area:

| Area quartile | Instances | Matched | matched mean IoU | GT-normalised IoU |
| --- | --- | --- | --- | --- |
| Q1_smallest | 34 | 14 | 0.592249 | 0.243867 |
| Q2 | 28 | 19 | 0.538743 | 0.365575 |
| Q3 | 26 | 23 | 0.636039 | 0.56265 |
| Q4_largest | 49 | 47 | 0.561457 | 0.53854 |

## 4. Cross-class context

`COMPUTED_RESULT`. The same decomposition for every class, so the `person` reading is calibrated rather than cherry-picked.

| Class | GT | matched | misses | matched mean IoU | GT-normalised IoU | IoU>=0.75 |
| --- | --- | --- | --- | --- | --- | --- |
| `helmet_loose` | 57 | 47 | 10 | 0.935424 | 0.771314 | 47 |
| `helmet_on_head` | 47 | 36 | 11 | 0.876559 | 0.671407 | 32 |
| `person` | 137 | 103 | 34 | 0.578106 | 0.434635 | 32 |
| `vest_loose` | 8 | 2 | 6 | 0.263543 | 0.065886 | 0 |
| `vest_on_body` | 55 | 48 | 7 | 0.702667 | 0.613237 | 29 |

`vest_loose` remains `DESCRIPTIVE_HIGH_UNCERTAINTY`: it holds 8 validation instances on one image. Nothing about the population may be inferred from it, and it is reported here only for completeness.

## 5. Stratified diagnostics

`COMPUTED_RESULT`, all descriptive. These are pre-existing strata from the canonical and phase 8A records, not subgroups searched for after seeing the outcome.

### Canonical mask-area quartile

| Stratum | Instances | Matched | matched mean IoU | GT-normalised IoU |
| --- | --- | --- | --- | --- |
| Q1_smallest | 76 | 39 | 0.718246 | 0.368574 |
| Q2 | 76 | 62 | 0.750655 | 0.612376 |
| Q3 | 76 | 67 | 0.762971 | 0.67262 |
| Q4_largest | 76 | 68 | 0.641909 | 0.574339 |

### Phase 8A adapter-fidelity band

| Stratum | Instances | Matched | matched mean IoU | GT-normalised IoU |
| --- | --- | --- | --- | --- |
| adapter_iou_0.90_to_0.95 | 28 | 13 | 0.638228 | 0.29632 |
| adapter_iou_0.95_to_0.99 | 134 | 98 | 0.702904 | 0.514064 |
| adapter_iou_ge_0.99 | 138 | 124 | 0.740585 | 0.665453 |
| adapter_iou_lt_0.90 | 4 | 1 | 0.307059 | 0.076765 |

### Connected-component count

| Stratum | Instances | Matched | matched mean IoU | GT-normalised IoU |
| --- | --- | --- | --- | --- |
| 1_component | 240 | 186 | 0.755587 | 0.58558 |
| 2_or_more_components | 64 | 50 | 0.575639 | 0.449718 |

### Hole presence

| Stratum | Instances | Matched | matched mean IoU | GT-normalised IoU |
| --- | --- | --- | --- | --- |
| has_holes | 38 | 29 | 0.615893 | 0.470024 |
| no_holes | 266 | 207 | 0.731692 | 0.569399 |

- **Coverage and mask quality fail in opposite directions across size.** Detection coverage rises monotonically with canonical area (0.513, 0.816, 0.882, 0.895 from smallest to largest quartile), so misses concentrate in small masks. But matched mask IoU is *worst* in the largest quartile (0.642) and best in the middle two (0.751, 0.763). Large objects are reliably found and poorly delineated; small ones are missed outright.
- **`person` is weaker than every other class at every size.** Matched IoU by quartile, person against non-person: 0.592 vs 0.789, 0.539 vs 0.844, 0.636 vs 0.829, 0.561 vs 0.822. The deficit therefore survives size stratification and is not a size artefact.
- **Adapter fidelity does not track model error.** The 20 worst S0 instances have a mean phase 8A adapter IoU of 0.968 against 0.979 for the split as a whole, and the rank correlation between adapter IoU and S0 mask IoU over matched instances is 0.246 - weak and positive. Only 4 validation instances fall in the adapter-risk band at all.
- **Multi-component instances score lower, but the effect is largely `person`.** Matched IoU 0.576 for two-or-more components against 0.756 for one. Within `person` alone the gap narrows to 0.522 against 0.602, and single-component persons still sit far below single-component non-persons - so topology is a secondary contributor, not the explanation.
- **Hole-carrying instances score lower (0.616 against 0.732), and this is confounded** the same way phase 8A's was: holed instances are disproportionately large and disproportionately `person`. It is reported as an association, not an effect.

## 6. Adapter fidelity versus model error

`COMPUTED_RESULT`.

Phase 8A measured how much canonical geometry survives the YOLO label format. If that loss were driving S0's mask error, the instances S0 handles worst would tend to be the ones the adapter converted worst. They are not.

| Quantity | Value |
| --- | --- |
| Phase 8A mean adapter IoU, `person` (validation) | 0.975368 |
| Phase 8A mean adapter IoU, all validation | 0.97914 |
| Mean adapter IoU of the 20 worst S0 instances | 0.968401 |
| Spearman rank correlation, adapter IoU vs S0 IoU (matched) | 0.246 |

**Adapter conversion is unlikely to be the main explanation.** The worst S0 instances are near-average conversions, the rank correlation is weak, `person`'s mean adapter fidelity is 0.975 - higher than the split's own worst strata - and only 4 validation instances fall in the adapter-risk band. This is not a claim that the adapter has zero effect: it loses real geometry, phase 8A quantified that, and a small share of S0's error is certainly attributable to it. What the evidence rules out is adapter loss as the dominant cause.

## 6b. The overlap-resolved training target

`COMPUTED_RESULT`, labelled `POST_HOC_HYPOTHESIS_GENERATING`: this measurement was motivated by an observation made while reviewing images, not predeclared. It is the phase's substantive finding and also its most easily over-read one, so both scorings are given side by side.

Read from the installed source. `polygons2masks_overlap` sorts instances by descending area and paints them with a running maximum, so the smaller instance owns any shared pixel: a vest owns the pixels of the person wearing it, and that person's training target is the remainder. `SegmentationValidator._prepare_batch` builds its ground truth the same way, so the framework's own mask metric is measured against the same resolved target.

The canonical COCO masks the direct IoU diagnostic scores against do **not** resolve overlap - every instance keeps its full extent. So a model that faithfully reproduces its training target is charged for the difference.

| Measurement (`person`, validation) | Value |
| --- | --- |
| Canonical pixels also inside another class | 27.3% |
| **Missed pixels lying in that contested region** | **71.0%** |
| Rank correlation, contested fraction vs mask IoU | -0.5466 |
| Matched IoU, persons under 20% contested | 0.691414 (n=55) |
| Matched IoU, persons at or over 40% contested | 0.408306 (n=32) |

The same predictions and the same matching, scored against each target in turn. If the mechanism is real, `person` should move and the compact classes should not:

| Class | GT-normalised IoU vs canonical | vs overlap-resolved | delta |
| --- | --- | --- | --- |
| `helmet_loose` | 0.771314 | 0.771208 | -0.0001 |
| `helmet_on_head` | 0.671407 | 0.671407 | +0.0000 |
| `person` | 0.434635 | 0.50461 | +0.0700 |
| `vest_loose` | 0.065886 | 0.065399 | -0.0005 |
| `vest_on_body` | 0.613237 | 0.615059 | +0.0018 |

`person` matched-only mask IoU moves from 0.578106 to 0.67118 (+0.0931).

**Interpretation.** A substantial share of person's mask deficit is a target-versus-evaluation mismatch created by the frozen protocol, not a failure to learn: re-scoring the same predictions against the target the model was actually trained on recovers a large part of the gap for person and almost nothing for the other classes, which is the signature the mechanism predicts. It does not recover all of it, so a person-specific difficulty remains.

**Why this is not yet a finding.** Re-scoring changes the measuring stick, not the model. It demonstrates that the two targets disagree and that person is where they disagree most; it does not show that training without overlap resolution would produce better masks. Only a controlled experiment could, and none was run.

## 7. Deterministic qualitative review

`PREDECLARED_PROTOCOL`. The review set was chosen from the instance table by fixed rules, and its identifiers were recorded, **before any image was opened**. No example was swapped afterwards.

| Selection rule | Selected |
| --- | --- |
| `adapter_fidelity_risk` | 4 |
| `best_matched_person` | 5 |
| `misses_person` | 10 |
| `worst_helmet_loose` | 5 |
| `worst_helmet_on_head` | 5 |
| `worst_matched_person` | 10 |
| `worst_vest_on_body` | 5 |
| **distinct instances selected** | **43** |
| **of those, visually inspected in detail** | **6** |

The selection rules chose and rendered every instance listed above. A subset was then examined in detail and carries a recorded judgement; the count of judgements is the count of instances actually looked at, and it is reported separately from the count selected so neither is mistaken for the other. The population-level findings in this analysis rest on all canonical instances, not on the reviewed subset.

Figures overlay the canonical mask and the S0 prediction on the source image and stay in git-ignored runtime storage (`artifacts/segmentation/S0_error_review`): they render dataset imagery, which this repository does not publish.

### Manual mechanism census

Assigned only where the image visibly supported it. Several may apply to one instance, and an instance nothing explains stays `UNATTRIBUTED` rather than receiving the nearest-sounding label.

| Mechanism | Count |
| --- | --- |
| `UNDERSEGMENTATION` | 3 |
| `OVERSEGMENTATION` | 1 |
| `BOUNDARY_ERROR` | 0 |
| `INSTANCE_SEPARATION_ERROR` | 2 |
| `OCCLUSION_ASSOCIATED` | 3 |
| `TINY_MASK_ASSOCIATED` | 0 |
| `ADAPTER_FIDELITY_RISK` | 0 |
| `UNATTRIBUTED` | 1 |

- **Six instances were inspected in detail** out of the 43 the rules selected and rendered. That is stated rather than implied: the population-level findings above rest on all 304 instances, and the review's job was to check what the numbers meant, not to supply a census of its own.
- **The dominant `person` failure is visible and consistent.** In the worst matched cases the predicted mask covers the face, hands and trousers but omits the high-visibility jacket - precisely the region a `vest_on_body` instance also occupies.
- **It is not an assignment artefact.** On the multi-person images the best IoU available from *any* prediction is already low (0.141, 0.196, 0.243, 0.263), so one-to-one matching is not manufacturing the deficit.
- **Two of the six sit in indoor office scenes with fragmentary `person` annotations** - one covering only hair and hands while the face is plainly visible, another covering two hands of a background figure. These are data characteristics, and the frozen taxonomy has no flag for them, so they are recorded as `UNATTRIBUTED` with the evidence written out.
- **Genuine occlusion accounts for some misses**, such as an operator seated behind cab glass at low contrast.

## 8. Box versus mask, per class

`COMPUTED_RESULT`. Descriptive; no combined score is formed.

| Class | box AP@.50:.95 | mask AP@.50:.95 | gap | matched IoU | GT-norm IoU | coverage |
| --- | --- | --- | --- | --- | --- | --- |
| `helmet_loose` | 0.802332 | 0.789927 | 0.012405 | 0.935424 | 0.771314 | 0.824561 |
| `helmet_on_head` | 0.625702 | 0.586523 | 0.039179 | 0.876559 | 0.671407 | 0.765957 |
| `person` | 0.522227 | 0.271182 | 0.251045 | 0.578106 | 0.434635 | 0.751825 |
| `vest_loose` | 0.025059 | 0.001782 | 0.023277 | 0.263543 | 0.065886 | 0.25 |
| `vest_on_body` | 0.415462 | 0.390296 | 0.025166 | 0.702667 | 0.613237 | 0.872727 |

## 9. Hypotheses for a future experiment

`HYPOTHESIS`. None of these is established. Each is recorded with the evidence that supports and weakens it, so a later reviewer can disagree with the reading rather than with a conclusion.

### `DATA_OR_ANNOTATION_LIMITATION` - HYPOTHESIS_PARTLY_SUPPORTED

**Supported by.** Two of the six inspected instances are indoor office scenes with fragmentary `person` annotations - one covering only hair and hands beside a fully visible unannotated face, another covering two hands of a background figure against a whole-person prediction 36 times its area. The dataset's known limitations already include out-of-domain images. Annotation scope for `person` is visibly not uniform across the split.

**Weakened by.** Two instances cannot establish a rate. No systematic audit of annotation scope was run, and phases 4 and 5 accepted the annotations with documented limitations after a human review that did not flag this. It would also not explain the population-wide contested-pixel pattern, which holds on cleanly annotated construction images.

### `INPUT_RESOLUTION` - HYPOTHESIS_WEAKLY_SUPPORTED

**Supported by.** Detection misses concentrate in the smallest area quartile (37 of 68), and higher input resolution is the conventional remedy for small-object recall.

**Weakened by.** The mask-quality failure this phase set out to explain sits at the *other* end of the size distribution: the largest quartile has 0.895 coverage and the worst matched IoU. Resolution is least likely to be the constraint on objects that already span much of the frame. Phase 7C also found the analogous resolution hypothesis unsupported by the shape of the detection result.

### `MASK_SUPERVISION_RESOLUTION` - HYPOTHESIS_WEAKLY_SUPPORTED

**Supported by.** `mask_ratio: 4` downsamples the mask target fourfold before the loss sees it, so fine structure is not what is optimised. Matched mask IoU is worst in the largest area quartile (0.642), where a person spans much of the frame and thin extremities carry proportionally more of the boundary.

**Weakened by.** The dominant measured failure is whole-region exclusion, not contour imprecision: among poorly matched instances the automated pass flags 56 under-segmentations against 19 boundary errors, and for `person` specifically 49 against 14. A coarse supervision grid does not explain a prediction that omits an entire jacket. `mask_ratio` also applies identically to every class, yet the compact classes score 0.88-0.94.

### `MODEL_CAPACITY` - HYPOTHESIS_NOT_SPECIFICALLY_MOTIVATED

**Supported by.** YOLO11n-seg is the smallest model in its family, and `person` is the most variable class in the set. Nothing here rules capacity out.

**Weakened by.** No measurement in this analysis isolates capacity, and none could: capacity is only visible by comparison against a larger model, which this phase did not run. The class the model handles worst is also the one whose training target the overlap policy alters most, which is a confound a capacity experiment would inherit. **Absence of evidence here is not evidence that capacity cannot help** - it is the absence of a reason to try it first.

### `NO_CLEAR_INTERVENTION` - HYPOTHESIS_NOT_SUPPORTED

**Supported by.** S0 is a first baseline and some of its weakness may simply be the difficulty of the task on 303 training images.

**Weakened by.** One mechanism is measured, specific, and traceable to a single frozen protocol flag whose behaviour was read from the installed source. That is a concrete lead, whatever its eventual size.

### `OVERLAP_MASK_TARGET_MISMATCH` - HYPOTHESIS_WITH_THE_STRONGEST_SUPPORT

**Supported by.** The frozen protocol sets `overlap_mask: true`. Read from the installed source, `polygons2masks_overlap` sorts instances by descending area and paints them with a running maximum, so the **smaller** instance owns any shared pixel: a vest owns the pixels of the person wearing it, and that person's training target is the remainder. Measured consequences, over all 137 validation persons: 28.0% of canonical person pixels lie inside another class's canonical mask, but **71.0% of the pixels S0 misses on person lie in that contested region** - 2.5 times the base rate. Contested fraction correlates negatively with person mask IoU (Spearman -0.547); persons under 20% contested average 0.691 matched IoU against 0.408 for those over 40%. Re-scoring the same predictions against the overlap-resolved target the model was actually trained on raises person GT-normalised IoU from 0.4346 to 0.5046 and matched IoU from 0.578 to 0.671, while every other class moves by less than 0.002 - exactly as expected if the compact classes are the ones winning the contested pixels.

**Weakened by.** It does not close the gap. Even scored against its own training target, `person` matched IoU is 0.671, still far below `helmet_loose` at 0.935 and `helmet_on_head` at 0.877, so a material `person`-specific difficulty remains unexplained. The measurement is also post-hoc - motivated by an observation made while reviewing images - and re-scoring changes the measuring stick rather than the model, so it demonstrates a target/evaluation mismatch, not that removing the flag would improve anything.

## 10. Candidate interventions, assessed not chosen

### `mask_ratio: 2` at YOLO11n-seg / imgsz 768

**Verdict:** `WEAKLY_MOTIVATED`

The evidence does not point here. A clean one-variable intervention needs the variable to address the measured failure, and the measured failure is whole-region exclusion concentrated in pixels the training target assigns to another instance - not contour imprecision. Under-segmentation outnumbers boundary error 56 to 19 overall and 49 to 14 within `person`. Halving the mask-target downsampling would sharpen boundaries the model is already placing roughly correctly, on the classes that are already scoring 0.88-0.94, while leaving the jacket-shaped hole in the `person` masks exactly where it is. It is not ruled out - finer supervision could help at the margin - but running it first would spend the project's one controlled comparison on the weaker lead.

**One-variable against S0?** Yes, methodologically. `mask_ratio` changes the training target's resolution and nothing else in the frozen protocol, and it does not alter what the framework's validation compares against, so S0 and such an S1 would remain comparable on both the framework mask metric and the direct IoU diagnostic. The design is clean; the motivation is what is thin.

### YOLO11s-seg at imgsz 768

**Verdict:** `NOT_SPECIFICALLY_MOTIVATED`

Nothing measured here points at capacity. The class S0 handles worst is the class whose training target the overlap policy most alters, so a capacity experiment run now would inherit that confound and its result would be hard to read: a larger model might close part of the gap by better fitting an artefact of the target. There is also a specific precedent against reaching for capacity first - phase 7B varied exactly this on the detection side and D1 came in **below** D0 on the selection metric. That says nothing definitive about segmentation, but it does mean capacity is not a free bet in this project. **Absence of supporting evidence is not evidence that capacity cannot help**; it is a reason to sequence it after the mechanism that is measured.

**One-variable against S0?** Yes, methodologically - it mirrors D1's design exactly, varying the model and its consequential pretrained weights and nothing else.

## 11. Limitations

- **Descriptive, not causal.** No controlled experiment was run. Every mechanism named here is an association or an observation, and the one quantitative decomposition offered - re-scoring against the overlap-resolved target - changes the measuring stick, not the model.
- **One model, one run, one operating point.** All of it describes a single S0 checkpoint at conf 0.25 and NMS IoU 0.70. A different threshold would move the coverage figures.
- **Validation only, 304 instances, 65 images.** Small enough that per-class and per-stratum figures carry real sampling uncertainty, and `vest_loose`'s 8 instances carry so much that nothing about the population may be read from them.
- **Six instances were looked at.** The visual review confirms and illustrates the population-level measurements; it does not independently establish their frequency.
- **The strata are correlated.** Size, class, component count and hole presence are not independent in this dataset - large multi-component holed instances are disproportionately `person` - so no single stratum's effect is isolated.
- **The contested-pixel measurement was motivated by an observation made during review.** It is reported as a post-hoc, hypothesis-generating analysis over pre-existing canonical data, not as a predeclared test, and it needs a controlled experiment to become a finding.

## 12. Holdout compliance

`HOLDOUT_POLICY` · status `PROTECTED_NOT_ACCESSED`.

Phase 8D re-ran inference on the frozen validation split under the settings phase 8C froze, and inspected a deterministic sample of validation images. The holdout was not read, materialised, adapted, counted, predicted on, inspected or plotted; no holdout identifier, image, mask or statistic exists in any artifact this phase wrote.

## 13. Next decision

`PENDING_HUMAN_REVIEW`. Classification: `S1_HYPOTHESIS_READY_FOR_PROTOCOL_FREEZE`.

The preferred intervention is **`overlap_mask: false`**, and the evidence for it is the measured contested-pixel pattern rather than a preference among plausible knobs. It is recorded as a hypothesis with a named obstacle: flipping the flag changes what the framework's own mask metric is measured against, so phase 8E must decide in advance which metric adjudicates the comparison - the direct mask-IoU diagnostic against canonical COCO is comparable across the flag, and the framework's mask mAP is not.

No protocol is frozen here and no experiment is authorised. Phase 8E is a human-reviewed protocol freeze, and like phase 7A it must be written before the experiment it would decide exists.

One further observation belongs in that review rather than in this analysis: if the overlap policy stays as it is, then the project's direct mask-IoU diagnostic is charging the model for pixels it was trained to exclude. That is a reporting question about S0's published numbers as much as an experiment design question, and both readings are already recorded side by side above so that neither is quietly adopted.

---

Instance table `reports/segmentation_S0_error_instances.csv` · analysis `reports/segmentation_S0_error_analysis.json` · S0 checkpoint `d7b512b95fafdc8658fd802459d2c87d9ad3f11f922162a774dacf8ca75b87a3`.
