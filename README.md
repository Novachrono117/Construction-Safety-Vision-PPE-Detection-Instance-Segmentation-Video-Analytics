# Construction Safety Vision - PPE Detection, Instance Segmentation & Video Analytics

> **Status: both models frozen - detector YOLO11n @ 768 (D2), segmenter YOLO11n-seg @ 768 with `overlap_mask: false` (S1) - and compared on validation under a frozen protocol for both recognition and spatial information (phase 10B) and for latency and inference memory (phase 10C). The operational synthesis is pending and the holdout has never been evaluated.** The
> dataset is acquired, hashed, structurally verified, audited automatically (4A)
> and reviewed visually by people (4B). The canonical annotation snapshot is
> resolved (5A), the modelling population and its indivisible split units are
> built (5B), the **group-aware and class-aware constrained split is frozen at
> 303 / 65 / 65** over 433 modelling images (5C.2), and the COCO detection and
> instance-segmentation views of `train` and `validation` are materialised (5D).
> Phase 6A added a verified GPU runtime, a lossless YOLO detection adapter and
> the frozen D0 protocol; **phase 6B ran D0 once and reports its validation
> metrics**. Phase 7A froze the controlled comparison protocol - the class-support
> rule, the selection metric, the decision margin and the two experiments D1 and
> D2 - **before either experiment ran**. Phase 7B ran **D1 once** (below D0) and
> phase 7C ran **D2 once** (above D0 beyond the margin). Phase 7D applied that
> frozen rule mechanically and, after human review, **froze D2 - YOLO11n at imgsz
> 768 - as the final detector**; the selection is based entirely on the
> validation-only Phase 7 protocol and trained, evaluated and benchmarked
> nothing. The provider's
> split was **rejected for the final protocol** and is not reused. The `test` split is a **locked holdout**: it has never been
> evaluated, inspected, materialised or adapted, and **every number below is a
> validation number**. Phase 8A measured how much canonical instance-mask geometry
> survives the YOLO segmentation label format, and **phase 8B selected YOLO11n-seg,
> approved the audited label bytes and froze the S0 protocol**. Phase 8C then ran
> **S0 exactly once**: mask mAP@0.50:0.95 **0.407942** on validation, plus a
> predeclared direct instance-mask IoU diagnostic against the canonical COCO
> masks. Phase 8D diagnosed *why* the mask metric trails the box metric and found
> that **a large share of the `person` deficit is a target-versus-evaluation
> mismatch created by the frozen `overlap_mask: true` policy**, not a failure to
> learn. Phase 8E froze a **canonical COCO evaluator** so two models trained
> against different targets can be compared at all, evaluated S0 under it once,
> and froze **S1 as a one-variable `overlap_mask` intervention**. Phase 8F then
> ran **S1 exactly once**: canonical supported macro **0.559463** against S0's
> **0.484643**, delta **+0.074820**, classified `S1_IMPROVES_S0_BEYOND_MARGIN`
> under the margin frozen before the run, with the direct-IoU diagnostic moving
> the same way. **The aggregate is not a uniform effect**: `person` carries 88.6%
> of the gain while `helmet_loose` **regressed** -0.059035, and why any
> individual class moved is UNKNOWN. Phase 8G applied the frozen policy
> mechanically and, after human review, **froze S1 - YOLO11n-seg at imgsz 768
> with `overlap_mask: false` - as the final segmenter**. Both frozen models were
> selected on validation evidence alone. Phase 10A then **froze the
> detector-versus-segmenter comparison protocol** - what will be measured, at
> which settings, and what the measurements may not claim - **before running
> any of it**. Phase 10B then ran it: on canonical box localisation the
> segmenter scores **+0.020292** above the detector, but **that entire gain is
> carried by `vest_loose`**, the one-image high-uncertainty class - excluding
> it the delta is **-0.008040**. What masks add is elsewhere: a third of the
> median predicted box is not the object, and the box proxy for area and
> overlap is systematically inflated.

A reproducible computer-vision system for detecting and segmenting people and
personal protective equipment (PPE) in construction scenes, with a controlled
quantitative evaluation, an explicit error analysis, and inference on real
video footage.

Built as a graduate assignment in Computer Vision and Pattern Recognition, and
as a public technical portfolio project.

---

## Problem statement

Construction sites are among the most hazardous work environments, and a large
share of severe incidents involve missing or incorrectly worn protective
equipment. Manual supervision of PPE compliance does not scale: it is
intermittent, subjective, and cannot review footage after the fact.

The task addressed here is: **given an image or a video frame of a construction
scene, locate every person and every piece of protective equipment, and
distinguish equipment that is actually being worn from equipment that is merely
present in the scene.**

That distinction is what makes the problem interesting rather than routine. A
helmet lying on a bench and a helmet on a worker's head are visually similar
objects with opposite safety meanings, so a system that only detects "helmet" is
useless for compliance. The planned class set therefore separates worn from
loose equipment.

Secondary difficulties expected in this domain, to be confirmed empirically
during the dataset audit: small objects at distance, heavy occlusion in crowded
scenes, strong outdoor lighting variation, and class imbalance between people
and equipment.

## Planned architecture

```text
canonical source: instance segmentation annotations (polygons)
        |
        |  boxes derived mathematically from polygons
        v
one frozen split (train / val / test) shared by both tasks
        |
        +--> detection view  ----> fine-tuned modern detector ----+
        |                                                          |
        +--> segmentation view ---> fine-tuned instance segmenter -+
                                                                   |
                                                                   v
                                         single evaluation protocol
                                    (mAP@0.5, mAP@0.5:0.95, IoU,
                                     precision, recall, confusion matrix)
                                                                   |
                                                                   v
                                    error analysis  +  video inference
```

Two design decisions define this architecture:

1. **Instance segmentation is the single source of truth.** Bounding boxes are
   computed from the polygons rather than annotated separately, so the detector
   and the segmenter describe exactly the same objects. Any difference in their
   results is attributable to the models, not to differing labels.
2. **One split, frozen once.** Both tasks use identical image IDs per split, and
   the `test` split is a locked holdout, read exactly once after both models are
   frozen. Selection and tuning use validation data only.

## Current status

| Area | State |
| --- | --- |
| Repository foundation | Done (phase 2). |
| Dataset acquisition and provenance | Done (phase 3). Archive hashed, export structurally verified. |
| Automated audit + source EDA | Done (phase 4A). 436 originals acquired, measured and screened. |
| Manual visual audit | Done (phase 4B). 32 human decisions recorded and validated against the phase 4A manifests. |
| Canonical annotation snapshot | Done (phase 5A). The live source state, recovered read-only with complete geometry in original coordinates. |
| Canonical modelling population | Done (phase 5B). 433 modelling images, 2031 annotations retained. |
| Semantic duplicate groups | Done (phase 5B.1). All 11 near-duplicate candidates dispositioned; 11 groups, 422 split units. |
| Split candidates | Done (phase 5C.1). Six provisional candidates generated and compared. |
| Splits | **Frozen (phase 5C.2). `candidate_001` selected by human review; 303 / 65 / 65 images over 422 indivisible groups. Provider split not reused.** |
| Holdout | **Frozen and locked.** Never evaluated, inspected or materialised. Access needs two independent opt-ins. |
| Task datasets | Done (phase 5D). COCO detection + instance segmentation for `train` and `validation`: 368 images, 1726 annotations, byte-identical images, geometry round-trip verified. |
| Model-specific adapter | Done for detection (phase 6A). Lossless YOLO detection adapter, 1726/1726 boxes round-trip within 1e-4 px. Segmentation: the phase 8A adapter is **approved for controlled training** (phase 8B, `APPROVED_FOR_CONTROLLED_TRAINING`) by digest, and stays `MODEL_SPECIFIC_DERIVED_REPRESENTATION` - canonical ground truth is still COCO. |
| GPU runtime | Done (phase 6A). torch 2.11.0+cu128 on an RTX 5070 Laptop (sm_120), verified by executing real kernels. |
| D0 baseline protocol | Frozen (phase 6A). YOLO11n, imgsz 640, seed 42, metric hierarchy and checkpoint rule declared before training. |
| Detection model | **D0 trained (phase 6B).** YOLO11n, 100 epochs, one run, checkpoint selected by the predeclared rule. |
| Detection experiments | **All three complete.** D0 0.570142 · D1 (capacity, YOLO11s) 0.560017 `BELOW_D0` · D2 (resolution, imgsz 768) 0.594018 `IMPROVES_D0_BEYOND_MARGIN`, on `supported_macro_map50_95`. |
| Phase 7 winner | **Frozen (phase 7D): D2 - YOLO11n @ imgsz 768.** `CASE_B_VALIDATION_PERFORMANCE_LEADER`, derived mechanically by the frozen logic and accepted by human review. Selection is **validation-only**; no test number exists. |
| Final detector artifact | `reports/final_detector_manifest.json` · `final_detector_sha256` `84d30d64...`. The checkpoint itself is **not committed** (`LOCAL_IGNORED_FROZEN_ARTIFACT`), so a fresh clone must obtain or retrain the weights. |
| Segmentation architecture | **Selected (phase 8B): YOLO11n-seg** (`FINAL_SELECTED_FOR_S0`), by human review of the phase 8A audit. Mask R-CNN recorded as `NOT_SELECTED_FALLBACK`, never benchmarked. |
| S0 protocol | **Frozen (phase 8B).** imgsz 768, batch 8, 100 epochs, seed 42, mask metric hierarchy and checkpoint rule declared before training. Runtime proven by a one-epoch `NON_EXPERIMENTAL` smoke test. |
| Segmentation model | **S0 trained (phase 8C).** YOLO11n-seg, 100/100 epochs, one run, checkpoint chosen by the frozen native rule (epoch 59). |
| Segmentation metrics | **Validation only.** mask mAP@0.50:0.95 **0.407942** · mask mAP@0.50 0.579830 · `supported_macro_mask_map50_95` 0.509482. Box from the same model: mAP@0.50:0.95 0.478156. |
| Direct mask IoU | **Measured (phase 8C)** against canonical COCO masks at a predeclared operating point: `matched_mask_iou_mean` 0.717462, `gt_normalized_mask_iou` 0.556977, coverage 0.776316. |
| S0 error analysis | Done (phase 8D). 304 instances: 68 misses, 57 low-overlap, 39 moderate, 140 high-quality. `person` weaker than every class at every size quartile. |
| Canonical evaluation | Frozen (phase 8E). pycocotools `COCOeval` segm against canonical COCO masks. S0 reference: supported macro **0.484643**, all-class mAP@0.50:0.95 **0.388009**. |
| S1 | **Trained once (phase 8F).** One intentional variable: `overlap_mask` `true` -> `false`, 43 framework arguments inherited unchanged. 100/100 epochs, best epoch 77, `S1_experiment_sha256` `d14b98fb...`. |
| S0-vs-S1 comparison | **Computed (phase 8F).** Canonical supported macro: S0 0.484643 -> S1 **0.559463**, delta **+0.074820**, `S1_IMPROVES_S0_BEYOND_MARGIN` at the frozen 0.005 margin. Direct GT-normalised mask IoU 0.556977 -> **0.635356**; `CROSS_METRIC_DIRECTION_CONSISTENT`. |
| S1 per-class movement | `person` **+0.317316** (88.6% of the total gain), `helmet_on_head` +0.028340, `vest_on_body` +0.012659, `helmet_loose` **-0.059035** (a supported class regressed). `vest_loose` +0.035845 stays `DESCRIPTIVE_HIGH_UNCERTAINTY` and decides nothing. |
| S1 native metrics | Reported, **demoted**: mask mAP@0.50:0.95 0.458206, box 0.518779. `NOT_CROSS_TARGET_COMPARABLE_FOR_S0_S1_SELECTION` - `overlap_mask` reshapes the native validation target, so S0's and S1's native AP are never differenced. |
| Final segmenter | **Frozen (phase 8G): S1 - YOLO11n-seg @ imgsz 768, `overlap_mask: false`.** `SEGMENTER_FROZEN`, `PREDECLARED_CANONICAL_POLICY_PLUS_HUMAN_REVIEW`. Selection is **validation-only**; no test number exists. |
| Final segmenter artifact | `reports/final_segmenter_manifest.json` · `final_segmenter_sha256` `63ef4196...`. Checkpoint `29337d67...` (`LOCAL_IGNORED_FROZEN_ARTIFACT`), so a fresh clone must obtain the weights rather than retrain. S0's checkpoint and any `last.pt` are rejected **by digest**. |
| Segmenter trade-off | Published, not buried: `person` +0.317316 carries most of the gain, `helmet_loose` **regressed** -0.059035. Selection does not require every class to improve. Why any class moved is UNKNOWN. |
| Segmentation format fidelity | Measured (phase 8A). All 1726 development instances round-trip through the YOLO label format at median mask IoU 0.9846, mean 0.9731, P05 0.9182. Instance cardinality preserved 1726/1726. |
| Metrics | **Validation only.** all-class mAP@0.50:0.95 / supported macro - D0 0.4644 / 0.5701, D1 0.4711 / 0.5600, D2 0.4904 / 0.5940. No test metric exists. |
| Detector-vs-segmenter protocol | **Frozen (phase 10A)**, fingerprint `d92a1576...`. Validation only, both models at imgsz 768, AP at conf 0.001 and operational analysis at conf 0.25, FP32 for both. |
| Recognition comparison | **Computed (phase 10B), validation only.** Canonical box mAP@0.50:0.95: D2 **0.485390**, S1 **0.505682**, delta **+0.020292**. One external `COCOeval` over one ground truth; each model's own predicted boxes. |
| Recognition caveat | **The aggregate gain is carried entirely by `vest_loose`** (+0.026724 of the +0.020292). Over the four adequately supported classes the delta is **-0.008040** (`POST_HOC_DESCRIPTIVE_SUPPORT_SENSITIVITY`, descriptive only). `vest_on_body` -0.046421 and `helmet_loose` -0.018685 both declined. |
| Spatial information gain | Median mask-to-box fill ratio **0.664** and shape extent **0.672** - neither has a box-only equivalent. Box proxies overstate: area 237206 px against 144563 px measured; intersection 47281 px against 26828 px. Centroid displacement median **16.0 px**, P95 **126.9 px**. |
| Person-PPE association | At the frozen 0.50 containment floor, holding the model constant: **172 of 173 relationships classified** by the frozen taxonomy (coverage 0.994220), with 103 agreements and **0 mask-only** associations - the box proxy is close. Descriptive only; there is no association ground truth. |
| Association taxonomy | `FROZEN_TAXONOMY_NON_EXHAUSTIVE_FOR_OBSERVED_DATA`. Both rules associating to **different** people is recorded as a coverage exception (`BOTH_RULES_ASSOCIATE_DIFFERENT_PERSON`), never as a fifth category; 1 geometry-isolating and 17 pipeline-level. The phase 10A protocol was not modified. |
| Latency comparison | **Measured (phase 10C), this machine only.** `CONTROLLED_LOCAL_HARDWARE_BENCHMARK`, batch 1 at imgsz 768 in FP32 over the frozen 20-image validation subset, 4800 timed readings. Model inference: D2 **6.055740 ms**, S1 **7.777487 ms** (+1.721747 ms, +28.43%). End-to-end including mask reconstruction: D2 **9.157766 ms**, S1 **11.914757 ms** (+2.756991 ms, +30.11%). `ADDITIONAL_SEGMENTATION_PIPELINE_COST`, never pure mask-reconstruction cost. |
| Latency caveat | **The distribution is wide and the mean alone misleads.** Mean/median 1.331 (D2) and 1.429 (S1) at the model-inference boundary; block means span 4.32-9.98 ms (D2) and 5.12-11.41 ms (S1). Mobile-GPU DVFS/power-state behaviour contributes; `NO_SYNCHRONIZED_PER_OBSERVATION_POWER_STATE_TELEMETRY`, so the cause is **UNKNOWN**. Nothing was filtered, normalised or re-run. |
| Inference memory | **Measured (phase 10C).** `INFERENCE_MEMORY`, never training memory. Peak reserved: D2 **0.125 GiB**, S1 **0.296875 GiB** (ratio 2.375). Peak allocated: D2 **0.073403 GiB**, S1 **0.231621 GiB** (ratio 3.155473). Each measured in a dedicated process with the allocator empty beforehand. |
| Operational synthesis | Not started. Phase 10D. |
| Video inference | Not implemented. |
| Tracking (bonus) | Not started; deliberately deferred. |

What exists today: the project layout, a pinned environment, strict typed
configuration, the holdout protection guard, provenance primitives, a
dependency-free Roboflow acquisition client, COCO structural inspection, the
source audit / EDA / visual-review tooling, the phase 4B decision recorder, the
phase 5A geometry recovery and canonical-snapshot resolution, the phase 5B
modelling-population and grouping pipeline, the phase 5C.1 split search, the
phase 5C.2 split freeze with its fingerprinted manifest and guarded data access
layer, the phase 5D task-dataset materialiser with its geometry round-trip and
cross-task alignment validators, the phase 6A YOLO detection adapter with its
box-fidelity audit, the GPU runtime preflight, the frozen D0 baseline protocol,
the phase 6B D0 training and validation pipeline with its result bookkeeping,
the phase 7A comparison-policy freezer with its support rule, selection logic
and protocol-compatibility validator, the phase 7B experiment runner with its
inherited-protocol resolver and direct optimizer-evidence capture, the test
suite, and the planning documents (`reports/rubric_contract.md`,
`reports/roadmap.md`, `CLAUDE.md`).

### The dataset

*Construction PPE Compliance Detection* (Roboflow Universe,
`agis-workspace-8gs52`, version 4), acquired as a **COCO instance-segmentation**
export under **CC BY 4.0**. Classes verified in the annotations: `person`,
`helmet_loose`, `helmet_on_head`, `vest_loose`, `vest_on_body`.

> **742 exported images are not 742 independent samples.** The project holds
> **436 independent source images**; version 4 augments the train split x2
> offline (306 -> 612) and leaves validation (87) and test (43) unchanged. This
> was established from provider metadata, exact arithmetic and the provider's own
> export documentation - see
> [`reports/dataset_provenance.md`](reports/dataset_provenance.md).

Structural inspection found the export internally consistent: 742 image records
matching 742 files on disk, 3373 annotations, no dangling image or category
references, no duplicate ids. Segmentation geometry is a **mix of polygon (1570)
and RLE (1803)**.

### What the audit found (phase 4A)

All 436 originals were acquired at full resolution, measured and screened. See
[`reports/dataset_audit_report.md`](reports/dataset_audit_report.md) and
[`reports/eda_report.md`](reports/eda_report.md).

- **No exact duplicates.** 436 images, 436 unique content hashes.
- **11 near-duplicate candidates, 6 of them crossing a split boundary**, two of
  which are identical under both perceptual fingerprints. Phase 4A raised these
  as *candidates*, not established leakage; phase 4B confirmed them (below).
- **`vest_loose` appears in only 8 of 436 images** (45 instances): 7 train,
  1 valid, **0 test**. The rare class cannot be scored on the provider's test split.
- **17 source images carry no annotation.** Whether they are deliberate negatives
  or missing labels needed a person to look at them; phase 4B did.
- **The live source project now holds 76 more annotations than the frozen v4
  export**, so the two are not interchangeable and phase 5 must choose one.
- **The supplied COCO bbox disagrees materially with RLE segmentation geometry.**
  The preferred policy is to derive boxes from the geometry, but that decision is
  **provisional pending visual validation**.

### What the visual review decided (phase 4B)

The project owner and a technical reviewer looked at the phase 4A contact sheets
and answered the semantic questions counting cannot. The judgements are recorded
in [`reports/manual_audit_report.md`](reports/manual_audit_report.md) and, in
machine-readable form, in `reports/manual_audit_decisions.csv`. They are kept in
separate files from the computed artifacts on purpose: a human judgement is not a
measurement, and the two must never be quoted as if they were the same evidence.

- **All 6 cross-split near-duplicate pairs are semantic duplicates.** Same
  content, different bytes - the SHA-256 result stands, byte identity is *not*
  claimed. They form six groups that phase 5 must keep inside one split.
- **The provider split is `UNSUITABLE_FOR_FINAL_PROTOCOL`**: confirmed duplicates
  cross its boundaries, its test split has no `vest_loose` at all, and the rare
  class is too thin for a defensible per-class evaluation. **The split is
  rejected; the dataset is not.**
- **No widespread missing labels among the 17 zero-instance images.** Three are
  out of domain (a cartoon illustration, an office-like interior, a street scene)
  and become exclusion *candidates* for phase 5. Nothing was removed.
- **Segmentation-derived boxes are preferred, and now visually supported** - but
  the policy is `PREFERRED_AND_VISUALLY_SUPPORTED`, not applied. No annotation
  was converted.
- **The dataset is heterogeneous.** Site photography mixed with stock, posed
  portraits and product-style PPE images. It must be described as *mixed
  construction and PPE imagery*, and results on it do not demonstrate deployment
  performance on arbitrary construction-site video.

### Which annotations are canonical (phase 5A)

Phase 5A answered only one question: **which reproducible annotation state is the
source of truth**. It created no split, excluded no image and converted no
geometry. The decision and its evidence are in
[`reports/canonical_annotation_decision.md`](reports/canonical_annotation_decision.md)
and `reports/canonical_annotation_manifest.json`.

- **Decision: the live source project**, recovered read-only. 436 source images,
  2031 annotations, of which **2029 carry complete instance-segmentation
  geometry in original image coordinates** (1007 polygon, 1022 RLE).
- **Phase 4A's "1022 annotations have no geometry" was a consumption gap, not a
  provider limitation.** Those annotations carry their geometry inline as
  base64-wrapped, zlib-compressed COCO run-length encoding in a field the earlier
  walk did not read. No mutating call was made to recover it.
- **The decode is verified, not assumed.** Every recovered instance was
  re-measured and checked against the provider's own declared area and box; all
  2029 agree to **0.0 px**.
- **Version 4 was viable and was not chosen for coordinate fidelity reasons.**
  All 436 source images map to exactly one non-augmented v4 representation, so
  option A was available. It was rejected because v4 geometry is expressed after
  a stretch resize to 640x640, which would have to be inverted for every
  annotation and cannot recover what rasterisation discarded.
- **The drift is measured, and its meaning was corrected in phase 5B.** The live
  state has 76 more annotations and 6 fewer than v4 (net +70), and every one of
  the 76 additions lies at least 80% inside an annotation of its own class that
  v4 already had, at a median 0.44% of its area. This report originally inferred
  that they therefore add no coverage and are fragments; **that inference was
  withdrawn**. They are a mixture of degenerate slivers and legitimate
  corrections. `vest_loose` is identical in both states.
- **The two unrecognised annotations are resolved** as
  `VALID_BUT_UNSUPPORTED_GEOMETRY`: both fall on a real distant worker, and their
  v4 counterparts are rectangles the exporter derived from the same boxes, not
  masks anyone drew. Nothing is lost by adopting the live state.

Newer is not treated as more correct. Neither state was compared against an
independent ground truth, because none exists for this dataset.

### What may be modelled (phase 5B)

Phase 5B decided which images and annotations are eligible for a future split,
and which images must stay together in one. It created **no split**. Evidence in
[`reports/canonical_modeling_population_report.md`](reports/canonical_modeling_population_report.md)
and `reports/canonical_modeling_manifest.json`.

- **433 modelling images.** The 436 source images minus the 3 that phase 4B
  confirmed out of domain. Exclusion is logical: the files stay on disk and stay
  in the source provenance population, marked ineligible with a reason and a
  decision source. **0 annotations** were lost with them - all three were
  zero-instance, which was computed rather than assumed.
- **14 zero-instance images are retained**, not excluded. A person found no
  missing target label on them, so they are usable as negatives.
- **2031 annotations retained.** The 2 records with no provider segmentation are
  materialised as four-corner rectangles clipped to the canvas and recorded as
  `geometry_origin = SYNTHETIC_FROM_PROVIDER_BBOX` - a synthetic mask, labelled
  as one, never presented as a human-drawn outline.
- **427 indivisible split units** at the time of phase 5B: 421 singletons plus
  the 6 semantic duplicate groups phase 4B confirmed. A further 5 perceptual
  near-duplicate chains that nobody had reviewed were recorded as
  `UNCONFIRMED_GROUP_CANDIDATE` and **not** merged - a hash collision is not a
  confirmed duplicate. **Superseded by phase 5B.1**, which reviewed those 5 and
  confirmed them all, giving the current **422** units; quote 422, not 427.
- **`vest_loose` is untouched**: 45 instances across 8 images. Phase 5B recorded
  none of them as being in a duplicate group; **phase 5B.1 made that false** -
  two are in `manual_dup_010`, so the class spans 7 indivisible units.
- **The provider's rejected split appears in no artifact** a split designer
  reads.

**All 2031 annotations are retained; no automatic filter was adopted.** Phase 5B
was asked to turn the phase 5A observation about the 76 annotations added since
version 4 into a deterministic geometric rule. No rule works, because the 76 are
not one kind of thing: about half are degenerate slivers, and the rest are
**corrections that improve the labels** - a coarse polygon replaced by several
tighter ones, and in one image a single oversized `person` box covering two
people replaced by one box per person, which *is* new instance coverage.

This corrects the phase 5A reading, which described all 76 as fragments adding no
coverage; the containment measurement behind that was right, the inference from
it was not. Containment inside an older same-class annotation is not evidence of
error, because a coarse over-merged parent contains its own corrections by
definition.

The project owner therefore set `fragment_rule_status =
REJECTED_FOR_AUTOMATIC_FILTERING`. The 34 annotations the evaluated rule would
have selected keep `action = KEEP` and carry the **descriptive** flag
`NESTED_SAME_CLASS_CANDIDATE` - it records a geometric relationship, not a
defect. The failed experiment is preserved as negative evidence in
[`reports/fragment_rule_report.md`](reports/fragment_rule_report.md).

> **Reading the rule scores.** The 76 additions are a *historical annotation-drift
> reference set*, not ground truth for bad annotations. Precision against them
> measures agreement with drift, not detection of error.

### Semantic duplicate groups (phase 5B.1)

Phase 4B reviewed only the near-duplicate candidates that crossed a *provider*
split boundary, and then rejected that split. Once the split is rebuilt from
scratch, a duplicate pair that happened to sit inside one of the provider's
splits constrains the new split just as much, so the same-split candidates were
reviewed too.

- **All 11 phase 4A near-duplicate candidates now carry a human decision**: 6 in
  phase 4B, 5 in phase 5B.1. None was merged on perceptual distance alone.
- **11 confirmed groups** and **411 singletons** = **422 split units** covering
  all 433 modelling images. Groups are connected components of the confirmed
  relations, so a chain A~B, B~C forms one group rather than two overlapping
  pairs.
- **Two findings, both indivisible**, recorded distinctly in
  `group_manifest.csv`:
  - `EXACT_SEMANTIC_DUPLICATE` (10 groups) - the same frame stored twice;
  - `NEAR_DUPLICATE_SAME_SCENE` (1 group) - the same worker and scene at a
    different moment, correlated but not identical.

  Both are grouped because the point of grouping is statistical independence
  across splits, not image identity. All are **semantic**, not byte duplicates -
  the 436 source images still have distinct SHA-256 hashes.
- Group numbering continues from the phase 4B sequence; identifiers 001-006 still
  point at the same images.

**Phase 5C entry gate `MANUAL_DISPOSITION_OF_REMAINING_NEAR_DUPLICATE_CANDIDATES`
is CLOSED**, and `phase_5c_entry_readiness` is `READY_FOR_SPLIT_OPTIMIZATION`.
The last candidate had never been shown to a reviewer - it is the weakest of the
11 by perceptual distance, so it fell outside the phase 4A sheet's eight-pair cap,
and being same-split it was not on the cross-split sheet either. It was drawn on
its own in `reports/figures/review_o_remaining_near_duplicates.jpg` and decided
there.

### Provisional split candidates (phase 5C.1)

Phase 5C.1 searched for candidate train/validation/test assignments over the 422
groups. It **selected nothing and froze nothing** - selection was a separate,
later, human step (phase 5C.2, below). Evidence in
[`reports/split_candidate_report.md`](reports/split_candidate_report.md) and
`reports/split_candidates/`.

- **Six candidates**, all hitting the exact 70/15/15 target of **303 / 65 / 65**
  images with every hard constraint satisfied.
- **The unit assigned is the group, never the image**, so no confirmed duplicate
  pair can be separated - that is structural, not a penalty.
- **The provider's split is not read anywhere**: not as an input, an
  initialisation, or a target. The optimiser refuses to run if the feature table
  even carries such a column.
- **All five classes appear in all three splits**, at image and instance level.
  `vest_loose` is allocated 5/1/2 images in five candidates and 4/2/2 in the
  sixth; the 14 negatives are split 10/2/2 throughout.
- **The objective is normalised per class and averaged**, so `person` (914
  instances) cannot drown out `vest_loose` (45). Each component is reported
  separately rather than hidden inside one score.
- **Deterministic**: 192 restarts seeded from the project seed 42, re-run
  produces identical assignments, fingerprints and ranking.

One correction to the phase 5C.1 brief: it stated that no `vest_loose` image
belongs to a duplicate group. That held before phase 5B.1, which then confirmed
`manual_dup_010` as a same-scene pair - and two of the eight `vest_loose` images
are in it. The class therefore occupies **7** indivisible units, not 8, and moves
in chunks. All three declared allocation families remain feasible under that
constraint; family B (5/2/1) is searched but rejected by the requirement that the
holdout carry at least two images of the class.

### The frozen split (phase 5C.2)

**`candidate_001` was selected by human review and is now the project's
authoritative split.** Evidence in
[`reports/split_freeze_report.md`](reports/split_freeze_report.md), with the
machine-readable membership in `reports/split_manifest.json` and
`reports/final_split_assignments.csv`.

| Split | Images | Groups | Non-singleton groups | Negatives | Annotations |
| --- | --- | --- | --- | --- | --- |
| train | 303 | 294 | 9 | 10 | 1422 |
| validation | 65 | 63 | 2 | 2 | 304 |
| test | 65 | 65 | 0 | 2 | 305 |
| **total** | **433** | **422** | **11** | **14** | **2031** |

- **Selection was a human decision, not an optimiser output.** The candidates
  were predeclared and deterministically generated in phase 5C.1; a person then
  chose among them after reviewing the rare-class and evaluation trade-offs. The
  selection coincides with `algorithmic_best_candidate`, which does not remove
  the review step - the two are recorded separately.
- **All five classes appear in all three splits**, at image level and at instance
  level. `vest_loose` is allocated **5 / 1 / 2 images** and **30 / 8 / 7
  instances**.
- **Call it a group-aware and class-aware constrained split**, never a perfectly
  stratified one. Group indivisibility and the rare-class floors are hard
  constraints; proportionality is a scored preference that the group structure
  sometimes makes unreachable.
- **`vest_loose` limitation.** Validation holds exactly **one** `vest_loose`
  source image, so `vest_loose`-specific validation metrics carry high sampling
  uncertainty and must not be used in isolation for model or hyperparameter
  selection. The holdout holds two, so even the final per-class figures for that
  class carry an explicit small-sample limitation.
- **The provider's split is not reused** anywhere in the freeze, and no confirmed
  duplicate or same-scene group crosses a split boundary.
- **Fingerprints.** `split_assignment_sha256` covers every
  `(group, image, split)` triple; `holdout_sha256` covers the holdout's
  membership together with the population it was drawn from. Neither sees a
  timestamp, a path, a label or a metric, so both move exactly when membership
  moves.

**The `test` split is a locked holdout from this point.** Reading it requires two
independent opt-ins - `allow_test=True` in code **and**
`CSVISION_ALLOW_TEST_SPLIT=1` in the environment - and neither alone is
sufficient. It has not been evaluated, inspected, plotted, or used for any
decision, and it may be read once, in the final-evaluation phase, after both
models are frozen. `construction_safety_vision.data.split_freeze.FrozenSplits`
routes every request through that guard:

```python
from construction_safety_vision.data.split_freeze import load_frozen_splits

splits = load_frozen_splits("reports/split_manifest.json")
train = splits.image_ids("train", purpose="training")  # 303 ids
val = splits.image_ids("validation", purpose="model selection")  # 65 ids
splits.image_ids("test", purpose="peeking")  # HoldoutViolationError
```

**Phase 5C.2 froze membership only.** No image was copied, moved, resized or
preprocessed; no label file was written. Phase 5D, below, materialises the
detection and segmentation views from this membership.

### Canonical task datasets (phase 5D)

The frozen membership is materialised into two COCO views of the **same** images
and the **same** objects. Evidence in
[`reports/task_materialization_report.md`](reports/task_materialization_report.md),
with the machine-readable record in `reports/task_dataset_manifest.json`.

| Split | Images | Annotations | Zero-instance images | Status |
| --- | --- | --- | --- | --- |
| train | 303 | 1422 | 10 | materialised |
| validation | 65 | 304 | 2 | materialised |
| test | - | - | - | **`NOT_MATERIALIZED_PROTECTED_HOLDOUT`** |
| **development total** | **368** | **1726** | **12** | |

- **COCO is canonical for both tasks; no YOLO labels exist.** The canonical
  annotation state holds polygon geometry *and* compressed RLE, and COCO carries
  both natively. YOLO's segmentation format carries polygons only, so writing it
  would mean rasterising and re-polygonising every RLE mask - an approximation
  applied to the ground truth before a model has even been chosen.
  `model_specific_adapter` is `NOT_YET_SELECTED`.
- **Detection boxes are derived from the segmentation**, never copied from the
  provider. As a cross-check, every derived box was compared against the box
  phase 5A measured independently from the same geometry: **max delta 0.0 px**
  across all 1726 development annotations.
- **Images are copied byte-for-byte** - no resize, crop, re-encode, EXIF rotation
  or colour conversion - and both sides are hashed after the copy, so
  "byte-identical" is measured, not asserted: **368/368**.
- **Geometry preservation is verified, not claimed.** The emitted segmentation
  file is read back from disk and compared against the canonical state: RLE by
  **decoded mask** (identical `counts` strings would only prove a copy happened),
  polygons coordinate by coordinate. **1726 checked, 1726 matched, 0 mismatches**,
  covering 843 polygons, 881 RLE masks and the 2 synthetic rectangles.
- **The two views are aligned by an automated check**: same COCO image ids, same
  source image ids, same annotation ids, same categories, same boxes. The only
  intended difference is that the segmentation view carries mask geometry.
- **Zero-instance images are retained** as image records with no annotations -
  they are deliberate negatives, and dropping them would change the dataset.
- **COCO numeric ids are global**, assigned from a sorted ordering of the whole
  modelling population rather than per split, so the holdout can be materialised
  later without renumbering anything.

**The holdout was not materialised and nothing new was measured about it.** No
test image directory, no test COCO file, no test statistic. The code path that
will materialise it is the same one used here and requires both holdout opt-ins;
it is exercised only against synthetic fixtures in the test suite.

```bash
uv run python scripts/materialize_task_datasets.py               # train + validation
uv run python scripts/materialize_task_datasets.py --verify-only # validate, write nothing
```

Re-running is byte-stable: no timestamp enters any emitted file, ids come from a
sorted canonical ordering, and JSON is written with sorted keys. The datasets
themselves are git-ignored and re-derived by the command above.

### Detection baseline preparation (phase 6A)

Phase 6A prepares the first detection experiment and **deliberately does not run
it**. Evidence in
[`reports/detection_adapter_report.md`](reports/detection_adapter_report.md) and
[`reports/detection_runtime_report.md`](reports/detection_runtime_report.md).

**A lossless YOLO detection adapter.** Ultralytics reads its own label format, so
a derived representation is unavoidable - a *lossy* one is not. Canonical COCO
stays the ground truth; if the two ever disagree, the COCO file is right and the
adapter is broken. Every box is converted, written, **read back from the label
file on disk**, decoded to source pixels and compared with the canonical box:

| | train | validation | total |
| --- | --- | --- | --- |
| Images | 303 | 65 | 368 |
| Labels | 1422 | 304 | 1726 |
| Empty label files (negatives) | 10 | 2 | 12 |
| Boxes within tolerance | 1422 | 304 | **1726 / 1726** |
| Max round-trip error | 1.47e-06 px | 1.36e-06 px | tolerance 1e-4 px |

Class indices are the frozen canonical map (verified against
`class_map_sha256`), the placeholder category `object` appears nowhere, adapter
images are byte-identical to the canonical ones, and **no YOLO segmentation
labels were written** - that conversion is lossy for RLE masks and must be
fidelity-audited by a later phase.

**A GPU runtime that was verified, not assumed.** `torch.cuda.is_available()` is
a weak claim: this machine's GPU is Blackwell (`sm_120`), and only CUDA 12.8
builds carry kernels for it. So the preflight runs real kernels - a matmul
checked against the CPU, a convolution backward pass, an AMP autocast step - and
reports a blocked runtime rather than falling back to CPU if they fail.

| | |
| --- | --- |
| GPU | NVIDIA GeForce RTX 5070 Laptop, `sm_120`, 7.96 GiB |
| torch / torchvision | 2.11.0+cu128 / 0.26.0+cu128 (CUDA 12.8 build index) |
| ultralytics | 8.4.138 |
| NVIDIA driver | 610.88 |

**The D0 protocol is frozen before the experiment**, in
[`configs/detection_baseline.yaml`](configs/detection_baseline.yaml): YOLO11n
pretrained, imgsz 640, seed 42, every hyperparameter stated explicitly, the
checkpoint-selection rule fixed in advance, and the metric hierarchy declared -
primary `mAP@0.50:0.95`, then `mAP@0.50`, precision and recall. The parser
refuses a protocol that names the holdout or invents a primary metric.

**`vest_loose` carries a limitation recorded before any number exists.** It
occupies **1 validation image with 8 instances**, so its validation AP has high
sampling uncertainty and is not a usable selection signal: hyperparameters are
not tuned against it, models are not preferred because it improved, and it is
always reported with an explicit small-sample caveat.

**A smoke test, not an experiment.** One epoch on train and validation proved the
stack executes - weights load, dataset parses, forward and backward run on the
GPU, validation loader works, checkpoints written - in 62 s at 2.4 GiB peak. Its
metrics are marked `NON_EXPERIMENTAL` / `DO_NOT_REPORT_AS_MODEL_RESULT` and **no
number from it is recorded anywhere**. Reporting a one-epoch figure as a result
would be inventing a finding.

The holdout took no part in any of this: it has no adapter, no directory, no
label and no key in the Ultralytics dataset descriptor.

### D0 detection baseline (phase 6B)

**The first real model result, and it is a validation result.** D0 was specified
in full before it ran and was not tuned afterwards. Evidence in
[`reports/detection_D0_report.md`](reports/detection_D0_report.md), with the
machine-readable record in `reports/detection_D0_manifest.json`.

| Validation metric | D0 |
| --- | --- |
| **mAP@0.50:0.95** (primary) | **0.464429** |
| mAP@0.50 | 0.619373 |
| precision | 0.85568 |
| recall | 0.539992 |

| Class | AP@0.50 | AP@0.50:0.95 | precision | recall |
| --- | --- | --- | --- | --- |
| `helmet_loose` | 0.879592 | 0.792559 | 0.898364 | 0.807018 |
| `helmet_on_head` | 0.735806 | 0.561699 | 0.887663 | 0.672636 |
| `person` | 0.68902 | 0.491618 | 0.792384 | 0.583942 |
| `vest_loose` *(1 val image)* | 0.131486 | 0.041575 | 1.0 | 0.0 |
| `vest_on_body` | 0.660959 | 0.434691 | 0.699987 | 0.636364 |

- **One run, no tuning.** YOLO11n pretrained, 640 px, batch 16, seed 42, 100
  epochs completed. The checkpoint is the one the predeclared rule selected
  (best validation fitness, epoch 67), not one picked by comparing
  epochs afterwards. No second run was launched to see whether the number moved.
- **Declared policy is not effective configuration.** The protocol declares
  `optimizer: auto`; Ultralytics resolved that to **AdamW at lr0 ≈
  0.001111**, overriding the
  file's generic `lr0: 0.01`. The report states what actually ran and how that
  was established.
- **`vest_loose` carries a small-sample warning.** It has **1 validation image
  with 8 instances**; its precision of 1.0 alongside a recall of 0.0 is what one
  image's worth of evidence looks like, not a finding about the class. This
  limitation was recorded *before* the run.
- **Errors are about finding objects, not naming them.** The validation confusion
  matrix shows 96 undetected ground-truth objects and 71 unmatched predictions
  against only 3 class-to-class confusions - consistent with precision
  (0.85568) sitting well above recall (0.539992).
- **D0 is a reference point, not the project's detector.** No alternative model,
  image size or augmentation has been tried; phase 7A defines that comparison
  below, and it was defined before running it.

**The holdout was not touched.** Every number here is from the 65-image
validation split. The test split has no labels, no adapter and no key in the
dataset descriptor, and `CSVISION_ALLOW_TEST_SPLIT` was never set. D0's
validation performance is **not** an estimate of its test performance.

Metric curves, the PR/P/R/F1 curves and both confusion matrices are committed
under `reports/figures/detection/D0/`. Checkpoints are not committed: `best.pt`
is referenced by SHA-256 in the manifest and regenerated by re-running the
command.

```bash
uv run python scripts/train_detection_baseline.py --verify-only  # pre-flight only
uv run python scripts/train_detection_baseline.py                # the D0 experiment
```

### Controlled detection comparison protocol (phase 7A)

**D0 exists. D1 and D2 do not.** That ordering is what makes phase 7A a protocol
rather than a description: the deciding metric, the class-support rule, the
decision margin and the four selection cases are all frozen while the numbers
they will be applied to do not yet exist. Full policy in
[`reports/detection_comparison_policy.md`](reports/detection_comparison_policy.md),
machine-readable in `reports/detection_comparison_policy.json`, and the
experiment matrix in
[`configs/detection_experiments.yaml`](configs/detection_experiments.yaml).

**Why the metric changes for the comparison.** The all-class `mAP@0.50:0.95` is
the unweighted mean over all five classes, so a class standing on **one**
validation image carries a full fifth of it. Under the frozen split the five
classes do not carry comparable evidence:

| Class | Validation images | Validation instances | Classification |
| --- | --- | --- | --- |
| `helmet_loose` | 11 | 57 | `COMPARISON_SUPPORTED` |
| `helmet_on_head` | 26 | 47 | `COMPARISON_SUPPORTED` |
| `person` | 55 | 137 | `COMPARISON_SUPPORTED` |
| `vest_loose` | 1 | 8 | `DESCRIPTIVE_HIGH_UNCERTAINTY` |
| `vest_on_body` | 31 | 55 | `COMPARISON_SUPPORTED` |

A class is `COMPARISON_SUPPORTED` when **both** `validation_positive_images >= 5`
**and** `validation_instances >= 20`. The rule is a general threshold applied
mechanically to counts frozen in phase 5C.2 - it names no class, and excluding
`vest_loose` is its **output**, not its premise.

| Role | Metric | D0 (validation) |
| --- | --- | --- |
| Primary phase 7 selection | `supported_macro_map50_95` | **0.570142** |
| Official all-class reporting | `mAP@0.50:0.95` | **0.464429** |

The supported macro is the unweighted mean of per-class `AP@0.50:0.95` over the
four supported classes. **The two numbers are not interchangeable**: the gap of
0.105713 is the arithmetic effect of dropping one very low AP from an average,
not an improvement. The five-class figure stays mandatory and is never hidden,
and phase 6B's protocol is unchanged - D0's primary metric is still
`mAP@0.50:0.95`.

`vest_loose` remains a **required project class**: its precision, recall,
`AP@0.50` and `AP@0.50:0.95` are reported for every experiment. What it may not
do is decide which model wins.

| | D1 | D2 |
| --- | --- | --- |
| Question | Does more capacity help? | Does more input resolution help? |
| Intentional variable | `MODEL_CAPACITY` | `INPUT_RESOLUTION` |
| Model | YOLO11s (`yolo11s.pt`) | YOLO11n (`yolo11n.pt`, as D0) |
| `imgsz` | 640 (as D0) | **768** |
| Everything else | inherited from D0 | inherited from D0 |
| Status | `FROZEN_NOT_EXECUTED` | `FROZEN_NOT_EXECUTED` |

**One-variable discipline is structural, not a promise.** The candidates carry no
protocol of their own: they inherit `configs/detection_baseline.yaml` and declare
an override set, and the parser rejects a matrix whose override set differs from
its declared variable fields. D1's changed pretrained checkpoint is recorded as
*consequential* to the capacity change rather than as a second variable. Split,
labels, class map, epochs, batch, optimizer policy, patience, seed,
`deterministic`, augmentation policy, checkpoint rule and validation protocol are
identical across all three.

**Selection is decided in advance.** Ranked by `supported_macro_map50_95` with an
absolute margin of **0.005**:

| Case | Condition | Outcome |
| --- | --- | --- |
| A | nothing clears D0 by more than the margin | retain D0, the lower-complexity baseline |
| B | one candidate clears D0 **and** separates from the next-best *candidate* | that candidate leads on validation |
| C | a candidate clears D0 but does not separate from the next-best *candidate* | no winner; efficiency comparison required |
| D | execution or protocol failure | protocol review; **not** read as model inferiority |

The margin is an engineering decision threshold that prevents escalating to a
larger model for a trivial difference. It is **not** a significance test, and
`deterministic: true` reduces run-to-run variance without eliminating it - the
size of that variance on this setup is UNKNOWN, because no experiment is
repeated. Batch stays at **16** for all three; a genuine CUDA OOM stops the
experiment as `MEMORY_CONSTRAINT_REVIEW_REQUIRED` rather than being rescued by a
smaller batch, which would break the comparison.

**No holdout involvement.** Every phase 7 decision uses the 65-image validation
split. `CSVISION_ALLOW_TEST_SPLIT` was not set, no holdout image, label or count
was read, and no comparison artifact carries a holdout number.

```bash
uv run python scripts/freeze_detection_experiments.py --verify-only
uv run python scripts/freeze_detection_experiments.py
```

### D1 - capacity experiment (phase 7B)

**One run of YOLO11s, and it came out below the baseline on the deciding
metric.** Evidence in
[`reports/detection_D1_report.md`](reports/detection_D1_report.md), machine-readable
in `reports/detection_D1_manifest.json`, and the comparison table in
`reports/detection_experiment_results.json`.

D1 carried no protocol of its own: it inherited `configs/detection_baseline.yaml`
and applied a declared override set, and the runner **proved** before fetching a
single weight that the only differences from D0 were `model` (YOLO11s) and its
consequential `weight_identifier` (`yolo11s.pt`). Split, labels, class map,
epochs, batch 16, imgsz 640, optimizer policy, patience, seed 42,
`deterministic`, augmentation policy, checkpoint rule and validation protocol
were identical by inheritance. `optimizer: auto` resolved to **AdamW at lr0
0.001111** for both runs, read directly from the framework's own log line - so
the optimizer does not confound the comparison.

| Validation metric | D1 | D0 | Delta |
| --- | --- | --- | --- |
| **`supported_macro_map50_95`** (selection) | **0.560017** | **0.570142** | **-0.010125** |
| `mAP@0.50:0.95` (official, all class) | 0.471114 | 0.464429 | +0.006685 |
| `mAP@0.50` | 0.618243 | 0.619373 | -0.001130 |
| precision | 0.85964 | 0.85568 | +0.003960 |
| recall | 0.527798 | 0.539992 | -0.012194 |

**The two metrics moved in opposite directions, and that is the interesting
part.** Both are unweighted means over the same per-class APs; they differ only
in which classes they average. `vest_loose` - the class the support rule set
aside, standing on **one** validation image - gained +0.073925, contributing
+0.014785 to the five-class mean. Remove that single contribution and the
all-class delta becomes **-0.008100**, agreeing with the selection metric. So the
entire sign change in the official figure comes from the excluded class.

Per class, `AP@0.50:0.95`:

| Class | D1 | D0 | Delta | In selection metric |
| --- | --- | --- | --- | --- |
| `helmet_on_head` | 0.594100 | 0.561699 | +0.032401 | yes |
| `helmet_loose` | 0.798361 | 0.792559 | +0.005802 | yes |
| `person` | 0.496203 | 0.491618 | +0.004585 | yes |
| `vest_on_body` | 0.351404 | 0.434691 | **-0.083287** | yes |
| `vest_loose` *(1 val image)* | 0.115500 | 0.041575 | +0.073925 | no |

- **Not a uniformly worse model.** Three of the four supported classes improved.
  One class, `vest_on_body`, fell by more than the other gains combined, and its
  recall dropped 0.111868 - it lost detections rather than localisation quality.
  **Why is UNKNOWN**: one run cannot separate that from run-to-run variance, and
  diagnosing it would need the image-level error analysis this phase does not do.
- **The all-class improvement is reported, and it does not win the comparison.**
  The frozen policy names `supported_macro_map50_95` as the deciding metric, and
  it was frozen in phase 7A before this result existed. Quoting the all-class
  gain as a victory would be exactly the metric-shopping that policy exists to
  prevent.
- **No winner is declared.** D2 has not run, so the A/B/C selection logic is not
  applied. D0 remains preferred by default, not by comparison.

Execution: 100/100 epochs in 1187.8 s, best epoch 73 by the predeclared rule,
peak 4.06 GiB GPU at the frozen batch 16. YOLO11s is 9,458,752 parameters and
21.836 GFLOPs (D0's parameter count was not recorded, so no comparison is made).
The headline metric was cross-checked against the independently recomputed epoch
history at a delta of 7.4e-05. Checkpoints are referenced by SHA-256 and not
committed.

**The holdout took no part.** `CSVISION_ALLOW_TEST_SPLIT` was never set, the
adapter descriptor carries no holdout key, and no holdout image, label, count,
prediction or metric exists.

```bash
uv run python scripts/train_detection_experiment.py --experiment D1 --verify-only
uv run python scripts/train_detection_experiment.py --experiment D1 --memory-preflight
```

### D2 - input-resolution experiment (phase 7C)

**One run of YOLO11n at 768 px, and it beats the baseline beyond the margin.**
Evidence in [`reports/detection_D2_report.md`](reports/detection_D2_report.md),
machine-readable in `reports/detection_D2_manifest.json`.

D2 inherited D0's protocol and overrode exactly one field, `training.imgsz`
640 → 768. Capacity stayed YOLO11n and the run started from **the same
`yolo11n.pt` bytes D0 used** - verified by digest against D0's manifest and the
phase 6A provenance before training, and recorded as
`IDENTICAL_TO_REFERENCE_VERIFIED_BY_DIGEST`. `optimizer: auto` again resolved to
AdamW at lr0 0.001111, read directly from the framework log.

| Validation metric | D2 | D0 | Δ vs D0 | D1 | Δ vs D1 |
| --- | --- | --- | --- | --- | --- |
| **`supported_macro_map50_95`** | **0.594018** | 0.570142 | **+0.023876** | 0.560017 | +0.034001 |
| `mAP@0.50:0.95` (official) | 0.490386 | 0.464429 | +0.025957 | 0.471114 | +0.019272 |
| `mAP@0.50` | 0.646304 | 0.619373 | +0.026931 | 0.618243 | +0.028061 |
| precision | 0.926331 | 0.855680 | +0.070651 | 0.859640 | +0.066691 |
| recall | 0.516549 | 0.539992 | -0.023443 | 0.527798 | -0.011249 |

**Margin status: `IMPROVES_D0_BEYOND_MARGIN`.** Unlike D1, both metrics move the
same way here, so nothing turns on which is read.

Per class, `AP@0.50:0.95`:

| Class | D2 | D0 | Delta | In selection metric |
| --- | --- | --- | --- | --- |
| `vest_on_body` | 0.490705 | 0.434691 | +0.056014 | yes |
| `helmet_on_head` | 0.602340 | 0.561699 | +0.040641 | yes |
| `helmet_loose` | 0.814493 | 0.792559 | +0.021934 | yes |
| `person` | 0.468534 | 0.491618 | **-0.023084** | yes |
| `vest_loose` *(1 val image)* | 0.075857 | 0.041575 | +0.034282 | no |

**The small-object hypothesis is not supported by the shape of the result.**
D2 was motivated by phase 4A's frozen measurement that several PPE classes hold
many small objects. If that mechanism were driving the gain, the benefit should
concentrate in the classes with the largest small-object fractions. It does not:
ranking the four supported classes by small-object fraction against their AP
change gives a rank correlation of **-0.20**. The largest gain went to
`vest_on_body` (25.21% small - the *least* small-object-heavy of the four) and
the only decline was `person` (27.57% small). So resolution did improve the
selection metric beyond the margin - that is the controlled claim - but **why**
is not explained by the predeclared account, and a rank correlation over four
points tests nothing. A size-stratified evaluation would be needed, and this
phase does not perform one.

Precision rose 0.0707 while recall fell 0.0234. Both deserve more caution than
the AP figures: Ultralytics reports them at the F1-maximising operating point
rather than a fixed threshold, so a large move can partly reflect where that
point landed. No threshold was tuned.

Execution: 100/100 epochs in 911.8 s, best epoch 90 (fitness 0.508697), peak
3.40 GiB at the frozen batch 16. YOLO11n is 2,624,080 parameters and 6.673
GFLOPs as measured here (same capacity as D0; the resource difference is
resolution, not model size - D0's own parameter count was not recorded, so no
comparison against it is made). The headline metric was cross-checked against
the independently recomputed epoch history at a delta of 0.002944.

### Phase 7D - the detector is frozen

**Final detector: D2 - YOLO11n at imgsz 768.** Selected under the frozen
validation-only Phase 7 protocol; `reports/detection_selection_report.md` sets
out the full argument and `reports/final_detector_manifest.json` carries its
identity.

Phase 7D trained nothing, validated nothing, ran no inference, benchmarked no
latency, tuned no threshold and opened no image. Every number it used was read
from a committed result manifest.

The decision was **derived, not asserted**. The phase rebuilt each experiment's
record from its committed manifest, re-derived the class-support filter from the
frozen split, recomputed `supported_macro_map50_95`, and asked the frozen
selection engine which case held. It returned
`CASE_B_VALIDATION_PERFORMANCE_LEADER` with D2 leading and D1 as the
non-reference candidate runner-up; the
freeze refuses any other case rather than reconciling it, and a test asserts
that the winning experiment's id appears nowhere as a constant in the freeze
script.

| Experiment | `supported_macro_map50_95` (exact) | Delta vs D0 | Status |
| --- | --- | --- | --- |
| D0 | 0.570141750000 | - | reference |
| D1 | 0.560017000000 | -0.010124750000 | `BELOW_D0` |
| D2 | 0.594018000000 | +0.023876250000 | `IMPROVES_D0_BEYOND_MARGIN` |

D2 also exceeds D1 by 0.034001000000, more than the frozen 0.005 margin, so no
performance tie exists and **no efficiency tie-break was required** - the policy
demands one only in Case C. The existing `FRAMEWORK_VALIDATION_SPEED` values
remain descriptive and were not consulted.

**Three different rankings, kept apart.** Case B compares D1 and D2 as
controlled *challengers* against D0 as the *reference baseline*; D0 is never
ranked as a peer of the two, because it enters the rule through the separate
"clears the reference by more than the margin" test. So:

| Concept | Value |
| --- | --- |
| Validation performance leader (the frozen detector) | **D2** |
| Second-highest experiment overall | **D0** |
| Non-reference candidate runner-up | **D1** |

D1 being the other candidate does **not** make its metric second-highest
overall - the overall ordering is **D2 > D0 > D1**, recorded in every artifact
as `overall_validation_ranking`. Both orderings are true; they are about
different sets.

Human review was **confirmatory, not corrective**: it established that the
protocol held, that no disqualifying violation exists and that the rule was
correctly applied, then accepted the outcome. It did not override the policy,
add a metric or tie-breaker, re-run anything, or consult the holdout.

The frozen checkpoint is identified by digest rather than by path (SHA-256
`0466f872...`, 5502289 bytes) and copied byte-identically to the git-ignored
`artifacts/frozen/detection/D2_best.pt`, deliberately outside the run directory
a re-run would overwrite. `last.pt` is rejected by name as well as by digest.

**What this does not establish.** No statistical significance - the margin is an
engineering threshold, nothing was repeated, and run-to-run variance here is
UNKNOWN. No generalisation across datasets or seeds. **No support for the
small-object mechanism**: Phase 7C's class-level diagnostic gave a rank
correlation of -0.20, so the beyond-margin gain must never be read as confirming
that hypothesis. `vest_loose` is reported in full but played no part in the
rationale. D1 versus D2 is a difference, not a ranking. And **nothing at all
about test performance** - the holdout has never been evaluated.

No further detection experiment is authorised - not another resolution, another
capacity, a combination of the two, or any tuning prompted by these results.

```bash
uv run python scripts/freeze_final_detector.py --verify-only
```

```bash
uv run python scripts/train_detection_experiment.py --experiment D2 --verify-only
uv run python scripts/train_detection_experiment.py --experiment D2 --memory-preflight
```


## Phase 8A - what the YOLO segmentation format would cost

**No segmentation model was trained, evaluated or downloaded, and no
architecture was selected.** Phase 8A answered one question with numbers:
*would a YOLO segmentation label still describe the same object?*
`reports/segmentation_adapter_fidelity_report.md` is the full argument.

**The format is narrower than the data.** Read from the installed Ultralytics
8.4.138 source: one row is one class plus **one flat ring**, with no separator
between rings. So an interior hole cannot be expressed, and neither can a
disconnected mask. Both get flattened, and both *add* area.

**Every instance converted.** 1726 canonical annotations became 1726 parseable
rows - none split, merged or dropped - and the framework's own dataset scanner
read back exactly that count (303/1422 and 65/304, 12 negatives, 5 classes, 0
corrupt labels, no holdout split).

**Loss was decomposed rather than aggregated**, because pycocotools and OpenCV
do not rasterise identical geometry into identical pixels and blaming the format
for that would be wrong:

| Level | What it adds | Global mean IoU |
| --- | --- | --- |
| `control_iou` | rasteriser and contour convention alone | 0.986368 |
| `merged_iou` | component joining | 0.985525 |
| `mask_iou` | serialisation and the int32 snap | 0.973066 |

That decomposition changed the conclusion: component joining costs 0.000843 mean
IoU, while serialisation and coordinate quantisation cost 0.012458. The loss a
naive audit would have blamed on multi-component instances is mostly the integer
snap.

| Canonical representation | Instances | Mean IoU | Median IoU | Min IoU |
| --- | --- | --- | --- | --- |
| polygon | 843 | 0.978091 | 0.989090 | 0.307692 |
| compressed RLE | 881 | 0.968538 | 0.977699 | 0.743243 |
| synthetic rectangle | 2 | 0.849702 | 0.849702 | 0.761905 |

**Size dominates, not topology.** Mask-area quartiles run 0.9358 / 0.9768 /
0.9874 / 0.9924, and all 20 worst instances are masks of 4-59 px where one
boundary pixel is a large share of the area. 307 instances have more than one
component (maximum 78) and 180 carry 699 holes totalling 1,072,133 filled
pixels - but the with-versus-without-holes comparison is **confounded by size**
(median 148,640 px against 24,690 px) and is reported as such rather than as
"holes are free".

**Phase 8A decided nothing.** It recorded
`segmentation_architecture_selection: UNSELECTED_PENDING_FIDELITY_REVIEW`,
`segmentation_baseline: UNFROZEN` and `S0: NOT_DEFINED`. A high IoU distribution
is not an approval of YOLO segmentation and a low one is not a rejection; that
judgement was phase 8B's. The phase 8A artifacts are historical and immutable,
and phase 8B verifies their digests rather than editing them.

```bash
uv run python scripts/audit_segmentation_adapter.py --verify-only
```


## Phase 8B - the architecture decision and the frozen S0 protocol

**S0 was not trained.** Phase 8B selected an architecture, approved the audited
label bytes, froze the protocol, and proved the runtime executes.
[`reports/segmentation_S0_protocol.md`](reports/segmentation_S0_protocol.md) is
the full record.

**The architecture is YOLO11n-seg** (`FINAL_SELECTED_FOR_S0`), chosen by human
review of the phase 8A evidence for three reasons: the audited representation
parses and preserves all 1726 instances at quantified fidelity; the family
matches the frozen detector's YOLO11; and imgsz 768 matches its input
resolution, so a later detector-versus-segmenter comparison is not also a
resolution comparison. **A mask-native alternative such as Mask R-CNN is
`NOT_SELECTED_FALLBACK`** - never installed, trained or benchmarked here, so
nothing in this project says YOLO11n-seg is better than one.

**The adapter is approved, not promoted.** Status
`APPROVED_FOR_CONTROLLED_TRAINING`, role
`MODEL_SPECIFIC_DERIVED_REPRESENTATION`, conversion characterised as
`ACCEPTED_WITH_QUANTIFIED_APPROXIMATION` - **never lossless**, because no
instance round-trips exactly. Canonical ground truth remains COCO instance
segmentation. The approval attaches to specific bytes: the four phase 8A label
digests are re-verified against the files on disk before and after the smoke
test, and a mismatch stops the phase as `ADAPTER_FINGERPRINT_MISMATCH` rather
than triggering a silent rebuild.

**No instance was filtered.** All 1726 remain, including the 47 whose round-trip
IoU fell below 0.90. Excluding them after seeing the fidelity numbers would
change the modelling population in response to a model-format limitation.

| Frozen S0 setting | Value |
| --- | --- |
| Model / weights | YOLO11n-seg / `yolo11n-seg.pt` (SHA-256 `55ed65c5...`, 6,182,636 B) |
| imgsz | 768 (aligned with the frozen detector) |
| batch | 8 - a `PREDECLARED_EXECUTION_DECISION`; an OOM stops the phase, it is never reduced |
| epochs / patience / seed | 100 / 50 / 42 |
| Optimizer | `auto` (resolved to AdamW at lr0 0.001111, captured from the framework log) |
| Primary metric | `mask_mAP@0.50:0.95` |
| Macro metric | `supported_macro_mask_map50_95` - reported, **not** a selection metric yet |
| Checkpoint | `ULTRALYTICS_BEST_ON_VALIDATION_FITNESS` |

**The checkpoint rule was read before training, reviewed, and accepted.** For a
segmentation model the framework's validation fitness is
`SegmentMetrics.fitness = self.seg.fitness() + DetMetrics.fitness`, i.e. the
unweighted **sum of box mAP@0.50:0.95 and mask mAP@0.50:0.95** at weights 1.0 and
1.0 - so `best.pt` is **not** selected on the mask metric alone. The behaviour
was discovered before any full S0 result, returned for methodological review, and
the decision was to **keep the native composite**
(`checkpoint_selection_policy: ULTRALYTICS_NATIVE_SEGMENTATION_FITNESS`,
`checkpoint_selection_review_status: HUMAN_REVIEWED_AND_ACCEPTED_BEFORE_S0`)
rather than write a custom mask-only selector.

The consequence is stated rather than smoothed over:
`selection_metric_equals_primary_reporting_metric: false`. The project's headline
metric stays `MASK_MAP50_95`, so **the epoch S0 reports need not be the epoch
that maximised the reported metric**. No post-hoc mask-only checkpoint selection
is authorised and S0's checkpoint semantics are never reinterpreted afterwards.
Every future segmentation experiment compared directly with S0 inherits the same
policy unless a new protocol is frozen **before** it runs. None of this claims
the composite is scientifically superior to mask-only selection - nothing in this
project compares the two; it is an explicitly accepted baseline choice, and the
framework fitness is a **checkpoint-selection mechanism, never a reported
metric**.

**Mask and box metrics never merge.** A composite box-plus-mask score is refused
by the configuration parser, not merely discouraged.

**A direct instance-mask IoU diagnostic is still owed.** Ultralytics' mask AP
does not satisfy the assignment's IoU requirement, and its matching protocol
must be predeclared - not written after looking at S0's predictions. Phase 8B
records the requirement and executes nothing.

**The smoke test was engineering, not science.** One epoch,
`NON_EXPERIMENTAL` / `DO_NOT_REPORT_AS_MODEL_RESULT`: the model loaded, the
labels parsed, CUDA forward and backward ran, the validation loader worked, the
mask loss executed and a checkpoint reached disk, in ~51 s at a peak 3.20 GiB
reserved. **None of its accuracy numbers was recorded anywhere.**

```bash
uv run python scripts/freeze_segmentation_baseline.py --verify-only
```


## Phase 8C - the S0 segmentation baseline and a direct IoU result

**One training run, one validation, one diagnostic, and no decision.**
[`reports/segmentation_S0_report.md`](reports/segmentation_S0_report.md) is the
full record.

**Primary result, validation only: mask mAP@0.50:0.95 = 0.407942.** Secondary:
mask mAP@0.50 0.579830, mask precision 0.850509, mask recall 0.528435. The box
metrics the same model produces are reported separately and never merged: box
mAP@0.50:0.95 0.478156, mAP@0.50 0.647256. `supported_macro_mask_map50_95` is
0.509482 over the four classes the frozen phase 7 support rule admits -
**descriptive, not a selection metric**, because there is nothing to select
between.

| Class | mask AP@0.50:0.95 | mask AP@0.50 | box AP@0.50:0.95 |
| --- | --- | --- | --- |
| `helmet_loose` | 0.789927 | 0.909395 | 0.802332 |
| `helmet_on_head` | 0.586523 | 0.786113 | 0.625702 |
| `vest_on_body` | 0.390296 | 0.652652 | 0.415462 |
| `person` | 0.271182 | 0.539634 | 0.522227 |
| `vest_loose` | 0.001782 | 0.011359 | 0.025059 |

**`person` is the outlier worth naming.** Its mask AP@0.50:0.95 (0.271182) is
roughly half its box AP@0.50:0.95 (0.522227) - by far the widest box-to-mask gap
of any class. **Why is UNKNOWN.** It is consistent with large, frequently
occluded, irregular shapes being harder to delineate than to localise, but that
is a hypothesis; nothing here tests it, and the image-level error analysis that
could is a later, deliberate phase.

**The checkpoint was chosen by the framework's native composite**, epoch 59 at
box-plus-mask fitness 0.884180, verified to be the argmax of that composite
recomputed from `results.csv`. As phase 8B recorded and accepted, that is **not**
the primary reported metric, so the selected epoch need not be the one that
maximised mask mAP alone.

**The direct instance-mask IoU diagnostic** ran once, under a protocol frozen
before the first optimisation step, scoring against the **canonical COCO masks**
rather than the YOLO adapter, at a predeclared confidence of 0.25 and NMS IoU
0.70. Matching is one-to-one within an image and a class, solved with
`scipy.optimize.linear_sum_assignment` to maximise total IoU.

| Diagnostic | Value |
| --- | --- |
| `matched_mask_iou_mean` | 0.717462 |
| `gt_normalized_mask_iou` | 0.556977 |
| `gt_match_coverage` | 0.776316 |
| `gt_iou50_coverage` | 0.588816 |
| `gt_iou75_coverage` | 0.460526 |

304 canonical instances, 304 predictions, 236 overlapping assignments, 68
unmatched on each side. **The two headline numbers answer different questions**:
0.717462 is mask quality *where the model found something*, and 0.556977 divides
the same IoU sum by every canonical instance, so the 68 misses pull it down.
Neither is a COCO AP, and neither was used to change S0.

**Nothing was decided.** `segmentation_baseline_status: S0_COMPLETE`,
`final_segmenter: UNSELECTED_PENDING_REVIEW`. No S1, no alternative resolution or
batch, no threshold tuning, and the frozen detector was not retrained,
revalidated or run.

```bash
uv run python scripts/train_segmentation_baseline.py --verify-only
```


## Phase 8D - why the masks trail the boxes

**Nothing was trained.** Phase 8D re-ran inference on validation under the
settings phase 8C froze and reported the outcome one canonical instance at a
time. Its per-instance table reproduces the committed 8C aggregates exactly,
which is how it is known to describe the same run.
[`reports/segmentation_S0_error_analysis.md`](reports/segmentation_S0_error_analysis.md)
is the full argument.

**Coverage and mask quality fail in opposite directions.** Detection coverage
rises with object size (0.513 / 0.816 / 0.882 / 0.895 by area quartile), so
misses concentrate in small masks. But matched mask IoU is *worst* in the largest
quartile (0.642). Large objects are reliably found and poorly delineated; small
ones are missed outright. Reading either half alone gives the wrong story.

| Outcome band | Count | Share |
| --- | --- | --- |
| `HIGH_QUALITY_MASK` (IoU >= 0.75) | 140 | 46.1% |
| `DETECTION_MISS` | 68 | 22.4% |
| `LOW_OVERLAP_MASK` (IoU < 0.50) | 57 | 18.8% |
| `MODERATE_MASK` | 39 | 12.8% |

**`person` is weaker than every other class at every size.** Matched IoU by
quartile, person against non-person: 0.592/0.789, 0.539/0.844, 0.636/0.829,
0.561/0.822. The deficit is not a size artefact, and its box-minus-mask AP gap
(0.251) is an order above the next class (0.039).

**A large share of that deficit is a target-versus-evaluation mismatch, not a
failure to learn.** The frozen protocol sets `overlap_mask: true`, and the
framework's `polygons2masks_overlap` sorts instances by descending area with a
running maximum - so **the smaller instance owns any shared pixel**. A vest owns
the pixels of the person wearing it, and that person's training target is the
remainder. Measured over all 137 validation persons:

| | |
| --- | --- |
| Canonical person pixels also inside another class | 27.3% |
| **Missed pixels lying in that contested region** | **71.0%** |
| Spearman, contested fraction vs mask IoU | -0.547 |
| Matched IoU, persons <20% contested vs >=40% | 0.691 vs 0.408 |

Re-scoring the *same* predictions against the target the model was actually
trained on raises person GT-normalised IoU from 0.4346 to 0.5046 and matched IoU
from 0.5781 to 0.6712, while every other class moves by less than 0.002 -
exactly the signature the mechanism predicts, since the compact classes are the
ones winning the contested pixels.

**This is labelled `POST_HOC_HYPOTHESIS_GENERATING`, not a finding.** It was
motivated by an observation made while reviewing images, and re-scoring changes
the measuring stick rather than the model. It does not show that training without
overlap resolution would produce better masks, and it does not close the gap:
even against its own target, `person` matched IoU (0.671) stays far below
`helmet_loose` (0.935).

**Adapter conversion is not the bottleneck.** The 20 worst S0 instances average
adapter IoU 0.968 against 0.979 for the split, the rank correlation is 0.246, and
only 4 validation instances fall in the adapter-risk band. Not zero effect -
phase 8A quantified real loss - but not the dominant cause.

**Candidate interventions, assessed rather than chosen.** `mask_ratio: 2` is
`WEAKLY_MOTIVATED` (under-segmentation outnumbers boundary error 56:19, so finer
supervision does not address the measured failure). YOLO11s-seg is
`NOT_SPECIFICALLY_MOTIVATED` (nothing here isolates capacity - which is not
evidence that capacity cannot help). `overlap_mask: false` is best motivated by
the evidence but is **not a clean comparison**: the flag changes the validation
target as well as the training target, so framework mask mAP would not be
comparable across S0 and S1 while the direct IoU diagnostic would.

`S1_HYPOTHESIS_READY_FOR_PROTOCOL_FREEZE` - no protocol is frozen and no
experiment is authorised here.

```bash
uv run python scripts/analyze_segmentation_errors.py --verify-only
```


## Phase 8E - a common yardstick, and S1 frozen

**Nothing was trained.** Phase 8E froze a canonical evaluator, measured S0 under
it once, froze S1, and proved the frozen batch still fits.
[`reports/segmentation_canonical_comparison_reference.md`](reports/segmentation_canonical_comparison_reference.md)
and
[`reports/segmentation_comparison_policy.md`](reports/segmentation_comparison_policy.md)
are the record.

**Why a new evaluator was needed.** Phase 8D established that `overlap_mask`
decides both the training target *and* the framework's validation ground truth.
S0 trained with it on; S1 will train with it off. Their native mask AP would
therefore be measured against **different ground truth**, so differencing the two
numbers would compare the targets as much as the models. Native mask AP is
labelled `NOT_CROSS_TARGET_COMPARABLE_FOR_S0_S1_SELECTION` — demoted, never
suppressed.

**The canonical evaluator.** pycocotools `COCOeval` at `iouType='segm'` against
the canonical phase 5D COCO validation masks, which no training flag can move.
IoU 0.50:0.05:0.95, `maxDets` [1, 10, 100], imgsz 768, NMS IoU 0.70, conf
**0.001**, binary-mask RLE, canonical category ids used directly. The low
confidence is **not an operating point** — AP needs the low-scoring tail; the
phase 8C direct-IoU diagnostic keeps its own operational 0.25 and the two are
never mixed. Validated against synthetic fixtures before it was pointed at any
checkpoint, and deterministic across two independent executions.

**S0 canonical reference** (validation only, executed once, classified
`POST_S0_PRE_S1_CANONICAL_COMPARISON_REFERENCE_EVALUATION`):

| Metric | Value |
| --- | --- |
| **`CANONICAL_SUPPORTED_MACRO_MASK_MAP50_95`** | **0.484643** |
| `CANONICAL_ALL_CLASS_MASK_MAP50_95` | 0.388009 |
| `CANONICAL_ALL_CLASS_MASK_MAP50` | 0.537536 |

| Class | canonical AP@0.50:0.95 | native AP@0.50:0.95 |
| --- | --- | --- |
| `helmet_loose` | 0.793946 | 0.789927 |
| `helmet_on_head` | 0.609345 | 0.586523 |
| `vest_on_body` | 0.400246 | 0.390296 |
| `person` | **0.135036** | 0.271182 |
| `vest_loose` | 0.001474 | 0.001782 |

**The two columns are not comparable in absolute terms** and no difference
between them measures the model. What is legible is the shape: the compact
classes land close under both evaluators and `person` does not. That is the class
phase 8D identified, and the direction its overlap-target mechanism predicts —
**corroboration from an evaluator built for another purpose, not proof, and not a
prediction that S1 will be better.**

**S0's earlier results stand.** Native mask mAP@0.50:0.95 0.407942 remains a
valid `NATIVE_TARGET_EVALUATION`; direct GT-normalised mask IoU 0.556977 remains
a valid `CANONICAL_GT_RECOVERY_DIAGNOSTIC`. Phase 8E added a third measurement
and withdrew none.

**S1, frozen and not executed.** The only intentional difference from S0 is
`overlap_mask: true → false`; 43 other framework arguments are inherited
unchanged, and the parser refuses any second override. Same pretrained binary,
same adapter label bytes, same imgsz 768 / batch 8 / epochs 100 / seed 42 /
**mask_ratio 4**, same native checkpoint policy. Selection is frozen at margin
0.005 with three cases, `PRACTICALLY_EQUIVALENT` preferring S0 — decided in
advance. If the canonical AP and the direct IoU disagree in direction, the answer
is `CROSS_METRIC_DIRECTION_DISAGREEMENT`, recorded and never resolved by a
composite.

**One honest asymmetry.** Phase 7A's detection policy predated both its
candidates; this one could not, because the need for a common evaluator was
discovered by phase 8D. `protocol_timing` is `POST_S0_PRE_S1_PROTOCOL_FREEZE` and
the parser refuses any claim that it predated S0. No S1 number influenced any
rule.

```bash
uv run python scripts/freeze_segmentation_comparison.py --verify-only
```


## Phase 8F - S1, and what the comparison actually shows

**S1 ran exactly once**, under the protocol phase 8E froze before it existed.
The only intentional difference from S0 is `overlap_mask` `true -> false`;
43 framework arguments are resolved from S0's own protocol and inherited
unchanged, and the run's `args.yaml` was checked afterwards so the intervention
is known to have reached the trainer rather than merely been requested.

**A trap worth naming.** Read from the installed source:
`Model._reset_ckpt_args` keeps only `imgsz`, `data`, `task` and `single_cls`
from a checkpoint, and the framework default is `overlap_mask: True`. Validating
S1 without passing the flag would have scored its predictions against **S0's**
overlap-resolved target - silently, and with a plausible-looking number. The
runner passes it explicitly and refuses to continue if the resolved value is
not `False`.

**The primary result, validation only.**
`CANONICAL_SUPPORTED_MACRO_MASK_MAP50_95` **0.559463** against S0's committed
**0.484643**: delta **+0.074820**, classified `S1_IMPROVES_S0_BEYOND_MARGIN`
under the 0.005 margin frozen before the run. All-class canonical mAP@0.50:0.95
0.455034 against 0.388009. The secondary diagnostic moves the same way -
GT-normalised mask IoU 0.556977 -> **0.635356** under the unchanged phase 8C
protocol - so the outcome is `CROSS_METRIC_DIRECTION_CONSISTENT`.

**The aggregate is not a uniform effect, and reading it as one would be the
mistake.** `person` moved **+0.317316** and carries **88.6%** of the total gain
across the admitted classes; `helmet_loose` **regressed -0.059035**. The
decomposition is computed mechanically rather than written by hand, and the
report names the regression instead of leaving it inside the mean.

| Class | S0 canonical AP@0.50:0.95 | S1 | delta |
| --- | --- | --- | --- |
| `person` | 0.135036 | 0.452352 | **+0.317316** |
| `helmet_on_head` | 0.609345 | 0.637685 | +0.028340 |
| `vest_on_body` | 0.400246 | 0.412905 | +0.012659 |
| `helmet_loose` | 0.793946 | 0.734911 | **-0.059035** |
| `vest_loose` | 0.001474 | 0.037319 | +0.035845 (`DESCRIPTIVE_HIGH_UNCERTAINTY`) |

**What this does and does not establish.** The experiment is controlled: one
declared variable, everything else inherited and verified. So the movement is
attributable to `overlap_mask` **for this pair of runs**. It is not a
measurement of an effect size: nothing was repeated, so run-to-run variance on
this setup remains UNKNOWN, and the margin is an engineering threshold rather
than a significance test. The direction is consistent with the phase 8D
overlap-target mechanism, which remains `POST_HOC_HYPOTHESIS_GENERATING` - a
result consistent with a hypothesis does not confirm it, and nothing here
explains why `helmet_loose` fell.

**Native metrics are reported in full and demoted.** Mask mAP@0.50:0.95
0.458206, mAP@0.50 0.663319, precision 0.719937, recall 0.599587; box
mAP@0.50:0.95 0.518779. They are marked
`NOT_CROSS_TARGET_COMPARABLE_FOR_S0_S1_SELECTION` and never differenced against
S0's, because `overlap_mask` reshapes the native validation ground truth as well
as the training target. The two experiments' checkpoints were also selected by
the same native rule computed against **different** targets - a genuine
limitation of the comparison, and precisely why the decision is external.

**Determinism, measured rather than assumed.** Validation, the canonical
evaluation and the diagnostic were re-executed on the same frozen checkpoint to
re-render a prose addition, and the canonical, direct-IoU and results artifacts
came back **byte-identical**, with `S1_experiment_sha256` unchanged. Training
was not repeated: this is one experiment, not two.

**Nothing was selected.** `final_segmenter: UNSELECTED_PENDING_REVIEW`. No S1
re-run, no `mask_ratio` variant, no YOLO11s-seg, no resolution or batch change,
no threshold tuned, no metric added, no composite score.

```bash
uv run python scripts/train_segmentation_comparison.py --verify-only
uv run python scripts/train_segmentation_comparison.py --validate-only
```


## Phase 8G - freezing the segmenter

**Nothing was trained, evaluated or inferred.** Phase 8G reads the committed S0
and S1 artifacts, re-derives the comparison arithmetically, records the human
review that accepted the policy's answer, and freezes the selected checkpoint's
identity.

**The winner is derived, not asserted.** Both primary figures are read out of
each experiment's own canonical evaluation, the delta is recomputed, and the
frozen 0.005 margin is reapplied by the same function the policy names. No
experiment id appears in the freeze script as the answer, and a test asserts
that. Human review **confirms** the policy result; it does not override it, and
the script stops rather than record a selection the arithmetic does not
support.

**Selected: S1** - YOLO11n-seg, imgsz 768, batch 8, `mask_ratio` 4,
**`overlap_mask: false`**, best epoch 77. The primary metric moved 0.484643 ->
**0.559463**, a delta of **+0.074820**, roughly 15x the engineering margin; the
all-class canonical figure and the direct-IoU diagnostic both move the same way.

**The trade-off is in the manifest and the report.** `helmet_loose` regressed
-0.059035 while the aggregate rose, and `person` (+0.317316) carries most of the
gain. Selection does not require every class to improve - it requires the
predeclared metric to clear the predeclared margin. Why any individual class
moved is **UNKNOWN**, and the `person` result is recorded as
`CONSISTENT_WITH_PHASE_8D_OVERLAP_TARGET_HYPOTHESIS`, explicitly **not** as
proof of the mechanism.

**One rejection is specific to segmentation.** S0 and S1 are the same
architecture at the same size, produced by the same protocol, and their
checkpoints are the same number of bytes. What separates them is
`overlap_mask` - what the model was trained to predict. Loading S0's weights
expecting the frozen segmenter would work, load without complaint, and predict
against a different target. So `overlap_mask` is part of the recorded identity
and of the semantic fingerprint, and the accessor rejects S0's checkpoint **by
digest**, alongside any `last.pt`.

```bash
uv run python scripts/freeze_final_segmenter.py --verify-only
```


## Phase 10A - what the segmenter adds, and what it costs

**This phase produced no results, deliberately.** It froze the protocol for
comparing the two frozen models, before running any of it and after both models
were already frozen - so nothing in it could have been chosen to flatter
either.

**The question is not which model wins.** The detector emits a class, a
confidence and a box. The segmenter emits those plus an instance mask. They do
not produce the same output, so a single ranking would be meaningless. The
protocol asks what the mask *adds* and what it *costs*, along four axes:
recognition, spatial information, computational cost, and operational
person-PPE reasoning.

**Two confidence thresholds, and they are never mixed.** Average precision
integrates over the score curve and needs the low-scoring tail, so the AP
protocol uses **0.001**. The spatial and association analysis needs a model to
commit to a set of instances, so the operational protocol uses **0.25**. Each
belongs to one protocol; a number from one may not be reported under the
other's name.

**Both models are measured in one precision, pinned explicitly.** `half` is
deprecated in the installed ultralytics 8.4.138, and leaving its replacement
`quantize` unset delegates the choice to the runtime - which could differ
between the two models and would silently turn a latency comparison into a
precision comparison. So it is fixed at FP32 for both.

**Every mask quantity is paired with a box proxy, or declared to have none.**

| Mask measurement | Box proxy |
| --- | --- |
| `INSTANCE_AREA_PIXELS` | `BOX_AREA_PIXELS` |
| `MASK_TO_BOX_FILL_RATIO` | **`NO_BOX_ONLY_EQUIVALENT`** |
| `MASK_CENTROID` | `BOX_CENTER` |
| `SHAPE_EXTENT` | **`NO_BOX_ONLY_EQUIVALENT`** |
| `PERSON_PPE_MASK_INTERSECTION` | `BOX_INTERSECTION_AREA` |
| `PERSON_PPE_MASK_CONTAINMENT` | `BOX_INTERSECTION_OVER_PPE_BOX_AREA` |
| `VISIBLE_PPE_COVERAGE_PROXY` | `BOX_OVERLAP_DERIVED_COVERAGE` |

That pairing *is* the answer to "what does segmentation add": a quantity with a
good box proxy adds little, and one with no box equivalent is the actual gain.
Deciding which is which after seeing the numbers would be circular, so it is
decided now.

**The association analysis is descriptive and named as such.**
`SPATIAL_ASSOCIATION_ANALYSIS`, not compliance accuracy. The project holds no
canonical compliance ground truth, and the provider's `helmet_on_head` and
`vest_on_body` already encode a worn state - inventing a compliance label on
top of them would be manufacturing ground truth. Each person-PPE candidate pair
will fall into one of `BOX_AND_MASK_AGREE`, `BOX_ONLY_ASSOCIATION`,
`MASK_ONLY_ASSOCIATION` or `NEITHER_ASSOCIATION`.

**The latency benchmark is frozen before anything is timed.** batch 1, imgsz
768, FP32, 20 warmup iterations discarded, 30 timed repetitions over 20
benchmark images chosen by ranking validation ids by their own SHA-256 - so the
subset cannot have been picked for being easy or crowded, because no image was
opened to choose it. Execution interleaves the two models and reverses the
order on a second pass, because benchmarking one to completion first would
measure the laptop's thermal state as much as the model. Two boundaries are
reported: the forward pass alone, and end-to-end including NMS and **mask
reconstruction** - putting the latter outside the segmenter's measurement would
hide exactly the cost being quantified.

**No aggregate score.** Benefit and cost are reported side by side. A weighted
index invented once the numbers are visible would hide the trade-off.

```bash
uv run python scripts/freeze_detector_segmenter_comparison.py --verify-only
```


## Phase 10B - what the masks actually bought

**Validation only, and the cost half is still unmeasured.** This phase ran both
frozen models under the two protocols phase 10A froze, scored their boxes with
one external evaluator, and computed the frozen spatial quantities. No latency
benchmark - that is phase 10C.

**Precision parity was proved, not assumed.** Both models were probed at
runtime before any comparison number existed: backend FP16 flag `false`,
parameter dtypes `['torch.float32']`, input tensor `torch.float32`, autocast
`false`, no quantization config - identical for both. A comparison across two
precisions would have measured the precision.

### Recognition: the headline is misleading on its own

| Metric | D2 | S1 | delta |
| --- | --- | --- | --- |
| `CANONICAL_BOX_MAP50_95` | 0.485390 | 0.505682 | **+0.020292** |
| `CANONICAL_BOX_MAP50` | 0.641107 | 0.692955 | **+0.051848** |

The all-class figure is the unweighted mean of five per-class APs, so it
decomposes exactly:

| Class | D2 | S1 | delta | contributes |
| --- | --- | --- | --- | --- |
| `helmet_loose` | 0.767262 | 0.748577 | **-0.018685** | -0.003737 |
| `helmet_on_head` | 0.628022 | 0.629018 | +0.000996 | +0.000199 |
| `person` | 0.485281 | 0.517231 | +0.031950 | +0.006390 |
| `vest_loose` | 0.068034 | 0.201654 | +0.133620 | **+0.026724** |
| `vest_on_body` | 0.478350 | 0.431929 | **-0.046421** | -0.009284 |

**`vest_loose` alone contributes more than the entire aggregate delta**, and it
is the class the project already classifies `DESCRIPTIVE_HIGH_UNCERTAINTY` with
one validation image. Quoting "+0.020292" without this would be the
metric-shopping the project's phase 7A policy forbids.

Applying the project's **pre-existing** support rule as a descriptive
sensitivity check (`POST_HOC_DESCRIPTIVE_SUPPORT_SENSITIVITY` - not a frozen
metric, not a selection rule, not a significance test):

| | D2 | S1 | delta |
| --- | --- | --- | --- |
| All-class canonical box mAP@0.50:0.95 | 0.485390 | 0.505682 | **+0.020292** |
| Supported-class macro (4 classes) | 0.589729 | 0.581689 | **-0.008040** |

**The conclusion:** S1 retains broadly similar localisation capability to D2
while additionally producing masks, but the positive all-class delta is driven
by the highly uncertain `vest_loose` class and should **not** be read as robust
evidence that S1 is the superior object localiser. Neither reading selects a
model, and neither frozen model changes.

These are canonical-evaluator figures and are **not** the native framework
metrics either model reported in its own phase; the two must never be
differenced.

### Spatial information: this is where the mask earns its place

Two quantities have no box-only equivalent at all:

* **`MASK_TO_BOX_FILL_RATIO`** median **0.664** (P25 0.556, P75 0.775). On the
  median prediction, a third of the box is not the object.
* **`SHAPE_EXTENT`** median **0.672**. Instances do not fill even their own
  tight rectangle.

Where a box proxy does exist, it is **systematically biased rather than merely
noisy**, because a box counts background as object:

| Quantity | mask measurement | box proxy |
| --- | --- | --- |
| Instance area (mean px) | 144563 | 237206 |
| Person-PPE intersection (mean px) | 26828 | 47281 |
| Centroid vs box centre | median 16.0 px, P95 126.9 px, max 262.8 px | — |

### Association: the honest result is that the box proxy is close

Holding the model constant - S1's own boxes, so only the shape representation
varies - across 173 person-PPE relationships at the frozen 0.50 containment
floor: **103 agree, 3 box-only, 0 mask-only, 66 neither, 1 both-but-different-
person**. At this floor the mask changes almost no association decision. The
pipeline-level reading against the detector's boxes disagrees much more
(81/7/6/62/17), but that is confounded - different model, different instances -
and is reported as a separate question.

**The frozen taxonomy turned out to be non-exhaustive.** Phase 10A froze four
categories assuming a rule either associates or does not; two rules can both
associate and pick *different* people, which none of the four describes. That
state is recorded as a **coverage exception**
(`BOTH_RULES_ASSOCIATE_DIFFERENT_PERSON`,
`UNCLASSIFIED_BY_FROZEN_FOUR_CATEGORY_TAXONOMY`), **not** as a fifth peer
category, and it is excluded from the denominator the frozen percentages use.
The phase 10A protocol itself was not modified.

| Reading | classified | exceptions | total | coverage |
| --- | --- | --- | --- | --- |
| Geometry-isolating | 172 | 1 | 173 | 0.994220 |
| Pipeline-level | 156 | 17 | 173 | 0.901734 |

A different-person outcome is an **`ASSOCIATION_RULE_DISAGREEMENT`, not an
error** - there is no association ground truth to be wrong against. And the 17
pipeline-level exceptions must not be read as pure geometry: that comparison
also changes the model and the instances.

**No compliance claim is made anywhere.** There is no compliance ground truth
in this project, so the association analysis reports agreement between two
geometric rules and no accuracy. `VISIBLE_PPE_COVERAGE_PROXY` stays an
`INTERPRETIVE_OPERATIONAL_PROXY` - it is the one frozen quantity whose
definition is qualitative, and nothing here rests on it.

```bash
uv run python scripts/compare_detector_segmenter.py --preflight-only
```


## Phase 10C - what the masks cost

Phase 10B answered what the segmenter adds. This phase answers what it costs,
under the benchmark phase 10A froze before anything was timed. It trained
nothing, recomputed no average precision, reran no spatial or association
analysis, tuned no threshold and never touched the holdout.

**The measurement.** `CONTROLLED_LOCAL_HARDWARE_BENCHMARK`: batch 1 at imgsz
768 in FP32, over the frozen 20-image validation subset, 20 warmup iterations
discarded and 30 timed repetitions per block, in a symmetric interleaved order -
pass A times the detector then the segmenter on each image, pass B reverses
them over the same images in the same order. 80 timed blocks, **4800 timed
readings**, every one bracketed by an explicit `torch.cuda.synchronize()` on
both edges with the same primitive for both models. Raw timing fingerprint
`fce637ee...`, execution-plan fingerprint `9734f8f0...`.

**Two boundaries, never merged.**

| Boundary | D2 mean | S1 mean | Absolute | Relative | Throughput ratio |
| --- | --- | --- | --- | --- | --- |
| `MODEL_INFERENCE_LATENCY_MS` | 6.055740 ms | 7.777487 ms | **+1.721747 ms** | **+28.43%** | 0.778624 |
| `END_TO_END_MODEL_OUTPUT_LATENCY_MS` | 9.157766 ms | 11.914757 ms | **+2.756991 ms** | **+30.11%** | 0.768607 |

The narrow boundary is `BasePredictor.inference` on a tensor prepared outside
the timer. The wide one is `preprocess` plus `inference` plus `postprocess` -
and **the segmenter's mask reconstruction is inside it**, which is a fact read
from the installed source rather than an assumption: with `retina_masks`
enabled, `SegmentationPredictor.construct_result` calls
`ops.process_mask_native`, which combines the prototypes with the per-instance
coefficients and upsamples onto the original image canvas, all inside
`postprocess`. The run verifies for every benchmark image with an instance that
the returned masks are on the original canvas, and aborts otherwise. Excluding
that work would hide exactly the cost this comparison exists to quantify.

`images_per_second_from_mean` is `1000 / mean`, at batch 1, never the
reciprocal of the fastest repetition. It is a latency reciprocal, not batched
throughput under load.

**The label is `ADDITIONAL_SEGMENTATION_PIPELINE_COST`, not pure
mask-reconstruction cost.** YOLO11n and YOLO11n-seg differ in the mask branch
of the network as well as in postprocessing, and nothing here isolates the two.
`pure_mask_reconstruction_cost_isolated: false`.

**The distribution is wide, and the mean alone would mislead.** At the
model-inference boundary D2 has median 4.550700 ms against mean 6.055740 ms and
P90 9.959820 ms; S1 has median 5.444300 ms against mean 7.777487 ms and P90
11.439560 ms. A mean a third above its own median with a P90 near the maximum
is not a tail of stragglers. The committed per-block table shows the spread
separating **between** blocks rather than within them: a block's thirty
repetitions cluster, while block means span 4.318367-9.981530 ms for D2 and
5.118167-11.413863 ms for S1, and which level a block sits at does not follow
the image, the model or the pass.

This is recorded as `POST_HOC_HARDWARE_BEHAVIOR_DIAGNOSTIC` /
`POST_HOC_DIAGNOSTIC_ONLY` - written after the run, computed from all 4800
observations, replacing no frozen statistic and deciding nothing.
**`causal_attribution: UNKNOWN`.** The shape is *consistent with* mobile-GPU
DVFS and power-state behaviour, but that stays an `UNTESTED_HYPOTHESIS`:
`NO_SYNCHRONIZED_PER_OBSERVATION_POWER_STATE_TELEMETRY`, so no clock, P-state,
utilisation, temperature or power reading accompanied the timed regions and no
observation can be mapped to a device state. Both models show the same *kind*
of skew, and that is all that is claimed -
`proportionality_across_models_demonstrated: false`.

**Nothing in the protocol was adapted after the timings were seen**, which is
when a frozen protocol earns its keep. The subset, the 20 warmup iterations,
the 30 repetitions, the symmetric order, FP32, batch 1, imgsz 768 and both
boundaries are all unchanged; no observation was removed, no outlier rule was
introduced, no timing was normalised or rescaled, no power, clock or fan
setting was touched, and **the benchmark was not re-run**. Later prose
corrections went through `--rebuild-results`, which re-derives the artifacts
from the persisted 4800 observations, executes no model, takes no timing, and
refuses to write unless every frozen statistic and delta recomputes identically.

**Inference memory**, `INFERENCE_MEMORY` and never training memory:

| | D2 | S1 | Ratio |
| --- | --- | --- | --- |
| Peak allocated | 78815744 B (0.073403 GiB) | 248700928 B (0.231621 GiB) | 3.155473 |
| Peak reserved | 134217728 B (0.125 GiB) | 318767104 B (0.296875 GiB) | 2.375 |

Peak statistics are reset with `torch.cuda.reset_peak_memory_stats()` **after**
the frozen warmup, so the figure describes inference rather than the
allocator's warmup high-water mark. Each model is measured in a **dedicated
process**: peak CUDA statistics are device-global, and a diagnostic run before
any memory figure existed showed that releasing a model in-process still leaves
a 33554432-byte cuBLAS workspace allocated, which the next model measured would
have been charged for. `pre_load_allocated_bytes` is 0 for both, recorded as
the evidence that the isolation held rather than asserted.

**FP32 parity was proved at runtime, not assumed.** Both models: backend FP16
flag `false`, parameter dtypes `['torch.float32']`, input tensor
`torch.float32`, autocast during forward `false`, no quantization config,
`quantize` resolved to 32 - byte-identical between the two. A difference would
have stopped the phase as `PRECISION_PROTOCOL_MISMATCH`.

**One gap in the frozen protocol is recorded rather than papered over.**
`latency_protocol` declares batch, resolution, precision, warmup, repetitions,
membership and order, but **no confidence threshold**. The operational block is
the only frozen operating point - the protocol itself states that the AP
block's 0.001 is deliberately not one - so the benchmark runs at the
operational **0.25** and says so in every artifact. Latency at 0.001 was not
measured and is not claimed; it would differ, because a lower threshold pushes
more candidates through NMS and more masks through reconstruction.

**Static complexity is read, not recomputed**: D2 2624080 parameters / 6.673
GFLOPs, S1 2843583 / 9.8 (fused 2835543 / 9.6), taken from the committed
experiment manifests. `measured_at_benchmark_input_size: false` - both figures
come from framework paths that default to a **640** reference input, so they
describe the architectures at a different input size than the latency figures
do, and they must not be read as an explanation of the timings.

Limits worth stating plainly: a laptop GPU throttles; batch 1 measures latency,
not throughput under load; neither model is exported or quantised for
deployment; host transfer of the outputs is outside both boundaries for both
models, so a pipeline needing masks in host memory would pay more than these
figures show. **No claim of hardware-independent latency is made or
supported**, and no operational recommendation follows - that is phase 10D,
which has not started.

Evidence: [`reports/detector_segmenter_latency_comparison.json`](reports/detector_segmenter_latency_comparison.json),
[`reports/detector_segmenter_memory_comparison.json`](reports/detector_segmenter_memory_comparison.json),
[`reports/detector_segmenter_latency_blocks.csv`](reports/detector_segmenter_latency_blocks.csv),
[`reports/detector_segmenter_latency_report.md`](reports/detector_segmenter_latency_report.md).
Result fingerprints: latency `27c1705f...`, memory `7f452c8a...`.

```bash
uv run python scripts/benchmark_detector_segmenter.py --verify-only
uv run python scripts/benchmark_detector_segmenter.py --preflight-only
```


## Academic requirements

The assignment requires all of the following. Each is mapped to a verifiable
definition of done in [`reports/rubric_contract.md`](reports/rubric_contract.md).

- At least 300 annotated images, with a train/validation/test split
- Fine-tuning of a modern object detector
- Instance segmentation in the same domain
- mAP@0.5 and mAP@0.5:0.95
- IoU, precision and recall
- Confusion matrix
- Qualitative analysis of false positives and false negatives
- Inference on real video of at least 30 seconds
- Complete hyperparameter documentation
- An executable Colab notebook
- A technical report
- A reproducible GitHub repository
- A 5-8 minute video pitch
- Bonus (up to +0.5): ByteTrack tracking or an interactive demo

## Planned methodology

Work proceeds through 14 gated phases (see
[`reports/roadmap.md`](reports/roadmap.md)). In outline:

1. **Scope and contract.** Turn the rubric into falsifiable acceptance criteria
   and fix the experimental protocol before seeing any data.
2. **Foundation.** Reproducible skeleton, tooling, protocol primitives.
3. **Acquisition and provenance.** Record source, version, license and file
   hashes for everything downloaded.
4. **Audit and EDA.** Measure class distribution and annotation integrity;
   detect exact duplicates, near duplicates and video-frame leakage.
5. **Split freeze.** Freeze one partition, keeping duplicates and same-sequence
   frames inside a single split; derive the detection and segmentation views.
6-7. **Detection.** Baseline first, then controlled variations, then freeze one
   model selected on validation data.
8-9. **Segmentation.** The same procedure on the same splits.
10. **Controlled validation comparison.** One evaluation protocol for all
    models, with matching rules and thresholds stated.
11. **One-shot test evaluation.** The holdout is unlocked and read exactly once.
12. **Error analysis.** Per-class FP/FN review, with observations separated from
    hypotheses.
13. **Video inference.** Real footage, measured throughput, temporal failure
    modes. Tracking only as a bonus, after the required work is complete.
14. **Submission and reproducibility audit.** Verify the whole thing from a
    clean clone.

## Repository structure

```text
.
├── CLAUDE.md                  # Operating constitution: protocol rules for AI-assisted sessions
├── README.md
├── pyproject.toml             # Project metadata, dependencies, ruff and pytest configuration
├── .env.example               # Documented environment variables (no secrets)
├── .gitattributes             # LF normalisation, binary declarations
├── configs/                   # Versioned experiment configuration (single source of settings)
│   ├── project.yaml
│   ├── split_search.yaml      # Split-search protocol: targets, constraints, weights (5C.1)
│   ├── split_freeze.yaml      # Which candidate was selected, and what it must reproduce (5C.2)
│   ├── task_materialization.yaml # Copy, naming, bbox and geometry policies (5D)
│   ├── detection_adapter.yaml    # COCO -> YOLO detection adapter protocol (6A)
│   ├── detection_dataset.template.yaml # Portable Ultralytics dataset descriptor (6A)
│   ├── detection_baseline.yaml   # D0 protocol: model, hyperparameters, metrics (6A)
│   └── detection_experiments.yaml # Phase 7 matrix: D0/D1/D2, support rule, margin (7A)
│                                  #   frozen before the run; parser rejects a test reference
├── data/                      # Never committed; see data/README.md
│   ├── external/              # Provider archive, source originals, provenance record
│   ├── raw/                   # Extracted canonical export, untouched
│   ├── interim/               # Derived representations (source geometry, image stats)
│   └── processed/             # Canonical COCO task datasets for train + validation (5D)
├── notebooks/                 # Colab-executable notebooks (added by the phase that needs them)
├── reports/                   # The committed evidence; every number traces to a file here
│   ├── rubric_contract.md     # Rubric as a verifiable contract
│   ├── roadmap.md             # 14 phases with validation gates
│   ├── dataset_provenance.md  # What the dataset is, and how we know (phase 3)
│   ├── dataset_audit_report.md    # Automated audit (phase 4A)
│   ├── eda_report.md              # Source EDA (phase 4A)
│   ├── bbox_consistency_audit.md  # Supplied bbox vs segmentation geometry (phase 4A)
│   ├── manual_audit_report.md     # Human visual decisions (phase 4B)
│   ├── annotation_drift_report.md # Live source vs v4 snapshot (phase 5A)
│   ├── canonical_annotation_decision.md   # Which annotations are canonical (phase 5A)
│   ├── canonical_annotation_manifest.json # The same decision, machine-readable
│   ├── fragment_rule_report.md            # Can a rule identify the drifted annotations? (5B)
│   ├── canonical_modeling_population_report.md  # What may be modelled (phase 5B)
│   ├── canonical_modeling_manifest.json   # The population, machine-readable
│   ├── group_manifest.csv                 # Indivisible split units
│   ├── group_split_features.csv           # One row per group, for phase 5C
│   ├── split_candidate_report.md          # Provisional split candidates (phase 5C.1)
│   ├── split_candidates/                  # One assignment per candidate, a summary, the selection
│   ├── split_freeze_report.md             # Why this split, and what it cannot support (5C.2)
│   ├── split_manifest.json                # THE frozen split: membership + fingerprints (5C.2)
│   ├── final_split_assignments.csv        # The same membership, one row per image (5C.2)
│   ├── split_freeze.provenance.json       # How the freeze was produced (5C.2)
│   ├── task_materialization_report.md     # How the task datasets were built (5D)
│   ├── task_dataset_manifest.json         # Counts, fingerprints, holdout status (5D)
│   ├── task_materialization.provenance.json
│   ├── detection_adapter_report.md        # YOLO detection adapter + fidelity audit (6A)
│   ├── detection_adapter_manifest.json    # Adapter counts and fingerprints (6A)
│   ├── detection_runtime_report.md        # GPU runtime, weights, smoke test (6A)
│   ├── detection_D0_report.md             # D0 baseline result and its limits (6B)
│   ├── detection_D0_manifest.json         # D0 metrics, fingerprints, checkpoints (6B)
│   ├── detection_comparison_policy.md     # Phase 7 comparison rules, frozen (7A)
│   ├── detection_comparison_policy.json   # The same policy, machine-readable (7A)
│   ├── detection_comparison_reference.json # D0 reference values for phase 7 (7A)
│   ├── detection_D1_report.md             # D1 capacity experiment and its limits (7B)
│   ├── detection_D1_manifest.json         # D1 metrics, fingerprints, checkpoints (7B)
│   ├── detection_D2_report.md             # D2 resolution experiment and its limits (7C)
│   ├── detection_D2_manifest.json         # D2 metrics, fingerprints, checkpoints (7C)
│   ├── detection_experiment_results.json  # D0/D1/D2 table, final selection (7D)
│   ├── detection_selection_report.md      # Why D2 was frozen, and its limits (7D)
│   ├── final_detector_manifest.json       # Frozen detector identity + fingerprint (7D)
│   ├── detection_experiment_comparison.csv # Machine-readable D0/D1/D2 table (7D)
│   ├── segmentation_adapter_fidelity_report.md  # What the YOLO seg format costs (8A)
│   ├── segmentation_adapter_audit_manifest.json # Fidelity, topology, parser check (8A)
│   ├── segmentation_adapter_fidelity.csv        # One row per development annotation (8A)
│   ├── segmentation_adapter_approval.json       # Adapter approved for training, by digest (8B)
│   ├── segmentation_S0_manifest.json            # Frozen S0 protocol + smoke record (8B)
│   ├── segmentation_S0_protocol.md              # Architecture decision and S0 protocol (8B)
│   ├── segmentation_S0_result_manifest.json     # S0 metrics, fingerprints, checkpoints (8C)
│   ├── segmentation_S0_mask_iou.json            # Direct instance-mask IoU diagnostic (8C)
│   ├── segmentation_S0_report.md                # The S0 baseline result and its limits (8C)
│   ├── segmentation_S0_error_analysis.md        # Why the masks trail the boxes (8D)
│   ├── segmentation_S0_error_analysis.json      # Strata, hypotheses, candidates (8D)
│   ├── segmentation_S0_error_instances.csv      # One row per canonical validation instance (8D)
│   ├── segmentation_S0_canonical_evaluation.json # S0 under the common evaluator (8E)
│   ├── segmentation_canonical_comparison_reference.md # Why, and the S0 reference (8E)
│   ├── segmentation_comparison_policy.{json,md} # The frozen S0-vs-S1 policy (8E)
│   ├── segmentation_S1_protocol.md              # S1, frozen and not executed (8E)
│   ├── segmentation_S1_protocol_manifest.json   # The same, machine-readable (8E)
│   └── figures/               # Contact sheets, analytical plots, D0 metric curves
├── scripts/                   # Command-line entry points, one job each
│   ├── check_environment.py       # Environment, configuration and holdout-lock report
│   ├── download_dataset.py        # Acquire + hash + extract the canonical export
│   ├── inspect_dataset.py         # Structural inspection -> provenance report
│   ├── fetch_source_inventory.py  # Recover the 436-image source inventory
│   ├── download_source_images.py  # Acquire the source originals at full resolution
│   ├── audit_source_dataset.py    # Duplicates, near duplicates, zero-instance images
│   ├── audit_bbox_consistency.py  # Supplied bbox vs the geometry it claims to enclose
│   ├── eda_source_dataset.py      # Source-population statistics and plots
│   ├── write_audit_reports.py     # Audit + EDA measurements -> written reports
│   ├── build_review_package.py    # Contact sheets for the human visual review
│   ├── record_manual_audit.py     # Validate and record the phase 4B decisions
│   ├── recover_source_geometry.py # Complete live geometry, read-only and verified
│   ├── map_v4_sources.py          # Source images -> non-augmented v4 representations
│   ├── analyze_annotation_drift.py # Live source vs v4 snapshot, per image
│   ├── build_drift_figures.py     # Review sheets for the drift and the odd records
│   ├── resolve_canonical_snapshot.py # The canonical-snapshot decision + manifest
│   ├── analyze_fragment_rule.py   # Can a geometry rule identify the drifted annotations?
│   ├── build_modeling_population.py # Eligible images/annotations + indivisible groups
│   ├── build_remaining_duplicate_review.py # Candidates still awaiting a human decision
│   ├── optimize_split_candidates.py # Provisional split candidates over the groups
│   ├── freeze_split.py            # Verify the selected candidate and freeze it + the holdout
│   ├── materialize_task_datasets.py # Canonical COCO detection + segmentation views (5D)
│   ├── build_detection_adapter.py # COCO -> YOLO detection labels + fidelity audit (6A)
│   ├── detection_runtime_check.py # GPU preflight, weight provenance, smoke test (6A)
│   ├── train_detection_baseline.py # Run and record the D0 detection baseline (6B)
│   ├── freeze_detection_experiments.py # Freeze the phase 7 comparison protocol (7A)
│   ├── train_detection_experiment.py # Run one frozen phase 7 candidate (7B+)
│   ├── audit_segmentation_adapter.py # Measure what the YOLO seg format costs (8A)
│   ├── freeze_segmentation_baseline.py # Select the architecture, freeze S0, smoke test (8B)
│   ├── train_segmentation_baseline.py # Run S0 once, validate it, run the IoU diagnostic (8C)
│   ├── analyze_segmentation_errors.py # Per-instance S0 error attribution, trains nothing (8D)
│   └── freeze_segmentation_comparison.py # Canonical evaluator, S0 reference, S1 freeze (8E)
├── src/construction_safety_vision/
│   ├── config.py              # Strict typed configuration loading
│   ├── paths.py               # Repository layout, Colab support, long-path handling
│   ├── provenance.py          # Hashing and run provenance records
│   ├── splits.py              # Split identifiers and the holdout guard
│   ├── experiment.py          # Experiment protocols, declared before they run
│   ├── detection_results.py   # Result manifests, metric extraction, experiment fingerprints
│   ├── detection_comparison.py # Phase 7 support rule, selection metric, selection logic
│   ├── detection_freeze.py     # Frozen detector identity, checkpoint verification (7D)
│   ├── data/segmentation_adapter.py  # Canonical COCO masks -> YOLO seg rows (8A)
│   ├── data/segmentation_fidelity.py # Round-trip mask metrics and attribution (8A)
│   ├── segmentation_experiment.py # The S0 protocol schema, parsed strictly (8B)
│   ├── segmentation_run.py        # Runtime view, native fitness, checkpoint records (8C)
│   ├── mask_iou_evaluation.py     # The direct instance-mask IoU diagnostic (8C)
│   ├── segmentation_error_analysis.py # Error taxonomy, strata, deterministic review (8D)
│   ├── canonical_evaluation.py    # COCOeval against canonical masks, common to all models (8E)
│   ├── segmentation_comparison.py # The frozen S0-vs-S1 policy and one-variable contract (8E)
│   ├── detection_run.py       # Shared run primitives: weights, optimizer evidence, figures
│   └── data/                  # Acquisition, COCO inspection, geometry, drift, decision,
│                              # split search, the frozen split + its access layer,
│                              # task materialisation and the YOLO detection adapter
└── tests/
```

## Development principles

- **Correctness > Simplicity > Maintainability > Performance > Cleverness.**
- **The holdout is sacred.** The test split is read once, after the models are
  frozen. Enforced in code by `splits.assert_split_allowed`, which requires both
  an in-code opt-in and an environment unlock.
- **No fabricated evidence.** Every number comes from an executed run with a
  provenance record. Unknown values are written `TBD`, never estimated.
- **Causation requires an experiment.** Claims that a change caused an
  improvement require a controlled comparison; otherwise they are labelled as
  hypotheses.
- **Raw data is immutable.** Corrections are code in the pipeline, never manual
  edits.
- **Configuration is a file, not a memory.** Every hyperparameter lives in
  `configs/` and is snapshotted into each run's provenance record.
- **Dependencies must be justified.** Heavy frameworks arrive with the phase
  that needs them.
- **Documentation describes reality.** Planned work is labelled as planned.

## Reproducibility goals

The target is that a third party, starting from a clean clone, can reproduce
every reported number without contacting the author.

- Environment pinned by `pyproject.toml` + `uv.lock`, with a single documented
  setup command.
- One global seed, recorded in every run.
- Frozen split manifests, listing every image ID and its content hash.
- Provenance records tying each artifact to its inputs, configuration, code
  commit and environment.
- Datasets and checkpoints re-obtainable from recorded sources and hashes rather
  than committed.
- Notebooks that run top to bottom on a fresh Colab runtime.
- A final reproducibility audit performed from a clean clone (phase 14).

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.11+.

```bash
git clone https://github.com/Novachrono117/Construction-Safety-Vision-PPE-Detection-Instance-Segmentation-Video-Analytics.git
cd Construction-Safety-Vision-PPE-Detection-Instance-Segmentation-Video-Analytics
uv sync
cp .env.example .env          # then set ROBOFLOW_API_KEY to acquire the dataset
uv run python scripts/check_environment.py
```

Acquire the dataset. The key is read from the environment only, sent as an
`Authorization` header rather than a URL parameter, and never written to any
file, log or provenance record:

```bash
uv run python scripts/download_dataset.py
uv run python scripts/inspect_dataset.py
```

Reproduce the audit (phase 4). The first two steps contact the provider
read-only; the rest run offline:

```bash
uv run python scripts/fetch_source_inventory.py
uv run python scripts/download_source_images.py
uv run python scripts/audit_source_dataset.py
uv run python scripts/audit_bbox_consistency.py
uv run python scripts/eda_source_dataset.py
uv run python scripts/write_audit_reports.py
uv run python scripts/build_review_package.py
uv run python scripts/record_manual_audit.py
```

Reproduce the canonical-snapshot decision (phase 5A). Only the first step
contacts the provider, and only to read:

```bash
uv run python scripts/recover_source_geometry.py
uv run python scripts/map_v4_sources.py
uv run python scripts/analyze_annotation_drift.py
uv run python scripts/build_drift_figures.py
uv run python scripts/resolve_canonical_snapshot.py
```

Build the canonical modelling population (phase 5B). Fully offline:

```bash
uv run python scripts/analyze_fragment_rule.py
uv run python scripts/build_modeling_population.py
uv run python scripts/build_remaining_duplicate_review.py
```

Search for provisional split candidates (phase 5C.1). Selects and freezes
nothing:

```bash
uv run python scripts/optimize_split_candidates.py
```

Checks:

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest
```

## Results

Not available. No model has been trained and no evaluation has been run. This
section will be filled by phases 10-12, from committed metrics files.

## License and attribution

The **software** license of this repository is not yet chosen.

The **dataset** is a separate matter and is licensed **CC BY 4.0** by AGIs
Workspace. It is not redistributed here; `scripts/download_dataset.py` obtains it
from the original source. Attribution:

```text
Construction PPE Compliance Detection [dataset], version 4.
AGIs Workspace, Roboflow Universe. Licensed CC BY 4.0.
https://universe.roboflow.com/agis-workspace-8gs52/construction-ppe-compliance-detection/dataset/4
```
