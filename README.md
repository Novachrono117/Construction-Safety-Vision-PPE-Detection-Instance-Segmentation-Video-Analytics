# Construction Safety Vision - PPE Detection, Instance Segmentation & Video Analytics

> **Status: canonical task datasets materialised (phase 5D of 14).** The dataset
> is acquired, hashed, structurally verified, audited automatically (4A) and
> reviewed visually by people (4B). The canonical annotation snapshot is resolved
> (5A), the modelling population and its indivisible split units are built (5B),
> the **group-aware and class-aware constrained split is frozen at 303 / 65 / 65**
> over 433 modelling images (5C.2), and the **COCO detection and instance-
> segmentation views of `train` and `validation` are reproducibly materialised**
> (5D). The provider's split was **rejected for the final protocol** and is not
> reused. The `test` split is a **locked holdout**: it has never been evaluated,
> inspected or materialised. **No model has been trained and no results exist
> yet.** Every metric section is intentionally empty until a real, recorded run
> produces it.

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
| Model-specific adapter | `NOT_YET_SELECTED`. No YOLO labels written; a future adapter must pass a geometry-fidelity audit first. |
| Detection model | Not trained. |
| Segmentation model | Not trained. |
| Metrics | **None.** No evaluation has been run. |
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
cross-task alignment validators, the test suite, and the planning documents
(`reports/rubric_contract.md`, `reports/roadmap.md`, `CLAUDE.md`).

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
│   └── task_materialization.yaml # Copy, naming, bbox and geometry policies (5D)
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
│   └── figures/               # Contact sheets and analytical plots
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
│   └── materialize_task_datasets.py # Canonical COCO detection + segmentation views (5D)
├── src/construction_safety_vision/
│   ├── config.py              # Strict typed configuration loading
│   ├── paths.py               # Repository layout, Colab support, long-path handling
│   ├── provenance.py          # Hashing and run provenance records
│   ├── splits.py              # Split identifiers and the holdout guard
│   └── data/                  # Acquisition, COCO inspection, geometry, drift, decision,
│                              # split search, and the frozen split + its access layer
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
