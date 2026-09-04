# Roadmap

Version: 1.6 · Current phase: **6 - Detection baseline** (6A complete: adapter, runtime and D0 protocol frozen; 6B - the D0 run - next)

Fourteen phases, executed in order. Each phase has a validation gate: the gate
must pass before the next phase starts, and a gate is passed only by evidence
that exists in the repository. "Academic mapping" links the phase to the rubric
criteria defined in [`rubric_contract.md`](rubric_contract.md).

A phase may be revisited, but never silently: reopening a completed phase
invalidates the gates that depended on it, and the reopening is recorded in the
change log at the bottom of this file.

## Status overview

| # | Phase | Status |
| --- | --- | --- |
| 1 | Scope and rubric contract | done |
| 2 | Repository foundation | done |
| 3 | Dataset acquisition and provenance | done |
| 4 | Dataset/annotation audit and EDA | complete (4A automated, 4B visual review) |
| 5 | Split freeze and task-specific dataset generation | complete (5A, 5B, 5B.1, 5C.1, 5C.2 split frozen 303/65/65, 5D COCO development datasets materialised) |
| 6 | Detection baseline | 6A complete (YOLO detection adapter, GPU runtime, D0 protocol frozen, smoke test passed) · 6B **next**: run D0. Not run, no metrics |
| 7 | Detection experiments and model freeze | not started |
| 8 | Segmentation baseline | not started |
| 9 | Segmentation experiments and model freeze | not started |
| 10 | Controlled validation comparison | not started |
| 11 | One-shot final test evaluation | not started |
| 12 | Error analysis | not started |
| 13 | Video inference and tracking | not started |
| 14 | Submission package and reproducibility audit | not started |

---

## Phase 1 - Scope and rubric contract

- **Objective.** Convert the assignment rubric into an explicit, verifiable
  contract, so that "done" has an objective meaning for every deliverable, and
  fix the experimental protocol before any data is seen.
- **Inputs.** Course rubric; assignment brief; candidate dataset description.
- **Outputs.** `reports/rubric_contract.md`; this roadmap; the protocol rules
  recorded in `CLAUDE.md`.
- **Validation gate.** Every rubric criterion has a weight, a planned artifact,
  a responsible phase and a falsifiable definition of done. The holdout protocol
  is written down before any dataset exists.
- **Academic mapping.** Foundation for C1-C7; graded indirectly through C6.

## Phase 2 - Repository foundation

- **Objective.** Build a minimal, professional, reproducible skeleton: `src`
  layout, pinned environment, linting, tests, typed configuration, and the code
  primitives the protocol depends on (paths, splits guard, provenance).
- **Inputs.** Phase 1 outputs; tooling decisions (Python, uv, ruff, pytest).
- **Outputs.** `pyproject.toml`, `uv.lock`, `.gitignore`, `.gitattributes`,
  `.env.example`, `configs/project.yaml`, `src/construction_safety_vision/`
  (`paths`, `config`, `splits`, `provenance`), `tests/`,
  `scripts/check_environment.py`, `README.md`, `CLAUDE.md`.
- **Validation gate.** `ruff check`, `ruff format --check` and `pytest` all pass
  on a clean environment; the holdout guard is covered by tests; the README
  describes only what exists.
- **Academic mapping.** C6 (report and repository).

## Phase 3 - Dataset acquisition and provenance

- **Objective.** Acquire the source dataset in its canonical instance
  segmentation form and record exactly what was obtained, from where, under
  which license.
- **Inputs.** `configs/project.yaml` (declared dataset); `ROBOFLOW_API_KEY` from
  the environment.
- **Outputs (as delivered).** `scripts/download_dataset.py` and
  `scripts/inspect_dataset.py`; `src/construction_safety_vision/data/`
  (`roboflow`, `acquisition`, `coco`, `versioning`); the archive and its record
  in `data/external/` (archive untracked, record committed); the extracted
  export in `data/raw/` (untracked);
  [`reports/dataset_provenance.md`](dataset_provenance.md) and
  `reports/dataset_provenance.json`.
- **Validation gate (passed).** The download is reproducible from the recorded
  version and its SHA-256; annotation files are hashed; the license (CC BY 4.0)
  permits public release of derived work and of the report;
  `configs/project.yaml` carries the exact provider coordinates and version URL;
  the export is structurally sound (no dangling image or category references, no
  duplicate ids, no missing files); the difference between the project-level and
  version-level image counts is explained by evidence rather than assumed.
- **Academic mapping.** C1 (problem and dataset).

> **Key outcome.** Version 4 contains 742 images but only **436 independent
> source images**: the train split is offline-augmented x2 (306 -> 612) while
> validation and test are unchanged. Every later phase must treat 436 as the
> independent population. Provenance-level facts are verified; annotation
> *quality* is not, and remains phase 4's job.

## Phase 4 - Dataset/annotation audit and EDA

- **Objective.** Replace every planning hypothesis with a measurement, and find
  the defects that would otherwise silently corrupt the evaluation.
- **Inputs.** `data/raw/`; the acquisition provenance record;
  `reports/dataset_provenance.md` (structure and open items already established).
- **Outputs.** `notebooks/01_dataset_audit.ipynb`; `reports/dataset_audit.md`;
  audit tables in `data/interim/`; figures for class distribution, object size
  distribution and objects per image.
- **Validation gate.** The **436 independent source images** are recovered from
  the 742 exported records, so that augmented variants are never counted as
  independent samples; annotation integrity checked (empty or degenerate
  polygons, out-of-bounds coordinates, self-intersections, RLE decoding,
  class-name inconsistencies); the 28 image records with no annotations are
  classified as deliberate negatives or as missing labels; exact duplicates
  detected by content hash; near duplicates detected by perceptual hash with a
  documented threshold; potential video-frame sequences identified and grouped;
  the >= 300 annotated image requirement re-confirmed on the de-duplicated
  population; the absence of `vest_loose` from the provider's test split is
  quantified and its consequence for reporting stated.
- **Academic mapping.** C1; feeds the limitations section of C4 and C6.

> Phase 3 set `dataset.verified: true` for *provenance-level* facts only
> (source, license, format, class list, artifact integrity). Annotation quality,
> duplication and split suitability are decided here.

### Phase 4A - automated audit (complete)

Delivered `scripts/audit_source_dataset.py`, `scripts/audit_bbox_consistency.py`,
`scripts/eda_source_dataset.py`, `scripts/build_review_package.py` and
`scripts/write_audit_reports.py`; `reports/dataset_audit_report.md`,
`reports/eda_report.md`, `reports/bbox_consistency_audit.md`, their JSON
counterparts, `reports/source_image_manifest.jsonl` and the visual-review
package (`reports/manual_review_manifest.csv` plus `reports/figures/review_*`).

### Phase 4B - manual visual audit (complete)

- **Objective.** Answer the semantic questions the automated audit deliberately
  left open, and record the answers so they can be traced rather than trusted.
- **Inputs.** The phase 4A contact sheets and manifests; judgements made by the
  project owner with a technical reviewer, outside this repository.
- **Outputs (as delivered).** `scripts/record_manual_audit.py` and
  `src/construction_safety_vision/data/manualaudit.py`;
  [`reports/manual_audit_report.md`](manual_audit_report.md) and
  `reports/manual_audit_decisions.csv` (32 decisions).
- **Validation gate (passed).** Every recorded subject resolves to an id in the
  phase 4A manifests and appears on the figure its decision cites; every verdict
  is drawn from a controlled vocabulary; semantic-duplicate group ids are a
  function of their sorted members alone; no absolute path or credential
  material is present. Verified by
  `uv run python scripts/record_manual_audit.py --check`.
- **Outcome.** Six cross-split semantic duplicate pairs confirmed; the provider
  split classified `UNSUITABLE_FOR_FINAL_PROTOCOL`; the dataset itself
  `ACCEPTED_WITH_DOCUMENTED_LIMITATIONS`. Twelve entry constraints (`P5-01` to
  `P5-12`) are handed to phase 5. Nothing was split, excluded, converted or
  deleted.

## Phase 5 - Split freeze and task-specific dataset generation

- **Objective.** Freeze one canonical train/validation/test partition and derive
  the detection and segmentation views from it, so both tasks are trained and
  evaluated on identical images.
- **Inputs.** Audited dataset; duplicate and sequence groups from phase 4;
  `split_ratios` and `seed` from the configuration.
- **Entry conditions.** The twelve constraints `P5-01` to `P5-12` recorded in
  [`manual_audit_report.md`](manual_audit_report.md). In particular the provider
  split is not reusable and the six confirmed semantic-duplicate groups must stay
  intact. The current-source-vs-v4 annotation drift was the remaining blocker and
  is now resolved by subphase 5A.

### Phase 5A - canonical annotation snapshot (complete)

- **Question.** Which reproducible annotation state becomes the source of truth?
  Not which images, not which split - only which annotations are authoritative.
- **Outcome.** `CURRENT_COMPLETE_GEOMETRY`. The live source project was recovered
  read-only: 436 source images, 2031 annotations, 2029 of them carrying complete
  instance-segmentation geometry (1007 polygon, 1022 RLE) in **original image
  coordinates**. Phase 4A's "1022 annotations lack geometry" was a consumption
  gap; the geometry travels inline as base64-wrapped zlib-compressed COCO RLE.
- **Verification.** Every recovered instance was re-measured and checked against
  the provider's declared area and box; all 2029 agree to 0.0 px.
- **Version 4 was viable and was not chosen.** All 436 source images map to
  exactly one non-augmented v4 representation, so option A was available. It was
  rejected because v4 geometry is expressed after a stretch resize to 640x640
  that would have to be inverted for every annotation.
- **Drift.** Net +70 annotations (76 added, 6 removed) across 67 images. Every
  addition lies at least 80% inside an existing same-class annotation at a median
  0.44% of its area, so none covers a previously unlabelled object. `vest_loose`
  is unchanged.
- **Outputs.** [`canonical_annotation_decision.md`](canonical_annotation_decision.md),
  `canonical_annotation_manifest.json`,
  [`annotation_drift_report.md`](annotation_drift_report.md),
  `annotation_drift.csv`, `v4_source_mapping.csv`, `source_geometry_summary.json`,
  and review sheets L, M and N under `figures/`.
- **Explicitly not done.** No split, no exclusion, no duplicate grouping, no
  holdout freeze, no YOLO labels, no coordinate conversion, no model.

### Phase 5B - canonical modelling population (complete)

- **Question.** Which images and annotations may enter a future split, and which
  images must stay together when one is designed? No split is created here.
- **Population.** 436 source images minus 3 out-of-domain exclusions confirmed by
  phase 4B = **433 modelling images**, of which 14 carry no annotation and are
  retained deliberately as negatives. The 3 excluded images stay on disk and stay
  in the source provenance population; exclusion is logical, never physical.
- **Annotations.** All 2031 canonical annotations are accounted for. The 3
  excluded images were zero-instance, so **0** annotations were lost with them.
  The 2 `VALID_BUT_UNSUPPORTED_GEOMETRY` records are materialised as four-corner
  rectangles clipped to the canvas and labelled
  `geometry_origin = SYNTHETIC_FROM_PROVIDER_BBOX`.
- **Split units.** After phase 5B.1: **422 groups**, being 11 confirmed groups
  (22 images) plus 411 singletons. Groups are connected components of the
  confirmed relations, so a transitive chain forms one group rather than
  overlapping pairs.
- **Rare class.** `vest_loose` is untouched: 45 instances across 8 images, none of
  them in a duplicate group.
- **Nested annotations: all retained.** Phase 5B was asked to turn the phase 5A
  observation about the 76 annotations added since version 4 into a deterministic
  geometric rule. No rule reaches an acceptable precision and recall together,
  because the 76 are **not one kind of thing**: roughly half are degenerate
  slivers, and the rest are corrections that *improve* the labels. This corrects
  the phase 5A reading, which called all 76 fragments adding no coverage. The
  owner set `fragment_rule_status = REJECTED_FOR_AUTOMATIC_FILTERING`: **0
  excluded, all 2031 retained**, with the 34 evaluated candidates carrying the
  descriptive flag `NESTED_SAME_CLASS_CANDIDATE`. The failed experiment is kept
  as negative evidence in [`fragment_rule_report.md`](fragment_rule_report.md).
  Note that the 76 are a historical drift reference set, not error ground truth.
- **Outputs.** [`canonical_modeling_population_report.md`](canonical_modeling_population_report.md),
  `canonical_modeling_manifest.json`, `canonical_modeling_population.csv`,
  `canonical_annotation_actions.csv`, `group_manifest.csv`,
  `group_split_features.csv`, `unconfirmed_group_candidates.csv`,
  `fragment_rule_analysis.csv`, and `figures/fragment_rule_distribution.png`.
- **Explicitly not done.** No split, no holdout, no YOLO labels, no coordinate
  conversion, no model. The provider's rejected split appears in no artifact a
  split designer reads.

### Phase 5B.1 - near-duplicate disposition (complete)

- **Why.** Phase 4B reviewed only the cross-split candidates, then rejected the
  provider split. With the split rebuilt from scratch, a same-split duplicate
  constrains it just as much.
- **Reconciliation.** Phase 4A raised 11 candidates: 6 cross-split, 5 same-split.
  `review_h` showed all 6 cross-split pairs. `review_g` showed the 8 candidates
  with the smallest perceptual distance - 4 of those 6 plus 4 same-split - so it
  displayed 8 pairs of which only 4 were new. The 11th candidate has the largest
  distance of all and fell outside the cap, appearing on neither sheet.
- **Decided.** All 5 same-split pairs dispositioned as `manual_dup_007` to
  `manual_dup_011` in [`manual_audit_decisions.csv`](manual_audit_decisions.csv),
  addendum 14b in [`manual_audit_report.md`](manual_audit_report.md): 4
  `EXACT_SEMANTIC_DUPLICATE` (HIGH) and 1 `NEAR_DUPLICATE_SAME_SCENE` (MEDIUM).
- **The two findings differ and both group.** The same frame stored twice, versus
  the same worker and scene at a different moment. Both are split-indivisible
  because grouping serves statistical independence, not image identity; the
  distinction is preserved in `group_basis`.
- **The last candidate had never been shown to a reviewer.** It was drawn alone in
  `figures/review_o_remaining_near_duplicates.jpg` and decided there.

### Phase 5C.1 - provisional split candidates (complete)

- **What it did.** Searched for candidate train/validation/test assignments over
  the 422 indivisible groups and compared them. **It selected nothing and froze
  nothing.**
- **Result.** Six candidates, every one hitting the exact 70/15/15 target of
  303 / 65 / 65 images with all hard constraints satisfied: all five classes in
  all three splits at image and instance level, `vest_loose` at 5/1/2 (five
  candidates) or 4/2/2 (one), negatives at 10/2/2.
- **Objective.** Normalised per class and averaged, so a 914-instance class
  cannot outweigh a 45-instance one; each component reported separately rather
  than folded into one opaque score.
- **Determinism.** 192 restarts seeded from the project seed 42; re-running
  reproduces identical assignments, fingerprints, scores and ranking.
- **Correction carried forward.** The brief stated no `vest_loose` image belongs
  to a duplicate group. Phase 5B.1 made that false: two of the eight are in
  `manual_dup_010`, so the class occupies 7 indivisible units and moves in
  chunks. All declared families stay feasible under it.
- **Outputs.** [`split_candidate_report.md`](split_candidate_report.md),
  `split_candidates/summary.csv`, one assignment file per candidate, and
  `configs/split_search.yaml`.
- **Explicitly not done.** No candidate selected, no holdout frozen, no holdout
  fingerprint, no candidate test set evaluated, no YOLO dataset, no model.

### Phase 5C.2 - split selection and freeze (complete)

- **What it did.** Promoted one predeclared candidate to the project's
  authoritative split and locked the holdout. **It froze membership only** - no
  image was copied, moved, resized or preprocessed, no label file was written,
  and `data/processed/` is untouched.
- **Selection.** `candidate_001`, chosen by a person reviewing the predeclared
  deterministic candidates (`HUMAN_REVIEW_OF_PREDECLARED_DETERMINISTIC_CANDIDATES`,
  decision source `PROJECT_OWNER_REVIEW`). It coincides with
  `algorithmic_best_candidate`; the two are recorded separately because a
  coincidence of outcome does not replace the review step.
- **Result.** **303 / 65 / 65 images** over **294 / 63 / 65 groups**, 422 groups
  in total, 2031 annotations at 1422 / 304 / 305. All five classes present in
  all three splits at image and instance level. `vest_loose` at **5 / 1 / 2
  images** and **30 / 8 / 7 instances**; negatives at 10 / 2 / 2. No group
  crosses a boundary; no excluded out-of-domain image appears.
- **Verification before writing.** The freeze re-derives the candidate's
  assignment digest, cross-checks it against the phase 5C.1 summary, and checks
  the phase 5B population and group fingerprints. Any disagreement is a stop:
  nothing is written and the phase classifies `INVALID_CANDIDATE`.
- **Fingerprints.** `split_assignment_sha256` over sorted
  `(group_id, source_image_id, split)`; `holdout_sha256` over the holdout's group
  ids, image ids and `modeling_population_sha256`. Both exclude timestamps,
  paths, labels, metrics and the provider split. The freeze is idempotent:
  re-running produces byte-identical artifacts.
- **Holdout.** Locked. Two independent opt-ins are required
  (`allow_test=True` **and** `CSVISION_ALLOW_TEST_SPLIT=1`), neither sufficient
  alone, and `FrozenSplits` routes every request through the existing guard. The
  variable is not set anywhere in this repository.
- **Outputs.** [`split_freeze_report.md`](split_freeze_report.md),
  `split_manifest.json`, `final_split_assignments.csv`,
  `split_candidates/selection.csv`, `split_freeze.provenance.json`,
  `configs/split_freeze.yaml`, and the
  `construction_safety_vision.data.split_freeze` module.
- **Explicitly not done.** No model, no YOLO or COCO dataset, no physical
  train/val/test image copies, no inference, no holdout evaluation and no
  holdout inspection.
- **Academic mapping.** C1.

### Phase 5D - task-specific dataset generation (complete)

- **What it did.** Materialised the frozen membership into two COCO views of the
  same images and the same objects, for **`train` and `validation` only**. The
  claim it makes is that it added nothing: same bytes, same annotations, same
  class order.
- **Result.** **368 development images and 1726 annotations** (train 303/1422,
  validation 65/304), matching the frozen membership exactly. Zero-instance
  images retained as annotation-free records (10 train, 2 validation).
- **Canonical formats.** `canonical_detection_format: COCO`,
  `canonical_segmentation_format: COCO_INSTANCE_SEGMENTATION`,
  `model_specific_adapter: NOT_YET_SELECTED`. **No YOLO labels were written**:
  the canonical state holds polygon *and* compressed RLE, and only COCO carries
  both, so a YOLO conversion would approximate the ground truth before a model
  had been chosen.
- **Images byte-preserved.** 368/368 byte-identical, hashed on both sides after a
  binary copy. No resize, crop, re-encode, EXIF rotation or colour conversion; a
  destination holding different bytes is a hard failure, never a silent
  overwrite.
- **Boxes derived from segmentation.** Every detection box comes from the
  canonical geometry through the phase 4A implementation. Cross-checked against
  the box phase 5A measured independently from the same geometry: **max delta
  0.0 px** over all 1726 annotations. The provider's bbox is used nowhere.
- **Geometry round-trip verified.** The emitted segmentation file is read back
  from disk and compared with the canonical state - RLE by decoded mask,
  polygons coordinate by coordinate. **1726 checked, 1726 matched, 0
  mismatches**: 843 polygons, 881 RLE masks, 2 synthetic rectangles.
- **Cross-task alignment verified.** Both views hold the same COCO image ids,
  source image ids, annotation ids, categories and boxes, per split.
- **Deterministic.** COCO ids come from a sorted ordering of the whole modelling
  population (so the holdout can be materialised later without renumbering), no
  timestamp enters an emitted file, and re-running reproduces every artifact byte
  for byte.
- **Holdout untouched.** `test` is `NOT_MATERIALIZED_PROTECTED_HOLDOUT`. No test
  image directory, COCO file, contact sheet or statistic was produced, and no new
  knowledge about it was computed. The path that will materialise it is the same
  function used here and needs both opt-ins; it is exercised only against
  synthetic fixtures.
- **Outputs.** [`task_materialization_report.md`](task_materialization_report.md),
  `task_dataset_manifest.json`, `task_materialization.provenance.json`,
  `configs/task_materialization.yaml`, the
  `construction_safety_vision.data.materialization` and
  `.coco_materialization` modules, and the git-ignored datasets under
  `data/processed/canonical/`.
- **Explicitly not done.** No model, no training framework installed, no
  inference, no YOLO labels, no holdout materialisation.
- **Academic mapping.** C1, and the alignment requirement of C3.

> **Future model-adapter gate.** If a YOLO segmentation stack is selected later,
> then before any training run the adapter must convert the canonical masks to
> the required polygon format, rasterise the result, compare it against the
> canonical masks, report per-instance mask IoU and area error, identify
> disconnected-component and hole cases, and be rejected or reconsidered if the
> loss is material. That audit is a phase 8 / model-adapter concern and has not
> been performed.

## Phase 6 - Detection baseline

- **Objective.** Establish a first honest reference point with default settings,
  before any tuning, so later changes can be judged against something.
- **Inputs.** `data/processed/detection/`; a pretrained modern detector;
  `configs/detection_baseline.yaml`.
- **Outputs.** Baseline run artifacts (untracked) plus their provenance record;
  validation metrics in `reports/metrics/detection_baseline.json`; training
  curves; a short baseline note.
- **Validation gate.** The run is reproducible from the committed configuration;
  every hyperparameter is recorded; metrics are computed on validation only;
  training and validation curves are inspected for obvious failure (divergence,
  collapse, degenerate predictions).
- **Academic mapping.** C2.

### Phase 6A - adapter, runtime and baseline protocol (complete)

- **What it did.** Prepared the first detection experiment and **deliberately did
  not run it**: a working GPU runtime, a lossless model-specific detection
  adapter, the frozen D0 protocol, and a minimal smoke test.
- **ML dependencies installed.** torch 2.11.0+cu128, torchvision 0.26.0+cu128,
  ultralytics 8.4.138, locked in `uv.lock`. torch resolves from the CUDA 12.8
  build index declared in `pyproject.toml`, for a hardware reason rather than a
  preference: the GPU here is Blackwell (`sm_120`) and only cu128 builds carry
  kernels for it.
- **Runtime verified, not assumed.** `torch.cuda.is_available()` is a weak claim,
  so the preflight executes a matmul checked against the CPU, a convolution
  backward pass and an AMP autocast step. RTX 5070 Laptop, `sm_120`, 7.96 GiB,
  driver 610.88, `sm_120` present in the compiled arch list, AMP working. A GPU
  visible but unusable would have been reported `BLOCKED_FOR_GPU` rather than
  silently downgraded to CPU.
- **Lossless detection adapter.** 368 images and 1726 labels derived from the
  canonical COCO detection dataset. Every box converted, written, **read back
  from the label file on disk**, decoded and compared: **1726/1726 within
  tolerance, 0 mismatches, max delta 1.47e-06 px** against a declared 1e-4 px.
  Adapter images byte-identical to the canonical ones; class indices the frozen
  map, verified against `class_map_sha256`; `object` absent; the 12 negatives
  preserved as empty label files.
- **Detection only.** No YOLO segmentation labels were written. RLE masks cannot
  become YOLO polygons without loss, and that conversion needs its own fidelity
  audit; a box has no such problem, which is why this adapter can be *proven*
  lossless rather than assumed to be.
- **D0 protocol frozen before the experiment** in
  `configs/detection_baseline.yaml`: YOLO11n pretrained (`yolo11n.pt`,
  fingerprinted), imgsz 640, batch 16, 100 epochs, seed 42, every hyperparameter
  stated, checkpoint rule `ULTRALYTICS_BEST_ON_VALIDATION_FITNESS` fixed in
  advance, and the metric hierarchy declared - primary `mAP@0.50:0.95`, then
  `mAP@0.50`, precision, recall. The parser rejects a protocol that names the
  holdout or invents a primary metric.
- **`vest_loose` limitation recorded before any number exists.** 1 validation
  image, 8 instances: not a usable selection signal, never tuned against, always
  reported with an explicit small-sample caveat.
- **Smoke test, not an experiment.** One epoch on train and validation: OK in
  62 s, 2.4 GiB peak, checkpoints written. Marked `NON_EXPERIMENTAL` /
  `DO_NOT_REPORT_AS_MODEL_RESULT`; **no metric from it is recorded anywhere** and
  nothing was tuned from it. batch 16 needed no reduction.
- **Outputs.** [`detection_adapter_report.md`](detection_adapter_report.md),
  `detection_adapter_manifest.json`,
  [`detection_runtime_report.md`](detection_runtime_report.md), the two
  provenance records, `configs/detection_adapter.yaml`,
  `configs/detection_dataset.template.yaml`, `configs/detection_baseline.yaml`,
  and the `yolo_detection_adapter` and `experiment` modules.
- **Explicitly not done.** The full D0 run, any hyperparameter tuning, any
  segmentation work, any holdout access. The holdout has no adapter, no
  directory, no label and no key in the Ultralytics dataset descriptor.
- **Academic mapping.** C2, and the reproducibility requirement of C6.

### Phase 6B - D0 detection baseline run (not started)

- **Entry conditions.** The frozen D0 protocol, the audited adapter, and a
  verified GPU runtime - all delivered by 6A.
- **Outputs.** The D0 run under the git-ignored `artifacts/`, its metrics against
  the predeclared hierarchy, PR curve, confusion matrix and loss curves, a
  provenance record, and a written baseline report.
- **Validation gate.** Trained on `train` only; every reported metric computed on
  `validation` only; the checkpoint chosen by the predeclared rule rather than by
  inspection; the resolved Ultralytics arguments logged in full; `vest_loose`
  reported with its small-sample caveat. The holdout is not touched.

## Phase 7 - Detection experiments and model freeze

- **Objective.** Improve on the baseline through controlled variations, then
  freeze one final detection model.
- **Inputs.** Baseline results; a written list of hypotheses to test.
- **Outputs.** `configs/detection_*.yaml` per run;
  `reports/detection_experiments.md` with a comparison table; the frozen
  checkpoint identified by hash; the selection rationale.
- **Validation gate.** Each run differs from its comparison point by documented
  factors only; the same seed policy is applied throughout; selection uses
  validation metrics only; run-to-run variability is acknowledged and, where a
  claim depends on a small difference, either supported by repeated runs or
  stated as a hypothesis; the frozen checkpoint is recorded and not retrained
  afterwards.
- **Academic mapping.** C2; feeds C4.

## Phase 8 - Segmentation baseline

- **Objective.** Establish the instance-segmentation reference point on the same
  frozen splits.
- **Inputs.** `data/processed/segmentation/`; a pretrained segmentation model;
  `configs/segmentation_baseline.yaml`.
- **Outputs.** Baseline run artifacts and provenance; validation mask metrics in
  `reports/metrics/segmentation_baseline.json`; qualitative mask samples.
- **Validation gate.** Same gate as phase 6, plus a re-verified image-ID
  alignment with the detection dataset and a visual check that predicted masks
  are in the correct coordinate space.
- **Academic mapping.** C3.

## Phase 9 - Segmentation experiments and model freeze

- **Objective.** Improve segmentation through controlled variations and freeze
  one final model.
- **Inputs.** Segmentation baseline; hypothesis list.
- **Outputs.** `configs/segmentation_*.yaml`;
  `reports/segmentation_experiments.md`; the frozen checkpoint and its hash.
- **Validation gate.** Same gate as phase 7, applied to mask metrics; box and
  mask metrics are reported separately and never conflated.
- **Academic mapping.** C3; feeds C4.

## Phase 10 - Controlled validation comparison

- **Objective.** Produce the complete, comparable validation-set evaluation of
  both frozen models under one evaluation protocol, and confirm the pipeline is
  ready for the single holdout run.
- **Inputs.** Both frozen checkpoints; the validation split; one evaluation
  implementation used for every model.
- **Outputs.** `reports/evaluation.md` (validation section); metrics JSON per
  model; confusion matrices; precision-recall curves.
- **Validation gate.** mAP@0.5, mAP@0.5:0.95, IoU, precision and recall are
  computed for both tasks with the matching rule, IoU threshold, confidence
  threshold and averaging scheme stated; the confusion matrix documents how
  unmatched predictions and unmatched ground truth are counted; the evaluation
  code path is identical across models; a dry run on validation proves the test
  evaluation will execute without needing a second attempt.
- **Academic mapping.** C4.

## Phase 11 - One-shot final test evaluation

- **Objective.** Measure generalisation on the locked holdout, exactly once.
- **Inputs.** Frozen checkpoints; frozen evaluation code; the test split,
  unlocked by the documented double opt-in.
- **Outputs.** `reports/metrics/test_*.json`; the test section of
  `reports/evaluation.md`; a provenance record capturing the unlock, the
  timestamp, the checkpoint hashes and the code commit.
- **Validation gate.** No model, threshold or preprocessing choice changes after
  this run; validation and test results are reported side by side, and any gap
  is discussed rather than explained away; the run is recorded as the single
  holdout evaluation. If the pipeline fails mid-run for technical reasons, the
  incident is documented in full before any re-run.
- **Academic mapping.** C4.

## Phase 12 - Error analysis

- **Objective.** Explain where and how the models fail, qualitatively and
  quantitatively, without inventing causes.
- **Inputs.** Predictions from phases 10 and 11; the audit findings from phase 4.
- **Outputs.** `reports/error_analysis.md`; a curated FP/FN gallery in
  `reports/figures/errors/`; per-class and per-condition breakdowns (object
  size, occlusion, crowding, lighting) where the data supports them.
- **Validation gate.** At least one false positive and one false negative per
  class are analysed with image, prediction, ground truth and a labelled
  hypothesis; systematic patterns are supported by counts, not impressions;
  class-confusion pairs from the confusion matrix are explained or explicitly
  left open; every causal statement is marked as hypothesis unless an experiment
  supports it.
- **Academic mapping.** C4.

## Phase 13 - Video inference and tracking

- **Objective.** Demonstrate the frozen models on real construction footage and
  characterise temporal behaviour. Tracking is a bonus, attempted only after the
  mandatory deliverables are complete.
- **Inputs.** Frozen checkpoints; a construction video of >= 30 s with a
  recorded source and license.
- **Outputs.** `scripts/run_video_inference.py`; the annotated output video;
  `reports/video_analysis.md` with measured FPS and hardware; optionally
  ByteTrack integration and `reports/tracking_notes.md`.
- **Validation gate.** The video meets the duration and provenance requirements;
  inference runs end to end from a documented command; FPS is measured, not
  estimated; at least three temporal failure modes are described with
  timestamps; if tracking is added, identity stability is demonstrated and its
  effect on those failure modes is discussed.
- **Academic mapping.** C5; bonus B1.

## Phase 14 - Submission package and reproducibility audit

- **Objective.** Assemble the deliverables and prove the repository reproduces
  from a clean clone.
- **Inputs.** All prior outputs.
- **Outputs.** `reports/technical_report.md` (and PDF); final `README.md`;
  Colab-verified notebooks; `reports/pitch_script.md`; the recorded pitch;
  `reports/reproducibility_audit.md`.
- **Validation gate.** A clean clone plus the documented setup commands
  reproduces the environment; at least one notebook runs top to bottom on a
  fresh Colab runtime; every number in the report resolves to a committed
  metrics file and provenance record; no dataset, checkpoint or large binary is
  committed; the rubric contract is walked item by item and each definition of
  done is checked off against a real artifact; the pitch runs 5-8 minutes and
  states nothing the evaluation does not support.
- **Academic mapping.** C6, C7; final verification of C1-C5.

---

## Change log

| Date | Change |
| --- | --- |
| 2026-09-01 | Roadmap created during the foundation phase (phases 1-14 defined). |
| 2026-09-02 | Phase 4A completed: 436 source originals acquired and audited. No exact duplicates; 6 cross-split near-duplicate candidates pending visual confirmation; vest_loose present in only 8 images and absent from the provider test split; source project has drifted 76 annotations ahead of the frozen v4 export. Provider split classified UNDETERMINED_PENDING_VISUAL_REVIEW. |
| 2026-09-02 | Phase 4B completed, closing phase 4: the manual visual audit was recorded in `manual_audit_decisions.csv` (32 decisions) and `manual_audit_report.md`. Six cross-split pairs confirmed as semantic duplicates; three zero-instance images marked out-of-domain exclusion candidates; segmentation-derived boxes preferred but not applied; provider split reclassified UNSUITABLE_FOR_FINAL_PROTOCOL; dataset ACCEPTED_WITH_DOCUMENTED_LIMITATIONS. Twelve entry constraints handed to phase 5. |
| 2026-09-01 | Phases 2 and 3 completed. Phase 3 established that version 4 holds 436 independent source images plus offline-augmented train variants; phase 4 gates updated to work from that population. |
| 2026-09-04 | Phase 5A completed: canonical annotation snapshot resolved as `CURRENT_COMPLETE_GEOMETRY`. The live source state was recovered read-only with complete geometry for 2029 of 2031 annotations in original image coordinates; phase 4A's "1022 annotations lack geometry" was a consumption gap, not a provider limitation. Decode verified against the provider's own areas and boxes (agreement 0.0 px on all 2029). Version 4 was viable - all 436 source images map to exactly one non-augmented representation - but was rejected because its geometry is expressed after a stretch resize to 640x640. Measured drift: net +70 (76 added, 6 removed) over 67 images, with every addition lying inside an existing same-class annotation and covering no new object. The two unrecognised records classified VALID_BUT_UNSUPPORTED_GEOMETRY. Two measured corrections to earlier figures: the v4 source-snapshot annotation total is 1961, not the 1955 previously inferred, and the drift is +70 net rather than +76. No split created, no image excluded, no geometry converted. |
| 2026-09-04 | Phase 5B built the canonical modelling population: 433 modelling images (436 source minus 3 confirmed out-of-domain), 2031 annotations retained, 2 geometry-less records materialised as labelled synthetic rectangles, and 427 indivisible split units (421 singletons + 6 confirmed semantic duplicate groups). `vest_loose` untouched at 45 instances over 8 images. **One decision returned for review**: no deterministic geometry rule reproduces the 76 version-4 additions, because they are a mixture of degenerate slivers and legitimate re-annotations - the best rule reaches precision 0.97 but recall 0.43. This corrects the phase 5A reading that all 76 were coverage-free fragments. Nothing was excluded on that basis. Phase 5B classified NEEDS_FRAGMENT_RULE_REVIEW at that point and returned the question; no split, holdout or model exists. **Superseded by the next entry.** |
| 2026-09-04 | Phase 5B closed as READY_FOR_SPLIT_DESIGN after owner review. Automatic nested-annotation filtering was **rejected** (`REJECTED_FOR_AUTOMATIC_FILTERING`): containment inside an older same-class annotation is not evidence of error, and the evaluated rules cannot separate fragments from legitimate instance splits, geometry refinements and corrections of previously merged objects. All 2031 canonical annotations retained, 0 excluded; the 34 evaluated candidates carry the descriptive flag NESTED_SAME_CLASS_CANDIDATE. Recorded explicitly: the 76 v4 additions are a historical annotation-drift reference set, not ground truth for bad annotations, so rule precision against them measures agreement with drift rather than annotation correctness. Phase 5C carries one open entry gate: manual disposition of the 5 remaining near-duplicate candidates before any split is frozen. |
| 2026-09-04 | Phase 5B.1 dispositioned the outstanding near-duplicate candidates. Reconciled the counts: phase 4A raised 11 candidates (6 cross-split, 5 same-split); `review_h` showed all 6 cross-split pairs and `review_g` showed the 8 smallest-distance candidates, which were 4 of those 6 plus 4 same-split - so `review_g` held 8 pairs of which only 4 were novel, and the 11th candidate fell outside the cap and appeared on no sheet at all. The 4 novel same-split pairs were confirmed EXACT_SEMANTIC_DUPLICATE (HIGH) as manual_dup_007-010, giving 10 confirmed groups, 413 singletons and 423 split units; groups are now computed as connected components so a transitive chain forms one group. `chain-007` remains undecided and unmerged, drawn alone in figures/review_o_remaining_near_duplicates.jpg. Gate MANUAL_DISPOSITION_OF_REMAINING_NEAR_DUPLICATE_CANDIDATES stays OPEN and phase 5C is BLOCKED_ON_MANUAL_DISPOSITION. No split, holdout or model exists. |
| 2026-09-04 | Phase 5B.1 closed. The final outstanding candidate (chain-007, `66p9gzaQFGcmQA2v40Of` ~ `pbOZlgeseTpoTXwVJjAh`) was decided NEAR_DUPLICATE_SAME_SCENE (MEDIUM, GROUP_TOGETHER): the same worker and scene at a different moment rather than the same frame, but correlated enough that separating them across splits would risk leakage. Grouping serves statistical independence, not image identity, so it is indivisible like an exact duplicate while being recorded as a different finding via `group_basis`. All 11 phase 4A candidates now carry a human disposition (6 in 4B, 5 in 5B.1); zero outstanding. Final structure: 433 modelling images, 2031 annotations, 11 confirmed groups (22 images) + 411 singletons = 422 split units, largest group 2 images. Gate MANUAL_DISPOSITION_OF_REMAINING_NEAR_DUPLICATE_CANDIDATES is CLOSED and phase 5C entry readiness is READY_FOR_SPLIT_OPTIMIZATION. No split, holdout or model exists. |
| 2026-09-04 | Phase 5C.1 generated six provisional split candidates over the 422 indivisible groups. All reach the exact 70/15/15 target (303/65/65 images) with every hard constraint satisfied: five classes in all three splits at image and instance level, vest_loose 5/1/2 or 4/2/2, negatives 10/2/2. The objective is normalised per class and averaged so the frequent classes cannot outweigh the rare one, and each component is reported separately. Search is deterministic: 192 restarts seeded from the project seed 42, 49 feasible, 49 unique, re-run byte-identical. The provider split is read nowhere - the optimiser refuses to run if the feature table carries such a column. A correction: the brief stated no vest_loose image belongs to a duplicate group, which phase 5B.1 made false (two of the eight are in manual_dup_010), so the class occupies 7 indivisible units. Family B (5/2/1) is searched but rejected by the two-image holdout floor. algorithmic_best_candidate is candidate_001; final_selected_candidate remains UNSELECTED_PENDING_REVIEW. No split frozen, no holdout fingerprint, no model. |
| 2026-09-04 | Phase 5C.2 froze the canonical split. `candidate_001` was selected by human review of the six predeclared deterministic candidates (`HUMAN_REVIEW_OF_PREDECLARED_DETERMINISTIC_CANDIDATES`, `PROJECT_OWNER_REVIEW`); it coincides with `algorithmic_best_candidate`, and the two are recorded separately because a coincidence of outcome does not replace the review step. Frozen at 303/65/65 images over 294/63/65 groups (422 total, 11 non-singleton at 9/2/0), 2031 annotations at 1422/304/305, negatives 10/2/2, all five classes in all three splits at image and instance level, vest_loose 5/1/2 images and 30/8/7 instances. Verification ran before any write: the candidate re-derives its recorded digest, agrees with the phase 5C.1 summary, and matches the phase 5B population and group fingerprints. New fingerprints `split_assignment_sha256` `a230869ff4cb45f53654d67357a27f2a0def6d8ba79f9fa4880f0fdda2f046cc` and `holdout_sha256` `bb7ed43b20a84644d5a3917c6d0ead688132f82a30052b06ae7ad121e4851a00` cover membership only - no timestamp, path, label, metric or provider split - and the freeze is idempotent. Recorded as a protocol limitation: validation holds a single vest_loose image, so vest_loose validation metrics must not drive model selection on their own, and the split is a group-aware and class-aware constrained split, not a perfectly stratified one. **From this point the test split is a locked holdout**, requiring both `allow_test=True` and `CSVISION_ALLOW_TEST_SPLIT=1`; the variable was not set and the holdout has never been evaluated or inspected. Membership only: no image copied, no label written, `data/processed/` untouched, no model, no YOLO dataset, no inference. |
| 2026-09-04 | Phase 5D materialised the canonical task datasets for the development splits only. Two COCO views of the same images and the same objects: `canonical_detection_format: COCO`, `canonical_segmentation_format: COCO_INSTANCE_SEGMENTATION`, `model_specific_adapter: NOT_YET_SELECTED`. **368 images and 1726 annotations** (train 303/1422, validation 65/304), derived from the frozen manifest and verified against the canonical population. Images copied byte-for-byte, 368/368 verified by hashing both sides - no resize, crop, re-encode, EXIF rotation or colour conversion. Detection boxes derived from the canonical segmentation, never from the provider's bbox, and cross-checked against the independent phase 5A measurement at **max delta 0.0 px**. Geometry preservation measured rather than claimed: the emitted segmentation file is read back from disk and compared with the canonical state, RLE by decoded mask and polygons coordinate by coordinate - **1726 checked, 1726 matched, 0 mismatches** across 843 polygons, 881 RLE masks and 2 synthetic rectangles. Cross-task alignment verified for both splits. COCO ids are global over the whole modelling population, so the holdout can be materialised later without renumbering; no timestamp enters an emitted file and re-running is byte-identical. **No YOLO labels were written**: only COCO carries both polygon and RLE natively, so converting now would approximate the ground truth before a model exists; a future adapter must pass a documented mask-IoU fidelity audit first. The holdout is `NOT_MATERIALIZED_PROTECTED_HOLDOUT` - no test directory, COCO file or statistic was produced and no new knowledge about it was computed; CSVISION_ALLOW_TEST_SPLIT was not set. No model trained, no framework installed, no inference. Phase 6 not started. |
| 2026-09-04 | Phase 6A prepared the detection baseline without running it. Installed and locked torch 2.11.0+cu128, torchvision 0.26.0+cu128 and ultralytics 8.4.138; torch comes from the CUDA 12.8 index for a hardware reason, not a preference - the GPU is Blackwell (sm_120) and only cu128 builds carry its kernels. The runtime was verified by executing real kernels (matmul checked against CPU, conv backward, AMP autocast) rather than by reading `torch.cuda.is_available()`: RTX 5070 Laptop, sm_120, 7.96 GiB, driver 610.88, AMP OK. Built a **lossless** YOLO detection adapter over 368 images and 1726 labels: every box converted, written, read back from the label file on disk, decoded and compared against canonical - **1726/1726 within tolerance, 0 mismatches, max delta 1.47e-06 px** against a declared 1e-4 px. Adapter images byte-identical to canonical, class indices the frozen map verified against class_map_sha256, placeholder `object` absent, 12 negatives preserved as empty label files, and no YOLO segmentation labels written anywhere. Froze the D0 protocol before the experiment: YOLO11n pretrained (yolo11n.pt fingerprinted 0ebbc80d..., 5613764 B), imgsz 640, batch 16, 100 epochs, seed 42, every hyperparameter stated, checkpoint rule fixed in advance, metric hierarchy declared with primary mAP@0.50:0.95, and the vest_loose small-sample limitation (1 validation image, 8 instances) recorded before any number exists. A one-epoch smoke test proved the stack executes (OK, 62 s, 2.4 GiB peak, checkpoints written) and is marked NON_EXPERIMENTAL / DO_NOT_REPORT_AS_MODEL_RESULT - no metric from it is recorded and nothing was tuned from it. **The full D0 run was not performed**; no model result exists. The holdout has no adapter, no directory, no label and no key in the dataset descriptor; CSVISION_ALLOW_TEST_SPLIT was not set. Phase 6B not started. |
