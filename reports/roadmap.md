# Roadmap

Version: 2.0 · Current phase: **8 - Segmentation baseline** (8A adapter fidelity audited · 8B YOLO11n-seg selected and the S0 protocol frozen · 8C S0 trained once, mask mAP@0.50:0.95 0.407942 on validation · **8D per-instance error analysis · 8E canonical comparison protocol and S1 frozen · 8F S1 trained once, canonical supported macro 0.559463 against S0's 0.484643, delta +0.074820, `S1_IMPROVES_S0_BEYOND_MARGIN` · **8G the final segmenter is FROZEN - S1, YOLO11n-seg at imgsz 768 with `overlap_mask: false`**). **Phase 10B is complete: the detector-versus-segmenter recognition and spatial comparison has run on validation.** Both models are frozen, every selection and comparison is validation-only, the latency benchmark is still pending (10C), and the holdout has never been evaluated. Phase 7 is complete: 7A-7C ran all three detection experiments and **7D froze the final detector**, D2 - YOLO11n at imgsz 768, selected on validation only.

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
| 6 | Detection baseline | complete (6A adapter/runtime/protocol · 6B D0 trained and validated, mAP@0.50:0.95 0.464429 on validation) |
| 7 | Detection experiments and model freeze | complete (7A protocol · 7B D1 `BELOW_D0` · 7C D2 `IMPROVES_D0_BEYOND_MARGIN` · 7D `DETECTOR_FROZEN`: D2, YOLO11n @ 768, `CASE_B_VALIDATION_PERFORMANCE_LEADER`) |
| 8 | Segmentation baseline, experiments and model freeze | complete (8A adapter audited · 8B `S0_PROTOCOL_FROZEN` · 8C `S0_SEGMENTATION_BASELINE_COMPLETE` · 8D error analysis · 8E `S1_PROTOCOL_FROZEN` · 8F `S1_CONTROLLED_EXPERIMENT_COMPLETE`: S1 trained once with `overlap_mask: false` as the only intentional difference, delta +0.074820, `S1_IMPROVES_S0_BEYOND_MARGIN` · **8G `SEGMENTER_FROZEN`: S1 selected by the predeclared canonical policy plus human review, `final_segmenter_sha256` `63ef4196...`**) |
| 9 | Segmentation experiments and model freeze | **delivered by phases 8E-8G** - the controlled variation (S1), the comparison protocol and the model freeze all happened there. No separate phase 9 work remains; the next phase of actual work is 10, the controlled validation comparison of the two frozen models. |
| 10 | Controlled validation comparison | in progress (10A `DETECTOR_SEGMENTER_COMPARISON_PROTOCOL_FROZEN`, fingerprint `d92a1576...` · **10B `DETECTOR_SEGMENTER_VALIDATION_COMPARISON_COMPLETE`**: canonical box comparison and spatial-information analysis on validation, FP32 parity proved at runtime · 10C latency and memory benchmark - **not started** · 10D operational synthesis - not started) |
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

### Phase 6B - D0 detection baseline run (complete)

- **What it did.** Ran the predeclared D0 experiment exactly once and reported
  its validation performance. Nothing was tuned, before or after.
- **Result (validation only).** primary **mAP@0.50:0.95 =
  0.464429**, mAP@0.50 0.619373, precision 0.85568,
  recall 0.539992, on the 65-image / 304-annotation validation split.
- **Execution.** 100 of 100 epochs, no early stopping, 772.7 s
  wall clock on an RTX 5070 Laptop. Best epoch **67** (validation
  fitness 0.481862), selected by the predeclared
  `ULTRALYTICS_BEST_ON_VALIDATION_FITNESS` rule and cross-checked against an
  independently recomputed fitness curve from `results.csv`.
- **Declared policy vs effective settings.** `optimizer: auto` resolved to
  **AdamW at lr0 0.001111**,
  overriding the file's generic `lr0: 0.01`. Ultralytics persists neither the
  chosen optimizer nor its learning rate, so the value was re-derived from the
  framework's own selection rule and corroborated against the learning rate the
  run actually logged; the manifest records that provenance explicitly rather
  than presenting it as if the framework had reported it.
- **`vest_loose`.** AP@0.50 0.131486, precision 1.0, recall 0.0 - on **1
  validation image with 8 instances**. Marked `HIGH_SAMPLING_UNCERTAINTY`; the
  limitation was predeclared, not constructed afterwards.
- **Confusion matrix.** 96 undetected ground-truth objects and 71 unmatched
  predictions against 3 class-to-class confusions: at this operating point the
  difficulty is finding objects rather than naming them, consistent with recall
  sitting well below precision. No image-level error analysis was performed.
- **Outputs.** [`detection_D0_report.md`](detection_D0_report.md),
  `detection_D0_manifest.json`, the pre-run and result provenance records, and
  seven metric-only figures under `figures/detection/D0/`. Checkpoints are
  referenced by SHA-256 and not committed.
- **Explicitly not done.** No second training run, no hyperparameter tuning, no
  alternative model or image size, no segmentation work, no holdout access.
- **Academic mapping.** C2, C3.

> **Reproducibility note.** D0 was run **once**, deliberately. `deterministic:
> true` reduces run-to-run variance but does not eliminate it, so a second run
> would measure noise rather than establish anything. A reproducibility study, if
> wanted, is its own predeclared experiment.

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

### Phase 7A - comparison protocol freeze (complete)

Delivered `configs/detection_experiments.yaml`,
`src/construction_safety_vision/detection_comparison.py`,
`scripts/freeze_detection_experiments.py`,
[`reports/detection_comparison_policy.md`](detection_comparison_policy.md),
`reports/detection_comparison_policy.json`,
`reports/detection_comparison_reference.json`,
`reports/detection_experiments.provenance.json` and two test modules.

- **Objective.** Fix how D1 and D2 will be judged while neither exists, so the
  deciding metric cannot be chosen once the numbers are visible.
- **Gate (passed).** The class-support rule is a general threshold applied
  mechanically to the phase 5C.2 counts and names no class; D0's
  `supported_macro_map50_95` is computed from committed per-class metrics and
  verified against an independent hand computation; both candidates inherit D0's
  protocol and are proven to differ only in their declared fields; the four
  selection cases and the 0.005 margin are frozen and covered by tests; the
  official all-class metric is retained; the rare class stays reported;
  `CSVISION_ALLOW_TEST_SPLIT` was not set and no holdout artifact exists;
  **no model was trained**.
- **Academic mapping.** C2; feeds C4 and C6.

### Phase 7B - D1 capacity experiment (complete)

Delivered `scripts/train_detection_experiment.py`,
`src/construction_safety_vision/detection_run.py`,
[`reports/detection_D1_report.md`](detection_D1_report.md),
`reports/detection_D1_manifest.json`,
`reports/detection_experiment_results.json`, two provenance records, seven
metric-only figures and two test modules.

- **Objective.** Measure the effect of raising YOLO11 capacity from n to s with
  the frozen data, resolution and training protocol held fixed.
- **Result (validation only).** `supported_macro_map50_95` **0.560017** against
  D0's 0.570142, a delta of **-0.010125**, classified **`BELOW_D0`**. The
  official all-class `mAP@0.50:0.95` **rose** to 0.471114 (+0.006685), and the
  divergence is fully explained by the excluded class: `vest_loose` contributes
  +0.014785 to the five-class mean, and removing it leaves -0.008100.
- **Gate (passed).** Exactly one run, 100/100 epochs, checkpoint by the
  predeclared rule (epoch 73); the one-variable contract was proven before any
  weight was fetched and only `model` and `weight_identifier` differ; the
  optimizer was captured directly from the framework log (AdamW, lr0 0.001111 -
  the same resolution D0 got); the headline metric was cross-checked against the
  epoch history at 7.4e-05; `CSVISION_ALLOW_TEST_SPLIT` was not set and no
  holdout artifact exists; **no winner declared, D2 untouched**.
- **Academic mapping.** C2; feeds C4.

### Phase 7C - D2 resolution experiment (complete)

Delivered [`reports/detection_D2_report.md`](detection_D2_report.md),
`reports/detection_D2_manifest.json`, two provenance records, seven metric-only
figures, the updated `reports/detection_experiment_results.json` and two test
modules.

- **Objective.** Measure the effect of raising the input resolution from 640 to
  768 with YOLO11n capacity and the rest of the protocol held fixed.
- **Result (validation only).** `supported_macro_map50_95` **0.594018** against
  D0's 0.570142, delta **+0.023876**, classified
  **`IMPROVES_D0_BEYOND_MARGIN`**. The official all-class `mAP@0.50:0.95` rose
  to 0.490386 (+0.025957), so unlike D1 both metrics agree in direction.
  Against D1: +0.034001 on the selection metric.
- **Gate (passed).** Exactly one run, 100/100 epochs, checkpoint by the
  predeclared rule (epoch 90); the one-variable contract proven before any
  weight was touched with `training.imgsz` the sole difference; the pretrained
  `yolo11n.pt` verified **identical to D0's bytes** by digest; the optimizer
  captured directly from the framework log (AdamW, lr0 0.001111); the headline
  metric cross-checked against the epoch history at 0.002944;
  `CSVISION_ALLOW_TEST_SPLIT` unset and no holdout artifact; **no detector
  selected**.
- **Academic mapping.** C2; feeds C4.

### Phase 7D - final detector selection (complete)

- **Objective.** Apply the frozen Phase 7A selection policy exactly as written,
  record the human-reviewed decision, and freeze the selected checkpoint's
  identity for downstream work.
- **Classification.** `DETECTOR_FROZEN`.
- **What ran.** Nothing. No training, no validation, no inference, no efficiency
  benchmark, no threshold tuning, no image opened. Every number was read from a
  committed result manifest; the only computation was arithmetic over those
  numbers and SHA-256 over files already on disk.
- **Outcome.** The frozen selection engine, run over records rebuilt from the
  committed manifests, produced `CASE_B_VALIDATION_PERFORMANCE_LEADER` with **D2**
  leading and D1 as the non-reference **candidate** runner-up - the overall
  ordering across all three is **D2 > D0 > D1**, since the reference outscores
  D1 but is never ranked as a candidate. `selection_status: FINAL_SELECTED`,
  `selection_method: PREDECLARED_POLICY_PLUS_HUMAN_REVIEW`. Human review was
  confirmatory, not corrective: it established that the protocol held and the
  rule was correctly applied, and did not override it.
- **Frozen detector.** D2 - YOLO11n at imgsz 768, batch 16, seed 42, best epoch
  90. Checkpoint SHA-256 `0466f872...`, 5502289 bytes, copied byte-identically to
  the git-ignored `artifacts/frozen/detection/D2_best.pt`.
  `final_detector_sha256` `84d30d64...`.
- **Outputs.** `reports/final_detector_manifest.json`,
  `reports/detection_selection_report.md`,
  `reports/detection_experiment_comparison.csv`, the finalised
  `reports/detection_experiment_results.json`, one provenance record, plus
  `src/construction_safety_vision/detection_freeze.py` and 112 tests.
- **Validation gate.** Passed: the six historical D0/D1/D2 artifacts and the
  frozen Phase 7A policy are byte-identical before and after; the case and leader
  are the selection engine's output rather than constants in the freeze script;
  the runtime checkpoint's digest matches the committed manifest; the frozen copy
  hashes equal; artifacts are byte-identical on re-run;
  `CSVISION_ALLOW_TEST_SPLIT` unset and no holdout artifact exists.
- **Limits.** The result supports only that, under this frozen dataset, split,
  stack and one-run-per-configuration protocol, YOLO11n at 768 was the best
  validation detector among D0/D1/D2. It establishes no statistical
  significance, no generalisation across datasets or seeds, no support for the
  small-object mechanism, and nothing whatever about test performance.
- **Academic mapping.** C2; feeds C4.

## Phase 8 - Segmentation baseline

- **Objective.** Establish the instance-segmentation reference point on the same
  frozen splits.
- **Inputs.** The canonical COCO instance segmentation frozen in phase 5D; a
  pretrained segmentation model; `configs/segmentation_baseline.yaml`.
- **Outputs.** Baseline run artifacts and provenance; validation mask metrics;
  qualitative mask samples.
- **Validation gate.** Same gate as phase 6, plus a re-verified image-ID
  alignment with the detection dataset and a visual check that predicted masks
  are in the correct coordinate space.
- **Academic mapping.** C3.

### Phase 8A - YOLO segmentation adapter fidelity audit (complete)

- **Objective.** Measure, before any architecture is chosen, how much canonical
  COCO instance-mask geometry survives the Ultralytics YOLO segmentation label
  format. Answer it with numbers rather than assumption.
- **Classification.** `SEGMENTATION_ADAPTER_AUDIT_COMPLETE`.
- **What ran.** No segmentation model was trained, evaluated or downloaded, no
  weights were fetched, no architecture was selected and no detector was
  touched. Development splits only.
- **Format contract, read from the installed Ultralytics 8.4.138 source.** One
  row is one class plus **one flat ring**; there is no separator between rings,
  so **interior holes cannot be expressed** (`RETR_EXTERNAL` in the framework's
  own mask converter, and `fillPoly` with no even-odd subtraction) and a
  **disconnected mask cannot be expressed** either. Coordinates are parsed as
  float32 and rasterised through an `int32` truncation. A latent cardinality
  hazard was found and checked rather than assumed away: `verify_image_label`
  drops duplicate `(class, box)` rows with `np.unique`, so two same-class
  instances sharing a box would silently collapse into one. **0 collisions** in
  this development set.
- **Conversion.** Framework primitives where they exist: `merge_multi_segment`
  for multi-component instances, the same contour settings Ultralytics'
  converter uses, and `polygon2mask` for reconstruction. **1726 canonical
  annotations became 1726 rows** - none split, merged or dropped.
- **Method.** Each instance is measured against the label **as written to disk**
  and decomposed across three levels, so no stage is charged for another's cost:
  `control_iou` (rasteriser and contour convention alone) 0.986368 mean,
  `merged_iou` (adds component joining) 0.985525, `mask_iou` (adds serialisation
  and quantisation) 0.973066.
- **Result.** Mask IoU median 0.984576, mean 0.973066, P05 0.918176, minimum
  0.307692. Bands: 34.88% at or above 0.99, 50.52% in [0.95, 0.99), 11.88% in
  [0.90, 0.95), 2.72% below 0.90, and **none exact** - the rasteriser convention
  alone prevents that. By representation: polygon 0.978091 mean, RLE 0.968538,
  synthetic rectangle 0.849702 (n=2).
- **Topology.** 307 instances have more than one component (maximum 78); 180
  carry 699 interior holes totalling 1072133 filled pixels, largest single
  hole-area fraction 17.65%.
- **The dominant driver is size, not topology.** Mask-area quartiles run 0.9358
  / 0.9768 / 0.9874 / 0.9924, and every one of the 20 worst instances is a mask
  of 4-59 px where a single boundary pixel is a large share of the area. The
  hole comparison is explicitly recorded as **confounded by size** rather than
  read as "holes are free".
- **Validation gate.** Passed: 368 development images and 1726 annotations
  verified against the canonical documents; instance cardinality preserved and
  re-verified by the framework's own dataset scanner (303/1422 and 65/304, 12
  negatives, 5 classes, 0 corrupt labels, no holdout split); artifacts
  byte-identical on re-run; `CSVISION_ALLOW_TEST_SPLIT` unset.
- **No decision.** `segmentation_architecture_selection:
  UNSELECTED_PENDING_FIDELITY_REVIEW`, `segmentation_baseline: UNFROZEN`, `S0:
  NOT_DEFINED`. A mask-native alternative is recorded as a future option, not
  selected.
- **Academic mapping.** C3.

### Phase 8B - architecture decision and S0 protocol freeze (complete)

- **Objective.** Record the human architecture decision the 8A evidence was
  produced for, formally approve the already-audited label bytes, freeze the S0
  protocol before any S0 number exists, and prove the segmentation runtime
  executes - without running S0.
- **Classification.** `S0_PROTOCOL_FROZEN`.
- **What ran.** One `NON_EXPERIMENTAL` one-epoch smoke test. **S0 was not
  trained** (`s0_execution_status: NOT_EXECUTED_PROTOCOL_ONLY`), the frozen
  detector was not retrained, revalidated or run, and the holdout was not
  touched.
- **Architecture.** **YOLO11n-seg**, `FINAL_SELECTED_FOR_S0`, basis
  `PHASE_8A_QUANTITATIVE_FIDELITY_AUDIT_PLUS_HUMAN_REVIEW`. Three stated
  reasons: the audited representation parses and preserves all 1726 instances at
  quantified fidelity; the family matches the frozen detector's YOLO11; imgsz
  768 matches its input resolution. A mask-native alternative such as Mask R-CNN
  is `NOT_SELECTED_FALLBACK` - never installed, trained or benchmarked, so
  nothing here claims YOLO11n-seg is better than one.
- **Adapter approval.** `APPROVED_FOR_CONTROLLED_TRAINING`, role
  `MODEL_SPECIFIC_DERIVED_REPRESENTATION`, canonical ground truth still
  `COCO_INSTANCE_SEGMENTATION`, conversion characterised as
  `ACCEPTED_WITH_QUANTIFIED_APPROXIMATION` - **never lossless**, because no
  instance round-trips exactly. The approval attaches to bytes: all four phase 8A
  label digests were re-verified against the files on disk before and after the
  smoke test, and a mismatch stops the phase as `ADAPTER_FINGERPRINT_MISMATCH`
  rather than rebuilding. No label was regenerated and no conversion algorithm
  was written.
- **No filtering.** All 1726 development instances retained, including the 47
  below IoU 0.90. Filtering after observing fidelity would change the modelling
  population in response to a model-format limitation.
- **Frozen S0 protocol.** `configs/segmentation_baseline.yaml`: imgsz 768,
  batch 8, 100 epochs, patience 50, seed 42, deterministic, AMP,
  `optimizer: auto`, augmentation
  `ULTRALYTICS_DEFAULT_SEGMENTATION_TRAINING_POLICY`. Batch 8 is a
  `PREDECLARED_EXECUTION_DECISION`; a genuine OOM stops the phase as
  `MEMORY_CONSTRAINT_REVIEW_REQUIRED` and is never rescued.
- **Segmentation-specific arguments recorded and verified** against the
  installed ultralytics 8.4.138 effective configuration, not documentation:
  `overlap_mask` true, `mask_ratio` 4, `retina_masks` false, `dropout` 0.0,
  `max_det` 300, `single_cls` false, `rect` false, `multi_scale` 0.0, plus the
  complete augmentation set.
- **Checkpoint rule, established before training and then reviewed.**
  `ULTRALYTICS_BEST_ON_VALIDATION_FITNESS`, and for a segmentation model that
  fitness is `SegmentMetrics.fitness = self.seg.fitness() + DetMetrics.fitness`
  - the unweighted **sum of box and mask mAP@0.50:0.95** at weights 1.0 and 1.0.
  `best.pt` is therefore **not** selected on the mask metric alone. The
  behaviour was returned for methodological review before S0 and the decision
  was to keep it: `checkpoint_selection_policy:
  ULTRALYTICS_NATIVE_SEGMENTATION_FITNESS`,
  `checkpoint_selection_review_status: HUMAN_REVIEWED_AND_ACCEPTED_BEFORE_S0`,
  `checkpoint_selection_semantics: BOX_MAP50_95_PLUS_MASK_MAP50_95`.
- **The selector and the reported metric are different things, deliberately.**
  `primary_scientific_reporting_metric: MASK_MAP50_95` with
  `selection_metric_equals_primary_reporting_metric: false`, so the epoch S0
  reports need not be the epoch that maximised the reported metric. No custom
  mask-only selector is authorised, S0 is never retrospectively re-read as a
  mask-only-selected epoch, and the framework fitness is never published as a
  headline number. **No claim is made that the composite is scientifically
  superior to mask-only selection** - nothing here compares the two.
- **Protocol invariant.** Every future segmentation experiment compared directly
  with S0 must select its checkpoint by
  `ULTRALYTICS_NATIVE_SEGMENTATION_FITNESS`, unless a new comparison protocol is
  human-reviewed and frozen **before** any affected experiment runs.
- **Metric hierarchy.** Primary `mask_mAP@0.50:0.95`; mask and box families
  reported side by side and never merged; a composite box-plus-mask score is
  refused by the parser. `supported_macro_mask_map50_95` is reported but is
  **not** a selection metric - no segmentation comparison protocol exists yet.
  `vest_loose` stays `DESCRIPTIVE_HIGH_UNCERTAINTY` under the unchanged phase 7A
  support rule and is reported in full without deciding anything.
- **Owed later.** A direct instance-mask IoU diagnostic under a **predeclared**
  matching protocol. Phase 8B records the requirement and executes nothing;
  writing that protocol after seeing S0's predictions is explicitly forbidden.
- **Validation gate.** Passed: phase 8A and phase 7D artifacts byte-identical
  before and after; adapter fingerprints and cardinality verified twice;
  `yolo11n-seg.pt` fingerprinted before use; CUDA preflight on sm_120; smoke test
  succeeded and wrote a checkpoint; `CSVISION_ALLOW_TEST_SPLIT` unset; no holdout
  adapter, label or statistic exists.
- **No result.** No S0 metric was produced, and the smoke test's numbers are
  `DO_NOT_REPORT_AS_MODEL_RESULT` and were not recorded anywhere.
- **Academic mapping.** C3.

### Phase 8C - S0 baseline training and the direct mask-IoU diagnostic (complete)

- **Objective.** Run S0 exactly once under the frozen protocol, validate the
  natively selected checkpoint once, and execute a direct instance-mask IoU
  diagnostic frozen before the first optimisation step - satisfying the
  assignment's explicit IoU requirement, which mask AP does not.
- **Classification.** `S0_SEGMENTATION_BASELINE_COMPLETE`.
- **What ran.** One full training run (100/100 epochs, `ALL_EPOCHS_COMPLETED`,
  1106.85 s), one authoritative validation, one direct IoU diagnostic. The frozen
  detector was not trained, validated, run or re-thresholded; the holdout was not
  touched.
- **Primary result (validation only).** **mask mAP@0.50:0.95 0.407942.**
  Secondary: mask mAP@0.50 0.579830, precision 0.850509, recall 0.528435.
- **Box metrics from the same model, reported separately and never merged.**
  mAP@0.50:0.95 0.478156, mAP@0.50 0.647256, precision 0.886160, recall 0.552147.
- **`supported_macro_mask_map50_95` 0.509482** over `helmet_loose`,
  `helmet_on_head`, `person` and `vest_on_body`, under the unchanged phase 7A
  support rule. **Descriptive only** - S0 is the sole segmentation experiment, so
  it declares no winner.
- **Per-class mask AP@0.50:0.95.** `helmet_loose` 0.789927, `helmet_on_head`
  0.586523, `vest_on_body` 0.390296, `person` 0.271182, `vest_loose` 0.001782.
  **`person`'s mask AP is roughly half its box AP (0.522227)** - the widest
  box-to-mask gap of any class, and **why is UNKNOWN**: occlusion and irregular
  outline are a hypothesis, and the image-level analysis that would test it is a
  later phase.
- **Checkpoint.** Best epoch **59** at native composite fitness **0.884180**,
  confirmed to be the argmax of the composite recomputed from `results.csv`, so
  the frozen rule provably chose it rather than being assumed to have.
  `optimizer: auto` resolved to AdamW at lr0 0.001111, captured from the log.
- **Direct instance-mask IoU diagnostic**, frozen before training and executed
  once against the **canonical COCO masks** at a predeclared conf 0.25 / NMS IoU
  0.70 / imgsz 768, matched one-to-one per image and class by
  `scipy.optimize.linear_sum_assignment`: `matched_mask_iou_mean` **0.717462**,
  `gt_normalized_mask_iou` **0.556977**, coverage 0.776316, IoU>=0.50 0.588816,
  IoU>=0.75 0.460526, over 304 canonical instances and 304 predictions with 236
  overlapping assignments and 68 unmatched on each side. The two headlines answer
  different questions and neither is a COCO AP.
- **Validation gate.** Passed: phase 8A, 8B and 7D artifacts byte-identical
  before and after; adapter digests verified before and after training; the
  audited directory left free of caches; `CSVISION_ALLOW_TEST_SPLIT` unset; only
  metric figures committed.
- **Nothing was decided.** `segmentation_baseline_status: S0_COMPLETE`,
  `final_segmenter: UNSELECTED_PENDING_REVIEW`. No S1, no alternative resolution
  or batch, no threshold tuning.
- **Academic mapping.** C3, and the IoU half of C4.

### Phase 8D - S0 validation error analysis (complete)

- **Objective.** Characterise why S0's mask quality trails its box quality,
  especially for `person`, and produce evidence for a future controlled
  experiment. No training.
- **Classification.** `S1_HYPOTHESIS_READY_FOR_PROTOCOL_FREEZE`.
- **What ran.** Inference on the frozen validation split under the settings
  phase 8C froze, re-used unchanged. The per-instance table reproduces the
  committed 8C aggregates exactly, which is how it is known to describe the same
  run; **the 8C figures were not regenerated**.
- **Outcome census (304 instances).** `HIGH_QUALITY_MASK` 140, `DETECTION_MISS`
  68, `LOW_OVERLAP_MASK` 57, `MODERATE_MASK` 39. The four bands partition.
- **Coverage and mask quality fail in opposite directions across size.** Coverage
  rises monotonically with area (0.513 / 0.816 / 0.882 / 0.895), so misses
  concentrate in small masks; matched IoU is **worst in the largest quartile**
  (0.642). Large objects are found and poorly delineated, small ones missed.
- **`person` is weaker than every class at every quartile** (0.592 / 0.539 /
  0.636 / 0.561 against non-person 0.789 / 0.844 / 0.829 / 0.822), so the deficit
  survives size stratification. Its box-minus-mask gap is 0.251045 against
  0.039179 for the next class.
- **The substantive finding: a target-versus-evaluation mismatch.**
  `overlap_mask: true` plus `polygons2masks_overlap`'s descending-area running
  maximum means the **smaller** instance owns shared pixels, so a vest owns the
  pixels of the person wearing it. 27.3% of canonical person pixels are
  contested; **71.0% of the pixels S0 misses on person lie there**. Spearman
  -0.5466; 0.691 matched IoU under 20% contested against 0.408 over 40%.
  Re-scoring the same predictions against the overlap-resolved target moves
  person GT-normalised IoU 0.434635 -> 0.504610 and matched IoU 0.578106 ->
  0.671180, other classes by under 0.002.
- **Labelled `POST_HOC_HYPOTHESIS_GENERATING`, not a finding.** Motivated by an
  observation during review; re-scoring changes the measuring stick, not the
  model. It does not close the gap either - person matched IoU 0.671180 against
  its own target still trails `helmet_loose` 0.935.
- **Adapter conversion is not the dominant cause.** Worst-20 adapter mean
  0.968401 against 0.979140; rank correlation 0.246; 4 adapter-risk instances.
  Not zero effect - phase 8A quantified real loss.
- **Candidates assessed, none chosen.** `mask_ratio: 2` `WEAKLY_MOTIVATED`
  (under-segmentation outnumbers boundary error 56:19); YOLO11s-seg
  `NOT_SPECIFICALLY_MOTIVATED` (nothing isolates capacity, which is not evidence
  it cannot help); `overlap_mask: false` best motivated but **not a clean
  comparison**, because the flag changes the validation target too.
- **Validation gate.** Passed: phase 8A/8B/8C and 7D artifacts byte-identical;
  taxonomy frozen before review; review set selected before any image was opened;
  6 of 43 selected instances visually inspected, recorded separately; causal
  labels refused by the recorder; `CSVISION_ALLOW_TEST_SPLIT` unset.
- **Nothing was decided.** `final_segmenter: UNSELECTED_PENDING_REVIEW`, no S1
  protocol frozen, no model trained.
- **Academic mapping.** C4 error analysis.

### Phase 8E - canonical comparison protocol and S1 freeze (complete)

- **Objective.** Freeze a common evaluator that two models trained against
  different targets can both be scored by, evaluate S0 under it once, freeze S1
  as a one-variable intervention, and establish that its batch fits. No training.
- **Classification.** `S1_PROTOCOL_FROZEN`.
- **Why a new evaluator.** `overlap_mask` decides the training target **and** the
  framework's validation ground truth, so S0's and S1's native mask AP would be
  measured against different targets. Native mask AP is therefore
  `NOT_CROSS_TARGET_COMPARABLE_FOR_S0_S1_SELECTION` - demoted, never suppressed.
- **The evaluator.** pycocotools `COCOeval` `segm` against the canonical phase 5D
  COCO validation masks; IoU 0.50:0.05:0.95, `maxDets` [1, 10, 100], imgsz 768,
  NMS IoU 0.70, conf 0.001, `retina_masks: true`, binary-mask RLE, canonical
  category ids used directly with no remapping. Fingerprint `282eb0ec...`.
  Validated on synthetic fixtures before touching any checkpoint, and
  deterministic across two independent executions.
- **conf 0.001 is not an operating point.** AP needs the low-scoring tail; the
  phase 8C direct-IoU diagnostic keeps its own operational 0.25 and the two are
  never mixed. The model proposes up to 300 candidates while COCOeval scores at
  its conventional 100 - both recorded.
- **S0 canonical reference, executed once**
  (`POST_S0_PRE_S1_CANONICAL_COMPARISON_REFERENCE_EVALUATION`): supported macro
  **0.484643**, all-class mAP@0.50:0.95 **0.388009**, mAP@0.50 **0.537536**. Per
  class AP@0.50:0.95: `helmet_loose` 0.793946, `helmet_on_head` 0.609345,
  `vest_on_body` 0.400246, `person` 0.135036, `vest_loose` 0.001474.
- **The two evaluators are not comparable in absolute terms**, but the shape is:
  compact classes land close under both, `person` does not (canonical 0.135036
  against native 0.271182). That **corroborates** the phase 8D overlap-target
  mechanism from an evaluator built for another purpose. It is not proof and not
  a prediction about S1.
- **S0's earlier results stand unrevised.** Native mask mAP@0.50:0.95 0.407942
  remains `NATIVE_TARGET_EVALUATION`; direct GT-normalised IoU 0.556977 remains
  `CANONICAL_GT_RECOVERY_DIAGNOSTIC`; the phase 8D re-score stays
  `HYPOTHESIS_GENERATING_DIAGNOSTIC_ONLY`. A third measurement was added and none
  withdrawn.
- **S1 frozen, not executed.** Only `overlap_mask: true -> false`; 43 other
  arguments inherited unchanged and any second override refused by the parser.
  Same pretrained binary, same adapter bytes, imgsz 768, batch 8, epochs 100,
  seed 42, mask_ratio 4, same native checkpoint policy.
- **Selection frozen at margin 0.005**, three cases, `PRACTICALLY_EQUIVALENT`
  preferring S0 - decided in advance. Direction disagreement between the
  canonical AP and the direct IoU is recorded as
  `CROSS_METRIC_DIRECTION_DISAGREEMENT`, never resolved by a composite, which
  both parsers refuse.
- **Feasibility.** One forward and backward pass at batch 8, imgsz 768,
  `overlap_mask: false`: peak 3.465 GiB reserved, mask target `[33, 192, 192]`.
  `NON_EXPERIMENTAL` - no optimizer step, no validation, no checkpoint, no metric.
- **Honest asymmetry.** Unlike phase 7A, this policy was frozen after its
  reference ran. `protocol_timing` is `POST_S0_PRE_S1_PROTOCOL_FREEZE` and the
  parser refuses any claim otherwise. No S1 number influenced any rule.
- **Validation gate.** Passed: phase 7D, 8A, 8B, 8C and 8D artifacts
  byte-identical; evaluator deterministic; `CSVISION_ALLOW_TEST_SPLIT` unset.
- **Academic mapping.** C4.

### Phase 8F - the S1 overlap-mask controlled experiment (complete)

- **Classification.** `S1_CONTROLLED_EXPERIMENT_COMPLETE`. Exactly one full S1
  training run, one authoritative native validation, one canonical evaluation
  and one direct mask-IoU diagnostic. S0 was read, never re-executed.
- **The intervention.** `overlap_mask` `true -> false`, resolved in code from
  S0's own protocol with 43 framework arguments inherited unchanged. The run's
  own `args.yaml` was checked afterwards, so the intervention is known to have
  reached the trainer rather than merely been requested.
- **A trap that had to be avoided.** Read from the installed source:
  `Model._reset_ckpt_args` keeps only `imgsz`, `data`, `task` and `single_cls`
  from a checkpoint, and the framework default is `overlap_mask: True`.
  Validating S1 without passing the flag would have scored it against **S0's**
  target. The runner passes it explicitly and refuses to continue if the
  resolved value is not `False`.
- **Execution.** YOLO11n-seg from the same `yolo11n-seg.pt` (`55ed65c5...`),
  imgsz 768, batch 8, seed 42, mask_ratio 4, 100/100 epochs in 1195.05 s, no
  early stop, peak 4.102 GiB reserved. Best epoch **77** at native composite
  fitness 0.974520, verified as the argmax of the composite recomputed from
  `results.csv`. `optimizer: auto` resolved to **AdamW at lr0 0.001111,
  momentum 0.9**, captured directly from the framework log - the same as S0, so
  the optimizer does not confound the comparison. `best.pt` `29337d67...`,
  `last.pt` `06a9af1d...`, neither committed.
- **Primary result (validation only).** `CANONICAL_SUPPORTED_MACRO_MASK_MAP50_95`
  **0.559463** against S0's committed **0.484643**: delta **+0.074820**,
  classified **`S1_IMPROVES_S0_BEYOND_MARGIN`** under the frozen 0.005 margin.
  All-class canonical mAP@0.50:0.95 0.455034 (S0 0.388009), mAP@0.50 0.634023.
- **The aggregate is not a uniform effect, and the report says so.** `person`
  moved +0.317316 and carries 88.6% of the total gain across admitted classes,
  while `helmet_loose` **regressed** -0.059035. The decomposition is computed
  mechanically, not written by hand. Why any individual class moved is UNKNOWN:
  the experiment varied one flag and measured the outcome; it tested no
  per-class mechanism.
- **Secondary diagnostic agrees.** Direct GT-normalised mask IoU 0.635356
  (S0 0.556977, +0.078379), matched mask IoU 0.794849 (+0.077387), coverage
  0.799342 (+0.023026), under the unchanged phase 8C protocol at conf 0.25.
  `CROSS_METRIC_DIRECTION_CONSISTENT` under a rule frozen before S1 ran.
- **Native metrics reported, demoted.** Mask mAP@0.50:0.95 0.458206, mAP@0.50
  0.663319, precision 0.719937, recall 0.599587; box mAP@0.50:0.95 0.518779.
  Marked `NATIVE_TARGET_METRIC` /
  `NOT_CROSS_TARGET_COMPARABLE_FOR_S0_S1_SELECTION`: `overlap_mask` reshapes the
  native validation ground truth, so S0's and S1's native AP are measured
  against different targets and are **never differenced**. The descriptive
  native supported macro 0.549273 decides nothing.
- **Determinism, measured.** Validation, the canonical evaluation and the
  diagnostic were re-executed on the same frozen checkpoint to re-render a
  prose addition, and the canonical, direct-IoU and results artifacts came back
  **byte-identical**, with `S1_experiment_sha256` unchanged. Training was not
  repeated; this is one experiment, not two.
- **Nothing was selected.** `final_segmenter: UNSELECTED_PENDING_REVIEW`. No
  S1 re-run, no `mask_ratio` variant, no YOLO11s-seg, no resolution or batch
  change, no threshold tuned, no metric added, no composite score.
- **Validation gate.** Passed: every phase 7D and 8A-8E artifact byte-identical
  before and after; adapter fingerprints unchanged; S0's checkpoint bytes
  unchanged; four validators pass; `CSVISION_ALLOW_TEST_SPLIT` unset and no
  holdout image, identifier, prediction or statistic exists in any artifact;
  eleven metric-only figures committed and no dataset imagery; 1814 tests pass.
- **`S1_experiment_sha256`** `d14b98fbea6402691245037a296a7da4c040299535c2743d994b65a58acaa402`.

### Phase 8G - final segmenter selection and model freeze (complete)

- **Classification.** `SEGMENTER_FROZEN`. No training, no evaluation, no
  inference, no benchmark, no threshold. The phase reads committed artifacts,
  re-derives the comparison arithmetically, records the human review, and
  freezes the selected checkpoint's identity.
- **The winner was derived, not asserted.** The two primary figures were read
  out of each experiment's committed canonical evaluation, the delta recomputed,
  and the frozen 0.005 margin reapplied by the same `classify_delta` the policy
  names. No experiment id appears in the freeze script as the answer, and a test
  asserts that.
- **Selected: S1** - YOLO11n-seg, imgsz 768, batch 8, mask_ratio 4,
  **`overlap_mask: false`**, best epoch 77. `selection_status: FINAL_SELECTED`,
  `selection_method: PREDECLARED_CANONICAL_POLICY_PLUS_HUMAN_REVIEW`,
  `margin_classification: S1_IMPROVES_S0_BEYOND_MARGIN`.
- **The evidence.** `CANONICAL_SUPPORTED_MACRO_MASK_MAP50_95` 0.484643 -> **0.559463**,
  delta **+0.074820**, roughly 15x the engineering margin. Canonical all-class
  mAP@0.50:0.95 0.388009 -> 0.455034 moves the same way, and the secondary
  direct GT-normalised mask IoU 0.556977 -> 0.635356 does too
  (`CROSS_METRIC_DIRECTION_CONSISTENT`). Native framework AP did **not**
  arbitrate: `overlap_mask` changes the native validation target, so S0's and
  S1's native numbers are measured against different ground truth.
- **The trade-off is published, not buried.** `helmet_loose` **regressed
  -0.059035** while the aggregate rose, and `person` carries most of the gain
  (+0.317316). Selection does not require every class to improve; it requires
  the predeclared metric to clear the predeclared margin. Why any individual
  class moved is **UNKNOWN**.
- **The person result stays non-causal.**
  `CONSISTENT_WITH_PHASE_8D_OVERLAP_TARGET_HYPOTHESIS`, explicitly **not**
  `PROOF_OF_CAUSAL_MECHANISM`. Phase 8D was `POST_HOC_HYPOTHESIS_GENERATING` and
  this phase ran no experiment to test the mechanism.
- **Checkpoint identity.** S1's `best.pt`, SHA-256 `29337d67...`, 6041685 bytes,
  with a byte-identical immutable copy at
  `artifacts/frozen/segmentation/S1_best.pt`, outside the run directory a re-run
  would overwrite. **Two checkpoints are rejected by identity**: S0's `best.pt`
  (`d7b512b9...`) and any `last.pt`. S0 and S1 are the same architecture and the
  same number of bytes, so a digest check is the only thing that separates them.
- **`final_segmenter_sha256`** `63ef41961ba3fedfef46658754940fb6ce075cdd93087b926d7c6620402579a7`,
  computed over the selected experiment, model, imgsz, batch, mask_ratio,
  **`overlap_mask`**, checkpoint digest, experiment digest, policy and evaluator
  digests, split and class-map digests, and the adapter fingerprints.
- **Validation gate.** Passed: freeze generation run twice with all four
  semantic artifacts byte-identical; every phase 7D and 8A-8F artifact
  byte-identical before and after; the accessor loads the manifest, recomputes
  its fingerprint, and rejects S0's checkpoint and `last.pt` against the real
  files on disk; `CSVISION_ALLOW_TEST_SPLIT` unset and no holdout identifier,
  metric or artifact exists; no model binary committed; 1888 tests pass.
- **Not done here, deliberately.** No latency, throughput or memory benchmark.
  The standardised detector-versus-segmenter comparison is a later phase, and
  the final holdout evaluation remains a separate single-shot phase.

## Phase 9 - Segmentation experiments and model freeze (delivered by phases 8E-8G)

**Numbering note, recorded rather than silently corrected.** This phase was
planned before the segmentation work was broken into 8A-8G, and its objective -
improve segmentation through a controlled variation and freeze one final model -
was met there: 8E froze the comparison protocol and the S1 candidate, 8F ran the
single controlled experiment, and 8G selected and froze the model. The phase is
therefore complete in substance under different numbers, and the roadmap is not
renumbered because earlier artifacts reference these phase numbers. The next
phase of actual work is **phase 10**, the controlled validation comparison of
the two frozen models, which is also where the standardised
detector-versus-segmenter operational comparison belongs.

- **Objective.** Improve segmentation through controlled variations and freeze
  one final model.
- **Delivered as.** `configs/segmentation_comparison.yaml` and
  `configs/segmentation_canonical_evaluation.yaml` (8E);
  `reports/segmentation_S1_report.md` (8F);
  `reports/final_segmenter_manifest.json` and
  `reports/segmentation_selection_report.md` (8G). The frozen checkpoint is
  referenced by SHA-256 `29337d67...` and is not committed.
- **Validation gate.** Met by the 8E-8G gates: mask and box metrics are reported
  separately and never conflated, and a composite is refused by both parsers.
- **Academic mapping.** C3; feeds C4.

## Phase 10 - Controlled validation comparison

Broken into four sub-phases, so the protocol is frozen before anything is
measured:

| Sub-phase | Scope | Status |
| --- | --- | --- |
| 10A | Freeze the detector-versus-segmenter comparison protocol | **complete** |
| 10B | Controlled validation recognition and spatial comparison | **complete** |
| 10C | Controlled latency and memory benchmark | not started |
| 10D | Operational and spatial-value synthesis | not started |

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

### Phase 10A - the detector-versus-segmenter comparison protocol (complete)

- **Classification.** `DETECTOR_SEGMENTER_COMPARISON_PROTOCOL_FROZEN`. No model
  was executed, no prediction produced, no latency measured and no image pixel
  read. Protocol fingerprint
  `d92a157637baf52f9a7248bf24257d5a177e8496b7727ffb5afc874ab96d1c4d`.
- **The question is not which model wins.** D2 emits a class, a confidence and
  a box; S1 emits those plus an instance mask. They do not produce the same
  output, so a single ranking would be meaningless. The protocol asks what the
  mask adds and what it costs, along four axes: recognition, spatial
  information, computational cost, and operational person-PPE reasoning.
- **Both models verified by digest, neither executed.** D2 `0466f872...` at
  imgsz 768, S1 `29337d67...` at imgsz 768 with `overlap_mask: false`. The two
  share the comparison input size, so resolution is held constant rather than
  becoming a confound.
- **Population.** Validation only: the frozen 65 images / 304 annotations, read
  from the canonical phase 5D document, membership fingerprint `54e4ae8a...`.
  Both models see the same images in the same order; nothing is re-split.
- **Two thresholds, never mixed.** AP inference at conf **0.001**, because
  average precision integrates over the score curve and needs the low-scoring
  tail; operational inference at conf **0.25**, because the spatial and
  association analysis needs a working point. Both at NMS IoU 0.70, max_det
  300, no TTA, and **FP32 for both models** - pinned as `quantize: 32`, because
  `half` is deprecated in the installed ultralytics 8.4.138 and leaving
  `quantize` unset would let the runtime choose, silently turning the latency
  comparison into a precision comparison.
- **Recognition is judged by one external evaluator.** pycocotools `COCOeval`
  at `iouType='bbox'` against the canonical phase 5D boxes, IoU 0.50:0.05:0.95,
  `maxDets` [1, 10, 100]. The two models run through different framework
  validation paths, so their native box metrics are not guaranteed to be
  computed identically; one external evaluator removes that doubt.
- **S1's boxes are S1's own.** `segmenter_boxes_derived_from_masks: false`.
  Re-deriving them from its masks would improve their consistency with the mask
  branch and would then measure a post-processing choice this project invented.
- **Every mask quantity is paired with a box proxy, or declared to have none.**
  Seven spatial features are frozen; `MASK_TO_BOX_FILL_RATIO` and
  `SHAPE_EXTENT` are marked `NO_BOX_ONLY_EQUIVALENT`. That pairing is the whole
  answer to "what does segmentation add", and deciding it after seeing numbers
  would be circular.
- **The association analysis is descriptive, and named so.**
  `SPATIAL_ASSOCIATION_ANALYSIS`, not compliance accuracy: the project holds no
  canonical compliance ground truth, and `helmet_on_head` / `vest_on_body`
  already encode a provider-level state. Four disagreement categories partition
  every candidate pair. No embeddings, no tracking, no learned rule, no
  collapsed classes.
- **Latency is frozen before anything is timed.** batch 1, imgsz 768, FP32,
  **20 warmup iterations discarded**, **30 timed repetitions** over **20
  benchmark images** selected by `STABLE_SHA256_RANK_OF_IMAGE_ID` - ranked by
  the digest of the identifier, so the subset cannot have been picked for being
  easy or crowded. Ordered fingerprint `45059c2c...`. Execution is interleaved
  and symmetric in both directions, because benchmarking one model to
  completion first would measure the laptop's thermal state as well.
- **Two timing boundaries, and mask reconstruction is inside the second.**
  `MODEL_INFERENCE_LATENCY_MS` is the forward pass;
  `END_TO_END_MODEL_OUTPUT_LATENCY_MS` includes preprocessing, NMS and - for
  the segmenter - mask reconstruction. Excluding it would hide precisely the
  cost this comparison exists to quantify. Explicit `torch.cuda.synchronize()`
  on both edges of every timed region, same primitive for both models.
- **No aggregate score.** `aggregate_benefit_score: false`,
  `winner_declared: false`. Benefit and cost are reported side by side, and
  neither frozen model changes as a result.
- **Validation gate.** Passed: the freeze is idempotent across two runs, both
  freeze manifests recompute, every phase 7D/8A-8G artifact is byte-identical,
  the committed membership reproduces its fingerprint and the benchmark subset
  reproduces the frozen selection rule, `CSVISION_ALLOW_TEST_SPLIT` unset, and
  a test asserts the freeze script contains no inference or timing call path at
  all.

### Phase 10B - the validation recognition and spatial comparison (complete)

- **Classification.** `DETECTOR_SEGMENTER_VALIDATION_COMPARISON_COMPLETE`. Both
  frozen models ran controlled inference on validation under the two protocols
  phase 10A froze. Neither model was trained, modified or re-thresholded, no
  latency or memory was measured, and the holdout was never read.
- **Precision parity was proved at runtime, not assumed.** Before any
  comparison number existed, both models were probed with a forward pre-hook:
  backend FP16 flag `false`, parameter dtypes `['torch.float32']`, input tensor
  `torch.float32`, autocast during forward `false`, no quantization config -
  **identical for both**. A comparison across two precisions would have
  measured the precision.
- **Canonical box comparison (validation only).** One external evaluator,
  `COCOeval` at `iouType='bbox'` against the canonical phase 5D boxes, both
  models' **own** predicted boxes:

  | Metric | D2 | S1 | delta |
  | --- | --- | --- | --- |
  | `CANONICAL_BOX_MAP50_95` | 0.485390 | 0.505682 | **+0.020292** |
  | `CANONICAL_BOX_MAP50` | 0.641107 | 0.692955 | **+0.051848** |

- **Support sensitivity, `POST_HOC_DESCRIPTIVE_SUPPORT_SENSITIVITY`.** Applying
  the project's pre-existing support rule as a descriptive check, over the four
  adequately supported classes: D2 **0.589729**, S1 **0.581689**, delta
  **-0.008040**. Not a frozen phase 10A metric, not a selection rule, not a
  significance test, and it changes no frozen number. **Conclusion:** S1
  retains broadly similar localisation capability to D2 while adding mask
  output, but the positive all-class delta is driven by the highly uncertain
  `vest_loose` class and is not robust evidence that S1 is the superior object
  localiser.
- **The aggregate delta is carried entirely by the rare class, and that is the
  substantive finding.** The all-class figure is the unweighted mean of the five
  per-class APs, so each class contributes its delta over five: `helmet_loose`
  -0.003737, `helmet_on_head` +0.000199, `person` +0.006390, **`vest_loose`
  +0.026724**, `vest_on_body` -0.009284. `vest_loose` alone contributes more
  than the entire +0.020292, and it is the class already classified
  `DESCRIPTIVE_HIGH_UNCERTAINTY` with one validation image. **Excluding it, the
  mean delta over the other four classes is -0.008040** - the segmenter sits
  *below* the detector. Two supported classes declined (`vest_on_body`
  -0.046421, `helmet_loose` -0.018685). Never quote the aggregate improvement
  on its own.
- **These figures are not the native framework metrics** either model reported
  in its own phase, and the two must never be differenced: different evaluator,
  different ground-truth document, different confidence.
- **Operational population.** At conf 0.25: D2 269 predictions, S1 336, with
  **336 masks reconstructed on the original canvas and 0 exclusions**.
- **What the mask adds that a box cannot express.** `MASK_TO_BOX_FILL_RATIO`
  median **0.664** (P25 0.556, P75 0.775): on the median prediction, a third of
  the box is not the object. `SHAPE_EXTENT` median **0.672**: instances do not
  fill even their own tight rectangle. Both are `NO_BOX_ONLY_EQUIVALENT`.
- **Where a box proxy exists, it is systematically biased rather than merely
  noisy.** Mask area mean 144563 px against box-area proxy 237206 px; mask
  intersection mean 26828 px against box proxy 47281 px. The box proxy
  overstates in both cases, because a box counts background as object.
  Centroid displacement from the box centre: median **16.0 px**, P95 **126.9
  px**, max **262.8 px**.
- **For association, the box proxy is close - and saying otherwise would be the
  error.** Holding the model constant (S1's own boxes, so only the geometry
  varies): 103 agree, 3 box-only, **0 mask-only**, 66 neither, 1 both-but-
  different-person, over 173 candidate relationships. At a containment floor of
  0.50 the mask changes almost no association decision. The pipeline-level
  reading against D2's boxes disagrees far more (81 / 7 / 6 / 62 / 17), but
  that comparison is confounded: different model, different instances.
- **The frozen taxonomy was found to be non-exhaustive, and that is recorded
  as a coverage exception rather than patched.** Phase 10A froze four
  categories on the implicit assumption that a rule either associates or it
  does not; two rules can both associate and pick **different** people, which
  none of the four describes. That state is recorded as
  `association_taxonomy_status: FROZEN_TAXONOMY_NON_EXHAUSTIVE_FOR_OBSERVED_DATA`
  with `taxonomy_exception_type: BOTH_RULES_ASSOCIATE_DIFFERENT_PERSON`, status
  `UNCLASSIFIED_BY_FROZEN_FOUR_CATEGORY_TAXONOMY`. It is **not a fifth peer
  category** - adding one after seeing data is what a frozen taxonomy exists to
  prevent - and it is excluded from the denominator the frozen percentages use.
  The phase 10A protocol itself was **not modified**.
  - Geometry-isolating: **172 classified + 1 exception = 173**, taxonomy
    coverage **0.994220**. Frozen counts 103 / 3 / 0 / 66.
  - Pipeline-level: **156 classified + 17 exceptions = 173**, coverage
    **0.901734**. Frozen counts 81 / 7 / 6 / 62.
  - Counts are over all relationships; percentages use
    `classified_relationships`. Coverage is protocol bookkeeping, not a
    spatial-performance metric.
  - A different-person outcome is an `ASSOCIATION_RULE_DISAGREEMENT`, **not an
    association error**: there is no association ground truth to be wrong
    against. The 17 pipeline-level exceptions must **not** be attributed solely
    to geometry - that reading also varies the model and the instances.
  - This is a protocol-design limitation found during execution. It invalidates
    no canonical metric, no continuous spatial metric, no raw association
    decision and no model prediction.
- **11 of 349 PPE-person candidate pairs (3.2%)** had overlapping boxes whose
  masks shared no pixel at all - reported as a count and a fraction, not binned,
  because phase 10A declared no threshold for "strong" or "minimal".
- **`VISIBLE_PPE_COVERAGE_PROXY` remains `INTERPRETIVE_OPERATIONAL_PROXY`.** It
  is the one frozen quantity whose definition is qualitative; the
  implementation is a literal reading of the frozen sentence, and it is not a
  compliance measure. No compliance accuracy is claimed anywhere - the project
  holds no compliance ground truth.
- **Validation gate.** Passed: both result artifacts validate, the deltas
  recompute, the decomposition reproduces from the per-class table, every phase
  7D/8G/10A artifact is byte-identical, the semantic fingerprints reproduce
  across two independent executions, `CSVISION_ALLOW_TEST_SPLIT` unset, and no
  latency or memory figure appears in any artifact.
- **Result fingerprints.** box `68c9826a...`, spatial `9877b88d...`.

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
| 2026-09-04 | Phase 6B ran the D0 detection baseline once and reported it. YOLO11n pretrained, 640 px, batch 16, seed 42, 100/100 epochs, 772.7 s on an RTX 5070 Laptop. Validation-only result: primary **mAP@0.50:0.95 0.464429**, mAP@0.50 0.619373, precision 0.85568, recall 0.539992. Checkpoint chosen by the predeclared ULTRALYTICS_BEST_ON_VALIDATION_FITNESS rule (epoch 67, fitness 0.481862) and cross-checked against a fitness curve recomputed independently from results.csv, so the published number provably belongs to best.pt rather than to the last epoch. A pre-run provenance record was written before the first optimisation step, so the protocol demonstrably preceded the result. `optimizer: auto` resolved to AdamW at lr0 0.001111, overriding the file's generic 0.01; because Ultralytics persists neither value, it was re-derived from the framework's own selection rule and corroborated against the observed learning-rate trace, and the manifest records that provenance instead of implying the framework reported it. vest_loose scored AP@0.50 0.131486 with precision 1.0 and recall 0.0 on its single validation image - marked HIGH_SAMPLING_UNCERTAINTY under a limitation predeclared before the run. The confusion matrix shows 96 missed objects and 71 unmatched predictions against 3 class-to-class confusions. **No tuning, one run, no second run to see whether the number moved.** The holdout was not adapted, loaded, evaluated or measured; CSVISION_ALLOW_TEST_SPLIT was not set. Seven metric-only figures committed; no dataset or prediction image and no checkpoint committed. Phase 7 not started. |
| 2026-09-08 | Phase 7A froze the controlled detection comparison protocol **before D1 or D2 existed**, which is the only thing that makes it a protocol rather than a description. A general class-support rule - `COMPARISON_SUPPORTED` requires both >= 5 positive validation images and >= 20 validation instances - was applied mechanically to the counts frozen in phase 5C.2 and classified `helmet_loose`, `helmet_on_head`, `person` and `vest_on_body` as supported and `vest_loose` (1 image, 8 instances) as `DESCRIPTIVE_HIGH_UNCERTAINTY`. The rule names no class; the exclusion is its output. The primary phase 7 selection metric is `supported_macro_map50_95`, the unweighted mean of per-class AP@0.50:0.95 over the supported classes; D0's value is **0.570142** (exactly 0.57014175), computed from the committed per-class metrics and checked against an independent exact-fraction computation. The five-class `mAP@0.50:0.95` remains `OFFICIAL_ALL_CLASS_REPORTING_METRIC` at 0.464429 and is never hidden - the 0.105713 gap is the arithmetic effect of dropping one very low AP from an average, not an improvement, and phase 6B's protocol was not rewritten. Two experiments were frozen and **neither was trained**: D1 varies `MODEL_CAPACITY` (YOLO11s, `yolo11s.pt` recorded as a consequential change, still `NOT_YET_FINGERPRINTED`) and D2 varies `INPUT_RESOLUTION` (imgsz 768), each inheriting D0's protocol wholesale and declaring only an override set - so the one-variable discipline is enforced by the parser rather than promised in prose, and both comparisons were verified to differ from D0 only in their declared fields. Selection logic frozen at four cases with an absolute margin of 0.005, explicitly not a significance test; Case C was widened to cover the region where only one candidate clears D0 while the two sit within the margin of each other, and that widening is recorded as predeclared rather than applied later. Batch is fixed at 16 for all three, with a genuine CUDA OOM stopping an experiment as `MEMORY_CONSTRAINT_REVIEW_REQUIRED` instead of being rescued. The phase 6B result-artifact rebuild was classified `NON_SELECTION_REVALIDATION` with `evidence_basis: MAINTAINER_DECLARED_PARTIALLY_CORROBORATED_ON_DISK`, introducing zero selection degrees of freedom and modifying no D0 metric; the corroborated half is that the git-ignored `artifacts/detection/D0_val/` exists and six of the seven committed D0 figures are byte-identical to its output, so a separate validation execution demonstrably happened, while the claim that nothing was varied between the two executions stays maintainer-declared because Ultralytics writes no `args.yaml` for a validation run. Artifacts are byte-identical on re-run. The holdout was not read, materialised, adapted, counted or plotted, and `CSVISION_ALLOW_TEST_SPLIT` was not set. No model trained, no D3 authorised, phase 7B not started. |
| 2026-09-08 | Phase 7B ran the D1 capacity experiment once. YOLO11s from `yolo11s.pt` (SHA-256 85a76fe8..., 19313732 B, obtained by the same ULTRALYTICS_ASSET_DOWNLOAD mechanism as D0's yolo11n.pt), every other field inherited from `configs/detection_baseline.yaml` through the phase 7A override mechanism - so the one-variable contract was a structural property, proven before a single weight was fetched: the only differences from D0 are `model` and its consequential `weight_identifier`. 100/100 epochs in 1187.8 s, best epoch 73 by the predeclared ULTRALYTICS_BEST_ON_VALIDATION_FITNESS rule, peak 4.06 GiB at the frozen batch 16 after a non-experimental memory preflight that was deleted afterwards. **Validation-only result: `supported_macro_map50_95` 0.560017 against D0's 0.570142, delta -0.010125, classified `BELOW_D0`.** The official all-class `mAP@0.50:0.95` moved the *other* way, up to 0.471114 (+0.006685), and the report sets out why rather than quoting whichever number flatters the run: both metrics are unweighted means over the same per-class APs and differ only in which classes they average, so `vest_loose` - the class the support rule excluded, standing on one validation image - contributes +0.014785 of the five-class delta, and removing that single contribution leaves -0.008100, agreeing with the selection metric. **The entire sign change in the official figure comes from the excluded class**, which is what the phase 7A policy was written to prevent being read as a win. Three of the four supported classes improved (helmet_on_head +0.032401, helmet_loose +0.005802, person +0.004585); `vest_on_body` fell -0.083287 with recall -0.111868, more than the other gains combined, and **why is recorded as UNKNOWN** - one run cannot separate it from run-to-run variance and the image-level analysis that would diagnose it is a later phase. `optimizer: auto` resolved to AdamW at lr0 0.001111, momentum 0.9, read **directly from the framework's own log line** rather than inferred; this required fixing a latent defect where the parser matched the raw log and could never succeed, because Ultralytics wraps the label in ANSI colour codes. Also fixed: attaching the log handler pre-created the run directory, which made the framework silently divert the run to `D1-2`; that first attempt was aborted, its partial artifacts and pre-run record deleted, no metric from it recorded, and the runner now fails if the framework ever writes outside the directory the result is read from. Headline metric cross-checked against the independently recomputed epoch history at delta 7.4e-05. **No winner declared and no selection case assigned**: `reports/detection_experiment_results.json` records D2 as FROZEN_NOT_EXECUTED with a null metrics block and no placeholder number. Nothing was tuned, D0 was neither retrained nor revalidated, no alternative imgsz or capacity was tried. The holdout was not adapted, loaded, evaluated, counted or inspected; CSVISION_ALLOW_TEST_SPLIT was not set. Seven metric-only figures committed; no dataset or prediction imagery and no checkpoint. Phase 7C not started. |
| 2026-09-08 | Phase 7C ran the D2 resolution experiment once, completing the frozen matrix. YOLO11n at imgsz 768, inheriting every other field from `configs/detection_baseline.yaml` with `training.imgsz` as the sole override - proven before any weight was touched. The pretrained checkpoint was verified **identical to D0's bytes** (`yolo11n.pt`, SHA-256 0ebbc80d..., 5613764 B) against both D0's manifest and the phase 6A provenance, recorded as `IDENTICAL_TO_REFERENCE_VERIFIED_BY_DIGEST`; that check is now enforced in code, and the converse is too - a candidate declaring new weights that loads the reference's bytes is refused as a variable never applied. 100/100 epochs in 911.8 s, best epoch 90 (fitness 0.508697), peak 3.40 GiB at the frozen batch 16 after a non-experimental memory preflight that was deleted afterwards. **Validation-only result: `supported_macro_map50_95` 0.594018 against D0's 0.570142, delta +0.023876, classified `IMPROVES_D0_BEYOND_MARGIN`;** all-class `mAP@0.50:0.95` 0.490386 (+0.025957), so unlike D1 both metrics agree in direction and nothing turns on which is read. Against D1: +0.034001 on the selection metric, recorded as a difference and explicitly not a ranking, since the two candidates differ from each other in two things at once. Three of four supported classes improved (vest_on_body +0.056014, helmet_on_head +0.040641, helmet_loose +0.021934); `person` fell -0.023084. **The predeclared small-object hypothesis is NOT supported by the shape of the result**: ranking the supported classes by their frozen small-object fraction against their AP change gives a rank correlation of -0.20, the largest gain went to the *least* small-object-heavy class and the only decline was the second-least, so resolution improved the selection metric without the mechanism the hypothesis proposed explaining it - recorded that way rather than letting a beyond-margin gain read as confirmation. Precision rose 0.070651 while recall fell 0.023443, both now carrying an explicit caveat that Ultralytics reports them at the F1-maximising operating point rather than a fixed threshold. `optimizer: auto` resolved to AdamW at lr0 0.001111 for all three experiments, captured directly from the framework log. **No detector was selected**: `reports/detection_experiment_results.json` records `final_selected_detector: UNSELECTED_PENDING_REVIEW` with the computed `CASE_B_VALIDATION_PERFORMANCE_LEADER` exposed as `advisory_only` and `preferred_experiment: null`. One engineering abort is recorded: a first D2 attempt was stopped at epoch 8 and deleted because the runner did not yet emit three required manifest fields (peer deltas, weight identity, the unselected sentinel); no metric from it was read or used, no protocol was adapted to any observed performance, and the corrected run started fresh - classified `ENGINEERING_ABORT_BEFORE_VALID_EXPERIMENT_COMPLETION`. Two provenance defects were corrected in code with a fenced metadata correction that proves only the `phase` key moved and preserves `created_at`: D2's manifest and both its provenance records said 7B. Also fixed: the runner's completion marker was hardcoded to D1, and the D1-specific prose in the report builder claimed "increasing capacity" and "D2 remains pending" for every experiment. The holdout was not adapted, loaded, evaluated, counted or inspected; CSVISION_ALLOW_TEST_SPLIT was not set. Seven metric-only figures committed; no dataset or prediction imagery and no checkpoint. Phase 7D not started. |
| 2026-09-08 | Phase 7D applied the frozen Phase 7A policy and froze the project's final detector. **No model was trained, validated, benchmarked or run, no threshold was tuned and no image was opened**; every number was read from a committed result manifest and the only computation was arithmetic over those numbers plus SHA-256 over files already on disk. The decision was **derived, not asserted**: each experiment's record was rebuilt from its committed manifest, the class-support filter re-derived from the frozen split, `supported_macro_map50_95` recomputed from the per-class metrics, and the frozen selection engine asked which case held - it returned `CASE_B_VALIDATION_PERFORMANCE_LEADER` with **D2** leading and D1 as the non-reference **candidate** runner-up, and the freeze script refuses any other case rather than reconciling it. A pre-push audit resolved that term: the frozen rule ranks candidates only and never treats the reference as a peer, so D1 being the other candidate does **not** make it second-highest overall - **the overall ordering is D2 > D0 > D1**, because D0 (0.570141750000) outscores D1 (0.560017000000). This was an ambiguous name, not a ranking bug; the Case B logic is correct and unchanged. Every artifact now publishes `reference_experiment`, `validation_performance_leader`, `candidate_runner_up` and an explicit `overall_validation_ranking` instead of one overloaded `runner_up` field, the CSV carries an `overall_validation_rank` column, and the report states the three concepts in separate rows. Exact values, recomputed: D0 0.570141750000, D1 0.560017000000, D2 0.594018000000; deltas D1-D0 -0.010124750000, D2-D0 +0.023876250000, D2-D1 +0.034001000000, against the frozen margin of 0.005. A test asserts that the winner's id appears nowhere as a constant in the freeze script, because "the policy chose D2" is only worth something if the script could not have written it. **Final detector: D2 - YOLO11n at imgsz 768**, batch 16, seed 42, best epoch 90, `selection_status: FINAL_SELECTED`, `selection_method: PREDECLARED_POLICY_PLUS_HUMAN_REVIEW`. Human review is recorded as **confirmatory, not corrective** - it established that the protocol held and the rule was correctly applied and accepted the outcome; had it disagreed, the correct action would have been to record the disagreement and stop, not to pick another model. Checkpoint identity verified against the runtime binary (SHA-256 0466f872..., 5502289 B) and copied byte-identically to the git-ignored `artifacts/frozen/detection/D2_best.pt`, deliberately outside the run directory a re-run would overwrite; `last.pt` is rejected by name as well as by digest. Semantic `final_detector_sha256` 84d30d64... covers experiment, family, imgsz, checkpoint, experiment and policy digests and the split/adapter/class-map fingerprints, and excludes timestamps, paths and usernames. **No efficiency tie-break was run**: the policy requires one only in Case C, and D2 separates from D1 by more than the margin - the existing FRAMEWORK_VALIDATION_SPEED values stay descriptive and were not consulted. **The six historical D0/D1/D2 artifacts and the frozen 7A policy are byte-identical before and after**, their digests recorded in the freeze manifest and re-verified by test. `reports/detection_experiment_results.json` moved from `UNSELECTED_PENDING_REVIEW` to the final reviewed state with every metric untouched. Limits recorded explicitly: no statistical significance, no generalisation across datasets or seeds, the small-object mechanism still unsupported, `vest_loose` reported in full but excluded from the rationale, D1-vs-D2 a difference and not a ranking, and nothing whatever about test performance. `binary_distribution_status: LOCAL_IGNORED_FROZEN_ARTIFACT` - a fresh clone cannot run inference without obtaining or retraining the weights, and that is stated rather than glossed. Artifacts byte-identical on re-run; `CSVISION_ALLOW_TEST_SPLIT` unset; no holdout adapter, label, materialisation, prediction or metric exists. 1188 tests pass. Phase 8 not started. |
| 2026-09-09 | Phase 8A audited how much canonical COCO instance-mask geometry survives the Ultralytics YOLO segmentation label format. **No segmentation model was trained, evaluated or downloaded, no weights were fetched, no architecture was selected and the frozen detector was not touched**; development splits only, holdout untouched. The format contract was established by reading the installed ultralytics 8.4.138 source rather than documentation: one row is one class plus **one flat ring**, with no separator between rings, so **interior holes cannot be expressed** (`RETR_EXTERNAL` in the framework's own mask converter, and `fillPoly` with no even-odd subtraction) and a **disconnected mask cannot be expressed** either; coordinates are parsed as float32 and rasterised through an `int32` truncation. A latent cardinality hazard was found and checked rather than assumed away - `verify_image_label` drops duplicate `(class, box)` rows with `np.unique`, so two same-class instances sharing a box would silently collapse into one; **0 collisions** here. Conversion used the framework's own primitives (`merge_multi_segment`, the same contour settings its converter uses, `polygon2mask` for reconstruction), and **1726 canonical annotations became 1726 rows with none split, merged or dropped**. Each instance was measured against the label **as written to disk** and decomposed across three levels so no stage is charged for another's cost: `control_iou` 0.986368 (rasteriser and contour convention alone), `merged_iou` 0.985525 (adds component joining), `mask_iou` 0.973066 (adds serialisation and quantisation). **That decomposition changed the conclusion**: component joining costs 0.000843 mean IoU while serialisation and the integer snap cost 0.012458, so the loss a naive audit would have blamed on multi-component instances is mostly coordinate quantisation. Mask IoU median 0.984576, P05 0.918176, minimum 0.307692; 34.88% at or above 0.99, 50.52% in [0.95, 0.99), 11.88% in [0.90, 0.95), 2.72% below 0.90, and **none exact** - pycocotools and OpenCV do not rasterise identical geometry into identical pixels, which is why the control exists and why that gap is not reported as format loss. By representation: polygon 0.978091, RLE 0.968538, synthetic rectangle 0.849702 (n=2), kept apart so one cannot vouch for another. Topology: 307 instances have more than one component (maximum 78); 180 carry 699 holes totalling 1072133 filled pixels. **The dominant driver is mask size, not topology** - area quartiles run 0.9358 / 0.9768 / 0.9874 / 0.9924 and all 20 worst instances are 4-59 px masks; the with-versus-without-holes comparison is recorded as **confounded by size** (median 148640 px against 24690 px) rather than read as 'holes are free'. Hole area is counted in exact pixels after `cv2.contourArea` was found to overstate a 20x20 hole as 441. The framework's own dataset scanner re-read 303/1422 and 65/304 with 12 negatives, 5 classes, 0 corrupt labels and no holdout split, so instance cardinality provably survived parsing. Artifacts byte-identical on re-run; images hard-linked with SHA-256 verified equal to the canonical bytes; adapter git-ignored and marked `AUDIT_ONLY` / `NOT_CANONICAL` / `NOT_YET_APPROVED_FOR_TRAINING`. **No architecture was selected**: `segmentation_architecture_selection: UNSELECTED_PENDING_FIDELITY_REVIEW`, `segmentation_baseline: UNFROZEN`, `S0: NOT_DEFINED`; a mask-native alternative is recorded as a future option, not chosen. No detector-versus-segmenter comparison, no latency benchmark. `CSVISION_ALLOW_TEST_SPLIT` unset; no holdout adapter, label, statistic or identifier exists. 1310 tests pass. Phase 8B not started. |
| 2026-09-09 | Phase 8B recorded the human architecture decision, approved the audited adapter and froze the S0 protocol. **S0 was not trained** (`NOT_EXECUTED_PROTOCOL_ONLY`); the only model that ran was a one-epoch `NON_EXPERIMENTAL` smoke test whose accuracy numbers are `DO_NOT_REPORT_AS_MODEL_RESULT` and were recorded nowhere. **Architecture: YOLO11n-seg**, `FINAL_SELECTED_FOR_S0`, basis `PHASE_8A_QUANTITATIVE_FIDELITY_AUDIT_PLUS_HUMAN_REVIEW` - the audited representation parses and preserves all 1726 instances at quantified fidelity, the family matches the frozen detector's YOLO11, and imgsz 768 matches its input resolution so a later detector-versus-segmenter comparison is not also a resolution comparison. A mask-native alternative is `NOT_SELECTED_FALLBACK`, never installed, trained or benchmarked, so nothing claims YOLO11n-seg is better than one. **The adapter is approved, not promoted**: `APPROVED_FOR_CONTROLLED_TRAINING`, role `MODEL_SPECIFIC_DERIVED_REPRESENTATION`, canonical ground truth still `COCO_INSTANCE_SEGMENTATION`, conversion `ACCEPTED_WITH_QUANTIFIED_APPROXIMATION` and never called lossless - 0 of 1726 instances round-trip exactly. The approval attaches to **bytes**: all four phase 8A label digests (`63a8145d...` train, `dae69290...` validation, `ff21c782...` development, `616701e1...` membership) were recomputed from the files on disk and matched before *and* after the smoke test, with 1726 rows at 1422/304 over 303/65 images and 12 negatives; a mismatch stops the phase as `ADAPTER_FINGERPRINT_MISMATCH` rather than rebuilding, and no label was regenerated and no conversion algorithm written. **All 1726 instances retained**, including the 47 below IoU 0.90 - filtering after seeing fidelity would change the modelling population to suit the tool. S0 frozen at imgsz 768, batch 8, 100 epochs, patience 50, seed 42, deterministic, AMP, `optimizer: auto`, framework-default augmentation; **batch 8 is a `PREDECLARED_EXECUTION_DECISION`** on an ~8 GiB GPU and an OOM stops the phase as `MEMORY_CONSTRAINT_REVIEW_REQUIRED` rather than being rescued. Every segmentation-specific argument (`overlap_mask` true, `mask_ratio` 4, `retina_masks`, `dropout`, `max_det`, `single_cls`, `rect`, `multi_scale`) and the complete augmentation set were **verified equal to the installed ultralytics 8.4.138 effective configuration**, not copied from documentation, and each declared training value is classified `FRAMEWORK_DEFAULT` or `PROJECT_OVERRIDE`. **The checkpoint rule's real behaviour was read from the installed source before training, returned for methodological review, and accepted**: `SegmentMetrics.fitness = self.seg.fitness() + DetMetrics.fitness` with weights [0, 0, 0, 1] each, so `best.pt` is selected on the unweighted **sum of box and mask mAP@0.50:0.95** at equal weight and *not* on the mask metric alone. Recorded as `checkpoint_selection_policy: ULTRALYTICS_NATIVE_SEGMENTATION_FITNESS`, `checkpoint_selection_review_status: HUMAN_REVIEWED_AND_ACCEPTED_BEFORE_S0`, `checkpoint_selection_semantics: BOX_MAP50_95_PLUS_MASK_MAP50_95`, box/mask component weights 1.0/1.0, and - stated rather than smoothed over - `primary_scientific_reporting_metric: MASK_MAP50_95` with `selection_metric_equals_primary_reporting_metric: false`, so the epoch S0 reports need not be the epoch that maximised the reported metric. No custom mask-only selector is authorised, S0's checkpoint semantics are never reinterpreted retroactively, the framework fitness is never published as a headline number, and **no claim is made that the composite beats mask-only selection** - nothing here compares the two. Protocol invariant: every future segmentation experiment compared directly with S0 inherits this checkpoint policy unless a new protocol is human-reviewed and frozen before the affected experiment runs. Metric hierarchy frozen with mask and box families kept apart and a composite box-plus-mask score refused by the parser; `supported_macro_mask_map50_95` is reported but explicitly **not** a selection metric, because no segmentation comparison protocol exists. `vest_loose` stays `DESCRIPTIVE_HIGH_UNCERTAINTY` under the unchanged phase 7A rule, whose thresholds are imported from `detection_comparison` rather than retyped. A direct instance-mask IoU diagnostic is recorded as owed under a **predeclared** matching protocol, and writing that protocol after seeing S0's predictions is forbidden. Pretrained `yolo11n-seg.pt` fingerprinted before use (SHA-256 `55ed65c5...`, 6182636 B, `ULTRALYTICS_ASSET_DOWNLOAD`). Smoke test: SUCCESS in 51.3 s, peak 3.20 GiB reserved at batch 8, `optimizer: auto` resolved to AdamW at lr0 0.001111 captured directly from the framework log, both checkpoints written, plots disabled so no dataset or prediction imagery was rendered; the framework label caches it created inside the audited directory were deleted afterwards so the audited bytes are exactly as phase 8A left them. **Phase 8A's five artifacts and phase 7D's five artifacts are byte-identical before and after**, their digests recorded in the approval and manifest and re-verified by test. The detector was not retrained, revalidated, run or re-thresholded. `CSVISION_ALLOW_TEST_SPLIT` unset; no holdout adapter, label, statistic or identifier exists, and the descriptor carries no `test` key. No model binary committed. 1396 tests pass. Phase 8C not started. |
| 2026-09-09 | Phase 8C ran the S0 segmentation baseline **exactly once** and executed the predeclared direct instance-mask IoU diagnostic. YOLO11n-seg from `yolo11n-seg.pt` (SHA-256 `55ed65c5...`), imgsz 768, batch 8, seed 42, 100/100 epochs in 1106.85 s, no early stop. The frozen phase 8B protocol was verified by fingerprint before the first optimisation step, and a pre-run provenance record was written before it, so the protocol demonstrably preceded the result. **Primary result, validation only: mask mAP@0.50:0.95 0.407942**; mask mAP@0.50 0.579830, precision 0.850509, recall 0.528435. Box metrics from the same model, kept in their own section and never merged: mAP@0.50:0.95 0.478156, mAP@0.50 0.647256. `supported_macro_mask_map50_95` 0.509482 over the four classes the unchanged phase 7A rule admits - **descriptive, declaring no winner**, because S0 is the only segmentation experiment. Per-class mask AP@0.50:0.95: helmet_loose 0.789927, helmet_on_head 0.586523, vest_on_body 0.390296, person 0.271182, vest_loose 0.001782. **The substantive observation is `person`**, whose mask AP is roughly half its box AP (0.522227) - the widest box-to-mask gap of any class; **why is UNKNOWN** and recorded that way rather than explained, since the image-level analysis that would test occlusion or outline irregularity is a later phase. The checkpoint was chosen by the frozen native composite (epoch 59, fitness 0.884180) and that choice was **verified, not assumed**: the composite was recomputed from results.csv as mAP50-95(B) + mAP50-95(M) and the recorded best epoch confirmed to be its argmax. `optimizer: auto` resolved to AdamW at lr0 0.001111 momentum 0.9, captured directly from the framework log. **The direct mask-IoU protocol was frozen BEFORE training** in `configs/segmentation_mask_iou_evaluation.yaml` (fingerprint `b912039c...`) and executed once afterwards against the **canonical COCO validation masks, never the YOLO adapter** - scoring against the adapter would fold its own measured 0.973066 round-trip error into the model's result. At the predeclared operating point (conf 0.25, NMS IoU 0.70, imgsz 768, max_det 300, retina_masks true, no TTA), matched one-to-one per image and per class by `scipy.optimize.linear_sum_assignment` maximising total IoU: **matched_mask_iou_mean 0.717462, gt_normalized_mask_iou 0.556977**, coverage 0.776316, IoU>=0.50 0.588816, IoU>=0.75 0.460526, over 304 canonical instances and 304 predictions with 236 overlapping assignments and 68 unmatched on each side. The two headlines are reported side by side because they answer different questions - mask quality where something was found, versus the same IoU sum spread over every instance including the 68 misses - and neither is a COCO AP. **scipy was added as a runtime dependency after a recorded methodological review**: `linear_sum_assignment` is the matching rule the protocol names, greedy matching would be a different rule and a hand-rolled Hungarian solver would put a non-trivial algorithm on a reported-metric path; the frozen ML stack was verified unchanged (torch 2.11.0+cu128, torchvision 0.26.0+cu128, ultralytics 8.4.138, numpy 2.5.2, CUDA 12.8). Training read a git-ignored hard-linked **runtime view** of the approved adapter, its label digests verified identical before and after, so the framework's `.cache` files never entered phase 8A's evidence - the audited directory ended the phase with zero caches and zero checkpoints. **One artifact-write re-execution is recorded and is not a second experiment**: the first attempt trained, validated and ran the diagnostic, then the sensitive-content scan refused to write, having caught machine-specific absolute paths copied out of the framework's args.yaml. No artifact was published; training was not repeated; validation and the diagnostic were re-executed deterministically on the same frozen checkpoint and **reproduced identical numbers**. Three recording defects were fixed in the same pass, none touching a metric: absolute paths are now rewritten repository-relative, the model-summary parser no longer mis-reads thousands separators (it had recorded 543 parameters instead of 2,843,583), and the effective validation arguments are resolved through the framework's own `get_cfg` because `Model.val` does not keep its validator. **Peak GPU memory for the training run is `NOT_PERSISTED_FOR_THIS_RUN`** - the measuring process exited before the artifact was written and it is not recoverable from any file, so it is recorded as absent rather than reconstructed from terminal output. `best.pt` SHA-256 `d7b512b9...` and `last.pt` `9b8a956f...`, 6041685 B each, neither committed. `S0_experiment_sha256` `1761007a...`. Phase 8A, 8B and 7D artifacts byte-identical before and after. **No final segmenter was selected** - `final_segmenter: UNSELECTED_PENDING_REVIEW`, no S1, no alternative resolution or batch, no threshold tuned, no metric added. The detector was not trained, validated, inferred with or benchmarked. `CSVISION_ALLOW_TEST_SPLIT` unset; no holdout adapter, label, prediction or statistic exists. Eleven metric-only figures committed; no dataset or prediction imagery and no checkpoint. 1513 tests pass. Phase 8D not started. |
| 2026-09-09 | Phase 8D analysed why S0's mask quality trails its box quality. **No model was trained, S0 was not re-validated and no threshold was changed**: inference was re-run on the frozen validation split under the settings phase 8C froze, read from the committed protocol and asserted rather than restated, and the resulting per-instance table **reproduces the committed 8C aggregates exactly** - which is how it is known to describe the same run. The 8C figures were not regenerated. Outcome census over all 304 canonical instances: 140 `HIGH_QUALITY_MASK`, 68 `DETECTION_MISS`, 57 `LOW_OVERLAP_MASK`, 39 `MODERATE_MASK`; the four bands partition, so they sum. **Coverage and mask quality fail in opposite directions across size** - coverage rises monotonically with canonical area (0.513 / 0.816 / 0.882 / 0.895 by quartile) so misses concentrate in small masks, while matched mask IoU is **worst in the largest quartile** (0.642 against 0.751 and 0.763 in the middle two). Large objects are reliably found and poorly delineated; small ones are missed outright, and reading either half alone gives the wrong story. **`person` is weaker than every other class at every size quartile** (0.592 / 0.539 / 0.636 / 0.561 against non-person 0.789 / 0.844 / 0.829 / 0.822), so its deficit is not a size artefact; its box-minus-mask AP gap is 0.251045 against 0.039179 for the next class. **The substantive finding is a target-versus-evaluation mismatch created by the frozen protocol.** `overlap_mask: true`, and the installed `polygons2masks_overlap` sorts instances by descending area with a running maximum, so the **smaller** instance owns any shared pixel - a vest owns the pixels of the person wearing it and that person's training target is the remainder. Measured over all 137 validation persons: 27.3% of canonical person pixels are contested by another class, but **71.0% of the pixels S0 misses on person lie in that contested region**, 2.5x the base rate; Spearman between contested fraction and person mask IoU is **-0.5466**; persons under 20% contested average 0.691414 matched IoU against 0.408306 over 40%. Re-scoring the **same** predictions with the **same** matching against the overlap-resolved target the model was actually trained on moves person GT-normalised IoU 0.434635 -> 0.504610 and matched IoU 0.578106 -> 0.671180, while every other class moves by less than 0.002 - exactly the signature the mechanism predicts, since the compact classes are the ones winning the contested pixels. **It is labelled `POST_HOC_HYPOTHESIS_GENERATING` and is explicitly not a finding**: it was motivated by an observation made while reviewing images, re-scoring changes the measuring stick rather than the model, and it does not close the gap - even against its own target person matched IoU 0.671180 trails `helmet_loose` 0.935424. **Adapter conversion is not the dominant cause**: the 20 worst S0 instances average adapter IoU 0.968401 against 0.979140 for the split, the adapter-versus-model rank correlation is 0.246, person's mean adapter fidelity is 0.975368 and only 4 validation instances fall in the adapter-risk band - which is not a claim of zero effect, since phase 8A quantified real loss. The error taxonomy, its thresholds and the deterministic selection rules were **frozen before any image was opened**, and the review set's identifiers were recorded first; automated mechanism flags are kept separate from human ones, and `INSTANCE_SEPARATION_ERROR` and `OCCLUSION_ASSOCIATED` are deliberately not assignable by arithmetic. **6 of the 43 selected instances were visually inspected**, and the artifacts record `instances_selected` and `instances_visually_inspected` as separate fields so neither is mistaken for the other. Two of the six are indoor office scenes with fragmentary `person` annotations - one covering only hair and hands beside a fully visible unannotated face - recorded as `UNATTRIBUTED` because the frozen taxonomy has no flag for an annotation-scope problem, and two instances establish no rate. Causal labels such as `MODEL_CAPACITY_LIMIT` are **refused by the recorder**, not merely discouraged, because this phase ran no controlled experiment. Candidates were assessed and none chosen: `mask_ratio: 2` `WEAKLY_MOTIVATED` (under-segmentation outnumbers boundary error 56:19 overall and 49:14 within person, so finer supervision does not address the measured failure); YOLO11s-seg `NOT_SPECIFICALLY_MOTIVATED` (nothing here isolates capacity, and **absence of evidence is not evidence that capacity cannot help** - though phase 7B's D1 came in below D0 on the analogous detection intervention); `overlap_mask: false` best motivated by the evidence but **not a clean comparison**, because `SegmentationValidator._prepare_batch` builds its ground truth with `masks == index` when the flag is set, so S0 and such an S1 would have incomparable framework mask mAP while the direct IoU diagnostic against canonical COCO would stay comparable. One documentation defect introduced in phase 8C was repaired: the README status block carried a truncated sentence fragment. `S0_experiment_sha256` and every phase 8A, 8B, 8C and 7D artifact are byte-identical before and after. **No final segmenter was selected, no S1 protocol was frozen and no experiment was authorised**; the detector was not touched; `CSVISION_ALLOW_TEST_SPLIT` unset and no holdout image, mask, identifier or statistic exists in any artifact. Review figures render dataset imagery and stay git-ignored. 1577 tests pass. Phase 8E not started. |
| 2026-09-09 | Phase 8E froze a canonical comparison protocol and S1. **No model was trained, S0 was not retrained and its published metrics were not regenerated.** The phase exists because of a problem phase 8D uncovered: `overlap_mask` decides the framework's training target **and** its validation ground truth (`SegmentationValidator._prepare_batch` builds GT with `masks == index`), so S0's and an `overlap_mask: false` S1's native mask AP would be measured against different targets, and differencing them would compare the targets as much as the models. Native mask AP is therefore labelled `NOT_CROSS_TARGET_COMPARABLE_FOR_S0_S1_SELECTION` - **demoted, never suppressed**; S1 must still report it in full. A common evaluator was frozen instead: pycocotools `COCOeval` at `iouType=segm` against the canonical phase 5D COCO validation masks, which no training flag can move. IoU 0.50:0.05:0.95, `maxDets` [1, 10, 100], imgsz 768, NMS IoU 0.70, conf **0.001**, `retina_masks: true`, no TTA, exact binary-mask RLE through pycocotools, and **canonical category ids used directly with no remapping** - verified rather than assumed, since id 0 is legal COCO and pycocotools handles it. Fingerprint `282eb0ec...`. The conf of 0.001 is explicitly **not an operating point**: AP integrates over the score curve and needs the low-scoring tail, while the phase 8C direct-IoU diagnostic keeps its own operational 0.25 and the two protocols are never mixed; likewise the model may propose 300 candidates while COCOeval scores at its conventional cap of 100, and both numbers are recorded so neither is read as constraining the other. The evaluator was validated on synthetic fixtures - perfect match scores 1.0, disjoint and wrong-class score 0.0, duplicates cannot raise the score, score ordering matters, RLE round-trips pixel-exactly - **before it was pointed at any checkpoint**, and it reproduced identically across two independent executions. **S0 canonical reference, executed exactly once** and classified `POST_S0_PRE_S1_CANONICAL_COMPARISON_REFERENCE_EVALUATION`: `CANONICAL_SUPPORTED_MACRO_MASK_MAP50_95` **0.484643**, all-class mAP@0.50:0.95 **0.388009**, mAP@0.50 **0.537536**, over 6838 scored detections; per class AP@0.50:0.95 helmet_loose 0.793946, helmet_on_head 0.609345, vest_on_body 0.400246, person 0.135036, vest_loose 0.001474. **The canonical and native numbers are not comparable in absolute terms and must never be differenced** - different ground truth, implementation and confidence - but the shape is legible: the compact classes land close under both evaluators while `person` does not (canonical 0.135036 against native 0.271182). That **corroborates** the phase 8D overlap-target mechanism from an evaluator built for another purpose; it is not proof, and it is not a prediction that S1 will be better. **S0's earlier results stand unrevised**: native mask mAP@0.50:0.95 0.407942 remains a valid `NATIVE_TARGET_EVALUATION`, direct GT-normalised mask IoU 0.556977 remains a valid `CANONICAL_GT_RECOVERY_DIAGNOSTIC`, and the phase 8D re-score stays `HYPOTHESIS_GENERATING_DIAGNOSTIC_ONLY` - a third measurement was added beside them and none withdrawn. **S1 is frozen and not executed** (`FROZEN_NOT_EXECUTED`): the only intentional difference from S0 is `overlap_mask` `true -> false`, with 43 other framework arguments inherited unchanged and any second override refused by the parser; same `yolo11n-seg.pt` (`55ed65c5...`, 6182636 B), same approved adapter label bytes, imgsz 768, batch 8, epochs 100, seed 42, **mask_ratio 4**, and the same `ULTRALYTICS_NATIVE_SEGMENTATION_FITNESS` checkpoint policy. Selection is frozen at margin **0.005** with three cases decided in advance - `S1_IMPROVES_S0_BEYOND_MARGIN`, `PRACTICALLY_EQUIVALENT` (S0 preferred, because the baseline exists and no material canonical advantage was shown), `S1_BELOW_S0` - and the margin is an engineering threshold, not a significance test. A direction disagreement between the canonical AP and the direct IoU is recorded as `CROSS_METRIC_DIRECTION_DISAGREEMENT`, ranked by the canonical metric, and **never resolved by a composite**, which both parsers refuse. Two limitations are recorded rather than glossed: this policy was frozen **after** its reference ran, unlike phase 7A's which predated both candidates - `protocol_timing` is `POST_S0_PRE_S1_PROTOCOL_FREEZE` and the parser refuses any claim otherwise, though no S1 number influenced any rule; and `overlap_mask` changes the treatment environment rather than a mere hyperparameter, since each experiment's `best.pt` is chosen by native fitness computed against its own target, which is exactly why the decision is made externally. **S1 memory feasibility established**: one forward and backward pass at batch 8, imgsz 768, `overlap_mask: false`, peak 3.465 GiB reserved, mask target `[33, 192, 192]` - one plane per instance instead of a single indexed map, which is the growth the check exists to size. `NON_EXPERIMENTAL`: **no optimizer step, no validation, no checkpoint written and no metric of any kind**. Every phase 7D, 8A, 8B, 8C and 8D artifact is byte-identical before and after. `CSVISION_ALLOW_TEST_SPLIT` unset; no holdout image, mask, identifier or statistic exists in any artifact. **No final segmenter selected** (`UNSELECTED`), S1 not trained. 1667 tests pass. Phase 8F not started. |
| 2026-09-10 | Phase 8F ran **S1 exactly once** under the protocol phase 8E froze before it existed, and applied the frozen comparison policy. **S0 was read, never re-executed**: its checkpoint bytes were verified by digest and its committed canonical, native and direct-IoU figures were quoted, not recomputed. The only intentional difference is `overlap_mask` `true -> false`, **resolved in code from S0's own protocol** with 43 framework arguments inherited unchanged and any second override refused by the parser; the resolved set was checked against phase 8E's `inherited_protocol` block *and* the run's own `args.yaml` afterwards, so the intervention is known to have reached the trainer rather than merely been requested. **One trap was found in the installed source and avoided**: `Model._reset_ckpt_args` keeps only `imgsz`, `data`, `task` and `single_cls` from a checkpoint and the framework default is `overlap_mask: True`, so validating S1 without passing the flag would have scored it against **S0's** target with a plausible-looking number; the runner passes it explicitly and refuses to continue if the resolved value is not `False`. Execution: YOLO11n-seg from the same `yolo11n-seg.pt` (`55ed65c5...`, 6182636 B), imgsz 768, batch 8, seed 42, mask_ratio 4, deterministic, AMP, 100/100 epochs in 1195.05 s with no early stop, peak **4.102 GiB** reserved - no OOM at the frozen batch. Best epoch **77** at native composite fitness **0.974520**, verified as the argmax of the composite recomputed from `results.csv`; `optimizer: auto` resolved to **AdamW at lr0 0.001111, momentum 0.9**, captured directly from the framework log and identical to S0's, so the optimizer does not confound the comparison. `best.pt` `29337d67...`, `last.pt` `06a9af1d...`, 6041685 B each, neither committed. **Primary result, validation only: `CANONICAL_SUPPORTED_MACRO_MASK_MAP50_95` 0.559463** against S0's committed 0.484643 - delta **+0.074820**, classified **`S1_IMPROVES_S0_BEYOND_MARGIN`** under the 0.005 margin frozen before the run. All-class canonical mAP@0.50:0.95 0.455034 (S0 0.388009), mAP@0.50 0.634023, over 5485 scored detections. The secondary diagnostic moves the same way under the **unchanged** phase 8C protocol at conf 0.25: GT-normalised mask IoU 0.556977 -> **0.635356** (+0.078379), matched mask IoU 0.717462 -> 0.794849 (+0.077387), coverage 0.776316 -> 0.799342 - `CROSS_METRIC_DIRECTION_CONSISTENT` under a boundary rule written before S1 ran (a disagreement needs strictly opposite signs; a zero delta has no direction). **The aggregate is not a uniform effect, and the artifacts say so mechanically rather than in prose**: `person` moved **+0.317316** and carries **88.6%** of the total gain across admitted classes, while `helmet_loose` **regressed -0.059035**; `helmet_on_head` +0.028340 and `vest_on_body` +0.012659. Each admitted class's contribution to the primary delta is recorded (-0.014759 / +0.007085 / +0.079329 / +0.003165) so the mean cannot be read as uniform. **Why any individual class moved is UNKNOWN** - the experiment varied one flag and measured the outcome; it tested no per-class mechanism, and the phase 8D overlap-target analysis it is consistent with remains `POST_HOC_HYPOTHESIS_GENERATING`. The comparison is controlled, so the movement is attributable to `overlap_mask` **for this pair of runs**; it is not an effect size, because nothing was repeated and run-to-run variance stays UNKNOWN. **Native metrics reported in full and demoted**: mask mAP@0.50:0.95 0.458206, mAP@0.50 0.663319, precision 0.719937, recall 0.599587; box mAP@0.50:0.95 0.518779, mAP@0.50 0.727308. Marked `NATIVE_TARGET_METRIC` / `NOT_CROSS_TARGET_COMPARABLE_FOR_S0_S1_SELECTION` and **never differenced** against S0's, because `overlap_mask` reshapes the native validation ground truth too; the descriptive native supported macro 0.549273 decides nothing, and the two checkpoints were selected by the same native rule computed against **different** targets - a recorded limitation, not a detail. `vest_loose` moved +0.035845 and stays `DESCRIPTIVE_HIGH_UNCERTAINTY`, reported in full and deciding nothing. **Determinism was measured, not assumed**: validation, the canonical evaluation and the diagnostic were re-executed on the same frozen checkpoint to re-render a prose addition, and the canonical, direct-IoU and results artifacts came back **byte-identical** with `S1_experiment_sha256` unchanged; training was not repeated and this is one experiment, not two. A re-render was also caught **downgrading measured evidence** - peak GPU memory and wall clock, which only the training process can measure - and the runner now carries them forward from the manifest that process itself wrote, labelled as carried rather than re-measured, instead of silently writing `NOT_PERSISTED_FOR_THIS_RUN`. Four validators run **before** anything is published: the result manifest, the canonical evaluation, the direct-IoU diagnostic, and the comparison, whose delta is recomputed from the committed S0 value rather than trusted. Two phase 8D/8E guard tests that asserted "no S1 result exists" were rewritten to assert what they were protecting - that phase 8E's protocol artifact still records `FROZEN_NOT_EXECUTED` and that the 8D analysis selected nothing - rather than a fact that is now false by authorisation. Every phase 7D and 8A-8E artifact is byte-identical before and after; adapter fingerprints and S0's checkpoint bytes unchanged. `CSVISION_ALLOW_TEST_SPLIT` unset at process, user and machine scope; **zero image identifiers of any kind** appear in any emitted artifact. Eleven metric-only figures committed; no dataset imagery, no checkpoint. **No final segmenter selected** (`UNSELECTED_PENDING_REVIEW`) - the frozen rule produces a classification, not a decision. No S1 re-run, no `mask_ratio` variant, no YOLO11s-seg, no resolution or batch change, no threshold tuned, no metric added, no composite score. `S1_experiment_sha256` `d14b98fb...`. 1814 tests pass. Phase 8G not started. |
| 2026-09-10 | Phase 8G selected and froze the project's final segmenter. **Nothing was trained, evaluated, inferred or benchmarked**: the phase reads the committed S0 and S1 artifacts, re-derives the comparison arithmetically, records the human review, and freezes the selected checkpoint's identity. **The winner was derived, not asserted**: both primary figures were read out of each experiment's own committed canonical evaluation, the delta recomputed, and the frozen 0.005 margin reapplied by the same `classify_delta` the policy names - no experiment id appears in the freeze script as the answer, and a test asserts that, alongside a test that the script contains no training, validation or inference call path at all. **Human review confirms the policy; it does not override it**, and the script stops as `INVALID_SELECTION_STATE` rather than record a selection the arithmetic does not support. **Selected: S1** - YOLO11n-seg, imgsz 768, batch 8, mask_ratio 4, **`overlap_mask: false`**, best epoch 77, `selection_status: FINAL_SELECTED`, `selection_method: PREDECLARED_CANONICAL_POLICY_PLUS_HUMAN_REVIEW`, `margin_classification: S1_IMPROVES_S0_BEYOND_MARGIN`. The evidence: `CANONICAL_SUPPORTED_MACRO_MASK_MAP50_95` 0.484643 -> **0.559463**, delta **+0.074820**, roughly 15x the engineering margin; canonical all-class mAP@0.50:0.95 0.388009 -> 0.455034 moves the same way; and the secondary direct GT-normalised mask IoU 0.556977 -> 0.635356 does too, giving `CROSS_METRIC_DIRECTION_CONSISTENT`. **Native framework AP did not arbitrate** - `overlap_mask` changes the native validation target, so the two experiments' native numbers are measured against different ground truth and are reported without being differenced. **The trade-off is published, not buried**: `helmet_loose` **regressed -0.059035** while the aggregate rose, and `person` (+0.317316) carries most of the gain. Selection does not require every class to improve; it requires the predeclared metric to clear the predeclared margin. **Why any individual class moved is UNKNOWN**, and the `person` result is recorded as `CONSISTENT_WITH_PHASE_8D_OVERLAP_TARGET_HYPOTHESIS` and explicitly **not** `PROOF_OF_CAUSAL_MECHANISM` - phase 8D was `POST_HOC_HYPOTHESIS_GENERATING` and this phase ran no experiment to test the mechanism. `vest_loose` moved +0.035845, stays `DESCRIPTIVE_HIGH_UNCERTAINTY`, is absent from the supported macro and decided nothing. **Checkpoint identity**: S1's `best.pt`, SHA-256 `29337d67...`, 6041685 bytes, verified against the committed manifest before anything was written, with a byte-identical immutable copy placed at `artifacts/frozen/segmentation/S1_best.pt`, outside the run directory a re-run would overwrite. **A rejection specific to segmentation**: S0 and S1 are the same architecture at the same size, produced by the same protocol, and their checkpoints are the same number of bytes - what separates them is what they were trained to predict. So `overlap_mask` is part of the recorded identity and of the semantic fingerprint, and the accessor rejects S0's checkpoint (`d7b512b9...`) **by digest** as well as any `last.pt`; both refusals are tested against the real files on disk. `final_segmenter_sha256` `63ef41961ba3fedfef46658754940fb6ce075cdd93087b926d7c6620402579a7`. **Idempotency was measured**: the freeze was generated twice and all four semantic artifacts came back byte-identical, after one defect was found and fixed - the manifest had carried `frozen_copy.action`, which records whether that invocation created the copy or found it present, and therefore described the run rather than the frozen model; it moved to the provenance record. Three earlier guard tests that asserted "nothing is selected" or "no S1 result exists" were rewritten to assert what they were protecting - that phases 8D, 8E and 8F each selected nothing themselves - rather than facts now false by authorisation. A roadmap numbering discrepancy is recorded rather than silently corrected: the planned phase 9, "segmentation experiments and model freeze", was delivered by phases 8E-8G, so the next phase of actual work is phase 10. Every phase 7D and 8A-8F artifact is byte-identical before and after. `CSVISION_ALLOW_TEST_SPLIT` unset at process, user and machine scope; no holdout identifier, metric or artifact exists anywhere in this phase's output. No model binary committed. **Both models are now frozen, and both selections rest entirely on validation evidence - the holdout has still never been evaluated.** 1888 tests pass. Phase 9/10 not started. |
| 2026-09-10 | Phase 10A froze the detector-versus-segmenter comparison protocol. **No model was executed, no prediction produced, no latency measured and no image pixel read**; the manifest records those as counts, all zero, and a test asserts the freeze script imports neither torch nor ultralytics and contains no timing call. Protocol fingerprint `d92a157637baf52f9a7248bf24257d5a177e8496b7727ffb5afc874ab96d1c4d`. **The framing is deliberate and is the first thing the protocol fixes**: D2 emits class + confidence + box, S1 emits those plus an instance mask, so they do not solve the same output task and a single ranking would be meaningless. The question is what the mask adds and what it costs, across recognition, spatial information, computational cost and operational person-PPE reasoning; `aggregate_benefit_score: false` and `winner_declared: false`, both refused by the parser. Both frozen models were verified by digest and left alone - D2 `0466f872...` and S1 `29337d67...`, **both at imgsz 768**, so resolution is held constant rather than becoming a confound. Population: **validation only**, the frozen 65 images / 304 annotations read from the canonical phase 5D document rather than restated, membership fingerprint `54e4ae8a...`, both models seeing the same images in the same order. **Two confidence thresholds are frozen and may never be mixed**: AP at **0.001**, because average precision integrates over the score curve and needs the low-scoring tail, and operational analysis at **0.25**, because the spatial and association work needs a model to commit to a set of instances. **Precision is pinned at FP32 for both models as `quantize: 32`, after reading the installed source**: `half` is deprecated in ultralytics 8.4.138 and leaving `quantize` unset delegates the choice to the runtime, which could differ between the two models and would silently turn the latency comparison into a precision comparison. Recognition is judged by **one external evaluator** - pycocotools `COCOeval` at `iouType='bbox'` against the canonical phase 5D boxes, IoU 0.50:0.05:0.95, `maxDets` [1, 10, 100] - because the two models run through different framework validation paths and their native box metrics are not guaranteed to be computed identically. **S1's boxes in the comparison are S1's own predicted boxes** (`segmenter_boxes_derived_from_masks: false`): re-deriving them from its masks would improve their consistency with the mask branch and would then measure a post-processing choice this project invented. **Seven spatial features are frozen and each is paired with a box proxy or explicitly marked `NO_BOX_ONLY_EQUIVALENT`** - that pairing is the whole answer to "what does segmentation add", and deciding it after seeing numbers would be circular; `MASK_TO_BOX_FILL_RATIO` and `SHAPE_EXTENT` have no box equivalent. The person-PPE work is named **`SPATIAL_ASSOCIATION_ANALYSIS`, not compliance accuracy**, because the project holds no canonical compliance ground truth and `helmet_on_head` / `vest_on_body` already encode a provider-level worn state; four categories partition every candidate pair, with no embeddings, no tracking, no learned rule and no collapsed classes. **The latency protocol is frozen down to the iteration count**: batch 1, imgsz 768, FP32, 20 warmup iterations discarded, 30 timed repetitions over 20 benchmark images selected by `STABLE_SHA256_RANK_OF_IMAGE_ID` (ordered fingerprint `45059c2c...`) - ranked by the digest of the identifier, so the subset cannot have been chosen for being easy or crowded, because no image was opened to choose it. Execution **interleaves the two models and reverses the order on a second pass**, because benchmarking one to completion first would measure the laptop's thermal state as well; `randomized: false` and the runner aborts on any deviation from fourteen frozen fairness invariants. **Two timing boundaries are defined and mask reconstruction sits inside the segmenter's end-to-end one** - excluding it would hide precisely the cost this comparison exists to quantify, and the parser refuses that. Explicit `torch.cuda.synchronize()` on both edges of every timed region with the same primitive for both models, because CUDA work is asynchronous and an unsynchronised reading measures dispatch rather than execution. Memory is measured at batch 1 with peak stats reset **after** warmup, and training memory may not be substituted. The future result is pre-labelled `CONTROLLED_LOCAL_HARDWARE_BENCHMARK`, valid for this machine and runtime only. The freeze is **idempotent**: run twice, all four artifacts came back byte-identical. Every phase 7D and 8A-8G artifact is byte-identical before and after; both freeze manifests recompute their fingerprints; the committed membership reproduces its digest and the benchmark subset reproduces the frozen selection rule. `CSVISION_ALLOW_TEST_SPLIT` unset at process, user and machine scope; no holdout identifier, image or statistic exists anywhere in this phase's output. 1971 tests pass. Phase 10B not started. |
| 2026-09-10 | Phase 10B ran the detector-versus-segmenter recognition and spatial comparison on validation, under the protocol phase 10A froze. **Neither model was trained, modified or re-thresholded, no latency or memory was measured, and the holdout was never read.** **Effective FP32 parity was proved at runtime rather than assumed from the configuration**: both models were probed with a forward pre-hook before any comparison number existed, giving backend FP16 flag `false`, parameter dtypes `['torch.float32']`, input tensor `torch.float32`, autocast during forward `false` and no quantization config - **byte-identical between the two**; a difference would have stopped the phase as `PRECISION_PROTOCOL_MISMATCH`, because a comparison across two precisions measures the precision. Recognition was judged by **one external evaluator** (`COCOeval`, `iouType='bbox'`, canonical phase 5D boxes, IoU 0.50:0.05:0.95, maxDets [1, 10, 100]) over each model's **own** predicted boxes - nothing was derived from a mask. **Canonical box mAP@0.50:0.95: D2 0.485390, S1 0.505682, delta +0.020292**; mAP@0.50 0.641107 -> 0.692955, delta +0.051848. **The substantive finding is that the aggregate gain is carried entirely by the rare class, and quoting it alone would be metric-shopping.** The all-class figure is the unweighted mean of the five per-class APs, so it decomposes exactly: `helmet_loose` -0.003737, `helmet_on_head` +0.000199, `person` +0.006390, **`vest_loose` +0.026724**, `vest_on_body` -0.009284. `vest_loose` alone contributes more than the whole +0.020292, and it is the class already classified `DESCRIPTIVE_HIGH_UNCERTAINTY` with one validation image; **excluding it the mean delta over the other four classes is -0.008040 and the segmenter sits below the detector**. Two supported classes declined (`vest_on_body` -0.046421, `helmet_loose` -0.018685) and why any class moved is UNKNOWN. These canonical-evaluator figures are **not** the native framework metrics either model reported in its own phase and must never be differenced against them. Operational inference at conf 0.25 produced 269 D2 predictions and 336 S1 predictions, with **336 masks reconstructed on the original canvas and 0 exclusions** - a mask whose shape did not match its source image would have been excluded and counted, never resized into agreement. **Where masks earn their place is representation, not association**: `MASK_TO_BOX_FILL_RATIO` median 0.664 (P25 0.556, P75 0.775) and `SHAPE_EXTENT` median 0.672, both `NO_BOX_ONLY_EQUIVALENT` - a third of the median predicted box is not the object, and instances do not fill even their own tight rectangle. Where a box proxy exists it is **systematically inflated rather than merely noisy**, because a box counts background as object: mask area mean 144563 px against box-proxy 237206 px, person-PPE intersection 26828 px against 47281 px; centroid displacement from the box centre median 16.0 px, P95 126.9 px, max 262.8 px. **For association the honest result is that the box proxy is close**: holding the model constant at the frozen 0.50 containment floor, 103 agree, 3 box-only, **0 mask-only**, 66 neither and 1 both-but-different-person over 173 relationships - the mask changes almost no association decision. The pipeline-level reading against D2's boxes disagrees far more (81/7/6/62/17) but is confounded by being a different model with different instances, and is reported as a separate question rather than as the geometry comparison. **A gap in the frozen taxonomy was found by applying it**: phase 10A froze four categories assuming a rule either associates or does not, but two rules can both associate and pick different people; that case is counted as `BOTH_ASSOCIATED_DIFFERENT_PERSON` alongside the four rather than absorbed into one of them, and **no fifth category was added to the frozen protocol**. 11 of 349 candidate pairs (3.2%) had overlapping boxes whose masks shared no pixel, reported as a count and a fraction because phase 10A declared no threshold for 'strong' or 'minimal' - inventing one now would be a post-hoc bin. `VISIBLE_PPE_COVERAGE_PROXY` stays `INTERPRETIVE_OPERATIONAL_PROXY`, the one frozen quantity whose definition is qualitative, and **no compliance accuracy is claimed anywhere** because the project holds no compliance ground truth. A half-pixel convention offset in the mask centroid (pixel indices against continuous box coordinates, ~0.71 px for a perfectly filling mask) is documented and pinned by test so a small non-zero value is not read as a real shift. Both result artifacts validate, every delta and the decomposition recompute, the semantic fingerprints reproduce across two independent executions (box `68c9826a...`, spatial `9877b88d...`), and every phase 7D, 8G and 10A artifact is byte-identical. `CSVISION_ALLOW_TEST_SPLIT` unset at process, user and machine scope; no holdout identifier, image or statistic exists in any artifact. **No latency or memory benchmark was executed** (`latency_measured: false`); that is phase 10C. 2026 tests pass. Phase 10C not started. |
| 2026-09-10 | Targeted pre-push correction to the phase 10B association result semantics. **No model was loaded or executed**: every count the correction needed was already recorded, so the restructure is arithmetic over committed numbers, and a test asserts the correction script imports neither torch nor ultralytics. Applying the phase 10A taxonomy to real predictions turned up a state its four categories do not describe - **both rules associate, to different people**. That is neither agreement, nor box-only, nor mask-only. It is now recorded as a **coverage exception** rather than as a fifth peer category: `association_taxonomy_status: FROZEN_TAXONOMY_NON_EXHAUSTIVE_FOR_OBSERVED_DATA`, `taxonomy_exception_type: BOTH_RULES_ASSOCIATE_DIFFERENT_PERSON`, status `UNCLASSIFIED_BY_FROZEN_FOUR_CATEGORY_TAXONOMY`. Adding a fifth category after seeing data is exactly what a frozen taxonomy exists to prevent, and it would also misdescribe the situation - the four were predeclared and this state was not. **The four frozen categories keep their names, their order and their counts, and the phase 10A protocol was not modified.** Denominators are now explicit: counts are over all relationships, percentages use `classified_relationships`, and taxonomy coverage is reported separately as protocol bookkeeping rather than as a spatial-performance metric. Geometry-isolating: **172 classified + 1 exception = 173**, coverage **0.994220**, frozen counts 103 / 3 / 0 / 66. Pipeline-level: **156 classified + 17 exceptions = 173**, coverage **0.901734**, frozen counts 81 / 7 / 6 / 62. A different-person outcome is labelled `ASSOCIATION_RULE_DISAGREEMENT` and explicitly **not an error**, because the project holds no person-PPE association ground truth for either rule to be wrong against; and the 17 pipeline-level exceptions are recorded as **not attributable solely to geometry**, since that reading also varies the model and the instances. A descriptive support sensitivity was added alongside the unchanged frozen metrics, applying the project's **pre-existing** support rule: D2 **0.589729**, S1 **0.581689**, delta **-0.008040** over the four adequately supported classes, labelled `POST_HOC_DESCRIPTIVE_SUPPORT_SENSITIVITY` and flagged as not a frozen metric, not a selection rule and not a significance test. The recorded conclusion is that **S1 retains broadly similar localisation capability to D2 while adding mask output, but the positive all-class delta is driven by the highly uncertain vest_loose class and is not robust evidence that S1 is the superior object localiser**. One defect is recorded rather than hidden: phase 10B's first-pass row records did not store which person each rule selected, which is why the correction had to work from counts - sufficient for everything this correction needed, and the runner now records both selected person indices so a future execution can reproduce an exception down to the person. The correction is idempotent: run twice, the artifacts come back byte-identical. Every phase 10A artifact and both model freezes are byte-identical; no canonical metric, continuous spatial metric, raw association decision or model prediction changed. `CSVISION_ALLOW_TEST_SPLIT` unset, holdout untouched, no latency benchmark. 2041 tests pass. Folded into the unpushed phase 10B commit. |
