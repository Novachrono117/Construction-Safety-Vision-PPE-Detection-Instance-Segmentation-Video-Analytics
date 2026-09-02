# Roadmap

Version: 1.3 · Current phase: **5 - Split freeze and task-specific dataset generation** (not started; phase 4 complete)

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
| 4 | Dataset/annotation audit and EDA | done (4A automated, 4B visual review) |
| 5 | Split freeze and task-specific dataset generation | **next** · not started |
| 6 | Detection baseline | not started |
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
  split is not reusable, the six confirmed semantic-duplicate groups must stay
  intact, and the current-source-vs-v4 annotation drift must be resolved before
  anything is frozen.
- **Outputs.** `data/processed/splits/{train,val,test}.manifest.csv` (image ID,
  file hash, group ID); `data/processed/detection/` and
  `data/processed/segmentation/`; the box-from-polygon derivation code; a
  provenance record for the freeze; `reports/split_report.md`.
- **Validation gate.** Splits are disjoint by image ID and by duplicate/sequence
  group (no near-duplicate or same-sequence frame crosses a split boundary);
  realised proportions match the configured ratios within a stated tolerance;
  every class appears in every split, or its absence is explicitly justified;
  an automated check confirms the detection and segmentation datasets contain
  the same image IDs per split; boxes are verified to be the tight bounds of
  their polygons on a sampled subset; the manifests are hashed and committed.
  **From this point the test split is locked.**
- **Academic mapping.** C1, and the alignment requirement of C3.

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
