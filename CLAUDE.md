# CLAUDE.md - Operating constitution for this repository

This file governs every Claude Code session in this repository. It refines the
global engineering guidelines and, on conflict inside this repository, it wins.

This is an academic project **and** a public portfolio project. Its value comes
entirely from the honesty of its evidence: a wrong number that looks right is
worse than no number at all. Every rule below exists to protect that.

Read alongside: [`reports/rubric_contract.md`](reports/rubric_contract.md)
(what must be delivered) and [`reports/roadmap.md`](reports/roadmap.md) (in what
order, and the gate each phase must pass).

Repository: <https://github.com/Novachrono117/Construction-Safety-Vision-PPE-Detection-Instance-Segmentation-Video-Analytics>

---

## 1. Test data is protected

The `test` split is a locked holdout. Since phase 5C.2 it is a **frozen, named
set of images**, recorded in `reports/split_manifest.json` under
`holdout_sha256`. It is no longer a plan; it is protected data.

- It may be read **once**, in phase 11, after both models are frozen. Every phase
  in between - 5D, 6, 7, 8, 9 and 10 - develops on `train` and `validation` only.
- Until then: no training on it, no metrics from it, no threshold tuning on it,
  no "just to check" inspection, no plotting it, no looking at its images.
- Model and hyperparameter selection use the **validation** split only.
- Access goes through `construction_safety_vision.splits.assert_split_allowed`,
  and to the frozen ids through
  `construction_safety_vision.data.split_freeze.FrozenSplits`, which routes to
  the same guard. Both require two independent opt-ins: `allow_test=True` in code
  **and** `CSVISION_ALLOW_TEST_SPLIT=1` in the environment. Never set that
  variable "to make an error go away", never weaken the guard, never add a
  helper that bypasses it, and never bypass it by reading the `test` section of
  `split_manifest.json` or the `test` rows of `final_split_assignments.csv`
  directly. Those files record membership so it can be audited, not so it can be
  used.
- If the holdout is touched by accident, say so immediately and in full. A
  disclosed leak is a limitation; a hidden one is fraud.

## 2. No fabricated metrics

- Never write a number that was not produced by an executed run. This includes
  examples, placeholders that look plausible, "typical" values, and numbers
  carried over from a paper, a tutorial, or a previous session's memory.
- Never state dataset statistics (image counts, class balance, split sizes)
  before the audit phase has measured them.
- If a value is unknown, write `TBD` and say which phase will produce it.
- Every reported metric must be traceable to a committed metrics file and a
  provenance record. If it cannot be traced, it cannot be reported.
- Report failures as plainly as successes: a run that diverged, a class the
  model cannot detect, or a metric worse than the baseline all get written down.

## 3. No causal claims without experimental support

- Distinguish `FACT` (measured), `LIKELY` (strongly indicated), `HYPOTHESIS`
  (plausible, untested) and `UNKNOWN`. Label the weaker ones explicitly.
- "Augmentation X improved mAP" is a claim about causation. It requires a
  controlled comparison in which only X differed. Otherwise write "run B, which
  differed by X and by Y, scored higher; the cause is untested".
- Two runs differing by a fraction of a point do not establish an ordering.
  Acknowledge run-to-run variance instead of ranking noise.
- Explanations of error patterns are hypotheses until an experiment tests them.

## 4. No hidden manual dataset changes

- `data/raw/` is immutable. Never edit, delete, re-encode or "fix" a raw file.
- Every correction, filter or relabel is implemented as code in the
  preprocessing pipeline, so it is visible, reviewable and reproducible.
- Never hand-edit an annotation file, a manifest or a metrics file.
- Removing images (duplicates, corrupt files, unlabelled samples) is a pipeline
  step with a recorded count and a stated criterion, never an ad-hoc deletion.

## 5. All preprocessing must be reproducible

- Everything under `data/interim/` and `data/processed/` must be re-derivable
  from `data/raw/` by running committed code with a committed configuration.
- No transformation lives only in a notebook cell or a shell one-liner.
- Every stochastic step draws from the configured seed and records it.
- If a step cannot be reproduced, it is a defect: fix the pipeline rather than
  keeping the output.

## 6. Split alignment across tasks

- Instance segmentation is the canonical annotation source. Detection boxes are
  **derived mathematically** from the polygons, never annotated separately.
- Detection and segmentation use exactly the same image IDs in `train`, `val`
  and `test`.
- Alignment is verified by an automated check, not by assertion, and re-verified
  whenever either dataset view is regenerated.
- Splits are frozen in phase 5 and recorded in manifests. Changing a split after
  the freeze invalidates every result produced under it; if it must happen, the
  affected results are recomputed or withdrawn, and the change is documented.
- Duplicate and same-video-sequence images must not straddle a split boundary.

## 7. Data and model provenance must be recorded

- Every acquisition, derivation, training run and evaluation writes a
  provenance record (`construction_safety_vision.provenance`): source, code
  commit, configuration snapshot, environment, input and output file hashes.
- Datasets and checkpoints are identified by hash, never by "the latest run".
- A dataset without a recorded source, version and license is not used.
- Never commit downloaded data or trained weights; commit the record that makes
  them re-obtainable.

## 8. New dependencies require justification

- Prefer the standard library, then the already-installed set.
- Adding a dependency requires stating what it does, why the existing set is
  insufficient, and what it costs (size, build requirements, Colab
  compatibility, license, maintenance).
- Heavy frameworks are installed by the phase that needs them, not in advance.
- Pin through `pyproject.toml` plus `uv.lock`. Never `pip install` into a
  session without recording it.

## 9. Tests and linting before reporting completion

Run, from the repository root, before claiming any work is done:

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest
```

- Report the actual output. If a command was not run, say so and say why.
- "It should work" is not a result. Either it was executed, or it was not.
- Add tests for logic that protects the protocol (splits, alignment, derivation,
  metric plumbing). Do not add tests that only inflate the count.

## 10. Version control discipline

- **Never commit, push, tag, or rewrite history unless explicitly asked in the
  current session.** Approval given for one commit does not extend to the next.
- Never modify global git configuration.
- Never force-push, never `git reset --hard` on work you did not create, never
  delete branches on your own initiative.
- Commit messages in English, small and atomic, describing what changed and why.

## 11. Binary and model artifacts stay out of the repository

- Never commit: datasets, images, videos, `.pt`/`.pth`/`.onnx`/`.engine`
  checkpoints, archives, or notebook outputs containing embedded binaries.
- `.gitignore` enforces this; do not add exceptions casually.
- Large outputs are referenced by provenance record plus an external link.
- Small text artifacts (manifests, metrics JSON, provenance records) are
  committed, because they are the evidence.

## 12. Documentation must match the repository

- The README describes what exists **now**, including the honest project status.
- Never document a script, notebook, dataset or result that has not been
  created. Planned work is labelled as planned.
- When code changes, update the documentation that describes it in the same
  change.
- When a phase completes, update the status table in `reports/roadmap.md`.

## 13. Every experiment records its configuration

- No hyperparameter is passed only on a command line or typed into a notebook
  cell. It lives in a file under `configs/`.
- Configuration parsing is strict: unknown keys raise. Do not relax that to make
  a file load.
- To run a variant, add a new configuration file rather than editing one whose
  results are already reported.
- The report documents the complete hyperparameter set, not just the ones that
  were changed.

---

## Working agreements

- **Language.** Code, identifiers, docstrings, comments, commit messages and the
  technical README are in English. Conversation with the maintainer is in
  Brazilian Portuguese. Academic deliverables may have a Portuguese version.
- **Read before write.** Inspect the existing structure and conventions before
  editing. Do not assume a file, function or field exists.
- **Root cause over symptom.** Diagnose before changing code; no random edits in
  the hope that something works.
- **Scope.** Out-of-scope problems are reported and classified
  (`BLOCKER / HIGH / MEDIUM / LOW`), not silently fixed. Never change an API,
  schema or split silently.
- **Honesty over agreement.** If a proposed decision is technically wrong, say
  so, give the reason and offer a better option.
- **Secrets.** Never hardcode keys or tokens. Use `.env` (git-ignored) and keep
  `.env.example` in sync.
- **Phase discipline.** Work the current roadmap phase. Do not start a later
  phase because it seems more interesting, and do not start bonus work before
  the mandatory deliverables are complete.

## Current state (keep this accurate)

- **Phase:** 8D complete - the split is frozen, the holdout is locked, **the
  final detector is FROZEN (D2, YOLO11n at imgsz 768)**, the segmentation adapter
  is audited and approved, **S0 has been trained once**, and its validation error
  analysis is done (`S1_HYPOTHESIS_READY_FOR_PROTOCOL_FREEZE`). **No final
  segmenter is selected** (`final_segmenter: UNSELECTED_PENDING_REVIEW`) and **no
  S1 protocol is frozen**. Phase 8E - the human-reviewed S1 protocol freeze - has
  not started; do not start it unprompted.
- **Dataset:** acquired. Roboflow Universe `agis-workspace-8gs52/
  construction-ppe-compliance-detection` v4, COCO instance segmentation, CC BY
  4.0. Archive SHA-256
  `5c0c35f79be251af349f289ab300a8c706f260f9a5b52f66facfbd23466538f6`. Evidence in
  `reports/dataset_provenance.md`.
- **`dataset.verified: true` means provenance only.** Source, license, format,
  class list and artifact integrity were verified. It is not a statement about
  annotation quality; phases 4 and 5A examined that separately and their findings
  are below.
- **436, not 742.** The export contains 742 images, of which only **436 are
  independent source images**; the train split is offline-augmented x2 (306 ->
  612). Never quote 742 as a sample count, never let augmented variants of one
  source image land in different splits, and never treat them as independent in
  any statistic.
- **Canonical annotations are the LIVE source project, not the v4 export.**
  Decision `CURRENT_COMPLETE_GEOMETRY`, recorded in
  `reports/canonical_annotation_manifest.json`. 436 source images, **2031
  annotations**, of which 2029 carry complete geometry (1007 polygon, 1022 RLE)
  in **original image coordinates**. Quote these numbers, not the export's.
- **The v4 export's 3373 annotations are not the annotation count.** That figure
  counts the augmented copies too. The v4 *source snapshot* - one non-augmented
  representation per source image - holds **1961**, measured in phase 5A. An
  earlier inferred value of 1955 was superseded.
- **Geometry is recovered, not missing.** Phase 4A's "1022 annotations expose no
  geometry" was a consumption gap. A `mask`-type annotation carries its geometry
  inline as base64(zlib(COCO RLE counts)) against the full original canvas. Code
  that reads only `points` will silently drop half the dataset.
- **Two annotations have no segmentation at all** (image `OQJwjQoYsf1KUgr9G0V8`,
  ids `I` and `J`), classified `VALID_BUT_UNSUPPORTED_GEOMETRY`. They are real
  objects. Never drop them silently; phase 5B must materialise them explicitly.
- **The 76 additions since v4 are NOT all fragments, and all are retained.**
  Phase 5A said they were fragments; phase 5B measured otherwise and that reading
  is withdrawn. About half are degenerate slivers; the rest are corrections that
  *improve* the labels (a coarse polygon replaced by several tighter ones; one
  oversized `person` box covering two people replaced by one box each - which is
  new instance coverage). No deterministic rule separates them, so the owner set
  `fragment_rule_status = REJECTED_FOR_AUTOMATIC_FILTERING`. **All 2031
  annotations are retained; 0 excluded.** Never re-propose an automatic filter
  here, and never describe the live annotations as "better" merely for being
  newer.
- **The v4 drift set is NOT error ground truth.** The 76 additions are a
  *historical annotation-drift reference set*. Precision or recall measured
  against them describes agreement with drift, not annotation correctness. A rule
  at precision 1.0 selects annotations that all changed since v4 - not
  annotations that are all wrong. Do not make that inference.
- **Terminology:** the flag is `NESTED_SAME_CLASS_CANDIDATE` and it is
  descriptive. Do not call these annotations fragments unless a specific one has
  been established as such by manual review.
- **Known dataset limitations:** `vest_loose` is rare (45 instances, 8 images)
  and absent from the provider's test split; six confirmed semantic-duplicate
  groups must stay inside one split; three zero-instance images are out-of-domain
  exclusion candidates; the export declares a placeholder category `object` with
  no annotations that must be excluded from the class map without shifting
  indices; the provider's stored boxes disagree with its own geometry inside the
  v4 export by up to 123.5 px, so detection boxes are always derived from
  segmentation.
- **Provider split is rejected** for the final protocol
  (`UNSUITABLE_FOR_FINAL_PROTOCOL`), but the dataset is accepted with documented
  limitations. It must not appear in any artifact a split designer reads;
  `group_split_features.csv` deliberately omits it.
- **Modelling population (phase 5B):** 433 eligible images (436 minus 3 confirmed
  out-of-domain, which stay on disk and in the provenance population), 2031
  retained annotations, and 427 indivisible split units at that time (421
  singletons + 6 confirmed semantic duplicate groups). **The 427 is superseded by
  phase 5B.1's 422** - quote 422. Quote 433 for modelling and 436 for
  provenance; they are different populations and must not be conflated.
- **11 confirmed groups**, 411 singletons, **422 split units**. Phase 4B
  confirmed 6 cross-split pairs; phase 5B.1 confirmed 5 same-split pairs, because
  the provider split was rejected and a same-split relation constrains the new
  split just as much. Group numbering continues rather than restarting: 001-006
  still mean what they meant.
- **Two grouping bases, both indivisible, recorded in `group_manifest.csv`:**
  `EXACT_SEMANTIC_DUPLICATE` (10 groups, same frame stored twice) and
  `NEAR_DUPLICATE_SAME_SCENE` (1 group, `manual_dup_008`, same worker and scene at
  a different moment). Do not call the second an exact duplicate; do not treat it
  as separable either. Grouping serves statistical independence across splits,
  not image identity.
- **Zero unresolved near-duplicate candidates.** All 11 carry a human decision;
  none was merged on perceptual distance alone.
- **Duplicate groups are connected components**, not pairs. If A~B and B~C are
  both confirmed, they are one group {A, B, C}; grouping pairwise would let a
  split separate A from C while honouring each relation.
- **The 2 geometry-less records are materialised**, not dropped: four-corner
  rectangles clipped to the canvas, marked
  `geometry_origin = SYNTHETIC_FROM_PROVIDER_BBOX`. Never describe them as
  human-drawn segmentation.
- **Models:** **D0 exists** (detection). No segmentation model. The phase 6A
  one-epoch smoke test is `NON_EXPERIMENTAL` / `DO_NOT_REPORT_AS_MODEL_RESULT`
  and none of its numbers were recorded - never quote it as a result.
- **D0 result (validation only):** primary **mAP@0.50:0.95 0.464429**,
  mAP@0.50 0.619373, precision 0.85568, recall 0.539992.
  100/100 epochs, best epoch 67, `d0_experiment_sha256`
  `cbd79fd2f2f70eb31ede61b813f991e973bb5d2f69c223a3826ee6aeadd0ffb3`. **Every one of these is a validation number and
  says nothing about test performance.** Quote them only with that qualifier.
- **D0 was run once and must not be re-run to improve it.** It is the baseline
  reference for phase 7. Do not retune it, do not swap the model, and do not
  average several runs - `deterministic: true` reduces variance but does not
  remove it, so a second run would measure noise.
- **`optimizer: auto` resolved to AdamW at lr0
  0.001111**, not the file's
  `lr0: 0.01`. Ultralytics persists neither, so the manifest records the value
  together with how it was established
  (`resolved_optimizer_determination.source`). Never quote a config value as
  "what the run used" when the protocol declares a policy.
- **`vest_loose` scored precision 1.0 with recall 0.0 on 1 validation image.**
  That is what one image's evidence looks like, not a finding. It is marked
  `HIGH_SAMPLING_UNCERTAINTY`; never rank it against the other classes and never
  tune against it.
- **The split is FROZEN (phase 5C.2).** `reports/split_manifest.json` is the
  single authoritative membership: **303 / 65 / 65 images** over **294 / 63 / 65
  groups**, 2031 annotations at 1422 / 304 / 305, negatives 10 / 2 / 2. Read it
  through `construction_safety_vision.data.split_freeze.load_frozen_splits`,
  never by parsing a candidate file or re-running the search. Changing it
  invalidates every result produced under it.
- **Fingerprints of the freeze:** `split_assignment_sha256`
  `a230869ff4cb45f53654d67357a27f2a0def6d8ba79f9fa4880f0fdda2f046cc`,
  `holdout_sha256`
  `bb7ed43b20a84644d5a3917c6d0ead688132f82a30052b06ae7ad121e4851a00`. They differ
  from `candidate_assignment_sha256` by design: the candidate digest covers
  `(group, split)`, the freeze digest covers `(group, image, split)`.
- **`test` IS NOW THE PROJECT'S HOLDOUT and it is LOCKED.** It has never been
  evaluated, inspected or plotted. For the whole of phases 5D through 10 it must
  not be used for model selection, architecture selection, hyperparameter
  tuning, augmentation tuning, image-size tuning, threshold tuning, qualitative
  model debugging or error-driven iteration - and it must not be looked at.
  Phase 11 reads it once, after both models are frozen. Access needs
  `allow_test=True` **and** `CSVISION_ALLOW_TEST_SPLIT=1`; never set that
  variable, never weaken the guard, never add a helper that bypasses it, and
  never read the frozen test ids straight out of the manifest to sidestep
  `FrozenSplits`.
- **The provider's `test` split is still not the holdout** and is still rejected.
  Do not conflate the two.
- **Candidates 002-006 are preserved as `NON_SELECTED_PROVISIONAL_CANDIDATE`**,
  recorded in `reports/split_candidates/selection.csv`. Their column is still
  `provisional_split` and their historical scores must never be rewritten. Do not
  freeze, evaluate or partially adopt one.
- **Neither the split search nor the freeze reads the provider split** - not as
  input, initialisation or target. `optimize_split_candidates.py` refuses to run
  if the group feature table carries such a column. Keep it that way.
- **The rare class is named in configuration**, not hardcoded: `rare_class:
  vest_loose` in `configs/split_search.yaml` and `configs/split_freeze.yaml`. It
  occupies **7** indivisible units over 8 images, because two of them sit in
  `manual_dup_010`, so it moves in chunks and cannot be freely rebalanced.
- **`vest_loose` in the frozen split: 5 / 1 / 2 images, 30 / 8 / 7 instances.**
  Validation holds **one** image of it, so a `vest_loose` validation metric has
  high sampling uncertainty and **must not be used on its own** to select a model
  or a hyperparameter; rely on predeclared global/macro criteria and treat the
  per-class number as supporting evidence. The holdout holds two, so even the
  final per-class figure carries an explicit small-sample limitation.
- **Call it a "group-aware and class-aware constrained split".** Never call it
  perfectly stratified, and never claim statistical independence beyond the
  duplicate screening actually performed: two perceptual fingerprints plus human
  review of every candidate they raised. Nothing establishes that two images in
  different splits do not share a site, a day, a camera or a worker.
- **Task datasets (phase 5D): COCO is canonical for both tasks.**
  `canonical_detection_format: COCO`, `canonical_segmentation_format:
  COCO_INSTANCE_SEGMENTATION`, `model_specific_adapter: NOT_YET_SELECTED`.
  Materialised under `data/processed/canonical/` (git-ignored, re-derivable by
  `scripts/materialize_task_datasets.py`): **train 303 images / 1422
  annotations, validation 65 / 304**, totalling 368 / 1726.
- **No YOLO labels exist, and none may be written casually.** The canonical state
  holds polygon *and* compressed RLE; only COCO carries both. Any future
  RLE-to-polygon adapter must first convert, rasterise, compare against the
  canonical masks, report per-instance mask IoU and area error, and flag
  disconnected-component and hole cases - and be rejected if the loss is
  material. That is a phase 8 concern; it has not been done, so make no claim
  about how lossy it would be.
- **Detection boxes are derived from segmentation, never from the provider bbox.**
  Verified against the independent phase 5A measurement at max delta 0.0 px over
  all 1726 development annotations. Do not reintroduce the provider's box.
- **Images in `data/processed/` are byte-identical copies**, 368/368 verified. No
  resize, crop, re-encode, EXIF rotation or colour conversion happens anywhere in
  the materialisation. Do not add one.
- **The holdout is `NOT_MATERIALIZED_PROTECTED_HOLDOUT`.** There is no
  `data/processed/canonical/images/test/` and no `*_test.coco.json`, and nothing
  new has been measured about the test set. Materialising it runs the same
  function with the same configuration - never a split-specific branch - and
  needs both opt-ins.
- **Detection adapter (phase 6A) is DERIVED, never canonical.** YOLO labels live
  under `data/processed/adapters/yolo_detection/` (git-ignored). If a YOLO label
  and the canonical COCO file ever disagree, **the COCO file is right and the
  adapter is broken** - regenerate it, never edit it. It was proven lossless:
  1726/1726 boxes round-trip within 1e-4 px, max observed 1.47e-06 px.
- **The RLE-to-polygon fidelity audit HAS been done (phase 8A), and its numbers
  are the only ones that may be quoted.** All 1726 development annotations
  round-trip through the YOLO segmentation format: mask IoU mean 0.973066,
  median 0.984576, P05 0.918176, minimum 0.307692; **none exact**. By canonical
  representation - polygon 0.978091 (n=843), RLE 0.968538 (n=881), synthetic
  rectangle 0.849702 (n=2). Evidence in
  `reports/segmentation_adapter_fidelity_report.md` and
  `reports/segmentation_adapter_audit_manifest.json`. Never aggregate the strata
  into one figure; that is what the report exists to prevent.
- **The audit-only segmentation adapter is NOT the project's dataset.**
  `data/processed/adapters/yolo_segmentation_audit/` is git-ignored and marked
  `AUDIT_ONLY` / `NOT_CANONICAL` / `NOT_YET_APPROVED_FOR_TRAINING`. Do not train
  on it, do not promote it, and do not describe it as the segmentation dataset.
- **The YOLO segmentation format cannot express a hole or a disconnected mask,
  and that is a structural fact read from the installed source**, not an
  inference: one row is one class plus one flat ring with no separator, the
  framework's own mask converter uses `RETR_EXTERNAL`, and `polygon2mask` fills
  with `cv2.fillPoly` and no even-odd subtraction. 307 development instances
  have more than one component (max 78); 180 carry 699 holes totalling 1072133
  filled pixels.
- **Loss is decomposed into three levels and must not be collapsed.**
  `control_iou` 0.986368 (pycocotools-versus-OpenCV rasterisation alone),
  `merged_iou` 0.985525 (adds component joining), `mask_iou` 0.973066 (adds
  serialisation and the int32 snap). Component joining costs 0.000843 mean IoU;
  serialisation and quantisation cost 0.012458. **Never report the
  control-level gap as YOLO format loss** - it exists before the format is
  involved.
- **Mask size, not topology, dominates the fidelity distribution.** Area
  quartiles run 0.9358 / 0.9768 / 0.9874 / 0.9924 and all 20 worst instances are
  4-59 px masks. The with-versus-without-holes comparison is **confounded by
  size** (median 148640 px against 24690 px) and must never be quoted as
  evidence that filling holes is free.
- **Phase 8A selected nothing; phase 8B did.** 8A's numbers are evidence, not a
  decision: a high IoU distribution is not an approval and a low one is not a
  rejection. The choice was made by human review in 8B and is recorded in
  `reports/segmentation_adapter_approval.json`, never by re-reading 8A's
  distribution. The five phase 8A artifacts are **historical and immutable** -
  their digests are recorded in the 8B approval and re-verified by test, so do
  not regenerate them, do not edit the fidelity report to say the adapter is now
  selected, and route any factual clarification through a new artifact.
- **The segmentation architecture is SELECTED (phase 8B): YOLO11n-seg from
  `yolo11n-seg.pt`** (SHA-256 `55ed65c56c91713d23e8402371c6c49a6fd84f257f7dce452e8d70e41dcbe152`,
  6182636 bytes), `segmentation_architecture_selection: FINAL_SELECTED_FOR_S0`,
  basis `PHASE_8A_QUANTITATIVE_FIDELITY_AUDIT_PLUS_HUMAN_REVIEW`. **Mask R-CNN is
  `NOT_SELECTED_FALLBACK`** and was never installed, trained or benchmarked here
  - never write that YOLO11n-seg beat it, or that it is better in general. The
  three recorded reasons are audited-format viability, YOLO11 family alignment
  with the frozen detector, and imgsz 768 matching its input resolution.
- **The audited adapter is APPROVED, not promoted.**
  `APPROVED_FOR_CONTROLLED_TRAINING`, role
  `MODEL_SPECIFIC_DERIVED_REPRESENTATION`, canonical ground truth still
  `COCO_INSTANCE_SEGMENTATION`. The conversion is
  `ACCEPTED_WITH_QUANTIFIED_APPROXIMATION` and **never lossless** - 0 of 1726
  instances round-trip exactly. Training uses the **exact phase 8A bytes**, whose
  digests are `63a8145d...` (train), `dae69290...` (validation), `ff21c782...`
  (development) and `616701e1...` (membership); a differing digest is
  `ADAPTER_FINGERPRINT_MISMATCH` and **stops the phase - never rebuild silently,
  and never write a new conversion algorithm**.
- **All 1726 development instances stay in S0**, including the 47 below round-trip
  IoU 0.90, the tiny masks, the multi-component masks, the holed masks and the 2
  synthetic rectangles. Filtering after observing adapter fidelity would change
  the canonical modelling population to suit a model format. Never propose it.
- **S0's protocol is frozen in `configs/segmentation_baseline.yaml`**: YOLO11n-seg,
  imgsz 768, batch 8, epochs 100, patience 50, seed 42, deterministic, AMP,
  `optimizer: auto`, `ULTRALYTICS_DEFAULT_SEGMENTATION_TRAINING_POLICY`. Editing
  it after S0 runs invalidates the result; a variant is a new file. **Batch 8 is a
  `PREDECLARED_EXECUTION_DECISION`** - never try 16 "to see", and treat a genuine
  OOM as `MEMORY_CONSTRAINT_REVIEW_REQUIRED` rather than lowering anything.
- **S0's primary metric is `mask_mAP@0.50:0.95`, and mask and box never merge.**
  Box metrics from the segmenter are reported in their own section; a composite
  box-plus-mask score is refused by the parser, not merely discouraged.
  `supported_macro_mask_map50_95` is reported by S0 but is **not** a
  winner-selection metric until a segmentation comparison protocol is frozen -
  and `vest_loose` stays `DESCRIPTIVE_HIGH_UNCERTAINTY` under the unchanged phase
  7A support rule, reported in full and deciding nothing.
- **`best.pt` for a segmentation model is NOT selected on the mask metric alone,
  and that was REVIEWED AND ACCEPTED before S0.** Read from the installed source
  before training: `SegmentMetrics.fitness = self.seg.fitness() +
  DetMetrics.fitness`, the unweighted **sum of box and mask mAP@0.50:0.95**
  (weights 1.0 and 1.0). Recorded as `checkpoint_selection_policy:
  ULTRALYTICS_NATIVE_SEGMENTATION_FITNESS`, `checkpoint_selection_review_status:
  HUMAN_REVIEWED_AND_ACCEPTED_BEFORE_S0`, `checkpoint_selection_semantics:
  BOX_MAP50_95_PLUS_MASK_MAP50_95`.
- **`selection_metric_equals_primary_reporting_metric` is `false`, deliberately.**
  The checkpoint is chosen on the composite; the project reports
  `primary_scientific_reporting_metric: MASK_MAP50_95`. So the epoch S0 reports
  need not be the epoch that maximised the reported metric. Never close that gap
  after the fact: **no custom mask-only checkpoint selector is authorised**
  (`custom_mask_only_selector_authorized: false`), and **S0 may never be
  retrospectively re-read as a mask-only-selected epoch**
  (`retrospective_reinterpretation_allowed: false`).
- **The framework fitness is a checkpoint-selection mechanism, not a metric.**
  Never publish it as a headline number, never introduce a combined box+mask
  project metric, and **never claim the composite is scientifically superior to
  mask-only selection** - nothing here compares the two. It is an accepted
  baseline protocol choice.
- **PROTOCOL INVARIANT: every future segmentation experiment compared directly
  with S0 must use `ULTRALYTICS_NATIVE_SEGMENTATION_FITNESS`** for checkpoint
  selection, unless a new comparison protocol is human-reviewed and frozen
  **before** any affected experiment runs. A new rule applies to experiments
  frozen under it; it never rewrites S0's semantics.
- **A direct instance-mask IoU diagnostic is OWED and its protocol is NOT
  written.** Ultralytics' mask AP does not satisfy the assignment's IoU
  requirement. The matching rule, inference settings, unmatched-prediction
  handling and averaging scheme must be **predeclared**; writing them after
  looking at S0's predictions is forbidden.
- **The phase 8B smoke test is `NON_EXPERIMENTAL` / `DO_NOT_REPORT_AS_MODEL_RESULT`
  and none of its numbers was recorded** - never quote it as a segmentation
  result, and never tune against it. What it establishes is engineering only:
  the model loads, labels parse, CUDA forward/backward runs, the validation
  loader works, the mask loss executes, a checkpoint is written, ~51 s at peak
  3.20 GiB reserved, `optimizer: auto` resolving to AdamW at lr0 0.001111.
- **S0 was run ONCE and must not be re-run, retuned or averaged.**
  `S0_experiment_sha256`
  `1761007ab1fd3937a56a870618c11115307988f2031598980e8dfea0e218f7bc`. YOLO11n-seg,
  imgsz 768, batch 8, 100/100 epochs, `ALL_EPOCHS_COMPLETED` (no early stop),
  training time 1106.85 s, best epoch **59** at native composite fitness
  **0.884180**, verified as the argmax of the composite recomputed from
  `results.csv`. `optimizer: auto` resolved to **AdamW at lr0 0.001111,
  momentum 0.9**, captured directly from the framework log. `best.pt` SHA-256
  `d7b512b95fafdc8658fd802459d2c87d9ad3f11f922162a774dacf8ca75b87a3`, 6041685 B;
  `last.pt` `9b8a956f...`, same size. Neither is committed.
- **S0 result (validation only): primary mask mAP@0.50:0.95 `0.407942`.** Mask
  mAP@0.50 0.579830, mask precision 0.850509, mask recall 0.528435.
  `supported_macro_mask_map50_95` **0.509482** over `helmet_loose`,
  `helmet_on_head`, `person`, `vest_on_body` - **descriptive, not a winner**.
  Box from the same model, reported separately and never merged: mAP@0.50:0.95
  0.478156, mAP@0.50 0.647256, precision 0.886160, recall 0.552147. **Every one
  of these is a validation number and says nothing about test performance.**
- **Per-class mask AP@0.50:0.95: `helmet_loose` 0.789927, `helmet_on_head`
  0.586523, `vest_on_body` 0.390296, `person` 0.271182, `vest_loose` 0.001782.**
  `person`'s mask AP is roughly half its box AP (0.522227) - the widest
  box-to-mask gap of any class. **Why is UNKNOWN**; occlusion and irregular shape
  are a hypothesis, and the image-level analysis that would test it is a later
  phase. Do not explain it and do not fix it.
- **The direct instance-mask IoU diagnostic ran once, and its numbers are the
  only ones that may be quoted.** `matched_mask_iou_mean` **0.717462**,
  `gt_normalized_mask_iou` **0.556977**, `gt_match_coverage` 0.776316,
  `gt_iou50_coverage` 0.588816, `gt_iou75_coverage` 0.460526, over 304 canonical
  instances and 304 predictions with 236 overlapping assignments and 68 unmatched
  on each side. Protocol fingerprint
  `b912039ca77b36959f74fcdbaed109dbf5e3bd707709556f95a19085c7f34d80`, frozen in
  `configs/segmentation_mask_iou_evaluation.yaml` **before the first optimisation
  step**.
- **The two direct-IoU headlines are NOT interchangeable, and neither is an AP.**
  `matched_mask_iou_mean` describes mask quality *where the model produced an
  overlapping same-class instance*; `gt_normalized_mask_iou` divides the same IoU
  sum by *every* canonical instance, so the 68 misses lower it. Never quote the
  first as the project's IoU, never call either a COCO AP, and never use either
  to change S0.
- **Direct IoU scores against CANONICAL COCO masks, never the YOLO adapter**, at
  a predeclared operating point (conf 0.25, NMS IoU 0.70, imgsz 768, max_det 300,
  `retina_masks: true`, no TTA). Matching is one-to-one per image and per class
  via `scipy.optimize.linear_sum_assignment` maximising total IoU; an assigned
  pair sharing no pixel is not a match; an unmatched GT contributes zero. **No
  threshold was swept and none may be.**
- **`scipy` was added in phase 8C as a runtime dependency**, after a recorded
  methodological review, because `linear_sum_assignment` is the matching rule the
  diagnostic freezes. Greedy matching and a hand-rolled Hungarian solver were both
  considered and rejected. Verified to cause no drift in the frozen ML stack.
- **The training run reads a git-ignored runtime VIEW of the approved adapter**
  (`data/processed/adapters/yolo_segmentation_s0_runtime/`), hard-linked with
  label bytes verified identical to the approved digests before and after
  training. It exists so the framework's `.cache` files never land inside phase
  8A's evidence. The audited directory must stay free of caches and checkpoints.
- **One artifact-write re-execution is recorded, and it is not a second
  experiment.** The first attempt trained, validated and ran the diagnostic, then
  refused to write: the sensitive-content scan caught machine-specific absolute
  paths copied out of the framework's `args.yaml`. No artifact was published.
  Training was **not** repeated; validation and the diagnostic were re-executed
  deterministically on the same frozen checkpoint and reproduced identical
  numbers. Never present this as a repeat measurement.
- **Peak GPU memory for the S0 training run is `NOT_PERSISTED_FOR_THIS_RUN`.**
  The measuring process exited before the artifact was written and the figure is
  not recoverable from any file. Do not reconstruct it from terminal scrollback -
  a number that cannot be traced to an artifact is not evidence. What is
  established is that batch 8 completed without an out-of-memory event.
- **NO final segmenter is selected and NO S1 is authorised.**
  `final_segmenter: UNSELECTED_PENDING_REVIEW`. Do not freeze S0 as the final
  segmenter, do not train an alternative model, resolution or batch, do not tune
  augmentation or thresholds, and do not add a metric. Any further segmentation
  experiment needs a comparison protocol frozen first, exactly as phase 7A did
  for detection.
- **Phase 8D trained nothing.** It re-ran inference on validation under the
  phase 8C settings and reported one row per canonical instance. Its per-instance
  table reproduces the committed 8C aggregates exactly, which is how it is known
  to describe the same run; **the 8C figures were not regenerated and are not
  revised**.
- **Outcome census over all 304 validation instances:** `DETECTION_MISS` 68,
  `LOW_OVERLAP_MASK` 57, `MODERATE_MASK` 39, `HIGH_QUALITY_MASK` 140. The four
  bands partition, so they sum to 304.
- **Coverage and mask quality fail in OPPOSITE directions across size, and
  conflating them is the easy mistake.** Detection coverage rises monotonically
  with canonical area (0.513 / 0.816 / 0.882 / 0.895 by quartile) - misses
  concentrate in small masks. Matched mask IoU is **worst in the largest
  quartile** (0.642 against 0.751 and 0.763 in the middle). Large objects are
  reliably found and poorly delineated; small ones are missed outright.
- **`person` is weaker than every other class at EVERY size quartile** - 0.592 /
  0.539 / 0.636 / 0.561 against non-person 0.789 / 0.844 / 0.829 / 0.822. The
  deficit survives size stratification, so never explain it as a size artefact.
  Its box-minus-mask AP gap is **0.251045**, an order above every other class
  (next is `helmet_on_head` at 0.039179).
- **A large share of `person`'s mask deficit is a TARGET-VERSUS-EVALUATION
  MISMATCH created by the frozen protocol.** `overlap_mask: true`, and
  `polygons2masks_overlap` sorts by descending area with a running maximum, so
  **the smaller instance owns shared pixels** - a vest owns the pixels of the
  person wearing it, and that person's training target is the remainder.
  Measured: 27.3% of canonical person pixels are contested, but **71.0% of the
  pixels S0 misses on person lie in that contested region**; contested fraction
  versus person mask IoU has Spearman **-0.5466**; under-20%-contested persons
  average 0.691 matched IoU against 0.408 for over-40%. Re-scoring the same
  predictions against the overlap-resolved target raises person GT-normalised IoU
  **0.434635 -> 0.504610** and matched IoU **0.578106 -> 0.671180**, while every
  other class moves by less than 0.002.
- **That analysis is `POST_HOC_HYPOTHESIS_GENERATING` and is NOT a finding.**
  It was motivated by an observation made while reviewing images, and re-scoring
  changes the measuring stick rather than the model. It does **not** show that
  training without overlap resolution would produce better masks - only a
  controlled experiment could. And it does not close the gap: even against its
  own target, `person` matched IoU 0.671180 remains far below `helmet_loose`
  0.935 and `helmet_on_head` 0.877.
- **Adapter conversion is NOT the main explanation for S0's mask error.** The 20
  worst S0 instances average adapter IoU 0.968401 against 0.979140 for the split;
  the adapter-versus-model rank correlation is 0.246; `person`'s mean adapter
  fidelity is 0.975368; only 4 validation instances fall in the adapter-risk
  band. **This is not a claim that the adapter has zero effect** - phase 8A
  quantified real loss and some of S0's error is certainly attributable to it.
- **`mask_ratio: 2` is `WEAKLY_MOTIVATED` and YOLO11s-seg is
  `NOT_SPECIFICALLY_MOTIVATED`.** Under-segmentation outnumbers boundary error 56
  to 19 overall and 49 to 14 within `person`, so finer supervision does not
  address the measured failure; and nothing here isolates capacity. **Absence of
  evidence for capacity is not evidence that capacity cannot help** - it is a
  reason to sequence it later.
- **The preferred S1 hypothesis is `overlap_mask: false`, and it is NOT a clean
  comparison.** The flag changes the training target **and** the validation
  target - `SegmentationValidator._prepare_batch` builds its ground truth with
  `masks == index` when the flag is set - so S0 and such an S1 would have
  incomparable framework mask mAP. The direct mask-IoU diagnostic **would** stay
  comparable, because it always scores against canonical COCO. Phase 8E must
  decide in advance which metric adjudicates; do not start the experiment first.
- **Only 6 of the 43 selected instances were visually inspected**, and the
  artifacts record `instances_selected` and `instances_visually_inspected`
  separately. Never quote the selected count as the number reviewed. The
  population findings rest on all 304 instances, not on the reviewed subset.
- **Two of the six inspected instances are indoor office scenes with fragmentary
  `person` annotations** (one covering only hair and hands beside a fully visible
  unannotated face). Recorded as `UNATTRIBUTED` because the frozen taxonomy has
  no flag for an annotation-scope problem. Two instances establish no rate.
- **Causal labels are refused by the recorder, not merely discouraged.**
  `MODEL_CAPACITY_LIMIT`, `INSUFFICIENT_TRAINING`, `NEEDS_MORE_DATA` and
  `ARCHITECTURE_LIMIT` raise if written into a manual attribution, because this
  phase ran no controlled experiment and could not establish any of them.
- **The ML stack is pinned for a hardware reason.** torch 2.11.0+cu128 from the
  CUDA 12.8 index, because the GPU is Blackwell (`sm_120`) and older builds see
  the device but have no kernels for it. If CUDA ever reports unavailable, that
  is `BLOCKED_FOR_GPU` - **never fall back to CPU and call it the same
  experiment**.
- **D0's protocol lives in `configs/detection_baseline.yaml` and it has been
  run once.** YOLO11n, imgsz 640, batch 16, seed 42, primary metric
  `mAP@0.50:0.95`, checkpoint rule predeclared. It is now the phase 7 reference
  and is also the protocol D1 and D2 inherit, so **editing it invalidates the
  frozen comparison** - a variant is a new entry in
  `configs/detection_experiments.yaml`, never an edit here. Do not tune any
  hyperparameter against validation, and do not add a metric after seeing
  results; the parser rejects both.
- **`vest_loose` has 1 validation image and 8 instances.** Its validation AP is
  not a usable selection signal. Never tune against it, never prefer a model
  because it improved, and always report it with an explicit small-sample caveat.
- **Phase 7's deciding metric is `supported_macro_map50_95`, not the all-class
  mAP.** The unweighted mean of per-class AP@0.50:0.95 over the classes a frozen
  support rule admits: `COMPARISON_SUPPORTED` needs **both** >= 5 positive
  validation images **and** >= 20 validation instances. Applied to the phase 5C.2
  counts it admits `helmet_loose`, `helmet_on_head`, `person` and
  `vest_on_body`, and classifies `vest_loose` (1 image, 8 instances)
  `DESCRIPTIVE_HIGH_UNCERTAINTY`. **The rule names no class** - never hardcode
  the exclusion, never relax a threshold, never add a metric after a result.
- **D0's phase 7 reference: `supported_macro_map50_95` 0.570142** (exactly
  0.57014175), alongside the unchanged official all-class **mAP@0.50:0.95
  0.464429**. Both are validation numbers. The 0.105713 gap is the arithmetic
  effect of dropping one very low AP from an average - **not an improvement**, and
  the two figures are never compared against each other. The five-class metric
  stays `OFFICIAL_ALL_CLASS_REPORTING_METRIC` and is never hidden; phase 6B's
  protocol was not rewritten.
- **`vest_loose` is still a required class and still reported in full** -
  precision, recall, AP@0.50, AP@0.50:0.95 - for every experiment. What it may
  not do is decide a winner: never tune for it, never prefer or reject a model
  because it moved, never rank it against the supported classes, never consult
  the holdout to resolve its uncertainty.
- **D1 and D2 both exist.** D1 varied `MODEL_CAPACITY` (YOLO11s from
  `yolo11s.pt`, SHA-256 `85a76fe8...`, 19313732 B); D2 varied
  `INPUT_RESOLUTION` (imgsz 768) from **the same `yolo11n.pt` bytes D0 used**,
  verified by digest. Both inherit D0's protocol
  from `configs/detection_baseline.yaml` and declare only an override set, so the
  one-variable discipline is enforced by the parser. Batch stays 16 for all
  three: a genuine CUDA OOM **stops** the experiment as
  `MEMORY_CONSTRAINT_REVIEW_REQUIRED` and is never rescued by a smaller batch,
  auto-batch, gradient accumulation, a different imgsz or a different model.
- **D1 result (validation only): `supported_macro_map50_95` 0.560017, delta
  -0.010125 vs D0, classified `BELOW_D0`.** All-class `mAP@0.50:0.95` 0.471114
  (+0.006685), mAP@0.50 0.618243, precision 0.85964, recall 0.527798. 100/100
  epochs, best epoch 73, `D1_experiment_sha256`
  `0589d4c0dabedfe0e6c72da3d248883d750347ce60b122b71aaa44e939158503`. **D1 was
  run once and must not be re-run, retuned or averaged.**
- **The two D1 metrics moved in OPPOSITE directions, and the reason is
  arithmetic, not a paradox.** Both are unweighted means over the same per-class
  APs; they differ only in which classes they average. `vest_loose` (excluded,
  one validation image) gained +0.073925 and so contributes +0.014785 to the
  five-class mean; remove that and the all-class delta is **-0.008100**, agreeing
  with the selection metric. **Never quote D1's all-class improvement as evidence
  it beat D0** - that is the metric-shopping the phase 7A policy forbids - and
  never suppress the all-class figure either.
- **The substantive D1 finding is `vest_on_body`, not the aggregate.** Three of
  four supported classes improved (helmet_on_head +0.032401, helmet_loose
  +0.005802, person +0.004585); `vest_on_body` fell -0.083287 with recall
  -0.111868, more than the other gains combined. **Why is UNKNOWN** - one run
  cannot separate it from run-to-run variance, and diagnosing it needs the
  image-level error analysis that is a later phase. Do not explain it; do not
  fix it.
- **`optimizer: auto` resolved to AdamW at lr0 0.001111 for BOTH D0 and D1**, so
  the optimizer does not confound the comparison. D1's value was read
  **directly** from the framework log (`FRAMEWORK_LOG_LINE_DIRECT_CAPTURE`); D0's
  was inferred, because its run predates the log capture. When adding an
  experiment, prefer direct capture and label an inference as inferred.
- **Selection is decided in advance: margin 0.005, four cases.** A - nothing
  clears D0 by more than the margin, retain D0. B - one candidate clears D0 and
  separates from the next-best *candidate*, it leads. C - a candidate clears D0 but does not
  separate, no winner, efficiency comparison required. D - execution or protocol
  failure, protocol review, **never** read as model inferiority. The margin is an
  engineering threshold, **not a significance test**; run-to-run variance on this
  setup is UNKNOWN because nothing is repeated.
- **D2 result (validation only): `supported_macro_map50_95` 0.594018, delta
  +0.023876 vs D0, classified `IMPROVES_D0_BEYOND_MARGIN`.** All-class
  `mAP@0.50:0.95` 0.490386 (+0.025957), mAP@0.50 0.646304, precision 0.926331,
  recall 0.516549. 100/100 epochs, best epoch 90, `D2_experiment_sha256`
  `8417f64f3c01c8994291f1fb58837059a9db6a814fd5dbbb955a44dcc1979685`. Against D1:
  +0.034001 on the selection metric. **D2 was run once and must not be re-run.**
  Unlike D1, both metrics move the same way, so nothing turns on which is read.
- **D2's small-object hypothesis is NOT supported by the shape of the result, and
  saying otherwise would be the error.** Ranking the four supported classes by
  their frozen small-object fraction against their AP change gives a rank
  correlation of **-0.20**: the largest gain went to `vest_on_body` (the *least*
  small-object-heavy) and the only decline was `person`. Resolution improved the
  selection metric beyond the margin - that is the controlled claim - but the
  proposed mechanism does not explain it. Never present the beyond-margin gain as
  confirming the hypothesis, and never run a size-stratified study to rescue it
  without a new reviewed protocol.
- **Deltas between D1 and D2 are differences, not a ranking.** They differ from
  each other in *two* things at once (each varies a different field relative to
  D0), so nothing between them is attributable to either variable. Only each
  candidate's comparison with D0 is controlled.
- **The detector is FROZEN (phase 7D): D2, YOLO11n at imgsz 768.**
  `selection_status: FINAL_SELECTED`, `selection_method:
  PREDECLARED_POLICY_PLUS_HUMAN_REVIEW`, `policy_case:
  CASE_B_VALIDATION_PERFORMANCE_LEADER`. Recorded in
  `reports/final_detector_manifest.json` with `final_detector_sha256`
  `84d30d64a1e7ebfc4e4763643ef3a3b5c8a5ce3cedc74bbefdb8031d48235a6e`;
  `reports/detection_experiment_results.json` now carries the same final state.
  **The selection is validation-only and says nothing about test performance.**
  D0 is no longer the default - it is the reference experiment, not the
  project's detector.
- **The winner was derived, not asserted, and it must stay that way.** The case
  and the leader are the output of
  `construction_safety_vision.detection_comparison.compare` run over records
  rebuilt from the committed manifests. `scripts/freeze_final_detector.py` names
  no experiment id as the answer and refuses any case other than the single-
  leader one; a test asserts the id appears nowhere as a constant. Never
  hardcode the winner into the comparison engine or the freeze script.
- **"Runner-up" in phase 7 means the next-best CANDIDATE, never the
  second-highest experiment overall.** The frozen rule ranks D1 and D2 against
  each other and each against D0; **D0 is never a peer in that ranking**, so the
  candidate ordering (D2 > D1) and the overall ordering (**D2 > D0 > D1**) are
  different true statements about different sets. D0 outscores D1. Every
  artifact publishes `reference_experiment`, `validation_performance_leader`,
  `candidate_runner_up` and `overall_validation_ranking` rather than one
  overloaded `runner_up` field - do not reintroduce that field, and never write
  "D1 was second" without the qualifier.
- **Reach for the frozen checkpoint only through
  `construction_safety_vision.detection_freeze`.** It resolves by digest, not by
  path: SHA-256
  `0466f872a9de22898c70d8834cf1fcfb3d77f6c9fb27e81ee6248d3d19cbc206`, 5502289
  bytes, with a byte-identical git-ignored copy at
  `artifacts/frozen/detection/D2_best.pt` kept outside the run directory a
  re-run would overwrite. **Never use `last.pt`, never use D0's or D1's
  weights**, and never retrain to replace a missing binary - a re-run produces
  different bytes under the same experiment name, which is the substitution the
  accessor exists to catch. A missing artifact is
  `BLOCKED_MISSING_MODEL_ARTIFACT`, not a reason to train.
- **No efficiency tie-break was run, and none is owed for phase 7.** The policy
  requires one only in Case C; D2 separates from D1 by 0.034001, beyond the
  margin. `FRAMEWORK_VALIDATION_SPEED` values stay descriptive and were not
  consulted. A standardised detector-versus-segmenter latency study is still
  required by the project's scientific question, but it is a later phase.
- **No threshold was tuned and none may be, casually.** The frozen object is
  model architecture + checkpoint + input resolution. If the video
  demonstration later needs an operating confidence, that is a separate,
  predeclared, validation-only decision; no holdout data may participate.
- **The phase 7D artifacts are additive; the historical ones did not move.**
  `reports/final_detector_manifest.json`,
  `reports/detection_selection_report.md` and
  `reports/detection_experiment_comparison.csv` are new. The six D0/D1/D2
  manifests and reports and the frozen 7A policy are byte-identical, their
  digests recorded in the freeze manifest and re-verified by test. A later
  factual clarification to any of them is an **addendum**, never an edit.
- **Precision and recall carry an operating-point caveat.** Ultralytics reports
  one precision/recall pair at the F1-maximising point, not at a fixed
  confidence, so a large move in either can partly reflect where that point
  landed. Read them as a hint about the precision/recall balance, never as a
  threshold-independent property. No threshold was ever tuned.
- **The frozen phase 7 policy artifact must NOT be regenerated.** D1's and D2's
  manifests record its digest, so rewriting it to refresh execution status would
  invalidate their provenance. Its per-experiment `status` fields say what was
  true *when it was frozen*; live status lives in
  `reports/detection_experiment_results.json`.
- **`--rebuild-report` re-renders prose from a committed manifest** without
  training, validating or recomputing a metric, and asserts the manifest is
  byte-identical afterwards. The single exception is a `--phase` metadata
  correction, which is fenced: it proves only the `phase` key moved and preserves
  `created_at`. Use it for prose fixes; never to change a number.
- **Run primitives shared by phase 7 live in
  `src/construction_safety_vision/detection_run.py`**, and
  `scripts/train_detection_baseline.py` still carries its own older copies -
  including an optimizer regex that cannot match, because it does not strip the
  ANSI codes Ultralytics wraps its log label in. That defect cannot affect D0's
  published numbers (D0 is frozen and never re-run) and is left deliberately
  rather than modifying a reported experiment's script; classified `LOW`, to
  converge in a later cleanup phase, never inside an experiment phase.
- **Ultralytics diverts a run to `<id>-2` if its output directory already
  exists**, and attaching the log handler creates that directory. The runner
  therefore passes `exist_ok=True` and relies on its own pre-run existence check,
  then verifies after training that the run landed where the result is read from.
  Do not "fix" this by setting `exist_ok=False`.
- **No D3 is authorised.** After a D1 or D2 result appears, do not try YOLO11m,
  imgsz 896/960, optimizer or LR tuning, augmentation changes, oversampling, loss
  weighting or class weighting, and do not add a metric or tie-breaker. Any
  further experiment needs a new, explicitly reviewed protocol frozen first.
- **The phase 6B result-artifact rebuild is `NON_SELECTION_REVALIDATION`**, with
  `evidence_basis: MAINTAINER_DECLARED_PARTIALLY_CORROBORATED_ON_DISK`. It
  re-executed validation of the same `best.pt` under the same configuration while
  repairing provenance, introduced zero selection degrees of freedom and modified
  no metric. **Corroborated:** the git-ignored `artifacts/detection/D0_val/`
  exists and six of the seven committed D0 figures are byte-identical to its
  output, so a separate validation execution demonstrably happened and the
  committed figures come from it. **Not corroborated:** that nothing was varied
  between the two executions - Ultralytics writes no `args.yaml` for a validation
  run. Keep those two apart, and never present the rebuild as a second run, a
  replication, or evidence about run-to-run variance.
- **`artifacts/` is git-ignored and holds weights and runs.** Never commit
  `best.pt`, `last.pt`, optimizer state or caches. A report references a
  checkpoint by SHA-256 and by its provenance record.
- **Metric figures only in `reports/figures/`.** The framework also writes
  `train_batch*.jpg`, `val_batch*.jpg` and `labels.jpg`, which render dataset
  imagery and prediction montages. Those stay in `artifacts/`. Committing them
  would publish dataset images and pre-empt the deliberate error-analysis stage.
- **No image-level error analysis has been done.** Confusion-matrix counts are
  fair game; opening a validation image to explain an individual failure is a
  later, deliberate phase. Do not start it early.
- **Long paths:** 137 of the export's 742 image files exceed the Windows
  `MAX_PATH` limit on this machine. Open them through
  `construction_safety_vision.paths.long_path`, never with a bare path.
- **Credentials:** `ROBOFLOW_API_KEY` is required by the provider-facing scripts
  and is read from the environment only. It must never be written to
  `.env.example`, a provenance record, a log line, or any committed file.
