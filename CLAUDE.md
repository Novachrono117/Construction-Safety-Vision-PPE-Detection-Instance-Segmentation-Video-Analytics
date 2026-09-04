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

- **Phase:** 5C.2 complete - **the split is frozen and the holdout is locked**.
  `final_selected_candidate` is `candidate_001`, selected by human review of the
  six predeclared phase 5C.1 candidates. Phase 5D (task-specific dataset
  materialisation) has not started; do not start it unprompted.
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
- **Models:** none trained. No metrics exist.
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
- **Phase 5C.2 froze membership only.** No image was copied, moved, resized or
  preprocessed, no label file was written, and `data/processed/` is untouched.
  Phase 5D materialises the detection and segmentation views.
- **Long paths:** 137 of the export's 742 image files exceed the Windows
  `MAX_PATH` limit on this machine. Open them through
  `construction_safety_vision.paths.long_path`, never with a bare path.
- **Credentials:** `ROBOFLOW_API_KEY` is required by the provider-facing scripts
  and is read from the environment only. It must never be written to
  `.env.example`, a provenance record, a log line, or any committed file.
