# scripts/

Command-line entry points. Anything that produces a reported number runs from
here (or from a notebook that only calls into `src/`), never from an
undocumented one-off shell invocation.

| Script | Phase | Purpose |
| --- | --- | --- |
| `check_environment.py` | 2 | Report interpreter, platform, repository root, configuration validity and holdout lock state. |
| `download_dataset.py` | 3 | Acquire the canonical COCO instance-segmentation export, hash it while streaming, extract it safely, and write its provenance record. Idempotent. |
| `inspect_dataset.py` | 3 | Structurally inspect the extracted export and write `reports/dataset_provenance.{md,json}`. Read-only; no EDA. |
| `fetch_source_inventory.py` | 4A | Walk the provider's source inventory and recover the 436 independent images with their per-class annotation counts. |
| `download_source_images.py` | 4A | Acquire the 436 source originals at full resolution into the git-ignored external layer. |
| `audit_source_dataset.py` | 4A | Exact duplicates, perceptual near-duplicate candidates, zero-instance images, split-crossing checks. |
| `audit_bbox_consistency.py` | 4A | Compare each supplied COCO `bbox` against the box its own segmentation implies. |
| `eda_source_dataset.py` | 4A | Source-population statistics and the plots under `reports/figures/`. |
| `write_audit_reports.py` | 4A | Turn the audit and EDA measurements into the written reports. Re-run after phase 4B. |
| `build_review_package.py` | 4A | Contact sheets so a person can answer the questions counting cannot. |
| `record_manual_audit.py` | 4B | Validate the declared human judgements against the manifests and record them. |
| `recover_source_geometry.py` | 5A | Recover complete instance-segmentation geometry for the live source state, read-only, verifying every instance against the provider's own area and box. |
| `map_v4_sources.py` | 5A | Map each source image to its one non-augmented version-4 representation, by filename identity plus a pixel comparison against a deterministic re-render. |
| `analyze_annotation_drift.py` | 5A | Per-image comparison of the live annotations against the version-4 snapshot, counting additions and removals separately. |
| `build_drift_figures.py` | 5A | Review sheets L, M and N: the drift, the geometry-less records, and every added annotation. |
| `resolve_canonical_snapshot.py` | 5A | Apply the decision rule and write `reports/canonical_annotation_{decision.md,manifest.json}`. |
| `analyze_fragment_rule.py` | 5B | Test whether a geometry rule computed from the current state alone can identify the annotations added since version 4. Scores candidates against the v4 diff; exits non-zero when none is acceptable. |
| `build_modeling_population.py` | 5B | Decide which images and annotations may be modelled, materialise geometry-less records, and build the indivisible split units. Creates no split. |

## Rules

- Every script takes its settings from a file in `configs/` and writes a
  provenance record for what it produced.
- Scripts that touch the `test` split must go through
  `construction_safety_vision.splits.assert_split_allowed` and require the
  documented double opt-in.
- Prefer `uv run python scripts/<name>.py` so the pinned environment is used.

`download_dataset.py` requires `ROBOFLOW_API_KEY` in the environment. The key is
read from the environment only, sent as an `Authorization` header rather than a
URL parameter, and never written to any file, log or provenance record.

Phase 4A scripts run in this order: `fetch_source_inventory` -> `download_source_images`
-> `audit_source_dataset` -> `audit_bbox_consistency` -> `eda_source_dataset` ->
`build_review_package` -> `write_audit_reports`. The first two need
`ROBOFLOW_API_KEY`; the rest are offline.

`record_manual_audit.py` (phase 4B) turns the human visual review of that package
into `reports/manual_audit_decisions.csv` and `reports/manual_audit_report.md`. It
reads no dataset image and makes no judgement of its own: the judgements are
declared as data at the top of the file, and the script resolves them against the
phase 4A manifests, refusing anything that does not resolve unambiguously. Run it
with `--check` to re-validate the committed record without rewriting it.
`write_audit_reports.py` then cross-references the outcome; re-run it after
recording. Both are offline.

Phase 5A scripts run in this order: `recover_source_geometry` -> `map_v4_sources`
-> `analyze_annotation_drift` -> `build_drift_figures` ->
`resolve_canonical_snapshot`. Only the first needs `ROBOFLOW_API_KEY`, and it only
reads: it issues one documented GET per source image and makes no mutating call.
It writes the bulk geometry to the git-ignored interim layer and a counts-only
summary to `reports/`.

`resolve_canonical_snapshot.py` refuses to write a manifest that is internally
inconsistent or that contains a signed URL, a credential or an absolute local
path. That check is a gate, not a formality: the manifest is the evidence a
reader checks the project's headline numbers against.

Phase 5B scripts run in this order: `analyze_fragment_rule` ->
`build_modeling_population`. Both are offline and read only committed artifacts,
so every decision they encode is traceable to a file rather than to a list typed
into the script. `analyze_fragment_rule.py` uses the version-4 export **only** to
score candidate rules; no rule it evaluates may read the export, the provider
split, or a list of identifiers.

`build_modeling_population.py` excludes logically and never deletes: an excluded
image keeps its file, keeps its place in the source provenance population of 436,
and gains a reason and a decision source. It refuses to write a manifest whose
invariants are broken.

## Planned scripts

Modelling-population construction, split freeze, training, evaluation, error
analysis and video inference scripts are added by their respective roadmap
phases. None are stubbed in advance.
