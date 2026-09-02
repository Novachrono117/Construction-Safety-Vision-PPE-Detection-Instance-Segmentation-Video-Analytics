# scripts/

Command-line entry points. Anything that produces a reported number runs from
here (or from a notebook that only calls into `src/`), never from an
undocumented one-off shell invocation.

| Script | Phase | Purpose |
| --- | --- | --- |
| `check_environment.py` | 2 | Report interpreter, platform, repository root, configuration validity and holdout lock state. |
| `download_dataset.py` | 3 | Acquire the canonical COCO instance-segmentation export, hash it while streaming, extract it safely, and write its provenance record. Idempotent. |
| `inspect_dataset.py` | 3 | Structurally inspect the extracted export and write `reports/dataset_provenance.{md,json}`. Read-only; no EDA. |

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
-> `audit_source_dataset` -> `eda_source_dataset` -> `build_review_package` ->
`write_audit_reports`. The first two need `ROBOFLOW_API_KEY`; the rest are offline.

`record_manual_audit.py` (phase 4B) turns the human visual review of that package
into `reports/manual_audit_decisions.csv` and `reports/manual_audit_report.md`. It
reads no dataset image and makes no judgement of its own: the judgements are
declared as data at the top of the file, and the script resolves them against the
phase 4A manifests, refusing anything that does not resolve unambiguously. Run it
with `--check` to re-validate the committed record without rewriting it.
`write_audit_reports.py` then cross-references the outcome; re-run it after
recording. Both are offline.

## Planned scripts

Split freeze, training, evaluation, error analysis and video inference
scripts are added by their respective roadmap phases. None are stubbed in
advance.
