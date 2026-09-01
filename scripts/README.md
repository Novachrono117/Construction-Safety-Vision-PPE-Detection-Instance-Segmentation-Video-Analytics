# scripts/

Command-line entry points. Anything that produces a reported number runs from
here (or from a notebook that only calls into `src/`), never from an
undocumented one-off shell invocation.

| Script | Phase | Purpose |
| --- | --- | --- |
| `check_environment.py` | 2 | Report interpreter, platform, repository root, configuration validity and holdout lock state. |

## Rules

- Every script takes its settings from a file in `configs/` and writes a
  provenance record for what it produced.
- Scripts that touch the `test` split must go through
  `construction_safety_vision.splits.assert_split_allowed` and require the
  documented double opt-in.
- Prefer `uv run python scripts/<name>.py` so the pinned environment is used.

## Planned scripts

Acquisition, audit, split freeze, training, evaluation, error analysis and video
inference scripts are added by their respective roadmap phases. None are stubbed
in advance.
