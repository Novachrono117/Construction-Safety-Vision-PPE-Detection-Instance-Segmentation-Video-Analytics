"""Validate or explicitly generate Phase 14A metadata without executing inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from construction_safety_vision.academic_colab_delivery import COMPLETE, validate, write_reports


def main() -> int:
    """Reconcile human validation evidence; generation needs an explicit quality record."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--quality-record", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.write:
        if args.quality_record is None:
            parser.error("--write requires --quality-record with observed local gate results")
        write_reports(root, json.loads(args.quality_record.read_text(encoding="utf-8")))
    problems = validate(root)
    for problem in problems:
        print(problem)
    print("ACADEMIC_COLAB_VALIDATION_FAILED" if problems else COMPLETE)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
