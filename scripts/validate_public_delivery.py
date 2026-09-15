"""Generate or validate delivery metadata; read no model, imagery or holdout membership."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from construction_safety_vision.delivery_status import (
    COMPLETE,
    historical_changes,
    validate_delivery,
    write_metadata,
)


def main() -> int:
    """Validate by default; generate new delivery metadata/provenance only on request."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write", action="store_true", help="regenerate delivery metadata and its provenance"
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if "CSVISION_ALLOW_TEST_SPLIT" in os.environ:
        print("BLOCKED: holdout environment variable must be absent")
        return 1
    if historical_changes(root):
        print("SCIENTIFIC_STATE_MISMATCH: historical artifacts changed")
        return 1
    if args.write:
        write_metadata(root)
    problems = validate_delivery(root)
    if problems:
        for problem in problems:
            print(problem)
        print("TRUTH_PASS_INCOMPLETE")
        return 1
    print(COMPLETE)
    print("Metadata and links valid; known stale probes 0; historical artifacts unchanged.")
    print("Existing sensitive-data scanner: no findings in public delivery text.")
    print("No model executed, no checkpoint or holdout content read, no metrics recomputed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
