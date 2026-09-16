"""Validate Phase 13A metadata and optional synthetic videos; never execute a model."""

import argparse
from pathlib import Path

from construction_safety_vision.video_delivery import validate_report

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--with-videos", action="store_true")
    args = parser.parse_args()
    problems = validate_report(Path(__file__).resolve().parents[1], with_videos=args.with_videos)
    for problem in problems:
        print(problem)
    print("VIDEO_RUNTIME_VALIDATION_FAILED" if problems else "VIDEO_RUNTIME_VALIDATED")
    raise SystemExit(bool(problems))
