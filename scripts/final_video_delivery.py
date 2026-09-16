"""Extract, assemble or validate Phase 13B evidence; never execute a model."""

import argparse
from pathlib import Path

from construction_safety_vision.final_video_delivery import build, extract, validate

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("extract", "build", "validate"))
    parser.add_argument("--with-video", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.action == "validate":
        problems = validate(root, with_video=args.with_video)
        for problem in problems:
            print(problem)
        print("FINAL_VIDEO_VALIDATION_FAILED" if problems else "FINAL_VIDEO_VALIDATED")
        raise SystemExit(bool(problems))
    (extract if args.action == "extract" else build)(root)
