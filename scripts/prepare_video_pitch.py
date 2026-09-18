"""Validate the video-pitch package and refresh the live delivery tracker.

Executes no model, reads no checkpoint and never opens holdout content. It records
no pitch and publishes nothing: the pitch stays READY_TO_RECORD until a human
recording exists at an accessible link.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from construction_safety_vision.delivery_status import STATUS_PATH, build_status
from construction_safety_vision.pitch_delivery import (
    PITCH_READY,
    load_manifest,
    spoken_words,
    validate,
)


def main() -> int:
    """Validate by default; refresh the delivery tracker only on request."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write", action="store_true", help="refresh the live delivery tracker from the manifest"
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]

    if "CSVISION_ALLOW_TEST_SPLIT" in os.environ:
        print("BLOCKED: holdout environment variable must be absent")
        return 1

    problems = validate(root)
    if problems:
        for problem in problems:
            print(problem)
        print("VIDEO_PITCH_PACKAGE_INVALID")
        return 1

    if args.write:
        status = build_status(root)
        (root / STATUS_PATH).write_text(
            json.dumps(status, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )

    manifest = load_manifest(root)
    words = spoken_words(root)
    rate = manifest["duration"]["assumed_rate_words_per_minute"]
    seconds = words / rate * 60
    print(PITCH_READY)
    print(
        f"{len(manifest['spoken_claims'])} spoken numbers verified against committed artifacts; "
        f"{words} spoken words, about {int(seconds // 60)}:{int(seconds % 60):02d} at {rate} wpm."
    )
    print(
        "Video segment "
        f"{manifest['video_segment']['selected_start_seconds']:.0f}-"
        f"{manifest['video_segment']['selected_end_seconds']:.0f}s, anchored on frozen "
        "screenshot instants, unedited, contains a narrated visible failure."
    )
    print("No model executed, no holdout content read, no metric recomputed, no video published.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
