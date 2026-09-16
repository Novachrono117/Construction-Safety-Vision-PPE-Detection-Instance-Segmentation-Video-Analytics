"""Run frozen D2/S1 on an ordinary local video; no threshold, training or tracking options."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from construction_safety_vision.video_models import MODES, VideoRuntimeError
from construction_safety_vision.video_runtime import run_video


def build_parser() -> argparse.ArgumentParser:
    """Expose delivery controls, never scientific thresholds."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=tuple(MODES), required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-frames", type=int, help="Engineering prefix only, not a full video")
    parser.add_argument("--no-fps-overlay", action="store_true")
    parser.add_argument(
        "--source-metadata", type=Path, help="Source attribution JSON; see delivery/VIDEO.md"
    )
    return parser


def main() -> int:
    """Execute the public runtime and return a clear failure instead of opaque codec errors."""
    args = build_parser().parse_args()
    try:
        metadata = (
            json.loads(args.source_metadata.read_text(encoding="utf-8"))
            if args.source_metadata
            else None
        )
        record = run_video(
            args.input,
            args.output,
            mode=args.mode,
            device=args.device,
            root=Path(__file__).resolve().parents[1],
            overwrite=args.overwrite,
            max_frames=args.max_frames,
            fps_overlay=not args.no_fps_overlay,
            attribution=metadata,
        )
    except (VideoRuntimeError, OSError, ValueError) as exc:
        message = str(exc) if isinstance(exc, VideoRuntimeError) else type(exc).__name__
        print(f"VIDEO_RUNTIME_FAILED: {message}")
        return 1
    print(
        f"{record['status']}: {record['processed_frames']} frames; "
        f"source {record['source']['fps']:.3f} fps; "
        f"demo loop {record['processing_throughput_fps']:.3f} fps"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
