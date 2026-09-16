"""Build Phase 13A reports from persisted engineering evidence without model execution."""

from pathlib import Path

from construction_safety_vision.video_delivery import build_report

if __name__ == "__main__":
    build_report(Path(__file__).resolve().parents[1])
    print("VIDEO_RUNTIME_REPORT_WRITTEN")
