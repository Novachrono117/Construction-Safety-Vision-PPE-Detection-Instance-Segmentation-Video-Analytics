"""Metadata-only protocol tests; no model, dataset or video is opened."""

import copy
import json
import subprocess
from pathlib import Path

import pytest

from construction_safety_vision.final_video_delivery import REPORT, validate, validate_payload
from construction_safety_vision.video_delivery import APPROVED_RUNTIME_COMMIT
from construction_safety_vision.video_models import frozen_settings

ROOT = Path(__file__).resolve().parents[1]


def test_committed_final_video_metadata_is_consistent():
    assert validate(ROOT) == []


def test_screenshot_writer_supports_unicode_windows_paths(tmp_path):
    import cv2
    import numpy as np

    from construction_safety_vision.final_video_delivery import write_jpeg

    path = tmp_path / "construction_\u2014_\u00e1.jpg"
    write_jpeg(path, np.full((20, 40, 3), 150, dtype=np.uint8))
    decoded = cv2.imdecode(np.frombuffer(path.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
    assert decoded.shape == (20, 40, 3)


@pytest.mark.parametrize(
    "path,value,expected",
    [
        (("source", "source_type"), "SYNTHETIC_ENGINEERING_FIXTURE", "source/license"),
        (("source", "license"), "", "source/license"),
        (("source", "url"), "?".join(("https://example.org/source", "page=1")), "Non-public"),
        (("source", "normalization", "pixel_sequence_identical"), False, "Lossless"),
        (("execution", "status"), "ENGINEERING_PREFIX_COMPLETE", "Incomplete"),
        (("execution", "full_source_processed"), False, "Incomplete"),
        (("execution", "processed_frames"), 1, "Frame accounting"),
        (("execution", "output_video", "duration_seconds"), 10, "Output metadata"),
        (("execution", "inference", "settings", "conf"), 0.3, "Frozen runtime"),
        (("execution", "mode"), "detector", "Frozen runtime"),
        (("execution", "requested_device"), "cpu", "Frozen runtime"),
        (("execution", "output", "sha256"), "invalid", "Invalid artifact"),
        (("execution", "inference", "models", "D2", "predict_invocations"), 1, "invocation"),
        (
            ("execution", "inference", "models", "S1", "effective_dtypes"),
            ["torch.float16"],
            "invocation",
        ),
        (("execution", "processing_throughput_fps"), -1, "Demo timing"),
        (("execution_plan", "selected_range", "start_frame_inclusive"), 25, "plan/accounting"),
        (("engineering_retry_count",), 1, "plan/accounting"),
        (("gap_010_status",), "RESOLVED", "Delivery status"),
        (("holdout_accessed",), True, "Phase boundary"),
        (("threshold_changed",), True, "Phase boundary"),
        (("tracking",), True, "Phase boundary"),
        (("media_license",), "AGPL-3.0", "Rights/audio"),
        (("temporal_review", "observations"), [], "Temporal coverage"),
    ],
)
def test_rejects_false_delivery_claims(path, value, expected):
    p = copy.deepcopy(json.loads((ROOT / REPORT).read_text()))
    target = p
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    assert any(expected in problem for problem in validate_payload(p, frozen_settings(ROOT)))


def test_rejects_screenshot_timestamp_and_frame_reselection():
    p = json.loads((ROOT / REPORT).read_text())
    p["screenshots"]["screenshots"][0]["timestamp_seconds"] += 1
    p["screenshots"]["screenshots"][1]["frame_zero_based"] += 1
    problems = validate_payload(p, frozen_settings(ROOT))
    assert "Screenshot timestamp mismatch" in problems
    assert "Screenshot selection mismatch" in problems


def test_only_authorized_gaps_change_and_historical_evidence_is_preserved():
    before = json.loads(
        subprocess.check_output(
            [
                "git",
                "show",
                f"{APPROVED_RUNTIME_COMMIT}:reports/delivery_gap_resolution_status.json",
            ],
            cwd=ROOT,
        )
    )
    after = json.loads((ROOT / "reports/delivery_gap_resolution_status.json").read_text())
    for old, new in zip(before["gaps"], after["gaps"], strict=True):
        if old["gap_id"] not in {"GAP-002", "GAP-010"}:
            assert old == new
    assert (
        subprocess.check_output(
            [
                "git",
                "diff",
                "--name-only",
                APPROVED_RUNTIME_COMMIT,
                "--",
                "reports/video_runtime_foundation.*",
                "reports/qualitative_validation_gallery*",
                "reports/figures/qualitative",
                "configs",
                "src/construction_safety_vision/video_runtime.py",
                "src/construction_safety_vision/video_models.py",
                "src/construction_safety_vision/video_render.py",
                "scripts/run_video_demo.py",
            ],
            cwd=ROOT,
            text=True,
        )
        == ""
    )
