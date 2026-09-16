import json
from pathlib import Path

import pytest

from construction_safety_vision.video_delivery import REPORT, validate_report

ROOT = Path(__file__).resolve().parents[1]


def test_committed_runtime_evidence_validates_without_model_or_video_reads():
    assert validate_report(ROOT) == []


@pytest.mark.parametrize(
    "defect,expected",
    [
        ("status", "Smoke completion status mismatch"),
        ("count", "Frame accounting mismatch"),
        ("timing", "Demo timing arithmetic mismatch"),
        ("precision", "Invocation/precision mismatch"),
        ("identity", "Frozen checkpoint mismatch"),
        ("holdout", "Forbidden phase behavior"),
        ("gap", "Completion/gap boundary mismatch"),
    ],
)
def test_validator_rejects_false_runtime_evidence(monkeypatch, defect, expected):
    original = Path.read_text
    payload = json.loads((ROOT / REPORT).read_text())
    run = payload["smoke"]["executions"][0]
    if defect == "status":
        run["status"] = "FAILED"
    if defect == "count":
        run["decoded_frames"] += 1
    if defect == "timing":
        run["processing_throughput_fps"] += 1
    if defect == "precision":
        run["inference"]["models"]["D2"]["effective_dtypes"] = ["torch.float16"]
    if defect == "identity":
        run["inference"]["models"]["S1"]["identity"]["sha256"] = "0" * 64
    if defect == "holdout":
        payload["holdout_accessed"] = True
    if defect == "gap":
        payload["gap_002_status"] = "RESOLVED"

    def read(path, *args, **kwargs):
        return json.dumps(payload) if path == ROOT / REPORT else original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read)
    assert any(expected in p for p in validate_report(ROOT))


def test_unrelated_gaps_and_phase12c_artifacts_preserved():
    import subprocess

    from construction_safety_vision.video_delivery import APPROVED_RUNTIME_COMMIT, BASELINE

    before = json.loads(
        subprocess.check_output(
            ["git", "show", f"{BASELINE}:reports/delivery_gap_resolution_status.json"], cwd=ROOT
        )
    )
    current = json.loads(
        subprocess.check_output(
            [
                "git",
                "show",
                f"{APPROVED_RUNTIME_COMMIT}:reports/delivery_gap_resolution_status.json",
            ],
            cwd=ROOT,
        )
    )
    for old, new in zip(before["gaps"], current["gaps"], strict=True):
        if old["gap_id"] != "GAP-010":
            assert old == new
    changed = subprocess.check_output(
        [
            "git",
            "diff",
            "--name-only",
            BASELINE,
            "--",
            "reports/qualitative_validation_gallery*",
            "reports/figures/qualitative",
            "configs/qualitative_validation_gallery.yaml",
        ],
        cwd=ROOT,
        text=True,
    )
    assert changed == ""
