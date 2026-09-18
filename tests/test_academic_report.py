"""Regression protection for reported evidence, without data/model access."""

import json
import subprocess
from pathlib import Path

import pytest

from construction_safety_vision import academic_report as report
from construction_safety_vision.delivery_status import STATUS_PATH, build_status

ROOT = Path(__file__).resolve().parents[1]


def test_report_matches_committed_headlines_and_figures():
    result = report.validate(ROOT)
    assert result["status"] == "PASS"
    assert result["models_executed"] == result["metrics_recomputed"] == 0
    assert not result["holdout_accessed"]


@pytest.mark.parametrize("key", sorted(report.REQUIRED))
def test_stale_or_missing_report_headline_is_rejected(monkeypatch, key):
    original = Path.read_text

    def stale(path, *args, **kwargs):
        text = original(path, *args, **kwargs)
        if path == ROOT / report.REPORT:
            text = text.replace(f"<!-- claim:{key} -->", f"<!-- claim:{key} -->stale ", 1)
        return text

    monkeypatch.setattr(Path, "read_text", stale)
    with pytest.raises(ValueError, match="Report claim mismatch"):
        report.validate(ROOT)


def test_historical_source_mutation_is_rejected(monkeypatch):
    original = Path.read_bytes

    def changed(path):
        value = original(path)
        return value + b" " if path == ROOT / "reports/final_test_detector.json" else value

    monkeypatch.setattr(Path, "read_bytes", changed)
    with pytest.raises(ValueError, match="Historical aggregate modified"):
        report.validate(ROOT)


def test_report_closes_only_its_gap_and_preserves_exact_colab_partial():
    before = json.loads(
        subprocess.check_output(["git", "show", f"{report.BASELINE}:{STATUS_PATH}"], cwd=ROOT)
    )
    after = build_status(ROOT)
    # GAP-007 is the video pitch, moved to READY_TO_RECORD by a later delivery phase.
    # This test still pins what the report phase itself was allowed to change.
    later_phase_gaps = {"GAP-007"}
    for old, new in zip(before["gaps"], after["gaps"], strict=True):
        if old["gap_id"] == "GAP-003":
            assert old["current_status"] == "OPEN"
            assert new["current_status"] == "RESOLVED"
        elif old["gap_id"] not in later_phase_gaps:
            assert old == new
    audit = after["assignment_exact_wording_audit"]
    assert audit["colab_exact_wording"]["classification"] == "COLAB_EXACT_WORDING_PARTIAL"
    assert audit["colab_exact_wording"]["status"] == "PARTIAL"
    assert audit["pitch"]["status"] == "OPEN"
    assert before["scientific_boundary"] == after["scientific_boundary"]
    assert before["phase_14a"] == after["phase_14a"]
