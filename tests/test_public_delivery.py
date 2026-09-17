"""Public delivery checks do not read model bytes, imagery or holdout membership."""

import ast
import csv
import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from construction_safety_vision.delivery_status import (
    PROVENANCE_PATH,
    STATUS_PATH,
    build_status,
    historical_changes,
    validate_delivery,
    validate_public_license,
)

ROOT = Path(__file__).resolve().parents[1]


def test_public_truth_links_metadata_and_scanner_pass():
    assert validate_delivery(ROOT) == []


def test_phase_12b_tracker_is_reproducible_and_preserves_all_original_gaps():
    payload = json.loads((ROOT / STATUS_PATH).read_text(encoding="utf-8"))
    assert payload == build_status(ROOT)
    with (ROOT / "reports/final_delivery_gap_register.csv").open(encoding="utf-8") as handle:
        original = list(csv.DictReader(handle))
    assert [x["gap_id"] for x in payload["gaps"]] == [x["gap_id"] for x in original]
    for gap, prior in zip(payload["gaps"], original, strict=True):
        assert gap["original_severity"] == prior["severity"]
        assert gap["remaining_action"]
    states = {gap["gap_id"]: gap["current_status"] for gap in payload["gaps"]}
    assert states["GAP-008"] == "OPEN"
    assert states["GAP-012"] == "PARTIALLY_RESOLVED"
    assert states["GAP-006"] == "RESOLVED"
    assert historical_changes(ROOT) == []


def test_delivery_modules_have_no_model_or_dataset_accessor_path():
    files = (
        "src/construction_safety_vision/checkpoint_delivery.py",
        "src/construction_safety_vision/delivery_status.py",
        "scripts/verify_checkpoint.py",
        "scripts/validate_public_delivery.py",
    )
    for relative in files:
        text = (ROOT / relative).read_text(encoding="utf-8")
        for node in ast.walk(ast.parse(text)):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                modules = [node.module or ""]
            else:
                continue
            assert not any(
                module.startswith(("torch", "ultralytics", "cv2", "PIL")) for module in modules
            ), relative
        for accessor in ("load_frozen_splits", "FrozenSplits", "authorize_final_holdout_access"):
            assert accessor not in text, relative


def test_colab_resolution_preserves_unrelated_gaps_and_historical_accounting():
    current = build_status(ROOT)
    prior = json.loads(
        subprocess.check_output(
            [
                "git",
                "show",
                "8ce5d0375903e3e3760873e0a75ddce37cc1149b:" + STATUS_PATH,
            ],
            cwd=ROOT,
        )
    )
    before = {gap["gap_id"]: gap["current_status"] for gap in prior["gaps"]}
    after = {gap["gap_id"]: gap["current_status"] for gap in current["gaps"]}
    assert {key for key in before if before[key] != after[key]} == {"GAP-004"}
    assert after["GAP-004"] == "RESOLVED"
    assert after["GAP-008"] == after["GAP-014"] == "OPEN"
    assert after["GAP-010"] == "PARTIALLY_RESOLVED"
    assert current["scientific_boundary"] == prior["scientific_boundary"]
    assert (
        next(entry for entry in current["requirement_updates"] if entry["requirement_id"] == "R18")[
            "current_state"
        ]
        == "COMPLETE"
    )


@pytest.mark.parametrize(
    "category",
    [
        "LOCAL_IMPLEMENTATION_VERIFICATION",
        "REAL_COLAB_MODE_A_VALIDATION",
        "REAL_COLAB_MODE_B_VALIDATION",
    ],
)
def test_colab_resolution_requires_every_separate_validation_pass(monkeypatch, category):
    report_path = ROOT / "reports/academic_colab_delivery.json"
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    payload["validation_evidence"][category]["status"] = "NOT_VERIFIED"
    original_read = Path.read_text

    def substituted_read(path, *args, **kwargs):
        if path == report_path:
            return json.dumps(payload)
        return original_read(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", substituted_read)
    current = build_status(ROOT)
    assert (
        next(gap for gap in current["gaps"] if gap["gap_id"] == "GAP-004")["current_status"]
        == "OPEN"
    )
    assert not any(entry["requirement_id"] == "R18" for entry in current["requirement_updates"])


def test_delivery_provenance_identifies_the_actual_sources_and_outputs():
    # Phase 12B is approved history. Later delivery phases update the live README,
    # tracker and builder; its provenance still identifies the exact 12B bytes.
    approved_phase_12b = "73098b7c62d9cb183cb3114af6209a65f1c532e7"
    record = json.loads((ROOT / PROVENANCE_PATH).read_text(encoding="utf-8"))
    for entry in record["inputs"] + record["outputs"]:
        body = subprocess.check_output(
            ["git", "show", f"{approved_phase_12b}:{entry['path']}"], cwd=ROOT
        )
        assert hashlib.sha256(body).hexdigest() == entry["sha256"]
        assert len(body) == entry["size_bytes"]
    assert record["details"]["models_executed"] == 0
    assert record["details"]["holdout_accessed"] is False


def test_repository_license_is_canonical_and_publicly_consistent():
    assert validate_public_license(ROOT) == []
    assert build_status(ROOT)["repository_license"]["project_repository_license"] == "AGPL-3.0"


@pytest.mark.parametrize("defect", ["canonical_text", "readme_license", "new_mit_claim"])
def test_license_validator_rejects_modified_text_or_conflicting_public_claims(tmp_path, defect):
    for relative in ("LICENSE", "README.md", "reports/README.md"):
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / relative).read_bytes())
    (tmp_path / "delivery").mkdir()
    for source in (ROOT / "delivery").glob("*.md"):
        (tmp_path / "delivery" / source.name).write_bytes(source.read_bytes())
    assert validate_public_license(tmp_path) == []
    if defect == "canonical_text":
        with (tmp_path / "LICENSE").open("a", encoding="utf-8") as handle:
            handle.write("\nUnapproved modification to the canonical body.\n")
    elif defect == "readme_license":
        path = tmp_path / "README.md"
        path.write_text(
            path.read_text(encoding="utf-8").replace("GNU AGPL-3.0", "MIT"), encoding="utf-8"
        )
    else:
        (tmp_path / "delivery/new_public_page.md").write_text(
            "Project code is MIT licensed.\n", encoding="utf-8"
        )
    assert validate_public_license(tmp_path)
