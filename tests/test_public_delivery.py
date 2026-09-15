"""Public delivery checks do not read model bytes, imagery or holdout membership."""

import ast
import csv
import hashlib
import json
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


def test_delivery_provenance_identifies_the_actual_sources_and_outputs():
    record = json.loads((ROOT / PROVENANCE_PATH).read_text(encoding="utf-8"))
    for entry in record["inputs"] + record["outputs"]:
        path = ROOT / entry["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"]
        assert path.stat().st_size == entry["size_bytes"]
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
