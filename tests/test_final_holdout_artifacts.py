"""Tests for the committed phase 11A protocol artifacts.

These read what phase 11A actually wrote, re-derive it from the committed
config, and check that it carries a protocol and no result. No holdout data is
opened and no model is loaded.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from construction_safety_vision.final_holdout_evaluation import (
    DETECTOR_CHECKPOINT_SHA256,
    ENVIRONMENT_GATE,
    FROZEN_NOT_EXECUTED,
    HOLDOUT_STATUS,
    PHASE,
    PROTOCOL_FROZEN,
    RESULT_FINGERPRINT_FIELDS,
    SEGMENTER_CHECKPOINT_SHA256,
    TEST_ANNOTATIONS,
    TEST_IMAGES,
    load_protocol,
    phase_11b_steps,
    protocol_fingerprint,
    validate_manifest,
    validate_protocol,
)
from construction_safety_vision.final_holdout_report import render
from construction_safety_vision.paths import ProjectPaths

PATHS = ProjectPaths.from_root()
ROOT = PATHS.root

CONFIG = ROOT / "configs" / "final_holdout_evaluation.yaml"
MANIFEST = PATHS.reports / "final_holdout_evaluation_protocol.json"
REPORT = PATHS.reports / "final_holdout_evaluation_protocol.md"
PROVENANCE = PATHS.reports / "final_holdout_evaluation.provenance.json"

RESULT_ARTIFACTS = (
    "final_test_detector.json",
    "final_test_segmenter.json",
    "final_test_direct_iou.json",
    "final_test_evaluation.md",
    "final_test_evaluation.provenance.json",
)


@pytest.fixture(scope="module")
def manifest() -> dict[str, Any]:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def test_every_protocol_artifact_exists() -> None:
    for path in (CONFIG, MANIFEST, REPORT, PROVENANCE):
        assert path.is_file(), path.name


def test_result_artifacts_are_all_present_or_all_absent() -> None:
    """Phase 11A defines schemas and creates nothing; phase 11B creates them all.

    A partial set would mean an evaluation that half-wrote its results, which
    the frozen failure policy handles by rebuilding from the persisted
    predictions rather than by leaving the repository in that state.
    """
    present = [name for name in RESULT_ARTIFACTS if (PATHS.reports / name).exists()]
    assert present in ([], list(RESULT_ARTIFACTS)), present


def test_the_phase_11a_protocol_artifacts_still_carry_no_result(
    manifest: dict[str, Any],
) -> None:
    """Whatever phase 11B produced, the protocol document remains a protocol."""
    assert manifest["results_present"] is False
    assert manifest["status"] == FROZEN_NOT_EXECUTED
    assert manifest["test"]["status"] == HOLDOUT_STATUS


def test_the_committed_manifest_validates(manifest: dict[str, Any]) -> None:
    assert validate_manifest(manifest) == []


def test_the_committed_protocol_validates() -> None:
    assert validate_protocol(load_protocol(CONFIG).raw) == []


def test_the_manifest_fingerprint_rederives(manifest: dict[str, Any]) -> None:
    assert manifest["protocol_fingerprint"] == protocol_fingerprint(load_protocol(CONFIG).raw)


def test_the_committed_report_rederives(manifest: dict[str, Any]) -> None:
    assert REPORT.read_text(encoding="utf-8") == render(manifest)


def test_status_phase_and_classification(manifest: dict[str, Any]) -> None:
    assert manifest["status"] == FROZEN_NOT_EXECUTED
    assert manifest["phase"] == PHASE == "11A"
    assert manifest["classification"] == PROTOCOL_FROZEN
    assert manifest["executes_in_phase"] == "11B"


def test_model_identities_in_the_manifest(manifest: dict[str, Any]) -> None:
    assert manifest["detector"]["checkpoint_sha256"] == DETECTOR_CHECKPOINT_SHA256
    assert manifest["segmenter"]["checkpoint_sha256"] == SEGMENTER_CHECKPOINT_SHA256
    assert manifest["detector"]["executed_in_this_phase"] is False
    assert manifest["segmenter"]["executed_in_this_phase"] is False
    assert manifest["segmenter"]["overlap_mask"] is False


def test_every_execution_count_is_zero(manifest: dict[str, Any]) -> None:
    for field in (
        "models_executed",
        "models_trained",
        "test_predictions_produced",
        "test_metrics_computed",
        "test_images_read",
        "test_annotations_read",
        "test_identifiers_recorded",
        "latency_measurements_taken",
        "thresholds_tuned",
        "figures_generated",
    ):
        assert manifest[field] == 0, field
    assert manifest["results_present"] is False
    assert manifest["holdout_unlocked"] is False


def test_holdout_is_protected_in_every_artifact(manifest: dict[str, Any]) -> None:
    assert manifest["test"]["status"] == HOLDOUT_STATUS
    assert manifest["test_policy"] == HOLDOUT_STATUS
    provenance = json.loads(PROVENANCE.read_text(encoding="utf-8"))
    assert provenance["details"]["holdout"] == HOLDOUT_STATUS
    assert provenance["details"]["holdout_unlocked"] is False
    assert provenance["details"]["models_executed"] == 0
    assert provenance["details"]["test_identifiers_recorded"] == 0


def test_aggregate_population_recorded_without_membership(manifest: dict[str, Any]) -> None:
    population = manifest["test_population"]
    assert population["aggregate_counts"]["images"] == TEST_IMAGES == 65
    assert population["aggregate_counts"]["annotations"] == TEST_ANNOTATIONS == 305
    assert population["membership_section_read"] is False
    assert population["verified_against_frozen_manifest"] is True
    for field in population["aggregate_source_fields"]:
        assert field.startswith("actual_") and field.endswith("_counts")


def test_no_holdout_identifier_in_any_artifact() -> None:
    """Nothing phase 11A wrote may contain a per-image holdout identifier."""
    import csv

    frozen = ROOT / "reports" / "final_split_assignments.csv"
    with frozen.open(encoding="utf-8", newline="") as handle:
        test_ids = {
            row["source_image_id"] for row in csv.DictReader(handle) if row["split"] == "test"
        }
    assert test_ids, "the fixture that makes this test meaningful is missing"
    for path in (CONFIG, MANIFEST, REPORT, PROVENANCE):
        text = path.read_text(encoding="utf-8")
        leaked = sorted(image_id for image_id in test_ids if image_id in text)
        assert leaked == [], (path.name, leaked)


def test_phase_11b_execution_order_is_recorded(manifest: dict[str, Any]) -> None:
    assert tuple(manifest["phase_11b_execution_order"]) == phase_11b_steps()
    assert manifest["runner"] == "scripts/evaluate_final_holdout.py"


def test_result_fingerprint_fields_are_declared(manifest: dict[str, Any]) -> None:
    assert tuple(manifest["fingerprints"]["result_fields"]) == RESULT_FINGERPRINT_FIELDS


def test_report_states_the_protocol_and_no_result() -> None:
    text = REPORT.read_text(encoding="utf-8")
    assert "FROZEN_NOT_EXECUTED" in text
    assert "PREDECLARED_PROTOCOL" in text
    assert "This phase evaluated nothing." in text
    assert "PROTECTED_NOT_ACCESSED" in text
    assert ENVIRONMENT_GATE in text
    # A protocol document must not read like a results document.
    for forbidden in ("mAP@0.50:0.95 of 0.", "test mAP was", "achieved on the test set"):
        assert forbidden not in text


def test_report_discloses_the_confusion_matrix_provenance() -> None:
    text = REPORT.read_text(encoding="utf-8")
    assert "prior_canonical_protocol_existed: false" in text
    assert "EXISTING_VALIDATION_CONVENTION_READ_FROM_INSTALLED_SOURCE" in text
    assert "ultralytics==8.4.138" in text


def test_report_preserves_the_one_shot_prohibitions() -> None:
    text = REPORT.read_text(encoding="utf-8")
    assert "reads_permitted: 1" in text
    assert "re-running the evaluation to obtain a different number" in text
    assert "reporting the better of two holdout runs" in text
    assert "second_run_may_be_presented_as_the_first: false" in text


def test_report_names_phase_11b_without_starting_it(manifest: dict[str, Any]) -> None:
    text = REPORT.read_text(encoding="utf-8")
    assert "Phase 11B execution contract" in text
    assert manifest["status"] == FROZEN_NOT_EXECUTED
    assert "refuses to execute" in text
