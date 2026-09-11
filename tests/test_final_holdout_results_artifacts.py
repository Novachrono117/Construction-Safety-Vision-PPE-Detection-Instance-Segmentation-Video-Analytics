"""Tests for the committed phase 11B holdout result artifacts.

These read what phase 11B actually wrote and check that it is internally
consistent, that every fingerprint re-derives, and that nothing it published
identifies a holdout image. **No model is loaded and no holdout image is
opened**: the only holdout content read here is the frozen id list, and it is
read solely to prove that none of those ids appears in a committed file.

The suite skips itself entirely before phase 11B has run.
"""

from __future__ import annotations

import csv
import json
from typing import Any

import pytest

from construction_safety_vision.final_holdout_evaluation import (
    CLASSES,
    CONFUSION_MATRIX_CONF,
    CONFUSION_MATRIX_IOU,
    DETECTOR_CHECKPOINT_SHA256,
    DIRECT_IOU_PROTOCOL_FINGERPRINT,
    OPERATIONAL_CONF,
    RARE_CLASS,
    SEGMENTER_CHECKPOINT_SHA256,
    TEST_ANNOTATIONS,
    TEST_IMAGES,
)
from construction_safety_vision.final_holdout_execution import (
    result_fingerprint,
    validate_result,
)
from construction_safety_vision.paths import ProjectPaths

PATHS = ProjectPaths.from_root()
ROOT = PATHS.root

PROTOCOL_FINGERPRINT = "a5a328b3a8e49b06fb8fd9e792abcf43ccdd9aac5422729814dac0dbadc1daef"

DETECTOR = PATHS.reports / "final_test_detector.json"
SEGMENTER = PATHS.reports / "final_test_segmenter.json"
DIRECT_IOU = PATHS.reports / "final_test_direct_iou.json"
REPORT = PATHS.reports / "final_test_evaluation.md"
PROVENANCE = PATHS.reports / "final_test_evaluation.provenance.json"
PER_CLASS = PATHS.reports / "final_test_per_class.csv"

pytestmark = pytest.mark.skipif(
    not PROVENANCE.is_file(), reason="phase 11B has not been executed in this checkout"
)


def _read(path: Any) -> dict[str, Any]:
    """Parse one committed JSON artifact.

    Args:
        path: The artifact.

    Returns:
        Its content.
    """
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def detector() -> dict[str, Any]:
    return _read(DETECTOR)


@pytest.fixture(scope="module")
def segmenter() -> dict[str, Any]:
    return _read(SEGMENTER)


@pytest.fixture(scope="module")
def direct_iou() -> dict[str, Any]:
    return _read(DIRECT_IOU)


@pytest.fixture(scope="module")
def provenance() -> dict[str, Any]:
    return _read(PROVENANCE)


@pytest.fixture(scope="module")
def holdout_ids() -> set[str]:
    with (PATHS.reports / "final_split_assignments.csv").open(encoding="utf-8", newline="") as fh:
        return {row["source_image_id"] for row in csv.DictReader(fh) if row["split"] == "test"}


# --- presence and fingerprints ----------------------------------------------------------


def test_every_result_artifact_exists() -> None:
    for path in (DETECTOR, SEGMENTER, DIRECT_IOU, REPORT, PROVENANCE, PER_CLASS):
        assert path.is_file(), path.name


def test_every_result_fingerprint_rederives(
    detector: dict[str, Any], segmenter: dict[str, Any], direct_iou: dict[str, Any]
) -> None:
    """A committed digest must cover the file it sits in."""
    for payload in (detector, segmenter, direct_iou):
        recorded = payload["result_sha256"]
        scope = {key: value for key, value in payload.items() if key != "result_sha256"}
        assert result_fingerprint(scope) == recorded


def test_every_result_names_the_frozen_protocol(
    detector: dict[str, Any], segmenter: dict[str, Any], direct_iou: dict[str, Any]
) -> None:
    for payload in (detector, segmenter, direct_iou):
        assert payload["protocol_fingerprint"] == PROTOCOL_FINGERPRINT
        assert payload["split"] == "test"


def test_every_result_validates(
    detector: dict[str, Any],
    segmenter: dict[str, Any],
    direct_iou: dict[str, Any],
    holdout_ids: set[str],
) -> None:
    for payload in (detector, segmenter, direct_iou):
        assert (
            validate_result(
                payload,
                protocol_fingerprint=PROTOCOL_FINGERPRINT,
                prediction_sha256=payload["prediction_fingerprint"],
                holdout_identifiers=sorted(holdout_ids),
            )
            == []
        )


def test_the_ledger_and_the_results_name_one_prediction_set(
    detector: dict[str, Any], segmenter: dict[str, Any], provenance: dict[str, Any]
) -> None:
    """A report whose prediction fingerprint differs is a second run, not a rebuild."""
    recorded = provenance["details"]["fingerprints"]
    assert detector["prediction_fingerprint"] == recorded["detector_test_prediction_sha256"]
    assert segmenter["prediction_fingerprint"] == recorded["segmenter_test_prediction_sha256"]
    assert detector["result_sha256"] == recorded["detector_test_result_sha256"]
    assert segmenter["result_sha256"] == recorded["segmenter_test_result_sha256"]


# --- no leak ------------------------------------------------------------------------------


def test_no_committed_artifact_identifies_a_holdout_image(holdout_ids: set[str]) -> None:
    assert holdout_ids, "the fixture that makes this test meaningful is missing"
    for path in (DETECTOR, SEGMENTER, DIRECT_IOU, REPORT, PROVENANCE, PER_CLASS):
        text = path.read_text(encoding="utf-8")
        leaked = sorted(identifier for identifier in holdout_ids if identifier in text)
        assert leaked == [], (path.name, len(leaked))


def test_no_holdout_imagery_or_prediction_was_committed(provenance: dict[str, Any]) -> None:
    details = provenance["details"]
    assert details["test_imagery_committed"] is False
    assert details["predictions_committed"] is False
    assert details["test_identifiers_committed"] == 0


# --- the models and the population --------------------------------------------------------


def test_the_frozen_models_were_the_ones_executed(
    detector: dict[str, Any], segmenter: dict[str, Any]
) -> None:
    assert detector["model"]["checkpoint_sha256"] == DETECTOR_CHECKPOINT_SHA256
    assert segmenter["model"]["checkpoint_sha256"] == SEGMENTER_CHECKPOINT_SHA256
    assert segmenter["model"]["overlap_mask"] is False
    for payload in (detector, segmenter):
        assert payload["model"]["imgsz"] == 768
        assert payload["model"]["trained_in_this_phase"] is False
        assert payload["model"]["modified_in_this_phase"] is False


def test_the_whole_frozen_holdout_was_evaluated(
    detector: dict[str, Any], segmenter: dict[str, Any]
) -> None:
    for payload in (detector, segmenter):
        population = payload["population"]
        assert population["images"] == TEST_IMAGES == 65
        assert population["annotations"] == TEST_ANNOTATIONS == 305
        assert population["evaluated"] == "ALL_FROZEN_TEST_IMAGES"
    assert detector["population"]["sampling_permitted"] is False
    assert detector["population"]["exclusions_permitted"] is False


def test_both_models_saw_the_same_images(
    detector: dict[str, Any], segmenter: dict[str, Any], direct_iou: dict[str, Any]
) -> None:
    membership = detector["population"]["membership_sha256"]
    assert segmenter["population"]["membership_sha256"] == membership
    assert direct_iou["population"]["membership_sha256"] == membership


# --- metric integrity -----------------------------------------------------------------------


def test_the_object_level_census_partitions_the_holdout(
    detector: dict[str, Any], segmenter: dict[str, Any]
) -> None:
    for payload in (detector, segmenter):
        objects = payload["object_level"]
        assert sum(objects["taxonomy_census"].values()) == objects["ground_truth"]
        assert objects["ground_truth"] == TEST_ANNOTATIONS
        assert objects["true_positives"] + objects["false_negatives"] == TEST_ANNOTATIONS
        assert objects["confidence_threshold"] == OPERATIONAL_CONF
        assert objects["iou_threshold"] == 0.50


def test_the_detector_records_no_mask_quality_failure(detector: dict[str, Any]) -> None:
    """The category is structurally unavailable to a model that emits no mask."""
    assert detector["object_level"]["taxonomy_census"]["MASK_QUALITY_FAILURE"] == 0


def test_the_confusion_matrices_use_the_frozen_framework_semantics(
    detector: dict[str, Any], segmenter: dict[str, Any]
) -> None:
    for payload in (detector, segmenter):
        matrix = payload["confusion_matrix"]
        assert matrix["conf"] == CONFUSION_MATRIX_CONF
        assert matrix["iou_threshold"] == CONFUSION_MATRIX_IOU
        assert matrix["orientation"] == "ROWS_ARE_PREDICTED_COLUMNS_ARE_GROUND_TRUTH"
        assert matrix["implementation_version"] == "ultralytics==8.4.138"
        assert matrix["labels"] == [*CLASSES, "background"]
        assert len(matrix["matrix"]) == len(CLASSES) + 1
        assert all(len(row) == len(CLASSES) + 1 for row in matrix["matrix"])
        # Every ground-truth instance lands in exactly one column.
        assert sum(sum(row) for row in matrix["matrix"]) >= TEST_ANNOTATIONS


def test_every_frozen_class_is_reported_and_none_is_collapsed(
    detector: dict[str, Any], segmenter: dict[str, Any]
) -> None:
    assert set(detector["canonical_box"]["per_class"]) == set(CLASSES)
    assert set(segmenter["canonical_mask"]["per_class"]) == set(CLASSES)
    assert set(segmenter["canonical_box"]["per_class"]) == set(CLASSES)


def test_the_rare_class_is_reported_in_full_and_decides_nothing(
    detector: dict[str, Any], segmenter: dict[str, Any]
) -> None:
    for payload in (detector, segmenter):
        support = payload["support_on_holdout"][RARE_CLASS]
        assert support["status"] == "DESCRIPTIVE_HIGH_UNCERTAINTY"
        assert support["supported"] is False
    assert RARE_CLASS not in detector["supported_macro"]["admitted_classes"]
    assert RARE_CLASS not in segmenter["supported_macro_mask"]["admitted_classes"]
    # Reported in full nonetheless.
    assert RARE_CLASS in detector["canonical_box"]["per_class"]
    assert RARE_CLASS in segmenter["canonical_mask"]["per_class"]


def test_the_support_rule_was_reused_not_reinvented(detector: dict[str, Any]) -> None:
    macro = detector["supported_macro"]
    assert macro["rule_origin"] == "PREEXISTING_PHASE_7A_SUPPORT_RULE_REUSED_UNCHANGED"
    assert macro["names_no_class"] is True
    assert macro["role_on_test"] == "DESCRIPTIVE_CAVEAT_ONLY_NEVER_A_SELECTION_RULE"


def test_the_segmenter_boxes_are_its_own(segmenter: dict[str, Any]) -> None:
    assert segmenter["canonical_box"]["boxes_derived_from_masks"] is False
    assert segmenter["canonical_box"]["boxes_source"] == "EACH_MODELS_OWN_PREDICTED_BOXES"
    # Box and mask AP come from one prediction set, so they score the same instances.
    assert (
        segmenter["canonical_box"]["detections_scored"]
        == segmenter["canonical_mask"]["detections_scored"]
    )


def test_mask_and_box_never_merge(segmenter: dict[str, Any]) -> None:
    assert segmenter["mask_and_box_never_merged"] is True
    assert "canonical_mask" in segmenter
    assert "canonical_box" in segmenter


def test_the_localisation_comparison_declares_no_winner(segmenter: dict[str, Any]) -> None:
    comparison = segmenter["localisation_comparison_with_detector"]
    assert comparison["label"] == "DESCRIPTIVE_ONLY"
    for flag in ("winner_declared", "composite_score", "model_selection_follows"):
        assert comparison[flag] is False
    assert comparison["significance_test"] is False
    delta = comparison["delta_map50_95"]
    assert delta == pytest.approx(
        comparison["segmenter_map50_95"] - comparison["detector_map50_95"], abs=1e-6
    )


def test_the_direct_iou_diagnostic_is_phase_8cs_unchanged(direct_iou: dict[str, Any]) -> None:
    assert direct_iou["direct_iou_protocol_fingerprint"] == DIRECT_IOU_PROTOCOL_FINGERPRINT
    assert direct_iou["unchanged_from_phase_8c"] is True
    assert direct_iou["new_rule_invented"] is False
    assert direct_iou["inference"]["conf"] == OPERATIONAL_CONF
    assert direct_iou["diagnostic"]["role"] == "SECONDARY_CANONICAL_DIAGNOSTIC"
    assert direct_iou["diagnostic"]["is_primary_segmenter_metric"] is False
    assert direct_iou["diagnostic"]["is_an_average_precision"] is False
    overall = direct_iou["diagnostic"]["global"]
    assert overall["gt_count"] == TEST_ANNOTATIONS
    assert overall["matched_count"] + overall["unmatched_gt"] == TEST_ANNOTATIONS


def test_the_two_direct_iou_headlines_are_not_interchangeable(direct_iou: dict[str, Any]) -> None:
    overall = direct_iou["diagnostic"]["global"]
    # The GT-normalised figure divides by every instance, so the misses lower it.
    assert overall["gt_normalized_mask_iou"] <= overall["matched_mask_iou_mean"]
    caveat = direct_iou["diagnostic"]["headlines_are_not_interchangeable"].lower()
    assert "neither is a coco ap" in caveat


# --- the per-class table and the report ------------------------------------------------------


def test_the_per_class_table_covers_every_class() -> None:
    with PER_CLASS.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["class"] for row in rows] == list(CLASSES)
    for row in rows:
        assert row["support_status"] in {"COMPARISON_SUPPORTED", "DESCRIPTIVE_HIGH_UNCERTAINTY"}


def test_the_report_states_the_one_shot_facts_and_no_winner() -> None:
    text = REPORT.read_text(encoding="utf-8")
    for required in (
        "TEST_EVALUATION_COMPLETE",
        "FINAL_TEST_OBSERVED",
        "DESCRIPTIVE_GENERALIZATION_COMPARISON",
        "No test-driven tuning",
        "no winner was declared",
        "Holdout reads performed | 1",
    ):
        assert required in text, required
    for forbidden in ("the better model is", "we therefore select", "composite score of"):
        assert forbidden not in text.lower(), forbidden


def test_the_report_carries_the_required_sections() -> None:
    text = REPORT.read_text(encoding="utf-8")
    for heading in (
        "## 1. Holdout policy",
        "## 2. One-shot execution status",
        "## 3. Final model identities",
        "## 4. Test population",
        "## 5. Detector canonical bounding-box results",
        "## 6. Segmenter canonical mask results",
        "## 7. Segmenter canonical bounding-box results",
        "## 8. Detector versus segmenter localisation on the holdout",
        "## 9. Direct instance-mask IoU diagnostic",
        "## 10. Confusion matrix",
        "## 11. Object-level TP / FP / FN",
        "## 12. Deterministically selected qualitative FP/FN examples",
        "## 13. Relationship to the validation conclusions",
        "## 14. Final limitations",
        "## 15. No test-driven tuning",
    ):
        assert heading in text, heading


def test_the_committed_figures_exist_and_carry_no_imagery() -> None:
    root = PATHS.reports / "figures" / "final_test"
    for model in ("d2", "s1"):
        for name in ("confusion_matrix.png", "confusion_matrix_normalized.png"):
            assert (root / model / name).is_file(), f"{model}/{name}"
    # Only confusion matrices may be committed from this phase.
    assert sorted(path.name for path in root.rglob("*.png")) == [
        "confusion_matrix.png",
        "confusion_matrix.png",
        "confusion_matrix_normalized.png",
        "confusion_matrix_normalized.png",
    ]


# --- the one-shot record ----------------------------------------------------------------------


def test_the_provenance_records_one_attempt_and_no_rerun(provenance: dict[str, Any]) -> None:
    details = provenance["details"]
    assert details["classification"] == "TEST_EVALUATION_COMPLETE"
    assert details["attempt"] == 1
    assert details["attempt_counter_resettable"] is False
    assert details["ledger_state"] == "COMPLETE"
    assert details["holdout_reads_permitted"] == 1
    assert details["holdout_reads_performed"] == 1
    assert details["prediction_reruns"] == 0
    assert details["ledger"]["history"][-1]["state"] == "COMPLETE"


def test_the_provenance_records_that_nothing_was_trained_or_tuned(
    provenance: dict[str, Any],
) -> None:
    details = provenance["details"]
    for field in (
        "models_trained",
        "thresholds_tuned",
        "thresholds_swept",
        "latency_measurements_taken",
        "memory_measurements_taken",
        "new_spatial_metrics",
        "significance_tests",
    ):
        assert details[field] == 0, field
    for field in ("winner_declared", "composite_score", "model_selection_follows"):
        assert details[field] is False, field
    assert details["association_analysis_repeated"] is False


def test_the_post_test_policy_is_closed(provenance: dict[str, Any]) -> None:
    policy = provenance["details"]["post_test_policy"]
    assert policy["state"] == "FINAL_TEST_OBSERVED"
    for field in (
        "model_selection",
        "hyperparameter_tuning",
        "threshold_tuning",
        "data_cleaning_for_performance",
    ):
        assert policy[field] == "CLOSED", field


def test_metrics_came_from_the_persisted_predictions(provenance: dict[str, Any]) -> None:
    details = provenance["details"]
    assert details["metrics_derived_from"] == "PERSISTED_PREDICTIONS_ONLY"
    assert details["model_invoked_during_metric_computation"] is False
    assert details["model_invoked_during_report_generation"] is False
    assert details["inference_passes"] == 3
    assert details["models_executed"] == 2


def test_the_two_recorded_protocol_notes_are_disclosed(provenance: dict[str, Any]) -> None:
    """A gap and a tension in the frozen protocol are recorded, not smoothed over."""
    gap = provenance["details"]["protocol_gap_resolution"]
    assert gap["label"] == "PRE_EXECUTION_PROTOCOL_GAP_RESOLUTION"
    assert gap["resolved_before_any_holdout_number_existed"] is True
    assert gap["applied_identically_to_both_models"] is True
    assert gap["phase_11a_declared_this_resolution"] is False

    note = provenance["details"]["qualitative_publication_note"]
    assert note["label"] == "PROTOCOL_COVERAGE_NOTE"
    assert note["protocol_edited_to_accommodate_this"] is False


def test_the_holdout_integrity_check_passed_before_any_model_ran(
    provenance: dict[str, Any],
) -> None:
    integrity = provenance["details"]["integrity"]
    assert integrity["performed_before_model_execution"] is True
    assert integrity["problems"] == 0
    assert integrity["silent_removal_permitted"] is False
    assert integrity["split_specific_branch_used"] is False
    assert integrity["materialised_by"] == "scripts/materialize_task_datasets.py"
    assert integrity["byte_identical_copies"] == integrity["image_copies"] == TEST_IMAGES
    assert integrity["images_materialised"] == TEST_IMAGES
    assert integrity["annotations_materialised"] == TEST_ANNOTATIONS
    assert integrity["geometry_round_trip_mismatches"] == 0
    assert integrity["detection_segmentation_alignment_problems"] == 0
