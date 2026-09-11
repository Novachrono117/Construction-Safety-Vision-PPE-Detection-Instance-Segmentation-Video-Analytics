"""Tests for the phase 11B holdout execution machinery.

Every test here uses synthetic fixtures. **No holdout data is opened, no model
is loaded and no inference runs**, so the suite can be run at any time without
touching the protected split or needing either authorisation gate.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from construction_safety_vision import final_holdout_execution as execution
from construction_safety_vision.final_holdout_evaluation import (
    CLASSES,
    OPERATIONAL_CONF,
    QUALITATIVE_CATEGORIES,
    QualitativeCandidate,
)
from construction_safety_vision.paths import ProjectPaths

PATHS = ProjectPaths.from_root()
ROOT = PATHS.root

MODULE = ROOT / "src" / "construction_safety_vision" / "final_holdout_execution.py"
RESULTS_MODULE = ROOT / "src" / "construction_safety_vision" / "final_holdout_results.py"
RUNNER_SCRIPT = ROOT / "scripts" / "evaluate_final_holdout.py"

CLASS_MAP = {name: index for index, name in enumerate(CLASSES)}


def _mask(height: int, width: int, box: tuple[int, int, int, int]) -> np.ndarray:
    """Build a rectangular boolean mask.

    Args:
        height: Canvas height.
        width: Canvas width.
        box: ``(x1, y1, x2, y2)`` integer bounds.

    Returns:
        The mask.
    """
    mask = np.zeros((height, width), dtype=bool)
    x1, y1, x2, y2 = box
    mask[y1:y2, x1:x2] = True
    return mask


def _population(annotations: list[dict[str, Any]]) -> execution.HoldoutPopulation:
    """Build a one-image synthetic population.

    Args:
        annotations: COCO annotations for that image.

    Returns:
        The population.
    """
    images = [{"id": 1, "file_name": "img.jpg", "height": 100, "width": 100}]
    categories = [{"id": index, "name": name} for name, index in CLASS_MAP.items()]
    document = {"images": images, "annotations": annotations, "categories": categories}
    return execution.HoldoutPopulation(
        detection=document,
        segmentation=document,
        image_order=("img",),
        files={"img": Path("img.jpg")},
        sizes={"img": (100, 100)},
        coco_ids={"img": 1},
        class_map=CLASS_MAP,
        membership_sha256=execution.membership_fingerprint(["img"]),
        integrity={"problems": 0},
    )


def _instance(
    class_name: str,
    score: float,
    box: tuple[float, float, float, float],
    mask: np.ndarray | None = None,
) -> execution.PredictedInstance:
    """Build one predicted instance.

    Args:
        class_name: Its class.
        score: Its confidence.
        box: ``(x1, y1, x2, y2)``.
        mask: Its mask, or ``None``.

    Returns:
        The instance.
    """
    return execution.PredictedInstance(
        image_id="img",
        class_name=class_name,
        score=score,
        box=box,
        mask_rle=execution.encode_mask(mask) if mask is not None else None,
    )


# --- fingerprints and geometry ----------------------------------------------------------


def test_membership_fingerprint_is_order_independent() -> None:
    assert execution.membership_fingerprint(["b", "a"]) == execution.membership_fingerprint(
        ["a", "b"]
    )
    assert execution.membership_fingerprint(["a"]) != execution.membership_fingerprint(["b"])


def test_digest_payload_is_deterministic_and_key_order_independent() -> None:
    first = execution.digest_payload({"a": 1, "b": [2, 3]})
    second = execution.digest_payload({"b": [2, 3], "a": 1})
    assert first == second
    assert len(first) == 64
    assert first != execution.digest_payload({"a": 1, "b": [2, 4]})


def test_mask_round_trips_through_rle() -> None:
    mask = _mask(100, 100, (10, 10, 40, 50))
    assert np.array_equal(execution.decode_mask(execution.encode_mask(mask)), mask)


def test_box_iou_matches_hand_computation() -> None:
    assert execution.box_iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)
    assert execution.box_iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    # 100 and 100 with a 25 overlap: 25 / 175.
    assert execution.box_iou((0, 0, 10, 10), (5, 5, 15, 15)) == pytest.approx(25 / 175)


# --- the operating point ----------------------------------------------------------------


def test_operating_point_filters_at_the_frozen_confidence() -> None:
    per_image = {
        "img": [
            _instance("person", 0.90, (0, 0, 10, 10)),
            _instance("person", OPERATIONAL_CONF, (0, 0, 10, 10)),
            _instance("person", 0.24999, (0, 0, 10, 10)),
            _instance("person", 0.001, (0, 0, 10, 10)),
        ]
    }
    kept = execution.at_operating_point(per_image)
    assert [instance.score for instance in kept["img"]] == [0.90, OPERATIONAL_CONF]


def test_the_gap_resolution_is_labelled_rather_than_attributed_to_phase_11a() -> None:
    assert execution.GAP_RESOLUTION == "PRE_EXECUTION_PROTOCOL_GAP_RESOLUTION"
    assert "DERIVED_FROM_DECLARED_AP_PASS" in execution.DERIVED_FROM_AP_PASS


# --- object-level matching and the frozen taxonomy --------------------------------------


def test_a_clean_match_is_a_true_positive_and_well_handled() -> None:
    population = _population(
        [
            {
                "id": 1,
                "image_id": 1,
                "category_id": CLASS_MAP["person"],
                "bbox": [10, 10, 30, 40],
                "segmentation": execution.encode_mask(_mask(100, 100, (10, 10, 40, 50))),
                "area": 1200,
                "iscrowd": 0,
            }
        ]
    )
    truth = execution.canonical_instances(population, with_masks=True)
    predictions = {"img": [_instance("person", 0.9, (10, 10, 40, 50))]}
    outcome = execution.object_level_outcomes(
        predictions, truth, population, with_predicted_masks=False
    )
    assert outcome["true_positives"] == 1
    assert outcome["false_positives"] == 0
    assert outcome["false_negatives"] == 0
    assert outcome["taxonomy_census"]["WELL_HANDLED_INSTANCE"] == 1


def test_the_four_failure_categories_are_evaluated_in_the_frozen_order() -> None:
    gt_mask = _mask(100, 100, (10, 10, 40, 50))
    population = _population(
        [
            {
                "id": 1,
                "image_id": 1,
                "category_id": CLASS_MAP["person"],
                "bbox": [10, 10, 30, 40],
                "segmentation": execution.encode_mask(gt_mask),
                "area": 1200,
                "iscrowd": 0,
            }
        ]
    )
    truth = execution.canonical_instances(population, with_masks=True)

    # Nothing overlaps at all -> DETECTION_MISS.
    miss = execution.object_level_outcomes(
        {"img": []}, truth, population, with_predicted_masks=False
    )
    assert miss["taxonomy_census"]["DETECTION_MISS"] == 1

    # An overlapping prediction of another class -> CLASSIFICATION_MISMATCH.
    mismatch = execution.object_level_outcomes(
        {"img": [_instance("vest_on_body", 0.9, (10, 10, 40, 50))]},
        truth,
        population,
        with_predicted_masks=False,
    )
    assert mismatch["taxonomy_census"]["CLASSIFICATION_MISMATCH"] == 1

    # Nothing of any class reaches 0.50, so a weak same-class overlap is still a
    # miss: DETECTION_MISS is evaluated first and wins.
    weak = execution.object_level_outcomes(
        {"img": [_instance("person", 0.9, (30, 30, 60, 70))]},
        truth,
        population,
        with_predicted_masks=False,
    )
    assert weak["taxonomy_census"]["DETECTION_MISS"] == 1
    assert weak["taxonomy_census"]["LOCALIZATION_FAILURE"] == 0

    # Matched box, poor mask -> MASK_QUALITY_FAILURE, but only when masks exist.
    poor = _mask(100, 100, (10, 10, 18, 50))
    quality = execution.object_level_outcomes(
        {"img": [_instance("person", 0.9, (10, 10, 40, 50), poor)]},
        truth,
        population,
        with_predicted_masks=True,
    )
    assert quality["taxonomy_census"]["MASK_QUALITY_FAILURE"] == 1
    # The same prediction judged without masks is well handled: the category is
    # structurally unavailable to a model that emits no mask.
    without = execution.object_level_outcomes(
        {"img": [_instance("person", 0.9, (10, 10, 40, 50), poor)]},
        truth,
        population,
        with_predicted_masks=False,
    )
    assert without["taxonomy_census"]["MASK_QUALITY_FAILURE"] == 0


def test_the_census_sums_to_the_ground_truth_count() -> None:
    annotations = [
        {
            "id": index,
            "image_id": 1,
            "category_id": CLASS_MAP["person"],
            "bbox": [index * 20, 0, 15, 15],
            "segmentation": execution.encode_mask(
                _mask(100, 100, (index * 20, 0, index * 20 + 15, 15))
            ),
            "area": 225,
            "iscrowd": 0,
        }
        for index in range(1, 4)
    ]
    population = _population(annotations)
    truth = execution.canonical_instances(population, with_masks=True)
    outcome = execution.object_level_outcomes(
        {"img": [_instance("person", 0.9, (20, 0, 35, 15))]},
        truth,
        population,
        with_predicted_masks=False,
    )
    assert sum(outcome["taxonomy_census"].values()) == outcome["ground_truth"] == 3


def test_matching_is_one_to_one_so_a_second_prediction_is_a_false_positive() -> None:
    population = _population(
        [
            {
                "id": 1,
                "image_id": 1,
                "category_id": CLASS_MAP["person"],
                "bbox": [10, 10, 30, 40],
                "segmentation": execution.encode_mask(_mask(100, 100, (10, 10, 40, 50))),
                "area": 1200,
                "iscrowd": 0,
            }
        ]
    )
    truth = execution.canonical_instances(population, with_masks=True)
    outcome = execution.object_level_outcomes(
        {
            "img": [
                _instance("person", 0.9, (10, 10, 40, 50)),
                _instance("person", 0.8, (11, 11, 41, 51)),
            ]
        },
        truth,
        population,
        with_predicted_masks=False,
    )
    assert outcome["true_positives"] == 1
    assert outcome["false_positives"] == 1


# --- the deterministic qualitative selection --------------------------------------------


def test_every_frozen_category_has_a_declared_sort_direction() -> None:
    assert tuple(execution.QUALITATIVE_DESCENDING) == QUALITATIVE_CATEGORIES


def test_selection_is_deterministic_and_never_reuses_an_instance() -> None:
    candidates = [
        QualitativeCandidate(
            category="LOWEST_IOU_MATCHED_INSTANCE",
            rank_value=0.4,
            canonical_annotation_id=2,
            canonical_prediction_index=0,
            source_image_id="b",
        ),
        QualitativeCandidate(
            category="LOWEST_IOU_MATCHED_INSTANCE",
            rank_value=0.4,
            canonical_annotation_id=1,
            canonical_prediction_index=0,
            source_image_id="a",
        ),
        QualitativeCandidate(
            category="SEGMENTATION_UNDER_COVERAGE",
            rank_value=0.1,
            canonical_annotation_id=1,
            canonical_prediction_index=0,
            source_image_id="a",
        ),
    ]
    first = execution.qualitative_selection(
        candidates, applicable=dict.fromkeys(QUALITATIVE_CATEGORIES, True)
    )
    second = execution.qualitative_selection(
        list(reversed(candidates)), applicable=dict.fromkeys(QUALITATIVE_CATEGORIES, True)
    )
    assert first == second
    # The tie on 0.4 resolves by annotation id, so id 1 ranks first.
    chosen = first["categories"]["LOWEST_IOU_MATCHED_INSTANCE"]["selected"]
    assert [entry["canonical_annotation_id"] for entry in chosen] == [1, 2]
    # Instance 1 was claimed by the earlier category, so the later one skips it.
    assert first["categories"]["SEGMENTATION_UNDER_COVERAGE"]["selected"] == []


def test_an_underfilled_category_records_its_shortfall_and_is_never_topped_up() -> None:
    candidates = [
        QualitativeCandidate(
            category="HIGHEST_CONFIDENCE_FALSE_POSITIVE",
            rank_value=0.9,
            canonical_annotation_id=-1,
            canonical_prediction_index=0,
            source_image_id="a",
        )
    ]
    selection = execution.qualitative_selection(
        candidates, applicable=dict.fromkeys(QUALITATIVE_CATEGORIES, True)
    )
    shortfall = selection["shortfalls"]["HIGHEST_CONFIDENCE_FALSE_POSITIVE"]
    assert shortfall == {"selected": 1, "quota": 3, "available": 1}
    assert selection["topping_up_from_another_category_permitted"] is False
    assert selection["images_inspected_to_design_this_rule"] == 0
    assert selection["images_browsed_before_selection"] == 0


def test_a_category_that_does_not_apply_is_marked_rather_than_left_empty() -> None:
    selection = execution.qualitative_selection(
        [],
        applicable={
            category: category != "SEGMENTATION_OVER_COVERAGE"
            for category in QUALITATIVE_CATEGORIES
        },
    )
    assert selection["categories"]["SEGMENTATION_OVER_COVERAGE"]["applicable"] is False


# --- redaction and result validation ----------------------------------------------------


def test_redaction_removes_every_identifier_at_every_depth() -> None:
    payload = {
        "keep": 1,
        "source_image_id": "leak",
        "nested": [{"canonical_annotation_id": 7, "value": 2}],
        "images_with_predictions": ["leak"],
    }
    cleaned = execution.redact_identifiers(payload)
    assert cleaned == {"keep": 1, "nested": [{"value": 2}]}


def test_a_result_naming_a_winner_or_a_composite_is_refused() -> None:
    base = {
        "protocol_fingerprint": "p",
        "prediction_fingerprint": "q",
        "split": "test",
        "metrics_derived_from": "PERSISTED_PREDICTIONS_ONLY",
    }
    assert (
        execution.validate_result(
            base, protocol_fingerprint="p", prediction_sha256="q", holdout_identifiers=()
        )
        == []
    )
    for key in ("winner_declared", "composite_score", "threshold_tuned_on_test"):
        assert execution.validate_result(
            {**base, key: True},
            protocol_fingerprint="p",
            prediction_sha256="q",
            holdout_identifiers=(),
        )


def test_a_leaked_holdout_identifier_is_caught() -> None:
    payload = {
        "protocol_fingerprint": "p",
        "prediction_fingerprint": "q",
        "split": "test",
        "metrics_derived_from": "PERSISTED_PREDICTIONS_ONLY",
        "note": "something about aBcDeF123",
    }
    problems = execution.validate_result(
        payload,
        protocol_fingerprint="p",
        prediction_sha256="q",
        holdout_identifiers=("aBcDeF123",),
    )
    assert problems and "holdout identifier" in problems[0]


def test_a_result_from_a_different_prediction_set_is_refused() -> None:
    payload = {
        "protocol_fingerprint": "p",
        "prediction_fingerprint": "OTHER",
        "split": "test",
        "metrics_derived_from": "PERSISTED_PREDICTIONS_ONLY",
    }
    assert execution.validate_result(
        payload, protocol_fingerprint="p", prediction_sha256="q", holdout_identifiers=()
    )


# --- the support rule -------------------------------------------------------------------


def test_the_support_rule_is_the_unchanged_phase_7a_rule_and_names_no_class() -> None:
    annotations = [
        {
            "id": index,
            "image_id": 1,
            "category_id": CLASS_MAP["person"],
            "bbox": [0, 0, 5, 5],
            "segmentation": execution.encode_mask(_mask(100, 100, (0, 0, 5, 5))),
            "area": 25,
            "iscrowd": 0,
        }
        for index in range(1, 25)
    ]
    admitted = execution.support(_population(annotations))
    # One image, so the image threshold refuses it even at 24 instances.
    assert admitted["person"]["instances"] == 24
    assert admitted["person"]["images"] == 1
    assert admitted["person"]["supported"] is False
    assert admitted["person"]["status"] == "DESCRIPTIVE_HIGH_UNCERTAINTY"
    macro = execution.supported_macro({"person": {"AP@0.50:0.95": 0.5}}, admitted)
    assert macro["value"] is None
    assert macro["names_no_class"] is True
    assert macro["rule_origin"] == "PREEXISTING_PHASE_7A_SUPPORT_RULE_REUSED_UNCHANGED"
    assert macro["role_on_test"] == "DESCRIPTIVE_CAVEAT_ONLY_NEVER_A_SELECTION_RULE"


# --- prediction persistence -------------------------------------------------------------


def test_a_prediction_fingerprint_changes_with_a_prediction_and_not_with_a_path() -> None:
    population = _population([])
    raw = {
        "per_image": {"img": [_instance("person", 0.9, (0, 0, 10, 10))]},
        "total": 1,
        "names": {},
        "masks_reconstructed": 0,
        "excluded": [],
        "images": 1,
    }
    settings = {"imgsz": 768, "conf": 0.001}
    document = execution.prediction_document(
        raw,
        population,
        model={"experiment": "D2"},
        protocol_fingerprint="p",
        settings=settings,
        pass_name="AP",
    )
    first = execution.prediction_fingerprint(document)

    moved = {**document, "written_to": "some-machine-specific-run-directory"}
    assert execution.prediction_fingerprint(moved) == first

    changed = {
        **document,
        "images_with_predictions": [
            {
                **document["images_with_predictions"][0],
                "predictions": [
                    {
                        **document["images_with_predictions"][0]["predictions"][0],
                        "score": 0.8,
                    }
                ],
            }
        ],
    }
    assert execution.prediction_fingerprint(changed) != first


def test_metrics_are_computed_from_the_persisted_document() -> None:
    population = _population([])
    raw = {
        "per_image": {"img": [_instance("person", 0.9, (1.5, 2.5, 10.25, 20.75))]},
        "total": 1,
        "names": {},
        "masks_reconstructed": 0,
        "excluded": [],
        "images": 1,
    }
    document = execution.prediction_document(
        raw,
        population,
        model={"experiment": "D2"},
        protocol_fingerprint="p",
        settings={"conf": 0.001},
        pass_name="AP",
    )
    reloaded = execution.load_predictions(document)
    assert reloaded["img"][0].box == (1.5, 2.5, 10.25, 20.75)
    assert reloaded["img"][0].score == 0.9


# --- the code boundaries the phase relies on --------------------------------------------


def test_no_phase_11b_file_writes_the_environment_gate() -> None:
    """No file in this phase may set, export or modify the gate variable."""
    for path in (MODULE, RESULTS_MODULE, RUNNER_SCRIPT):
        text = path.read_text(encoding="utf-8")
        for forbidden in (
            "os.environ[",
            "os.putenv",
            "environ.update",
            "monkeypatch.setenv",
        ):
            assert forbidden not in text, (path.name, forbidden)


def test_the_runner_stays_the_authorisation_boundary_and_not_the_implementation() -> None:
    """The runner must hold no inference or split-enumeration call path."""
    text = RUNNER_SCRIPT.read_text(encoding="utf-8")
    for forbidden in ("import torch", "from torch", "YOLO(", ".train(", "model.val("):
        assert forbidden not in text, forbidden
    assert "final_holdout_execution" in text


def test_no_phase_11b_file_trains_a_model() -> None:
    for path in (MODULE, RESULTS_MODULE, RUNNER_SCRIPT):
        text = path.read_text(encoding="utf-8")
        for forbidden in (".train(", "trainer", "optimizer"):
            assert forbidden not in text, (path.name, forbidden)


def test_the_holdout_is_materialised_by_the_phase_5d_function() -> None:
    """Never a copy of it: a second implementation could drift."""
    text = MODULE.read_text(encoding="utf-8")
    assert execution.MATERIALIZER == "scripts/materialize_task_datasets.py"
    assert "materialize_split(" in text
    assert "split_specific_branch_used" in text


def test_localization_failure_fires_when_the_only_match_was_awarded_elsewhere() -> None:
    """The frozen cascade's third category, under the recorded resolution."""
    annotations = []
    for index, box in enumerate(((0, 0, 20, 20), (2, 2, 22, 22)), start=1):
        x1, y1, x2, y2 = box
        annotations.append(
            {
                "id": index,
                "image_id": 1,
                "category_id": CLASS_MAP["person"],
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "segmentation": execution.encode_mask(_mask(100, 100, box)),
                "area": (x2 - x1) * (y2 - y1),
                "iscrowd": 0,
            }
        )
    population = _population(annotations)
    truth = execution.canonical_instances(population, with_masks=True)
    outcome = execution.object_level_outcomes(
        {"img": [_instance("person", 0.9, (0, 0, 20, 20))]},
        truth,
        population,
        with_predicted_masks=False,
    )
    assert outcome["true_positives"] == 1
    assert outcome["taxonomy_census"]["LOCALIZATION_FAILURE"] == 1
    assert outcome["taxonomy_census"]["DETECTION_MISS"] == 0


def test_the_taxonomy_cascade_resolution_is_recorded_not_silent() -> None:
    resolution = execution.TAXONOMY_CASCADE_RESOLUTION
    assert resolution["resolved_before_any_holdout_number_existed"] is True
    assert resolution["phase_11a_protocol_edited"] is False
    assert resolution["categories_added"] == 0
    assert len(resolution["ambiguities"]) == 2
