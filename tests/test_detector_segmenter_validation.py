"""Tests for the phase 10B validation comparison.

Two halves. The first exercises the spatial measurements, the association rule
and the validators on synthetic fixtures - masks small enough to reason about
by hand, so a wrong formula shows up as a wrong number rather than a plausible
one. The second checks the committed artifacts describe the comparison that
actually ran.

No test here loads a model, runs inference, reads a dataset image or touches
the holdout.
"""

from __future__ import annotations

import csv
import json
from typing import Any

import numpy as np
import pytest

from construction_safety_vision.data.canonical import scan_for_sensitive
from construction_safety_vision.detector_segmenter_analysis import (
    AGREE,
    ASSOCIATION_DISAGREEMENT,
    BOX_ONLY,
    FEATURE_SEMANTICS,
    GEOMETRY_ISOLATING,
    INTERPRETIVE_PROXY,
    MASK_ONLY,
    NEITHER,
    NO_BOX_PROXY,
    PIPELINE_LEVEL,
    PURE_GEOMETRIC,
    RARE_CLASS,
    RARE_CLASS_STATUS,
    STATISTIC_NAMES,
    TAXONOMY_EXCEPTION,
    TAXONOMY_EXHAUSTIVE,
    TAXONOMY_NON_EXHAUSTIVE,
    Instance,
    associate,
    association_category,
    box_containment,
    box_intersection,
    centroid_displacement,
    decompose_all_class_delta,
    describe,
    instance_area_pixels,
    mask_centroid,
    mask_containment,
    mask_intersection,
    mask_to_box_fill_ratio,
    result_fingerprint,
    shape_extent,
    supported_class_sensitivity,
    tally,
    tally_from_counts,
    validate_box_comparison,
    validate_spatial_comparison,
)
from construction_safety_vision.detector_segmenter_comparison import (
    ASSOCIATION_CATEGORIES,
    SPATIAL_FEATURES,
    load_comparison_protocol,
)
from construction_safety_vision.paths import ProjectPaths

BOX_JSON = "detector_segmenter_box_comparison.json"
BOX_CSV = "detector_segmenter_box_comparison.csv"
SPATIAL_JSON = "detector_segmenter_spatial_comparison.json"
REPORT_MD = "detector_segmenter_validation_comparison.md"
EXAMPLES_CSV = "detector_segmenter_comparison_examples.csv"

PROTOCOL_SHA = "d92a157637baf52f9a7248bf24257d5a177e8496b7727ffb5afc874ab96d1c4d"
DETECTOR_SHA = "0466f872a9de22898c70d8834cf1fcfb3d77f6c9fb27e81ee6248d3d19cbc206"
SEGMENTER_SHA = "29337d671459d0f742fb713613cc41d0eafcb8a21f5846ac0da65b2ffc024f20"
MEMBERSHIP_SHA = "54e4ae8afd711314b472649b1564286d0a6d3d8f13e32c035d31402cbfa5f152"


@pytest.fixture(scope="module")
def paths() -> ProjectPaths:
    """Resolve the project layout.

    Returns:
        The project paths.
    """
    return ProjectPaths.from_root()


def _rect(height: int, width: int, y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
    """Build a boolean mask with one filled rectangle.

    Args:
        height: Canvas height.
        width: Canvas width.
        y0: Top row, inclusive.
        y1: Bottom row, exclusive.
        x0: Left column, inclusive.
        x1: Right column, exclusive.

    Returns:
        The mask.
    """
    mask = np.zeros((height, width), dtype=bool)
    mask[y0:y1, x0:x1] = True
    return mask


# --- the spatial measurements, on masks small enough to check by hand ----------------


def test_instance_area_counts_foreground_pixels() -> None:
    """A 4x5 rectangle is 20 pixels, whatever the canvas."""
    instance = Instance("img", "person", 0.9, (0, 0, 10, 10), _rect(20, 20, 2, 6, 3, 8))

    assert instance_area_pixels(instance) == 20


def test_fill_ratio_is_mask_area_over_the_models_own_box() -> None:
    """A 20-pixel mask inside a 100-pixel box fills a fifth of it."""
    instance = Instance("img", "person", 0.9, (0.0, 0.0, 10.0, 10.0), _rect(20, 20, 2, 6, 3, 8))

    assert mask_to_box_fill_ratio(instance) == pytest.approx(0.2)


def test_fill_ratio_is_none_for_a_degenerate_box() -> None:
    """A ratio over zero area is undefined, not zero and not infinite."""
    instance = Instance("img", "person", 0.9, (5.0, 5.0, 5.0, 5.0), _rect(20, 20, 2, 6, 3, 8))

    assert mask_to_box_fill_ratio(instance) is None


def test_centroid_is_the_mean_of_the_foreground() -> None:
    """Rows 2-5 and columns 3-7 centre on (5.0, 3.5) in (x, y)."""
    instance = Instance("img", "person", 0.9, (0, 0, 10, 10), _rect(20, 20, 2, 6, 3, 8))

    assert mask_centroid(instance) == pytest.approx((5.0, 3.5))


def test_shape_extent_is_one_for_a_filled_rectangle() -> None:
    """A rectangle fills its own bounding rectangle exactly."""
    instance = Instance("img", "person", 0.9, (0, 0, 10, 10), _rect(20, 20, 2, 6, 3, 8))

    assert shape_extent(instance) == pytest.approx(1.0)


def test_shape_extent_falls_below_one_for_a_non_rectangular_shape() -> None:
    """An L-shape leaves part of its bounding rectangle empty."""
    mask = np.zeros((10, 10), dtype=bool)
    mask[0:6, 0:2] = True
    mask[4:6, 0:6] = True
    instance = Instance("img", "person", 0.9, (0, 0, 10, 10), mask)

    extent = shape_extent(instance)

    assert extent is not None
    assert 0.0 < extent < 1.0
    assert extent == pytest.approx(int(mask.sum()) / 36.0)


def test_shape_extent_uses_the_masks_own_rectangle_not_the_predicted_box() -> None:
    """A wildly wrong predicted box must not change a shape property."""
    mask = _rect(20, 20, 2, 6, 3, 8)
    tight = Instance("img", "person", 0.9, (3.0, 2.0, 8.0, 6.0), mask)
    loose = Instance("img", "person", 0.9, (0.0, 0.0, 20.0, 20.0), mask)

    assert shape_extent(tight) == shape_extent(loose)


def test_mask_intersection_and_containment() -> None:
    """A 20-pixel PPE mask half inside a person mask is 0.5 contained."""
    ppe = Instance("img", "helmet_on_head", 0.9, (0, 0, 10, 10), _rect(20, 20, 0, 4, 0, 5))
    person = Instance("img", "person", 0.9, (0, 0, 20, 20), _rect(20, 20, 2, 12, 0, 5))

    assert mask_intersection(ppe, person) == 10
    assert mask_containment(ppe, person) == pytest.approx(0.5)


def test_mask_intersection_refuses_mismatched_canvases() -> None:
    """Two masks on different canvases would measure the mismatch."""
    ppe = Instance("img", "helmet_on_head", 0.9, (0, 0, 5, 5), _rect(20, 20, 0, 4, 0, 5))
    person = Instance("img", "person", 0.9, (0, 0, 5, 5), _rect(10, 10, 0, 4, 0, 5))

    from construction_safety_vision.detector_segmenter_analysis import SpatialAnalysisError

    with pytest.raises(SpatialAnalysisError, match="different canvases"):
        mask_intersection(ppe, person)


def test_box_proxies_mirror_the_mask_quantities() -> None:
    """The box counterparts use the same shape of definition."""
    ppe = Instance("img", "helmet_on_head", 0.9, (0.0, 0.0, 10.0, 10.0))
    person = Instance("img", "person", 0.9, (5.0, 0.0, 15.0, 10.0))

    assert box_intersection(ppe, person) == pytest.approx(50.0)
    assert box_containment(ppe, person) == pytest.approx(0.5)


def test_centroid_displacement_is_the_half_pixel_floor_for_a_centred_mask() -> None:
    """A mask filling its box sits at the convention floor, not at exactly zero.

    The centroid averages pixel indices, so it lands at pixel centres, while
    the box is in continuous coordinates. A perfectly filling mask is therefore
    half a pixel off in each axis - about 0.71 px. Asserting the exact value
    pins the convention, so a future change to it fails here rather than
    quietly shifting every displacement in the report.
    """
    instance = Instance("img", "person", 0.9, (0.0, 0.0, 10.0, 10.0), _rect(10, 10, 0, 10, 0, 10))

    assert centroid_displacement(instance) == pytest.approx(0.5 * 2**0.5, abs=1e-9)


def test_centroid_displacement_grows_when_the_mask_sits_off_centre() -> None:
    """An off-centre shape is where the box centre becomes a poor proxy."""
    mask = _rect(20, 20, 0, 4, 0, 4)
    instance = Instance("img", "person", 0.9, (0.0, 0.0, 20.0, 20.0), mask)

    displacement = centroid_displacement(instance)

    assert displacement is not None
    assert displacement > 5.0


# --- the frozen association rule --------------------------------------------------------


def test_association_picks_the_most_overlapping_person_above_the_floor() -> None:
    """The rule associates with the best candidate that clears the floor."""
    ppe = Instance("i", "helmet_on_head", 0.9, (0, 0, 10, 10), _rect(20, 20, 0, 10, 0, 10))
    mostly = Instance("i", "person", 0.9, (0, 0, 20, 20), _rect(20, 20, 0, 10, 0, 9))
    barely = Instance("i", "person", 0.9, (0, 0, 20, 20), _rect(20, 20, 0, 10, 0, 6))

    assert associate(ppe, [barely, mostly], containment_floor=0.5, use_masks=True) == 1


def test_association_returns_none_below_the_floor() -> None:
    """A PPE instance barely touching a person is not associated with them."""
    ppe = Instance("i", "helmet_on_head", 0.9, (0, 0, 10, 10), _rect(20, 20, 0, 10, 0, 10))
    person = Instance("i", "person", 0.9, (0, 0, 20, 20), _rect(20, 20, 0, 10, 0, 2))

    assert associate(ppe, [person], containment_floor=0.5, use_masks=True) is None


def test_association_is_order_independent() -> None:
    """A deterministic rule cannot depend on how the candidates were listed."""
    ppe = Instance("i", "helmet_on_head", 0.9, (0, 0, 10, 10), _rect(20, 20, 0, 10, 0, 10))
    people = [
        Instance("i", "person", 0.9, (0, 0, 20, 20), _rect(20, 20, 0, 10, 0, 7)),
        Instance("i", "person", 0.9, (0, 0, 20, 20), _rect(20, 20, 0, 10, 0, 10)),
    ]

    forward = associate(ppe, people, containment_floor=0.5, use_masks=True)
    backward = associate(ppe, list(reversed(people)), containment_floor=0.5, use_masks=True)

    assert people[forward] is list(reversed(people))[backward]


@pytest.mark.parametrize(
    ("mask_choice", "box_choice", "expected"),
    [
        (0, 0, AGREE),
        (None, None, NEITHER),
        (0, None, MASK_ONLY),
        (None, 0, BOX_ONLY),
        (0, 1, TAXONOMY_EXCEPTION),
    ],
)
def test_association_categories(
    mask_choice: int | None, box_choice: int | None, expected: str
) -> None:
    """Including the state the four frozen categories do not cover."""
    assert association_category(mask_choice, box_choice) == expected


def test_the_frozen_four_are_unchanged() -> None:
    """Phase 10A's categories keep their names, their order and their membership."""
    assert ASSOCIATION_CATEGORIES == (
        "BOX_AND_MASK_AGREE",
        "BOX_ONLY_ASSOCIATION",
        "MASK_ONLY_ASSOCIATION",
        "NEITHER_ASSOCIATION",
    )


def test_the_exception_is_not_a_fifth_peer_category() -> None:
    """It is outside the frozen taxonomy, not a new member of it."""
    assert TAXONOMY_EXCEPTION not in ASSOCIATION_CATEGORIES
    assert association_category(0, 1) not in ASSOCIATION_CATEGORIES


def test_the_tally_separates_classified_from_exceptions() -> None:
    """The frozen four keep their own denominator; the exception is excluded."""
    result = tally([AGREE, AGREE, NEITHER, BOX_ONLY, TAXONOMY_EXCEPTION])

    assert result["total_relationships"] == 5
    assert result["classified_relationships"] == 4
    assert result["taxonomy_exceptions"] == 1
    assert result["taxonomy_coverage"] == pytest.approx(0.8)
    assert result["counts_partition"] is True
    assert result["frozen_category_counts"][AGREE] == 2
    assert result["frozen_category_percentages"][AGREE] == pytest.approx(50.0)
    assert result["percentage_denominator"] == "classified_relationships"
    assert TAXONOMY_EXCEPTION not in result["frozen_category_counts"]


def test_percentages_use_the_classified_denominator_not_the_total() -> None:
    """Mixing an undeclared state into the denominator would change their meaning."""
    result = tally([AGREE, TAXONOMY_EXCEPTION])

    assert result["frozen_category_percentages"][AGREE] == pytest.approx(100.0)
    assert result["taxonomy_coverage"] == pytest.approx(0.5)


def test_the_taxonomy_status_reflects_whether_anything_fell_outside() -> None:
    """Exhaustive when the four cover everything observed, non-exhaustive otherwise."""
    assert tally([AGREE, NEITHER])["taxonomy_status"] == TAXONOMY_EXHAUSTIVE
    assert tally([AGREE, TAXONOMY_EXCEPTION])["taxonomy_status"] == TAXONOMY_NON_EXHAUSTIVE


def test_the_exception_is_a_disagreement_not_an_error() -> None:
    """There is no association ground truth, so no rule can be called wrong."""
    result = tally([TAXONOMY_EXCEPTION])
    exception = result["taxonomy_exception"]

    assert exception["type"] == TAXONOMY_EXCEPTION
    assert exception["status"] == "UNCLASSIFIED_BY_FROZEN_FOUR_CATEGORY_TAXONOMY"
    assert exception["is_a_fifth_frozen_category"] is False
    assert exception["reading"] == ASSOCIATION_DISAGREEMENT
    assert "error" not in exception["reading"].lower()


def test_reaggregating_from_counts_reproduces_the_tally() -> None:
    """A recorded result can be restructured without re-running the models."""
    categories = [AGREE, AGREE, NEITHER, BOX_ONLY, TAXONOMY_EXCEPTION, TAXONOMY_EXCEPTION]
    counts = {
        AGREE: 2,
        BOX_ONLY: 1,
        MASK_ONLY: 0,
        NEITHER: 1,
        TAXONOMY_EXCEPTION: 2,
    }

    assert tally_from_counts(counts) == tally(categories)


def test_the_support_sensitivity_is_descriptive_and_selects_nothing() -> None:
    """A sensitivity check, explicitly not a metric and not a selection rule."""
    per_class = {
        "a": {"D2_AP@0.50:0.95": 0.4, "S1_AP@0.50:0.95": 0.3},
        "b": {"D2_AP@0.50:0.95": 0.6, "S1_AP@0.50:0.95": 0.7},
        RARE_CLASS: {"D2_AP@0.50:0.95": 0.0, "S1_AP@0.50:0.95": 0.9},
    }
    result = supported_class_sensitivity(per_class, ["a", "b"])

    assert result["label"] == "POST_HOC_DESCRIPTIVE_SUPPORT_SENSITIVITY"
    assert result["admitted_classes"] == ["a", "b"]
    assert result["excluded_classes"] == [RARE_CLASS]
    assert result["D2_supported_macro"] == pytest.approx(0.5)
    assert result["S1_supported_macro"] == pytest.approx(0.5)
    assert result["delta"] == pytest.approx(0.0)
    assert result["is_a_frozen_phase_10a_metric"] is False
    assert result["is_a_selection_rule"] is False
    assert result["is_a_significance_test"] is False
    assert result["changes_any_frozen_number"] is False


# --- statistics and decomposition ----------------------------------------------------------


def test_describe_reports_the_frozen_statistic_set() -> None:
    """The same set for every continuous quantity, and no inferential test."""
    result = describe([1.0, 2.0, 3.0, 4.0])

    assert set(result) == set(STATISTIC_NAMES)
    assert result["count"] == 4
    assert result["median"] == pytest.approx(2.5)
    assert "p_value" not in result
    assert "significant" not in result


def test_describe_handles_an_empty_sample() -> None:
    """An empty sample has a count and no invented statistics."""
    result = describe([])

    assert result["count"] == 0
    assert result["mean"] is None


def test_the_decomposition_is_exact_and_sums_to_the_delta() -> None:
    """The all-class figure is an unweighted mean, so this is exact arithmetic."""
    per_class = {
        "a": {"delta_AP@0.50:0.95": 0.10},
        "b": {"delta_AP@0.50:0.95": -0.05},
        RARE_CLASS: {"delta_AP@0.50:0.95": 0.40},
        "d": {"delta_AP@0.50:0.95": 0.0},
        "e": {"delta_AP@0.50:0.95": 0.05},
    }
    result = decompose_all_class_delta(per_class)
    expected = sum(row["delta_AP@0.50:0.95"] for row in per_class.values()) / 5

    assert result["class_count"] == 5
    assert result["total_contribution"] == pytest.approx(expected)
    assert result["declined_classes"] == ["b"]


def test_the_decomposition_separates_the_rare_class() -> None:
    """A one-image class can move an unweighted mean without meaning anything.

    This fixture is the case that matters: the rare class alone contributes
    more than the whole aggregate, and the other class actually declined. The
    flag has to catch that, because it is the difference between "the model
    improved" and "one high-uncertainty class moved".
    """
    per_class = {
        "a": {"delta_AP@0.50:0.95": -0.02},
        RARE_CLASS: {"delta_AP@0.50:0.95": 0.40},
    }
    result = decompose_all_class_delta(per_class)

    assert result["rare_class"] == RARE_CLASS
    assert result["rare_class_delta"] == pytest.approx(0.40)
    assert result["delta_excluding_rare_class"] == pytest.approx(-0.02)
    assert result["rare_class_contribution_exceeds_total"] is True
    assert result["declined_classes"] == ["a"]


def test_the_rare_class_flag_stays_false_when_it_is_not_carrying_the_result() -> None:
    """The flag must not fire on every aggregate that happens to include it."""
    per_class = {
        "a": {"delta_AP@0.50:0.95": 0.30},
        "b": {"delta_AP@0.50:0.95": 0.30},
        RARE_CLASS: {"delta_AP@0.50:0.95": 0.03},
    }
    result = decompose_all_class_delta(per_class)

    assert result["rare_class_contribution_exceeds_total"] is False
    assert result["declined_classes"] == []


def test_the_fingerprint_is_deterministic_and_content_sensitive() -> None:
    """Same content hashes the same; changed content does not."""
    values = {"a": 1, "b": [1, 2]}

    assert result_fingerprint(values) == result_fingerprint({"b": [1, 2], "a": 1})
    assert result_fingerprint(values) != result_fingerprint({"a": 2, "b": [1, 2]})


# --- the committed artifacts ------------------------------------------------------------------


@pytest.fixture(scope="module")
def box(paths: ProjectPaths) -> dict[str, Any]:
    """Read the committed box comparison.

    Args:
        paths: Project layout.

    Returns:
        The artifact.
    """
    return json.loads((paths.reports / BOX_JSON).read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def spatial(paths: ProjectPaths) -> dict[str, Any]:
    """Read the committed spatial comparison.

    Args:
        paths: Project layout.

    Returns:
        The artifact.
    """
    return json.loads((paths.reports / SPATIAL_JSON).read_text(encoding="utf-8"))


def test_both_artifacts_validate(box: dict, spatial: dict) -> None:
    """The committed results pass their own validators."""
    assert (
        validate_box_comparison(
            box,
            protocol_fingerprint=PROTOCOL_SHA,
            detector_sha256=DETECTOR_SHA,
            segmenter_sha256=SEGMENTER_SHA,
            membership_sha256=MEMBERSHIP_SHA,
        )
        == []
    )
    assert (
        validate_spatial_comparison(
            spatial, protocol_fingerprint=PROTOCOL_SHA, membership_sha256=MEMBERSHIP_SHA
        )
        == []
    )


def test_the_frozen_protocol_and_models_are_the_committed_ones(
    paths: ProjectPaths, box: dict
) -> None:
    """Everything is keyed to the phase 10A freeze."""
    protocol = load_comparison_protocol(paths.configs / "detector_segmenter_comparison.yaml")

    assert box["protocol_fingerprint"] == protocol.fingerprint() == PROTOCOL_SHA
    assert box["detector"]["checkpoint_sha256"] == DETECTOR_SHA
    assert box["segmenter"]["checkpoint_sha256"] == SEGMENTER_SHA
    assert box["detector"]["imgsz"] == box["segmenter"]["imgsz"] == 768
    assert box["detector"]["trained_in_this_phase"] is False
    assert box["segmenter"]["modified_in_this_phase"] is False


def test_the_precision_preflight_proved_parity(box: dict) -> None:
    """Runtime evidence, not a configuration value."""
    precision = box["precision_preflight"]

    assert precision["status"] == "EFFECTIVE_PRECISION_PARITY_VERIFIED"
    assert precision["identical_across_models"] is True
    for experiment in ("D2", "S1"):
        evidence = precision["evidence"][experiment]
        assert evidence["backend_fp16_flag"] is False
        assert evidence["parameter_dtypes"] == ["torch.float32"]
        assert evidence["input_dtype"] == "torch.float32"
        assert evidence["autocast_enabled_during_forward"] is False
        assert evidence["quantization_config_present"] is False
    assert precision["evidence"]["D2"] == precision["evidence"]["S1"]


def test_the_two_inference_protocols_kept_their_thresholds(box: dict, spatial: dict) -> None:
    """0.001 for AP, 0.25 for operational, never crossed."""
    assert box["inference"]["conf"] == 0.001
    assert spatial["inference"]["conf"] == 0.25
    assert box["inference"]["iou"] == spatial["inference"]["iou"] == 0.70
    assert box["inference"]["imgsz"] == spatial["inference"]["imgsz"] == 768
    assert box["inference"]["precision"] == spatial["inference"]["precision"] == "FP32"
    assert box["inference"]["augment"] is False
    assert box["inference"]["tta"] is False


def test_the_same_validation_population_was_used(box: dict, spatial: dict) -> None:
    """One membership, both models, no sampling."""
    for payload in (box, spatial):
        assert payload["population"]["split"] == "validation"
        assert payload["population"]["membership_sha256"] == MEMBERSHIP_SHA
        assert payload["population"]["identical_images_for_both_models"] is True
    assert box["population"]["images"] == 65


def test_the_segmenter_boxes_are_its_own(box: dict) -> None:
    """Nothing was derived from a mask to tidy up the box comparison."""
    assert box["segmenter_boxes_derived_from_masks"] is False
    assert box["segmenter_boxes_source"] == "THE_SEGMENTERS_OWN_PREDICTED_BOXES"


def test_the_box_evaluator_is_the_frozen_one(box: dict) -> None:
    """Standard COCO bbox semantics against canonical boxes."""
    evaluator = box["cocoeval"]

    assert evaluator["iou_type"] == "bbox"
    assert evaluator["max_dets"] == [1, 10, 100]
    assert len(evaluator["iou_thresholds"]) == 10
    assert box["ground_truth"] == "CANONICAL_COCO_DETECTION_BOXES"


def test_the_box_deltas_recompute(box: dict) -> None:
    """Every delta is derived, not restated."""
    for name in ("CANONICAL_BOX_MAP50_95", "CANONICAL_BOX_MAP50"):
        block = box["metrics"][name]
        assert block["delta"] == pytest.approx(block["S1"] - block["D2"], abs=1e-9)
    for row in box["per_class"].values():
        assert row["delta_AP@0.50:0.95"] == pytest.approx(
            row["S1_AP@0.50:0.95"] - row["D2_AP@0.50:0.95"], abs=1e-6
        )


def test_the_decomposition_recomputes_from_the_per_class_table(box: dict) -> None:
    """The aggregate is decomposed exactly, and the rare class is separated."""
    decomposition = box["all_class_delta_decomposition"]
    recomputed = decompose_all_class_delta(box["per_class"])

    assert decomposition["contributions"] == recomputed["contributions"]
    assert decomposition["total_contribution"] == pytest.approx(
        box["metrics"]["CANONICAL_BOX_MAP50_95"]["delta"], abs=1e-6
    )
    assert decomposition["rare_class"] == RARE_CLASS
    assert decomposition["delta_excluding_rare_class"] is not None


def test_the_rare_class_carries_its_warning_everywhere(box: dict, spatial: dict) -> None:
    """vest_loose stays descriptive and decides nothing."""
    assert box["per_class"][RARE_CLASS]["status"] == RARE_CLASS_STATUS
    assert box["rare_class"]["status"] == RARE_CLASS_STATUS
    assert spatial["rare_class"]["status"] == RARE_CLASS_STATUS


def test_no_winner_and_no_aggregate_score(box: dict) -> None:
    """A descriptive comparison, and the artifact says so."""
    assert box["winner_declared"] is False
    assert box["aggregate_score"] is False
    assert box["models_trained"] == 0
    assert box["thresholds_tuned"] == 0


def test_exactly_the_seven_frozen_spatial_features(spatial: dict) -> None:
    """No feature added, none dropped, semantics preserved."""
    assert set(spatial["spatial_features"]) == set(SPATIAL_FEATURES)
    assert spatial["new_metrics_introduced"] == 0
    for name, block in spatial["spatial_features"].items():
        assert block["semantics"] == FEATURE_SEMANTICS[name]
    assert (
        spatial["spatial_features"]["VISIBLE_PPE_COVERAGE_PROXY"]["semantics"] == INTERPRETIVE_PROXY
    )
    geometric = [name for name, value in FEATURE_SEMANTICS.items() if value == PURE_GEOMETRIC]
    assert len(geometric) == 6


def test_the_coverage_proxy_keeps_its_limitation(spatial: dict) -> None:
    """The one qualitative definition is flagged as such wherever it appears."""
    coverage = spatial["spatial_features"]["VISIBLE_PPE_COVERAGE_PROXY"]

    assert coverage["definition_is_qualitative_in_the_frozen_protocol"] is True
    assert coverage["is_not_a_compliance_measure"] is True
    assert spatial["compliance_accuracy_claimed"] is False


def test_no_box_proxy_was_invented_where_none_exists(spatial: dict) -> None:
    """The two quantities frozen without an equivalent stay without one."""
    for name in ("MASK_TO_BOX_FILL_RATIO", "SHAPE_EXTENT"):
        block = spatial["box_proxies"][name]
        assert block["proxy"] == NO_BOX_PROXY
        assert block["statistics"] is None


def test_every_other_feature_has_a_paired_comparison(spatial: dict) -> None:
    """Where a proxy exists, both sides are measured on the same instances."""
    for name in (
        "INSTANCE_AREA_PIXELS",
        "PERSON_PPE_MASK_INTERSECTION",
        "PERSON_PPE_MASK_CONTAINMENT",
        "VISIBLE_PPE_COVERAGE_PROXY",
    ):
        block = spatial["box_proxies"][name]
        assert block["proxy"] != NO_BOX_PROXY
        assert set(block["statistics"]) == {"mask_measurement", "box_proxy"}
        assert block["statistics"]["mask_measurement"]["count"] > 0


def test_the_association_used_the_frozen_rule(spatial: dict) -> None:
    """Floor 0.50, deterministic, both box sources labelled."""
    association = spatial["association"]

    assert association["containment_floor"] == 0.50
    assert association["deterministic"] is True
    assert association["geometry_isolating"]["box_source"] == GEOMETRY_ISOLATING
    assert association["pipeline_level"]["box_source"] == PIPELINE_LEVEL


def test_the_frozen_four_survived_the_correction(spatial: dict) -> None:
    """The phase 10A categories were not renamed, extended or reduced."""
    association = spatial["association"]

    assert association["frozen_categories"] == list(ASSOCIATION_CATEGORIES)
    assert association["frozen_categories_unchanged"] is True
    assert association["fifth_peer_category_added"] is False


def test_the_coverage_arithmetic_is_exact(spatial: dict) -> None:
    """Classified plus exceptions equals the total, in every block."""
    for key in ("geometry_isolating", "pipeline_level"):
        block = spatial["association"][key]
        assert block["counts_partition"] is True
        assert (
            block["classified_relationships"] + block["taxonomy_exceptions"]
            == block["total_relationships"]
        )
        assert sum(block["frozen_category_counts"].values()) == block["classified_relationships"]
        assert set(block["frozen_category_counts"]) == set(ASSOCIATION_CATEGORIES)
        assert TAXONOMY_EXCEPTION not in block["frozen_category_counts"]
        # Recorded to six decimals, so compare at that precision.
        assert block["taxonomy_coverage"] == pytest.approx(
            block["classified_relationships"] / block["total_relationships"], abs=1e-6
        )
    for group in ("per_relationship", "per_class"):
        for block in spatial["association"][group].values():
            assert (
                block["classified_relationships"] + block["taxonomy_exceptions"]
                == block["total_relationships"]
            )


def test_the_percentage_denominator_is_stated(spatial: dict) -> None:
    """Percentages are among classified relationships, and say so."""
    for key in ("geometry_isolating", "pipeline_level"):
        block = spatial["association"][key]
        assert block["percentage_denominator"] == "classified_relationships"
        classified = block["classified_relationships"]
        for name, count in block["frozen_category_counts"].items():
            assert block["frozen_category_percentages"][name] == pytest.approx(
                100.0 * count / classified, abs=1e-3
            )


def test_the_exception_is_recorded_as_a_coverage_exception(spatial: dict) -> None:
    """Outside the taxonomy, counted separately, and not called an error."""
    association = spatial["association"]

    assert association["association_taxonomy_status"] == TAXONOMY_NON_EXHAUSTIVE
    assert association["taxonomy_exception_type"] == TAXONOMY_EXCEPTION
    assert association["taxonomy_exception_reading"] == ASSOCIATION_DISAGREEMENT
    assert "was not modified" in association["taxonomy_exception_note"]
    assert "protocol-design limitation" in association["taxonomy_exception_note"]
    assert "invalidates no canonical metric" in association["taxonomy_exception_note"]
    assert "not an association error" in association["not_an_error"]
    for key in ("geometry_isolating", "pipeline_level"):
        exception = spatial["association"][key]["taxonomy_exception"]
        assert exception["is_a_fifth_frozen_category"] is False
        assert exception["status"] == "UNCLASSIFIED_BY_FROZEN_FOUR_CATEGORY_TAXONOMY"


def test_the_two_exception_counts_stay_separate(spatial: dict) -> None:
    """Geometry-isolating and pipeline-level are different questions."""
    geometry = spatial["association"]["geometry_isolating"]
    pipeline = spatial["association"]["pipeline_level"]

    assert geometry["taxonomy_exceptions"] == 1
    assert pipeline["taxonomy_exceptions"] == 17
    assert geometry["classified_relationships"] == 172
    assert pipeline["classified_relationships"] == 156
    assert geometry["total_relationships"] == pipeline["total_relationships"] == 173


def test_pipeline_exceptions_are_not_attributed_to_geometry_alone(spatial: dict) -> None:
    """A different model with different instances is a second explanation."""
    note = spatial["association"]["geometry_isolating_versus_pipeline_level"]

    assert "must not be attributed solely to geometry" in note
    assert "different models produced different instances" in note


def test_the_frozen_counts_survived_the_correction(spatial: dict) -> None:
    """The correction changed presentation, never a measurement."""
    geometry = spatial["association"]["geometry_isolating"]["frozen_category_counts"]
    pipeline = spatial["association"]["pipeline_level"]["frozen_category_counts"]

    assert geometry == {
        "BOX_AND_MASK_AGREE": 103,
        "BOX_ONLY_ASSOCIATION": 3,
        "MASK_ONLY_ASSOCIATION": 0,
        "NEITHER_ASSOCIATION": 66,
    }
    assert pipeline == {
        "BOX_AND_MASK_AGREE": 81,
        "BOX_ONLY_ASSOCIATION": 7,
        "MASK_ONLY_ASSOCIATION": 6,
        "NEITHER_ASSOCIATION": 62,
    }


def test_the_committed_support_sensitivity_is_descriptive(box: dict) -> None:
    """Recorded, exact, and explicitly not a metric or a selection rule."""
    sensitivity = box["supported_class_sensitivity"]
    admitted = sensitivity["admitted_classes"]

    assert RARE_CLASS not in admitted
    assert len(admitted) == 4
    assert sensitivity["D2_supported_macro"] == pytest.approx(
        sum(box["per_class"][name]["D2_AP@0.50:0.95"] for name in admitted) / 4, abs=1e-6
    )
    assert sensitivity["S1_supported_macro"] == pytest.approx(
        sum(box["per_class"][name]["S1_AP@0.50:0.95"] for name in admitted) / 4, abs=1e-6
    )
    assert sensitivity["delta"] == pytest.approx(
        sensitivity["S1_supported_macro"] - sensitivity["D2_supported_macro"], abs=1e-6
    )
    assert sensitivity["delta"] < 0 < box["metrics"]["CANONICAL_BOX_MAP50_95"]["delta"]
    for flag in (
        "is_a_frozen_phase_10a_metric",
        "is_a_selection_rule",
        "is_a_significance_test",
        "changes_any_frozen_number",
    ):
        assert sensitivity[flag] is False


def test_the_localization_conclusion_is_hedged_correctly(box: dict) -> None:
    """The positive aggregate is not presented as robust superiority."""
    conclusion = box["localization_conclusion"]

    assert "broadly similar localisation capability" in conclusion
    assert RARE_CLASS in conclusion
    assert "should not be read as robust evidence" in conclusion
    assert box["rare_class"]["drives_the_aggregate_delta"] is True


def test_the_phase_10a_protocol_was_not_modified(paths: ProjectPaths) -> None:
    """A taxonomy found to be non-exhaustive is a limitation, not a licence."""
    protocol = load_comparison_protocol(paths.configs / "detector_segmenter_comparison.yaml")
    frozen = json.loads(
        (paths.reports / "detector_segmenter_comparison_protocol.json").read_text(encoding="utf-8")
    )

    assert protocol.fingerprint() == PROTOCOL_SHA
    assert tuple(protocol["association"]["disagreement_categories"]) == ASSOCIATION_CATEGORIES
    assert frozen["status"] == "FROZEN_NOT_EXECUTED"
    assert tuple(frozen["association"]["disagreement_categories"]) == ASSOCIATION_CATEGORIES


def test_the_correction_script_runs_no_model(paths: ProjectPaths) -> None:
    """The restructure is arithmetic over recorded counts."""
    source = (paths.root / "scripts" / "correct_association_taxonomy.py").read_text(
        encoding="utf-8"
    )

    for forbidden in ("import torch", "from ultralytics", "import ultralytics", ".predict("):
        assert forbidden not in source, forbidden


def test_the_operational_counts_and_mask_accounting(spatial: dict) -> None:
    """Every segmenter prediction is either measured or explicitly excluded."""
    counts = spatial["prediction_counts"]
    reconstruction = spatial["mask_reconstruction"]

    assert counts["D2"]["total"] > 0
    assert counts["S1"]["total"] > 0
    assert (
        reconstruction["masks_on_original_canvas"] + reconstruction["excluded_predictions"]
        == counts["S1"]["total"]
    )
    assert len(reconstruction["exclusions"]) == reconstruction["excluded_predictions"]


def test_no_post_hoc_bins_and_no_inferential_test(spatial: dict) -> None:
    """Continuous quantities stay continuous; no test was predeclared."""
    assert spatial["post_hoc_bins_created"] is False
    assert spatial["inferential_tests_run"] == 0
    assert set(spatial["statistic_set"]) == set(STATISTIC_NAMES)


def test_no_latency_or_memory_was_measured(box: dict, spatial: dict) -> None:
    """That is phase 10C."""
    assert box["latency_measured"] is False
    assert spatial["latency_measured"] is False
    for payload in (box, spatial):
        text = json.dumps(payload).lower()
        assert "latency_ms" not in text
        assert "peak_memory" not in text


def test_the_result_fingerprints_are_recorded(box: dict, spatial: dict) -> None:
    """Both results carry a deterministic semantic digest."""
    assert len(box["box_comparison_sha256"]) == 64
    assert len(spatial["spatial_comparison_sha256"]) == 64
    assert box["box_comparison_sha256"] != spatial["spatial_comparison_sha256"]


def test_the_box_csv_matches_the_artifact(paths: ProjectPaths, box: dict) -> None:
    """The table and the JSON cannot disagree."""
    with (paths.reports / BOX_CSV).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    all_class = {row["metric"]: row for row in rows if row["scope"] == "all_class"}
    for name in ("CANONICAL_BOX_MAP50_95", "CANONICAL_BOX_MAP50"):
        assert float(all_class[name]["D2"]) == pytest.approx(box["metrics"][name]["D2"])
        assert float(all_class[name]["S1"]) == pytest.approx(box["metrics"][name]["S1"])
        assert float(all_class[name]["delta_S1_minus_D2"]) == pytest.approx(
            box["metrics"][name]["delta"]
        )
    per_class = {row["scope"] for row in rows if row["metric"] == "CANONICAL_BOX_AP50_95"}
    assert per_class == set(box["per_class"])


def test_the_example_manifest_is_deterministic(paths: ProjectPaths) -> None:
    """Selected by a stable ordering of records, not by looking at images."""
    with (paths.reports / EXAMPLES_CSV).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert rows
    assert len(rows) <= 20
    assert {row["category_geometry_isolating"] for row in rows} <= {
        *ASSOCIATION_CATEGORIES,
        "BOTH_ASSOCIATED_DIFFERENT_PERSON",
        TAXONOMY_EXCEPTION,
    }
    assert AGREE not in {row["category_geometry_isolating"] for row in rows}


def test_no_artifact_mentions_the_holdout(paths: ProjectPaths, box: dict, spatial: dict) -> None:
    """Validation only, structurally."""
    for payload in (box, spatial):
        assert payload["test"]["status"] == "PROTECTED_NOT_ACCESSED"
        assert payload["holdout_accessed"] is False
    for name in (BOX_CSV, EXAMPLES_CSV):
        text = (paths.reports / name).read_text(encoding="utf-8")
        assert "test" not in text.lower().replace("latest", "")


def test_the_report_carries_no_sensitive_content_and_states_the_caveats(
    paths: ProjectPaths,
) -> None:
    """The reader is told what was measured and what it may not be read as."""
    text = (paths.reports / REPORT_MD).read_text(encoding="utf-8")

    assert scan_for_sensitive(text) == []
    for fragment in (
        "FROZEN_PROTOCOL",
        "CANONICAL_BOX_EVALUATION",
        "COMPUTED_RESULT",
        "MASK_MEASUREMENT",
        "BOX_PROXY",
        "SPATIAL_INFORMATION_GAIN",
        "OPERATIONAL_PROXY",
        "LIMITATION",
        "HOLDOUT_POLICY",
        "LATENCY_PENDING",
        "must never be differenced",
        "not an across-the-board improvement",
    ):
        assert fragment in text, fragment


def test_the_historical_artifacts_are_untouched(paths: ProjectPaths) -> None:
    """The two freezes and the phase 10A protocol were not regenerated."""
    from construction_safety_vision.provenance import sha256_file

    protocol = json.loads(
        (paths.reports / "detector_segmenter_comparison_protocol.json").read_text(encoding="utf-8")
    )

    assert protocol["status"] == "FROZEN_NOT_EXECUTED"
    assert protocol["models_executed_in_this_phase"] == 0
    for name, expected in protocol["historical_artifact_digests"].items():
        assert sha256_file(paths.root / name) == expected, name
