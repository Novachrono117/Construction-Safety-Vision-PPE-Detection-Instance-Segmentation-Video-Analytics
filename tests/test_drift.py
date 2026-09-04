"""Tests for measuring annotation drift between the two annotation states.

The property that matters most here is that a net count difference never stands
in for the amount of change: an image that gains one annotation and loses another
nets to zero and must still be reported as changed in both directions.
"""

from __future__ import annotations

import pytest

from construction_safety_vision.data.drift import (
    ADDED_AND_REMOVED,
    ADDED_ONLY,
    MODIFIED,
    REMOVED_ONLY,
    UNCHANGED,
    DriftInstance,
    bbox_iou,
    compare_image,
    containment,
    describe_addition,
    match_instances,
    normalise_bbox,
    summarise,
)


def instance(key: str, label: str, box: tuple[float, float, float, float]) -> DriftInstance:
    """Build a comparison instance.

    Args:
        key: Identifier.
        label: Class name.
        box: Normalised box.

    Returns:
        The instance.
    """
    return DriftInstance(key=key, label=label, bbox=box)


PERSON = (0.1, 0.1, 0.4, 0.6)
"""A large person box in the unit square."""

VEST = (0.2, 0.3, 0.2, 0.2)
"""A vest box lying inside the person box."""


def test_normalise_bbox_is_invariant_to_a_stretch_resize() -> None:
    # The export stretches every image to 640x640. A normalised box is the same
    # on both sides, which is why no coordinate inversion is needed.
    original = normalise_bbox([100, 200, 50, 80], width=1000, height=2000)
    resized = normalise_bbox([64, 64, 32, 25.6], width=640, height=640)
    assert original == pytest.approx(resized, abs=1e-6)


def test_normalise_bbox_rejects_a_degenerate_canvas() -> None:
    with pytest.raises(ValueError, match="Cannot normalise"):
        normalise_bbox([1, 2, 3, 4], width=0, height=10)


def test_bbox_iou_is_one_for_identical_boxes_and_zero_when_disjoint() -> None:
    assert bbox_iou(PERSON, PERSON) == pytest.approx(1.0)
    assert bbox_iou((0.0, 0.0, 0.1, 0.1), (0.5, 0.5, 0.1, 0.1)) == 0.0


def test_containment_is_asymmetric() -> None:
    # A small box wholly inside a large one is fully contained, but not the
    # reverse. IoU would report both as small and hide the relationship.
    assert containment(VEST, PERSON) == pytest.approx(1.0)
    assert containment(PERSON, VEST) < 0.2


def test_identical_annotations_are_unchanged() -> None:
    current = [instance("1", "person", PERSON)]
    drift = compare_image("img", "train", current, list(current))
    assert drift.match_status == UNCHANGED
    assert drift.delta == 0
    assert not drift.added
    assert not drift.removed


def test_an_extra_annotation_is_reported_as_added() -> None:
    drift = compare_image(
        "img",
        "train",
        [instance("1", "person", PERSON), instance("2", "vest_on_body", VEST)],
        [instance("a", "person", PERSON)],
    )
    assert drift.match_status == ADDED_ONLY
    assert drift.delta == 1
    assert drift.added_class_counts == {"vest_on_body": 1}


def test_a_missing_annotation_is_reported_as_removed() -> None:
    drift = compare_image(
        "img",
        "train",
        [instance("1", "person", PERSON)],
        [instance("a", "person", PERSON), instance("b", "vest_on_body", VEST)],
    )
    assert drift.match_status == REMOVED_ONLY
    assert drift.delta == -1
    assert drift.removed_class_counts == {"vest_on_body": 1}


def test_a_zero_net_delta_still_reports_both_directions() -> None:
    # The whole reason additions and removals are counted separately.
    elsewhere = (0.7, 0.7, 0.2, 0.2)
    drift = compare_image(
        "img",
        "train",
        [instance("1", "person", PERSON), instance("2", "vest_on_body", elsewhere)],
        [instance("a", "person", PERSON), instance("b", "vest_on_body", VEST)],
    )
    assert drift.delta == 0
    assert drift.match_status == ADDED_AND_REMOVED
    assert len(drift.added) == 1
    assert len(drift.removed) == 1


def test_a_relabelled_object_is_matched_and_reported_not_counted_twice() -> None:
    drift = compare_image(
        "img", "train", [instance("1", "helmet_loose", PERSON)], [instance("a", "person", PERSON)]
    )
    assert drift.relabelled == [("person", "helmet_loose")]
    assert not drift.added
    assert not drift.removed
    assert drift.match_status == MODIFIED


def test_same_class_pairs_are_matched_before_cross_class_ones() -> None:
    # Without the two-pass order a greedy cross-class match could pair the person
    # with the vest and invent a relabel that did not happen.
    current = [instance("1", "person", PERSON), instance("2", "vest_on_body", VEST)]
    snapshot = [instance("a", "vest_on_body", VEST), instance("b", "person", PERSON)]
    matches, added, removed = match_instances(current, snapshot)
    assert not added and not removed
    assert all(c.label == s.label for c, s, _ in matches)


def test_a_shifted_object_is_matched_but_flagged_as_reshaped() -> None:
    moved = (0.12, 0.12, 0.4, 0.6)
    drift = compare_image(
        "img", "train", [instance("1", "person", moved)], [instance("a", "person", PERSON)]
    )
    assert not drift.added and not drift.removed
    assert drift.reshaped == 1
    assert drift.match_status == MODIFIED


def test_describe_addition_detects_a_fragment_of_an_existing_object() -> None:
    fragment = (0.15, 0.15, 0.05, 0.05)
    enclosed, ratio = describe_addition(
        instance("x", "person", fragment), [instance("a", "person", PERSON)]
    )
    assert enclosed == pytest.approx(1.0)
    assert ratio < 0.05


def test_describe_addition_reports_no_container_when_the_class_is_new() -> None:
    enclosed, ratio = describe_addition(
        instance("x", "vest_loose", VEST), [instance("a", "person", PERSON)]
    )
    assert enclosed == 0.0
    assert ratio == 1.0


def test_summarise_separates_gross_change_from_net_change() -> None:
    elsewhere = (0.7, 0.7, 0.2, 0.2)
    drifts = [
        compare_image(
            "a",
            "train",
            [instance("1", "person", PERSON), instance("2", "vest_on_body", elsewhere)],
            [instance("a", "person", PERSON), instance("b", "vest_on_body", VEST)],
        ),
        compare_image("b", "valid", [instance("1", "person", PERSON)], []),
    ]
    summary = summarise(drifts)
    assert summary["net_delta"] == 1
    assert summary["gross_added"] == 2
    assert summary["gross_removed"] == 1
    assert summary["images_compared"] == 2
    assert summary["by_provider_split"]["valid"]["added"] == 1


def test_summarise_reports_overlap_against_object_size() -> None:
    # The evidence that separates rasterisation noise from real reshaping.
    drift = compare_image(
        "a",
        "train",
        [instance("1", "person", (0.12, 0.12, 0.4, 0.6))],
        [instance("a", "person", PERSON)],
    )
    summary = summarise([drift])
    assert summary["matched_pairs"] == 1
    assert summary["iou_by_object_size"]
    assert set(summary["iou_below"]) == {"0.99", "0.95", "0.90", "0.80", "0.70", "0.50"}


def test_summarise_counts_addition_placement() -> None:
    fragment = (0.15, 0.15, 0.05, 0.05)
    drift = compare_image(
        "a",
        "train",
        [instance("1", "person", PERSON), instance("2", "person", fragment)],
        [instance("a", "person", PERSON)],
    )
    placement = summarise([drift])["addition_placement"]
    assert placement["additions"] == 1
    assert placement["inside_same_class_annotation"] == 1
    assert placement["covering_a_new_object"] == 0
