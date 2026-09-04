"""Tests for the canonical modelling population and its split units.

The invariants here are the ones a split design will silently rely on: that every
eligible image belongs to exactly one group, that a confirmed duplicate pair
cannot be pulled apart, that nothing without geometry is retained, and that the
fingerprint moves when a decision moves and not when a timestamp does.
"""

from __future__ import annotations

import pytest

from construction_safety_vision.data.population import (
    CLASS_ORDER,
    ELIGIBLE,
    EXCLUDE_IMAGE,
    EXCLUDE_WITH_IMAGE,
    EXCLUDED,
    KEEP_ANNOTATION,
    KEEP_IMAGE,
    MATERIALISE_RECTANGLE,
    PROVIDER_GEOMETRY,
    SEMANTIC_DUPLICATE,
    SINGLETON,
    SYNTHETIC_GEOMETRY,
    AnnotationRecord,
    Group,
    ImageRecord,
    PopulationError,
    build_class_map,
    build_groups,
    connected_components,
    fingerprint_population,
    group_features,
    rectangle_polygon,
    validate_population,
)

SQUARE = [[0.0, 0.0, 10.0, 0.0, 10.0, 10.0, 0.0, 10.0]]
"""A trivial polygon used where the geometry itself does not matter."""

_UNSET = object()
"""Sentinel so a test can pass an explicitly empty geometry."""


def image(
    image_id: str, *, status: str = ELIGIBLE, group_id: str = "", annotations: int = 1
) -> ImageRecord:
    """Build an image record.

    Args:
        image_id: Source image identifier.
        status: Eligibility.
        group_id: Owning group.
        annotations: Canonical annotation count.

    Returns:
        The record.
    """
    return ImageRecord(
        source_image_id=image_id,
        name=f"{image_id}.jpg",
        width=100,
        height=100,
        status=status,
        action=KEEP_IMAGE if status == ELIGIBLE else EXCLUDE_IMAGE,
        reason="TEST",
        decision_source="TEST",
        annotation_count=annotations,
        eligible_annotation_count=annotations if status == ELIGIBLE else 0,
        group_id=group_id,
    )


def annotation(
    image_id: str,
    annotation_id: str = "1",
    *,
    label: str = "person",
    action: str = KEEP_ANNOTATION,
    segmentation: object = _UNSET,
    area: float | None = 100.0,
) -> AnnotationRecord:
    """Build an annotation record.

    Args:
        image_id: Owning image.
        annotation_id: Annotation identifier.
        label: Class name.
        action: Modelling action.
        segmentation: Canonical segmentation.
        area: Mask area.

    Returns:
        The record.
    """
    return AnnotationRecord(
        source_image_id=image_id,
        annotation_id=annotation_id,
        label=label,
        geometry_kind="POLYGON",
        action=action,
        reason="TEST",
        decision_source="TEST",
        geometry_origin=PROVIDER_GEOMETRY,
        segmentation=SQUARE if segmentation is _UNSET else segmentation,
        bbox=[0.0, 0.0, 10.0, 10.0],
        area=area,
    )


# --- class map ---------------------------------------------------------------


def test_class_map_is_explicit_and_excludes_the_placeholder_category() -> None:
    mapping = build_class_map()
    assert mapping == {
        "helmet_loose": 0,
        "helmet_on_head": 1,
        "person": 2,
        "vest_loose": 3,
        "vest_on_body": 4,
    }
    assert "object" not in mapping


def test_class_map_indices_do_not_depend_on_iteration_order() -> None:
    # The index must come from the declared order, never from a dict or from the
    # provider's category ids.
    assert list(build_class_map()) == list(CLASS_ORDER)


def test_class_map_rejects_a_duplicated_class() -> None:
    with pytest.raises(PopulationError, match="duplicate"):
        build_class_map(("person", "person"))


# --- synthetic rectangles ----------------------------------------------------


def test_rectangle_is_a_closed_four_corner_ring() -> None:
    ring = rectangle_polygon([10.0, 20.0, 30.0, 40.0], width=200, height=200)[0]
    assert len(ring) == 8
    corners = list(zip(ring[0::2], ring[1::2], strict=True))
    assert corners == [(10.0, 20.0), (40.0, 20.0), (40.0, 60.0), (10.0, 60.0)]
    assert len(set(corners)) == 4


def test_rectangle_is_clipped_to_the_canvas() -> None:
    ring = rectangle_polygon([-5.0, -5.0, 40.0, 40.0], width=20, height=20)[0]
    assert min(ring[0::2]) >= 0.0
    assert min(ring[1::2]) >= 0.0
    assert max(ring[0::2]) <= 20.0
    assert max(ring[1::2]) <= 20.0


def test_rectangle_matches_the_area_of_its_own_box() -> None:
    ring = rectangle_polygon([0.0, 0.0, 7.9, 6.0], width=1600, height=1200)[0]
    width = max(ring[0::2]) - min(ring[0::2])
    height = max(ring[1::2]) - min(ring[1::2])
    assert width * height == pytest.approx(7.9 * 6.0)


def test_rectangle_refuses_a_box_outside_the_canvas() -> None:
    with pytest.raises(PopulationError, match="no area inside"):
        rectangle_polygon([500.0, 500.0, 10.0, 10.0], width=100, height=100)


# --- grouping ----------------------------------------------------------------


def test_every_eligible_image_lands_in_exactly_one_group() -> None:
    groups = build_groups(["a", "b", "c", "d"], {"manual_dup_001": ["a", "b"]})
    members = [member for group in groups for member in group.members]
    assert sorted(members) == ["a", "b", "c", "d"]
    assert len(members) == len(set(members))


def test_a_confirmed_duplicate_pair_forms_one_indivisible_group() -> None:
    groups = build_groups(["a", "b", "c"], {"manual_dup_001": ["a", "b"]})
    duplicate = next(g for g in groups if g.group_type == SEMANTIC_DUPLICATE)
    assert duplicate.members == ["a", "b"]
    assert duplicate.split_indivisible


def test_unrelated_images_become_singletons() -> None:
    groups = build_groups(["a", "b", "c"], {"manual_dup_001": ["a", "b"]})
    singletons = [g for g in groups if g.group_type == SINGLETON]
    assert [g.members for g in singletons] == [["c"]]


def test_group_ids_are_deterministic_and_independent_of_input_order() -> None:
    first = build_groups(["c", "a", "b"], {"manual_dup_001": ["b", "a"]})
    second = build_groups(["a", "b", "c"], {"manual_dup_001": ["a", "b"]})
    assert [(g.group_id, g.members) for g in first] == [(g.group_id, g.members) for g in second]


def test_a_singleton_group_id_is_derived_from_its_image() -> None:
    groups = build_groups(["abc"], {})
    assert groups[0].group_id == "singleton-abc"


def test_a_degenerate_confirmed_group_is_rejected() -> None:
    with pytest.raises(PopulationError, match="need >= 2"):
        build_groups(["a"], {"manual_dup_001": ["a"]})


def test_a_confirmed_group_naming_an_excluded_image_is_rejected() -> None:
    with pytest.raises(PopulationError, match="not in the modelling population"):
        build_groups(["a"], {"manual_dup_001": ["a", "excluded"]})


def test_two_declarations_sharing_an_image_merge_rather_than_conflict() -> None:
    # Superseded behaviour: this used to raise. Sharing an image means the two
    # declarations describe one group, which is what connected components give.
    groups = build_groups(["a", "b", "c"], {"g1": ["a", "b"], "g2": ["a", "c"]})
    duplicates = [g for g in groups if g.group_type == SEMANTIC_DUPLICATE]
    assert len(duplicates) == 1
    assert duplicates[0].members == ["a", "b", "c"]


# --- group features ----------------------------------------------------------


def test_group_features_count_instances_across_all_members() -> None:
    group = Group("g", ["a", "b"], SEMANTIC_DUPLICATE, "TEST")
    row = group_features(
        group,
        {
            "a": [annotation("a", "1"), annotation("a", "2", label="vest_loose")],
            "b": [annotation("b", "1")],
        },
    )
    assert row["image_count"] == 2
    assert row["instances_person"] == 2
    assert row["instances_vest_loose"] == 1
    assert row["has_vest_loose"] == "true"
    assert row["has_helmet_loose"] == "false"
    assert row["total_instances"] == 3


def test_group_features_count_zero_instance_members() -> None:
    group = Group("g", ["a", "b"], SEMANTIC_DUPLICATE, "TEST")
    row = group_features(group, {"a": [annotation("a")]})
    assert row["zero_instance_image_count"] == 1


def test_group_features_carry_no_provider_split() -> None:
    # The provider split was rejected; leaking it here would let it bias a split
    # optimiser that has no reason to consult it.
    row = group_features(Group("g", ["a"], SINGLETON, "TEST"), {"a": [annotation("a")]})
    assert not [key for key in row if "split" in key.lower()]


# --- validation --------------------------------------------------------------


def test_a_sound_population_reports_no_problem() -> None:
    images = [image("a", group_id="singleton-a"), image("b", group_id="singleton-b")]
    annotations = [annotation("a"), annotation("b")]
    groups = build_groups(["a", "b"], {})
    assert validate_population(images, annotations, groups) == []


def test_an_eligible_image_with_no_group_is_reported() -> None:
    images = [image("a"), image("b")]
    groups = build_groups(["a"], {})
    problems = validate_population(images, [annotation("a")], groups)
    assert any("belong to no group" in problem for problem in problems)


def test_an_image_in_two_groups_is_reported() -> None:
    images = [image("a")]
    groups = [Group("g1", ["a"], SINGLETON, "TEST"), Group("g2", ["a"], SINGLETON, "TEST")]
    problems = validate_population(images, [annotation("a")], groups)
    assert any("belongs to groups" in problem for problem in problems)


def test_a_retained_annotation_without_geometry_is_reported() -> None:
    # The invariant that stops an UNSUPPORTED record slipping through unnoticed.
    images = [image("a", group_id="singleton-a")]
    groups = build_groups(["a"], {})
    problems = validate_population(images, [annotation("a", segmentation=None)], groups)
    assert any("no segmentation geometry" in problem for problem in problems)


def test_a_retained_annotation_with_an_unknown_class_is_reported() -> None:
    images = [image("a", group_id="singleton-a")]
    groups = build_groups(["a"], {})
    problems = validate_population(images, [annotation("a", label="object")], groups)
    assert any("not in the canonical class set" in problem for problem in problems)


def test_a_retained_annotation_with_no_area_is_reported() -> None:
    images = [image("a", group_id="singleton-a")]
    groups = build_groups(["a"], {})
    problems = validate_population(images, [annotation("a", area=0.0)], groups)
    assert any("non-positive area" in problem for problem in problems)


def test_an_annotation_excluded_with_its_image_is_not_required_to_be_sound() -> None:
    images = [image("a", group_id="singleton-a"), image("b", status=EXCLUDED)]
    groups = build_groups(["a"], {})
    excluded = annotation("b", action=EXCLUDE_WITH_IMAGE, segmentation=False)
    assert validate_population(images, [annotation("a"), excluded], groups) == []


def test_a_materialised_rectangle_counts_as_eligible_geometry() -> None:
    images = [image("a", group_id="singleton-a")]
    groups = build_groups(["a"], {})
    record = annotation("a", action=MATERIALISE_RECTANGLE)
    object.__setattr__(record, "geometry_origin", SYNTHETIC_GEOMETRY)
    assert record.is_eligible
    assert validate_population(images, [record], groups) == []


# --- fingerprint -------------------------------------------------------------


def _population() -> tuple[list[ImageRecord], list[AnnotationRecord], list[Group], dict[str, int]]:
    """Build a small sound population.

    Returns:
        Its images, annotations, groups and class map.
    """
    images = [image("a", group_id="singleton-a"), image("b", group_id="singleton-b")]
    annotations = [annotation("a"), annotation("b")]
    groups = build_groups(["a", "b"], {})
    return images, annotations, groups, build_class_map()


def test_the_fingerprint_is_stable_across_runs() -> None:
    first = fingerprint_population(*_population())
    second = fingerprint_population(*_population())
    assert first == second
    assert len(first["modeling_population_sha256"]) == 64


def test_the_fingerprint_does_not_depend_on_record_order() -> None:
    images, annotations, groups, class_map = _population()
    baseline = fingerprint_population(images, annotations, groups, class_map)
    shuffled = fingerprint_population(
        list(reversed(images)), list(reversed(annotations)), list(reversed(groups)), class_map
    )
    assert baseline == shuffled


def test_changing_an_image_decision_changes_the_fingerprint() -> None:
    images, annotations, groups, class_map = _population()
    baseline = fingerprint_population(images, annotations, groups, class_map)
    images[0].status = EXCLUDED
    assert fingerprint_population(images, annotations, groups, class_map) != baseline


def test_changing_annotation_geometry_changes_the_fingerprint() -> None:
    images, annotations, groups, class_map = _population()
    baseline = fingerprint_population(images, annotations, groups, class_map)
    annotations[0] = annotation("a", segmentation=[[1.0, 1.0, 5.0, 1.0, 5.0, 5.0, 1.0, 5.0]])
    assert fingerprint_population(images, annotations, groups, class_map) != baseline


def test_changing_group_membership_changes_the_fingerprint() -> None:
    images, annotations, groups, class_map = _population()
    baseline = fingerprint_population(images, annotations, groups, class_map)
    regrouped = build_groups(["a", "b"], {"manual_dup_001": ["a", "b"]})
    assert fingerprint_population(images, annotations, regrouped, class_map) != baseline


def test_changing_the_class_map_changes_the_fingerprint() -> None:
    images, annotations, groups, class_map = _population()
    baseline = fingerprint_population(images, annotations, groups, class_map)
    reordered = build_class_map(tuple(reversed(CLASS_ORDER)))
    assert fingerprint_population(images, annotations, groups, reordered) != baseline


# --- connected components (phase 5B.1) ---------------------------------------


def test_disjoint_relations_stay_separate_groups() -> None:
    components = connected_components({"g1": ["a", "b"], "g2": ["c", "d"]})
    assert components == [("g1", ["a", "b"]), ("g2", ["c", "d"])]


def test_transitive_relations_merge_into_one_component() -> None:
    # A~B and B~C means one group {A, B, C}. Two overlapping pairs would let a
    # split separate A from C while honouring each relation individually.
    components = connected_components({"g1": ["a", "b"], "g2": ["b", "c"]})
    assert components == [("g1", ["a", "b", "c"])]


def test_a_merged_component_keeps_the_lowest_declared_id() -> None:
    components = connected_components({"g9": ["b", "c"], "g2": ["a", "b"]})
    assert components == [("g2", ["a", "b", "c"])]


def test_a_long_chain_collapses_to_a_single_component() -> None:
    components = connected_components(
        {"g1": ["a", "b"], "g2": ["b", "c"], "g3": ["c", "d"], "g4": ["d", "e"]}
    )
    assert components == [("g1", ["a", "b", "c", "d", "e"])]


def test_components_are_independent_of_declaration_order() -> None:
    forward = connected_components({"g1": ["a", "b"], "g2": ["b", "c"], "g3": ["x", "y"]})
    backward = connected_components({"g3": ["y", "x"], "g2": ["c", "b"], "g1": ["b", "a"]})
    assert forward == backward


def test_transitive_relations_produce_one_group_not_two() -> None:
    groups = build_groups(["a", "b", "c", "d"], {"g1": ["a", "b"], "g2": ["b", "c"]})
    duplicates = [g for g in groups if g.group_type == SEMANTIC_DUPLICATE]
    assert len(duplicates) == 1
    assert duplicates[0].members == ["a", "b", "c"]
    members = [m for g in groups for m in g.members]
    assert sorted(members) == ["a", "b", "c", "d"]
    assert len(members) == len(set(members))


def test_a_transitively_merged_group_passes_validation() -> None:
    images = [
        image("a", group_id="g1"),
        image("b", group_id="g1"),
        image("c", group_id="g1"),
        image("d", group_id="singleton-d"),
    ]
    groups = build_groups(["a", "b", "c", "d"], {"g1": ["a", "b"], "g2": ["b", "c"]})
    annotations = [annotation(name) for name in ("a", "b", "c", "d")]
    assert validate_population(images, annotations, groups) == []


def test_group_features_aggregate_a_three_member_component() -> None:
    groups = build_groups(["a", "b", "c"], {"g1": ["a", "b"], "g2": ["b", "c"]})
    row = group_features(
        groups[0],
        {"a": [annotation("a")], "b": [annotation("b", label="vest_loose")], "c": []},
    )
    assert row["image_count"] == 3
    assert row["zero_instance_image_count"] == 1
    assert row["instances_person"] == 1
    assert row["instances_vest_loose"] == 1


def test_confirming_a_duplicate_pair_changes_the_group_fingerprint() -> None:
    images, annotations, groups, class_map = _population()
    baseline = fingerprint_population(images, annotations, groups, class_map)
    merged = build_groups(["a", "b"], {"manual_dup_007": ["a", "b"]})
    after = fingerprint_population(images, annotations, merged, class_map)
    assert after["groups_sha256"] != baseline["groups_sha256"]
    assert after["modeling_population_sha256"] != baseline["modeling_population_sha256"]
    # No annotation and no class index moved, so those digests must not move.
    assert after["annotations_sha256"] == baseline["annotations_sha256"]
    assert after["class_map_sha256"] == baseline["class_map_sha256"]


def test_a_group_records_the_verdict_behind_it() -> None:
    groups = build_groups(
        ["a", "b", "c"],
        {"g1": ["a", "b"]},
        {"g1": "NEAR_DUPLICATE_SAME_SCENE"},
    )
    duplicate = next(g for g in groups if g.group_type == SEMANTIC_DUPLICATE)
    assert duplicate.basis == "NEAR_DUPLICATE_SAME_SCENE"
    assert duplicate.csv_rows()[0]["group_basis"] == "NEAR_DUPLICATE_SAME_SCENE"


def test_a_merged_component_reports_every_verdict_that_produced_it() -> None:
    # A component built from two different findings must not claim only one.
    groups = build_groups(
        ["a", "b", "c"],
        {"g1": ["a", "b"], "g2": ["b", "c"]},
        {"g1": "EXACT_SEMANTIC_DUPLICATE", "g2": "NEAR_DUPLICATE_SAME_SCENE"},
    )
    duplicate = next(g for g in groups if g.group_type == SEMANTIC_DUPLICATE)
    assert duplicate.members == ["a", "b", "c"]
    assert duplicate.basis == "EXACT_SEMANTIC_DUPLICATE|NEAR_DUPLICATE_SAME_SCENE"


def test_a_group_without_a_declared_basis_is_left_empty() -> None:
    groups = build_groups(["a", "b"], {"g1": ["a", "b"]})
    duplicate = next(g for g in groups if g.group_type == SEMANTIC_DUPLICATE)
    assert duplicate.basis == ""


def test_a_near_duplicate_group_is_as_indivisible_as_an_exact_one() -> None:
    groups = build_groups(["a", "b"], {"g1": ["a", "b"]}, {"g1": "NEAR_DUPLICATE_SAME_SCENE"})
    duplicate = next(g for g in groups if g.group_type == SEMANTIC_DUPLICATE)
    assert duplicate.split_indivisible
    assert duplicate.csv_rows()[0]["split_indivisible"] == "true"
