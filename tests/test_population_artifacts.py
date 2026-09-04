"""Checks on the phase 5B artifacts that are actually committed.

These read the real files a phase 5C split designer will consume. The arithmetic
must reconcile across them, the provider's rejected split must not have leaked
into the inputs a split optimiser reads, and nothing unpublishable may have crept
into a report.

Each test skips when its artifact is absent, so a fresh clone that has not run
the phase 5B scripts is not reported as broken.
"""

from __future__ import annotations

import csv
import json

import pytest

from construction_safety_vision.data.canonical import scan_for_sensitive
from construction_safety_vision.data.population import (
    CLASS_ORDER,
    ELIGIBLE,
    EXCLUDED,
    MATERIALISE_RECTANGLE,
    SEMANTIC_DUPLICATE,
    SINGLETON,
    SYNTHETIC_GEOMETRY,
)
from construction_safety_vision.paths import ProjectPaths

SOURCE_POPULATION = 436
"""The independent source population every phase must preserve."""

CONFIRMED_DUPLICATE_GROUPS = 6
"""Semantic duplicate groups phase 4B confirmed."""

REPORTS = ProjectPaths.from_root().reports
"""Directory holding the committed evidence."""


def rows(name: str) -> list[dict]:
    """Load a committed CSV artifact, skipping the test when it is absent.

    Args:
        name: Filename under ``reports``.

    Returns:
        The parsed rows.
    """
    path = REPORTS / name
    if not path.is_file():
        pytest.skip(f"{name} not generated yet")
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def manifest() -> dict:
    """Load the committed modelling manifest, skipping when absent.

    Returns:
        The parsed manifest.
    """
    path = REPORTS / "canonical_modeling_manifest.json"
    if not path.is_file():
        pytest.skip("canonical_modeling_manifest.json not generated yet")
    return json.loads(path.read_text(encoding="utf-8"))


def test_the_source_population_is_preserved_in_full() -> None:
    population = rows("canonical_modeling_population.csv")
    assert len(population) == SOURCE_POPULATION
    assert len({row["source_image_id"] for row in population}) == SOURCE_POPULATION


def test_excluded_images_are_marked_not_deleted() -> None:
    population = rows("canonical_modeling_population.csv")
    excluded = [row for row in population if row["status"] == EXCLUDED]
    assert excluded, "expected the out-of-domain exclusions to be present"
    assert all(row["reason"] == "OUT_OF_DOMAIN_MANUAL_REVIEW" for row in excluded)
    assert all(row["decision_source"] == "PHASE_4B_HUMAN_AUDIT" for row in excluded)


def test_the_modelling_population_reconciles_with_the_manifest() -> None:
    population = rows("canonical_modeling_population.csv")
    summary = manifest()["population"]
    eligible = [row for row in population if row["status"] == ELIGIBLE]
    assert len(eligible) == summary["modeling_image_population"]
    assert len(population) - len(eligible) == summary["excluded_images"]
    assert len(population) == summary["source_provenance_population"]


def test_zero_instance_images_are_retained_rather_than_excluded() -> None:
    population = rows("canonical_modeling_population.csv")
    retained_empty = [
        row for row in population if row["status"] == ELIGIBLE and row["zero_instance"] == "true"
    ]
    assert retained_empty, "zero-instance images are useful negatives and must be retained"
    assert all(row["reason"] != "OUT_OF_DOMAIN_MANUAL_REVIEW" for row in retained_empty)


def test_annotation_actions_cover_every_canonical_annotation() -> None:
    actions = rows("canonical_annotation_actions.csv")
    summary = manifest()["population"]
    assert len(actions) == summary["canonical_annotations"]
    eligible = [row for row in actions if row["action"] in ("KEEP", MATERIALISE_RECTANGLE)]
    assert len(eligible) == summary["modeling_annotation_population"]


def test_no_retained_annotation_is_unsupported() -> None:
    actions = rows("canonical_annotation_actions.csv")
    retained = [row for row in actions if row["action"] in ("KEEP", MATERIALISE_RECTANGLE)]
    assert all(
        row["geometry_kind"] != "UNSUPPORTED" or row["action"] == MATERIALISE_RECTANGLE
        for row in retained
    )
    unsupported_kept = [
        row for row in retained if row["geometry_kind"] == "UNSUPPORTED" and row["action"] == "KEEP"
    ]
    assert not unsupported_kept


def test_geometry_less_annotations_are_materialised_and_labelled_synthetic() -> None:
    actions = rows("canonical_annotation_actions.csv")
    synthetic = [row for row in actions if row["action"] == MATERIALISE_RECTANGLE]
    assert synthetic, "expected the geometry-less records to be materialised"
    assert all(row["geometry_origin"] == SYNTHETIC_GEOMETRY for row in synthetic)
    assert all(row["reason"] == "SOURCE_GEOMETRY_UNAVAILABLE" for row in synthetic)
    assert len(synthetic) == manifest()["population"]["annotations_materialised_from_bbox"]


def test_automatic_fragment_filtering_was_rejected_and_nothing_dropped() -> None:
    # The owner rejected automatic filtering; every canonical annotation stays.
    summary = manifest()["population"]
    handling = manifest()["nested_annotation_handling"]
    assert summary["annotations_excluded_as_fragments"] == 0
    assert handling["applied"] is False
    assert handling["fragment_rule_status"] == "REJECTED_FOR_AUTOMATIC_FILTERING"
    assert handling["annotations_excluded"] == 0


def test_nested_candidates_are_flagged_descriptively_and_retained() -> None:
    actions = rows("canonical_annotation_actions.csv")
    flagged = [row for row in actions if row["review_flag"] == "NESTED_SAME_CLASS_CANDIDATE"]
    assert flagged, "expected the nested candidates to be flagged"
    assert all(row["action"] in ("KEEP", MATERIALISE_RECTANGLE) for row in flagged)
    # One action row per annotation: a flag must never become a second, conflicting action.
    keys = [(row["source_image_id"], row["annotation_id"]) for row in actions]
    assert len(keys) == len(set(keys))


def test_no_artifact_calls_a_nested_annotation_a_fragment() -> None:
    # Neutral terminology: nothing has been established to be a fragment.
    actions = rows("canonical_annotation_actions.csv")
    assert not [row for row in actions if "FRAGMENT" in row["review_flag"].upper()]


def test_the_retained_population_is_the_full_canonical_snapshot() -> None:
    summary = manifest()["population"]
    assert summary["modeling_annotation_population"] == summary["canonical_annotations"]


def test_the_phase_5c_entry_gate_is_recorded_and_open() -> None:
    gates = manifest()["phase_5c_entry_gates"]
    names = {gate["gate"] for gate in gates}
    assert "MANUAL_DISPOSITION_OF_REMAINING_NEAR_DUPLICATE_CANDIDATES" in names
    gate = next(g for g in gates if g["gate"].startswith("MANUAL_DISPOSITION"))
    assert gate["status"] == "OPEN"
    assert gate["candidates"] == len(rows("unconfirmed_group_candidates.csv"))


def test_the_phase_is_classified_ready_for_split_design() -> None:
    assert manifest()["phase_5b_classification"] == "READY_FOR_SPLIT_DESIGN"


def test_every_eligible_image_belongs_to_exactly_one_group() -> None:
    population = rows("canonical_modeling_population.csv")
    groups = rows("group_manifest.csv")
    eligible = {row["source_image_id"] for row in population if row["status"] == ELIGIBLE}
    members = [row["source_image_id"] for row in groups]
    assert sorted(members) == sorted(eligible)
    assert len(members) == len(set(members))


def test_excluded_images_appear_in_no_group() -> None:
    population = rows("canonical_modeling_population.csv")
    groups = rows("group_manifest.csv")
    excluded = {row["source_image_id"] for row in population if row["status"] == EXCLUDED}
    assert not (excluded & {row["source_image_id"] for row in groups})


def test_the_six_confirmed_duplicate_groups_are_represented() -> None:
    groups = rows("group_manifest.csv")
    duplicates: dict[str, list[str]] = {}
    for row in groups:
        if row["group_type"] == SEMANTIC_DUPLICATE:
            duplicates.setdefault(row["group_id"], []).append(row["source_image_id"])
    assert len(duplicates) == CONFIRMED_DUPLICATE_GROUPS
    assert all(len(members) >= 2 for members in duplicates.values())
    assert all(row["split_indivisible"] == "true" for row in groups)


def test_singleton_groups_hold_exactly_one_image() -> None:
    groups = rows("group_manifest.csv")
    sizes: dict[str, int] = {}
    types: dict[str, str] = {}
    for row in groups:
        sizes[row["group_id"]] = sizes.get(row["group_id"], 0) + 1
        types[row["group_id"]] = row["group_type"]
    assert all(sizes[gid] == 1 for gid, kind in types.items() if kind == SINGLETON)


def test_group_features_have_one_row_per_group() -> None:
    groups = rows("group_manifest.csv")
    features = rows("group_split_features.csv")
    assert len(features) == len({row["group_id"] for row in groups})
    assert {row["group_id"] for row in features} == {row["group_id"] for row in groups}


def test_group_features_carry_no_provider_split_column() -> None:
    # The provider split was formally rejected. A split optimiser must not be
    # able to read it, even accidentally.
    features = rows("group_split_features.csv")
    forbidden = {"split", "provider_split", "original_split"}
    assert not (forbidden & set(features[0]))


def test_group_feature_instance_counts_reconcile_with_the_manifest() -> None:
    features = rows("group_split_features.csv")
    by_class = manifest()["population"]["annotations_by_class"]
    for name in CLASS_ORDER:
        total = sum(int(row[f"instances_{name}"]) for row in features)
        assert total == by_class[name], f"class {name} does not reconcile"


def test_no_artifact_assigns_a_split() -> None:
    features = rows("group_split_features.csv")
    groups = rows("group_manifest.csv")
    population = rows("canonical_modeling_population.csv")
    for table in (features, groups, population):
        columns = set(table[0])
        assert not {"train", "val", "validation", "test"} & columns
    assert manifest()["split_assignment_created"] is False
    assert manifest()["holdout_frozen"] is False


def test_unconfirmed_candidates_are_recorded_and_not_merged() -> None:
    candidates = rows("unconfirmed_group_candidates.csv")
    groups = rows("group_manifest.csv")
    assert all(row["action"] == "NOT_MERGED" for row in candidates)
    assert all(row["status"] == "UNCONFIRMED_GROUP_CANDIDATE" for row in candidates)
    duplicate_ids = {row["group_id"] for row in groups if row["group_type"] == SEMANTIC_DUPLICATE}
    assert not (duplicate_ids & {row["candidate_id"] for row in candidates})


def test_the_rare_class_survived_every_action() -> None:
    rare = manifest()["rare_class"]
    assert rare["class"] == "vest_loose"
    assert rare["retained_instances"] == rare["canonical_instances"]
    assert rare["instances_lost_to_any_exclusion"] == 0
    assert rare["affected_by_any_exclusion"] is False
    assert rare["modeling_images"] > 0


def test_the_class_map_is_explicit_and_excludes_the_placeholder() -> None:
    class_map = manifest()["class_map"]
    assert class_map == {name: index for index, name in enumerate(CLASS_ORDER)}
    assert "object" not in class_map


def test_the_fingerprint_is_present_and_well_formed() -> None:
    fingerprints = manifest()["fingerprints"]
    assert set(fingerprints) >= {
        "images_sha256",
        "annotations_sha256",
        "groups_sha256",
        "class_map_sha256",
        "modeling_population_sha256",
    }
    assert all(len(value) == 64 for value in fingerprints.values())


def test_the_manifest_records_no_validation_problem() -> None:
    assert manifest()["validation_problems"] == []


@pytest.mark.parametrize(
    "name",
    [
        "canonical_modeling_population.csv",
        "canonical_annotation_actions.csv",
        "group_manifest.csv",
        "group_split_features.csv",
        "unconfirmed_group_candidates.csv",
        "canonical_modeling_manifest.json",
        "canonical_modeling_population_report.md",
        "fragment_rule_analysis.csv",
        "fragment_rule_report.md",
    ],
)
def test_committed_phase_5b_artifacts_carry_no_secret_or_local_path(name: str) -> None:
    path = REPORTS / name
    if not path.is_file():
        pytest.skip(f"{name} not generated yet")
    assert scan_for_sensitive(path.read_text(encoding="utf-8")) == []
