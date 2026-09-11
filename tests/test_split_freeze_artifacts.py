"""Tests that the committed frozen split is what the project says it is.

These read the real artifacts under ``reports/``. They deliberately assert on
*counts and invariants*, never on holdout identifiers: printing the holdout's
image ids into a test failure would defeat the guard the same tests check. The
one place the holdout's membership is exercised is through its fingerprint,
which reveals nothing about its contents.
"""

from __future__ import annotations

import csv
import json

import pytest

from construction_safety_vision.data.canonical import scan_for_sensitive
from construction_safety_vision.data.split_freeze import (
    FROZEN,
    MANIFEST_SCHEMA_VERSION,
    MANIFEST_SPLITS,
    NON_SELECTED,
    SELECTED,
    TEST,
    assignments_from_sections,
    fingerprint_holdout,
    fingerprint_split_assignment,
    load_frozen_splits,
    load_split_freeze_config,
    validate_manifest,
)
from construction_safety_vision.data.split_optimization import fingerprint_assignment
from construction_safety_vision.paths import ProjectPaths
from construction_safety_vision.splits import HOLDOUT_UNLOCK_ENV_VAR, HoldoutViolationError

EXPECTED_IMAGES = {"train": 303, "validation": 65, "test": 65}
EXPECTED_GROUPS = {"train": 294, "validation": 63, "test": 65}
EXPECTED_NEGATIVES = {"train": 10, "validation": 2, "test": 2}
EXPECTED_RARE_IMAGES = {"train": 5, "validation": 1, "test": 2}
EXPECTED_RARE_INSTANCES = {"train": 30, "validation": 8, "test": 7}
MODELLING_IMAGES = 433
MODELLING_ANNOTATIONS = 2031
SPLIT_UNITS = 422
NON_SINGLETON_GROUPS = 11
CLASSES = ("helmet_loose", "helmet_on_head", "person", "vest_loose", "vest_on_body")

LOCKED_ENV: dict[str, str] = {}
UNLOCKED_ENV = {HOLDOUT_UNLOCK_ENV_VAR: "1"}


@pytest.fixture(scope="module")
def paths() -> ProjectPaths:
    return ProjectPaths.from_root()


@pytest.fixture(scope="module")
def manifest(paths: ProjectPaths) -> dict:
    path = paths.reports / "split_manifest.json"
    if not path.is_file():
        pytest.skip("split_manifest.json not present; run scripts/freeze_split.py")
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def assignment_rows(paths: ProjectPaths) -> list[dict[str, str]]:
    path = paths.reports / "final_split_assignments.csv"
    if not path.is_file():
        pytest.skip("final_split_assignments.csv not present; run scripts/freeze_split.py")
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_csv(paths: ProjectPaths, name: str) -> list[dict[str, str]]:
    with (paths.reports / name).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


# --- manifest schema and status ---------------------------------------------


def test_manifest_is_valid(manifest):
    assert validate_manifest(manifest) == []


def test_manifest_declares_a_frozen_split(manifest):
    assert manifest["schema_version"] == MANIFEST_SCHEMA_VERSION
    assert manifest["status"] == FROZEN
    assert manifest["holdout_frozen"] is True
    assert manifest["created_from_candidate"] == "candidate_001"
    assert manifest["final_selected_candidate"] == "candidate_001"
    assert manifest["selection_status"] == SELECTED
    assert manifest["selection_method"] == "HUMAN_REVIEW_OF_PREDECLARED_DETERMINISTIC_CANDIDATES"


def test_manifest_records_that_no_data_was_materialised(manifest):
    assert manifest["physical_materialisation"]["performed"] is False
    assert manifest["provider_split_used"] is False


def test_manifest_carries_no_sensitive_content(paths):
    text = (paths.reports / "split_manifest.json").read_text(encoding="utf-8")
    assert scan_for_sensitive(text) == []


def test_freeze_report_carries_no_sensitive_content(paths):
    text = (paths.reports / "split_freeze_report.md").read_text(encoding="utf-8")
    assert scan_for_sensitive(text) == []


# --- the numbers ------------------------------------------------------------


def test_image_counts_are_exactly_303_65_65(manifest):
    assert manifest["actual_image_counts"] == EXPECTED_IMAGES
    assert sum(manifest["actual_image_counts"].values()) == MODELLING_IMAGES


def test_group_counts_match_the_422_split_units(manifest):
    assert manifest["actual_group_counts"] == EXPECTED_GROUPS
    assert sum(manifest["actual_group_counts"].values()) == SPLIT_UNITS
    assert sum(manifest["actual_non_singleton_group_counts"].values()) == NON_SINGLETON_GROUPS


def test_negative_images_are_10_2_2(manifest):
    assert manifest["actual_negative_image_counts"] == EXPECTED_NEGATIVES


def test_annotations_total_the_modelling_population(manifest):
    assert sum(manifest["actual_annotation_counts"].values()) == MODELLING_ANNOTATIONS


def test_every_class_appears_in_every_split(manifest):
    for split in MANIFEST_SPLITS:
        for name in CLASSES:
            assert manifest["images_with_class"][split][name] > 0
            assert manifest["instances_by_class"][split][name] > 0


def test_vest_loose_images_are_5_1_2(manifest):
    found = {split: manifest["images_with_class"][split]["vest_loose"] for split in MANIFEST_SPLITS}
    assert found == EXPECTED_RARE_IMAGES


def test_vest_loose_instances_are_30_8_7(manifest):
    found = {
        split: manifest["instances_by_class"][split]["vest_loose"] for split in MANIFEST_SPLITS
    }
    assert found == EXPECTED_RARE_INSTANCES
    assert sum(found.values()) == 45


def test_per_class_totals_match_the_population(paths, manifest):
    population = json.loads(
        (paths.reports / "canonical_modeling_manifest.json").read_text(encoding="utf-8")
    )
    by_class = population["population"]["annotations_by_class"]
    images_by_class = population["population"]["modeling_images_by_class"]
    for name in CLASSES:
        assert (
            sum(manifest["instances_by_class"][s][name] for s in MANIFEST_SPLITS) == by_class[name]
        )
        assert (
            sum(manifest["images_with_class"][s][name] for s in MANIFEST_SPLITS)
            == images_by_class[name]
        )


# --- coverage and integrity -------------------------------------------------


def test_assignment_csv_holds_one_row_per_modelling_image(assignment_rows):
    assert len(assignment_rows) == MODELLING_IMAGES
    assert len({row["source_image_id"] for row in assignment_rows}) == MODELLING_IMAGES
    assert set(assignment_rows[0]) == {"group_id", "source_image_id", "split"}


def test_assignment_csv_uses_only_the_canonical_vocabulary(assignment_rows):
    assert {row["split"] for row in assignment_rows} == set(MANIFEST_SPLITS)


def test_assignment_csv_is_deterministically_sorted(assignment_rows):
    keys = [(row["group_id"], row["source_image_id"]) for row in assignment_rows]
    assert keys == sorted(keys)


def test_assignment_csv_agrees_with_the_manifest(manifest, assignment_rows):
    from_manifest = {
        (row.group_id, row.source_image_id, row.split)
        for row in assignments_from_sections(manifest)
    }
    from_csv = {(row["group_id"], row["source_image_id"], row["split"]) for row in assignment_rows}
    assert from_manifest == from_csv


def test_every_eligible_image_is_assigned_exactly_once(paths, assignment_rows):
    eligible = {
        row["source_image_id"]
        for row in read_csv(paths, "canonical_modeling_population.csv")
        if row["status"] == "ELIGIBLE"
    }
    assigned = [row["source_image_id"] for row in assignment_rows]
    assert sorted(assigned) == sorted(eligible)
    assert len(assigned) == len(set(assigned))


def test_no_excluded_out_of_domain_image_appears(paths, assignment_rows):
    excluded = {
        row["source_image_id"]
        for row in read_csv(paths, "canonical_modeling_population.csv")
        if row["status"] != "ELIGIBLE"
    }
    assert len(excluded) == 3
    assert excluded.isdisjoint({row["source_image_id"] for row in assignment_rows})


def test_no_group_crosses_a_split_boundary(assignment_rows):
    splits_of: dict[str, set[str]] = {}
    for row in assignment_rows:
        splits_of.setdefault(row["group_id"], set()).add(row["split"])
    assert all(len(splits) == 1 for splits in splits_of.values())
    assert len(splits_of) == SPLIT_UNITS


def test_group_membership_matches_the_group_manifest(paths, assignment_rows):
    declared: dict[str, set[str]] = {}
    for row in read_csv(paths, "group_manifest.csv"):
        declared.setdefault(row["group_id"], set()).add(row["source_image_id"])
    frozen: dict[str, set[str]] = {}
    for row in assignment_rows:
        frozen.setdefault(row["group_id"], set()).add(row["source_image_id"])
    assert frozen == declared


def test_the_eleven_confirmed_groups_stay_whole(paths, assignment_rows):
    duplicates = {
        row["group_id"]
        for row in read_csv(paths, "group_manifest.csv")
        if row["group_type"] == "SEMANTIC_DUPLICATE"
    }
    assert len(duplicates) == NON_SINGLETON_GROUPS
    by_group: dict[str, set[str]] = {}
    for row in assignment_rows:
        by_group.setdefault(row["group_id"], set()).add(row["split"])
    assert all(len(by_group[group_id]) == 1 for group_id in duplicates)


# --- fingerprints -----------------------------------------------------------


def test_candidate_001_still_reproduces_its_recorded_fingerprint(paths, manifest):
    config = load_split_freeze_config(paths.configs / "split_freeze.yaml")
    candidate = {
        row["group_id"]: row["provisional_split"]
        for row in read_csv(paths, "split_candidates/candidate_001.csv")
    }
    recomputed = fingerprint_assignment(candidate)
    assert recomputed == config.expected_candidate_assignment_sha256
    assert recomputed == manifest["candidate_assignment_sha256"]


def test_candidate_fingerprint_agrees_with_the_phase_5c1_summary(paths, manifest):
    row = next(
        row
        for row in read_csv(paths, "split_candidates/summary.csv")
        if row["candidate_id"] == "candidate_001"
    )
    assert row["candidate_assignment_sha256"] == manifest["candidate_assignment_sha256"]


def test_split_fingerprint_is_reproducible_from_the_manifest(manifest):
    assignments = assignments_from_sections(manifest)
    assert fingerprint_split_assignment(assignments) == manifest["split_assignment_sha256"]


def test_holdout_fingerprint_is_reproducible_from_the_manifest(manifest):
    assignments = assignments_from_sections(manifest)
    recomputed = fingerprint_holdout(
        assignments, modeling_population_sha256=manifest["modeling_population_sha256"]
    )
    assert recomputed == manifest["holdout_sha256"]


def test_holdout_fingerprint_moves_if_test_membership_changes(manifest):
    # Move one holdout image into validation without naming it, and confirm the
    # holdout digest notices.
    assignments = list(assignments_from_sections(manifest))
    index = next(i for i, row in enumerate(assignments) if row.split == TEST)
    from construction_safety_vision.data.split_freeze import SplitAssignment

    assignments[index] = SplitAssignment(
        assignments[index].group_id, assignments[index].source_image_id, "validation"
    )
    recomputed = fingerprint_holdout(
        assignments, modeling_population_sha256=manifest["modeling_population_sha256"]
    )
    assert recomputed != manifest["holdout_sha256"]


def test_population_fingerprints_match_the_phase_5b_manifest(paths, manifest):
    population = json.loads(
        (paths.reports / "canonical_modeling_manifest.json").read_text(encoding="utf-8")
    )
    assert (
        manifest["modeling_population_sha256"]
        == population["fingerprints"]["modeling_population_sha256"]
    )
    assert manifest["groups_sha256"] == population["fingerprints"]["groups_sha256"]


def test_split_and_candidate_fingerprints_are_distinct(manifest):
    # They cover different content: (group, split) against (group, image, split).
    assert manifest["split_assignment_sha256"] != manifest["candidate_assignment_sha256"]


# --- holdout guard ----------------------------------------------------------


def test_frozen_splits_load_and_serve_the_development_splits(paths):
    splits = load_frozen_splits(paths.reports / "split_manifest.json")
    assert len(splits.image_ids("train", purpose="training", env=LOCKED_ENV)) == 303
    assert len(splits.image_ids("validation", purpose="model selection", env=LOCKED_ENV)) == 65
    assert len(splits.group_ids("train", purpose="training", env=LOCKED_ENV)) == 294


def test_the_frozen_holdout_is_locked_by_default(paths):
    splits = load_frozen_splits(paths.reports / "split_manifest.json")
    with pytest.raises(HoldoutViolationError, match="allow_test=True"):
        splits.image_ids("test", purpose="training", env=LOCKED_ENV)


def test_the_frozen_holdout_refuses_a_code_opt_in_alone(paths):
    splits = load_frozen_splits(paths.reports / "split_manifest.json")
    with pytest.raises(HoldoutViolationError, match=HOLDOUT_UNLOCK_ENV_VAR):
        splits.image_ids("test", purpose="evaluation", allow_test=True, env=LOCKED_ENV)


def test_the_frozen_holdout_refuses_an_environment_opt_in_alone(paths):
    splits = load_frozen_splits(paths.reports / "split_manifest.json")
    with pytest.raises(HoldoutViolationError, match="allow_test=True"):
        splits.image_ids("test", purpose="evaluation", env=UNLOCKED_ENV)


def test_the_declared_guard_policy_matches_the_enforced_one(manifest):
    guard = manifest["holdout_guard"]
    assert guard["protected_split"] == TEST
    assert guard["env_var"] == HOLDOUT_UNLOCK_ENV_VAR
    assert guard["requires_code_opt_in"] is True
    assert guard["requires_environment_opt_in"] is True
    assert guard["default_state"] == "LOCKED"


def test_the_environment_does_not_unlock_the_holdout_during_the_test_run():
    """The gate stays unset unless the one-shot evaluation has actually been run.

    Before phase 11B the variable must be absent: a stray unlock left over from
    development is exactly what this catches. Phase 11B is the one authorised
    condition under which a person deliberately sets it, and the committed
    evidence that it happened is the final evaluation's provenance record, so
    the assertion is relaxed only once that record exists.
    """
    import os

    from construction_safety_vision.paths import ProjectPaths

    executed = (
        ProjectPaths.from_root().reports / "final_test_evaluation.provenance.json"
    ).is_file()
    assert os.environ.get(HOLDOUT_UNLOCK_ENV_VAR, "") != "1" or executed


# --- candidate artifacts ----------------------------------------------------


def test_candidates_002_to_006_remain_preserved_and_non_selected(paths):
    rows = {row["candidate_id"]: row for row in read_csv(paths, "split_candidates/selection.csv")}
    assert rows["candidate_001"]["selection_status"] == SELECTED
    assert rows["candidate_001"]["selected_in"] == "5C.2"
    for index in range(2, 7):
        candidate_id = f"candidate_{index:03d}"
        assert rows[candidate_id]["selection_status"] == NON_SELECTED
        assert (paths.reports / "split_candidates" / f"{candidate_id}.csv").is_file()


def test_candidate_files_still_use_the_provisional_column(paths):
    for index in range(1, 7):
        rows = read_csv(paths, f"split_candidates/candidate_{index:03d}.csv")
        assert set(rows[0]) == {"group_id", "provisional_split"}


def test_candidate_report_records_the_selection(paths):
    text = (paths.reports / "split_candidate_report.md").read_text(encoding="utf-8")
    assert "UNSELECTED_PENDING_REVIEW" not in text
    assert "`candidate_001`" in text
    assert "phase 5C.2" in text


def test_candidate_scores_were_not_rewritten(paths, manifest):
    row = next(
        row
        for row in read_csv(paths, "split_candidates/summary.csv")
        if row["candidate_id"] == "candidate_001"
    )
    assert row["total_objective"] == "0.132203"
    assert row["train_images"] == "303"
    assert row["train_vest_loose_instances"] == "30"
