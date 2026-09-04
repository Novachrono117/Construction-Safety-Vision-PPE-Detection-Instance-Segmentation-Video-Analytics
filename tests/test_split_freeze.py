"""Tests for the frozen split manifest, its fingerprints and its access layer.

Everything here runs against synthetic fixtures. The one test that exercises a
fully unlocked holdout uses a manifest built in ``tmp_path``, so demonstrating
that both opt-ins together grant access never touches the project's real
protected split.
"""

from __future__ import annotations

import json

import pytest

from construction_safety_vision.data.split_freeze import (
    FROZEN,
    MANIFEST_SCHEMA_VERSION,
    MANIFEST_SPLITS,
    NON_SELECTED,
    SELECTED,
    FingerprintMismatchError,
    HoldoutGuardPolicy,
    SplitAssignment,
    SplitManifestError,
    assignment_rows,
    assignments_from_sections,
    fingerprint_holdout,
    fingerprint_split_assignment,
    load_frozen_splits,
    load_split_freeze_config,
    normalise_assignments,
    split_sections,
    stamp_selection_status,
    validate_manifest,
)
from construction_safety_vision.splits import (
    HOLDOUT_UNLOCK_ENV_VAR,
    HoldoutViolationError,
    Split,
)

POPULATION_SHA = "a" * 64
GROUPS_SHA = "b" * 64
CANDIDATE_SHA = "c" * 64

UNLOCKED_ENV = {HOLDOUT_UNLOCK_ENV_VAR: "1"}
LOCKED_ENV: dict[str, str] = {}


def make_assignments() -> list[SplitAssignment]:
    """Two-image group in train, singletons elsewhere."""
    return [
        SplitAssignment("pair_001", "img_a", "train"),
        SplitAssignment("pair_001", "img_b", "train"),
        SplitAssignment("single_c", "img_c", "train"),
        SplitAssignment("single_d", "img_d", "validation"),
        SplitAssignment("single_e", "img_e", "test"),
        SplitAssignment("single_f", "img_f", "test"),
    ]


def make_manifest(assignments: list[SplitAssignment], **overrides: object) -> dict:
    """Build a well-formed manifest over the given assignment."""
    sections = split_sections(assignments)
    manifest: dict = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "status": FROZEN,
        "created_from_candidate": "candidate_001",
        "selection_method": "HUMAN_REVIEW_OF_PREDECLARED_DETERMINISTIC_CANDIDATES",
        "seed": 42,
        "target_ratios": {"train": 0.7, "validation": 0.15, "test": 0.15},
        "actual_image_counts": {s: sections[s]["image_count"] for s in MANIFEST_SPLITS},
        "actual_group_counts": {s: sections[s]["group_count"] for s in MANIFEST_SPLITS},
        "modeling_population_sha256": POPULATION_SHA,
        "groups_sha256": GROUPS_SHA,
        "candidate_assignment_sha256": CANDIDATE_SHA,
        "split_assignment_sha256": fingerprint_split_assignment(assignments),
        "holdout_sha256": fingerprint_holdout(
            assignments, modeling_population_sha256=POPULATION_SHA
        ),
    }
    manifest.update({s: sections[s] for s in MANIFEST_SPLITS})
    manifest.update(overrides)
    return manifest


def write_manifest(tmp_path, manifest: dict, *, population: dict | None = None):
    """Write a manifest and the population manifest the loader cross-checks."""
    path = tmp_path / "split_manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    recorded = (
        population
        if population is not None
        else {
            "fingerprints": {
                "modeling_population_sha256": POPULATION_SHA,
                "groups_sha256": GROUPS_SHA,
            }
        }
    )
    (tmp_path / "canonical_modeling_manifest.json").write_text(
        json.dumps(recorded), encoding="utf-8"
    )
    return path


# --- fingerprints -----------------------------------------------------------


def test_split_fingerprint_is_deterministic_and_order_independent():
    assignments = make_assignments()
    first = fingerprint_split_assignment(assignments)
    second = fingerprint_split_assignment(list(reversed(assignments)))
    assert first == second
    assert first == fingerprint_split_assignment(assignments)


def test_holdout_fingerprint_is_deterministic_and_order_independent():
    assignments = make_assignments()
    first = fingerprint_holdout(assignments, modeling_population_sha256=POPULATION_SHA)
    second = fingerprint_holdout(
        list(reversed(assignments)), modeling_population_sha256=POPULATION_SHA
    )
    assert first == second


def test_split_fingerprint_moves_when_an_image_changes_split():
    assignments = make_assignments()
    moved = [
        SplitAssignment(row.group_id, row.source_image_id, "train")
        if row.source_image_id == "img_d"
        else row
        for row in assignments
    ]
    assert fingerprint_split_assignment(moved) != fingerprint_split_assignment(assignments)


def test_split_fingerprint_moves_when_group_membership_changes():
    assignments = make_assignments()
    regrouped = [
        SplitAssignment("single_b", row.source_image_id, row.split)
        if row.source_image_id == "img_b"
        else row
        for row in assignments
    ]
    assert fingerprint_split_assignment(regrouped) != fingerprint_split_assignment(assignments)


def test_holdout_fingerprint_moves_when_test_membership_changes():
    assignments = make_assignments()
    swapped = [
        SplitAssignment(row.group_id, row.source_image_id, "validation")
        if row.source_image_id == "img_f"
        else row
        for row in assignments
    ]
    assert fingerprint_holdout(
        swapped, modeling_population_sha256=POPULATION_SHA
    ) != fingerprint_holdout(assignments, modeling_population_sha256=POPULATION_SHA)


def test_holdout_fingerprint_ignores_changes_outside_the_holdout():
    # Moving an image between train and validation must not disturb the digest
    # that exists to detect drift in the protected split.
    assignments = make_assignments()
    moved = [
        SplitAssignment(row.group_id, row.source_image_id, "validation")
        if row.source_image_id == "img_c"
        else row
        for row in assignments
    ]
    assert fingerprint_holdout(
        moved, modeling_population_sha256=POPULATION_SHA
    ) == fingerprint_holdout(assignments, modeling_population_sha256=POPULATION_SHA)


def test_holdout_fingerprint_moves_with_the_population_it_was_drawn_from():
    assignments = make_assignments()
    assert fingerprint_holdout(
        assignments, modeling_population_sha256="d" * 64
    ) != fingerprint_holdout(assignments, modeling_population_sha256=POPULATION_SHA)


def test_unknown_split_name_is_rejected():
    with pytest.raises(SplitManifestError, match="unknown split name"):
        normalise_assignments([SplitAssignment("g", "i", "val")])


# --- manifest ---------------------------------------------------------------


def test_wellformed_manifest_has_no_problems():
    assert validate_manifest(make_manifest(make_assignments())) == []


def test_roundtrip_through_sections_preserves_the_assignment():
    assignments = normalise_assignments(make_assignments())
    assert assignments_from_sections(make_manifest(list(assignments))) == assignments


def test_missing_field_is_reported():
    manifest = make_manifest(make_assignments())
    del manifest["holdout_sha256"]
    assert any("missing required field" in problem for problem in validate_manifest(manifest))


def test_wrong_status_is_reported():
    manifest = make_manifest(make_assignments(), status="PROVISIONAL")
    assert any("is not 'FROZEN'" in problem for problem in validate_manifest(manifest))


def test_unsupported_schema_version_is_reported():
    manifest = make_manifest(make_assignments(), schema_version=99)
    assert any("schema_version" in problem for problem in validate_manifest(manifest))


def test_declared_counts_must_match_the_membership():
    manifest = make_manifest(make_assignments())
    manifest["actual_image_counts"]["train"] = 999
    assert any("actual_image_counts" in problem for problem in validate_manifest(manifest))


def test_group_straddling_a_boundary_is_reported():
    assignments = [
        SplitAssignment("pair_001", "img_a", "train"),
        SplitAssignment("pair_001", "img_b", "test"),
        SplitAssignment("single_d", "img_d", "validation"),
    ]
    manifest = make_manifest(assignments)
    assert any("is split across" in problem for problem in validate_manifest(manifest))


def test_stale_split_fingerprint_is_reported():
    manifest = make_manifest(make_assignments(), split_assignment_sha256="0" * 64)
    assert any(
        "split_assignment_sha256 does not match" in problem
        for problem in validate_manifest(manifest)
    )


def test_stale_holdout_fingerprint_is_reported():
    manifest = make_manifest(make_assignments(), holdout_sha256="0" * 64)
    assert any(
        "holdout_sha256 does not match" in problem for problem in validate_manifest(manifest)
    )


def test_assignment_rows_are_sorted_and_complete():
    rows = assignment_rows(make_assignments())
    assert len(rows) == 6
    assert [row["source_image_id"] for row in rows] == sorted(
        row["source_image_id"] for row in rows
    )
    assert set(rows[0]) == {"group_id", "source_image_id", "split"}


# --- loading ----------------------------------------------------------------


def test_missing_manifest_is_rejected(tmp_path):
    with pytest.raises(SplitManifestError, match="not found"):
        load_frozen_splits(tmp_path / "split_manifest.json")


def test_malformed_json_is_rejected(tmp_path):
    path = tmp_path / "split_manifest.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(SplitManifestError, match="not valid JSON"):
        load_frozen_splits(path, verify_population=False)


def test_malformed_manifest_is_rejected(tmp_path):
    manifest = make_manifest(make_assignments())
    manifest["train"] = {"group_count": 3}
    path = write_manifest(tmp_path, manifest)
    with pytest.raises(SplitManifestError, match="no 'groups' list"):
        load_frozen_splits(path)


def test_stale_fingerprint_raises_a_mismatch(tmp_path):
    path = write_manifest(tmp_path, make_manifest(make_assignments(), holdout_sha256="0" * 64))
    with pytest.raises(FingerprintMismatchError, match="holdout_sha256"):
        load_frozen_splits(path)


def test_modeling_population_fingerprint_mismatch_is_rejected(tmp_path):
    path = write_manifest(
        tmp_path,
        make_manifest(make_assignments()),
        population={
            "fingerprints": {
                "modeling_population_sha256": "9" * 64,
                "groups_sha256": GROUPS_SHA,
            }
        },
    )
    with pytest.raises(FingerprintMismatchError, match="modeling_population_sha256 disagrees"):
        load_frozen_splits(path)


def test_group_fingerprint_mismatch_is_rejected(tmp_path):
    path = write_manifest(
        tmp_path,
        make_manifest(make_assignments()),
        population={
            "fingerprints": {
                "modeling_population_sha256": POPULATION_SHA,
                "groups_sha256": "9" * 64,
            }
        },
    )
    with pytest.raises(FingerprintMismatchError, match="groups_sha256 disagrees"):
        load_frozen_splits(path)


def test_missing_population_manifest_is_rejected(tmp_path):
    path = tmp_path / "split_manifest.json"
    path.write_text(json.dumps(make_manifest(make_assignments())), encoding="utf-8")
    with pytest.raises(SplitManifestError, match="cannot be verified against the population"):
        load_frozen_splits(path)


# --- access layer -----------------------------------------------------------


def test_train_and_validation_load_without_any_override(tmp_path):
    splits = load_frozen_splits(write_manifest(tmp_path, make_manifest(make_assignments())))
    assert splits.image_ids("train", purpose="training", env=LOCKED_ENV) == (
        "img_a",
        "img_b",
        "img_c",
    )
    assert splits.image_ids("validation", purpose="model selection", env=LOCKED_ENV) == ("img_d",)
    assert splits.group_ids("train", purpose="training", env=LOCKED_ENV) == (
        "pair_001",
        "single_c",
    )
    assert splits.groups("train", purpose="training", env=LOCKED_ENV)["pair_001"] == (
        "img_a",
        "img_b",
    )


def test_validation_accepts_the_val_spelling(tmp_path):
    splits = load_frozen_splits(write_manifest(tmp_path, make_manifest(make_assignments())))
    assert splits.image_ids(Split.VAL, purpose="model selection", env=LOCKED_ENV) == ("img_d",)
    assert splits.image_ids("val", purpose="model selection", env=LOCKED_ENV) == ("img_d",)


def test_unknown_split_name_is_refused_by_the_loader(tmp_path):
    splits = load_frozen_splits(write_manifest(tmp_path, make_manifest(make_assignments())))
    with pytest.raises(ValueError, match="Unknown split"):
        splits.image_ids("trainval", purpose="training", env=LOCKED_ENV)


def test_test_split_is_denied_by_default(tmp_path):
    splits = load_frozen_splits(write_manifest(tmp_path, make_manifest(make_assignments())))
    with pytest.raises(HoldoutViolationError, match="allow_test=True"):
        splits.image_ids("test", purpose="training", env=LOCKED_ENV)


def test_code_opt_in_alone_is_insufficient(tmp_path):
    splits = load_frozen_splits(write_manifest(tmp_path, make_manifest(make_assignments())))
    with pytest.raises(HoldoutViolationError, match=HOLDOUT_UNLOCK_ENV_VAR):
        splits.image_ids("test", purpose="evaluation", allow_test=True, env=LOCKED_ENV)


def test_environment_opt_in_alone_is_insufficient(tmp_path):
    splits = load_frozen_splits(write_manifest(tmp_path, make_manifest(make_assignments())))
    with pytest.raises(HoldoutViolationError, match="allow_test=True"):
        splits.image_ids("test", purpose="evaluation", env=UNLOCKED_ENV)


def test_both_opt_ins_grant_access_to_a_synthetic_holdout(tmp_path):
    # A fixture holdout, never the project's. The point is that the guard opens
    # only when both opt-ins are present, not what the real holdout contains.
    splits = load_frozen_splits(write_manifest(tmp_path, make_manifest(make_assignments())))
    assert splits.image_ids(
        "test", purpose="final evaluation", allow_test=True, env=UNLOCKED_ENV
    ) == ("img_e", "img_f")


def test_group_accessor_is_guarded_too(tmp_path):
    splits = load_frozen_splits(write_manifest(tmp_path, make_manifest(make_assignments())))
    with pytest.raises(HoldoutViolationError):
        splits.groups("test", purpose="debugging", env=UNLOCKED_ENV)
    with pytest.raises(HoldoutViolationError):
        splits.group_ids("test", purpose="debugging", env=UNLOCKED_ENV)


# --- selection stamping -----------------------------------------------------


PROVISIONAL_BANNER = (
    "**No split is frozen and no candidate is selected.** Every assignment below is "
    "provisional. `final_selected_candidate` is `UNSELECTED_PENDING_REVIEW`; the holdout "
    "remains locked and no candidate test set has been evaluated."
)

PROVISIONAL_LIMITATION = (
    "* **No candidate is selected.** `final_selected_candidate` is "
    "`UNSELECTED_PENDING_REVIEW`. `algorithmic_best_candidate` names only the lowest scorer "
    "under the predeclared objective, which is `candidate_001`."
)

PROVISIONAL_REPORT = "\n\n".join(
    [
        "# Provisional Split Candidates",
        PROVISIONAL_BANNER,
        "## 13. Limitations",
        PROVISIONAL_LIMITATION,
    ]
)


def test_stamping_is_idempotent():
    text = PROVISIONAL_REPORT
    once = stamp_selection_status(text, selected="candidate_001", selected_in="phase 5C.2")
    twice = stamp_selection_status(once, selected="candidate_001", selected_in="phase 5C.2")
    assert once == twice
    assert "UNSELECTED_PENDING_REVIEW" not in once
    assert "candidate_001" in once
    assert "phase 5C.2" in once


def test_stamping_refuses_a_partially_recognised_report():
    # The banner alone is not enough: a report whose limitations section has
    # changed shape is one the substitution can no longer verify.
    with pytest.raises(SplitManifestError, match="refusing to guess"):
        stamp_selection_status(
            PROVISIONAL_BANNER, selected="candidate_001", selected_in="phase 5C.2"
        )


def test_stamping_refuses_an_unrecognised_report():
    with pytest.raises(SplitManifestError, match="refusing to guess"):
        stamp_selection_status("# something else\n", selected="c", selected_in="p")


def test_selection_vocabulary_is_distinct():
    assert SELECTED != NON_SELECTED


# --- freeze configuration ---------------------------------------------------


def test_freeze_config_parses(tmp_path):
    from construction_safety_vision.paths import ProjectPaths

    config = load_split_freeze_config(ProjectPaths.from_root().configs / "split_freeze.yaml")
    assert config.selected_candidate == "candidate_001"
    assert config.algorithmic_best_candidate == "candidate_001"
    assert config.rare_class == "vest_loose"
    assert config.expected_image_counts == {"train": 303, "validation": 65, "test": 65}
    assert config.expected_rare_class_images == {"train": 5, "validation": 1, "test": 2}
    assert config.expected_rare_class_instances == {"train": 30, "validation": 8, "test": 7}
    assert config.expected_negative_images == {"train": 10, "validation": 2, "test": 2}
    assert config.holdout_guard == HoldoutGuardPolicy(
        protected_split="test",
        env_var=HOLDOUT_UNLOCK_ENV_VAR,
        requires_code_opt_in=True,
        requires_environment_opt_in=True,
        default_state="LOCKED",
    )


VALID_FREEZE_YAML = """schema_version: 1
manifest_schema_version: 1
selection:
  selected_candidate: candidate_001
  selection_method: HUMAN_REVIEW_OF_PREDECLARED_DETERMINISTIC_CANDIDATES
  decision_source: PROJECT_OWNER_REVIEW
  algorithmic_best_candidate: candidate_001
  rare_class: vest_loose
expected_fingerprints:
  candidate_assignment_sha256: a
  modeling_population_sha256: b
  groups_sha256: c
expected_counts:
  images: {train: 303, validation: 65, test: 65}
  groups: {train: 294, validation: 63, test: 65}
  negative_images: {train: 10, validation: 2, test: 2}
  rare_class_images: {train: 5, validation: 1, test: 2}
  rare_class_instances: {train: 30, validation: 8, test: 7}
holdout_guard:
  protected_split: test
  env_var: CSVISION_ALLOW_TEST_SPLIT
  requires_code_opt_in: true
  requires_environment_opt_in: true
  default_state: LOCKED
"""


def test_freeze_config_rejects_an_unknown_key(tmp_path):
    from construction_safety_vision.config import ConfigError

    path = tmp_path / "split_freeze.yaml"
    path.write_text(VALID_FREEZE_YAML + "surprise: true\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="unknown key"):
        load_split_freeze_config(path)


def test_freeze_config_rejects_a_missing_split_count(tmp_path):
    from construction_safety_vision.config import ConfigError

    path = tmp_path / "split_freeze.yaml"
    path.write_text(
        VALID_FREEZE_YAML.replace(
            "images: {train: 303, validation: 65, test: 65}",
            "images: {train: 303, validation: 65}",
        ),
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="missing required key"):
        load_split_freeze_config(path)


def test_freeze_config_rejects_a_non_integer_count(tmp_path):
    from construction_safety_vision.config import ConfigError

    path = tmp_path / "split_freeze.yaml"
    path.write_text(
        VALID_FREEZE_YAML.replace(
            "images: {train: 303, validation: 65, test: 65}",
            "images: {train: many, validation: 65, test: 65}",
        ),
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="must be an integer"):
        load_split_freeze_config(path)


def test_freeze_config_rejects_a_missing_file(tmp_path):
    from construction_safety_vision.config import ConfigError

    with pytest.raises(ConfigError, match="not found"):
        load_split_freeze_config(tmp_path / "absent.yaml")
