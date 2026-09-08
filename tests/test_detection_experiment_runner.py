"""Tests for the pure protocol guards inside the Phase 7 experiment runner.

The runner is a script rather than a package module, so it is loaded by path.
Only its side-effect-free guards are exercised: nothing here trains, validates,
downloads a checkpoint or writes into `reports/`.

The weight-identity guard is the one worth pinning. A candidate that inherits
the reference's `weight_identifier` is asserting it starts from the *same bytes*;
checking only the file name would let a silently replaced asset add an
undeclared variable to a comparison that claims to have exactly one.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from construction_safety_vision.detection_comparison import (
    ExperimentDeclaration,
    load_experiment_matrix,
)
from construction_safety_vision.paths import ProjectPaths

REFERENCE_WEIGHTS = {
    "identifier": "yolo11n.pt",
    "sha256": "0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1",
    "size_bytes": 5613764,
    "committed": False,
}


@pytest.fixture(scope="module")
def paths() -> ProjectPaths:
    return ProjectPaths.from_root()


@pytest.fixture(scope="module")
def runner(paths: ProjectPaths):
    script = paths.scripts / "train_detection_experiment.py"
    spec = importlib.util.spec_from_file_location("phase7_runner", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def matrix(paths: ProjectPaths):
    return load_experiment_matrix(paths.configs / "detection_experiments.yaml")


def _declaration(**overrides) -> ExperimentDeclaration:
    base = {
        "experiment_id": "D9",
        "role": "CONTROLLED_EXPERIMENT",
        "question": "q",
        "status": "FROZEN_NOT_EXECUTED",
        "intentional_variable": "INPUT_RESOLUTION",
        "intentional_fields": ("training.imgsz",),
        "consequential_fields": (),
        "overrides": {"training.imgsz": 768},
        "inherits": "D0",
        "protocol_source": None,
        "result_manifest": None,
        "hypothesis": "h",
    }
    base.update(overrides)
    return ExperimentDeclaration(**base)


# --- inherited pretrained weights --------------------------------------------


def test_identical_inherited_weights_are_accepted(runner):
    verdict = runner.verify_inherited_weights(
        _declaration(), dict(REFERENCE_WEIGHTS), {"pretrained_weights": dict(REFERENCE_WEIGHTS)}
    )
    assert verdict == "IDENTICAL_TO_REFERENCE_VERIFIED_BY_DIGEST"


def test_a_changed_inherited_binary_is_refused(runner):
    weights = {**REFERENCE_WEIGHTS, "sha256": "f" * 64}
    with pytest.raises(runner.ExperimentRunError, match="asset changed"):
        runner.verify_inherited_weights(
            _declaration(), weights, {"pretrained_weights": dict(REFERENCE_WEIGHTS)}
        )


def test_a_changed_inherited_size_is_refused(runner):
    weights = {**REFERENCE_WEIGHTS, "size_bytes": 999}
    with pytest.raises(runner.ExperimentRunError, match="bytes"):
        runner.verify_inherited_weights(
            _declaration(), weights, {"pretrained_weights": dict(REFERENCE_WEIGHTS)}
        )


def test_a_different_inherited_identifier_is_refused(runner):
    weights = {**REFERENCE_WEIGHTS, "identifier": "yolo11m.pt"}
    with pytest.raises(runner.ExperimentRunError, match="resolved to"):
        runner.verify_inherited_weights(
            _declaration(), weights, {"pretrained_weights": dict(REFERENCE_WEIGHTS)}
        )


def test_a_declared_override_is_expected_to_differ(runner):
    declaration = _declaration(
        intentional_fields=("model",),
        consequential_fields=("weight_identifier",),
        overrides={"model": "YOLO11s", "weight_identifier": "yolo11s.pt"},
        intentional_variable="MODEL_CAPACITY",
    )
    weights = {"identifier": "yolo11s.pt", "sha256": "a" * 64, "size_bytes": 19313732}
    verdict = runner.verify_inherited_weights(
        declaration, weights, {"pretrained_weights": dict(REFERENCE_WEIGHTS)}
    )
    assert verdict == "DECLARED_OVERRIDE_DIFFERS_FROM_REFERENCE"


def test_a_declared_override_that_did_not_change_the_bytes_is_refused(runner):
    # Declaring a new checkpoint and then loading the reference's bytes means the
    # variable was never applied.
    declaration = _declaration(
        intentional_fields=("model",),
        consequential_fields=("weight_identifier",),
        overrides={"model": "YOLO11s", "weight_identifier": "yolo11s.pt"},
    )
    with pytest.raises(runner.ExperimentRunError, match="was not actually applied"):
        runner.verify_inherited_weights(
            declaration, dict(REFERENCE_WEIGHTS), {"pretrained_weights": dict(REFERENCE_WEIGHTS)}
        )


def test_an_unrecorded_reference_digest_is_reported_not_assumed(runner):
    verdict = runner.verify_inherited_weights(_declaration(), dict(REFERENCE_WEIGHTS), {})
    assert verdict == "REFERENCE_DIGEST_NOT_RECORDED_CANNOT_VERIFY"


def test_the_frozen_d2_declaration_inherits_the_reference_weights(matrix):
    declaration = matrix.declaration("D2")
    assert "weight_identifier" not in declaration.overrides
    assert declaration.overrides == {"training.imgsz": 768}


def test_the_frozen_d1_declaration_overrides_the_reference_weights(matrix):
    declaration = matrix.declaration("D1")
    assert declaration.overrides["weight_identifier"] == "yolo11s.pt"
    assert declaration.consequential_fields == ("weight_identifier",)


# --- selection state ----------------------------------------------------------


def test_selection_is_blocked_while_a_candidate_is_missing(runner, paths, matrix):
    state = runner.selection_state(
        paths,
        matrix,
        rows=[{"experiment_id": "D0", "metrics": {}}, {"experiment_id": "D1", "metrics": None}],
        pending=["D1"],
        reference={"class_map": {}},
    )
    assert state["status"] == "PENDING_INCOMPLETE_MATRIX"
    assert state["policy_case_candidate"] is None
    assert state["preferred_experiment"] is None
    assert "cannot run until" in state["reason"]


def test_a_complete_matrix_still_selects_nothing(runner, paths, matrix):
    manifests = [paths.reports / f"detection_{name}_manifest.json" for name in ("D0", "D1", "D2")]
    if not all(path.is_file() for path in manifests):
        pytest.skip("not every experiment has a committed result yet")
    d0 = paths.reports / "detection_D0_manifest.json"
    import json

    reference = json.loads(d0.read_text(encoding="utf-8"))
    state = runner.selection_state(
        paths,
        matrix,
        rows=[{"experiment_id": name, "metrics": {}} for name in ("D0", "D1", "D2")],
        pending=[],
        reference={"class_map": reference["class_map"]},
    )
    assert state["status"] == "PENDING_HUMAN_REVIEW"
    assert state["preferred_experiment"] is None
    assert state["advisory_only"] is True
    # A computed case may be exposed as an input to review, never as a decision.
    if state["policy_case_candidate"] is not None:
        assert state["policy_case_candidate"].startswith("CASE_")


# --- runner invariants --------------------------------------------------------


def test_the_completion_marker_names_the_experiment_that_ran(runner):
    # It used to be hardcoded to D1, which would have mislabelled every later
    # experiment's completion line.
    assert (
        runner.EXPERIMENT_COMPLETE_TEMPLATE.format(experiment_id="D2") == "D2_EXPERIMENT_COMPLETE"
    )
    assert (
        runner.EXPERIMENT_COMPLETE_TEMPLATE.format(experiment_id="D1") == "D1_EXPERIMENT_COMPLETE"
    )


def test_the_runner_declares_the_unselected_sentinel(runner):
    assert runner.UNSELECTED == "UNSELECTED_PENDING_REVIEW"


def test_the_margin_statuses_are_the_three_frozen_ones(runner):
    assert runner.IMPROVES == "IMPROVES_D0_BEYOND_MARGIN"
    assert runner.EQUIVALENT == "PRACTICALLY_EQUIVALENT_TO_D0"
    assert runner.BELOW == "BELOW_D0"


def test_the_runner_refuses_a_holdout_named_dataset_descriptor(runner, paths):
    # The descriptor is checked on parsed keys, not raw text: its comments
    # explain that it deliberately has no holdout key, and a substring match
    # would read that explanation as the violation it rules out.
    descriptor = paths.root / runner.DATASET_DESCRIPTOR
    if not descriptor.is_file():
        pytest.skip("the adapter dataset descriptor is git-ignored and not present")
    import yaml

    parsed = yaml.safe_load(descriptor.read_text(encoding="utf-8"))
    assert "test" not in {str(key).lower() for key in parsed}
    assert "test" in descriptor.read_text(encoding="utf-8").lower()  # in a comment only


def test_the_memory_preflight_directory_is_not_an_experiment_directory(runner):
    assert runner.MEMORY_PREFLIGHT_NAME.startswith("_")
    assert runner.MEMORY_PREFLIGHT_NAME not in ("D0", "D1", "D2")


def test_the_runner_only_writes_under_the_ignored_run_root(runner):
    assert runner.RUN_ROOT == "artifacts/detection"
    assert Path(runner.DATASET_DESCRIPTOR).parts[0] == "data"
