"""Tests for the Phase 7 controlled detection comparison logic.

These test the rules, not the results: the support rule's classification, the
selection metric's arithmetic, the margin boundary, the four selection cases and
the refusals that make a comparison invalid. No model is trained and no
checkpoint is loaded.

The one place a concrete number is asserted is D0's supported macro, computed
independently here from the committed per-class values. That is a test of the
implementation's arithmetic against a hand-derived figure, not a claim about how
well any model should perform.
"""

from __future__ import annotations

import copy
import json
from decimal import Decimal
from fractions import Fraction

import pytest

from construction_safety_vision.detection_comparison import (
    CASE_A,
    CASE_B,
    CASE_C,
    CASE_D,
    COMPARISON_SUPPORTED,
    DESCRIPTIVE_HIGH_UNCERTAINTY,
    IMPROVED,
    OFFICIAL_ALL_CLASS_METRIC,
    PRACTICAL_EQUIVALENCE_MARGIN,
    PRACTICALLY_EQUIVALENT,
    PRIMARY_SELECTION_METRIC,
    REGRESSED,
    SUPPORT_MIN_INSTANCES,
    SUPPORT_MIN_POSITIVE_IMAGES,
    ComparisonConfigError,
    ComparisonError,
    SupportRule,
    build_experiment_record,
    check_protocol_compatibility,
    class_support,
    classify_delta,
    compare,
    contains_forbidden_split,
    descriptive_class_names,
    flatten_protocol,
    load_experiment_matrix,
    protocol_differences,
    resolve_candidate_protocol,
    select,
    supported_class_names,
    supported_macro,
)
from construction_safety_vision.experiment import load_detection_baseline_config
from construction_safety_vision.paths import ProjectPaths
from construction_safety_vision.splits import HOLDOUT_UNLOCK_ENV_VAR, HoldoutViolationError

CLASS_MAP = {
    "helmet_loose": 0,
    "helmet_on_head": 1,
    "person": 2,
    "vest_loose": 3,
    "vest_on_body": 4,
}


@pytest.fixture(scope="module")
def paths() -> ProjectPaths:
    return ProjectPaths.from_root()


@pytest.fixture(scope="module")
def matrix(paths: ProjectPaths):
    return load_experiment_matrix(paths.configs / "detection_experiments.yaml")


@pytest.fixture(scope="module")
def baseline(paths: ProjectPaths):
    return load_detection_baseline_config(paths.configs / "detection_baseline.yaml")


@pytest.fixture(scope="module")
def split_manifest(paths: ProjectPaths) -> dict:
    return json.loads((paths.reports / "split_manifest.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def d0_manifest(paths: ProjectPaths) -> dict:
    path = paths.reports / "detection_D0_manifest.json"
    if not path.is_file():
        pytest.skip("detection_D0_manifest.json not present")
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def support(split_manifest: dict) -> tuple:
    return class_support(split_manifest, CLASS_MAP)


# --- the support rule ---------------------------------------------------------


@pytest.mark.parametrize(
    ("images", "instances", "expected"),
    [
        (5, 20, COMPARISON_SUPPORTED),
        (99, 999, COMPARISON_SUPPORTED),
        (4, 20, DESCRIPTIVE_HIGH_UNCERTAINTY),
        (5, 19, DESCRIPTIVE_HIGH_UNCERTAINTY),
        (4, 19, DESCRIPTIVE_HIGH_UNCERTAINTY),
        (1, 8, DESCRIPTIVE_HIGH_UNCERTAINTY),
        # Both floors are required, and neither substitutes for the other: many
        # instances inside two images is exactly the case the image floor exists
        # to catch.
        (2, 500, DESCRIPTIVE_HIGH_UNCERTAINTY),
        (500, 2, DESCRIPTIVE_HIGH_UNCERTAINTY),
    ],
)
def test_support_rule_classifies_on_both_floors(images, instances, expected):
    rule = SupportRule()
    assert rule.classify(positive_images=images, instances=instances) == expected


def test_support_rule_thresholds_are_the_frozen_ones():
    rule = SupportRule()
    assert (rule.min_positive_images, rule.min_instances) == (
        SUPPORT_MIN_POSITIVE_IMAGES,
        SUPPORT_MIN_INSTANCES,
    )
    assert (SUPPORT_MIN_POSITIVE_IMAGES, SUPPORT_MIN_INSTANCES) == (5, 20)


def test_support_rule_rejects_a_non_positive_threshold():
    with pytest.raises(ComparisonConfigError):
        SupportRule(min_positive_images=0)
    with pytest.raises(ComparisonConfigError):
        SupportRule(min_instances=-1)


def test_frozen_support_classifies_exactly_the_expected_classes(support):
    assert supported_class_names(support) == (
        "helmet_loose",
        "helmet_on_head",
        "person",
        "vest_on_body",
    )
    assert descriptive_class_names(support) == ("vest_loose",)


def test_the_rare_class_is_descriptive_because_of_its_support_not_its_name(support):
    rare = next(record for record in support if record.class_name == "vest_loose")
    assert (rare.positive_images, rare.instances) == (1, 8)
    assert rare.classification == DESCRIPTIVE_HIGH_UNCERTAINTY
    # The classification follows from the counts alone: fed the same counts under
    # any other name, the rule reaches the same verdict.
    assert (
        SupportRule().classify(positive_images=rare.positive_images, instances=rare.instances)
        == DESCRIPTIVE_HIGH_UNCERTAINTY
    )


def test_support_is_independent_of_class_map_ordering(split_manifest):
    shuffled = dict(reversed(list(CLASS_MAP.items())))
    reordered = class_support(split_manifest, shuffled)
    assert supported_class_names(reordered) == supported_class_names(
        class_support(split_manifest, CLASS_MAP)
    )
    # Records come back in class-index order whatever order the map was built in.
    assert [record.class_index for record in reordered] == [0, 1, 2, 3, 4]


def test_support_reads_validation_and_never_the_holdout(split_manifest, monkeypatch):
    monkeypatch.delenv(HOLDOUT_UNLOCK_ENV_VAR, raising=False)
    support = class_support(split_manifest, CLASS_MAP)
    counts = {record.class_name: record.instances for record in support}
    assert counts == split_manifest["instances_by_class"]["validation"]
    assert counts != split_manifest["instances_by_class"]["test"]


def test_support_refuses_a_class_with_no_recorded_support(split_manifest):
    with pytest.raises(ComparisonError, match="no recorded validation support"):
        class_support(split_manifest, {**CLASS_MAP, "invented_class": 5})


def test_support_refuses_a_manifest_without_validation_counts(split_manifest):
    broken = copy.deepcopy(split_manifest)
    del broken["images_with_class"]["validation"]
    with pytest.raises(ComparisonError, match="validation"):
        class_support(broken, CLASS_MAP)


def test_support_derivation_routes_through_the_holdout_guard(split_manifest, monkeypatch):
    # The guard call is the point: support derivation asks permission for the
    # split it is about to read, and the split it asks for is never the holdout.
    import construction_safety_vision.detection_comparison as module

    asked: list[tuple] = []
    original = module.assert_split_allowed

    def recording(split, **kwargs):
        asked.append((split, kwargs.get("allow_test")))
        return original(split, **kwargs)

    monkeypatch.setattr(module, "assert_split_allowed", recording)
    class_support(split_manifest, CLASS_MAP)
    assert asked == [("validation", False)]


def test_the_holdout_guard_refuses_the_protected_split_without_both_opt_ins():
    from construction_safety_vision.splits import assert_split_allowed

    with pytest.raises(HoldoutViolationError):
        assert_split_allowed("test", purpose="phase 7", allow_test=False, env={})
    with pytest.raises(HoldoutViolationError):
        assert_split_allowed("test", purpose="phase 7", allow_test=True, env={})


# --- the selection metric -----------------------------------------------------


def test_supported_macro_is_the_unweighted_mean():
    per_class = {
        "a": {"AP@0.50:0.95": 0.4},
        "b": {"AP@0.50:0.95": 0.6},
        "rare": {"AP@0.50:0.95": 0.0},
    }
    assert supported_macro(per_class, ["a", "b"]) == Decimal("0.5")
    # Unweighted: swapping which class holds which value cannot move the mean.
    swapped = {"a": {"AP@0.50:0.95": 0.6}, "b": {"AP@0.50:0.95": 0.4}}
    assert supported_macro(swapped, ["a", "b"]) == Decimal("0.5")


def test_supported_macro_ignores_the_descriptive_class():
    per_class = {
        "a": {"AP@0.50:0.95": 0.8},
        "rare": {"AP@0.50:0.95": 0.0},
    }
    assert supported_macro(per_class, ["a"]) == Decimal("0.8")


def test_supported_macro_is_independent_of_argument_order():
    per_class = {
        "a": {"AP@0.50:0.95": 0.1},
        "b": {"AP@0.50:0.95": 0.2},
        "c": {"AP@0.50:0.95": 0.3},
    }
    assert supported_macro(per_class, ["c", "a", "b"]) == supported_macro(
        per_class, ["a", "b", "c"]
    )


def test_supported_macro_refuses_an_empty_selection_set():
    with pytest.raises(ComparisonError, match="no class clears"):
        supported_macro({"a": {"AP@0.50:0.95": 0.5}}, [])


def test_supported_macro_refuses_a_missing_value():
    with pytest.raises(ComparisonError, match="has no"):
        supported_macro({"a": {"AP@0.50": 0.5}}, ["a"])
    with pytest.raises(ComparisonError, match="expected a number"):
        supported_macro({"a": {"AP@0.50:0.95": "NOT_EXPOSED_RELIABLY"}}, ["a"])


def test_d0_supported_macro_matches_an_independent_computation(d0_manifest, support):
    selection = supported_class_names(support)
    exact = sum(
        Fraction(str(d0_manifest["per_class_metrics"][name]["AP@0.50:0.95"])) for name in selection
    ) / len(selection)
    assert exact == Fraction(2280567, 4000000)
    record = build_experiment_record(d0_manifest, support)
    assert record.supported_macro == Decimal("0.57014175")
    assert round(float(record.supported_macro), 6) == 0.570142


def test_d0_record_preserves_the_official_all_class_metric(d0_manifest, support):
    record = build_experiment_record(d0_manifest, support)
    reported = d0_manifest["validation_metrics"]
    assert record.all_class_map50_95 == Decimal(str(reported[OFFICIAL_ALL_CLASS_METRIC]))
    for name, value in record.secondary.items():
        assert value == Decimal(str(reported[name]))
    payload = record.as_dict()
    assert payload["metrics"][OFFICIAL_ALL_CLASS_METRIC] == reported[OFFICIAL_ALL_CLASS_METRIC]
    assert payload["metrics"][PRIMARY_SELECTION_METRIC] == 0.570142


def test_the_descriptive_class_is_still_reported_in_full(d0_manifest, support):
    payload = build_experiment_record(d0_manifest, support).as_dict()
    assert "vest_loose" in payload["per_class_metrics"]
    assert set(payload["per_class_metrics"]["vest_loose"]) >= {
        "AP@0.50",
        "AP@0.50:0.95",
        "precision",
        "recall",
    }
    assert payload["descriptive_classes"] == ["vest_loose"]
    assert "vest_loose" not in payload["selection_metric_classes"]


def test_a_record_refuses_a_different_class_map(d0_manifest, support):
    broken = copy.deepcopy(d0_manifest)
    broken["class_map"] = {"person": 0, "helmet_on_head": 1}
    with pytest.raises(ComparisonError, match="class map"):
        build_experiment_record(broken, support)


def test_a_record_refuses_a_missing_per_class_table(d0_manifest, support):
    broken = copy.deepcopy(d0_manifest)
    del broken["per_class_metrics"]["person"]
    with pytest.raises(ComparisonError, match="missing class"):
        build_experiment_record(broken, support)


# --- the margin ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("delta", "expected"),
    [
        ("0.0060", IMPROVED),
        ("0.0050001", IMPROVED),
        # Exactly the margin is equivalence, not an improvement.
        ("0.0050", PRACTICALLY_EQUIVALENT),
        ("0.0049999", PRACTICALLY_EQUIVALENT),
        ("0.0", PRACTICALLY_EQUIVALENT),
        ("-0.0050", PRACTICALLY_EQUIVALENT),
        ("-0.0050001", REGRESSED),
        ("-0.2", REGRESSED),
    ],
)
def test_practical_equivalence_boundary(delta, expected):
    assert classify_delta(Decimal(delta)) == expected


def test_the_margin_is_the_frozen_value():
    assert Decimal("0.005") == PRACTICAL_EQUIVALENCE_MARGIN


# --- selection logic ----------------------------------------------------------


def _record(experiment_id: str, macro: str, d0_manifest: dict, support) -> object:
    """Build a record whose selection metric is a chosen value."""
    manifest = copy.deepcopy(d0_manifest)
    manifest["experiment_id"] = experiment_id
    selection = supported_class_names(support)
    for name in selection:
        manifest["per_class_metrics"][name]["AP@0.50:0.95"] = float(macro)
    return build_experiment_record(manifest, support)


def test_case_a_retains_the_reference_when_nothing_clears_the_margin(d0_manifest, support):
    reference = _record("D0", "0.500000", d0_manifest, support)
    candidates = [
        _record("D1", "0.505000", d0_manifest, support),
        _record("D2", "0.400000", d0_manifest, support),
    ]
    outcome = select(reference, candidates)
    assert outcome.case == CASE_A
    assert outcome.preferred_experiment == "D0"
    assert outcome.efficiency_comparison_required is False


def test_case_a_also_covers_a_regression(d0_manifest, support):
    reference = _record("D0", "0.500000", d0_manifest, support)
    candidates = [_record("D1", "0.100000", d0_manifest, support)]
    assert select(reference, candidates).case == CASE_A


def test_case_b_elects_a_separated_leader(d0_manifest, support):
    reference = _record("D0", "0.500000", d0_manifest, support)
    candidates = [
        _record("D1", "0.600000", d0_manifest, support),
        _record("D2", "0.510000", d0_manifest, support),
    ]
    outcome = select(reference, candidates)
    assert outcome.case == CASE_B
    assert outcome.preferred_experiment == "D1"
    assert outcome.runner_up == "D2"
    assert outcome.leader_separation == Decimal("0.09")


def test_case_c_declares_no_winner_when_the_candidates_tie(d0_manifest, support):
    reference = _record("D0", "0.500000", d0_manifest, support)
    candidates = [
        _record("D1", "0.600000", d0_manifest, support),
        _record("D2", "0.598000", d0_manifest, support),
    ]
    outcome = select(reference, candidates)
    assert outcome.case == CASE_C
    assert outcome.preferred_experiment is None
    assert outcome.efficiency_comparison_required is True
    assert {outcome.leader, outcome.runner_up} == {"D1", "D2"}


def test_case_c_covers_one_clearing_candidate_that_does_not_separate(d0_manifest, support):
    # D1 clears the reference by more than the margin; D2 does not. But the two
    # candidates sit within the margin of each other, so they cannot be ordered.
    reference = _record("D0", "0.500000", d0_manifest, support)
    candidates = [
        _record("D1", "0.506000", d0_manifest, support),
        _record("D2", "0.503000", d0_manifest, support),
    ]
    outcome = select(reference, candidates)
    assert outcome.case == CASE_C
    assert outcome.preferred_experiment is None
    assert outcome.efficiency_comparison_required is True


def test_case_b_when_a_single_candidate_clears_alone(d0_manifest, support):
    reference = _record("D0", "0.500000", d0_manifest, support)
    candidates = [_record("D1", "0.600000", d0_manifest, support)]
    outcome = select(reference, candidates)
    assert outcome.case == CASE_B
    assert outcome.preferred_experiment == "D1"
    assert outcome.leader_separation is None


def test_case_d_takes_priority_and_is_not_read_as_inferiority(d0_manifest, support):
    reference = _record("D0", "0.500000", d0_manifest, support)
    candidates = [_record("D1", "0.900000", d0_manifest, support)]
    outcome = select(reference, candidates, failures=["D2"])
    assert outcome.case == CASE_D
    assert outcome.preferred_experiment is None
    assert outcome.leader is None
    assert "not evidence of model inferiority" in outcome.rationale


def test_selection_is_deterministic_under_input_permutation(d0_manifest, support):
    reference = _record("D0", "0.500000", d0_manifest, support)
    first = _record("D1", "0.600000", d0_manifest, support)
    second = _record("D2", "0.510000", d0_manifest, support)
    assert (
        select(reference, [first, second]).as_dict() == select(reference, [second, first]).as_dict()
    )


def test_the_descriptive_class_cannot_change_the_outcome(d0_manifest, support):
    reference = _record("D0", "0.500000", d0_manifest, support)
    candidate = copy.deepcopy(d0_manifest)
    candidate["experiment_id"] = "D1"
    for name in supported_class_names(support):
        candidate["per_class_metrics"][name]["AP@0.50:0.95"] = 0.5
    # A perfect score on the descriptive class, and it still cannot win.
    for metric in ("AP@0.50", "AP@0.50:0.95", "precision", "recall"):
        candidate["per_class_metrics"]["vest_loose"][metric] = 1.0
    record = build_experiment_record(candidate, support)
    outcome = select(reference, [record])
    assert outcome.case == CASE_A
    assert outcome.preferred_experiment == "D0"


# --- protocol compatibility ---------------------------------------------------


def test_d1_differs_from_d0_only_in_capacity_and_its_weights(matrix, baseline):
    declaration = matrix.declaration("D1")
    reference = baseline.as_dict()
    resolved = resolve_candidate_protocol(reference, declaration)
    verdict = check_protocol_compatibility(
        reference, resolved, declaration, reference_experiment="D0"
    )
    assert verdict.compatible
    assert verdict.observed_differences == ("model", "weight_identifier")
    assert declaration.intentional_fields == ("model",)
    assert declaration.consequential_fields == ("weight_identifier",)
    assert resolved["model"] == "YOLO11s"
    assert resolved["weight_identifier"] == "yolo11s.pt"
    assert resolved["training.imgsz"] == reference["training"]["imgsz"]


def test_d2_differs_from_d0_only_in_imgsz(matrix, baseline):
    declaration = matrix.declaration("D2")
    reference = baseline.as_dict()
    resolved = resolve_candidate_protocol(reference, declaration)
    verdict = check_protocol_compatibility(
        reference, resolved, declaration, reference_experiment="D0"
    )
    assert verdict.compatible
    assert verdict.observed_differences == ("training.imgsz",)
    assert resolved["training.imgsz"] == 768
    assert resolved["model"] == reference["model"]
    assert resolved["weight_identifier"] == reference["weight_identifier"]


@pytest.mark.parametrize(
    "frozen_field",
    [
        "training.epochs",
        "training.batch",
        "training.optimizer",
        "training.patience",
        "training.deterministic",
        "training.augmentation",
        "training.close_mosaic",
        "seed",
        "dataset_config",
        "checkpoint_selection",
    ],
)
def test_an_undeclared_parameter_difference_invalidates_the_comparison(
    matrix, baseline, frozen_field
):
    declaration = matrix.declaration("D2")
    reference = baseline.as_dict()
    resolved = resolve_candidate_protocol(reference, declaration)
    current = resolved[frozen_field]
    resolved[frozen_field] = 12345 if not isinstance(current, str) else f"{current}-changed"
    verdict = check_protocol_compatibility(
        reference, resolved, declaration, reference_experiment="D0"
    )
    assert not verdict.compatible
    assert frozen_field in verdict.undeclared_differences


def test_a_declared_variable_that_was_never_applied_invalidates_the_comparison(matrix, baseline):
    declaration = matrix.declaration("D2")
    reference = baseline.as_dict()
    resolved = resolve_candidate_protocol(reference, declaration)
    resolved["training.imgsz"] = reference["training"]["imgsz"]
    verdict = check_protocol_compatibility(
        reference, resolved, declaration, reference_experiment="D0"
    )
    assert not verdict.compatible
    assert verdict.unapplied_declarations == ("training.imgsz",)


def test_an_override_of_an_unknown_field_is_refused(matrix, baseline):
    declaration = matrix.declaration("D2")
    broken = type(declaration)(
        **{
            **declaration.__dict__,
            "overrides": {"training.does_not_exist": 1},
            "intentional_fields": ("training.does_not_exist",),
        }
    )
    with pytest.raises(ComparisonError, match="does not declare"):
        resolve_candidate_protocol(baseline.as_dict(), broken)


def test_runtime_and_identity_fields_are_not_protocol_differences(baseline):
    reference = baseline.as_dict()
    candidate = {
        **reference,
        "experiment_id": "D9",
        "description": "a different question",
        "created_at": "2026-01-01T00:00:00+00:00",
        "save_dir": "artifacts/detection/D9",
    }
    assert protocol_differences(reference, candidate) == ()


def test_flattening_is_idempotent(baseline):
    once = flatten_protocol(baseline.as_dict())
    assert flatten_protocol(once) == once


# --- fingerprint refusals -----------------------------------------------------


def _comparison_inputs(matrix, d0_manifest, support):
    reference = build_experiment_record(d0_manifest, support)
    candidate_manifest = copy.deepcopy(d0_manifest)
    candidate_manifest["experiment_id"] = "D1"
    return reference, candidate_manifest


def test_a_split_fingerprint_mismatch_refuses_the_comparison(matrix, d0_manifest, support):
    reference, candidate_manifest = _comparison_inputs(matrix, d0_manifest, support)
    candidate_manifest["split_reference"]["split_assignment_sha256"] = "0" * 64
    candidate = build_experiment_record(candidate_manifest, support)
    with pytest.raises(ComparisonError, match="split_assignment_sha256"):
        compare(matrix, {"D0": reference, "D1": candidate})


def test_an_adapter_fingerprint_mismatch_refuses_the_comparison(matrix, d0_manifest, support):
    reference, candidate_manifest = _comparison_inputs(matrix, d0_manifest, support)
    candidate_manifest["adapter_manifest_sha256"] = "1" * 64
    candidate = build_experiment_record(candidate_manifest, support)
    with pytest.raises(ComparisonError, match="adapter_manifest_sha256"):
        compare(matrix, {"D0": reference, "D1": candidate})


def test_a_label_fingerprint_mismatch_refuses_the_comparison(matrix, d0_manifest, support):
    reference, candidate_manifest = _comparison_inputs(matrix, d0_manifest, support)
    candidate_manifest["dataset_fingerprints"]["yolo_label_sha256"]["train"] = "2" * 64
    candidate = build_experiment_record(candidate_manifest, support)
    with pytest.raises(ComparisonError, match="yolo_label_sha256"):
        compare(matrix, {"D0": reference, "D1": candidate})


def test_a_class_map_fingerprint_mismatch_refuses_the_comparison(matrix, d0_manifest, support):
    reference, candidate_manifest = _comparison_inputs(matrix, d0_manifest, support)
    candidate_manifest["dataset_fingerprints"]["class_map_sha256"] = "3" * 64
    candidate = build_experiment_record(candidate_manifest, support)
    with pytest.raises(ComparisonError, match="class_map_sha256"):
        compare(matrix, {"D0": reference, "D1": candidate})


def test_a_missing_fingerprint_refuses_the_comparison(d0_manifest, support):
    broken = copy.deepcopy(d0_manifest)
    del broken["dataset_fingerprints"]["class_map_sha256"]
    with pytest.raises(ComparisonError, match="missing data fingerprint"):
        build_experiment_record(broken, support)


def test_an_undeclared_experiment_cannot_be_compared(matrix, d0_manifest, support):
    reference = build_experiment_record(d0_manifest, support)
    stranger_manifest = copy.deepcopy(d0_manifest)
    stranger_manifest["experiment_id"] = "D9"
    stranger = build_experiment_record(stranger_manifest, support)
    with pytest.raises(ComparisonError, match="not declared in the matrix"):
        compare(matrix, {"D0": reference, "D9": stranger})


def test_an_incompatible_protocol_refuses_the_comparison(matrix, baseline, d0_manifest, support):
    reference = build_experiment_record(d0_manifest, support)
    declaration = matrix.declaration("D2")
    resolved = resolve_candidate_protocol(baseline.as_dict(), declaration)
    resolved["training.epochs"] = 250
    verdict = check_protocol_compatibility(
        baseline.as_dict(), resolved, declaration, reference_experiment="D0"
    )
    with pytest.raises(ComparisonError, match="protocol compatibility failed"):
        compare(matrix, {"D0": reference}, compatibility=[verdict])


def test_comparison_without_the_reference_is_refused(matrix, d0_manifest, support):
    manifest = copy.deepcopy(d0_manifest)
    manifest["experiment_id"] = "D1"
    with pytest.raises(ComparisonError, match="requires the reference record"):
        compare(matrix, {"D1": build_experiment_record(manifest, support)})


def test_comparison_is_deterministic(matrix, d0_manifest, support):
    reference = build_experiment_record(d0_manifest, support)
    first = compare(matrix, {"D0": reference})
    second = compare(matrix, {"D0": reference})
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)


def test_comparison_reports_an_unexecuted_candidate_as_such(matrix, d0_manifest, support):
    reference = build_experiment_record(d0_manifest, support)
    result = compare(matrix, {"D0": reference})
    statuses = {row["experiment_id"]: row["status"] for row in result["candidates"]}
    assert statuses == {"D1": "FROZEN_NOT_EXECUTED", "D2": "FROZEN_NOT_EXECUTED"}
    assert all(row["metrics"] is None for row in result["candidates"])
    assert result["selection"]["case"] == CASE_A


# --- the matrix configuration -------------------------------------------------


def test_the_matrix_declares_exactly_d0_d1_and_d2(matrix):
    assert [item.experiment_id for item in matrix.experiments] == ["D0", "D1", "D2"]
    assert [item.experiment_id for item in matrix.candidates] == ["D1", "D2"]
    assert matrix.reference_experiment == "D0"
    assert matrix.reference.protocol_source == "configs/detection_baseline.yaml"


def test_the_matrix_freezes_the_metrics_and_the_margin(matrix):
    assert matrix.primary_selection_metric == PRIMARY_SELECTION_METRIC
    assert matrix.official_all_class_metric == OFFICIAL_ALL_CLASS_METRIC
    assert matrix.practical_equivalence_margin == PRACTICAL_EQUIVALENCE_MARGIN
    assert matrix.support_rule == SupportRule()


def test_both_candidates_are_frozen_and_not_executed(matrix):
    assert [item.status for item in matrix.candidates] == [
        "FROZEN_NOT_EXECUTED",
        "FROZEN_NOT_EXECUTED",
    ]


def test_the_matrix_carries_no_reference_to_the_holdout(paths):
    text = (paths.configs / "detection_experiments.yaml").read_text(encoding="utf-8")
    assert "CSVISION_ALLOW_TEST_SPLIT" not in text
    raw = load_experiment_matrix(paths.configs / "detection_experiments.yaml").as_dict()
    assert not contains_forbidden_split(raw)


def test_the_matrix_declares_batch_16_and_stops_on_oom(matrix):
    assert matrix.memory_policy["controlled_batch"] == 16
    assert matrix.memory_policy["classification"] == "MEMORY_CONSTRAINT_REVIEW_REQUIRED"
    forbidden = set(matrix.memory_policy["forbidden_mitigations"])
    assert {"reduce_batch", "auto_batch", "gradient_accumulation", "change_imgsz"} <= forbidden


def test_d1_records_a_weight_fingerprint_requirement(matrix):
    weights = matrix.declaration("D1").pretrained_weights
    assert weights["identifier"] == "yolo11s.pt"
    assert weights["commit_binary"] is False
    assert {"sha256", "size_bytes", "ultralytics_version"} <= set(
        weights["required_before_training"]
    )


def test_the_matrix_fingerprint_is_stable(paths):
    first = load_experiment_matrix(paths.configs / "detection_experiments.yaml")
    second = load_experiment_matrix(paths.configs / "detection_experiments.yaml")
    assert first.fingerprint() == second.fingerprint()


# --- strict parsing -----------------------------------------------------------


@pytest.fixture()
def raw_matrix(paths) -> dict:
    import yaml

    return yaml.safe_load(
        (paths.configs / "detection_experiments.yaml").read_text(encoding="utf-8")
    )


def _parse(raw: dict):
    from construction_safety_vision.detection_comparison import parse_experiment_matrix

    return parse_experiment_matrix(raw)


def test_an_unknown_top_level_key_is_refused(raw_matrix):
    raw_matrix["extra_key"] = 1
    with pytest.raises(ComparisonConfigError, match="unknown key"):
        _parse(raw_matrix)


def test_a_relaxed_support_threshold_is_refused(raw_matrix):
    raw_matrix["support_rule"]["min_validation_positive_images"] = 1
    with pytest.raises(ComparisonConfigError, match="frozen thresholds"):
        _parse(raw_matrix)


def test_a_changed_margin_is_refused(raw_matrix):
    raw_matrix["practical_equivalence_margin_map50_95"] = 0.02
    with pytest.raises(ComparisonConfigError, match="margin"):
        _parse(raw_matrix)


def test_a_replaced_primary_metric_is_refused(raw_matrix):
    raw_matrix["metrics"]["primary_selection"] = "recall_weighted_composite"
    with pytest.raises(ComparisonConfigError, match="primary_selection"):
        _parse(raw_matrix)


def test_dropping_the_official_all_class_metric_is_refused(raw_matrix):
    raw_matrix["metrics"]["official_all_class"] = "supported_macro_map50_95"
    with pytest.raises(ComparisonConfigError, match="official_all_class"):
        _parse(raw_matrix)


def test_an_undeclared_override_is_refused(raw_matrix):
    entry = next(item for item in raw_matrix["experiments"] if item["experiment_id"] == "D2")
    entry["overrides"]["training.epochs"] = 250
    with pytest.raises(ComparisonConfigError, match="do not match the override set"):
        _parse(raw_matrix)


def test_a_declared_field_without_an_override_is_refused(raw_matrix):
    entry = next(item for item in raw_matrix["experiments"] if item["experiment_id"] == "D2")
    entry["intentional_fields"].append("training.batch")
    with pytest.raises(ComparisonConfigError, match="do not match the override set"):
        _parse(raw_matrix)


def test_a_candidate_that_does_not_inherit_the_reference_is_refused(raw_matrix):
    entry = next(item for item in raw_matrix["experiments"] if item["experiment_id"] == "D1")
    entry["inherits"] = "D2"
    with pytest.raises(ComparisonConfigError, match="must inherit from the reference"):
        _parse(raw_matrix)


def test_a_protocol_naming_the_holdout_is_refused(raw_matrix):
    raw_matrix["experiments"][0]["question"] = "test"
    with pytest.raises(ComparisonConfigError, match="protected split"):
        _parse(raw_matrix)


def test_a_duplicate_experiment_id_is_refused(raw_matrix):
    raw_matrix["experiments"].append(dict(raw_matrix["experiments"][-1]))
    with pytest.raises(ComparisonConfigError, match="more than once"):
        _parse(raw_matrix)


def test_an_identity_field_cannot_be_the_experimental_variable(raw_matrix):
    entry = next(item for item in raw_matrix["experiments"] if item["experiment_id"] == "D2")
    entry["intentional_fields"] = ["experiment_id"]
    entry["overrides"] = {"experiment_id": "D2"}
    with pytest.raises(ComparisonConfigError, match="not a protocol field"):
        _parse(raw_matrix)


def test_a_second_reference_is_refused(raw_matrix):
    entry = next(item for item in raw_matrix["experiments"] if item["experiment_id"] == "D1")
    entry["role"] = "REFERENCE_BASELINE"
    with pytest.raises(ComparisonConfigError, match="exactly one experiment"):
        _parse(raw_matrix)


def test_committing_the_pretrained_binary_is_refused(raw_matrix):
    entry = next(item for item in raw_matrix["experiments"] if item["experiment_id"] == "D1")
    entry["pretrained_weights"]["commit_binary"] = True
    with pytest.raises(ComparisonConfigError, match="commit_binary"):
        _parse(raw_matrix)
