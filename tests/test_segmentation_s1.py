"""Tests for the S1 resolution, arithmetic and validators.

Pure logic only: nothing here trains, infers on real data, loads a checkpoint or
touches the holdout. The point is that every rule S1's result rests on - the
one-variable contract, the margin arithmetic, the cross-metric direction
boundary, and the refusals that keep the native metric from deciding - can be
exercised without a GPU.

Most of these are refusals. The module's job is to make certain later moves
impossible: a second override, a declared variable that never applies, a
mask-only checkpoint selector, a composite score, a manifest that ranks on the
framework's own mask AP, or an artifact that names the protected split.
"""

from __future__ import annotations

import copy
from typing import Any

import pytest

from construction_safety_vision.canonical_evaluation import (
    ALL_CLASS_METRIC,
    ALL_CLASS_METRIC_50,
    BELOW,
    EQUIVALENT,
    IMPROVES,
    NATIVE_METRIC_STATUS,
    PRACTICAL_EQUIVALENCE_MARGIN,
    PRIMARY_METRIC,
    classify_delta,
)
from construction_safety_vision.segmentation_comparison import (
    CHECKPOINT_POLICY,
    ONE_VARIABLE_FIELD,
    REFERENCE,
)
from construction_safety_vision.segmentation_s1 import (
    CANONICAL_PRIMARY_METRIC,
    COMPLETE,
    CONSISTENT,
    DESCRIPTIVE_NATIVE_TARGET_METRIC,
    DISAGREEMENT,
    EXPERIMENT_ID,
    FINAL_SEGMENTER,
    HOLDOUT_STATUS,
    MANIFEST_SCHEMA_VERSION,
    NATIVE_TARGET_METRIC,
    PHASE,
    RARE_CLASS,
    RARE_CLASS_STATUS,
    SegmentationS1Error,
    aggregate_composition,
    contains_forbidden_split,
    cross_metric_direction,
    experiment_fingerprint,
    per_class_comparison,
    person_diagnostic,
    resolve_candidate_arguments,
    validate_canonical_evaluation,
    validate_comparison,
    validate_direct_iou,
    validate_s1_result_manifest,
    verify_inherited_protocol,
)

REFERENCE_ARGUMENTS: dict[str, Any] = {
    "epochs": 100,
    "imgsz": 768,
    "batch": 8,
    "seed": 42,
    "overlap_mask": True,
    "mask_ratio": 4,
    "optimizer": "auto",
    "patience": 50,
}

CANONICAL_FINGERPRINT = "a" * 64
DIRECT_FINGERPRINT = "b" * 64


# --- the one-variable contract ---------------------------------------------------


def test_resolution_flips_only_the_declared_field() -> None:
    """S1's arguments are S0's with exactly one value changed."""
    resolved = resolve_candidate_arguments(REFERENCE_ARGUMENTS, {ONE_VARIABLE_FIELD: False})
    arguments = resolved["arguments"]

    assert arguments[ONE_VARIABLE_FIELD] is False
    assert resolved["contract"]["one_variable"] is True
    assert resolved["contract"]["inherited_count"] == len(REFERENCE_ARGUMENTS) - 1
    changed = [name for name in REFERENCE_ARGUMENTS if arguments[name] != REFERENCE_ARGUMENTS[name]]
    assert changed == [ONE_VARIABLE_FIELD]


def test_mask_ratio_and_capacity_are_inherited_untouched() -> None:
    """The fields the phase deliberately does not vary come through unchanged."""
    arguments = resolve_candidate_arguments(REFERENCE_ARGUMENTS, {ONE_VARIABLE_FIELD: False})[
        "arguments"
    ]

    assert arguments["mask_ratio"] == 4
    assert arguments["imgsz"] == 768
    assert arguments["batch"] == 8
    assert arguments["seed"] == 42
    assert arguments["epochs"] == 100


def test_a_second_override_is_refused() -> None:
    """Two variables would make the result unattributable, so it cannot be run."""
    with pytest.raises(SegmentationS1Error, match="beyond"):
        resolve_candidate_arguments(
            REFERENCE_ARGUMENTS, {ONE_VARIABLE_FIELD: False, "mask_ratio": 2}
        )


def test_an_override_that_changes_nothing_is_refused() -> None:
    """A variable declared but never applied is not a controlled comparison."""
    with pytest.raises(SegmentationS1Error, match="never applied"):
        resolve_candidate_arguments(REFERENCE_ARGUMENTS, {ONE_VARIABLE_FIELD: True})


def test_inherited_protocol_drift_is_refused() -> None:
    """A drifted inherited field stops the phase rather than being absorbed."""
    frozen = {**REFERENCE_ARGUMENTS, ONE_VARIABLE_FIELD: True}
    resolved = resolve_candidate_arguments(REFERENCE_ARGUMENTS, {ONE_VARIABLE_FIELD: False})[
        "arguments"
    ]
    assert verify_inherited_protocol(resolved, frozen)["one_variable"] is True

    with pytest.raises(SegmentationS1Error, match="drifted"):
        verify_inherited_protocol(resolved, {**frozen, "mask_ratio": 2})


def test_an_intervention_that_failed_to_apply_is_refused() -> None:
    """If the flag did not flip, the run is not S1."""
    frozen = {**REFERENCE_ARGUMENTS, ONE_VARIABLE_FIELD: True}
    with pytest.raises(SegmentationS1Error, match="did not take effect"):
        verify_inherited_protocol(dict(REFERENCE_ARGUMENTS), frozen)


# --- the frozen margin -------------------------------------------------------------


@pytest.mark.parametrize(
    ("candidate", "expected"),
    [
        (0.484643, EQUIVALENT),
        (0.489643, EQUIVALENT),
        (0.479643, EQUIVALENT),
        (0.489644, IMPROVES),
        (0.479642, BELOW),
        (0.600000, IMPROVES),
        (0.300000, BELOW),
    ],
)
def test_margin_boundaries_are_inclusive_at_exactly_the_margin(
    candidate: float, expected: str
) -> None:
    """The band is closed: exactly +-0.005 is practical equivalence, not a win."""
    verdict = classify_delta(0.484643, candidate)

    assert verdict["verdict"] == expected
    assert verdict["margin"] == PRACTICAL_EQUIVALENCE_MARGIN
    assert verdict["margin_is_not_a_significance_test"] is True


def test_practical_equivalence_prefers_the_reference() -> None:
    """A tie was decided in advance in S0's favour."""
    verdict = classify_delta(0.484643, 0.486000)

    assert verdict["verdict"] == EQUIVALENT
    assert verdict["practically_equivalent_prefers_reference"] is True


# --- the cross-metric direction rule ------------------------------------------------


@pytest.mark.parametrize(
    ("primary", "direct", "expected"),
    [
        (0.02, 0.03, CONSISTENT),
        (-0.02, -0.03, CONSISTENT),
        (0.02, -0.03, DISAGREEMENT),
        (-0.02, 0.03, DISAGREEMENT),
        (0.0, 0.03, CONSISTENT),
        (0.02, 0.0, CONSISTENT),
        (0.0, 0.0, CONSISTENT),
    ],
)
def test_direction_requires_strictly_opposite_signs(
    primary: float, direct: float, expected: str
) -> None:
    """A zero delta has no direction, so it cannot disagree with anything."""
    assert cross_metric_direction(primary, direct)["status"] == expected


def test_direction_never_builds_a_composite() -> None:
    """A disagreement is recorded and read, never resolved by a weighted score."""
    result = cross_metric_direction(0.02, -0.03)

    assert result["status"] == DISAGREEMENT
    assert result["composite_created"] is False
    assert result["resolved_by_weighting"] is False
    assert result["ranked_by"] == PRIMARY_METRIC
    assert result["rule_frozen_before_s1"] is True


# --- the predeclared diagnostics ------------------------------------------------------


def test_person_diagnostic_reports_deltas_and_refuses_to_decide() -> None:
    """The person focus is mechanism evidence, never the global decision metric."""
    result = person_diagnostic(
        canonical_reference={"person": {"AP@0.50:0.95": 0.135036, "AP@0.50": 0.309623}},
        canonical_candidate={"person": {"AP@0.50:0.95": 0.200000, "AP@0.50": 0.400000}},
        direct_reference={"person": {"gt_normalized_mask_iou": 0.434635}},
        direct_candidate={"person": {"gt_normalized_mask_iou": 0.500000}},
    )

    assert result["canonical"]["delta_AP@0.50:0.95"] == pytest.approx(0.064964)
    assert result["direct_iou"]["gt_normalized_mask_iou"]["delta"] == pytest.approx(0.065365)
    assert result["may_select_final_model"] is False
    assert result["is_mechanism_evidence_not_the_decision"] is True


def test_person_diagnostic_tolerates_a_missing_figure() -> None:
    """A figure the diagnostic cannot compute is None, never invented."""
    result = person_diagnostic(
        canonical_reference={},
        canonical_candidate={"person": {"AP@0.50:0.95": 0.2}},
        direct_reference={},
        direct_candidate={},
    )

    assert result["canonical"]["delta_AP@0.50:0.95"] is None
    assert result["direct_iou"]["gt_match_coverage"]["delta"] is None


def test_per_class_comparison_keeps_the_rare_class_descriptive() -> None:
    """vest_loose is reported in full and carries its caveat."""
    table = per_class_comparison(
        {"person": {"AP@0.50:0.95": 0.1, "AP@0.50": 0.3}, RARE_CLASS: {"AP@0.50:0.95": 0.001}},
        {"person": {"AP@0.50:0.95": 0.2, "AP@0.50": 0.4}, RARE_CLASS: {"AP@0.50:0.95": 0.500}},
        {"person": {"supported": True}, RARE_CLASS: {"supported": False}},
    )

    assert table["person"]["delta_AP@0.50:0.95"] == pytest.approx(0.1)
    assert table[RARE_CLASS]["status"] == RARE_CLASS_STATUS
    assert "decides nothing" in table[RARE_CLASS]["caveat"]
    assert table[RARE_CLASS]["delta_AP@0.50:0.95"] == pytest.approx(0.499)


# --- the aggregate decomposition ------------------------------------------------------


def test_the_decomposition_names_a_class_that_regressed() -> None:
    """An unweighted mean can rise while a class inside it falls."""
    table = {
        "a": {"delta_AP@0.50:0.95": 0.4},
        "b": {"delta_AP@0.50:0.95": -0.1},
        "c": {"delta_AP@0.50:0.95": 0.02},
        "d": {"delta_AP@0.50:0.95": 0.02},
    }
    result = aggregate_composition(table, ["a", "b", "c", "d"])

    assert result["largest_gain"] == "a"
    assert result["largest_gain_delta"] == pytest.approx(0.4)
    assert result["largest_decline"] == "b"
    assert result["regressed_supported_classes"] == ["b"]


def test_the_contributions_sum_to_the_primary_delta() -> None:
    """Each class contributes its own delta divided by the admitted count."""
    table = {
        "a": {"delta_AP@0.50:0.95": 0.4},
        "b": {"delta_AP@0.50:0.95": -0.1},
        "c": {"delta_AP@0.50:0.95": 0.02},
        "d": {"delta_AP@0.50:0.95": 0.02},
    }
    admitted = ["a", "b", "c", "d"]
    result = aggregate_composition(table, admitted)
    expected = sum(table[name]["delta_AP@0.50:0.95"] for name in admitted) / len(admitted)

    assert sum(result["contribution_to_primary_delta"].values()) == pytest.approx(expected)


def test_the_dominant_share_measures_only_the_gains() -> None:
    """The share is of the total gain, so a decline cannot inflate it."""
    table = {"a": {"delta_AP@0.50:0.95": 0.3}, "b": {"delta_AP@0.50:0.95": 0.1}}
    result = aggregate_composition(table, ["a", "b"])

    assert result["dominant_class_share_of_total_gain"] == pytest.approx(0.75)


def test_the_decomposition_only_covers_admitted_classes() -> None:
    """The rare class contributes to no aggregate, so it is not decomposed."""
    table = {
        "a": {"delta_AP@0.50:0.95": 0.2},
        RARE_CLASS: {"delta_AP@0.50:0.95": 0.9},
    }
    result = aggregate_composition(table, ["a"])

    assert RARE_CLASS not in result["contribution_to_primary_delta"]
    assert result["largest_gain"] == "a"


def test_the_decomposition_survives_a_missing_delta() -> None:
    """A class with no computable delta is omitted, never invented."""
    result = aggregate_composition({"a": {"delta_AP@0.50:0.95": None}}, ["a"])

    assert result["contribution_to_primary_delta"] == {}
    assert result["largest_gain"] is None
    assert result["dominant_class_share_of_total_gain"] is None


# --- the experiment fingerprint ----------------------------------------------------------


def test_experiment_fingerprint_is_deterministic_and_order_independent() -> None:
    """The same semantics hash the same regardless of key order."""
    left = {"experiment_id": EXPERIMENT_ID, "overlap_mask": False, "seed": 42}
    right = {"seed": 42, "overlap_mask": False, "experiment_id": EXPERIMENT_ID}

    assert experiment_fingerprint(left) == experiment_fingerprint(right)


def test_experiment_fingerprint_changes_with_the_intervention() -> None:
    """Flipping the treatment produces a different experiment."""
    base = {"experiment_id": EXPERIMENT_ID, "overlap_mask": False}

    assert experiment_fingerprint(base) != experiment_fingerprint({**base, "overlap_mask": True})


def test_forbidden_split_is_found_at_any_depth() -> None:
    """A protected identifier cannot hide inside a nested structure."""
    assert contains_forbidden_split({"a": [{"b": "test"}]}) is True
    assert contains_forbidden_split({"test": 1}) is True
    assert contains_forbidden_split({"a": ["validation", "train"]}) is False


# --- validator fixtures --------------------------------------------------------------------


def _manifest() -> dict[str, Any]:
    """Build a minimally valid S1 result manifest.

    Returns:
        A manifest the validator accepts.
    """
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "phase": PHASE,
        "experiment_id": EXPERIMENT_ID,
        "status": COMPLETE,
        "reference_experiment": REFERENCE,
        "hypothesis": "a predeclared hypothesis",
        "intentional_difference": {
            "field": ONE_VARIABLE_FIELD,
            "reference_value": True,
            "candidate_value": False,
            "one_variable": True,
        },
        "treatment_semantics": "the treatment changes the target",
        "s1_protocol_fingerprint": "c" * 64,
        "comparison_policy_fingerprint": "d" * 64,
        "canonical_evaluator_fingerprint": CANONICAL_FINGERPRINT,
        "adapter": {"regenerated": False},
        "pretrained_weights": {"sha256": "e" * 64},
        "optimizer": {"actual_resolved_optimizer": "AdamW"},
        "effective_training_configuration": {
            "overlap_mask": False,
            "mask_ratio": 4,
            "imgsz": 768,
            "batch": 8,
        },
        "checkpoint_selection": {
            "policy": CHECKPOINT_POLICY,
            "mask_only_checkpoint_created": False,
        },
        "checkpoints": {"best": {"sha256": "f" * 64}},
        "execution": {"runs": 1},
        "native_metrics": {
            "label": NATIVE_TARGET_METRIC,
            "cross_target_status": NATIVE_METRIC_STATUS,
            "is_primary_for_selection": False,
        },
        "descriptive_native_supported_macro": {
            "label": DESCRIPTIVE_NATIVE_TARGET_METRIC,
            "used_for_selection": False,
            "value": 0.5,
        },
        "canonical_metrics": {ALL_CLASS_METRIC: 0.4, ALL_CLASS_METRIC_50: 0.5},
        "canonical_supported_macro": {
            "metric": PRIMARY_METRIC,
            "label": CANONICAL_PRIMARY_METRIC,
            "value": 0.5,
        },
        "primary_delta": {
            "reference": 0.484643,
            "candidate": 0.5,
            "delta": 0.015357,
            "margin": PRACTICAL_EQUIVALENCE_MARGIN,
            "verdict": IMPROVES,
            "margin_is_not_a_significance_test": True,
        },
        "direct_mask_iou": {"runs": 1},
        "cross_metric_direction": {"status": CONSISTENT, "composite_created": False},
        "person_diagnostic": {"may_select_final_model": False},
        "rare_class": {
            "name": RARE_CLASS,
            "status": RARE_CLASS_STATUS,
            "may_decide_anything": False,
        },
        "resources": {},
        "composite_box_mask_score": False,
        "final_segmenter": FINAL_SEGMENTER,
        "S1_experiment_sha256": "0" * 64,
        "models_trained_in_this_phase": 1,
        "alternative_segmenters_trained": 0,
        "imgsz_variants_tried": 0,
        "batch_variants_tried": 0,
        "thresholds_tuned": 0,
        "s0_retrained": False,
        "s0_revalidated": False,
        "holdout_accessed": False,
        "test": {"status": HOLDOUT_STATUS, "reason": "not accessed"},
    }


def _canonical() -> dict[str, Any]:
    """Build a minimally valid S1 canonical evaluation artifact.

    Returns:
        An artifact the validator accepts.
    """
    return {
        "experiment_id": EXPERIMENT_ID,
        "protocol_fingerprint": CANONICAL_FINGERPRINT,
        "split": "validation",
        "runs": 1,
        "inference": {
            "conf": 0.001,
            "iou": 0.70,
            "imgsz": 768,
            "max_det": 300,
            "augment": False,
            "retina_masks": True,
        },
        "cocoeval": {
            "iou_type": "segm",
            "iou_thresholds": [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95],
            "max_dets": [1, 10, 100],
        },
        "s0_reevaluated": False,
        "holdout_accessed": False,
        "test": {"status": HOLDOUT_STATUS, "reason": "not accessed"},
    }


def _direct() -> dict[str, Any]:
    """Build a minimally valid S1 direct-IoU artifact.

    Returns:
        An artifact the validator accepts.
    """
    return {
        "experiment_id": EXPERIMENT_ID,
        "protocol_fingerprint": DIRECT_FINGERPRINT,
        "split": "validation",
        "runs": 1,
        "inference": {"conf": 0.25, "iou": 0.70, "imgsz": 768, "max_det": 300},
        "ground_truth": {"is_the_yolo_adapter": False},
        "s0_rerun": False,
        "global": {
            "matched_mask_iou_mean": 0.7,
            "gt_normalized_mask_iou": 0.5,
            "gt_match_coverage": 0.8,
            "gt_iou50_coverage": 0.6,
            "gt_iou75_coverage": 0.4,
        },
        "holdout_accessed": False,
        "test": {"status": HOLDOUT_STATUS, "reason": "not accessed"},
    }


# --- the result-manifest validator ------------------------------------------------------------


def test_a_well_formed_manifest_validates() -> None:
    """The fixture is accepted, so every refusal below is about its own change."""
    assert validate_s1_result_manifest(_manifest()) == []


@pytest.mark.parametrize(
    ("path", "value", "fragment"),
    [
        (("effective_training_configuration", "overlap_mask"), True, "overlap_mask false"),
        (("effective_training_configuration", "mask_ratio"), 2, "mask_ratio"),
        (("effective_training_configuration", "imgsz"), 896, "imgsz"),
        (("effective_training_configuration", "batch"), 16, "batch"),
        (("checkpoint_selection", "mask_only_checkpoint_created"), True, "mask-only"),
        (("native_metrics", "is_primary_for_selection"), True, "native metric"),
        (("descriptive_native_supported_macro", "used_for_selection"), True, "native supported"),
        (("cross_metric_direction", "composite_created"), True, "composite"),
        (("person_diagnostic", "may_select_final_model"), True, "person diagnostic"),
        (("rare_class", "may_decide_anything"), True, "rare class"),
        (("execution", "runs"), 2, "exactly one valid S1 run"),
    ],
)
def test_manifest_refusals(path: tuple[str, str], value: Any, fragment: str) -> None:
    """Each protocol violation is caught rather than published."""
    manifest = _manifest()
    manifest[path[0]][path[1]] = value

    assert any(fragment in problem for problem in validate_s1_result_manifest(manifest))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("final_segmenter", "S1"),
        ("composite_box_mask_score", True),
        ("models_trained_in_this_phase", 2),
        ("alternative_segmenters_trained", 1),
        ("thresholds_tuned", 1),
        ("s0_retrained", True),
        ("s0_revalidated", True),
        ("holdout_accessed", True),
        ("status", "SOMETHING_ELSE"),
        ("phase", "8G"),
    ],
)
def test_manifest_top_level_refusals(field: str, value: Any) -> None:
    """A manifest that claims something the phase may not do is refused."""
    manifest = _manifest()
    manifest[field] = value

    assert validate_s1_result_manifest(manifest)


def test_manifest_refuses_the_protected_split() -> None:
    """A holdout identifier outside the declaration is caught."""
    manifest = _manifest()
    manifest["resources"] = {"split": "test"}

    assert any("protected split" in problem for problem in validate_s1_result_manifest(manifest))


def test_manifest_requires_every_reported_field() -> None:
    """A missing required block is named rather than silently tolerated."""
    manifest = _manifest()
    del manifest["direct_mask_iou"]

    problems = validate_s1_result_manifest(manifest)

    assert any("direct_mask_iou" in problem for problem in problems)


# --- the canonical and direct-IoU validators -------------------------------------------------


def test_canonical_artifact_validates_against_the_frozen_evaluator() -> None:
    """The fixture matches the frozen evaluator exactly."""
    assert (
        validate_canonical_evaluation(_canonical(), protocol_fingerprint=CANONICAL_FINGERPRINT)
        == []
    )


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("inference", "conf"), 0.25),
        (("inference", "iou"), 0.5),
        (("inference", "imgsz"), 640),
        (("inference", "max_det"), 100),
        (("inference", "augment"), True),
        (("inference", "retina_masks"), False),
        (("cocoeval", "iou_type"), "bbox"),
        (("cocoeval", "max_dets"), [1, 10, 300]),
        (("cocoeval", "iou_thresholds"), [0.5, 0.75]),
    ],
)
def test_canonical_artifact_refusals(path: tuple[str, str], value: Any) -> None:
    """Any departure from the frozen evaluator is caught."""
    payload = copy.deepcopy(_canonical())
    payload[path[0]][path[1]] = value

    assert validate_canonical_evaluation(payload, protocol_fingerprint=CANONICAL_FINGERPRINT)


def test_canonical_artifact_refuses_a_different_evaluator() -> None:
    """A protocol fingerprint that is not the committed one stops the phase."""
    problems = validate_canonical_evaluation(_canonical(), protocol_fingerprint="9" * 64)

    assert any("fingerprint" in problem for problem in problems)


def test_canonical_artifact_refuses_s0_reevaluation() -> None:
    """S0's canonical numbers are quoted, never recomputed."""
    payload = _canonical()
    payload["s0_reevaluated"] = True

    assert any(
        "S0 must not be re-evaluated" in problem
        for problem in validate_canonical_evaluation(
            payload, protocol_fingerprint=CANONICAL_FINGERPRINT
        )
    )


def test_direct_iou_artifact_validates_against_the_frozen_protocol() -> None:
    """The fixture matches the unchanged phase 8C protocol."""
    assert validate_direct_iou(_direct(), protocol_fingerprint=DIRECT_FINGERPRINT) == []


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("inference", "conf"), 0.001),
        (("inference", "iou"), 0.5),
        (("inference", "imgsz"), 640),
        (("inference", "max_det"), 100),
        (("ground_truth", "is_the_yolo_adapter"), True),
    ],
)
def test_direct_iou_refusals(path: tuple[str, str], value: Any) -> None:
    """The operating point and the ground truth are both fixed."""
    payload = copy.deepcopy(_direct())
    payload[path[0]][path[1]] = value

    assert validate_direct_iou(payload, protocol_fingerprint=DIRECT_FINGERPRINT)


def test_direct_iou_refuses_rerunning_the_reference() -> None:
    """S0's committed diagnostic is read, never re-executed."""
    payload = _direct()
    payload["s0_rerun"] = True

    assert any(
        "must not be re-run" in problem
        for problem in validate_direct_iou(payload, protocol_fingerprint=DIRECT_FINGERPRINT)
    )


def test_the_two_protocols_are_never_mixed() -> None:
    """The canonical conf and the diagnostic conf belong to different protocols."""
    canonical = _canonical()
    canonical["inference"]["conf"] = 0.25
    direct = _direct()
    direct["inference"]["conf"] = 0.001

    assert validate_canonical_evaluation(canonical, protocol_fingerprint=CANONICAL_FINGERPRINT)
    assert validate_direct_iou(direct, protocol_fingerprint=DIRECT_FINGERPRINT)


# --- the comparison validator -------------------------------------------------------------------


def _reference(value: float = 0.484643) -> dict[str, Any]:
    """Build a committed S0 canonical reference.

    Args:
        value: S0's canonical supported macro.

    Returns:
        The reference artifact shape the validator reads.
    """
    return {"supported_macro": {"value": value}}


def test_comparison_recomputes_the_delta_from_the_committed_reference() -> None:
    """The delta is derived, not trusted."""
    assert validate_comparison(_manifest(), s0_reference=_reference()) == []


def test_comparison_catches_a_quoted_reference_that_was_not_used() -> None:
    """A manifest naming a reference other than the committed one is refused."""
    manifest = _manifest()
    manifest["primary_delta"]["reference"] = 0.400000

    problems = validate_comparison(manifest, s0_reference=_reference())

    assert any("committed S0" in problem for problem in problems)


def test_comparison_catches_a_delta_that_does_not_recompute() -> None:
    """Arithmetic is checked rather than accepted."""
    manifest = _manifest()
    manifest["primary_delta"]["delta"] = 0.999

    assert any(
        "does not recompute" in problem
        for problem in validate_comparison(manifest, s0_reference=_reference())
    )


@pytest.mark.parametrize(
    ("candidate", "expected"),
    [(0.5, IMPROVES), (0.486, EQUIVALENT), (0.4, BELOW)],
)
def test_comparison_derives_the_verdict_independently(candidate: float, expected: str) -> None:
    """The recorded verdict must be the one the frozen margin produces."""
    manifest = _manifest()
    reference_value = 0.484643
    manifest["canonical_supported_macro"]["value"] = candidate
    manifest["primary_delta"] = {
        "reference": reference_value,
        "candidate": candidate,
        "delta": round(candidate - reference_value, 6),
        "margin": PRACTICAL_EQUIVALENCE_MARGIN,
        "verdict": expected,
        "margin_is_not_a_significance_test": True,
    }

    assert validate_comparison(manifest, s0_reference=_reference(reference_value)) == []

    manifest["primary_delta"]["verdict"] = "SOMETHING_ELSE"
    assert validate_comparison(manifest, s0_reference=_reference(reference_value))


def test_comparison_requires_the_final_segmenter_to_stay_unselected() -> None:
    """Running an experiment selects nothing."""
    manifest = _manifest()
    manifest["final_segmenter"] = "S1"

    assert any(
        "final_segmenter" in problem
        for problem in validate_comparison(manifest, s0_reference=_reference())
    )
