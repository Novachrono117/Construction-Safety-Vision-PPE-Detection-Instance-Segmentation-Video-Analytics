"""The S1 controlled segmentation experiment: resolution, arithmetic, validators.

Phase 8F. S1 carries no protocol of its own. Its arguments are S0's, read from
``configs/segmentation_baseline.yaml``, with the single override the phase 8E
comparison protocol declares. That resolution happens here rather than in a
script so it can be exercised without a GPU, a dataset or a training run.

Four decisions are fixed in this module rather than taken once numbers exist.

**Only ``overlap_mask`` may differ.** :func:`resolve_candidate_arguments`
re-derives the candidate's argument set from the reference's and refuses any
override outside the declared field, and refuses a declared override that does
not actually change the reference's value.

**The cross-metric direction rule.** The primary canonical AP and the secondary
direct GT-normalised IoU either move the same way or they do not, and the answer
to "they disagree" is to record it and rank by the primary metric. The rule is
written here, before S1 exists: a disagreement requires the two deltas to have
strictly opposite signs, and a delta of exactly zero has no direction and
therefore cannot disagree with anything. Fixing that boundary in advance is the
point; a rule chosen after seeing which way the numbers went would be a
narrative, not a protocol.

**The native metric may not decide.** ``overlap_mask`` changes the framework's
training target *and* its validation ground truth, so S0's and S1's native mask
AP are measured against different targets. :func:`validate_s1_result_manifest`
refuses a manifest that ranks on the native metric or that carries a composite
box-plus-mask score.

**The holdout may not appear.** Every validator refuses an artifact mentioning
the protected split, because a result file is the easiest place for a protected
identifier to leak into a public repository.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from construction_safety_vision.canonical_evaluation import (
    ALL_CLASS_METRIC,
    ALL_CLASS_METRIC_50,
    IOU_THRESHOLDS,
    IOU_TYPE,
    MAX_DETS,
    METRIC_PRECISION,
    NATIVE_METRIC_STATUS,
    PRACTICAL_EQUIVALENCE_MARGIN,
    PRIMARY_METRIC,
    SELECTION_CASES,
)
from construction_safety_vision.data.materialization import digest
from construction_safety_vision.segmentation_comparison import (
    CANDIDATE,
    CHECKPOINT_POLICY,
    ONE_VARIABLE_FIELD,
    REFERENCE,
    ComparisonConfigError,
    verify_one_variable_contract,
)

MANIFEST_SCHEMA_VERSION = 1
"""Schema version of ``reports/segmentation_S1_result_manifest.json``."""

PHASE = "8F"
"""The phase that executes S1."""

EXPERIMENT_ID = CANDIDATE
"""The single authorised candidate."""

RUN_NAME = "S1"
"""Directory name of the run, under ``artifacts/segmentation/``."""

RUNTIME_VIEW_ROOT = "data/processed/adapters/yolo_segmentation_s1_runtime"
"""Git-ignored runtime view of the approved adapter, where caches may live.

S1 gets its own view rather than sharing S0's for one reason: the framework
writes ``.cache`` files beside the labels it scans, and a cache S0 produced is
not evidence that S1's loader saw the approved bytes. The label bytes in both
views are verified identical to the approved digests, so this changes nothing
about the data and everything about what can be claimed afterwards.
"""

COMPLETE = "S1_CONTROLLED_EXPERIMENT_COMPLETE"
"""Classification of a phase that ran exactly one valid S1 experiment."""

MEMORY_CONSTRAINT_REVIEW_REQUIRED = "MEMORY_CONSTRAINT_REVIEW_REQUIRED"
S1_PROTOCOL_INVALID = "S1_PROTOCOL_INVALID"
TRAINING_FAILED = "TRAINING_FAILED"
CANONICAL_EVALUATION_FAILED = "CANONICAL_EVALUATION_FAILED"
DIRECT_IOU_EVALUATION_FAILED = "DIRECT_IOU_EVALUATION_FAILED"
PROTOCOL_VIOLATION = "PROTOCOL_VIOLATION"
BLOCKED = "BLOCKED"
ADAPTER_FINGERPRINT_MISMATCH = "ADAPTER_FINGERPRINT_MISMATCH"

ENGINEERING_ABORT = "ENGINEERING_ABORT_BEFORE_VALID_EXPERIMENT_COMPLETION"
"""Classification of a runner attempt abandoned before any metric was consulted."""

CONSISTENT = "CROSS_METRIC_DIRECTION_CONSISTENT"
DISAGREEMENT = "CROSS_METRIC_DIRECTION_DISAGREEMENT"

FINAL_SEGMENTER = "UNSELECTED_PENDING_REVIEW"
"""No segmenter is selected by running an experiment."""

NATIVE_TARGET_METRIC = "NATIVE_TARGET_METRIC"
"""Label attached to every framework mask metric S1 reports."""

DESCRIPTIVE_NATIVE_TARGET_METRIC = "DESCRIPTIVE_NATIVE_TARGET_METRIC"
"""Label attached to the native supported macro, kept for continuity with S0."""

CANONICAL_PRIMARY_METRIC = "CANONICAL_PRIMARY_METRIC"
"""Label attached to the metric that actually decides."""

RARE_CLASS = "vest_loose"
RARE_CLASS_STATUS = "DESCRIPTIVE_HIGH_UNCERTAINTY"

PERSON_CLASS = "person"
"""The predeclared phase 8D diagnostic focus. Evidence, never the decision."""

HOLDOUT_STATUS = "PROTECTED_NOT_ACCESSED"
FORBIDDEN_SPLIT = "test"

TREATMENT_SEMANTICS = (
    "S1 does not merely change a reporting flag. overlap_mask=false changes the framework's "
    "instance-mask target construction: the loader returns one mask plane per instance instead "
    "of a single indexed map in which the smaller instance owns contested pixels, and "
    "SegmentationValidator._prepare_batch builds its ground truth the same way. That "
    "consequence is part of the S1 treatment, not a side effect to be corrected for. Both "
    "experiments keep ULTRALYTICS_NATIVE_SEGMENTATION_FITNESS for selecting their own best.pt, "
    "so each checkpoint is chosen against its own target - which is precisely why the final "
    "S0-versus-S1 comparison is made externally, against canonical COCO masks that neither "
    "flag can move."
)
"""Recorded in every S1 artifact, and never reinterpreted once results exist."""


class SegmentationS1Error(RuntimeError):
    """Raised when the S1 experiment cannot be resolved or recorded as specified."""


# --- the one-variable contract -------------------------------------------------


def resolve_candidate_arguments(
    reference_arguments: Mapping[str, Any], overrides: Mapping[str, Any]
) -> dict[str, Any]:
    """Derive S1's framework arguments from S0's, applying the single override.

    The candidate's argument set is never written out independently: it is
    computed from the reference's, so a drift in the reference protocol shows up
    as a changed S1 argument rather than being silently masked by a second copy.

    Args:
        reference_arguments: S0's complete framework argument set.
        overrides: The candidate's declared overrides.

    Returns:
        The candidate's arguments and the verified contract, under
        ``arguments`` and ``contract``.

    Raises:
        SegmentationS1Error: If the contract does not hold.
    """
    try:
        contract = verify_one_variable_contract(reference_arguments, overrides)
    except ComparisonConfigError as exc:
        raise SegmentationS1Error(str(exc)) from exc

    arguments = dict(reference_arguments)
    arguments[ONE_VARIABLE_FIELD] = overrides[ONE_VARIABLE_FIELD]

    changed = sorted(
        name
        for name in set(arguments) | set(reference_arguments)
        if arguments.get(name) != reference_arguments.get(name)
    )
    if changed != [ONE_VARIABLE_FIELD]:  # pragma: no cover - defended, not expected
        msg = f"resolving S1 changed more than {ONE_VARIABLE_FIELD!r}: {changed}"
        raise SegmentationS1Error(msg)
    return {"arguments": arguments, "contract": contract}


def verify_inherited_protocol(
    resolved: Mapping[str, Any], frozen: Mapping[str, Any]
) -> dict[str, Any]:
    """Confirm the resolved arguments match what phase 8E recorded as inherited.

    Args:
        resolved: The candidate's resolved framework arguments.
        frozen: The ``inherited_protocol`` block from the phase 8E S1 manifest,
            which records the reference's values before the override.

    Returns:
        What was verified.

    Raises:
        SegmentationS1Error: If any inherited field drifted, or if the
            intentional field failed to flip.
    """
    drifted = [
        f"{name}: phase 8E recorded {frozen[name]!r}, resolved to {resolved.get(name)!r}"
        for name in sorted(frozen)
        if name != ONE_VARIABLE_FIELD and frozen[name] != resolved.get(name)
    ]
    if drifted:
        msg = "inherited fields drifted from the frozen S1 protocol: " + "; ".join(drifted)
        raise SegmentationS1Error(msg)
    if resolved.get(ONE_VARIABLE_FIELD) is not False:
        msg = (
            f"the intentional field did not take effect: {ONE_VARIABLE_FIELD} resolved to "
            f"{resolved.get(ONE_VARIABLE_FIELD)!r}, not False"
        )
        raise SegmentationS1Error(msg)
    return {
        "inherited_fields_verified": sorted(name for name in frozen if name != ONE_VARIABLE_FIELD),
        "inherited_count": len([name for name in frozen if name != ONE_VARIABLE_FIELD]),
        "intentional_field": ONE_VARIABLE_FIELD,
        "intentional_value": False,
        "one_variable": True,
    }


# --- the frozen cross-metric direction rule -------------------------------------


def cross_metric_direction(primary_delta: float, direct_delta: float) -> dict[str, Any]:
    """Compare the direction of the primary and secondary metrics.

    The rule, fixed before S1 existed: the two disagree only when their deltas
    have **strictly opposite signs**. A delta of exactly zero has no direction
    and therefore cannot disagree with anything. Recording that boundary in
    advance is what stops the classification being chosen once the numbers are
    visible.

    A disagreement is recorded and read by a human. It is never resolved by
    combining the two into a weighted score: the primary canonical metric ranks,
    and the direct IoU explains.

    Args:
        primary_delta: S1 minus S0 on the canonical supported macro AP.
        direct_delta: S1 minus S0 on the direct GT-normalised mask IoU.

    Returns:
        The classification and the arithmetic behind it.
    """
    primary = round(float(primary_delta), METRIC_PRECISION)
    direct = round(float(direct_delta), METRIC_PRECISION)
    opposed = (primary > 0 and direct < 0) or (primary < 0 and direct > 0)
    return {
        "primary_metric": PRIMARY_METRIC,
        "primary_delta": primary,
        "secondary_metric": "gt_normalized_mask_iou",
        "secondary_delta": direct,
        "status": DISAGREEMENT if opposed else CONSISTENT,
        "rule": (
            "A disagreement requires strictly opposite signs. A delta of exactly zero has no "
            "direction and does not disagree with anything."
        ),
        "rule_frozen_before_s1": True,
        "ranked_by": PRIMARY_METRIC,
        "composite_created": False,
        "resolved_by_weighting": False,
    }


def person_diagnostic(
    *,
    canonical_reference: Mapping[str, Any],
    canonical_candidate: Mapping[str, Any],
    direct_reference: Mapping[str, Any],
    direct_candidate: Mapping[str, Any],
) -> dict[str, Any]:
    """Compare the predeclared ``person`` focus across the two experiments.

    Phase 8D named ``person`` before S1 existed, so this is a predeclared
    diagnostic rather than a class picked out of the results. It is mechanism
    evidence: it may inform the human reading of *why* the aggregate moved, and
    it may never select the final model on its own.

    Args:
        canonical_reference: S0's canonical per-class block.
        canonical_candidate: S1's canonical per-class block.
        direct_reference: S0's direct-IoU per-class block.
        direct_candidate: S1's direct-IoU per-class block.

    Returns:
        The canonical and direct comparisons for ``person``.
    """

    def delta(candidate: Any, reference: Any) -> Any:
        if isinstance(candidate, (int, float)) and isinstance(reference, (int, float)):
            return round(float(candidate) - float(reference), METRIC_PRECISION)
        return None

    canonical_s0 = dict(canonical_reference.get(PERSON_CLASS, {}))
    canonical_s1 = dict(canonical_candidate.get(PERSON_CLASS, {}))
    direct_s0 = dict(direct_reference.get(PERSON_CLASS, {}))
    direct_s1 = dict(direct_candidate.get(PERSON_CLASS, {}))

    direct_keys = ("gt_normalized_mask_iou", "matched_mask_iou_mean", "gt_match_coverage")
    return {
        "class": PERSON_CLASS,
        "status": "PREDECLARED_PHASE_8D_DIAGNOSTIC_FOCUS",
        "canonical": {
            "S0_AP@0.50:0.95": canonical_s0.get("AP@0.50:0.95"),
            "S1_AP@0.50:0.95": canonical_s1.get("AP@0.50:0.95"),
            "delta_AP@0.50:0.95": delta(
                canonical_s1.get("AP@0.50:0.95"), canonical_s0.get("AP@0.50:0.95")
            ),
            "S0_AP@0.50": canonical_s0.get("AP@0.50"),
            "S1_AP@0.50": canonical_s1.get("AP@0.50"),
            "delta_AP@0.50": delta(canonical_s1.get("AP@0.50"), canonical_s0.get("AP@0.50")),
            "reference_source": "COMMITTED_PHASE_8E_CANONICAL_EVALUATION",
        },
        "direct_iou": {
            name: {
                "S0": direct_s0.get(name),
                "S1": direct_s1.get(name),
                "delta": delta(direct_s1.get(name), direct_s0.get(name)),
            }
            for name in direct_keys
        },
        "direct_reference_source": "COMMITTED_PHASE_8C_DIAGNOSTIC",
        "may_select_final_model": False,
        "is_mechanism_evidence_not_the_decision": True,
        "caveat": (
            "The phase 8D overlap-target analysis that motivated this focus is "
            "POST_HOC_HYPOTHESIS_GENERATING. A movement here is consistent with that mechanism; "
            "it does not confirm it, and one run cannot separate it from run-to-run variance."
        ),
    }


def per_class_comparison(
    reference: Mapping[str, Mapping[str, Any]],
    candidate: Mapping[str, Mapping[str, Any]],
    support: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Build the canonical per-class S0-versus-S1 table.

    Args:
        reference: S0's canonical per-class AP.
        candidate: S1's canonical per-class AP.
        support: The frozen support rule's verdict per class.

    Returns:
        Per class, both experiments' AP, the delta, and its comparison status.
    """
    table: dict[str, dict[str, Any]] = {}
    for name in sorted(set(reference) | set(candidate)):
        s0 = reference.get(name, {})
        s1 = candidate.get(name, {})
        supported = bool(support.get(name, {}).get("supported"))
        row: dict[str, Any] = {
            "supported": supported,
            "status": "COMPARISON_SUPPORTED" if supported else RARE_CLASS_STATUS,
        }
        for key in ("AP@0.50:0.95", "AP@0.50"):
            left, right = s0.get(key), s1.get(key)
            row[f"S0_{key}"] = left
            row[f"S1_{key}"] = right
            row[f"delta_{key}"] = (
                round(float(right) - float(left), METRIC_PRECISION)
                if isinstance(left, (int, float)) and isinstance(right, (int, float))
                else None
            )
        if not supported:
            row["caveat"] = (
                f"{name} does not meet the frozen support rule. Its movement is reported in "
                "full and decides nothing."
            )
        table[name] = row
    return table


def aggregate_composition(
    per_class_table: Mapping[str, Mapping[str, Any]], admitted: Sequence[str]
) -> dict[str, Any]:
    """Decompose the primary delta into the per-class movements that produced it.

    An unweighted mean can rise while a class inside it falls, and the mean on
    its own does not say so. This is arithmetic, not interpretation: each
    admitted class contributes its own delta divided by the number of admitted
    classes, and the largest gain, the largest decline and any regression are
    named so a reader is not left to notice them from a table.

    Args:
        per_class_table: The canonical per-class S0-versus-S1 comparison.
        admitted: The classes the frozen support rule admits.

    Returns:
        The contribution of each admitted class and the extremes.
    """
    contributions: dict[str, float] = {}
    for name in admitted:
        delta = per_class_table.get(name, {}).get("delta_AP@0.50:0.95")
        if isinstance(delta, (int, float)):
            contributions[name] = round(float(delta) / len(admitted), METRIC_PRECISION)

    deltas = {
        name: float(per_class_table[name]["delta_AP@0.50:0.95"])
        for name in admitted
        if isinstance(per_class_table.get(name, {}).get("delta_AP@0.50:0.95"), (int, float))
    }
    regressed = sorted(name for name, value in deltas.items() if value < 0)
    largest_gain = max(deltas, key=lambda name: deltas[name]) if deltas else None
    largest_decline = min(deltas, key=lambda name: deltas[name]) if deltas else None
    dominant_share = None
    total_positive = sum(value for value in deltas.values() if value > 0)
    if largest_gain is not None and total_positive > 0:
        dominant_share = round(deltas[largest_gain] / total_positive, METRIC_PRECISION)

    return {
        "admitted_classes": list(admitted),
        "contribution_to_primary_delta": contributions,
        "largest_gain": largest_gain,
        "largest_gain_delta": round(deltas[largest_gain], METRIC_PRECISION)
        if largest_gain
        else None,
        "largest_decline": largest_decline,
        "largest_decline_delta": round(deltas[largest_decline], METRIC_PRECISION)
        if largest_decline
        else None,
        "regressed_supported_classes": regressed,
        "dominant_class_share_of_total_gain": dominant_share,
        "note": (
            "Arithmetic only. The aggregate moving in one direction does not mean every class "
            "did, and the class carrying most of the movement is named so the mean is not read "
            "as a uniform effect. Why any individual class moved is UNKNOWN: the experiment "
            "varied one flag and measured the outcome, it did not test a per-class mechanism."
        ),
    }


# --- the experiment fingerprint --------------------------------------------------


def experiment_fingerprint(values: Mapping[str, Any]) -> str:
    """Hash S1's semantic identity.

    Covers what would make this a different experiment and nothing that would
    not: no timestamp, no path, no username, no wall-clock. Deterministic, so
    the same protocol over the same data producing the same checkpoint hashes
    the same on any machine.

    Args:
        values: The identity fields.

    Returns:
        A SHA-256 hex digest.
    """
    return digest(json.loads(json.dumps(values, sort_keys=True)))


# --- validators -------------------------------------------------------------------


def contains_forbidden_split(value: Any) -> bool:
    """Report whether a parsed value mentions the protected split.

    Args:
        value: Any parsed value.

    Returns:
        ``True`` when the protected split appears as a string or a key.
    """
    if isinstance(value, str):
        return value.strip().lower() == FORBIDDEN_SPLIT
    if isinstance(value, Mapping):
        return any(
            contains_forbidden_split(key) or contains_forbidden_split(item)
            for key, item in value.items()
        )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return any(contains_forbidden_split(item) for item in value)
    return False


def _holdout_problems(payload: Mapping[str, Any], *, label: str) -> list[str]:
    """Check an artifact declares and honours the holdout policy.

    Args:
        payload: The artifact.
        label: Its name, for the message.

    Returns:
        One description per problem.
    """
    problems: list[str] = []
    body = {key: value for key, value in payload.items() if key != "test"}
    if contains_forbidden_split(body):
        problems.append(f"{label} mentions the protected split outside its holdout declaration")
    if payload.get("test", {}).get("status") != HOLDOUT_STATUS:
        problems.append(f"{label} does not declare test status {HOLDOUT_STATUS!r}")
    if payload.get("holdout_accessed", False):
        problems.append(f"{label} records holdout access")
    return problems


def validate_s1_result_manifest(manifest: Mapping[str, Any]) -> list[str]:
    """Check the S1 result manifest for structural and protocol defects.

    Args:
        manifest: The parsed manifest.

    Returns:
        One description per problem found, empty when the manifest is valid.
    """
    problems: list[str] = []
    required = (
        "schema_version",
        "phase",
        "experiment_id",
        "status",
        "hypothesis",
        "intentional_difference",
        "treatment_semantics",
        "s1_protocol_fingerprint",
        "comparison_policy_fingerprint",
        "canonical_evaluator_fingerprint",
        "adapter",
        "pretrained_weights",
        "optimizer",
        "effective_training_configuration",
        "checkpoint_selection",
        "checkpoints",
        "execution",
        "native_metrics",
        "descriptive_native_supported_macro",
        "canonical_metrics",
        "canonical_supported_macro",
        "primary_delta",
        "direct_mask_iou",
        "cross_metric_direction",
        "person_diagnostic",
        "rare_class",
        "resources",
        "final_segmenter",
        "S1_experiment_sha256",
        "test",
    )
    missing = [name for name in required if name not in manifest]
    if missing:
        problems.append(f"missing required field(s): {', '.join(missing)}")
        return problems

    if int(manifest["schema_version"]) != MANIFEST_SCHEMA_VERSION:
        problems.append(f"schema_version must be {MANIFEST_SCHEMA_VERSION}")
    if manifest["phase"] != PHASE:
        problems.append(f"phase must be {PHASE!r}")
    if manifest["experiment_id"] != EXPERIMENT_ID:
        problems.append(f"experiment_id must be {EXPERIMENT_ID!r}")
    if manifest["status"] != COMPLETE:
        problems.append(f"status must be {COMPLETE!r}")

    difference = manifest["intentional_difference"]
    if difference.get("field") != ONE_VARIABLE_FIELD:
        problems.append(f"intentional_difference.field must be {ONE_VARIABLE_FIELD!r}")
    if (
        difference.get("reference_value") is not True
        or difference.get("candidate_value") is not False
    ):
        problems.append("intentional_difference must record overlap_mask true -> false")
    if not difference.get("one_variable", False):
        problems.append("intentional_difference must assert the one-variable contract")

    effective = manifest["effective_training_configuration"]
    if effective.get(ONE_VARIABLE_FIELD) is not False:
        problems.append("the effective configuration does not record overlap_mask false")
    if effective.get("mask_ratio") != 4:
        problems.append("mask_ratio must remain 4: it is not part of the S1 intervention")
    if effective.get("imgsz") != 768:
        problems.append("imgsz must remain 768")
    if effective.get("batch") != 8:
        problems.append("batch must remain 8")

    selection = manifest["checkpoint_selection"]
    if selection.get("policy") != CHECKPOINT_POLICY:
        problems.append(f"checkpoint_selection.policy must be {CHECKPOINT_POLICY!r}")
    if selection.get("mask_only_checkpoint_created", False):
        problems.append("no mask-only checkpoint selector is authorised")

    native = manifest["native_metrics"]
    if native.get("label") != NATIVE_TARGET_METRIC:
        problems.append(f"native_metrics.label must be {NATIVE_TARGET_METRIC!r}")
    if native.get("cross_target_status") != NATIVE_METRIC_STATUS:
        problems.append(f"native_metrics.cross_target_status must be {NATIVE_METRIC_STATUS!r}")
    if native.get("is_primary_for_selection", False):
        problems.append("the native metric may not be the selection metric")
    if manifest["descriptive_native_supported_macro"].get("label") != (
        DESCRIPTIVE_NATIVE_TARGET_METRIC
    ):
        problems.append(
            f"descriptive_native_supported_macro.label must be {DESCRIPTIVE_NATIVE_TARGET_METRIC!r}"
        )
    if manifest["descriptive_native_supported_macro"].get("used_for_selection", False):
        problems.append("the native supported macro may not select between S0 and S1")

    macro = manifest["canonical_supported_macro"]
    if macro.get("metric") != PRIMARY_METRIC:
        problems.append(f"canonical_supported_macro.metric must be {PRIMARY_METRIC!r}")
    if macro.get("label") != CANONICAL_PRIMARY_METRIC:
        problems.append(f"canonical_supported_macro.label must be {CANONICAL_PRIMARY_METRIC!r}")

    canonical = manifest["canonical_metrics"]
    for name in (ALL_CLASS_METRIC, ALL_CLASS_METRIC_50):
        if name not in canonical:
            problems.append(f"canonical_metrics must carry {name}")

    delta = manifest["primary_delta"]
    if delta.get("margin") != PRACTICAL_EQUIVALENCE_MARGIN:
        problems.append(f"primary_delta.margin must be {PRACTICAL_EQUIVALENCE_MARGIN}")
    if delta.get("verdict") not in SELECTION_CASES:
        problems.append(f"primary_delta.verdict must be one of {list(SELECTION_CASES)}")
    if delta.get("margin_is_not_a_significance_test") is not True:
        problems.append("primary_delta must state the margin is not a significance test")

    direction = manifest["cross_metric_direction"]
    if direction.get("status") not in (CONSISTENT, DISAGREEMENT):
        problems.append(
            f"cross_metric_direction.status must be one of {[CONSISTENT, DISAGREEMENT]}"
        )
    if direction.get("composite_created", False):
        problems.append("no composite score may be created")

    if manifest["person_diagnostic"].get("may_select_final_model", True):
        problems.append("the person diagnostic may not select the final model")

    rare = manifest["rare_class"]
    if rare.get("name") != RARE_CLASS or rare.get("status") != RARE_CLASS_STATUS:
        problems.append(f"rare_class must record {RARE_CLASS!r} as {RARE_CLASS_STATUS!r}")
    if rare.get("may_decide_anything", True):
        problems.append("the rare class may not decide anything")

    if manifest["final_segmenter"] != FINAL_SEGMENTER:
        problems.append(f"final_segmenter must be {FINAL_SEGMENTER!r}")
    if manifest.get("composite_box_mask_score") is not False:
        problems.append("composite_box_mask_score must be false")

    execution = manifest["execution"]
    if int(execution.get("runs", 0)) != 1:
        problems.append("exactly one valid S1 run is authorised")
    if int(manifest.get("models_trained_in_this_phase", 0)) != 1:
        problems.append("exactly one model may be trained in this phase")
    for field in (
        "alternative_segmenters_trained",
        "imgsz_variants_tried",
        "batch_variants_tried",
        "thresholds_tuned",
    ):
        if int(manifest.get(field, 1)) != 0:
            problems.append(f"{field} must be 0")
    if manifest.get("s0_retrained", True) or manifest.get("s0_revalidated", True):
        problems.append("S0 must not be retrained or revalidated in this phase")

    problems.extend(_holdout_problems(manifest, label="the S1 result manifest"))
    return problems


def validate_canonical_evaluation(
    payload: Mapping[str, Any], *, protocol_fingerprint: str
) -> list[str]:
    """Check an S1 canonical evaluation artifact against the frozen evaluator.

    Args:
        payload: The parsed artifact.
        protocol_fingerprint: The committed canonical evaluator fingerprint.

    Returns:
        One description per problem found.
    """
    problems: list[str] = []
    if payload.get("experiment_id") != EXPERIMENT_ID:
        problems.append(f"experiment_id must be {EXPERIMENT_ID!r}")
    if payload.get("protocol_fingerprint") != protocol_fingerprint:
        problems.append("the canonical evaluator fingerprint does not match the committed one")
    if payload.get("split") != "validation":
        problems.append("the canonical evaluation must run on the validation split")
    if int(payload.get("runs", 0)) != 1:
        problems.append("exactly one canonical evaluation is authorised")

    inference = payload.get("inference", {})
    if float(inference.get("conf", -1)) != 0.001:
        problems.append("canonical inference conf must be 0.001")
    if float(inference.get("iou", -1)) != 0.70:
        problems.append("canonical inference NMS IoU must be 0.70")
    if int(inference.get("imgsz", 0)) != 768:
        problems.append("canonical inference imgsz must be 768")
    if int(inference.get("max_det", 0)) != 300:
        problems.append("canonical inference max_det must be 300")
    if inference.get("augment") is not False:
        problems.append("canonical inference must not use augmentation")
    if inference.get("retina_masks") is not True:
        problems.append("canonical inference must use retina_masks")

    evaluator = payload.get("cocoeval", {})
    if evaluator.get("iou_type") != IOU_TYPE:
        problems.append(f"cocoeval.iou_type must be {IOU_TYPE!r}")
    if tuple(round(float(v), 2) for v in evaluator.get("iou_thresholds", ())) != IOU_THRESHOLDS:
        problems.append(f"cocoeval.iou_thresholds must be {list(IOU_THRESHOLDS)}")
    if tuple(int(v) for v in evaluator.get("max_dets", ())) != MAX_DETS:
        problems.append(f"cocoeval.max_dets must be {list(MAX_DETS)}")

    if payload.get("s0_reevaluated", True):
        problems.append("S0 must not be re-evaluated in this phase")
    problems.extend(_holdout_problems(payload, label="the S1 canonical evaluation"))
    return problems


def validate_direct_iou(payload: Mapping[str, Any], *, protocol_fingerprint: str) -> list[str]:
    """Check an S1 direct mask-IoU artifact against the frozen phase 8C protocol.

    Args:
        payload: The parsed artifact.
        protocol_fingerprint: The committed phase 8C protocol fingerprint.

    Returns:
        One description per problem found.
    """
    problems: list[str] = []
    if payload.get("experiment_id") != EXPERIMENT_ID:
        problems.append(f"experiment_id must be {EXPERIMENT_ID!r}")
    if payload.get("protocol_fingerprint") != protocol_fingerprint:
        problems.append("the direct-IoU protocol fingerprint is not the frozen phase 8C one")
    if payload.get("split") != "validation":
        problems.append("the direct-IoU diagnostic must run on the validation split")
    if int(payload.get("runs", 0)) != 1:
        problems.append("exactly one direct-IoU diagnostic is authorised")

    inference = payload.get("inference", {})
    if float(inference.get("conf", -1)) != 0.25:
        problems.append("the direct-IoU operating point is conf 0.25 and was not swept")
    if float(inference.get("iou", -1)) != 0.70:
        problems.append("direct-IoU NMS IoU must be 0.70")
    if int(inference.get("imgsz", 0)) != 768:
        problems.append("direct-IoU imgsz must be 768")
    if int(inference.get("max_det", 0)) != 300:
        problems.append("direct-IoU max_det must be 300")

    if payload.get("ground_truth", {}).get("is_the_yolo_adapter", True):
        problems.append("the direct-IoU diagnostic must score against canonical COCO masks")
    if payload.get("s0_rerun", True):
        problems.append("S0's direct-IoU diagnostic must not be re-run")

    required = (
        "matched_mask_iou_mean",
        "gt_normalized_mask_iou",
        "gt_match_coverage",
        "gt_iou50_coverage",
        "gt_iou75_coverage",
    )
    missing = [name for name in required if name not in payload.get("global", {})]
    if missing:
        problems.append(f"the direct-IoU result is missing: {', '.join(missing)}")

    problems.extend(_holdout_problems(payload, label="the S1 direct-IoU diagnostic"))
    return problems


def validate_comparison(
    manifest: Mapping[str, Any], *, s0_reference: Mapping[str, Any]
) -> list[str]:
    """Check the S0-versus-S1 comparison recomputes from the committed reference.

    The delta is recomputed here from the committed S0 value rather than read
    out of the manifest, so a manifest that quotes a reference it did not use is
    caught.

    Args:
        manifest: The parsed S1 result manifest.
        s0_reference: The committed phase 8E S0 canonical evaluation.

    Returns:
        One description per problem found.
    """
    problems: list[str] = []
    delta = manifest.get("primary_delta", {})
    reference = s0_reference.get("supported_macro", {}).get("value")
    candidate = manifest.get("canonical_supported_macro", {}).get("value")

    if reference is None or candidate is None:
        problems.append("the comparison is missing the reference or candidate primary metric")
        return problems

    if round(float(delta.get("reference", -1)), METRIC_PRECISION) != round(
        float(reference), METRIC_PRECISION
    ):
        problems.append(
            f"primary_delta.reference {delta.get('reference')!r} is not the committed S0 "
            f"canonical supported macro {reference!r}"
        )
    if round(float(delta.get("candidate", -1)), METRIC_PRECISION) != round(
        float(candidate), METRIC_PRECISION
    ):
        problems.append("primary_delta.candidate is not S1's own canonical supported macro")

    recomputed = round(float(candidate) - float(reference), METRIC_PRECISION)
    if round(float(delta.get("delta", -99)), METRIC_PRECISION) != recomputed:
        problems.append(f"primary_delta.delta does not recompute: expected {recomputed}")

    if recomputed > PRACTICAL_EQUIVALENCE_MARGIN:
        expected = SELECTION_CASES[0]
    elif recomputed < -PRACTICAL_EQUIVALENCE_MARGIN:
        expected = SELECTION_CASES[2]
    else:
        expected = SELECTION_CASES[1]
    if delta.get("verdict") != expected:
        problems.append(f"primary_delta.verdict must be {expected!r} for delta {recomputed}")

    if manifest.get("reference_experiment") != REFERENCE:
        problems.append(f"reference_experiment must be {REFERENCE!r}")
    if manifest.get("final_segmenter") != FINAL_SEGMENTER:
        problems.append(f"final_segmenter must remain {FINAL_SEGMENTER!r}")
    return problems
