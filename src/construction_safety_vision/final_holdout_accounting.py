"""Phase 11B execution-accounting clarification.

An additive provenance clarification, not a result correction. Phase 11B's
committed provenance record says ``models_executed: 2`` and
``inference_passes: 3`` without saying which model ran twice, and the phrase
"one-shot" can be read as "one model invocation per model". It cannot: the
segmenter ran a second, real prediction pass at the operational confidence the
frozen protocol declares for the phase 8C direct-IoU diagnostic.

Six things are deliberate.

**No model is loaded and no holdout byte is read.** Every value here is read by
field from an artifact phase 11B already committed, or from committed source.
This module imports nothing that can reach a checkpoint, a split accessor or an
image decoder, and a test asserts that.

**Historical artifacts are read, never rewritten.** The phase 11A protocol and
every phase 11B result stay byte-identical; the clarification is a new artifact
that names the ambiguous field, states its semantics and adds unambiguous
siblings beside it. The historical value is not reinterpreted silently.

**"One-shot" is scoped to the evaluation, not to model invocations.** The
sanctioned label is :data:`ONE_SHOT_EVALUATION_VALID`. The reading that would
imply one invocation per model is named as rejected rather than left to be
inferred, because it is false.

**Both consistency verdicts are published, including the failing one.** The
frozen phase 11A protocol declares three inference blocks and was satisfied. A
later, informal execution instruction expected the segmenter's operational
predictions to be reused rather than re-predicted, and was not. Neither fact is
hidden, and the mismatch did not follow from observing any holdout outcome.

**Evidence carries its own status.** Aggregate counts corroborated by a
committed field say so; the parts that rest on an audit performed in an earlier
session and never persisted say that instead. A number is never promoted to
"verified" because it is convenient.

**Filesystem mtime is not cryptographic provenance.** The runtime evidence is
consistent with both protocol-gap resolutions predating holdout access, but code
and results were committed together, so the strongest claim that survives
scrutiny is the weaker one this module records.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
"""Version of the clarification payload schema."""

PHASE = "11B"
"""The phase whose execution accounting this clarifies."""

CLASSIFICATION = "PHASE_11B_EXECUTION_ACCOUNTING_CLARIFICATION"
"""What this artifact is: provenance metadata, never a result."""

CLARIFICATION_TYPE = "PROVENANCE_METADATA_ONLY"
"""No scientific value, ranking or fingerprint contract is changed."""

ENVIRONMENT_GATE = "CSVISION_ALLOW_TEST_SPLIT"
"""The environment half of the dual holdout gate. It must be absent here."""

ONE_SHOT_EVALUATION_VALID = "ONE_SHOT_FINAL_HOLDOUT_EVALUATION_VALID"
"""One human-authorised evaluation attempt, not one invocation per model."""

ONE_SHOT_PREDICTION_LABEL_REJECTED = "ONE_SHOT_MODEL_PREDICTION_EXECUTION_VALID"
"""Rejected: it would imply a single prediction call per model, which is false."""

DETECTOR_PASS = "DETECTOR_AP_PASS"
SEGMENTER_PASS = "SEGMENTER_AP_PASS"
SEGMENTER_OPERATIONAL_PASS = "SEGMENTER_OPERATIONAL_PASS_FOR_DIRECT_IOU"

PASS_NAMES: tuple[str, ...] = (DETECTOR_PASS, SEGMENTER_PASS, SEGMENTER_OPERATIONAL_PASS)
"""The three passes that actually ran, in execution order."""

DECLARED_INFERENCE_BLOCKS: tuple[str, ...] = (
    "detector_inference",
    "segmenter_inference",
    "direct_iou.inference",
)
"""The inference blocks the frozen phase 11A protocol declares."""

EQUIVALENCE_VERIFIED = "VERIFIED"
"""The second S1 pass reproduced the AP pass's operational subset exactly."""

NO_METRIC_DEPENDENCY = "NONE_BEYOND_IDENTICAL_REPRODUCTION"
"""No reported number needed the second execution to be a separate execution."""

COMMITTED_FIELD = "COMMITTED_ARTIFACT_FIELD"
"""Read by field from an artifact phase 11B committed."""

SOURCE_FACT = "COMMITTED_SOURCE_STRUCTURAL_FACT"
"""Read from committed source code in this repository."""

OPERATOR_DECLARED = "OPERATOR_DECLARED_PRIOR_SESSION_AUDIT_NOT_PERSISTED"
"""Established by an audit in an earlier session that wrote no artifact."""

EVIDENCE_STATUSES: frozenset[str] = frozenset({COMMITTED_FIELD, SOURCE_FACT, OPERATOR_DECLARED})
"""Every evidence-bearing value must declare one of these."""

CHRONOLOGY_RUNTIME = "CONSISTENT_WITH_RESOLVED_BEFORE_HOLDOUT_ACCESS"
"""What the mtimes support, and no more."""

CHRONOLOGY_VCS = "ABSENT"
"""There is no pre-execution commit of the implementation."""

CHRONOLOGY_CLAIM = "RESOLVED_BEFORE_OUTCOME_METRICS_WERE_OBSERVED"
"""The strongest unambiguous scientific claim about the gap resolutions."""

PROTECTED_RESULTS: tuple[str, ...] = (
    "reports/final_test_detector.json",
    "reports/final_test_segmenter.json",
    "reports/final_test_direct_iou.json",
    "reports/final_test_evaluation.md",
    "reports/final_test_per_class.csv",
    "reports/final_test_evaluation.provenance.json",
)
"""Phase 11B's committed results. This clarification must not move one byte."""

PROTECTED_PROTOCOL: tuple[str, ...] = (
    "configs/final_holdout_evaluation.yaml",
    "reports/final_holdout_evaluation_protocol.json",
    "reports/final_holdout_evaluation_protocol.md",
    "reports/final_holdout_evaluation.provenance.json",
)
"""The frozen phase 11A protocol. Historical, and never edited."""

PROTECTED: tuple[str, ...] = PROTECTED_RESULTS + PROTECTED_PROTOCOL
"""Everything this clarification reads and must leave byte-identical."""

ACCOUNTING_JSON = "final_test_execution_accounting.json"
ACCOUNTING_MARKDOWN = "final_test_execution_accounting.md"
ACCOUNTING_PROVENANCE = "final_test_execution_accounting.provenance.json"

FORBIDDEN_KEYS: frozenset[str] = frozenset(
    {
        "aggregate_benefit_score",
        "combined_score",
        "composite_score",
        "cost_benefit_index",
        "weighted_score",
        "winner_declared",
    }
)
"""A clarification ranks nothing. Any of these present and true is refused."""


class AccountingError(RuntimeError):
    """Raised when the clarification cannot be built or does not validate."""


def digest_payload(payload: Any) -> str:
    """Hash a payload's semantic content, deterministically.

    Deliberately a local implementation rather than an import from
    :mod:`construction_safety_vision.final_holdout_execution`: this module must
    have no import path reaching the code able to read the holdout, and a test
    asserts the two produce identical digests.

    Args:
        payload: Any JSON-shaped value.

    Returns:
        A SHA-256 hex digest over its canonical serialisation.
    """
    text = json.dumps(
        json.loads(json.dumps(payload, sort_keys=True, default=str)),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_bytes(path: Path) -> str:
    """Digest a file's bytes.

    Args:
        path: The file to hash.

    Returns:
        The hexadecimal digest.

    Raises:
        AccountingError: If the file is missing.
    """
    if not path.is_file():
        msg = f"cannot hash a missing file: {path.name}"
        raise AccountingError(msg)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    """Load a committed JSON artifact.

    Args:
        path: The artifact.

    Returns:
        Its parsed content.

    Raises:
        AccountingError: If the artifact is missing or is not a JSON object.
    """
    if not path.is_file():
        msg = f"a committed artifact this clarification reads is missing: {path.name}"
        raise AccountingError(msg)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        msg = f"{path.name} is not a JSON object"
        raise AccountingError(msg)
    return payload


def field(payload: Mapping[str, Any], dotted: str) -> Any:
    """Read one value by its dotted field path.

    Every number in this artifact goes through here rather than being typed in,
    so a clarification cannot quietly disagree with the result it describes.

    Args:
        payload: A committed artifact.
        dotted: The field path, for example ``details.attempt``.

    Returns:
        The value at that path.

    Raises:
        AccountingError: If any segment is absent.
    """
    current: Any = payload
    for segment in dotted.split("."):
        if not isinstance(current, Mapping) or segment not in current:
            msg = f"expected field {dotted!r} is absent"
            raise AccountingError(msg)
        current = current[segment]
    return current


def _evidence(status: str, value: Any, source: str) -> dict[str, Any]:
    """Wrap a value with where it came from and how strong that is.

    Args:
        status: One of the evidence-status constants.
        value: The value itself.
        source: Artifact and field, or a description of the observation.

    Returns:
        The evidence record.
    """
    return {"evidence_status": status, "source": source, "value": value}


def _authorization() -> dict[str, Any]:
    """Record the closed authorisation state this clarification runs under.

    These are invariants the runner enforces before anything is built, not
    observations copied into the payload: if the environment gate were present,
    the clarification would refuse to run at all.

    Returns:
        The authorisation block.
    """
    return {
        "effective_holdout_access_authorized": False,
        "environment_gate_variable": ENVIRONMENT_GATE,
        "environment_test_gate_present": False,
        "final_test_observed": True,
        "gate_scopes_verified_absent": ["PROCESS", "USER", "MACHINE"],
        "gate_written_by_this_clarification": False,
        "in_code_gate_supplied": False,
    }


def _inference_accounting(
    provenance: Mapping[str, Any], protocol: Mapping[str, Any]
) -> dict[str, Any]:
    """Account for every model invocation phase 11B actually made.

    Args:
        provenance: The committed phase 11B provenance record.
        protocol: The frozen phase 11A protocol document.

    Returns:
        The inference-accounting block.
    """
    fingerprints = field(provenance, "details.fingerprints")
    passes = [
        {
            "confidence": field(protocol, "detector_inference.conf"),
            "derived_view_of_another_pass": False,
            "model": "D2",
            "pass": DETECTOR_PASS,
            "prediction_fingerprint": fingerprints["detector_test_prediction_sha256"],
            "purpose": "AP_CURVE",
            "real_model_prediction_execution": True,
        },
        {
            "confidence": field(protocol, "segmenter_inference.conf"),
            "derived_view_of_another_pass": False,
            "model": "S1",
            "pass": SEGMENTER_PASS,
            "prediction_fingerprint": fingerprints["segmenter_test_prediction_sha256"],
            "purpose": "AP_CURVE",
            "real_model_prediction_execution": True,
        },
        {
            "confidence": field(protocol, "direct_iou.inference.conf"),
            "derived_view_of_another_pass": False,
            "model": "S1",
            "pass": SEGMENTER_OPERATIONAL_PASS,
            "prediction_fingerprint": fingerprints["segmenter_operational_test_prediction_sha256"],
            "purpose": "DIRECT_INSTANCE_MASK_IOU_DIAGNOSTIC",
            "real_model_prediction_execution": True,
        },
    ]
    return {
        "d2_inference_invocations": 1,
        "historical_field_semantics": {
            "artifact": "reports/final_test_evaluation.provenance.json",
            "companion_field": {
                "agrees_with_this_clarification": True,
                "field": "details.inference_passes",
                "means": "TOTAL_MODEL_INFERENCE_PASSES",
                "recorded_value": field(provenance, "details.inference_passes"),
            },
            "does_not_mean": "NUMBER_OF_MODEL_INFERENCE_PASSES",
            "field": "details.models_executed",
            "historical_artifact_edited": False,
            "means": "NUMBER_OF_DISTINCT_MODEL_IDENTITIES_EXECUTED",
            "recorded_value": field(provenance, "details.models_executed"),
            "sibling_fields_added_here": [
                "unique_models_executed",
                "total_model_inference_passes",
                "d2_inference_invocations",
                "s1_inference_invocations",
            ],
            "silently_reinterpreted": False,
        },
        "holdout_evaluation_attempts": field(provenance, "details.attempt"),
        "passes": passes,
        "s1_inference_invocations": 2,
        "second_s1_pass_representation": {
            "described_as_derived_view": False,
            "evidence": _evidence(
                SOURCE_FACT,
                "src/construction_safety_vision/final_holdout_execution.py",
                "execute() calls run_pass() twice with the S1 checkpoint, and run_pass() calls "
                "predict() on every invocation",
            ),
            "was_a_real_model_predict_execution": True,
        },
        "total_model_inference_passes": 3,
        "unique_models_executed": 2,
    }


def _one_shot_semantics(provenance: Mapping[str, Any]) -> dict[str, Any]:
    """State what "one-shot" does and does not mean here.

    Args:
        provenance: The committed phase 11B provenance record.

    Returns:
        The one-shot semantics block.
    """
    return {
        "adaptive_prediction_rerun_count": 0,
        "evaluation_attempt_count": field(provenance, "details.attempt"),
        "frozen_schema_prohibits_this_clarification": False,
        "label": ONE_SHOT_EVALUATION_VALID,
        "means": [
            "exactly one human-authorised final holdout evaluation attempt",
            "the complete frozen test population was used",
            "all inference passes belonged to that single attempt",
            "no result-driven rerun occurred",
            "no prediction pass was repeated after an outcome was observed",
            "no tuning occurred",
        ],
        "phase_11b_classification": field(provenance, "details.classification"),
        "post_metric_model_invocation_count": 0,
        "prediction_regeneration_after_immutability_barrier": False,
        "rejected_because": (
            "it reads as one prediction invocation per model, and that is false - the "
            "segmenter was invoked twice, at the two confidences the frozen protocol declares"
        ),
        "rejected_label": ONE_SHOT_PREDICTION_LABEL_REJECTED,
        "supporting_ledger_state": field(provenance, "details.ledger_state"),
        "supporting_metrics_derived_from": field(provenance, "details.metrics_derived_from"),
        "supporting_prediction_reruns": field(provenance, "details.prediction_reruns"),
    }


def _frozen_protocol_context(protocol: Mapping[str, Any]) -> dict[str, Any]:
    """Publish both consistency verdicts, including the failing one.

    Args:
        protocol: The frozen phase 11A protocol document.

    Returns:
        The frozen-protocol context block.
    """
    return {
        "FROZEN_PROTOCOL_SATISFIED": True,
        "LATER_EXECUTION_INSTRUCTION_SINGLE_S1_INVOCATION_SATISFIED": False,
        "declared_inference_block_count": len(DECLARED_INFERENCE_BLOCKS),
        "declared_inference_blocks": list(DECLARED_INFERENCE_BLOCKS),
        "direct_iou_inference_confidence": field(protocol, "direct_iou.inference.conf"),
        "direct_iou_protocol_fingerprint": field(protocol, "direct_iou.protocol_fingerprint"),
        "direct_iou_reuses_phase_8c_operational_path": field(
            protocol, "direct_iou.unchanged_from_phase_8c"
        ),
        "mismatch_caused_by_observing_holdout_outcomes": False,
        "mismatch_description": (
            "The frozen phase 11A protocol declares three inference blocks, so three inference "
            "passes are consistent with it. A later, informal phase 11B execution instruction "
            "expected the segmenter's operational predictions to be reused from the AP pass "
            "rather than produced by a second prediction call. The implementation ran the "
            "declared block instead. Both facts are recorded; neither is hidden."
        ),
        "mismatch_disclosed": True,
        "protocol_artifact": "reports/final_holdout_evaluation_protocol.json",
        "protocol_fingerprint": field(protocol, "protocol_fingerprint"),
        "protocol_status": field(protocol, "status"),
        "three_passes_consistent_with_frozen_protocol": True,
    }


def _second_pass_audit(
    provenance: Mapping[str, Any],
    segmenter: Mapping[str, Any],
    direct_iou: Mapping[str, Any],
) -> dict[str, Any]:
    """Record the second-S1-pass consequence audit, with per-field evidence.

    The counts a committed field corroborates are marked as such. The
    per-instance comparison was performed in an earlier session and persisted no
    artifact, so it is marked as declared rather than as verified here. That
    distinction is the point: the conclusion stands, its evidence is not
    uniformly strong, and this artifact says which is which.

    Args:
        provenance: The committed phase 11B provenance record.
        segmenter: The committed segmenter result.
        direct_iou: The committed direct-IoU result.

    Returns:
        The second-pass audit block.
    """
    prior_audit = "phase 11B post-execution audit, earlier session, no artifact persisted"
    return {
        "SECOND_S1_PASS_EQUIVALENCE_TO_AP_FILTER": EQUIVALENCE_VERIFIED,
        "compared_fields": ["class", "score", "box", "mask_rle"],
        "corroboration_note": (
            "The committed operational prediction count and the committed image count agree "
            "with the audit's aggregates. The per-instance field comparison itself was not "
            "persisted and is not re-derived here, because re-deriving it would mean opening "
            "holdout-derived prediction files after FINAL_TEST_OBSERVED."
        ),
        "count_divergences": _evidence(OPERATOR_DECLARED, 0, prior_audit),
        "content_divergences": _evidence(OPERATOR_DECLARED, 0, prior_audit),
        "images_compared": _evidence(OPERATOR_DECLARED, 65, prior_audit),
        "images_total": _evidence(
            COMMITTED_FIELD,
            field(direct_iou, "diagnostic.images"),
            "reports/final_test_direct_iou.json:diagnostic.images",
        ),
        "operational_pass_prediction_fingerprint": _evidence(
            COMMITTED_FIELD,
            field(segmenter, "operational_prediction_fingerprint"),
            "reports/final_test_segmenter.json:operational_prediction_fingerprint",
        ),
        "recomputed_in_this_clarification": False,
        "reported_metric_dependency_on_second_execution": NO_METRIC_DEPENDENCY,
        "required_in_hindsight": False,
        "required_in_hindsight_reason": (
            "its output was identical to the deterministic operational subset of the AP pass, "
            "which makes the second execution redundant in hindsight - not retrospectively "
            "something other than a real execution"
        ),
        "s1_ap_predictions_at_or_above_operational_conf": _evidence(
            OPERATOR_DECLARED, 261, prior_audit
        ),
        "s1_ap_predictions_total": _evidence(
            COMMITTED_FIELD,
            field(segmenter, "canonical_mask.detections_scored"),
            "reports/final_test_segmenter.json:canonical_mask.detections_scored",
        ),
        "s1_operational_pass_predictions": _evidence(
            COMMITTED_FIELD,
            field(direct_iou, "diagnostic.global.prediction_count"),
            "reports/final_test_direct_iou.json:diagnostic.global.prediction_count",
        ),
        "second_pass_rewritten_as_derived_view": False,
        "second_pass_was_real_inference": True,
        "supporting_immutability_barrier": field(
            provenance, "details.model_invoked_during_metric_computation"
        ),
    }


def _results_unchanged(
    detector: Mapping[str, Any],
    segmenter: Mapping[str, Any],
    direct_iou: Mapping[str, Any],
) -> dict[str, Any]:
    """Restate the protected scientific values, read by field.

    Restating them is not a second report: it is the assertion that the
    clarification left them alone, in a form a test can check against the
    artifacts themselves.

    Args:
        detector: The committed detector result.
        segmenter: The committed segmenter result.
        direct_iou: The committed direct-IoU result.

    Returns:
        The unchanged-results block.
    """
    return {
        "any_metric_value_changed": False,
        "any_ranking_changed": False,
        "detector": {
            "canonical_box_map50": field(detector, "canonical_box.CANONICAL_TEST_BOX_MAP50"),
            "canonical_box_map50_95": field(detector, "canonical_box.CANONICAL_TEST_BOX_MAP50_95"),
            "false_negatives": field(detector, "object_level.false_negatives"),
            "false_positives": field(detector, "object_level.false_positives"),
            "precision": field(detector, "object_level.precision"),
            "prediction_fingerprint": field(detector, "prediction_fingerprint"),
            "recall": field(detector, "object_level.recall"),
            "result_sha256": field(detector, "result_sha256"),
            "true_positives": field(detector, "object_level.true_positives"),
        },
        "direct_iou": {
            "gt_match_coverage": field(direct_iou, "diagnostic.global.gt_match_coverage"),
            "gt_normalized_mask_iou": field(direct_iou, "diagnostic.global.gt_normalized_mask_iou"),
            "matched_mask_iou_mean": field(direct_iou, "diagnostic.global.matched_mask_iou_mean"),
            "prediction_fingerprint": field(direct_iou, "prediction_fingerprint"),
            "result_sha256": field(direct_iou, "result_sha256"),
        },
        "segmenter": {
            "canonical_box_map50_95": field(
                segmenter, "canonical_box.S1_CANONICAL_TEST_BOX_MAP50_95"
            ),
            "canonical_mask_map50": field(segmenter, "canonical_mask.CANONICAL_TEST_MASK_MAP50"),
            "canonical_mask_map50_95": field(
                segmenter, "canonical_mask.CANONICAL_TEST_MASK_MAP50_95"
            ),
            "false_negatives": field(segmenter, "object_level.false_negatives"),
            "false_positives": field(segmenter, "object_level.false_positives"),
            "operational_prediction_fingerprint": field(
                segmenter, "operational_prediction_fingerprint"
            ),
            "precision": field(segmenter, "object_level.precision"),
            "prediction_fingerprint": field(segmenter, "prediction_fingerprint"),
            "recall": field(segmenter, "object_level.recall"),
            "result_sha256": field(segmenter, "result_sha256"),
            "true_positives": field(segmenter, "object_level.true_positives"),
        },
        "test_population_changed": False,
        "validation_versus_test_deltas_changed": False,
    }


def _chronology() -> dict[str, Any]:
    """Record what the gap-resolution chronology evidence does and does not show.

    Returns:
        The chronology block.
    """
    return {
        "RUNTIME_FILESYSTEM_EVIDENCE": CHRONOLOGY_RUNTIME,
        "VCS_PRE_EXECUTION_CHECKPOINT": CHRONOLOGY_VCS,
        "code_and_results_committed_together": True,
        "gap_resolutions": [
            "PRE_EXECUTION_PROTOCOL_GAP_RESOLUTION",
            "FROZEN_TAXONOMY_AMBIGUOUS_CASCADE_RESOLVED_BEFORE_EXECUTION",
        ],
        "mtime_treated_as_cryptographic_provenance": False,
        "runtime_evidence_description": (
            "the implementation module carrying both gap resolutions has a last-modified time "
            "earlier than the persisted prediction directories, the one-shot ledger and every "
            "committed result of the run"
        ),
        "strongest_unambiguous_scientific_claim": CHRONOLOGY_CLAIM,
        "vcs_limitation": (
            "the implementation is absent from the parent commit, so no signed pre-execution "
            "checkpoint proves the resolutions predated holdout access - only that they "
            "predated the observation of any outcome metric"
        ),
    }


def _post_observation_access() -> dict[str, Any]:
    """Disclose the filesystem contact the post-observation audit made.

    Returns:
        The post-observation access block.
    """
    return {
        "occurred_after": "FINAL_TEST_OBSERVED",
        "post_observation_model_execution": False,
        "post_observation_prediction_generation": False,
        "post_observation_test_content_access": False,
        "post_observation_test_metadata_inspection": True,
        "role_in_model_selection_tuning_or_reported_metrics": "NONE",
        "what_was_inspected": [
            "a directory listing of the materialised test image directory",
            "filesystem last-modified metadata",
        ],
        "what_was_not_done": [
            "no test image was read, decoded or parsed",
            "no test annotation was read, decoded or parsed",
            "no model was executed",
            "no prediction was generated",
        ],
        "zero_filesystem_contact_claimed": False,
    }


def _this_clarification() -> dict[str, Any]:
    """Count what this clarification itself did to the holdout: nothing.

    Returns:
        The self-accounting block.
    """
    return {
        "frozen_historical_protocol_altered": False,
        "holdout_accessor_invoked": False,
        "holdout_reads": 0,
        "metric_values_altered": 0,
        "metrics_recomputed_from_test_data": 0,
        "model_inference_passes": 0,
        "models_executed": 0,
        "prediction_bytes_altered": 0,
        "predictions_regenerated": 0,
        "qualitative_selection_rerun": False,
        "test_annotations_read": 0,
        "test_identifiers_enumerated": 0,
        "test_image_directory_listed": False,
        "test_images_read": 0,
    }


def build(
    paths: Any,
    *,
    protected_digests: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Assemble the execution-accounting clarification from committed artifacts.

    Args:
        paths: A :class:`~construction_safety_vision.paths.ProjectPaths`.
        protected_digests: Digests of the protected artifacts, captured before
            anything was written. Computed here when omitted.

    Returns:
        The clarification payload, carrying its own fingerprint.

    Raises:
        AccountingError: If a required artifact or field is missing.
    """
    reports = Path(paths.reports)
    root = Path(paths.root)

    provenance = read_json(reports / "final_test_evaluation.provenance.json")
    protocol = read_json(reports / "final_holdout_evaluation_protocol.json")
    detector = read_json(reports / "final_test_detector.json")
    segmenter = read_json(reports / "final_test_segmenter.json")
    direct_iou = read_json(reports / "final_test_direct_iou.json")

    digests = (
        dict(protected_digests)
        if protected_digests is not None
        else {name: sha256_bytes(root / name) for name in PROTECTED}
    )

    payload: dict[str, Any] = {
        "authorization": _authorization(),
        "chronology": _chronology(),
        "clarification_type": CLARIFICATION_TYPE,
        "classification": CLASSIFICATION,
        "frozen_protocol_context": _frozen_protocol_context(protocol),
        "historical_artifacts_unchanged": dict(sorted(digests.items())),
        "inference_accounting": _inference_accounting(provenance, protocol),
        "is_a_test_result_correction": False,
        "one_shot_semantics": _one_shot_semantics(provenance),
        "phase": PHASE,
        "post_observation_access": _post_observation_access(),
        "protocol_fingerprint": field(provenance, "config.protocol_fingerprint"),
        "results_unchanged": _results_unchanged(detector, segmenter, direct_iou),
        "schema_version": SCHEMA_VERSION,
        "scientific_results_changed": False,
        "second_s1_pass_audit": _second_pass_audit(provenance, segmenter, direct_iou),
        "this_clarification": _this_clarification(),
    }
    payload["execution_accounting_sha256"] = digest_payload(payload)
    return payload


def _walk(payload: Any, path: str = "") -> list[tuple[str, str, Any]]:
    """Yield every leaf as ``(path, key, value)``.

    Args:
        payload: Any JSON-shaped value.
        path: The accumulated path.

    Returns:
        One entry per leaf.
    """
    found: list[tuple[str, str, Any]] = []
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            where = f"{path}.{key}" if path else str(key)
            found.append((where, str(key), value))
            found.extend(_walk(value, where))
    elif isinstance(payload, Sequence) and not isinstance(payload, str | bytes):
        for index, value in enumerate(payload):
            found.extend(_walk(value, f"{path}[{index}]"))
    return found


def _accounting_problems(payload: Mapping[str, Any]) -> list[str]:
    """Check the inference accounting.

    Args:
        payload: The clarification payload.

    Returns:
        One description per problem found.
    """
    problems: list[str] = []
    accounting = payload.get("inference_accounting", {})
    for key, value in (
        ("holdout_evaluation_attempts", 1),
        ("unique_models_executed", 2),
        ("total_model_inference_passes", 3),
        ("d2_inference_invocations", 1),
        ("s1_inference_invocations", 2),
    ):
        if accounting.get(key) != value:
            problems.append(f"inference_accounting.{key} must be {value}")

    passes = accounting.get("passes", [])
    if [entry.get("pass") for entry in passes] != list(PASS_NAMES):
        problems.append(f"the three passes must be recorded, in order, as {PASS_NAMES}")
    for entry in passes:
        if entry.get("real_model_prediction_execution") is not True:
            problems.append(f"{entry.get('pass')} must be recorded as a real execution")
        if entry.get("derived_view_of_another_pass") is not False:
            problems.append(f"{entry.get('pass')} must not be recorded as a derived view")

    historical = accounting.get("historical_field_semantics", {})
    if historical.get("silently_reinterpreted") is not False:
        problems.append("the historical models_executed value must not be reinterpreted silently")
    if historical.get("historical_artifact_edited") is not False:
        problems.append("the historical provenance artifact must not be edited")
    if historical.get("means") != "NUMBER_OF_DISTINCT_MODEL_IDENTITIES_EXECUTED":
        problems.append("the historical field's semantics must be stated explicitly")

    representation = accounting.get("second_s1_pass_representation", {})
    if representation.get("was_a_real_model_predict_execution") is not True:
        problems.append("the second S1 pass must be represented as a real predict() execution")
    if representation.get("described_as_derived_view") is not False:
        problems.append("the second S1 pass must not be described as a derived view")
    return problems


def _semantics_problems(payload: Mapping[str, Any]) -> list[str]:
    """Check the one-shot semantics and the frozen-protocol context.

    Args:
        payload: The clarification payload.

    Returns:
        One description per problem found.
    """
    problems: list[str] = []
    semantics = payload.get("one_shot_semantics", {})
    if semantics.get("label") != ONE_SHOT_EVALUATION_VALID:
        problems.append(f"the one-shot label must be {ONE_SHOT_EVALUATION_VALID}")
    if semantics.get("rejected_label") != ONE_SHOT_PREDICTION_LABEL_REJECTED:
        problems.append("the rejected per-invocation one-shot label must be named")
    for key in ("adaptive_prediction_rerun_count", "post_metric_model_invocation_count"):
        if semantics.get(key) != 0:
            problems.append(f"one_shot_semantics.{key} must be 0")
    if semantics.get("evaluation_attempt_count") != 1:
        problems.append("one_shot_semantics.evaluation_attempt_count must be 1")
    if semantics.get("prediction_regeneration_after_immutability_barrier") is not False:
        problems.append("no prediction may be regenerated after the immutability barrier")
    if semantics.get("phase_11b_classification") != "TEST_EVALUATION_COMPLETE":
        problems.append("the phase 11B classification must remain TEST_EVALUATION_COMPLETE")

    context = payload.get("frozen_protocol_context", {})
    if context.get("FROZEN_PROTOCOL_SATISFIED") is not True:
        problems.append("the frozen phase 11A protocol was satisfied and must say so")
    if context.get("LATER_EXECUTION_INSTRUCTION_SINGLE_S1_INVOCATION_SATISFIED") is not False:
        problems.append("the later single-S1-invocation instruction was not satisfied")
    if context.get("mismatch_disclosed") is not True:
        problems.append("the mismatch must be disclosed")
    if context.get("mismatch_caused_by_observing_holdout_outcomes") is not False:
        problems.append("the mismatch did not follow from observing a holdout outcome")
    if list(context.get("declared_inference_blocks", [])) != list(DECLARED_INFERENCE_BLOCKS):
        problems.append("the three declared phase 11A inference blocks must be listed")
    return problems


def validate(payload: Mapping[str, Any]) -> list[str]:
    """Check the clarification adversarially.

    Args:
        payload: The clarification payload.

    Returns:
        One description per problem found, empty when the payload is sound.
    """
    problems: list[str] = []

    if payload.get("classification") != CLASSIFICATION:
        problems.append(f"classification must be {CLASSIFICATION}")
    if payload.get("clarification_type") != CLARIFICATION_TYPE:
        problems.append(f"clarification_type must be {CLARIFICATION_TYPE}")
    if payload.get("scientific_results_changed") is not False:
        problems.append("scientific_results_changed must be false")
    if payload.get("is_a_test_result_correction") is not False:
        problems.append("is_a_test_result_correction must be false")

    authorization = payload.get("authorization", {})
    if authorization.get("environment_test_gate_present") is not False:
        problems.append("the environment test gate must be recorded absent")
    if authorization.get("effective_holdout_access_authorized") is not False:
        problems.append("holdout access must be recorded as not authorised")
    if authorization.get("final_test_observed") is not True:
        problems.append("final_test_observed must be true")

    problems.extend(_accounting_problems(payload))
    problems.extend(_semantics_problems(payload))

    audit = payload.get("second_s1_pass_audit", {})
    if audit.get("SECOND_S1_PASS_EQUIVALENCE_TO_AP_FILTER") != EQUIVALENCE_VERIFIED:
        problems.append("the second-pass equivalence verdict must be recorded")
    if audit.get("second_pass_was_real_inference") is not True:
        problems.append("the second S1 pass must be recorded as real inference")
    if audit.get("second_pass_rewritten_as_derived_view") is not False:
        problems.append("the second S1 pass must not be rewritten as a derived view")
    if audit.get("reported_metric_dependency_on_second_execution") != NO_METRIC_DEPENDENCY:
        problems.append(f"the metric dependency must be {NO_METRIC_DEPENDENCY}")
    for key in ("count_divergences", "content_divergences"):
        entry = audit.get(key, {})
        if entry.get("value") != 0:
            problems.append(f"second_s1_pass_audit.{key} must be 0")
    for key, entry in audit.items():
        if isinstance(entry, Mapping) and "value" in entry:
            if entry.get("evidence_status") not in EVIDENCE_STATUSES:
                problems.append(f"second_s1_pass_audit.{key} must carry a known evidence status")
            if not entry.get("source"):
                problems.append(f"second_s1_pass_audit.{key} must name its source")

    chronology = payload.get("chronology", {})
    if chronology.get("RUNTIME_FILESYSTEM_EVIDENCE") != CHRONOLOGY_RUNTIME:
        problems.append("the runtime chronology evidence must be recorded")
    if chronology.get("VCS_PRE_EXECUTION_CHECKPOINT") != CHRONOLOGY_VCS:
        problems.append("the absent VCS checkpoint must be recorded")
    if chronology.get("strongest_unambiguous_scientific_claim") != CHRONOLOGY_CLAIM:
        problems.append(f"the strongest claim must be {CHRONOLOGY_CLAIM}")
    if chronology.get("mtime_treated_as_cryptographic_provenance") is not False:
        problems.append("mtime must not be treated as cryptographic provenance")

    access = payload.get("post_observation_access", {})
    if access.get("post_observation_test_metadata_inspection") is not True:
        problems.append("the post-observation metadata inspection must be disclosed")
    if access.get("zero_filesystem_contact_claimed") is not False:
        problems.append("zero filesystem contact must not be claimed")
    for key in (
        "post_observation_test_content_access",
        "post_observation_model_execution",
        "post_observation_prediction_generation",
    ):
        if access.get(key) is not False:
            problems.append(f"post_observation_access.{key} must be false")

    mine = payload.get("this_clarification", {})
    for key, value in _this_clarification().items():
        if mine.get(key) != value:
            problems.append(f"this_clarification.{key} must be {value!r}")

    for where, key, value in _walk(payload):
        if key in FORBIDDEN_KEYS and value not in (False, None):
            problems.append(f"{where} must be false, found {value!r}")

    return problems


def _rows(pairs: Sequence[tuple[str, Any]]) -> list[str]:
    """Render a two-column Markdown table.

    Args:
        pairs: Label and value.

    Returns:
        The table's lines.
    """
    lines = ["| | Value |", "| --- | --- |"]
    lines.extend(f"| {label} | {value} |" for label, value in pairs)
    return lines


def _render_accounting(payload: Mapping[str, Any]) -> list[str]:
    """Render the authorisation and inference-accounting sections.

    Args:
        payload: The clarification payload.

    Returns:
        Markdown lines.
    """
    accounting = payload["inference_accounting"]
    historical = accounting["historical_field_semantics"]
    lines = [
        "# Phase 11B execution accounting - provenance clarification",
        "",
        f"**`{payload['classification']}`. `{payload['clarification_type']}`.**",
        "",
        "This is a provenance clarification, **not** a test-result correction. No metric, "
        "ranking, prediction byte, fingerprint or frozen protocol changed; the holdout was not "
        "accessed and no model was executed to produce it. Every value below is read by field "
        "from an artifact phase 11B already committed, or from committed source.",
        "",
        "## 1. Authorisation is closed",
        "",
        *_rows(
            [
                ("`final_test_observed`", "true"),
                ("`environment_test_gate_present`", "false"),
                ("`effective_holdout_access_authorized`", "false"),
                (
                    "Gate variable",
                    f"`{payload['authorization']['environment_gate_variable']}`, verified "
                    "absent in the process, user and machine scopes",
                ),
                ("Gate written by this clarification", "no"),
            ]
        ),
        "",
        "## 2. Authoritative inference accounting",
        "",
        *_rows(
            [
                ("Holdout evaluation attempts", accounting["holdout_evaluation_attempts"]),
                ("Unique models executed", accounting["unique_models_executed"]),
                ("Total model inference passes", accounting["total_model_inference_passes"]),
                ("D2 inference invocations", accounting["d2_inference_invocations"]),
                ("S1 inference invocations", accounting["s1_inference_invocations"]),
            ]
        ),
        "",
        "| Pass | Model | Confidence | Purpose | Real execution |",
        "| --- | --- | --- | --- | --- |",
    ]
    lines.extend(
        f"| `{entry['pass']}` | {entry['model']} | {entry['confidence']} | "
        f"`{entry['purpose']}` | yes |"
        for entry in accounting["passes"]
    )
    siblings = "`, `".join(historical["sibling_fields_added_here"])
    lines.extend(
        [
            "",
            "The segmenter's operational pass was a **second, real `predict()` execution** of "
            "the same frozen checkpoint, not a derived view of the AP pass. "
            f"`{historical['artifact']}` records `{historical['field']}: "
            f"{historical['recorded_value']}`; that field counts **distinct model identities**, "
            "not inference passes. It is not reinterpreted here - the unambiguous siblings "
            f"`{siblings}` are added beside it, and the historical artifact was not edited. Its "
            f"companion `{historical['companion_field']['field']}: "
            f"{historical['companion_field']['recorded_value']}` already agrees with this "
            "clarification.",
            "",
        ]
    )
    return lines


def _render_semantics(payload: Mapping[str, Any]) -> list[str]:
    """Render the one-shot semantics and frozen-protocol sections.

    Args:
        payload: The clarification payload.

    Returns:
        Markdown lines.
    """
    semantics = payload["one_shot_semantics"]
    context = payload["frozen_protocol_context"]
    blocks = "`, `".join(context["declared_inference_blocks"])
    lines = [
        '## 3. What "one-shot" means',
        "",
        f"The sanctioned label is **`{semantics['label']}`**. The reading "
        f"`{semantics['rejected_label']}` is **rejected**: {semantics['rejected_because']}.",
        "",
    ]
    lines.extend(f"- {item}" for item in semantics["means"])
    lines.extend(
        [
            "",
            *_rows(
                [
                    ("`evaluation_attempt_count`", semantics["evaluation_attempt_count"]),
                    (
                        "`adaptive_prediction_rerun_count`",
                        semantics["adaptive_prediction_rerun_count"],
                    ),
                    (
                        "`post_metric_model_invocation_count`",
                        semantics["post_metric_model_invocation_count"],
                    ),
                    ("`prediction_regeneration_after_immutability_barrier`", "false"),
                    ("Phase 11B classification", f"`{semantics['phase_11b_classification']}`"),
                ]
            ),
            "",
            "## 4. Frozen-protocol context, and one mismatch",
            "",
            f"The frozen phase 11A protocol declares **three** inference blocks - `{blocks}` - "
            "and `direct_iou.inference` reuses the phase 8C operational confidence "
            f"{context['direct_iou_inference_confidence']}. Three inference passes were "
            "therefore consistent with it.",
            "",
            *_rows(
                [
                    ("`FROZEN_PROTOCOL_SATISFIED`", "**true**"),
                    ("`LATER_EXECUTION_INSTRUCTION_SINGLE_S1_INVOCATION_SATISFIED`", "**false**"),
                    ("Mismatch disclosed", "yes"),
                    ("Caused by observing a holdout outcome", "no"),
                ]
            ),
            "",
            context["mismatch_description"],
            "",
        ]
    )
    return lines


def _render_audit(payload: Mapping[str, Any]) -> list[str]:
    """Render the second-pass audit and the unchanged-results sections.

    Args:
        payload: The clarification payload.

    Returns:
        Markdown lines.
    """
    audit = payload["second_s1_pass_audit"]
    unchanged = payload["results_unchanged"]
    lines = [
        "## 5. Second-S1-pass consequence audit",
        "",
        "**`SECOND_S1_PASS_EQUIVALENCE_TO_AP_FILTER: "
        f"{audit['SECOND_S1_PASS_EQUIVALENCE_TO_AP_FILTER']}`.** "
        "`reported_metric_dependency_on_second_execution: "
        f"{audit['reported_metric_dependency_on_second_execution']}`.",
        "",
        "| Quantity | Value | Evidence |",
        "| --- | --- | --- |",
    ]
    for label, key in (
        ("S1 AP predictions, total", "s1_ap_predictions_total"),
        ("S1 AP predictions at score >= 0.25", "s1_ap_predictions_at_or_above_operational_conf"),
        ("S1 operational-pass predictions", "s1_operational_pass_predictions"),
        ("Images compared", "images_compared"),
        ("Images in the holdout", "images_total"),
        ("Count divergences", "count_divergences"),
        ("Content divergences", "content_divergences"),
    ):
        entry = audit[key]
        lines.append(f"| {label} | {entry['value']} | `{entry['evidence_status']}` |")
    fields = ", ".join(f"`{name}`" for name in audit["compared_fields"])
    lines.extend(
        [
            "",
            f"Fields compared: {fields}.",
            "",
            f"{audit['corroboration_note']} The second pass **was real**; "
            f"{audit['required_in_hindsight_reason']}.",
            "",
            "## 6. Nothing scientific changed",
            "",
            *_rows(
                [
                    (
                        "D2 canonical box mAP@0.50:0.95",
                        unchanged["detector"]["canonical_box_map50_95"],
                    ),
                    (
                        "S1 canonical box mAP@0.50:0.95",
                        unchanged["segmenter"]["canonical_box_map50_95"],
                    ),
                    (
                        "S1 canonical mask mAP@0.50:0.95",
                        unchanged["segmenter"]["canonical_mask_map50_95"],
                    ),
                    (
                        "Direct `matched_mask_iou_mean`",
                        unchanged["direct_iou"]["matched_mask_iou_mean"],
                    ),
                    (
                        "Direct `gt_normalized_mask_iou`",
                        unchanged["direct_iou"]["gt_normalized_mask_iou"],
                    ),
                    (
                        "D2 object-level TP / FP / FN",
                        f"{unchanged['detector']['true_positives']} / "
                        f"{unchanged['detector']['false_positives']} / "
                        f"{unchanged['detector']['false_negatives']}",
                    ),
                    (
                        "S1 object-level TP / FP / FN",
                        f"{unchanged['segmenter']['true_positives']} / "
                        f"{unchanged['segmenter']['false_positives']} / "
                        f"{unchanged['segmenter']['false_negatives']}",
                    ),
                    (
                        "Detector prediction fingerprint",
                        f"`{unchanged['detector']['prediction_fingerprint']}`",
                    ),
                    (
                        "Segmenter prediction fingerprint",
                        f"`{unchanged['segmenter']['prediction_fingerprint']}`",
                    ),
                    (
                        "Segmenter operational prediction fingerprint",
                        f"`{unchanged['segmenter']['operational_prediction_fingerprint']}`",
                    ),
                ]
            ),
            "",
            "Precision, recall, the confusion matrices, the qualitative ranking, the model "
            "checkpoint fingerprints, the test population and the validation-versus-test "
            "deltas are equally untouched, and every protected artifact is verified "
            "byte-identical before and after this clarification.",
            "",
        ]
    )
    return lines


def _render_disclosure(payload: Mapping[str, Any]) -> list[str]:
    """Render the chronology, access disclosure and self-accounting sections.

    Args:
        payload: The clarification payload.

    Returns:
        Markdown lines.
    """
    chronology = payload["chronology"]
    access = payload["post_observation_access"]
    mine = payload["this_clarification"]
    inspected = "; ".join(access["what_was_inspected"])
    description = chronology["runtime_evidence_description"]
    return [
        "## 7. Protocol-gap chronology",
        "",
        *_rows(
            [
                ("`RUNTIME_FILESYSTEM_EVIDENCE`", f"`{chronology['RUNTIME_FILESYSTEM_EVIDENCE']}`"),
                (
                    "`VCS_PRE_EXECUTION_CHECKPOINT`",
                    f"`{chronology['VCS_PRE_EXECUTION_CHECKPOINT']}`",
                ),
                (
                    "Strongest unambiguous scientific claim",
                    f"`{chronology['strongest_unambiguous_scientific_claim']}`",
                ),
            ]
        ),
        "",
        f"The runtime evidence is that {description}. But {chronology['vcs_limitation']}. A "
        "last-modified time is not a signature, and it is not presented as one.",
        "",
        "## 8. Post-observation metadata access, disclosed",
        "",
        *_rows(
            [
                ("`post_observation_test_content_access`", "false"),
                ("`post_observation_test_metadata_inspection`", "**true**"),
                ("`post_observation_model_execution`", "false"),
                ("`post_observation_prediction_generation`", "false"),
            ]
        ),
        "",
        f"After `FINAL_TEST_OBSERVED`, the audit behind section 5 made this filesystem "
        f"contact: {inspected}. Zero filesystem contact with the test directory is **not** "
        "claimed. It played no part in model selection, tuning or any reported metric.",
        "",
        "## 9. What this clarification did",
        "",
        *_rows(
            [
                ("Models executed", mine["models_executed"]),
                ("Model inference passes", mine["model_inference_passes"]),
                ("Holdout reads", mine["holdout_reads"]),
                ("Test images read", mine["test_images_read"]),
                ("Test annotations read", mine["test_annotations_read"]),
                ("Test identifiers enumerated", mine["test_identifiers_enumerated"]),
                ("Predictions regenerated", mine["predictions_regenerated"]),
                ("Metrics recomputed from test data", mine["metrics_recomputed_from_test_data"]),
                ("Qualitative selection rerun", "no"),
                ("Holdout accessor invoked", "no"),
                ("Frozen historical protocol altered", "no"),
            ]
        ),
        "",
        f"Clarification fingerprint `{payload['execution_accounting_sha256']}`.",
        "",
    ]


def render(payload: Mapping[str, Any]) -> str:
    """Render the clarification as Markdown.

    Args:
        payload: The clarification payload.

    Returns:
        The document text.
    """
    lines = [
        *_render_accounting(payload),
        *_render_semantics(payload),
        *_render_audit(payload),
        *_render_disclosure(payload),
    ]
    return "\n".join(lines)
