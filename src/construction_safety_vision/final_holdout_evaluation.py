"""The frozen protocol for the project's single holdout evaluation.

Phase 11A writes down, in advance, everything phase 11B will do to the test
split: which checkpoints, at which settings, judged by which evaluator, with
which confusion-matrix semantics, which failure states, and which qualitative
examples. Then it stops. **No holdout data is read here, and nothing in this
module can read any.**

Six things are deliberate.

**Parsing is strict and the validator is adversarial.** An unknown key raises.
So does a wrong checkpoint digest, a changed confidence, NMS IoU, ``max_det``,
image size or precision, enabled TTA, a swapped evaluator, a changed direct-IoU
protocol, a non-deterministic qualitative rule, a composite score, a declared
winner, a script that could unlock the environment, any test metric, and any
status other than ``FROZEN_NOT_EXECUTED``. Each of those is a way the final
number could be quietly moved after someone had seen it.

**The dual gate is preserved, never re-implemented.**
:func:`authorize_final_holdout_access` delegates to the existing
:func:`~construction_safety_vision.splits.assert_split_allowed`, so there is one
guard in this repository rather than two that can drift apart. It adds one
restriction on top: the caller must be the declared final-evaluation runner, so
a development script holding both opt-ins is still refused.

**No code may satisfy its own precondition.** The runner reads
``CSVISION_ALLOW_TEST_SPLIT``; nothing in this repository writes it. A test
asserts that neither this module nor the runner assigns to ``os.environ``.

**The one-shot ledger is an append-only state machine.** Its transitions are
enumerated, illegal transitions raise, and there is no way to reset the attempt
counter. A second run is a numbered, justified attempt or it does not happen.

**A write failure is not a prediction failure.** If a complete fingerprinted
prediction set exists and only the report failed, the recovery is to rebuild
metrics from the persisted predictions - never to run a model again. The two
cases have separate names and separate policies, settled before either can
occur.

**The confusion-matrix semantics are read from the installed framework, not
assumed.** ``DetectionValidator`` sets its matrix confidence to 0.25 and calls
``ConfusionMatrix.process_batch`` without ``iou_thres``, so the signature
default of 0.45 applies; matching is class-agnostic IoU with the class pair then
recorded, and the matrix is ``(nc+1, nc+1)`` with rows predicted and columns
ground truth. Those are the values that produced every committed validation
matrix in this repository, so freezing them changes nothing and invents nothing.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from construction_safety_vision.splits import (
    HOLDOUT_UNLOCK_ENV_VAR,
    Split,
    assert_split_allowed,
    holdout_unlocked,
)

PHASE = "11A"
EXECUTES_IN_PHASE = "11B"

PROTOCOL = "FINAL_HOLDOUT_ONE_SHOT_EVALUATION"
FROZEN_NOT_EXECUTED = "FROZEN_NOT_EXECUTED"

PROTOCOL_FROZEN = "FINAL_HOLDOUT_EVALUATION_PROTOCOL_FROZEN"
MODEL_IDENTITY_MISMATCH = "MODEL_IDENTITY_MISMATCH"
INVALID_HOLDOUT_PROTOCOL = "INVALID_HOLDOUT_PROTOCOL"
HOLDOUT_ACCESS_VIOLATION = "HOLDOUT_ACCESS_VIOLATION"
BLOCKED = "BLOCKED"

HOLDOUT_STATUS = "PROTECTED_NOT_ACCESSED"

CONFIG_PATH = "configs/final_holdout_evaluation.yaml"

ENVIRONMENT_GATE = HOLDOUT_UNLOCK_ENV_VAR
"""Re-exported from the single guard so artifacts name one variable, not two."""

CODE_GATE = "allow_test=True"

# --- the frozen model identities ----------------------------------------------------------

DETECTOR_EXPERIMENT = "D2"
SEGMENTER_EXPERIMENT = "S1"

DETECTOR_CHECKPOINT_SHA256 = "0466f872a9de22898c70d8834cf1fcfb3d77f6c9fb27e81ee6248d3d19cbc206"
SEGMENTER_CHECKPOINT_SHA256 = "29337d671459d0f742fb713613cc41d0eafcb8a21f5846ac0da65b2ffc024f20"
FINAL_SEGMENTER_SHA256 = "63ef41961ba3fedfef46658754940fb6ce075cdd93087b926d7c6620402579a7"
FINAL_DETECTOR_SHA256 = "84d30d64a1e7ebfc4e4763643ef3a3b5c8a5ce3cedc74bbefdb8031d48235a6e"

IMGSZ = 768
"""Both models. Frozen since phase 7D and 8G; never swept on the holdout."""

# --- the two confidences, which are different things ---------------------------------------

AP_CONF = 0.001
"""Average precision integrates over the score curve and needs its tail."""

OPERATIONAL_CONF = 0.25
"""The project's only operating point. Frozen since phase 10A, never tuned."""

NMS_IOU = 0.70
MAX_DET = 300
PRECISION = "FP32"
QUANTIZE = 32

# --- evaluators ------------------------------------------------------------------------------

COCOEVAL = "pycocotools.cocoeval.COCOeval"
IOU_TYPE_BBOX = "bbox"
IOU_TYPE_SEGM = "segm"
IOU_THRESHOLDS: tuple[float, ...] = (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95)
MAX_DETS: tuple[int, ...] = (1, 10, 100)

DIRECT_IOU_PROTOCOL_FINGERPRINT = "b912039ca77b36959f74fcdbaed109dbf5e3bd707709556f95a19085c7f34d80"
"""Phase 8C's frozen diagnostic, reused unchanged. No new rule is invented."""

CANONICAL_EVALUATOR_FINGERPRINT = "282eb0ec125c47687340325266a2ea397bc0f85e239290a26a5c1ec8d8721e64"
"""Phase 8E's canonical mask-AP evaluator."""

# --- the confusion matrix, read from the installed source -----------------------------------

CONFUSION_MATRIX_IMPLEMENTATION = "ultralytics.utils.metrics.ConfusionMatrix"
CONFUSION_MATRIX_CONF = 0.25
CONFUSION_MATRIX_IOU = 0.45
CONFUSION_MATRIX_ORIENTATION = "ROWS_ARE_PREDICTED_COLUMNS_ARE_GROUND_TRUTH"
CONFUSION_MATRIX_MATCHING = "CLASS_AGNOSTIC_IOU_THEN_CLASS_PAIR_RECORDED"

CLASSES: tuple[str, ...] = (
    "helmet_loose",
    "helmet_on_head",
    "person",
    "vest_loose",
    "vest_on_body",
)
"""Canonical class-map index order. Never collapsed, never reordered."""

RARE_CLASS = "vest_loose"
RARE_CLASS_STATUS = "DESCRIPTIVE_HIGH_UNCERTAINTY"

# --- object-level matching --------------------------------------------------------------------

OBJECT_MATCH_IOU = 0.50

SEGMENTATION_FAILURE_CATEGORIES: tuple[str, ...] = (
    "DETECTION_MISS",
    "CLASSIFICATION_MISMATCH",
    "LOCALIZATION_FAILURE",
    "MASK_QUALITY_FAILURE",
)
"""Evaluated in this order; the first that matches wins."""

# --- qualitative selection -----------------------------------------------------------------------

QUALITATIVE_CATEGORIES: tuple[str, ...] = (
    "HIGHEST_CONFIDENCE_FALSE_POSITIVE",
    "HIGHEST_CONFIDENCE_CLASSIFICATION_MISMATCH",
    "LOWEST_IOU_MATCHED_INSTANCE",
    "LARGEST_MISSED_INSTANCE",
    "SEGMENTATION_UNDER_COVERAGE",
    "SEGMENTATION_OVER_COVERAGE",
)

EXAMPLES_PER_CATEGORY = 3

TIE_BREAKING: tuple[str, ...] = (
    "the ranked quantity itself",
    "canonical_annotation_id ascending",
    "canonical_prediction_index ascending",
    "source_image_id lexicographic ascending",
)
"""Ends in an identifier, so the order is total on any machine."""

# --- the one-shot ledger ----------------------------------------------------------------------

LEDGER_STATES: tuple[str, ...] = (
    "NOT_STARTED",
    "AUTHORIZED",
    "MODEL_IDENTITY_VERIFIED",
    "TEST_LOADED",
    "DETECTOR_PREDICTION_STARTED",
    "DETECTOR_PREDICTION_COMPLETE",
    "SEGMENTER_PREDICTION_STARTED",
    "SEGMENTER_PREDICTION_COMPLETE",
    "PREDICTIONS_PERSISTED",
    "METRICS_COMPUTED",
    "ARTIFACTS_WRITTEN",
    "COMPLETE",
    "FAILED_REQUIRES_HUMAN_REVIEW",
)

TERMINAL_STATES: frozenset[str] = frozenset({"COMPLETE", "FAILED_REQUIRES_HUMAN_REVIEW"})

LEDGER_TRANSITIONS: dict[str, tuple[str, ...]] = {
    "NOT_STARTED": ("AUTHORIZED", "FAILED_REQUIRES_HUMAN_REVIEW"),
    "AUTHORIZED": ("MODEL_IDENTITY_VERIFIED", "FAILED_REQUIRES_HUMAN_REVIEW"),
    "MODEL_IDENTITY_VERIFIED": ("TEST_LOADED", "FAILED_REQUIRES_HUMAN_REVIEW"),
    "TEST_LOADED": ("DETECTOR_PREDICTION_STARTED", "FAILED_REQUIRES_HUMAN_REVIEW"),
    "DETECTOR_PREDICTION_STARTED": (
        "DETECTOR_PREDICTION_COMPLETE",
        "FAILED_REQUIRES_HUMAN_REVIEW",
    ),
    "DETECTOR_PREDICTION_COMPLETE": (
        "SEGMENTER_PREDICTION_STARTED",
        "FAILED_REQUIRES_HUMAN_REVIEW",
    ),
    "SEGMENTER_PREDICTION_STARTED": (
        "SEGMENTER_PREDICTION_COMPLETE",
        "FAILED_REQUIRES_HUMAN_REVIEW",
    ),
    "SEGMENTER_PREDICTION_COMPLETE": ("PREDICTIONS_PERSISTED", "FAILED_REQUIRES_HUMAN_REVIEW"),
    # Once predictions are persisted and fingerprinted, a later failure is a
    # reporting failure. The recovery path re-enters at METRICS_COMPUTED from the
    # persisted artifacts; it never returns to a prediction state.
    "PREDICTIONS_PERSISTED": ("METRICS_COMPUTED", "FAILED_REQUIRES_HUMAN_REVIEW"),
    "METRICS_COMPUTED": ("ARTIFACTS_WRITTEN", "FAILED_REQUIRES_HUMAN_REVIEW"),
    "ARTIFACTS_WRITTEN": ("COMPLETE", "FAILED_REQUIRES_HUMAN_REVIEW"),
    "COMPLETE": (),
    "FAILED_REQUIRES_HUMAN_REVIEW": (),
}

REBUILD_ENTRY_STATE = "PREDICTIONS_PERSISTED"
"""A report rebuild resumes here. It never re-enters a prediction state."""

# --- failure states -----------------------------------------------------------------------------

FAILURE_STATES: tuple[str, ...] = (
    "TEST_EVALUATION_COMPLETE",
    "BLOCKED_MISSING_MODEL_ARTIFACT",
    "MODEL_IDENTITY_MISMATCH",
    "TEST_UNLOCK_AUTHORIZATION_MISSING",
    "TEST_DATA_INTEGRITY_FAILURE",
    "PREDICTION_EXECUTION_FAILED_BEFORE_RESULTS",
    "EVALUATION_ARTIFACT_WRITE_FAILURE_AFTER_VALID_PREDICTIONS",
    "PROTOCOL_VIOLATION",
)

RERUN_INFERENCE_ON: frozenset[str] = frozenset()
"""Empty on purpose: no failure state authorises re-running inference."""

REBUILD_FROM_PREDICTIONS_ON: frozenset[str] = frozenset(
    {"EVALUATION_ARTIFACT_WRITE_FAILURE_AFTER_VALID_PREDICTIONS"}
)

HUMAN_REVIEW_REQUIRED_ON: frozenset[str] = frozenset(
    {
        "BLOCKED_MISSING_MODEL_ARTIFACT",
        "MODEL_IDENTITY_MISMATCH",
        "TEST_DATA_INTEGRITY_FAILURE",
        "PREDICTION_EXECUTION_FAILED_BEFORE_RESULTS",
        "PROTOCOL_VIOLATION",
    }
)

# --- report artifacts -----------------------------------------------------------------------------

REPORT_ARTIFACTS: tuple[str, ...] = (
    "reports/final_test_detector.json",
    "reports/final_test_segmenter.json",
    "reports/final_test_direct_iou.json",
    "reports/final_test_evaluation.md",
    "reports/final_test_evaluation.provenance.json",
)

RESULT_FINGERPRINT_FIELDS: tuple[str, ...] = (
    "detector_test_result_sha256",
    "segmenter_test_result_sha256",
    "direct_iou_test_result_sha256",
    "qualitative_selection_sha256",
)

PREDICTION_FINGERPRINT_FIELDS: tuple[str, ...] = (
    "detector_test_prediction_sha256",
    "segmenter_test_prediction_sha256",
)

FINGERPRINT_EXCLUDES: tuple[str, ...] = (
    "timestamp",
    "username",
    "hostname",
    "absolute filesystem path",
    "machine-specific run directory",
)

# --- aggregate population facts, frozen in phase 5C.2 ---------------------------------------------

TEST_IMAGES = 65
TEST_ANNOTATIONS = 305
SPLIT_ASSIGNMENT_SHA256 = "a230869ff4cb45f53654d67357a27f2a0def6d8ba79f9fa4880f0fdda2f046cc"
HOLDOUT_SHA256 = "bb7ed43b20a84644d5a3917c6d0ead688132f82a30052b06ae7ad121e4851a00"
CLASS_MAP_SHA256 = "596dab5b5756e3d4253924f84bf46d37a87d80b6ec830da781a08a93cd823753"

RUNNER = "scripts/evaluate_final_holdout.py"
"""The only caller the accessor grants the holdout to."""

ACCESS_PURPOSE = "FINAL_ONE_SHOT_HOLDOUT_EVALUATION"


class HoldoutProtocolError(RuntimeError):
    """Raised when the frozen protocol cannot be parsed or is inconsistent."""


class PhaseNotAuthorisedError(RuntimeError):
    """Raised when phase 11B work is attempted while only 11A is authorised."""


# --- parsing --------------------------------------------------------------------------------------

TOP_LEVEL_KEYS: frozenset[str] = frozenset(
    {
        "schema_version",
        "protocol",
        "status",
        "phase",
        "executes_in_phase",
        "objective",
        "one_shot",
        "detector",
        "segmenter",
        "alternate_checkpoint_permitted",
        "retraining_to_replace_a_missing_binary_permitted",
        "authorization",
        "test_population",
        "classes",
        "class_collapsing_permitted",
        "class_map_source",
        "rare_class",
        "rare_class_status",
        "support_rule",
        "detector_inference",
        "segmenter_inference",
        "canonical_bbox_evaluator",
        "canonical_segm_evaluator",
        "detector_metrics",
        "segmenter_metrics",
        "precision_recall_semantics",
        "direct_iou",
        "confusion_matrix",
        "object_level_matching",
        "qualitative_selection",
        "spatial_analysis_on_test",
        "computational_cost",
        "prediction_persistence",
        "fingerprints",
        "one_shot_ledger",
        "failure_policy",
        "validation_versus_test",
        "reports",
        "figures",
        "final_comparison",
        "prohibited",
        "test_policy",
    }
)
"""Strict: an unknown top-level key raises rather than being ignored."""

FORBIDDEN_VALUE_KEYS: tuple[str, ...] = (
    "winner_declared",
    "composite_score",
    "weighted_ranking",
    "aggregate_benefit_score",
    "model_selection_follows",
    "automatic_environment_unlock_permitted",
    "runner_may_write_environment",
    "alternate_checkpoint_permitted",
    "exclusions_permitted",
    "sampling_permitted",
    "manual_removal_permitted",
    "manual_cherry_picking_permitted",
    "browsing_then_choosing_permitted",
    "threshold_sweep_permitted",
    "confidence_tuning_on_test_permitted",
    "new_spatial_metrics_permitted",
    "latency_benchmark_permitted_in_phase_11b",
    "silent_rerun_permitted",
    "automatic_rerun_permitted",
    "attempt_counter_resettable",
    "placeholder_values_permitted",
    "predictions_rerun_on_rebuild",
    "significance_test",
    "class_collapsing_permitted",
    "committed",
    "tta",
    "augment",
)
"""Every one must be false wherever it appears, at any depth."""

TEST_LEAK_MARKERS: tuple[str, ...] = (
    "test_map",
    "test_metrics",
    "test_result",
    "test_ap",
    "test_predictions_list",
    "test_image_ids",
)
"""Key fragments whose presence would mean a result or an id had leaked in."""


@dataclass(frozen=True)
class ModelIdentity:
    """One frozen model's authoritative identity.

    Attributes:
        experiment: Experiment id, such as ``D2``.
        model: Architecture name.
        imgsz: Frozen input resolution.
        checkpoint_sha256: The digest that resolves the checkpoint.
        manifest: Repo-relative freeze manifest.
        extras: Task-specific frozen fields, such as ``overlap_mask``.
    """

    experiment: str
    model: str
    imgsz: int
    checkpoint_sha256: str
    manifest: str
    extras: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FinalHoldoutProtocol:
    """The parsed frozen protocol.

    Attributes:
        raw: The whole parsed document, for artifact rendering.
        detector: The frozen detector's identity.
        segmenter: The frozen segmenter's identity.
    """

    raw: Mapping[str, Any]
    detector: ModelIdentity
    segmenter: ModelIdentity

    @property
    def status(self) -> str:
        """The protocol's frozen status.

        Returns:
            The recorded status string.
        """
        return str(self.raw["status"])

    @property
    def test_images(self) -> int:
        """Aggregate holdout image count, frozen in phase 5C.2.

        Returns:
            The count, read from the protocol rather than from membership.
        """
        return int(self.raw["test_population"]["images"])

    @property
    def test_annotations(self) -> int:
        """Aggregate holdout annotation count, frozen in phase 5C.2.

        Returns:
            The count.
        """
        return int(self.raw["test_population"]["annotations"])


def _identity(block: Mapping[str, Any], *, reserved: Iterable[str]) -> ModelIdentity:
    """Build a model identity from one protocol block.

    Args:
        block: The ``detector`` or ``segmenter`` mapping.
        reserved: Keys consumed into named fields.

    Returns:
        The identity, with every remaining key preserved as an extra.
    """
    consumed = set(reserved)
    return ModelIdentity(
        experiment=str(block["experiment"]),
        model=str(block["model"]),
        imgsz=int(block["imgsz"]),
        checkpoint_sha256=str(block["checkpoint_sha256"]),
        manifest=str(block["manifest"]),
        extras={key: value for key, value in sorted(block.items()) if key not in consumed},
    )


def load_protocol(path: str | Path) -> FinalHoldoutProtocol:
    """Parse and structurally check the frozen protocol file.

    Args:
        path: Path to ``configs/final_holdout_evaluation.yaml``.

    Returns:
        The parsed protocol.

    Raises:
        HoldoutProtocolError: If the file is missing, is not a mapping, carries
            an unknown top-level key, or declares the wrong protocol or status.
    """
    location = Path(path)
    if not location.is_file():
        msg = f"frozen holdout protocol not found: {location.name}"
        raise HoldoutProtocolError(msg)
    raw = yaml.safe_load(location.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        msg = f"{location.name} must contain a YAML mapping"
        raise HoldoutProtocolError(msg)

    unknown = sorted(set(raw) - TOP_LEVEL_KEYS)
    if unknown:
        msg = f"unknown key(s) in the frozen holdout protocol: {unknown}"
        raise HoldoutProtocolError(msg)
    missing = sorted(TOP_LEVEL_KEYS - set(raw))
    if missing:
        msg = f"the frozen holdout protocol is missing required key(s): {missing}"
        raise HoldoutProtocolError(msg)

    if raw["protocol"] != PROTOCOL:
        msg = f"protocol must be {PROTOCOL}, found {raw['protocol']!r}"
        raise HoldoutProtocolError(msg)
    if raw["status"] != FROZEN_NOT_EXECUTED:
        msg = f"status must be {FROZEN_NOT_EXECUTED}, found {raw['status']!r}"
        raise HoldoutProtocolError(msg)

    reserved = ("experiment", "model", "imgsz", "checkpoint_sha256", "manifest")
    return FinalHoldoutProtocol(
        raw=raw,
        detector=_identity(raw["detector"], reserved=reserved),
        segmenter=_identity(raw["segmenter"], reserved=reserved),
    )


# --- the dual gate --------------------------------------------------------------------------------


def authorize_final_holdout_access(
    *,
    purpose: str,
    caller: str,
    allow_test: bool = False,
    env: dict[str, str] | None = None,
) -> Split:
    """Authorise the one-shot holdout read, or refuse.

    Both existing opt-ins are required and neither is sufficient alone; that
    check is delegated to the project's single guard rather than reimplemented.
    One further restriction applies here: the caller must be the declared final
    evaluation runner, so a generic development script that happens to hold both
    opt-ins is still refused.

    Args:
        purpose: Must be :data:`ACCESS_PURPOSE`.
        caller: Repo-relative path of the calling script.
        allow_test: The in-code opt-in.
        env: Environment mapping to read. Defaults to ``os.environ``.

    Returns:
        The authorised split.

    Raises:
        HoldoutProtocolError: If the purpose or the caller is not the declared
            final evaluation.
        HoldoutViolationError: If either opt-in is missing. Raised by the
            project guard, unchanged.
    """
    if purpose != ACCESS_PURPOSE:
        msg = (
            f"the holdout is reachable only for {ACCESS_PURPOSE!r}; refused for {purpose!r}. "
            "Development access to the test split does not exist."
        )
        raise HoldoutProtocolError(msg)
    if caller != RUNNER:
        msg = (
            f"the holdout is reachable only from {RUNNER!r}; refused for {caller!r}. "
            "Holding both opt-ins does not make a generic script the final evaluation."
        )
        raise HoldoutProtocolError(msg)
    return assert_split_allowed(Split.TEST, purpose=purpose, allow_test=allow_test, env=env)


def holdout_is_locked(env: dict[str, str] | None = None) -> bool:
    """Report whether the environment gate is absent.

    Args:
        env: Environment mapping to read. Defaults to ``os.environ``.

    Returns:
        ``True`` when the holdout is locked.
    """
    return not holdout_unlocked(env)


# --- the one-shot ledger --------------------------------------------------------------------------


class LedgerTransitionError(RuntimeError):
    """Raised on an illegal one-shot ledger transition."""


@dataclass
class OneShotLedger:
    """Append-only record of the single holdout evaluation.

    There is deliberately no way to reset the attempt counter and no way to
    rewrite an entry. A rerun after an infrastructure failure is a new, numbered
    attempt carrying a written justification, never a silent restart.

    Attributes:
        attempt: Attempt number, starting at 1.
        justification: Why this attempt exists. Required beyond the first.
        state: Current state.
        history: Every transition, in order.
    """

    attempt: int = 1
    justification: str = "FIRST_AND_ONLY_AUTHORISED_ATTEMPT"
    state: str = "NOT_STARTED"
    history: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate the opening state.

        Raises:
            LedgerTransitionError: If the attempt is not positive, or a repeat
                attempt carries no justification.
        """
        if self.attempt < 1:
            msg = "the one-shot attempt counter starts at 1 and never resets"
            raise LedgerTransitionError(msg)
        if self.attempt > 1 and self.justification == "FIRST_AND_ONLY_AUTHORISED_ATTEMPT":
            msg = (
                f"attempt {self.attempt} requires a written human justification; a rerun is "
                "never presented as the original one-shot evaluation"
            )
            raise LedgerTransitionError(msg)
        self.history.append({"state": self.state, "detail": "ledger opened"})

    def advance(self, state: str, *, detail: str = "") -> None:
        """Move to the next state, recording the transition.

        Args:
            state: Target state.
            detail: Short note recorded with the transition.

        Raises:
            LedgerTransitionError: If the state is unknown, the current state is
                terminal, or the transition is not permitted.
        """
        if state not in LEDGER_STATES:
            msg = f"unknown ledger state: {state!r}"
            raise LedgerTransitionError(msg)
        if self.state in TERMINAL_STATES:
            msg = (
                f"the ledger is terminal at {self.state!r}; a further transition would be a "
                "second evaluation rather than a continuation"
            )
            raise LedgerTransitionError(msg)
        if state not in LEDGER_TRANSITIONS[self.state]:
            msg = f"illegal one-shot transition: {self.state!r} -> {state!r}"
            raise LedgerTransitionError(msg)
        self.state = state
        self.history.append({"state": state, "detail": detail})

    def fail(self, classification: str, *, detail: str = "") -> None:
        """Record a failure and move to the review state.

        Args:
            classification: One of :data:`FAILURE_STATES`.
            detail: What happened.

        Raises:
            LedgerTransitionError: If the classification is not predeclared.
        """
        if classification not in FAILURE_STATES:
            msg = f"undeclared failure classification: {classification!r}"
            raise LedgerTransitionError(msg)
        self.advance("FAILED_REQUIRES_HUMAN_REVIEW", detail=f"{classification}: {detail}")

    def may_rebuild_from_predictions(self) -> bool:
        """Report whether a report rebuild is possible from persisted work.

        Returns:
            ``True`` once predictions are persisted and fingerprinted.
        """
        return any(entry["state"] == REBUILD_ENTRY_STATE for entry in self.history)

    def as_record(self) -> dict[str, Any]:
        """Render the ledger for the provenance artifact.

        Returns:
            A JSON-serialisable summary.
        """
        return {
            "attempt": self.attempt,
            "justification": self.justification,
            "state": self.state,
            "append_only": True,
            "attempt_counter_resettable": False,
            "history": list(self.history),
        }


def recovery_action(classification: str) -> str:
    """Map a failure classification to its predeclared recovery.

    Args:
        classification: One of :data:`FAILURE_STATES`.

    Returns:
        The recovery action name.

    Raises:
        LedgerTransitionError: If the classification is not predeclared.
    """
    if classification not in FAILURE_STATES:
        msg = f"undeclared failure classification: {classification!r}"
        raise LedgerTransitionError(msg)
    if classification == "TEST_EVALUATION_COMPLETE":
        return "NONE_SUCCESS"
    if classification in REBUILD_FROM_PREDICTIONS_ON:
        return "REBUILD_METRICS_AND_REPORTS_FROM_PERSISTED_PREDICTIONS"
    if classification == "TEST_UNLOCK_AUTHORIZATION_MISSING":
        return "REFUSE_AND_CHANGE_NOTHING"
    return "STOP_PRESERVE_EVIDENCE_REQUIRE_HUMAN_REVIEW"


def inference_rerun_permitted(classification: str) -> bool:
    """Report whether a failure authorises re-running inference.

    Always ``False``. The function exists so the rule is asserted by test rather
    than left implicit.

    Args:
        classification: One of :data:`FAILURE_STATES`.

    Returns:
        ``False``.

    Raises:
        LedgerTransitionError: If the classification is not predeclared.
    """
    if classification not in FAILURE_STATES:
        msg = f"undeclared failure classification: {classification!r}"
        raise LedgerTransitionError(msg)
    return classification in RERUN_INFERENCE_ON


# --- deterministic qualitative selection ----------------------------------------------------------


@dataclass(frozen=True)
class QualitativeCandidate:
    """One candidate for the qualitative gallery.

    Attributes:
        category: Which frozen category it qualifies for.
        rank_value: The ranked quantity.
        canonical_annotation_id: Ground-truth id, or -1 for a false positive.
        canonical_prediction_index: Prediction index, or -1 for a miss.
        source_image_id: The image the instance belongs to.
    """

    category: str
    rank_value: float
    canonical_annotation_id: int
    canonical_prediction_index: int
    source_image_id: str


def _sort_key(candidate: QualitativeCandidate, *, descending: bool) -> tuple[Any, ...]:
    """Build the frozen total-order sort key.

    Args:
        candidate: The candidate to rank.
        descending: Whether the ranked quantity sorts descending.

    Returns:
        A tuple ordering by the ranked quantity, then by the frozen tie-breaks.
    """
    primary = -candidate.rank_value if descending else candidate.rank_value
    return (
        primary,
        candidate.canonical_annotation_id,
        candidate.canonical_prediction_index,
        candidate.source_image_id,
    )


def select_qualitative_examples(
    candidates: Sequence[QualitativeCandidate],
    *,
    descending: Mapping[str, bool],
    per_category: int = EXAMPLES_PER_CATEGORY,
) -> dict[str, list[QualitativeCandidate]]:
    """Apply the frozen deterministic selection rule.

    Categories are resolved in :data:`QUALITATIVE_CATEGORIES` order, and an
    instance already selected by an earlier category is skipped by every later
    one, so the gallery never shows the same failure twice under two names. An
    image may appear more than once, because one image can hold several distinct
    failures.

    Args:
        candidates: Every qualifying candidate, in any order.
        descending: Per category, whether its ranked quantity sorts descending.
        per_category: Quota per category.

    Returns:
        Category -> the selected candidates, in rank order.

    Raises:
        HoldoutProtocolError: If a candidate names an unfrozen category or a
            category has no declared sort direction.
    """
    unknown = sorted({c.category for c in candidates} - set(QUALITATIVE_CATEGORIES))
    if unknown:
        msg = f"candidate(s) in undeclared qualitative category: {unknown}"
        raise HoldoutProtocolError(msg)

    selected: dict[str, list[QualitativeCandidate]] = {}
    claimed: set[tuple[int, int]] = set()
    for category in QUALITATIVE_CATEGORIES:
        if category not in descending:
            msg = f"category {category!r} has no declared sort direction"
            raise HoldoutProtocolError(msg)
        pool = [c for c in candidates if c.category == category]
        pool.sort(key=lambda c: _sort_key(c, descending=descending[category]))
        chosen: list[QualitativeCandidate] = []
        for candidate in pool:
            identity = (candidate.canonical_annotation_id, candidate.canonical_prediction_index)
            if identity in claimed:
                continue
            claimed.add(identity)
            chosen.append(candidate)
            if len(chosen) == per_category:
                break
        selected[category] = chosen
    return selected


# --- fingerprints ---------------------------------------------------------------------------------


def protocol_fingerprint(payload: Mapping[str, Any]) -> str:
    """Hash the protocol's scientific and operational content.

    Covers every decision that could change a holdout result or its
    interpretation, and excludes everything machine-specific, so the same
    protocol fingerprints identically on another machine.

    Args:
        payload: The parsed protocol document.

    Returns:
        A SHA-256 hex digest.
    """
    scope = {
        key: payload[key] for key in sorted(payload) if key not in {"objective", "schema_version"}
    }
    text = json.dumps(
        json.loads(json.dumps(scope, sort_keys=True, default=str)),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# --- the validator --------------------------------------------------------------------------------


def _walk(payload: Any, path: str = "") -> list[tuple[str, str, Any]]:
    """Yield every key/value pair in a nested payload.

    Args:
        payload: Any JSON-shaped value.
        path: Accumulated dotted path.

    Returns:
        ``(path, key, value)`` for every mapping entry reached.
    """
    found: list[tuple[str, str, Any]] = []
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            where = f"{path}.{key}" if path else str(key)
            found.append((where, str(key), value))
            found.extend(_walk(value, where))
    elif isinstance(payload, list):
        for index, value in enumerate(payload):
            found.extend(_walk(value, f"{path}[{index}]"))
    return found


def _model_problems(raw: Mapping[str, Any]) -> list[str]:
    """Check both frozen model identities.

    Args:
        raw: The parsed protocol.

    Returns:
        One problem per mismatch.
    """
    problems: list[str] = []
    detector = raw["detector"]
    segmenter = raw["segmenter"]
    if detector["experiment"] != DETECTOR_EXPERIMENT:
        problems.append(f"detector must be {DETECTOR_EXPERIMENT}")
    if detector["checkpoint_sha256"] != DETECTOR_CHECKPOINT_SHA256:
        problems.append("the detector checkpoint digest is not the frozen one")
    if segmenter["experiment"] != SEGMENTER_EXPERIMENT:
        problems.append(f"segmenter must be {SEGMENTER_EXPERIMENT}")
    if segmenter["checkpoint_sha256"] != SEGMENTER_CHECKPOINT_SHA256:
        problems.append("the segmenter checkpoint digest is not the frozen one")
    if segmenter.get("final_segmenter_sha256") != FINAL_SEGMENTER_SHA256:
        problems.append("the segmenter freeze fingerprint is not the frozen one")
    if segmenter.get("overlap_mask") is not False:
        problems.append("the segmenter's overlap_mask must be false; it is part of its identity")
    for name, block in (("detector", detector), ("segmenter", segmenter)):
        if block["imgsz"] != IMGSZ:
            problems.append(f"{name} imgsz must be {IMGSZ}")
    if raw["alternate_checkpoint_permitted"] is not False:
        problems.append("an alternate checkpoint must never be permitted")
    if raw["retraining_to_replace_a_missing_binary_permitted"] is not False:
        problems.append("retraining to replace a missing binary must never be permitted")
    return problems


def _inference_problems(raw: Mapping[str, Any]) -> list[str]:
    """Check both frozen inference blocks.

    Args:
        raw: The parsed protocol.

    Returns:
        One problem per mismatch.
    """
    problems: list[str] = []
    for name in ("detector_inference", "segmenter_inference"):
        block = raw[name]
        if block["imgsz"] != IMGSZ:
            problems.append(f"{name}.imgsz must be {IMGSZ}")
        if block["conf"] != AP_CONF:
            problems.append(f"{name}.conf must be the AP confidence {AP_CONF}")
        if block["iou"] != NMS_IOU:
            problems.append(f"{name}.iou must be {NMS_IOU}")
        if block["max_det"] != MAX_DET:
            problems.append(f"{name}.max_det must be {MAX_DET}")
        if block["precision"] != PRECISION:
            problems.append(f"{name}.precision must be {PRECISION}")
        if block.get("quantize") != QUANTIZE:
            problems.append(f"{name}.quantize must be {QUANTIZE}")
        if block["augment"] is not False or block["tta"] is not False:
            problems.append(f"{name} must disable augmentation and TTA")
        if block.get("half") is not False:
            problems.append(f"{name}.half must be false")
    if raw["segmenter_inference"].get("retina_masks") is not True:
        problems.append("segmenter_inference.retina_masks must stay true")
    if raw["segmenter_inference"].get("mask_reconstruction_change_permitted") is not False:
        problems.append("mask reconstruction behaviour may never be changed")
    return problems


def _evaluator_problems(raw: Mapping[str, Any]) -> list[str]:
    """Check both canonical evaluators and the direct-IoU reference.

    Args:
        raw: The parsed protocol.

    Returns:
        One problem per mismatch.
    """
    problems: list[str] = []
    for name, iou_type in (
        ("canonical_bbox_evaluator", IOU_TYPE_BBOX),
        ("canonical_segm_evaluator", IOU_TYPE_SEGM),
    ):
        block = raw[name]
        if block["implementation"] != COCOEVAL:
            problems.append(f"{name} must use {COCOEVAL}")
        if block["iou_type"] != iou_type:
            problems.append(f"{name}.iou_type must be {iou_type}")
        if tuple(block["iou_thresholds"]) != IOU_THRESHOLDS:
            problems.append(f"{name} must use the standard COCO IoU sweep")
        if tuple(block["max_dets"]) != MAX_DETS:
            problems.append(f"{name}.max_dets must be {list(MAX_DETS)}")
    if raw["canonical_bbox_evaluator"].get("boxes_from_masks") is not False:
        problems.append("the segmenter's boxes must be its own, never derived from its masks")

    direct = raw["direct_iou"]
    if direct["protocol_fingerprint"] != DIRECT_IOU_PROTOCOL_FINGERPRINT:
        problems.append("the direct-IoU protocol fingerprint is not phase 8C's")
    if direct["unchanged_from_phase_8c"] is not True:
        problems.append("the direct-IoU protocol must be phase 8C's, unchanged")
    if direct["new_rule_invented"] is not False:
        problems.append("no new direct-IoU rule may be invented for the holdout")
    if direct["inference"]["conf"] != OPERATIONAL_CONF:
        problems.append(f"the direct-IoU diagnostic keeps its operational {OPERATIONAL_CONF}")
    if direct["role"] != "SECONDARY_CANONICAL_DIAGNOSTIC":
        problems.append("the direct-IoU diagnostic is secondary, never the primary metric")
    if direct["is_primary_segmenter_metric"] is not False:
        problems.append("the direct-IoU diagnostic must not become the primary metric")
    return problems


def _confusion_matrix_problems(raw: Mapping[str, Any]) -> list[str]:
    """Check the frozen confusion-matrix semantics.

    Args:
        raw: The parsed protocol.

    Returns:
        One problem per mismatch.
    """
    problems: list[str] = []
    block = raw["confusion_matrix"]
    if block["implementation"] != CONFUSION_MATRIX_IMPLEMENTATION:
        problems.append(f"the confusion matrix must use {CONFUSION_MATRIX_IMPLEMENTATION}")
    if block["conf"] != CONFUSION_MATRIX_CONF:
        problems.append(f"the confusion-matrix confidence must be {CONFUSION_MATRIX_CONF}")
    if block["iou_threshold"] != CONFUSION_MATRIX_IOU:
        problems.append(f"the confusion-matrix IoU must be {CONFUSION_MATRIX_IOU}")
    if block["orientation"] != CONFUSION_MATRIX_ORIENTATION:
        problems.append("the confusion-matrix orientation was changed")
    if block["matching"] != CONFUSION_MATRIX_MATCHING:
        problems.append("the confusion-matrix matching rule was changed")
    if tuple(block["class_order"]) != CLASSES:
        problems.append("the confusion-matrix class order must be the canonical class map order")
    if block["matrix_dimension"] != len(CLASSES) + 1:
        problems.append("the confusion matrix must carry the background row and column")
    if block["threshold_chosen_after_seeing_test"] is not False:
        problems.append("the confusion-matrix threshold must be frozen before any test access")
    return problems


def _qualitative_problems(raw: Mapping[str, Any]) -> list[str]:
    """Check the deterministic qualitative-selection rule.

    Args:
        raw: The parsed protocol.

    Returns:
        One problem per mismatch.
    """
    problems: list[str] = []
    block = raw["qualitative_selection"]
    if block["deterministic"] is not True:
        problems.append("the qualitative selection must be deterministic")
    if block["random_sampling_permitted"] is not False:
        problems.append("random sampling of qualitative examples is not permitted")
    if block["manual_cherry_picking_permitted"] is not False:
        problems.append("manual cherry-picking is not permitted")
    if block["selection_occurs_before_any_human_views_a_test_image"] is not True:
        problems.append("selection must precede any human viewing of a test image")
    if block["images_inspected_to_design_this_rule"] != 0:
        problems.append("no image may have been inspected while designing the selection rule")
    if tuple(block["tie_breaking"]) != TIE_BREAKING:
        problems.append("the tie-breaking chain was changed")
    if block["tie_breaking_is_total_order"] is not True:
        problems.append("the tie-breaking chain must be a total order")
    if block["examples_per_category"] != EXAMPLES_PER_CATEGORY:
        problems.append(f"examples per category must be {EXAMPLES_PER_CATEGORY}")
    if tuple(block["categories"]) != QUALITATIVE_CATEGORIES:
        problems.append("the qualitative categories were changed")
    if block["topping_up_from_another_category_permitted"] is not False:
        problems.append("an underfilled category may not be topped up from another")
    return problems


def _ledger_problems(raw: Mapping[str, Any]) -> list[str]:
    """Check the one-shot ledger and failure policy.

    Args:
        raw: The parsed protocol.

    Returns:
        One problem per mismatch.
    """
    problems: list[str] = []
    ledger = raw["one_shot_ledger"]
    if tuple(ledger["states"]) != LEDGER_STATES:
        problems.append("the one-shot ledger states were changed")
    if ledger["append_only"] is not True:
        problems.append("the one-shot ledger must be append-only")
    if ledger["attempt_counter_resettable"] is not False:
        problems.append("the attempt counter must never be resettable")
    rerun = ledger["rerun_policy"]
    if rerun["automatic_rerun_permitted"] is not False:
        problems.append("automatic rerun is never permitted")
    if rerun["human_authorization_required"] is not True:
        problems.append("a rerun requires human authorisation")
    if rerun["second_run_may_be_presented_as_the_first"] is not False:
        problems.append("a second run may never be presented as the first")

    policy = raw["failure_policy"]
    if tuple(policy["states"]) != FAILURE_STATES:
        problems.append("the predeclared failure states were changed")
    if policy["silent_rerun_permitted"] is not False:
        problems.append("a silent rerun is never permitted")
    partial = policy["partial_prediction_policy"]
    for flag, expected in (
        ("preserve_evidence", True),
        ("stop", True),
        ("require_human_review", True),
        ("automatic_restart_permitted", False),
        ("present_second_run_as_original_permitted", False),
    ):
        if partial[flag] is not expected:
            problems.append(f"partial_prediction_policy.{flag} must be {expected}")
    return problems


def validate_protocol(payload: Mapping[str, Any]) -> list[str]:
    """Check the frozen protocol against every rule it must satisfy.

    The validator reads no holdout data and cannot: it operates entirely on the
    protocol document.

    Args:
        payload: The parsed protocol.

    Returns:
        One description per problem found; empty when the protocol is sound.
    """
    problems: list[str] = []
    if payload.get("status") != FROZEN_NOT_EXECUTED:
        problems.append(f"status must be {FROZEN_NOT_EXECUTED}")
    if payload.get("phase") != PHASE:
        problems.append(f"phase must be {PHASE}")
    if payload.get("test_policy") != HOLDOUT_STATUS:
        problems.append(f"test_policy must be {HOLDOUT_STATUS}")
    if tuple(payload.get("classes", ())) != CLASSES:
        problems.append("the five frozen classes were changed or reordered")
    if payload.get("rare_class") != RARE_CLASS:
        problems.append(f"rare_class must be {RARE_CLASS}")

    auth = payload["authorization"]
    if auth["activated_in_this_phase"] is not False:
        problems.append("phase 11A must not activate either authorisation gate")
    if auth["either_alone_is_refused"] is not True:
        problems.append("the dual gate must refuse either opt-in alone")
    if auth["automatic_environment_unlock_permitted"] is not False:
        problems.append("no script may unlock the environment automatically")
    if auth["runner_may_write_environment"] is not False:
        problems.append("the runner may read the environment gate but never write it")
    if auth["generic_development_access_permitted"] is not False:
        problems.append("generic development access to the holdout must be refused")

    population = payload["test_population"]
    if population["images"] != TEST_IMAGES:
        problems.append(f"the frozen holdout holds {TEST_IMAGES} images")
    if population["annotations"] != TEST_ANNOTATIONS:
        problems.append(f"the frozen holdout holds {TEST_ANNOTATIONS} annotations")
    if population["evaluate"] != "ALL_FROZEN_TEST_IMAGES":
        problems.append("phase 11B must evaluate the whole frozen holdout")
    if population["per_image_composition_inspected_in_phase_11a"] is not False:
        problems.append("phase 11A must not inspect per-image holdout composition")
    if population["technical_validation"]["silent_removal_permitted"] is not False:
        problems.append("an unreadable file may never be removed silently")

    one_shot = payload["one_shot"]
    if one_shot["reads_permitted"] != 1:
        problems.append("exactly one holdout read is permitted")
    if one_shot["policy"] != "ONE_SHOT_FINAL_HOLDOUT_EVALUATION":
        problems.append("the one-shot policy label was changed")

    if payload["computational_cost"]["rerun_on_test"] is not False:
        problems.append("the latency benchmark must not be rerun on the holdout")
    if payload["spatial_analysis_on_test"]["repeat_full_phase_10b_exploration"] is not False:
        problems.append("the phase 10B exploratory spatial study is not repeated on the holdout")

    comparison = payload["validation_versus_test"]
    if comparison["metrics_must_already_exist"] is not True:
        problems.append("only already-existing metrics may be compared across splits")
    if comparison["new_metric_invented_for_the_comparison"] is not False:
        problems.append("no new metric may be invented for the validation-test comparison")

    if payload["reports"]["created_in_phase_11a"] is not False:
        problems.append("the result reports must not exist in phase 11A")
    for artifact in REPORT_ARTIFACTS:
        declared = json.dumps(payload["reports"])
        if artifact not in declared:
            problems.append(f"the report schema does not declare {artifact}")

    fingerprints = payload["fingerprints"]
    if fingerprints["rebuild_policy"]["predictions_rerun_on_rebuild"] is not False:
        problems.append("a report rebuild must never rerun predictions")
    if fingerprints["deterministic_recomputation_required"] is not True:
        problems.append("fingerprints must recompute deterministically")
    for excluded in FINGERPRINT_EXCLUDES:
        if excluded not in fingerprints["excludes"]:
            problems.append(f"fingerprints must exclude {excluded}")

    problems.extend(_model_problems(payload))
    problems.extend(_inference_problems(payload))
    problems.extend(_evaluator_problems(payload))
    problems.extend(_confusion_matrix_problems(payload))
    problems.extend(_qualitative_problems(payload))
    problems.extend(_ledger_problems(payload))

    for path, key, value in _walk(payload):
        if key in FORBIDDEN_VALUE_KEYS and value not in (False, None):
            problems.append(f"{path} must be false, found {value!r}")
        lowered = str(key).lower()
        # A boolean here is a policy declaration ("this was not invented after
        # seeing the test results"), not a leaked value. Only something that
        # could *be* a result or an identifier counts.
        if not isinstance(value, bool):
            for marker in TEST_LEAK_MARKERS:
                if marker in lowered:
                    problems.append(f"{path} looks like a holdout result or identifier")
    return problems


def validate_manifest(manifest: Mapping[str, Any]) -> list[str]:
    """Check the committed protocol manifest.

    Args:
        manifest: The parsed manifest.

    Returns:
        One description per problem found.
    """
    problems: list[str] = []
    if manifest.get("status") != FROZEN_NOT_EXECUTED:
        problems.append(f"manifest status must be {FROZEN_NOT_EXECUTED}")
    if manifest.get("phase") != PHASE:
        problems.append(f"manifest phase must be {PHASE}")
    for field_name in (
        "models_executed",
        "test_predictions_produced",
        "test_metrics_computed",
        "test_images_read",
        "test_annotations_read",
        "test_identifiers_recorded",
    ):
        if manifest.get(field_name) != 0:
            problems.append(f"{field_name} must be 0 in a protocol-only phase")
    if manifest.get("test", {}).get("status") != HOLDOUT_STATUS:
        problems.append(f"test.status must be {HOLDOUT_STATUS}")
    if manifest.get("holdout_unlocked") is not False:
        problems.append("the holdout must be locked")
    if manifest.get("results_present") is not False:
        problems.append("a protocol manifest may carry no result")
    detector = manifest.get("detector", {})
    segmenter = manifest.get("segmenter", {})
    if detector.get("checkpoint_sha256") != DETECTOR_CHECKPOINT_SHA256:
        problems.append("the manifest's detector digest is not the frozen one")
    if segmenter.get("checkpoint_sha256") != SEGMENTER_CHECKPOINT_SHA256:
        problems.append("the manifest's segmenter digest is not the frozen one")
    if not manifest.get("protocol_fingerprint"):
        problems.append("the manifest carries no protocol fingerprint")
    return problems


def phase_11b_steps() -> tuple[str, ...]:
    """The frozen execution order of the phase 11B runner.

    Declared here so the order is testable in phase 11A and cannot drift when
    phase 11B is eventually authorised.

    Returns:
        The ordered step names.
    """
    return (
        "VALIDATE_AUTHORIZATION",
        "VERIFY_FROZEN_MODEL_BINARIES",
        "VERIFY_PROTOCOL_FINGERPRINT",
        "CREATE_ONE_SHOT_LEDGER",
        "LOAD_TEST_SPLIT_ONCE",
        "RUN_DETECTOR_PREDICTIONS",
        "PERSIST_AND_FINGERPRINT_DETECTOR_PREDICTIONS",
        "RUN_SEGMENTER_PREDICTIONS",
        "PERSIST_AND_FINGERPRINT_SEGMENTER_PREDICTIONS",
        "COMPUTE_CANONICAL_METRICS",
        "COMPUTE_DIRECT_MASK_IOU",
        "COMPUTE_FROZEN_CONFUSION_MATRIX",
        "SELECT_QUALITATIVE_EXAMPLES_DETERMINISTICALLY",
        "GENERATE_REPORTS",
        "LOCK_RESULT_PROVENANCE",
    )
