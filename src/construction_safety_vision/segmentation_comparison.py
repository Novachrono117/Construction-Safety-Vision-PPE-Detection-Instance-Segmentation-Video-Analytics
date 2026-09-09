"""The frozen S0-versus-S1 segmentation comparison protocol.

Phase 8E. Written before S1 exists, which is the only thing that makes it a
protocol; and written after S0 ran, which is recorded rather than hidden.

The shape mirrors what phase 7A did for detection, and for the same reason: a
candidate carries **no protocol of its own**. It names the experiment it inherits
and a set of dotted-path overrides, and the parser refuses any override outside
the single field the candidate declares it is testing. So "only one thing
differs" is checkable rather than asserted.

Two things are specific to this comparison and are enforced here.

**The deciding metric cannot be the framework's own.** Phase 8D established from
the installed source that ``overlap_mask`` changes both the training target and
the validation ground truth. S0 and an ``overlap_mask: false`` S1 would therefore
have their native mask AP measured against different targets, so that number is
marked ``NOT_CROSS_TARGET_COMPARABLE_FOR_S0_S1_SELECTION`` and the canonical
COCO evaluator decides instead. The parser refuses a protocol that promotes the
native metric.

**Disagreement is recorded, not resolved.** If the canonical AP and the direct
IoU diagnostic move in opposite directions, the answer is
``CROSS_METRIC_DIRECTION_DISAGREEMENT`` and a human reading, never a new weighted
score invented once the numbers are visible. A composite is refused at parse
time.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from construction_safety_vision.canonical_evaluation import (
    ALL_CLASS_METRIC,
    ALL_CLASS_METRIC_50,
    NATIVE_METRIC_STATUS,
    PRACTICAL_EQUIVALENCE_MARGIN,
    PRIMARY_METRIC,
)
from construction_safety_vision.config import ConfigError, check_keys

CONFIG_SCHEMA_VERSION = 1
"""Schema version of ``configs/segmentation_comparison.yaml``."""

REFERENCE = "S0"
"""The experiment the candidate is measured against."""

CANDIDATE = "S1"
"""The single authorised candidate."""

ONE_VARIABLE_FIELD = "overlap_mask"
"""The only field S1 may differ from S0 in."""

S1_STATUS = "FROZEN_NOT_EXECUTED"
"""S1's state at the end of this phase."""

FINAL_SEGMENTER = "UNSELECTED"
"""No segmenter is chosen by freezing a protocol."""

CHECKPOINT_POLICY = "ULTRALYTICS_NATIVE_SEGMENTATION_FITNESS"
"""Both experiments keep the same checkpoint rule, unchanged from phase 8B."""

TIMING = "POST_S0_PRE_S1_PROTOCOL_FREEZE"
"""When this protocol was frozen, stated so nobody reads it as predating S0."""

FORBIDDEN_SPLIT = "test"

SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class ComparisonConfigError(ConfigError):
    """Raised when the comparison protocol is missing, malformed or unsafe."""


@dataclass(frozen=True)
class ComparisonConfig:
    """The frozen segmentation comparison protocol.

    Attributes:
        raw: The validated configuration mapping, exactly as parsed.
    """

    raw: dict[str, Any]

    def __getitem__(self, key: str) -> Any:
        """Read a configuration value.

        Args:
            key: Configuration key.

        Returns:
            The value.
        """
        return self.raw[key]

    @property
    def candidate(self) -> dict[str, Any]:
        """The candidate's inheritance and override declaration.

        Returns:
            The candidate block.
        """
        return dict(self.raw["candidate"])

    def as_dict(self) -> dict[str, Any]:
        """Serialise the protocol for hashing and reporting.

        Returns:
            A JSON-serialisable mapping.
        """
        return json.loads(json.dumps(self.raw, sort_keys=True))

    def fingerprint(self) -> str:
        """Hash the protocol, so a result can name the rules that produced it.

        Returns:
            A SHA-256 hex digest over the configuration content only.
        """
        text = json.dumps(self.as_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(text.encode("utf-8")).hexdigest()


_CONFIG_KEYS: tuple[str, ...] = (
    "schema_version",
    "protocol_timing",
    "reference_experiment",
    "candidate",
    "canonical_evaluation_config",
    "canonical_evaluation_sha256",
    "metrics",
    "margin",
    "selection_logic",
    "disagreement_policy",
    "rare_class",
    "rare_class_status",
    "final_segmenter",
    "test_policy",
)

_CANDIDATE_KEYS: tuple[str, ...] = (
    "experiment_id",
    "question",
    "inherits",
    "intentional_field",
    "overrides",
    "status",
    "checkpoint_policy",
    "adapter_identity",
    "required_reporting",
)

_METRIC_KEYS: tuple[str, ...] = (
    "primary",
    "all_class",
    "secondary_direct_iou",
    "native_framework",
    "native_framework_status",
    "composite_score",
)


def _contains_forbidden_split(value: Any) -> bool:
    """Report whether a parsed value mentions the protected split.

    Args:
        value: Any parsed configuration value.

    Returns:
        ``True`` when the protected split appears as a string or a key.
    """
    if isinstance(value, str):
        return value.strip().lower() == FORBIDDEN_SPLIT
    if isinstance(value, Mapping):
        return any(
            _contains_forbidden_split(key) or _contains_forbidden_split(item)
            for key, item in value.items()
        )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return any(_contains_forbidden_split(item) for item in value)
    return False


def load_comparison_config(path: str | Path) -> ComparisonConfig:
    """Load and validate the frozen comparison protocol.

    Args:
        path: Configuration file.

    Returns:
        The parsed protocol.

    Raises:
        ComparisonConfigError: If the file is missing, malformed, names the
            protected split, promotes the native metric, introduces a composite
            score, declares a result for the unexecuted candidate, or lets the
            candidate override more than its one declared field.
    """
    config_path = Path(path).expanduser()
    if not config_path.is_file():
        msg = f"Comparison protocol not found: {config_path.name}"
        raise ComparisonConfigError(msg)
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        msg = f"Comparison protocol is not valid YAML: {config_path.name} ({exc})"
        raise ComparisonConfigError(msg) from exc
    if not isinstance(raw, Mapping):
        msg = f"Comparison protocol must be a mapping: {config_path.name}"
        raise ComparisonConfigError(msg)

    try:
        check_keys(raw, required=_CONFIG_KEYS, context=config_path.name)
    except ConfigError as exc:
        raise ComparisonConfigError(str(exc)) from exc

    if int(raw["schema_version"]) != CONFIG_SCHEMA_VERSION:
        msg = f"schema_version must be {CONFIG_SCHEMA_VERSION}, got {raw['schema_version']!r}"
        raise ComparisonConfigError(msg)
    if str(raw["protocol_timing"]) != TIMING:
        msg = (
            f"protocol_timing must be {TIMING!r}. This protocol was frozen after S0 ran and "
            "before S1 exists, and must not be presented as having predated S0."
        )
        raise ComparisonConfigError(msg)
    if str(raw["reference_experiment"]) != REFERENCE:
        msg = f"reference_experiment must be {REFERENCE!r}"
        raise ComparisonConfigError(msg)
    if _contains_forbidden_split(dict(raw)):
        msg = "the comparison protocol references the protected split"
        raise ComparisonConfigError(msg)
    if str(raw["final_segmenter"]) != FINAL_SEGMENTER:
        msg = f"final_segmenter must be {FINAL_SEGMENTER!r}: freezing a protocol selects nothing"
        raise ComparisonConfigError(msg)
    if not SHA256_PATTERN.match(str(raw["canonical_evaluation_sha256"])):
        msg = "canonical_evaluation_sha256 is not a SHA-256 digest"
        raise ComparisonConfigError(msg)

    if float(raw["margin"]) != PRACTICAL_EQUIVALENCE_MARGIN:
        msg = (
            f"margin must be {PRACTICAL_EQUIVALENCE_MARGIN}, got {raw['margin']!r}. Relaxing it "
            "after a candidate exists is the failure this file is written to prevent."
        )
        raise ComparisonConfigError(msg)

    metrics = raw["metrics"]
    if not isinstance(metrics, Mapping):
        msg = "metrics: must be a mapping"
        raise ComparisonConfigError(msg)
    try:
        check_keys(metrics, required=_METRIC_KEYS, context="metrics")
    except ConfigError as exc:
        raise ComparisonConfigError(str(exc)) from exc
    if str(metrics["primary"]) != PRIMARY_METRIC:
        msg = f"metrics.primary must be {PRIMARY_METRIC!r}, got {metrics['primary']!r}"
        raise ComparisonConfigError(msg)
    for name in (ALL_CLASS_METRIC, ALL_CLASS_METRIC_50):
        if name not in metrics["all_class"]:
            msg = f"metrics.all_class must include {name!r}: it is a required reporting metric"
            raise ComparisonConfigError(msg)
    if str(metrics["native_framework_status"]) != NATIVE_METRIC_STATUS:
        msg = (
            f"metrics.native_framework_status must be {NATIVE_METRIC_STATUS!r}. The framework's "
            "own mask AP is measured against a target overlap_mask reshapes, so it cannot "
            "arbitrate this comparison."
        )
        raise ComparisonConfigError(msg)
    if not metrics["native_framework"]:
        msg = (
            "metrics.native_framework must list the framework metrics S1 still reports. They are "
            "not suppressed; they are demoted."
        )
        raise ComparisonConfigError(msg)
    if metrics["composite_score"] is not False:
        msg = (
            "metrics.composite_score must be false. Inventing a weighting once the numbers are "
            "visible is exactly what the disagreement policy exists to prevent."
        )
        raise ComparisonConfigError(msg)

    candidate = raw["candidate"]
    if not isinstance(candidate, Mapping):
        msg = "candidate: must be a mapping"
        raise ComparisonConfigError(msg)
    try:
        check_keys(candidate, required=_CANDIDATE_KEYS, context="candidate")
    except ConfigError as exc:
        raise ComparisonConfigError(str(exc)) from exc
    if str(candidate["experiment_id"]) != CANDIDATE:
        msg = f"candidate.experiment_id must be {CANDIDATE!r}"
        raise ComparisonConfigError(msg)
    if str(candidate["inherits"]) != REFERENCE:
        msg = f"candidate.inherits must be {REFERENCE!r}: S1 carries no protocol of its own"
        raise ComparisonConfigError(msg)
    if str(candidate["intentional_field"]) != ONE_VARIABLE_FIELD:
        msg = f"candidate.intentional_field must be {ONE_VARIABLE_FIELD!r}"
        raise ComparisonConfigError(msg)
    if str(candidate["status"]) != S1_STATUS:
        msg = f"candidate.status must be {S1_STATUS!r} while no S1 result exists"
        raise ComparisonConfigError(msg)
    if str(candidate["checkpoint_policy"]) != CHECKPOINT_POLICY:
        msg = (
            f"candidate.checkpoint_policy must be {CHECKPOINT_POLICY!r}. Changing the checkpoint "
            "rule as well would make the comparison two-variable."
        )
        raise ComparisonConfigError(msg)

    overrides = candidate["overrides"]
    if not isinstance(overrides, Mapping):
        msg = "candidate.overrides: must be a mapping"
        raise ComparisonConfigError(msg)
    if set(overrides) != {ONE_VARIABLE_FIELD}:
        msg = (
            f"candidate.overrides must contain exactly {{{ONE_VARIABLE_FIELD!r}}}, got "
            f"{sorted(overrides)}. Any other override makes this a multi-variable comparison, "
            "and the result would not be attributable to the declared intervention."
        )
        raise ComparisonConfigError(msg)
    if overrides[ONE_VARIABLE_FIELD] is not False:
        msg = f"candidate.overrides.{ONE_VARIABLE_FIELD} must be false"
        raise ComparisonConfigError(msg)

    if "result" in candidate or "metrics" in candidate:
        msg = "candidate must carry no result: S1 has not been executed"
        raise ComparisonConfigError(msg)

    logic = raw["selection_logic"]
    if not isinstance(logic, Mapping):
        msg = "selection_logic: must be a mapping"
        raise ComparisonConfigError(msg)
    for case in ("improves", "practically_equivalent", "below"):
        if case not in logic:
            msg = f"selection_logic must declare the {case!r} case before S1 exists"
            raise ComparisonConfigError(msg)

    disagreement = raw["disagreement_policy"]
    if not isinstance(disagreement, Mapping) or disagreement.get("create_composite") is not False:
        msg = "disagreement_policy.create_composite must be false"
        raise ComparisonConfigError(msg)

    return ComparisonConfig(raw=json.loads(json.dumps(raw)))


def verify_one_variable_contract(
    reference: Mapping[str, Any], candidate_overrides: Mapping[str, Any]
) -> dict[str, Any]:
    """Prove the candidate differs from the reference in exactly one field.

    Checked against the reference protocol's own values rather than against a
    restatement of them, so a drifted inheritance cannot pass by matching a
    second copy of itself.

    Args:
        reference: The frozen reference experiment's effective arguments.
        candidate_overrides: The candidate's declared overrides.

    Returns:
        What was verified, field by field.

    Raises:
        ComparisonConfigError: If the candidate overrides anything else, or if
            its declared override does not actually change the reference's value
            - a variable declared but never applied is as invalid as an
            undeclared one.
    """
    extra = sorted(set(candidate_overrides) - {ONE_VARIABLE_FIELD})
    if extra:
        msg = f"the candidate overrides fields beyond {ONE_VARIABLE_FIELD!r}: {extra}"
        raise ComparisonConfigError(msg)

    reference_value = reference.get(ONE_VARIABLE_FIELD)
    candidate_value = candidate_overrides[ONE_VARIABLE_FIELD]
    if reference_value == candidate_value:
        msg = (
            f"the candidate declares {ONE_VARIABLE_FIELD} = {candidate_value!r}, which is what "
            "the reference already uses. A variable that is declared but never applied is not a "
            "controlled comparison."
        )
        raise ComparisonConfigError(msg)

    inherited = {
        key: value for key, value in sorted(reference.items()) if key != ONE_VARIABLE_FIELD
    }
    return {
        "intentional_field": ONE_VARIABLE_FIELD,
        "reference_value": reference_value,
        "candidate_value": candidate_value,
        "overrides_declared": sorted(candidate_overrides),
        "fields_inherited_unchanged": sorted(inherited),
        "inherited_count": len(inherited),
        "one_variable": True,
    }
