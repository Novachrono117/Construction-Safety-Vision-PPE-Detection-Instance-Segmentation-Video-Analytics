"""Phase 7 controlled detection comparison: the rules, frozen before the results.

D0 exists. D1 and D2 do not. That ordering is the only reason this module can be
trusted, and it is why every threshold in it is a constant rather than an
argument discovered later.

Three specific ways a model comparison lies, and what is done here about each.

**The metric is chosen after the numbers.** With five classes and four global
metrics there are enough orderings available that some framing will favour
whichever run one prefers. So the primary selection metric is named here, once,
before any comparative result exists: :data:`PRIMARY_SELECTION_METRIC`. The
all-class metric stays mandatory alongside it - a policy that quietly drops the
official number would be worse than no policy - but it does not decide the
winner.

**A class with almost no validation evidence casts a full vote.** ``vest_loose``
holds one positive validation image and eight instances under the frozen split.
Its AP is not a measurement of how well a model detects loose vests; it is a
measurement of one image. Averaging it into the deciding metric hands a fifth of
the decision to that image. The support rule
(:data:`SUPPORT_MIN_POSITIVE_IMAGES`, :data:`SUPPORT_MIN_INSTANCES`) is applied
mechanically to the frozen validation support to decide which classes may vote,
and it is written as a general threshold rather than as "exclude vest_loose", so
that it is a rule and not a name.

**A fraction of a point is read as an ordering.** ``deterministic: true`` reduces
run-to-run variance; it does not remove it. So an improvement no larger than
:data:`PRACTICAL_EQUIVALENCE_MARGIN` is classified
:data:`PRACTICALLY_EQUIVALENT` and the lower-complexity baseline is retained.
That margin is an engineering decision threshold, not a significance test, and
this module never calls it one.

Comparison also refuses rather than warns. Two runs are comparable only if they
saw the same data and differ only where the protocol declared they would, so a
fingerprint mismatch or an undeclared hyperparameter difference raises
:class:`ComparisonError` instead of producing a table with a footnote.

Nothing here reads, names or needs the holdout.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

import yaml

from construction_safety_vision.config import ConfigError, check_keys
from construction_safety_vision.data.materialization import digest
from construction_safety_vision.splits import assert_split_allowed

CONFIG_SCHEMA_VERSION = 1
"""Schema version of ``configs/detection_experiments.yaml``."""

POLICY_SCHEMA_VERSION = 1
"""Schema version of the emitted comparison policy and reference artifacts."""

METRIC_PRECISION = 6
"""Decimal places every metric is normalised to before it is compared.

The result manifests store metrics rounded to six places. Comparing at that
precision, through :class:`~decimal.Decimal`, makes the margin boundary exact:
a delta of precisely 0.005 is reproducibly *not* an improvement rather than
depending on how binary floating point happened to land.
"""

SUPPORT_MIN_POSITIVE_IMAGES = 5
"""Minimum positive validation source images for a class to vote on selection."""

SUPPORT_MIN_INSTANCES = 20
"""Minimum validation instances for a class to vote on selection."""

COMPARISON_SUPPORTED = "COMPARISON_SUPPORTED"
"""A class whose frozen validation support clears both thresholds."""

DESCRIPTIVE_HIGH_UNCERTAINTY = "DESCRIPTIVE_HIGH_UNCERTAINTY"
"""A class reported in full but excluded from the deciding metric."""

SUPPORT_CLASSIFICATIONS: tuple[str, ...] = (
    COMPARISON_SUPPORTED,
    DESCRIPTIVE_HIGH_UNCERTAINTY,
)
"""The only two support classifications."""

PRIMARY_SELECTION_METRIC = "supported_macro_map50_95"
"""The single metric that decides the Phase 7 validation-performance ranking.

The unweighted arithmetic mean of per-class AP@0.50:0.95 over the classes
classified :data:`COMPARISON_SUPPORTED`. Unweighted on purpose: a
support-weighted mean would let ``person``, the most frequent class, absorb the
decision, which is the opposite of what a PPE-compliance evaluation cares about.
"""

OFFICIAL_ALL_CLASS_METRIC = "mAP@0.50:0.95"
"""The mandatory all-class metric, reported for every experiment.

Not the selection metric, and never hidden. The class-support filter is a
statement about which classes carry enough validation evidence to *order two
models*; it is not a statement that the excluded class does not count.
"""

SECONDARY_METRICS: tuple[str, ...] = ("mAP@0.50", "precision", "recall")
"""Supporting global metrics, reported for every experiment."""

PER_CLASS_METRICS: tuple[str, ...] = ("AP@0.50", "AP@0.50:0.95", "precision", "recall")
"""Metrics reported for every class, supported or descriptive."""

PRACTICAL_EQUIVALENCE_MARGIN = Decimal("0.005")
"""Absolute margin below which a difference is not treated as an ordering.

An engineering decision threshold that prevents escalating to a larger model for
a trivial validation difference. It is **not** a significance test and no claim
of statistical significance may be derived from it.
"""

IMPROVED = "IMPROVED_BEYOND_MARGIN"
"""A candidate that beats the reference by more than the margin."""

PRACTICALLY_EQUIVALENT = "PRACTICALLY_EQUIVALENT_ON_THIS_VALIDATION_SET"
"""A candidate within the margin of the reference, in either direction."""

REGRESSED = "REGRESSED_BEYOND_MARGIN"
"""A candidate that falls short of the reference by more than the margin."""

DELTA_CLASSIFICATIONS: tuple[str, ...] = (IMPROVED, PRACTICALLY_EQUIVALENT, REGRESSED)
"""The only three per-candidate outcomes against the reference."""

CASE_A = "CASE_A_RETAIN_REFERENCE"
"""No candidate clears the reference by more than the margin."""

CASE_B = "CASE_B_VALIDATION_PERFORMANCE_LEADER"
"""One candidate clears the reference and separates from the runner-up."""

CASE_C = "CASE_C_PERFORMANCE_TIE_REQUIRES_EFFICIENCY_COMPARISON"
"""A candidate clears the reference but does not separate from the runner-up."""

CASE_D = "CASE_D_EXECUTION_OR_PROTOCOL_FAILURE"
"""At least one experiment failed to execute or violated its protocol."""

SELECTION_CASES: tuple[str, ...] = (CASE_A, CASE_B, CASE_C, CASE_D)
"""The four selection outcomes, fixed before any candidate result exists."""

MEMORY_CONSTRAINT_REVIEW_REQUIRED = "MEMORY_CONSTRAINT_REVIEW_REQUIRED"
"""Classification for an experiment that cannot run at the controlled batch."""

FROZEN_NOT_EXECUTED = "FROZEN_NOT_EXECUTED"
"""An experiment whose protocol is frozen and which has not been run."""

EXECUTED = "EXECUTED"
"""An experiment with a committed result manifest."""

EXPERIMENT_STATUSES: tuple[str, ...] = (FROZEN_NOT_EXECUTED, EXECUTED)
"""Statuses an experiment declaration may carry."""

REFERENCE_ROLE = "REFERENCE_BASELINE"
"""Role of the experiment every other experiment is measured against."""

CANDIDATE_ROLE = "CONTROLLED_EXPERIMENT"
"""Role of an experiment that varies exactly one declared thing."""

EXPERIMENT_ROLES: tuple[str, ...] = (REFERENCE_ROLE, CANDIDATE_ROLE)
"""The only two roles in the experiment matrix."""

EXPERIMENT_ID_PATTERN = re.compile(r"^[A-Z]\d+[A-Za-z0-9_-]*$")
"""Experiment ids look like ``D0``, ``D1``, ``D2``."""

FORBIDDEN_SPLIT = "test"
"""The protected split, which no comparison artifact may name."""

VALIDATION_SPLIT = "validation"
"""The only split Phase 7 selection may read."""

NON_PROTOCOL_FIELDS: frozenset[str] = frozenset({"experiment_id", "description"})
"""Protocol fields that differ between experiments by design.

Each experiment has its own identifier and its own question. Neither changes
what the training run does, so neither counts as a protocol difference.
"""

RUNTIME_FIELDS: frozenset[str] = frozenset(
    {
        "created_at",
        "timestamp",
        "run_name",
        "project",
        "name",
        "save_dir",
        "output_dir",
        "device",
        "exist_ok",
    }
)
"""Fields describing when and where a run happened, not what it was.

Excluded from the protocol comparison so that a different timestamp or output
directory can never be mistaken for a broken controlled experiment.
"""

FINGERPRINT_PATHS: tuple[str, ...] = (
    "adapter_manifest_sha256",
    "split_reference.split_assignment_sha256",
    "dataset_fingerprints.adapter_config_sha256",
    "dataset_fingerprints.class_map_sha256",
    "dataset_fingerprints.modeling_population_sha256",
    "dataset_fingerprints.task_materialization_config_sha256",
    "dataset_fingerprints.yolo_label_sha256.train",
    "dataset_fingerprints.yolo_label_sha256.validation",
)
"""Result-manifest paths that must be identical across compared experiments.

If any of these differs, the two runs did not see the same data and no
difference between their metrics is attributable to the declared variable.
"""

EXPERIMENT_FINGERPRINT_KEYS: tuple[str, ...] = ("experiment_sha256", "d0_experiment_sha256")
"""Accepted spellings of a result manifest's experiment fingerprint.

``d0_experiment_sha256`` is the Phase 6B spelling. A generic key is accepted so
a D1 or D2 manifest need not pretend to be D0.
"""


class ComparisonError(RuntimeError):
    """Raised when two experiments cannot be compared honestly."""


class ComparisonConfigError(ConfigError):
    """Raised when the experiment matrix is missing, malformed or unsafe."""


def _decimal(value: Any, *, context: str) -> Decimal:
    """Normalise a metric to an exact decimal at the project's precision.

    Args:
        value: Raw metric value.
        context: Field name, used in the error message.

    Returns:
        The value as a :class:`~decimal.Decimal` rounded to
        :data:`METRIC_PRECISION` places.

    Raises:
        ComparisonError: If the value is not a finite real number. A missing or
            non-numeric metric is refused rather than coerced to zero: those
            mean different things and confusing them would invent a result.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        msg = f"{context}: expected a number, got {value!r}"
        raise ComparisonError(msg)
    number = float(value)
    if number != number or number in (float("inf"), float("-inf")):
        msg = f"{context}: expected a finite number, got {value!r}"
        raise ComparisonError(msg)
    return Decimal(str(round(number, METRIC_PRECISION)))


def _as_float(value: Decimal) -> float:
    """Convert an exact decimal metric back to a JSON-serialisable float.

    Args:
        value: The exact value.

    Returns:
        The value as a float rounded to :data:`METRIC_PRECISION` places.
    """
    return round(float(value), METRIC_PRECISION)


def contains_forbidden_split(value: Any) -> bool:
    """Report whether a parsed value names the protected split.

    Args:
        value: Any parsed value.

    Returns:
        ``True`` when the protected split name appears as a string or a key.
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


# --- class support ------------------------------------------------------------


@dataclass(frozen=True)
class SupportRule:
    """The threshold deciding which classes may decide a comparison.

    Attributes:
        min_positive_images: Minimum positive validation source images.
        min_instances: Minimum validation instances.
    """

    min_positive_images: int = SUPPORT_MIN_POSITIVE_IMAGES
    min_instances: int = SUPPORT_MIN_INSTANCES

    def __post_init__(self) -> None:
        """Validate the thresholds.

        Raises:
            ComparisonConfigError: If either threshold is not a positive integer.
        """
        for name, value in (
            ("min_positive_images", self.min_positive_images),
            ("min_instances", self.min_instances),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                msg = f"support_rule.{name} must be a positive integer, got {value!r}"
                raise ComparisonConfigError(msg)

    def classify(self, *, positive_images: int, instances: int) -> str:
        """Classify one class from its frozen validation support.

        Both thresholds must be cleared. A class can hold many instances inside
        very few images - which is what makes an image-count floor a separate
        requirement rather than a proxy for the instance count.

        Args:
            positive_images: Validation source images containing the class.
            instances: Validation instances of the class.

        Returns:
            :data:`COMPARISON_SUPPORTED` or :data:`DESCRIPTIVE_HIGH_UNCERTAINTY`.
        """
        clears = positive_images >= self.min_positive_images and instances >= self.min_instances
        return COMPARISON_SUPPORTED if clears else DESCRIPTIVE_HIGH_UNCERTAINTY

    def as_dict(self) -> dict[str, Any]:
        """Serialise the rule.

        Returns:
            A JSON-serialisable mapping.
        """
        return {
            "min_validation_positive_images": self.min_positive_images,
            "min_validation_instances": self.min_instances,
            "split": VALIDATION_SPLIT,
            "classifications": list(SUPPORT_CLASSIFICATIONS),
        }


@dataclass(frozen=True)
class ClassSupport:
    """One class's frozen validation support and what it may be used for.

    Attributes:
        class_name: The frozen class name.
        class_index: Its index in the frozen class map.
        positive_images: Validation source images containing the class.
        instances: Validation instances of the class.
        classification: Support classification under the frozen rule.
    """

    class_name: str
    class_index: int
    positive_images: int
    instances: int
    classification: str

    @property
    def supported(self) -> bool:
        """Whether this class may take part in the deciding metric.

        Returns:
            ``True`` when the class is :data:`COMPARISON_SUPPORTED`.
        """
        return self.classification == COMPARISON_SUPPORTED

    def as_dict(self) -> dict[str, Any]:
        """Serialise the support record.

        Returns:
            A JSON-serialisable mapping.
        """
        return {
            "class_name": self.class_name,
            "class_index": self.class_index,
            "validation_positive_images": self.positive_images,
            "validation_instances": self.instances,
            "classification": self.classification,
        }


def class_support(
    split_manifest: Mapping[str, Any],
    class_map: Mapping[str, int],
    *,
    rule: SupportRule | None = None,
    env: dict[str, str] | None = None,
) -> tuple[ClassSupport, ...]:
    """Derive each class's support from the frozen split manifest.

    Reads the manifest's ``validation`` counts only, through the holdout guard,
    and never indexes its ``test`` section. Class names come from the frozen
    class map rather than from whatever keys the manifest happens to carry, so a
    class missing from the counts is an error rather than a silent zero.

    Args:
        split_manifest: Parsed ``reports/split_manifest.json``.
        class_map: The frozen class name to index mapping.
        rule: Support rule. Defaults to the frozen thresholds.
        env: Environment mapping to read. Defaults to ``os.environ``.

    Returns:
        One record per class, ordered by class index.

    Raises:
        ComparisonError: If the manifest lacks the validation counts, or a class
            in the frozen map has no recorded support.
    """
    assert_split_allowed(
        VALIDATION_SPLIT,
        purpose="phase 7 class-support derivation",
        allow_test=False,
        env=env,
    )
    support_rule = SupportRule() if rule is None else rule
    if not class_map:
        msg = "class_support: the frozen class map is empty"
        raise ComparisonError(msg)

    images = _validation_counts(split_manifest, "images_with_class")
    instances = _validation_counts(split_manifest, "instances_by_class")

    records: list[ClassSupport] = []
    for name in sorted(class_map, key=lambda key: (class_map[key], key)):
        if name not in images or name not in instances:
            msg = (
                f"class_support: class {name!r} is in the frozen class map but has no "
                "recorded validation support in the split manifest"
            )
            raise ComparisonError(msg)
        positive_images = int(images[name])
        count = int(instances[name])
        records.append(
            ClassSupport(
                class_name=name,
                class_index=int(class_map[name]),
                positive_images=positive_images,
                instances=count,
                classification=support_rule.classify(
                    positive_images=positive_images, instances=count
                ),
            )
        )
    return tuple(records)


def _validation_counts(split_manifest: Mapping[str, Any], section: str) -> Mapping[str, Any]:
    """Return one per-class count section for the validation split only.

    Args:
        split_manifest: Parsed split manifest.
        section: Section name, e.g. ``images_with_class``.

    Returns:
        The mapping of class name to count for ``validation``.

    Raises:
        ComparisonError: If the section or its validation entry is missing or
            malformed.
    """
    block = split_manifest.get(section)
    if not isinstance(block, Mapping):
        msg = f"split manifest is missing the {section!r} section"
        raise ComparisonError(msg)
    counts = block.get(VALIDATION_SPLIT)
    if not isinstance(counts, Mapping) or not counts:
        msg = f"split manifest section {section!r} has no {VALIDATION_SPLIT!r} counts"
        raise ComparisonError(msg)
    return counts


def supported_class_names(support: Iterable[ClassSupport]) -> tuple[str, ...]:
    """Return the names of the classes that may decide a comparison.

    Args:
        support: Support records.

    Returns:
        Supported class names, sorted, so the selection set is independent of
        the order the records arrived in.
    """
    return tuple(sorted(record.class_name for record in support if record.supported))


def descriptive_class_names(support: Iterable[ClassSupport]) -> tuple[str, ...]:
    """Return the names of the classes reported but excluded from selection.

    Args:
        support: Support records.

    Returns:
        Descriptive class names, sorted.
    """
    return tuple(sorted(record.class_name for record in support if not record.supported))


def supported_macro(
    per_class_metrics: Mapping[str, Mapping[str, Any]],
    supported: Sequence[str],
    *,
    metric: str = "AP@0.50:0.95",
) -> Decimal:
    """Average one per-class metric over the supported classes.

    Args:
        per_class_metrics: Per-class metric tables keyed by class name.
        supported: Names of the classes that may contribute.
        metric: Per-class metric to average.

    Returns:
        The unweighted arithmetic mean, exact at :data:`METRIC_PRECISION`.

    Raises:
        ComparisonError: If the supported set is empty, or a supported class has
            no usable value for the metric.
    """
    if not supported:
        msg = (
            "supported_macro: no class clears the frozen support rule, so no selection "
            "metric can be computed. That is a protocol failure, not a score of zero."
        )
        raise ComparisonError(msg)
    total = Decimal(0)
    for name in sorted(supported):
        table = per_class_metrics.get(name)
        if not isinstance(table, Mapping) or metric not in table:
            msg = f"supported_macro: class {name!r} has no {metric!r} value"
            raise ComparisonError(msg)
        total += _decimal(table[metric], context=f"per_class_metrics.{name}.{metric}")
    return (total / Decimal(len(supported))).quantize(Decimal(1).scaleb(-METRIC_PRECISION * 2))


# --- experiment matrix configuration -----------------------------------------


@dataclass(frozen=True)
class ExperimentDeclaration:
    """One frozen entry of the Phase 7 experiment matrix.

    Attributes:
        experiment_id: Stable identifier, e.g. ``D1``.
        role: :data:`REFERENCE_BASELINE` or :data:`CONTROLLED_EXPERIMENT`.
        question: The question the experiment answers, written before it runs.
        status: Whether a result exists yet.
        intentional_variable: The one thing this experiment varies.
        intentional_fields: Protocol field paths the variable changes directly.
        consequential_fields: Field paths that follow necessarily from it.
        overrides: Field path to value, applied over the inherited protocol.
        inherits: Experiment whose protocol this one starts from, if any.
        protocol_source: Repository-relative protocol file, for the reference.
        result_manifest: Repository-relative result manifest, once it exists.
        hypothesis: The predeclared expectation, labelled as untested.
        pretrained_weights: The starting-checkpoint policy, when the experiment
            changes it. Fingerprinting requirements are recorded here before the
            run so the binary cannot be adopted by name alone.
    """

    experiment_id: str
    role: str
    question: str
    status: str
    intentional_variable: str
    intentional_fields: tuple[str, ...]
    consequential_fields: tuple[str, ...]
    overrides: dict[str, Any]
    inherits: str | None
    protocol_source: str | None
    result_manifest: str | None
    hypothesis: str
    pretrained_weights: dict[str, Any] | None = None

    @property
    def declared_difference_fields(self) -> frozenset[str]:
        """Field paths this experiment is permitted to differ in.

        Returns:
            The union of the intentional and consequential field paths.
        """
        return frozenset(self.intentional_fields) | frozenset(self.consequential_fields)

    def as_dict(self) -> dict[str, Any]:
        """Serialise the declaration.

        Returns:
            A JSON-serialisable mapping.
        """
        return {
            "experiment_id": self.experiment_id,
            "role": self.role,
            "question": self.question,
            "status": self.status,
            "intentional_variable": self.intentional_variable,
            "intentional_fields": list(self.intentional_fields),
            "consequential_fields": list(self.consequential_fields),
            "overrides": dict(self.overrides),
            "inherits": self.inherits,
            "protocol_source": self.protocol_source,
            "result_manifest": self.result_manifest,
            "hypothesis": self.hypothesis,
            "pretrained_weights": (
                None if self.pretrained_weights is None else dict(self.pretrained_weights)
            ),
        }


@dataclass(frozen=True)
class ExperimentMatrix:
    """The complete, frozen Phase 7 comparison protocol.

    Attributes:
        schema_version: Version of this configuration schema.
        phase: Roadmap phase that froze the matrix.
        task: Task name.
        reference_experiment: Id of the baseline every candidate is measured on.
        support_rule: The frozen class-support threshold.
        primary_selection_metric: The deciding metric's name.
        official_all_class_metric: The mandatory all-class metric's name.
        secondary_metrics: Supporting global metrics.
        per_class_metrics: Metrics reported for every class.
        diagnostics: Diagnostic artifacts every experiment must produce.
        practical_equivalence_margin: The frozen decision margin.
        memory_policy: What happens if the controlled batch does not fit.
        experiments: The declarations, in declaration order.
    """

    schema_version: int
    phase: str
    task: str
    reference_experiment: str
    support_rule: SupportRule
    primary_selection_metric: str
    official_all_class_metric: str
    secondary_metrics: tuple[str, ...]
    per_class_metrics: tuple[str, ...]
    diagnostics: tuple[str, ...]
    practical_equivalence_margin: Decimal
    memory_policy: dict[str, Any]
    experiments: tuple[ExperimentDeclaration, ...]

    def declaration(self, experiment_id: str) -> ExperimentDeclaration:
        """Look up one declaration by id.

        Args:
            experiment_id: The id to find.

        Returns:
            The matching declaration.

        Raises:
            ComparisonError: If the matrix declares no such experiment. An
                undeclared experiment cannot be compared: the matrix is the
                record of what was planned in advance.
        """
        for declaration in self.experiments:
            if declaration.experiment_id == experiment_id:
                return declaration
        known = [item.experiment_id for item in self.experiments]
        msg = f"experiment {experiment_id!r} is not declared in the matrix; declared: {known}"
        raise ComparisonError(msg)

    @property
    def reference(self) -> ExperimentDeclaration:
        """The reference declaration.

        Returns:
            The declaration of :attr:`reference_experiment`.
        """
        return self.declaration(self.reference_experiment)

    @property
    def candidates(self) -> tuple[ExperimentDeclaration, ...]:
        """The controlled experiments, excluding the reference.

        Returns:
            The candidate declarations, in declaration order.
        """
        return tuple(item for item in self.experiments if item.role == CANDIDATE_ROLE)

    def as_dict(self) -> dict[str, Any]:
        """Serialise the matrix for hashing and reporting.

        Returns:
            A JSON-serialisable mapping.
        """
        return {
            "schema_version": self.schema_version,
            "phase": self.phase,
            "task": self.task,
            "reference_experiment": self.reference_experiment,
            "support_rule": self.support_rule.as_dict(),
            "metrics": {
                "primary_selection": self.primary_selection_metric,
                "official_all_class": self.official_all_class_metric,
                "secondary": list(self.secondary_metrics),
                "per_class": list(self.per_class_metrics),
                "diagnostics": list(self.diagnostics),
            },
            "practical_equivalence_margin": str(self.practical_equivalence_margin),
            "memory_policy": dict(self.memory_policy),
            "experiments": [item.as_dict() for item in self.experiments],
        }

    def fingerprint(self) -> str:
        """Hash the matrix, so a comparison can name the rules that produced it.

        Returns:
            A SHA-256 hex digest over the matrix content only.
        """
        return digest(self.as_dict())


_MATRIX_KEYS: tuple[str, ...] = (
    "schema_version",
    "phase",
    "task",
    "reference_experiment",
    "support_rule",
    "metrics",
    "practical_equivalence_margin_map50_95",
    "memory_policy",
    "experiments",
)

_SUPPORT_RULE_KEYS: tuple[str, ...] = (
    "min_validation_positive_images",
    "min_validation_instances",
    "rationale",
)

_METRICS_KEYS: tuple[str, ...] = (
    "primary_selection",
    "official_all_class",
    "secondary",
    "per_class",
    "diagnostics",
)

_MEMORY_POLICY_KEYS: tuple[str, ...] = (
    "controlled_batch",
    "on_cuda_oom",
    "classification",
    "forbidden_mitigations",
    "rationale",
)

_EXPERIMENT_KEYS: tuple[str, ...] = (
    "experiment_id",
    "role",
    "question",
    "status",
    "intentional_variable",
    "intentional_fields",
    "consequential_fields",
    "overrides",
    "hypothesis",
)

_EXPERIMENT_OPTIONAL_KEYS: tuple[str, ...] = (
    "inherits",
    "protocol_source",
    "result_manifest",
    "pretrained_weights",
)


def load_experiment_matrix(path: str | Path) -> ExperimentMatrix:
    """Load and validate the Phase 7 experiment matrix.

    Args:
        path: ``configs/detection_experiments.yaml``.

    Returns:
        The parsed matrix.

    Raises:
        ComparisonConfigError: If the file is missing, malformed, names the
            protected split, declares a metric outside the frozen set, or
            describes an experiment whose declared variable does not match its
            override set.
    """
    config_path = Path(path).expanduser()
    if not config_path.is_file():
        msg = f"Detection experiment matrix not found: {config_path.name}"
        raise ComparisonConfigError(msg)
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        msg = f"Detection experiment matrix is not valid YAML: {config_path.name} ({exc})"
        raise ComparisonConfigError(msg) from exc
    if not isinstance(raw, Mapping):
        msg = f"Detection experiment matrix must be a mapping: {config_path.name}"
        raise ComparisonConfigError(msg)
    return parse_experiment_matrix(raw)


def parse_experiment_matrix(raw: Mapping[str, Any]) -> ExperimentMatrix:
    """Validate a parsed experiment matrix mapping.

    Args:
        raw: The parsed mapping.

    Returns:
        The validated matrix.

    Raises:
        ComparisonConfigError: If the matrix is malformed or unsafe.
    """
    try:
        check_keys(raw, required=_MATRIX_KEYS, context="detection_experiments")
    except ConfigError as exc:
        raise ComparisonConfigError(str(exc)) from exc

    if contains_forbidden_split(dict(raw)):
        msg = (
            "detection_experiments references the protected split. Phase 7 selection happens "
            "on validation only, and the holdout may not appear in a comparison protocol."
        )
        raise ComparisonConfigError(msg)

    version = raw["schema_version"]
    if version != CONFIG_SCHEMA_VERSION:
        msg = (
            f"detection_experiments.schema_version {version!r} is unsupported; "
            f"expected {CONFIG_SCHEMA_VERSION}"
        )
        raise ComparisonConfigError(msg)

    support = _parse_support_rule(raw["support_rule"])
    metrics = _parse_metrics(raw["metrics"])
    margin = _parse_margin(raw["practical_equivalence_margin_map50_95"])
    memory = _parse_memory_policy(raw["memory_policy"])
    experiments = _parse_experiments(raw["experiments"])

    reference_id = str(raw["reference_experiment"])
    references = [item for item in experiments if item.role == REFERENCE_ROLE]
    if len(references) != 1 or references[0].experiment_id != reference_id:
        msg = (
            "detection_experiments: exactly one experiment must carry role "
            f"{REFERENCE_ROLE!r} and it must be {reference_id!r}"
        )
        raise ComparisonConfigError(msg)
    if references[0].protocol_source is None:
        msg = f"detection_experiments: the reference {reference_id!r} must name a protocol_source"
        raise ComparisonConfigError(msg)

    for candidate in experiments:
        if candidate.role != CANDIDATE_ROLE:
            continue
        if candidate.inherits != reference_id:
            msg = (
                f"detection_experiments.{candidate.experiment_id}: must inherit from the "
                f"reference {reference_id!r}, got {candidate.inherits!r}. A candidate that "
                "starts from anything else is not a one-variable comparison against D0."
            )
            raise ComparisonConfigError(msg)
        declared = candidate.declared_difference_fields
        overridden = frozenset(candidate.overrides)
        if declared != overridden:
            msg = (
                f"detection_experiments.{candidate.experiment_id}: the declared variable fields "
                f"{sorted(declared)} do not match the override set {sorted(overridden)}. "
                "An override that is not declared is an undeclared second variable; a declared "
                "field with no override is a variable that was never actually applied."
            )
            raise ComparisonConfigError(msg)
        if not candidate.intentional_fields:
            msg = (
                f"detection_experiments.{candidate.experiment_id}: a controlled experiment must "
                "declare at least one intentional field"
            )
            raise ComparisonConfigError(msg)

    return ExperimentMatrix(
        schema_version=int(version),
        phase=str(raw["phase"]),
        task=str(raw["task"]),
        reference_experiment=reference_id,
        support_rule=support,
        primary_selection_metric=metrics["primary_selection"],
        official_all_class_metric=metrics["official_all_class"],
        secondary_metrics=tuple(metrics["secondary"]),
        per_class_metrics=tuple(metrics["per_class"]),
        diagnostics=tuple(metrics["diagnostics"]),
        practical_equivalence_margin=margin,
        memory_policy=memory,
        experiments=experiments,
    )


def _parse_support_rule(data: Any) -> SupportRule:
    """Parse and validate the class-support rule.

    Args:
        data: Raw ``support_rule`` mapping.

    Returns:
        The parsed rule.

    Raises:
        ComparisonConfigError: If it is malformed, or its thresholds differ from
            the frozen constants.
    """
    if not isinstance(data, Mapping):
        msg = "detection_experiments.support_rule: must be a mapping"
        raise ComparisonConfigError(msg)
    try:
        check_keys(data, required=_SUPPORT_RULE_KEYS, context="detection_experiments.support_rule")
    except ConfigError as exc:
        raise ComparisonConfigError(str(exc)) from exc
    rule = SupportRule(
        min_positive_images=_int(
            data["min_validation_positive_images"],
            context="detection_experiments.support_rule.min_validation_positive_images",
        ),
        min_instances=_int(
            data["min_validation_instances"],
            context="detection_experiments.support_rule.min_validation_instances",
        ),
    )
    frozen = SupportRule()
    if rule != frozen:
        msg = (
            "detection_experiments.support_rule must state the frozen thresholds "
            f"({frozen.min_positive_images} images, {frozen.min_instances} instances). "
            "Relaxing them after D0 would let the support filter be tuned to a result."
        )
        raise ComparisonConfigError(msg)
    return rule


def _parse_metrics(data: Any) -> dict[str, Any]:
    """Parse and validate the metric declaration.

    Args:
        data: Raw ``metrics`` mapping.

    Returns:
        The validated metric names and lists.

    Raises:
        ComparisonConfigError: If it is malformed, or a metric name differs from
            the frozen one.
    """
    if not isinstance(data, Mapping):
        msg = "detection_experiments.metrics: must be a mapping"
        raise ComparisonConfigError(msg)
    try:
        check_keys(data, required=_METRICS_KEYS, context="detection_experiments.metrics")
    except ConfigError as exc:
        raise ComparisonConfigError(str(exc)) from exc

    expected = {
        "primary_selection": PRIMARY_SELECTION_METRIC,
        "official_all_class": OFFICIAL_ALL_CLASS_METRIC,
    }
    for key, value in expected.items():
        if str(data[key]) != value:
            msg = (
                f"detection_experiments.metrics.{key} must be {value!r}, got {data[key]!r}. "
                "The deciding metric is frozen; a new one would have to be argued for in "
                "advance, not selected after seeing a result."
            )
            raise ComparisonConfigError(msg)
    if tuple(str(item) for item in data["secondary"]) != SECONDARY_METRICS:
        msg = f"detection_experiments.metrics.secondary must be {list(SECONDARY_METRICS)}"
        raise ComparisonConfigError(msg)
    if tuple(str(item) for item in data["per_class"]) != PER_CLASS_METRICS:
        msg = f"detection_experiments.metrics.per_class must be {list(PER_CLASS_METRICS)}"
        raise ComparisonConfigError(msg)
    diagnostics = data["diagnostics"]
    if not isinstance(diagnostics, list) or not diagnostics:
        msg = "detection_experiments.metrics.diagnostics: must be a non-empty list"
        raise ComparisonConfigError(msg)
    return {
        "primary_selection": PRIMARY_SELECTION_METRIC,
        "official_all_class": OFFICIAL_ALL_CLASS_METRIC,
        "secondary": [str(item) for item in data["secondary"]],
        "per_class": [str(item) for item in data["per_class"]],
        "diagnostics": [str(item) for item in diagnostics],
    }


def _parse_margin(value: Any) -> Decimal:
    """Parse and validate the practical-equivalence margin.

    Args:
        value: Raw margin value.

    Returns:
        The margin as an exact decimal.

    Raises:
        ComparisonConfigError: If it differs from the frozen margin.
    """
    try:
        margin = Decimal(str(value))
    except (ArithmeticError, ValueError) as exc:
        msg = f"detection_experiments.practical_equivalence_margin_map50_95: {exc}"
        raise ComparisonConfigError(msg) from exc
    if margin != PRACTICAL_EQUIVALENCE_MARGIN:
        msg = (
            "detection_experiments.practical_equivalence_margin_map50_95 must be "
            f"{PRACTICAL_EQUIVALENCE_MARGIN}, got {value!r}. The margin is frozen before the "
            "results; changing it afterwards would choose the winner."
        )
        raise ComparisonConfigError(msg)
    return margin


def _parse_memory_policy(data: Any) -> dict[str, Any]:
    """Parse and validate the batch/memory policy.

    Args:
        data: Raw ``memory_policy`` mapping.

    Returns:
        The validated policy.

    Raises:
        ComparisonConfigError: If it is malformed or names the wrong
            classification.
    """
    if not isinstance(data, Mapping):
        msg = "detection_experiments.memory_policy: must be a mapping"
        raise ComparisonConfigError(msg)
    try:
        check_keys(
            data,
            required=_MEMORY_POLICY_KEYS,
            context="detection_experiments.memory_policy",
        )
    except ConfigError as exc:
        raise ComparisonConfigError(str(exc)) from exc
    if str(data["classification"]) != MEMORY_CONSTRAINT_REVIEW_REQUIRED:
        msg = (
            "detection_experiments.memory_policy.classification must be "
            f"{MEMORY_CONSTRAINT_REVIEW_REQUIRED!r}"
        )
        raise ComparisonConfigError(msg)
    mitigations = data["forbidden_mitigations"]
    if not isinstance(mitigations, list) or not mitigations:
        msg = "detection_experiments.memory_policy.forbidden_mitigations: must be a non-empty list"
        raise ComparisonConfigError(msg)
    return {
        "controlled_batch": _int(
            data["controlled_batch"],
            context="detection_experiments.memory_policy.controlled_batch",
        ),
        "on_cuda_oom": str(data["on_cuda_oom"]),
        "classification": MEMORY_CONSTRAINT_REVIEW_REQUIRED,
        "forbidden_mitigations": [str(item) for item in mitigations],
        "rationale": str(data["rationale"]),
    }


def _parse_experiments(data: Any) -> tuple[ExperimentDeclaration, ...]:
    """Parse and validate the experiment declarations.

    Args:
        data: Raw ``experiments`` list.

    Returns:
        The parsed declarations, in declaration order.

    Raises:
        ComparisonConfigError: If the list is malformed, an id is invalid or
            duplicated, or a declaration carries an unknown key.
    """
    if not isinstance(data, list) or not data:
        msg = "detection_experiments.experiments: must be a non-empty list"
        raise ComparisonConfigError(msg)
    declarations: list[ExperimentDeclaration] = []
    seen: set[str] = set()
    for index, entry in enumerate(data):
        context = f"detection_experiments.experiments[{index}]"
        if not isinstance(entry, Mapping):
            msg = f"{context}: must be a mapping"
            raise ComparisonConfigError(msg)
        try:
            check_keys(
                entry,
                required=_EXPERIMENT_KEYS,
                optional=_EXPERIMENT_OPTIONAL_KEYS,
                context=context,
            )
        except ConfigError as exc:
            raise ComparisonConfigError(str(exc)) from exc

        experiment_id = str(entry["experiment_id"])
        if not EXPERIMENT_ID_PATTERN.match(experiment_id):
            msg = (
                f"{context}.experiment_id {experiment_id!r} is not a valid identifier; "
                "expected a task letter followed by a number, such as 'D1'"
            )
            raise ComparisonConfigError(msg)
        if experiment_id in seen:
            msg = f"{context}.experiment_id {experiment_id!r} is declared more than once"
            raise ComparisonConfigError(msg)
        seen.add(experiment_id)

        role = str(entry["role"])
        if role not in EXPERIMENT_ROLES:
            msg = f"{context}.role {role!r} is not one of {list(EXPERIMENT_ROLES)}"
            raise ComparisonConfigError(msg)
        status = str(entry["status"])
        if status not in EXPERIMENT_STATUSES:
            msg = f"{context}.status {status!r} is not one of {list(EXPERIMENT_STATUSES)}"
            raise ComparisonConfigError(msg)

        overrides = entry["overrides"]
        if not isinstance(overrides, Mapping):
            msg = f"{context}.overrides: must be a mapping of field path to value"
            raise ComparisonConfigError(msg)
        for field_path in overrides:
            _check_field_path(str(field_path), context=f"{context}.overrides")

        for key in ("intentional_fields", "consequential_fields"):
            value = entry[key]
            if not isinstance(value, list):
                msg = f"{context}.{key}: must be a list of field paths"
                raise ComparisonConfigError(msg)
            for field_path in value:
                _check_field_path(str(field_path), context=f"{context}.{key}")

        declarations.append(
            ExperimentDeclaration(
                experiment_id=experiment_id,
                role=role,
                question=str(entry["question"]),
                status=status,
                intentional_variable=str(entry["intentional_variable"]),
                intentional_fields=tuple(str(item) for item in entry["intentional_fields"]),
                consequential_fields=tuple(str(item) for item in entry["consequential_fields"]),
                overrides={str(key): value for key, value in overrides.items()},
                inherits=_optional_str(entry.get("inherits")),
                protocol_source=_optional_str(entry.get("protocol_source")),
                result_manifest=_optional_str(entry.get("result_manifest")),
                hypothesis=str(entry["hypothesis"]),
                pretrained_weights=_parse_pretrained_weights(
                    entry.get("pretrained_weights"), context=context
                ),
            )
        )
    return tuple(declarations)


_PRETRAINED_WEIGHT_KEYS: tuple[str, ...] = (
    "identifier",
    "source_mechanism",
    "fingerprint_status",
    "commit_binary",
    "required_before_training",
)


def _parse_pretrained_weights(data: Any, *, context: str) -> dict[str, Any] | None:
    """Parse an experiment's starting-checkpoint policy.

    Args:
        data: Raw ``pretrained_weights`` mapping, or ``None``.
        context: Configuration section, used in error messages.

    Returns:
        The validated policy, or ``None`` when the experiment inherits the
        reference's weights unchanged.

    Raises:
        ComparisonConfigError: If the mapping is malformed, or declares that the
            binary should be committed. Checkpoints are referenced by digest and
            provenance record, never committed.
    """
    if data is None:
        return None
    if not isinstance(data, Mapping):
        msg = f"{context}.pretrained_weights: must be a mapping"
        raise ComparisonConfigError(msg)
    try:
        check_keys(
            data,
            required=_PRETRAINED_WEIGHT_KEYS,
            context=f"{context}.pretrained_weights",
        )
    except ConfigError as exc:
        raise ComparisonConfigError(str(exc)) from exc
    if data["commit_binary"] is not False:
        msg = f"{context}.pretrained_weights.commit_binary must be false; weights are not committed"
        raise ComparisonConfigError(msg)
    required = data["required_before_training"]
    if not isinstance(required, list) or not required:
        msg = f"{context}.pretrained_weights.required_before_training: must be a non-empty list"
        raise ComparisonConfigError(msg)
    return {
        "identifier": str(data["identifier"]),
        "source_mechanism": str(data["source_mechanism"]),
        "fingerprint_status": str(data["fingerprint_status"]),
        "commit_binary": False,
        "required_before_training": [str(item) for item in required],
    }


def _check_field_path(field_path: str, *, context: str) -> None:
    """Reject a field path that names a non-protocol or runtime field.

    Args:
        field_path: Dotted protocol field path.
        context: Configuration section, used in the error message.

    Raises:
        ComparisonConfigError: If the path is empty, or names a field excluded
            from the protocol comparison. Declaring such a field as the
            experimental variable would make the comparison vacuous.
    """
    if not field_path or field_path.startswith(".") or field_path.endswith("."):
        msg = f"{context}: {field_path!r} is not a valid dotted field path"
        raise ComparisonConfigError(msg)
    head = field_path.split(".", 1)[0]
    leaf = field_path.rsplit(".", 1)[-1]
    if head in NON_PROTOCOL_FIELDS or leaf in RUNTIME_FIELDS:
        msg = (
            f"{context}: {field_path!r} is not a protocol field. Identifiers, descriptions, "
            "timestamps and output paths are excluded from the protocol comparison, so they "
            "cannot be an experimental variable."
        )
        raise ComparisonConfigError(msg)


def _optional_str(value: Any) -> str | None:
    """Normalise an optional string field.

    Args:
        value: Raw value.

    Returns:
        The string, or ``None`` when absent or empty.
    """
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _int(value: Any, *, context: str) -> int:
    """Parse an integer configuration value strictly.

    Args:
        value: Raw value.
        context: Field name, used in the error message.

    Returns:
        The integer.

    Raises:
        ComparisonConfigError: If the value is not an integer.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        msg = f"{context}: must be an integer, got {value!r}"
        raise ComparisonConfigError(msg)
    return value


# --- protocol resolution and compatibility -----------------------------------


def flatten_protocol(protocol: Mapping[str, Any], *, prefix: str = "") -> dict[str, Any]:
    """Flatten a protocol mapping to dotted leaf paths.

    Non-protocol and runtime fields are dropped, so identity, prose, timestamps
    and output locations can never register as protocol differences. Flattening
    is idempotent: an already-flat mapping keyed by dotted paths passes through
    unchanged, so callers need not track which form they hold.

    Args:
        protocol: A protocol mapping, e.g. a parsed baseline configuration.
        prefix: Path prefix used during recursion.

    Returns:
        Leaf values keyed by dotted path.
    """
    flat: dict[str, Any] = {}
    for key, value in protocol.items():
        name = str(key)
        if not prefix and name in NON_PROTOCOL_FIELDS:
            continue
        if name in RUNTIME_FIELDS:
            continue
        path = f"{prefix}{name}"
        if isinstance(value, Mapping):
            flat.update(flatten_protocol(value, prefix=f"{path}."))
        elif isinstance(value, (list, tuple)):
            flat[path] = list(value)
        else:
            flat[path] = value
    return flat


def resolve_candidate_protocol(
    reference_protocol: Mapping[str, Any],
    declaration: ExperimentDeclaration,
) -> dict[str, Any]:
    """Apply a candidate's declared overrides to the reference protocol.

    Inheritance is the mechanism that makes the one-variable discipline
    structural rather than aspirational: the candidate has no protocol of its
    own to drift from the baseline, only a declared override set.

    Args:
        reference_protocol: The reference experiment's flattened protocol.
        declaration: The candidate declaration.

    Returns:
        The candidate's flattened protocol.

    Raises:
        ComparisonError: If an override names a field the reference protocol
            does not have. A typo there would otherwise add a field instead of
            changing one, and the run would silently keep the baseline value.
    """
    flat = dict(flatten_protocol(reference_protocol))
    for field_path, value in declaration.overrides.items():
        if field_path not in flat:
            msg = (
                f"{declaration.experiment_id}: override {field_path!r} names a field the "
                f"reference protocol does not declare. Known fields: {sorted(flat)}"
            )
            raise ComparisonError(msg)
        flat[field_path] = value
    flat["experiment_id"] = declaration.experiment_id
    return {key: flat[key] for key in sorted(flat) if key not in NON_PROTOCOL_FIELDS}


def protocol_differences(
    reference: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> tuple[str, ...]:
    """List the field paths on which two flattened protocols differ.

    Args:
        reference: The reference protocol, flattened.
        candidate: The candidate protocol, flattened.

    Returns:
        The differing field paths, sorted. A field present in only one of the
        two counts as a difference.
    """
    left = flatten_protocol(reference)
    right = flatten_protocol(candidate)
    differing = {key for key in set(left) | set(right) if left.get(key) != right.get(key)}
    return tuple(sorted(differing))


@dataclass(frozen=True)
class ProtocolCompatibility:
    """The verdict on whether two protocols form a one-variable comparison.

    Attributes:
        experiment_id: The candidate's id.
        reference_experiment: The reference's id.
        intentional_variable: The one thing the candidate was meant to vary.
        declared_fields: Field paths the candidate was permitted to differ in.
        observed_differences: Field paths on which the protocols actually differ.
        undeclared_differences: Observed differences that were not declared.
        unapplied_declarations: Declared fields that do not actually differ.
        compatible: Whether the comparison is valid.
    """

    experiment_id: str
    reference_experiment: str
    intentional_variable: str
    declared_fields: tuple[str, ...]
    observed_differences: tuple[str, ...]
    undeclared_differences: tuple[str, ...]
    unapplied_declarations: tuple[str, ...]
    compatible: bool

    def as_dict(self) -> dict[str, Any]:
        """Serialise the verdict.

        Returns:
            A JSON-serialisable mapping.
        """
        return {
            "experiment_id": self.experiment_id,
            "reference_experiment": self.reference_experiment,
            "intentional_variable": self.intentional_variable,
            "declared_fields": list(self.declared_fields),
            "observed_differences": list(self.observed_differences),
            "undeclared_differences": list(self.undeclared_differences),
            "unapplied_declarations": list(self.unapplied_declarations),
            "compatible": self.compatible,
        }


def check_protocol_compatibility(
    reference_protocol: Mapping[str, Any],
    candidate_protocol: Mapping[str, Any],
    declaration: ExperimentDeclaration,
    *,
    reference_experiment: str,
) -> ProtocolCompatibility:
    """Verify that a candidate differs from the reference only where declared.

    An undeclared difference makes the comparison invalid: any metric change
    could then be attributed to either variable, which is exactly the confound a
    controlled experiment exists to avoid. A declared field that does not
    actually differ is equally a defect - it means the experiment did not apply
    its own variable.

    Args:
        reference_protocol: The reference protocol.
        candidate_protocol: The candidate protocol.
        declaration: The candidate declaration.
        reference_experiment: Id of the reference.

    Returns:
        The verdict, whether or not it is compatible.
    """
    declared = declaration.declared_difference_fields
    observed = protocol_differences(reference_protocol, candidate_protocol)
    undeclared = tuple(sorted(set(observed) - declared))
    unapplied = tuple(sorted(declared - set(observed)))
    return ProtocolCompatibility(
        experiment_id=declaration.experiment_id,
        reference_experiment=reference_experiment,
        intentional_variable=declaration.intentional_variable,
        declared_fields=tuple(sorted(declared)),
        observed_differences=observed,
        undeclared_differences=undeclared,
        unapplied_declarations=unapplied,
        compatible=not undeclared and not unapplied,
    )


# --- experiment records -------------------------------------------------------


def _path_value(data: Mapping[str, Any], path: str) -> Any:
    """Read a dotted path out of a nested mapping.

    Args:
        data: The mapping.
        path: Dotted path.

    Returns:
        The value, or ``None`` when any segment is absent.
    """
    current: Any = data
    for segment in path.split("."):
        if not isinstance(current, Mapping) or segment not in current:
            return None
        current = current[segment]
    return current


def data_fingerprints(manifest: Mapping[str, Any]) -> dict[str, str]:
    """Extract the data fingerprints a comparison must match on.

    Args:
        manifest: A parsed result manifest.

    Returns:
        Fingerprint values keyed by their dotted manifest path.

    Raises:
        ComparisonError: If any required fingerprint is missing. A comparison
            that cannot prove both runs saw the same data is refused rather than
            annotated.
    """
    found: dict[str, str] = {}
    missing: list[str] = []
    for path in FINGERPRINT_PATHS:
        value = _path_value(manifest, path)
        if not isinstance(value, str) or not value:
            missing.append(path)
        else:
            found[path] = value
    if missing:
        experiment = manifest.get("experiment_id", "<unknown>")
        msg = f"{experiment}: result manifest is missing data fingerprint(s): {missing}"
        raise ComparisonError(msg)
    return found


def experiment_fingerprint_of(manifest: Mapping[str, Any]) -> str:
    """Read a result manifest's experiment fingerprint.

    Args:
        manifest: A parsed result manifest.

    Returns:
        The recorded fingerprint.

    Raises:
        ComparisonError: If no accepted fingerprint key carries a value.
    """
    for key in EXPERIMENT_FINGERPRINT_KEYS:
        value = manifest.get(key)
        if isinstance(value, str) and value:
            return value
    experiment = manifest.get("experiment_id", "<unknown>")
    msg = (
        f"{experiment}: result manifest carries no experiment fingerprint; expected one of "
        f"{list(EXPERIMENT_FINGERPRINT_KEYS)}"
    )
    raise ComparisonError(msg)


@dataclass(frozen=True)
class ExperimentRecord:
    """One experiment's comparison record, derived from its result manifest.

    Attributes:
        experiment_id: The experiment's id.
        model: Model family and size.
        imgsz: Model input size.
        parameters: Model parameter count, or ``None`` when not recorded.
        best_checkpoint_sha256: The reported checkpoint's digest.
        experiment_sha256: The experiment fingerprint.
        supported_macro: The primary selection metric.
        all_class_map50_95: The official all-class metric.
        secondary: The secondary global metrics.
        per_class: The per-class metric tables.
        support: The frozen class-support records.
        fingerprints: The data fingerprints the comparison matches on.
        diagnostics: Diagnostic values carried through for interpretation.
    """

    experiment_id: str
    model: str
    imgsz: int
    parameters: int | None
    best_checkpoint_sha256: str
    experiment_sha256: str
    supported_macro: Decimal
    all_class_map50_95: Decimal
    secondary: dict[str, Decimal]
    per_class: dict[str, dict[str, Any]]
    support: tuple[ClassSupport, ...]
    fingerprints: dict[str, str]
    diagnostics: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        """Serialise the record.

        Returns:
            A JSON-serialisable mapping.
        """
        return {
            "experiment_id": self.experiment_id,
            "model": self.model,
            "imgsz": self.imgsz,
            "parameters": self.parameters,
            "best_checkpoint_sha256": self.best_checkpoint_sha256,
            "experiment_sha256": self.experiment_sha256,
            "metrics": {
                PRIMARY_SELECTION_METRIC: _as_float(self.supported_macro),
                OFFICIAL_ALL_CLASS_METRIC: _as_float(self.all_class_map50_95),
                **{name: _as_float(value) for name, value in self.secondary.items()},
            },
            "selection_metric_classes": list(supported_class_names(self.support)),
            "descriptive_classes": list(descriptive_class_names(self.support)),
            "per_class_metrics": {
                name: dict(table) for name, table in sorted(self.per_class.items())
            },
            "class_support": [record.as_dict() for record in self.support],
            "data_fingerprints": dict(self.fingerprints),
            "diagnostics": dict(self.diagnostics),
        }


def build_experiment_record(
    manifest: Mapping[str, Any],
    support: Sequence[ClassSupport],
) -> ExperimentRecord:
    """Derive an experiment's comparison record from its result manifest.

    Every number is read from the committed manifest. Nothing is recomputed from
    a checkpoint, and nothing is filled in from a default.

    Args:
        manifest: A parsed result manifest.
        support: The frozen class-support records.

    Returns:
        The comparison record.

    Raises:
        ComparisonError: If the manifest's class map disagrees with the support
            records, or a required metric is missing.
    """
    experiment_id = str(manifest.get("experiment_id", ""))
    if not EXPERIMENT_ID_PATTERN.match(experiment_id):
        msg = f"result manifest experiment_id {experiment_id!r} is not a valid identifier"
        raise ComparisonError(msg)

    class_map = manifest.get("class_map")
    if not isinstance(class_map, Mapping):
        msg = f"{experiment_id}: result manifest carries no class_map"
        raise ComparisonError(msg)
    declared = {record.class_name: record.class_index for record in support}
    if {str(key): int(value) for key, value in class_map.items()} != declared:
        msg = (
            f"{experiment_id}: the manifest class map {dict(class_map)!r} disagrees with the "
            f"frozen class map {declared!r}. Comparing per-class metrics across different "
            "class maps would compare different classes."
        )
        raise ComparisonError(msg)

    per_class = manifest.get("per_class_metrics")
    if not isinstance(per_class, Mapping):
        msg = f"{experiment_id}: result manifest carries no per_class_metrics"
        raise ComparisonError(msg)
    missing = sorted(set(declared) - set(per_class))
    if missing:
        msg = f"{experiment_id}: per_class_metrics is missing class(es) {missing}"
        raise ComparisonError(msg)

    global_metrics = manifest.get("validation_metrics")
    if not isinstance(global_metrics, Mapping):
        msg = f"{experiment_id}: result manifest carries no validation_metrics"
        raise ComparisonError(msg)

    supported = supported_class_names(support)
    macro = supported_macro(per_class, supported)
    all_class = _decimal(
        global_metrics.get(OFFICIAL_ALL_CLASS_METRIC),
        context=f"{experiment_id}.validation_metrics.{OFFICIAL_ALL_CLASS_METRIC}",
    )
    secondary = {
        name: _decimal(
            global_metrics.get(name), context=f"{experiment_id}.validation_metrics.{name}"
        )
        for name in SECONDARY_METRICS
    }

    imgsz = _path_value(manifest, "resolved_training_arguments.imgsz")
    if imgsz is None:
        imgsz = _path_value(manifest, "validation_configuration.imgsz")
    if isinstance(imgsz, bool) or not isinstance(imgsz, int):
        msg = f"{experiment_id}: result manifest does not record an integer imgsz"
        raise ComparisonError(msg)

    parameters = manifest.get("model_parameters")
    if isinstance(parameters, bool) or not isinstance(parameters, int):
        parameters = None

    return ExperimentRecord(
        experiment_id=experiment_id,
        model=str(manifest.get("model", "")),
        imgsz=imgsz,
        parameters=parameters,
        best_checkpoint_sha256=str(_path_value(manifest, "best_checkpoint.sha256") or ""),
        experiment_sha256=experiment_fingerprint_of(manifest),
        supported_macro=macro,
        all_class_map50_95=all_class,
        secondary=secondary,
        per_class={str(name): dict(table) for name, table in per_class.items()},
        support=tuple(support),
        fingerprints=data_fingerprints(manifest),
        diagnostics=_diagnostics(manifest),
    )


def _diagnostics(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Collect the diagnostic values carried into a comparison record.

    Recall in particular. D0's recall sits far below its precision, which is a
    reading worth tracking across experiments - but it explains a movement in
    the selection metric rather than deciding the winner itself.

    Args:
        manifest: A parsed result manifest.

    Returns:
        A JSON-serialisable mapping of the available diagnostics.
    """
    keys = (
        "best_epoch",
        "epochs_completed",
        "early_stopped",
        "termination_mode",
        "resolved_optimizer",
        "training_duration_seconds",
    )
    diagnostics: dict[str, Any] = {key: manifest.get(key) for key in keys if key in manifest}
    for path, name in (
        ("resolved_optimizer_determination.effective_lr0", "effective_lr0"),
        ("framework_validation_speed_ms_per_image.inference", "validation_inference_ms_per_image"),
    ):
        value = _path_value(manifest, path)
        if value is not None:
            diagnostics[name] = value
    if "confusion_matrix" in manifest:
        diagnostics["confusion_matrix_recorded"] = True
    return diagnostics


# --- comparison and selection -------------------------------------------------


def classify_delta(delta: Decimal, *, margin: Decimal = PRACTICAL_EQUIVALENCE_MARGIN) -> str:
    """Classify a candidate's difference from the reference.

    The margin boundary is inclusive on the equivalence side: a delta of exactly
    the margin is practically equivalent, not an improvement.

    Args:
        delta: Candidate metric minus reference metric.
        margin: The practical-equivalence margin.

    Returns:
        One of :data:`DELTA_CLASSIFICATIONS`.
    """
    if delta > margin:
        return IMPROVED
    if delta < -margin:
        return REGRESSED
    return PRACTICALLY_EQUIVALENT


@dataclass(frozen=True)
class SelectionOutcome:
    """The frozen selection logic's verdict on a set of experiments.

    Attributes:
        case: One of :data:`SELECTION_CASES`.
        preferred_experiment: The experiment the rule retains or elects.
        leader: The best-scoring candidate, or ``None`` when there is none.
        runner_up: The next-best candidate, or ``None``.
        leader_separation: Leader minus runner-up, or ``None``.
        rationale: Why this case applies, in one sentence.
        efficiency_comparison_required: Whether a later benchmark must decide.
    """

    case: str
    preferred_experiment: str | None
    leader: str | None
    runner_up: str | None
    leader_separation: Decimal | None
    rationale: str
    efficiency_comparison_required: bool

    def as_dict(self) -> dict[str, Any]:
        """Serialise the verdict.

        Returns:
            A JSON-serialisable mapping.
        """
        return {
            "case": self.case,
            "preferred_experiment": self.preferred_experiment,
            "leader": self.leader,
            "runner_up": self.runner_up,
            "leader_separation": (
                None if self.leader_separation is None else _as_float(self.leader_separation)
            ),
            "rationale": self.rationale,
            "efficiency_comparison_required": self.efficiency_comparison_required,
        }


def select(
    reference: ExperimentRecord,
    candidates: Sequence[ExperimentRecord],
    *,
    margin: Decimal = PRACTICAL_EQUIVALENCE_MARGIN,
    failures: Sequence[str] = (),
) -> SelectionOutcome:
    """Apply the frozen Phase 7 selection logic.

    The four cases, fixed before any candidate result existed:

    * **A** - no candidate clears the reference by more than the margin. The
      reference is retained, because it is the lower-complexity model and the
      difference does not establish an ordering.
    * **B** - a candidate clears the reference by more than the margin *and*
      separates from the runner-up by more than the margin. It is the
      validation-performance leader.
    * **C** - a candidate clears the reference but does not separate from the
      runner-up. No winner is declared from a difference that small; a later
      controlled efficiency comparison decides.
    * **D** - an execution or protocol failure. A failure is not evidence that
      the model is worse; it returns for protocol review.

    Case C covers slightly more ground than "both candidates cleared the
    reference and tied": it also catches the case where only one cleared it but
    the two candidates are within the margin of each other. The reasoning is the
    same in both - the candidates cannot be ordered - and leaving that region
    undefined would mean deciding it after seeing the numbers.

    Args:
        reference: The reference record.
        candidates: The candidate records.
        margin: The practical-equivalence margin.
        failures: Ids of experiments that failed to execute or violated their
            protocol.

    Returns:
        The verdict.
    """
    if failures:
        return SelectionOutcome(
            case=CASE_D,
            preferred_experiment=None,
            leader=None,
            runner_up=None,
            leader_separation=None,
            rationale=(
                f"execution or protocol failure in {sorted(failures)}; a failed run is not "
                "evidence of model inferiority and no ranking is derived from it"
            ),
            efficiency_comparison_required=False,
        )

    ranked = sorted(
        candidates,
        key=lambda record: (-record.supported_macro, record.experiment_id),
    )
    improving = [
        record
        for record in ranked
        if classify_delta(record.supported_macro - reference.supported_macro, margin=margin)
        == IMPROVED
    ]
    if not improving:
        return SelectionOutcome(
            case=CASE_A,
            preferred_experiment=reference.experiment_id,
            leader=None,
            runner_up=None,
            leader_separation=None,
            rationale=(
                f"no candidate improves {PRIMARY_SELECTION_METRIC} over "
                f"{reference.experiment_id} by more than {margin}, so the lower-complexity "
                "baseline is retained"
            ),
            efficiency_comparison_required=False,
        )

    leader = ranked[0]
    runner_up = ranked[1] if len(ranked) > 1 else None
    separation = None if runner_up is None else leader.supported_macro - runner_up.supported_macro
    if separation is None or separation > margin:
        return SelectionOutcome(
            case=CASE_B,
            preferred_experiment=leader.experiment_id,
            leader=leader.experiment_id,
            runner_up=None if runner_up is None else runner_up.experiment_id,
            leader_separation=separation,
            rationale=(
                f"{leader.experiment_id} improves {PRIMARY_SELECTION_METRIC} over "
                f"{reference.experiment_id} by more than {margin} and separates from every "
                "other candidate by more than the same margin"
            ),
            efficiency_comparison_required=False,
        )
    return SelectionOutcome(
        case=CASE_C,
        preferred_experiment=None,
        leader=leader.experiment_id,
        runner_up=runner_up.experiment_id,
        leader_separation=separation,
        rationale=(
            f"{leader.experiment_id} and {runner_up.experiment_id} lie within {margin} of each "
            f"other on {PRIMARY_SELECTION_METRIC}, so validation performance does not order "
            "them; a later controlled efficiency comparison decides"
        ),
        efficiency_comparison_required=True,
    )


def compare(
    matrix: ExperimentMatrix,
    records: Mapping[str, ExperimentRecord],
    *,
    compatibility: Sequence[ProtocolCompatibility] = (),
    failures: Sequence[str] = (),
) -> dict[str, Any]:
    """Build the complete comparison record for the declared experiments.

    Args:
        matrix: The frozen experiment matrix.
        records: Comparison records keyed by experiment id. The reference must be
            present; candidates without a record are reported as not executed.
        compatibility: Protocol-compatibility verdicts to embed and enforce.
        failures: Ids of experiments that failed to execute.

    Returns:
        A JSON-serialisable comparison record.

    Raises:
        ComparisonError: If the reference record is absent, an experiment is not
            declared in the matrix, a candidate's data fingerprints differ from
            the reference's, or a compatibility verdict is incompatible.
    """
    reference_id = matrix.reference_experiment
    if reference_id not in records:
        msg = f"comparison requires the reference record {reference_id!r}"
        raise ComparisonError(msg)
    for experiment_id in records:
        matrix.declaration(experiment_id)

    reference = records[reference_id]
    margin = matrix.practical_equivalence_margin
    verdicts = {item.experiment_id: item for item in compatibility}
    incompatible = sorted(item.experiment_id for item in compatibility if not item.compatible)
    if incompatible:
        msg = (
            f"protocol compatibility failed for {incompatible}; a comparison whose runs differ "
            "in an undeclared parameter cannot attribute any metric change to its variable"
        )
        raise ComparisonError(msg)

    rows: list[dict[str, Any]] = []
    present: list[ExperimentRecord] = []
    for declaration in matrix.candidates:
        record = records.get(declaration.experiment_id)
        if record is None:
            rows.append(
                {
                    "experiment_id": declaration.experiment_id,
                    "status": FROZEN_NOT_EXECUTED,
                    "intentional_variable": declaration.intentional_variable,
                    "metrics": None,
                    "delta": None,
                }
            )
            continue
        if record.fingerprints != reference.fingerprints:
            differing = sorted(
                path
                for path in set(record.fingerprints) | set(reference.fingerprints)
                if record.fingerprints.get(path) != reference.fingerprints.get(path)
            )
            msg = (
                f"{record.experiment_id} and {reference_id} disagree on data fingerprint(s) "
                f"{differing}. They did not train and validate on the same data, so no metric "
                "difference between them is attributable to the declared variable."
            )
            raise ComparisonError(msg)
        present.append(record)
        delta = record.supported_macro - reference.supported_macro
        rows.append(
            {
                "experiment_id": record.experiment_id,
                "status": EXECUTED,
                "intentional_variable": declaration.intentional_variable,
                "metrics": record.as_dict()["metrics"],
                "delta": {
                    PRIMARY_SELECTION_METRIC: _as_float(delta),
                    OFFICIAL_ALL_CLASS_METRIC: _as_float(
                        record.all_class_map50_95 - reference.all_class_map50_95
                    ),
                    "classification": classify_delta(delta, margin=margin),
                },
                "protocol_compatibility": (
                    verdicts[record.experiment_id].as_dict()
                    if record.experiment_id in verdicts
                    else None
                ),
            }
        )

    outcome = select(reference, present, margin=margin, failures=failures)
    return {
        "schema_version": POLICY_SCHEMA_VERSION,
        "experiment_matrix_sha256": matrix.fingerprint(),
        "reference_experiment": reference_id,
        "primary_selection_metric": PRIMARY_SELECTION_METRIC,
        "official_all_class_metric": OFFICIAL_ALL_CLASS_METRIC,
        "practical_equivalence_margin": str(matrix.practical_equivalence_margin),
        "reference": reference.as_dict(),
        "candidates": rows,
        "selection": outcome.as_dict(),
    }
