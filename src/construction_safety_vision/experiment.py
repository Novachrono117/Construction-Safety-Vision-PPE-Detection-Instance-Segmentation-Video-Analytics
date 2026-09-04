"""Experiment protocols, declared before the experiment runs.

An experiment whose settings are written down afterwards is not an experiment,
it is a description of whatever happened. So the baseline protocol lives in a
versioned file, parsed strictly, and names its metric hierarchy *before* any
model is trained - which is what makes "the primary metric is mAP@0.50:0.95" a
commitment rather than a choice made once the numbers were visible.

Two rules are enforced here rather than trusted.

**No protocol may name the holdout.** A training configuration is one of the
easiest places for the protected split to enter by accident, so a configuration
that mentions it is rejected at parse time.

**The metric hierarchy is fixed and ordered.** A primary metric, then
secondaries. Adding a composite metric after seeing results is the classic way
to turn a disappointing run into a successful one, and the parser refuses to
load a protocol whose primary metric is not among the declared, allowed set.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from construction_safety_vision.config import ConfigError, check_keys
from construction_safety_vision.data.materialization import digest

CONFIG_SCHEMA_VERSION = 1
"""Schema version of ``configs/detection_baseline.yaml``."""

EXPERIMENT_ID_PATTERN = re.compile(r"^[A-Z]\d+[A-Za-z0-9_-]*$")
"""Experiment ids look like ``D0``, ``D1``, ``S0``: a task letter then a number."""

DEVELOPMENT_SPLITS: tuple[str, ...] = ("train", "validation")
"""The only splits an experiment protocol may name."""

FORBIDDEN_SPLIT = "test"
"""The protected split, which no protocol may reference."""

ALLOWED_PRIMARY_METRICS: tuple[str, ...] = (
    "mAP@0.50:0.95",
    "mask_mAP@0.50:0.95",
)
"""Primary metrics a protocol may declare.

Deliberately short. The primary metric of a detection or segmentation experiment
is the standard COCO summary metric; anything else would need arguing for in
advance, in writing, not selecting from a menu after the fact.
"""


class ExperimentConfigError(ConfigError):
    """Raised when an experiment protocol is missing, malformed or unsafe."""


@dataclass(frozen=True)
class MetricHierarchy:
    """The metrics an experiment commits to reporting, in order.

    Attributes:
        primary: The single metric that decides the experiment's headline result.
        secondary: Supporting global metrics.
        per_class: Metrics reported for each class.
        artifacts: Diagnostic outputs preserved alongside the numbers.
    """

    primary: str
    secondary: tuple[str, ...]
    per_class: tuple[str, ...]
    artifacts: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        """Serialise for the manifest and the protocol fingerprint.

        Returns:
            A JSON-serialisable mapping.
        """
        return {
            "primary": self.primary,
            "secondary": list(self.secondary),
            "per_class": list(self.per_class),
            "artifacts": list(self.artifacts),
        }


@dataclass(frozen=True)
class DetectionBaselineConfig:
    """The frozen protocol of one detection experiment.

    Attributes:
        schema_version: Version of this configuration schema.
        experiment_id: Stable identifier, e.g. ``D0``.
        task: Task name.
        description: What question the experiment answers.
        model: Model family and size.
        weights: Whether training starts from pretrained weights.
        weight_identifier: The exact pretrained checkpoint name.
        dataset_config: Repository-relative adapter dataset descriptor.
        splits: Splits the experiment may read.
        seed: Master seed.
        device_requirement: What hardware the full run requires.
        training: The complete hyperparameter set.
        checkpoint_selection: The rule naming which checkpoint is reported.
        metrics: The predeclared metric hierarchy.
        rare_class: The class whose results carry a small-sample limitation.
        rare_class_limitation: The limitation, recorded before training.
    """

    schema_version: int
    experiment_id: str
    task: str
    description: str
    model: str
    weights: str
    weight_identifier: str
    dataset_config: str
    splits: tuple[str, ...]
    seed: int
    device_requirement: str
    training: dict[str, Any]
    checkpoint_selection: str
    metrics: MetricHierarchy
    rare_class: str
    rare_class_limitation: str

    def as_dict(self) -> dict[str, Any]:
        """Serialise the protocol for hashing and reporting.

        Returns:
            A JSON-serialisable mapping of the protocol content.
        """
        return {
            "schema_version": self.schema_version,
            "experiment_id": self.experiment_id,
            "task": self.task,
            "description": self.description,
            "model": self.model,
            "weights": self.weights,
            "weight_identifier": self.weight_identifier,
            "dataset_config": self.dataset_config,
            "splits": list(self.splits),
            "seed": self.seed,
            "device_requirement": self.device_requirement,
            "training": self.training,
            "checkpoint_selection": self.checkpoint_selection,
            "metrics": self.metrics.as_dict(),
            "rare_class": self.rare_class,
            "rare_class_limitation": self.rare_class_limitation,
        }

    def fingerprint(self) -> str:
        """Hash the protocol, so a result can name the rules that produced it.

        Returns:
            A SHA-256 hex digest over the protocol content only.
        """
        return digest(self.as_dict())


REQUIRED_TRAINING_KEYS: tuple[str, ...] = (
    "epochs",
    "imgsz",
    "batch",
    "optimizer",
    "lr0",
    "lrf",
    "momentum",
    "weight_decay",
    "warmup_epochs",
    "patience",
    "augmentation",
    "amp",
    "deterministic",
    "workers",
    "pretrained",
    "cos_lr",
    "close_mosaic",
    "val",
)
"""Every hyperparameter the protocol must state explicitly.

Listed exhaustively on purpose. "We used the defaults" is not a record: the
defaults change between library versions, and a report that cannot name the
learning rate it trained at cannot be reproduced.
"""


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


def _check_top_level(raw: Mapping[str, Any]) -> None:
    """Validate the protocol's top-level keys.

    Args:
        raw: The parsed mapping.

    Raises:
        ConfigError: If keys are missing or unknown.
    """
    check_keys(
        raw,
        required=(
            "schema_version",
            "experiment_id",
            "task",
            "description",
            "model",
            "weights",
            "weight_identifier",
            "dataset_config",
            "splits",
            "seed",
            "device_requirement",
            "training",
            "checkpoint_selection",
            "metrics",
            "rare_class",
            "rare_class_limitation",
        ),
        context="detection_baseline",
    )


def load_detection_baseline_config(path: str | Path) -> DetectionBaselineConfig:
    """Load and validate a detection experiment protocol.

    Args:
        path: Configuration file.

    Returns:
        The parsed protocol.

    Raises:
        ExperimentConfigError: If the file is missing, malformed, names the
            protected split, or declares a metric hierarchy outside the allowed
            set.
    """
    config_path = Path(path).expanduser()
    if not config_path.is_file():
        msg = f"Detection baseline configuration not found: {config_path.name}"
        raise ExperimentConfigError(msg)
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        msg = f"Detection baseline configuration is not valid YAML: {config_path.name} ({exc})"
        raise ExperimentConfigError(msg) from exc
    if not isinstance(raw, Mapping):
        msg = f"Detection baseline configuration must be a mapping: {config_path.name}"
        raise ExperimentConfigError(msg)

    try:
        _check_top_level(raw)
    except ConfigError as exc:
        raise ExperimentConfigError(str(exc)) from exc

    experiment_id = str(raw["experiment_id"])
    if not EXPERIMENT_ID_PATTERN.match(experiment_id):
        msg = (
            f"detection_baseline.experiment_id {experiment_id!r} is not a valid identifier; "
            "expected a task letter followed by a number, such as 'D0'"
        )
        raise ExperimentConfigError(msg)

    splits = raw["splits"]
    if not isinstance(splits, list) or tuple(splits) != DEVELOPMENT_SPLITS:
        msg = (
            f"detection_baseline.splits must be exactly {list(DEVELOPMENT_SPLITS)}, got "
            f"{splits!r}. An experiment protocol never names the holdout."
        )
        raise ExperimentConfigError(msg)
    if _contains_forbidden_split(dict(raw)):
        msg = (
            "detection_baseline references the protected split. The holdout may not appear in "
            "a training protocol, a dataset descriptor or a metric declaration."
        )
        raise ExperimentConfigError(msg)

    training = raw["training"]
    if not isinstance(training, Mapping):
        msg = "detection_baseline.training: must be a mapping"
        raise ExperimentConfigError(msg)
    try:
        check_keys(training, required=REQUIRED_TRAINING_KEYS, context="detection_baseline.training")
    except ConfigError as exc:
        raise ExperimentConfigError(str(exc)) from exc

    metrics = _parse_metrics(raw["metrics"])

    return DetectionBaselineConfig(
        schema_version=int(raw["schema_version"]),
        experiment_id=experiment_id,
        task=str(raw["task"]),
        description=str(raw["description"]),
        model=str(raw["model"]),
        weights=str(raw["weights"]),
        weight_identifier=str(raw["weight_identifier"]),
        dataset_config=str(raw["dataset_config"]),
        splits=DEVELOPMENT_SPLITS,
        seed=int(raw["seed"]),
        device_requirement=str(raw["device_requirement"]),
        training=dict(training),
        checkpoint_selection=str(raw["checkpoint_selection"]),
        metrics=metrics,
        rare_class=str(raw["rare_class"]),
        rare_class_limitation=str(raw["rare_class_limitation"]),
    )


def _parse_metrics(data: Any) -> MetricHierarchy:
    """Parse and validate the predeclared metric hierarchy.

    Args:
        data: Raw ``metrics`` mapping.

    Returns:
        The parsed hierarchy.

    Raises:
        ExperimentConfigError: If it is malformed, or the primary metric is not
            one of the allowed summary metrics.
    """
    if not isinstance(data, Mapping):
        msg = "detection_baseline.metrics: must be a mapping"
        raise ExperimentConfigError(msg)
    try:
        check_keys(
            data,
            required=("primary", "secondary", "per_class", "artifacts"),
            context="detection_baseline.metrics",
        )
    except ConfigError as exc:
        raise ExperimentConfigError(str(exc)) from exc
    primary = str(data["primary"])
    if primary not in ALLOWED_PRIMARY_METRICS:
        msg = (
            f"detection_baseline.metrics.primary {primary!r} is not one of "
            f"{list(ALLOWED_PRIMARY_METRICS)}. A bespoke primary metric would have to be "
            "argued for in advance, not selected after seeing results."
        )
        raise ExperimentConfigError(msg)
    for name in ("secondary", "per_class", "artifacts"):
        if not isinstance(data[name], list) or not data[name]:
            msg = f"detection_baseline.metrics.{name}: must be a non-empty list"
            raise ExperimentConfigError(msg)
    if primary in data["secondary"]:
        msg = "detection_baseline.metrics: the primary metric must not repeat as a secondary"
        raise ExperimentConfigError(msg)
    return MetricHierarchy(
        primary=primary,
        secondary=tuple(str(value) for value in data["secondary"]),
        per_class=tuple(str(value) for value in data["per_class"]),
        artifacts=tuple(str(value) for value in data["artifacts"]),
    )
