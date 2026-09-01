"""Typed experiment configuration.

Every experiment is driven by a versioned YAML file under ``configs/`` and is
parsed into frozen dataclasses. Parsing is strict: unknown keys raise instead of
being ignored, so a typo in a configuration file cannot silently turn an
experiment into a different one than the report claims.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from construction_safety_vision.paths import ProjectRootNotFoundError, find_project_root

CANONICAL_TASKS: frozenset[str] = frozenset({"instance_segmentation", "object_detection"})
"""Tasks accepted as the canonical annotation format of the source dataset."""

RATIO_TOLERANCE = 1e-6
"""Absolute tolerance used when checking that split ratios sum to one."""


class ConfigError(ValueError):
    """Raised when a configuration file is missing, malformed or incomplete."""


def _check_keys(
    data: Mapping[str, Any],
    *,
    required: Iterable[str],
    optional: Iterable[str] = (),
    context: str,
) -> None:
    """Validate that a mapping contains exactly the expected keys.

    Args:
        data: Mapping to validate.
        required: Keys that must be present.
        optional: Keys that may be present.
        context: Name of the configuration section, used in error messages.

    Raises:
        ConfigError: If keys are missing or unknown.
    """
    required_keys = set(required)
    allowed = required_keys | set(optional)
    present = set(data)
    missing = sorted(required_keys - present)
    unknown = sorted(present - allowed)
    if missing:
        msg = f"{context}: missing required key(s) {missing}"
        raise ConfigError(msg)
    if unknown:
        msg = f"{context}: unknown key(s) {unknown}; allowed keys are {sorted(allowed)}"
        raise ConfigError(msg)


@dataclass(frozen=True)
class SplitRatios:
    """Target proportions of the train/validation/test partition.

    These are the intended proportions. If the source dataset ships its own
    splits, the audit phase decides whether to adopt them or to re-split; either
    way the frozen result is recorded and must match this configuration.

    Attributes:
        train: Fraction assigned to training.
        val: Fraction assigned to validation.
        test: Fraction assigned to the locked holdout.
    """

    train: float
    val: float
    test: float

    def __post_init__(self) -> None:
        """Validate the ratios.

        Raises:
            ConfigError: If any ratio is outside the open interval (0, 1) or
                they do not sum to one.
        """
        for name, value in self.as_dict().items():
            if not 0.0 < value < 1.0:
                msg = f"split_ratios.{name} must be in the open interval (0, 1), got {value}"
                raise ConfigError(msg)
        total = self.train + self.val + self.test
        if abs(total - 1.0) > RATIO_TOLERANCE:
            msg = f"split_ratios must sum to 1.0, got {total}"
            raise ConfigError(msg)

    def as_dict(self) -> dict[str, float]:
        """Return the ratios keyed by split name.

        Returns:
            Mapping of split name to ratio.
        """
        return {"train": self.train, "val": self.val, "test": self.test}

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> SplitRatios:
        """Build split ratios from a raw mapping.

        Args:
            data: Mapping with ``train``, ``val`` and ``test`` keys.

        Returns:
            The parsed ratios.

        Raises:
            ConfigError: If keys are missing, unknown or not numeric.
        """
        _check_keys(data, required=("train", "val", "test"), context="split_ratios")
        try:
            train = float(data["train"])
            val = float(data["val"])
            test = float(data["test"])
        except (TypeError, ValueError) as exc:
            msg = f"split_ratios: values must be numeric ({exc})"
            raise ConfigError(msg) from exc
        return cls(train=train, val=val, test=test)


@dataclass(frozen=True)
class DatasetCandidate:
    """Declared source dataset and the claims that still need verification.

    Everything here is a planning hypothesis until the audit phase confirms it
    against the downloaded files. ``verified`` is flipped to ``True`` only by
    that audit, and reports must not present unverified fields as facts.

    Attributes:
        name: Human-readable dataset name.
        source: Hosting platform identifier.
        canonical_task: Annotation format treated as the single source of truth.
        expected_classes: Class names expected in the annotations.
        source_url: Canonical URL of the dataset version. Empty until acquired.
        verified: Whether the audit phase confirmed the declared fields.
        notes: Free-form provenance notes.
        workspace: Provider workspace slug used to address the dataset.
        project_slug: Provider project slug.
        version: Provider dataset version number. ``0`` means unselected.
        export_format: Provider export format slug used for acquisition.
    """

    name: str
    source: str
    canonical_task: str
    expected_classes: tuple[str, ...]
    source_url: str = ""
    verified: bool = False
    notes: str = ""
    workspace: str = ""
    project_slug: str = ""
    version: int = 0
    export_format: str = ""

    def __post_init__(self) -> None:
        """Validate the declared dataset fields.

        Raises:
            ConfigError: If the task is unknown, or the class list is empty or
                contains duplicates.
        """
        if self.canonical_task not in CANONICAL_TASKS:
            msg = (
                f"dataset.canonical_task must be one of {sorted(CANONICAL_TASKS)}, "
                f"got {self.canonical_task!r}"
            )
            raise ConfigError(msg)
        if not self.expected_classes:
            msg = "dataset.expected_classes must not be empty"
            raise ConfigError(msg)
        duplicates = sorted(
            {name for name in self.expected_classes if self.expected_classes.count(name) > 1}
        )
        if duplicates:
            msg = f"dataset.expected_classes contains duplicates: {duplicates}"
            raise ConfigError(msg)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> DatasetCandidate:
        """Build a dataset declaration from a raw mapping.

        Args:
            data: Mapping describing the dataset.

        Returns:
            The parsed declaration.

        Raises:
            ConfigError: If keys are missing, unknown or malformed.
        """
        _check_keys(
            data,
            required=("name", "source", "canonical_task", "expected_classes"),
            optional=(
                "source_url",
                "verified",
                "notes",
                "workspace",
                "project_slug",
                "version",
                "export_format",
            ),
            context="dataset",
        )
        classes = data["expected_classes"]
        if isinstance(classes, str) or not isinstance(classes, Iterable):
            msg = "dataset.expected_classes must be a list of class names"
            raise ConfigError(msg)
        try:
            version = int(data.get("version", 0))
        except (TypeError, ValueError) as exc:
            msg = f"dataset.version must be an integer ({exc})"
            raise ConfigError(msg) from exc
        return cls(
            name=str(data["name"]),
            source=str(data["source"]),
            canonical_task=str(data["canonical_task"]),
            expected_classes=tuple(str(name) for name in classes),
            source_url=str(data.get("source_url", "")),
            verified=bool(data.get("verified", False)),
            notes=str(data.get("notes", "")),
            workspace=str(data.get("workspace", "")),
            project_slug=str(data.get("project_slug", "")),
            version=version,
            export_format=str(data.get("export_format", "")),
        )


@dataclass(frozen=True)
class ExperimentConfig:
    """Top-level configuration shared by every phase of the project.

    Attributes:
        project_name: Identifier used in run names and provenance records.
        seed: Global random seed for every stochastic step.
        split_ratios: Target train/validation/test proportions.
        dataset: Declared source dataset.
        holdout_split: Name of the split protected from development use.
    """

    project_name: str
    seed: int
    split_ratios: SplitRatios
    dataset: DatasetCandidate
    holdout_split: str = "test"
    source_path: Path | None = field(default=None, compare=False, repr=False)

    @classmethod
    def from_mapping(
        cls,
        data: Mapping[str, Any],
        *,
        source_path: Path | None = None,
    ) -> ExperimentConfig:
        """Build the configuration from a raw mapping.

        Args:
            data: Parsed YAML mapping.
            source_path: Path the mapping was loaded from, kept for provenance.

        Returns:
            The parsed configuration.

        Raises:
            ConfigError: If the mapping is malformed.
        """
        _check_keys(
            data,
            required=("project_name", "seed", "split_ratios", "dataset"),
            optional=("holdout_split",),
            context="config",
        )
        for section in ("split_ratios", "dataset"):
            if not isinstance(data[section], Mapping):
                msg = f"config.{section} must be a mapping"
                raise ConfigError(msg)
        try:
            seed = int(data["seed"])
        except (TypeError, ValueError) as exc:
            msg = f"config.seed must be an integer ({exc})"
            raise ConfigError(msg) from exc
        return cls(
            project_name=str(data["project_name"]),
            seed=seed,
            split_ratios=SplitRatios.from_mapping(data["split_ratios"]),
            dataset=DatasetCandidate.from_mapping(data["dataset"]),
            holdout_split=str(data.get("holdout_split", "test")),
            source_path=source_path,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialise the configuration for provenance records.

        ``source_path`` is emitted relative to the repository root. Provenance
        records are committed, and an absolute path would leak the local user's
        directory layout while helping nobody reproduce the run.

        Returns:
            A JSON-serialisable snapshot of the configuration.
        """
        payload = asdict(self)
        payload["dataset"]["expected_classes"] = list(self.dataset.expected_classes)
        payload["source_path"] = self.relative_source_path()
        return payload

    def relative_source_path(self) -> str | None:
        """Return the configuration path relative to the repository root.

        Returns:
            A repository-relative POSIX path, the bare file name when the file
            lies outside the repository, or ``None`` for in-memory configurations.
        """
        if self.source_path is None:
            return None
        resolved = Path(self.source_path).expanduser().resolve()
        try:
            return resolved.relative_to(find_project_root()).as_posix()
        except (ValueError, ProjectRootNotFoundError):
            return resolved.name


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    """Load and validate an experiment configuration file.

    Args:
        path: Path to a YAML configuration file.

    Returns:
        The parsed configuration.

    Raises:
        ConfigError: If the file is missing, is not valid YAML, or does not
            contain a top-level mapping.
    """
    config_path = Path(path).expanduser()
    if not config_path.is_file():
        msg = f"Configuration file not found: {config_path}"
        raise ConfigError(msg)
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        msg = f"Configuration file is not valid YAML: {config_path} ({exc})"
        raise ConfigError(msg) from exc
    if not isinstance(raw, Mapping):
        msg = f"Configuration file must contain a top-level mapping: {config_path}"
        raise ConfigError(msg)
    return ExperimentConfig.from_mapping(raw, source_path=config_path)
