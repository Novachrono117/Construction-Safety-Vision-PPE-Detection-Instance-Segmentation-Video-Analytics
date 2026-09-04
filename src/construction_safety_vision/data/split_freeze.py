"""The frozen, authoritative train/validation/test membership.

Phase 5C.1 produced *provisional* candidates over the 422 indivisible groups and
selected none. Phase 5C.2 turns exactly one of them into the project's split.
From that point the assignment is no longer an optimisation output that a later
run may improve on: it is the protocol, and every result the project reports is
conditioned on it.

Three properties make that claim checkable rather than asserted.

**Identity is content, not order.** The manifest names each split by its groups
and their member images, sorted, so nothing depends on filesystem order, on the
order the optimiser emitted, or on a path that exists only on one machine.

**Fingerprints move when membership moves.** ``split_assignment_sha256`` covers
every ``(group, image, split)`` triple, so moving one image, reassigning one
group or changing one group's membership changes it. ``holdout_sha256`` covers
the holdout alone, so drift in the protected set is visible even when the rest of
the split is untouched. Neither digest sees a timestamp, a path, a label or a
metric.

**The holdout stays locked by default.** The loader routes ``test`` through
:func:`construction_safety_vision.splits.assert_split_allowed`, which needs both
an in-code opt-in and an environment opt-in. This module adds no way around it.

Nothing here reads an image, derives a label, materialises a dataset or consults
the provider's rejected split. It freezes membership and nothing else.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from construction_safety_vision.config import ConfigError, check_keys
from construction_safety_vision.splits import Split, assert_split_allowed

MANIFEST_SCHEMA_VERSION = 1
"""Schema version of ``reports/split_manifest.json``."""

FROZEN = "FROZEN"
"""The only manifest status under which the split may be used."""

TRAIN = "train"
VALIDATION = "validation"
TEST = "test"

MANIFEST_SPLITS: tuple[str, ...] = (TRAIN, VALIDATION, TEST)
"""Split vocabulary of the manifest, in a fixed order for deterministic output.

The manifest spells the middle split ``validation`` rather than the ``val`` short
form used in directory names, because the manifest is the document a person
reads. :meth:`construction_safety_vision.splits.Split.parse` accepts both.
"""

SPLIT_BY_ENUM: dict[Split, str] = {
    Split.TRAIN: TRAIN,
    Split.VAL: VALIDATION,
    Split.TEST: TEST,
}
"""Canonical manifest name of each split."""

REQUIRED_MANIFEST_FIELDS: tuple[str, ...] = (
    "schema_version",
    "status",
    "created_from_candidate",
    "selection_method",
    "seed",
    "target_ratios",
    "actual_image_counts",
    "actual_group_counts",
    "modeling_population_sha256",
    "groups_sha256",
    "candidate_assignment_sha256",
    "split_assignment_sha256",
    "holdout_sha256",
    TRAIN,
    VALIDATION,
    TEST,
)
"""Fields a manifest must carry before it may be loaded."""


class SplitManifestError(RuntimeError):
    """Raised when a split manifest is missing, malformed or inconsistent."""


class FingerprintMismatchError(SplitManifestError):
    """Raised when a recorded fingerprint does not match the content it covers."""


@dataclass(frozen=True)
class HoldoutGuardPolicy:
    """The declared access policy for the protected split.

    Recorded rather than inferred, so the manifest states the rule a reader can
    then check against :mod:`construction_safety_vision.splits`.

    Attributes:
        protected_split: The split the guard protects.
        env_var: Environment variable that supplies the second opt-in.
        requires_code_opt_in: Whether an in-code ``allow_test=True`` is required.
        requires_environment_opt_in: Whether the environment opt-in is required.
        default_state: The state with neither opt-in present.
    """

    protected_split: str
    env_var: str
    requires_code_opt_in: bool
    requires_environment_opt_in: bool
    default_state: str

    def as_dict(self) -> dict[str, Any]:
        """Serialise for the manifest.

        Returns:
            A JSON-serialisable mapping.
        """
        return {
            "protected_split": self.protected_split,
            "env_var": self.env_var,
            "requires_code_opt_in": self.requires_code_opt_in,
            "requires_environment_opt_in": self.requires_environment_opt_in,
            "default_state": self.default_state,
        }


@dataclass(frozen=True)
class SplitFreezeConfig:
    """Which candidate is frozen, and what it must look like when it is.

    Every expectation is declared here rather than computed at freeze time. That
    is the point: the freeze either reproduces the reviewed candidate exactly or
    it fails, and it cannot quietly freeze a different assignment that happens to
    be feasible.

    Attributes:
        schema_version: Version of this configuration schema.
        manifest_schema_version: Schema version to stamp on the manifest.
        selected_candidate: Candidate promoted to the authoritative split.
        selection_method: How the candidate was chosen.
        selection_decision_source: Who chose it.
        algorithmic_best_candidate: Best candidate under the phase 5C.1 objective.
        rare_class: The class whose allocation the freeze re-verifies.
        expected_candidate_assignment_sha256: Fingerprint the candidate must
            still reproduce.
        expected_modeling_population_sha256: Population fingerprint the split is
            defined over.
        expected_groups_sha256: Group fingerprint the split is defined over.
        expected_image_counts: Images per split.
        expected_group_counts: Groups per split.
        expected_negative_images: Zero-annotation images per split.
        expected_rare_class_images: Rare-class images per split.
        expected_rare_class_instances: Rare-class instances per split.
        holdout_guard: The declared holdout access policy.
    """

    schema_version: int
    manifest_schema_version: int
    selected_candidate: str
    selection_method: str
    selection_decision_source: str
    algorithmic_best_candidate: str
    rare_class: str
    expected_candidate_assignment_sha256: str
    expected_modeling_population_sha256: str
    expected_groups_sha256: str
    expected_image_counts: dict[str, int]
    expected_group_counts: dict[str, int]
    expected_negative_images: dict[str, int]
    expected_rare_class_images: dict[str, int]
    expected_rare_class_instances: dict[str, int]
    holdout_guard: HoldoutGuardPolicy


def _split_counts(data: Any, *, context: str) -> dict[str, int]:
    """Parse a per-split integer mapping, requiring every split exactly once.

    Args:
        data: Raw mapping from the configuration file.
        context: Section name, used in error messages.

    Returns:
        The parsed counts.

    Raises:
        ConfigError: If a split is missing, unknown, or the value is not an
            integer.
    """
    if not isinstance(data, Mapping):
        msg = f"{context}: must be a mapping of split name to integer"
        raise ConfigError(msg)
    check_keys(data, required=MANIFEST_SPLITS, context=context)
    counts: dict[str, int] = {}
    for split in MANIFEST_SPLITS:
        try:
            counts[split] = int(data[split])
        except (TypeError, ValueError) as exc:
            msg = f"{context}.{split}: must be an integer ({exc})"
            raise ConfigError(msg) from exc
    return counts


def load_split_freeze_config(path: str | Path) -> SplitFreezeConfig:
    """Load and validate the split-freeze protocol.

    Parsing is strict: an unknown key raises rather than being ignored, so a typo
    cannot silently drop an expectation the freeze is supposed to enforce.

    Args:
        path: Configuration file.

    Returns:
        The parsed protocol.

    Raises:
        ConfigError: If the file is missing, is not valid YAML, or is malformed.
    """
    config_path = Path(path).expanduser()
    if not config_path.is_file():
        msg = f"Split-freeze configuration not found: {config_path.name}"
        raise ConfigError(msg)
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        msg = f"Split-freeze configuration is not valid YAML: {config_path.name} ({exc})"
        raise ConfigError(msg) from exc
    if not isinstance(raw, Mapping):
        msg = f"Split-freeze configuration must contain a top-level mapping: {config_path.name}"
        raise ConfigError(msg)

    check_keys(
        raw,
        required=(
            "schema_version",
            "manifest_schema_version",
            "selection",
            "expected_fingerprints",
            "expected_counts",
            "holdout_guard",
        ),
        context="split_freeze",
    )
    selection = raw["selection"]
    if not isinstance(selection, Mapping):
        msg = "split_freeze.selection: must be a mapping"
        raise ConfigError(msg)
    check_keys(
        selection,
        required=(
            "selected_candidate",
            "selection_method",
            "decision_source",
            "algorithmic_best_candidate",
            "rare_class",
        ),
        context="split_freeze.selection",
    )
    fingerprints = raw["expected_fingerprints"]
    if not isinstance(fingerprints, Mapping):
        msg = "split_freeze.expected_fingerprints: must be a mapping"
        raise ConfigError(msg)
    check_keys(
        fingerprints,
        required=(
            "candidate_assignment_sha256",
            "modeling_population_sha256",
            "groups_sha256",
        ),
        context="split_freeze.expected_fingerprints",
    )
    counts = raw["expected_counts"]
    if not isinstance(counts, Mapping):
        msg = "split_freeze.expected_counts: must be a mapping"
        raise ConfigError(msg)
    check_keys(
        counts,
        required=(
            "images",
            "groups",
            "negative_images",
            "rare_class_images",
            "rare_class_instances",
        ),
        context="split_freeze.expected_counts",
    )
    guard = raw["holdout_guard"]
    if not isinstance(guard, Mapping):
        msg = "split_freeze.holdout_guard: must be a mapping"
        raise ConfigError(msg)
    check_keys(
        guard,
        required=(
            "protected_split",
            "env_var",
            "requires_code_opt_in",
            "requires_environment_opt_in",
            "default_state",
        ),
        context="split_freeze.holdout_guard",
    )

    return SplitFreezeConfig(
        schema_version=int(raw["schema_version"]),
        manifest_schema_version=int(raw["manifest_schema_version"]),
        selected_candidate=str(selection["selected_candidate"]),
        selection_method=str(selection["selection_method"]),
        selection_decision_source=str(selection["decision_source"]),
        algorithmic_best_candidate=str(selection["algorithmic_best_candidate"]),
        rare_class=str(selection["rare_class"]),
        expected_candidate_assignment_sha256=str(fingerprints["candidate_assignment_sha256"]),
        expected_modeling_population_sha256=str(fingerprints["modeling_population_sha256"]),
        expected_groups_sha256=str(fingerprints["groups_sha256"]),
        expected_image_counts=_split_counts(counts["images"], context="expected_counts.images"),
        expected_group_counts=_split_counts(counts["groups"], context="expected_counts.groups"),
        expected_negative_images=_split_counts(
            counts["negative_images"], context="expected_counts.negative_images"
        ),
        expected_rare_class_images=_split_counts(
            counts["rare_class_images"], context="expected_counts.rare_class_images"
        ),
        expected_rare_class_instances=_split_counts(
            counts["rare_class_instances"], context="expected_counts.rare_class_instances"
        ),
        holdout_guard=HoldoutGuardPolicy(
            protected_split=str(guard["protected_split"]),
            env_var=str(guard["env_var"]),
            requires_code_opt_in=bool(guard["requires_code_opt_in"]),
            requires_environment_opt_in=bool(guard["requires_environment_opt_in"]),
            default_state=str(guard["default_state"]),
        ),
    )


def _digest(payload: Any) -> str:
    """Hash a JSON-serialisable payload deterministically.

    Args:
        payload: Content to hash.

    Returns:
        Its SHA-256 hex digest.
    """
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True, order=True)
class SplitAssignment:
    """One modelling image, the group it belongs to and the split it landed in.

    Attributes:
        group_id: The indivisible unit the split assigned.
        source_image_id: Provider source-image identifier. The stable key.
        split: One of :data:`MANIFEST_SPLITS`.
    """

    group_id: str
    source_image_id: str
    split: str


def normalise_assignments(assignments: Iterable[SplitAssignment]) -> tuple[SplitAssignment, ...]:
    """Order assignments deterministically and reject unknown split names.

    Args:
        assignments: Assignments in any order.

    Returns:
        The assignments sorted by group id then image id.

    Raises:
        SplitManifestError: If any row names a split outside the vocabulary.
    """
    ordered = tuple(sorted(assignments))
    unknown = sorted({row.split for row in ordered} - set(MANIFEST_SPLITS))
    if unknown:
        msg = f"unknown split name(s) {unknown}; expected {list(MANIFEST_SPLITS)}"
        raise SplitManifestError(msg)
    return ordered


def fingerprint_split_assignment(assignments: Iterable[SplitAssignment]) -> str:
    """Fingerprint the frozen membership by its content alone.

    The digest covers every ``(group, image, split)`` triple, so it moves when an
    image changes split, when a group changes split, and when a group's
    membership changes. No timestamp, machine path, label, metric or provider
    split enters it.

    Args:
        assignments: The full assignment.

    Returns:
        A SHA-256 hex digest.
    """
    ordered = normalise_assignments(assignments)
    return _digest([[row.group_id, row.source_image_id, row.split] for row in ordered])


def fingerprint_holdout(
    assignments: Iterable[SplitAssignment], *, modeling_population_sha256: str
) -> str:
    """Fingerprint the holdout's identity, so membership drift is detectable.

    Only what defines membership enters the digest: the holdout's group ids, its
    image ids, and the fingerprint of the population they were drawn from. No
    label and no metric, because the purpose is to notice that the protected set
    changed, not to record what a model did on it.

    Args:
        assignments: The full assignment.
        modeling_population_sha256: Fingerprint of the population the split was
            drawn from, tying the holdout to the data it was defined over.

    Returns:
        A SHA-256 hex digest.
    """
    ordered = normalise_assignments(assignments)
    holdout = [row for row in ordered if row.split == TEST]
    return _digest(
        {
            "modeling_population_sha256": modeling_population_sha256,
            "split": TEST,
            "group_ids": sorted({row.group_id for row in holdout}),
            "source_image_ids": sorted({row.source_image_id for row in holdout}),
        }
    )


def split_sections(assignments: Iterable[SplitAssignment]) -> dict[str, dict[str, Any]]:
    """Build the per-split manifest sections.

    Each split is described by its groups and their members rather than by a flat
    image list, so group integrity is provable from the manifest alone.

    Args:
        assignments: The full assignment.

    Returns:
        One section per split, keyed by manifest split name.
    """
    ordered = normalise_assignments(assignments)
    sections: dict[str, dict[str, Any]] = {}
    for split in MANIFEST_SPLITS:
        members: dict[str, list[str]] = {}
        for row in ordered:
            if row.split == split:
                members.setdefault(row.group_id, []).append(row.source_image_id)
        groups = [
            {"group_id": group_id, "source_image_ids": sorted(members[group_id])}
            for group_id in sorted(members)
        ]
        sections[split] = {
            "group_count": len(groups),
            "image_count": sum(len(group["source_image_ids"]) for group in groups),
            "non_singleton_group_count": sum(
                1 for group in groups if len(group["source_image_ids"]) > 1
            ),
            "groups": groups,
        }
    return sections


def assignments_from_sections(manifest: Mapping[str, Any]) -> tuple[SplitAssignment, ...]:
    """Read the assignment back out of a manifest.

    Args:
        manifest: The parsed manifest.

    Returns:
        The assignment, deterministically ordered.

    Raises:
        SplitManifestError: If a section is malformed.
    """
    rows: list[SplitAssignment] = []
    for split in MANIFEST_SPLITS:
        section = manifest.get(split)
        if not isinstance(section, Mapping):
            msg = f"manifest section {split!r} is missing or is not a mapping"
            raise SplitManifestError(msg)
        groups = section.get("groups")
        if not isinstance(groups, list):
            msg = f"manifest section {split!r} has no 'groups' list"
            raise SplitManifestError(msg)
        for group in groups:
            if not isinstance(group, Mapping):
                msg = f"manifest section {split!r} holds a non-mapping group entry"
                raise SplitManifestError(msg)
            group_id = group.get("group_id")
            members = group.get("source_image_ids")
            if not isinstance(group_id, str) or not group_id:
                msg = f"manifest section {split!r} holds a group without a group_id"
                raise SplitManifestError(msg)
            if not isinstance(members, list) or not members:
                msg = f"group {group_id!r} in section {split!r} has no source_image_ids"
                raise SplitManifestError(msg)
            rows.extend(
                SplitAssignment(group_id=group_id, source_image_id=str(member), split=split)
                for member in members
            )
    return normalise_assignments(rows)


def validate_manifest(manifest: Mapping[str, Any]) -> list[str]:
    """Check a split manifest for structural and arithmetic defects.

    Args:
        manifest: The parsed manifest.

    Returns:
        One description per problem found, empty when the manifest is sound.
    """
    problems: list[str] = []
    missing = [name for name in REQUIRED_MANIFEST_FIELDS if name not in manifest]
    if missing:
        problems.append(f"missing required field(s): {', '.join(missing)}")
        return problems

    if manifest["schema_version"] != MANIFEST_SCHEMA_VERSION:
        problems.append(
            f"schema_version {manifest['schema_version']!r} is not the supported "
            f"{MANIFEST_SCHEMA_VERSION}"
        )
    if manifest["status"] != FROZEN:
        problems.append(f"status {manifest['status']!r} is not {FROZEN!r}")

    try:
        assignments = assignments_from_sections(manifest)
    except SplitManifestError as exc:
        problems.append(str(exc))
        return problems

    seen: dict[str, str] = {}
    for row in assignments:
        if row.source_image_id in seen:
            problems.append(
                f"image {row.source_image_id!r} appears in "
                f"{seen[row.source_image_id]!r} and {row.split!r}"
            )
        seen[row.source_image_id] = row.split

    group_splits: dict[str, set[str]] = {}
    for row in assignments:
        group_splits.setdefault(row.group_id, set()).add(row.split)
    for group_id, splits in sorted(group_splits.items()):
        if len(splits) > 1:
            problems.append(f"group {group_id!r} is split across {sorted(splits)}")

    problems.extend(_count_problems(manifest, assignments))
    problems.extend(_fingerprint_problems(manifest, assignments))
    return problems


def _count_problems(
    manifest: Mapping[str, Any], assignments: Sequence[SplitAssignment]
) -> list[str]:
    """Check the declared counts against the membership they describe.

    Args:
        manifest: The parsed manifest.
        assignments: The assignment read back from it.

    Returns:
        One description per disagreement.
    """
    problems: list[str] = []
    sections = split_sections(assignments)
    for split in MANIFEST_SPLITS:
        declared_images = manifest["actual_image_counts"].get(split)
        declared_groups = manifest["actual_group_counts"].get(split)
        if declared_images != sections[split]["image_count"]:
            problems.append(
                f"actual_image_counts[{split!r}] is {declared_images!r} but the section "
                f"holds {sections[split]['image_count']}"
            )
        if declared_groups != sections[split]["group_count"]:
            problems.append(
                f"actual_group_counts[{split!r}] is {declared_groups!r} but the section "
                f"holds {sections[split]['group_count']}"
            )
    return problems


def _fingerprint_problems(
    manifest: Mapping[str, Any], assignments: Sequence[SplitAssignment]
) -> list[str]:
    """Recompute the manifest's own fingerprints from its membership.

    Args:
        manifest: The parsed manifest.
        assignments: The assignment read back from it.

    Returns:
        One description per fingerprint that does not match.
    """
    problems: list[str] = []
    recomputed = fingerprint_split_assignment(assignments)
    if recomputed != manifest["split_assignment_sha256"]:
        problems.append(
            "split_assignment_sha256 does not match the membership it covers "
            f"(recomputed {recomputed})"
        )
    recomputed_holdout = fingerprint_holdout(
        assignments, modeling_population_sha256=str(manifest["modeling_population_sha256"])
    )
    if recomputed_holdout != manifest["holdout_sha256"]:
        problems.append(
            f"holdout_sha256 does not match the holdout it covers (recomputed {recomputed_holdout})"
        )
    return problems


@dataclass(frozen=True)
class FrozenSplits:
    """Read access to the frozen membership, with the holdout still guarded.

    Attributes:
        manifest: The parsed manifest.
        assignments: The full assignment, deterministically ordered.
    """

    manifest: Mapping[str, Any]
    assignments: tuple[SplitAssignment, ...]

    @property
    def split_assignment_sha256(self) -> str:
        """The verified fingerprint of the whole assignment.

        Returns:
            The digest recorded in the manifest.
        """
        return str(self.manifest["split_assignment_sha256"])

    @property
    def holdout_sha256(self) -> str:
        """The verified fingerprint of the holdout's membership.

        Returns:
            The digest recorded in the manifest.
        """
        return str(self.manifest["holdout_sha256"])

    def _rows(
        self,
        split: str | Split,
        *,
        purpose: str,
        allow_test: bool,
        env: dict[str, str] | None,
    ) -> tuple[SplitAssignment, ...]:
        """Authorise a split and return its assignment rows.

        Args:
            split: Split being requested, in any accepted spelling.
            purpose: Short reason, recorded in the guard's error message.
            allow_test: Explicit in-code opt-in for the holdout.
            env: Environment mapping to read. Defaults to ``os.environ``.

        Returns:
            The rows of that split.
        """
        resolved = assert_split_allowed(split, purpose=purpose, allow_test=allow_test, env=env)
        name = SPLIT_BY_ENUM[resolved]
        return tuple(row for row in self.assignments if row.split == name)

    def image_ids(
        self,
        split: str | Split,
        *,
        purpose: str,
        allow_test: bool = False,
        env: dict[str, str] | None = None,
    ) -> tuple[str, ...]:
        """Return the source image ids of one split.

        Args:
            split: Split being requested.
            purpose: Short reason, recorded in the guard's error message.
            allow_test: Explicit in-code opt-in for the holdout.
            env: Environment mapping to read. Defaults to ``os.environ``.

        Returns:
            The image ids, sorted.
        """
        rows = self._rows(split, purpose=purpose, allow_test=allow_test, env=env)
        return tuple(sorted(row.source_image_id for row in rows))

    def group_ids(
        self,
        split: str | Split,
        *,
        purpose: str,
        allow_test: bool = False,
        env: dict[str, str] | None = None,
    ) -> tuple[str, ...]:
        """Return the group ids of one split.

        Args:
            split: Split being requested.
            purpose: Short reason, recorded in the guard's error message.
            allow_test: Explicit in-code opt-in for the holdout.
            env: Environment mapping to read. Defaults to ``os.environ``.

        Returns:
            The group ids, sorted and deduplicated.
        """
        rows = self._rows(split, purpose=purpose, allow_test=allow_test, env=env)
        return tuple(sorted({row.group_id for row in rows}))

    def groups(
        self,
        split: str | Split,
        *,
        purpose: str,
        allow_test: bool = False,
        env: dict[str, str] | None = None,
    ) -> dict[str, tuple[str, ...]]:
        """Return one split's groups with their member images.

        Args:
            split: Split being requested.
            purpose: Short reason, recorded in the guard's error message.
            allow_test: Explicit in-code opt-in for the holdout.
            env: Environment mapping to read. Defaults to ``os.environ``.

        Returns:
            Member image ids keyed by group id.
        """
        rows = self._rows(split, purpose=purpose, allow_test=allow_test, env=env)
        members: dict[str, list[str]] = {}
        for row in rows:
            members.setdefault(row.group_id, []).append(row.source_image_id)
        return {group_id: tuple(sorted(members[group_id])) for group_id in sorted(members)}


def load_frozen_splits(
    manifest_path: str | Path,
    *,
    population_manifest_path: str | Path | None = None,
    verify_population: bool = True,
) -> FrozenSplits:
    """Load and verify the frozen split manifest.

    Verification is not optional decoration. A manifest whose fingerprints do not
    cover its own content, or that was frozen over a different modelling
    population than the one on disk, would silently invalidate every downstream
    result, so both are checked before any caller sees an image id.

    Args:
        manifest_path: ``reports/split_manifest.json``.
        population_manifest_path: ``reports/canonical_modeling_manifest.json``.
            Defaults to the sibling file of ``manifest_path``.
        verify_population: Whether to cross-check the modelling-population and
            group fingerprints against that manifest. Turn it off only in tests
            that build a manifest with no population beside it.

    Returns:
        The verified accessor.

    Raises:
        SplitManifestError: If the file is missing, unparsable or inconsistent.
        FingerprintMismatchError: If a recorded fingerprint does not match.
    """
    path = Path(manifest_path)
    if not path.is_file():
        msg = f"{path.name} not found; run scripts/freeze_split.py first"
        raise SplitManifestError(msg)
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        msg = f"{path.name} is not valid JSON ({exc})"
        raise SplitManifestError(msg) from exc
    if not isinstance(manifest, Mapping):
        msg = f"{path.name} must contain a top-level mapping"
        raise SplitManifestError(msg)

    problems = validate_manifest(manifest)
    if problems:
        msg = f"{path.name} is not a usable frozen split: {'; '.join(problems)}"
        if any("sha256" in problem for problem in problems):
            raise FingerprintMismatchError(msg)
        raise SplitManifestError(msg)

    if verify_population:
        population_path = (
            Path(population_manifest_path)
            if population_manifest_path is not None
            else path.parent / "canonical_modeling_manifest.json"
        )
        _verify_population_fingerprints(manifest, population_path)

    return FrozenSplits(manifest=manifest, assignments=assignments_from_sections(manifest))


def _verify_population_fingerprints(
    manifest: Mapping[str, Any], population_manifest_path: Path
) -> None:
    """Check the split against the population manifest it claims to be built on.

    Args:
        manifest: The parsed split manifest.
        population_manifest_path: ``reports/canonical_modeling_manifest.json``.

    Raises:
        SplitManifestError: If the population manifest is missing or unparsable.
        FingerprintMismatchError: If either fingerprint disagrees.
    """
    if not population_manifest_path.is_file():
        msg = (
            f"{population_manifest_path.name} not found; the frozen split cannot be "
            "verified against the population it was drawn from"
        )
        raise SplitManifestError(msg)
    try:
        population = json.loads(population_manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        msg = f"{population_manifest_path.name} is not valid JSON ({exc})"
        raise SplitManifestError(msg) from exc
    recorded = population.get("fingerprints", {})
    for name in ("modeling_population_sha256", "groups_sha256"):
        expected = recorded.get(name)
        found = manifest.get(name)
        if expected != found:
            msg = (
                f"{name} disagrees: the split manifest records {found!r} but "
                f"{population_manifest_path.name} records {expected!r}. The split was frozen "
                "over a different modelling population; recompute it or withdraw the results "
                "produced under it."
            )
            raise FingerprintMismatchError(msg)


NON_SELECTED = "NON_SELECTED_PROVISIONAL_CANDIDATE"
"""Status of every candidate the human review did not select."""

SELECTED = "FINAL_SELECTED"
"""Status of the one candidate that became the authoritative split."""

_PROVISIONAL_BANNER = (
    "**No split is frozen and no candidate is selected.** Every assignment below is "
    "provisional. `final_selected_candidate` is `UNSELECTED_PENDING_REVIEW`; the holdout "
    "remains locked and no candidate test set has been evaluated."
)

_PROVISIONAL_LIMITATION = (
    "* **No candidate is selected.** `final_selected_candidate` is "
    "`UNSELECTED_PENDING_REVIEW`. `algorithmic_best_candidate` names only the lowest scorer "
    "under the predeclared objective, which is `candidate_001`."
)


def stamp_selection_status(text: str, *, selected: str, selected_in: str) -> str:
    """Record the phase 5C.2 selection outcome in the phase 5C.1 report.

    The candidate report is a phase 5C.1 artifact and its scores, rankings and
    reasoning are left exactly as they were measured. Only the two sentences that
    assert *no candidate is selected* are rewritten, because after the freeze
    they are false, and a report that states a falsehood about its own status is
    worse than one that is out of date.

    The rewrite is idempotent: applying it to an already-stamped report returns
    the text unchanged, so re-running the freeze is safe.

    Args:
        text: Current contents of ``reports/split_candidate_report.md``.
        selected: The selected candidate id.
        selected_in: The phase that selected it.

    Returns:
        The stamped text.

    Raises:
        SplitManifestError: If the report carries neither the provisional
            sentences nor an existing stamp, meaning its shape changed and the
            substitution can no longer be verified.
    """
    banner = (
        f"**The split is frozen.** `final_selected_candidate` is `{selected}`, selected in "
        f"{selected_in} by human review of these predeclared candidates; the other candidates "
        f"remain as `{NON_SELECTED}` and are kept for comparison. The scores, rankings and "
        "reasoning below are the phase 5C.1 measurements and are unchanged. The authoritative "
        "assignment is `reports/split_manifest.json`; the holdout is locked."
    )
    limitation = (
        f"* **The selected candidate is `{selected}`**, chosen in {selected_in}. "
        "`algorithmic_best_candidate` named only the lowest scorer under the predeclared "
        "objective; the selection among the predeclared candidates was a human decision, "
        "recorded in `reports/split_freeze_report.md`."
    )
    stamped = text
    for old, new in ((_PROVISIONAL_BANNER, banner), (_PROVISIONAL_LIMITATION, limitation)):
        if old in stamped:
            stamped = stamped.replace(old, new, 1)
        elif new not in stamped:
            msg = (
                "reports/split_candidate_report.md carries neither the provisional status "
                "sentence nor its phase 5C.2 replacement; refusing to guess where the "
                "selection outcome belongs"
            )
            raise SplitManifestError(msg)
    return stamped


def assignment_rows(assignments: Sequence[SplitAssignment]) -> list[dict[str, str]]:
    """Serialise the assignment for ``reports/final_split_assignments.csv``.

    Args:
        assignments: The full assignment.

    Returns:
        One row per image, deterministically ordered.
    """
    return [
        {"group_id": row.group_id, "source_image_id": row.source_image_id, "split": row.split}
        for row in normalise_assignments(assignments)
    ]
