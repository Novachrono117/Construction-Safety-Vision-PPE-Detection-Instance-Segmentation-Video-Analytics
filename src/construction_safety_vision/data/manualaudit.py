"""Recording and validation of the phase 4B human visual audit.

Phase 4A measured the dataset; it deliberately answered no semantic question.
Phase 4B carries the answers a person gave after looking at the committed
contact sheets. Those answers are judgements, not measurements, so they are
stored apart from the computed artifacts and are never presented as numbers.

This module holds the machinery that keeps such a record trustworthy:

* a controlled vocabulary, so a decision cannot be spelled two ways;
* deterministic identifiers for semantic-duplicate groups, so re-running the
  recorder cannot renumber them;
* tolerant resolution of hand-transcribed short identifiers, which refuses
  anything ambiguous rather than guessing;
* validation of the written record against the phase 4A manifests.

No dataset file, annotation or figure is read or modified here.
"""

from __future__ import annotations

import csv
import json
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

FIELDNAMES: tuple[str, ...] = (
    "decision_id",
    "review_type",
    "subject_scope",
    "group_id",
    "subject_id_a",
    "subject_id_b",
    "provider_split_a",
    "provider_split_b",
    "decision",
    "confidence",
    "rationale",
    "evidence_figure",
    "phase5_action",
    "notes",
)
"""Column order of ``reports/manual_audit_decisions.csv``."""

REVIEW_TYPES: frozenset[str] = frozenset(
    {
        "cross_split_near_duplicate",
        "zero_instance_image",
        "bbox_vs_segmentation",
        "class_semantics",
        "class_data_diversity",
        "dataset_domain",
        "provider_split",
        "annotation_snapshot_drift",
        "source_annotation_geometry",
    }
)
"""Kinds of review a row may describe."""

SUBJECT_SCOPES: frozenset[str] = frozenset({"pair", "image", "annotation", "set"})
"""What a row is about. ``set`` rows carry no subject identifier."""

DECISIONS: frozenset[str] = frozenset(
    {
        "EXACT_SEMANTIC_DUPLICATE",
        "OUT_OF_DOMAIN",
        "NO_OBVIOUS_MISSING_TARGET_LABEL",
        "SEGMENTATION_GEOMETRY_PREFERRED",
        "BOTH_ACCEPTABLE",
        "UNCERTAIN",
        "VALID_OVERALL",
        "VERY_LOW",
        "ACCEPTABLE_WITH_DOMAIN_HETEROGENEITY",
        "UNSUITABLE_FOR_FINAL_PROTOCOL",
        "UNRESOLVED_CANONICALIZATION",
        "MANUAL_REVIEW_STILL_REQUIRED",
    }
)
"""Every verdict a human review may record."""

CONFIDENCE_LEVELS: frozenset[str] = frozenset({"HIGH", "MEDIUM", "LOW", "NOT_STATED"})
"""Confidence as stated by the reviewers.

``NOT_STATED`` is not ``LOW``: it means the reviewers gave a verdict without
qualifying it, and inventing a level would be fabricating evidence.
"""

PHASE5_ACTIONS: frozenset[str] = frozenset(
    {
        "GROUP_TOGETHER",
        "EXCLUDE_CANDIDATE",
        "RETAIN_CANDIDATE",
        "PREFER_SEGMENTATION_DERIVED_BBOX",
        "ENFORCE_RARE_CLASS_COVERAGE",
        "REDESIGN_SPLIT",
        "RESOLVE_BEFORE_FREEZE",
        "PHASE5_CANONICALIZATION_ITEM",
        "CONSTRAIN_EXTERNAL_VALIDITY_CLAIMS",
        "NONE",
    }
)
"""Consequences a row hands to phase 5.

Recording an action is not performing it: nothing in this module splits,
excludes or converts anything.
"""

PROVIDER_SPLITS: frozenset[str] = frozenset({"train", "valid", "test"})
"""Split names used by the provider export."""

DUPLICATE_GROUP_PREFIX = "manual_dup_"
"""Prefix of a visually confirmed semantic-duplicate group identifier."""

MAX_SHORT_ID_DISTANCE = 2
"""Largest edit distance tolerated when resolving a transcribed short id."""

MIN_SHORT_ID_MARGIN = 2
"""Required gap between the best and the second-best short-id candidate.

A transcription that fits two identifiers almost equally well is ambiguous and
is rejected rather than resolved.
"""

_CONFUSABLES = str.maketrans({"0": "o", "1": "l", "5": "s", "8": "b", "i": "l"})
"""Characters that a human reading a small caption confuses with one another."""

_SECRET_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"ROBOFLOW_API_KEY\s*[=:]", re.IGNORECASE),
    re.compile(r"\bapi[_-]?key\b\s*[=:]", re.IGNORECASE),
    re.compile(r"X-Amz-(Signature|Credential|Security-Token)", re.IGNORECASE),
    re.compile(r"[?&](token|signature|sig|key|access_key)=", re.IGNORECASE),
    re.compile(r"\bBearer\s+[A-Za-z0-9._-]{8,}"),
)
"""Shapes of credential material and signed provider URLs."""

_ABSOLUTE_PATH_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^[A-Za-z]:[\\/]"),
    re.compile(r"^\\\\"),
    re.compile(r"^/[A-Za-z]"),
    re.compile(r"^file://", re.IGNORECASE),
)
"""Shapes of a machine-local path, which must never enter a committed artifact."""


class ManualAuditError(RuntimeError):
    """Raised when a manual-audit record cannot be built or trusted."""


@dataclass(frozen=True)
class ManualDecision:
    """One recorded human judgement.

    Attributes:
        decision_id: Stable identifier of this row.
        review_type: Which review the judgement belongs to.
        subject_scope: Whether the row concerns a pair, an image, an annotation
            or a whole review set.
        decision: The verdict, from :data:`DECISIONS`.
        confidence: Confidence as stated by the reviewers.
        rationale: Why the reviewers decided this.
        phase5_action: The consequence handed to phase 5.
        group_id: Semantic-duplicate group, when the row belongs to one.
        subject_id_a: First subject identifier, empty for ``set`` rows.
        subject_id_b: Second subject identifier, used by ``pair`` rows.
        provider_split_a: Provider split of ``subject_id_a``.
        provider_split_b: Provider split of ``subject_id_b``.
        evidence_figure: Repository-relative figure path(s), ``|``-separated.
        notes: Traceability detail; never a substitute for the rationale.
    """

    decision_id: str
    review_type: str
    subject_scope: str
    decision: str
    confidence: str
    rationale: str
    phase5_action: str
    group_id: str = ""
    subject_id_a: str = ""
    subject_id_b: str = ""
    provider_split_a: str = ""
    provider_split_b: str = ""
    evidence_figure: str = ""
    notes: str = ""

    def as_row(self) -> dict[str, str]:
        """Render the decision as a CSV row.

        Returns:
            A mapping keyed by :data:`FIELDNAMES`.
        """
        return {name: getattr(self, name) for name in FIELDNAMES}


def _levenshtein(left: str, right: str) -> int:
    """Compute the edit distance between two short strings.

    Args:
        left: First string.
        right: Second string.

    Returns:
        The number of single-character insertions, deletions or substitutions
        needed to turn ``left`` into ``right``.
    """
    if left == right:
        return 0
    previous = list(range(len(right) + 1))
    for i, a in enumerate(left, start=1):
        current = [i]
        for j, b in enumerate(right, start=1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (a != b)))
        previous = current
    return previous[-1]


def _fold(value: str) -> str:
    """Normalise an identifier for comparison against a human transcription.

    Case and a handful of visually confusable characters carry no information
    when someone reads an eight-character caption off a contact sheet, so both
    sides are folded before they are compared.

    Args:
        value: Identifier or transcription.

    Returns:
        The folded form.
    """
    return value.strip().lower().translate(_CONFUSABLES)


@dataclass(frozen=True)
class ShortIdMatch:
    """Resolution of a transcribed short identifier.

    Attributes:
        transcription: What the reviewers wrote down.
        image_id: The full source image identifier it resolves to.
        distance: Edit distance between the folded forms.
        runner_up_distance: Distance of the next-best candidate, or ``-1`` when
            the pool held a single candidate.
    """

    transcription: str
    image_id: str
    distance: int
    runner_up_distance: int


def resolve_short_id(transcription: str, pool: dict[str, str]) -> ShortIdMatch:
    """Resolve a hand-transcribed short id against a restricted candidate pool.

    The reviewers copied short ids off image captions, so a transcription may
    differ from the real identifier in case or in a confusable character. This
    resolves such a transcription only when one candidate fits clearly better
    than every other; otherwise it raises, because recording the wrong image
    would attach a human judgement to an image nobody judged.

    Args:
        transcription: The short id as written by the reviewers.
        pool: Mapping of short id to full image id, restricted to the images
            that were actually part of the relevant review set.

    Returns:
        The resolved match, carrying the distance that justified it.

    Raises:
        ManualAuditError: If the pool is empty, if no candidate is close
            enough, or if two candidates fit comparably well.
    """
    if not pool:
        msg = f"cannot resolve {transcription!r}: the candidate pool is empty"
        raise ManualAuditError(msg)
    target = _fold(transcription)
    scored = sorted(
        ((_levenshtein(target, _fold(short)), short) for short in pool),
        key=lambda item: (item[0], item[1]),
    )
    best_distance, best_short = scored[0]
    runner_up = scored[1][0] if len(scored) > 1 else -1
    if best_distance > MAX_SHORT_ID_DISTANCE:
        msg = (
            f"cannot resolve {transcription!r}: closest candidate {best_short!r} "
            f"is {best_distance} edits away (limit {MAX_SHORT_ID_DISTANCE})"
        )
        raise ManualAuditError(msg)
    if runner_up >= 0 and runner_up - best_distance < MIN_SHORT_ID_MARGIN:
        msg = (
            f"ambiguous transcription {transcription!r}: {best_short!r} at "
            f"distance {best_distance} and {scored[1][1]!r} at distance {runner_up}"
        )
        raise ManualAuditError(msg)
    return ShortIdMatch(
        transcription=transcription,
        image_id=pool[best_short],
        distance=best_distance,
        runner_up_distance=runner_up,
    )


def assign_duplicate_group_ids(groups: Iterable[Sequence[str]]) -> dict[tuple[str, ...], str]:
    """Assign deterministic identifiers to semantic-duplicate groups.

    The identifier is a function of the sorted member ids alone, so the same
    set of confirmed duplicates always produces the same numbering, whatever
    order the reviewers listed them in.

    Args:
        groups: Iterable of member-id sequences.

    Returns:
        Mapping from the sorted member tuple to its group id.

    Raises:
        ManualAuditError: If a group repeats a member, holds fewer than two
            members, or is declared twice.
    """
    normalised: list[tuple[str, ...]] = []
    for members in groups:
        listed = [member.strip() for member in members]
        if any(not member for member in listed):
            msg = f"duplicate group {listed!r} contains an empty member id"
            raise ManualAuditError(msg)
        if len(set(listed)) != len(listed):
            msg = f"duplicate group {listed!r} repeats a member id"
            raise ManualAuditError(msg)
        if len(listed) < 2:
            msg = f"duplicate group {listed!r} needs at least two members"
            raise ManualAuditError(msg)
        normalised.append(tuple(sorted(listed)))
    if len(set(normalised)) != len(normalised):
        msg = "the same duplicate group was declared more than once"
        raise ManualAuditError(msg)
    return {
        members: f"{DUPLICATE_GROUP_PREFIX}{index:03d}"
        for index, members in enumerate(sorted(normalised), start=1)
    }


def load_source_image_index(path: Path) -> dict[str, dict[str, Any]]:
    """Load the phase 4A source-image manifest.

    Args:
        path: ``reports/source_image_manifest.jsonl``.

    Returns:
        Mapping of image id to its manifest record.

    Raises:
        ManualAuditError: If the manifest is missing or repeats an image id.
    """
    if not path.is_file():
        msg = f"source image manifest not found: {path}"
        raise ManualAuditError(msg)
    index: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            image_id = record["image_id"]
            if image_id in index:
                msg = f"source image manifest repeats image id {image_id}"
                raise ManualAuditError(msg)
            index[image_id] = record
    return index


def load_review_index(path: Path) -> dict[str, list[dict[str, str]]]:
    """Load the phase 4A visual-review manifest, keyed by full image id.

    Args:
        path: ``reports/manual_review_manifest.csv``.

    Returns:
        Mapping of image id to the review rows that reference it. An image may
        appear on more than one contact sheet, hence the list.

    Raises:
        ManualAuditError: If the manifest is missing.
    """
    if not path.is_file():
        msg = f"manual review manifest not found: {path}"
        raise ManualAuditError(msg)
    index: dict[str, list[dict[str, str]]] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            index.setdefault(row["image_id"], []).append(row)
    return index


def review_pool(review_index: dict[str, list[dict[str, str]]], reason: str) -> dict[str, str]:
    """Build a short-id lookup restricted to one review reason.

    Args:
        review_index: Output of :func:`load_review_index`.
        reason: The ``reason`` value that defines the review set, e.g.
            ``zero_instance``.

    Returns:
        Mapping of short id to full image id for that review set.
    """
    pool: dict[str, str] = {}
    for image_id, rows in review_index.items():
        for row in rows:
            if row.get("reason") == reason:
                pool[row["short_id"]] = image_id
    return pool


def read_decisions(path: Path) -> list[dict[str, str]]:
    """Read a recorded decision file.

    Args:
        path: ``reports/manual_audit_decisions.csv``.

    Returns:
        The rows, in file order.

    Raises:
        ManualAuditError: If the file is missing or its header is wrong.
    """
    if not path.is_file():
        msg = f"manual audit decisions not found: {path}"
        raise ManualAuditError(msg)
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        header = tuple(reader.fieldnames or ())
        if header != FIELDNAMES:
            msg = f"unexpected header in {path.name}: {header}"
            raise ManualAuditError(msg)
        return list(reader)


def write_decisions(path: Path, decisions: Sequence[ManualDecision]) -> None:
    """Write the decision file.

    Args:
        path: Destination CSV.
        decisions: Decisions to record, in the order they should appear.
    """
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        for decision in decisions:
            writer.writerow(decision.as_row())


def _check_vocabulary(row: dict[str, str], where: str, problems: list[str]) -> None:
    """Check one row against the controlled vocabulary.

    Args:
        row: Decision row.
        where: Human-readable row label used in messages.
        problems: List that failures are appended to.
    """
    checks = (
        ("review_type", REVIEW_TYPES),
        ("subject_scope", SUBJECT_SCOPES),
        ("decision", DECISIONS),
        ("confidence", CONFIDENCE_LEVELS),
        ("phase5_action", PHASE5_ACTIONS),
    )
    for field_name, allowed in checks:
        value = row.get(field_name, "")
        if value not in allowed:
            problems.append(f"{where}: {field_name}={value!r} is outside the controlled vocabulary")
    for field_name in ("provider_split_a", "provider_split_b"):
        value = row.get(field_name, "")
        if value and value not in PROVIDER_SPLITS:
            problems.append(f"{where}: {field_name}={value!r} is not a provider split")


def _check_subjects(
    row: dict[str, str],
    where: str,
    source_index: dict[str, dict[str, Any]],
    review_index: dict[str, list[dict[str, str]]],
    problems: list[str],
) -> None:
    """Check that a row's subjects exist and carry the recorded splits.

    A row that cites a contact sheet must name a subject that actually appears
    on it: the judgement was made by looking at that sheet, so a subject the
    reviewers could not have seen there is a mapping error. A row that cites no
    figure only has to name an image the source manifest knows.

    Args:
        row: Decision row.
        where: Human-readable row label used in messages.
        source_index: Output of :func:`load_source_image_index`.
        review_index: Output of :func:`load_review_index`.
        problems: List that failures are appended to.
    """
    scope = row.get("subject_scope", "")
    figures = {figure for figure in (row.get("evidence_figure", "") or "").split("|") if figure}
    subjects = [("subject_id_a", "provider_split_a")]
    if scope == "pair":
        subjects.append(("subject_id_b", "provider_split_b"))
    for id_field, split_field in subjects:
        image_id = row.get(id_field, "")
        if scope == "set":
            if image_id:
                problems.append(f"{where}: a set-level row must not name {id_field}")
            continue
        if not image_id:
            problems.append(f"{where}: {id_field} is empty for a {scope}-level row")
            continue
        record = source_index.get(image_id)
        if record is None:
            problems.append(f"{where}: {id_field}={image_id!r} is not in the source manifest")
            continue
        if figures:
            shown_on = {entry.get("contact_sheet", "") for entry in review_index.get(image_id, [])}
            missing = figures - shown_on
            if missing:
                problems.append(
                    f"{where}: {id_field}={image_id!r} does not appear on {sorted(missing)}"
                )
        recorded_split = row.get(split_field, "")
        if recorded_split and recorded_split != record.get("split"):
            problems.append(
                f"{where}: {split_field}={recorded_split!r} contradicts the manifest "
                f"({record.get('split')!r})"
            )
    if scope == "pair" and row.get("subject_id_a") == row.get("subject_id_b"):
        problems.append(f"{where}: a pair row names the same image twice")


def _check_hygiene(row: dict[str, str], where: str, root: Path, problems: list[str]) -> None:
    """Check a row for absent figures, absolute paths and credential material.

    Args:
        row: Decision row.
        where: Human-readable row label used in messages.
        root: Repository root, used to resolve figure paths.
        problems: List that failures are appended to.
    """
    for figure in filter(None, (row.get("evidence_figure", "") or "").split("|")):
        if not (root / figure).is_file():
            problems.append(f"{where}: evidence figure {figure!r} does not exist")
    for field_name, value in row.items():
        for token in (value or "").split():
            if any(pattern.search(token) for pattern in _ABSOLUTE_PATH_PATTERNS):
                problems.append(f"{where}: {field_name} contains an absolute path {token!r}")
        if any(pattern.search(value or "") for pattern in _SECRET_PATTERNS):
            problems.append(f"{where}: {field_name} looks like credential material")


def validate_decisions(
    rows: Sequence[dict[str, str]],
    *,
    source_index: dict[str, dict[str, Any]],
    review_index: dict[str, list[dict[str, str]]],
    root: Path,
) -> list[str]:
    """Validate a recorded decision file against the phase 4A manifests.

    Args:
        rows: Decision rows.
        source_index: Output of :func:`load_source_image_index`.
        review_index: Output of :func:`load_review_index`.
        root: Repository root, used to resolve evidence-figure paths.

    Returns:
        A list of problems; empty when the record is internally consistent and
        every reference resolves.
    """
    problems: list[str] = []
    if not rows:
        return ["no decisions were recorded"]

    seen_ids: set[str] = set()
    groups: dict[str, list[str]] = {}
    for index, row in enumerate(rows, start=1):
        decision_id = row.get("decision_id", "")
        where = f"row {index} ({decision_id or 'unnamed'})"
        if not decision_id:
            problems.append(f"{where}: decision_id is empty")
        elif decision_id in seen_ids:
            problems.append(f"{where}: decision_id is not unique")
        seen_ids.add(decision_id)
        if not row.get("rationale", "").strip():
            problems.append(f"{where}: rationale is empty")
        _check_vocabulary(row, where, problems)
        _check_subjects(row, where, source_index, review_index, problems)
        _check_hygiene(row, where, root, problems)

        group_id = row.get("group_id", "")
        if group_id:
            if not group_id.startswith(DUPLICATE_GROUP_PREFIX):
                problems.append(f"{where}: group_id={group_id!r} has an unexpected prefix")
            members = [row.get("subject_id_a", ""), row.get("subject_id_b", "")]
            groups.setdefault(group_id, []).extend(member for member in members if member)

    for group_id, members in sorted(groups.items()):
        if len(members) < 2:
            problems.append(f"group {group_id}: fewer than two member ids")
        if len(set(members)) != len(members):
            problems.append(f"group {group_id}: an image id appears more than once")
    return problems
