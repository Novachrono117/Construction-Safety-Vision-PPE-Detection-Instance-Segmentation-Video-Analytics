"""The canonical annotation snapshot: which annotation state the project builds on.

Phase 5A answers one question - not which images are used, not how they are
split, but *which reproducible annotation state* every later phase derives from.
Two candidates exist: the frozen version-4 export, and the provider's live source
project. This module holds the decision vocabulary, the manifest schema, and the
checks that keep the manifest safe to commit.

The manifest is evidence, so it is validated rather than trusted: a missing field
is a defect, and so is a credential, a signed URL or a machine-specific path that
should never reach a public repository.
"""

from __future__ import annotations

import re
from typing import Any

CURRENT_COMPLETE_GEOMETRY = "CURRENT_COMPLETE_GEOMETRY"
"""The live source state, recovered read-only with complete geometry."""

FROZEN_V4_SNAPSHOT = "FROZEN_V4_SNAPSHOT"
"""The frozen version-4 export."""

BLOCKED = "BLOCKED"
"""Neither candidate could be established safely."""

DATASET_REJECTED = "DATASET_REJECTED"
"""The dataset is unsuitable for the task itself."""

DECISIONS = (CURRENT_COMPLETE_GEOMETRY, FROZEN_V4_SNAPSHOT, BLOCKED, DATASET_REJECTED)
"""Every decision this phase may reach."""

PHASE_CLASSIFICATION = {
    CURRENT_COMPLETE_GEOMETRY: "COMPLETE_CURRENT",
    FROZEN_V4_SNAPSHOT: "COMPLETE_V4",
    BLOCKED: "BLOCKED",
    DATASET_REJECTED: "DATASET_REJECTED",
}
"""How each decision classifies the phase itself."""

REQUIRED_FIELDS = (
    "decision",
    "phase_classification",
    "source_project",
    "source_image_count",
    "canonical_snapshot",
    "annotation_count",
    "geometry_representation_counts",
    "category_map",
    "acquisition_method",
    "artifact_fingerprints",
    "drift_summary",
    "limitations",
    "git_commit",
)
"""Fields the manifest must carry to be usable as evidence."""

SIGNED_URL_PATTERN = re.compile(
    r"https?://[^\s\"']*(?:\?|&|x-amz-|signature=|token=|key=|expires=)", re.IGNORECASE
)
"""A URL carrying a query string or a credential-shaped parameter."""

ABSOLUTE_PATH_PATTERN = re.compile(
    # The drive letter must not be preceded by another letter, or the "s:/" inside
    # "https://" matches and every legitimate public URL is reported as a path.
    r"(?<![A-Za-z0-9])[A-Za-z]:[\\/]|(?:^|[\s\"'])(?:/home/|/Users/|/root/|\\\\)",
    re.MULTILINE,
)
"""A machine-specific filesystem path."""

CREDENTIAL_PATTERN = re.compile(
    r"(?:api[_-]?key|secret|password|bearer)\s*[:=]\s*[\"']?[A-Za-z0-9_\-]{8,}", re.IGNORECASE
)
"""An assignment that looks like a credential value."""


class CanonicalDecisionError(ValueError):
    """Raised when a canonical decision or its manifest is not well formed."""


def scan_for_sensitive(text: str) -> list[str]:
    """Find content that must never be committed.

    Args:
        text: Serialised manifest or report text.

    Returns:
        One description per problem found, empty when the text is clean.
    """
    problems: list[str] = []
    if SIGNED_URL_PATTERN.search(text):
        problems.append("contains a URL with a query string or credential-shaped parameter")
    if ABSOLUTE_PATH_PATTERN.search(text):
        problems.append("contains an absolute machine-specific filesystem path")
    if CREDENTIAL_PATTERN.search(text):
        problems.append("contains something shaped like a credential assignment")
    return problems


def validate_manifest(manifest: dict[str, Any]) -> list[str]:
    """Check a canonical manifest for structural defects.

    Args:
        manifest: The manifest mapping.

    Returns:
        One description per problem found, empty when the manifest is valid.
    """
    problems: list[str] = []
    missing = [field for field in REQUIRED_FIELDS if field not in manifest]
    if missing:
        problems.append(f"missing required field(s): {', '.join(missing)}")

    decision = manifest.get("decision")
    if decision not in DECISIONS:
        problems.append(f"decision {decision!r} is not one of {DECISIONS}")
    elif manifest.get("phase_classification") != PHASE_CLASSIFICATION[decision]:
        problems.append(
            f"phase_classification {manifest.get('phase_classification')!r} does not match "
            f"decision {decision!r}, which maps to {PHASE_CLASSIFICATION[decision]!r}"
        )

    counts = manifest.get("geometry_representation_counts")
    total = manifest.get("annotation_count")
    if isinstance(counts, dict) and isinstance(total, int):
        summed = sum(int(v) for v in counts.values())
        if summed != total:
            problems.append(
                f"geometry_representation_counts sum to {summed} but annotation_count is {total}; "
                "an annotation has been dropped or double counted"
            )
    return problems


def assert_manifest_committable(manifest: dict[str, Any], serialised: str) -> None:
    """Fail loudly rather than commit a defective or unsafe manifest.

    Args:
        manifest: The manifest mapping.
        serialised: Its serialised form, which is what would reach the repository.

    Raises:
        CanonicalDecisionError: If the manifest is malformed or unsafe to commit.
    """
    problems = validate_manifest(manifest) + scan_for_sensitive(serialised)
    if problems:
        listed = "; ".join(problems)
        msg = f"Canonical manifest is not fit to commit: {listed}"
        raise CanonicalDecisionError(msg)
