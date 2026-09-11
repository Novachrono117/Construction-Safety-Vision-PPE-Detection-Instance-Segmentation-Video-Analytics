"""Shared fixtures and helpers for the test suite.

The one helper here answers a question several suites need: has the single
authorised holdout evaluation happened yet?

Until phase 11B runs, no holdout image or annotation may exist on disk, and
several phases assert exactly that. Phase 11B materialises the holdout - through
the phase 5D function, under both authorisation gates, as its frozen protocol
requires - so from that point the same directories legitimately exist. Gating
those assertions on the committed evidence that the evaluation happened keeps
the protection for every other state of the repository, including the one that
matters most: a holdout materialised by accident during development.
"""

from __future__ import annotations

from construction_safety_vision.paths import ProjectPaths

FINAL_EVALUATION_PROVENANCE = "final_test_evaluation.provenance.json"
"""Committed evidence that the one-shot holdout evaluation was executed."""


def holdout_has_been_evaluated(paths: ProjectPaths | None = None) -> bool:
    """Report whether phase 11B has run in this checkout.

    Args:
        paths: Project layout. Resolved from the repository root when omitted.

    Returns:
        ``True`` once the final evaluation's provenance record exists.
    """
    resolved = paths if paths is not None else ProjectPaths.from_root()
    return (resolved.reports / FINAL_EVALUATION_PROVENANCE).is_file()
