"""The phase 11B one-shot holdout evaluation runner - structure only, not authorised.

Phase 11A declares this runner's shape so the execution order is fixed and
testable before anything can run. **It does not execute.** Every step that would
touch a model or the holdout raises :class:`PhaseNotAuthorisedError`, and the
authorisation preflight runs first regardless, so a stray invocation refuses on
the gates rather than on a missing implementation.

Three things are deliberate.

**It reads the environment gate and never writes it.** Nothing in this file
assigns to ``os.environ``; a test asserts that. A person sets
``CSVISION_ALLOW_TEST_SPLIT=1`` deliberately, immediately before phase 11B, or
the holdout stays closed. Code that can satisfy its own precondition is not a
gate.

**The order is frozen here, not discovered later.** :func:`plan` returns the
step sequence from the committed protocol module, so phase 11B cannot quietly
reorder persistence after metric computation - which is the reordering that
would make a report rebuild impossible.

**Predictions are persisted and fingerprinted before any metric exists.** That
is what lets an artifact-write failure be recovered by rebuilding rather than by
running the models a second time, and it is why those two failures have separate
names in the frozen policy.

Usage (phase 11B only, after explicit human authorisation):
    uv run python scripts/evaluate_final_holdout.py --plan
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from construction_safety_vision.final_holdout_evaluation import (
    ACCESS_PURPOSE,
    BLOCKED,
    CODE_GATE,
    CONFIG_PATH,
    ENVIRONMENT_GATE,
    PHASE,
    RUNNER,
    HoldoutProtocolError,
    OneShotLedger,
    PhaseNotAuthorisedError,
    authorize_final_holdout_access,
    holdout_is_locked,
    load_protocol,
    phase_11b_steps,
    protocol_fingerprint,
    validate_protocol,
)
from construction_safety_vision.paths import ProjectPaths

EXECUTION_AUTHORISED = False
"""Flipped only when phase 11B is explicitly authorised, in that phase's own change."""

NOT_AUTHORISED = "PHASE_11B_NOT_AUTHORISED"
TEST_UNLOCK_MISSING = "TEST_UNLOCK_AUTHORIZATION_MISSING"


def plan() -> tuple[str, ...]:
    """Return the frozen phase 11B execution order.

    Returns:
        The ordered step names, from the committed protocol module.
    """
    return phase_11b_steps()


def preflight(*, allow_test: bool, env: dict[str, str] | None = None) -> None:
    """Check both authorisation gates before anything else happens.

    Args:
        allow_test: The in-code opt-in, supplied by the phase 11B caller.
        env: Environment mapping to read. Defaults to ``os.environ``.

    Raises:
        HoldoutProtocolError: If the caller is not this runner or the purpose is
            not the final evaluation.
        HoldoutViolationError: If either gate is missing.
    """
    authorize_final_holdout_access(
        purpose=ACCESS_PURPOSE, caller=RUNNER, allow_test=allow_test, env=env
    )


def run(*, allow_test: bool = False) -> int:
    """Execute the one-shot evaluation. Not authorised in phase 11A.

    The authorisation preflight runs first, so an invocation without both gates
    refuses on the gates rather than reaching the phase guard below.

    Args:
        allow_test: The in-code opt-in.

    Returns:
        Never returns in phase 11A.

    Raises:
        PhaseNotAuthorisedError: Always, while :data:`EXECUTION_AUTHORISED` is
            false.
    """
    preflight(allow_test=allow_test)
    if not EXECUTION_AUTHORISED:
        msg = (
            "phase 11B is not authorised. This runner's structure is frozen by phase 11A and "
            "its execution is a separate, explicitly reviewed phase. No model is loaded, no "
            "holdout data is read and no artifact is written here."
        )
        raise PhaseNotAuthorisedError(msg)
    # Phase 11B implements the frozen steps here, in the order plan() returns,
    # advancing the ledger at each transition. Deliberately absent in 11A: an
    # untested inference path that could be run by accident is worse than none.
    raise PhaseNotAuthorisedError(NOT_AUTHORISED)  # pragma: no cover


def main(argv: Sequence[str] | None = None) -> int:
    """Print the frozen execution plan, or refuse to run.

    Args:
        argv: Command-line arguments.

    Returns:
        Process exit status.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plan",
        action="store_true",
        help="print the frozen phase 11B execution order and exit without running",
    )
    args = parser.parse_args(argv)

    paths = ProjectPaths.from_root()
    try:
        protocol = load_protocol(paths.root / CONFIG_PATH)
    except HoldoutProtocolError as exc:
        print(f"{BLOCKED}: {exc}", file=sys.stderr)
        return 2
    problems = validate_protocol(protocol.raw)
    if problems:
        print(f"{BLOCKED}: the frozen protocol does not validate: {problems}", file=sys.stderr)
        return 2

    print(f"protocol    {CONFIG_PATH}")
    print(f"fingerprint {protocol_fingerprint(protocol.raw)}")
    print(f"status      {protocol.status}")
    print(f"detector    {protocol.detector.experiment}  {protocol.detector.checkpoint_sha256}")
    print(f"segmenter   {protocol.segmenter.experiment}  {protocol.segmenter.checkpoint_sha256}")
    print(f"population  {protocol.test_images} images / {protocol.test_annotations} annotations")
    print()
    print("frozen phase 11B execution order:")
    for index, step in enumerate(plan(), start=1):
        print(f"  {index:>2}. {step}")
    print()
    print(f"ledger      {OneShotLedger().state} (not opened)")
    print(f"code gate   {CODE_GATE}")
    print(f"env gate    {ENVIRONMENT_GATE}={'unset' if holdout_is_locked() else 'SET'}")

    if args.plan:
        print(f"PLAN ONLY: phase {PHASE} froze this order. Nothing was executed.")
        return 0

    if holdout_is_locked():
        print(
            f"{TEST_UNLOCK_MISSING}: {ENVIRONMENT_GATE} is not set. The holdout stays closed "
            "and nothing was changed. This runner never sets it; a person must.",
            file=sys.stderr,
        )
        return 2
    print(
        f"{NOT_AUTHORISED}: phase 11B has not been authorised. Phase {PHASE} froze this "
        "runner's structure only.",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    sys.exit(main())
