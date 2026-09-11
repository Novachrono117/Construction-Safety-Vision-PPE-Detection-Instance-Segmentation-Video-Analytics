"""The phase 11B one-shot holdout evaluation runner.

Phase 11A declared this runner's shape so the execution order was fixed and
testable before anything could run. Phase 11B authorises it. The structure is
unchanged: the authorisation preflight still fires first, the frozen step order
still comes from the committed protocol module, and predictions are still
persisted and fingerprinted before any metric exists.

Four things are deliberate.

**It reads the environment gate and never writes it.** Nothing in this file
assigns to the process environment; a test asserts that. A person sets
``CSVISION_ALLOW_TEST_SPLIT=1`` deliberately, immediately before phase 11B, or
the holdout stays closed. Code that can satisfy its own precondition is not a
gate.

**It is the authorisation boundary, not the implementation.** The pipeline lives
in :mod:`construction_safety_vision.final_holdout_execution`, so this file stays
free of any inference, model-loading or split-enumeration call path and remains
the one place the holdout accessor grants access to.

**The order is frozen, not discovered.** :func:`plan` returns the step sequence
from the committed protocol module, so phase 11B cannot quietly reorder
persistence after metric computation - the reordering that would make a report
rebuild impossible.

**A failure is classified, never improvised.** Every exit below maps to one of
the failure states phase 11A named, and none of them authorises re-running
inference. A second attempt requires a number and a written human justification
or the ledger does not construct.

Usage (phase 11B only, after explicit human authorisation):
    uv run python scripts/evaluate_final_holdout.py --plan
    uv run python scripts/evaluate_final_holdout.py --execute
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
    DIRECT_IOU_PROTOCOL_FINGERPRINT,
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

EXECUTION_AUTHORISED = True
"""Flipped in phase 11B's own change, after explicit human authorisation."""

NOT_AUTHORISED = "PHASE_11B_NOT_AUTHORISED"
TEST_UNLOCK_MISSING = "TEST_UNLOCK_AUTHORIZATION_MISSING"
PROTOCOL_VIOLATION = "PROTOCOL_VIOLATION"
COMPLETE = "TEST_EVALUATION_COMPLETE"

EXPECTED_PROTOCOL = "a5a328b3a8e49b06fb8fd9e792abcf43ccdd9aac5422729814dac0dbadc1daef"
"""The phase 11A fingerprint this runner refuses to execute without."""


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
        env: Environment mapping to read. Defaults to the process environment.

    Raises:
        HoldoutProtocolError: If the caller is not this runner or the purpose is
            not the final evaluation.
        HoldoutViolationError: If either gate is missing.
    """
    authorize_final_holdout_access(
        purpose=ACCESS_PURPOSE, caller=RUNNER, allow_test=allow_test, env=env
    )


def run(*, allow_test: bool = False, attempt: int = 1, justification: str | None = None) -> int:
    """Execute the one-shot holdout evaluation.

    The authorisation preflight runs first, so an invocation without both gates
    refuses on the gates rather than reaching anything that could read the
    holdout.

    Args:
        allow_test: The in-code opt-in.
        attempt: The attempt number. Never resets.
        justification: Required beyond the first attempt.

    Returns:
        Process exit status.

    Raises:
        PhaseNotAuthorisedError: If execution has not been authorised.
    """
    from construction_safety_vision import final_holdout_execution as execution

    preflight(allow_test=allow_test)
    if not EXECUTION_AUTHORISED:
        msg = (
            "phase 11B is not authorised. This runner's structure is frozen by phase 11A and "
            "its execution is a separate, explicitly reviewed phase."
        )
        raise PhaseNotAuthorisedError(msg)

    paths = ProjectPaths.from_root()
    protocol = load_protocol(paths.root / CONFIG_PATH)
    problems = validate_protocol(protocol.raw)
    if problems:
        print(f"{PROTOCOL_VIOLATION}: {problems}", file=sys.stderr)
        return 2
    fingerprint = protocol_fingerprint(protocol.raw)
    if fingerprint != EXPECTED_PROTOCOL:
        print(
            f"{PROTOCOL_VIOLATION}: the protocol hashes to {fingerprint}, not the frozen "
            f"{EXPECTED_PROTOCOL}",
            file=sys.stderr,
        )
        return 2

    ledger = (
        OneShotLedger(attempt=attempt, justification=justification)
        if justification is not None
        else OneShotLedger(attempt=attempt)
    )
    ledger.advance(
        "AUTHORIZED",
        detail=(
            f"{CODE_GATE} supplied in code and {ENVIRONMENT_GATE} supplied externally by a "
            f"person; protocol {fingerprint} verified"
        ),
    )

    try:
        identifiers = execution.frozen_holdout_identifiers(paths, allow_test=allow_test)
        bundle = execution.execute(paths, protocol.raw, fingerprint, ledger, allow_test=allow_test)
    except execution.TestDataIntegrityError as exc:
        ledger.fail("TEST_DATA_INTEGRITY_FAILURE", detail=str(exc))
        execution.write_ledger(paths, ledger)
        print(f"TEST_DATA_INTEGRITY_FAILURE: {exc}", file=sys.stderr)
        return 3
    except execution.PredictionExecutionError as exc:
        ledger.fail("PREDICTION_EXECUTION_FAILED_BEFORE_RESULTS", detail=str(exc))
        execution.write_ledger(paths, ledger)
        print(f"PREDICTION_EXECUTION_FAILED_BEFORE_RESULTS: {exc}", file=sys.stderr)
        return 3
    except execution.HoldoutExecutionError as exc:
        ledger.fail(PROTOCOL_VIOLATION, detail=str(exc))
        execution.write_ledger(paths, ledger)
        print(f"{PROTOCOL_VIOLATION}: {exc}", file=sys.stderr)
        return 3

    try:
        written = execution.write_artifacts(
            paths,
            bundle,
            ledger,
            protocol_fingerprint=fingerprint,
            direct_iou_protocol_fingerprint=DIRECT_IOU_PROTOCOL_FINGERPRINT,
            holdout_identifiers=identifiers,
        )
    except Exception as exc:
        ledger.fail(
            "EVALUATION_ARTIFACT_WRITE_FAILURE_AFTER_VALID_PREDICTIONS",
            detail=f"{exc}. The persisted predictions are intact; rebuild from them.",
        )
        execution.write_ledger(paths, ledger)
        print(
            "EVALUATION_ARTIFACT_WRITE_FAILURE_AFTER_VALID_PREDICTIONS: "
            f"{exc}. Do NOT rerun inference; rebuild from the persisted predictions.",
            file=sys.stderr,
        )
        return 4

    ledger.advance("ARTIFACTS_WRITTEN", detail=f"{len(written['committed'])} committed artifacts")
    ledger.advance("COMPLETE", detail=COMPLETE)
    provenance = execution.write_provenance(
        paths, bundle, written, ledger, protocol_fingerprint=fingerprint
    )

    payload = written["report_payload"]
    detector_box = payload["detector"]["canonical_box"]
    segmenter_mask = payload["segmenter"]["canonical_mask"]
    segmenter_box = payload["segmenter"]["canonical_box"]
    print(f"classification  {COMPLETE}")
    print(f"attempt         {ledger.attempt}  ({ledger.state})")
    print(
        f"population      {payload['population']['images']} images / "
        f"{payload['population']['annotations']} annotations"
    )
    print(f"D2 box  mAP@0.50:0.95  {detector_box['CANONICAL_TEST_BOX_MAP50_95']}")
    print(f"S1 mask mAP@0.50:0.95  {segmenter_mask['CANONICAL_TEST_MASK_MAP50_95']}")
    print(f"S1 box  mAP@0.50:0.95  {segmenter_box['S1_CANONICAL_TEST_BOX_MAP50_95']}")
    print(f"provenance      {provenance.relative_to(paths.root)}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Print the frozen execution plan, or execute the one-shot evaluation.

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
    parser.add_argument(
        "--execute",
        action="store_true",
        help="run the one-shot holdout evaluation; requires both authorisation gates",
    )
    parser.add_argument(
        "--attempt",
        type=int,
        default=1,
        help="attempt number; never resets, and beyond 1 a justification is required",
    )
    parser.add_argument(
        "--justification",
        default=None,
        help="written human justification, required for any attempt beyond the first",
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
    print(f"code gate   {CODE_GATE}")
    print(f"env gate    {ENVIRONMENT_GATE}={'unset' if holdout_is_locked() else 'SET'}")
    print()

    if args.plan or not args.execute:
        print(f"PLAN ONLY: phase {PHASE} froze this order. Nothing was executed.")
        return 0

    if holdout_is_locked():
        print(
            f"{TEST_UNLOCK_MISSING}: {ENVIRONMENT_GATE} is not set. The holdout stays closed "
            "and nothing was changed. This runner never sets it; a person must.",
            file=sys.stderr,
        )
        return 2

    return run(allow_test=True, attempt=args.attempt, justification=args.justification)


if __name__ == "__main__":
    sys.exit(main())
