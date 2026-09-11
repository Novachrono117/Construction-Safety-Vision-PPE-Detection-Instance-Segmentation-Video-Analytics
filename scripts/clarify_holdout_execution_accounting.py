"""Write the phase 11B execution-accounting provenance clarification.

Phase 11B's committed provenance says ``models_executed: 2`` and
``inference_passes: 3`` without saying which model ran twice, and "one-shot" can
be misread as one prediction call per model. It was not: the segmenter ran a
second, real prediction pass at the operational confidence the frozen protocol
declares for the phase 8C direct-IoU diagnostic. This script records that
precisely, as an additive artifact.

Five things are deliberate.

**It refuses to run while the holdout gate is open.** ``CSVISION_ALLOW_TEST_SPLIT``
must be absent from the process environment. A clarification is written after
authorisation has closed, and this script never sets the variable - a test
asserts it does not assign to the environment.

**It loads no model and reads no holdout byte.** Every value comes from an
artifact phase 11B already committed. A test asserts this file imports neither
torch nor ultralytics, and that it names no split accessor.

**It never edits a historical artifact.** The phase 11A protocol and every phase
11B result are digested before and after, and the script fails if one moved.
Clarifying an ambiguous field means adding unambiguous siblings in a new
artifact, not rewriting the record that carries the ambiguity.

**It is idempotent.** Running it twice produces byte-identical outputs, because
the payload is derived from committed evidence rather than assembled from
whatever the session happens to know.

**It declares nothing it cannot source.** Aggregate counts a committed field
corroborates carry ``COMMITTED_ARTIFACT_FIELD``; the per-instance comparison
performed in an earlier session, which persisted no artifact, carries
``OPERATOR_DECLARED_PRIOR_SESSION_AUDIT_NOT_PERSISTED`` instead.

Usage:
    uv run python scripts/clarify_holdout_execution_accounting.py --verify-only
    uv run python scripts/clarify_holdout_execution_accounting.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path

from construction_safety_vision.data.canonical import scan_for_sensitive
from construction_safety_vision.final_holdout_accounting import (
    ACCOUNTING_JSON,
    ACCOUNTING_MARKDOWN,
    ACCOUNTING_PROVENANCE,
    CLASSIFICATION,
    ENVIRONMENT_GATE,
    PROTECTED,
    AccountingError,
    build,
    render,
    sha256_bytes,
    validate,
)
from construction_safety_vision.paths import ProjectPaths
from construction_safety_vision.provenance import ProvenanceRecord, git_commit

BLOCKED = "BLOCKED"
"""Printed when the clarification refuses to write."""

CLARIFIED = "PHASE_11B_EXECUTION_ACCOUNTING_CLARIFIED"
"""Printed on success."""

NEWLINE = "\n"
"""Artifacts are written with LF endings on every platform."""


def gate_is_open(env: dict[str, str] | None = None) -> bool:
    """Report whether the holdout environment gate is present.

    Args:
        env: Environment mapping to read. Defaults to the process environment.

    Returns:
        ``True`` when the gate variable is set to anything at all.
    """
    source = os.environ if env is None else env
    return source.get(ENVIRONMENT_GATE) is not None


def write_json(path: Path, payload: object) -> str:
    """Write a JSON artifact with LF endings and return its digest.

    Args:
        path: Destination file.
        payload: The document.

    Returns:
        The SHA-256 of the bytes written.
    """
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + NEWLINE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline=NEWLINE)
    return sha256_bytes(path)


def main(argv: Sequence[str] | None = None) -> int:
    """Build, validate and write the execution-accounting clarification.

    Args:
        argv: Command-line arguments.

    Returns:
        Process exit status.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="build and validate the clarification without writing anything",
    )
    args = parser.parse_args(argv)

    if gate_is_open():
        print(
            f"{BLOCKED}: {ENVIRONMENT_GATE} is set in this process. A provenance clarification "
            "is written after holdout authorisation has closed. This script never unsets it; "
            "relaunch the session from a clean environment.",
            file=sys.stderr,
        )
        return 2

    paths = ProjectPaths.from_root()
    before = {name: sha256_bytes(paths.root / name) for name in PROTECTED}

    try:
        payload = build(paths, protected_digests=before)
    except AccountingError as exc:
        print(f"{BLOCKED}: {exc}", file=sys.stderr)
        return 2

    problems = validate(payload)
    if problems:
        print(f"{BLOCKED}: the clarification does not validate:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 2

    document = render(payload)
    findings = scan_for_sensitive(json.dumps(payload, ensure_ascii=False))
    findings.extend(scan_for_sensitive(document))
    if findings:
        print(f"{BLOCKED}: sensitive content: {findings}", file=sys.stderr)
        return 2

    if args.verify_only:
        print(f"{CLASSIFICATION} validates")
        print(f"fingerprint {payload['execution_accounting_sha256']}")
        print("nothing written")
        return 0

    json_sha = write_json(paths.reports / ACCOUNTING_JSON, payload)
    (paths.reports / ACCOUNTING_MARKDOWN).write_text(document, encoding="utf-8", newline=NEWLINE)

    after = {name: sha256_bytes(paths.root / name) for name in PROTECTED}
    changed = sorted(name for name in before if before[name] != after[name])
    if changed:
        print(f"{BLOCKED}: protected artifacts changed: {changed}", file=sys.stderr)
        return 2

    record = ProvenanceRecord.create(
        "final_test_execution_accounting",
        phase=11,
        repo_root=paths.root,
        config={"protocol_fingerprint": payload["protocol_fingerprint"]},
        details={
            "classification": CLASSIFICATION,
            "clarification_type": payload["clarification_type"],
            "phase": payload["phase"],
            "execution_accounting_sha256": payload["execution_accounting_sha256"],
            "is_a_test_result_correction": False,
            "scientific_results_changed": False,
            "historical_artifacts_unchanged": True,
            "holdout_accessed": False,
            "holdout_accessor_invoked": False,
            "models_loaded": 0,
            "models_executed": 0,
            "model_inference_passes": 0,
            "predictions_regenerated": 0,
            "metrics_recomputed": 0,
            "test_images_read": 0,
            "test_annotations_read": 0,
            "test_identifiers_enumerated": 0,
            "environment_test_gate_present": False,
            "environment_gate_written": False,
            "clarified": {
                "unique_models_executed": payload["inference_accounting"]["unique_models_executed"],
                "total_model_inference_passes": payload["inference_accounting"][
                    "total_model_inference_passes"
                ],
                "d2_inference_invocations": payload["inference_accounting"][
                    "d2_inference_invocations"
                ],
                "s1_inference_invocations": payload["inference_accounting"][
                    "s1_inference_invocations"
                ],
                "one_shot_label": payload["one_shot_semantics"]["label"],
                "frozen_protocol_satisfied": payload["frozen_protocol_context"][
                    "FROZEN_PROTOCOL_SATISFIED"
                ],
                "later_single_s1_invocation_instruction_satisfied": payload[
                    "frozen_protocol_context"
                ]["LATER_EXECUTION_INSTRUCTION_SINGLE_S1_INVOCATION_SATISFIED"],
            },
        },
    )
    for name in PROTECTED:
        record.add_input(paths.root / name, relative_to=paths.root)
    for name in (ACCOUNTING_JSON, ACCOUNTING_MARKDOWN):
        record.add_output(paths.reports / name, relative_to=paths.root)
    record.write_json(paths.reports / ACCOUNTING_PROVENANCE)

    print(CLARIFIED)
    print(f"clarification reports/{ACCOUNTING_JSON}  sha256 {json_sha}")
    print(f"report        reports/{ACCOUNTING_MARKDOWN}")
    print(f"provenance    reports/{ACCOUNTING_PROVENANCE}")
    print(f"fingerprint   {payload['execution_accounting_sha256']}")
    print(f"protected     {len(PROTECTED)} artifacts verified byte-identical")
    print(f"commit        {git_commit(paths.root)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
