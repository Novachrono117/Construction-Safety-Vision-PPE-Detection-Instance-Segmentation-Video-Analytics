"""Phase 12A - audit the repository's academic and portfolio delivery readiness.

Read-only. This script loads no model, reads no holdout content, recomputes no
metric and modifies no scientific artifact. It measures the repository, applies
the declared findings, and writes four artifacts that say what must be improved
before final delivery.

Four things are deliberate.

**It refuses to run while the holdout gate is open.** ``CSVISION_ALLOW_TEST_SPLIT``
must be absent. An audit is performed with the experiment closed, and this
script never sets the variable - a test asserts it does not assign to the
environment.

**Measurements and judgements are written to separate blocks.** The inventory,
the deliverable checks, the stale-claim probes and the documentation-consistency
checks re-derive on every run; the persona verdicts and gap severities are
editorial and labelled as such. Fix the README and the next run says so.

**It writes only its own artifacts.** Every pre-existing scientific artifact is
digested before and after, and the script fails if one moved.

**It is idempotent.** Running it twice produces byte-identical outputs.

Usage:
    uv run python scripts/audit_delivery_readiness.py --verify-only
    uv run python scripts/audit_delivery_readiness.py
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path

from construction_safety_vision.data.canonical import scan_for_sensitive
from construction_safety_vision.delivery_audit import (
    AUDIT_JSON,
    AUDIT_MARKDOWN,
    AUDIT_PROVENANCE,
    CLASSIFICATION_COMPLETE,
    COMPLIANCE_CSV,
    ENVIRONMENT_GATE,
    GAP_CSV,
    AuditError,
    build,
    classify,
    compliance_rows,
    gap_rows,
    render,
    tracked_files,
    validate,
)
from construction_safety_vision.paths import ProjectPaths
from construction_safety_vision.provenance import ProvenanceRecord, git_commit, sha256_file

BLOCKED = "BLOCKED"
"""Printed when the audit refuses to write."""

NEWLINE = "\n"
"""Artifacts are written with LF endings on every platform."""

OWN_ARTIFACTS: frozenset[str] = frozenset(
    {
        f"reports/{AUDIT_JSON}",
        f"reports/{AUDIT_MARKDOWN}",
        f"reports/{GAP_CSV}",
        f"reports/{COMPLIANCE_CSV}",
        f"reports/{AUDIT_PROVENANCE}",
    }
)
"""The only files this phase may create or change."""


def gate_is_open(env: dict[str, str] | None = None) -> bool:
    """Report whether the holdout environment gate is present.

    Args:
        env: Environment mapping to read. Defaults to the process environment.

    Returns:
        ``True`` when the gate variable is set to anything at all.
    """
    source = os.environ if env is None else env
    return source.get(ENVIRONMENT_GATE) is not None


def existing_digests(root: Path) -> dict[str, str]:
    """Digest every tracked file this phase must not change.

    Args:
        root: Repository root.

    Returns:
        Path to SHA-256, excluding the audit's own artifacts.
    """
    return {
        name: sha256_file(root / name)
        for name in tracked_files(root)
        if name not in OWN_ARTIFACTS and (root / name).is_file()
    }


def write_csv(path: Path, rows: Sequence[Sequence[str]]) -> str:
    """Write a CSV artifact with LF endings and return its digest.

    Args:
        path: Destination file.
        rows: Rows, header first.

    Returns:
        The SHA-256 of the bytes written.
    """
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator=NEWLINE)
    writer.writerows(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(buffer.getvalue(), encoding="utf-8", newline=NEWLINE)
    return sha256_file(path)


def main(argv: Sequence[str] | None = None) -> int:
    """Build, validate and write the phase 12A audit artifacts.

    Args:
        argv: Command-line arguments.

    Returns:
        Process exit status.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="build and validate the audit without writing anything",
    )
    args = parser.parse_args(argv)

    if gate_is_open():
        print(
            f"{BLOCKED}: {ENVIRONMENT_GATE} is set in this process. The audit runs with the "
            "experiment closed. This script never unsets it; relaunch from a clean "
            "environment.",
            file=sys.stderr,
        )
        return 2

    paths = ProjectPaths.from_root()
    before = existing_digests(paths.root)

    try:
        payload = build(paths, env=dict(os.environ))
    except AuditError as exc:
        print(f"{BLOCKED}: {exc}", file=sys.stderr)
        return 2

    problems = validate(payload)
    if problems:
        print(f"{BLOCKED}: the audit does not validate:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 2

    classification = classify(payload)
    if classification != CLASSIFICATION_COMPLETE:
        print(f"{classification}: the audit did not complete cleanly", file=sys.stderr)
        return 3

    document = render(payload)
    findings = scan_for_sensitive(json.dumps(payload, ensure_ascii=False))
    findings.extend(scan_for_sensitive(document))
    if findings:
        print(f"{BLOCKED}: sensitive content: {findings}", file=sys.stderr)
        return 2

    if args.verify_only:
        print(f"{classification}")
        print(f"fingerprint {payload['repository_audit_sha256']}")
        print("nothing written")
        return 0

    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + NEWLINE
    (paths.reports / AUDIT_JSON).write_text(text, encoding="utf-8", newline=NEWLINE)
    (paths.reports / AUDIT_MARKDOWN).write_text(document, encoding="utf-8", newline=NEWLINE)
    gap_sha = write_csv(paths.reports / GAP_CSV, gap_rows(payload))
    compliance_sha = write_csv(paths.reports / COMPLIANCE_CSV, compliance_rows(payload))

    after = existing_digests(paths.root)
    changed = sorted(name for name in before if before.get(name) != after.get(name))
    if changed:
        print(f"{BLOCKED}: pre-existing artifacts changed: {changed}", file=sys.stderr)
        return 2

    summary = payload["summary"]
    record = ProvenanceRecord.create(
        "final_repository_audit",
        phase=12,
        repo_root=paths.root,
        config={"phase": payload["phase"]},
        details={
            "classification": classification,
            "phase": payload["phase"],
            "repository_audit_sha256": payload["repository_audit_sha256"],
            "audit_only": True,
            "contains_new_scientific_results": False,
            "scientific_results_modified": False,
            "holdout_accessed": False,
            "models_executed": 0,
            "model_inference_passes": 0,
            "metrics_recomputed": 0,
            "test_accessed": False,
            "environment_test_gate_present": False,
            "readme_rewritten": False,
            "report_generated": False,
            "video_started": False,
            "pre_existing_artifacts_unchanged": True,
            "summary": summary,
            "readiness": payload["readiness"],
        },
    )
    for name in (AUDIT_JSON, AUDIT_MARKDOWN, GAP_CSV, COMPLIANCE_CSV):
        record.add_output(paths.reports / name, relative_to=paths.root)
    record.write_json(paths.reports / AUDIT_PROVENANCE)

    print(classification)
    print(f"audit       reports/{AUDIT_JSON}  sha256 {sha256_file(paths.reports / AUDIT_JSON)}")
    print(f"report      reports/{AUDIT_MARKDOWN}")
    print(f"gaps        reports/{GAP_CSV}  sha256 {gap_sha}")
    print(f"compliance  reports/{COMPLIANCE_CSV}  sha256 {compliance_sha}")
    print(f"fingerprint {payload['repository_audit_sha256']}")
    print(
        f"summary     {summary['compliance']['COMPLETE']} complete / "
        f"{summary['compliance']['PARTIAL']} partial / {summary['compliance']['MISSING']} "
        f"missing; P0 {summary['gaps']['P0']}, P1 {summary['gaps']['P1']}, "
        f"P2 {summary['gaps']['P2']}, P3 {summary['gaps']['P3']}"
    )
    print(f"unchanged   {len(before)} pre-existing tracked files verified byte-identical")
    print(f"commit      {git_commit(paths.root)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
