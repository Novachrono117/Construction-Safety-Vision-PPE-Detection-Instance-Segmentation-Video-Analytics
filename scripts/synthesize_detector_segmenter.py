"""Synthesise the committed phase 10B and 10C evidence into the phase 10D artifacts.

Phase 10D answers the project's scientific question from evidence that already
exists. It **loads no model**. A test asserts this file imports neither torch
nor ultralytics and contains no training, inference, timing or holdout call
path, because the one way a synthesis phase can go wrong is by quietly
producing a new measurement and presenting it as an old one.

Four things are deliberate.

**Numbers come from artifacts, not from prose.** Every value is read out of a
committed JSON artifact by field. Nothing is transcribed from a report, rounded
by hand or recomputed from predictions - there are no predictions here.

**Historical artifacts are read, digested and verified unchanged.** Phase 10A's
protocol, phase 10B's corrected results, phase 10C's benchmark and both model
freezes are inputs. If any of their bytes differ before and after this run, the
phase stops rather than writing.

**The validator runs before anything is written.** It compares every headline
value against its source, checks that each caveat is present with the right
polarity, and refuses a composite score, a declared winner, a compliance-accuracy
claim or any holdout content.

**The holdout is not read, and there is nothing here that could read it.** This
phase's inputs are validation-derived result artifacts that themselves record
the holdout as untouched.

Usage:
    uv run python scripts/synthesize_detector_segmenter.py
    uv run python scripts/synthesize_detector_segmenter.py --verify-only
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from construction_safety_vision.data.canonical import scan_for_sensitive
from construction_safety_vision.detector_segmenter_synthesis import (
    BLOCKED,
    CLAIM_VALIDATION_FAILED,
    HISTORICAL,
    PHASE,
    SOURCE_ARTIFACT_MISMATCH,
    SYNTHESIS_COMPLETE,
    TRADEOFF_FIELDS,
    SynthesisError,
    build_synthesis,
    load_sources,
    render_markdown,
    tradeoff_rows,
    validate_synthesis,
)
from construction_safety_vision.paths import ProjectPaths
from construction_safety_vision.provenance import ProvenanceRecord, git_commit, sha256_file
from construction_safety_vision.splits import HOLDOUT_UNLOCK_ENV_VAR

SYNTHESIS_JSON = "detector_segmenter_scientific_synthesis.json"
SYNTHESIS_MD = "detector_segmenter_scientific_synthesis.md"
TRADEOFF_CSV = "detector_segmenter_tradeoff.csv"

NEWLINE = "\n"
"""Written explicitly so artifacts do not pick up platform line endings."""


def write_json(path: Path, payload: Any) -> str:
    """Write a JSON artifact deterministically and return its digest.

    Args:
        path: Destination file.
        payload: JSON-serialisable content.

    Returns:
        The SHA-256 digest of the bytes written.
    """
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + NEWLINE
    path.write_text(text, encoding="utf-8", newline=NEWLINE)
    return sha256_file(path)


def render_tradeoff_csv(rows: Sequence[Mapping[str, str]]) -> str:
    """Render the trade-off table as CSV text.

    Args:
        rows: Rows keyed by the declared trade-off fields.

    Returns:
        The CSV document.
    """
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(TRADEOFF_FIELDS), lineterminator=NEWLINE)
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    return buffer.getvalue()


def holdout_is_locked() -> bool:
    """Report whether the holdout opt-in is absent from the environment.

    Returns:
        True when the opt-in variable is unset.
    """
    return os.environ.get(HOLDOUT_UNLOCK_ENV_VAR) is None


def main(argv: Sequence[str] | None = None) -> int:
    """Build, validate and write the phase 10D synthesis artifacts.

    Args:
        argv: Command-line arguments.

    Returns:
        Process exit status.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="derive and validate the synthesis without writing anything",
    )
    args = parser.parse_args(argv)

    paths = ProjectPaths.from_root()

    if not holdout_is_locked():
        print(
            f"{BLOCKED}: {HOLDOUT_UNLOCK_ENV_VAR} is set. Phase 10D reads no holdout data and "
            "must not run with the opt-in present.",
            file=sys.stderr,
        )
        return 2

    try:
        before = {name: sha256_file(paths.root / name) for name in HISTORICAL}
        sources = load_sources(paths.root)
        payload = build_synthesis(paths.root, sources)
        rows = tradeoff_rows(payload)
    except (SynthesisError, KeyError, OSError) as exc:
        print(f"{SOURCE_ARTIFACT_MISMATCH}: {exc}", file=sys.stderr)
        return 2

    problems = validate_synthesis(payload, sources)
    if problems:
        print(f"{CLAIM_VALIDATION_FAILED}: the synthesis does not validate:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 2

    report = render_markdown(payload)
    table = render_tradeoff_csv(rows)
    findings = (
        scan_for_sensitive(json.dumps(payload))
        + scan_for_sensitive(report)
        + scan_for_sensitive(table)
    )
    if findings:
        print(f"{BLOCKED}: sensitive content: {findings}", file=sys.stderr)
        return 2

    recognition = payload["recognition"]
    association = payload["association"]
    latency = payload["cost"]["latency"]
    end_to_end = latency["boundaries"]["END_TO_END_MODEL_OUTPUT_LATENCY_MS"]
    print(
        "recognition  all-class "
        f"{recognition['all_class']['CANONICAL_BOX_MAP50_95']['delta']:+f}  supported "
        f"{recognition['supported_class_sensitivity']['delta']:+f}"
    )
    print(
        "spatial      fill-ratio median "
        f"{payload['spatial_representation']['MASK_TO_BOX_FILL_RATIO']['median']}  "
        f"mask-only quantities {payload['spatial_representation']['mask_only_quantities']}"
    )
    print(
        "association  mask-only "
        f"{association['mask_only_associations']}  exceptions "
        f"{association['geometry_isolating']['taxonomy_exceptions']}  total "
        f"{association['geometry_isolating']['total_relationships']}"
    )
    print(
        "cost         end-to-end "
        f"{end_to_end['absolute_latency_delta_ms']:+f} ms "
        f"({end_to_end['relative_latency_cost']:+f})  peak reserved ratio "
        f"{payload['cost']['memory']['peak_reserved']['ratio']}"
    )
    print(f"claims       {len(payload['claim_register'])} registered")
    print("models       0 loaded, 0 executed (synthesis of committed evidence only)")
    print(f"holdout      {payload['holdout_policy']['status']}")

    if args.verify_only:
        print("VERIFIED: the synthesis derives from committed evidence. Nothing written.")
        return 0

    synthesis_sha = write_json(paths.reports / SYNTHESIS_JSON, payload)
    (paths.reports / SYNTHESIS_MD).write_text(report, encoding="utf-8", newline=NEWLINE)
    (paths.reports / TRADEOFF_CSV).write_text(table, encoding="utf-8", newline=NEWLINE)

    after = {name: sha256_file(paths.root / name) for name in HISTORICAL}
    changed = sorted(name for name in before if before[name] != after[name])
    if changed:
        print(f"{BLOCKED}: historical artifacts changed: {changed}", file=sys.stderr)
        return 2

    record = ProvenanceRecord.create(
        name="detector_segmenter_scientific_synthesis",
        phase=10,
        config={"protocol_fingerprint": payload["protocol_fingerprint"]},
        details={
            "phase": PHASE,
            "classification": SYNTHESIS_COMPLETE,
            "synthesis_sha256": payload["synthesis_sha256"],
            "source_artifacts": payload["source_artifacts"],
            "execution": payload["execution"],
            "historical_artifacts_unchanged": True,
            "holdout": payload["holdout_policy"]["status"],
            "claims_registered": len(payload["claim_register"]),
            "next_phase": payload["next_phase"],
        },
        repo_root=paths.root,
    )
    record.write_json(paths.reports / "detector_segmenter_synthesis.provenance.json")

    print(SYNTHESIS_COMPLETE)
    print(f"synthesis   reports/{SYNTHESIS_JSON}  sha256 {synthesis_sha}")
    print(f"report      reports/{SYNTHESIS_MD}")
    print(f"tradeoff    reports/{TRADEOFF_CSV}  {len(rows)} rows")
    print(f"fingerprint {payload['synthesis_sha256']}")
    print(f"commit      {git_commit(paths.root)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
