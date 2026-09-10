"""Re-express the phase 10B association results under a corrected taxonomy.

A targeted result-semantics correction. Phase 10A froze four association
categories on the implicit assumption that a rule either associates or it does
not; applying them to real predictions turned up a fifth *state* - both rules
associate, to different people - that none of the four describes.

This script fixes how that state is reported. It does not change what was
measured.

Four things are deliberate.

**No model is loaded.** Every count the correction needs was already committed
by phase 10B, so the restructure is arithmetic over recorded numbers. A test
asserts this file imports neither torch nor ultralytics.

**The frozen four are untouched.** Their names, their counts and their meaning
are exactly what phase 10B recorded. What changes is that the undeclared state
is reported as an exception to the taxonomy's coverage rather than as a fifth
peer category, and is excluded from the denominator the frozen percentages use.

**The phase 10A protocol is historical.** It is read, verified byte-identical,
and never rewritten. A taxonomy discovered to be non-exhaustive is a recorded
limitation of that protocol, not a licence to edit it.

**A different-person outcome is a disagreement, not an error.** There is no
person-PPE association ground truth in this project, so neither rule's answer
can be called wrong.

Usage:
    uv run python scripts/correct_association_taxonomy.py
    uv run python scripts/correct_association_taxonomy.py --verify-only
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from construction_safety_vision.data.canonical import scan_for_sensitive
from construction_safety_vision.detection_comparison import (
    SUPPORT_MIN_INSTANCES,
    SUPPORT_MIN_POSITIVE_IMAGES,
)
from construction_safety_vision.detector_segmenter_analysis import (
    ASSOCIATION_DISAGREEMENT,
    RARE_CLASS,
    RARE_CLASS_STATUS,
    TAXONOMY_EXCEPTION,
    TAXONOMY_EXHAUSTIVE,
    TAXONOMY_NON_EXHAUSTIVE,
    result_fingerprint,
    supported_class_sensitivity,
    tally_from_counts,
    validate_box_comparison,
    validate_spatial_comparison,
)
from construction_safety_vision.detector_segmenter_comparison import ASSOCIATION_CATEGORIES
from construction_safety_vision.paths import ProjectPaths
from construction_safety_vision.provenance import ProvenanceRecord, git_commit, sha256_file

BOX_JSON = "detector_segmenter_box_comparison.json"
SPATIAL_JSON = "detector_segmenter_spatial_comparison.json"
PROTOCOL_JSON = "detector_segmenter_comparison_protocol.json"
REPORT_MD = "detector_segmenter_validation_comparison.md"

LEGACY_EXCEPTION_KEY = "BOTH_ASSOCIATED_DIFFERENT_PERSON"
"""What phase 10B's first pass called the state, before the correction."""

NEWLINE = "\n"
"""Written explicitly so artifacts do not pick up platform line endings."""

CORRECTED = "ASSOCIATION_TAXONOMY_CORRECTED"
BLOCKED = "BLOCKED"

HISTORICAL: tuple[str, ...] = (
    "configs/detector_segmenter_comparison.yaml",
    "reports/detector_segmenter_comparison_protocol.json",
    "reports/detector_segmenter_comparison_protocol.md",
    "reports/detector_segmenter_comparison_membership.csv",
    "reports/detector_segmenter_latency_membership.csv",
    "reports/final_detector_manifest.json",
    "reports/final_segmenter_manifest.json",
)
"""Phase 10A and the two freezes. Read, verified, never rewritten."""


class CorrectionError(RuntimeError):
    """Raised when the correction cannot be made from recorded values alone."""


def read_json(path: Path) -> dict[str, Any]:
    """Read a committed JSON artifact.

    Args:
        path: File to read.

    Returns:
        The parsed mapping.

    Raises:
        CorrectionError: If the file is missing or malformed.
    """
    if not path.is_file():
        msg = f"required input not found: {path.name}"
        raise CorrectionError(msg)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        msg = f"{path.name} must contain a JSON object"
        raise CorrectionError(msg)
    return data


def write_json(path: Path, payload: Any) -> str:
    """Write a JSON artifact deterministically and return its digest.

    Args:
        path: Destination file.
        payload: JSON-serialisable content.

    Returns:
        The SHA-256 digest of the bytes written.
    """
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    path.write_text(text, encoding="utf-8", newline="\n")
    return sha256_file(path)


def normalise(counts: Mapping[str, Any]) -> dict[str, int]:
    """Map a recorded count block onto the corrected outcome names.

    Args:
        counts: Counts as phase 10B first recorded them.

    Returns:
        Counts keyed by the corrected names.

    Raises:
        CorrectionError: If an unrecognised outcome appears, which would mean
            silently dropping observations.
    """
    corrected: dict[str, int] = {}
    for name, value in counts.items():
        key = TAXONOMY_EXCEPTION if name == LEGACY_EXCEPTION_KEY else name
        if key not in (*ASSOCIATION_CATEGORIES, TAXONOMY_EXCEPTION):
            msg = f"unrecognised association outcome in the recorded counts: {name!r}"
            raise CorrectionError(msg)
        corrected[key] = int(value)
    return corrected


def rebuild_block(recorded: Mapping[str, Any]) -> dict[str, Any]:
    """Rebuild one association block under the corrected taxonomy.

    Args:
        recorded: The block phase 10B wrote.

    Returns:
        The corrected block, carrying the same observations.

    Raises:
        CorrectionError: If the rebuilt totals do not match what was recorded.
    """
    # Accept either shape, so the correction is idempotent: run twice and the
    # second run reproduces the first byte for byte rather than failing on its
    # own output.
    if "counts" in recorded:
        counts = normalise(recorded["counts"])
        total = int(recorded["total"])
    else:
        counts = {
            **{name: int(value) for name, value in recorded["frozen_category_counts"].items()},
            TAXONOMY_EXCEPTION: int(recorded["taxonomy_exceptions"]),
        }
        total = int(recorded["total_relationships"])

    rebuilt = tally_from_counts(counts)
    if rebuilt["total_relationships"] != total:
        msg = (
            f"re-aggregation changed the total: recorded {total}, rebuilt "
            f"{rebuilt['total_relationships']}"
        )
        raise CorrectionError(msg)
    if not rebuilt["counts_partition"]:
        msg = "the rebuilt counts do not partition the relationships"
        raise CorrectionError(msg)
    for name in ASSOCIATION_CATEGORIES:
        if rebuilt["frozen_category_counts"][name] != counts.get(name, 0):
            msg = f"re-aggregation changed the count for {name}"
            raise CorrectionError(msg)
    if "box_source" in recorded:
        rebuilt = {"box_source": recorded["box_source"], **rebuilt}
    return rebuilt


def correct_association(association: Mapping[str, Any]) -> dict[str, Any]:
    """Rebuild the whole association section under the corrected taxonomy.

    Args:
        association: The section phase 10B wrote.

    Returns:
        The corrected section.
    """
    corrected: dict[str, Any] = {
        "containment_floor": association["containment_floor"],
        "deterministic": association["deterministic"],
        "frozen_categories": list(ASSOCIATION_CATEGORIES),
        "frozen_categories_unchanged": True,
        "fifth_peer_category_added": False,
    }
    for key in ("geometry_isolating", "pipeline_level"):
        corrected[key] = rebuild_block(association[key])
    for group in ("per_relationship", "per_class"):
        corrected[group] = {
            name: rebuild_block(block) for name, block in sorted(association[group].items())
        }

    exceptions = (
        corrected["geometry_isolating"]["taxonomy_exceptions"]
        + corrected["pipeline_level"]["taxonomy_exceptions"]
    )
    corrected["association_taxonomy_status"] = (
        TAXONOMY_NON_EXHAUSTIVE if exceptions else TAXONOMY_EXHAUSTIVE
    )
    corrected["taxonomy_exception_type"] = TAXONOMY_EXCEPTION
    corrected["taxonomy_exception_reading"] = ASSOCIATION_DISAGREEMENT
    corrected["taxonomy_exception_note"] = (
        "Phase 10A froze four categories on the implicit assumption that a rule either "
        "associates or it does not. Two rules can both associate and pick different people, "
        "and none of the four is true of that. It is recorded as an exception to the "
        "taxonomy's coverage - counted on its own, excluded from the classified denominator - "
        "rather than as a fifth peer category, because adding a category after seeing data is "
        "what a frozen taxonomy exists to prevent. The phase 10A protocol is historical and "
        "was not modified. This is a protocol-design limitation found during execution; it "
        "invalidates no canonical metric, no continuous spatial metric, no raw association "
        "decision and no model prediction."
    )
    corrected["not_an_error"] = (
        "A different-person outcome is an ASSOCIATION_RULE_DISAGREEMENT, not an association "
        "error. The project holds no person-PPE association ground truth, so neither rule's "
        "answer can be called wrong."
    )
    corrected["geometry_isolating_versus_pipeline_level"] = (
        "The geometry-isolating reading holds the model and its instances fixed and varies "
        "only the shape representation, so its exceptions are attributable to mask-versus-box "
        "geometry. The pipeline-level reading compares the frozen detector's outputs with the "
        "frozen segmenter's, so its exceptions reflect BOTH representational geometry AND the "
        "fact that different models produced different instances. The pipeline-level "
        "exceptions must not be attributed solely to geometry."
    )
    corrected["row_level_person_identity_recorded"] = False
    corrected["row_level_person_identity_note"] = (
        "Phase 10B's row-level records preserve each disagreement's image, class, score and "
        "both containment values, but not which person index each rule selected. That is "
        "enough to locate and count every exception, which is all this correction needed, and "
        "it is why no model was re-run. The runner now records both selected person indices so "
        "a future execution can reproduce an exception down to the person."
    )
    return corrected


def render_report(paths: ProjectPaths, box: Mapping[str, Any], spatial: Mapping[str, Any]) -> str:
    """Re-render the phase 10B report from the corrected artifacts.

    The renderer lives in the phase 10B runner, so it is loaded from that file
    rather than duplicated - a second copy would be free to drift from the one
    a fresh run uses. Loading the module runs no inference: the runner imports
    ultralytics and torch only inside the functions that need them, and this
    calls neither.

    Args:
        paths: Project layout.
        box: The corrected box comparison.
        spatial: The corrected spatial comparison.

    Returns:
        The rendered report.

    Raises:
        CorrectionError: If the renderer cannot be loaded.
    """
    source = paths.root / "scripts" / "compare_detector_segmenter.py"
    spec = importlib.util.spec_from_file_location("_phase_10b_runner", source)
    if spec is None or spec.loader is None:
        msg = f"cannot load the phase 10B renderer from {source.name}"
        raise CorrectionError(msg)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    return runner.build_report(box, spatial, commit=git_commit(paths.root))


def main(argv: list[str] | None = None) -> int:
    """Apply the association-taxonomy correction to the committed results.

    Args:
        argv: Command-line arguments.

    Returns:
        Process exit status.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verify-only", action="store_true", help="derive the correction without writing"
    )
    args = parser.parse_args(argv)

    paths = ProjectPaths.from_root()
    try:
        historical = {name: sha256_file(paths.root / name) for name in HISTORICAL}
        spatial = read_json(paths.reports / SPATIAL_JSON)
        box = read_json(paths.reports / BOX_JSON)
        protocol = read_json(paths.reports / PROTOCOL_JSON)
        if protocol["status"] != "FROZEN_NOT_EXECUTED":
            msg = "the phase 10A protocol manifest is not in its frozen state"
            raise CorrectionError(msg)

        corrected = correct_association(spatial["association"])

        support = read_json(paths.reports / "segmentation_S1_canonical_evaluation.json")["support"]
        admitted = [
            name
            for name, row in sorted(support.items())
            if row["images"] >= SUPPORT_MIN_POSITIVE_IMAGES
            and row["instances"] >= SUPPORT_MIN_INSTANCES
        ]
        sensitivity = supported_class_sensitivity(box["per_class"], admitted)
    except CorrectionError as exc:
        print(f"{BLOCKED}: {exc}", file=sys.stderr)
        return 2

    geometry = corrected["geometry_isolating"]
    pipeline = corrected["pipeline_level"]
    print(
        f"geometry   {geometry['classified_relationships']} classified + "
        f"{geometry['taxonomy_exceptions']} exception(s) = "
        f"{geometry['total_relationships']}  coverage {geometry['taxonomy_coverage']}"
    )
    print(
        f"pipeline   {pipeline['classified_relationships']} classified + "
        f"{pipeline['taxonomy_exceptions']} exception(s) = "
        f"{pipeline['total_relationships']}  coverage {pipeline['taxonomy_coverage']}"
    )
    print(f"status     {corrected['association_taxonomy_status']}")
    print(
        f"support    D2 {sensitivity['D2_supported_macro']}  S1 "
        f"{sensitivity['S1_supported_macro']}  delta {sensitivity['delta']}  "
        f"over {sensitivity['admitted_classes']}"
    )
    print("models     0 loaded, 0 executed (arithmetic over recorded counts only)")

    if args.verify_only:
        print("VERIFIED: the correction derives from recorded counts. Nothing written.")
        return 0

    spatial["association"] = corrected
    spatial["spatial_comparison_sha256"] = result_fingerprint(
        {
            "protocol_fingerprint": spatial["protocol_fingerprint"],
            "detector_checkpoint": spatial["detector"]["checkpoint_sha256"],
            "segmenter_checkpoint": spatial["segmenter"]["checkpoint_sha256"],
            "membership": spatial["population"]["membership_sha256"],
            "inference": spatial["inference"],
            "prediction_counts": spatial["prediction_counts"],
            "spatial_features": spatial["spatial_features"],
            "box_proxies": spatial["box_proxies"],
            "association": spatial["association"],
            "geometry_disagreement": spatial["geometry_disagreement"],
        }
    )
    box["supported_class_sensitivity"] = sensitivity
    box["rare_class"]["drives_the_aggregate_delta"] = True
    box["localization_conclusion"] = (
        "S1 retains broadly similar localisation capability to D2 while additionally producing "
        f"masks. The positive all-class delta of "
        f"{box['metrics']['CANONICAL_BOX_MAP50_95']['delta']:+f} is driven by "
        f"{RARE_CLASS}, which is already frozen as {RARE_CLASS_STATUS}, and should not be read "
        "as robust evidence that S1 is the superior object localiser: over the adequately "
        f"supported classes the same comparison gives {sensitivity['delta']:+f}. Neither "
        "reading selects a model, and neither frozen model changes."
    )

    problems = validate_spatial_comparison(
        spatial,
        protocol_fingerprint=spatial["protocol_fingerprint"],
        membership_sha256=spatial["population"]["membership_sha256"],
    ) + validate_box_comparison(
        box,
        protocol_fingerprint=box["protocol_fingerprint"],
        detector_sha256=box["detector"]["checkpoint_sha256"],
        segmenter_sha256=box["segmenter"]["checkpoint_sha256"],
        membership_sha256=box["population"]["membership_sha256"],
    )
    if problems:
        print(f"{BLOCKED}: the corrected artifacts do not validate:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 2

    findings = scan_for_sensitive(json.dumps(spatial)) + scan_for_sensitive(json.dumps(box))
    if findings:
        print(f"{BLOCKED}: sensitive content: {findings}", file=sys.stderr)
        return 2

    try:
        report = render_report(paths, box, spatial)
    except CorrectionError as exc:
        print(f"{BLOCKED}: {exc}", file=sys.stderr)
        return 2
    if scan_for_sensitive(report):
        print(f"{BLOCKED}: sensitive content in the report", file=sys.stderr)
        return 2

    spatial_sha = write_json(paths.reports / SPATIAL_JSON, spatial)
    box_sha = write_json(paths.reports / BOX_JSON, box)
    (paths.reports / REPORT_MD).write_text(report, encoding="utf-8", newline=NEWLINE)

    after = {name: sha256_file(paths.root / name) for name in HISTORICAL}
    changed = sorted(name for name in historical if historical[name] != after[name])
    if changed:
        print(f"{BLOCKED}: historical artifacts changed: {changed}", file=sys.stderr)
        return 2

    record = ProvenanceRecord.create(
        name="association_taxonomy_correction",
        phase=10,
        config={"protocol_fingerprint": spatial["protocol_fingerprint"]},
        details={
            "phase": "10B",
            "classification": CORRECTED,
            "association_taxonomy_status": corrected["association_taxonomy_status"],
            "taxonomy_exception_type": TAXONOMY_EXCEPTION,
            "geometry_isolating": {
                "classified": geometry["classified_relationships"],
                "exceptions": geometry["taxonomy_exceptions"],
                "coverage": geometry["taxonomy_coverage"],
            },
            "pipeline_level": {
                "classified": pipeline["classified_relationships"],
                "exceptions": pipeline["taxonomy_exceptions"],
                "coverage": pipeline["taxonomy_coverage"],
            },
            "supported_class_sensitivity": sensitivity,
            "frozen_categories_unchanged": True,
            "fifth_peer_category_added": False,
            "phase_10a_protocol_modified": False,
            "models_loaded": 0,
            "models_executed": 0,
            "inference_run": False,
            "spatial_comparison_sha256": spatial["spatial_comparison_sha256"],
            "historical_artifacts_unchanged": True,
        },
        repo_root=paths.root,
    )
    record.write_json(paths.reports / "association_taxonomy_correction.provenance.json")

    print(CORRECTED)
    print(f"spatial    reports/{SPATIAL_JSON}  sha256 {spatial_sha}")
    print(f"box        reports/{BOX_JSON}  sha256 {box_sha}")
    print(f"report     reports/{REPORT_MD}  re-rendered from the corrected artifacts")
    print(f"fingerprint {spatial['spatial_comparison_sha256']}")
    print(f"commit      {git_commit(paths.root)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
