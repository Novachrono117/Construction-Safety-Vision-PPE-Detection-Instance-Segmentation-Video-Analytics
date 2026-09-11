"""Freeze the one-shot final holdout evaluation protocol - and execute none of it.

Phase 11A predeclares everything phase 11B will do to the test split, then
stops. **It reads no holdout data.** A test asserts that this file imports
neither torch nor ultralytics, never enumerates a test identifier, and never
writes the environment gate.

Four things are deliberate.

**The protocol is frozen before the first holdout read, not after.** Every
threshold, evaluator, matching rule, confusion-matrix setting and qualitative
ranking is written down while no holdout number exists and none can. That is
what makes them a protocol rather than choices made with a result in view.

**Aggregate population facts only.** The holdout's image and annotation counts
were frozen in phase 5C.2, long before any model existed, and are read from the
committed manifest's aggregate count fields. The `test` membership section is
never opened, so no identifier enters this phase.

**Neither gate is touched.** The script refuses to run if the environment
variable is set, because a protocol-only phase has no business executing with
the holdout unlocked. Nothing here can set it; only a person can.

**Historical artifacts are read, digested and verified unchanged.** Both model
freezes, the phase 10A protocol, the 10B/10C results, the 10D synthesis, the
split manifest and the canonical fingerprints are inputs. If any byte moves, the
phase stops rather than writing.

Usage:
    uv run python scripts/freeze_final_holdout_evaluation.py
    uv run python scripts/freeze_final_holdout_evaluation.py --verify-only
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from construction_safety_vision.data.canonical import scan_for_sensitive
from construction_safety_vision.final_holdout_evaluation import (
    AP_CONF,
    BLOCKED,
    CLASS_MAP_SHA256,
    CLASSES,
    CODE_GATE,
    CONFUSION_MATRIX_CONF,
    CONFUSION_MATRIX_IOU,
    ENVIRONMENT_GATE,
    EXAMPLES_PER_CATEGORY,
    FAILURE_STATES,
    HOLDOUT_SHA256,
    HOLDOUT_STATUS,
    INVALID_HOLDOUT_PROTOCOL,
    LEDGER_STATES,
    MAX_DET,
    NMS_IOU,
    OPERATIONAL_CONF,
    PHASE,
    PRECISION,
    PREDICTION_FINGERPRINT_FIELDS,
    PROTOCOL_FROZEN,
    QUALITATIVE_CATEGORIES,
    RESULT_FINGERPRINT_FIELDS,
    RUNNER,
    SEGMENTATION_FAILURE_CATEGORIES,
    SPLIT_ASSIGNMENT_SHA256,
    TEST_ANNOTATIONS,
    TEST_IMAGES,
    HoldoutProtocolError,
    holdout_is_locked,
    load_protocol,
    phase_11b_steps,
    protocol_fingerprint,
    validate_manifest,
    validate_protocol,
)
from construction_safety_vision.paths import ProjectPaths
from construction_safety_vision.provenance import ProvenanceRecord, git_commit, sha256_file

PROTOCOL_JSON = "final_holdout_evaluation_protocol.json"
PROTOCOL_MD = "final_holdout_evaluation_protocol.md"
CONFIG = "configs/final_holdout_evaluation.yaml"

NEWLINE = "\n"
"""Written explicitly so artifacts do not pick up platform line endings."""

HISTORICAL: tuple[str, ...] = (
    "configs/detector_segmenter_comparison.yaml",
    "configs/segmentation_canonical_evaluation.yaml",
    "configs/segmentation_mask_iou_evaluation.yaml",
    "configs/split_freeze.yaml",
    "reports/final_detector_manifest.json",
    "reports/final_detector.provenance.json",
    "reports/final_segmenter_manifest.json",
    "reports/final_segmenter.provenance.json",
    "reports/detector_segmenter_comparison_protocol.json",
    "reports/detector_segmenter_comparison_protocol.md",
    "reports/detector_segmenter_box_comparison.json",
    "reports/detector_segmenter_spatial_comparison.json",
    "reports/detector_segmenter_validation_comparison.md",
    "reports/detector_segmenter_latency_comparison.json",
    "reports/detector_segmenter_memory_comparison.json",
    "reports/detector_segmenter_latency_report.md",
    "reports/detector_segmenter_scientific_synthesis.json",
    "reports/detector_segmenter_scientific_synthesis.md",
    "reports/detector_segmenter_tradeoff.csv",
    "reports/split_manifest.json",
    "reports/task_dataset_manifest.json",
    "reports/canonical_annotation_manifest.json",
    "reports/canonical_modeling_manifest.json",
    "reports/segmentation_S1_canonical_evaluation.json",
    "reports/segmentation_S1_mask_iou.json",
    "reports/detection_D2_manifest.json",
)
"""Read, digested and verified byte-identical. Never regenerated here."""

AGGREGATE_FIELDS: tuple[str, ...] = (
    "actual_image_counts",
    "actual_annotation_counts",
    "actual_group_counts",
    "actual_negative_image_counts",
)
"""The only fields read from the split manifest. Never its membership sections."""


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


def read_aggregate_population(root: Path) -> dict[str, int]:
    """Read the holdout's aggregate counts without touching its membership.

    Only the manifest's aggregate count fields are opened. The ``test`` section
    that carries per-image membership is never read, so no holdout identifier
    enters this phase.

    Args:
        root: Repository root.

    Returns:
        The aggregate counts for the protected split.

    Raises:
        HoldoutProtocolError: If the recorded counts disagree with the frozen
            figures this protocol declares.
    """
    manifest = json.loads((root / "reports" / "split_manifest.json").read_text(encoding="utf-8"))
    counts = {field: int(manifest[field]["test"]) for field in AGGREGATE_FIELDS}
    aggregate = {
        "images": counts["actual_image_counts"],
        "annotations": counts["actual_annotation_counts"],
        "groups": counts["actual_group_counts"],
        "negative_images": counts["actual_negative_image_counts"],
    }
    if aggregate["images"] != TEST_IMAGES or aggregate["annotations"] != TEST_ANNOTATIONS:
        msg = (
            "the frozen split manifest disagrees with the declared aggregate holdout "
            f"population: {aggregate['images']} images / {aggregate['annotations']} "
            f"annotations against {TEST_IMAGES} / {TEST_ANNOTATIONS}"
        )
        raise HoldoutProtocolError(msg)
    if manifest["split_assignment_sha256"] != SPLIT_ASSIGNMENT_SHA256:
        msg = "the frozen split assignment fingerprint has changed"
        raise HoldoutProtocolError(msg)
    if manifest["holdout_sha256"] != HOLDOUT_SHA256:
        msg = "the frozen holdout fingerprint has changed"
        raise HoldoutProtocolError(msg)
    return aggregate


def build_manifest(
    raw: Mapping[str, Any], *, fingerprint: str, aggregate: Mapping[str, int]
) -> dict[str, Any]:
    """Assemble the committed protocol manifest.

    Args:
        raw: The parsed protocol document.
        fingerprint: The protocol fingerprint.
        aggregate: The holdout's aggregate counts.

    Returns:
        The manifest, carrying no result and no holdout identifier.
    """
    detector = raw["detector"]
    segmenter = raw["segmenter"]
    return {
        "schema_version": 1,
        "phase": PHASE,
        "executes_in_phase": raw["executes_in_phase"],
        "protocol": raw["protocol"],
        "status": raw["status"],
        "classification": PROTOCOL_FROZEN,
        "protocol_config": CONFIG,
        "protocol_fingerprint": fingerprint,
        "objective": raw["objective"],
        "detector": {
            "experiment": detector["experiment"],
            "model": detector["model"],
            "imgsz": detector["imgsz"],
            "checkpoint_sha256": detector["checkpoint_sha256"],
            "checkpoint_bytes": detector["checkpoint_bytes"],
            "identity_fingerprint": detector["identity_fingerprint"],
            "manifest": detector["manifest"],
            "frozen_in_phase": detector["frozen_in_phase"],
            "accessor": detector["accessor"],
            "modified_in_this_phase": False,
            "executed_in_this_phase": False,
        },
        "segmenter": {
            "experiment": segmenter["experiment"],
            "model": segmenter["model"],
            "imgsz": segmenter["imgsz"],
            "mask_ratio": segmenter["mask_ratio"],
            "overlap_mask": segmenter["overlap_mask"],
            "checkpoint_sha256": segmenter["checkpoint_sha256"],
            "checkpoint_bytes": segmenter["checkpoint_bytes"],
            "final_segmenter_sha256": segmenter["final_segmenter_sha256"],
            "identity_fingerprint": segmenter["identity_fingerprint"],
            "manifest": segmenter["manifest"],
            "frozen_in_phase": segmenter["frozen_in_phase"],
            "accessor": segmenter["accessor"],
            "modified_in_this_phase": False,
            "executed_in_this_phase": False,
        },
        "one_shot": raw["one_shot"],
        "authorization": {
            **raw["authorization"],
            "environment_gate_variable": ENVIRONMENT_GATE,
            "code_gate": CODE_GATE,
            "environment_gate_currently_set": False,
        },
        "test_population": {
            **{k: v for k, v in raw["test_population"].items() if k != "technical_validation"},
            "technical_validation": raw["test_population"]["technical_validation"],
            "verified_against_frozen_manifest": True,
            "aggregate_source_fields": list(AGGREGATE_FIELDS),
            "membership_section_read": False,
            "aggregate_counts": dict(aggregate),
        },
        "classes": list(CLASSES),
        "support_rule": raw["support_rule"],
        "detector_inference": raw["detector_inference"],
        "segmenter_inference": raw["segmenter_inference"],
        "canonical_bbox_evaluator": raw["canonical_bbox_evaluator"],
        "canonical_segm_evaluator": raw["canonical_segm_evaluator"],
        "detector_metrics": raw["detector_metrics"],
        "segmenter_metrics": raw["segmenter_metrics"],
        "precision_recall_semantics": raw["precision_recall_semantics"],
        "direct_iou": raw["direct_iou"],
        "confusion_matrix": raw["confusion_matrix"],
        "object_level_matching": {
            **raw["object_level_matching"],
            "segmentation_failure_category_order": list(SEGMENTATION_FAILURE_CATEGORIES),
        },
        "qualitative_selection": raw["qualitative_selection"],
        "spatial_analysis_on_test": raw["spatial_analysis_on_test"],
        "computational_cost": raw["computational_cost"],
        "prediction_persistence": raw["prediction_persistence"],
        "fingerprints": {
            **raw["fingerprints"],
            "prediction_fields": list(PREDICTION_FINGERPRINT_FIELDS),
            "result_fields": list(RESULT_FINGERPRINT_FIELDS),
        },
        "one_shot_ledger": raw["one_shot_ledger"],
        "failure_policy": raw["failure_policy"],
        "validation_versus_test": raw["validation_versus_test"],
        "reports": raw["reports"],
        "figures": raw["figures"],
        "final_comparison": raw["final_comparison"],
        "prohibited": raw["prohibited"],
        "phase_11b_execution_order": list(phase_11b_steps()),
        "runner": RUNNER,
        "class_map_sha256": CLASS_MAP_SHA256,
        "split_assignment_sha256": SPLIT_ASSIGNMENT_SHA256,
        "holdout_sha256": HOLDOUT_SHA256,
        "holdout_unlocked": False,
        "results_present": False,
        "models_executed": 0,
        "models_trained": 0,
        "test_predictions_produced": 0,
        "test_metrics_computed": 0,
        "test_images_read": 0,
        "test_annotations_read": 0,
        "test_identifiers_recorded": 0,
        "latency_measurements_taken": 0,
        "thresholds_tuned": 0,
        "figures_generated": 0,
        "historical_artifacts_unchanged": True,
        "test": {
            "status": HOLDOUT_STATUS,
            "images_read": 0,
            "annotations_read": 0,
            "identifiers_recorded": 0,
            "predictions": 0,
            "metrics": 0,
            "reason": (
                "Phase 11A predeclared the final evaluation and executed none of it. The "
                "holdout was not unlocked, materialised, adapted, enumerated, predicted on or "
                "inspected; only the aggregate counts frozen in phase 5C.2 were read, from the "
                "manifest's aggregate count fields rather than its membership sections."
            ),
        },
        "test_policy": raw["test_policy"],
    }


def _table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    """Render a Markdown table.

    Args:
        header: Column titles.
        rows: Row cells, already stringified.

    Returns:
        The table's lines.
    """
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return lines


def render_report(manifest: Mapping[str, Any]) -> str:
    """Render the human-readable protocol document.

    Args:
        manifest: The assembled protocol manifest.

    Returns:
        The Markdown document.
    """
    from construction_safety_vision.final_holdout_report import render

    return render(manifest)


def main(argv: Sequence[str] | None = None) -> int:
    """Freeze the protocol, or verify that it derives.

    Args:
        argv: Command-line arguments.

    Returns:
        Process exit status.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="derive and validate the protocol without writing anything",
    )
    args = parser.parse_args(argv)

    paths = ProjectPaths.from_root()

    # A protocol-only phase has no business running with the holdout unlocked.
    if not holdout_is_locked():
        print(
            f"{BLOCKED}: {ENVIRONMENT_GATE} is set. Phase 11A freezes a protocol and reads no "
            "holdout data; it must not run with the gate open.",
            file=sys.stderr,
        )
        return 2
    if os.environ.get(ENVIRONMENT_GATE) is not None:
        print(f"{BLOCKED}: {ENVIRONMENT_GATE} is present in the environment.", file=sys.stderr)
        return 2

    try:
        before = {name: sha256_file(paths.root / name) for name in HISTORICAL}
        protocol = load_protocol(paths.root / CONFIG)
        aggregate = read_aggregate_population(paths.root)
        fingerprint = protocol_fingerprint(protocol.raw)
        manifest = build_manifest(protocol.raw, fingerprint=fingerprint, aggregate=aggregate)
    except (HoldoutProtocolError, KeyError, OSError) as exc:
        print(f"{INVALID_HOLDOUT_PROTOCOL}: {exc}", file=sys.stderr)
        return 2

    problems = validate_protocol(protocol.raw) + validate_manifest(manifest)
    if problems:
        print(f"{INVALID_HOLDOUT_PROTOCOL}: the protocol does not validate:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 2

    # Recompute from a fresh parse, so the fingerprint is shown to be a
    # deterministic function of the committed file rather than of this process.
    if protocol_fingerprint(load_protocol(paths.root / CONFIG).raw) != fingerprint:
        print(f"{INVALID_HOLDOUT_PROTOCOL}: the fingerprint is not deterministic", file=sys.stderr)
        return 2

    report = render_report(manifest)
    findings = scan_for_sensitive(json.dumps(manifest)) + scan_for_sensitive(report)
    if findings:
        print(f"{BLOCKED}: sensitive content: {findings}", file=sys.stderr)
        return 2

    print(f"detector    {manifest['detector']['experiment']} @ {manifest['detector']['imgsz']}")
    print(f"segmenter   {manifest['segmenter']['experiment']} @ {manifest['segmenter']['imgsz']}")
    print(f"population  {aggregate['images']} images / {aggregate['annotations']} annotations")
    print(f"detector AP conf {AP_CONF}  NMS {NMS_IOU}  max_det {MAX_DET}  {PRECISION}")
    print(f"direct IoU  operational conf {OPERATIONAL_CONF} (phase 8C protocol, unchanged)")
    print(f"confusion   conf {CONFUSION_MATRIX_CONF}  IoU {CONFUSION_MATRIX_IOU}")
    print(
        f"qualitative {len(QUALITATIVE_CATEGORIES)} categories x "
        f"{EXAMPLES_PER_CATEGORY} deterministic"
    )
    print(f"ledger      {len(LEDGER_STATES)} states, {len(FAILURE_STATES)} failure states")
    print(f"holdout     {manifest['test']['status']}  (gate {ENVIRONMENT_GATE} unset)")
    print("models      0 loaded, 0 executed (protocol only)")

    if args.verify_only:
        print("VERIFIED: the protocol derives and validates. Nothing written.")
        return 0

    manifest_sha = write_json(paths.reports / PROTOCOL_JSON, manifest)
    (paths.reports / PROTOCOL_MD).write_text(report, encoding="utf-8", newline=NEWLINE)

    after = {name: sha256_file(paths.root / name) for name in HISTORICAL}
    changed = sorted(name for name in before if before[name] != after[name])
    if changed:
        print(f"{BLOCKED}: historical artifacts changed: {changed}", file=sys.stderr)
        return 2

    record = ProvenanceRecord.create(
        name="final_holdout_evaluation_protocol",
        phase=11,
        config={"protocol_fingerprint": fingerprint, "protocol_config": CONFIG},
        details={
            "phase": PHASE,
            "classification": PROTOCOL_FROZEN,
            "status": manifest["status"],
            "protocol_fingerprint": fingerprint,
            "detector": manifest["detector"]["checkpoint_sha256"],
            "segmenter": manifest["segmenter"]["checkpoint_sha256"],
            "holdout": HOLDOUT_STATUS,
            "holdout_unlocked": False,
            "models_executed": 0,
            "test_predictions_produced": 0,
            "test_metrics_computed": 0,
            "test_images_read": 0,
            "test_identifiers_recorded": 0,
            "historical_artifacts_unchanged": True,
            "executes_in_phase": manifest["executes_in_phase"],
        },
        repo_root=paths.root,
    )
    record.write_json(paths.reports / "final_holdout_evaluation.provenance.json")

    print(PROTOCOL_FROZEN)
    print(f"manifest    reports/{PROTOCOL_JSON}  sha256 {manifest_sha}")
    print(f"report      reports/{PROTOCOL_MD}")
    print(f"fingerprint {fingerprint}")
    print(f"commit      {git_commit(paths.root)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
