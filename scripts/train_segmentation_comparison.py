"""Run the S1 controlled overlap-mask segmentation experiment once.

Phase 8F. It executes exactly one full S1 training run under the protocol frozen
in phase 8E, one authoritative native validation of the checkpoint the
framework's own rule selected, one canonical COCO evaluation under the frozen
common evaluator, and one direct instance-mask IoU diagnostic under the
unchanged phase 8C protocol. It selects no final segmenter, tunes nothing,
retrains nothing, and never touches the holdout.

Five things here are less obvious than they look.

**S1 has no protocol of its own.** Its framework arguments are resolved from
S0's, in code, with the single override the phase 8E comparison protocol
declares. A drift in S0's protocol therefore surfaces as a changed S1 argument
rather than being masked by a second copy of the same values.

**The native validation is given ``overlap_mask: false`` explicitly.** Read from
the installed source: ``Model._reset_ckpt_args`` keeps only ``imgsz``, ``data``,
``task`` and ``single_cls`` from a checkpoint, so the flag is *not* carried over
from training, and the framework default is ``True``. Validating S1 without
passing it would score S1's predictions against S0's target - the exact
confusion this phase exists to avoid.

**The decision is made against canonical COCO masks, never the framework's own
mask AP.** ``overlap_mask`` reshapes both the training target and the native
validation ground truth, so S0's and S1's native numbers are measured against
different targets. They are reported in full and marked
``NOT_PRIMARY_FOR_CROSS_TARGET_SELECTION``.

**S0 is read, never executed.** Its checkpoint bytes are verified and its
committed metrics are quoted. Nothing about S0 is recomputed, and no phase 8A-8E
artifact is regenerated - all are digested before and after and required to be
byte-identical.

Requires:

* ``configs/segmentation_baseline.yaml``                    phase 8B
* ``configs/segmentation_mask_iou_evaluation.yaml``         phase 8C
* ``configs/segmentation_canonical_evaluation.yaml``        phase 8E
* ``configs/segmentation_comparison.yaml``                  phase 8E
* ``reports/segmentation_S1_protocol_manifest.json``        phase 8E
* ``reports/segmentation_S0_canonical_evaluation.json``     phase 8E
* ``data/processed/adapters/yolo_segmentation_audit/``      phase 8A
* ``data/processed/canonical/annotations/``                 phase 5D

Writes:
    reports/segmentation_S1_result_manifest.json
    reports/segmentation_S1_canonical_evaluation.json
    reports/segmentation_S1_mask_iou.json
    reports/segmentation_S1_report.md
    reports/segmentation_experiment_results.json
    reports/segmentation_S1.provenance.json
    reports/segmentation_S1_prerun.provenance.json
    reports/figures/segmentation_S1_*.png
    artifacts/segmentation/S1/                              (git-ignored)
    data/processed/adapters/yolo_segmentation_s1_runtime/   (git-ignored)

Usage:
    uv run python scripts/train_segmentation_comparison.py
    uv run python scripts/train_segmentation_comparison.py --verify-only
    uv run python scripts/train_segmentation_comparison.py --validate-only
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from construction_safety_vision.canonical_evaluation import (
    ALL_CLASS_METRIC,
    ALL_CLASS_METRIC_50,
    NATIVE_METRIC_STATUS,
    PRIMARY_METRIC,
    CanonicalEvaluationError,
    classify_delta,
    evaluate_canonical,
    load_canonical_evaluation_config,
    prediction_record,
)
from construction_safety_vision.canonical_evaluation import (
    supported_macro as canonical_supported_macro,
)
from construction_safety_vision.canonical_evaluation import (
    validation_support as canonical_validation_support,
)
from construction_safety_vision.config import ConfigError
from construction_safety_vision.data.canonical import scan_for_sensitive
from construction_safety_vision.detection_freeze import FinalDetectorError, load_final_detector
from construction_safety_vision.detection_run import (
    RunError,
    capture_framework_log,
    ensure_pretrained_weights,
    optimizer_evidence_from_log,
    release_framework_log,
    runtime_facts,
)
from construction_safety_vision.mask_iou_evaluation import (
    GROUND_TRUTH_SOURCE,
    GT_IOU50_COVERAGE,
    GT_IOU75_COVERAGE,
    GT_MATCH_COVERAGE,
    GT_NORMALIZED_MASK_IOU,
    MATCHED_MASK_IOU_MEAN,
    MATCHING_ALGORITHM,
    MATCHING_SCOPE,
    UNMATCHED_GT_POLICY,
    ZERO_OVERLAP_POLICY,
    MaskInstance,
    accumulate,
    load_mask_iou_config,
    match_image,
)
from construction_safety_vision.mask_iou_evaluation import (
    PROTOCOL_NAME as DIRECT_IOU_PROTOCOL,
)
from construction_safety_vision.paths import ProjectPaths, long_path
from construction_safety_vision.provenance import ProvenanceRecord, git_commit, sha256_file
from construction_safety_vision.segmentation_comparison import (
    CHECKPOINT_POLICY,
    ONE_VARIABLE_FIELD,
    REFERENCE,
    ComparisonConfigError,
    load_comparison_config,
)
from construction_safety_vision.segmentation_experiment import (
    ADAPTER_ROLE,
    CANONICAL_GROUND_TRUTH,
    CHECKPOINT_SELECTION_SEMANTICS,
    SUPPORT_RULE,
    SegmentationBaselineConfig,
    SegmentationExperimentConfigError,
    load_segmentation_baseline_config,
)
from construction_safety_vision.segmentation_run import (
    FITNESS_TOLERANCE,
    METRIC_PRECISION,
    NOT_EXPOSED,
    RUN_ROOT,
    SegmentationRunError,
    adapter_counts,
    adapter_fingerprints,
    best_epoch_by_native_fitness,
    build_runtime_view,
    checkpoint_record,
    copy_metric_figures,
    model_complexity,
    native_fitness_curve,
    read_adapter_labels,
    read_epoch_history,
    relativise,
    resolved_training_arguments,
    rounded,
    summarise_curves,
    training_duration,
)
from construction_safety_vision.segmentation_s1 import (
    BLOCKED,
    CANONICAL_EVALUATION_FAILED,
    CANONICAL_PRIMARY_METRIC,
    COMPLETE,
    DESCRIPTIVE_NATIVE_TARGET_METRIC,
    DIRECT_IOU_EVALUATION_FAILED,
    EXPERIMENT_ID,
    FINAL_SEGMENTER,
    HOLDOUT_STATUS,
    MANIFEST_SCHEMA_VERSION,
    MEMORY_CONSTRAINT_REVIEW_REQUIRED,
    NATIVE_TARGET_METRIC,
    PHASE,
    PROTOCOL_VIOLATION,
    RARE_CLASS_STATUS,
    RUN_NAME,
    RUNTIME_VIEW_ROOT,
    S1_PROTOCOL_INVALID,
    TRAINING_FAILED,
    TREATMENT_SEMANTICS,
    SegmentationS1Error,
    aggregate_composition,
    cross_metric_direction,
    experiment_fingerprint,
    per_class_comparison,
    person_diagnostic,
    resolve_candidate_arguments,
    validate_canonical_evaluation,
    validate_comparison,
    validate_direct_iou,
    validate_s1_result_manifest,
    verify_inherited_protocol,
)
from construction_safety_vision.splits import HOLDOUT_UNLOCK_ENV_VAR, holdout_unlocked

BASELINE_YAML = "segmentation_baseline.yaml"
MASK_IOU_YAML = "segmentation_mask_iou_evaluation.yaml"
CANONICAL_YAML = "segmentation_canonical_evaluation.yaml"
COMPARISON_YAML = "segmentation_comparison.yaml"

APPROVAL_JSON = "segmentation_adapter_approval.json"
AUDIT_MANIFEST_JSON = "segmentation_adapter_audit_manifest.json"
S1_PROTOCOL_JSON = "segmentation_S1_protocol_manifest.json"
S0_RESULT_JSON = "segmentation_S0_result_manifest.json"
S0_MASK_IOU_JSON = "segmentation_S0_mask_iou.json"
S0_CANONICAL_JSON = "segmentation_S0_canonical_evaluation.json"
POLICY_JSON = "segmentation_comparison_policy.json"

RESULT_MANIFEST_JSON = "segmentation_S1_result_manifest.json"
CANONICAL_JSON = "segmentation_S1_canonical_evaluation.json"
MASK_IOU_JSON = "segmentation_S1_mask_iou.json"
REPORT_MD = "segmentation_S1_report.md"
RESULTS_JSON = "segmentation_experiment_results.json"
PROVENANCE_JSON = "segmentation_S1.provenance.json"
PRERUN_PROVENANCE_JSON = "segmentation_S1_prerun.provenance.json"

EXPECTED_S0_CHECKPOINT = "d7b512b95fafdc8658fd802459d2c87d9ad3f11f922162a774dacf8ca75b87a3"
EXPECTED_PRETRAINED = "55ed65c56c91713d23e8402371c6c49a6fd84f257f7dce452e8d70e41dcbe152"
EXPECTED_PRETRAINED_BYTES = 6182636
EXPECTED_COMPARISON_FINGERPRINT = "a74609c1c58371f91de2b1699e695fcbdf3fed6ac9a68237c190cfb1d1393f16"
EXPECTED_CANONICAL_FINGERPRINT = "282eb0ec125c47687340325266a2ea397bc0f85e239290a26a5c1ec8d8721e64"

ADAPTER_FINGERPRINT_MISMATCH = "ADAPTER_FINGERPRINT_MISMATCH"

SUPPORTED = "COMPARISON_SUPPORTED"
NATIVE_SUPPORTED_MACRO_METRIC = "supported_macro_mask_map50_95"

HOLDOUT_REASON = (
    "Phase 8F trained S1 on the frozen train split, validated it on the frozen validation "
    "split, and ran the canonical COCO evaluation and the direct mask-IoU diagnostic on that "
    "same validation split. The holdout was not read, materialised, adapted, counted, "
    "predicted on, inspected or plotted; no holdout identifier, label, prediction or statistic "
    "exists in any artifact this phase wrote, and the dataset descriptor the framework read "
    "carries no holdout key."
)

HISTORICAL: tuple[str, ...] = (
    "reports/final_detector_manifest.json",
    "reports/detection_experiment_results.json",
    "reports/segmentation_adapter_audit_manifest.json",
    "reports/segmentation_adapter_fidelity_report.md",
    "reports/segmentation_adapter_fidelity.csv",
    "reports/segmentation_adapter_audit.provenance.json",
    "configs/segmentation_adapter_audit.yaml",
    "reports/segmentation_adapter_approval.json",
    "reports/segmentation_S0_manifest.json",
    "reports/segmentation_S0_protocol.md",
    "configs/segmentation_baseline.yaml",
    "configs/segmentation_mask_iou_evaluation.yaml",
    "reports/segmentation_S0_result_manifest.json",
    "reports/segmentation_S0_mask_iou.json",
    "reports/segmentation_S0_report.md",
    "reports/segmentation_S0_error_analysis.json",
    "reports/segmentation_S0_error_analysis.md",
    "reports/segmentation_S0_error_instances.csv",
    "configs/segmentation_canonical_evaluation.yaml",
    "configs/segmentation_comparison.yaml",
    "reports/segmentation_S0_canonical_evaluation.json",
    "reports/segmentation_canonical_comparison_reference.md",
    "reports/segmentation_comparison_policy.json",
    "reports/segmentation_comparison_policy.md",
    "reports/segmentation_S1_protocol.md",
    "reports/segmentation_S1_protocol_manifest.json",
)
"""Every artifact this phase must leave byte-identical."""

MASK_KEYS = {
    "precision": "metrics/precision(M)",
    "recall": "metrics/recall(M)",
    "mAP@0.50": "metrics/mAP50(M)",
    "mAP@0.50:0.95": "metrics/mAP50-95(M)",
}
BOX_KEYS = {
    "precision": "metrics/precision(B)",
    "recall": "metrics/recall(B)",
    "mAP@0.50": "metrics/mAP50(B)",
    "mAP@0.50:0.95": "metrics/mAP50-95(B)",
}


class ExperimentError(RuntimeError):
    """Raised when a required input is missing or an invariant fails."""


class AdapterMismatchError(ExperimentError):
    """Raised when the label bytes are not the approved ones."""


class ProtocolMismatchError(ExperimentError):
    """Raised when a frozen protocol or policy has moved."""


class MemoryConstraintError(ExperimentError):
    """Raised when the frozen batch size exhausts device memory."""


# --- helpers ------------------------------------------------------------------


def read_json(path: Path) -> dict[str, Any]:
    """Read a committed JSON artifact.

    Args:
        path: File to read.

    Returns:
        The parsed mapping.

    Raises:
        ExperimentError: If the file is missing or is not a JSON object.
    """
    if not path.is_file():
        msg = f"required input not found: {path.name}"
        raise ExperimentError(msg)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        msg = f"{path.name} is not valid JSON ({exc})"
        raise ExperimentError(msg) from exc
    if not isinstance(data, dict):
        msg = f"{path.name} must contain a JSON object"
        raise ExperimentError(msg)
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
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")
    return sha256_file(path)


# --- preflight ----------------------------------------------------------------


def verify_historical(paths: ProjectPaths) -> dict[str, str]:
    """Digest every artifact this phase must not change.

    Args:
        paths: Project layout.

    Returns:
        Digests keyed by repository-relative name.

    Raises:
        ExperimentError: If one is absent.
    """
    digests: dict[str, str] = {}
    for name in HISTORICAL:
        path = paths.root / name
        if not path.is_file():
            msg = f"historical artifact missing: {name}"
            raise ExperimentError(msg)
        digests[name] = sha256_file(path)
    return digests


def verify_detector(paths: ProjectPaths) -> dict[str, Any]:
    """Verify the frozen detector is untouched, then leave it alone.

    Args:
        paths: Project layout.

    Returns:
        What was verified.

    Raises:
        ExperimentError: If the manifest does not recompute or the bytes differ.
    """
    try:
        detector = load_final_detector(paths.reports)
    except FinalDetectorError as exc:
        raise ExperimentError(str(exc)) from exc
    if detector.recompute_fingerprint() != detector.fingerprint:
        msg = "the final detector manifest does not recompute"
        raise ExperimentError(msg)
    state = "ABSENT_ON_THIS_MACHINE"
    present = [path for path in detector.candidate_paths(paths.root) if path.is_file()]
    if present:
        if sha256_file(present[0]) != detector.checkpoint_sha256:
            msg = "the frozen detector checkpoint bytes changed"
            raise ExperimentError(msg)
        state = "PRESENT_VERIFIED"
    return {
        "selected_experiment": detector.selected_experiment,
        "model": detector.model,
        "imgsz": detector.imgsz,
        "checkpoint_sha256": detector.checkpoint_sha256,
        "checkpoint_state": state,
        "trained": False,
        "validated": False,
        "status": "UNCHANGED_DETECTION_BLOCK_CLOSED",
    }


def verify_policies(
    paths: ProjectPaths,
    baseline: SegmentationBaselineConfig,
    comparison: Any,
    canonical_protocol: Any,
    direct_protocol: Any,
) -> dict[str, Any]:
    """Verify every frozen phase 8B/8C/8E policy is exactly the committed one.

    Args:
        paths: Project layout.
        baseline: S0's protocol.
        comparison: The phase 8E comparison protocol.
        canonical_protocol: The frozen canonical evaluator.
        direct_protocol: The frozen phase 8C direct-IoU protocol.

    Returns:
        The verified fingerprints.

    Raises:
        ProtocolMismatchError: If any fingerprint or required field has moved.
    """
    comparison_fingerprint = comparison.fingerprint()
    canonical_fingerprint = canonical_protocol.fingerprint()

    if comparison_fingerprint != EXPECTED_COMPARISON_FINGERPRINT:
        msg = (
            f"{S1_PROTOCOL_INVALID}: the comparison protocol hashes to "
            f"{comparison_fingerprint}, not the frozen {EXPECTED_COMPARISON_FINGERPRINT}"
        )
        raise ProtocolMismatchError(msg)
    if canonical_fingerprint != EXPECTED_CANONICAL_FINGERPRINT:
        msg = (
            f"{S1_PROTOCOL_INVALID}: the canonical evaluator hashes to {canonical_fingerprint}, "
            f"not the frozen {EXPECTED_CANONICAL_FINGERPRINT}"
        )
        raise ProtocolMismatchError(msg)
    if comparison["canonical_evaluation_sha256"] != canonical_fingerprint:
        msg = f"{S1_PROTOCOL_INVALID}: the comparison protocol names a different evaluator"
        raise ProtocolMismatchError(msg)
    if comparison["metrics"]["primary"] != PRIMARY_METRIC:
        msg = f"{S1_PROTOCOL_INVALID}: the primary metric is not {PRIMARY_METRIC}"
        raise ProtocolMismatchError(msg)

    manifest = read_json(paths.reports / S1_PROTOCOL_JSON)
    if manifest["comparison_protocol_fingerprint"] != comparison_fingerprint:
        msg = f"{S1_PROTOCOL_INVALID}: the S1 protocol manifest names a different comparison"
        raise ProtocolMismatchError(msg)
    if manifest["canonical_evaluation_fingerprint"] != canonical_fingerprint:
        msg = f"{S1_PROTOCOL_INVALID}: the S1 protocol manifest names a different evaluator"
        raise ProtocolMismatchError(msg)
    if manifest["status"] != "FROZEN_NOT_EXECUTED":
        msg = (
            f"{S1_PROTOCOL_INVALID}: the frozen S1 protocol records status "
            f"{manifest['status']!r}, not FROZEN_NOT_EXECUTED. Exactly one S1 run is authorised."
        )
        raise ProtocolMismatchError(msg)
    if manifest["checkpoint_policy"] != CHECKPOINT_POLICY:
        msg = f"{S1_PROTOCOL_INVALID}: the S1 checkpoint policy is not {CHECKPOINT_POLICY}"
        raise ProtocolMismatchError(msg)

    policy = read_json(paths.reports / POLICY_JSON)
    if policy.get("margin") != 0.005:
        msg = f"{S1_PROTOCOL_INVALID}: the frozen margin is not 0.005"
        raise ProtocolMismatchError(msg)

    if (
        direct_protocol.fingerprint()
        != read_json(paths.reports / S0_MASK_IOU_JSON)["protocol_fingerprint"]
    ):
        msg = (
            f"{S1_PROTOCOL_INVALID}: the direct-IoU protocol no longer matches the one S0's "
            "committed diagnostic ran under, so the two diagnostics would not be comparable"
        )
        raise ProtocolMismatchError(msg)

    return {
        "s0_protocol_fingerprint": baseline.fingerprint(),
        "comparison_protocol_fingerprint": comparison_fingerprint,
        "canonical_evaluator_fingerprint": canonical_fingerprint,
        "direct_iou_protocol_fingerprint": direct_protocol.fingerprint(),
        "s1_protocol_manifest_sha256": sha256_file(paths.reports / S1_PROTOCOL_JSON),
        "comparison_policy_sha256": sha256_file(paths.reports / POLICY_JSON),
        "s1_status_before_run": manifest["status"],
        "margin": policy["margin"],
        "primary_metric": PRIMARY_METRIC,
        "reference_experiment": REFERENCE,
        "candidate_experiment": EXPERIMENT_ID,
    }


def verify_adapter(
    paths: ProjectPaths, config: SegmentationBaselineConfig, *, stage: str
) -> dict[str, Any]:
    """Verify the approved label bytes are the ones on disk.

    Args:
        paths: Project layout.
        config: S0's protocol, which names the approved digests.
        stage: Where in the phase this runs, used in the error message.

    Returns:
        The verified fingerprints and cardinality.

    Raises:
        AdapterMismatchError: If any digest or count differs.
        ExperimentError: If the adapter is not present.
    """
    root = paths.root / Path(config.dataset_config).parent
    if not root.is_dir():
        msg = f"the approved adapter is not on this machine: {config.dataset_config}"
        raise ExperimentError(msg)
    labels = read_adapter_labels(root)
    observed = adapter_fingerprints(labels)
    mismatched = sorted(name for name, value in observed.items() if value != config.adapter[name])
    if mismatched:
        detail = "; ".join(
            f"{name}: expected {config.adapter[name]}, observed {observed[name]}"
            for name in mismatched
        )
        msg = f"{ADAPTER_FINGERPRINT_MISMATCH} at {stage}: {detail}. Labels are never rebuilt."
        raise AdapterMismatchError(msg)

    counts = adapter_counts(labels)
    wrong = sorted(name for name, value in counts.items() if value != config.adapter[name])
    if wrong:
        detail = "; ".join(
            f"{name}: expected {config.adapter[name]}, observed {counts[name]}" for name in wrong
        )
        msg = f"{ADAPTER_FINGERPRINT_MISMATCH} at {stage}: cardinality disagrees: {detail}"
        raise AdapterMismatchError(msg)
    return {"fingerprints": observed, "counts": counts, "root": root}


def verify_no_holdout(paths: ProjectPaths, config: SegmentationBaselineConfig) -> None:
    """Refuse to continue if any holdout artifact exists.

    Args:
        paths: Project layout.
        config: S0's protocol.

    Raises:
        ExperimentError: If a holdout artifact is present.
    """
    adapter_root = paths.root / Path(config.dataset_config).parent
    forbidden = [
        adapter_root / "images" / "test",
        adapter_root / "labels" / "test",
        paths.root / RUNTIME_VIEW_ROOT / "images" / "test",
        paths.root / RUNTIME_VIEW_ROOT / "labels" / "test",
        paths.data_processed / "canonical" / "images" / "test",
        paths.data_processed / "canonical" / "annotations" / "segmentation_test.coco.json",
    ]
    present = [path for path in forbidden if path.exists()]
    if present:
        names = ", ".join(path.relative_to(paths.root).as_posix() for path in present)
        msg = f"holdout artifacts exist and must not: {names}"
        raise ExperimentError(msg)


def verify_s0_reference(paths: ProjectPaths) -> dict[str, Any]:
    """Verify S0's frozen checkpoint and read its committed numbers.

    S0 is read, never executed: no model is loaded, no metric recomputed. The
    checkpoint is hashed because a re-run under the same name would produce
    different bytes, and that substitution is what this check exists to catch.

    Args:
        paths: Project layout.

    Returns:
        S0's verified identity and its committed reference metrics.

    Raises:
        ExperimentError: If the checkpoint is missing or its bytes differ.
    """
    result = read_json(paths.reports / S0_RESULT_JSON)
    canonical = read_json(paths.reports / S0_CANONICAL_JSON)
    direct = read_json(paths.reports / S0_MASK_IOU_JSON)

    checkpoint = paths.root / result["execution"]["run_directory"] / "weights" / "best.pt"
    if not checkpoint.is_file():
        msg = (
            f"the frozen S0 checkpoint is not on this machine: {checkpoint.name}. It is "
            "git-ignored; obtain the artifact rather than retraining."
        )
        raise ExperimentError(msg)
    observed = sha256_file(checkpoint)
    if observed != EXPECTED_S0_CHECKPOINT or observed != result["checkpoints"]["best"]["sha256"]:
        msg = f"the S0 checkpoint bytes are not the frozen ones: observed {observed}"
        raise ExperimentError(msg)

    return {
        "experiment_id": REFERENCE,
        "checkpoint_sha256": observed,
        "checkpoint_verified": True,
        "retrained": False,
        "revalidated": False,
        "recomputed": False,
        "source": "COMMITTED_ARTIFACTS_READ_NOT_EXECUTED",
        "canonical": {
            "supported_macro": canonical["supported_macro"]["value"],
            "all_class_map50_95": canonical["canonical"]["all_class_map50_95"],
            "all_class_map50": canonical["canonical"]["all_class_map50"],
            "per_class": dict(canonical["canonical"]["per_class"]),
            "artifact_sha256": sha256_file(paths.reports / S0_CANONICAL_JSON),
        },
        "native": {
            "mask": dict(result["mask_metrics"]),
            "box": dict(result["box_metrics_from_segmenter"]),
            "supported_macro": result["supported_macro_mask"]["value"],
            "per_class": dict(result["per_class_metrics"]),
            "status": "NATIVE_TARGET_EVALUATION",
            "cross_target_comparable": False,
        },
        "direct_iou": {
            "global": dict(direct["global"]),
            "per_class": dict(direct["per_class"]),
            "protocol_fingerprint": direct["protocol_fingerprint"],
            "rerun_in_this_phase": False,
        },
        "experiment_sha256": result["S0_experiment_sha256"],
        "best_epoch": result["checkpoint_selection"]["best_epoch"],
    }


# --- canonical ground truth ---------------------------------------------------


def load_canonical_validation(paths: ProjectPaths, document: str) -> dict[str, Any]:
    """Read the canonical COCO validation document.

    Args:
        paths: Project layout.
        document: Repository-relative path from a frozen protocol.

    Returns:
        The parsed COCO document plus its digest.

    Raises:
        ExperimentError: If it is missing.
    """
    path = paths.root / document
    if not path.is_file():
        msg = f"canonical validation annotations not found: {document}"
        raise ExperimentError(msg)
    payload = json.loads(Path(long_path(path)).read_text(encoding="utf-8"))
    return {"document": payload, "sha256": sha256_file(path), "relative_path": document}


def canonical_instances(
    coco: Mapping[str, Any], class_map: Mapping[str, int]
) -> dict[str, list[MaskInstance]]:
    """Decode every canonical validation annotation into a full-canvas mask.

    A local copy of phase 8C's decoder rather than an import: that phase's script
    is not importable and is deliberately left alone, being the producer of S0's
    committed evidence. The decoding is exercised against the same canonical
    document, and the instance count it yields is checked against the frozen
    adapter cardinality before anything is scored.

    Args:
        coco: The parsed COCO document.
        class_map: The frozen class map.

    Returns:
        Instances keyed by image file stem.

    Raises:
        ExperimentError: If a category is not in the frozen class map.
    """
    from pycocotools import mask as mask_utils

    categories = {int(entry["id"]): str(entry["name"]) for entry in coco["categories"]}
    images = {
        int(entry["id"]): (
            Path(str(entry["file_name"])).stem,
            int(entry["height"]),
            int(entry["width"]),
        )
        for entry in coco["images"]
    }
    instances: dict[str, list[MaskInstance]] = {stem: [] for stem, _, _ in images.values()}

    for annotation in coco["annotations"]:
        name = categories[int(annotation["category_id"])]
        if name not in class_map:
            msg = f"canonical annotation carries a category outside the frozen class map: {name}"
            raise ExperimentError(msg)
        stem, height, width = images[int(annotation["image_id"])]
        segmentation = annotation["segmentation"]
        if isinstance(segmentation, list):
            rle = mask_utils.merge(mask_utils.frPyObjects(segmentation, height, width))
        elif isinstance(segmentation.get("counts"), list):
            rle = mask_utils.frPyObjects(segmentation, height, width)
        else:
            rle = dict(segmentation)
            if isinstance(rle["counts"], str):
                rle["counts"] = rle["counts"].encode("utf-8")
        mask = mask_utils.decode(rle).astype(bool)
        if mask.ndim == 3:  # pragma: no cover - merge already flattens, kept defensive
            mask = mask.any(axis=2)
        instances[stem].append(MaskInstance(class_id=class_map[name], mask=mask))
    return instances


# --- the experiment -----------------------------------------------------------


def train(
    paths: ProjectPaths,
    arguments: Mapping[str, Any],
    weights: Mapping[str, Any],
    descriptor: Path,
) -> dict[str, Any]:
    """Execute the single full S1 training run.

    Args:
        paths: Project layout.
        arguments: The resolved S1 framework arguments.
        weights: The pretrained checkpoint record.
        descriptor: The runtime view's dataset descriptor.

    Returns:
        Facts about the execution, with no metric interpretation.

    Raises:
        MemoryConstraintError: If the frozen batch exhausts device memory. The
            batch is never reduced to rescue the run.
        ExperimentError: If the run fails otherwise, or lands somewhere other
            than the directory the result is read from.
    """
    import torch
    from ultralytics import YOLO

    run_directory = paths.root / RUN_ROOT / RUN_NAME
    if (run_directory / "weights" / "best.pt").is_file():
        msg = (
            f"a completed S1 run already exists at {RUN_ROOT}/{RUN_NAME}. Exactly one valid full "
            "run is authorised; overwriting it would destroy the experiment being reported. "
            "Move or quarantine it deliberately if a re-run has been reviewed."
        )
        raise ExperimentError(msg)
    run_directory.mkdir(parents=True, exist_ok=True)

    call = dict(arguments)
    call["data"] = str(descriptor.resolve())
    call["project"] = str((paths.root / RUN_ROOT).resolve())
    call["name"] = RUN_NAME
    # Ultralytics diverts a run to `<name>-2` when the output directory already
    # exists, and attaching the log handler creates it. The pre-run check above
    # is what guards against overwriting a completed run; this only stops the
    # framework silently writing somewhere the result is not read from, which is
    # verified after training.
    call["exist_ok"] = True
    call["plots"] = True

    handler = capture_framework_log(run_directory / "framework.log")
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    failure: str | None = None
    try:
        model = YOLO(str(paths.root / weights["relative_path"]))
        model.train(**call)
    except torch.cuda.OutOfMemoryError as exc:
        release_framework_log(handler)
        msg = (
            f"{MEMORY_CONSTRAINT_REVIEW_REQUIRED}: the frozen batch of {arguments['batch']} "
            f"exhausted device memory ({exc}). The batch is not reduced, auto-batch is not "
            "enabled, gradient accumulation is not introduced and imgsz is not lowered: the "
            "experiment stops for review."
        )
        raise MemoryConstraintError(msg) from exc
    except Exception as exc:  # any failure stops the experiment and is reported as one
        failure = f"{type(exc).__name__}: {exc}"
    elapsed = time.perf_counter() - started
    peak_reserved = int(torch.cuda.max_memory_reserved())
    peak_allocated = int(torch.cuda.max_memory_allocated())
    release_framework_log(handler)

    if failure is not None:
        msg = f"{TRAINING_FAILED}: the S1 training run did not complete ({failure})"
        raise ExperimentError(msg)

    if not (run_directory / "weights" / "best.pt").is_file():
        msg = (
            f"{TRAINING_FAILED}: training reported success but wrote no best.pt in "
            f"{RUN_ROOT}/{RUN_NAME}. The framework may have diverted the run elsewhere."
        )
        raise ExperimentError(msg)

    log_text = (run_directory / "framework.log").read_text(encoding="utf-8", errors="replace")
    return {
        "run_directory": f"{RUN_ROOT}/{RUN_NAME}",
        "wall_clock_seconds": round(elapsed, 3),
        "peak_gpu_memory_reserved_bytes": peak_reserved,
        "peak_gpu_memory_reserved_gib": round(peak_reserved / 1024**3, 3),
        "peak_gpu_memory_allocated_bytes": peak_allocated,
        "peak_gpu_memory_source": "TORCH_CUDA_MAX_MEMORY_RESERVED_IN_THE_TRAINING_PROCESS",
        "log_text": log_text,
    }


def verify_checkpoint_selection(
    history: Sequence[Mapping[str, str]], resolved: Mapping[str, Any]
) -> dict[str, Any]:
    """Confirm the recorded checkpoint is the frozen rule's argmax.

    Ultralytics writes ``best.pt`` by its own composite fitness but records only
    the terms, so the composite is recomputed here and its argmax compared
    against what the run actually did.

    Args:
        history: Parsed ``results.csv`` rows.
        resolved: The framework's recorded arguments.

    Returns:
        The selection evidence.

    Raises:
        ExperimentError: If the composite cannot be reconstructed.
    """
    curve = native_fitness_curve(history)
    best_epoch, best_fitness = best_epoch_by_native_fitness(history)
    if best_epoch is None:
        msg = "results.csv carried no usable epoch, so the checkpoint rule cannot be verified"
        raise ExperimentError(msg)
    ranked = sorted(curve, key=lambda item: (-item[1], item[0]))
    runner_up = ranked[1] if len(ranked) > 1 else (None, None)
    return {
        "policy": CHECKPOINT_POLICY,
        "semantics": CHECKPOINT_SELECTION_SEMANTICS,
        "best_epoch": best_epoch,
        "best_native_fitness": round(float(best_fitness), METRIC_PRECISION),
        "next_best_epoch": runner_up[0],
        "next_best_native_fitness": (
            round(float(runner_up[1]), METRIC_PRECISION) if runner_up[1] is not None else None
        ),
        "epochs_logged": len(curve),
        "epochs_configured": int(resolved.get("epochs", 0)) or None,
        "reconstruction": (
            "The composite is recomputed from results.csv as metrics/mAP50-95(B) + "
            "metrics/mAP50-95(M), which is exactly SegmentMetrics.fitness for the installed "
            "version, and the recorded best epoch is its argmax."
        ),
        "tolerance": FITNESS_TOLERANCE,
        "manual_epoch_selection": False,
        "mask_only_checkpoint_created": False,
        "identical_policy_to_reference": True,
        "computed_against_own_target": True,
        "verified": True,
    }


def validate_native(
    paths: ProjectPaths, arguments: Mapping[str, Any], checkpoint: Path
) -> dict[str, Any]:
    """Run the single authoritative native validation of the selected checkpoint.

    ``overlap_mask`` is passed explicitly. Read from the installed source:
    ``Model._reset_ckpt_args`` keeps only ``imgsz``, ``data``, ``task`` and
    ``single_cls`` from a checkpoint, and the framework default is ``True``, so
    omitting it would validate S1 against S0's target.

    Args:
        paths: Project layout.
        arguments: The resolved S1 framework arguments.
        checkpoint: The selected ``best.pt``.

    Returns:
        Global and per-class mask and box metrics, plus the effective settings.

    Raises:
        ExperimentError: If validation fails.
    """
    from ultralytics import YOLO
    from ultralytics.cfg import get_cfg

    output = paths.root / RUN_ROOT / f"{RUN_NAME}_val"
    if output.exists():
        shutil.rmtree(output)

    overrides = {
        "data": str((paths.root / RUNTIME_VIEW_ROOT / "dataset.yaml").resolve()),
        "split": "val",
        "imgsz": arguments["imgsz"],
        "batch": arguments["batch"],
        "overlap_mask": arguments[ONE_VARIABLE_FIELD],
        "mask_ratio": arguments["mask_ratio"],
        "project": str((paths.root / RUN_ROOT).resolve()),
        "name": f"{RUN_NAME}_val",
        "exist_ok": True,
        "plots": True,
    }

    model = YOLO(str(checkpoint))
    try:
        metrics = model.val(**overrides)
    except Exception as exc:  # a failed validation is a failed experiment
        msg = f"{TRAINING_FAILED}: the authoritative validation did not complete ({exc})"
        raise ExperimentError(msg) from exc

    results = {key: float(value) for key, value in metrics.results_dict.items()}
    names = metrics.names if isinstance(metrics.names, dict) else dict(enumerate(metrics.names))

    def family(keys: Mapping[str, str]) -> dict[str, Any]:
        return {label: rounded(results.get(key, NOT_EXPOSED)) for label, key in keys.items()}

    per_class: dict[str, dict[str, Any]] = {}
    for position, class_index in enumerate(metrics.box.ap_class_index.tolist()):
        name = str(names[int(class_index)])
        box_precision, box_recall, box_ap50, box_ap = metrics.box.class_result(position)
        mask_precision, mask_recall, mask_ap50, mask_ap = metrics.seg.class_result(position)
        per_class[name] = {
            "mask": {
                "precision": rounded(mask_precision),
                "recall": rounded(mask_recall),
                "AP@0.50": rounded(mask_ap50),
                "AP@0.50:0.95": rounded(mask_ap),
            },
            "box": {
                "precision": rounded(box_precision),
                "recall": rounded(box_recall),
                "AP@0.50": rounded(box_ap50),
                "AP@0.50:0.95": rounded(box_ap),
            },
        }

    # The metrics object carries no arguments and `Model.val` does not keep the
    # validator it built, so the effective settings are re-resolved through the
    # framework's own `get_cfg` with the same overrides the call was given.
    resolved_val = get_cfg(overrides={**overrides, "mode": "val", "task": "segment"})
    fields = (
        "conf",
        "iou",
        "max_det",
        "imgsz",
        "batch",
        "half",
        "augment",
        "rect",
        "single_cls",
        "split",
        "overlap_mask",
        "mask_ratio",
        "retina_masks",
    )
    effective = {
        name: relativise(getattr(resolved_val, name, NOT_EXPOSED), paths.root) for name in fields
    }
    effective["source"] = "RESOLVED_FROM_INSTALLED_CFG_WITH_THE_SAME_OVERRIDES"
    effective["overlap_mask_passed_explicitly"] = True
    effective["why_overlap_mask_is_passed"] = (
        "Model._reset_ckpt_args keeps only imgsz, data, task and single_cls from a checkpoint, "
        "and the framework default is True. Omitting the flag would have validated S1 against "
        "S0's overlap-resolved target instead of its own."
    )

    return {
        "mask": family(MASK_KEYS),
        "box": family(BOX_KEYS),
        "per_class": per_class,
        "framework_fitness": rounded(results.get("fitness")),
        "speed": {key: rounded(value) for key, value in dict(metrics.speed).items()},
        "effective_arguments": effective,
        "output_directory": f"{RUN_ROOT}/{RUN_NAME}_val",
        "runs": 1,
    }


def native_supported_macro(
    per_class: Mapping[str, Mapping[str, Any]], support: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    """Apply the frozen support rule to the native mask AP values.

    Kept for descriptive continuity with S0 and explicitly barred from
    selecting between the two experiments: their native targets differ.

    Args:
        per_class: Native per-class mask and box metrics.
        support: The frozen support rule's verdict per class.

    Returns:
        The macro metric, the admitted classes and the arithmetic behind it.
    """
    classification: dict[str, str] = {}
    admitted: list[str] = []
    for name, counts in sorted(support.items()):
        if counts["supported"]:
            classification[name] = SUPPORTED
            admitted.append(name)
        else:
            classification[name] = RARE_CLASS_STATUS
    values = [
        float(per_class[name]["mask"]["AP@0.50:0.95"])
        for name in admitted
        if name in per_class and isinstance(per_class[name]["mask"]["AP@0.50:0.95"], (int, float))
    ]
    return {
        "metric": NATIVE_SUPPORTED_MACRO_METRIC,
        "label": DESCRIPTIVE_NATIVE_TARGET_METRIC,
        "value": round(sum(values) / len(values), METRIC_PRECISION) if values else None,
        "rule": SUPPORT_RULE,
        "rule_origin": "FROZEN_IN_PHASE_7A_REUSED_UNCHANGED",
        "classification": classification,
        "admitted_classes": admitted,
        "contributing_values": [round(item, METRIC_PRECISION) for item in values],
        "used_for_selection": False,
        "reason": (
            "Computed for descriptive continuity with S0's reported figure. It may not select "
            "between S0 and S1: overlap_mask changes the native target, so the two experiments' "
            "native mask AP is measured against different ground truth."
        ),
    }


def run_canonical(
    paths: ProjectPaths,
    protocol: Any,
    checkpoint: Path,
    ground_truth: Mapping[str, Any],
    class_map: Mapping[str, int],
) -> dict[str, Any]:
    """Execute the frozen canonical COCO evaluation exactly once.

    Args:
        paths: Project layout.
        protocol: The frozen canonical evaluator.
        checkpoint: S1's selected ``best.pt``.
        ground_truth: The canonical COCO validation document.
        class_map: The frozen class map.

    Returns:
        The canonical result and the settings it ran under.

    Raises:
        ExperimentError: If prediction fails or an image is not canonical.
        CanonicalEvaluationError: If a predicted mask is off-canvas.
    """
    from ultralytics import YOLO

    inference = protocol.inference
    images = {
        Path(str(entry["file_name"])).stem: int(entry["id"]) for entry in ground_truth["images"]
    }
    images_root = paths.root / RUNTIME_VIEW_ROOT / "images" / "val"
    if not images_root.is_dir():
        msg = f"the validation image view is not on this machine: {RUNTIME_VIEW_ROOT}/images/val"
        raise ExperimentError(msg)

    model = YOLO(str(checkpoint))
    detections: list[dict[str, Any]] = []
    for image_path in sorted(path for path in images_root.iterdir() if path.suffix != ".txt"):
        stem = image_path.stem
        if stem not in images:
            msg = f"predicted image {stem} has no canonical validation entry"
            raise ExperimentError(msg)
        try:
            outputs = model.predict(
                source=str(image_path),
                imgsz=inference["imgsz"],
                conf=inference["conf"],
                iou=inference["iou"],
                max_det=inference["max_det"],
                retina_masks=inference["retina_masks"],
                augment=inference["augment"],
                agnostic_nms=inference["agnostic_nms"],
                half=inference["half"],
                verbose=False,
                save=False,
                stream=False,
            )
        except Exception as exc:
            msg = f"{CANONICAL_EVALUATION_FAILED}: prediction failed on one image ({exc})"
            raise ExperimentError(msg) from exc

        result = outputs[0]
        if result.masks is None:
            continue
        masks = result.masks.data.cpu().numpy().astype(bool)
        classes = result.boxes.cls.cpu().numpy().astype(int).tolist()
        scores = result.boxes.conf.cpu().numpy().astype(float).tolist()
        for mask, class_index, score in zip(masks, classes, scores, strict=True):
            detections.append(
                prediction_record(
                    image_id=images[stem],
                    category_id=int(class_index),
                    mask=mask,
                    score=float(score),
                )
            )

    return evaluate_canonical(ground_truth, detections, class_map=class_map)


def run_direct_iou(
    paths: ProjectPaths,
    protocol: Any,
    checkpoint: Path,
    ground_truth: Mapping[str, list[MaskInstance]],
    class_map: Mapping[str, int],
) -> dict[str, Any]:
    """Execute the frozen phase 8C direct mask-IoU diagnostic exactly once.

    Args:
        paths: Project layout.
        protocol: The frozen diagnostic protocol, unchanged from phase 8C.
        checkpoint: S1's selected ``best.pt``.
        ground_truth: Canonical instances keyed by image stem.
        class_map: The frozen class map.

    Returns:
        The diagnostic result plus the settings it actually ran under.

    Raises:
        ExperimentError: If prediction fails or an image has no canonical entry.
    """
    from ultralytics import YOLO

    inference = protocol.inference
    images_root = paths.root / RUNTIME_VIEW_ROOT / "images" / "val"
    image_paths = sorted(path for path in images_root.iterdir() if path.suffix != ".txt")
    if not image_paths:
        msg = "the runtime view carries no validation images to predict on"
        raise ExperimentError(msg)

    model = YOLO(str(checkpoint))
    matchings = []
    predicted_total = 0
    for image_path in image_paths:
        stem = image_path.stem
        if stem not in ground_truth:
            msg = f"predicted image {stem} has no canonical validation entry"
            raise ExperimentError(msg)
        try:
            outputs = model.predict(
                source=str(image_path),
                imgsz=inference["imgsz"],
                conf=inference["conf"],
                iou=inference["iou"],
                max_det=inference["max_det"],
                retina_masks=inference["retina_masks"],
                augment=inference["augment"],
                agnostic_nms=inference["agnostic_nms"],
                half=inference["half"],
                verbose=False,
                save=False,
                save_txt=False,
                save_conf=False,
                stream=False,
            )
        except Exception as exc:
            msg = f"{DIRECT_IOU_EVALUATION_FAILED}: prediction failed on one image ({exc})"
            raise ExperimentError(msg) from exc

        result = outputs[0]
        predictions: list[MaskInstance] = []
        if result.masks is not None:
            masks = result.masks.data.cpu().numpy().astype(bool)
            classes = result.boxes.cls.cpu().numpy().astype(int).tolist()
            for mask, class_index in zip(masks, classes, strict=True):
                predictions.append(MaskInstance(class_id=int(class_index), mask=mask))
        predicted_total += len(predictions)
        matchings.append(match_image(stem, ground_truth[stem], predictions))

    outcome = accumulate(matchings)
    names = {index: name for name, index in class_map.items()}
    rendered = outcome.as_dict(names)
    if rendered["global"]["prediction_count"] != predicted_total:  # pragma: no cover - invariant
        msg = "the diagnostic lost a prediction between matching and accumulation"
        raise ExperimentError(msg)
    return {
        "protocol": DIRECT_IOU_PROTOCOL,
        "protocol_fingerprint": protocol.fingerprint(),
        "ground_truth_source": GROUND_TRUTH_SOURCE,
        "matching_algorithm": MATCHING_ALGORITHM,
        "inference": {key: value for key, value in inference.items() if key != "threshold_policy"},
        "runs": 1,
        **rendered,
    }


# --- artifacts ------------------------------------------------------------------


def build_result_manifest(
    *,
    baseline: SegmentationBaselineConfig,
    comparison: Any,
    policies: Mapping[str, Any],
    contract: Mapping[str, Any],
    inheritance: Mapping[str, Any],
    adapter: Mapping[str, Any],
    approval_sha256: str,
    canonical_fingerprints: Mapping[str, Any],
    detector: Mapping[str, Any],
    historical: Mapping[str, str],
    runtime: Mapping[str, Any],
    runtime_view: Mapping[str, Any],
    weights: Mapping[str, Any],
    execution: Mapping[str, Any],
    resolved: Mapping[str, Any],
    optimizer: Mapping[str, Any] | None,
    selection: Mapping[str, Any],
    checkpoints: Mapping[str, Any],
    native: Mapping[str, Any],
    native_macro: Mapping[str, Any],
    canonical: Mapping[str, Any],
    canonical_macro: Mapping[str, Any],
    canonical_sha256: str,
    delta: Mapping[str, Any],
    direct: Mapping[str, Any],
    direct_sha256: str,
    direction: Mapping[str, Any],
    person: Mapping[str, Any],
    per_class_table: Mapping[str, Any],
    reference: Mapping[str, Any],
    support: Mapping[str, Any],
    complexity: Mapping[str, Any],
    figures: Sequence[str],
    curves: Mapping[str, Any],
    hypothesis: str,
) -> dict[str, Any]:
    """Assemble the S1 result manifest.

    Args:
        baseline: S0's protocol, which S1 inherits.
        comparison: The frozen phase 8E comparison protocol.
        policies: The verified policy fingerprints.
        contract: The verified one-variable contract.
        inheritance: What the inherited-protocol check confirmed.
        adapter: Verified label fingerprints and cardinality.
        approval_sha256: Digest of the phase 8B adapter approval.
        canonical_fingerprints: Class map, split and annotation digests.
        detector: The frozen detector verification.
        historical: Digests of every artifact this phase must not change.
        runtime: Software and hardware facts.
        runtime_view: The training-time dataset view record.
        weights: The pretrained checkpoint record.
        execution: Training execution facts.
        resolved: The framework's recorded arguments.
        optimizer: Captured optimizer evidence.
        selection: Checkpoint-selection verification.
        checkpoints: best and last checkpoint records.
        native: The authoritative native validation result.
        native_macro: The descriptive native supported macro.
        canonical: The canonical COCO evaluation result.
        canonical_macro: The canonical supported macro.
        canonical_sha256: Digest of the canonical evaluation artifact.
        delta: The primary delta and its margin classification.
        direct: The direct mask-IoU result.
        direct_sha256: Digest of the direct mask-IoU artifact.
        direction: The cross-metric direction classification.
        person: The predeclared person diagnostic.
        per_class_table: The canonical per-class S0-versus-S1 comparison.
        reference: S0's verified identity and committed metrics.
        support: The frozen support rule's verdict per class.
        complexity: Model parameter and FLOP counts.
        figures: Committed metric-only figure names.
        curves: Observations about the training dynamics.
        hypothesis: The predeclared S1 hypothesis, quoted from phase 8E.

    Returns:
        The manifest.
    """
    scheduler = "COSINE" if resolved.get("cos_lr") else "LINEAR_LAMBDA_LR0_TO_LR0_TIMES_LRF"
    effective_names = sorted(
        {
            "model",
            "imgsz",
            "batch",
            "epochs",
            "optimizer",
            "lr0",
            "lrf",
            "momentum",
            "weight_decay",
            "warmup_epochs",
            "warmup_momentum",
            "warmup_bias_lr",
            "cos_lr",
            "patience",
            "amp",
            "deterministic",
            "workers",
            "seed",
            "device",
            "overlap_mask",
            "mask_ratio",
            "retina_masks",
            "dropout",
            "max_det",
            "single_cls",
            "rect",
            "multi_scale",
            "close_mosaic",
            "mosaic",
            "mixup",
            "cutmix",
            "copy_paste",
            "copy_paste_mode",
            "auto_augment",
            "erasing",
            "hsv_h",
            "hsv_s",
            "hsv_v",
            "degrees",
            "translate",
            "scale",
            "shear",
            "perspective",
            "flipud",
            "fliplr",
            "bgr",
        }
    )
    effective = {name: resolved.get(name, NOT_EXPOSED) for name in effective_names}

    identity = {
        "experiment_id": EXPERIMENT_ID,
        "s0_protocol_fingerprint": baseline.fingerprint(),
        "comparison_protocol_fingerprint": policies["comparison_protocol_fingerprint"],
        "canonical_evaluator_fingerprint": policies["canonical_evaluator_fingerprint"],
        "direct_iou_protocol_fingerprint": policies["direct_iou_protocol_fingerprint"],
        "adapter_approval_sha256": approval_sha256,
        "adapter_fingerprints": dict(adapter["fingerprints"]),
        "class_map_sha256": canonical_fingerprints["class_map_sha256"],
        "split_assignment_sha256": canonical_fingerprints["split_assignment_sha256"],
        "canonical_task_manifest_sha256": canonical_fingerprints["canonical_task_manifest_sha256"],
        "pretrained_weight_sha256": weights["sha256"],
        "checkpoint_selection_policy": CHECKPOINT_POLICY,
        "checkpoint_selection_semantics": CHECKPOINT_SELECTION_SEMANTICS,
        "best_checkpoint_sha256": checkpoints["best"]["sha256"],
        "intentional_difference": {
            "field": ONE_VARIABLE_FIELD,
            "reference_value": contract["reference_value"],
            "candidate_value": contract["candidate_value"],
        },
        "critical_training_config": {
            "model": baseline.model,
            "imgsz": effective.get("imgsz"),
            "batch": effective.get("batch"),
            "epochs": effective.get("epochs"),
            "seed": effective.get("seed"),
            "optimizer_policy": baseline.training["optimizer"],
            "patience": effective.get("patience"),
            "deterministic": effective.get("deterministic"),
            "amp": effective.get("amp"),
            "overlap_mask": effective.get("overlap_mask"),
            "mask_ratio": effective.get("mask_ratio"),
        },
    }

    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "phase": PHASE,
        "experiment_id": EXPERIMENT_ID,
        "status": COMPLETE,
        "task": "segmentation",
        "reference_experiment": REFERENCE,
        "experiment_question": comparison.candidate["question"].strip(),
        "hypothesis": hypothesis,
        "hypothesis_label": "PREDECLARED_S1_HYPOTHESIS",
        "experiment_role": "CONTROLLED_CANDIDATE_NOT_THE_FINAL_SEGMENTER",
        "metrics_are_validation_only": True,
        "architecture": baseline.architecture,
        "model": baseline.model,
        "intentional_difference": {
            "field": ONE_VARIABLE_FIELD,
            "reference_value": contract["reference_value"],
            "candidate_value": contract["candidate_value"],
            "label": "ONE_VARIABLE_INTERVENTION",
            "one_variable": bool(contract["one_variable"]),
            "fields_inherited_unchanged": list(contract["fields_inherited_unchanged"]),
            "inherited_count": contract["inherited_count"],
            "resolved_from": "configs/" + BASELINE_YAML,
            "declared_in": "configs/" + COMPARISON_YAML,
        },
        "inherited_protocol_verified": dict(inheritance),
        "treatment_semantics": TREATMENT_SEMANTICS,
        "treatment_semantics_label": "PREDECLARED_TREATMENT_DESCRIPTION",
        "s1_protocol_fingerprint": policies["comparison_protocol_fingerprint"],
        "comparison_policy_fingerprint": policies["comparison_policy_sha256"],
        "canonical_evaluator_fingerprint": policies["canonical_evaluator_fingerprint"],
        "direct_iou_protocol_fingerprint": policies["direct_iou_protocol_fingerprint"],
        "s0_protocol_fingerprint": policies["s0_protocol_fingerprint"],
        "frozen_policies": dict(policies),
        "adapter": {
            "role": ADAPTER_ROLE,
            "canonical_ground_truth": CANONICAL_GROUND_TRUTH,
            "approval_sha256": approval_sha256,
            "fingerprints": dict(adapter["fingerprints"]),
            "counts": dict(adapter["counts"]),
            "identical_to_reference": True,
            "regenerated": False,
            "modified": False,
            "filtered": False,
        },
        "runtime_view": dict(runtime_view),
        "canonical_fingerprints": dict(canonical_fingerprints),
        "historical_artifact_digests": dict(historical),
        "historical_artifacts_unchanged": True,
        "frozen_detector": dict(detector),
        "runtime": dict(runtime),
        "pretrained_weights": dict(weights),
        "optimizer": {
            "declared_policy": baseline.training["optimizer"],
            "declared_optimizer_policy": baseline.training["optimizer"],
            "actual_resolved_optimizer": (optimizer.get("optimizer") if optimizer else NOT_EXPOSED),
            "effective_lr0": optimizer.get("effective_lr0") if optimizer else NOT_EXPOSED,
            "effective_momentum": (
                optimizer.get("effective_momentum") if optimizer else NOT_EXPOSED
            ),
            "resolution_evidence": (
                optimizer.get("source") if optimizer else "FRAMEWORK_LOG_LINE_ABSENT"
            ),
            "inferred": bool(optimizer.get("inferred")) if optimizer else False,
            "evidence_line": optimizer.get("evidence") if optimizer else None,
            "declared_values_are_not_what_ran": (
                "optimizer: auto overrides the file's lr0 and momentum. The declared values are "
                "the protocol's policy inputs, never a statement of what ran."
            ),
            "weight_decay": resolved.get("weight_decay", NOT_EXPOSED),
            "warmup_epochs": resolved.get("warmup_epochs", NOT_EXPOSED),
            "warmup_momentum": resolved.get("warmup_momentum", NOT_EXPOSED),
            "warmup_bias_lr": resolved.get("warmup_bias_lr", NOT_EXPOSED),
            "lrf": resolved.get("lrf", NOT_EXPOSED),
            "scheduler": scheduler,
        },
        "effective_training_configuration": effective,
        "effective_training_configuration_note": (
            "Read back from the framework's own args.yaml. Absolute paths are rewritten "
            "repository-relative so a committed artifact does not name this machine."
        ),
        "checkpoint_selection": dict(selection),
        "checkpoints": dict(checkpoints),
        "execution": {key: value for key, value in execution.items() if key != "log_text"},
        "training_curves": dict(curves),
        "model_complexity": dict(complexity),
        "native_validation": {
            "runs": native["runs"],
            "split": "validation",
            "effective_arguments": dict(native["effective_arguments"]),
            "output_directory": native["output_directory"],
        },
        "native_metrics": {
            "label": NATIVE_TARGET_METRIC,
            "cross_target_status": NATIVE_METRIC_STATUS,
            "is_primary_for_selection": False,
            "mask": dict(native["mask"]),
            "box": dict(native["box"]),
            "per_class": dict(native["per_class"]),
            "framework_fitness_at_validation": native["framework_fitness"],
            "why_not_primary": (
                "overlap_mask changes the framework's validation ground truth as well as its "
                "training target - SegmentationValidator._prepare_batch builds its masks with "
                "`masks == index` only when the flag is set - so S0's and S1's native mask AP "
                "are measured against different targets. Reported in full, never differenced "
                "against S0's, and never used to rank."
            ),
        },
        "descriptive_native_supported_macro": dict(native_macro),
        "canonical_evaluation": {
            "protocol": "CANONICAL_COCO_MASK_AP_EVALUATION",
            "protocol_fingerprint": policies["canonical_evaluator_fingerprint"],
            "protocol_config": "configs/" + CANONICAL_YAML,
            "result_artifact": "reports/" + CANONICAL_JSON,
            "result_sha256": canonical_sha256,
            "runs": 1,
            "unchanged_from_phase_8e": True,
        },
        "canonical_metrics": {
            ALL_CLASS_METRIC: canonical["all_class_map50_95"],
            ALL_CLASS_METRIC_50: canonical["all_class_map50"],
            "per_class": dict(canonical["per_class"]),
            "detections_scored": canonical["detections"],
            "label": CANONICAL_PRIMARY_METRIC,
        },
        "canonical_supported_macro": {**dict(canonical_macro), "label": CANONICAL_PRIMARY_METRIC},
        "primary_delta": dict(delta),
        "canonical_all_class_delta": {
            "S0": reference["canonical"]["all_class_map50_95"],
            "S1": canonical["all_class_map50_95"],
            "delta": round(
                float(canonical["all_class_map50_95"])
                - float(reference["canonical"]["all_class_map50_95"]),
                METRIC_PRECISION,
            ),
            "is_not_the_selection_metric": True,
        },
        "canonical_per_class_comparison": dict(per_class_table),
        "aggregate_composition": aggregate_composition(
            per_class_table, canonical_macro["admitted_classes"]
        ),
        "direct_mask_iou": {
            "protocol": DIRECT_IOU_PROTOCOL,
            "protocol_fingerprint": policies["direct_iou_protocol_fingerprint"],
            "protocol_config": "configs/" + MASK_IOU_YAML,
            "result_artifact": "reports/" + MASK_IOU_JSON,
            "result_sha256": direct_sha256,
            "label": "DIRECT_IOU_DIAGNOSTIC",
            "runs": 1,
            "unchanged_from_phase_8c": True,
            "headline": {
                name: direct["global"][name]
                for name in (
                    MATCHED_MASK_IOU_MEAN,
                    GT_NORMALIZED_MASK_IOU,
                    GT_MATCH_COVERAGE,
                    GT_IOU50_COVERAGE,
                    GT_IOU75_COVERAGE,
                )
            },
            "counts": {
                name: direct["global"][name]
                for name in (
                    "gt_count",
                    "prediction_count",
                    "matched_count",
                    "unmatched_gt",
                    "unmatched_predictions",
                )
            },
            "deltas_vs_reference": {
                name: round(
                    float(direct["global"][name]) - float(reference["direct_iou"]["global"][name]),
                    METRIC_PRECISION,
                )
                for name in (
                    MATCHED_MASK_IOU_MEAN,
                    GT_NORMALIZED_MASK_IOU,
                    GT_MATCH_COVERAGE,
                    GT_IOU50_COVERAGE,
                    GT_IOU75_COVERAGE,
                )
            },
            "is_not_average_precision": (
                "Neither figure is a COCO AP. AP is a ranking-sensitive average over IoU "
                "thresholds; these are mask overlap at one predeclared operating point."
            ),
        },
        "cross_metric_direction": dict(direction),
        "person_diagnostic": dict(person),
        "rare_class": {
            "name": baseline.rare_class,
            "status": RARE_CLASS_STATUS,
            "support_rule": SUPPORT_RULE,
            "validation_support": dict(support.get(baseline.rare_class, {})),
            "reported_in_full": True,
            "may_decide_anything": False,
            "limitation": baseline.rare_class_limitation.strip(),
        },
        "support": dict(support),
        "reference_metrics": {
            "experiment_id": REFERENCE,
            "canonical_supported_macro": reference["canonical"]["supported_macro"],
            "canonical_all_class_map50_95": reference["canonical"]["all_class_map50_95"],
            "canonical_all_class_map50": reference["canonical"]["all_class_map50"],
            "canonical_per_class": dict(reference["canonical"]["per_class"]),
            "direct_iou_global": dict(reference["direct_iou"]["global"]),
            "native_mask": dict(reference["native"]["mask"]),
            "source": reference["source"],
            "retrained": False,
            "revalidated": False,
            "recomputed": False,
        },
        "resources": {
            "training_wall_clock_seconds": execution.get("wall_clock_seconds"),
            "training_seconds_from_results_csv": execution.get("training_seconds"),
            "framework_reported_hours": execution.get("framework_reported_hours"),
            "gpu_name": runtime.get("gpu_name"),
            "gpu_arch": runtime.get("gpu_arch"),
            "epochs_configured": execution.get("epochs_configured"),
            "epochs_completed": execution.get("epochs_completed"),
            "best_epoch": selection["best_epoch"],
            "batch": effective.get("batch"),
            "imgsz": effective.get("imgsz"),
            "peak_gpu_memory_reserved_bytes": execution.get("peak_gpu_memory_reserved_bytes"),
            "peak_gpu_memory_reserved_gib": execution.get("peak_gpu_memory_reserved_gib"),
            "peak_gpu_memory_source": execution.get("peak_gpu_memory_source"),
            "parameters": complexity.get("parameters"),
            "gflops": complexity.get("gflops"),
            "fused_parameters": complexity.get("fused_parameters"),
            "fused_gflops": complexity.get("fused_gflops"),
            "framework_validation_speed_ms_per_image": dict(native["speed"]),
            "framework_validation_speed_label": "FRAMEWORK_VALIDATION_SPEED",
            "standardised_latency_benchmark": "NOT_RUN_IN_THIS_PHASE",
        },
        "committed_figures": list(figures),
        "composite_box_mask_score": False,
        "final_segmenter": FINAL_SEGMENTER,
        "final_segmenter_note": (
            "The frozen rule produces a classification, not a frozen segmenter. Selecting the "
            "project's final segmenter remains a reviewed human decision, exactly as phase 7D "
            "was for detection."
        ),
        "pending_human_review": True,
        "S1_experiment_sha256": experiment_fingerprint(identity),
        "experiment_fingerprint_inputs": sorted(identity),
        "test": {"status": HOLDOUT_STATUS, "reason": HOLDOUT_REASON},
        "holdout_accessed": False,
        "detector_touched": False,
        "s0_retrained": False,
        "s0_revalidated": False,
        "s0_artifacts_regenerated": False,
        "models_trained_in_this_phase": 1,
        "alternative_segmenters_trained": 0,
        "imgsz_variants_tried": 0,
        "batch_variants_tried": 0,
        "thresholds_tuned": 0,
        "mask_ratio_variants_tried": 0,
    }


def build_results_artifact(
    manifest: Mapping[str, Any], reference: Mapping[str, Any]
) -> dict[str, Any]:
    """Build the live segmentation-results artifact.

    Args:
        manifest: The assembled S1 result manifest.
        reference: S0's verified identity and committed metrics.

    Returns:
        The live results payload.
    """
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "phase": PHASE,
        "task": "segmentation",
        "reference_experiment": REFERENCE,
        "candidate_experiment": EXPERIMENT_ID,
        "primary_metric": PRIMARY_METRIC,
        "margin": manifest["primary_delta"]["margin"],
        "primary_comparison_status": "COMPUTED",
        "experiments": [
            {
                "experiment_id": REFERENCE,
                "experiment_phase": "8C",
                "status": "S0_COMPLETE",
                "experiment_sha256": reference["experiment_sha256"],
                "role": "REFERENCE",
                "intentional_variable": "NONE_REFERENCE",
                "overlap_mask": True,
                "canonical_supported_macro": reference["canonical"]["supported_macro"],
                "canonical_all_class_map50_95": reference["canonical"]["all_class_map50_95"],
                "canonical_all_class_map50": reference["canonical"]["all_class_map50"],
                "canonical_per_class": dict(reference["canonical"]["per_class"]),
                "native_mask_map50_95": reference["native"]["mask"]["mAP@0.50:0.95"],
                "native_supported_macro": reference["native"]["supported_macro"],
                "direct_gt_normalized_mask_iou": reference["direct_iou"]["global"][
                    GT_NORMALIZED_MASK_IOU
                ],
                "direct_matched_mask_iou_mean": reference["direct_iou"]["global"][
                    MATCHED_MASK_IOU_MEAN
                ],
                "margin_status": "REFERENCE",
                "delta_primary_vs_reference": 0.0,
                "checkpoint_sha256": reference["checkpoint_sha256"],
                "retrained_in_phase_8f": False,
                "revalidated_in_phase_8f": False,
            },
            {
                "experiment_id": EXPERIMENT_ID,
                "experiment_phase": PHASE,
                "status": COMPLETE,
                "experiment_sha256": manifest["S1_experiment_sha256"],
                "role": "CANDIDATE",
                "intentional_variable": ONE_VARIABLE_FIELD,
                "overlap_mask": False,
                "canonical_supported_macro": manifest["canonical_supported_macro"]["value"],
                "canonical_all_class_map50_95": manifest["canonical_metrics"][ALL_CLASS_METRIC],
                "canonical_all_class_map50": manifest["canonical_metrics"][ALL_CLASS_METRIC_50],
                "canonical_per_class": dict(manifest["canonical_metrics"]["per_class"]),
                "native_mask_map50_95": manifest["native_metrics"]["mask"]["mAP@0.50:0.95"],
                "native_supported_macro": manifest["descriptive_native_supported_macro"]["value"],
                "direct_gt_normalized_mask_iou": manifest["direct_mask_iou"]["headline"][
                    GT_NORMALIZED_MASK_IOU
                ],
                "direct_matched_mask_iou_mean": manifest["direct_mask_iou"]["headline"][
                    MATCHED_MASK_IOU_MEAN
                ],
                "margin_status": manifest["primary_delta"]["verdict"],
                "delta_primary_vs_reference": manifest["primary_delta"]["delta"],
                "checkpoint_sha256": manifest["checkpoints"]["best"]["sha256"],
                "retrained_in_phase_8f": True,
                "revalidated_in_phase_8f": True,
            },
        ],
        "cross_metric_direction": dict(manifest["cross_metric_direction"]),
        "native_metric_status": NATIVE_METRIC_STATUS,
        "descriptive_classes": [
            name for name, row in manifest["support"].items() if not row["supported"]
        ],
        "composite_score": False,
        "final_segmenter": FINAL_SEGMENTER,
        "final_segmenter_note": manifest["final_segmenter_note"],
        "pending_human_review": True,
        "further_experiments_authorised": 0,
        "test": {"status": HOLDOUT_STATUS, "reason": HOLDOUT_REASON},
        "holdout_accessed": False,
    }


def build_report(manifest: Mapping[str, Any], *, commit: str | None) -> str:
    """Render the S1 report.

    Args:
        manifest: The assembled result manifest.
        commit: Repository commit, when available.

    Returns:
        The report text.
    """
    native = manifest["native_metrics"]
    canonical = manifest["canonical_metrics"]
    macro = manifest["canonical_supported_macro"]
    delta = manifest["primary_delta"]
    direct = manifest["direct_mask_iou"]
    direction = manifest["cross_metric_direction"]
    person = manifest["person_diagnostic"]
    selection = manifest["checkpoint_selection"]
    execution = manifest["execution"]
    optimizer = manifest["optimizer"]
    weights = manifest["pretrained_weights"]
    adapter = manifest["adapter"]
    reference = manifest["reference_metrics"]
    resources = manifest["resources"]
    rare = manifest["rare_class"]
    curves = manifest["training_curves"]
    difference = manifest["intentional_difference"]

    lines: list[str] = []
    add = lines.append

    add("# S1 - overlap-preserving instance-mask targets")
    add("")
    add(
        f"Phase {manifest['phase']} · classification `{manifest['status']}` · experiment "
        f"`{manifest['experiment_id']}` · reference `{manifest['reference_experiment']}` · "
        f"final segmenter `{manifest['final_segmenter']}`"
    )
    add("")
    add(
        "**Every number here is a validation number.** The holdout has never been evaluated, "
        "and nothing below says anything about test performance. S1 is a controlled candidate, "
        "not the project's final segmenter."
    )
    add("")
    if commit:
        add(f"Repository commit at production time: `{commit}`.")
        add("")

    add("## 1. Experimental question")
    add("")
    add(f"> {manifest['experiment_question']}")
    add("")
    add(f"`{manifest['experiment_role']}`.")
    add("")

    add("## 2. Phase 8D motivation")
    add("")
    add(f"`PREDECLARED_S1_HYPOTHESIS`. {manifest['hypothesis']}")
    add("")

    add("## 3. Phase 8E common evaluator")
    add("")
    add(
        "`CANONICAL_PRIMARY_METRIC`. Both experiments are scored against the canonical phase 5D "
        "COCO validation masks with `pycocotools` `COCOeval` at `iouType='segm'`, IoU "
        "0.50:0.05:0.95, `maxDets` [1, 10, 100], conf 0.001, NMS IoU 0.70, imgsz 768, "
        "`retina_masks: true`, no TTA. Evaluator fingerprint "
        f"`{manifest['canonical_evaluator_fingerprint']}`, unchanged from phase 8E and verified "
        "before S1 trained."
    )
    add("")
    add(
        "conf 0.001 is **not** an operating point: average precision integrates over the score "
        "curve and needs the low-scoring tail. The direct-IoU diagnostic keeps its own frozen "
        "operational 0.25, and the two protocols are never mixed."
    )
    add("")

    add("## 4. One-variable contract")
    add("")
    add(
        f"`ONE_VARIABLE_INTERVENTION`. S1 carries no protocol of its own. Its framework "
        f"arguments are resolved from `{difference['resolved_from']}` with the single override "
        f"declared in `{difference['declared_in']}`: **`{difference['field']}` "
        f"{str(difference['reference_value']).lower()} -> "
        f"{str(difference['candidate_value']).lower()}**. "
        f"{difference['inherited_count']} other framework arguments are inherited unchanged, "
        "and the parser refuses any second override."
    )
    add("")
    add("| Held constant | Value |")
    add("| --- | --- |")
    for name in ("model", "imgsz", "batch", "epochs", "seed", "mask_ratio", "patience"):
        value = manifest["effective_training_configuration"].get(name)
        add(f"| {name} | {value} |")
    add(f"| pretrained weights | `{weights['identifier']}` SHA-256 `{weights['sha256']}` |")
    add(f"| adapter label bytes | identical to S0, `{adapter['identical_to_reference']}` |")
    add(f"| checkpoint policy | `{selection['policy']}` |")
    add("")

    add("## 5. Training target difference")
    add("")
    add(f"`LIMITATION`. {manifest['treatment_semantics']}")
    add("")

    add("## 6. Runtime provenance")
    add("")
    add(
        f"Adapter fingerprints re-verified before and after training against the phase 8A "
        f"approved digests: {adapter['counts']['instances']} instances over "
        f"{adapter['counts']['train_images']} train and "
        f"{adapter['counts']['validation_images']} validation images, "
        f"`regenerated: {adapter['regenerated']}`, `filtered: {adapter['filtered']}`. Training "
        "read a hard-linked runtime view of those bytes so the framework's `.cache` files did "
        "not land inside phase 8A's evidence."
    )
    add("")
    add(
        f"Every phase 8A-8E artifact was digested before and after this phase and is "
        f"byte-identical: `{manifest['historical_artifacts_unchanged']}`, "
        f"{len(manifest['historical_artifact_digests'])} artifacts. S0 was not retrained, not "
        "revalidated and not recomputed; its committed numbers are quoted."
    )
    add("")

    add("## 7. Effective training configuration")
    add("")
    add("| Field | Value |")
    add("| --- | --- |")
    for name in sorted(manifest["effective_training_configuration"]):
        add(f"| {name} | {manifest['effective_training_configuration'][name]} |")
    add("")
    add(
        f"`DECLARED_OPTIMIZER_POLICY` `{optimizer['declared_optimizer_policy']}` resolved to "
        f"`ACTUAL_RESOLVED_OPTIMIZER` **{optimizer['actual_resolved_optimizer']}** at lr0 "
        f"{optimizer['effective_lr0']}, momentum {optimizer['effective_momentum']}. "
        f"`RESOLUTION_EVIDENCE` `{optimizer['resolution_evidence']}`. Weight decay "
        f"{optimizer['weight_decay']}, warmup {optimizer['warmup_epochs']} epochs, scheduler "
        f"`{optimizer['scheduler']}`, final LR factor {optimizer['lrf']}."
    )
    add("")
    add(f"{optimizer['declared_values_are_not_what_ran']}")
    add("")

    add("## 8. Checkpoint-selection limitation")
    add("")
    add(
        f"`LIMITATION`. S1 keeps S0's rule, `{selection['policy']}` "
        f"(`{selection['semantics']}`), unchanged. Because `overlap_mask` alters the native "
        "target, each experiment's fitness is computed against its own target, so the two "
        "checkpoints are selected under different fitness definitions. That is a genuine "
        "limitation of the comparison, not a detail - and it is exactly why the decision is "
        "made externally, against canonical masks. No mask-only checkpoint selector was "
        f"created: `mask_only_checkpoint_created: {selection['mask_only_checkpoint_created']}`."
    )
    add("")

    add("## 9. Training execution")
    add("")
    add(
        f"`COMPUTED_RESULT`. One run, {execution['epochs_completed']}/"
        f"{execution['epochs_configured']} epochs, termination "
        f"`{execution['termination']}`, best epoch **{selection['best_epoch']}** at native "
        f"composite fitness **{selection['best_native_fitness']}**, verified as the argmax of "
        "the composite recomputed from `results.csv`. Engineering aborts: "
        f"{execution['engineering_aborts']}. Resumed: {execution['resumed']}."
    )
    add("")
    add(
        f"`best.pt` SHA-256 `{manifest['checkpoints']['best']['sha256']}`, "
        f"{manifest['checkpoints']['best']['size_bytes']} bytes; `last.pt` "
        f"`{manifest['checkpoints']['last']['sha256']}`, "
        f"{manifest['checkpoints']['last']['size_bytes']} bytes. Neither is committed."
    )
    add("")

    add("## 10. Native framework results")
    add("")
    add(f"`{native['label']}`. Validation split, one run, `overlap_mask: false` passed explicitly.")
    add("")
    add("| Family | precision | recall | mAP@0.50 | mAP@0.50:0.95 |")
    add("| --- | --- | --- | --- | --- |")
    for family_name, family in (("mask", native["mask"]), ("box", native["box"])):
        add(
            f"| {family_name} | {family['precision']} | {family['recall']} | "
            f"{family['mAP@0.50']} | {family['mAP@0.50:0.95']} |"
        )
    add("")
    add("| Class | native mask AP@0.50 | native mask AP@0.50:0.95 |")
    add("| --- | --- | --- |")
    for name in sorted(native["per_class"]):
        row = native["per_class"][name]["mask"]
        add(f"| {name} | {row['AP@0.50']} | {row['AP@0.50:0.95']} |")
    add("")
    add(
        f"Descriptive native supported macro: "
        f"**{manifest['descriptive_native_supported_macro']['value']}** "
        f"(`{manifest['descriptive_native_supported_macro']['label']}`, over "
        f"{manifest['descriptive_native_supported_macro']['admitted_classes']}). "
        f"{manifest['descriptive_native_supported_macro']['reason']}"
    )
    add("")

    add("## 11. Why native AP is not the primary cross-target metric")
    add("")
    add(f"`{native['cross_target_status']}`. {native['why_not_primary']}")
    add("")

    add("## 12. Canonical S1 results")
    add("")
    add(
        f"`CANONICAL_PRIMARY_METRIC`. One evaluation, "
        f"{canonical['detections_scored']} detections scored."
    )
    add("")
    add(f"* {ALL_CLASS_METRIC}: **{canonical[ALL_CLASS_METRIC]}**")
    add(f"* {ALL_CLASS_METRIC_50}: **{canonical[ALL_CLASS_METRIC_50]}**")
    add("")

    add("## 13. Canonical S0-versus-S1 comparison")
    add("")
    add("| Class | S0 AP@0.50:0.95 | S1 AP@0.50:0.95 | delta | support |")
    add("| --- | --- | --- | --- | --- |")
    for name in sorted(manifest["canonical_per_class_comparison"]):
        row = manifest["canonical_per_class_comparison"][name]
        add(
            f"| {name} | {row['S0_AP@0.50:0.95']} | {row['S1_AP@0.50:0.95']} | "
            f"{row['delta_AP@0.50:0.95']} | `{row['status']}` |"
        )
    add("")
    add(
        f"All-class mAP@0.50:0.95: S0 {manifest['canonical_all_class_delta']['S0']} -> S1 "
        f"{manifest['canonical_all_class_delta']['S1']}, delta "
        f"{manifest['canonical_all_class_delta']['delta']}. This is reported, not used to rank."
    )
    add("")
    composition = manifest["aggregate_composition"]
    add(
        f"**The aggregate is not a uniform effect.** `{composition['largest_gain']}` moved "
        f"{composition['largest_gain_delta']:+f} and carries "
        f"{composition['dominant_class_share_of_total_gain']:.1%} of the total gain across "
        f"admitted classes, while `{composition['largest_decline']}` moved "
        f"{composition['largest_decline_delta']:+f}."
    )
    add("")
    if composition["regressed_supported_classes"]:
        add(
            "A supported class **regressed**: "
            + ", ".join(f"`{name}`" for name in composition["regressed_supported_classes"])
            + ". It is named here rather than left inside the mean. Each admitted class "
            "contributes its own delta divided by the four admitted classes: "
            + ", ".join(
                f"`{name}` {value:+f}"
                for name, value in sorted(composition["contribution_to_primary_delta"].items())
            )
            + "."
        )
        add("")
    add(f"`LIMITATION`. {composition['note']}")
    add("")

    add("## 14. Primary supported macro")
    add("")
    add(
        f"`CANONICAL_PRIMARY_METRIC` `{macro['metric']}`, the unweighted mean canonical mask "
        f"AP@0.50:0.95 over the classes the frozen phase 7A support rule admits "
        f"({macro['admitted_classes']}). The rule names no class; the admitted set is its "
        "output."
    )
    add("")
    add(f"* S0 (committed phase 8E reference): **{reference['canonical_supported_macro']}**")
    add(f"* S1: **{macro['value']}**")
    add(f"* delta: **{delta['delta']}**")
    add("")

    add("## 15. Margin classification")
    add("")
    add(
        f"`COMPUTED_RESULT`. Frozen margin {delta['margin']} absolute AP. Classification: "
        f"**`{delta['verdict']}`**."
    )
    add("")
    add(
        "The margin is an engineering practical-equivalence threshold, **not a significance "
        "test**: nothing is repeated, so run-to-run variance on this setup remains UNKNOWN. "
        "Under the frozen rule, `PRACTICALLY_EQUIVALENT` prefers S0, decided in advance."
    )
    add("")

    add("## 16. Person diagnostic")
    add("")
    add(
        f"`{person['status']}`. Canonical AP@0.50:0.95 S0 {person['canonical']['S0_AP@0.50:0.95']} "
        f"-> S1 {person['canonical']['S1_AP@0.50:0.95']}, delta "
        f"{person['canonical']['delta_AP@0.50:0.95']}."
    )
    add("")
    add("| Direct-IoU figure | S0 | S1 | delta |")
    add("| --- | --- | --- | --- |")
    for name in sorted(person["direct_iou"]):
        row = person["direct_iou"][name]
        add(f"| {name} | {row['S0']} | {row['S1']} | {row['delta']} |")
    add("")
    add(f"`LIMITATION`. {person['caveat']} It may not select the final model.")
    add("")

    add("## 17. Per-class canonical comparison")
    add("")
    add("| Class | S0 AP@0.50 | S1 AP@0.50 | delta |")
    add("| --- | --- | --- | --- |")
    for name in sorted(manifest["canonical_per_class_comparison"]):
        row = manifest["canonical_per_class_comparison"][name]
        add(f"| {name} | {row['S0_AP@0.50']} | {row['S1_AP@0.50']} | {row['delta_AP@0.50']} |")
    add("")

    add("## 18. Direct mask-IoU")
    add("")
    add(
        f"`DIRECT_IOU_DIAGNOSTIC`. One run under the unchanged phase 8C protocol "
        f"(`{direct['protocol_fingerprint']}`): canonical COCO ground truth, conf 0.25, NMS IoU "
        "0.70, imgsz 768, max_det 300, per-image per-class one-to-one Hungarian matching "
        "maximising total IoU. No threshold was swept."
    )
    add("")
    add("| Figure | S0 | S1 | delta |")
    add("| --- | --- | --- | --- |")
    for name in (
        MATCHED_MASK_IOU_MEAN,
        GT_NORMALIZED_MASK_IOU,
        GT_MATCH_COVERAGE,
        GT_IOU50_COVERAGE,
        GT_IOU75_COVERAGE,
    ):
        add(
            f"| {name} | {reference['direct_iou_global'][name]} | {direct['headline'][name]} | "
            f"{direct['deltas_vs_reference'][name]} |"
        )
    add("")
    add(f"`LIMITATION`. {direct['is_not_average_precision']}")
    add("")

    add("## 19. Cross-metric direction")
    add("")
    add(
        f"`COMPUTED_RESULT`. Primary delta {direction['primary_delta']}, secondary "
        f"(GT-normalised direct mask IoU) delta {direction['secondary_delta']}: "
        f"**`{direction['status']}`**."
    )
    add("")
    add(
        f"Rule, frozen before S1 ran: {direction['rule']} Ranked by "
        f"`{direction['ranked_by']}`; no composite was created "
        f"(`composite_created: {direction['composite_created']}`)."
    )
    add("")

    add("## 20. Training dynamics")
    add("")
    add(
        f"`COMPUTED_RESULT`. {curves.get('epochs_logged')} epochs logged. Native composite "
        f"fitness first {curves.get('native_fitness_first')}, best "
        f"{curves.get('native_fitness_best')} at epoch {curves.get('best_epoch')}, last "
        f"{curves.get('native_fitness_last')}; {curves.get('epochs_after_best')} epochs ran "
        f"after the best without exceeding it "
        f"(`improved_after_best: {curves.get('improved_after_best')}`)."
    )
    add("")
    add("| Loss / LR column | first epoch | last epoch |")
    add("| --- | --- | --- |")
    for name in curves.get("loss_and_lr_columns", []):
        add(f"| {name} | {curves['first_epoch'][name]} | {curves['last_epoch'][name]} |")
    add("")
    add(f"{curves.get('observation', '')}")
    add("")

    add("## 21. Resource facts")
    add("")
    add("| Fact | Value |")
    add("| --- | --- |")
    add(f"| Training wall clock (s) | {resources['training_wall_clock_seconds']} |")
    add(f"| results.csv cumulative time (s) | {resources['training_seconds_from_results_csv']} |")
    add(f"| GPU | {resources['gpu_name']} ({resources['gpu_arch']}) |")
    add(
        f"| Epochs configured / completed | {resources['epochs_configured']} / "
        f"{resources['epochs_completed']} |"
    )
    add(f"| Best epoch | {resources['best_epoch']} |")
    add(f"| batch / imgsz | {resources['batch']} / {resources['imgsz']} |")
    add(f"| Peak GPU memory reserved (GiB) | {resources['peak_gpu_memory_reserved_gib']} |")
    add(f"| Parameters / GFLOPs | {resources['parameters']} / {resources['gflops']} |")
    add(
        f"| Fused parameters / GFLOPs | {resources['fused_parameters']} / "
        f"{resources['fused_gflops']} |"
    )
    add("")
    add(
        f"`FRAMEWORK_VALIDATION_SPEED` "
        f"{resources['framework_validation_speed_ms_per_image']} ms per image. Measured by the "
        "framework during validation, not a production latency benchmark: a standardised "
        "detector-versus-segmenter latency study is "
        f"`{resources['standardised_latency_benchmark']}`."
    )
    add("")

    add("## 22. vest_loose limitation")
    add("")
    add(f"`LIMITATION` `{rare['status']}`. {rare['limitation']}")
    add("")
    add(
        f"It is reported in full in every table above and decides nothing: "
        f"`may_decide_anything: {rare['may_decide_anything']}`."
    )
    add("")

    add("## 23. Holdout compliance")
    add("")
    add(f"`HOLDOUT_POLICY` `{manifest['test']['status']}`. {manifest['test']['reason']}")
    add("")

    add("## 24. Interpretation limits")
    add("")
    add("`LIMITATION`.")
    add("")
    add(
        "* Every figure is a **validation** figure. Nothing here says anything about test "
        "performance."
    )
    add(
        "* One run each. Run-to-run variance on this setup is UNKNOWN, because nothing was "
        "repeated. The margin is an engineering threshold, not a significance test."
    )
    add(
        "* The comparison protocol was frozen **after** S0 ran "
        "(`POST_S0_PRE_S1_PROTOCOL_FREEZE`), unlike phase 7A's, which predated its candidates. "
        "No S1 number influenced any rule, but the asymmetry is real and is recorded rather "
        "than smoothed over."
    )
    add(
        "* The two experiments' checkpoints were selected by the same policy computed against "
        "**different targets**. That is part of the treatment, and it is why the decision is "
        "external."
    )
    add(
        "* The phase 8D analysis that motivated S1 is `POST_HOC_HYPOTHESIS_GENERATING`. A "
        "movement consistent with it is not confirmation of the mechanism."
    )
    add(
        "* Native and canonical numbers are **not** comparable and must never be differenced: "
        "different ground truth, different implementation, different confidence."
    )
    add("")

    add("## 25. Final segmenter")
    add("")
    add(
        f"`PENDING_HUMAN_REVIEW`. `final_segmenter: {manifest['final_segmenter']}`. "
        f"{manifest['final_segmenter_note']} No further segmentation experiment is authorised: "
        "no S1 re-run, no `mask_ratio` variant, no YOLO11s-seg, no resolution or batch change, "
        "no threshold tuning and no change to the canonical evaluator."
    )
    add("")

    return "\n".join(lines) + "\n"


# --- entry point --------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """Run the S1 controlled experiment once and report it honestly.

    Args:
        argv: Command-line arguments.

    Returns:
        Process exit status.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="run every precondition check without training, evaluating or writing",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="re-run the four validators against the committed S1 artifacts and exit",
    )
    parser.add_argument(
        "--from-completed-run",
        action="store_true",
        help=(
            "reuse the completed S1 training run instead of training, re-executing the "
            "downstream evaluations deterministically on the same frozen checkpoint. For "
            "recovering from an artifact-write failure or re-rendering an artifact. It never "
            "trains, never changes a metric and never produces a second experiment."
        ),
    )
    parser.add_argument(
        "--reuse-reason",
        default="",
        help="why the completed run is being reused, recorded verbatim in the manifest",
    )
    args = parser.parse_args(argv)

    paths = ProjectPaths.from_root()
    if holdout_unlocked():
        print(
            f"REFUSED: {HOLDOUT_UNLOCK_ENV_VAR} is set. This phase reads development data only "
            "and declines to run in an environment where the holdout is unlocked.",
            file=sys.stderr,
        )
        return 2

    started_at = datetime.now(UTC).isoformat(timespec="seconds")

    # --- preflight -------------------------------------------------------------
    try:
        historical = verify_historical(paths)
        baseline = load_segmentation_baseline_config(paths.configs / BASELINE_YAML)
        comparison = load_comparison_config(paths.configs / COMPARISON_YAML)
        canonical_protocol = load_canonical_evaluation_config(paths.configs / CANONICAL_YAML)
        direct_protocol = load_mask_iou_config(paths.configs / MASK_IOU_YAML)
        policies = verify_policies(paths, baseline, comparison, canonical_protocol, direct_protocol)

        resolution = resolve_candidate_arguments(
            baseline.training_arguments(), comparison.candidate["overrides"]
        )
        arguments, contract = resolution["arguments"], resolution["contract"]
        protocol_manifest = read_json(paths.reports / S1_PROTOCOL_JSON)
        inheritance = verify_inherited_protocol(arguments, protocol_manifest["inherited_protocol"])

        detector = verify_detector(paths)
        verify_no_holdout(paths, baseline)
        adapter = verify_adapter(paths, baseline, stage="preflight")
        approval = read_json(paths.reports / APPROVAL_JSON)
        approval_sha256 = sha256_file(paths.reports / APPROVAL_JSON)
        if approval["status"] != "APPROVED_FOR_CONTROLLED_TRAINING":
            msg = f"the adapter is not approved for training: {approval['status']}"
            raise ExperimentError(msg)

        audit = read_json(paths.reports / AUDIT_MANIFEST_JSON)
        class_map = {name: int(index) for name, index in audit["class_map"].items()}
        canonical_fingerprints = {
            "class_map_sha256": str(audit["class_map_sha256"]),
            "split_assignment_sha256": str(
                audit["canonical_fingerprints"]["split_assignment_sha256"]
            ),
            "canonical_task_manifest_sha256": str(audit["canonical_task_manifest_sha256"]),
        }
        for protocol in (canonical_protocol, direct_protocol):
            if protocol["class_map_sha256"] != canonical_fingerprints["class_map_sha256"]:
                msg = "a frozen protocol's class map digest disagrees with the frozen record"
                raise ExperimentError(msg)
            if (
                protocol["split_assignment_sha256"]
                != canonical_fingerprints["split_assignment_sha256"]
            ):
                msg = "a frozen protocol's split digest disagrees with the frozen record"
                raise ExperimentError(msg)

        reference = verify_s0_reference(paths)

        document = load_canonical_validation(
            paths, str(canonical_protocol["ground_truth_document"])
        )
        if str(direct_protocol["ground_truth_document"]) != document["relative_path"]:
            msg = "the two frozen protocols name different canonical ground-truth documents"
            raise ExperimentError(msg)
        canonical_fingerprints["validation_annotations_sha256"] = document["sha256"]

        ground_truth = canonical_instances(document["document"], class_map)
        decoded = sum(len(items) for items in ground_truth.values())
        if decoded != adapter["counts"]["validation_instances"]:
            msg = (
                f"the canonical validation document decodes {decoded} instances but the frozen "
                f"adapter records {adapter['counts']['validation_instances']}"
            )
            raise ExperimentError(msg)
        support = canonical_validation_support(document["document"], class_map)
    except (
        ConfigError,
        ComparisonConfigError,
        SegmentationExperimentConfigError,
        SegmentationRunError,
        SegmentationS1Error,
        ExperimentError,
    ) as exc:
        if isinstance(exc, AdapterMismatchError):
            classification = ADAPTER_FINGERPRINT_MISMATCH
        elif isinstance(exc, (ProtocolMismatchError, ComparisonConfigError, SegmentationS1Error)):
            classification = S1_PROTOCOL_INVALID
        else:
            classification = BLOCKED
        print(f"{classification}: {exc}", file=sys.stderr)
        return 2

    # --- the validators, on their own ------------------------------------------
    if args.validate_only:
        try:
            manifest = read_json(paths.reports / RESULT_MANIFEST_JSON)
            canonical_payload = read_json(paths.reports / CANONICAL_JSON)
            direct_payload = read_json(paths.reports / MASK_IOU_JSON)
        except ExperimentError as exc:
            print(f"{BLOCKED}: {exc}", file=sys.stderr)
            return 2
        checks = {
            "S1 result validator": validate_s1_result_manifest(manifest),
            "canonical evaluator validator": validate_canonical_evaluation(
                canonical_payload,
                protocol_fingerprint=policies["canonical_evaluator_fingerprint"],
            ),
            "direct-IoU validator": validate_direct_iou(
                direct_payload,
                protocol_fingerprint=policies["direct_iou_protocol_fingerprint"],
            ),
            "S0/S1 comparison validator": validate_comparison(
                manifest, s0_reference=read_json(paths.reports / S0_CANONICAL_JSON)
            ),
        }
        failed = False
        for name, problems in checks.items():
            if problems:
                failed = True
                print(f"{name}: FAILED")
                for problem in problems:
                    print(f"  - {problem}")
            else:
                print(f"{name}: PASS")
        print(f"holdout    {HOLDOUT_STATUS}")
        return 2 if failed else 0

    try:
        runtime = runtime_facts()
    except RunError as exc:
        print(f"{BLOCKED}: {exc}", file=sys.stderr)
        return 2

    print(
        f"protocol   {EXPERIMENT_ID} inherits {REFERENCE}  one variable: {ONE_VARIABLE_FIELD} "
        f"{contract['reference_value']} -> {contract['candidate_value']}  "
        f"{contract['inherited_count']} fields inherited  VERIFIED"
    )
    print(f"comparison {policies['comparison_protocol_fingerprint']}  margin {policies['margin']}")
    print(f"evaluator  {policies['canonical_evaluator_fingerprint']}  conf 0.001")
    print(f"direct IoU {policies['direct_iou_protocol_fingerprint']}  conf 0.25")
    print(f"adapter    {adapter['counts']['instances']} instances  fingerprints VERIFIED")
    print(f"S0 ckpt    {reference['checkpoint_sha256']}  VERIFIED (read, not executed)")
    print(f"S0 primary {reference['canonical']['supported_macro']}  (committed phase 8E)")
    print(f"detector   {detector['selected_experiment']}  {detector['status']}")
    print(f"canonical  {len(ground_truth)} validation images  {decoded} GT instances")
    print(f"support    admitted {[n for n in sorted(support) if support[n]['supported']]}")
    print(
        f"runtime    torch {runtime['torch']}  ultralytics {runtime['ultralytics']}  "
        f"{runtime['gpu_name']}"
    )

    if args.verify_only:
        print("VERIFIED: preconditions hold. Nothing trained, evaluated or written.")
        print(f"holdout    {HOLDOUT_STATUS}")
        return 0

    try:
        weights = ensure_pretrained_weights(paths, baseline.weight_identifier)
    except RunError as exc:
        print(f"{BLOCKED}: {exc}", file=sys.stderr)
        return 2
    if weights["sha256"] != EXPECTED_PRETRAINED or weights["size_bytes"] != (
        EXPECTED_PRETRAINED_BYTES
    ):
        print(
            f"{BLOCKED}: S1 would not start from the same pretrained binary S0 used: observed "
            f"{weights['sha256']} ({weights['size_bytes']} bytes)",
            file=sys.stderr,
        )
        return 2
    print(f"weights    {weights['identifier']}  sha256 {weights['sha256']}  VERIFIED")

    # --- runtime view -----------------------------------------------------------
    # Rebuilt from the approved adapter unless a completed run is being reused, in which
    # case the view the run actually read is kept and re-verified rather than replaced.
    try:
        if args.from_completed_run and (paths.root / RUNTIME_VIEW_ROOT).is_dir():
            runtime_view = {
                "root": Path(RUNTIME_VIEW_ROOT).name,
                "descriptor": "dataset.yaml",
                "committed": False,
                "image_transformation": "NONE",
                "label_conversion": "NONE",
                "transfer_modes": ["REUSED_EXISTING_VIEW"],
                "image_counts": {
                    "train": adapter["counts"]["train_images"],
                    "validation": adapter["counts"]["validation_images"],
                },
            }
        else:
            runtime_view = build_runtime_view(
                source_root=adapter["root"],
                destination_root=paths.root / RUNTIME_VIEW_ROOT,
                class_map=class_map,
            )
        view_labels = read_adapter_labels(paths.root / RUNTIME_VIEW_ROOT)
        view_fingerprints = adapter_fingerprints(view_labels)
        drifted = sorted(
            name
            for name, value in view_fingerprints.items()
            if value != adapter["fingerprints"][name]
        )
        if drifted:
            msg = (
                f"{ADAPTER_FINGERPRINT_MISMATCH}: the runtime view's labels differ from the "
                f"approved bytes: {drifted}"
            )
            raise AdapterMismatchError(msg)
        if adapter_counts(view_labels) != adapter["counts"]:
            msg = f"{ADAPTER_FINGERPRINT_MISMATCH}: the runtime view's cardinality differs"
            raise AdapterMismatchError(msg)
    except (SegmentationRunError, ExperimentError) as exc:
        classification = (
            ADAPTER_FINGERPRINT_MISMATCH if isinstance(exc, AdapterMismatchError) else BLOCKED
        )
        print(f"{classification}: {exc}", file=sys.stderr)
        return 2
    runtime_view["labels_verified_against_approved_digests"] = True
    print(f"view       {RUNTIME_VIEW_ROOT}  {runtime_view['transfer_modes']}  labels VERIFIED")

    descriptor = paths.root / RUNTIME_VIEW_ROOT / "dataset.yaml"
    run_directory = paths.root / RUN_ROOT / RUN_NAME

    hypothesis = str(protocol_manifest["motivation"])

    # --- pre-run provenance ------------------------------------------------------
    prerun = ProvenanceRecord.create(
        name="segmentation_S1_prerun",
        phase=8,
        config={
            "segmentation_baseline": f"configs/{BASELINE_YAML}",
            "segmentation_comparison": f"configs/{COMPARISON_YAML}",
            "canonical_evaluation": f"configs/{CANONICAL_YAML}",
            "direct_mask_iou_protocol": f"configs/{MASK_IOU_YAML}",
        },
        details={
            "phase": PHASE,
            "experiment_id": EXPERIMENT_ID,
            "state": "BEFORE_FIRST_OPTIMIZATION_STEP",
            "started_at": started_at,
            "intentional_difference": (
                f"{ONE_VARIABLE_FIELD} {contract['reference_value']} -> "
                f"{contract['candidate_value']}"
            ),
            "one_variable_contract": contract,
            "inherited_protocol_verified": inheritance,
            "frozen_policies": policies,
            "adapter_approval_sha256": approval_sha256,
            "adapter_fingerprints": adapter["fingerprints"],
            "adapter_counts": adapter["counts"],
            "canonical_fingerprints": canonical_fingerprints,
            "pretrained_weight_sha256": weights["sha256"],
            "pretrained_weight_bytes": weights["size_bytes"],
            "checkpoint_selection_policy": CHECKPOINT_POLICY,
            "checkpoint_selection_semantics": CHECKPOINT_SELECTION_SEMANTICS,
            "reference_checkpoint_sha256": reference["checkpoint_sha256"],
            "reference_canonical_supported_macro": reference["canonical"]["supported_macro"],
            "effective_training_arguments": dict(sorted(arguments.items())),
            "runtime": runtime,
            "historical_artifact_digests": historical,
            "holdout": HOLDOUT_STATUS,
            "models_trained_so_far": 0,
        },
        repo_root=paths.root,
    )
    for name in (BASELINE_YAML, COMPARISON_YAML, CANONICAL_YAML, MASK_IOU_YAML):
        prerun.add_input(paths.configs / name, relative_to=paths.root)
    prerun.add_input(paths.reports / S1_PROTOCOL_JSON, relative_to=paths.root)
    prerun.write_json(paths.reports / PRERUN_PROVENANCE_JSON)
    print(f"prerun     reports/{PRERUN_PROVENANCE_JSON}  written before the first step")

    # --- the single training run --------------------------------------------------
    #
    # `--from-completed-run` exists for one situation, the one phase 8C actually
    # hit: everything ran, and the artifact write was refused at the last step.
    # Re-training to recover from that would replace the experiment with a
    # different one. Validation, the canonical evaluation and the diagnostic are
    # deterministic functions of a fixed checkpoint, fixed data and fixed
    # settings, so re-executing them reproduces the same numbers; what changes is
    # only whether they can be written down. It is never a second experiment, and
    # it is never a way to retry a run whose numbers were disliked.
    if args.from_completed_run:
        if not (run_directory / "weights" / "best.pt").is_file():
            print(
                f"{BLOCKED}: --from-completed-run needs a completed S1 run at "
                f"{RUN_ROOT}/{RUN_NAME}",
                file=sys.stderr,
            )
            return 2
        log_path = run_directory / "framework.log"
        execution = {
            "run_directory": f"{RUN_ROOT}/{RUN_NAME}",
            "wall_clock_seconds": None,
            "peak_gpu_memory_reserved_bytes": "NOT_PERSISTED_FOR_THIS_RUN",
            "peak_gpu_memory_reserved_gib": "NOT_PERSISTED_FOR_THIS_RUN",
            "peak_gpu_memory_allocated_bytes": "NOT_PERSISTED_FOR_THIS_RUN",
            "peak_gpu_memory_source": "NOT_MEASURED_IN_THIS_PROCESS",
            "log_text": (
                log_path.read_text(encoding="utf-8", errors="replace") if log_path.is_file() else ""
            ),
        }
        # Peak device memory and wall clock are measured inside the training
        # process and cannot be recovered once it has exited. Re-rendering must
        # not silently downgrade them to "not persisted": that would destroy real
        # evidence in order to make a prose change. They are carried forward from
        # the manifest the measuring process itself wrote - an artifact on disk,
        # not terminal scrollback - and labelled as carried rather than measured
        # here.
        previous_path = paths.reports / RESULT_MANIFEST_JSON
        if previous_path.is_file():
            try:
                previous = read_json(previous_path)["execution"]
            except (ExperimentError, KeyError):
                previous = {}
            carried = [
                name
                for name in (
                    "wall_clock_seconds",
                    "peak_gpu_memory_reserved_bytes",
                    "peak_gpu_memory_reserved_gib",
                    "peak_gpu_memory_allocated_bytes",
                )
                if isinstance(previous.get(name), (int, float))
            ]
            for name in carried:
                execution[name] = previous[name]
            if carried:
                execution["peak_gpu_memory_source"] = previous.get(
                    "peak_gpu_memory_source", NOT_EXPOSED
                )
                execution["measured_fields_carried_forward"] = sorted(carried)
                execution["measured_fields_carried_forward_source"] = (
                    "THE_TRAINING_RUNS_OWN_RESULT_MANIFEST"
                )
        reuse = {
            "occurred": True,
            "what_was_reexecuted": [
                "the authoritative native validation",
                "the canonical COCO evaluation",
                "the direct mask-IoU diagnostic",
            ],
            "what_was_not_reexecuted": (
                "training - the single S1 run and its weights are unchanged, and the "
                "checkpoint the frozen rule selected is the same file"
            ),
            "why_this_is_not_a_second_experiment": (
                "Validation, the canonical evaluation and the diagnostic are deterministic "
                "functions of a fixed checkpoint, fixed data and fixed settings, all of which "
                "are unchanged."
            ),
            "reason": args.reuse_reason.strip() or "NOT_STATED",
        }
        print("training   REUSED the completed S1 run (nothing was trained)")
    else:
        try:
            execution = train(paths, arguments, weights, descriptor)
        except MemoryConstraintError as exc:
            print(f"{MEMORY_CONSTRAINT_REVIEW_REQUIRED}: {exc}", file=sys.stderr)
            return 2
        except ExperimentError as exc:
            print(f"{TRAINING_FAILED}: {exc}", file=sys.stderr)
            return 2
        reuse = {"occurred": False}
        print(
            f"training   COMPLETE  {execution['wall_clock_seconds']} s  peak "
            f"{execution['peak_gpu_memory_reserved_gib']} GiB"
        )
    execution["reused_completed_run"] = bool(args.from_completed_run)
    execution["artifact_write_reexecution"] = reuse

    # --- checkpoint selection -------------------------------------------------------
    try:
        history = read_epoch_history(run_directory)
        resolved = resolved_training_arguments(run_directory, paths.root)
        selection = verify_checkpoint_selection(history, resolved)
        checkpoints = {
            "best": checkpoint_record(run_directory / "weights" / "best.pt"),
            "last": checkpoint_record(run_directory / "weights" / "last.pt"),
        }
    except (SegmentationRunError, ExperimentError) as exc:
        print(f"{TRAINING_FAILED}: {exc}", file=sys.stderr)
        return 2

    if resolved.get(ONE_VARIABLE_FIELD) is not False:
        print(
            f"{PROTOCOL_VIOLATION}: the framework recorded {ONE_VARIABLE_FIELD}="
            f"{resolved.get(ONE_VARIABLE_FIELD)!r} for this run, not False. The intervention "
            "did not take effect and the run is not S1.",
            file=sys.stderr,
        )
        return 2
    if resolved.get("mask_ratio") != 4:
        print(
            f"{PROTOCOL_VIOLATION}: mask_ratio resolved to {resolved.get('mask_ratio')!r}, "
            "not the inherited 4",
            file=sys.stderr,
        )
        return 2

    execution.update(training_duration(history, execution.get("log_text", "")))
    configured = int(arguments["epochs"])
    logged = selection["epochs_logged"]
    early_stopped = logged < configured
    execution["termination"] = (
        "EARLY_STOPPING_AT_FROZEN_PATIENCE" if early_stopped else "ALL_EPOCHS_COMPLETED"
    )
    execution["early_stopping"] = early_stopped
    execution["epochs_configured"] = configured
    execution["epochs_completed"] = logged
    execution["engineering_aborts"] = 0
    execution["resumed"] = False
    # Two different facts, kept apart because one field cannot carry both: the
    # training run itself was a single fresh run (always true - a completed run is
    # never overwritten and resume is not authorised here), and this particular
    # invocation may have reused it to re-render an artifact without training.
    execution["training_was_a_fresh_single_run"] = True
    execution["this_invocation_trained"] = not bool(args.from_completed_run)
    execution["runs"] = 1
    print(
        f"checkpoint best epoch {selection['best_epoch']}  native fitness "
        f"{selection['best_native_fitness']}  {execution['termination']}"
    )

    optimizer = optimizer_evidence_from_log(execution.get("log_text", ""))

    # --- authoritative native validation ------------------------------------------
    try:
        native = validate_native(paths, arguments, run_directory / "weights" / "best.pt")
    except ExperimentError as exc:
        print(f"{TRAINING_FAILED}: {exc}", file=sys.stderr)
        return 2
    if native["effective_arguments"]["overlap_mask"] is not False:
        print(
            f"{PROTOCOL_VIOLATION}: the native validation resolved overlap_mask to "
            f"{native['effective_arguments']['overlap_mask']!r}; S1 would have been scored "
            "against S0's target",
            file=sys.stderr,
        )
        return 2
    print(
        f"native     mask mAP@0.50:0.95 {native['mask']['mAP@0.50:0.95']}  "
        f"box mAP@0.50:0.95 {native['box']['mAP@0.50:0.95']}  (NATIVE_TARGET_METRIC)"
    )

    native_macro = native_supported_macro(native["per_class"], support)
    print(f"native macro {native_macro['value']}  (descriptive, decides nothing)")

    # --- the single canonical evaluation --------------------------------------------
    try:
        canonical = run_canonical(
            paths,
            canonical_protocol,
            run_directory / "weights" / "best.pt",
            document["document"],
            class_map,
        )
    except CanonicalEvaluationError as exc:
        print(f"{CANONICAL_EVALUATION_FAILED}: {exc}", file=sys.stderr)
        return 2
    except ExperimentError as exc:
        print(f"{CANONICAL_EVALUATION_FAILED}: {exc}", file=sys.stderr)
        return 2

    macro = canonical_supported_macro(canonical["per_class"], support)
    delta = classify_delta(float(reference["canonical"]["supported_macro"]), float(macro["value"]))
    print(
        f"canonical  all-class mAP@0.50:0.95 {canonical['all_class_map50_95']}  "
        f"mAP@0.50 {canonical['all_class_map50']}  supported macro {macro['value']}"
    )
    print(
        f"primary    delta {delta['delta']} vs S0 "
        f"{reference['canonical']['supported_macro']}  -> {delta['verdict']}"
    )

    # --- the single direct mask-IoU diagnostic ----------------------------------------
    try:
        direct = run_direct_iou(
            paths,
            direct_protocol,
            run_directory / "weights" / "best.pt",
            ground_truth,
            class_map,
        )
    except ExperimentError as exc:
        print(f"{DIRECT_IOU_EVALUATION_FAILED}: {exc}", file=sys.stderr)
        return 2
    print(
        f"direct IoU matched mean {direct['global'][MATCHED_MASK_IOU_MEAN]}  "
        f"GT-normalised {direct['global'][GT_NORMALIZED_MASK_IOU]}  "
        f"coverage {direct['global'][GT_MATCH_COVERAGE]}"
    )

    direction = cross_metric_direction(
        delta["delta"],
        float(direct["global"][GT_NORMALIZED_MASK_IOU])
        - float(reference["direct_iou"]["global"][GT_NORMALIZED_MASK_IOU]),
    )
    print(f"direction  {direction['status']}")

    person = person_diagnostic(
        canonical_reference=reference["canonical"]["per_class"],
        canonical_candidate=canonical["per_class"],
        direct_reference=reference["direct_iou"]["per_class"],
        direct_candidate=direct["per_class"],
    )
    per_class_table = per_class_comparison(
        reference["canonical"]["per_class"], canonical["per_class"], support
    )

    # --- post-run verification ---------------------------------------------------------
    try:
        verify_adapter(paths, baseline, stage="after training")
        after = verify_historical(paths)
        changed = sorted(name for name in historical if historical[name] != after.get(name))
        if changed:
            msg = f"historical artifacts changed during this phase: {changed}"
            raise ExperimentError(msg)
        if (
            sha256_file(paths.root / reference_checkpoint_path(paths))
            != (reference["checkpoint_sha256"])
        ):
            msg = "the S0 checkpoint bytes changed during this phase"
            raise ExperimentError(msg)
    except (AdapterMismatchError, ExperimentError) as exc:
        classification = (
            ADAPTER_FINGERPRINT_MISMATCH if isinstance(exc, AdapterMismatchError) else BLOCKED
        )
        print(f"{classification}: {exc}", file=sys.stderr)
        return 2

    # --- artifacts -----------------------------------------------------------------------
    figures = copy_metric_figures(
        paths.figures / "segmentation_S1",
        sorted((paths.root / RUN_ROOT / f"{RUN_NAME}_val").glob("*.png"))
        + sorted(run_directory.glob("*.png")),
    )
    complexity = model_complexity(execution.get("log_text", ""))
    curves = summarise_curves(history, int(selection["best_epoch"]))

    canonical_payload = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "phase": PHASE,
        "experiment_id": EXPERIMENT_ID,
        "protocol": "CANONICAL_COCO_MASK_AP_EVALUATION",
        "protocol_fingerprint": policies["canonical_evaluator_fingerprint"],
        "protocol_config": f"configs/{CANONICAL_YAML}",
        "protocol_unchanged_from_phase_8e": True,
        "checkpoint": {
            "name": checkpoints["best"]["name"],
            "sha256": checkpoints["best"]["sha256"],
            "size_bytes": checkpoints["best"]["size_bytes"],
            "selected_by": CHECKPOINT_POLICY,
        },
        "ground_truth": {
            "source": "CANONICAL_COCO_INSTANCE_SEGMENTATION",
            "document": document["relative_path"],
            "sha256": document["sha256"],
            "is_the_yolo_adapter": False,
        },
        "split": "validation",
        "split_assignment_sha256": canonical_fingerprints["split_assignment_sha256"],
        "class_map_sha256": canonical_fingerprints["class_map_sha256"],
        "inference": {
            key: value
            for key, value in canonical_protocol.inference.items()
            if key not in ("threshold_policy", "checkpoint")
        },
        "conf_is_not_an_operating_point": (
            "0.001 preserves the low-scoring tail average precision integrates over. The "
            "operational threshold lives in the separate direct-IoU diagnostic, at 0.25, and "
            "the two protocols are never mixed, averaged or swapped."
        ),
        "prediction_cap_versus_metric_cap": {
            "model_side_max_det": canonical_protocol.inference["max_det"],
            "cocoeval_max_dets": list(canonical["cocoeval"]["max_dets"]),
            "note": (
                "The model is given room to propose and the metric applies its conventional "
                "cap when scoring. Two different numbers, both recorded."
            ),
        },
        "cocoeval": dict(canonical["cocoeval"]),
        "mask_encoding": "PYCOCOTOOLS_BINARY_MASK_RLE_ON_ORIGINAL_CANVAS",
        "category_id_mapping": "CANONICAL_CATEGORY_IDS_USED_DIRECTLY_NO_REMAPPING",
        "canonical": {
            "all_class_map50_95": canonical["all_class_map50_95"],
            "all_class_map50": canonical["all_class_map50"],
            "per_class": dict(canonical["per_class"]),
            "detections_scored": canonical["detections"],
        },
        "supported_macro": dict(macro),
        "support": dict(support),
        "reference_comparison": {
            "reference_experiment": REFERENCE,
            "reference_supported_macro": reference["canonical"]["supported_macro"],
            "reference_artifact": f"reports/{S0_CANONICAL_JSON}",
            "reference_artifact_sha256": reference["canonical"]["artifact_sha256"],
            **dict(delta),
        },
        "s0_reevaluated": False,
        "s1_retuned": False,
        "runs": 1,
        "contains_raw_masks_or_imagery": False,
        "test": {"status": HOLDOUT_STATUS, "reason": HOLDOUT_REASON},
        "holdout_accessed": False,
    }
    canonical_sha256 = write_json(paths.reports / CANONICAL_JSON, canonical_payload)

    direct_payload = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "phase": PHASE,
        "experiment_id": EXPERIMENT_ID,
        "protocol": DIRECT_IOU_PROTOCOL,
        "protocol_fingerprint": policies["direct_iou_protocol_fingerprint"],
        "protocol_config": f"configs/{MASK_IOU_YAML}",
        "protocol_unchanged_from_phase_8c": True,
        "checkpoint": {
            "name": checkpoints["best"]["name"],
            "sha256": checkpoints["best"]["sha256"],
            "size_bytes": checkpoints["best"]["size_bytes"],
            "selected_by": CHECKPOINT_POLICY,
        },
        "ground_truth": {
            "source": GROUND_TRUTH_SOURCE,
            "document": document["relative_path"],
            "sha256": document["sha256"],
            "is_the_yolo_adapter": False,
        },
        "split": "validation",
        "split_assignment_sha256": canonical_fingerprints["split_assignment_sha256"],
        "class_map_sha256": canonical_fingerprints["class_map_sha256"],
        "inference": direct["inference"],
        "matching": {
            "algorithm": MATCHING_ALGORITHM,
            "implementation": "scipy.optimize.linear_sum_assignment",
            "scope": MATCHING_SCOPE,
            "zero_overlap_policy": ZERO_OVERLAP_POLICY,
            "unmatched_gt_policy": UNMATCHED_GT_POLICY,
            "deterministic": True,
        },
        "images": direct["images"],
        "global": direct["global"],
        "per_class": direct["per_class"],
        "reference_comparison": {
            "reference_experiment": REFERENCE,
            "reference_artifact": f"reports/{S0_MASK_IOU_JSON}",
            "reference_global": dict(reference["direct_iou"]["global"]),
            "deltas": {
                name: round(
                    float(direct["global"][name]) - float(reference["direct_iou"]["global"][name]),
                    METRIC_PRECISION,
                )
                for name in (
                    MATCHED_MASK_IOU_MEAN,
                    GT_NORMALIZED_MASK_IOU,
                    GT_MATCH_COVERAGE,
                    GT_IOU50_COVERAGE,
                    GT_IOU75_COVERAGE,
                )
            },
        },
        "rare_class_warning": {
            "class": baseline.rare_class,
            "status": RARE_CLASS_STATUS,
            "detail": (
                "vest_loose holds one validation source image and eight instances under the "
                "frozen split. Its direct-IoU figures carry high sampling uncertainty, are "
                "reported for completeness, and must not be used to tune, rank or select."
            ),
        },
        "s0_rerun": False,
        "thresholds_swept": 0,
        "runs": 1,
        "contains_raw_masks_or_imagery": False,
        "test": {"status": HOLDOUT_STATUS, "reason": HOLDOUT_REASON},
        "holdout_accessed": False,
    }
    direct_sha256 = write_json(paths.reports / MASK_IOU_JSON, direct_payload)

    manifest = build_result_manifest(
        baseline=baseline,
        comparison=comparison,
        policies=policies,
        contract=contract,
        inheritance=inheritance,
        adapter=adapter,
        approval_sha256=approval_sha256,
        canonical_fingerprints=canonical_fingerprints,
        detector=detector,
        historical=historical,
        runtime=runtime,
        runtime_view=runtime_view,
        weights=weights,
        execution=execution,
        resolved=resolved,
        optimizer=optimizer,
        selection=selection,
        checkpoints=checkpoints,
        native=native,
        native_macro=native_macro,
        canonical=canonical,
        canonical_macro=macro,
        canonical_sha256=canonical_sha256,
        delta=delta,
        direct=direct,
        direct_sha256=direct_sha256,
        direction=direction,
        person=person,
        per_class_table=per_class_table,
        reference=reference,
        support=support,
        complexity=complexity,
        figures=figures,
        curves=curves,
        hypothesis=hypothesis,
    )

    # --- the validators, before anything is published -------------------------------------
    problems: list[str] = []
    problems += [f"result manifest: {item}" for item in validate_s1_result_manifest(manifest)]
    problems += [
        f"canonical evaluation: {item}"
        for item in validate_canonical_evaluation(
            canonical_payload, protocol_fingerprint=policies["canonical_evaluator_fingerprint"]
        )
    ]
    problems += [
        f"direct IoU: {item}"
        for item in validate_direct_iou(
            direct_payload, protocol_fingerprint=policies["direct_iou_protocol_fingerprint"]
        )
    ]
    problems += [
        f"comparison: {item}"
        for item in validate_comparison(
            manifest, s0_reference=read_json(paths.reports / S0_CANONICAL_JSON)
        )
    ]
    if problems:
        print(f"{PROTOCOL_VIOLATION}: the emitted artifacts do not validate:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 2

    results = build_results_artifact(manifest, reference)
    report = build_report(manifest, commit=git_commit(paths.root))

    findings = (
        scan_for_sensitive(report)
        + scan_for_sensitive(json.dumps(manifest))
        + scan_for_sensitive(json.dumps(canonical_payload))
        + scan_for_sensitive(json.dumps(direct_payload))
        + scan_for_sensitive(json.dumps(results))
    )
    if findings:
        print(f"{BLOCKED}: sensitive content in the emitted artifacts: {findings}", file=sys.stderr)
        return 2

    manifest_sha256 = write_json(paths.reports / RESULT_MANIFEST_JSON, manifest)
    results_sha256 = write_json(paths.reports / RESULTS_JSON, results)
    (paths.reports / REPORT_MD).write_text(report, encoding="utf-8", newline="\n")

    record = ProvenanceRecord.create(
        name="segmentation_S1_experiment",
        phase=8,
        config={
            "segmentation_baseline": f"configs/{BASELINE_YAML}",
            "segmentation_comparison": f"configs/{COMPARISON_YAML}",
            "canonical_evaluation": f"configs/{CANONICAL_YAML}",
            "direct_mask_iou_protocol": f"configs/{MASK_IOU_YAML}",
        },
        details={
            "phase": PHASE,
            "classification": COMPLETE,
            "experiment_id": EXPERIMENT_ID,
            "reference_experiment": REFERENCE,
            "S1_experiment_sha256": manifest["S1_experiment_sha256"],
            "intentional_difference": (
                f"{ONE_VARIABLE_FIELD} {contract['reference_value']} -> "
                f"{contract['candidate_value']}"
            ),
            "epochs_configured": configured,
            "epochs_completed": logged,
            "best_epoch": selection["best_epoch"],
            "best_native_fitness": selection["best_native_fitness"],
            "checkpoint_selection_policy": CHECKPOINT_POLICY,
            "best_checkpoint_sha256": checkpoints["best"]["sha256"],
            "primary_metric": PRIMARY_METRIC,
            "primary_value": macro["value"],
            "primary_reference": reference["canonical"]["supported_macro"],
            "primary_delta": delta["delta"],
            "margin_classification": delta["verdict"],
            "cross_metric_direction": direction["status"],
            "native_mask_map50_95": native["mask"]["mAP@0.50:0.95"],
            "direct_mask_iou_gt_normalized": direct["global"][GT_NORMALIZED_MASK_IOU],
            "direct_mask_iou_matched_mean": direct["global"][MATCHED_MASK_IOU_MEAN],
            "adapter_fingerprints": adapter["fingerprints"],
            "adapter_regenerated": False,
            "models_trained_in_this_phase": 1,
            "s0_retrained": False,
            "s0_revalidated": False,
            "final_segmenter": FINAL_SEGMENTER,
            "detector_touched": False,
            "holdout_accessed": False,
            "historical_artifacts_unchanged": True,
        },
        repo_root=paths.root,
    )
    for name in (BASELINE_YAML, COMPARISON_YAML, CANONICAL_YAML, MASK_IOU_YAML):
        record.add_input(paths.configs / name, relative_to=paths.root)
    for name in (S1_PROTOCOL_JSON, S0_CANONICAL_JSON, S0_RESULT_JSON, S0_MASK_IOU_JSON):
        record.add_input(paths.reports / name, relative_to=paths.root)
    for name in (RESULT_MANIFEST_JSON, CANONICAL_JSON, MASK_IOU_JSON, REPORT_MD, RESULTS_JSON):
        record.add_output(paths.reports / name, relative_to=paths.root)
    record.write_json(paths.reports / PROVENANCE_JSON)

    print(COMPLETE)
    print(f"primary    {PRIMARY_METRIC} {macro['value']}  (validation only)")
    print(f"delta      {delta['delta']}  margin {delta['margin']}  -> {delta['verdict']}")
    print(f"direction  {direction['status']}")
    print(f"fingerprint S1_experiment_sha256 {manifest['S1_experiment_sha256']}")
    print(f"manifest   reports/{RESULT_MANIFEST_JSON}  sha256 {manifest_sha256}")
    print(f"canonical  reports/{CANONICAL_JSON}  sha256 {canonical_sha256}")
    print(f"diagnostic reports/{MASK_IOU_JSON}  sha256 {direct_sha256}")
    print(f"results    reports/{RESULTS_JSON}  sha256 {results_sha256}")
    print(f"report     reports/{REPORT_MD}")
    print(f"segmenter  {FINAL_SEGMENTER}")
    print(f"holdout    {HOLDOUT_STATUS}")
    return 0


def reference_checkpoint_path(paths: ProjectPaths) -> str:
    """Locate S0's frozen checkpoint, relative to the repository root.

    Args:
        paths: Project layout.

    Returns:
        The repository-relative path.
    """
    result = read_json(paths.reports / S0_RESULT_JSON)
    return f"{result['execution']['run_directory']}/weights/best.pt"


if __name__ == "__main__":
    sys.exit(main())
