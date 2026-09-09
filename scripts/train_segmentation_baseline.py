"""Run the S0 segmentation baseline once and report it honestly.

Phase 8C. It performs exactly one full training run under the protocol frozen in
phase 8B, one authoritative validation of the checkpoint the framework's own rule
selected, and one direct instance-mask IoU diagnostic under the protocol frozen
before the first optimisation step. It selects no final segmenter, tunes nothing,
and never touches the holdout.

Four things here are less obvious than they look.

**The training data is a hard-linked view, not the audited directory itself.**
Ultralytics writes ``.cache`` files beside the labels it scans, and the audited
adapter is phase 8A evidence. So the run reads a disposable view whose label
bytes are verified identical to the approved digests first - identical bytes, not
a regenerated conversion.

**The selected checkpoint is verified against the rule, not assumed.** The
framework writes ``best.pt`` by its own composite fitness and records the terms
but not the fitness. So the composite is recomputed from ``results.csv`` and the
recorded best epoch is confirmed to be its argmax. Otherwise "the frozen rule
chose this checkpoint" would be an article of faith.

**The direct IoU diagnostic scores against canonical COCO masks, never the YOLO
adapter.** The adapter's own round-trip error was measured at mean mask IoU
0.973066; scoring against it would fold that approximation into the model's
result by an amount nobody could separate afterwards.

**Nothing is decided.** S0 is a baseline. There is no second segmentation
experiment, so there is nothing to select between, and the final segmenter stays
``UNSELECTED_PENDING_REVIEW``.

Requires:

* ``configs/segmentation_baseline.yaml``                 phase 8B
* ``configs/segmentation_mask_iou_evaluation.yaml``      phase 8C
* ``reports/segmentation_adapter_approval.json``         phase 8B
* ``reports/segmentation_S0_manifest.json``              phase 8B
* ``data/processed/adapters/yolo_segmentation_audit/``   phase 8A
* ``data/processed/canonical/annotations/``              phase 5D

Writes:
    reports/segmentation_S0_result_manifest.json
    reports/segmentation_S0_mask_iou.json
    reports/segmentation_S0_report.md
    reports/segmentation_S0.provenance.json
    reports/segmentation_S0_prerun.provenance.json
    reports/figures/segmentation_S0_*.png
    artifacts/segmentation/S0/                           (git-ignored)
    data/processed/adapters/yolo_segmentation_s0_runtime/ (git-ignored)

Usage:
    uv run python scripts/train_segmentation_baseline.py
    uv run python scripts/train_segmentation_baseline.py --verify-only
    uv run python scripts/train_segmentation_baseline.py --diagnostic-only
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from construction_safety_vision.config import ConfigError
from construction_safety_vision.data.canonical import scan_for_sensitive
from construction_safety_vision.data.materialization import digest
from construction_safety_vision.detection_comparison import (
    SUPPORT_MIN_INSTANCES,
    SUPPORT_MIN_POSITIVE_IMAGES,
)
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
    PROTOCOL_NAME,
    MaskInstance,
    accumulate,
    load_mask_iou_config,
    match_image,
)
from construction_safety_vision.paths import ProjectPaths, long_path
from construction_safety_vision.provenance import ProvenanceRecord, git_commit, sha256_file
from construction_safety_vision.segmentation_experiment import (
    ADAPTER_ROLE,
    CANONICAL_GROUND_TRUTH,
    CHECKPOINT_SELECTION_POLICY,
    CHECKPOINT_SELECTION_SEMANTICS,
    PRIMARY_SCIENTIFIC_REPORTING_METRIC,
    SUPPORT_RULE,
    SegmentationBaselineConfig,
    SegmentationExperimentConfigError,
    load_segmentation_baseline_config,
)
from construction_safety_vision.segmentation_run import (
    FITNESS_TOLERANCE,
    RUN_ROOT,
    RUNTIME_VIEW_ROOT,
    SegmentationRunError,
    adapter_counts,
    adapter_fingerprints,
    best_epoch_by_native_fitness,
    build_runtime_view,
    checkpoint_record,
    copy_metric_figures,
    native_fitness_curve,
    read_adapter_labels,
    read_epoch_history,
)
from construction_safety_vision.splits import HOLDOUT_UNLOCK_ENV_VAR, holdout_unlocked

BASELINE_YAML = "segmentation_baseline.yaml"
MASK_IOU_YAML = "segmentation_mask_iou_evaluation.yaml"

APPROVAL_JSON = "segmentation_adapter_approval.json"
PROTOCOL_MANIFEST_JSON = "segmentation_S0_manifest.json"
AUDIT_MANIFEST_JSON = "segmentation_adapter_audit_manifest.json"
FINAL_DETECTOR_JSON = "final_detector_manifest.json"
TASK_MANIFEST_JSON = "task_dataset_manifest.json"
SPLIT_MANIFEST_JSON = "split_manifest.json"

RESULT_MANIFEST_JSON = "segmentation_S0_result_manifest.json"
MASK_IOU_JSON = "segmentation_S0_mask_iou.json"
REPORT_MD = "segmentation_S0_report.md"
PROVENANCE_JSON = "segmentation_S0.provenance.json"
PRERUN_PROVENANCE_JSON = "segmentation_S0_prerun.provenance.json"

SCHEMA_VERSION = 1
PHASE = "8C"
EXPERIMENT_ID = "S0"
RUN_NAME = "S0"

COMPLETE = "S0_SEGMENTATION_BASELINE_COMPLETE"
ADAPTER_FINGERPRINT_MISMATCH = "ADAPTER_FINGERPRINT_MISMATCH"
S0_PROTOCOL_MISMATCH = "S0_PROTOCOL_MISMATCH"
MEMORY_CONSTRAINT_REVIEW_REQUIRED = "MEMORY_CONSTRAINT_REVIEW_REQUIRED"
TRAINING_FAILED = "TRAINING_FAILED"
DIRECT_IOU_EVALUATION_FAILED = "DIRECT_IOU_EVALUATION_FAILED"
PROTOCOL_VIOLATION = "PROTOCOL_VIOLATION"
BLOCKED = "BLOCKED"

BASELINE_STATUS = "S0_COMPLETE"
FINAL_SEGMENTER = "UNSELECTED_PENDING_REVIEW"

HOLDOUT_STATUS = "PROTECTED_NOT_ACCESSED"
HOLDOUT_REASON = (
    "Phase 8C trained S0 on the frozen train split, validated it on the frozen validation "
    "split, and ran the direct mask-IoU diagnostic on that same validation split. The holdout "
    "was not read, materialised, adapted, counted, predicted on, inspected or plotted; no "
    "holdout identifier, label, prediction or statistic exists in any artifact this phase "
    "wrote, and the dataset descriptor the framework read carries no holdout key."
)

RARE_CLASS_STATUS = "DESCRIPTIVE_HIGH_UNCERTAINTY"
SUPPORTED = "COMPARISON_SUPPORTED"
SUPPORTED_MACRO_METRIC = "supported_macro_mask_map50_95"
NOT_EXPOSED = "NOT_EXPOSED_RELIABLY"

METRIC_PRECISION = 6

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
    """Raised when the frozen S0 protocol has moved."""


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


def relativise(value: Any, root: Path) -> Any:
    """Strip machine-specific absolute paths out of a recorded value.

    Ultralytics records absolute paths for ``model``, ``data`` and its output
    directories, and those name this machine and this user. A committed artifact
    must not, so a path under the repository root becomes repository-relative and
    anything else absolute becomes a sentinel. The sensitive-content scan catches
    a leak either way; this stops there being one to catch.

    Args:
        value: A recorded configuration value.
        root: Repository root.

    Returns:
        The value, with any absolute path rewritten.
    """
    if not isinstance(value, str):
        return value
    text = value.replace("\\", "/")
    base = str(root).replace("\\", "/").rstrip("/")
    if text.startswith(base + "/"):
        return text[len(base) + 1 :]
    if re.match(r"^[A-Za-z]:/", text) or text.startswith(("/home/", "/Users/", "/root/", "//")):
        return "EXTERNAL_ABSOLUTE_PATH_REDACTED"
    return value


def rounded(value: Any) -> Any:
    """Round a metric to the reported precision, passing non-numbers through.

    Args:
        value: A metric or a sentinel.

    Returns:
        The rounded value, or the input unchanged.
    """
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return round(float(value), METRIC_PRECISION)
    return value


# --- preflight ----------------------------------------------------------------


def verify_detector(paths: ProjectPaths) -> dict[str, Any]:
    """Verify the frozen detector is untouched, then leave it alone.

    Nothing is loaded into a model, validated or inferred with: the detection
    block is closed, and this is verification rather than use.

    Args:
        paths: Project layout.

    Returns:
        What was verified.

    Raises:
        ExperimentError: If the manifest does not recompute or the checkpoint
            bytes differ.
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
        "inference_run": False,
        "benchmarked": False,
        "threshold_changed": False,
        "status": "UNCHANGED_DETECTION_BLOCK_CLOSED",
    }


def verify_historical(paths: ProjectPaths) -> dict[str, str]:
    """Digest every phase 8A and 8B artifact that must not change.

    Args:
        paths: Project layout.

    Returns:
        Digests keyed by repository-relative name.

    Raises:
        ExperimentError: If one is absent.
    """
    names = [
        f"reports/{AUDIT_MANIFEST_JSON}",
        "reports/segmentation_adapter_fidelity_report.md",
        "reports/segmentation_adapter_fidelity.csv",
        "reports/segmentation_adapter_audit.provenance.json",
        "configs/segmentation_adapter_audit.yaml",
        f"reports/{APPROVAL_JSON}",
        f"reports/{PROTOCOL_MANIFEST_JSON}",
        "reports/segmentation_S0_protocol.md",
        f"configs/{BASELINE_YAML}",
        f"reports/{FINAL_DETECTOR_JSON}",
    ]
    digests: dict[str, str] = {}
    for name in names:
        path = paths.root / name
        if not path.is_file():
            msg = f"historical artifact missing: {name}"
            raise ExperimentError(msg)
        digests[name] = sha256_file(path)
    return digests


def verify_protocol(paths: ProjectPaths, config: SegmentationBaselineConfig) -> dict[str, Any]:
    """Verify the loaded protocol is the one phase 8B froze.

    Args:
        paths: Project layout.
        config: The parsed protocol.

    Returns:
        What was verified.

    Raises:
        ProtocolMismatchError: If any frozen field has moved.
    """
    manifest = read_json(paths.reports / PROTOCOL_MANIFEST_JSON)
    if manifest["protocol_fingerprint"] != config.fingerprint():
        msg = (
            f"the S0 protocol fingerprint changed: phase 8B recorded "
            f"{manifest['protocol_fingerprint']}, the configuration now hashes to "
            f"{config.fingerprint()}. The frozen protocol is not edited."
        )
        raise ProtocolMismatchError(msg)

    expected = {
        "experiment_id": (config.experiment_id, EXPERIMENT_ID),
        "architecture": (config.architecture, "YOLO11n-seg"),
        "imgsz": (config.training["imgsz"], 768),
        "batch": (config.training["batch"], 8),
        "epochs": (config.training["epochs"], 100),
        "seed": (config.seed, 42),
        "deterministic": (config.training["deterministic"], True),
        "amp": (config.training["amp"], True),
        "optimizer": (config.training["optimizer"], "auto"),
        "patience": (config.training["patience"], 50),
        "checkpoint_selection_policy": (
            config.checkpoint_selection_policy,
            CHECKPOINT_SELECTION_POLICY,
        ),
        "checkpoint_selection_semantics": (
            config.checkpoint_selection_semantics,
            CHECKPOINT_SELECTION_SEMANTICS,
        ),
        "primary_scientific_reporting_metric": (
            config.primary_scientific_reporting_metric,
            PRIMARY_SCIENTIFIC_REPORTING_METRIC,
        ),
    }
    wrong = [
        f"{name}: expected {want!r}, got {have!r}"
        for name, (have, want) in expected.items()
        if have != want
    ]
    if wrong:
        msg = "the frozen S0 protocol does not match phase 8B: " + "; ".join(wrong)
        raise ProtocolMismatchError(msg)

    recorded = manifest["segmentation_arguments"]
    drifted = [
        f"{name}: frozen {recorded[name]!r}, configured {config.segmentation_arguments[name]!r}"
        for name in sorted(recorded)
        if recorded[name] != config.segmentation_arguments.get(name)
    ]
    if drifted:
        msg = "segmentation-specific arguments drifted from phase 8B: " + "; ".join(drifted)
        raise ProtocolMismatchError(msg)

    return {
        "protocol_fingerprint": config.fingerprint(),
        "phase_8b_manifest_sha256": sha256_file(paths.reports / PROTOCOL_MANIFEST_JSON),
        "verified_fields": sorted(expected),
        "segmentation_arguments_verified": True,
    }


def verify_adapter(
    paths: ProjectPaths, config: SegmentationBaselineConfig, *, stage: str
) -> dict[str, Any]:
    """Verify the approved label bytes are the ones on disk.

    Args:
        paths: Project layout.
        config: The frozen protocol.
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
        config: The frozen protocol.

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


# --- canonical ground truth ---------------------------------------------------


def load_canonical_validation(paths: ProjectPaths, document: str) -> dict[str, Any]:
    """Read the canonical COCO validation document.

    Args:
        paths: Project layout.
        document: Repository-relative path from the frozen diagnostic protocol.

    Returns:
        The parsed COCO document plus its digest.

    Raises:
        ExperimentError: If it is missing or names the protected split.
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
            rles = mask_utils.frPyObjects(segmentation, height, width)
            rle = mask_utils.merge(rles)
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
    config: SegmentationBaselineConfig,
    weights: Mapping[str, Any],
    descriptor: Path,
) -> dict[str, Any]:
    """Execute the single full S0 training run.

    Args:
        paths: Project layout.
        config: The frozen protocol.
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
            f"a completed S0 run already exists at {RUN_ROOT}/{RUN_NAME}. Exactly one valid full "
            "run is authorised; overwriting it would destroy the experiment being reported. "
            "Move or quarantine it deliberately if a re-run has been reviewed."
        )
        raise ExperimentError(msg)
    run_directory.mkdir(parents=True, exist_ok=True)

    arguments = config.training_arguments()
    arguments["data"] = str(descriptor.resolve())
    arguments["project"] = str((paths.root / RUN_ROOT).resolve())
    arguments["name"] = RUN_NAME
    # Ultralytics diverts a run to `<name>-2` when the output directory already
    # exists, and attaching the log handler creates it. The pre-run check above
    # is what guards against overwriting a completed run; this only stops the
    # framework silently writing somewhere the result is not read from, which is
    # verified after training.
    arguments["exist_ok"] = True
    arguments["plots"] = True

    handler = capture_framework_log(run_directory / "framework.log")
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    failure: str | None = None
    try:
        model = YOLO(str(paths.root / weights["relative_path"]))
        model.train(**arguments)
    except torch.cuda.OutOfMemoryError as exc:
        release_framework_log(handler)
        msg = (
            f"{MEMORY_CONSTRAINT_REVIEW_REQUIRED}: the frozen batch of "
            f"{config.training['batch']} exhausted device memory ({exc}). The batch is not "
            "reduced, auto-batch is not enabled, gradient accumulation is not introduced and "
            "imgsz is not lowered: the experiment stops for review."
        )
        raise MemoryConstraintError(msg) from exc
    except Exception as exc:  # any failure stops the experiment and is reported as one
        failure = f"{type(exc).__name__}: {exc}"
    elapsed = time.perf_counter() - started
    peak_reserved = int(torch.cuda.max_memory_reserved())
    peak_allocated = int(torch.cuda.max_memory_allocated())
    release_framework_log(handler)

    if failure is not None:
        msg = f"{TRAINING_FAILED}: the S0 training run did not complete ({failure})"
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
        "log_text": log_text,
    }


def resolved_training_arguments(run_directory: Path) -> dict[str, Any]:
    """Read the arguments the framework recorded for the run.

    Args:
        run_directory: The run's output directory.

    Returns:
        The parsed ``args.yaml``, or an empty mapping when absent.
    """
    import yaml

    path = run_directory / "args.yaml"
    if not path.is_file():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, Mapping):
        return {}
    root = ProjectPaths.from_root().root
    return {key: relativise(value, root) for key, value in data.items()}


def verify_checkpoint_selection(
    history: Sequence[Mapping[str, str]], resolved: Mapping[str, Any]
) -> dict[str, Any]:
    """Confirm the recorded checkpoint is the frozen rule's argmax.

    Ultralytics writes ``best.pt`` by its own composite fitness but records only
    the terms, so the composite is recomputed here and its argmax compared
    against what the run actually did. Without this the claim that the frozen
    rule chose the checkpoint would rest on nothing.

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
        "policy": CHECKPOINT_SELECTION_POLICY,
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
        "verified": True,
    }


def validate(
    paths: ProjectPaths, config: SegmentationBaselineConfig, checkpoint: Path
) -> dict[str, Any]:
    """Run the single authoritative validation of the selected checkpoint.

    Args:
        paths: Project layout.
        config: The frozen protocol.
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
        "imgsz": config.training["imgsz"],
        "batch": config.training["batch"],
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
    # framework's own `get_cfg` with the same overrides the call was given. That
    # reproduces exactly what the validator received - including the defaults the
    # call did not name, which are the ones a report is most likely to get wrong -
    # and anyone can re-derive it. Recording only the requested settings would
    # leave `conf` and `iou` unstated, which are precisely the material ones.
    resolved_val = get_cfg(overrides={**overrides, "mode": "val", "task": "segment"})
    names = (
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
        name: relativise(getattr(resolved_val, name, NOT_EXPOSED), paths.root) for name in names
    }
    effective["source"] = "RESOLVED_FROM_INSTALLED_CFG_WITH_THE_SAME_OVERRIDES"

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


def supported_macro(
    per_class: Mapping[str, Mapping[str, Any]], support: Mapping[str, Mapping[str, int]]
) -> dict[str, Any]:
    """Apply the frozen support rule to the mask AP values.

    Args:
        per_class: Per-class mask and box metrics.
        support: Validation image and instance counts per class.

    Returns:
        The macro metric, the admitted classes and the arithmetic behind it.
    """
    classification: dict[str, str] = {}
    admitted: list[str] = []
    for name, counts in sorted(support.items()):
        images = int(counts["images"])
        instances = int(counts["instances"])
        if images >= SUPPORT_MIN_POSITIVE_IMAGES and instances >= SUPPORT_MIN_INSTANCES:
            classification[name] = SUPPORTED
            admitted.append(name)
        else:
            classification[name] = RARE_CLASS_STATUS
    values = [
        float(per_class[name]["mask"]["AP@0.50:0.95"])
        for name in admitted
        if name in per_class and isinstance(per_class[name]["mask"]["AP@0.50:0.95"], (int, float))
    ]
    value = round(sum(values) / len(values), METRIC_PRECISION) if values else None
    return {
        "metric": SUPPORTED_MACRO_METRIC,
        "value": value,
        "rule": SUPPORT_RULE,
        "rule_origin": "FROZEN_IN_PHASE_7A_REUSED_UNCHANGED",
        "classification": classification,
        "admitted_classes": admitted,
        "contributing_values": [round(item, METRIC_PRECISION) for item in values],
        "is_a_selection_metric": False,
        "descriptive_for_s0": True,
        "reason": (
            "S0 is the only segmentation experiment, so there is nothing to select between. "
            "This value is descriptive and declares no winner."
        ),
    }


def run_direct_iou(
    paths: ProjectPaths,
    protocol: Any,
    checkpoint: Path,
    ground_truth: Mapping[str, list[MaskInstance]],
    class_map: Mapping[str, int],
) -> dict[str, Any]:
    """Execute the frozen direct mask-IoU diagnostic exactly once.

    Args:
        paths: Project layout.
        protocol: The frozen diagnostic protocol.
        checkpoint: The selected ``best.pt``.
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

    result = accumulate(matchings)
    names = {index: name for name, index in class_map.items()}
    rendered = result.as_dict(names)
    if rendered["global"]["prediction_count"] != predicted_total:  # pragma: no cover - invariant
        msg = "the diagnostic lost a prediction between matching and accumulation"
        raise ExperimentError(msg)
    return {
        "protocol": PROTOCOL_NAME,
        "protocol_fingerprint": protocol.fingerprint(),
        "ground_truth_source": GROUND_TRUTH_SOURCE,
        "matching_algorithm": MATCHING_ALGORITHM,
        "inference": {key: value for key, value in inference.items() if key != "threshold_policy"},
        "runs": 1,
        **rendered,
    }


def validation_support(
    ground_truth: Mapping[str, list[MaskInstance]], class_map: Mapping[str, int]
) -> dict[str, dict[str, int]]:
    """Count positive validation images and instances per class.

    The frozen support rule needs both, and deriving them from the canonical
    document rather than restating phase 5C.2's numbers means the rule is applied
    to the data actually evaluated.

    Args:
        ground_truth: Canonical instances keyed by image stem.
        class_map: The frozen class map.

    Returns:
        ``{class_name: {"images": n, "instances": m}}`` for every frozen class.
    """
    support = {name: {"images": 0, "instances": 0} for name in class_map}
    names = {index: name for name, index in class_map.items()}
    for instances in ground_truth.values():
        seen: set[int] = set()
        for instance in instances:
            name = names[int(instance.class_id)]
            support[name]["instances"] += 1
            seen.add(int(instance.class_id))
        for class_index in seen:
            support[names[class_index]]["images"] += 1
    return support


def experiment_fingerprint(values: Mapping[str, Any]) -> str:
    """Hash the experiment's semantic identity.

    Covers everything that would make this a different experiment and nothing
    that would not: no timestamp, no path, no username, no wall-clock. Two runs
    of the same protocol over the same data producing the same checkpoint hash
    the same, and any change to the protocol, the data, the weights, the
    checkpoint rule or the diagnostic protocol changes it.

    Args:
        values: The identity fields.

    Returns:
        A SHA-256 hex digest.
    """
    return digest(json.loads(json.dumps(values, sort_keys=True)))


def build_result_manifest(
    *,
    config: SegmentationBaselineConfig,
    protocol: Any,
    verified_protocol: Mapping[str, Any],
    adapter: Mapping[str, Any],
    approval_sha256: str,
    canonical: Mapping[str, Any],
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
    validation: Mapping[str, Any],
    macro: Mapping[str, Any],
    diagnostic: Mapping[str, Any],
    diagnostic_sha256: str,
    complexity: Mapping[str, Any],
    figures: Sequence[str],
    curves: Mapping[str, Any],
) -> dict[str, Any]:
    """Assemble the S0 result manifest.

    Args:
        config: The frozen protocol.
        protocol: The frozen diagnostic protocol.
        verified_protocol: What the protocol check confirmed.
        adapter: Verified label fingerprints and cardinality.
        approval_sha256: Digest of the phase 8B adapter approval.
        canonical: Canonical dataset fingerprints.
        detector: The frozen detector verification.
        historical: Digests of the phase 8A and 8B artifacts.
        runtime: Software and hardware facts.
        runtime_view: The training-time dataset view record.
        weights: The pretrained checkpoint record.
        execution: Training execution facts.
        resolved: The framework's recorded arguments.
        optimizer: Captured optimizer evidence.
        selection: Checkpoint-selection verification.
        checkpoints: best and last checkpoint records.
        validation: The authoritative validation result.
        macro: The supported mask macro result.
        diagnostic: The direct mask-IoU result.
        diagnostic_sha256: Digest of the direct mask-IoU artifact.
        complexity: Model parameter and FLOP counts.
        figures: Committed metric-only figure names.
        curves: Observations about the training dynamics.

    Returns:
        The manifest.
    """
    identity = {
        "experiment_id": EXPERIMENT_ID,
        "phase_8b_protocol_fingerprint": config.fingerprint(),
        "adapter_approval_sha256": approval_sha256,
        "adapter_fingerprints": dict(adapter["fingerprints"]),
        "class_map_sha256": canonical["class_map_sha256"],
        "split_assignment_sha256": canonical["split_assignment_sha256"],
        "canonical_task_manifest_sha256": canonical["canonical_task_manifest_sha256"],
        "pretrained_weight_sha256": weights["sha256"],
        "checkpoint_selection_policy": CHECKPOINT_SELECTION_POLICY,
        "checkpoint_selection_semantics": CHECKPOINT_SELECTION_SEMANTICS,
        "best_checkpoint_sha256": checkpoints["best"]["sha256"],
        "direct_mask_iou_protocol_fingerprint": protocol.fingerprint(),
        "critical_training_config": {
            "model": config.model,
            "imgsz": config.training["imgsz"],
            "batch": config.training["batch"],
            "epochs": config.training["epochs"],
            "seed": config.seed,
            "optimizer_policy": config.training["optimizer"],
            "patience": config.training["patience"],
            "deterministic": config.training["deterministic"],
            "amp": config.training["amp"],
            "segmentation_arguments": dict(config.segmentation_arguments),
            "augmentation_arguments": dict(config.augmentation_arguments),
        },
    }

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

    return {
        "schema_version": SCHEMA_VERSION,
        "phase": PHASE,
        "experiment_id": EXPERIMENT_ID,
        "status": COMPLETE,
        "task": "segmentation",
        "experiment_question": config.description.strip(),
        "experiment_role": "BASELINE_NOT_THE_FINAL_SEGMENTER",
        "segmentation_baseline_status": BASELINE_STATUS,
        "final_segmenter": FINAL_SEGMENTER,
        "final_segmenter_note": (
            "S0 is the project's only segmentation experiment. There is nothing to select "
            "between and no segmentation comparison protocol has been frozen, so no final "
            "segmenter is chosen here."
        ),
        "metrics_are_validation_only": True,
        "architecture": config.architecture,
        "model": config.model,
        "imgsz": config.training["imgsz"],
        "batch": config.training["batch"],
        "seed": config.seed,
        "phase_8b_protocol_fingerprint": config.fingerprint(),
        "phase_8b_protocol_verified": dict(verified_protocol),
        "adapter": {
            "role": ADAPTER_ROLE,
            "canonical_ground_truth": CANONICAL_GROUND_TRUTH,
            "approval_sha256": approval_sha256,
            "fingerprints": dict(adapter["fingerprints"]),
            "counts": dict(adapter["counts"]),
            "regenerated": False,
            "modified": False,
        },
        "runtime_view": dict(runtime_view),
        "canonical_fingerprints": dict(canonical),
        "historical_artifact_digests": dict(historical),
        "historical_artifacts_unchanged": True,
        "frozen_detector": dict(detector),
        "runtime": dict(runtime),
        "pretrained_weights": dict(weights),
        "optimizer": {
            "declared_policy": config.training["optimizer"],
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
        "effective_training_configuration": {
            name: resolved.get(name, NOT_EXPOSED) for name in effective_names
        },
        "effective_training_configuration_note": (
            "Read back from the framework's own args.yaml. Absolute paths are rewritten "
            "repository-relative so a committed artifact does not name this machine."
        ),
        "checkpoint_selection": dict(selection),
        "checkpoints": dict(checkpoints),
        "execution": {key: value for key, value in execution.items() if key != "log_text"},
        "training_curves": dict(curves),
        "model_complexity": dict(complexity),
        "validation": {
            "runs": validation["runs"],
            "split": "validation",
            "effective_arguments": dict(validation["effective_arguments"]),
            "output_directory": validation["output_directory"],
        },
        "primary_scientific_result": {
            "metric": "mask_mAP@0.50:0.95",
            "value": validation["mask"]["mAP@0.50:0.95"],
            "validation_only": True,
        },
        "mask_metrics": dict(validation["mask"]),
        "box_metrics_from_segmenter": dict(validation["box"]),
        "box_metrics_note": (
            "Reported separately and never merged with the mask family. The framework's "
            "composite fitness remains checkpoint-selection metadata, not a scientific score."
        ),
        "per_class_metrics": dict(validation["per_class"]),
        "framework_fitness_at_validation": validation["framework_fitness"],
        "supported_macro_mask": dict(macro),
        "rare_class": {
            "name": config.rare_class,
            "status": RARE_CLASS_STATUS,
            "support_rule": SUPPORT_RULE,
            "reported_in_full": True,
            "may_decide_anything": False,
            "limitation": config.rare_class_limitation.strip(),
        },
        "direct_mask_iou": {
            "protocol": PROTOCOL_NAME,
            "protocol_fingerprint": protocol.fingerprint(),
            "protocol_config": "configs/" + MASK_IOU_YAML,
            "result_artifact": "reports/" + MASK_IOU_JSON,
            "result_sha256": diagnostic_sha256,
            "ground_truth_source": GROUND_TRUTH_SOURCE,
            "runs": 1,
            "headline": {
                name: diagnostic["global"][name]
                for name in (
                    MATCHED_MASK_IOU_MEAN,
                    GT_NORMALIZED_MASK_IOU,
                    GT_MATCH_COVERAGE,
                    GT_IOU50_COVERAGE,
                    GT_IOU75_COVERAGE,
                )
            },
            "counts": {
                name: diagnostic["global"][name]
                for name in (
                    "gt_count",
                    "prediction_count",
                    "matched_count",
                    "unmatched_gt",
                    "unmatched_predictions",
                )
            },
            "is_not_average_precision": (
                "Neither figure is a COCO AP. AP is a ranking-sensitive average over IoU "
                "thresholds; these are mask overlap at one predeclared operating point."
            ),
        },
        "framework_validation_speed": {
            "values_ms_per_image": dict(validation["speed"]),
            "label": "FRAMEWORK_VALIDATION_SPEED",
            "caveat": (
                "Measured by the framework during validation, not a production latency "
                "benchmark. No standardised latency study was run in this phase."
            ),
        },
        "committed_figures": list(figures),
        "S0_experiment_sha256": experiment_fingerprint(identity),
        "experiment_fingerprint_inputs": sorted(identity),
        "test": {"status": HOLDOUT_STATUS, "reason": HOLDOUT_REASON},
        "holdout_accessed": False,
        "detector_touched": False,
        "models_trained_in_this_phase": 1,
        "alternative_segmenters_trained": 0,
        "imgsz_variants_tried": 0,
        "batch_variants_tried": 0,
        "thresholds_tuned": 0,
    }


SUMMARY_PATTERN = re.compile(
    r"summary(?P<fused>\s*\(fused\))?:\s*(?P<layers>[\d,]+)\s+layers,\s*"
    r"(?P<parameters>[\d,]+)\s+parameters.*?(?P<gflops>[\d.]+)\s+GFLOPs"
)
"""The framework's own model summary line.

Parsed with a regex rather than by splitting on commas: the counts carry
thousands separators, so ``2,843,583 parameters`` splits into three fields and a
naive parser silently records 543.
"""

EPOCH_TIME_COLUMN = "time"
"""Cumulative training seconds, as ``results.csv`` names it."""


def model_complexity(log_text: str) -> dict[str, Any]:
    """Read parameter, layer and FLOP counts out of the framework's summary lines.

    Both the unfused training model and the fused inference model are recorded:
    they differ, and quoting one as the other would misstate the model.

    Args:
        log_text: Captured framework log.

    Returns:
        The counts, with ``None`` where the line was absent.
    """
    found: dict[str, dict[str, Any]] = {}
    for match in SUMMARY_PATTERN.finditer(log_text):
        key = "fused" if match.group("fused") else "unfused"
        found.setdefault(
            key,
            {
                "layers": int(match.group("layers").replace(",", "")),
                "parameters": int(match.group("parameters").replace(",", "")),
                "gflops": float(match.group("gflops")),
            },
        )
    unfused = found.get("unfused", {})
    fused = found.get("fused", {})
    return {
        "parameters": unfused.get("parameters"),
        "layers": unfused.get("layers"),
        "gflops": unfused.get("gflops"),
        "fused_parameters": fused.get("parameters"),
        "fused_layers": fused.get("layers"),
        "fused_gflops": fused.get("gflops"),
        "source": "FRAMEWORK_MODEL_SUMMARY_LINE",
    }


def training_duration(history: Sequence[Mapping[str, str]], log_text: str) -> dict[str, Any]:
    """Recover the training run's duration from artifacts that are on disk.

    Deliberately not taken from a wall-clock timer held in the running process:
    a figure that cannot be pointed at a file is a figure a reader cannot check.
    ``results.csv`` carries the framework's own cumulative time, and the log line
    corroborates it independently.

    Args:
        history: Parsed ``results.csv`` rows.
        log_text: Captured framework log.

    Returns:
        The duration and where each figure came from.
    """
    cumulative = None
    if history and EPOCH_TIME_COLUMN in history[-1]:
        try:
            cumulative = round(float(history[-1][EPOCH_TIME_COLUMN]), 3)
        except (TypeError, ValueError):
            cumulative = None
    match = re.search(r"(\d+) epochs completed in ([\d.]+) hours", log_text)
    return {
        "training_seconds": cumulative,
        "training_seconds_source": "FRAMEWORK_RESULTS_CSV_CUMULATIVE_TIME",
        "framework_reported_hours": float(match.group(2)) if match else None,
        "framework_reported_epochs": int(match.group(1)) if match else None,
        "framework_reported_source": "FRAMEWORK_LOG_LINE_DIRECT_CAPTURE",
    }


def summarise_curves(history: Sequence[Mapping[str, str]], best_epoch: int) -> dict[str, Any]:
    """Describe the training dynamics without opening a single image.

    Args:
        history: Parsed ``results.csv`` rows.
        best_epoch: The epoch the frozen rule selected.

    Returns:
        Loss endpoints, the fitness trajectory and where the best epoch sits.
    """
    if not history:
        return {}
    columns = [name for name in history[0] if name.startswith(("train/", "val/", "lr/"))]
    first, last = history[0], history[-1]

    def value(row: Mapping[str, str], name: str) -> Any:
        try:
            return round(float(row[name]), METRIC_PRECISION)
        except (KeyError, TypeError, ValueError):
            return NOT_EXPOSED

    curve = native_fitness_curve(history)
    tail = [item for epoch, item in curve if epoch > best_epoch]
    return {
        "epochs_logged": len(history),
        "loss_and_lr_columns": sorted(columns),
        "first_epoch": {name: value(first, name) for name in sorted(columns)},
        "last_epoch": {name: value(last, name) for name in sorted(columns)},
        "native_fitness_first": round(curve[0][1], METRIC_PRECISION) if curve else None,
        "native_fitness_last": round(curve[-1][1], METRIC_PRECISION) if curve else None,
        "native_fitness_best": (
            round(max(item for _, item in curve), METRIC_PRECISION) if curve else None
        ),
        "best_epoch": best_epoch,
        "epochs_after_best": len(tail),
        "improved_after_best": False,
        "observation": (
            "Read from results.csv only. No validation image was opened: image-level error "
            "analysis is a later, deliberate phase and starting it here would pre-empt it."
        ),
    }


def build_report(
    manifest: Mapping[str, Any], diagnostic: Mapping[str, Any], *, commit: str | None
) -> str:
    """Render the S0 report.

    Args:
        manifest: The assembled result manifest.
        diagnostic: The direct mask-IoU result.
        commit: Repository commit, when available.

    Returns:
        The report text.
    """
    mask = manifest["mask_metrics"]
    box = manifest["box_metrics_from_segmenter"]
    per_class = manifest["per_class_metrics"]
    macro = manifest["supported_macro_mask"]
    selection = manifest["checkpoint_selection"]
    execution = manifest["execution"]
    runtime = manifest["runtime"]
    weights = manifest["pretrained_weights"]
    optimizer = manifest["optimizer"]
    adapter = manifest["adapter"]
    rare = manifest["rare_class"]
    direct = manifest["direct_mask_iou"]
    curves = manifest["training_curves"]
    complexity = manifest["model_complexity"]
    checkpoints = manifest["checkpoints"]

    lines: list[str] = []
    add = lines.append

    add("# S0 - YOLO11n-seg segmentation baseline")
    add("")
    add(
        f"Phase {manifest['phase']} · classification `{manifest['status']}` · experiment "
        f"`{manifest['experiment_id']}` · baseline status "
        f"`{manifest['segmentation_baseline_status']}` · final segmenter "
        f"`{manifest['final_segmenter']}`"
    )
    add("")
    add(
        "**Every number here is a validation number.** The holdout has never been evaluated, "
        "and nothing below says anything about test performance. S0 is a baseline, not the "
        "project's final segmenter."
    )
    add("")
    if commit:
        add(f"Repository commit at production time: `{commit}`.")
        add("")

    add("## 1. Experimental question")
    add("")
    add(f"> {manifest['experiment_question']}")
    add("")
    add(f"`{manifest['experiment_role']}`. {manifest['final_segmenter_note']}")
    add("")

    add("## 2. Frozen S0 protocol")
    add("")
    add(
        "`PREDECLARED_PROTOCOL`. Frozen in phase 8B before S0 trained, protocol fingerprint "
        f"`{manifest['phase_8b_protocol_fingerprint']}`, verified against the committed phase 8B "
        "manifest before the first optimisation step."
    )
    add("")
    add("| Field | Value |")
    add("| --- | --- |")
    add(f"| Architecture | {manifest['architecture']} |")
    add(f"| imgsz | {manifest['imgsz']} |")
    add(f"| batch | {manifest['batch']} |")
    add(f"| seed | {manifest['seed']} |")
    add(f"| Pretrained weights | `{weights['identifier']}` SHA-256 `{weights['sha256']}` |")
    add("")

    add("## 3. Canonical representation versus the model adapter")
    add("")
    add(
        f"`MODEL_ADAPTER_LIMITATION`. Canonical ground truth is "
        f"`{adapter['canonical_ground_truth']}`; the labels S0 trained on are a "
        f"`{adapter['role']}` approved in phase 8B. The approval attaches to bytes, and those "
        "bytes were re-verified against the approved digests before training and again after "
        f"it: {adapter['counts']['instances']} instances over "
        f"{adapter['counts']['train_images']} train and "
        f"{adapter['counts']['validation_images']} validation images, "
        f"`regenerated: {adapter['regenerated']}`, `modified: {adapter['modified']}`."
    )
    add("")
    add(
        "Training read a hard-linked runtime view of those bytes rather than the audited "
        "directory itself, so the framework's `.cache` files did not land inside phase 8A's "
        "evidence. Image membership, image bytes and label bytes are identical; no label was "
        "converted and no coordinate was rewritten."
    )
    add("")

    add("## 4. Phase 8A fidelity context")
    add("")
    add(
        "`MODEL_ADAPTER_LIMITATION`. The YOLO segmentation format cannot express an interior "
        "hole or a disconnected mask, and phase 8A measured what that costs: mean mask IoU "
        "0.973066, median 0.984576, P05 0.918176, and **no instance round-trips exactly**. "
        "Every metric below is therefore a measurement of a model trained on an approximated "
        "representation of the canonical masks - which is one reason the direct IoU diagnostic "
        "in section 17 scores against the canonical COCO masks rather than against the adapter."
    )
    add("")

    add("## 5. Runtime provenance")
    add("")
    add(
        f"python {runtime['python']}, torch {runtime['torch']}, torchvision "
        f"{runtime['torchvision']}, ultralytics {runtime['ultralytics']}, CUDA "
        f"{runtime['cuda_runtime']}, {runtime['gpu_name']} ({runtime['gpu_arch']}, "
        f"{runtime['gpu_total_memory_bytes'] / 1024**3:.2f} GiB)."
    )
    add("")
    add(
        "A pre-run provenance record was written before the first optimisation step, so the "
        "protocol demonstrably preceded the result."
    )
    add("")

    add("## 6. Effective training configuration")
    add("")
    add("`COMPUTED_RESULT`, read back from the framework's own recorded arguments.")
    add("")
    add("| Argument | Effective value |")
    add("| --- | --- |")
    for name in sorted(manifest["effective_training_configuration"]):
        add(f"| `{name}` | {manifest['effective_training_configuration'][name]} |")
    add("")
    add(
        f"**Optimizer.** Declared policy `{optimizer['declared_policy']}`, which resolved to "
        f"**{optimizer['actual_resolved_optimizer']}** at lr0 {optimizer['effective_lr0']} and "
        f"momentum {optimizer['effective_momentum']}, evidence "
        f"`{optimizer['resolution_evidence']}`. {optimizer['declared_values_are_not_what_ran']} "
        f"Scheduler `{optimizer['scheduler']}`, weight decay {optimizer['weight_decay']}, "
        f"warmup {optimizer['warmup_epochs']} epochs."
    )
    add("")

    add("## 7. Native checkpoint-selection semantics")
    add("")
    add(
        f"`FRAMEWORK_CHECKPOINT_POLICY`. Policy `{selection['policy']}`, semantics "
        f"`{selection['semantics']}` - the unweighted sum of box and mask mAP@0.50:0.95, "
        "reviewed and accepted in phase 8B. **The checkpoint is therefore not selected on the "
        "primary reported metric**, and the epoch S0 reports need not be the epoch that "
        "maximised mask mAP@0.50:0.95 on its own."
    )
    add("")
    add(f"{selection['reconstruction']}")
    add("")
    add(
        f"`manual_epoch_selection: {selection['manual_epoch_selection']}` · "
        f"`mask_only_checkpoint_created: {selection['mask_only_checkpoint_created']}`"
    )
    add("")

    add("## 8. Training execution")
    add("")
    add("`COMPUTED_RESULT`. Exactly one full run.")
    add("")
    add("| Field | Value |")
    add("| --- | --- |")
    add(f"| Epochs configured | {selection['epochs_configured']} |")
    add(f"| Epochs logged | {selection['epochs_logged']} |")
    add(f"| Termination | {execution['termination']} |")
    add(f"| Early stopping | {execution['early_stopping']} |")
    add(
        f"| Training time | {execution['training_seconds']} s "
        f"(`{execution['training_seconds_source']}`; the framework's own log reports "
        f"{execution['framework_reported_hours']} hours for "
        f"{execution['framework_reported_epochs']} epochs) |"
    )
    add(f"| Peak GPU memory reserved | {execution['peak_gpu_memory_reserved_gib']} |")
    add(f"| Engineering aborts | {execution['engineering_aborts']} |")
    add(f"| Resumed | {execution['resumed']} |")
    add(f"| Run directory | `{execution['run_directory']}` (git-ignored) |")
    add("")
    if execution.get("peak_gpu_memory_note"):
        add(execution["peak_gpu_memory_note"])
        add("")
    reexecution = execution.get("artifact_write_reexecution", {})
    if reexecution.get("occurred"):
        add(
            "**One artifact-write re-execution is recorded.** "
            + reexecution["reason"]
            + " "
            + reexecution["fix"]
            + " Re-executed: "
            + ", ".join(reexecution["what_was_reexecuted"])
            + ". Not re-executed: "
            + reexecution["what_was_not_reexecuted"]
            + ". "
            + reexecution["why_this_is_not_a_second_experiment"]
        )
        add("")

    add("## 9. Checkpoint selection")
    add("")
    add(
        f"`FRAMEWORK_CHECKPOINT_POLICY`. Best epoch **{selection['best_epoch']}** at native "
        f"composite fitness {selection['best_native_fitness']}; next best epoch "
        f"{selection['next_best_epoch']} at {selection['next_best_native_fitness']}. The "
        "recorded best epoch was confirmed to be the argmax of the recomputed composite, so "
        "the frozen rule provably chose it."
    )
    add("")
    add("| Checkpoint | SHA-256 | Bytes |")
    add("| --- | --- | --- |")
    for name in ("best", "last"):
        entry = checkpoints[name]
        add(f"| `{entry['name']}` | `{entry['sha256']}` | {entry['size_bytes']} |")
    add("")
    add("Neither weight file is committed; the repository records the digest, not the binary.")
    add("")

    add("## 10. Primary mask result")
    add("")
    add(
        f"`PRIMARY_SCIENTIFIC_RESULT`. **mask mAP@0.50:0.95 = "
        f"{mask['mAP@0.50:0.95']}** on the frozen validation split."
    )
    add("")
    add(
        "One run, one configuration, no tuning. `deterministic: true` reduces run-to-run "
        "variance without removing it, so this is a single measurement and not an estimate "
        "with an interval."
    )
    add("")

    add("## 11. Secondary mask metrics")
    add("")
    add("| Metric | Value |")
    add("| --- | --- |")
    add(f"| mask mAP@0.50 | {mask['mAP@0.50']} |")
    add(f"| mask precision | {mask['precision']} |")
    add(f"| mask recall | {mask['recall']} |")
    add("")
    add(
        "**Operating-point caveat.** Ultralytics reports one precision/recall pair at the "
        "F1-maximising point rather than at a fixed confidence, so both figures partly reflect "
        "where that point landed. Read them as a hint about the precision/recall balance, never "
        "as threshold-independent properties. No threshold was tuned."
    )
    add("")

    add("## 12. Per-class mask metrics")
    add("")
    add("| Class | precision | recall | AP@0.50 | AP@0.50:0.95 |")
    add("| --- | --- | --- | --- | --- |")
    for name in sorted(per_class):
        entry = per_class[name]["mask"]
        add(
            f"| `{name}` | {entry['precision']} | {entry['recall']} | {entry['AP@0.50']} | "
            f"{entry['AP@0.50:0.95']} |"
        )
    add("")

    add("## 13. Box metrics from the segmenter")
    add("")
    add(f"`COMPUTED_RESULT`. {manifest['box_metrics_note']}")
    add("")
    add("| Metric | Value |")
    add("| --- | --- |")
    add(f"| box mAP@0.50:0.95 | {box['mAP@0.50:0.95']} |")
    add(f"| box mAP@0.50 | {box['mAP@0.50']} |")
    add(f"| box precision | {box['precision']} |")
    add(f"| box recall | {box['recall']} |")
    add("")
    add("| Class | box precision | box recall | box AP@0.50 | box AP@0.50:0.95 |")
    add("| --- | --- | --- | --- | --- |")
    for name in sorted(per_class):
        entry = per_class[name]["box"]
        add(
            f"| `{name}` | {entry['precision']} | {entry['recall']} | {entry['AP@0.50']} | "
            f"{entry['AP@0.50:0.95']} |"
        )
    add("")
    add(
        "**These are not comparable to the frozen detector's numbers.** D2 is a different "
        "model trained under a different protocol, and no controlled detector-versus-segmenter "
        "comparison has been run. Reading the two side by side would be comparing two "
        "experiments that differ in more than one thing."
    )
    add("")

    add("## 14. Supported mask macro")
    add("")
    add(
        f"`COMPUTED_RESULT`. `{macro['metric']}` = **{macro['value']}**, the unweighted mean of "
        f"per-class mask AP@0.50:0.95 over the classes the frozen support rule admits "
        f"({', '.join('`' + name + '`' for name in macro['admitted_classes'])})."
    )
    add("")
    add(f"Rule (`{macro['rule_origin']}`): {macro['rule']}.")
    add("")
    add(
        f"`is_a_selection_metric: {macro['is_a_selection_metric']}`. {macro['reason']} "
        "**No segmentation winner is declared.**"
    )
    add("")

    add("## 15. The `vest_loose` limitation")
    add("")
    add(f"`LIMITATION` · status `{rare['status']}`.")
    add("")
    add(rare["limitation"])
    add("")
    add(
        f"It is reported in full above and in the diagnostic below "
        f"(`reported_in_full: {rare['reported_in_full']}`), and it decides nothing "
        f"(`may_decide_anything: {rare['may_decide_anything']}`)."
    )
    add("")

    add("## 16. Direct mask-IoU protocol")
    add("")
    add(
        f"`PREDECLARED_PROTOCOL`. `{direct['protocol']}`, frozen in "
        f"`{direct['protocol_config']}` **before the first optimisation step**, fingerprint "
        f"`{direct['protocol_fingerprint']}`."
    )
    add("")
    add(
        f"Ground truth is `{direct['ground_truth_source']}` - the canonical phase 5D COCO "
        "validation masks, **never the YOLO adapter**, whose own approximation would otherwise "
        "be folded into the model's score."
    )
    add("")
    add("| Setting | Value |")
    add("| --- | --- |")
    for name in sorted(diagnostic["inference"]):
        add(f"| `{name}` | {diagnostic['inference'][name]} |")
    add(f"| matching | `{diagnostic['matching_algorithm']}` |")
    add("")
    add(
        "The confidence and NMS IoU are a **predeclared operational diagnostic threshold**, "
        "fixed before any S0 performance number existed. No sweep, no second threshold, and no "
        "test-time augmentation. Matching is one-to-one within an image and a class, solved to "
        "maximise total IoU; an assigned pair sharing no pixel is discarded rather than counted "
        "as a match."
    )
    add("")

    add("## 17. Direct mask-IoU results")
    add("")
    add("`DIRECT_IOU_DIAGNOSTIC`, executed exactly once on the selected checkpoint.")
    add("")
    global_result = diagnostic["global"]
    add("| Diagnostic | Value |")
    add("| --- | --- |")
    add(f"| `matched_mask_iou_mean` | **{global_result[MATCHED_MASK_IOU_MEAN]}** |")
    add(f"| `gt_normalized_mask_iou` | **{global_result[GT_NORMALIZED_MASK_IOU]}** |")
    add(f"| `gt_match_coverage` | {global_result[GT_MATCH_COVERAGE]} |")
    add(f"| `gt_iou50_coverage` | {global_result[GT_IOU50_COVERAGE]} |")
    add(f"| `gt_iou75_coverage` | {global_result[GT_IOU75_COVERAGE]} |")
    add("")
    add("| Count | Value |")
    add("| --- | --- |")
    add(f"| Canonical GT instances | {global_result['gt_count']} |")
    add(f"| Predicted instances | {global_result['prediction_count']} |")
    add(f"| Assignments with positive overlap | {global_result['matched_count']} |")
    add(f"| Unmatched GT | {global_result['unmatched_gt']} |")
    add(f"| Unmatched predictions | {global_result['unmatched_predictions']} |")
    add("")
    add("Per class:")
    add("")
    add(
        "| Class | GT | pred | matched | matched mean IoU | GT-normalised IoU | IoU>=0.50 | "
        "IoU>=0.75 |"
    )
    add("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for name in sorted(diagnostic["per_class"]):
        entry = diagnostic["per_class"][name]
        add(
            f"| `{name}` | {entry['gt_count']} | {entry['prediction_count']} | "
            f"{entry['matched_count']} | {entry[MATCHED_MASK_IOU_MEAN]} | "
            f"{entry[GT_NORMALIZED_MASK_IOU]} | {entry[GT_IOU50_COVERAGE]} | "
            f"{entry[GT_IOU75_COVERAGE]} |"
        )
    add("")
    add(
        f"`{rare['name']}` remains `{rare['status']}` here too: its direct-IoU figures stand on "
        "one validation image and must not be used to tune, rank or select anything."
    )
    add("")

    add("## 18. How AP and direct IoU relate")
    add("")
    add("`LIMITATION`. The two answer different questions and are not interchangeable.")
    add("")
    add(
        "- **`matched_mask_iou_mean`** answers, approximately: *when S0 produced an overlapping "
        "same-class instance, how similar was its mask to the canonical one?* It says nothing "
        "about the instances S0 never found."
    )
    add(
        "- **`gt_normalized_mask_iou`** answers, approximately: *across every canonical "
        "instance, counting a miss as zero, how much same-class mask overlap did S0 recover at "
        "the frozen operating point?* It folds mask quality and coverage into one figure."
    )
    add(
        "- **mask mAP@0.50:0.95** is neither. It averages precision over recall at a range of "
        "IoU thresholds and is sensitive to confidence ranking, which the two diagnostics above "
        "ignore entirely."
    )
    add("")
    add(
        f"{direct['is_not_average_precision']} None of the three may be substituted for another, "
        "and none of them was used to change S0."
    )
    add("")

    add("## 19. Training dynamics")
    add("")
    add("`COMPUTED_RESULT`, read from `results.csv` only.")
    add("")
    add("| Quantity | First epoch | Last epoch |")
    add("| --- | --- | --- |")
    for name in sorted(curves.get("loss_and_lr_columns", [])):
        add(f"| `{name}` | {curves['first_epoch'][name]} | {curves['last_epoch'][name]} |")
    add("")
    add(
        f"Native composite fitness ran {curves.get('native_fitness_first')} at the first logged "
        f"epoch to {curves.get('native_fitness_last')} at the last, peaking at "
        f"{curves.get('native_fitness_best')} on epoch {curves.get('best_epoch')}, with "
        f"{curves.get('epochs_after_best')} epochs logged after it."
    )
    add("")
    add(curves.get("observation", ""))
    add("")

    add("## 20. Confusion and metric figures")
    add("")
    add(
        "`COMPUTED_RESULT`. Metric-only figures are committed under `reports/figures/`: "
        + ", ".join(f"`{name}`" for name in manifest["committed_figures"])
        + "."
    )
    add("")
    add(
        "The framework also writes `train_batch*.jpg`, `val_batch*.jpg` and `labels.jpg`, which "
        "render dataset imagery and prediction montages. Those stay in the git-ignored run "
        "directory: committing them would publish dataset images and pre-empt the deliberate "
        "error-analysis phase. **No validation image was opened to explain an individual "
        "failure**, and no image-level error analysis was performed."
    )
    add("")

    add("## 21. Resource use")
    add("")
    add("| Field | Value |")
    add("| --- | --- |")
    add(f"| Parameters | {complexity.get('parameters')} |")
    add(f"| GFLOPs | {complexity.get('gflops')} |")
    add(f"| Layers | {complexity.get('layers')} |")
    add(f"| Training time | {execution['training_seconds']} s |")
    add(f"| Peak GPU memory reserved | {execution['peak_gpu_memory_reserved_gib']} |")
    add(f"| GPU | {runtime['gpu_name']} ({runtime['gpu_arch']}) |")
    add("")
    speed = manifest["framework_validation_speed"]
    add(
        f"`{speed['label']}` (ms per image): "
        + ", ".join(
            f"{name} {value}" for name, value in sorted(speed["values_ms_per_image"].items())
        )
        + f". {speed['caveat']}"
    )
    add("")

    add("## 22. Holdout compliance")
    add("")
    add(f"`HOLDOUT_POLICY` · status `{manifest['test']['status']}`.")
    add("")
    add(manifest["test"]["reason"])
    add("")

    add("## 23. Limitations")
    add("")
    add("`LIMITATION`. What this experiment does and does not establish.")
    add("")
    add(
        "- **One run, one configuration.** Nothing was repeated, so run-to-run variance on this "
        "setup is UNKNOWN and every figure is a single measurement."
    )
    add(
        "- **Validation only.** The holdout has never been evaluated. Nothing here predicts "
        "test performance."
    )
    add(
        "- **No comparison exists.** S0 is the only segmentation experiment, so no claim about "
        "architecture, resolution or capacity is supported by it."
    )
    add(
        "- **Trained on an approximated representation.** The YOLO adapter cannot express holes "
        "or disconnected masks; phase 8A measured the cost and phase 8B accepted it, but the "
        "model never saw the canonical geometry."
    )
    add(
        "- **The checkpoint was not selected on the reported metric.** The native composite "
        "includes box mAP@0.50:0.95, which is a reviewed and accepted property of this "
        "baseline, not an oversight."
    )
    add(
        "- **Direct IoU is reported at one operating point.** Confidence 0.25 and NMS IoU 0.70 "
        "were predeclared; a different threshold would give different coverage figures, and no "
        "sweep was run to find a flattering one."
    )
    add(
        f"- **`{rare['name']}` stands on one validation image.** Every figure for it carries "
        "high sampling uncertainty."
    )
    add(
        "- **No latency benchmark.** The framework validation speed above is descriptive; the "
        "standardised detector-versus-segmenter study the project's question needs is a later "
        "phase."
    )
    add("")

    add("## 24. Next decision")
    add("")
    add(
        f"`PENDING_HUMAN_REVIEW`. `segmentation_baseline_status: "
        f"{manifest['segmentation_baseline_status']}`, `final_segmenter: "
        f"{manifest['final_segmenter']}`."
    )
    add("")
    add(
        "S0 is **not** frozen as the final segmenter, and no S1 or alternative model, "
        "resolution or batch was trained. The next phase is human review of this result and "
        "the planning of a segmentation experiment protocol - which, like phase 7A, must be "
        "frozen before the experiments it would decide exist."
    )
    add("")
    add("---")
    add("")
    add(
        f"`S0_experiment_sha256` `{manifest['S0_experiment_sha256']}` · direct mask-IoU result "
        f"`{direct['result_artifact']}` SHA-256 `{direct['result_sha256']}`."
    )
    add("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Run the S0 segmentation baseline and its direct mask-IoU diagnostic.

    Args:
        argv: Command-line arguments.

    Returns:
        Process exit status.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="run every precondition check without fetching weights, training or writing",
    )
    parser.add_argument(
        "--diagnostic-only",
        action="store_true",
        help=(
            "reuse the completed S0 run and re-render the artifacts, without training. For "
            "recovering from a failure after training; it never re-runs the experiment."
        ),
    )
    args = parser.parse_args(argv)
    if args.verify_only and args.diagnostic_only:
        print(
            "REFUSED: --verify-only and --diagnostic-only are mutually exclusive.", file=sys.stderr
        )
        return 2

    paths = ProjectPaths.from_root()
    if holdout_unlocked():
        print(
            f"REFUSED: {HOLDOUT_UNLOCK_ENV_VAR} is set. This experiment reads development data "
            "only and declines to run in an environment where the holdout is unlocked.",
            file=sys.stderr,
        )
        return 2

    started_at = datetime.now(UTC).isoformat(timespec="seconds")

    # --- preflight -------------------------------------------------------------
    try:
        historical = verify_historical(paths)
        config = load_segmentation_baseline_config(paths.configs / BASELINE_YAML)
        protocol = load_mask_iou_config(paths.configs / MASK_IOU_YAML)
        verified_protocol = verify_protocol(paths, config)
        detector = verify_detector(paths)
        verify_no_holdout(paths, config)
        adapter = verify_adapter(paths, config, stage="preflight")
        approval = read_json(paths.reports / APPROVAL_JSON)
        approval_sha256 = sha256_file(paths.reports / APPROVAL_JSON)
        audit = read_json(paths.reports / AUDIT_MANIFEST_JSON)
        class_map = {name: int(index) for name, index in audit["class_map"].items()}
        canonical = {
            "class_map_sha256": str(audit["class_map_sha256"]),
            "split_assignment_sha256": str(
                audit["canonical_fingerprints"]["split_assignment_sha256"]
            ),
            "canonical_task_manifest_sha256": str(audit["canonical_task_manifest_sha256"]),
        }
        if protocol["class_map_sha256"] != canonical["class_map_sha256"]:
            msg = "the diagnostic protocol's class map digest disagrees with the frozen record"
            raise ExperimentError(msg)
        if protocol["split_assignment_sha256"] != canonical["split_assignment_sha256"]:
            msg = "the diagnostic protocol's split digest disagrees with the frozen record"
            raise ExperimentError(msg)
        if approval["status"] != "APPROVED_FOR_CONTROLLED_TRAINING":
            msg = f"the adapter is not approved for training: {approval['status']}"
            raise ExperimentError(msg)
        ground_truth_document = load_canonical_validation(
            paths, str(protocol["ground_truth_document"])
        )
        ground_truth = canonical_instances(ground_truth_document["document"], class_map)
        canonical["validation_annotations_sha256"] = ground_truth_document["sha256"]
        support = validation_support(ground_truth, class_map)
    except (
        ConfigError,
        SegmentationExperimentConfigError,
        SegmentationRunError,
        ExperimentError,
    ) as exc:
        if isinstance(exc, AdapterMismatchError):
            classification = ADAPTER_FINGERPRINT_MISMATCH
        elif isinstance(exc, ProtocolMismatchError):
            classification = S0_PROTOCOL_MISMATCH
        else:
            classification = BLOCKED
        print(f"{classification}: {exc}", file=sys.stderr)
        return 2

    try:
        runtime = runtime_facts()
    except RunError as exc:
        print(f"{BLOCKED}: {exc}", file=sys.stderr)
        return 2

    gt_total = sum(len(items) for items in ground_truth.values())
    print(
        f"protocol   {config.experiment_id}  {config.architecture}  imgsz "
        f"{config.training['imgsz']}  batch {config.training['batch']}  VERIFIED"
    )
    print(f"adapter    {adapter['counts']['instances']} instances  fingerprints VERIFIED")
    print(f"detector   {detector['selected_experiment']}  {detector['status']}")
    print(f"canonical  {len(ground_truth)} validation images  {gt_total} GT instances")
    print(f"diagnostic protocol sha256 {protocol.fingerprint()}")
    print(
        f"runtime    torch {runtime['torch']}  ultralytics {runtime['ultralytics']}  "
        f"{runtime['gpu_name']}"
    )

    if args.verify_only:
        print("VERIFIED: preconditions hold. Nothing fetched, trained or written.")
        print(f"holdout    {HOLDOUT_STATUS}")
        return 0

    try:
        weights = ensure_pretrained_weights(paths, config.weight_identifier)
    except RunError as exc:
        print(f"{BLOCKED}: {exc}", file=sys.stderr)
        return 2
    if weights["sha256"] != approval["architecture_decision"].get(
        "weight_sha256", weights["sha256"]
    ):
        print(f"{BLOCKED}: the pretrained checkpoint is not the approved one", file=sys.stderr)
        return 2
    print(f"weights    {weights['identifier']}  sha256 {weights['sha256']}")

    # --- runtime view -----------------------------------------------------------
    try:
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
    print(f"view       {RUNTIME_VIEW_ROOT}  {runtime_view['transfer_modes']}  labels VERIFIED")

    descriptor = paths.root / RUNTIME_VIEW_ROOT / "dataset.yaml"
    run_directory = paths.root / RUN_ROOT / RUN_NAME

    # --- pre-run provenance ------------------------------------------------------
    if not args.diagnostic_only:
        prerun = ProvenanceRecord.create(
            name="segmentation_S0_prerun",
            phase=8,
            config={
                "segmentation_baseline": f"configs/{BASELINE_YAML}",
                "direct_mask_iou_protocol": f"configs/{MASK_IOU_YAML}",
                "protocol_fingerprint": config.fingerprint(),
                "direct_mask_iou_protocol_fingerprint": protocol.fingerprint(),
            },
            details={
                "phase": PHASE,
                "experiment_id": EXPERIMENT_ID,
                "state": "BEFORE_FIRST_OPTIMIZATION_STEP",
                "started_at": started_at,
                "architecture": config.architecture,
                "imgsz": config.training["imgsz"],
                "batch": config.training["batch"],
                "epochs": config.training["epochs"],
                "seed": config.seed,
                "device_requirement": config.device_requirement,
                "adapter_approval_sha256": approval_sha256,
                "adapter_fingerprints": adapter["fingerprints"],
                "adapter_counts": adapter["counts"],
                "canonical_fingerprints": canonical,
                "pretrained_weight_sha256": weights["sha256"],
                "pretrained_weight_bytes": weights["size_bytes"],
                "checkpoint_selection_policy": CHECKPOINT_SELECTION_POLICY,
                "checkpoint_selection_semantics": CHECKPOINT_SELECTION_SEMANTICS,
                "runtime": runtime,
                "historical_artifact_digests": historical,
                "holdout": HOLDOUT_STATUS,
                "models_trained_so_far": 0,
            },
            repo_root=paths.root,
        )
        prerun.add_input(paths.configs / BASELINE_YAML, relative_to=paths.root)
        prerun.add_input(paths.configs / MASK_IOU_YAML, relative_to=paths.root)
        prerun.add_input(paths.reports / APPROVAL_JSON, relative_to=paths.root)
        prerun.write_json(paths.reports / PRERUN_PROVENANCE_JSON)
        print(f"prerun     reports/{PRERUN_PROVENANCE_JSON}  written before the first step")

    # --- the single training run --------------------------------------------------
    if args.diagnostic_only:
        if not (run_directory / "weights" / "best.pt").is_file():
            print(f"{BLOCKED}: --diagnostic-only needs a completed S0 run", file=sys.stderr)
            return 2
        execution = {
            "run_directory": f"{RUN_ROOT}/{RUN_NAME}",
            "wall_clock_seconds": None,
            "peak_gpu_memory_reserved_bytes": None,
            "peak_gpu_memory_reserved_gib": None,
            "peak_gpu_memory_allocated_bytes": None,
            "log_text": (run_directory / "framework.log").read_text(
                encoding="utf-8", errors="replace"
            )
            if (run_directory / "framework.log").is_file()
            else "",
            "reused_completed_run": True,
            "artifact_write_reexecution": {
                "occurred": True,
                "what_was_reexecuted": [
                    "the authoritative framework validation",
                    "the direct mask-IoU diagnostic",
                ],
                "what_was_not_reexecuted": "training - the single S0 run and its weights are "
                "unchanged, and the checkpoint the frozen rule selected is the same file",
                "reason": (
                    "The first attempt trained, validated and ran the diagnostic successfully, "
                    "then refused to write its artifacts: the sensitive-content scan found "
                    "machine-specific absolute paths that Ultralytics records in args.yaml and "
                    "the runner was copying verbatim into the effective-configuration block. No "
                    "artifact was written, so nothing was published and nothing was corrected "
                    "after the fact."
                ),
                "fix": (
                    "Absolute paths recorded by the framework are now rewritten "
                    "repository-relative before they reach an artifact. No metric, threshold, "
                    "hyperparameter or selection rule was touched."
                ),
                "why_this_is_not_a_second_experiment": (
                    "Validation and the diagnostic are deterministic functions of a fixed "
                    "checkpoint, fixed data and fixed settings, all of which are unchanged. "
                    "Re-executing them reproduces the same numbers; what changed is only "
                    "whether they could be written down."
                ),
            },
        }
        print("training   REUSED the completed S0 run (nothing was trained)")
    else:
        try:
            execution = train(paths, config, weights, descriptor)
        except MemoryConstraintError as exc:
            print(f"{MEMORY_CONSTRAINT_REVIEW_REQUIRED}: {exc}", file=sys.stderr)
            return 2
        except ExperimentError as exc:
            print(f"{TRAINING_FAILED}: {exc}", file=sys.stderr)
            return 2
        execution["reused_completed_run"] = False
        execution["artifact_write_reexecution"] = {"occurred": False}
        print(
            f"training   COMPLETE  {execution['wall_clock_seconds']} s  peak "
            f"{execution['peak_gpu_memory_reserved_gib']} GiB"
        )

    # --- checkpoint selection -------------------------------------------------------
    try:
        history = read_epoch_history(run_directory)
        resolved = resolved_training_arguments(run_directory)
        selection = verify_checkpoint_selection(history, resolved)
        checkpoints = {
            "best": checkpoint_record(run_directory / "weights" / "best.pt"),
            "last": checkpoint_record(run_directory / "weights" / "last.pt"),
        }
    except (SegmentationRunError, ExperimentError) as exc:
        print(f"{TRAINING_FAILED}: {exc}", file=sys.stderr)
        return 2

    duration = training_duration(history, execution.get("log_text", ""))
    execution.update(duration)
    if execution.get("peak_gpu_memory_reserved_bytes") is None:
        execution["peak_gpu_memory_reserved_bytes"] = "NOT_PERSISTED_FOR_THIS_RUN"
        execution["peak_gpu_memory_reserved_gib"] = "NOT_PERSISTED_FOR_THIS_RUN"
        execution["peak_gpu_memory_note"] = (
            "The training process measured peak device memory but exited before the artifact "
            "could be written, and the figure is not recoverable from any file on disk. It is "
            "recorded as not persisted rather than reconstructed from terminal output, because "
            "a number a reader cannot trace to an artifact is not evidence. The frozen batch of "
            "8 completed without an out-of-memory event, which is what the protocol required."
        )

    configured = int(config.training["epochs"])
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
    execution["runs"] = 1
    print(
        f"checkpoint best epoch {selection['best_epoch']}  native fitness "
        f"{selection['best_native_fitness']}  {execution['termination']}"
    )

    optimizer = optimizer_evidence_from_log(execution.get("log_text", ""))

    # --- authoritative validation -----------------------------------------------------
    try:
        validation = validate(paths, config, run_directory / "weights" / "best.pt")
    except ExperimentError as exc:
        print(f"{TRAINING_FAILED}: {exc}", file=sys.stderr)
        return 2
    print(
        f"validation mask mAP@0.50:0.95 {validation['mask']['mAP@0.50:0.95']}  "
        f"box mAP@0.50:0.95 {validation['box']['mAP@0.50:0.95']}"
    )

    macro = supported_macro(validation["per_class"], support)
    print(f"macro      {macro['metric']} {macro['value']}  over {macro['admitted_classes']}")

    # --- the direct mask-IoU diagnostic -------------------------------------------------
    try:
        diagnostic = run_direct_iou(
            paths, protocol, run_directory / "weights" / "best.pt", ground_truth, class_map
        )
    except ExperimentError as exc:
        print(f"{DIRECT_IOU_EVALUATION_FAILED}: {exc}", file=sys.stderr)
        return 2
    print(
        f"direct IoU matched mean {diagnostic['global'][MATCHED_MASK_IOU_MEAN]}  "
        f"GT-normalised {diagnostic['global'][GT_NORMALIZED_MASK_IOU]}  "
        f"coverage {diagnostic['global'][GT_MATCH_COVERAGE]}"
    )

    # --- post-run verification ------------------------------------------------------
    try:
        verify_adapter(paths, config, stage="after training")
        after = verify_historical(paths)
        changed = sorted(name for name in historical if historical[name] != after.get(name))
        if changed:
            msg = f"historical artifacts changed during this phase: {changed}"
            raise ExperimentError(msg)
    except (AdapterMismatchError, ExperimentError) as exc:
        classification = (
            ADAPTER_FINGERPRINT_MISMATCH if isinstance(exc, AdapterMismatchError) else BLOCKED
        )
        print(f"{classification}: {exc}", file=sys.stderr)
        return 2

    # --- artifacts ---------------------------------------------------------------------
    # Priority order matters. The authoritative validation is the single reported
    # evaluation, so its curves and confusion matrices are the ones that provably
    # correspond to the published metrics; the training directory contributes only
    # `results.png`, the per-epoch curve it alone produces.
    figures = copy_metric_figures(
        paths.figures / "segmentation_S0",
        sorted((paths.root / RUN_ROOT / f"{RUN_NAME}_val").glob("*.png"))
        + sorted(run_directory.glob("*.png")),
    )
    complexity = model_complexity(execution.get("log_text", ""))

    curves = summarise_curves(history, int(selection["best_epoch"]))

    diagnostic_payload = {
        "schema_version": SCHEMA_VERSION,
        "phase": PHASE,
        "experiment_id": EXPERIMENT_ID,
        "protocol": PROTOCOL_NAME,
        "protocol_fingerprint": protocol.fingerprint(),
        "protocol_config": f"configs/{MASK_IOU_YAML}",
        "checkpoint": {
            "name": checkpoints["best"]["name"],
            "sha256": checkpoints["best"]["sha256"],
            "size_bytes": checkpoints["best"]["size_bytes"],
            "selected_by": CHECKPOINT_SELECTION_POLICY,
        },
        "ground_truth": {
            "source": GROUND_TRUTH_SOURCE,
            "document": ground_truth_document["relative_path"],
            "sha256": ground_truth_document["sha256"],
            "is_the_yolo_adapter": False,
        },
        "split": "validation",
        "split_assignment_sha256": canonical["split_assignment_sha256"],
        "class_map_sha256": canonical["class_map_sha256"],
        "inference": diagnostic["inference"],
        "matching": {
            "algorithm": MATCHING_ALGORITHM,
            "implementation": "scipy.optimize.linear_sum_assignment",
            "scope": "PER_IMAGE_PER_CLASS_ONE_TO_ONE",
            "zero_overlap_policy": "ASSIGNED_PAIRS_WITH_ZERO_IOU_ARE_NOT_MATCHES",
            "unmatched_gt_policy": "CONTRIBUTES_ZERO_TO_GT_NORMALIZED_MASK_IOU",
            "deterministic": True,
        },
        "images": diagnostic["images"],
        "global": diagnostic["global"],
        "per_class": diagnostic["per_class"],
        "rare_class_warning": {
            "class": config.rare_class,
            "status": RARE_CLASS_STATUS,
            "detail": (
                "vest_loose holds one validation source image and eight instances under the "
                "frozen split. Its direct-IoU figures carry high sampling uncertainty, are "
                "reported for completeness, and must not be used to tune, rank or select."
            ),
        },
        "runs": 1,
        "contains_raw_masks_or_imagery": False,
        "test": {"status": HOLDOUT_STATUS, "reason": HOLDOUT_REASON},
    }
    diagnostic_sha256 = write_json(paths.reports / MASK_IOU_JSON, diagnostic_payload)

    manifest = build_result_manifest(
        config=config,
        protocol=protocol,
        verified_protocol=verified_protocol,
        adapter=adapter,
        approval_sha256=approval_sha256,
        canonical=canonical,
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
        validation=validation,
        macro=macro,
        diagnostic=diagnostic,
        diagnostic_sha256=diagnostic_sha256,
        complexity=complexity,
        figures=figures,
        curves=curves,
    )
    report = build_report(manifest, diagnostic, commit=git_commit(paths.root))

    findings = (
        scan_for_sensitive(report)
        + scan_for_sensitive(json.dumps(manifest))
        + scan_for_sensitive(json.dumps(diagnostic_payload))
    )
    if findings:
        print(f"{BLOCKED}: sensitive content in the emitted artifacts: {findings}", file=sys.stderr)
        return 2

    manifest_sha256 = write_json(paths.reports / RESULT_MANIFEST_JSON, manifest)
    (paths.reports / REPORT_MD).write_text(report, encoding="utf-8", newline="\n")

    record = ProvenanceRecord.create(
        name="segmentation_S0_baseline",
        phase=8,
        config={
            "segmentation_baseline": f"configs/{BASELINE_YAML}",
            "direct_mask_iou_protocol": f"configs/{MASK_IOU_YAML}",
            "protocol_fingerprint": config.fingerprint(),
            "direct_mask_iou_protocol_fingerprint": protocol.fingerprint(),
        },
        details={
            "phase": PHASE,
            "classification": COMPLETE,
            "experiment_id": EXPERIMENT_ID,
            "S0_experiment_sha256": manifest["S0_experiment_sha256"],
            "architecture": config.architecture,
            "epochs_configured": configured,
            "epochs_completed": logged,
            "best_epoch": selection["best_epoch"],
            "best_native_fitness": selection["best_native_fitness"],
            "checkpoint_selection_policy": CHECKPOINT_SELECTION_POLICY,
            "best_checkpoint_sha256": checkpoints["best"]["sha256"],
            "primary_metric": "mask_mAP@0.50:0.95",
            "primary_value": validation["mask"]["mAP@0.50:0.95"],
            "supported_macro_mask_map50_95": macro["value"],
            "direct_mask_iou_gt_normalized": diagnostic["global"][GT_NORMALIZED_MASK_IOU],
            "direct_mask_iou_matched_mean": diagnostic["global"][MATCHED_MASK_IOU_MEAN],
            "adapter_fingerprints": adapter["fingerprints"],
            "adapter_regenerated": False,
            "models_trained_in_this_phase": 1,
            "alternative_segmenters_trained": 0,
            "final_segmenter": FINAL_SEGMENTER,
            "detector_touched": False,
            "holdout_accessed": False,
            "historical_artifacts_unchanged": True,
        },
        repo_root=paths.root,
    )
    for name in (BASELINE_YAML, MASK_IOU_YAML):
        record.add_input(paths.configs / name, relative_to=paths.root)
    for name in (APPROVAL_JSON, PROTOCOL_MANIFEST_JSON, AUDIT_MANIFEST_JSON, FINAL_DETECTOR_JSON):
        record.add_input(paths.reports / name, relative_to=paths.root)
    for name in (RESULT_MANIFEST_JSON, MASK_IOU_JSON, REPORT_MD):
        record.add_output(paths.reports / name, relative_to=paths.root)
    record.write_json(paths.reports / PROVENANCE_JSON)

    print(COMPLETE)
    print(f"primary    mask mAP@0.50:0.95 {validation['mask']['mAP@0.50:0.95']}  (validation only)")
    print(
        f"direct IoU GT-normalised {diagnostic['global'][GT_NORMALIZED_MASK_IOU]}  "
        f"matched mean {diagnostic['global'][MATCHED_MASK_IOU_MEAN]}"
    )
    print(f"fingerprint S0_experiment_sha256 {manifest['S0_experiment_sha256']}")
    print(f"manifest   reports/{RESULT_MANIFEST_JSON}  sha256 {manifest_sha256}")
    print(f"diagnostic reports/{MASK_IOU_JSON}  sha256 {diagnostic_sha256}")
    print(f"report     reports/{REPORT_MD}")
    print(f"segmenter  {FINAL_SEGMENTER}")
    print(f"holdout    {HOLDOUT_STATUS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
