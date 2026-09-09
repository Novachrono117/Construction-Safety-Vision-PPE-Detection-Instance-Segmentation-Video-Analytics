"""Freeze the S0 segmentation baseline protocol and prove the runtime works.

Phase 8B. It records a human architecture decision, formally approves the label
bytes phase 8A audited, freezes the S0 protocol, and runs one minimal
non-experimental smoke test. It does **not** run S0, and it reports no model
performance: a single epoch's AP is not a result and is never recorded as one.

The order matters, and it is the point of the script:

* every phase 8A artifact is digested **before** anything else happens, and
  digested again at the end, so "the historical evidence is unchanged" is a
  measurement rather than an intention;
* the frozen detector's manifest and checkpoint bytes are verified and then left
  entirely alone - nothing here trains, validates or infers with it;
* the runtime label bytes are checked against phase 8A's fingerprints before a
  single weight is fetched, so the approval attaches to bytes that provably are
  the audited ones, and a mismatch stops the phase instead of silently
  regenerating them;
* the segmentation-specific framework arguments in the protocol are verified
  against the **installed** effective configuration, so the file cannot claim a
  default the library does not have;
* the checkpoint rule's actual behaviour is read out of the installed source
  before training, because ``best.pt`` for a segmentation model is selected on
  the sum of the mask and box summary metrics, not on the mask metric alone,
  and discovering that after the run would be discovering it too late.

Runs offline apart from obtaining the standard pretrained checkpoint through
Ultralytics' own asset mechanism.

Requires:

* ``configs/segmentation_baseline.yaml``
* ``reports/segmentation_adapter_audit_manifest.json``   phase 8A
* ``reports/final_detector_manifest.json``               phase 7D
* ``reports/task_dataset_manifest.json``                 phase 5D
* ``reports/split_manifest.json``                        phase 5C.2
* ``data/processed/adapters/yolo_segmentation_audit/``   phase 8A

Writes:
    reports/segmentation_adapter_approval.json
    reports/segmentation_S0_manifest.json
    reports/segmentation_S0_protocol.md
    reports/segmentation_S0_protocol.provenance.json
    artifacts/segmentation/S0_smoke/                     (git-ignored)
    artifacts/weights/yolo11n-seg.pt                     (git-ignored)

Usage:
    uv run python scripts/freeze_segmentation_baseline.py
    uv run python scripts/freeze_segmentation_baseline.py --verify-only
"""

from __future__ import annotations

import argparse
import inspect
import json
import re
import shutil
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from construction_safety_vision.config import ConfigError
from construction_safety_vision.data.canonical import scan_for_sensitive
from construction_safety_vision.data.materialization import digest
from construction_safety_vision.data.segmentation_adapter import label_fingerprint
from construction_safety_vision.detection_freeze import (
    FinalDetectorError,
    load_final_detector,
)
from construction_safety_vision.detection_run import (
    RunError,
    capture_framework_log,
    ensure_pretrained_weights,
    optimizer_evidence_from_log,
    release_framework_log,
    runtime_facts,
)
from construction_safety_vision.paths import ProjectPaths
from construction_safety_vision.provenance import ProvenanceRecord, git_commit, sha256_file
from construction_safety_vision.segmentation_experiment import (
    ADAPTER_ROLE,
    CANONICAL_GROUND_TRUTH,
    RARE_CLASS_CLASSIFICATION,
    SUPPORT_RULE,
    SegmentationBaselineConfig,
    SegmentationExperimentConfigError,
    load_segmentation_baseline_config,
)
from construction_safety_vision.splits import HOLDOUT_UNLOCK_ENV_VAR, holdout_unlocked

CONFIG_YAML = "segmentation_baseline.yaml"

AUDIT_MANIFEST_JSON = "segmentation_adapter_audit_manifest.json"
AUDIT_REPORT_MD = "segmentation_adapter_fidelity_report.md"
AUDIT_TABLE_CSV = "segmentation_adapter_fidelity.csv"
AUDIT_PROVENANCE_JSON = "segmentation_adapter_audit.provenance.json"
AUDIT_CONFIG_YAML = "segmentation_adapter_audit.yaml"

FINAL_DETECTOR_JSON = "final_detector_manifest.json"
TASK_MANIFEST_JSON = "task_dataset_manifest.json"
SPLIT_MANIFEST_JSON = "split_manifest.json"

APPROVAL_JSON = "segmentation_adapter_approval.json"
S0_MANIFEST_JSON = "segmentation_S0_manifest.json"
S0_REPORT_MD = "segmentation_S0_protocol.md"
S0_PROVENANCE_JSON = "segmentation_S0_protocol.provenance.json"

SCHEMA_VERSION = 1
PHASE = "8B"

FROZEN = "S0_PROTOCOL_FROZEN"
ADAPTER_FINGERPRINT_MISMATCH = "ADAPTER_FINGERPRINT_MISMATCH"
MEMORY_CONSTRAINT_REVIEW_REQUIRED = "MEMORY_CONSTRAINT_REVIEW_REQUIRED"
SEGMENTATION_RUNTIME_BLOCKED = "SEGMENTATION_RUNTIME_BLOCKED"
BLOCKED = "BLOCKED"

APPROVED = "APPROVED_FOR_CONTROLLED_TRAINING"
APPROXIMATION = "ACCEPTED_WITH_QUANTIFIED_APPROXIMATION"
SELECTION_BASIS = "PHASE_8A_QUANTITATIVE_FIDELITY_AUDIT_PLUS_HUMAN_REVIEW"
ARCHITECTURE_SELECTED = "FINAL_SELECTED_FOR_S0"
FALLBACK_STATUS = "NOT_SELECTED_FALLBACK"

SMOKE_STATUS_NON_EXPERIMENTAL = "NON_EXPERIMENTAL"
SMOKE_REPORTING_BAN = "DO_NOT_REPORT_AS_MODEL_RESULT"
SMOKE_RUN_NAME = "S0_smoke"
SEGMENTATION_RUN_ROOT = "artifacts/segmentation"
SMOKE_EPOCHS = 1

S0_EXECUTION_STATUS = "NOT_EXECUTED_PROTOCOL_ONLY"

HOLDOUT_STATUS = "PROTECTED_NOT_ACCESSED"
HOLDOUT_REASON = (
    "Phase 8B selected an architecture, approved already-audited label bytes, froze the S0 "
    "protocol and ran a one-epoch engineering smoke test on the development splits. The "
    "holdout was not read, materialised, adapted, converted, counted, inspected or predicted "
    "on; no holdout label, statistic or identifier exists in any artifact this phase wrote, "
    "and no holdout adapter directory was created. Nothing about choosing an architecture or "
    "proving that a training loop executes depends on which images are held out."
)

FRAMEWORK_EVIDENCE = "INSTALLED_PACKAGE_SOURCE_INSPECTION"
FRAMEWORK_CONFIG_EVIDENCE = "INSTALLED_EFFECTIVE_CONFIGURATION"

FITNESS_COMBINED = "COMBINED_MASK_AND_BOX_MAP50_95_UNWEIGHTED_SUM"

DIRECT_MASK_IOU_REQUIREMENT = "REQUIRED_FUTURE_PREDECLARED_EVALUATION"

PROJECT_OVERRIDE = "PROJECT_OVERRIDE"
FRAMEWORK_DEFAULT = "FRAMEWORK_DEFAULT"

FITNESS_WEIGHT_PATTERN = re.compile(r"w\s*=\s*\[([^\]]*)\]")


class FreezeError(RuntimeError):
    """Raised when a required input is missing or an invariant fails."""


class AdapterMismatchError(FreezeError):
    """Raised when the runtime label bytes are not the audited ones."""


class MemoryConstraintError(FreezeError):
    """Raised when the frozen batch size exhausts device memory."""


# --- small helpers ------------------------------------------------------------


def read_json(path: Path) -> dict[str, Any]:
    """Read a committed JSON artifact.

    Args:
        path: File to read.

    Returns:
        The parsed mapping.

    Raises:
        FreezeError: If the file is missing or is not a JSON object.
    """
    if not path.is_file():
        msg = f"required input not found: {path.name}"
        raise FreezeError(msg)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        msg = f"{path.name} is not valid JSON ({exc})"
        raise FreezeError(msg) from exc
    if not isinstance(data, dict):
        msg = f"{path.name} must contain a JSON object"
        raise FreezeError(msg)
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


def historical_digests(paths: ProjectPaths) -> dict[str, str]:
    """Digest every phase 8A artifact that must not change.

    Args:
        paths: Project layout.

    Returns:
        Digests keyed by file name.

    Raises:
        FreezeError: If one of them is absent.
    """
    files = {
        f"reports/{AUDIT_MANIFEST_JSON}": paths.reports / AUDIT_MANIFEST_JSON,
        f"reports/{AUDIT_REPORT_MD}": paths.reports / AUDIT_REPORT_MD,
        f"reports/{AUDIT_TABLE_CSV}": paths.reports / AUDIT_TABLE_CSV,
        f"reports/{AUDIT_PROVENANCE_JSON}": paths.reports / AUDIT_PROVENANCE_JSON,
        f"configs/{AUDIT_CONFIG_YAML}": paths.configs / AUDIT_CONFIG_YAML,
    }
    digests: dict[str, str] = {}
    for name, path in files.items():
        if not path.is_file():
            msg = f"phase 8A artifact missing: {name}"
            raise FreezeError(msg)
        digests[name] = sha256_file(path)
    return digests


def detector_digests(paths: ProjectPaths) -> dict[str, str]:
    """Digest the phase 7D artifacts that must not change.

    Args:
        paths: Project layout.

    Returns:
        Digests keyed by file name.

    Raises:
        FreezeError: If one of them is absent.
    """
    names = (
        FINAL_DETECTOR_JSON,
        "detection_selection_report.md",
        "detection_experiment_comparison.csv",
        "detection_experiment_results.json",
        "final_detector.provenance.json",
    )
    digests: dict[str, str] = {}
    for name in names:
        path = paths.reports / name
        if not path.is_file():
            msg = f"phase 7D artifact missing: reports/{name}"
            raise FreezeError(msg)
        digests[f"reports/{name}"] = sha256_file(path)
    return digests


def compare_digests(before: Mapping[str, str], after: Mapping[str, str], *, label: str) -> None:
    """Assert two digest sets are identical.

    Args:
        before: Digests recorded at entry.
        after: Digests recorded at exit.
        label: What the set describes, used in the error message.

    Raises:
        FreezeError: If any digest moved.
    """
    changed = sorted(name for name in before if before[name] != after.get(name))
    if changed:
        msg = f"{label} changed during this phase: {changed}. They are historical and immutable."
        raise FreezeError(msg)


# --- verification -------------------------------------------------------------


def verify_detector(paths: ProjectPaths) -> dict[str, Any]:
    """Verify the frozen detector is exactly as phase 7D left it.

    Reads the manifest, recomputes its semantic fingerprint from its own
    contents, and hashes the checkpoint on disk. Nothing is loaded into a model,
    nothing is validated and nothing is inferred with: the detection block is
    closed, and this is verification, not use.

    Args:
        paths: Project layout.

    Returns:
        What was verified.

    Raises:
        FreezeError: If the manifest is unreadable, the fingerprint does not
            recompute, or the checkpoint bytes differ from the recorded digest.
    """
    try:
        detector = load_final_detector(paths.reports)
    except FinalDetectorError as exc:
        raise FreezeError(str(exc)) from exc

    recomputed = detector.recompute_fingerprint()
    if recomputed != detector.fingerprint:
        msg = (
            f"the final detector manifest does not recompute: recorded {detector.fingerprint}, "
            f"recomputed {recomputed}"
        )
        raise FreezeError(msg)

    checkpoint_state = "PRESENT_VERIFIED"
    candidates = detector.candidate_paths(paths.root)
    present = [path for path in candidates if path.is_file()]
    if not present:
        checkpoint_state = "ABSENT_ON_THIS_MACHINE"
    else:
        observed = sha256_file(present[0])
        if observed != detector.checkpoint_sha256:
            msg = (
                f"the frozen detector checkpoint bytes changed: recorded "
                f"{detector.checkpoint_sha256}, observed {observed}"
            )
            raise FreezeError(msg)

    return {
        "selected_experiment": detector.selected_experiment,
        "model": detector.model,
        "imgsz": detector.imgsz,
        "checkpoint_sha256": detector.checkpoint_sha256,
        "checkpoint_size_bytes": detector.checkpoint_size_bytes,
        "checkpoint_state": checkpoint_state,
        "final_detector_sha256": detector.fingerprint,
        "fingerprint_recomputed": True,
        "retrained": False,
        "revalidated": False,
        "inference_run": False,
        "threshold_changed": False,
        "status": "UNCHANGED_DETECTION_BLOCK_CLOSED",
    }


def read_adapter_labels(root: Path, directories: Mapping[str, str]) -> dict[str, dict[str, str]]:
    """Read the adapter's label files off disk, keyed the way phase 8A keyed them.

    Args:
        root: The adapter root directory.
        directories: Canonical split name to on-disk directory name.

    Returns:
        Label text keyed by split then image stem.

    Raises:
        FreezeError: If a split directory is missing.
    """
    labels: dict[str, dict[str, str]] = {}
    for split, directory in directories.items():
        label_dir = root / "labels" / directory
        if not label_dir.is_dir():
            msg = f"adapter label directory not found: labels/{directory}"
            raise FreezeError(msg)
        labels[split] = {
            path.stem: path.read_text(encoding="utf-8") for path in sorted(label_dir.glob("*.txt"))
        }
    return labels


def verify_adapter(
    paths: ProjectPaths, config: SegmentationBaselineConfig, *, stage: str
) -> dict[str, Any]:
    """Verify the runtime label bytes are the ones phase 8A audited.

    Args:
        paths: Project layout.
        config: The S0 protocol.
        stage: Where in the phase this check runs, used in error messages.

    Returns:
        The verified fingerprints and cardinality.

    Raises:
        AdapterMismatchError: If any digest or count differs from the protocol.
        FreezeError: If the adapter is not on disk.
    """
    root = paths.root / Path(config.dataset_config).parent
    if not root.is_dir():
        msg = (
            f"the audited adapter is not on this machine: {config.dataset_config}. It is "
            "git-ignored and re-derivable by scripts/audit_segmentation_adapter.py, but "
            "regenerating it here would defeat the fingerprint check - obtain the audited "
            "bytes, or re-run phase 8A deliberately and confirm the digests still match."
        )
        raise FreezeError(msg)

    directories = {"train": "train", "validation": "val"}
    labels = read_adapter_labels(root, directories)

    observed = {
        "labels_train_sha256": label_fingerprint(labels["train"]),
        "labels_validation_sha256": label_fingerprint(labels["validation"]),
        "labels_development_sha256": label_fingerprint(
            {f"{split}/{stem}": text for split in labels for stem, text in labels[split].items()}
        ),
        "image_membership_sha256": digest(
            {split: sorted(labels[split]) for split in sorted(labels)}
        ),
    }
    mismatched = sorted(name for name, value in observed.items() if value != config.adapter[name])
    if mismatched:
        detail = "; ".join(
            f"{name}: expected {config.adapter[name]}, observed {observed[name]}"
            for name in mismatched
        )
        msg = (
            f"{ADAPTER_FINGERPRINT_MISMATCH} at {stage}: {detail}. The approved bytes are the "
            "ones phase 8A measured; a differing digest means a different conversion, and it is "
            "not silently rebuilt."
        )
        raise AdapterMismatchError(msg)

    counts = {
        "train_instances": sum(
            len([line for line in text.splitlines() if line.strip()])
            for text in labels["train"].values()
        ),
        "validation_instances": sum(
            len([line for line in text.splitlines() if line.strip()])
            for text in labels["validation"].values()
        ),
        "train_images": len(labels["train"]),
        "validation_images": len(labels["validation"]),
        "negative_images": sum(
            1 for split in labels for text in labels[split].values() if not text.strip()
        ),
    }
    counts["instances"] = counts["train_instances"] + counts["validation_instances"]
    wrong = sorted(name for name, value in counts.items() if value != config.adapter[name])
    if wrong:
        detail = "; ".join(
            f"{name}: expected {config.adapter[name]}, observed {counts[name]}" for name in wrong
        )
        msg = (
            f"{ADAPTER_FINGERPRINT_MISMATCH} at {stage}: adapter cardinality disagrees with the "
            f"protocol: {detail}"
        )
        raise AdapterMismatchError(msg)

    descriptor = read_dataset_descriptor(root / "dataset.yaml")
    return {"fingerprints": observed, "counts": counts, "descriptor_keys": descriptor}


def read_dataset_descriptor(path: Path) -> list[str]:
    """Read the adapter descriptor's keys and refuse one that names the holdout.

    Args:
        path: The ``dataset.yaml`` file.

    Returns:
        Its sorted top-level keys.

    Raises:
        FreezeError: If it is missing or carries a holdout entry.
    """
    import yaml

    if not path.is_file():
        msg = f"adapter descriptor not found: {path.name}"
        raise FreezeError(msg)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, Mapping):
        msg = f"{path.name} must contain a mapping"
        raise FreezeError(msg)
    if "test" in data:
        msg = (
            f"{path.name} declares a holdout split. A dataset descriptor is exactly the kind of "
            "place the protected split reaches a training loop by accident."
        )
        raise FreezeError(msg)
    return sorted(str(key) for key in data)


def verify_no_holdout_artifacts(paths: ProjectPaths, config: SegmentationBaselineConfig) -> None:
    """Refuse to continue if a holdout adapter or annotation view exists.

    Args:
        paths: Project layout.
        config: The S0 protocol.

    Raises:
        FreezeError: If any holdout artifact is present.
    """
    root = paths.root / Path(config.dataset_config).parent
    forbidden = [
        root / "images" / "test",
        root / "labels" / "test",
        paths.data_processed / "canonical" / "images" / "test",
    ]
    present = [path for path in forbidden if path.exists()]
    if present:
        names = ", ".join(path.relative_to(paths.root).as_posix() for path in present)
        msg = f"holdout artifacts exist and must not: {names}"
        raise FreezeError(msg)


def verify_canonical_references(
    paths: ProjectPaths, config: SegmentationBaselineConfig
) -> dict[str, Any]:
    """Verify the protocol's class-map and split digests against the frozen records.

    Args:
        paths: Project layout.
        config: The S0 protocol.

    Returns:
        What was verified.

    Raises:
        FreezeError: If a digest disagrees with the committed record.
    """
    audit = read_json(paths.reports / AUDIT_MANIFEST_JSON)
    recorded_class_map = str(audit["class_map_sha256"])
    recorded_split = str(audit["canonical_fingerprints"]["split_assignment_sha256"])
    if config.class_map_sha256 != recorded_class_map:
        msg = (
            f"class_map_sha256 {config.class_map_sha256} disagrees with the phase 8A record "
            f"{recorded_class_map}"
        )
        raise FreezeError(msg)
    if config.split_assignment_sha256 != recorded_split:
        msg = (
            f"split_assignment_sha256 {config.split_assignment_sha256} disagrees with the "
            f"phase 8A record {recorded_split}"
        )
        raise FreezeError(msg)

    split_manifest = read_json(paths.reports / SPLIT_MANIFEST_JSON)
    if str(split_manifest.get("split_assignment_sha256")) != config.split_assignment_sha256:
        msg = "split_assignment_sha256 disagrees with reports/split_manifest.json"
        raise FreezeError(msg)

    return {
        "class_map_sha256": config.class_map_sha256,
        "split_assignment_sha256": config.split_assignment_sha256,
        "class_map": audit["class_map"],
        "canonical_task_manifest_sha256": audit["canonical_task_manifest_sha256"],
    }


def verify_audit_evidence(
    paths: ProjectPaths, config: SegmentationBaselineConfig
) -> dict[str, Any]:
    """Read the phase 8A fidelity evidence the architecture decision rests on.

    Every figure is read from the committed manifest; none is recomputed, and
    none is re-aggregated across the strata the audit deliberately kept apart.

    Args:
        paths: Project layout.
        config: The S0 protocol.

    Returns:
        The evidence summary.

    Raises:
        FreezeError: If the audit manifest's digest disagrees with the protocol,
            or the audit did not preserve instance cardinality.
    """
    path = paths.reports / AUDIT_MANIFEST_JSON
    observed = sha256_file(path)
    if observed != config.adapter["audit_manifest_sha256"]:
        msg = (
            f"the phase 8A audit manifest digest changed: protocol declares "
            f"{config.adapter['audit_manifest_sha256']}, observed {observed}"
        )
        raise FreezeError(msg)

    audit = read_json(path)
    cardinality = audit["instance_cardinality"]
    preserved = bool(cardinality["preserved"])
    if not preserved or cardinality["model_instance_rows"] != config.adapter["instances"]:
        msg = "the phase 8A audit did not preserve instance cardinality at the declared count"
        raise FreezeError(msg)

    global_fidelity = audit["fidelity"]["global"]
    bands = global_fidelity["iou_bands"]
    instances = int(global_fidelity["instances"])
    at_least_090 = instances - int(bands["iou_lt_0.90"]["count"])
    at_least_095 = int(bands["iou_ge_0.99"]["count"]) + int(bands["iou_0.95_to_0.99"]["count"])

    return {
        "audit_manifest_sha256": observed,
        "audit_phase": audit["phase"],
        "instances": instances,
        "instance_cardinality_preserved": bool(cardinality["preserved"]),
        "instances_dropped": int(cardinality["instances_dropped"]),
        "instances_merged": int(cardinality["instances_merged"]),
        "instances_split": int(cardinality["instances_split"]),
        "parser_corrupt_labels": int(audit["parser_validation"]["corrupt_labels"]),
        "parser_test_split_present": bool(audit["parser_validation"]["test_split_present"]),
        "mask_iou_mean": global_fidelity["mask_iou"]["mean"],
        "mask_iou_median": global_fidelity["mask_iou"]["median"],
        "mask_iou_p05": global_fidelity["mask_iou"]["p05"],
        "mask_iou_min": global_fidelity["mask_iou"]["min"],
        "mask_iou_exact_instances": int(audit["fidelity"]["exact_instances"]),
        "control_iou_mean": global_fidelity["control_iou"]["mean"],
        "merged_iou_mean": global_fidelity["merged_iou"]["mean"],
        "instances_iou_ge_0.90": at_least_090,
        "percent_iou_ge_0.90": round(100.0 * at_least_090 / instances, 4),
        "instances_iou_ge_0.95": at_least_095,
        "percent_iou_ge_0.95": round(100.0 * at_least_095 / instances, 4),
        "instances_iou_lt_0.90": int(bands["iou_lt_0.90"]["count"]),
        "by_geometry_type_mask_iou_mean": {
            name: audit["fidelity"]["by_geometry_type"][name]["mask_iou"]["mean"]
            for name in sorted(audit["fidelity"]["by_geometry_type"])
        },
        "instances_with_multiple_components": int(
            audit["topology"]["instances_with_multiple_components"]
        ),
        "instances_with_holes": int(audit["topology"]["instances_with_holes"]),
        "total_hole_pixels": int(audit["topology"]["total_hole_pixels"]),
    }


# --- installed framework behaviour --------------------------------------------


def installed_segmentation_defaults() -> dict[str, Any]:
    """Read the installed framework's effective segmentation training defaults.

    Args:
        None.

    Returns:
        The effective configuration as a plain mapping.

    Raises:
        FreezeError: If the installed configuration cannot be resolved.
    """
    try:
        from ultralytics.cfg import get_cfg
    except ImportError as exc:  # pragma: no cover - the package is a hard dependency
        msg = f"{SEGMENTATION_RUNTIME_BLOCKED}: ultralytics is not importable ({exc})"
        raise FreezeError(msg) from exc
    resolved = get_cfg(overrides={"task": "segment", "mode": "train"})
    return {key: value for key, value in vars(resolved).items() if not key.startswith("_")}


def classify_declared_arguments(
    declared: Mapping[str, Any], defaults: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    """Compare each declared argument against the installed default.

    Args:
        declared: Values the protocol fixes.
        defaults: The installed effective configuration.

    Returns:
        Per-argument declared value, installed default and classification.
    """
    classified: dict[str, dict[str, Any]] = {}
    for name in sorted(declared):
        default = defaults.get(name, "NOT_EXPOSED")
        classified[name] = {
            "declared": declared[name],
            "installed_default": default,
            "classification": FRAMEWORK_DEFAULT if declared[name] == default else PROJECT_OVERRIDE,
        }
    return classified


def assert_matches_defaults(
    declared: Mapping[str, Any], defaults: Mapping[str, Any], *, block: str
) -> None:
    """Assert a declared block reproduces the installed defaults exactly.

    Args:
        declared: Values the protocol fixes.
        defaults: The installed effective configuration.
        block: Which configuration block is being checked.

    Raises:
        FreezeError: If a declared value differs from the installed default, or
            names an argument the installed version does not expose.
    """
    problems: list[str] = []
    for name in sorted(declared):
        if name not in defaults:
            problems.append(f"{name}: not exposed by the installed version")
        elif declared[name] != defaults[name]:
            problems.append(f"{name}: declared {declared[name]!r}, installed {defaults[name]!r}")
    if problems:
        msg = (
            f"{SEGMENTATION_RUNTIME_BLOCKED}: {block} does not match the installed effective "
            f"configuration: {'; '.join(problems)}. The protocol records the framework's own "
            "defaults; it does not invent them, and it is not adjusted to hide a version change."
        )
        raise FreezeError(msg)


def _fitness_source(cls: type) -> str:
    """Read one class's own ``fitness`` definition out of the installed package.

    The framework declares ``fitness`` as a property on some metric classes and
    as a plain method on others, so the attribute is unwrapped rather than
    assumed. ``cls.__dict__`` is used deliberately: an inherited definition
    would be a different class's evidence.

    Args:
        cls: The metric class.

    Returns:
        Its source text.

    Raises:
        FreezeError: If the class does not define ``fitness`` itself.
    """
    attribute = cls.__dict__.get("fitness")
    if attribute is None:
        msg = (
            f"{SEGMENTATION_RUNTIME_BLOCKED}: the installed {cls.__name__} does not define "
            "fitness; the checkpoint rule cannot be established from this version."
        )
        raise FreezeError(msg)
    function = getattr(attribute, "fget", attribute)
    return inspect.getsource(function).strip()


def fitness_behaviour() -> dict[str, Any]:
    """Establish what the installed segmentation trainer's ``best.pt`` optimises.

    Read from the installed source rather than from documentation, and checked
    structurally rather than asserted: if a future version changes the
    definition, the checks below fail instead of the report going stale.

    Args:
        None.

    Returns:
        The fitness definition, its evidence and its classification.

    Raises:
        FreezeError: If the installed definition is not the one this protocol
            was written against.
    """
    try:
        from ultralytics.utils.metrics import DetMetrics, Metric, SegmentMetrics
    except ImportError as exc:  # pragma: no cover - the package is a hard dependency
        msg = f"{SEGMENTATION_RUNTIME_BLOCKED}: cannot import the metric classes ({exc})"
        raise FreezeError(msg) from exc

    segment_source = _fitness_source(SegmentMetrics)
    metric_source = _fitness_source(Metric)
    detection_source = _fitness_source(DetMetrics)

    if "self.seg.fitness()" not in segment_source or "DetMetrics.fitness" not in segment_source:
        msg = (
            f"{SEGMENTATION_RUNTIME_BLOCKED}: the installed SegmentMetrics.fitness is not the "
            "mask-plus-box sum this protocol was written against. Do not train until the "
            "checkpoint rule has been re-established."
        )
        raise FreezeError(msg)

    match = FITNESS_WEIGHT_PATTERN.search(metric_source)
    if match is None:
        msg = f"{SEGMENTATION_RUNTIME_BLOCKED}: cannot read the fitness weight vector"
        raise FreezeError(msg)
    weights = [float(value) for value in match.group(1).split(",")]
    if weights != [0.0, 0.0, 0.0, 1.0]:
        msg = (
            f"{SEGMENTATION_RUNTIME_BLOCKED}: the installed fitness weights are {weights}, not "
            "[0.0, 0.0, 0.0, 1.0]. The checkpoint rule must be re-established before training."
        )
        raise FreezeError(msg)

    return {
        "classification": FITNESS_COMBINED,
        "driven_by": "BOTH_MASK_AND_BOX",
        "definition": (
            "SegmentMetrics.fitness returns self.seg.fitness() + DetMetrics.fitness, and each "
            "term is Metric.fitness, a weighted mean of [precision, recall, mAP@0.50, "
            "mAP@0.50:0.95] with weights [0.0, 0.0, 0.0, 1.0]. So the framework's validation "
            "fitness is the unweighted SUM of mask mAP@0.50:0.95 and box mAP@0.50:0.95, and "
            "best.pt is not selected on the mask metric alone."
        ),
        "component_weights": weights,
        "consequence": (
            "The reported checkpoint can differ from the epoch that maximised the mask metric "
            "on its own. This is recorded before training rather than discovered after it, and "
            "it is NOT replaced by a private checkpoint rule: substituting one would make S0 "
            "incomparable to any experiment that used the framework's own. If a later phase "
            "judges the misalignment material, that is a methodological review, not a quiet fix."
        ),
        "evidence": FRAMEWORK_EVIDENCE,
        "source": {
            "SegmentMetrics.fitness": segment_source,
            "DetMetrics.fitness": detection_source,
            "Metric.fitness": metric_source,
        },
    }


# --- smoke test ---------------------------------------------------------------


def run_smoke_test(
    paths: ProjectPaths, config: SegmentationBaselineConfig, weights: Mapping[str, Any]
) -> dict[str, Any]:
    """Run one minimal non-experimental epoch to prove the runtime works.

    Purpose only: the model loads, the segmentation labels parse, a CUDA forward
    and backward pass runs, the validation loader works, the mask loss executes,
    and a checkpoint reaches disk. No metric it produces is a result.

    Args:
        paths: Project layout.
        config: The S0 protocol.
        weights: The pretrained checkpoint record.

    Returns:
        What happened, deliberately without a single accuracy figure.

    Raises:
        MemoryConstraintError: If the frozen batch size exhausts device memory.
            The batch is never reduced to rescue the run.
        FreezeError: If the run fails for any other reason.
    """
    import torch
    from ultralytics import YOLO

    run_directory = paths.root / SEGMENTATION_RUN_ROOT / SMOKE_RUN_NAME
    if run_directory.exists():
        shutil.rmtree(run_directory)
    run_directory.mkdir(parents=True, exist_ok=True)

    arguments = config.training_arguments()
    arguments["epochs"] = SMOKE_EPOCHS
    # Deviations from the frozen protocol, both recorded: one epoch instead of
    # 100, and no plots. The plots would render dataset imagery and prediction
    # montages that phase 8B has no business producing, and the smoke test does
    # not need them to prove the loop executes.
    arguments["plots"] = False
    arguments["data"] = str((paths.root / config.dataset_config).resolve())
    arguments["project"] = str((paths.root / SEGMENTATION_RUN_ROOT).resolve())
    arguments["name"] = SMOKE_RUN_NAME
    arguments["exist_ok"] = True

    handler = capture_framework_log(run_directory / "framework.log")
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    failure: str | None = None
    try:
        model = YOLO(str(paths.root / weights["relative_path"]))
        model.train(**arguments)
    except torch.cuda.OutOfMemoryError as exc:
        elapsed = time.perf_counter() - started
        release_framework_log(handler)
        msg = (
            f"{MEMORY_CONSTRAINT_REVIEW_REQUIRED}: the frozen batch of {config.training['batch']} "
            f"exhausted device memory after {elapsed:.1f} s ({exc}). The batch is not reduced, "
            "auto-batch is not enabled, gradient accumulation is not introduced and imgsz is "
            "not lowered: the protocol stops for review instead."
        )
        raise MemoryConstraintError(msg) from exc
    except Exception as exc:  # any failure is a blocked runtime, and is reported as one
        failure = f"{type(exc).__name__}: {exc}"
    elapsed = time.perf_counter() - started
    peak_reserved = int(torch.cuda.max_memory_reserved())
    peak_allocated = int(torch.cuda.max_memory_allocated())
    release_framework_log(handler)

    log_text = ""
    log_path = run_directory / "framework.log"
    if log_path.is_file():
        log_text = log_path.read_text(encoding="utf-8", errors="replace")

    if failure is not None:
        msg = (
            f"{SEGMENTATION_RUNTIME_BLOCKED}: the one-epoch smoke test did not complete "
            f"({failure}). Nothing is frozen and no S0 run is authorised."
        )
        raise FreezeError(msg)

    best = run_directory / "weights" / "best.pt"
    last = run_directory / "weights" / "last.pt"
    optimizer = optimizer_evidence_from_log(log_text)

    record = {
        "status": "SUCCESS",
        "classification": SMOKE_STATUS_NON_EXPERIMENTAL,
        "reporting_ban": SMOKE_REPORTING_BAN,
        "purpose": (
            "Prove the model loads, the segmentation labels parse, CUDA forward and backward "
            "run, the validation loader works, the mask loss executes and a checkpoint reaches "
            "disk. Nothing else."
        ),
        "epochs": SMOKE_EPOCHS,
        "imgsz": config.training["imgsz"],
        "batch": config.training["batch"],
        "seed": config.seed,
        "deviations_from_s0": [
            f"epochs {SMOKE_EPOCHS} instead of {config.training['epochs']}",
            "plots disabled, so no dataset or prediction imagery is rendered",
        ],
        "runtime_seconds": round(elapsed, 3),
        "peak_gpu_memory_reserved_bytes": peak_reserved,
        "peak_gpu_memory_reserved_gib": round(peak_reserved / 1024**3, 3),
        "peak_gpu_memory_allocated_bytes": peak_allocated,
        "out_of_memory": False,
        "checkpoint_written": best.is_file(),
        "last_checkpoint_written": last.is_file(),
        "run_directory": (run_directory.relative_to(paths.root)).as_posix(),
        "run_directory_committed": False,
        "metrics_recorded": False,
        "metrics_recorded_reason": (
            "A one-epoch run's AP is not a model result. Recording it would create a number the "
            "project would then have to explain, and inviting a comparison against it is exactly "
            "the tuning this phase forbids."
        ),
        "optimizer_selected": optimizer,
    }
    if optimizer is None:
        record["optimizer_selected"] = {
            "optimizer": "NOT_CAPTURED",
            "source": "FRAMEWORK_LOG_LINE_ABSENT",
            "inferred": False,
        }
    return record


def recorded_smoke_test(paths: ProjectPaths, weights: Mapping[str, Any]) -> dict[str, Any]:
    """Reuse the smoke record the committed manifest already carries.

    For a protocol clarification that changes no runtime behaviour, re-running
    the smoke test would replace a recorded runtime and peak-memory figure with
    a different one for no reason, and would train a model this phase has no
    business training twice. So the record is read back instead - and checked,
    because a reused record is only trustworthy if it provably describes the
    same configuration.

    Args:
        paths: Project layout.
        weights: The pretrained checkpoint record, re-derived from the file on
            disk so a substituted binary cannot pass unnoticed.

    Returns:
        The recorded smoke-test block, unchanged.

    Raises:
        FreezeError: If no manifest exists to reuse, if it carries no successful
            smoke record, or if the weights on disk are not the ones it ran on.
    """
    path = paths.reports / S0_MANIFEST_JSON
    if not path.is_file():
        msg = (
            f"--rebuild-protocol needs an existing reports/{S0_MANIFEST_JSON} to reuse. There is "
            "none, so there is no smoke record to carry forward: run the freeze normally."
        )
        raise FreezeError(msg)
    manifest = read_json(path)
    smoke = manifest.get("smoke_test")
    if not isinstance(smoke, dict) or smoke.get("status") != "SUCCESS":
        msg = "the committed manifest carries no successful smoke record to reuse"
        raise FreezeError(msg)
    if smoke.get("metrics_recorded") is not False:
        msg = "the committed smoke record claims to carry metrics; it must not"
        raise FreezeError(msg)
    recorded_weights = manifest.get("pretrained_weights", {})
    if recorded_weights.get("sha256") != weights["sha256"]:
        msg = (
            "the pretrained checkpoint on disk is not the one the recorded smoke test ran on: "
            f"recorded {recorded_weights.get('sha256')}, observed {weights['sha256']}. Reusing "
            "the record would attribute one run's evidence to different bytes."
        )
        raise FreezeError(msg)
    return dict(smoke)


def clear_dataset_caches(paths: ProjectPaths, config: SegmentationBaselineConfig) -> list[str]:
    """Remove the label caches the framework writes into the audited directory.

    The caches are derived, git-ignored and harmless, but the audited adapter is
    evidence and is left exactly as phase 8A wrote it.

    Args:
        paths: Project layout.
        config: The S0 protocol.

    Returns:
        The repository-relative paths removed.
    """
    root = paths.root / Path(config.dataset_config).parent
    removed: list[str] = []
    for cache in sorted(root.rglob("*.cache")):
        removed.append(cache.relative_to(paths.root).as_posix())
        cache.unlink()
    return removed


# --- artifacts ----------------------------------------------------------------


def build_approval(
    *,
    config: SegmentationBaselineConfig,
    evidence: Mapping[str, Any],
    adapter: Mapping[str, Any],
    historical: Mapping[str, str],
) -> dict[str, Any]:
    """Assemble the adapter-approval manifest.

    Args:
        config: The S0 protocol.
        evidence: The phase 8A fidelity evidence.
        adapter: The verified runtime fingerprints and counts.
        historical: Digests of the phase 8A artifacts.

    Returns:
        The manifest.
    """
    return {
        "schema_version": SCHEMA_VERSION,
        "phase": PHASE,
        "status": APPROVED,
        "adapter_review_status": APPROVED,
        "adapter_role": ADAPTER_ROLE,
        "canonical_ground_truth": CANONICAL_GROUND_TRUTH,
        "derived_adapter": "YOLO_SEGMENTATION",
        "conversion_characterisation": APPROXIMATION,
        "architecture_selection_basis": SELECTION_BASIS,
        "reviewer": "project maintainer",
        "review_method": "HUMAN_REVIEW_OF_PHASE_8A_QUANTITATIVE_EVIDENCE",
        "what_the_approval_is": (
            "A decision that the measured approximation is acceptable for controlled "
            "segmentation training under this project's protocol. It approves specific label "
            "bytes, identified by digest."
        ),
        "what_the_approval_is_not": [
            "it does not make the YOLO segmentation labels canonical ground truth",
            "it does not describe the conversion as lossless: no instance round-trips exactly",
            "it does not turn phase 8A's measurements into an endorsement of YOLO segmentation "
            "over a mask-native architecture, which was never benchmarked here",
            "it does not authorise regenerating the labels by a different conversion",
        ],
        "phase_8a_audit_manifest_sha256": evidence["audit_manifest_sha256"],
        "phase_8a_artifact_digests": dict(historical),
        "adapter_fingerprints": {
            "labels_train_sha256": config.adapter["labels_train_sha256"],
            "labels_validation_sha256": config.adapter["labels_validation_sha256"],
            "labels_development_sha256": config.adapter["labels_development_sha256"],
            "image_membership_sha256": config.adapter["image_membership_sha256"],
        },
        "adapter_fingerprints_verified_on_disk": True,
        "adapter_bytes_regenerated": False,
        "conversion_algorithm_changed": False,
        "cardinality": dict(adapter["counts"]),
        "all_instances_retained": True,
        "fidelity_based_filtering": "NONE",
        "no_filtering_rationale": (
            "Excluding instances after observing adapter fidelity would change the canonical "
            "modelling population in response to a model-format limitation - choosing the data "
            "to suit the tool. All 1726 development instances remain, including the 47 whose "
            "round-trip IoU fell below 0.90, the tiny masks, the multi-component masks, the "
            "masks with holes and the 2 synthetic rectangles."
        ),
        "fidelity_summary": {
            "instances": evidence["instances"],
            "instance_cardinality_preserved": evidence["instance_cardinality_preserved"],
            "instances_dropped": evidence["instances_dropped"],
            "instances_merged": evidence["instances_merged"],
            "instances_split": evidence["instances_split"],
            "parser_corrupt_labels": evidence["parser_corrupt_labels"],
            "mask_iou_mean": evidence["mask_iou_mean"],
            "mask_iou_median": evidence["mask_iou_median"],
            "mask_iou_p05": evidence["mask_iou_p05"],
            "mask_iou_min": evidence["mask_iou_min"],
            "mask_iou_exact_instances": evidence["mask_iou_exact_instances"],
            "percent_iou_ge_0.90": evidence["percent_iou_ge_0.90"],
            "percent_iou_ge_0.95": evidence["percent_iou_ge_0.95"],
            "instances_iou_lt_0.90": evidence["instances_iou_lt_0.90"],
            "control_iou_mean": evidence["control_iou_mean"],
            "merged_iou_mean": evidence["merged_iou_mean"],
            "by_canonical_representation_mask_iou_mean": evidence["by_geometry_type_mask_iou_mean"],
        },
        "fidelity_reading_rules": [
            "the three levels are never collapsed: control_iou is rasterisation convention "
            "alone and exists before the YOLO format is involved, merged_iou adds component "
            "joining, mask_iou adds serialisation and the int32 snap",
            "the per-representation means are never aggregated into one figure",
            "mask size, not topology, dominates the distribution; the with-versus-without-holes "
            "comparison is confounded by size and is not evidence that filling holes is free",
        ],
        "known_approximation_categories": [
            {
                "category": "SERIALIZATION_AND_QUANTIZATION",
                "detail": (
                    "Coordinates are written at six decimal places and rasterised through an "
                    "int32 truncation. This is the largest contributor: 0.012458 mean IoU."
                ),
            },
            {
                "category": "COMPONENT_JOIN_APPROXIMATION",
                "detail": (
                    "A disconnected mask cannot be expressed: one row is one flat ring, so the "
                    f"{evidence['instances_with_multiple_components']} multi-component instances "
                    "are bridged with zero-width connectors. Costs 0.000843 mean IoU."
                ),
            },
            {
                "category": "HOLE_FILL_APPROXIMATION",
                "detail": (
                    "An interior ring cannot be expressed, so holes are filled: "
                    f"{evidence['instances_with_holes']} instances carry holes totalling "
                    f"{evidence['total_hole_pixels']} filled pixels."
                ),
            },
            {
                "category": "RASTER_CONVENTION_DIFFERENCE",
                "detail": (
                    "pycocotools and OpenCV do not rasterise identical geometry into identical "
                    "pixels, which is why no instance round-trips exactly and why this level is "
                    "never reported as YOLO format loss."
                ),
            },
            {
                "category": "SMALL_MASK_SENSITIVITY",
                "detail": (
                    "All 20 worst instances are 4-59 px masks, where a single boundary pixel is "
                    "a large share of the area. This is the primary fidelity-risk stratum."
                ),
            },
        ],
        "architecture_decision": {
            "status": ARCHITECTURE_SELECTED,
            "selected_architecture": "YOLO11n-seg",
            "weight_identifier": config.weight_identifier,
            "fallback_architecture": config.fallback_architecture,
            "fallback_status": FALLBACK_STATUS,
            "not_a_claim_of_superiority": (
                "No mask-native architecture was installed, trained or benchmarked in this "
                "project. YOLO11n-seg is not asserted to be better; it is the architecture "
                "chosen under this evidence and these constraints."
            ),
        },
        "test": {"status": HOLDOUT_STATUS, "reason": HOLDOUT_REASON},
    }


def build_manifest(
    *,
    config: SegmentationBaselineConfig,
    config_sha256: str,
    evidence: Mapping[str, Any],
    adapter: Mapping[str, Any],
    canonical: Mapping[str, Any],
    detector: Mapping[str, Any],
    runtime: Mapping[str, Any],
    weights: Mapping[str, Any],
    defaults: Mapping[str, Any],
    fitness: Mapping[str, Any],
    smoke: Mapping[str, Any],
    historical: Mapping[str, str],
    detector_artifacts: Mapping[str, str],
) -> dict[str, Any]:
    """Assemble the S0 protocol manifest.

    Args:
        config: The S0 protocol.
        config_sha256: Digest of the configuration file's bytes.
        evidence: The phase 8A fidelity evidence.
        adapter: The verified runtime fingerprints and counts.
        canonical: The verified class-map and split references.
        detector: The frozen detector verification.
        runtime: Software and hardware facts.
        weights: The pretrained checkpoint record.
        defaults: The installed effective segmentation configuration.
        fitness: The checkpoint rule's established behaviour.
        smoke: The smoke-test record.
        historical: Digests of the phase 8A artifacts.
        detector_artifacts: Digests of the phase 7D artifacts.

    Returns:
        The manifest.
    """
    return {
        "schema_version": SCHEMA_VERSION,
        "phase": PHASE,
        "classification": FROZEN,
        "experiment_id": config.experiment_id,
        "task": config.task,
        "experiment_question": config.description.strip(),
        "experiment_role": "BASELINE_NOT_THE_FINAL_SEGMENTER",
        "s0_execution_status": S0_EXECUTION_STATUS,
        "s0_result_metrics": None,
        "s0_result_metrics_reason": (
            "Phase 8B freezes the protocol and proves the runtime executes. S0 has not been "
            "trained, so there is no performance number to record and none is invented."
        ),
        "protocol_config": f"configs/{CONFIG_YAML}",
        "protocol_config_sha256": config_sha256,
        "protocol_fingerprint": config.fingerprint(),
        "architecture": {
            "status": ARCHITECTURE_SELECTED,
            "selected": config.architecture,
            "model": config.model,
            "family_alignment": (
                "YOLO11, the same family as the frozen detector, so a later operational "
                "comparison is between comparable things rather than between two frameworks."
            ),
            "resolution_alignment": (
                f"imgsz {config.training['imgsz']}, the frozen detector's input resolution, so a "
                "later detector-versus-segmenter comparison is not also a resolution comparison."
            ),
            "fallback_architecture": config.fallback_architecture,
            "fallback_status": FALLBACK_STATUS,
            "selection_basis": SELECTION_BASIS,
        },
        "adapter": {
            "role": ADAPTER_ROLE,
            "canonical_ground_truth": CANONICAL_GROUND_TRUTH,
            "review_status": APPROVED,
            "conversion_characterisation": APPROXIMATION,
            "dataset_config": config.dataset_config,
            "fingerprints": dict(adapter["fingerprints"]),
            "counts": dict(adapter["counts"]),
            "descriptor_keys": list(adapter["descriptor_keys"]),
            "all_instances_retained": True,
            "fidelity_based_filtering": "NONE",
            "regenerated": False,
        },
        "phase_8a_evidence": dict(evidence),
        "phase_8a_artifact_digests": dict(historical),
        "phase_8a_artifacts_unchanged": True,
        "canonical_references": dict(canonical),
        "frozen_detector": dict(detector),
        "detection_artifact_digests": dict(detector_artifacts),
        "detection_artifacts_unchanged": True,
        "runtime": dict(runtime),
        "pretrained_weights": dict(weights),
        "training": dict(config.training),
        "segmentation_arguments": dict(config.segmentation_arguments),
        "augmentation_policy": config.augmentation_policy,
        "augmentation_arguments": dict(config.augmentation_arguments),
        "installed_defaults_evidence": FRAMEWORK_CONFIG_EVIDENCE,
        "argument_classification": {
            "training": classify_declared_arguments(
                {key: value for key, value in config.training.items() if key != "augmentation"},
                defaults,
            ),
            "seed": {
                "declared": config.seed,
                "installed_default": defaults.get("seed"),
                "classification": (
                    FRAMEWORK_DEFAULT if config.seed == defaults.get("seed") else PROJECT_OVERRIDE
                ),
            },
            "segmentation_arguments": classify_declared_arguments(
                config.segmentation_arguments, defaults
            ),
            "augmentation_arguments": classify_declared_arguments(
                config.augmentation_arguments, defaults
            ),
        },
        "batch_decision": config.batch_decision.strip(),
        "checkpoint_selection": config.checkpoint_selection.strip(),
        "checkpoint_selection_policy": {
            "checkpoint_selection_policy": config.checkpoint_selection_policy,
            "checkpoint_selection_review_status": config.checkpoint_selection_review_status,
            "checkpoint_selection_semantics": config.checkpoint_selection_semantics,
            "checkpoint_selection_box_component_weight": (
                config.checkpoint_selection_box_component_weight
            ),
            "checkpoint_selection_mask_component_weight": (
                config.checkpoint_selection_mask_component_weight
            ),
            "primary_scientific_reporting_metric": config.primary_scientific_reporting_metric,
            "selection_metric_equals_primary_reporting_metric": (
                config.selection_metric_equals_primary_reporting_metric
            ),
            "divergence_is_intentional": True,
            "custom_mask_only_selector_authorized": False,
            "retrospective_reinterpretation_allowed": False,
            "framework_fitness_is_a_reporting_metric": False,
            "rationale": list(config.checkpoint_selection_rationale),
            "future_comparison_constraint": (config.checkpoint_selection_future_constraint.strip()),
            "superiority_claim": (
                "NONE. Keeping the native composite is an explicitly accepted baseline protocol "
                "choice, not a finding that it beats mask-only selection. No experiment in this "
                "project compares the two, and none could support such a claim."
            ),
            "established_from": FRAMEWORK_EVIDENCE,
        },
        "checkpoint_fitness_behaviour": dict(fitness),
        "metrics": config.metrics.as_dict(),
        "rare_class": {
            "name": config.rare_class,
            "support_rule": SUPPORT_RULE,
            "support_rule_origin": "FROZEN_IN_PHASE_7A_REUSED_UNCHANGED",
            "classification": RARE_CLASS_CLASSIFICATION,
            "validation_images": 1,
            "validation_instances": 8,
            "reported_in_full": True,
            "may_decide_a_winner": False,
            "limitation": config.rare_class_limitation.strip(),
        },
        "supported_macro_mask_policy": {
            "metric": config.metrics.supported_macro,
            "definition": (
                "The unweighted mean of per-class mask AP@0.50:0.95 over the classes the frozen "
                "support rule admits. The rule names no class; the admitted set is its output."
            ),
            "s0_reports_it": True,
            "is_a_selection_metric": False,
            "reason": (
                "S0 is a baseline and there is nothing to select between. It becomes a "
                "selection metric only when a segmentation comparison protocol is frozen, "
                "before the experiments it would decide exist."
            ),
        },
        "direct_mask_iou_requirement": {
            "status": DIRECT_MASK_IOU_REQUIREMENT,
            "why": (
                "The academic deliverable requires IoU. Ultralytics' mask AP is an averaged "
                "detection-style metric over IoU thresholds and is not a direct instance-mask "
                "IoU diagnostic, so reporting AP alone would not satisfy it."
            ),
            "what_is_owed": (
                "After S0's checkpoint is selected, the project must report an explicit "
                "instance-mask IoU diagnostic under a protocol that predeclares the inference "
                "settings, the prediction-to-ground-truth matching rule, how unmatched "
                "predictions and unmatched ground truth are counted, and the averaging scheme."
            ),
            "defined_here": False,
            "executed_here": False,
            "must_not": (
                "That protocol must not be written after looking at S0's predictions. Choosing "
                "a matching rule once the failures are visible is how a diagnostic becomes a "
                "flattering one."
            ),
            "phase": "a later phase, alongside the phase 10 evaluation protocol",
        },
        "smoke_test": dict(smoke),
        "test": {"status": HOLDOUT_STATUS, "reason": HOLDOUT_REASON},
        "models_trained_in_this_phase": 0,
        "smoke_runs_in_this_phase": 1,
        "detector_touched": False,
        "holdout_accessed": False,
    }


def build_report(manifest: Mapping[str, Any], *, commit: str | None) -> str:
    """Render the S0 protocol report.

    Args:
        manifest: The assembled manifest.
        commit: Repository commit, when available.

    Returns:
        The report text.
    """
    evidence = manifest["phase_8a_evidence"]
    adapter = manifest["adapter"]
    smoke = manifest["smoke_test"]
    runtime = manifest["runtime"]
    weights = manifest["pretrained_weights"]
    training = manifest["training"]
    fitness = manifest["checkpoint_fitness_behaviour"]
    metrics = manifest["metrics"]
    rare = manifest["rare_class"]
    counts = adapter["counts"]

    lines: list[str] = []
    add = lines.append

    add("# S0 - segmentation baseline protocol")
    add("")
    add(
        f"Phase {manifest['phase']} · classification `{manifest['classification']}` · "
        f"experiment `{manifest['experiment_id']}` · "
        f"S0 execution `{manifest['s0_execution_status']}`"
    )
    add("")
    add(
        "This report freezes a protocol. It contains **no S0 performance result**, because S0 "
        "has not been trained. The only model that ran in this phase was a one-epoch "
        f"`{smoke['classification']}` smoke test whose metrics are "
        f"`{smoke['reporting_ban']}` and were not recorded."
    )
    add("")
    if commit:
        add(f"Repository commit at production time: `{commit}`.")
        add("")

    add("## 1. Architecture decision")
    add("")
    add(
        f"`HUMAN_ARCHITECTURE_DECISION` · status "
        f"`{manifest['architecture']['status']}` · selected "
        f"**{manifest['architecture']['selected']}** "
        f"(`{weights['identifier']}`)."
    )
    add("")
    add(
        "The decision is the maintainer's, taken on the phase 8A evidence. Phase 8A itself "
        "selected nothing and could not: a high IoU distribution is not an approval and a low "
        "one is not a rejection."
    )
    add("")
    add("| Field | Value |")
    add("| --- | --- |")
    add(f"| Selected architecture | {manifest['architecture']['selected']} |")
    add(f"| Weight identifier | `{weights['identifier']}` |")
    add(f"| Selection basis | `{manifest['architecture']['selection_basis']}` |")
    add(f"| Fallback | {manifest['architecture']['fallback_architecture']} |")
    add(f"| Fallback status | `{manifest['architecture']['fallback_status']}` |")
    add("")

    add("## 2. Phase 8A evidence")
    add("")
    add("`AUDIT_EVIDENCE`. Every figure below is read from the committed phase 8A manifest.")
    add("")
    add("| Measurement | Value |")
    add("| --- | --- |")
    add(f"| Development instances | {evidence['instances']} |")
    add(
        f"| Instance cardinality preserved | {evidence['instance_cardinality_preserved']} "
        f"({evidence['instances_dropped']} dropped, {evidence['instances_merged']} merged, "
        f"{evidence['instances_split']} split) |"
    )
    add(f"| Framework parser corrupt labels | {evidence['parser_corrupt_labels']} |")
    add(f"| Mask IoU mean | {evidence['mask_iou_mean']:.6f} |")
    add(f"| Mask IoU median | {evidence['mask_iou_median']:.6f} |")
    add(f"| Mask IoU P05 | {evidence['mask_iou_p05']:.6f} |")
    add(f"| Mask IoU minimum | {evidence['mask_iou_min']:.6f} |")
    add(
        f"| Instances at IoU >= 0.90 | {evidence['instances_iou_ge_0.90']} "
        f"({evidence['percent_iou_ge_0.90']}%) |"
    )
    add(
        f"| Instances at IoU >= 0.95 | {evidence['instances_iou_ge_0.95']} "
        f"({evidence['percent_iou_ge_0.95']}%) |"
    )
    add(f"| Instances below IoU 0.90 | {evidence['instances_iou_lt_0.90']} |")
    add(f"| Exactly-converted instances | {evidence['mask_iou_exact_instances']} |")
    add(
        f"| Control-level IoU mean (rasteriser convention alone) | "
        f"{evidence['control_iou_mean']:.6f} |"
    )
    add(f"| Merged-level IoU mean (adds component joining) | {evidence['merged_iou_mean']:.6f} |")
    add("")
    add("Mask IoU by canonical representation, never aggregated into one figure:")
    add("")
    add("| Representation | Mean mask IoU |")
    add("| --- | --- |")
    for name, value in evidence["by_geometry_type_mask_iou_mean"].items():
        add(f"| `{name}` | {value:.6f} |")
    add("")
    add(
        f"Topology: {evidence['instances_with_multiple_components']} instances have more than "
        f"one component and {evidence['instances_with_holes']} carry holes totalling "
        f"{evidence['total_hole_pixels']} filled pixels. Neither is the dominant cost - "
        "component joining costs 0.000843 mean IoU while serialisation and the integer snap "
        "cost 0.012458."
    )
    add("")

    add("## 3. Why YOLO11n-seg was accepted")
    add("")
    add("`HUMAN_ARCHITECTURE_DECISION`. Four reasons, stated so a reader can disagree with them:")
    add("")
    add(
        "1. **The audited representation works and its cost is quantified.** All "
        f"{evidence['instances']} development instances convert to exactly "
        f"{evidence['instances']} rows, the framework's own parser reads them with "
        f"{evidence['parser_corrupt_labels']} corrupt labels, and the approximation is measured "
        "rather than assumed."
    )
    add("2. **Family alignment.** " + manifest["architecture"]["family_alignment"])
    add("3. **Resolution alignment.** " + manifest["architecture"]["resolution_alignment"])
    add(
        "4. **The mask-native alternative stays available.** It is recorded as "
        f"`{manifest['architecture']['fallback_status']}`, not discarded."
    )
    add("")
    add(
        f"The conversion is characterised as `{adapter['conversion_characterisation']}`, never "
        "as lossless: **no instance round-trips exactly**, and the reason is partly the "
        "rasteriser convention rather than the format."
    )
    add("")
    add(
        "No mask-native architecture was installed, trained or benchmarked in this project, so "
        "nothing here says YOLO11n-seg is better than one. It is the architecture chosen under "
        "this evidence and these constraints."
    )
    add("")

    add("## 4. Why canonical COCO remains authoritative")
    add("")
    add(
        f"`MODEL_ADAPTER_LIMITATION`. Canonical ground truth is "
        f"`{adapter['canonical_ground_truth']}`; the YOLO labels are a "
        f"`{adapter['role']}`. Approving them for training does not promote them. If a YOLO "
        "label and the canonical COCO document ever disagree, the COCO document is right and "
        "the adapter is broken: regenerate it from the canonical source, never hand-edit it, "
        "and never re-derive the canonical document from a label."
    )
    add("")

    add("## 5. Known adapter limitations")
    add("")
    add("`MODEL_ADAPTER_LIMITATION`.")
    add("")
    for entry in manifest.get("known_approximation_categories", []):
        add(f"- **`{entry['category']}`** - {entry['detail']}")
    add("")

    add("## 6. Why no instance filtering occurred")
    add("")
    add(
        f"All {counts['instances']} development instances remain. Excluding the "
        f"{evidence['instances_iou_lt_0.90']} instances whose round-trip IoU fell below 0.90, or "
        "the tiny, multi-component, holed or synthetic-rectangle masks, would change the "
        "canonical modelling population in response to a model-format limitation - choosing the "
        "data to suit the tool, and quietly making every later metric describe an easier "
        "dataset than the project claims to have."
    )
    add("")

    add("## 7. S0 experimental question")
    add("")
    add(f"> {manifest['experiment_question']}")
    add("")
    add(
        f"S0 is a **baseline** (`{manifest['experiment_role']}`). It is not the project's final "
        "segmenter, and no comparison protocol exists for it yet."
    )
    add("")

    add("## 8. Frozen data")
    add("")
    add("`PREDECLARED_PROTOCOL`. The exact bytes phase 8A audited, verified on disk.")
    add("")
    add("| Field | Value |")
    add("| --- | --- |")
    add(f"| Dataset descriptor | `{adapter['dataset_config']}` |")
    add(f"| Descriptor keys | {', '.join(f'`{key}`' for key in adapter['descriptor_keys'])} |")
    add(f"| Train images / instances | {counts['train_images']} / {counts['train_instances']} |")
    add(
        f"| Validation images / instances | {counts['validation_images']} / "
        f"{counts['validation_instances']} |"
    )
    add(f"| Total instances | {counts['instances']} |")
    add(f"| Negative images | {counts['negative_images']} |")
    add(f"| Train label digest | `{adapter['fingerprints']['labels_train_sha256']}` |")
    add(f"| Validation label digest | `{adapter['fingerprints']['labels_validation_sha256']}` |")
    add(f"| Development label digest | `{adapter['fingerprints']['labels_development_sha256']}` |")
    add(f"| Image membership digest | `{adapter['fingerprints']['image_membership_sha256']}` |")
    add(f"| Class map digest | `{manifest['canonical_references']['class_map_sha256']}` |")
    add(f"| Split digest | `{manifest['canonical_references']['split_assignment_sha256']}` |")
    add("")
    add(
        "The labels were **not regenerated**. A differing digest stops the phase as "
        f"`{ADAPTER_FINGERPRINT_MISMATCH}` rather than triggering a rebuild, because a rebuild "
        "would silently replace measured bytes with unmeasured ones."
    )
    add("")

    add("## 9. Frozen training configuration")
    add("")
    add("`PREDECLARED_PROTOCOL`. Complete, not just the fields that were changed.")
    add("")
    add("| Argument | Value | Against installed default |")
    add("| --- | --- | --- |")
    classification = manifest["argument_classification"]
    add(
        f"| `seed` | {manifest['argument_classification']['seed']['declared']} | "
        f"`{manifest['argument_classification']['seed']['classification']}` |"
    )
    for name in sorted(classification["training"]):
        entry = classification["training"][name]
        add(f"| `{name}` | {entry['declared']} | `{entry['classification']}` |")
    add(f"| `augmentation` | {training['augmentation']} | policy |")
    add("")
    add(f"**Batch.** {manifest['batch_decision']}")
    add("")
    add(
        f"Device requirement is `CUDA_GPU_REQUIRED`. Runtime verified: python "
        f"{runtime['python']}, torch {runtime['torch']}, torchvision {runtime['torchvision']}, "
        f"ultralytics {runtime['ultralytics']}, CUDA {runtime['cuda_runtime']}, "
        f"{runtime['gpu_name']} ({runtime['gpu_arch']}, "
        f"{runtime['gpu_total_memory_bytes'] / 1024**3:.2f} GiB)."
    )
    add("")
    add(
        f"Pretrained weights `{weights['identifier']}`, obtained by "
        f"`{weights['source_mechanism']}` under ultralytics {weights['ultralytics_version']}, "
        f"SHA-256 `{weights['sha256']}`, {weights['size_bytes']} bytes. Git-ignored: the "
        "repository commits the record, not the binary."
    )
    add("")

    add("## 10. Segmentation-specific framework arguments")
    add("")
    add(
        f"`PREDECLARED_PROTOCOL`, evidence `{manifest['installed_defaults_evidence']}`. Read "
        "from the installed configuration, not from online documentation, and verified equal to "
        "it at freeze time."
    )
    add("")
    add("| Argument | Value | Why it is recorded |")
    add("| --- | --- | --- |")
    reasons = {
        "overlap_mask": "overlapping instance masks are rasterised into one indexed map, so an "
        "occluded instance's target is what remains visible",
        "mask_ratio": "the mask target is downsampled by this factor before the loss sees it",
        "retina_masks": "affects mask resolution at inference, not training targets",
        "dropout": "regularisation, exposed by the framework",
        "max_det": "caps detections per image during validation",
        "single_cls": "would collapse the class map if set",
        "rect": "would change letterboxing if set",
        "multi_scale": "would vary input resolution during training if set",
    }
    for name in sorted(manifest["segmentation_arguments"]):
        add(
            f"| `{name}` | {manifest['segmentation_arguments'][name]} | "
            f"{reasons.get(name, 'recorded so a reader can tell it was left alone')} |"
        )
    add("")
    add(
        "**Augmentation policy** `" + manifest["augmentation_policy"] + "`. The installed "
        "version's own defaults, enumerated in full so the report does not depend on a library "
        "version's documentation:"
    )
    add("")
    add("| Argument | Value |")
    add("| --- | --- |")
    for name in sorted(manifest["augmentation_arguments"]):
        add(f"| `{name}` | {manifest['augmentation_arguments'][name]} |")
    add("")
    add(
        "These are **model training transformations, not canonical preprocessing**. The "
        "canonical images and masks are unchanged, and augmentation is not applied to "
        "validation. Nothing here was tuned, and tuning it before S0 exists would make S0 a "
        "tuned result rather than a baseline."
    )
    add("")

    add("## 11. Metric hierarchy")
    add("")
    add(
        "`PREDECLARED_PROTOCOL`. Fixed before training, so the headline number cannot be chosen "
        "once results are visible."
    )
    add("")
    add(f"- **Primary:** `{metrics['primary']}`")
    add(f"- **Secondary mask:** {', '.join(f'`{name}`' for name in metrics['secondary_mask'])}")
    add(
        "- **Box metrics from the segmenter:** "
        + ", ".join(f"`{name}`" for name in metrics["box_from_segmenter"])
    )
    add(f"- **Per class (mask):** {', '.join(f'`{name}`' for name in metrics['per_class_mask'])}")
    add(f"- **Per class (box):** {', '.join(f'`{name}`' for name in metrics['per_class_box'])}")
    add(f"- **Macro over supported classes:** `{metrics['supported_macro']}`")
    add("")
    add(
        "Mask and box families are reported side by side and never merged. A composite "
        "box-plus-mask score is refused by the configuration parser, not merely discouraged: "
        "inventing one after the fact is how a weak mask result hides behind a strong box one."
    )
    add("")
    add(
        f"`{metrics['supported_macro']}` is the unweighted mean of per-class mask AP@0.50:0.95 "
        "over the classes the frozen phase 7 support rule admits. **S0 reports it; it is not a "
        "winner-selection metric**, because there is nothing yet to select between and the "
        "comparison protocol that would use it has not been frozen."
    )
    add("")

    add("## 12. The `vest_loose` limitation")
    add("")
    add(
        f"`PREDECLARED_PROTOCOL`. The support rule is `{rare['support_rule_origin']}`: "
        f"{rare['support_rule']}. Applied to the frozen split, `{rare['name']}` holds "
        f"{rare['validation_images']} validation image and {rare['validation_instances']} "
        f"instances, so it is classified `{rare['classification']}`."
    )
    add("")
    add(rare["limitation"])
    add("")

    add("## 13. Checkpoint-selection behaviour")
    add("")
    policy = manifest["checkpoint_selection_policy"]
    add(
        f"`HUMAN_ARCHITECTURE_DECISION` and `PREDECLARED_PROTOCOL`, evidence "
        f"`{fitness['evidence']}`. The rule is `ULTRALYTICS_BEST_ON_VALIDATION_FITNESS`. What "
        "that means for a segmentation model was established from the installed source "
        "**before** any full S0 result existed, returned for methodological review, and "
        "accepted."
    )
    add("")
    add("### 13.1 What the framework actually optimises")
    add("")
    add(f"- **Classification:** `{fitness['classification']}` (driven by `{fitness['driven_by']}`)")
    add(f"- **Definition:** {fitness['definition']}")
    add(f"- **Consequence:** {fitness['consequence']}")
    add("")
    add("### 13.2 The reviewed decision")
    add("")
    add(
        "**Accepted: the native Ultralytics composite fitness selects S0's checkpoint.** No "
        "custom mask-only checkpoint selector was written, and none is authorised."
    )
    add("")
    add("| Field | Value |")
    add("| --- | --- |")
    add(f"| `checkpoint_selection_policy` | `{policy['checkpoint_selection_policy']}` |")
    add(
        f"| `checkpoint_selection_review_status` | "
        f"`{policy['checkpoint_selection_review_status']}` |"
    )
    add(f"| `checkpoint_selection_semantics` | `{policy['checkpoint_selection_semantics']}` |")
    add(
        f"| `checkpoint_selection_box_component_weight` | "
        f"{policy['checkpoint_selection_box_component_weight']} |"
    )
    add(
        f"| `checkpoint_selection_mask_component_weight` | "
        f"{policy['checkpoint_selection_mask_component_weight']} |"
    )
    add(
        f"| `primary_scientific_reporting_metric` | "
        f"`{policy['primary_scientific_reporting_metric']}` |"
    )
    add(
        f"| `selection_metric_equals_primary_reporting_metric` | "
        f"**{policy['selection_metric_equals_primary_reporting_metric']}** - intentional |"
    )
    add("")
    add(
        "That last row is the whole point of this section. **The checkpoint is selected on a "
        "box-plus-mask composite while the project reports mask mAP@0.50:0.95**, so the epoch "
        "S0 reports need not be the epoch that maximised the reported metric. That divergence "
        "is recorded as a known property of the protocol rather than left for a reader to "
        "discover."
    )
    add("")
    add("### 13.3 Why the native fitness was kept")
    add("")
    for entry in policy["rationale"]:
        add(f"- {entry}")
    add("")
    add(f"**Superiority claim:** {policy['superiority_claim']}")
    add("")
    add(
        "The framework fitness is a **checkpoint-selection mechanism, not a scientific headline "
        "metric**: `framework_fitness_is_a_reporting_metric: "
        f"{policy['framework_fitness_is_a_reporting_metric']}`. No combined box-plus-mask number "
        "is introduced for reporting anywhere in this project, and the metric hierarchy in "
        "section 11 is unchanged by this decision."
    )
    add("")
    add("### 13.4 Protocol invariant for future comparisons")
    add("")
    add(f"`PREDECLARED_PROTOCOL`. {policy['future_comparison_constraint']}")
    add("")
    add(
        "Two things follow and are enforced by the configuration parser: no post-hoc mask-only "
        "checkpoint selection is authorised "
        f"(`custom_mask_only_selector_authorized: "
        f"{policy['custom_mask_only_selector_authorized']}`), and S0's checkpoint semantics are "
        "never reinterpreted after the fact "
        f"(`retrospective_reinterpretation_allowed: "
        f"{policy['retrospective_reinterpretation_allowed']}`)."
    )
    add("")

    add("## 14. Future direct mask-IoU requirement")
    add("")
    requirement = manifest["direct_mask_iou_requirement"]
    add(f"`FUTURE_EVALUATION_REQUIREMENT` · status `{requirement['status']}`.")
    add("")
    add(requirement["why"])
    add("")
    add(f"**What is owed.** {requirement['what_is_owed']}")
    add("")
    add(f"**What must not happen.** {requirement['must_not']}")
    add("")
    add(
        f"Phase 8B defines the requirement only: `defined_here: {requirement['defined_here']}`, "
        f"`executed_here: {requirement['executed_here']}`. It belongs to "
        f"{requirement['phase']}."
    )
    add("")

    add("## 15. Smoke-test result")
    add("")
    add(f"`{smoke['classification']}` · `{smoke['reporting_ban']}` · status `{smoke['status']}`.")
    add("")
    add(smoke["purpose"])
    add("")
    add("| Field | Value |")
    add("| --- | --- |")
    add(f"| Epochs | {smoke['epochs']} |")
    add(f"| imgsz | {smoke['imgsz']} |")
    add(f"| Batch | {smoke['batch']} |")
    add(f"| Seed | {smoke['seed']} |")
    add(f"| Runtime | {smoke['runtime_seconds']} s |")
    add(
        f"| Peak GPU memory reserved | {smoke['peak_gpu_memory_reserved_gib']} GiB "
        f"({smoke['peak_gpu_memory_reserved_bytes']} bytes) |"
    )
    add(f"| Out of memory | {smoke['out_of_memory']} |")
    add(f"| Optimizer selected | {smoke['optimizer_selected'].get('optimizer')} |")
    add(f"| Optimizer evidence | `{smoke['optimizer_selected'].get('source')}` |")
    add(f"| `best.pt` written | {smoke['checkpoint_written']} |")
    add(f"| `last.pt` written | {smoke['last_checkpoint_written']} |")
    add(f"| Run directory | `{smoke['run_directory']}` (git-ignored) |")
    add(f"| Accuracy metrics recorded | {smoke['metrics_recorded']} |")
    add("")
    add("Deviations from the frozen S0 protocol, both deliberate:")
    for deviation in smoke["deviations_from_s0"]:
        add(f"- {deviation}")
    add("")
    add(smoke["metrics_recorded_reason"])
    add("")

    add("## 16. Holdout policy")
    add("")
    add(f"`HOLDOUT_POLICY` · status `{manifest['test']['status']}`.")
    add("")
    add(manifest["test"]["reason"])
    add("")
    add(
        "The adapter descriptor carries no holdout key, no holdout adapter directory exists, "
        "and the phase runs under an environment where the unlock variable is not set. Phase 11 "
        "reads the holdout once, after both models are frozen."
    )
    add("")

    add("## 17. Fallback status")
    add("")
    add(
        f"`{FALLBACK_STATUS}`. {manifest['architecture']['fallback_architecture']}. Nothing "
        "about it has been implemented, installed, benchmarked or measured, so no statement in "
        "this report compares the two. It remains available if a later phase judges the "
        "approximation material."
    )
    add("")

    add("## 18. Next phase")
    add("")
    add(
        f"Phase 8C: run S0 once under this frozen protocol. S0 is currently "
        f"`{manifest['s0_execution_status']}` and this report contains no S0 metric. Do not "
        "tune, do not vary a hyperparameter to see whether the number moves, and do not begin "
        "8C without an explicit instruction."
    )
    add("")
    add("---")
    add("")
    add(
        f"Protocol fingerprint `{manifest['protocol_fingerprint']}` · configuration "
        f"`{manifest['protocol_config']}` SHA-256 `{manifest['protocol_config_sha256']}`."
    )
    add("")
    return "\n".join(lines)


# --- entry point --------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """Freeze the S0 segmentation baseline protocol.

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
        "--rebuild-protocol",
        action="store_true",
        help=(
            "re-render the protocol artifacts from the committed manifest's smoke record, "
            "without training anything. For a protocol clarification only: the smoke record is "
            "reused verbatim and the rebuild refuses if it would change."
        ),
    )
    args = parser.parse_args(argv)
    if args.verify_only and args.rebuild_protocol:
        print(
            "REFUSED: --verify-only and --rebuild-protocol are mutually exclusive.", file=sys.stderr
        )
        return 2

    paths = ProjectPaths.from_root()
    if holdout_unlocked():
        print(
            f"REFUSED: {HOLDOUT_UNLOCK_ENV_VAR} is set. This phase reads development data only "
            "and declines to run in an environment where the holdout is unlocked.",
            file=sys.stderr,
        )
        return 2

    try:
        historical = historical_digests(paths)
        detector_artifacts = detector_digests(paths)
        config = load_segmentation_baseline_config(paths.configs / CONFIG_YAML)
        config_sha256 = sha256_file(paths.configs / CONFIG_YAML)
        evidence = verify_audit_evidence(paths, config)
        detector = verify_detector(paths)
        verify_no_holdout_artifacts(paths, config)
        canonical = verify_canonical_references(paths, config)
        adapter = verify_adapter(paths, config, stage="entry")
        defaults = installed_segmentation_defaults()
        assert_matches_defaults(
            config.segmentation_arguments, defaults, block="segmentation_arguments"
        )
        assert_matches_defaults(
            config.augmentation_arguments, defaults, block="augmentation_arguments"
        )
        fitness = fitness_behaviour()
    except (ConfigError, SegmentationExperimentConfigError, FreezeError) as exc:
        classification = (
            ADAPTER_FINGERPRINT_MISMATCH if isinstance(exc, AdapterMismatchError) else BLOCKED
        )
        print(f"{classification}: {exc}", file=sys.stderr)
        return 2

    try:
        runtime = runtime_facts()
    except RunError as exc:
        print(f"{SEGMENTATION_RUNTIME_BLOCKED}: {exc}", file=sys.stderr)
        return 2

    print(
        f"config     {config.experiment_id}  {config.architecture}  imgsz "
        f"{config.training['imgsz']}  batch {config.training['batch']}"
    )
    print(
        f"adapter    {adapter['counts']['instances']} instances  "
        f"{adapter['counts']['train_images']}/{adapter['counts']['validation_images']} images  "
        f"fingerprints VERIFIED"
    )
    print(
        f"detector   {detector['selected_experiment']} {detector['model']} "
        f"imgsz {detector['imgsz']}  {detector['status']}"
    )
    print(f"fitness    {fitness['classification']}")
    print(
        f"runtime    torch {runtime['torch']}  ultralytics {runtime['ultralytics']}  "
        f"{runtime['gpu_name']} {runtime['gpu_arch']}"
    )

    if args.verify_only:
        print("VERIFIED: preconditions hold. No weights fetched, nothing trained, nothing written.")
        print(f"protocol   sha256 {config.fingerprint()}")
        print(f"holdout    {HOLDOUT_STATUS}")
        return 0

    try:
        weights = ensure_pretrained_weights(paths, config.weight_identifier)
    except RunError as exc:
        print(f"{SEGMENTATION_RUNTIME_BLOCKED}: {exc}", file=sys.stderr)
        return 2
    print(
        f"weights    {weights['identifier']}  sha256 {weights['sha256']}  "
        f"{weights['size_bytes']} bytes"
    )

    if args.rebuild_protocol:
        try:
            smoke = recorded_smoke_test(paths, weights)
        except FreezeError as exc:
            print(f"{BLOCKED}: {exc}", file=sys.stderr)
            return 2
        print(
            f"smoke      REUSED from the committed manifest  {smoke['runtime_seconds']} s  "
            "(nothing was trained)"
        )
    else:
        try:
            smoke = run_smoke_test(paths, config, weights)
        except MemoryConstraintError as exc:
            print(f"{MEMORY_CONSTRAINT_REVIEW_REQUIRED}: {exc}", file=sys.stderr)
            return 2
        except FreezeError as exc:
            print(f"{SEGMENTATION_RUNTIME_BLOCKED}: {exc}", file=sys.stderr)
            return 2
        print(
            f"smoke      {smoke['status']}  {smoke['runtime_seconds']} s  peak "
            f"{smoke['peak_gpu_memory_reserved_gib']} GiB  "
            f"optimizer {smoke['optimizer_selected'].get('optimizer')}"
        )

    try:
        removed = clear_dataset_caches(paths, config)
        verify_adapter(
            paths,
            config,
            stage="after the smoke test" if not args.rebuild_protocol else "after the rebuild",
        )
        compare_digests(historical, historical_digests(paths), label="phase 8A artifacts")
        compare_digests(detector_artifacts, detector_digests(paths), label="phase 7D artifacts")
    except (AdapterMismatchError, FreezeError) as exc:
        classification = (
            ADAPTER_FINGERPRINT_MISMATCH if isinstance(exc, AdapterMismatchError) else BLOCKED
        )
        print(f"{classification}: {exc}", file=sys.stderr)
        return 2
    if removed:
        print(f"caches     removed {len(removed)} framework label cache file(s)")

    approval = build_approval(
        config=config, evidence=evidence, adapter=adapter, historical=historical
    )
    manifest = build_manifest(
        config=config,
        config_sha256=config_sha256,
        evidence=evidence,
        adapter=adapter,
        canonical=canonical,
        detector=detector,
        runtime=runtime,
        weights=weights,
        defaults=defaults,
        fitness=fitness,
        smoke=smoke,
        historical=historical,
        detector_artifacts=detector_artifacts,
    )
    manifest["known_approximation_categories"] = approval["known_approximation_categories"]
    report = build_report(manifest, commit=git_commit(paths.root))

    findings = (
        scan_for_sensitive(report)
        + scan_for_sensitive(json.dumps(manifest))
        + scan_for_sensitive(json.dumps(approval))
    )
    if findings:
        print(f"{BLOCKED}: sensitive content in the emitted artifacts: {findings}", file=sys.stderr)
        return 2

    approval_digest = write_json(paths.reports / APPROVAL_JSON, approval)
    manifest_digest = write_json(paths.reports / S0_MANIFEST_JSON, manifest)
    (paths.reports / S0_REPORT_MD).write_text(report, encoding="utf-8", newline="\n")

    record = ProvenanceRecord.create(
        name="segmentation_baseline_protocol_freeze",
        phase=8,
        config={
            "segmentation_baseline": f"configs/{CONFIG_YAML}",
            "protocol_fingerprint": config.fingerprint(),
            "protocol_config_sha256": config_sha256,
        },
        details={
            "phase": PHASE,
            "classification": FROZEN,
            "experiment_id": config.experiment_id,
            "architecture": config.architecture,
            "architecture_selection": ARCHITECTURE_SELECTED,
            "fallback_status": FALLBACK_STATUS,
            "adapter_review_status": APPROVED,
            "adapter_role": ADAPTER_ROLE,
            "canonical_ground_truth": CANONICAL_GROUND_TRUTH,
            "conversion_characterisation": APPROXIMATION,
            "adapter_fingerprints": adapter["fingerprints"],
            "adapter_counts": adapter["counts"],
            "adapter_regenerated": False,
            "all_instances_retained": True,
            "fidelity_based_filtering": "NONE",
            "pretrained_weight_sha256": weights["sha256"],
            "checkpoint_fitness_behaviour": fitness["classification"],
            "smoke_status": smoke["status"],
            "smoke_classification": SMOKE_STATUS_NON_EXPERIMENTAL,
            "smoke_metrics_recorded": False,
            "s0_execution_status": S0_EXECUTION_STATUS,
            "models_trained_in_this_phase": 0,
            "detector_touched": False,
            "holdout_accessed": False,
            "phase_8a_artifacts_unchanged": True,
            "detection_artifacts_unchanged": True,
        },
        repo_root=paths.root,
    )
    record.add_input(paths.configs / CONFIG_YAML, relative_to=paths.root)
    for name in (
        AUDIT_MANIFEST_JSON,
        AUDIT_REPORT_MD,
        AUDIT_TABLE_CSV,
        FINAL_DETECTOR_JSON,
        TASK_MANIFEST_JSON,
        SPLIT_MANIFEST_JSON,
    ):
        record.add_input(paths.reports / name, relative_to=paths.root)
    for name in (APPROVAL_JSON, S0_MANIFEST_JSON, S0_REPORT_MD):
        record.add_output(paths.reports / name, relative_to=paths.root)
    record.write_json(paths.reports / S0_PROVENANCE_JSON)

    print(FROZEN)
    print(f"approval   reports/{APPROVAL_JSON}  sha256 {approval_digest}")
    print(f"manifest   reports/{S0_MANIFEST_JSON}  sha256 {manifest_digest}")
    print(f"report     reports/{S0_REPORT_MD}")
    print(f"S0         {S0_EXECUTION_STATUS}  (no performance metric exists)")
    print(f"holdout    {HOLDOUT_STATUS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
