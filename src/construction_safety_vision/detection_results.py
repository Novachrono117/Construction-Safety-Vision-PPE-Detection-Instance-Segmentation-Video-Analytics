"""Recording a detection experiment's result so it can be audited later.

A metric is only evidence if you can say exactly what produced it. This module
holds the parts of that bookkeeping which are pure functions - the experiment
fingerprint, the manifest schema, the metric extraction from a framework result
dictionary - so they can be tested without a GPU, a dataset or a training run.

Three rules are enforced here rather than trusted.

**The fingerprint covers the experiment, not the run.** It hashes the protocol,
the pretrained weights, the dataset, the split, the critical resolved training
arguments and the selected checkpoint. It deliberately excludes timestamps,
paths and machine identity, so re-running the same experiment on another machine
produces the same fingerprint and a changed hyperparameter does not.

**A result may not mention the holdout.** The manifest is validated for that,
because a results file is one of the easiest places for a protected identifier to
leak into a public repository.

**A missing metric is recorded as missing.** If the installed framework does not
expose a per-class number reliably, the manifest says ``NOT_EXPOSED_RELIABLY``
rather than carrying a value derived by guesswork.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

MANIFEST_SCHEMA_VERSION = 1
"""Schema version of ``reports/detection_D0_manifest.json``."""

NOT_EXPOSED = "NOT_EXPOSED_RELIABLY"
"""Recorded where the installed framework does not expose a metric dependably."""

TEST_PROTECTED = "PROTECTED_NOT_ACCESSED"
"""The only acceptable test status for a development experiment."""

HIGH_SAMPLING_UNCERTAINTY = "HIGH_SAMPLING_UNCERTAINTY"
"""Marker attached to a class whose validation support is too small to trust."""

COMPLETE = "D0_BASELINE_COMPLETE"
"""Status of a finished, validated baseline run."""

TRAINING_FAILED = "TRAINING_FAILED"
"""Status of a run that did not finish."""

PROTOCOL_INPUT_MISMATCH = "PROTOCOL_INPUT_MISMATCH"
"""Status when a frozen input fingerprint no longer matches."""

STATUSES: tuple[str, ...] = (COMPLETE, TRAINING_FAILED, PROTOCOL_INPUT_MISMATCH)
"""Every status a result manifest may declare."""

FORBIDDEN_SPLIT = "test"
"""The protected split, which no result artifact may carry data for."""

PRIMARY_METRIC = "mAP@0.50:0.95"
"""The predeclared primary metric of a detection experiment."""

SECONDARY_METRICS: tuple[str, ...] = ("mAP@0.50", "precision", "recall")
"""The predeclared secondary metrics, in order."""

PER_CLASS_METRICS: tuple[str, ...] = ("AP@0.50", "AP@0.50:0.95", "precision", "recall")
"""The predeclared per-class metrics, in order."""

CRITICAL_ARGUMENT_KEYS: tuple[str, ...] = (
    "model",
    "imgsz",
    "epochs",
    "batch",
    "optimizer",
    "lr0",
    "lrf",
    "momentum",
    "weight_decay",
    "warmup_epochs",
    "patience",
    "amp",
    "deterministic",
    "seed",
    "close_mosaic",
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
    "mosaic",
    "mixup",
    "cutmix",
    "copy_paste",
    "erasing",
    "box",
    "cls",
    "dfl",
)
"""Resolved arguments that define the experiment.

Anything in this list changes what the experiment *is*, so it enters the
fingerprint. Arguments describing where output went, how verbose the logs were,
or how many dataloader workers ran do not.
"""

REQUIRED_MANIFEST_FIELDS: tuple[str, ...] = (
    "schema_version",
    "experiment_id",
    "status",
    "model",
    "pretrained_weights",
    "git_commit",
    "baseline_config_sha256",
    "adapter_manifest_sha256",
    "dataset_fingerprints",
    "split_reference",
    "train_images",
    "validation_images",
    "test",
    "epochs_configured",
    "epochs_completed",
    "best_epoch",
    "early_stopped",
    "resolved_optimizer",
    "best_checkpoint",
    "last_checkpoint",
    "validation_metrics",
    "per_class_metrics",
    "rare_class",
    "training_duration_seconds",
    "d0_experiment_sha256",
)
"""Fields a result manifest must carry."""


class ResultError(RuntimeError):
    """Raised when an experiment result cannot be recorded honestly."""


def digest(payload: Any) -> str:
    """Hash a JSON-serialisable payload deterministically.

    Args:
        payload: Content to hash.

    Returns:
        Its SHA-256 hex digest.
    """
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def critical_arguments(resolved: Mapping[str, Any]) -> dict[str, Any]:
    """Extract the resolved arguments that define the experiment.

    Args:
        resolved: The framework's complete resolved argument mapping.

    Returns:
        Only the arguments in :data:`CRITICAL_ARGUMENT_KEYS` that are present,
        with values normalised for stable hashing.
    """
    selected: dict[str, Any] = {}
    for key in CRITICAL_ARGUMENT_KEYS:
        if key not in resolved:
            continue
        value = resolved[key]
        if isinstance(value, float):
            value = round(value, 10)
        elif not isinstance(value, (str, int, bool, type(None))):
            value = str(value)
        selected[key] = value
    return selected


def experiment_fingerprint(
    *,
    baseline_config_sha256: str,
    pretrained_weights_sha256: str,
    adapter_manifest_sha256: str,
    split_assignment_sha256: str,
    resolved_arguments: Mapping[str, Any],
    best_checkpoint_sha256: str,
) -> str:
    """Fingerprint the experiment by everything that defines it.

    Excludes timestamps, filesystem paths and machine identity on purpose: the
    same experiment run on another machine must fingerprint the same, and a
    changed hyperparameter must not.

    Args:
        baseline_config_sha256: Digest of the frozen protocol.
        pretrained_weights_sha256: Digest of the starting checkpoint.
        adapter_manifest_sha256: Digest of the dataset the model trained on.
        split_assignment_sha256: Digest of the frozen split membership.
        resolved_arguments: The framework's resolved arguments.
        best_checkpoint_sha256: Digest of the selected checkpoint.

    Returns:
        A SHA-256 hex digest.
    """
    return digest(
        {
            "baseline_config_sha256": baseline_config_sha256,
            "pretrained_weights_sha256": pretrained_weights_sha256,
            "adapter_manifest_sha256": adapter_manifest_sha256,
            "split_assignment_sha256": split_assignment_sha256,
            "critical_arguments": critical_arguments(resolved_arguments),
            "best_checkpoint_sha256": best_checkpoint_sha256,
        }
    )


def _finite(value: Any) -> float | str:
    """Convert a framework metric to a JSON-safe number.

    Args:
        value: Raw metric value.

    Returns:
        The rounded float, or :data:`NOT_EXPOSED` when it is absent or not a
        finite number.
    """
    try:
        number = float(value)
    except (TypeError, ValueError):
        return NOT_EXPOSED
    if number != number or number in (float("inf"), float("-inf")):
        return NOT_EXPOSED
    return round(number, 6)


def global_metrics(results: Mapping[str, Any]) -> dict[str, float | str]:
    """Extract the global validation metrics from a framework result mapping.

    Args:
        results: The framework's ``results_dict``-style mapping.

    Returns:
        The predeclared global metrics, keyed by the project's metric names.
    """
    mapping = {
        PRIMARY_METRIC: "metrics/mAP50-95(B)",
        "mAP@0.50": "metrics/mAP50(B)",
        "precision": "metrics/precision(B)",
        "recall": "metrics/recall(B)",
    }
    return {name: _finite(results.get(key)) for name, key in mapping.items()}


def per_class_metrics(
    class_names: Sequence[str],
    *,
    class_indices: Sequence[int],
    precision: Sequence[Any],
    recall: Sequence[Any],
    ap50: Sequence[Any],
    ap: Sequence[Any],
) -> dict[str, dict[str, float | str]]:
    """Assemble per-class metrics, keyed by class name.

    The framework reports per-class arrays indexed by the classes it actually
    evaluated, not by the class map. A class the validation split does not
    contain, or that the framework skipped, is recorded as
    :data:`NOT_EXPOSED` rather than silently given a zero - those mean different
    things, and a zero would read as "the model failed at it".

    Args:
        class_names: The frozen class names, ordered by class index.
        class_indices: Class indices the framework reported, in array order.
        precision: Per-class precision, in array order.
        recall: Per-class recall, in array order.
        ap50: Per-class AP@0.50, in array order.
        ap: Per-class AP@0.50:0.95, in array order.

    Returns:
        One entry per class name.
    """
    by_index = {int(value): position for position, value in enumerate(class_indices)}
    result: dict[str, dict[str, float | str]] = {}
    for index, name in enumerate(class_names):
        position = by_index.get(index)
        if position is None:
            result[name] = dict.fromkeys(PER_CLASS_METRICS, NOT_EXPOSED)
            continue
        result[name] = {
            "AP@0.50": _finite(ap50[position]) if position < len(ap50) else NOT_EXPOSED,
            "AP@0.50:0.95": _finite(ap[position]) if position < len(ap) else NOT_EXPOSED,
            "precision": _finite(precision[position]) if position < len(precision) else NOT_EXPOSED,
            "recall": _finite(recall[position]) if position < len(recall) else NOT_EXPOSED,
        }
    return result


def ultralytics_fitness(map50: float, map50_95: float) -> float:
    """Reproduce the framework's validation fitness.

    Ultralytics ranks checkpoints by ``0.1 * mAP@0.50 + 0.9 * mAP@0.50:0.95``.
    Recomputing it from the epoch history is how the best epoch recorded in the
    checkpoint gets cross-checked against an independent artifact.

    Args:
        map50: mAP@0.50.
        map50_95: mAP@0.50:0.95.

    Returns:
        The fitness value.
    """
    return 0.1 * map50 + 0.9 * map50_95


def contains_forbidden_split(value: Any) -> bool:
    """Report whether a parsed value carries the protected split as a key.

    Args:
        value: Any parsed value.

    Returns:
        ``True`` when a mapping key equals the protected split name.
    """
    if isinstance(value, Mapping):
        return any(
            (isinstance(key, str) and key.strip().lower() == FORBIDDEN_SPLIT)
            or contains_forbidden_split(item)
            for key, item in value.items()
        )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return any(contains_forbidden_split(item) for item in value)
    return False


def validate_result_manifest(
    manifest: Mapping[str, Any], *, class_names: Sequence[str]
) -> list[str]:
    """Check a result manifest for defects that would make it misleading.

    Args:
        manifest: The parsed manifest.
        class_names: The frozen class names.

    Returns:
        One description per problem found, empty when the manifest is sound.
    """
    problems: list[str] = []
    missing = [name for name in REQUIRED_MANIFEST_FIELDS if name not in manifest]
    if missing:
        problems.append(f"missing required field(s): {', '.join(missing)}")
        return problems

    if manifest["schema_version"] != MANIFEST_SCHEMA_VERSION:
        problems.append(f"schema_version {manifest['schema_version']!r} is unsupported")
    if manifest["status"] not in STATUSES:
        problems.append(f"status {manifest['status']!r} is not one of {list(STATUSES)}")

    problems.extend(_validate_test_protection(manifest))
    problems.extend(_validate_metrics(manifest, class_names))
    problems.extend(_validate_checkpoints(manifest))

    if not manifest["d0_experiment_sha256"]:
        problems.append("d0_experiment_sha256 is empty")
    if manifest["resolved_optimizer"] in (None, "", NOT_EXPOSED):
        problems.append("resolved_optimizer was not recorded; optimizer=auto must resolve")
    return problems


def _validate_test_protection(manifest: Mapping[str, Any]) -> list[str]:
    """Check that the manifest carries no holdout data.

    Args:
        manifest: The parsed manifest.

    Returns:
        One description per problem found.
    """
    problems: list[str] = []
    test = manifest["test"]
    if not isinstance(test, Mapping) or test.get("status") != TEST_PROTECTED:
        problems.append(f"test status must be {TEST_PROTECTED!r}")
    elif set(test) - {"status", "reason"}:
        problems.append(f"the test entry carries unexpected field(s): {sorted(set(test))}")
    without_test = {key: value for key, value in manifest.items() if key != "test"}
    if contains_forbidden_split(without_test):
        problems.append("the manifest carries a 'test' keyed section outside the status entry")
    return problems


def _validate_metrics(manifest: Mapping[str, Any], class_names: Sequence[str]) -> list[str]:
    """Check the metric hierarchy and the per-class table.

    Args:
        manifest: The parsed manifest.
        class_names: The frozen class names.

    Returns:
        One description per problem found.
    """
    problems: list[str] = []
    metrics = manifest["validation_metrics"]
    if not isinstance(metrics, Mapping):
        problems.append("validation_metrics must be a mapping")
        return problems
    if PRIMARY_METRIC not in metrics:
        problems.append(f"validation_metrics is missing the primary metric {PRIMARY_METRIC!r}")
    for name in SECONDARY_METRICS:
        if name not in metrics:
            problems.append(f"validation_metrics is missing the secondary metric {name!r}")

    per_class = manifest["per_class_metrics"]
    if not isinstance(per_class, Mapping):
        problems.append("per_class_metrics must be a mapping")
        return problems
    if set(per_class) != set(class_names):
        problems.append(
            f"per_class_metrics covers {sorted(per_class)}, the frozen class map is "
            f"{sorted(class_names)}"
        )
    for name, entry in sorted(per_class.items()):
        if not isinstance(entry, Mapping):
            problems.append(f"per_class_metrics[{name!r}] is not a mapping")
            continue
        for metric in PER_CLASS_METRICS:
            if metric not in entry:
                problems.append(f"per_class_metrics[{name!r}] is missing {metric!r}")

    rare = manifest["rare_class"]
    if not isinstance(rare, Mapping):
        problems.append("rare_class must be a mapping")
    else:
        if rare.get("name") not in class_names:
            problems.append("rare_class.name is not one of the frozen classes")
        if rare.get("warning") != HIGH_SAMPLING_UNCERTAINTY:
            problems.append(f"rare_class.warning must be {HIGH_SAMPLING_UNCERTAINTY!r}")
    return problems


def _validate_checkpoints(manifest: Mapping[str, Any]) -> list[str]:
    """Check that both checkpoints are fingerprinted and none is committed.

    Args:
        manifest: The parsed manifest.

    Returns:
        One description per problem found.
    """
    problems: list[str] = []
    for name in ("best_checkpoint", "last_checkpoint"):
        entry = manifest[name]
        if not isinstance(entry, Mapping):
            problems.append(f"{name} must be a mapping")
            continue
        if not entry.get("sha256"):
            problems.append(f"{name}.sha256 is missing; a checkpoint must be fingerprinted")
        if not isinstance(entry.get("size_bytes"), int) or entry["size_bytes"] <= 0:
            problems.append(f"{name}.size_bytes is missing or not positive")
        if entry.get("committed") is not False:
            problems.append(f"{name}.committed must be false; weights are never committed")
    return problems
