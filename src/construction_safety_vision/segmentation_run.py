"""Run primitives for the project's segmentation experiments.

Phase 8C for S0, extended in phase 8F for S1. Sibling of
:mod:`construction_safety_vision.detection_run`, which
already supplies everything the two tasks share - pretrained-weight
fingerprinting, runtime facts, framework log capture and optimizer evidence -
and is imported rather than copied.

What lives here is what segmentation needs and detection does not.

**A runtime dataset view.** Ultralytics writes ``.cache`` files next to the
labels it scans. The audited adapter is phase 8A evidence, so the training run
gets a hard-linked view of it instead and the framework's transient files land
there. The view is verified to carry byte-identical labels before anything is
trained; if it did not, training would be measuring different data than the one
the approval covers.

**Native segmentation fitness, reconstructed.** The framework's checkpoint rule
for a segmentation model is the sum of box and mask mAP@0.50:0.95. Ultralytics
records the per-epoch terms in ``results.csv`` but not the fitness itself, so the
selected epoch is only checkable by recomputing the composite from those columns
and confirming the recorded best is its argmax. Without that step, "the frozen
rule chose this checkpoint" is an assumption.

.. note::
   ``scripts/freeze_segmentation_baseline.py`` (phase 8B) carries its own copies
   of the adapter-fingerprint helpers. It is deliberately not refactored onto
   this module: phase 8B is complete and its committed artifacts were produced
   by that script, so putting a newer implementation under its name would risk
   an untested change beneath published evidence for no benefit. The duplication
   is recorded here rather than hidden, classified ``LOW``, and belongs to a
   later cleanup phase rather than an experiment phase.
"""

from __future__ import annotations

import csv
import os
import re
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from construction_safety_vision.data.materialization import digest, sha256_bytes
from construction_safety_vision.data.segmentation_adapter import label_fingerprint
from construction_safety_vision.data.yolo_detection_adapter import (
    ADAPTER_ROOT_PLACEHOLDER,
    dataset_yaml,
)

RUN_ROOT = "artifacts/segmentation"
"""Git-ignored parent of every segmentation run directory."""

RUNTIME_VIEW_ROOT = "data/processed/adapters/yolo_segmentation_s0_runtime"
"""Git-ignored runtime view of the approved adapter, where caches may live."""

SPLIT_DIRECTORIES: Mapping[str, str] = {"train": "train", "validation": "val"}
"""Canonical split name to the directory name Ultralytics expects."""

BOX_FITNESS_COLUMN = "metrics/mAP50-95(B)"
"""Per-epoch box mAP@0.50:0.95 as ``results.csv`` names it."""

MASK_FITNESS_COLUMN = "metrics/mAP50-95(M)"
"""Per-epoch mask mAP@0.50:0.95 as ``results.csv`` names it."""

FITNESS_TOLERANCE = 1e-6
"""Numeric slack when confirming the recorded best epoch is the composite argmax.

``results.csv`` stores rounded text, so an exact comparison against the value the
trainer held in memory would fail on serialisation alone.
"""

METRIC_FIGURES: tuple[str, ...] = (
    "confusion_matrix.png",
    "confusion_matrix_normalized.png",
    "BoxPR_curve.png",
    "BoxP_curve.png",
    "BoxR_curve.png",
    "BoxF1_curve.png",
    "MaskPR_curve.png",
    "MaskP_curve.png",
    "MaskR_curve.png",
    "MaskF1_curve.png",
    "results.png",
)
"""Metric-only plots that may be committed.

An allowlist, not a filter. The framework also writes ``train_batch*.jpg``,
``val_batch*.jpg`` and ``labels.jpg``, which render dataset imagery and
prediction montages; committing those would publish the dataset and pre-empt the
deliberate error-analysis phase.
"""


class SegmentationRunError(RuntimeError):
    """Raised when the segmentation experiment cannot execute as specified."""


# --- the approved adapter -----------------------------------------------------


def read_adapter_labels(
    root: Path, directories: Mapping[str, str] = SPLIT_DIRECTORIES
) -> dict[str, dict[str, str]]:
    """Read an adapter's label files, keyed the way phase 8A keyed them.

    Args:
        root: The adapter root directory.
        directories: Canonical split name to on-disk directory name.

    Returns:
        Label text keyed by split then image stem.

    Raises:
        SegmentationRunError: If a split directory is missing.
    """
    labels: dict[str, dict[str, str]] = {}
    for split, directory in directories.items():
        label_dir = root / "labels" / directory
        if not label_dir.is_dir():
            msg = f"adapter label directory not found: {label_dir}"
            raise SegmentationRunError(msg)
        labels[split] = {
            path.stem: path.read_text(encoding="utf-8") for path in sorted(label_dir.glob("*.txt"))
        }
    return labels


def adapter_fingerprints(labels: Mapping[str, Mapping[str, str]]) -> dict[str, str]:
    """Fingerprint a label set exactly as phase 8A did.

    Args:
        labels: Label text keyed by split then image stem.

    Returns:
        The four digests the approval records.
    """
    return {
        "labels_train_sha256": label_fingerprint(labels["train"]),
        "labels_validation_sha256": label_fingerprint(labels["validation"]),
        "labels_development_sha256": label_fingerprint(
            {f"{split}/{stem}": text for split in labels for stem, text in labels[split].items()}
        ),
        "image_membership_sha256": digest(
            {split: sorted(labels[split]) for split in sorted(labels)}
        ),
    }


def adapter_counts(labels: Mapping[str, Mapping[str, str]]) -> dict[str, int]:
    """Count images, instance rows and negatives in a label set.

    Args:
        labels: Label text keyed by split then image stem.

    Returns:
        The cardinality the approval records.
    """
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
    return counts


# --- the runtime view ---------------------------------------------------------


def transfer_file(source: Path, destination: Path) -> str:
    """Place a file at the destination without copying its bytes if avoidable.

    Args:
        source: Existing file.
        destination: Where it should appear.

    Returns:
        ``"HARDLINK"`` or ``"COPY"``, whichever the filesystem allowed.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.unlink()
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)
        return "COPY"
    return "HARDLINK"


def build_runtime_view(
    *,
    source_root: Path,
    destination_root: Path,
    class_map: Mapping[str, int],
    directories: Mapping[str, str] = SPLIT_DIRECTORIES,
) -> dict[str, Any]:
    """Materialise a training-time view of the approved adapter.

    Same image membership, same image bytes, same label bytes. Nothing is
    converted, re-normalised or re-encoded: the view exists so the framework's
    ``.cache`` files land somewhere disposable instead of inside phase 8A's
    evidence, not to change anything about the data.

    Args:
        source_root: The approved adapter root.
        destination_root: Where the view should live. Rebuilt from scratch.
        class_map: The frozen class map, for the descriptor.
        directories: Canonical split name to on-disk directory name.

    Returns:
        What was written, including the transfer modes used.

    Raises:
        SegmentationRunError: If a source image or label is missing, or a
            transferred file's bytes differ from the source's.
    """
    if destination_root.exists():
        shutil.rmtree(destination_root)
    destination_root.mkdir(parents=True, exist_ok=True)

    transfer_modes: set[str] = set()
    counts: dict[str, int] = {}
    for split, directory in directories.items():
        source_images = source_root / "images" / directory
        source_labels = source_root / "labels" / directory
        if not source_images.is_dir() or not source_labels.is_dir():
            msg = f"approved adapter is incomplete: {split} images or labels are missing"
            raise SegmentationRunError(msg)

        written = 0
        for label in sorted(source_labels.glob("*.txt")):
            matches = sorted(source_images.glob(f"{label.stem}.*"))
            if not matches:
                msg = f"approved adapter has a label with no image: {split}/{label.stem}"
                raise SegmentationRunError(msg)
            image = matches[0]
            image_target = destination_root / "images" / directory / image.name
            label_target = destination_root / "labels" / directory / label.name
            transfer_modes.add(transfer_file(image, image_target))
            transfer_modes.add(transfer_file(label, label_target))
            if sha256_bytes(image_target) != sha256_bytes(image):
                msg = f"runtime view image differs from the approved adapter: {image.name}"
                raise SegmentationRunError(msg)
            written += 1
        counts[split] = written

    descriptor = dataset_yaml(
        path_value=str(destination_root.resolve()),
        train_directory=f"images/{directories['train']}",
        validation_directory=f"images/{directories['validation']}",
        class_map=class_map,
    )
    (destination_root / "dataset.yaml").write_text(descriptor, encoding="utf-8", newline="\n")
    template = dataset_yaml(
        path_value=ADAPTER_ROOT_PLACEHOLDER,
        train_directory=f"images/{directories['train']}",
        validation_directory=f"images/{directories['validation']}",
        class_map=class_map,
    )
    (destination_root / "README.md").write_text(
        "# YOLO segmentation adapter - S0 RUNTIME VIEW\n\n"
        "Generated by `scripts/train_segmentation_baseline.py` (phase 8C). A hard-linked view "
        "of the phase 8A audited, phase 8B approved adapter, created so that the framework's\n"
        "`.cache` files and other transient writes do not land inside phase 8A's evidence.\n\n"
        "- image membership, image bytes and label bytes are identical to the approved adapter\n"
        "- no label conversion, no coordinate rewrite, no re-encoding\n"
        "- development splits only; the holdout has no directory here and must not be added\n"
        "- git-ignored and disposable: delete it and re-run the script to rebuild\n\n"
        "The canonical COCO instance segmentation frozen in phase 5D remains the ground truth.\n\n"
        "Portable descriptor template:\n\n"
        "```yaml\n" + template + "```\n",
        encoding="utf-8",
        newline="\n",
    )
    return {
        "root": destination_root.name,
        "descriptor": "dataset.yaml",
        "image_counts": counts,
        "transfer_modes": sorted(transfer_modes),
        "label_conversion": "NONE",
        "image_transformation": "NONE",
        "committed": False,
    }


# --- native checkpoint selection ----------------------------------------------


def read_epoch_history(run_directory: Path) -> list[dict[str, str]]:
    """Read the framework's per-epoch metric log.

    Args:
        run_directory: The run's output directory.

    Returns:
        One mapping per epoch, with whitespace stripped from the headers.

    Raises:
        SegmentationRunError: If the log is missing.
    """
    path = run_directory / "results.csv"
    if not path.is_file():
        msg = f"the run wrote no results.csv: {path}"
        raise SegmentationRunError(msg)
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [
            {(key or "").strip(): (value or "").strip() for key, value in row.items()}
            for row in reader
        ]


def native_fitness_curve(history: Sequence[Mapping[str, str]]) -> list[tuple[int, float]]:
    """Recompute the framework's segmentation fitness for every epoch.

    ``SegmentMetrics.fitness`` is ``seg.fitness() + DetMetrics.fitness``, and
    each term weights ``[precision, recall, mAP@0.50, mAP@0.50:0.95]`` by
    ``[0, 0, 0, 1]``. So the composite is simply the sum of the two
    mAP@0.50:0.95 columns, which is what makes it recoverable from the log at
    all.

    Args:
        history: Parsed ``results.csv`` rows.

    Returns:
        ``(epoch, fitness)`` pairs, skipping any row missing a term.

    Raises:
        SegmentationRunError: If neither column is present, which would mean the
            installed framework no longer logs what the checkpoint rule needs.
    """
    if not history:
        return []
    headers = set(history[0])
    missing = [name for name in (BOX_FITNESS_COLUMN, MASK_FITNESS_COLUMN) if name not in headers]
    if missing:
        msg = (
            f"results.csv does not expose {missing}, so the native composite fitness cannot be "
            "reconstructed and the selected checkpoint cannot be verified against the frozen rule"
        )
        raise SegmentationRunError(msg)

    curve: list[tuple[int, float]] = []
    for row in history:
        try:
            epoch = int(float(row["epoch"]))
            value = float(row[BOX_FITNESS_COLUMN]) + float(row[MASK_FITNESS_COLUMN])
        except (KeyError, TypeError, ValueError):
            continue
        curve.append((epoch, value))
    return curve


def best_epoch_by_native_fitness(
    history: Sequence[Mapping[str, str]],
) -> tuple[int | None, float | None]:
    """Find the epoch the frozen native rule selects.

    Ties go to the earlier epoch, which is the framework's own behaviour: it
    replaces the best checkpoint only on a strict improvement.

    Args:
        history: Parsed ``results.csv`` rows.

    Returns:
        The best epoch and its composite fitness, or ``(None, None)`` when the
        log is empty.
    """
    curve = native_fitness_curve(history)
    if not curve:
        return None, None
    best_epoch, best_value = curve[0]
    for epoch, value in curve[1:]:
        if value > best_value:
            best_epoch, best_value = epoch, value
    return best_epoch, best_value


def checkpoint_record(path: Path) -> dict[str, Any]:
    """Fingerprint a checkpoint by its bytes.

    Args:
        path: The checkpoint file.

    Returns:
        Its digest, size and repository-relative name.

    Raises:
        SegmentationRunError: If it does not exist.
    """
    if not path.is_file():
        msg = f"expected checkpoint not found: {path}"
        raise SegmentationRunError(msg)
    return {
        "name": path.name,
        "sha256": sha256_bytes(path),
        "size_bytes": path.stat().st_size,
        "committed": False,
    }


def copy_metric_figures(figures_directory: Path, sources: Sequence[Path]) -> list[str]:
    """Copy only the metric-only plots out of a run directory.

    The first candidate offering a given name wins, and later ones are skipped
    rather than overwriting it. Training and validation directories both emit
    e.g. ``confusion_matrix.png``, and silently overwriting one with the other
    would leave a figure whose provenance nobody could state.

    Args:
        figures_directory: Destination directory.
        sources: Candidate files, in priority order.

    Returns:
        The names copied, sorted and unique.
    """
    figures_directory.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    for source in sources:
        if source.name not in METRIC_FIGURES or source.name in copied:
            continue
        shutil.copy2(source, figures_directory / source.name)
        copied.append(source.name)
    return sorted(copied)


# --- run reporting, shared from phase 8F onwards --------------------------------
#
# Phase 8C's ``scripts/train_segmentation_baseline.py`` carries its own copies of
# the functions below. They are lifted here rather than imported from there
# because a script is not importable, and phase 8C's script is deliberately left
# alone: it produced S0's committed evidence, S0 is never re-run, and putting a
# newer implementation under its name would risk an untested change beneath
# published numbers. Same reasoning, same classification (``LOW``) and same
# resolution as the phase 8B note above: a later cleanup phase, never an
# experiment phase.

METRIC_PRECISION = 6
"""Decimal places every reported metric is rounded to."""

NOT_EXPOSED = "NOT_EXPOSED_RELIABLY"
"""Recorded where the installed framework does not expose a value dependably."""

EPOCH_TIME_COLUMN = "time"
"""Cumulative training seconds, as ``results.csv`` names it."""

REDACTED_PATH = "EXTERNAL_ABSOLUTE_PATH_REDACTED"
"""Stand-in for an absolute path that lies outside the repository."""

_DRIVE_LETTER_PATH = re.compile(r"^[A-Za-z]:/")
_EXTERNAL_PREFIXES = ("/home/", "/Users/", "/root/", "//")

SUMMARY_PATTERN = re.compile(
    r"summary(?P<fused>\s*\(fused\))?:\s*(?P<layers>[\d,]+)\s+layers,\s*"
    r"(?P<parameters>[\d,]+)\s+parameters.*?(?P<gflops>[\d.]+)\s+GFLOPs"
)
"""The framework's own model summary line.

Parsed with a regex rather than by splitting on commas: the counts carry
thousands separators, so ``2,843,583 parameters`` splits into three fields and a
naive parser silently records 543.
"""

EPOCHS_COMPLETED_PATTERN = re.compile(r"(\d+) epochs completed in ([\d.]+) hours")
"""The framework's end-of-training line, used to corroborate the duration."""


def relativise(value: Any, root: Path) -> Any:
    """Strip machine-specific absolute paths out of a recorded value.

    Ultralytics records absolute paths for ``model``, ``data`` and its output
    directories, and those name this machine and this user. A committed artifact
    must not, so a path under the repository root becomes repository-relative and
    anything else absolute becomes a sentinel.

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
    if _DRIVE_LETTER_PATH.match(text) or text.startswith(_EXTERNAL_PREFIXES):
        return REDACTED_PATH
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


def resolved_training_arguments(run_directory: Path, root: Path) -> dict[str, Any]:
    """Read the arguments the framework recorded for a run.

    Args:
        run_directory: The run's output directory.
        root: Repository root, for path rewriting.

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
    return {key: relativise(value, root) for key, value in data.items()}


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
    """Recover a training run's duration from artifacts that are on disk.

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
    match = EPOCHS_COMPLETED_PATTERN.search(log_text)
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
    best = max((item for _, item in curve), default=None)
    return {
        "epochs_logged": len(history),
        "loss_and_lr_columns": sorted(columns),
        "first_epoch": {name: value(first, name) for name in sorted(columns)},
        "last_epoch": {name: value(last, name) for name in sorted(columns)},
        "native_fitness_first": round(curve[0][1], METRIC_PRECISION) if curve else None,
        "native_fitness_last": round(curve[-1][1], METRIC_PRECISION) if curve else None,
        "native_fitness_best": round(best, METRIC_PRECISION) if best is not None else None,
        "best_epoch": best_epoch,
        "epochs_after_best": len(tail),
        "improved_after_best": bool(
            best is not None and tail and max(tail) > best + FITNESS_TOLERANCE
        ),
        "observation": (
            "Read from results.csv only. No validation image was opened: image-level error "
            "analysis is a later, deliberate phase and starting it here would pre-empt it."
        ),
    }
