"""Run primitives for the S0 segmentation experiment.

Phase 8C. Sibling of :mod:`construction_safety_vision.detection_run`, which
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
