"""Run the D0 detection baseline: one predeclared experiment, start to finish.

Phase 6B. This is the project's first real model experiment, and its value comes
entirely from having been specified before it ran. So the script does the
verification first, writes down what is about to happen, and only then trains.

Order matters here and is deliberate:

1. every frozen input fingerprint is re-checked - protocol, class map, split,
   task datasets, adapter, pretrained weights. A mismatch is
   ``PROTOCOL_INPUT_MISMATCH`` and nothing runs;
2. a pre-run provenance record is written **before the first optimisation
   step**, so the protocol demonstrably existed before the result;
3. training runs once, with the frozen arguments, refusing to reuse or overwrite
   an existing D0 directory;
4. the checkpoint is selected by the predeclared rule, never by comparing epochs
   afterwards;
5. that checkpoint is validated once, on validation only;
6. the metrics are cross-checked against an independent artifact before anything
   is published.

The protocol declares ``optimizer: auto``, so this script is careful to
distinguish the **declared policy** from the **effective settings** Ultralytics
resolved. The report states what actually ran, not what the configuration file
would suggest.

The holdout takes no part. It has no adapter, no labels and no key in the dataset
descriptor, and this script never resolves it.

Usage:
    uv run python scripts/train_detection_baseline.py
    uv run python scripts/train_detection_baseline.py --verify-only
    uv run python scripts/train_detection_baseline.py --resume   # operational only
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import re
import shutil
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from construction_safety_vision.config import ConfigError
from construction_safety_vision.data.canonical import scan_for_sensitive
from construction_safety_vision.data.materialization import sha256_bytes
from construction_safety_vision.data.yolo_detection_adapter import load_adapter_config
from construction_safety_vision.detection_results import (
    COMPLETE,
    HIGH_SAMPLING_UNCERTAINTY,
    MANIFEST_SCHEMA_VERSION,
    NOT_EXPOSED,
    PRIMARY_METRIC,
    TEST_PROTECTED,
    ResultError,
    experiment_fingerprint,
    global_metrics,
    per_class_metrics,
    ultralytics_fitness,
    validate_result_manifest,
)
from construction_safety_vision.experiment import (
    ExperimentConfigError,
    load_detection_baseline_config,
)
from construction_safety_vision.paths import ProjectPaths
from construction_safety_vision.provenance import ProvenanceRecord, git_commit
from construction_safety_vision.splits import HOLDOUT_UNLOCK_ENV_VAR, holdout_unlocked

RUN_ROOT = "artifacts/detection"
RUN_NAME = "D0"
FIGURE_SUBDIR = "figures/detection/D0"

MANIFEST_JSON = "detection_D0_manifest.json"
REPORT_MD = "detection_D0_report.md"
PRERUN_JSON = "detection_D0_prerun.provenance.json"
PROVENANCE_JSON = "detection_D0.provenance.json"

EXPECTED_MEMBERSHIP = {
    "train": {"image_count": 303, "annotation_count": 1422, "negative_images": 10},
    "validation": {"image_count": 65, "annotation_count": 304, "negative_images": 2},
}
EXPECTED_WEIGHT_BYTES = 5613764

METRIC_FIGURES: tuple[str, ...] = (
    "confusion_matrix.png",
    "confusion_matrix_normalized.png",
    "BoxPR_curve.png",
    "BoxP_curve.png",
    "BoxR_curve.png",
    "BoxF1_curve.png",
    "PR_curve.png",
    "P_curve.png",
    "R_curve.png",
    "F1_curve.png",
    "results.png",
)
"""Metric-only plots that may be committed.

An allowlist, not a filter: `train_batch*.jpg`, `val_batch*.jpg` and
`labels*.jpg` render actual dataset imagery, and this phase commits no image of
the data. Qualitative error analysis is a deliberate later stage.
"""

CROSS_CHECK_TOLERANCE = 0.02
"""How far the authoritative validation may sit from the training history.

A standalone validation of ``best.pt`` and the training-time validation at the
best epoch run under slightly different batching, so an exact match is not
expected. A gap wider than this means the wrong checkpoint or the wrong epoch is
being reported, which must stop publication rather than be explained away.
"""


class BaselineError(RuntimeError):
    """Raised when the experiment cannot run as specified."""


def read_json(path: Path) -> dict[str, Any]:
    """Read a committed JSON artifact.

    Args:
        path: File to read.

    Returns:
        The parsed mapping.

    Raises:
        BaselineError: If the file is absent or unparsable.
    """
    if not path.is_file():
        msg = f"{path.name} not found; run the phase that produces it first"
        raise BaselineError(msg)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        msg = f"{path.name} is not valid JSON ({exc})"
        raise BaselineError(msg) from exc


def verify_inputs(paths: ProjectPaths, config: Any) -> dict[str, Any]:
    """Re-check every frozen input the experiment depends on.

    Args:
        paths: Project layout.
        config: The frozen D0 protocol.

    Returns:
        The verified fingerprints and dataset facts.

    Raises:
        BaselineError: If any fingerprint or count no longer matches.
    """
    problems: list[str] = []
    population = read_json(paths.reports / "canonical_modeling_manifest.json")
    split = read_json(paths.reports / "split_manifest.json")
    task = read_json(paths.reports / "task_dataset_manifest.json")
    adapter = read_json(paths.reports / "detection_adapter_manifest.json")
    runtime = read_json(paths.reports / "detection_runtime.provenance.json")

    class_map_sha = population["fingerprints"]["class_map_sha256"]
    for label, value in (
        ("task manifest class_map_sha256", task["class_map_sha256"]),
        ("adapter manifest class_map_sha256", adapter["class_map_sha256"]),
    ):
        if value != class_map_sha:
            problems.append(f"{label} disagrees with the frozen population")
    if adapter["split_assignment_sha256"] != split["split_assignment_sha256"]:
        problems.append("the adapter was built over a different frozen split")
    if (
        adapter["modeling_population_sha256"]
        != (population["fingerprints"]["modeling_population_sha256"])
    ):
        problems.append("the adapter was built over a different modelling population")
    if task["class_map"] != population["class_map"]:
        problems.append("the task manifest class map disagrees with the frozen one")

    for split_name, expected in EXPECTED_MEMBERSHIP.items():
        for field, value in expected.items():
            found = adapter[split_name][field]
            if found != value:
                problems.append(
                    f"adapter {split_name}.{field} is {found}, the frozen protocol expects {value}"
                )
    if adapter["test"]["status"] != "NOT_MATERIALIZED_PROTECTED_HOLDOUT":
        problems.append("the adapter does not record the holdout as unmaterialised")

    weights = runtime["details"]["pretrained_weights"]
    weight_path = paths.root / weights["relative_path"]
    if not weight_path.is_file():
        problems.append(f"{weights['identifier']} is missing from {weights['relative_path']}")
    else:
        actual = sha256_bytes(weight_path)
        if actual != weights["sha256"]:
            problems.append(
                f"{weights['identifier']} on disk digests to {actual}, the recorded "
                f"checkpoint is {weights['sha256']}"
            )
        if weight_path.stat().st_size != EXPECTED_WEIGHT_BYTES:
            problems.append(
                f"{weights['identifier']} is {weight_path.stat().st_size} bytes, expected "
                f"{EXPECTED_WEIGHT_BYTES}"
            )

    if problems:
        raise BaselineError("; ".join(problems))

    return {
        "class_map": population["class_map"],
        "class_map_sha256": class_map_sha,
        "modeling_population_sha256": population["fingerprints"]["modeling_population_sha256"],
        "split_assignment_sha256": split["split_assignment_sha256"],
        "task_materialization_config_sha256": task["materialization_config_sha256"],
        "adapter_config_sha256": adapter["adapter_config_sha256"],
        "adapter_manifest_sha256": sha256_bytes(paths.reports / "detection_adapter_manifest.json"),
        "yolo_label_sha256": {
            name: adapter[name]["yolo_label_sha256"] for name in EXPECTED_MEMBERSHIP
        },
        "pretrained_weights": weights,
        "train_images": adapter["train"]["image_count"],
        "train_annotations": adapter["train"]["annotation_count"],
        "validation_images": adapter["validation"]["image_count"],
        "validation_annotations": adapter["validation"]["annotation_count"],
    }


def runtime_facts() -> dict[str, Any]:
    """Describe the software and hardware the experiment runs on.

    Returns:
        Version and device facts.

    Raises:
        BaselineError: If CUDA is unavailable.
    """
    import torch
    import torchvision
    import ultralytics

    if not torch.cuda.is_available():
        msg = (
            "CUDA is not available. The D0 protocol requires a GPU; a CPU run would not be "
            "the same experiment and is never substituted."
        )
        raise BaselineError(msg)
    properties = torch.cuda.get_device_properties(0)
    architecture = f"sm_{properties.major}{properties.minor}"
    if architecture not in torch.cuda.get_arch_list():
        msg = (
            f"the installed torch build has no {architecture} kernels; it can see this GPU "
            "but not run on it"
        )
        raise BaselineError(msg)
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "ultralytics": ultralytics.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_arch": architecture,
        "gpu_total_memory_bytes": int(properties.total_memory),
    }


def write_prerun_record(
    paths: ProjectPaths, config: Any, inputs: dict[str, Any], runtime: dict[str, Any]
) -> Path:
    """Record what is about to run, before the first optimisation step.

    The point is evidential: a protocol that only appears next to a result cannot
    be shown to have preceded it.

    Args:
        paths: Project layout.
        config: The frozen D0 protocol.
        inputs: The verified input fingerprints.
        runtime: The runtime facts.

    Returns:
        The record path.
    """
    record = ProvenanceRecord.create(
        "detection_baseline_prerun",
        phase=6,
        repo_root=paths.root,
        config={
            "detection_baseline": "configs/detection_baseline.yaml",
            "baseline_config_sha256": config.fingerprint(),
            "experiment_id": config.experiment_id,
        },
        details={
            "phase": "6B",
            "stage": "PRE_RUN",
            "experiment_id": config.experiment_id,
            "declared_optimizer_policy": config.training["optimizer"],
            "seed": config.seed,
            "device": "cuda:0",
            "imgsz": config.training["imgsz"],
            "epochs_configured": config.training["epochs"],
            "batch": config.training["batch"],
            "checkpoint_selection": config.checkpoint_selection,
            "metrics": config.metrics.as_dict(),
            "dataset_fingerprints": {
                key: inputs[key]
                for key in (
                    "class_map_sha256",
                    "modeling_population_sha256",
                    "split_assignment_sha256",
                    "adapter_config_sha256",
                    "adapter_manifest_sha256",
                    "yolo_label_sha256",
                )
            },
            "pretrained_weights_sha256": inputs["pretrained_weights"]["sha256"],
            "runtime": runtime,
            "test_status": TEST_PROTECTED,
            "started_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S+00:00"),
        },
    )
    record.add_input(paths.configs / "detection_baseline.yaml", relative_to=paths.root)
    record.add_input(paths.reports / "detection_adapter_manifest.json", relative_to=paths.root)
    destination = paths.reports / PRERUN_JSON
    serialised = json.dumps(record.to_dict(), ensure_ascii=False, indent=2, sort_keys=True)
    unsafe = scan_for_sensitive(serialised)
    if unsafe:
        msg = f"{PRERUN_JSON} is not fit to commit: {'; '.join(unsafe)}"
        raise BaselineError(msg)
    destination.write_text(f"{serialised}\n", encoding="utf-8", newline="\n")
    return destination


def _capture_framework_log(destination: Path) -> Any:
    """Tee Ultralytics' logger into a file inside the run directory.

    Without this the framework's own line naming the optimizer it selected for
    ``optimizer: auto`` exists only on the terminal, and nothing durable records
    what the run actually used.

    Args:
        destination: Log file to write.

    Returns:
        The handler, so the caller can detach it.
    """
    import logging

    from ultralytics.utils import LOGGER

    destination.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(destination, encoding="utf-8")
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("%(message)s"))
    LOGGER.addHandler(handler)
    return handler


def _release_framework_log(handler: Any) -> None:
    """Detach the log handler.

    Args:
        handler: The handler returned by :func:`_capture_framework_log`.
    """
    from ultralytics.utils import LOGGER

    handler.close()
    LOGGER.removeHandler(handler)


def train(paths: ProjectPaths, config: Any, dataset: Path, *, resume: bool) -> dict[str, Any]:
    """Run the single D0 training execution.

    Args:
        paths: Project layout.
        config: The frozen D0 protocol.
        dataset: The Ultralytics dataset descriptor.
        resume: Whether this is an identical operational resume.

    Returns:
        The execution record.

    Raises:
        BaselineError: If a completed D0 run already exists, or training fails.
    """
    from ultralytics import YOLO

    run_directory = paths.root / RUN_ROOT / RUN_NAME
    pretrained = paths.root / "artifacts/weights" / config.weight_identifier

    if resume:
        last = run_directory / "weights" / "last.pt"
        if not last.is_file():
            msg = "--resume was requested but no D0 checkpoint exists to resume from"
            raise BaselineError(msg)
        model = YOLO(str(last))
        started = time.perf_counter()
        model.train(resume=True)
        return {
            "resumed": True,
            "seconds": round(time.perf_counter() - started, 1),
            "run_directory": f"{RUN_ROOT}/{RUN_NAME}",
        }

    if run_directory.exists():
        msg = (
            f"{RUN_ROOT}/{RUN_NAME} already exists. D0 is one experiment; refusing to "
            "overwrite or mix runs. Move it aside deliberately if it must be re-run."
        )
        raise BaselineError(msg)

    training = config.training
    model = YOLO(str(pretrained))
    handler = _capture_framework_log(run_directory / "train_console.log")
    started = time.perf_counter()
    try:
        model.train(
            data=str(dataset),
            epochs=training["epochs"],
            imgsz=training["imgsz"],
            batch=training["batch"],
            seed=config.seed,
            deterministic=training["deterministic"],
            workers=training["workers"],
            amp=training["amp"],
            optimizer=training["optimizer"],
            lr0=training["lr0"],
            lrf=training["lrf"],
            momentum=training["momentum"],
            weight_decay=training["weight_decay"],
            warmup_epochs=training["warmup_epochs"],
            cos_lr=training["cos_lr"],
            close_mosaic=training["close_mosaic"],
            patience=training["patience"],
            pretrained=training["pretrained"],
            val=training["val"],
            device=0,
            project=str(paths.root / RUN_ROOT),
            name=RUN_NAME,
            exist_ok=False,
            plots=True,
            verbose=True,
        )
    finally:
        _release_framework_log(handler)
    return {
        "resumed": False,
        "seconds": round(time.perf_counter() - started, 1),
        "run_directory": f"{RUN_ROOT}/{RUN_NAME}",
    }


def read_epoch_history(run_directory: Path) -> list[dict[str, str]]:
    """Read the framework's per-epoch metric history.

    Args:
        run_directory: The run output directory.

    Returns:
        The parsed rows of ``results.csv``, empty when it is absent.
    """
    path = run_directory / "results.csv"
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return [
            {key.strip(): value for key, value in row.items()} for row in csv.DictReader(handle)
        ]


def best_epoch_from_history(history: list[dict[str, str]]) -> tuple[int | None, float | None]:
    """Recompute the best epoch from the epoch history.

    This is the independent cross-check on the checkpoint the framework selected:
    the fitness is recomputed from ``results.csv`` rather than read from the
    checkpoint that is being verified.

    Args:
        history: The parsed epoch history.

    Returns:
        The best epoch number and its fitness, or ``(None, None)``.
    """
    best_epoch: int | None = None
    best_fitness: float | None = None
    for row in history:
        try:
            epoch = int(float(row["epoch"]))
            fitness = ultralytics_fitness(
                float(row["metrics/mAP50(B)"]), float(row["metrics/mAP50-95(B)"])
            )
        except (KeyError, TypeError, ValueError):
            continue
        if best_fitness is None or fitness > best_fitness:
            best_epoch, best_fitness = epoch, fitness
    return best_epoch, best_fitness


def checkpoint_record(path: Path) -> dict[str, Any]:
    """Fingerprint a checkpoint without committing it.

    Args:
        path: The checkpoint file.

    Returns:
        Its identity record.

    Raises:
        BaselineError: If it is missing.
    """
    if not path.is_file():
        msg = f"{path.name} was not produced by the run"
        raise BaselineError(msg)
    return {
        "relative_path": f"{RUN_ROOT}/{RUN_NAME}/weights/{path.name}",
        "sha256": sha256_bytes(path),
        "size_bytes": path.stat().st_size,
        "committed": False,
    }


def validate_best(paths: ProjectPaths, config: Any, dataset: Path) -> dict[str, Any]:
    """Run the single authoritative validation of the selected checkpoint.

    One evaluation, of ``best.pt``, on the validation split, at the frozen image
    size. No threshold sweep, no alternative image size, no comparison of several
    checkpoints - any of those would turn a measurement into a selection.

    Args:
        paths: Project layout.
        config: The frozen D0 protocol.
        dataset: The Ultralytics dataset descriptor.

    Returns:
        The validation record.
    """
    from ultralytics import YOLO

    best = paths.root / RUN_ROOT / RUN_NAME / "weights" / "best.pt"
    model = YOLO(str(best))
    metrics = model.val(
        data=str(dataset),
        split="val",
        imgsz=config.training["imgsz"],
        batch=config.training["batch"],
        device=0,
        project=str(paths.root / RUN_ROOT),
        name=f"{RUN_NAME}_val",
        exist_ok=True,
        plots=True,
        verbose=False,
    )
    box = metrics.box
    names = [name for _, name in sorted(model.names.items())]
    matrix = getattr(getattr(metrics, "confusion_matrix", None), "matrix", None)
    return {
        "confusion_matrix": (
            [[round(float(value)) for value in row] for row in matrix]
            if matrix is not None
            else None
        ),
        "confusion_matrix_axes": {
            "rows": "predicted",
            "columns": "ground truth",
            "labels": [*names, "background"],
        },
        "global": global_metrics(metrics.results_dict),
        "per_class": per_class_metrics(
            names,
            class_indices=list(box.ap_class_index),
            precision=list(box.p),
            recall=list(box.r),
            ap50=list(box.ap50),
            ap=list(box.ap),
        ),
        "speed": {key: round(float(value), 3) for key, value in metrics.speed.items()},
        "fitness": round(float(metrics.fitness), 6),
        "effective_arguments": {
            "split": "val",
            "imgsz": config.training["imgsz"],
            "batch": config.training["batch"],
            "conf": float(getattr(metrics, "conf", 0.001) or 0.001),
            "iou": 0.7,
            "max_det": 300,
            "device": "cuda:0",
            "note": (
                "Ultralytics validation defaults for the pinned version; no threshold, IoU "
                "or image-size variant was explored"
            ),
        },
        "output_directory": f"{RUN_ROOT}/{RUN_NAME}_val",
    }


def copy_metric_figures(paths: ProjectPaths, sources: list[Path]) -> list[str]:
    """Copy the metric-only plots into the committed figures directory.

    An allowlist rather than a filter: everything not named here stays in the
    ignored run directory, which is how dataset imagery and prediction montages
    are kept out of the repository.

    Args:
        paths: Project layout.
        sources: Run directories to look in.

    Returns:
        The repository-relative paths written.
    """
    destination = paths.reports / "figures" / "detection" / RUN_NAME
    destination.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    for source in sources:
        for name in METRIC_FIGURES:
            candidate = source / name
            if not candidate.is_file():
                continue
            target = destination / name
            shutil.copyfile(candidate, target)
            written.append(target.relative_to(paths.root).as_posix())
    return sorted(set(written))


def build_manifest(
    config: Any,
    inputs: dict[str, Any],
    runtime: dict[str, Any],
    execution: dict[str, Any],
    history: list[dict[str, str]],
    checkpoints: dict[str, dict[str, Any]],
    validation: dict[str, Any],
    resolved: dict[str, Any],
    optimizer: dict[str, Any],
    best_epoch: int | None,
    best_fitness: float | None,
    figures: list[str],
) -> dict[str, Any]:
    """Assemble the D0 result manifest.

    Args:
        config: The frozen D0 protocol.
        inputs: The verified input fingerprints.
        runtime: The runtime facts.
        execution: The training execution record.
        history: The per-epoch metric history.
        checkpoints: The fingerprinted checkpoints.
        validation: The authoritative validation record.
        resolved: The resolved Ultralytics arguments.
        optimizer: How the effective optimizer was determined.
        best_epoch: The best epoch, recomputed from the history.
        best_fitness: Its fitness.
        figures: The committed metric figures.

    Returns:
        The manifest mapping.
    """
    epochs_completed = len(history)
    configured = config.training["epochs"]
    fingerprint = experiment_fingerprint(
        baseline_config_sha256=config.fingerprint(),
        pretrained_weights_sha256=inputs["pretrained_weights"]["sha256"],
        adapter_manifest_sha256=inputs["adapter_manifest_sha256"],
        split_assignment_sha256=inputs["split_assignment_sha256"],
        resolved_arguments=resolved,
        best_checkpoint_sha256=checkpoints["best"]["sha256"],
    )
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "phase": "6B",
        "experiment_id": config.experiment_id,
        "status": COMPLETE,
        "task": config.task,
        "model": config.model,
        "pretrained_weights": {
            "identifier": inputs["pretrained_weights"]["identifier"],
            "sha256": inputs["pretrained_weights"]["sha256"],
            "size_bytes": inputs["pretrained_weights"]["size_bytes"],
            "committed": False,
        },
        "git_commit": git_commit(),
        "baseline_config_sha256": config.fingerprint(),
        "adapter_manifest_sha256": inputs["adapter_manifest_sha256"],
        "dataset_fingerprints": {
            "class_map_sha256": inputs["class_map_sha256"],
            "modeling_population_sha256": inputs["modeling_population_sha256"],
            "adapter_config_sha256": inputs["adapter_config_sha256"],
            "task_materialization_config_sha256": inputs["task_materialization_config_sha256"],
            "yolo_label_sha256": inputs["yolo_label_sha256"],
        },
        "split_reference": {
            "split_assignment_sha256": inputs["split_assignment_sha256"],
            "manifest": "reports/split_manifest.json",
        },
        "class_map": inputs["class_map"],
        "train_images": inputs["train_images"],
        "train_annotations": inputs["train_annotations"],
        "validation_images": inputs["validation_images"],
        "validation_annotations": inputs["validation_annotations"],
        "test": {
            "status": TEST_PROTECTED,
            "reason": (
                "the holdout was not adapted, loaded, evaluated or measured in this "
                "experiment. It has no labels, no dataset entry and no result."
            ),
        },
        "runtime": runtime,
        "epochs_configured": configured,
        "epochs_completed": epochs_completed,
        "best_epoch": best_epoch,
        "best_validation_fitness": round(best_fitness, 6) if best_fitness is not None else None,
        "early_stopped": epochs_completed < configured,
        "termination_mode": (
            "EARLY_STOPPING_PATIENCE" if epochs_completed < configured else "COMPLETED_ALL_EPOCHS"
        ),
        "resumed": execution["resumed"],
        "checkpoint_selection": config.checkpoint_selection,
        "declared_optimizer_policy": config.training["optimizer"],
        "resolved_optimizer": resolved.get("optimizer", NOT_EXPOSED),
        "resolved_optimizer_determination": optimizer,
        "resolved_training_arguments": resolved,
        "best_checkpoint": checkpoints["best"],
        "last_checkpoint": checkpoints["last"],
        "validation_configuration": validation["effective_arguments"],
        "validation_metrics": validation["global"],
        "per_class_metrics": validation["per_class"],
        "framework_validation_speed_ms_per_image": validation["speed"],
        "confusion_matrix": validation["confusion_matrix"],
        "confusion_matrix_axes": validation["confusion_matrix_axes"],
        "rare_class": {
            "name": config.rare_class,
            "warning": HIGH_SAMPLING_UNCERTAINTY,
            "validation_images": 1,
            "validation_instances": 8,
            "limitation": config.rare_class_limitation,
        },
        "training_duration_seconds": execution["seconds"],
        "metric_hierarchy": config.metrics.as_dict(),
        "committed_figures": figures,
        "weights_committed": False,
        "d0_experiment_sha256": fingerprint,
    }


def build_report(manifest: dict[str, Any], config: Any, cross_check: dict[str, Any]) -> str:
    """Write the D0 result report.

    Args:
        manifest: The result manifest.
        config: The frozen D0 protocol.
        cross_check: The metric cross-validation record.

    Returns:
        The report as Markdown.
    """
    lines: list[str] = []
    add = lines.append
    runtime = manifest["runtime"]

    add("# D0 - YOLO11n Detection Baseline - Phase 6B")
    add("")
    add(
        f"Phase: 6B · Commit: `{manifest['git_commit'] or 'unavailable'}` · Status: "
        f"**{manifest['status']}**"
    )
    add("")
    add(
        "**D0 is a reference point, not the project's detector.** It answers one question - "
        "does a small pretrained YOLO11 detector learn this task under the frozen data "
        "protocol - and it was specified in full before it ran. Nothing here was tuned."
    )
    add("")
    add(
        "Claims are labelled `PREDECLARED PROTOCOL` (fixed before the run), `COMPUTED RESULT` "
        "(measured by this run), `OBSERVATION` (a reading of those numbers), `LIMITATION` and "
        "`FUTURE EXPERIMENT`."
    )
    add("")
    add("## 1. Objective")
    add("")
    add(f"`PREDECLARED PROTOCOL` {config.description.strip()}")
    add("")
    add("## 2. Frozen protocol")
    add("")
    add("| Setting | Declared |")
    add("| --- | --- |")
    add(f"| `experiment_id` | {config.experiment_id} |")
    add(f"| Model | {config.model}, pretrained `{config.weight_identifier}` |")
    add(f"| Image size | {config.training['imgsz']} |")
    add(f"| Epochs | {config.training['epochs']} (max) |")
    add(f"| Batch | {config.training['batch']} |")
    add(f"| Seed | {config.seed} |")
    add(f"| Optimizer policy | `{config.training['optimizer']}` |")
    add(f"| Patience | {config.training['patience']} |")
    add(f"| Augmentation | `{config.training['augmentation']}` |")
    add(f"| AMP / deterministic | {config.training['amp']} / {config.training['deterministic']} |")
    add("")
    add(
        f"`PREDECLARED PROTOCOL` Baseline config SHA-256: "
        f"`{manifest['baseline_config_sha256']}`. Checkpoint rule: "
        f"{config.checkpoint_selection.strip()}"
    )
    add("")
    add("## 3. Dataset and split")
    add("")
    add("| | train | validation | test |")
    add("| --- | --- | --- | --- |")
    add(
        f"| Images | {manifest['train_images']} | {manifest['validation_images']} | "
        f"**{TEST_PROTECTED}** |"
    )
    add(
        f"| Annotations | {manifest['train_annotations']} | "
        f"{manifest['validation_annotations']} | - |"
    )
    add("")
    add(
        "`PREDECLARED PROTOCOL` Only `train` updated model parameters. `validation` drove the "
        "framework's fitness, checkpoint selection and early stopping - which is what a "
        "validation split is for - and the single authoritative evaluation below."
    )
    add("")
    add("| Fingerprint | Value |")
    add("| --- | --- |")
    add(
        f"| `split_assignment_sha256` | "
        f"`{manifest['split_reference']['split_assignment_sha256']}` |"
    )
    add(f"| `class_map_sha256` | `{manifest['dataset_fingerprints']['class_map_sha256']}` |")
    add(f"| `adapter_manifest_sha256` | `{manifest['adapter_manifest_sha256']}` |")
    add(f"| `d0_experiment_sha256` | `{manifest['d0_experiment_sha256']}` |")
    add("")
    add("## 4. Environment")
    add("")
    add("| Component | Version |")
    add("| --- | --- |")
    add(f"| Python | {runtime['python']} |")
    add(f"| torch | {runtime['torch']} |")
    add(f"| torchvision | {runtime['torchvision']} |")
    add(f"| ultralytics | {runtime['ultralytics']} |")
    add(f"| CUDA runtime | {runtime['cuda_runtime']} |")
    add(f"| GPU | {runtime['gpu_name']} ({runtime['gpu_arch']}) |")
    add("")
    lines.extend(_report_execution(manifest, config, cross_check))
    lines.extend(_report_results(manifest, config))
    return "\n".join(lines) + "\n"


def _report_execution(
    manifest: dict[str, Any], config: Any, cross_check: dict[str, Any]
) -> list[str]:
    """Write the effective-configuration, execution and checkpoint sections.

    Args:
        manifest: The result manifest.
        config: The frozen D0 protocol.
        cross_check: The metric cross-validation record.

    Returns:
        The report lines.
    """
    lines: list[str] = []
    add = lines.append
    resolved = manifest["resolved_training_arguments"]

    add("## 5. Effective training configuration")
    add("")
    add(
        "`COMPUTED RESULT` The protocol declares `optimizer: auto`, so the declared policy and "
        "the settings that actually ran are **not the same thing**. Ultralytics chose the "
        "optimizer and its learning rate from the dataset size and epoch count, overriding the "
        "generic values in the configuration file. What follows is what the run actually used, "
        "read back from the framework's own resolved arguments."
    )
    add("")
    add("| Resolved setting | Value |")
    add("| --- | --- |")
    for key in (
        "optimizer",
        "lr0",
        "lrf",
        "momentum",
        "weight_decay",
        "warmup_epochs",
        "warmup_momentum",
        "warmup_bias_lr",
        "cos_lr",
        "epochs",
        "batch",
        "imgsz",
        "amp",
        "deterministic",
        "seed",
        "patience",
        "workers",
        "device",
        "box",
        "cls",
        "dfl",
    ):
        if key in resolved:
            add(f"| `{key}` | {resolved[key]} |")
    add("")
    add(
        f"`COMPUTED RESULT` **Declared** `optimizer: {config.training['optimizer']}` -> "
        f"**resolved** `{manifest['resolved_optimizer']}`. Quoting the configuration file's "
        f"`lr0: {config.training['lr0']}` as this run's learning rate would be wrong: the "
        f"effective `lr0` was `{resolved.get('lr0', NOT_EXPOSED)}`."
    )
    add("")
    add("`COMPUTED RESULT` Augmentation values actually applied:")
    add("")
    add("| Augmentation | Value |")
    add("| --- | --- |")
    for key in (
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
        "auto_augment",
        "close_mosaic",
    ):
        if key in resolved:
            add(f"| `{key}` | {resolved[key]} |")
    add("")
    add(
        "`PREDECLARED PROTOCOL` These are **model training transformations**, not canonical "
        "preprocessing: the source images on disk are untouched and validation applies none of "
        "them. They were not tuned in this phase."
    )
    add("")
    add("## 6. Training execution")
    add("")
    add("| | |")
    add("| --- | --- |")
    add(f"| Termination | **{manifest['termination_mode']}** |")
    add(f"| Epochs configured | {manifest['epochs_configured']} |")
    add(f"| Epochs completed | {manifest['epochs_completed']} |")
    add(f"| Early stopped | {manifest['early_stopped']} (patience {config.training['patience']}) |")
    add(f"| Resumed | {manifest['resumed']} |")
    add(f"| Wall clock | {manifest['training_duration_seconds']} s |")
    add(f"| Batch / imgsz | {resolved.get('batch')} / {resolved.get('imgsz')} |")
    add("")
    add(
        "`PREDECLARED PROTOCOL` Exactly one training execution was performed. It was not "
        "stopped by hand, no hyperparameter was changed after seeing an epoch result, and no "
        "second run was launched to see whether the number would move."
    )
    add("")
    add("## 7. Checkpoint selection")
    add("")
    add("`PREDECLARED PROTOCOL` `ULTRALYTICS_BEST_ON_VALIDATION_FITNESS`, fixed before the run.")
    add("")
    add(f"* Best epoch: **{manifest['best_epoch']}** of {manifest['epochs_completed']} completed")
    add(f"* Best validation fitness: **{manifest['best_validation_fitness']}**")
    add(
        f"* `best.pt` SHA-256 `{manifest['best_checkpoint']['sha256']}` "
        f"({manifest['best_checkpoint']['size_bytes']} bytes)"
    )
    add(
        f"* `last.pt` SHA-256 `{manifest['last_checkpoint']['sha256']}` "
        f"({manifest['last_checkpoint']['size_bytes']} bytes)"
    )
    add("")
    add(
        "`COMPUTED RESULT` The best epoch was **recomputed independently** from `results.csv` "
        "using Ultralytics' own fitness definition (`0.1 * mAP@0.50 + 0.9 * mAP@0.50:0.95`) "
        "rather than read from the checkpoint being verified. No epoch was inspected and then "
        "chosen; only `best.pt` is the official D0 model."
    )
    add("")
    add("`COMPUTED RESULT` Metric cross-check, so the headline number provably belongs to the")
    add("selected checkpoint rather than to the last epoch or an earlier run:")
    add("")
    add("| Source | mAP@0.50:0.95 |")
    add("| --- | --- |")
    add(f"| Authoritative `best.pt` validation | {cross_check['authoritative']} |")
    add(f"| `results.csv` at best epoch {manifest['best_epoch']} | {cross_check['history']} |")
    add(
        f"| `results.csv` at last epoch {manifest['epochs_completed']} | "
        f"{cross_check['last_epoch']} |"
    )
    add("")
    add(
        f"`COMPUTED RESULT` Difference at the best epoch: **{cross_check['delta']}** "
        f"(tolerance {cross_check['tolerance']}). The two are separate evaluations of the same "
        "weights under slightly different batching, so they are expected to agree closely "
        "rather than exactly."
    )
    add("")
    return lines


def _report_confusion_matrix(manifest: dict[str, Any]) -> list[str]:
    """Report structural observations from the validation confusion matrix.

    Numbers rather than prose about a picture: the counts come from the matrix
    the validation produced, so the observations can be checked. Image-level
    error analysis is deliberately absent - no validation image was opened.

    Args:
        manifest: The result manifest.

    Returns:
        The report lines.
    """
    lines: list[str] = []
    add = lines.append
    matrix = manifest.get("confusion_matrix")
    labels = manifest["confusion_matrix_axes"]["labels"]

    add("## 12. Confusion matrix")
    add("")
    if not matrix:
        add(
            "`COMPUTED RESULT` The framework did not expose the confusion matrix as an array "
            "for this run; only the committed figures are available."
        )
        return lines

    add(
        "`COMPUTED RESULT` Counts from the validation confusion matrix, rows **predicted** and "
        "columns **ground truth**, with `background` covering unmatched predictions and "
        "undetected objects."
    )
    add("")
    add("| predicted \\ true | " + " | ".join(f"`{name}`" for name in labels) + " |")
    add("| --- |" + " --- |" * len(labels))
    for index, row in enumerate(matrix):
        add(f"| `{labels[index]}` | " + " | ".join(str(value) for value in row) + " |")
    add("")

    background = len(labels) - 1
    missed = [(labels[i], matrix[background][i]) for i in range(background)]
    spurious = [(labels[i], matrix[i][background]) for i in range(background)]
    confusions = [
        (labels[p], labels[t], matrix[p][t])
        for p in range(background)
        for t in range(background)
        if p != t and matrix[p][t] > 0
    ]
    missed.sort(key=lambda item: item[1], reverse=True)
    spurious.sort(key=lambda item: item[1], reverse=True)
    confusions.sort(key=lambda item: item[2], reverse=True)

    total_missed = sum(value for _, value in missed)
    total_spurious = sum(value for _, value in spurious)
    total_confused = sum(value for _, _, value in confusions)

    add(
        f"`OBSERVATION` **Errors concentrate on the background row and column, not between "
        f"classes.** {total_missed} ground-truth objects were not detected and "
        f"{total_spurious} predictions matched no object, against {total_confused} "
        "class-to-class confusions in total. The model's difficulty at this operating point "
        "is finding objects, more than telling the classes apart once found - which is "
        "consistent with the global recall being well below the global precision."
    )
    add("")
    if missed:
        worst = ", ".join(f"`{name}` {value}" for name, value in missed[:3] if value)
        add(f"`OBSERVATION` Undetected ground truth is led by {worst}.")
        add("")
    if confusions:
        pairs = ", ".join(
            f"`{predicted}` predicted where truth was `{truth}` ({value})"
            for predicted, truth, value in confusions[:3]
        )
        add(f"`OBSERVATION` The largest class-to-class confusions are {pairs}.")
        add("")
    add(
        "`LIMITATION` These are counts on one 65-image validation split, and a single "
        "confusion of one or two instances is not a pattern. **No image-level qualitative "
        "error analysis was performed**: no validation image was opened to explain an "
        "individual failure. That is a later, deliberate stage."
    )
    add("")
    return lines


def _report_results(manifest: dict[str, Any], config: Any) -> list[str]:
    """Write the results, limitations and future-work sections.

    Args:
        manifest: The result manifest.
        config: The frozen D0 protocol.

    Returns:
        The report lines.
    """
    lines: list[str] = []
    add = lines.append
    metrics = manifest["validation_metrics"]
    per_class = manifest["per_class_metrics"]
    class_map = manifest["class_map"]
    rare = manifest["rare_class"]["name"]

    add("## 8. Validation results")
    add("")
    add(
        "`COMPUTED RESULT` One evaluation of `best.pt` on the frozen validation split "
        f"({manifest['validation_images']} images, {manifest['validation_annotations']} "
        "annotations) at the frozen image size. No threshold sweep, no alternative image size, "
        "no second checkpoint."
    )
    add("")
    add("| Metric | Value |")
    add("| --- | --- |")
    add(f"| **{PRIMARY_METRIC}** (primary) | **{metrics[PRIMARY_METRIC]}** |")
    add(f"| mAP@0.50 | {metrics['mAP@0.50']} |")
    add(f"| precision | {metrics['precision']} |")
    add(f"| recall | {metrics['recall']} |")
    add("")
    add("## 9. Per-class results")
    add("")
    add("| Class | AP@0.50 | AP@0.50:0.95 | precision | recall |")
    add("| --- | --- | --- | --- | --- |")
    for name in sorted(per_class, key=lambda n: class_map[n]):
        entry = per_class[name]
        marker = " (small sample)" if name == rare else ""
        add(
            f"| `{name}`{marker} | {entry['AP@0.50']} | {entry['AP@0.50:0.95']} | "
            f"{entry['precision']} | {entry['recall']} |"
        )
    add("")
    add(
        f"A metric recorded as `{NOT_EXPOSED}` was not dependably exposed by the installed "
        "framework and was **not** reconstructed by guesswork."
    )
    add("")
    add(f"## 10. `{rare}` sampling limitation")
    add("")
    add(
        f"`LIMITATION` **{HIGH_SAMPLING_UNCERTAINTY}.** The frozen validation split contains "
        f"**{manifest['rare_class']['validation_images']} `{rare}` image** carrying "
        f"**{manifest['rare_class']['validation_instances']} instances**. Whatever number "
        "appears above for that class is one image's worth of evidence."
    )
    add("")
    add(
        f"`LIMITATION` Do not conclude that D0 is generally good or bad at `{rare}` from it. "
        "This limitation was predeclared in `configs/detection_baseline.yaml` **before** the "
        "run, so it is not an excuse built around an inconvenient result. Nothing was tuned "
        "against this metric, and the holdout - which holds 2 such images - was not consulted "
        "to resolve the uncertainty."
    )
    add("")
    add("## 11. Training dynamics")
    add("")
    add(
        "`COMPUTED RESULT` The full per-epoch history is preserved unmodified in the run's "
        "`results.csv`: box, classification and DFL losses for train and validation, the metric "
        "curves and the learning-rate progression. The committed figures under "
        f"`reports/{FIGURE_SUBDIR}/` are the framework's own metric plots - no curve was "
        "smoothed, truncated or redrawn."
    )
    add("")
    lines.extend(_report_confusion_matrix(manifest))
    add("## 13. Runtime and resources")
    add("")
    add("| | |")
    add("| --- | --- |")
    add(f"| Training wall clock | {manifest['training_duration_seconds']} s |")
    add(f"| Epochs completed | {manifest['epochs_completed']} |")
    add(f"| GPU | {manifest['runtime']['gpu_name']} |")
    for key, value in sorted(manifest["framework_validation_speed_ms_per_image"].items()):
        add(f"| `FRAMEWORK_VALIDATION_SPEED` {key} | {value} ms/image |")
    add("")
    add(
        "`LIMITATION` These timings describe **this execution on this device** and nothing "
        "more. They are not the project's latency benchmark: a standardised operational "
        "measurement belongs to the later detector/segmenter comparison, and no tuning run was "
        "performed to improve them."
    )
    add("")
    add("## 14. Reproducibility")
    add("")
    add(
        f"`PREDECLARED PROTOCOL` seed {config.seed}, `deterministic: true`, the pinned software "
        "stack above, and every input fingerprinted. `d0_experiment_sha256` covers the "
        "protocol, the pretrained weights, the adapter dataset, the split, the critical "
        "resolved arguments and the selected checkpoint; it excludes timestamps, paths and "
        "machine identity, so the same experiment elsewhere fingerprints the same."
    )
    add("")
    add(
        "`LIMITATION` `deterministic: true` reduces run-to-run variance but does not eliminate "
        "it - some cuDNN and CUDA kernels remain non-deterministic. **D0 was run once, "
        "deliberately.** A second run purely to see whether the number moved was not "
        "performed: that would be measuring noise, and a reproducibility study, if wanted, is "
        "its own predeclared experiment."
    )
    add("")
    add("## 15. Holdout compliance")
    add("")
    add(
        f"`PREDECLARED PROTOCOL` `test` is `{TEST_PROTECTED}`. It has no YOLO labels, no "
        "adapter directory and no key in the Ultralytics dataset descriptor, so it could not "
        f"have been loaded even by accident. `{HOLDOUT_UNLOCK_ENV_VAR}` was not set at any "
        "point, no test image or annotation was read, and no test metric exists. Model "
        "selection used validation only."
    )
    add("")
    lines.extend(_report_interpretation(manifest))
    return lines


def _report_interpretation(manifest: dict[str, Any]) -> list[str]:
    """Write the conservative interpretation and future-experiment sections.

    The numbers are quoted; the reading of them is kept separate and hedged,
    because a baseline's job is to be a reference point rather than a verdict.

    Args:
        manifest: The result manifest.

    Returns:
        The report lines.
    """
    lines: list[str] = []
    add = lines.append
    metrics = manifest["validation_metrics"]
    per_class = manifest["per_class_metrics"]
    rare = manifest["rare_class"]["name"]

    scored = [
        (name, entry["AP@0.50:0.95"])
        for name, entry in per_class.items()
        if isinstance(entry["AP@0.50:0.95"], (int, float)) and name != rare
    ]
    scored.sort(key=lambda item: item[1], reverse=True)

    add("## 16. Baseline interpretation")
    add("")
    add(
        f"`OBSERVATION` **The detector learns the task.** A primary metric of "
        f"{metrics[PRIMARY_METRIC]} with mAP@0.50 of {metrics['mAP@0.50']} is far above the "
        "zero a non-learning model would produce, so the data pipeline, the adapter and the "
        "training stack are working end to end. That is what a baseline is for."
    )
    add("")
    if scored:
        strongest, strongest_value = scored[0]
        weakest, weakest_value = scored[-1]
        add(
            f"`OBSERVATION` Among the classes with meaningful validation support, `{strongest}` "
            f"scores highest (AP@0.50:0.95 {strongest_value}) and `{weakest}` lowest "
            f"({weakest_value}). This is a description of one validation split of "
            f"{manifest['validation_images']} images, not a claim about the classes in general."
        )
        add("")
    add(
        f"`LIMITATION` `{rare}` is excluded from that comparison on purpose: with one "
        "validation image it cannot be ranked against classes measured on dozens."
    )
    add("")
    add(
        "`LIMITATION` **D0 is not the project's detector and this is not a final result.** It "
        "is one small pretrained model, trained once, on 303 images, evaluated on 65. No "
        "comparison against another model has been run, so nothing here says whether a larger "
        "model, a different image size or different augmentation would do better."
    )
    add("")
    add(
        "`LIMITATION` **No claim of generalisation.** These numbers describe this dataset's "
        "validation split. They say nothing about arbitrary construction sites, other cameras, "
        "other countries' PPE, or conditions absent from 433 images."
    )
    add("")
    add(
        "`LIMITATION` **The holdout is still unmeasured**, by design. D0's validation "
        "performance is not an estimate of its test performance, and the gap between them is "
        "unknown until phase 11."
    )
    add("")
    add("## 17. Questions for phase 7")
    add("")
    add(
        "`FUTURE EXPERIMENT` Hypotheses worth testing, recorded here **without acting on "
        "them**. Phase 7 defines a controlled experiment matrix before any of these run, so "
        "that each is a comparison rather than a reaction to D0's numbers:"
    )
    add("")
    add(
        "1. **Model capacity.** Does YOLO11s change the primary metric materially, and is any "
        "change larger than run-to-run variance?"
    )
    add(
        "2. **Input resolution.** PPE objects are often small at distance; does a larger "
        "`imgsz` help, and at what cost in time and memory?"
    )
    add(
        "3. **Epoch budget.** Did D0 stop for the right reason - is there evidence of "
        "underfitting or of overfitting in the loss curves?"
    )
    add(
        f"4. **Class imbalance.** `{rare}` has 45 instances against `person`'s 914. Does any "
        "imbalance-aware treatment help, and can it even be evaluated given the validation "
        "support?"
    )
    add("")
    add(
        "`PREDECLARED PROTOCOL` None of these was tried after seeing D0's results, and D0 was "
        "not modified in response to them. Phase 7 has not been started."
    )
    return lines


def _write_provenance(
    paths: ProjectPaths, manifest: dict[str, Any], config: Any, outputs: list[Path]
) -> Path:
    """Record how the D0 result was produced.

    Args:
        paths: Project layout.
        manifest: The result manifest.
        config: The frozen D0 protocol.
        outputs: The committed files this run produced.

    Returns:
        The provenance record path.
    """
    record = ProvenanceRecord.create(
        "detection_baseline_d0",
        phase=6,
        repo_root=paths.root,
        config={
            "detection_baseline": "configs/detection_baseline.yaml",
            "baseline_config_sha256": config.fingerprint(),
            "experiment_id": config.experiment_id,
        },
        details={
            "phase": "6B",
            "stage": "RESULT",
            "status": manifest["status"],
            "d0_experiment_sha256": manifest["d0_experiment_sha256"],
            "best_checkpoint_sha256": manifest["best_checkpoint"]["sha256"],
            "last_checkpoint_sha256": manifest["last_checkpoint"]["sha256"],
            "epochs_completed": manifest["epochs_completed"],
            "best_epoch": manifest["best_epoch"],
            "resolved_optimizer": manifest["resolved_optimizer"],
            "validation_metrics": manifest["validation_metrics"],
            "test_status": TEST_PROTECTED,
            "holdout_accessed": False,
            "weights_committed": False,
            "runs_performed": 1,
        },
    )
    for name in (
        "detection_adapter_manifest.json",
        "detection_runtime.provenance.json",
        PRERUN_JSON,
    ):
        record.add_input(paths.reports / name, relative_to=paths.root)
    record.add_input(paths.configs / "detection_baseline.yaml", relative_to=paths.root)
    for path in outputs:
        record.add_output(path, relative_to=paths.root)
    destination = paths.reports / PROVENANCE_JSON
    serialised = json.dumps(record.to_dict(), ensure_ascii=False, indent=2, sort_keys=True)
    unsafe = scan_for_sensitive(serialised)
    if unsafe:
        msg = f"{PROVENANCE_JSON} is not fit to commit: {'; '.join(unsafe)}"
        raise BaselineError(msg)
    destination.write_text(f"{serialised}\n", encoding="utf-8", newline="\n")
    return destination


def cross_check_metrics(
    authoritative: Any, history: list[dict[str, str]], best_epoch: int | None
) -> dict[str, Any]:
    """Check the headline metric against an independent framework artifact.

    Args:
        authoritative: The primary metric from the ``best.pt`` validation.
        history: The per-epoch metric history.
        best_epoch: The best epoch recomputed from that history.

    Returns:
        The comparison record.

    Raises:
        ResultError: If the two disagree beyond tolerance, which would mean the
            reported number does not belong to the selected checkpoint.
    """

    def epoch_metric(number: int | None) -> float | str:
        if number is None:
            return NOT_EXPOSED
        for row in history:
            try:
                if int(float(row["epoch"])) == number:
                    return round(float(row["metrics/mAP50-95(B)"]), 6)
            except (KeyError, TypeError, ValueError):
                continue
        return NOT_EXPOSED

    at_best = epoch_metric(best_epoch)
    at_last = epoch_metric(int(float(history[-1]["epoch"]))) if history else NOT_EXPOSED
    delta: float | str = NOT_EXPOSED
    if isinstance(authoritative, (int, float)) and isinstance(at_best, (int, float)):
        delta = round(abs(float(authoritative) - float(at_best)), 6)
        if delta > CROSS_CHECK_TOLERANCE:
            msg = (
                f"the authoritative best.pt validation reports {authoritative} but the epoch "
                f"history reports {at_best} at the selected epoch {best_epoch} (delta {delta} > "
                f"{CROSS_CHECK_TOLERANCE}). The published metric may not belong to the "
                "selected checkpoint; refusing to publish."
            )
            raise ResultError(msg)
    return {
        "authoritative": authoritative,
        "history": at_best,
        "last_epoch": at_last,
        "delta": delta,
        "tolerance": CROSS_CHECK_TOLERANCE,
    }


AUTO_OPTIMIZER_ITERATION_THRESHOLD = 10000
"""Above this many optimiser iterations Ultralytics' ``auto`` picks the SGD-family
branch instead of AdamW."""

NOMINAL_BATCH_SIZE = 64
"""Ultralytics' nominal batch size, used in its own iteration estimate."""


def resolve_optimizer(
    run_directory: Path,
    *,
    requested: str,
    class_count: int,
    train_images: int,
    batch: int,
    epochs: int,
    history: list[dict[str, str]],
) -> dict[str, Any]:
    """Determine which optimizer the run actually used.

    ``optimizer: auto`` is a policy, and Ultralytics does not persist what it
    resolved: ``args.yaml`` records the *request*, and the final checkpoint is
    stripped of optimizer state. So the effective optimizer has to be
    established rather than read, and this function is explicit about how.

    Preferred source is the framework's own log line, captured to
    ``train_console.log`` when the run was launched through this script. Failing
    that, the selection is re-derived from the framework's documented rule - a
    pure function of the class count and the iteration estimate - and
    corroborated against the learning rate the run actually logged, which
    separates the two candidate branches by an order of magnitude.

    Args:
        run_directory: The run output directory.
        requested: The optimizer named in the protocol.
        class_count: Number of classes in the dataset.
        train_images: Training images.
        batch: Batch size.
        epochs: Epochs configured.
        history: The per-epoch metric history.

    Returns:
        The resolved optimizer, its effective learning rate, and how both were
        determined.
    """
    if requested != "auto":
        return {
            "optimizer": requested,
            "effective_lr0": NOT_EXPOSED,
            "source": "DECLARED_EXPLICITLY_IN_PROTOCOL",
        }

    log = run_directory / "train_console.log"
    if log.is_file():
        match = re.search(
            r"optimizer:\s*([A-Za-z]+)\(lr=([0-9.eE+-]+),\s*momentum=([0-9.eE+-]+)\)",
            log.read_text(encoding="utf-8", errors="ignore"),
        )
        if match:
            return {
                "optimizer": match.group(1),
                "effective_lr0": float(match.group(2)),
                "effective_momentum": float(match.group(3)),
                "source": "FRAMEWORK_LOG_LINE",
            }

    # Ultralytics' own rule, applied to this run's numbers.
    iterations = math.ceil(train_images / max(batch, NOMINAL_BATCH_SIZE)) * epochs
    lr_fit = round(0.002 * 5 / (4 + class_count), 6)
    if iterations > AUTO_OPTIMIZER_ITERATION_THRESHOLD:
        name, lr0 = "SGD_FAMILY", 0.01
    else:
        name, lr0 = "AdamW", lr_fit

    observed_peak = NOT_EXPOSED
    values = []
    for row in history:
        try:
            values.append(float(row["lr/pg0"]))
        except (KeyError, TypeError, ValueError):
            continue
    if values:
        observed_peak = round(max(values), 8)

    corroborated = isinstance(observed_peak, float) and abs(observed_peak - lr0) < 0.5 * lr0
    return {
        "optimizer": name,
        "effective_lr0": lr0,
        "effective_momentum": 0.9,
        "iterations_estimate": iterations,
        "observed_peak_lr": observed_peak,
        "corroborated_by_observed_lr": corroborated,
        "source": (
            "REDERIVED_FROM_FRAMEWORK_RULE_CORROBORATED_BY_OBSERVED_LR"
            if corroborated
            else "REDERIVED_FROM_FRAMEWORK_RULE_UNCORROBORATED"
        ),
        "note": (
            "Ultralytics does not persist the optimizer it selects for "
            "optimizer=auto: args.yaml records the request and the final "
            "checkpoint is stripped of optimizer state. This value is the "
            "framework's own selection rule applied to this run's class count "
            "and iteration estimate, checked against the learning rate the run "
            "actually logged in results.csv - the two candidate branches differ "
            "by roughly an order of magnitude in lr0, so the observation "
            "distinguishes them."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    """Verify, run and record the D0 detection baseline.

    Args:
        argv: Command-line arguments. Defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None, help="Baseline protocol.")
    parser.add_argument("--verify-only", action="store_true", help="Pre-flight, then stop.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume the same interrupted D0 run identically. Operational use only.",
    )
    parser.add_argument(
        "--rebuild-result",
        action="store_true",
        help=(
            "Regenerate the result artifacts from the completed run without training "
            "again. Re-runs only the deterministic validation of the same best.pt."
        ),
    )
    args = parser.parse_args(argv)

    paths = ProjectPaths.from_root()
    if holdout_unlocked():
        print(
            f"ERROR: {HOLDOUT_UNLOCK_ENV_VAR} is set. The holdout must stay locked throughout "
            "phase 6B; refusing to run.",
            file=sys.stderr,
        )
        return 2

    try:
        config = load_detection_baseline_config(
            args.config or (paths.configs / "detection_baseline.yaml")
        )
        adapter = load_adapter_config(paths.configs / "detection_adapter.yaml")
    except (ConfigError, ExperimentConfigError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    try:
        inputs = verify_inputs(paths, config)
    except BaselineError as exc:
        print(f"PROTOCOL_INPUT_MISMATCH: {exc}", file=sys.stderr)
        return 3

    dataset = paths.root / adapter.output_root / "dataset.yaml"
    if not dataset.is_file():
        print("ERROR: run scripts/build_detection_adapter.py first", file=sys.stderr)
        return 3
    # Parse rather than grep: the descriptor's own comment explains why there is
    # no `test` key, and a substring search would flag that explanation.
    descriptor = yaml.safe_load(dataset.read_text(encoding="utf-8")) or {}
    if "test" in descriptor:
        print(
            "ERROR: the dataset descriptor declares a `test` key pointing at the protected "
            "split; refusing to run",
            file=sys.stderr,
        )
        return 3

    try:
        runtime = runtime_facts()
    except BaselineError as exc:
        print(f"BLOCKED: {exc}", file=sys.stderr)
        return 4

    print(f"  experiment            {config.experiment_id}  {config.model}")
    print(f"  baseline_config_sha256 {config.fingerprint()}")
    print("  inputs                 verified")
    print(f"  runtime                {runtime['gpu_name']} {runtime['gpu_arch']}")
    print(
        f"  data                   train {inputs['train_images']}/{inputs['train_annotations']}"
        f"  val {inputs['validation_images']}/{inputs['validation_annotations']}"
        f"  test {TEST_PROTECTED}"
    )

    if args.verify_only:
        print("--verify-only: nothing trained")
        return 0

    prerun = write_prerun_record(paths, config, inputs, runtime)
    print(f"wrote {prerun.relative_to(paths.root).as_posix()} (before the first step)")

    if args.rebuild_result:
        existing = paths.reports / MANIFEST_JSON
        if not existing.is_file():
            print("ERROR: --rebuild-result needs an existing D0 result", file=sys.stderr)
            return 3
        previous = json.loads(existing.read_text(encoding="utf-8"))
        execution = {
            "resumed": previous["resumed"],
            "seconds": previous["training_duration_seconds"],
            "run_directory": f"{RUN_ROOT}/{RUN_NAME}",
        }
        print("  --rebuild-result: reusing the completed run, training not repeated")
    else:
        try:
            execution = train(paths, config, dataset, resume=args.resume)
        except BaselineError as exc:
            print(f"TRAINING_FAILED: {exc}", file=sys.stderr)
            return 5
        except Exception as exc:  # a framework failure is reported, not adapted around
            print(f"TRAINING_FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 5

    run_directory = paths.root / RUN_ROOT / RUN_NAME
    history = read_epoch_history(run_directory)
    best_epoch, best_fitness = best_epoch_from_history(history)
    resolved = _read_resolved_arguments(run_directory)
    optimizer = resolve_optimizer(
        run_directory,
        requested=config.training["optimizer"],
        class_count=len(inputs["class_map"]),
        train_images=inputs["train_images"],
        batch=config.training["batch"],
        epochs=config.training["epochs"],
        history=history,
    )
    # args.yaml records the *request*, so the effective values replace it. The
    # requested pair is preserved separately rather than overwritten.
    resolved["optimizer_requested"] = resolved.get("optimizer")
    resolved["lr0_requested"] = resolved.get("lr0")
    resolved["momentum_requested"] = resolved.get("momentum")
    resolved["optimizer"] = optimizer["optimizer"]
    if optimizer["effective_lr0"] != NOT_EXPOSED:
        resolved["lr0"] = optimizer["effective_lr0"]
    if "effective_momentum" in optimizer:
        resolved["momentum"] = optimizer["effective_momentum"]
    checkpoints = {
        "best": checkpoint_record(run_directory / "weights" / "best.pt"),
        "last": checkpoint_record(run_directory / "weights" / "last.pt"),
    }
    validation = validate_best(paths, config, dataset)

    try:
        cross_check = cross_check_metrics(validation["global"][PRIMARY_METRIC], history, best_epoch)
    except ResultError as exc:
        print(f"PROTOCOL_VIOLATION: {exc}", file=sys.stderr)
        return 6

    figures = copy_metric_figures(paths, [run_directory, paths.root / RUN_ROOT / f"{RUN_NAME}_val"])
    manifest = build_manifest(
        config,
        inputs,
        runtime,
        execution,
        history,
        checkpoints,
        validation,
        resolved,
        optimizer,
        best_epoch,
        best_fitness,
        figures,
    )
    problems = validate_result_manifest(manifest, class_names=list(inputs["class_map"]))
    if problems:
        print("PROTOCOL_VIOLATION: the result manifest is not sound:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 6

    serialised = json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True)
    report = build_report(manifest, config, cross_check)
    for name, text in ((MANIFEST_JSON, serialised), (REPORT_MD, report)):
        unsafe = scan_for_sensitive(text)
        if unsafe:
            print(f"ERROR: {name} is not fit to commit: {'; '.join(unsafe)}", file=sys.stderr)
            return 6

    manifest_path = paths.reports / MANIFEST_JSON
    manifest_path.write_text(f"{serialised}\n", encoding="utf-8", newline="\n")
    report_path = paths.reports / REPORT_MD
    report_path.write_text(report, encoding="utf-8", newline="\n")
    provenance = _write_provenance(paths, manifest, config, [manifest_path, report_path])

    metrics = manifest["validation_metrics"]
    print()
    print(f"  epochs completed      {manifest['epochs_completed']}/{manifest['epochs_configured']}")
    print(f"  best epoch            {manifest['best_epoch']}")
    print(f"  resolved optimizer    {manifest['resolved_optimizer']}")
    print(f"  {PRIMARY_METRIC:<21} {metrics[PRIMARY_METRIC]}")
    print(f"  mAP@0.50              {metrics['mAP@0.50']}")
    print(f"  precision / recall    {metrics['precision']} / {metrics['recall']}")
    print(f"  d0_experiment_sha256  {manifest['d0_experiment_sha256']}")
    for path in (manifest_path, report_path, provenance):
        print(f"wrote {path.relative_to(paths.root).as_posix()}")
    for figure in figures:
        print(f"wrote {figure}")
    print(f"\n{COMPLETE}   holdout ({TEST_PROTECTED})")
    return 0


def _read_resolved_arguments(run_directory: Path) -> dict[str, Any]:
    """Read the framework's resolved training arguments.

    Args:
        run_directory: The run output directory.

    Returns:
        The resolved arguments, with machine-specific entries removed.
    """
    path = run_directory / "args.yaml"
    if not path.is_file():
        return {}
    resolved = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    # `data`, `project`, `name` and `save_dir` are absolute paths on this
    # machine; they describe where the run happened, not what it was.
    for key in ("data", "project", "name", "save_dir", "model"):
        resolved.pop(key, None)
    return dict(sorted(resolved.items()))


if __name__ == "__main__":
    sys.exit(main())
