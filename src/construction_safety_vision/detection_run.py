"""Shared primitives for executing one frozen detection experiment.

Phase 7 runs more than one experiment against the same protocol, so the parts
that must behave identically across runs live here rather than inside a
per-experiment script. If D1 and D2 fingerprinted their weights differently, or
established their optimizer differently, the comparison between them would be
measuring the bookkeeping as well as the model.

Three things in here are less obvious than they look.

**A pretrained checkpoint is identified by its bytes, not its name.** Two files
called ``yolo11s.pt`` are not necessarily the same weights, and an experiment
that cannot name the digest it started from cannot be reproduced.

**``optimizer: auto`` has to be established, not read.** Ultralytics records the
*request* in ``args.yaml`` and strips optimizer state from the final checkpoint,
so what actually ran survives only in the framework's own log line. That line is
colourised, so the escape codes must be removed before it can be parsed -
without that step the parse fails silently and the run falls back to inference,
which is a weaker kind of evidence for no reason. Direct capture is preferred
and inference is labelled as inference; see
:func:`optimizer_evidence_from_log` and :func:`rederive_auto_optimizer`.

**The committed figures are an allowlist, not a filter.** ``train_batch*.jpg``,
``val_batch*.jpg`` and ``labels.jpg`` render real dataset imagery and prediction
montages. Committing them would publish the dataset and pre-empt the deliberate
error-analysis phase, so only the plots named in :data:`METRIC_FIGURES` may
leave the ignored run directory.

.. note::
   ``scripts/train_detection_baseline.py`` (phase 6B) carries its own copies of
   several of these functions. It is deliberately not refactored onto this
   module: D0 is frozen and must never be re-run, so putting a newer
   implementation under its name would change nothing about its committed result
   while risking an untested change to a reported experiment. The duplication is
   a known cost, recorded here rather than hidden, and the two should converge in
   a later cleanup phase - not inside an experiment phase.
"""

from __future__ import annotations

import csv
import logging
import math
import re
import shutil
from pathlib import Path
from typing import Any

import yaml

from construction_safety_vision.data.materialization import sha256_bytes
from construction_safety_vision.detection_results import (
    NOT_EXPOSED,
    ResultError,
    ultralytics_fitness,
)
from construction_safety_vision.paths import ProjectPaths

ARTIFACTS_ROOT = "artifacts"
"""Git-ignored root for weights and run directories."""

WEIGHTS_SUBDIR = "weights"
"""Where pretrained checkpoints are kept, under :data:`ARTIFACTS_ROOT`."""

RUN_ROOT = "artifacts/detection"
"""Git-ignored parent of every detection run directory."""

WEIGHT_SOURCE_MECHANISM = "ULTRALYTICS_ASSET_DOWNLOAD"
"""How pretrained checkpoints are obtained, recorded identically for every run.

The same string the phase 6A runtime check recorded for ``yolo11n.pt``, so a
later comparison can show that D0 and its candidates started from checkpoints
obtained the same way.
"""

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
"""Metric-only plots that may be committed. An allowlist, not a filter."""

CROSS_CHECK_TOLERANCE = 0.02
"""How far the authoritative validation may sit from the training history.

A standalone validation of ``best.pt`` and the training-time validation at the
best epoch run under slightly different batching, so an exact match is not
expected. A wider gap means the wrong checkpoint or the wrong epoch is being
reported, which must stop publication rather than be explained away.
"""

AUTO_OPTIMIZER_ITERATION_THRESHOLD = 10000
"""Above this many optimiser iterations Ultralytics' ``auto`` leaves AdamW."""

NOMINAL_BATCH_SIZE = 64
"""Ultralytics' nominal batch size, used in its own iteration estimate."""

ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")
"""Colour escape sequences the framework wraps its log labels in."""

OPTIMIZER_LOG_PATTERN = re.compile(
    r"optimizer:\s*([A-Za-z_][A-Za-z0-9_]*)\(lr=([0-9.eE+-]+),\s*momentum=([0-9.eE+-]+)\)"
)
"""The framework's own line naming the optimizer it built.

Matched against ANSI-stripped text. The label is colourised in the real log, so
matching the raw bytes fails and the caller silently loses its best evidence.
"""

FRAMEWORK_LOG_LINE = "FRAMEWORK_LOG_LINE_DIRECT_CAPTURE"
"""Optimizer established by reading the framework's own log line."""

DECLARED_EXPLICITLY = "DECLARED_EXPLICITLY_IN_PROTOCOL"
"""Optimizer named outright by the protocol, so nothing had to be established."""

REDERIVED_CORROBORATED = "INFERRED_FROM_FRAMEWORK_RULE_CORROBORATED_BY_OBSERVED_LR"
"""Optimizer inferred from the framework's rule and checked against the run."""

REDERIVED_UNCORROBORATED = "INFERRED_FROM_FRAMEWORK_RULE_UNCORROBORATED"
"""Optimizer inferred with no observation available to check it against."""

TRAIN_CONSOLE_LOG = "train_console.log"
"""Filename the framework's log is teed into inside the run directory."""


class RunError(RuntimeError):
    """Raised when an experiment cannot execute as specified."""


# --- pretrained weights -------------------------------------------------------


def ensure_pretrained_weights(paths: ProjectPaths, identifier: str) -> dict[str, Any]:
    """Obtain a pretrained checkpoint and fingerprint it.

    Fetches through Ultralytics' own asset mechanism, then moves the file under
    ``artifacts/weights/`` so the project owns its location rather than
    depending on the working directory the run happened to start in.

    Args:
        paths: Project layout.
        identifier: Weight file name, e.g. ``yolo11s.pt``.

    Returns:
        The weight provenance record, never the bytes.

    Raises:
        RunError: If the checkpoint cannot be obtained.
    """
    destination = paths.root / ARTIFACTS_ROOT / WEIGHTS_SUBDIR / identifier
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.is_file():
        from ultralytics import YOLO

        model = YOLO(identifier)
        source = Path(getattr(model, "ckpt_path", "") or identifier)
        if not source.is_file():
            msg = f"Ultralytics did not leave {identifier} on disk where it could be recorded"
            raise RunError(msg)
        destination.write_bytes(source.read_bytes())
        if source.resolve() != destination.resolve():
            source.unlink(missing_ok=True)

    import ultralytics

    return {
        "identifier": identifier,
        "source_mechanism": WEIGHT_SOURCE_MECHANISM,
        "ultralytics_version": ultralytics.__version__,
        "relative_path": destination.relative_to(paths.root).as_posix(),
        "sha256": sha256_bytes(destination),
        "size_bytes": destination.stat().st_size,
        "committed": False,
    }


# --- runtime ------------------------------------------------------------------


def runtime_facts() -> dict[str, Any]:
    """Describe the software and hardware the experiment runs on.

    Returns:
        Version and device facts.

    Raises:
        RunError: If CUDA is unavailable, or the installed torch build has no
            kernels for this device. Either is a blocked experiment rather than
            a reason to fall back to CPU, which would not be the same run.
    """
    import platform

    import torch
    import torchvision
    import ultralytics

    if not torch.cuda.is_available():
        msg = (
            "CUDA is not available. This protocol requires a GPU; a CPU run would not be the "
            "same experiment and is never substituted."
        )
        raise RunError(msg)
    properties = torch.cuda.get_device_properties(0)
    architecture = f"sm_{properties.major}{properties.minor}"
    if architecture not in torch.cuda.get_arch_list():
        msg = (
            f"the installed torch build has no {architecture} kernels; it can see this GPU "
            "but not run on it"
        )
        raise RunError(msg)
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


def capture_framework_log(destination: Path) -> logging.Handler:
    """Tee the framework's logger into a file inside the run directory.

    Without this the line naming the optimizer selected for ``optimizer: auto``
    exists only on the terminal, and nothing durable records what ran.

    Args:
        destination: Log file to write.

    Returns:
        The handler, so the caller can detach it.
    """
    from ultralytics.utils import LOGGER

    destination.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(destination, encoding="utf-8")
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("%(message)s"))
    LOGGER.addHandler(handler)
    return handler


def release_framework_log(handler: logging.Handler) -> None:
    """Detach the log handler.

    Args:
        handler: The handler returned by :func:`capture_framework_log`.
    """
    from ultralytics.utils import LOGGER

    handler.close()
    LOGGER.removeHandler(handler)


# --- optimizer evidence -------------------------------------------------------


def strip_ansi(text: str) -> str:
    """Remove colour escape sequences from framework log text.

    Args:
        text: Raw log text.

    Returns:
        The same text without ANSI escapes.
    """
    return ANSI_ESCAPE.sub("", text)


def optimizer_evidence_from_log(log_text: str) -> dict[str, Any] | None:
    """Read the optimizer the framework actually built, from its own log.

    This is the strongest available evidence: the framework naming its own
    choice, at the moment it made it. Everything else is inference.

    Args:
        log_text: Contents of the captured training log.

    Returns:
        The optimizer, its effective learning rate and momentum, or ``None``
        when the log carries no such line.
    """
    match = OPTIMIZER_LOG_PATTERN.search(strip_ansi(log_text))
    if match is None:
        return None
    return {
        "optimizer": match.group(1),
        "effective_lr0": float(match.group(2)),
        "effective_momentum": float(match.group(3)),
        "source": FRAMEWORK_LOG_LINE,
        "evidence": strip_ansi(match.group(0)).strip(),
        "inferred": False,
    }


def rederive_auto_optimizer(
    *,
    class_count: int,
    train_images: int,
    batch: int,
    epochs: int,
    history: list[dict[str, str]],
    ultralytics_version: str,
) -> dict[str, Any]:
    """Infer which optimizer ``auto`` selected, labelled as an inference.

    The fallback for a run whose log line is unavailable. It applies the
    framework's own documented rule to this run's numbers and checks the answer
    against the learning rate the run logged - the two candidate branches differ
    by roughly an order of magnitude in ``lr0``, so the observation distinguishes
    them.

    Args:
        class_count: Number of classes in the dataset.
        train_images: Training images.
        batch: Batch size.
        epochs: Epochs configured.
        history: The per-epoch metric history.
        ultralytics_version: Installed framework version, recorded because the
            rule and the name of its high-iteration branch are version-specific.

    Returns:
        The inferred optimizer and the evidence behind the inference.
    """
    iterations = math.ceil(train_images / max(batch, NOMINAL_BATCH_SIZE)) * epochs
    lr_fit = round(0.002 * 5 / (4 + class_count), 6)
    if iterations > AUTO_OPTIMIZER_ITERATION_THRESHOLD:
        name, lr0 = "MuSGD", 0.01
    else:
        name, lr0 = "AdamW", lr_fit

    observed_peak: float | str = NOT_EXPOSED
    values: list[float] = []
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
        "source": REDERIVED_CORROBORATED if corroborated else REDERIVED_UNCORROBORATED,
        "inferred": True,
        "ultralytics_version": ultralytics_version,
        "note": (
            "Ultralytics does not persist the optimizer it selects for optimizer=auto: "
            "args.yaml records the request and the final checkpoint is stripped of optimizer "
            "state. This value is the framework's own selection rule applied to this run's "
            "class count and iteration estimate, checked against the learning rate the run "
            "logged in results.csv. It is an inference, not a reading, and the branch names "
            "are specific to the recorded framework version."
        ),
    }


def optimizer_evidence(
    run_directory: Path,
    *,
    requested: str,
    class_count: int,
    train_images: int,
    batch: int,
    epochs: int,
    history: list[dict[str, str]],
    ultralytics_version: str,
) -> dict[str, Any]:
    """Establish what optimizer the run used, preferring direct capture.

    Args:
        run_directory: The run output directory.
        requested: The optimizer named in the protocol.
        class_count: Number of classes in the dataset.
        train_images: Training images.
        batch: Batch size.
        epochs: Epochs configured.
        history: The per-epoch metric history.
        ultralytics_version: Installed framework version.

    Returns:
        The resolved optimizer and how it was established.
    """
    if requested != "auto":
        return {
            "optimizer": requested,
            "effective_lr0": NOT_EXPOSED,
            "source": DECLARED_EXPLICITLY,
            "inferred": False,
        }
    log = run_directory / TRAIN_CONSOLE_LOG
    if log.is_file():
        direct = optimizer_evidence_from_log(log.read_text(encoding="utf-8", errors="ignore"))
        if direct is not None:
            direct["log"] = f"{run_directory.name}/{TRAIN_CONSOLE_LOG}"
            return direct
    return rederive_auto_optimizer(
        class_count=class_count,
        train_images=train_images,
        batch=batch,
        epochs=epochs,
        history=history,
        ultralytics_version=ultralytics_version,
    )


# --- run artifacts ------------------------------------------------------------


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

    The independent cross-check on the checkpoint the framework selected: the
    fitness is recomputed from ``results.csv`` rather than read out of the
    checkpoint being verified.

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


def checkpoint_record(path: Path, *, run_name: str) -> dict[str, Any]:
    """Fingerprint a checkpoint without committing it.

    Args:
        path: The checkpoint file.
        run_name: Run directory name, used to build the recorded path.

    Returns:
        Its identity record.

    Raises:
        RunError: If it is missing.
    """
    if not path.is_file():
        msg = f"{path.name} was not produced by the run"
        raise RunError(msg)
    return {
        "relative_path": f"{RUN_ROOT}/{run_name}/weights/{path.name}",
        "sha256": sha256_bytes(path),
        "size_bytes": path.stat().st_size,
        "committed": False,
    }


def resolved_training_arguments(run_directory: Path) -> dict[str, Any]:
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
    # `data`, `project`, `name`, `save_dir` and `model` are absolute paths on
    # this machine; they describe where the run happened, not what it was.
    for key in ("data", "project", "name", "save_dir", "model"):
        resolved.pop(key, None)
    return dict(sorted(resolved.items()))


def copy_metric_figures(paths: ProjectPaths, sources: list[Path], *, run_name: str) -> list[str]:
    """Copy the metric-only plots into the committed figures directory.

    Args:
        paths: Project layout.
        sources: Run directories to look in.
        run_name: Experiment id, used as the figures subdirectory.

    Returns:
        The repository-relative paths written.
    """
    destination = paths.reports / "figures" / "detection" / run_name
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


def model_complexity(model: Any) -> dict[str, Any]:
    """Describe a model's size, when the framework exposes it reliably.

    Relevant here because D1 varies capacity: parameter count and FLOPs are
    descriptive facts about what the intentional variable actually changed. They
    are recorded, not used to choose a winner.

    Args:
        model: A loaded Ultralytics model.

    Returns:
        Parameter count, layer count and GFLOPs where available, with
        :data:`NOT_EXPOSED` where not.
    """
    facts: dict[str, Any] = {
        "parameters": NOT_EXPOSED,
        "layers": NOT_EXPOSED,
        "gflops": NOT_EXPOSED,
        "source": NOT_EXPOSED,
    }
    try:
        from ultralytics.utils.torch_utils import get_flops, get_num_params

        underlying = getattr(model, "model", model)
        parameters = get_num_params(underlying)
        if isinstance(parameters, int) and parameters > 0:
            facts["parameters"] = parameters
        flops = get_flops(underlying)
        if isinstance(flops, (int, float)) and flops > 0:
            facts["gflops"] = round(float(flops), 3)
        layers = sum(1 for _ in underlying.modules())
        facts["layers"] = layers
        facts["source"] = "ULTRALYTICS_TORCH_UTILS"
    except Exception:  # descriptive only; never fail a run over a nice-to-have fact
        facts["source"] = NOT_EXPOSED
    return facts
