"""Benchmark the frozen detector and segmenter for latency and inference memory.

Phase 10C. It executes only the cost half of the comparison phase 10A froze:
two timing boundaries over a deterministic 20-image validation subset, and a
controlled inference-memory measurement. It trains nothing, evaluates no
average precision, reruns no spatial or association analysis, tunes no
threshold and never touches the holdout.

Six things here are deliberate.

**The execution order is derived from the frozen membership, not written out.**
Pass A times the detector then the segmenter on each benchmark image in the
frozen order; pass B reverses the two models over the same images in the same
order. Both passes feed the reported distribution, so residual thermal or
ordering drift falls on both models rather than on whichever ran second.

**Both boundaries are measured through the same framework calls.** The
framework's own prediction pipeline is ``preprocess`` then ``inference`` then
``postprocess``; the detector and the segmenter differ only in which
``postprocess`` override runs, and the segmenter's reconstructs its masks on
the original canvas inside that call. ``MODEL_INFERENCE_LATENCY_MS`` times
``inference`` on a tensor prepared outside the timer;
``END_TO_END_MODEL_OUTPUT_LATENCY_MS`` times all three.

**Every timed region is synchronised on both edges.** CUDA work is
asynchronous, so an unsynchronised wall-clock reading measures how long it took
to queue the work rather than to run it, and the faster-to-queue model would
appear faster.

**Image decode is hoisted out, identically for both models.** Each benchmark
image is decoded once, with the framework's own reader, and the same decoded
array is handed to both models. Disk-cache variation cannot land on one model
and not the other.

**Memory is measured in a dedicated process per model.** Peak CUDA statistics
are device-global, and releasing a model in-process still leaves a cuBLAS
workspace allocated that the next model measured would be charged for.

**One confidence had to be chosen, and it is chosen here rather than after the
fact.** ``latency_protocol`` freezes batch, resolution, precision, warmup,
repetitions, membership and order, but no confidence threshold. The
operational block is the only frozen operating point - the AP block's 0.001 is
declared in the protocol itself to be deliberately not one - so the benchmark
runs at the operational 0.25 and says so in every artifact. Latency at 0.001
was not measured and is not claimed.

Requires:

* ``configs/detector_segmenter_comparison.yaml``               phase 10A
* ``reports/detector_segmenter_comparison_protocol.json``      phase 10A
* ``reports/detector_segmenter_latency_membership.csv``        phase 10A
* ``reports/final_detector_manifest.json``                     phase 7D
* ``reports/final_segmenter_manifest.json``                    phase 8G
* ``data/processed/canonical/`` detection + images              phase 5D

Writes:
    reports/detector_segmenter_latency_comparison.json
    reports/detector_segmenter_memory_comparison.json
    reports/detector_segmenter_latency_blocks.csv
    reports/detector_segmenter_latency_report.md
    reports/detector_segmenter_cost_benchmark.provenance.json
    artifacts/benchmark/                                       (git-ignored)

Usage:
    uv run python scripts/benchmark_detector_segmenter.py
    uv run python scripts/benchmark_detector_segmenter.py --verify-only
    uv run python scripts/benchmark_detector_segmenter.py --preflight-only
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from construction_safety_vision.config import ConfigError
from construction_safety_vision.data.canonical import scan_for_sensitive
from construction_safety_vision.detection_freeze import (
    FinalDetectorError,
    load_final_detector,
)
from construction_safety_vision.detection_freeze import (
    load_checkpoint_path as detector_checkpoint,
)
from construction_safety_vision.detector_segmenter_analysis import METRIC_PRECISION
from construction_safety_vision.detector_segmenter_comparison import (
    BENCHMARK_LABEL,
    COMPARISON_IMGSZ,
    DETECTOR_EXPERIMENT,
    END_TO_END_LATENCY,
    EVALUATION_SPLIT,
    HOLDOUT_STATUS,
    LATENCY_BATCH,
    MODEL_INFERENCE_LATENCY,
    PRECISION,
    PRECISION_ARGUMENT,
    PRECISION_VALUE,
    SEGMENTER_EXPERIMENT,
    SELECTION_RULE,
    TIMED_ITERATIONS_PER_IMAGE,
    WARMUP_ITERATIONS,
    ComparisonProtocolError,
    load_comparison_protocol,
    membership_fingerprint,
    ordered_fingerprint,
    select_benchmark_images,
    stable_rank,
    validate_protocol_manifest,
)
from construction_safety_vision.detector_segmenter_cost import (
    BENCHMARK_EXECUTION_FAILED,
    BLOCK_FIELDS,
    BLOCKED,
    COMPLEXITY_REFERENCE_INPUT,
    COST_BENCHMARK_COMPLETE,
    EXECUTION_PASSES,
    HOST_TRANSFER_BOUNDARY,
    INFERENCE_MEMORY,
    LATENCY_BOUNDARY_BLOCKED,
    MEMORY_FIELDS,
    MEMORY_ISOLATION,
    MEMORY_ISOLATION_DIAGNOSTIC,
    MEMORY_MEASUREMENT_BLOCKED,
    MODEL_IDENTITY_MISMATCH,
    MODELS,
    NO_POWER_TELEMETRY,
    OBSERVATION_FIELDS,
    PERCENTILE_CONVENTION,
    PHASE,
    PRECISION_PROTOCOL_MISMATCH,
    PROTOCOL_VIOLATION,
    SEGMENTATION_COST_LABEL,
    STATIC_MODEL_COMPLEXITY,
    STD_CONVENTION,
    SYNCHRONIZATION_PRIMITIVE,
    TELEMETRY_NOTE,
    TIMING_BOUNDARIES,
    TIMING_PRIMITIVE,
    TRAINING_MEMORY_LABEL,
    WARMUP_PATH,
    WARMUP_SCHEDULE,
    CostBenchmarkError,
    ExecutionStep,
    MemoryIsolationError,
    PrecisionParityError,
    as_gib,
    assert_precision_parity,
    build_cost_deltas,
    build_execution_plan,
    build_protocol_stability,
    describe_distribution_shape,
    describe_latency,
    dumps,
    execution_plan_fingerprint,
    expected_observation_count,
    images_per_second_from_mean,
    latency_result_fingerprint,
    memory_delta,
    memory_result_fingerprint,
    precision_evidence,
    render_rows,
    timing_dataset_fingerprint,
    validate_latency_result,
    validate_memory_result,
)
from construction_safety_vision.paths import ProjectPaths, long_path
from construction_safety_vision.provenance import ProvenanceRecord, git_commit, sha256_file
from construction_safety_vision.segmentation_freeze import (
    FinalSegmenterError,
    load_final_segmenter,
)
from construction_safety_vision.segmentation_freeze import (
    load_checkpoint_path as segmenter_checkpoint,
)
from construction_safety_vision.splits import HOLDOUT_UNLOCK_ENV_VAR, holdout_unlocked

CONFIG_YAML = "detector_segmenter_comparison.yaml"
PROTOCOL_JSON = "detector_segmenter_comparison_protocol.json"
BENCHMARK_CSV = "detector_segmenter_latency_membership.csv"

LATENCY_JSON = "detector_segmenter_latency_comparison.json"
MEMORY_JSON = "detector_segmenter_memory_comparison.json"
BLOCKS_CSV = "detector_segmenter_latency_blocks.csv"
REPORT_MD = "detector_segmenter_latency_report.md"
PROVENANCE_JSON = "detector_segmenter_cost_benchmark.provenance.json"

RUNTIME_ROOT = "artifacts/benchmark"

SCHEMA_VERSION = 1

EXPECTED_PROTOCOL = "d92a157637baf52f9a7248bf24257d5a177e8496b7727ffb5afc874ab96d1c4d"
EXPECTED_BENCHMARK = "45059c2cdda1285b4fa6dbfdcfbeac6fedcad551e551e6810b1369af90f7f162"

DETECTOR = DETECTOR_EXPERIMENT
SEGMENTER = SEGMENTER_EXPERIMENT

CONFIDENCE_SOURCE = "operational_inference"
CONFIDENCE_DECISION = (
    "The frozen latency_protocol declares batch, resolution, precision, warmup, repetitions, "
    "membership and execution order, but no confidence threshold. The operational block is the "
    "only frozen operating point - the protocol itself states that the AP block's 0.001 is "
    "deliberately not one, because average precision needs the low-scoring tail - so the "
    "benchmark runs at the operational 0.25. Recorded as a gap in the frozen protocol and "
    "settled before any timing existed; latency at 0.001 was not measured and is not claimed, "
    "and it would differ, because a lower threshold pushes more candidates through NMS and, for "
    "the segmenter, more masks through reconstruction."
)

MASK_OUTPUT_DECISION = (
    "retina_masks is enabled for the segmenter, so masks are reconstructed on the original "
    "image canvas by ops.process_mask_native inside postprocess. That is the project's usable "
    "segmentation output everywhere else, and the alternative would time masks at the "
    "letterboxed model resolution. It is ignored by the detector, which predicts no mask."
)

HOLDOUT_REASON = (
    "Phase 10C timed both frozen models on the frozen 20-image validation benchmark subset "
    "only. The holdout was not read, materialised, adapted, counted, predicted on, timed or "
    "inspected; no holdout identifier, image, prediction, timing or statistic exists in any "
    "artifact this phase wrote."
)

HISTORICAL: tuple[str, ...] = (
    "configs/detector_segmenter_comparison.yaml",
    "reports/detector_segmenter_comparison_protocol.json",
    "reports/detector_segmenter_comparison_protocol.md",
    "reports/detector_segmenter_comparison_membership.csv",
    "reports/detector_segmenter_latency_membership.csv",
    "reports/detector_segmenter_box_comparison.json",
    "reports/detector_segmenter_box_comparison.csv",
    "reports/detector_segmenter_spatial_comparison.json",
    "reports/detector_segmenter_validation_comparison.md",
    "reports/detector_segmenter_comparison_examples.csv",
    "reports/final_detector_manifest.json",
    "reports/detection_selection_report.md",
    "reports/detection_experiment_results.json",
    "reports/detection_D2_manifest.json",
    "reports/final_segmenter_manifest.json",
    "reports/segmentation_selection_report.md",
    "reports/segmentation_experiment_results.json",
    "reports/segmentation_S0_result_manifest.json",
    "reports/segmentation_S1_result_manifest.json",
    "reports/segmentation_S1_canonical_evaluation.json",
    "reports/split_manifest.json",
    "reports/task_dataset_manifest.json",
)
"""Every artifact this phase must leave byte-identical."""


class BenchmarkError(RuntimeError):
    """Raised when a required input is missing or an invariant fails."""


class ModelIdentityError(BenchmarkError):
    """Raised when a frozen model is not the one the protocol names."""


class BoundaryError(BenchmarkError):
    """Raised when a frozen timing boundary cannot be isolated equivalently."""


# --- helpers ------------------------------------------------------------------------


def read_json(path: Path) -> dict[str, Any]:
    """Read a committed JSON artifact.

    Args:
        path: File to read.

    Returns:
        The parsed mapping.

    Raises:
        BenchmarkError: If the file is missing or is not a JSON object.
    """
    if not path.is_file():
        msg = f"required input not found: {path.name}"
        raise BenchmarkError(msg)
    data = json.loads(Path(long_path(path)).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        msg = f"{path.name} must contain a JSON object"
        raise BenchmarkError(msg)
    return data


def write_text(path: Path, text: str) -> str:
    """Write a text artifact deterministically and return its digest.

    Args:
        path: Destination file.
        text: Content.

    Returns:
        The SHA-256 digest of the bytes written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")
    return sha256_file(path)


def historical_digests(paths: ProjectPaths) -> dict[str, str]:
    """Digest every artifact this phase must not change.

    Args:
        paths: Project layout.

    Returns:
        Digests keyed by repository-relative name.

    Raises:
        BenchmarkError: If one is absent.
    """
    digests: dict[str, str] = {}
    for name in HISTORICAL:
        path = paths.root / name
        if not path.is_file():
            msg = f"historical artifact missing: {name}"
            raise BenchmarkError(msg)
        digests[name] = sha256_file(path)
    return digests


# --- the frozen benchmark population ---------------------------------------------------


def load_benchmark_subset(paths: ProjectPaths, protocol: Any) -> dict[str, Any]:
    """Read the frozen benchmark subset and verify it has not moved.

    The subset is checked three ways: the committed order reproduces the frozen
    ordered fingerprint, the ids reproduce the frozen selection rule when
    re-derived from the validation membership, and every id resolves to an
    image on this machine. Nothing is reselected and no image is opened to
    choose it.

    Args:
        paths: Project layout.
        protocol: The frozen comparison protocol.

    Returns:
        The ordered ids, their files and the verified fingerprints.

    Raises:
        BenchmarkError: If the subset, its order or its source has changed.
    """
    latency = protocol.latency
    declared = protocol["population"]

    document = paths.root / str(declared["source"])
    digest = sha256_file(document)
    if digest != declared["source_sha256"]:
        msg = f"the canonical validation document changed: {digest}"
        raise BenchmarkError(msg)
    payload = json.loads(Path(long_path(document)).read_text(encoding="utf-8"))
    entries = sorted(payload["images"], key=lambda entry: str(entry["file_name"]))
    validation_ids = [Path(str(entry["file_name"])).stem for entry in entries]
    if membership_fingerprint(validation_ids) != declared["membership_sha256"]:
        msg = "the validation membership changed"
        raise BenchmarkError(msg)

    table = paths.reports / BENCHMARK_CSV
    if not table.is_file():
        msg = f"the frozen benchmark membership is missing: {BENCHMARK_CSV}"
        raise BenchmarkError(msg)
    with Path(long_path(table)).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if [int(row["benchmark_order"]) for row in rows] != list(range(len(rows))):
        msg = "the committed benchmark order is not a contiguous zero-based sequence"
        raise BenchmarkError(msg)
    if any(row["split"] != EVALUATION_SPLIT for row in rows):
        msg = f"every benchmark image must come from the {EVALUATION_SPLIT} split"
        raise BenchmarkError(msg)

    image_ids = [str(row["source_image_id"]) for row in rows]
    if len(image_ids) != int(latency["benchmark_image_count"]):
        msg = (
            f"expected {latency['benchmark_image_count']} benchmark images, found {len(image_ids)}"
        )
        raise BenchmarkError(msg)
    fingerprint = ordered_fingerprint(image_ids)
    if fingerprint != latency["benchmark_membership_sha256"]:
        msg = f"the benchmark subset or its order changed: {fingerprint}"
        raise BenchmarkError(msg)
    if fingerprint != EXPECTED_BENCHMARK:
        msg = f"the benchmark subset does not reproduce the frozen fingerprint {EXPECTED_BENCHMARK}"
        raise BenchmarkError(msg)
    if any(row["stable_rank_sha256"] != stable_rank(row["source_image_id"]) for row in rows):
        msg = "a committed stable rank does not reproduce from its image id"
        raise BenchmarkError(msg)
    if tuple(image_ids) != select_benchmark_images(validation_ids, len(image_ids)):
        msg = (
            f"the committed subset does not reproduce {SELECTION_RULE} over the validation "
            "membership"
        )
        raise BenchmarkError(msg)

    images_root = paths.data_processed / "canonical" / "images" / EVALUATION_SPLIT
    lookup = {Path(str(entry["file_name"])).stem: str(entry["file_name"]) for entry in entries}
    files: dict[str, Path] = {}
    for image_id in image_ids:
        if image_id not in lookup:
            msg = f"a benchmark image is not in the validation membership: {image_id}"
            raise BenchmarkError(msg)
        candidate = images_root / lookup[image_id]
        if not candidate.is_file():
            msg = f"benchmark image not on this machine: {lookup[image_id]}"
            raise BenchmarkError(msg)
        files[image_id] = candidate

    return {
        "image_ids": tuple(image_ids),
        "files": files,
        "benchmark_membership_sha256": fingerprint,
        "validation_membership_sha256": declared["membership_sha256"],
        "validation_images": len(validation_ids),
        "document_sha256": digest,
        "document_path": str(declared["source"]),
    }


# --- hardware and runtime provenance ---------------------------------------------------


def runtime_provenance() -> dict[str, Any]:
    """Record the machine and software stack the benchmark ran on.

    Non-invasive: nothing about the power plan, clocks or fan curves is
    changed, and nothing machine-identifying beyond the GPU model and driver is
    recorded.

    Returns:
        The runtime and hardware description.
    """
    import torch

    gpu: dict[str, Any] = {"cuda_available": bool(torch.cuda.is_available())}
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability(0)
        properties = torch.cuda.get_device_properties(0)
        gpu.update(
            {
                "name": torch.cuda.get_device_name(0),
                "compute_capability": f"{major}.{minor}",
                "total_memory_bytes": int(properties.total_memory),
                "multi_processor_count": int(properties.multi_processor_count),
            }
        )

    driver = "UNKNOWN"
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        if completed.returncode == 0 and completed.stdout.strip():
            driver = completed.stdout.strip().splitlines()[0].strip()
    except (OSError, subprocess.SubprocessError):
        driver = "UNKNOWN"

    power: dict[str, Any] = {"source": "UNKNOWN", "battery_status_raw": None}
    try:
        completed = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "(Get-CimInstance Win32_Battery | Select-Object -First 1"
                " -ExpandProperty BatteryStatus)",
            ],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        raw = completed.stdout.strip()
        if completed.returncode == 0 and raw.isdigit():
            power = {
                "source": "AC_POWER" if int(raw) == 2 else f"BATTERY_STATUS_{raw}",
                "battery_status_raw": int(raw),
                "interpretation": (
                    "Win32_Battery.BatteryStatus 2 documents that the system has access to AC "
                    "power, so the battery is not discharging."
                ),
            }
    except (OSError, subprocess.SubprocessError):
        power = {"source": "UNKNOWN", "battery_status_raw": None}

    import ultralytics

    return {
        "python": platform.python_version(),
        "platform": platform.system(),
        "platform_release": platform.release(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "ultralytics": ultralytics.__version__,
        "gpu": gpu,
        "driver_version": driver,
        "power": power,
        "power_settings_changed_by_this_phase": False,
        "thermal_correction_applied": False,
        "per_observation_power_state_telemetry": NO_POWER_TELEMETRY,
        "telemetry_synchronous_with_timed_blocks": False,
        "telemetry_note": TELEMETRY_NOTE,
    }


# --- the latency benchmark -------------------------------------------------------------


def prepare_model(checkpoint: Path, settings: Mapping[str, Any], image: Any, *, masks: bool) -> Any:
    """Load one frozen model and drive its predictor to a benchmark-ready state.

    Model construction, the checkpoint read, the CUDA transfer and the
    framework's own first-call warmup all happen here, outside every timed
    region, exactly as the frozen protocol requires.

    Args:
        checkpoint: The frozen checkpoint.
        settings: The frozen inference block to run under.
        image: A decoded source image, used only to initialise the predictor.
        masks: Whether to request original-canvas mask reconstruction.

    Returns:
        The loaded model, with ``model.predictor`` set up.
    """
    from ultralytics import YOLO

    model = YOLO(str(checkpoint))
    call = {
        "source": image.copy(),
        "imgsz": settings["imgsz"],
        "conf": settings["conf"],
        "iou": settings["iou"],
        "max_det": settings["max_det"],
        "augment": settings["augment"],
        settings["precision_argument"]: settings["precision_value"],
        "verbose": False,
        "save": False,
        "show": False,
        "visualize": False,
        "stream": False,
    }
    if masks:
        call["retina_masks"] = True
    model.predict(**call)
    return model


def check_predictor(model: Any, settings: Mapping[str, Any], *, name: str) -> dict[str, Any]:
    """Verify one predictor honours every frozen inference invariant.

    Args:
        model: The loaded model.
        settings: The frozen inference block.
        name: The experiment id, for messages.

    Returns:
        The resolved argument evidence.

    Raises:
        BenchmarkError: If any frozen setting did not take effect.
    """
    predictor = model.predictor
    args = predictor.args
    resolved = {
        "imgsz": int(args.imgsz) if isinstance(args.imgsz, int) else list(args.imgsz),
        "conf": float(args.conf),
        "iou": float(args.iou),
        "max_det": int(args.max_det),
        "augment": bool(args.augment),
        "retina_masks": bool(getattr(args, "retina_masks", False)),
        PRECISION_ARGUMENT: getattr(args, PRECISION_ARGUMENT, None),
        "batch": int(getattr(args, "batch", LATENCY_BATCH)),
        "device": str(predictor.device),
        "warmup_done_before_timing": bool(predictor.done_warmup),
        "predictor": type(predictor).__name__,
        "postprocess_owner": type(predictor).postprocess.__qualname__,
    }
    if resolved["imgsz"] != COMPARISON_IMGSZ:
        msg = f"{name}: imgsz resolved to {resolved['imgsz']}, not {COMPARISON_IMGSZ}"
        raise BenchmarkError(msg)
    if resolved["conf"] != float(settings["conf"]) or resolved["iou"] != float(settings["iou"]):
        msg = f"{name}: the frozen confidence or NMS IoU did not take effect"
        raise BenchmarkError(msg)
    if resolved["max_det"] != int(settings["max_det"]):
        msg = f"{name}: max_det resolved to {resolved['max_det']}"
        raise BenchmarkError(msg)
    if resolved["augment"]:
        msg = f"{name}: augmentation is enabled and no test-time augmentation is authorised"
        raise BenchmarkError(msg)
    if int(resolved[PRECISION_ARGUMENT]) != PRECISION_VALUE:
        msg = f"{name}: {PRECISION_ARGUMENT} resolved to {resolved[PRECISION_ARGUMENT]}"
        raise BenchmarkError(msg)
    if not resolved["device"].startswith("cuda"):
        msg = (
            f"{name}: the predictor is on {resolved['device']}. A CPU fallback would not be the "
            "same benchmark; this is BLOCKED_FOR_GPU, not a substitute measurement."
        )
        raise BenchmarkError(msg)
    if not resolved["warmup_done_before_timing"]:
        msg = f"{name}: the framework's own first-call warmup has not completed"
        raise BenchmarkError(msg)
    return resolved


def run_block(
    model: Any,
    step: ExecutionStep,
    image: Any,
    path: Path,
    *,
    masks: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Warm up and time one model on one image, at both frozen boundaries.

    The block runs the frozen 20 warmup iterations over the full end-to-end
    path first - the widest of the two boundaries, so the narrower one is
    covered too - then times the model-inference boundary and the end-to-end
    boundary, 30 repetitions each. Warmup output is discarded.

    Args:
        model: The prepared model.
        step: The execution step this block corresponds to.
        image: The decoded source image, shared with the other model.
        path: The image's path, used only to populate the framework's batch.
        masks: Whether this model produces masks.

    Returns:
        The raw observations, and the block's structural evidence.

    Raises:
        BoundaryError: If a boundary cannot be materialised as frozen.
    """
    import torch

    predictor = model.predictor
    predictor.batch = [[str(path)], [image], [""]]

    for _ in range(WARMUP_ITERATIONS):
        tensor = predictor.preprocess([image])
        results = predictor.postprocess(predictor.inference(tensor), tensor, [image])
        _ = materialize(results, masks=masks)
    torch.cuda.synchronize()

    probe_tensor = predictor.preprocess([image])
    probe = predictor.postprocess(predictor.inference(probe_tensor), probe_tensor, [image])
    evidence = describe_outputs(probe, image, masks=masks)
    torch.cuda.synchronize()

    observations: list[dict[str, Any]] = []

    def record(timing_type: str, repetition: int, latency_ms: float) -> None:
        observations.append(
            {
                "model": step.model,
                "benchmark_rank": step.benchmark_rank,
                "source_image_id": step.source_image_id,
                "execution_pass": step.execution_pass,
                "execution_position": step.position,
                "repetition": repetition,
                "timing_type": timing_type,
                "latency_ms": round(latency_ms, METRIC_PRECISION),
            }
        )

    # MODEL_INFERENCE: the forward pass on an already-prepared input tensor.
    prepared = predictor.preprocess([image])
    torch.cuda.synchronize()
    for repetition in range(TIMED_ITERATIONS_PER_IMAGE):
        torch.cuda.synchronize()
        start = time.perf_counter()
        raw = predictor.inference(prepared)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        record(MODEL_INFERENCE_LATENCY, repetition, elapsed * 1000.0)
        del raw

    # END_TO_END: preprocessing, forward, NMS/postprocessing and, for the
    # segmenter, mask reconstruction onto the original canvas.
    for repetition in range(TIMED_ITERATIONS_PER_IMAGE):
        torch.cuda.synchronize()
        start = time.perf_counter()
        tensor = predictor.preprocess([image])
        outputs = predictor.postprocess(predictor.inference(tensor), tensor, [image])
        materialized = materialize(outputs, masks=masks)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        record(END_TO_END_LATENCY, repetition, elapsed * 1000.0)
        if materialized is None:
            msg = (
                f"{step.model}: the end-to-end boundary produced no usable output on "
                f"{step.source_image_id}"
            )
            raise BoundaryError(msg)
        del tensor, outputs, materialized

    evidence["input_tensor_shape"] = list(prepared.shape)
    evidence["input_tensor_dtype"] = str(prepared.dtype)
    del prepared, probe_tensor, probe
    return observations, evidence


def materialize(results: Sequence[Any], *, masks: bool) -> Any:
    """Force the frozen boundary's outputs into existence before the timer stops.

    Python is lazy enough that a wrapper object can be created without its
    contents being computed. Touching the shapes here makes sure the boundary
    times the work rather than the wrapper: the detector's boundary ends when
    class, confidence and box are available, the segmenter's when the instance
    masks are available too.

    Args:
        results: The framework's ``Results`` objects.
        masks: Whether instance masks are part of this model's output.

    Returns:
        The materialised counts, or ``None`` when the boundary produced nothing.
    """
    result = results[0]
    boxes = result.boxes
    if boxes is None:
        return None
    detected = int(boxes.data.shape[0])
    if not masks:
        return (detected, int(boxes.data.shape[1]))
    if detected == 0:
        return (0, 0, 0)
    if result.masks is None:
        return None
    shape = tuple(int(value) for value in result.masks.data.shape)
    return (detected, shape[0], shape[1] * shape[2])


def describe_outputs(results: Sequence[Any], image: Any, *, masks: bool) -> dict[str, Any]:
    """Describe what one model's frozen boundary actually produced.

    Args:
        results: The framework's ``Results`` objects.
        image: The decoded source image.
        masks: Whether instance masks are expected.

    Returns:
        Structural evidence about the output, with no prediction content.
    """
    result = results[0]
    detected = 0 if result.boxes is None else int(result.boxes.data.shape[0])
    evidence: dict[str, Any] = {
        "instances": detected,
        "boxes_available": result.boxes is not None,
        "masks_available": result.masks is not None,
    }
    if masks and result.masks is not None:
        shape = tuple(int(value) for value in result.masks.data.shape)
        evidence["mask_shape"] = list(shape)
        evidence["masks_on_original_canvas"] = shape[1:] == tuple(int(v) for v in image.shape[:2])
        evidence["mask_device"] = str(result.masks.data.device).split(":")[0]
    return evidence


def benchmark_latency(
    checkpoints: Mapping[str, Path],
    subset: Mapping[str, Any],
    settings: Mapping[str, Any],
) -> dict[str, Any]:
    """Execute the frozen symmetric interleaved latency benchmark.

    Args:
        checkpoints: Resolved checkpoint paths, keyed by experiment.
        subset: The verified benchmark subset.
        settings: The frozen inference block to run under.

    Returns:
        The raw observations, per-block summaries and structural evidence.

    Raises:
        BenchmarkError: If a fairness invariant is violated at runtime.
    """
    import torch
    from ultralytics.utils.patches import imread

    if not torch.cuda.is_available():
        msg = (
            "BLOCKED_FOR_GPU: CUDA reports unavailable. The benchmark is not re-run on CPU and "
            "called the same measurement."
        )
        raise BenchmarkError(msg)

    image_ids = list(subset["image_ids"])
    decoded: dict[str, Any] = {}
    for image_id in image_ids:
        array = imread(long_path(subset["files"][image_id]))
        if array is None:
            msg = f"could not decode benchmark image {image_id}"
            raise BenchmarkError(msg)
        decoded[image_id] = array

    plan = build_execution_plan(image_ids)
    models = {
        DETECTOR: prepare_model(
            checkpoints[DETECTOR], settings, decoded[image_ids[0]], masks=False
        ),
        SEGMENTER: prepare_model(
            checkpoints[SEGMENTER], settings, decoded[image_ids[0]], masks=True
        ),
    }
    resolved = {
        experiment: check_predictor(models[experiment], settings, name=experiment)
        for experiment in MODELS
    }
    if resolved[DETECTOR]["device"] != resolved[SEGMENTER]["device"]:
        msg = "the two models are not on the same device"
        raise BenchmarkError(msg)

    observations: list[dict[str, Any]] = []
    blocks: list[dict[str, Any]] = []
    shapes: dict[str, dict[str, list[int]]] = {experiment: {} for experiment in MODELS}
    outputs: dict[str, dict[str, Any]] = {experiment: {} for experiment in MODELS}
    started = datetime.now(UTC).isoformat(timespec="seconds")

    for step in plan:
        rows, evidence = run_block(
            models[step.model],
            step,
            decoded[step.source_image_id],
            subset["files"][step.source_image_id],
            masks=step.model == SEGMENTER,
        )
        observations.extend(rows)
        shapes[step.model][step.source_image_id] = evidence["input_tensor_shape"]
        outputs[step.model].setdefault(step.source_image_id, evidence)
        for boundary in TIMING_BOUNDARIES:
            values = [row["latency_ms"] for row in rows if row["timing_type"] == boundary]
            statistics = describe_latency(values)
            blocks.append(
                {
                    "execution_position": step.position,
                    "execution_pass": step.execution_pass,
                    "benchmark_rank": step.benchmark_rank,
                    "source_image_id": step.source_image_id,
                    "model": step.model,
                    "timing_type": boundary,
                    "count": statistics["count"],
                    "mean": statistics["mean"],
                    "median": statistics["median"],
                    "std": statistics["std"],
                    "min": statistics["min"],
                    "max": statistics["max"],
                }
            )

    finished = datetime.now(UTC).isoformat(timespec="seconds")

    # Fairness: the same image must reach both networks as the same tensor.
    divergent = sorted(
        image_id
        for image_id in image_ids
        if shapes[DETECTOR].get(image_id) != shapes[SEGMENTER].get(image_id)
    )
    if divergent:
        msg = (
            f"the two models received differently shaped inputs for {divergent}. The benchmark "
            "aborts rather than comparing them at different effective resolutions."
        )
        raise BenchmarkError(msg)

    mask_images = [
        image_id for image_id in image_ids if outputs[SEGMENTER][image_id].get("instances", 0) > 0
    ]
    off_canvas = [
        image_id
        for image_id in mask_images
        if not outputs[SEGMENTER][image_id].get("masks_on_original_canvas")
    ]
    if off_canvas:
        msg = (
            f"the segmenter's masks were not reconstructed on the original canvas for "
            f"{off_canvas}, so the end-to-end boundary would not include the full mask cost"
        )
        raise BoundaryError(msg)

    for experiment in MODELS:
        del models[experiment]
    models.clear()
    torch.cuda.empty_cache()

    return {
        "observations": observations,
        "blocks": blocks,
        "plan": plan,
        "resolved_arguments": resolved,
        "input_tensor_shapes": {image_id: shapes[DETECTOR][image_id] for image_id in image_ids},
        "identical_input_tensor_shapes": True,
        "output_evidence": {
            DETECTOR: {
                "instances_total": sum(
                    outputs[DETECTOR][image_id]["instances"] for image_id in image_ids
                ),
                "masks_available": False,
                "boundary_ends_at": "CLASS_CONFIDENCE_BOX_AVAILABLE",
            },
            SEGMENTER: {
                "instances_total": sum(
                    outputs[SEGMENTER][image_id]["instances"] for image_id in image_ids
                ),
                "images_with_masks": len(mask_images),
                "masks_on_original_canvas": True,
                "boundary_ends_at": "CLASS_CONFIDENCE_BOX_AND_INSTANCE_MASK_AVAILABLE",
            },
        },
        "started_at": started,
        "finished_at": finished,
    }


# --- the controlled inference-memory measurement ----------------------------------------


def measure_memory(
    checkpoint: Path,
    subset: Mapping[str, Any],
    settings: Mapping[str, Any],
    *,
    masks: bool,
) -> dict[str, Any]:
    """Measure one model's inference memory, with nothing else resident.

    Runs in a process that has loaded no other model, so the device-global peak
    statistics can be attributed to this model alone. Peak statistics are reset
    after the frozen warmup, so the figure describes inference rather than the
    allocator's warmup high-water mark.

    Args:
        checkpoint: The frozen checkpoint.
        subset: The verified benchmark subset.
        settings: The frozen inference block.
        masks: Whether this model produces masks.

    Returns:
        The measured memory block.

    Raises:
        MemoryIsolationError: If the allocator is not empty before the load.
    """
    import torch
    from ultralytics.utils.patches import imread

    if not torch.cuda.is_available():
        msg = "BLOCKED_FOR_GPU: CUDA reports unavailable, so no inference memory can be measured"
        raise MemoryIsolationError(msg)

    pre_allocated = int(torch.cuda.memory_allocated())
    pre_reserved = int(torch.cuda.memory_reserved())
    if pre_allocated != 0:
        msg = (
            f"{MEMORY_MEASUREMENT_BLOCKED}: {pre_allocated} bytes were already allocated on the "
            "device before this model was loaded, so its peak could not be attributed to it"
        )
        raise MemoryIsolationError(msg)

    image_ids = list(subset["image_ids"])
    decoded = {
        image_id: imread(long_path(Path(subset["files"][image_id]))) for image_id in image_ids
    }

    model = prepare_model(checkpoint, settings, decoded[image_ids[0]], masks=masks)
    predictor = model.predictor

    for _ in range(WARMUP_ITERATIONS):
        image = decoded[image_ids[0]]
        predictor.batch = [[str(subset["files"][image_ids[0]])], [image], [""]]
        tensor = predictor.preprocess([image])
        results = predictor.postprocess(predictor.inference(tensor), tensor, [image])
        _ = materialize(results, masks=masks)
        del tensor, results
    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()
    baseline_allocated = int(torch.cuda.memory_allocated())
    baseline_reserved = int(torch.cuda.memory_reserved())

    for image_id in image_ids:
        image = decoded[image_id]
        predictor.batch = [[str(subset["files"][image_id])], [image], [""]]
        tensor = predictor.preprocess([image])
        results = predictor.postprocess(predictor.inference(tensor), tensor, [image])
        _ = materialize(results, masks=masks)
        del tensor, results
    torch.cuda.synchronize()

    peak_allocated = int(torch.cuda.max_memory_allocated())
    peak_reserved = int(torch.cuda.max_memory_reserved())

    return {
        "pre_load_allocated_bytes": pre_allocated,
        "pre_load_reserved_bytes": pre_reserved,
        "baseline_allocated_bytes": baseline_allocated,
        "baseline_reserved_bytes": baseline_reserved,
        "peak_memory_allocated_bytes": peak_allocated,
        "peak_memory_reserved_bytes": peak_reserved,
        "peak_memory_allocated_gib": as_gib(peak_allocated),
        "peak_memory_reserved_gib": as_gib(peak_reserved),
        "other_model_resident": False,
        "images_measured": len(image_ids),
        "peak_stats_reset_after_warmup": True,
        "measurement": INFERENCE_MEMORY,
    }


def memory_in_child(paths: ProjectPaths, experiment: str) -> dict[str, Any]:
    """Measure one model's inference memory in a dedicated child process.

    Args:
        paths: Project layout.
        experiment: Which frozen model to measure.

    Returns:
        The measured memory block.

    Raises:
        BenchmarkError: If the child process fails or emits no result.
    """
    completed = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--measure-memory", experiment],
        capture_output=True,
        text=True,
        cwd=str(paths.root),
        check=False,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip().splitlines()
        msg = f"the {experiment} memory measurement failed: {detail[-1] if detail else 'no output'}"
        raise BenchmarkError(msg)
    for line in reversed(completed.stdout.strip().splitlines()):
        if line.startswith("{"):
            return json.loads(line)
    msg = f"the {experiment} memory measurement produced no result"
    raise BenchmarkError(msg)


# --- artifacts --------------------------------------------------------------------------


def build_latency_artifact(
    *,
    protocol: Any,
    models: Mapping[str, Any],
    subset: Mapping[str, Any],
    execution: Mapping[str, Any],
    precision: Mapping[str, Any],
    runtime: Mapping[str, Any],
    complexity: Mapping[str, Any],
    settings: Mapping[str, Any],
) -> dict[str, Any]:
    """Assemble the latency result artifact.

    Args:
        protocol: The frozen comparison protocol.
        models: The two frozen model identities.
        subset: The verified benchmark subset.
        execution: The benchmark's observations and evidence.
        precision: The verified precision-parity declaration.
        runtime: The hardware and software provenance.
        complexity: The static model-complexity block.
        settings: The frozen inference block the benchmark ran under.

    Returns:
        The artifact, without its own fingerprint.
    """
    observations = execution["observations"]
    latency: dict[str, Any] = {}
    for experiment in MODELS:
        block: dict[str, Any] = {"throughput": {}}
        for boundary in TIMING_BOUNDARIES:
            values = [
                row["latency_ms"]
                for row in observations
                if row["model"] == experiment and row["timing_type"] == boundary
            ]
            statistics = describe_latency(values)
            block[boundary] = statistics
            block["throughput"][boundary] = images_per_second_from_mean(statistics["mean"])
        latency[experiment] = block

    deltas = build_cost_deltas(latency[DETECTOR], latency[SEGMENTER])

    return {
        "post_hoc_distribution_diagnostic": describe_distribution_shape(observations, latency),
        "protocol_stability_after_observation": build_protocol_stability(),
        "schema_version": SCHEMA_VERSION,
        "phase": PHASE,
        "status": COST_BENCHMARK_COMPLETE,
        "benchmark_status": BENCHMARK_LABEL,
        "protocol": protocol["protocol"],
        "protocol_fingerprint": protocol.fingerprint(),
        "protocol_config": f"configs/{CONFIG_YAML}",
        "detector": dict(models[DETECTOR]),
        "segmenter": dict(models[SEGMENTER]),
        "benchmark": {
            "split": EVALUATION_SPLIT,
            "benchmark_image_count": len(subset["image_ids"]),
            "benchmark_selection_rule": SELECTION_RULE,
            "benchmark_membership_sha256": subset["benchmark_membership_sha256"],
            "benchmark_membership_artifact": f"reports/{BENCHMARK_CSV}",
            "validation_membership_sha256": subset["validation_membership_sha256"],
            "validation_images": subset["validation_images"],
            "images_reselected": False,
            "images_inspected_to_select": 0,
            "execution_plan_sha256": execution_plan_fingerprint(execution["plan"]),
            "execution_passes": list(EXECUTION_PASSES),
            "execution_blocks": len(execution["plan"]),
            "interleaved": True,
            "symmetric": True,
            "randomized": False,
            "order_changed_after_observing_timings": False,
            "identical_images_for_both_models": True,
            "identical_order_for_both_models": True,
            "identical_input_tensor_shapes": execution["identical_input_tensor_shapes"],
            "input_tensor_shapes": execution["input_tensor_shapes"],
            "started_at": execution["started_at"],
            "finished_at": execution["finished_at"],
        },
        "conditions": {
            "batch": LATENCY_BATCH,
            "imgsz": COMPARISON_IMGSZ,
            "precision": PRECISION,
            "precision_argument": PRECISION_ARGUMENT,
            "precision_value": PRECISION_VALUE,
            "conf": float(settings["conf"]),
            "confidence_source": CONFIDENCE_SOURCE,
            "confidence_decision": CONFIDENCE_DECISION,
            "iou": float(settings["iou"]),
            "max_det": int(settings["max_det"]),
            "augment": False,
            "tta": False,
            "visualization": False,
            "predictions_saved_during_timing": False,
            "warmup_iterations": WARMUP_ITERATIONS,
            "warmup_schedule": WARMUP_SCHEDULE,
            "warmup_path": WARMUP_PATH,
            "warmup_outputs_discarded": True,
            "warmup_count_changed_after_observing_timings": False,
            "timed_iterations_per_image": TIMED_ITERATIONS_PER_IMAGE,
            "repetitions_changed_after_observing_variance": False,
            "timing_primitive": TIMING_PRIMITIVE,
            "synchronization_primitive": SYNCHRONIZATION_PRIMITIVE,
            "cuda_synchronize_before_timed_region": True,
            "cuda_synchronize_after_timed_region": True,
            "same_primitive_for_both_models": True,
            "percentile_convention": PERCENTILE_CONVENTION,
            "std_convention": STD_CONVENTION,
            "model_load_excluded_from_timing": True,
            "disk_decode_excluded_from_timing": True,
            "decode_reader": "ultralytics.utils.patches.imread",
            "decode_shared_between_models": True,
            "torch_compile": False,
            "tensorrt": False,
            "onnx": False,
            "recompiled_for_one_model_only": False,
            "mask_output_decision": MASK_OUTPUT_DECISION,
            "resolved_arguments": execution["resolved_arguments"],
        },
        "timing_boundaries": {
            MODEL_INFERENCE_LATENCY: {
                "definition": (
                    "The forward pass on an already-prepared input tensor. The tensor is built "
                    "outside the timer and reused across the block's repetitions, so the "
                    "measurement is the model core."
                ),
                "framework_call": "BasePredictor.inference",
                "preprocessing_included": False,
                "postprocessing_included": False,
                "mask_reconstruction_included": False,
                "disk_io_included": False,
                "model_load_included": False,
                "same_call_for_both_models": True,
            },
            END_TO_END_LATENCY: {
                "definition": (
                    "What a caller waits for: input preprocessing, the forward pass, NMS and "
                    "postprocessing, and for the segmenter the mask reconstruction that exposes "
                    "final instance masks on the original canvas."
                ),
                "framework_calls": [
                    "BasePredictor.preprocess",
                    "BasePredictor.inference",
                    "DetectionPredictor.postprocess / SegmentationPredictor.postprocess",
                ],
                "preprocessing_included": True,
                "postprocessing_included": True,
                "segmenter_mask_reconstruction_included": True,
                "mask_reconstruction_call": "ultralytics.utils.ops.process_mask_native",
                "mask_reconstruction_site": "SegmentationPredictor.construct_result",
                "outputs_materialized_before_end_timestamp": True,
                "disk_io_included": False,
                "model_load_included": False,
                "host_transfer": HOST_TRANSFER_BOUNDARY,
                "detector_boundary_ends_at": execution["output_evidence"][DETECTOR][
                    "boundary_ends_at"
                ],
                "segmenter_boundary_ends_at": execution["output_evidence"][SEGMENTER][
                    "boundary_ends_at"
                ],
            },
        },
        "boundaries_combined": False,
        "output_evidence": execution["output_evidence"],
        "precision_preflight": dict(precision),
        "runtime": dict(runtime),
        "latency": latency,
        "deltas": deltas,
        "combined_latency_score": False,
        "winner_declared": False,
        "raw_timings": {
            "schema": list(OBSERVATION_FIELDS),
            "observations": len(observations),
            "expected_observations": expected_observation_count(),
            "observations_per_model_per_boundary": len(observations)
            // (len(MODELS) * len(TIMING_BOUNDARIES)),
            "timing_dataset_sha256": timing_dataset_fingerprint(observations),
            "row_level_artifact": f"{RUNTIME_ROOT}/latency_observations.csv",
            "row_level_artifact_committed": False,
            "block_summary_artifact": f"reports/{BLOCKS_CSV}",
            "holdout_identifiers_stored": False,
        },
        "model_complexity": dict(complexity),
        "segmentation_cost_label": SEGMENTATION_COST_LABEL,
        "pure_mask_reconstruction_cost_isolated": False,
        "segmentation_cost_note": (
            "YOLO11n and YOLO11n-seg differ in the mask branch of the network as well as in "
            "postprocessing, and this benchmark isolates neither from the other. The measured "
            "difference is the cost of the whole segmentation pipeline relative to the whole "
            "detection pipeline."
        ),
        "models_trained_in_this_phase": 0,
        "models_modified_in_this_phase": 0,
        "thresholds_tuned": 0,
        "ap_metrics_recomputed": False,
        "spatial_analysis_rerun": False,
        "association_analysis_rerun": False,
        "mask_iou_diagnostic_rerun": False,
        "protocol_adapted_after_results": False,
        "limitations": [
            "A laptop GPU throttles. These figures describe this machine under whatever power "
            "and thermal state it was in, and are labelled CONTROLLED_LOCAL_HARDWARE_BENCHMARK "
            "rather than presented as a property of either architecture.",
            "Mobile-GPU DVFS and power-state behaviour contributes to the observed latency "
            "distribution: block means span a wide range and the mean sits well above the "
            "median for both models at both boundaries. No GPU clock, P-state, utilisation or "
            "power telemetry accompanied the timed regions "
            "(NO_SYNCHRONIZED_PER_OBSERVATION_POWER_STATE_TELEMETRY), so no observation can be "
            "mapped to a device state and the cause of any feature of the distribution is "
            "UNKNOWN. Nothing was normalised, filtered or re-run in response.",
            "Batch 1 measures latency, not throughput under load. No batched or concurrent "
            "scenario is covered, and images_per_second is derived from the mean of a batch-1 "
            "latency rather than measured as a sustained rate.",
            "Neither model is exported, quantised or otherwise optimised for deployment. These "
            "are the PyTorch checkpoints as trained.",
            "One run of each model was ever trained, and each was timed once under this "
            "protocol. The distribution describes repetition-to-repetition variation on this "
            "machine, not run-to-run variance of training or of the benchmark itself.",
            "Copying the outputs to host memory is outside both boundaries, for both models "
            "alike, because the frozen boundary lists neither includes nor excludes it. The "
            "segmenter's outputs are far larger, so a pipeline that needs them in host memory "
            "would pay more than these figures show.",
            "The frozen latency protocol declares no confidence threshold. The benchmark uses "
            "the operational 0.25 and says so; a lower threshold would push more candidates "
            "through NMS and more masks through reconstruction, and that was not measured.",
            "The committed parameter and GFLOPs figures are static complexity at the "
            "framework's default 640 reference input, not at the benchmark's 768.",
        ],
        "test": {
            "status": HOLDOUT_STATUS,
            "reason": HOLDOUT_REASON,
            "images_read": 0,
            "predictions": 0,
            "timings": 0,
            "statistics": 0,
        },
        "holdout_accessed": False,
    }


def build_memory_artifact(
    *,
    protocol: Any,
    models: Mapping[str, Any],
    memory: Mapping[str, Any],
    runtime: Mapping[str, Any],
    settings: Mapping[str, Any],
) -> dict[str, Any]:
    """Assemble the inference-memory result artifact.

    Args:
        protocol: The frozen comparison protocol.
        models: The two frozen model identities.
        memory: The per-model measured memory blocks.
        runtime: The hardware and software provenance.
        settings: The frozen inference block the measurement ran under.

    Returns:
        The artifact, without its own fingerprint.
    """
    return {
        "schema_version": SCHEMA_VERSION,
        "phase": PHASE,
        "status": COST_BENCHMARK_COMPLETE,
        "measurement": INFERENCE_MEMORY,
        "benchmark_status": BENCHMARK_LABEL,
        "protocol": protocol["protocol"],
        "protocol_fingerprint": protocol.fingerprint(),
        "detector": dict(models[DETECTOR]),
        "segmenter": dict(models[SEGMENTER]),
        "conditions": {
            "batch": LATENCY_BATCH,
            "imgsz": COMPARISON_IMGSZ,
            "precision": PRECISION,
            "precision_argument": PRECISION_ARGUMENT,
            "precision_value": PRECISION_VALUE,
            "conf": float(settings["conf"]),
            "confidence_source": CONFIDENCE_SOURCE,
            "warmup_iterations": WARMUP_ITERATIONS,
            "peak_stats_reset_after_warmup": True,
            "reset_call": "torch.cuda.reset_peak_memory_stats",
            "isolation_method": MEMORY_ISOLATION,
            "isolation_diagnostic": MEMORY_ISOLATION_DIAGNOSTIC,
            "models_resident_during_measurement": 1,
            "images_measured": memory[DETECTOR]["images_measured"],
        },
        "runtime": dict(runtime),
        "memory": {experiment: dict(memory[experiment]) for experiment in MODELS},
        "delta": memory_delta(memory[DETECTOR], memory[SEGMENTER]),
        "training_memory_reused": False,
        "training_memory": TRAINING_MEMORY_LABEL,
        "training_memory_note": (
            "S0's and S1's training peaks and the smoke tests' figures are a different quantity "
            "measured under a different protocol, and are never compared with these."
        ),
        "models_trained_in_this_phase": 0,
        "limitations": [
            "Peak CUDA statistics are device-global, so each model is measured in a process "
            "that has loaded no other model. pre_load_allocated_bytes records that the "
            "allocator was empty when the model was constructed.",
            "The figures include the framework's own allocations and the CUDA context's "
            "workspaces as the allocator reports them; they are not a parameter-memory "
            "calculation.",
            "Batch 1 at imgsz 768 only. A larger batch or resolution is a different "
            "measurement and was not taken.",
        ],
        "test": {
            "status": HOLDOUT_STATUS,
            "reason": HOLDOUT_REASON,
            "images_read": 0,
            "statistics": 0,
        },
        "holdout_accessed": False,
    }


def build_complexity(paths: ProjectPaths) -> dict[str, Any]:
    """Read the committed static model-complexity figures.

    Nothing is recomputed and no architecture is instantiated to obtain these.
    The two figures came from different framework paths in different phases -
    ``get_flops`` for the detector, the framework's model summary line for the
    segmenter - and both of those default to a 640 reference input, so the
    complexity figures and the latency figures describe the models at different
    input sizes. Stated rather than silently paired.

    Args:
        paths: Project layout.

    Returns:
        The static complexity block.

    Raises:
        BenchmarkError: If a committed manifest does not carry the figures.
    """
    detector = read_json(paths.reports / "detection_D2_manifest.json")
    segmenter = read_json(paths.reports / "segmentation_S1_result_manifest.json")
    try:
        detector_block = detector["model_complexity"]
        segmenter_block = segmenter["model_complexity"]
    except KeyError as exc:
        msg = f"a committed manifest carries no model_complexity block ({exc})"
        raise BenchmarkError(msg) from exc

    return {
        "label": STATIC_MODEL_COMPLEXITY,
        "recomputed_in_this_phase": False,
        "source": "FROZEN_EXPERIMENT_ARTIFACTS",
        "measured_at_benchmark_input_size": False,
        "reference_input_size": COMPLEXITY_REFERENCE_INPUT,
        "reference_input_note": (
            "ultralytics.utils.torch_utils.get_flops and model_info both default to imgsz=640, "
            "and both committed figures came from those paths. The benchmark runs at 768, so "
            "these are not the FLOPs of the timed configuration."
        ),
        "comparable_basis": "UNFUSED_PARAMETER_COUNTS",
        "comparable_basis_note": (
            "The detector's committed count is unfused, so it is paired with the segmenter's "
            "unfused count. The segmenter's fused figures are reported alongside rather than "
            "substituted."
        ),
        DETECTOR: {
            "source_artifact": "reports/detection_D2_manifest.json",
            "source_method": detector_block.get("source"),
            "parameters": detector_block.get("parameters"),
            "gflops": detector_block.get("gflops"),
            "layers": detector_block.get("layers"),
        },
        SEGMENTER: {
            "source_artifact": "reports/segmentation_S1_result_manifest.json",
            "source_method": segmenter_block.get("source"),
            "parameters": segmenter_block.get("parameters"),
            "gflops": segmenter_block.get("gflops"),
            "layers": segmenter_block.get("layers"),
            "fused_parameters": segmenter_block.get("fused_parameters"),
            "fused_gflops": segmenter_block.get("fused_gflops"),
            "fused_layers": segmenter_block.get("fused_layers"),
        },
        "parameter_delta": (
            int(segmenter_block["parameters"]) - int(detector_block["parameters"])
            if detector_block.get("parameters") and segmenter_block.get("parameters")
            else None
        ),
        "gflops_delta_at_reference_input": (
            round(float(segmenter_block["gflops"]) - float(detector_block["gflops"]), 3)
            if detector_block.get("gflops") and segmenter_block.get("gflops")
            else None
        ),
    }


def build_report(
    latency: Mapping[str, Any], memory: Mapping[str, Any], *, commit: str | None
) -> str:
    """Render the latency and inference-memory report.

    Args:
        latency: The committed latency artifact.
        memory: The committed memory artifact.
        commit: The repository commit, when available.

    Returns:
        The report as Markdown.
    """
    lines: list[str] = []

    def add(text: str = "") -> None:
        lines.append(text)

    detector = latency["latency"][DETECTOR]
    segmenter = latency["latency"][SEGMENTER]
    deltas = latency["deltas"]
    inference_delta = deltas[MODEL_INFERENCE_LATENCY]
    end_to_end_delta = deltas[END_TO_END_LATENCY]
    runtime = latency["runtime"]
    conditions = latency["conditions"]
    benchmark = latency["benchmark"]
    complexity = latency["model_complexity"]

    add("# Detector versus segmenter - computational cost (phase 10C)")
    add()
    add(f"Status: **`{latency['status']}`** - `{latency['benchmark_status']}`")
    add()
    add(
        "This report measures only the cost half of the frozen phase 10A comparison. "
        "`FROZEN_PROTOCOL`: no model was trained, no average precision was recomputed, no "
        "spatial or association analysis was rerun, no threshold was tuned, and the holdout "
        "was not touched."
    )
    add()
    if commit:
        add(f"Repository commit at write time: `{commit}`.")
        add()

    add("## 1. Benchmark objective")
    add()
    add(
        "Phase 10A asked what the mask adds and what it costs. Phase 10B answered the first "
        "half on validation. This phase answers the second: how much latency and how much "
        "inference memory the frozen segmenter requires relative to the frozen detector, "
        "under one symmetric benchmark on one machine."
    )
    add()
    add(
        "It is still not a contest. D2 emits a class, a confidence and a box; S1 emits those "
        f"plus an instance mask. `winner_declared: {json.dumps(latency['winner_declared'])}`, "
        f"`combined_latency_score: {json.dumps(latency['combined_latency_score'])}`, and both "
        "are refused by the validator rather than merely discouraged."
    )
    add()

    add("## 2. Frozen phase 10A protocol")
    add()
    add("| Field | Value |")
    add("| --- | --- |")
    add(f"| Protocol | `{latency['protocol']}` |")
    add(f"| Configuration | `{latency['protocol_config']}` |")
    add(f"| Protocol fingerprint | `{latency['protocol_fingerprint']}` |")
    add(
        f"| Protocol adapted after results | "
        f"`{json.dumps(latency['protocol_adapted_after_results'])}` |"
    )
    add()
    add(
        "Every frozen quantity the benchmark obeys - batch, resolution, precision, warmup "
        "count, timed repetitions, benchmark membership, execution order, synchronisation and "
        "the two timing boundaries - is read from that configuration rather than restated as "
        "a constant in the runner, and the run aborts if any of them fails to take effect."
    )
    add()
    add(
        "**One gap in the frozen protocol is recorded rather than papered over.** "
        "`latency_protocol` declares no confidence threshold. `LIMITATION`: "
        + conditions["confidence_decision"]
    )
    add()

    add("## 3. Frozen model identities")
    add()
    add("| | Detector | Segmenter |")
    add("| --- | --- | --- |")
    add(
        f"| Experiment | `{latency['detector']['experiment']}` | "
        f"`{latency['segmenter']['experiment']}` |"
    )
    add(f"| Architecture | `{latency['detector']['model']}` | `{latency['segmenter']['model']}` |")
    add(f"| Input resolution | {latency['detector']['imgsz']} | {latency['segmenter']['imgsz']} |")
    add(
        f"| Checkpoint SHA-256 | `{latency['detector']['checkpoint_sha256']}` | "
        f"`{latency['segmenter']['checkpoint_sha256']}` |"
    )
    add(
        f"| Trained in this phase | "
        f"`{json.dumps(latency['detector']['trained_in_this_phase'])}` | "
        f"`{json.dumps(latency['segmenter']['trained_in_this_phase'])}` |"
    )
    add()
    add(
        "Both were resolved by digest through their freeze accessors, not by path, and the "
        "binary on disk was re-hashed before anything ran. The segmenter also carries "
        f"`overlap_mask: {json.dumps(latency['segmenter'].get('overlap_mask'))}` and "
        f"`mask_ratio: {latency['segmenter'].get('mask_ratio')}`, which is part of its "
        "identity rather than a detail: S0 and S1 are the same architecture at the same size "
        "and their checkpoints are the same number of bytes, so only the digest separates "
        "them."
    )
    add()

    add("## 4. Hardware and runtime")
    add()
    add("| Field | Value |")
    add("| --- | --- |")
    add(f"| GPU | {runtime['gpu'].get('name')} |")
    add(f"| Compute capability | {runtime['gpu'].get('compute_capability')} |")
    add(f"| Device memory | {as_gib(int(runtime['gpu'].get('total_memory_bytes', 0)))} GiB |")
    add(f"| Driver | {runtime['driver_version']} |")
    add(f"| CUDA runtime | {runtime['torch_cuda']} |")
    add(f"| torch | {runtime['torch']} |")
    add(f"| ultralytics | {runtime['ultralytics']} |")
    add(f"| Python | {runtime['python']} |")
    add(f"| Platform | {runtime['platform']} {runtime['platform_release']} |")
    add(f"| cuDNN benchmark mode | `{json.dumps(runtime['cudnn_benchmark'])}` |")
    add(f"| Power source | `{runtime['power']['source']}` |")
    add(f"| Benchmark start | {benchmark['started_at']} |")
    add(f"| Benchmark end | {benchmark['finished_at']} |")
    add()

    add("## 5. Precision parity")
    add()
    precision = latency["precision_preflight"]
    add(f"Status: **`{precision['status']}`**, frozen intent `{precision['frozen_intent']}`.")
    add()
    add("| Evidence | D2 | S1 |")
    add("| --- | --- | --- |")
    for field in (
        "backend_fp16_flag",
        "parameter_dtypes",
        "input_dtype",
        "autocast_enabled_during_forward",
        "quantization_config_present",
        "resolved_precision_value",
        "device_type",
    ):
        add(
            f"| `{field}` | `{json.dumps(precision['evidence'][DETECTOR].get(field))}` | "
            f"`{json.dumps(precision['evidence'][SEGMENTER].get(field))}` |"
        )
    add()
    add(
        "`FROZEN_PROTOCOL`: this is probed, not read off the configuration. A configuration "
        "value is an intention; the evidence above is what the runtime did with the tensor "
        "that reached the network, captured by a forward pre-hook before any timing existed. "
        "A difference between the two models would have stopped the phase as "
        f"`{PRECISION_PROTOCOL_MISMATCH}`, because a latency comparison across two precisions "
        "measures the precision."
    )
    add()

    add("## 6. Benchmark population")
    add()
    add("| Field | Value |")
    add("| --- | --- |")
    add(f"| Split | `{benchmark['split']}` |")
    add(f"| Images | {benchmark['benchmark_image_count']} |")
    add(f"| Selection rule | `{benchmark['benchmark_selection_rule']}` |")
    add(f"| Ordered membership fingerprint | `{benchmark['benchmark_membership_sha256']}` |")
    add(f"| Membership artifact | `{benchmark['benchmark_membership_artifact']}` |")
    add(f"| Drawn from validation membership | `{benchmark['validation_membership_sha256']}` |")
    add(f"| Reselected | `{json.dumps(benchmark['images_reselected'])}` |")
    add(f"| Images inspected to select | {benchmark['images_inspected_to_select']} |")
    add()
    add(
        "The subset was verified three independent ways before anything ran: the committed "
        "order reproduces the frozen ordered fingerprint, every committed rank reproduces "
        "from its own image id, and re-deriving the frozen selection rule over the frozen "
        "validation membership returns the same twenty ids in the same order. **No image was "
        "opened to choose it** - that is the point of ranking by the digest of the "
        "identifier, and it is why the subset cannot have been picked for being easy, "
        "crowded or visually interesting."
    )
    add()
    add(
        "Both models received identical decoded input: each image is decoded once with the "
        f"framework's own reader (`{conditions['decode_reader']}`) outside every timed "
        "region, and the same array is handed to both. "
        f"`identical_input_tensor_shapes: "
        f"{json.dumps(benchmark['identical_input_tensor_shapes'])}` - the run aborts if the "
        "two models ever receive differently shaped tensors for one image, because that "
        "would make the benchmark a resolution comparison."
    )
    add()

    add("## 7. Deterministic execution order")
    add()
    add("| Field | Value |")
    add("| --- | --- |")
    add(f"| Passes | `{'`, `'.join(benchmark['execution_passes'])}` |")
    add(f"| Timed blocks | {benchmark['execution_blocks']} |")
    add(f"| Interleaved | `{json.dumps(benchmark['interleaved'])}` |")
    add(f"| Symmetric | `{json.dumps(benchmark['symmetric'])}` |")
    add(f"| Randomized | `{json.dumps(benchmark['randomized'])}` |")
    add(f"| Execution-plan fingerprint | `{benchmark['execution_plan_sha256']}` |")
    add(
        f"| Order changed after observing timings | "
        f"`{json.dumps(benchmark['order_changed_after_observing_timings'])}` |"
    )
    add()
    add(
        "Pass A walks the twenty benchmark images in the frozen order and times the detector "
        "then the segmenter on each. Pass B walks the same images in the same order and times "
        "the segmenter then the detector. Both passes feed the reported distribution, so "
        "residual thermal or ordering drift falls on both models equally instead of on "
        "whichever ran second. Running one model to completion and then the other would have "
        "measured the laptop's thermal state as much as the models."
    )
    add()
    add(
        "The plan is **derived** from the frozen membership by "
        "`detector_segmenter_cost.build_execution_plan` rather than written out, so it cannot "
        "drift away from the images. Its fingerprint changes if the membership, the order or "
        "the pass structure changes, and the validator recomputes it from the executed "
        "blocks."
    )
    add()

    add("## 8. Warmup")
    add()
    add("| Field | Value |")
    add("| --- | --- |")
    add(f"| Iterations | {conditions['warmup_iterations']} |")
    add(f"| Schedule | `{conditions['warmup_schedule']}` |")
    add(f"| Path exercised | `{conditions['warmup_path']}` |")
    add(f"| Outputs discarded | `{json.dumps(conditions['warmup_outputs_discarded'])}` |")
    add(
        f"| Count changed after observing timings | "
        f"`{json.dumps(conditions['warmup_count_changed_after_observing_timings'])}` |"
    )
    add()
    add(
        "The protocol freezes *how many* warmup iterations there are, not when they run. The "
        "full twenty are applied immediately before every timed block, for both models "
        "equally - declared before the run, not chosen after seeing a distribution. A new "
        "input shape can carry a one-off kernel-selection cost, and a schedule that warmed "
        "once at the start would have pushed that cost into the first timed repetition of "
        "every image."
    )
    add()
    add(
        "Warmup exercises the **full end-to-end path**, mask reconstruction included, which "
        "is the wider of the two boundaries; the narrower model-inference boundary is a "
        "strict subset of it, so one warmup set covers both."
    )
    add()

    add("## 9. Timing methodology")
    add()
    add("| Field | Value |")
    add("| --- | --- |")
    add(f"| Batch | {conditions['batch']} |")
    add(f"| Input resolution | {conditions['imgsz']} |")
    add(
        f"| Precision | `{conditions['precision']}` "
        f"(`{conditions['precision_argument']}: {conditions['precision_value']}`) |"
    )
    add(f"| Confidence | {conditions['conf']} (`{conditions['confidence_source']}`) |")
    add(f"| NMS IoU | {conditions['iou']} |")
    add(f"| max_det | {conditions['max_det']} |")
    add(f"| Timed repetitions per block | {conditions['timed_iterations_per_image']} |")
    add(f"| Timing primitive | `{conditions['timing_primitive']}` |")
    add(f"| Synchronisation primitive | `{conditions['synchronization_primitive']}` |")
    add(
        f"| Synchronise before timed region | "
        f"`{json.dumps(conditions['cuda_synchronize_before_timed_region'])}` |"
    )
    add(
        f"| Synchronise after timed region | "
        f"`{json.dumps(conditions['cuda_synchronize_after_timed_region'])}` |"
    )
    add(
        f"| Same primitive for both models | "
        f"`{json.dumps(conditions['same_primitive_for_both_models'])}` |"
    )
    add(f"| Percentile convention | `{conditions['percentile_convention']}` |")
    add(f"| Standard-deviation convention | `{conditions['std_convention']}` |")
    add(
        f"| Model load inside timing | "
        f"`{json.dumps(not conditions['model_load_excluded_from_timing'])}` |"
    )
    add(
        f"| Disk decode inside timing | "
        f"`{json.dumps(not conditions['disk_decode_excluded_from_timing'])}` |"
    )
    add(
        f"| Saving predictions during timing | "
        f"`{json.dumps(conditions['predictions_saved_during_timing'])}` |"
    )
    add(
        f"| torch.compile / TensorRT / ONNX | `{json.dumps(conditions['torch_compile'])}` / "
        f"`{json.dumps(conditions['tensorrt'])}` / `{json.dumps(conditions['onnx'])}` |"
    )
    add()
    add(
        "**CUDA work is asynchronous**, so every timed region is bracketed by an explicit "
        "`torch.cuda.synchronize()` on both edges, with the same primitive for both models. "
        "Without it a wall-clock reading measures how long the work took to *queue*, and "
        "whichever model dispatches faster would look faster regardless of how long it "
        "actually runs."
    )
    add()
    add(
        "Both models are timed through **structurally the same framework calls**. The "
        "framework's own prediction pipeline is `preprocess`, `inference`, `postprocess`; the "
        "detector and the segmenter share the first two verbatim and differ only in which "
        "`postprocess` override runs - and `SegmentationPredictor.postprocess` is a subclass "
        "of the detector's. Nothing here benchmarks one model through a high-level API and "
        "the other through a low-level path; an inability to isolate a boundary equivalently "
        f"would have stopped the phase as `{LATENCY_BOUNDARY_BLOCKED}`."
    )
    add()
    add(
        "Lazy evaluation is defeated deliberately: before each end timestamp the outputs are "
        "materialised, so the boundary times the work rather than the construction of a "
        "wrapper object."
    )
    add()

    boundaries = latency["timing_boundaries"]
    inference_boundary = boundaries[MODEL_INFERENCE_LATENCY]
    add("## 10. `MODEL_INFERENCE_LATENCY` boundary")
    add()
    add(inference_boundary["definition"])
    add()
    add("| Field | Value |")
    add("| --- | --- |")
    add(f"| Framework call | `{inference_boundary['framework_call']}` |")
    add(
        f"| Preprocessing included | `{json.dumps(inference_boundary['preprocessing_included'])}` |"
    )
    add(
        f"| Postprocessing included | "
        f"`{json.dumps(inference_boundary['postprocessing_included'])}` |"
    )
    add(
        f"| Mask reconstruction included | "
        f"`{json.dumps(inference_boundary['mask_reconstruction_included'])}` |"
    )
    add(f"| Disk I/O included | `{json.dumps(inference_boundary['disk_io_included'])}` |")
    add(f"| Model load included | `{json.dumps(inference_boundary['model_load_included'])}` |")
    add(
        f"| Same call for both models | "
        f"`{json.dumps(inference_boundary['same_call_for_both_models'])}` |"
    )
    add()

    end_boundary = boundaries[END_TO_END_LATENCY]
    add("## 11. `END_TO_END_MODEL_OUTPUT_LATENCY` boundary")
    add()
    add(end_boundary["definition"])
    add()
    add("| Field | Value |")
    add("| --- | --- |")
    add(f"| Framework calls | `{'`, `'.join(end_boundary['framework_calls'])}` |")
    add(f"| Preprocessing included | `{json.dumps(end_boundary['preprocessing_included'])}` |")
    add(
        f"| NMS and postprocessing included | "
        f"`{json.dumps(end_boundary['postprocessing_included'])}` |"
    )
    add(
        f"| Segmenter mask reconstruction included | "
        f"`{json.dumps(end_boundary['segmenter_mask_reconstruction_included'])}` |"
    )
    add(f"| Mask reconstruction call | `{end_boundary['mask_reconstruction_call']}` |")
    add(f"| Mask reconstruction site | `{end_boundary['mask_reconstruction_site']}` |")
    add(
        f"| Outputs materialised before end timestamp | "
        f"`{json.dumps(end_boundary['outputs_materialized_before_end_timestamp'])}` |"
    )
    add(f"| Detector boundary ends at | `{end_boundary['detector_boundary_ends_at']}` |")
    add(f"| Segmenter boundary ends at | `{end_boundary['segmenter_boundary_ends_at']}` |")
    add(f"| Host transfer | `{end_boundary['host_transfer']}` |")
    add()
    add(
        "**Mask reconstruction is inside the segmenter's timer, and that is a fact read from "
        "the installed source rather than an assumption.** With `retina_masks` enabled, "
        "`SegmentationPredictor.construct_result` calls "
        "`ultralytics.utils.ops.process_mask_native`, which combines the prototypes with the "
        "per-instance coefficients and upsamples the result onto the original image canvas - "
        "all inside `postprocess`, and therefore inside this boundary. The run verifies, for "
        "every benchmark image that produced an instance, that the returned masks are on the "
        "original canvas, and aborts otherwise. Excluding this work would hide precisely the "
        "cost the comparison exists to quantify."
    )
    add()
    add(
        f"`LIMITATION`: `{end_boundary['host_transfer']}`. The frozen boundary lists "
        "preprocessing, the forward pass, NMS/postprocessing and mask reconstruction; copying "
        "the resulting tensors into host memory appears in neither its includes nor its "
        "excludes, so it is left outside the timer for **both** models. That is symmetric, "
        "but the segmenter's outputs are far larger than the detector's, so a pipeline that "
        "needs them in host memory would pay more than the figures below show. No third "
        "boundary was invented to cover it."
    )
    add()

    def table(block: Mapping[str, Any], boundary: str) -> None:
        statistics = block[boundary]
        add("| Statistic | Value (ms) |")
        add("| --- | --- |")
        for field in ("count", "mean", "median", "std", "p50", "p90", "p95", "p99", "min", "max"):
            add(f"| {field} | {statistics[field]} |")
        add(f"| images/sec from mean | {block['throughput'][boundary]} |")
        add()

    add("## 12. D2 latency")
    add()
    add("`COMPUTED_RESULT`. `MODEL_INFERENCE_LATENCY`:")
    add()
    table(detector, MODEL_INFERENCE_LATENCY)
    add("`END_TO_END_MODEL_OUTPUT_LATENCY`:")
    add()
    table(detector, END_TO_END_LATENCY)

    add("## 13. S1 latency")
    add()
    add("`COMPUTED_RESULT`. `MODEL_INFERENCE_LATENCY`:")
    add()
    table(segmenter, MODEL_INFERENCE_LATENCY)
    add("`END_TO_END_MODEL_OUTPUT_LATENCY`:")
    add()
    table(segmenter, END_TO_END_LATENCY)

    add("## 14. Absolute latency cost")
    add()
    add("| Boundary | D2 mean (ms) | S1 mean (ms) | Absolute delta (ms) |")
    add("| --- | --- | --- | --- |")
    for boundary in TIMING_BOUNDARIES:
        block = deltas[boundary]
        add(
            f"| `{boundary}` | {block['detector_mean_latency_ms']} | "
            f"{block['segmenter_mean_latency_ms']} | **{block['absolute_latency_delta_ms']}** |"
        )
    add()
    add(
        "`absolute_latency_delta_ms` is the segmenter's mean minus the detector's at an "
        "identical measurement boundary. The two boundaries are never combined into a single "
        "figure."
    )
    add()

    add("## 15. Relative latency cost")
    add()
    add("| Boundary | Relative S1 cost | Reading |")
    add("| --- | --- | --- |")
    for boundary in TIMING_BOUNDARIES:
        block = deltas[boundary]
        percent = round(100.0 * float(block["relative_latency_cost"]), 2)
        direction = "more" if percent >= 0 else "less"
        add(
            f"| `{boundary}` | **{block['relative_latency_cost']}** | S1 requires "
            f"{abs(percent)}% {direction} mean latency than D2 |"
        )
    add()
    add("`relative_latency_cost` is `(S1 mean / D2 mean) - 1`, recomputed by the validator.")
    add()

    add("## 16. Throughput")
    add()
    add("| Boundary | D2 images/sec | S1 images/sec | Throughput ratio |")
    add("| --- | --- | --- | --- |")
    for boundary in TIMING_BOUNDARIES:
        block = deltas[boundary]
        add(
            f"| `{boundary}` | {block['detector_images_per_second_from_mean']} | "
            f"{block['segmenter_images_per_second_from_mean']} | "
            f"**{block['throughput_ratio']}** |"
        )
    add()
    add(
        "`images_per_second_from_mean` is `1000 / mean_latency_ms` at **batch 1**, derived "
        "from the mean and never from the fastest repetition: one lucky iteration describes a "
        "scheduling accident, not a rate either model sustains. `LIMITATION`: this is a "
        "batch-1 latency reciprocal, not batched throughput under load, and it must not be "
        "quoted as one."
    )
    add()

    add("## 17. Mask-reconstruction cost context")
    add()
    add(
        f"The end-to-end delta is **{end_to_end_delta['absolute_latency_delta_ms']} ms** "
        f"while the model-inference delta is "
        f"**{inference_delta['absolute_latency_delta_ms']} ms**. The difference between those "
        "two numbers is where the extra segmentation work outside the forward pass lands: NMS "
        "over mask coefficients, prototype combination, and the upsample onto the original "
        "canvas."
    )
    add()
    add(
        f"`{SEGMENTATION_COST_LABEL}` is the only label this benchmark supports for the "
        "difference. It is **not** `PURE_MASK_RECONSTRUCTION_CAUSAL_COST`: "
        f"`pure_mask_reconstruction_cost_isolated: "
        f"{json.dumps(latency['pure_mask_reconstruction_cost_isolated'])}`. "
        + latency["segmentation_cost_note"]
    )
    add()

    add("## 18. Inference memory")
    add()
    add(
        f"`{INFERENCE_MEMORY}`, batch {memory['conditions']['batch']} at imgsz "
        f"{memory['conditions']['imgsz']} in `{memory['conditions']['precision']}`."
    )
    add()
    add("| Field | D2 | S1 |")
    add("| --- | --- | --- |")
    for field in (
        "pre_load_allocated_bytes",
        "baseline_allocated_bytes",
        "baseline_reserved_bytes",
        "peak_memory_allocated_bytes",
        "peak_memory_reserved_bytes",
        "peak_memory_allocated_gib",
        "peak_memory_reserved_gib",
    ):
        add(
            f"| `{field}` | {memory['memory'][DETECTOR][field]} | "
            f"{memory['memory'][SEGMENTER][field]} |"
        )
    add()
    add("| Derived | Value |")
    add("| --- | --- |")
    for field, value in sorted(memory["delta"].items()):
        if field == "measurement":
            continue
        add(f"| `{field}` | {value} |")
    add()
    add(
        "`FROZEN_PROTOCOL`: peak statistics are reset with "
        f"`{memory['conditions']['reset_call']}` **after** the frozen "
        f"{memory['conditions']['warmup_iterations']} warmup iterations, so the figure "
        "describes inference rather than the allocator's warmup high-water mark."
    )
    add()
    add(
        f"Isolation is `{memory['conditions']['isolation_method']}`, with "
        "`models_resident_during_measurement: "
        f"{memory['conditions']['models_resident_during_measurement']}`. "
        + memory["conditions"]["isolation_diagnostic"]
        + " `pre_load_allocated_bytes` is recorded as the evidence that the isolation held, "
        "rather than asserted."
    )
    add()
    add(
        f"`LIMITATION`: `{memory['training_memory']}`. S0's and S1's training peaks and the "
        "phase 8B and S1 feasibility smoke tests measured a different quantity under a "
        "different protocol; they are never compared with these figures, and "
        f"`training_memory_reused: {json.dumps(memory['training_memory_reused'])}`."
    )
    add()

    add("## 19. Static model complexity")
    add()
    add(
        f"`{complexity['label']}`, `recomputed_in_this_phase: "
        f"{json.dumps(complexity['recomputed_in_this_phase'])}`."
    )
    add()
    add("| Field | D2 | S1 |")
    add("| --- | --- | --- |")
    add(
        f"| Parameters | {complexity[DETECTOR]['parameters']} | "
        f"{complexity[SEGMENTER]['parameters']} |"
    )
    add(f"| GFLOPs | {complexity[DETECTOR]['gflops']} | {complexity[SEGMENTER]['gflops']} |")
    add(f"| Layers | {complexity[DETECTOR]['layers']} | {complexity[SEGMENTER]['layers']} |")
    add(f"| Fused parameters | - | {complexity[SEGMENTER]['fused_parameters']} |")
    add(f"| Fused GFLOPs | - | {complexity[SEGMENTER]['fused_gflops']} |")
    add(
        f"| Source method | `{complexity[DETECTOR]['source_method']}` | "
        f"`{complexity[SEGMENTER]['source_method']}` |"
    )
    add(
        f"| Source artifact | `{complexity[DETECTOR]['source_artifact']}` | "
        f"`{complexity[SEGMENTER]['source_artifact']}` |"
    )
    add()
    add(
        f"Parameter delta: **{complexity['parameter_delta']}**. GFLOPs delta at the reference "
        f"input: **{complexity['gflops_delta_at_reference_input']}**."
    )
    add()
    add(
        f"`LIMITATION`: `measured_at_benchmark_input_size: "
        f"{json.dumps(complexity['measured_at_benchmark_input_size'])}`, "
        f"`reference_input_size: {complexity['reference_input_size']}`. "
        + complexity["reference_input_note"]
        + " The two figures also came from different framework paths in different phases. "
        + complexity["comparable_basis_note"]
        + " These are static architecture counts, not runtime measurements, and they must not "
        "be read as an explanation of the latency figures above."
    )
    add()

    add("## 20. Thermal and power limitations")
    add()
    add(f"`{latency['benchmark_status']}`. `LIMITATION`, and not a small one:")
    add()
    add(
        f"- Power source at benchmark time: `{runtime['power']['source']}`. "
        "`power_settings_changed_by_this_phase: "
        f"{json.dumps(runtime['power_settings_changed_by_this_phase'])}` - no Windows power "
        "plan, GPU clock, fan curve, undervolt or performance mode was altered for this "
        "phase."
    )
    add(
        "- `thermal_correction_applied: "
        f"{json.dumps(runtime['thermal_correction_applied'])}`. No thermal-correction "
        "mathematics was introduced. A laptop GPU throttles, and the symmetric interleaved "
        "order is the only mitigation applied: it spreads drift across both models rather "
        "than removing it."
    )
    add(
        "- These numbers are valid for this machine, this driver, this runtime and this "
        "protocol. **No claim of hardware-independent latency is made or supported.**"
    )
    add()
    diagnostic = latency["post_hoc_distribution_diagnostic"]
    stability = latency["protocol_stability_after_observation"]
    add(
        f"**The observed latency distribution is wide, and the mean alone would mislead.** "
        f"`{diagnostic['label']}` / `{diagnostic['status']}` - written after the benchmark ran, "
        "so it is a diagnostic and not a finding. It is computed from "
        f"`{diagnostic['computed_from']}` ({diagnostic['observations_used']} observations), "
        f"`observations_discarded: {diagnostic['observations_discarded']}`, and it "
        f"`replaces_frozen_statistics: "
        f"{json.dumps(diagnostic['replaces_frozen_statistics'])}`."
    )
    add()

    def short_name(boundary: str) -> str:
        return "inference" if boundary == MODEL_INFERENCE_LATENCY else "end-to-end"

    add("| Evidence | Boundary | D2 | S1 |")
    add("| --- | --- | --- | --- |")
    for field in (
        "mean",
        "median",
        "mean_to_median_ratio",
        "p90",
        "min",
        "max",
        "timed_blocks",
        "block_mean_min",
        "block_mean_max",
    ):
        for boundary in TIMING_BOUNDARIES:
            detector_value = diagnostic["evidence"][DETECTOR][boundary][field]
            segmenter_value = diagnostic["evidence"][SEGMENTER][boundary][field]
            add(f"| `{field}` | {short_name(boundary)} | {detector_value} | {segmenter_value} |")
    add()
    add("| Per-pass mean of block means | Boundary | D2 | S1 |")
    add("| --- | --- | --- | --- |")
    for boundary in TIMING_BOUNDARIES:
        for execution_pass in EXECUTION_PASSES:
            detector_value = diagnostic["evidence"][DETECTOR][boundary][
                "per_pass_block_mean_of_means"
            ][execution_pass]
            segmenter_value = diagnostic["evidence"][SEGMENTER][boundary][
                "per_pass_block_mean_of_means"
            ][execution_pass]
            add(
                f"| `{execution_pass}` | {short_name(boundary)} | {detector_value} | "
                f"{segmenter_value} |"
            )
    add()
    add(diagnostic["reading"])
    add()
    add(
        f"**Both models show the same kind of skew, and that is all that is claimed.** "
        f"`proportionality_across_models_demonstrated: "
        f"{json.dumps(diagnostic['proportionality_across_models_demonstrated'])}`. "
        + diagnostic["proportionality_note"]
    )
    add()
    add(
        f"**`LIMITATION` `{NO_POWER_TELEMETRY}`.** "
        f"`telemetry_synchronous_with_timed_blocks: "
        f"{json.dumps(diagnostic['telemetry_synchronous_with_timed_blocks'])}`. "
        + diagnostic["telemetry_note"]
    )
    add()
    add(
        f"So `causal_attribution: {diagnostic['causal_attribution']}` and the mechanism stays "
        f"`{diagnostic['hypothesis_status']}`. "
        + diagnostic["hypothesis"]
        + " Mobile-GPU DVFS and power-state behaviour contributes to the observed latency "
        "distribution on this machine; which observation sat at which device state is not "
        "something this benchmark can say."
    )
    add()
    add(
        "**Nothing in the protocol was adapted after the timings were seen**, which is exactly "
        "when a frozen protocol earns its keep:"
    )
    add()
    add("| After observing the distribution | Value |")
    add("| --- | --- |")
    for field in (
        "benchmark_subset_changed",
        "warmup_iterations_changed",
        "timed_repetitions_changed",
        "execution_order_changed",
        "precision_changed",
        "batch_changed",
        "imgsz_changed",
        "timing_boundaries_changed",
        "observations_removed",
        "outlier_rejection_introduced",
        "observations_normalized_or_rescaled",
        "power_clock_or_fan_setting_changed",
        "benchmark_rerun",
        "statistical_definitions_changed",
    ):
        add(f"| `{field}` | `{json.dumps(stability[field])}` |")
    add()
    add(stability["note"])
    add()
    add(
        "The symmetric interleaved order was the one mitigation the frozen protocol provided, "
        "and the per-pass table above is what there is to say about how it behaved. It spreads "
        "the machine's state across both models rather than removing it: the means in sections "
        "14 to 16 carry real machine variance, and a difference of this size measured on a "
        "quieter machine could look different."
    )
    add()

    add("## 21. Relationship to phase 10B")
    add()
    add(
        "Phase 10B's numbers are referred to here only as context, and **none was "
        f"recomputed**: `ap_metrics_recomputed: "
        f"{json.dumps(latency['ap_metrics_recomputed'])}`, `spatial_analysis_rerun: "
        f"{json.dumps(latency['spatial_analysis_rerun'])}`, `association_analysis_rerun: "
        f"{json.dumps(latency['association_analysis_rerun'])}`, `mask_iou_diagnostic_rerun: "
        f"{json.dumps(latency['mask_iou_diagnostic_rerun'])}`."
    )
    add()
    add(
        "For the benefit half of the trade-off, read "
        "`reports/detector_segmenter_validation_comparison.md`. Its committed reading stands "
        "unchanged: on canonical box localisation S1 retains broadly similar capability to "
        "D2, the positive all-class delta is carried by the highly uncertain `vest_loose` "
        "class, and excluding that class the descriptive support sensitivity puts S1 "
        "**below** D2. What masks demonstrably added there was representation - a third of "
        "the median predicted box is not the object - rather than new person-PPE association "
        "discovery at the frozen containment floor."
    )
    add()
    add(
        "This phase adds the cost side of that same trade-off and nothing else. It changes no "
        "frozen model, no frozen protocol and no phase 10B number."
    )
    add()

    add("## 22. Holdout compliance")
    add()
    add(f"`HOLDOUT_POLICY`: **`{latency['test']['status']}`**.")
    add()
    add(latency["test"]["reason"])
    add()
    add("| Field | Value |")
    add("| --- | --- |")
    add(f"| Holdout images read | {latency['test']['images_read']} |")
    add(f"| Holdout predictions | {latency['test']['predictions']} |")
    add(f"| Holdout timings | {latency['test']['timings']} |")
    add(f"| Holdout statistics | {latency['test']['statistics']} |")
    add(f"| `holdout_accessed` | `{json.dumps(latency['holdout_accessed'])}` |")
    add(
        f"| Raw timings store holdout identifiers | "
        f"`{json.dumps(latency['raw_timings']['holdout_identifiers_stored'])}` |"
    )
    add()
    add(
        f"The runner refuses to start at all when `{HOLDOUT_UNLOCK_ENV_VAR}` is set, and the "
        "benchmark subset is drawn from the frozen validation membership by fingerprint, so "
        "there is no path by which a holdout image could enter it."
    )
    add()

    add("## 23. Interpretation limits")
    add()
    add("What this phase measured, and only this:")
    add()
    for limitation in latency["limitations"]:
        add(f"- `LIMITATION` {limitation}")
    add()
    add(
        "What it did **not** establish: that the latency difference is caused by mask "
        "reconstruction alone; that either model is faster in general or on other hardware; "
        "that either model should be preferred; or anything at all about test performance. "
        "`winner_declared: false` and `combined_latency_score: false` remain refused by both "
        "parsers, and neither frozen model changed."
    )
    add()

    add("## 24. Next phase")
    add()
    add(
        "Phase 10D - the operational and scientific synthesis - reads phase 10B's benefit "
        "figures alongside this phase's cost figures and states the trade-off. It has **not** "
        "started, and no synthesis, recommendation or deployment reading is offered here."
    )
    add()
    add("The holdout stays locked until phase 11.")
    add()

    add("## Fingerprints")
    add()
    add("| Artifact | Fingerprint |")
    add("| --- | --- |")
    add(f"| Phase 10A protocol | `{latency['protocol_fingerprint']}` |")
    add(f"| Benchmark membership (ordered) | `{benchmark['benchmark_membership_sha256']}` |")
    add(f"| Execution plan | `{benchmark['execution_plan_sha256']}` |")
    add(f"| Raw timing dataset | `{latency['raw_timings']['timing_dataset_sha256']}` |")
    add(f"| Latency result | `{latency['latency_result_sha256']}` |")
    add(f"| Memory result | `{memory['memory_result_sha256']}` |")
    add()
    add(
        f"Raw observations: **{latency['raw_timings']['observations']}** timed readings "
        f"({latency['raw_timings']['observations_per_model_per_boundary']} per model per "
        f"boundary), row-level in `{latency['raw_timings']['row_level_artifact']}` "
        "(git-ignored) and summarised per block in "
        f"`{latency['raw_timings']['block_summary_artifact']}`."
    )
    add()

    return "\n".join(lines) + "\n"


# --- entry point ------------------------------------------------------------------------


def load_persisted_observations(paths: ProjectPaths) -> list[dict[str, Any]]:
    """Read the raw timings this phase already recorded.

    Args:
        paths: Project layout.

    Returns:
        The observations, in the executed order, typed as recorded.

    Raises:
        BenchmarkError: If the file is absent or short.
    """
    table = paths.root / RUNTIME_ROOT / "latency_observations.csv"
    if not table.is_file():
        msg = (
            f"the persisted raw timings are missing: {RUNTIME_ROOT}/latency_observations.csv. "
            "They cannot be regenerated without re-timing, which this mode exists to avoid."
        )
        raise BenchmarkError(msg)
    with Path(long_path(table)).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != expected_observation_count():
        msg = (
            f"the persisted raw timings hold {len(rows)} observations, not the expected "
            f"{expected_observation_count()}"
        )
        raise BenchmarkError(msg)
    return [
        {
            "model": row["model"],
            "benchmark_rank": int(row["benchmark_rank"]),
            "source_image_id": row["source_image_id"],
            "execution_pass": row["execution_pass"],
            "execution_position": int(row["execution_position"]),
            "repetition": int(row["repetition"]),
            "timing_type": row["timing_type"],
            "latency_ms": float(row["latency_ms"]),
        }
        for row in rows
    ]


def rebuild_results() -> int:
    """Re-derive the result artifacts from the persisted raw timings.

    **No model is executed and no measurement is taken.** The complete set of
    timed observations this phase recorded is read back, every frozen statistic
    and derived quantity is recomputed from it under the unchanged definitions,
    and the recomputed values are asserted identical to the committed ones
    before anything is written. That assertion is the point: it proves the
    rebuild changed presentation and not measurement.

    Fields that can only be measured inside the benchmark process - the
    precision evidence, the runtime and hardware provenance, the output
    evidence, the benchmark timestamps and the whole memory measurement - are
    carried forward from the committed artifacts rather than re-derived. A
    rebuild may never downgrade measured evidence.

    Returns:
        Process exit status.
    """
    paths = ProjectPaths.from_root()
    if holdout_unlocked():
        print(f"REFUSED: {HOLDOUT_UNLOCK_ENV_VAR} is set", file=sys.stderr)
        return 2

    try:
        committed_latency = read_json(paths.reports / LATENCY_JSON)
        committed_memory = read_json(paths.reports / MEMORY_JSON)
        protocol = load_comparison_protocol(paths.configs / CONFIG_YAML)
        if protocol.fingerprint() != EXPECTED_PROTOCOL:
            msg = "the comparison protocol is not the frozen one"
            raise BenchmarkError(msg)
        subset = load_benchmark_subset(paths, protocol)
        plan = build_execution_plan(list(subset["image_ids"]))
        plan_fingerprint = execution_plan_fingerprint(plan)
        observations = load_persisted_observations(paths)
        digest = timing_dataset_fingerprint(observations)
        if digest != committed_latency["raw_timings"]["timing_dataset_sha256"]:
            msg = (
                "the persisted raw timings do not reproduce the committed timing fingerprint, "
                f"so they are not the timings behind the committed result: {digest}"
            )
            raise BenchmarkError(msg)
        complexity = build_complexity(paths)
    except (ConfigError, ComparisonProtocolError, CostBenchmarkError, BenchmarkError) as exc:
        print(f"{BLOCKED}: {exc}", file=sys.stderr)
        return 2

    blocks: list[dict[str, Any]] = []
    for step in plan:
        for boundary in TIMING_BOUNDARIES:
            values = [
                row["latency_ms"]
                for row in observations
                if row["execution_position"] == step.position and row["timing_type"] == boundary
            ]
            statistics = describe_latency(values)
            blocks.append(
                {
                    "execution_position": step.position,
                    "execution_pass": step.execution_pass,
                    "benchmark_rank": step.benchmark_rank,
                    "source_image_id": step.source_image_id,
                    "model": step.model,
                    "timing_type": boundary,
                    "count": statistics["count"],
                    "mean": statistics["mean"],
                    "median": statistics["median"],
                    "std": statistics["std"],
                    "min": statistics["min"],
                    "max": statistics["max"],
                }
            )

    # A carried-forward runtime block from an earlier artifact version may predate these
    # fields. They record what the code did *not* sample, which is knowable without
    # measuring anything, so backfilling them downgrades no evidence.
    carried_runtime = dict(committed_latency["runtime"])
    carried_runtime.setdefault("per_observation_power_state_telemetry", NO_POWER_TELEMETRY)
    carried_runtime.setdefault("telemetry_synchronous_with_timed_blocks", False)
    carried_runtime.setdefault("telemetry_note", TELEMETRY_NOTE)

    execution = {
        "observations": observations,
        "blocks": blocks,
        "plan": plan,
        "resolved_arguments": committed_latency["conditions"]["resolved_arguments"],
        "input_tensor_shapes": committed_latency["benchmark"]["input_tensor_shapes"],
        "identical_input_tensor_shapes": committed_latency["benchmark"][
            "identical_input_tensor_shapes"
        ],
        "output_evidence": committed_latency["output_evidence"],
        "started_at": committed_latency["benchmark"]["started_at"],
        "finished_at": committed_latency["benchmark"]["finished_at"],
    }

    latency_payload = build_latency_artifact(
        protocol=protocol,
        models={
            DETECTOR: committed_latency["detector"],
            SEGMENTER: committed_latency["segmenter"],
        },
        subset=subset,
        execution=execution,
        precision=committed_latency["precision_preflight"],
        runtime=carried_runtime,
        complexity=complexity,
        settings=protocol.operational_inference,
    )

    # The proof that this rebuilt presentation and not measurement.
    unchanged = ("latency", "deltas")
    drifted = [key for key in unchanged if latency_payload[key] != committed_latency[key]]
    if drifted:
        print(
            f"{PROTOCOL_VIOLATION}: recomputing from the persisted timings changed {drifted}. "
            "A rebuild may not alter a measured value.",
            file=sys.stderr,
        )
        return 2
    if (
        latency_payload["raw_timings"]["timing_dataset_sha256"]
        != (committed_latency["raw_timings"]["timing_dataset_sha256"])
    ):
        print(f"{PROTOCOL_VIOLATION}: the raw timing fingerprint moved", file=sys.stderr)
        return 2

    memory_payload = {
        key: value for key, value in committed_memory.items() if key != "memory_result_sha256"
    }
    memory_payload["runtime"] = carried_runtime
    memory_payload["measured_fields_carried_forward"] = sorted(MEMORY_FIELDS)
    memory_payload["remeasured_in_this_rebuild"] = False

    latency_payload["latency_result_sha256"] = latency_result_fingerprint(latency_payload)
    memory_payload["memory_result_sha256"] = memory_result_fingerprint(memory_payload)

    problems = validate_latency_result(
        latency_payload,
        protocol_fingerprint=protocol.fingerprint(),
        detector_sha256=committed_latency["detector"]["checkpoint_sha256"],
        segmenter_sha256=committed_latency["segmenter"]["checkpoint_sha256"],
        benchmark_sha256=subset["benchmark_membership_sha256"],
        execution_plan_sha256=plan_fingerprint,
    ) + validate_memory_result(
        memory_payload,
        protocol_fingerprint=protocol.fingerprint(),
        detector_sha256=committed_latency["detector"]["checkpoint_sha256"],
        segmenter_sha256=committed_latency["segmenter"]["checkpoint_sha256"],
    )
    if problems:
        print(f"{PROTOCOL_VIOLATION}: the rebuilt artifacts do not validate:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 2

    report = build_report(latency_payload, memory_payload, commit=git_commit(paths.root))
    blocks_csv = render_rows(blocks, BLOCK_FIELDS)
    findings = (
        scan_for_sensitive(report)
        + scan_for_sensitive(blocks_csv)
        + scan_for_sensitive(dumps(latency_payload))
        + scan_for_sensitive(dumps(memory_payload))
    )
    if findings:
        print(f"{BLOCKED}: sensitive content in the artifacts: {findings}", file=sys.stderr)
        return 2

    write_text(paths.reports / LATENCY_JSON, dumps(latency_payload))
    write_text(paths.reports / MEMORY_JSON, dumps(memory_payload))
    write_text(paths.reports / BLOCKS_CSV, blocks_csv)
    write_text(paths.reports / REPORT_MD, report)

    record = ProvenanceRecord.create(
        name="detector_segmenter_cost_benchmark",
        phase=10,
        config={
            "detector_segmenter_comparison": f"configs/{CONFIG_YAML}",
            "protocol_fingerprint": protocol.fingerprint(),
        },
        details={
            "phase": PHASE,
            "classification": COST_BENCHMARK_COMPLETE,
            "benchmark_status": BENCHMARK_LABEL,
            "rebuilt_from_persisted_timings": True,
            "models_executed_in_this_rebuild": 0,
            "timings_taken_in_this_rebuild": 0,
            "benchmark_rerun": False,
            "observations_discarded": 0,
            "frozen_statistics_unchanged": True,
            "statistical_definitions_changed": False,
            "detector_checkpoint_sha256": committed_latency["detector"]["checkpoint_sha256"],
            "segmenter_checkpoint_sha256": committed_latency["segmenter"]["checkpoint_sha256"],
            "benchmark_membership_sha256": subset["benchmark_membership_sha256"],
            "execution_plan_sha256": plan_fingerprint,
            "timing_dataset_sha256": digest,
            "latency_result_sha256": latency_payload["latency_result_sha256"],
            "memory_result_sha256": memory_payload["memory_result_sha256"],
            "timed_observations": len(observations),
            "models_trained": 0,
            "thresholds_tuned": 0,
            "ap_metrics_recomputed": False,
            "spatial_analysis_rerun": False,
            "holdout_accessed": False,
        },
        repo_root=paths.root,
    )
    record.add_input(paths.configs / CONFIG_YAML, relative_to=paths.root)
    for name in (
        "final_detector_manifest.json",
        "final_segmenter_manifest.json",
        PROTOCOL_JSON,
        BENCHMARK_CSV,
    ):
        record.add_input(paths.reports / name, relative_to=paths.root)
    for name in (LATENCY_JSON, MEMORY_JSON, BLOCKS_CSV, REPORT_MD):
        record.add_output(paths.reports / name, relative_to=paths.root)
    record.write_json(paths.reports / PROVENANCE_JSON)

    print("RESULTS_REBUILT_FROM_PERSISTED_TIMINGS")
    print("no model executed, no timing taken, no observation discarded")
    print(f"timings sha {digest}  (unchanged)")
    print("frozen latency statistics and deltas recomputed identical")
    print(f"latency sha {latency_payload['latency_result_sha256']}")
    print(f"memory sha  {memory_payload['memory_result_sha256']}")
    print(f"report      reports/{REPORT_MD}")
    return 0


def rebuild_report() -> int:
    """Re-render the report from the committed result artifacts.

    Nothing is timed, measured, inferred or recomputed: the two committed JSON
    artifacts are the only inputs, and both are asserted byte-identical
    afterwards. This exists so a prose correction never requires a second
    benchmark - re-timing after seeing a result is the move the frozen
    repetition count exists to prevent.

    Returns:
        Process exit status.
    """
    paths = ProjectPaths.from_root()
    latency_path = paths.reports / LATENCY_JSON
    memory_path = paths.reports / MEMORY_JSON
    try:
        latency_payload = read_json(latency_path)
        memory_payload = read_json(memory_path)
    except BenchmarkError as exc:
        print(f"{BLOCKED}: {exc}", file=sys.stderr)
        return 2

    before = (sha256_file(latency_path), sha256_file(memory_path))
    report = build_report(latency_payload, memory_payload, commit=git_commit(paths.root))
    findings = scan_for_sensitive(report)
    if findings:
        print(f"{BLOCKED}: sensitive content in the report: {findings}", file=sys.stderr)
        return 2
    write_text(paths.reports / REPORT_MD, report)

    after = (sha256_file(latency_path), sha256_file(memory_path))
    if before != after:
        print(
            f"{PROTOCOL_VIOLATION}: a result artifact changed during a report rebuild",
            file=sys.stderr,
        )
        return 2

    print("REPORT_REBUILT_FROM_COMMITTED_RESULTS")
    print("nothing timed, measured or recomputed; both result artifacts byte-identical")
    print(f"latency sha {latency_payload['latency_result_sha256']}")
    print(f"memory sha  {memory_payload['memory_result_sha256']}")
    print(f"report      reports/{REPORT_MD}")
    return 0


def run_memory_child(experiment: str) -> int:
    """Measure one model's inference memory and print it as JSON.

    Args:
        experiment: Which frozen model to measure.

    Returns:
        Process exit status.
    """
    paths = ProjectPaths.from_root()
    if holdout_unlocked():
        print(f"REFUSED: {HOLDOUT_UNLOCK_ENV_VAR} is set", file=sys.stderr)
        return 2
    try:
        protocol = load_comparison_protocol(paths.configs / CONFIG_YAML)
        if protocol.fingerprint() != EXPECTED_PROTOCOL:
            msg = "the comparison protocol is not the frozen one"
            raise BenchmarkError(msg)
        subset = load_benchmark_subset(paths, protocol)
        if experiment == DETECTOR:
            checkpoint = detector_checkpoint(load_final_detector(paths.reports), paths.root)
            declared = protocol["detector"]["checkpoint_sha256"]
        else:
            checkpoint = segmenter_checkpoint(load_final_segmenter(paths.reports), paths.root)
            declared = protocol["segmenter"]["checkpoint_sha256"]
        if sha256_file(checkpoint) != declared:
            msg = f"{MODEL_IDENTITY_MISMATCH}: the {experiment} checkpoint is not the frozen one"
            raise BenchmarkError(msg)
        measured = measure_memory(
            checkpoint,
            subset,
            protocol.operational_inference,
            masks=experiment == SEGMENTER,
        )
    except (
        ConfigError,
        ComparisonProtocolError,
        FinalDetectorError,
        FinalSegmenterError,
        BenchmarkError,
        MemoryIsolationError,
        CostBenchmarkError,
    ) as exc:
        print(f"{BLOCKED}: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(measured, sort_keys=True))
    return 0


def main(argv: list[str] | None = None) -> int:
    """Run the controlled cost benchmark.

    Args:
        argv: Command-line arguments.

    Returns:
        Process exit status.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true", help="check preconditions only")
    parser.add_argument(
        "--preflight-only", action="store_true", help="verify precision parity and stop"
    )
    parser.add_argument(
        "--measure-memory",
        choices=MODELS,
        help="internal: measure one model's inference memory in this process and print JSON",
    )
    parser.add_argument(
        "--rebuild-report",
        action="store_true",
        help=(
            "re-render the report from the committed result artifacts without timing, "
            "measuring or recomputing anything"
        ),
    )
    parser.add_argument(
        "--rebuild-results",
        action="store_true",
        help=(
            "re-derive the result artifacts from the persisted raw timings, executing no "
            "model and taking no new measurement"
        ),
    )
    args = parser.parse_args(argv)

    if args.measure_memory:
        return run_memory_child(args.measure_memory)
    if args.rebuild_report:
        return rebuild_report()
    if args.rebuild_results:
        return rebuild_results()

    paths = ProjectPaths.from_root()
    if holdout_unlocked():
        print(
            f"REFUSED: {HOLDOUT_UNLOCK_ENV_VAR} is set. This phase benchmarks the validation "
            "subset only and declines to run in an environment where the holdout is unlocked.",
            file=sys.stderr,
        )
        return 2

    try:
        historical = historical_digests(paths)
        protocol = load_comparison_protocol(paths.configs / CONFIG_YAML)
        if protocol.fingerprint() != EXPECTED_PROTOCOL:
            msg = (
                f"the comparison protocol hashes to {protocol.fingerprint()}, not the frozen "
                f"{EXPECTED_PROTOCOL}"
            )
            raise BenchmarkError(msg)

        frozen_manifest = read_json(paths.reports / PROTOCOL_JSON)
        detector_identity = load_final_detector(paths.reports)
        segmenter_identity = load_final_segmenter(paths.reports)
        checkpoints = {
            DETECTOR: detector_checkpoint(detector_identity, paths.root),
            SEGMENTER: segmenter_checkpoint(segmenter_identity, paths.root),
        }
        for experiment, identity, declared in (
            (DETECTOR, detector_identity, protocol["detector"]["checkpoint_sha256"]),
            (SEGMENTER, segmenter_identity, protocol["segmenter"]["checkpoint_sha256"]),
        ):
            if identity.checkpoint_sha256 != declared:
                msg = f"the frozen {experiment} checkpoint is not the one the protocol names"
                raise ModelIdentityError(msg)
            on_disk = sha256_file(checkpoints[experiment])
            if on_disk != declared:
                msg = (
                    f"the {experiment} binary on disk hashes to {on_disk}, not the frozen "
                    f"{declared}"
                )
                raise ModelIdentityError(msg)

        subset = load_benchmark_subset(paths, protocol)
        plan = build_execution_plan(list(subset["image_ids"]))
        plan_fingerprint = execution_plan_fingerprint(plan)

        problems = validate_protocol_manifest(
            frozen_manifest,
            protocol_fingerprint=protocol.fingerprint(),
            detector_sha256=protocol["detector"]["checkpoint_sha256"],
            segmenter_sha256=protocol["segmenter"]["checkpoint_sha256"],
            membership_sha256=subset["validation_membership_sha256"],
            benchmark_sha256=subset["benchmark_membership_sha256"],
        )
        if problems:
            msg = "the frozen phase 10A manifest does not validate: " + "; ".join(problems)
            raise BenchmarkError(msg)

        complexity = build_complexity(paths)
        models = {
            DETECTOR: {
                "experiment": detector_identity.selected_experiment,
                "model": detector_identity.model,
                "imgsz": detector_identity.imgsz,
                "checkpoint_sha256": detector_identity.checkpoint_sha256,
                "identity_fingerprint": detector_identity.fingerprint,
                "trained_in_this_phase": False,
                "modified_in_this_phase": False,
            },
            SEGMENTER: {
                "experiment": segmenter_identity.selected_experiment,
                "model": segmenter_identity.model,
                "imgsz": segmenter_identity.imgsz,
                "overlap_mask": segmenter_identity.overlap_mask,
                "mask_ratio": segmenter_identity.mask_ratio,
                "checkpoint_sha256": segmenter_identity.checkpoint_sha256,
                "identity_fingerprint": segmenter_identity.fingerprint,
                "trained_in_this_phase": False,
                "modified_in_this_phase": False,
            },
        }
    except (
        ConfigError,
        ComparisonProtocolError,
        FinalDetectorError,
        FinalSegmenterError,
        CostBenchmarkError,
        BenchmarkError,
    ) as exc:
        classification = MODEL_IDENTITY_MISMATCH if isinstance(exc, ModelIdentityError) else BLOCKED
        print(f"{classification}: {exc}", file=sys.stderr)
        return 2

    settings = protocol.operational_inference
    print(f"protocol   {protocol.fingerprint()}  VERIFIED")
    print(f"detector   {DETECTOR}  {models[DETECTOR]['checkpoint_sha256'][:16]}...")
    print(f"segmenter  {SEGMENTER}  {models[SEGMENTER]['checkpoint_sha256'][:16]}...")
    print(
        f"benchmark  {len(subset['image_ids'])} validation images  "
        f"{subset['benchmark_membership_sha256'][:16]}..."
    )
    print(f"plan       {len(plan)} blocks  {plan_fingerprint[:16]}...")
    print(
        f"conditions batch {LATENCY_BATCH}  imgsz {COMPARISON_IMGSZ}  {PRECISION}  "
        f"conf {settings['conf']}  warmup {WARMUP_ITERATIONS}  "
        f"reps {TIMED_ITERATIONS_PER_IMAGE}"
    )

    if args.verify_only:
        print("VERIFIED: preconditions hold. Nothing timed, measured or written.")
        print(f"holdout    {HOLDOUT_STATUS}")
        return 0

    try:
        runtime = runtime_provenance()
        from ultralytics.utils.patches import imread

        probe = imread(long_path(subset["files"][subset["image_ids"][0]]))
        captured = {
            experiment: precision_evidence(checkpoints[experiment], settings, probe.copy())
            for experiment in MODELS
        }
        precision = assert_precision_parity(captured)
    except PrecisionParityError as exc:
        print(f"{exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"{BLOCKED}: the precision preflight could not run ({exc})", file=sys.stderr)
        return 2

    evidence = precision["evidence"][DETECTOR]
    print(
        f"precision  {precision['status']}  params {evidence['parameter_dtypes']}  input "
        f"{evidence['input_dtype']}  autocast {evidence['autocast_enabled_during_forward']}"
    )
    print(f"gpu        {runtime['gpu'].get('name')}  driver {runtime['driver_version']}")

    if args.preflight_only:
        print("PREFLIGHT ONLY: precision parity verified. Nothing timed or written.")
        return 0

    try:
        execution = benchmark_latency(checkpoints, subset, settings)
    except BoundaryError as exc:
        print(f"{LATENCY_BOUNDARY_BLOCKED}: {exc}", file=sys.stderr)
        return 2
    except (BenchmarkError, CostBenchmarkError) as exc:
        print(f"{BLOCKED}: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"{BENCHMARK_EXECUTION_FAILED}: {exc}", file=sys.stderr)
        return 2

    if execution_plan_fingerprint(execution["plan"]) != plan_fingerprint:
        print(f"{PROTOCOL_VIOLATION}: the executed order is not the derived plan", file=sys.stderr)
        return 2
    print(
        f"timed      {len(execution['observations'])} observations over "
        f"{len(execution['blocks']) // len(TIMING_BOUNDARIES)} blocks"
    )

    try:
        memory = {experiment: memory_in_child(paths, experiment) for experiment in MODELS}
    except BenchmarkError as exc:
        print(f"{MEMORY_MEASUREMENT_BLOCKED}: {exc}", file=sys.stderr)
        return 2
    print(
        f"memory     D2 {memory[DETECTOR]['peak_memory_reserved_gib']} GiB reserved  "
        f"S1 {memory[SEGMENTER]['peak_memory_reserved_gib']} GiB reserved"
    )

    latency_payload = build_latency_artifact(
        protocol=protocol,
        models=models,
        subset=subset,
        execution=execution,
        precision=precision,
        runtime=runtime,
        complexity=complexity,
        settings=settings,
    )
    memory_payload = build_memory_artifact(
        protocol=protocol,
        models=models,
        memory=memory,
        runtime=runtime,
        settings=settings,
    )
    latency_payload["latency_result_sha256"] = latency_result_fingerprint(latency_payload)
    memory_payload["memory_result_sha256"] = memory_result_fingerprint(memory_payload)

    problems = validate_latency_result(
        latency_payload,
        protocol_fingerprint=protocol.fingerprint(),
        detector_sha256=models[DETECTOR]["checkpoint_sha256"],
        segmenter_sha256=models[SEGMENTER]["checkpoint_sha256"],
        benchmark_sha256=subset["benchmark_membership_sha256"],
        execution_plan_sha256=plan_fingerprint,
    ) + validate_memory_result(
        memory_payload,
        protocol_fingerprint=protocol.fingerprint(),
        detector_sha256=models[DETECTOR]["checkpoint_sha256"],
        segmenter_sha256=models[SEGMENTER]["checkpoint_sha256"],
    )
    if problems:
        print(f"{PROTOCOL_VIOLATION}: the emitted artifacts do not validate:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 2

    report = build_report(latency_payload, memory_payload, commit=git_commit(paths.root))
    blocks_csv = render_rows(execution["blocks"], BLOCK_FIELDS)
    observations_csv = render_rows(execution["observations"], OBSERVATION_FIELDS)

    findings = (
        scan_for_sensitive(report)
        + scan_for_sensitive(blocks_csv)
        + scan_for_sensitive(dumps(latency_payload))
        + scan_for_sensitive(dumps(memory_payload))
    )
    if findings:
        print(f"{BLOCKED}: sensitive content in the emitted artifacts: {findings}", file=sys.stderr)
        return 2

    latency_sha = write_text(paths.reports / LATENCY_JSON, dumps(latency_payload))
    memory_sha = write_text(paths.reports / MEMORY_JSON, dumps(memory_payload))
    write_text(paths.reports / BLOCKS_CSV, blocks_csv)
    write_text(paths.reports / REPORT_MD, report)

    # Row-level observations stay in ignored runtime space; the committed
    # artifacts carry the aggregates, the per-block summaries and the
    # fingerprint that ties them to these timings.
    runtime_root = paths.root / RUNTIME_ROOT
    runtime_root.mkdir(parents=True, exist_ok=True)
    (runtime_root / "latency_observations.csv").write_text(
        observations_csv, encoding="utf-8", newline="\n"
    )

    after = historical_digests(paths)
    changed = sorted(name for name in historical if historical[name] != after.get(name))
    if changed:
        print(f"{PROTOCOL_VIOLATION}: historical artifacts changed: {changed}", file=sys.stderr)
        return 2

    record = ProvenanceRecord.create(
        name="detector_segmenter_cost_benchmark",
        phase=10,
        config={
            "detector_segmenter_comparison": f"configs/{CONFIG_YAML}",
            "protocol_fingerprint": protocol.fingerprint(),
        },
        details={
            "phase": PHASE,
            "classification": COST_BENCHMARK_COMPLETE,
            "benchmark_status": BENCHMARK_LABEL,
            "detector_checkpoint_sha256": models[DETECTOR]["checkpoint_sha256"],
            "segmenter_checkpoint_sha256": models[SEGMENTER]["checkpoint_sha256"],
            "benchmark_membership_sha256": subset["benchmark_membership_sha256"],
            "execution_plan_sha256": plan_fingerprint,
            "timing_dataset_sha256": latency_payload["raw_timings"]["timing_dataset_sha256"],
            "latency_result_sha256": latency_payload["latency_result_sha256"],
            "memory_result_sha256": memory_payload["memory_result_sha256"],
            "precision_status": precision["status"],
            "timed_observations": latency_payload["raw_timings"]["observations"],
            "models_trained": 0,
            "models_modified": 0,
            "thresholds_tuned": 0,
            "ap_metrics_recomputed": False,
            "spatial_analysis_rerun": False,
            "holdout_accessed": False,
            "historical_artifacts_unchanged": True,
        },
        repo_root=paths.root,
    )
    record.add_input(paths.configs / CONFIG_YAML, relative_to=paths.root)
    for name in (
        "final_detector_manifest.json",
        "final_segmenter_manifest.json",
        PROTOCOL_JSON,
        BENCHMARK_CSV,
    ):
        record.add_input(paths.reports / name, relative_to=paths.root)
    for name in (LATENCY_JSON, MEMORY_JSON, BLOCKS_CSV, REPORT_MD):
        record.add_output(paths.reports / name, relative_to=paths.root)
    record.write_json(paths.reports / PROVENANCE_JSON)

    inference_delta = latency_payload["deltas"][MODEL_INFERENCE_LATENCY]
    end_to_end_delta = latency_payload["deltas"][END_TO_END_LATENCY]
    print(COST_BENCHMARK_COMPLETE)
    print(
        f"inference  D2 {inference_delta['detector_mean_latency_ms']} ms  "
        f"S1 {inference_delta['segmenter_mean_latency_ms']} ms  "
        f"delta {inference_delta['absolute_latency_delta_ms']} ms  "
        f"relative {inference_delta['relative_latency_cost']}"
    )
    print(
        f"end-to-end D2 {end_to_end_delta['detector_mean_latency_ms']} ms  "
        f"S1 {end_to_end_delta['segmenter_mean_latency_ms']} ms  "
        f"delta {end_to_end_delta['absolute_latency_delta_ms']} ms  "
        f"relative {end_to_end_delta['relative_latency_cost']}"
    )
    print(f"latency sha {latency_payload['latency_result_sha256']}")
    print(f"memory sha  {memory_payload['memory_result_sha256']}")
    print(f"timings sha {latency_payload['raw_timings']['timing_dataset_sha256']}")
    print(
        f"artifacts  reports/{LATENCY_JSON} ({latency_sha[:16]}...)  "
        f"reports/{MEMORY_JSON} ({memory_sha[:16]}...)"
    )
    print(f"report     reports/{REPORT_MD}")
    print(f"holdout    {HOLDOUT_STATUS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
