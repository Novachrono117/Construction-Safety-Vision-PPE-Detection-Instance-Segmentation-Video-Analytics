"""The computational-cost half of the detector-versus-segmenter comparison.

Phase 10C. Phase 10A froze what would be measured; phase 10B measured the
recognition and spatial halves. This module holds everything the cost benchmark
needs that is not a model call: the deterministic execution plan, the frozen
latency statistics, the derived cost quantities, the raw-observation schema and
the two result validators.

Nothing here trains, evaluates average precision, analyses a mask or reads an
image. It is arithmetic and bookkeeping, which is why it can be tested with
synthetic timings instead of by benchmarking a real model in a unit test.

Five decisions are worth stating explicitly, because each is a way the
benchmark could quietly stop being fair.

**The execution plan is built, not written down.** :func:`build_execution_plan`
derives the symmetric interleaved order from the frozen benchmark membership,
so the order cannot drift away from the images. Its fingerprint changes if
either the membership, its order, or the pass structure changes.

**The two timing boundaries never merge.** ``MODEL_INFERENCE_LATENCY_MS`` is
the forward pass on an already-prepared input tensor;
``END_TO_END_MODEL_OUTPUT_LATENCY_MS`` is preprocessing plus forward plus
postprocessing, which for the segmenter includes mask reconstruction. They are
reported side by side and a combined score is refused by the validator, not
merely discouraged.

**Throughput comes from the mean, never from the fastest iteration.** One
lucky repetition is not a throughput figure. ``1000 / mean_latency_ms`` at batch
1, and the validator recomputes it.

**One percentile convention, stated.** ``numpy.percentile`` with linear
interpolation, and the sample standard deviation with ``ddof=1``. Two
conventions applied to two models would be a way to pick a favourable number.

**Precision parity is probed, not read off the configuration.** A configuration
value is an intention; :func:`precision_evidence` records what the runtime
actually did with the tensor that reached the network.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from construction_safety_vision.detector_segmenter_analysis import (
    METRIC_PRECISION,
    result_fingerprint,
)
from construction_safety_vision.detector_segmenter_comparison import (
    BENCHMARK_IMAGE_COUNT,
    BENCHMARK_LABEL,
    DETECTOR_EXPERIMENT,
    END_TO_END_LATENCY,
    HOLDOUT_STATUS,
    LATENCY_BATCH,
    LATENCY_STATISTICS,
    MODEL_INFERENCE_LATENCY,
    PRECISION,
    PRECISION_ARGUMENT,
    PRECISION_VALUE,
    SEGMENTER_EXPERIMENT,
    TIMED_ITERATIONS_PER_IMAGE,
    WARMUP_ITERATIONS,
    ordered_fingerprint,
)

PHASE = "10C"

COST_BENCHMARK_COMPLETE = "DETECTOR_SEGMENTER_COST_BENCHMARK_COMPLETE"
PRECISION_PROTOCOL_MISMATCH = "PRECISION_PROTOCOL_MISMATCH"
MODEL_IDENTITY_MISMATCH = "MODEL_IDENTITY_MISMATCH"
LATENCY_BOUNDARY_BLOCKED = "LATENCY_BOUNDARY_IMPLEMENTATION_BLOCKED"
MEMORY_MEASUREMENT_BLOCKED = "MEMORY_MEASUREMENT_PROTOCOL_BLOCKED"
BENCHMARK_EXECUTION_FAILED = "BENCHMARK_EXECUTION_FAILED"
PROTOCOL_VIOLATION = "PROTOCOL_VIOLATION"
BLOCKED = "BLOCKED"

TIMING_BOUNDARIES: tuple[str, ...] = (MODEL_INFERENCE_LATENCY, END_TO_END_LATENCY)
"""The two frozen boundaries, in the order every artifact reports them."""

MODELS: tuple[str, ...] = (DETECTOR_EXPERIMENT, SEGMENTER_EXPERIMENT)
"""The two frozen experiments, detector first."""

PASS_FORWARD = "PASS_A_DETECTOR_FIRST"
PASS_REVERSE = "PASS_B_SEGMENTER_FIRST"

EXECUTION_PASSES: tuple[str, ...] = (PASS_FORWARD, PASS_REVERSE)
"""The two symmetric passes. Every image is timed under both orderings."""

PASS_ORDER: dict[str, tuple[str, ...]] = {
    PASS_FORWARD: MODELS,
    PASS_REVERSE: tuple(reversed(MODELS)),
}
"""Which model runs first within each pass, per the frozen interleaved design."""

WARMUP_SCHEDULE = "PER_MODEL_PER_IMAGE_PER_TIMED_BLOCK"
"""How the frozen warmup count is applied.

The protocol freezes *how many* warmup iterations there are, not when they run.
Applying the full 20 immediately before every timed block, for both models
equally, is declared here rather than chosen after seeing a distribution: a new
input shape can trigger a one-off kernel-selection cost, and a schedule that
warmed once at the start would push that cost into the first timed repetition
of every image.
"""

WARMUP_PATH = "FULL_END_TO_END_PATH_INCLUDING_MASK_RECONSTRUCTION"
"""Warmup exercises the widest timed path, so the narrower one is covered too."""

PERCENTILE_CONVENTION = "NUMPY_PERCENTILE_LINEAR_INTERPOLATION"
STD_CONVENTION = "SAMPLE_STANDARD_DEVIATION_DDOF_1"

TIMING_PRIMITIVE = "time.perf_counter"
SYNCHRONIZATION_PRIMITIVE = "torch.cuda.synchronize"

STATISTIC_FIELDS: tuple[str, ...] = ("count", *LATENCY_STATISTICS)
"""Every statistic reported for a latency distribution, count first."""

OBSERVATION_FIELDS: tuple[str, ...] = (
    "model",
    "benchmark_rank",
    "source_image_id",
    "execution_pass",
    "execution_position",
    "repetition",
    "timing_type",
    "latency_ms",
)
"""The schema of one raw timed observation."""

BLOCK_FIELDS: tuple[str, ...] = (
    "execution_position",
    "execution_pass",
    "benchmark_rank",
    "source_image_id",
    "model",
    "timing_type",
    "count",
    "mean",
    "median",
    "std",
    "min",
    "max",
)
"""The schema of the committed per-block latency summary."""

INFERENCE_MEMORY = "INFERENCE_MEMORY"
"""What the memory artifact measures. Never training memory."""

TRAINING_MEMORY_LABEL = "NOT_MEASURED_TRAINING_MEMORY_IS_A_DIFFERENT_QUANTITY"

MEMORY_ISOLATION = "SINGLE_MODEL_RESIDENCY_IN_A_DEDICATED_PROCESS"
"""How the two models' inference memory is kept apart.

``torch.cuda.max_memory_allocated`` is a device-global figure, so two resident
models would each be charged for the other's parameters. Releasing the first
model inside one process is not enough either: a diagnostic run before any
memory figure existed showed that ``del`` plus ``gc.collect()`` plus
``torch.cuda.empty_cache()`` still leaves a 33554432-byte cuBLAS workspace
allocated, which would then be charged to whichever model was measured second.
Each model is therefore measured in its own process, where the allocator starts
at zero and the other model cannot be resident at all. The frozen fairness
invariant asks for the same process only "where practical", and this is the
case where it is not.
"""

MEMORY_ISOLATION_DIAGNOSTIC = (
    "An in-process measurement was rejected before any memory figure was published: after "
    "releasing one model with del, gc.collect() and torch.cuda.empty_cache(), "
    "torch.cuda.memory_allocated() still reported 33554432 bytes of cuBLAS workspace, which "
    "the next model measured would have been charged for. The decision follows from that "
    "structural fact, not from any model's memory being preferable."
)

MEMORY_FIELDS: tuple[str, ...] = (
    "pre_load_allocated_bytes",
    "pre_load_reserved_bytes",
    "baseline_allocated_bytes",
    "baseline_reserved_bytes",
    "peak_memory_allocated_bytes",
    "peak_memory_reserved_bytes",
    "peak_memory_allocated_gib",
    "peak_memory_reserved_gib",
)
"""What is recorded for each model's inference memory.

``pre_load`` is the allocator before the model is constructed - the evidence
that the isolation held. ``baseline`` is the allocator at the moment peak
statistics are reset, after warmup, with the model resident.
"""

STATIC_MODEL_COMPLEXITY = "STATIC_MODEL_COMPLEXITY"
"""Parameter and FLOP counts. Not a runtime measurement."""

COMPLEXITY_REFERENCE_INPUT = "FRAMEWORK_DEFAULT_640_NOT_THE_BENCHMARK_768"
"""The input size the committed GFLOPs figures were computed at.

Read from the installed source: ``ultralytics.utils.torch_utils.get_flops`` and
``model_info`` both default to ``imgsz=640``, and both committed figures came
from those paths. The benchmark runs at 768, so the complexity figures describe
the architectures at a different reference input than the latency figures
describe. Stated rather than silently paired.
"""

DISTRIBUTION_DIAGNOSTIC_LABEL = "POST_HOC_HARDWARE_BEHAVIOR_DIAGNOSTIC"
"""What a description of the observed latency distribution's shape is.

It was written after the benchmark ran, so it is a diagnostic and not a
finding. It replaces no frozen statistic, changes no frozen number, and is
never a selection rule.
"""

DISTRIBUTION_DIAGNOSTIC_STATUS = "POST_HOC_DIAGNOSTIC_ONLY"

NO_POWER_TELEMETRY = "NO_SYNCHRONIZED_PER_OBSERVATION_POWER_STATE_TELEMETRY"
"""Whether GPU clock, P-state or utilisation telemetry accompanied the timings.

It did not. Nothing about the device's clock or power state was sampled during
or between the timed regions, so no observation can be mapped to a power state
and the cause of any feature of the distribution cannot be attributed from the
evidence this benchmark holds.
"""

DVFS_HYPOTHESIS = (
    "The shape is consistent with mobile-GPU DVFS and power-state behaviour on this laptop: "
    "a block's thirty repetitions cluster tightly, while different blocks sit at different "
    "levels independently of the image, the model and the pass. That is a hypothesis about a "
    "class of mechanism, not a measurement of one. No clock, P-state, utilisation, temperature "
    "or power reading was taken during the timed regions, so nothing here maps an observation "
    "to a device state, and no alternative explanation was excluded."
)

DVFS_HYPOTHESIS_STATUS = "UNTESTED_HYPOTHESIS"

TELEMETRY_NOTE = (
    "No GPU clock, P-state, utilisation, temperature or power reading was sampled during or "
    "between the timed regions. Sampling one inside a timed region would have added its own "
    "cost to the measurement; sampling one outside could not be mapped to an individual "
    "observation. The consequence is recorded rather than worked around: no latency "
    "observation can be attributed to a device state."
)
"""Why no device-state reading accompanies the timings, and what follows."""

CAUSAL_ATTRIBUTION = "UNKNOWN"
"""What caused any individual latency mode. Not established by this benchmark."""

PROTOCOL_STABILITY_FIELDS: tuple[str, ...] = (
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
)
"""Every way the protocol could have been bent after the timings were seen.

Each must be recorded, and each must be false. The list exists because the
temptation a frozen benchmark protects against arrives precisely when the
distribution turns out to be untidy.
"""

SEGMENTATION_COST_LABEL = "ADDITIONAL_SEGMENTATION_PIPELINE_COST"
"""What the latency difference may be called.

Deliberately not ``PURE_MASK_RECONSTRUCTION_CAUSAL_COST``: YOLO11n and
YOLO11n-seg differ in the mask branch of the network as well as in
postprocessing, and nothing in this benchmark isolates the two. The measured
quantity is the cost of the whole segmentation pipeline relative to the whole
detection pipeline.
"""

HOST_TRANSFER_BOUNDARY = "DEVICE_TO_HOST_TRANSFER_OUTSIDE_BOTH_BOUNDARIES"
"""Where the timed region stops, for both models alike.

The frozen boundary lists preprocessing, the forward pass, NMS/postprocessing
and mask reconstruction. Copying the resulting tensors to host memory is in
neither the ``includes`` nor the ``excludes`` list, so it is left outside both
boundaries and outside both models - which is symmetric, but does mean the
segmenter's larger outputs would cost more to move than the numbers here show.
Recorded as a limitation rather than resolved by inventing a third boundary.
"""

GIB = 1024**3


class CostBenchmarkError(RuntimeError):
    """Raised when the cost benchmark cannot run as frozen."""


class PrecisionParityError(CostBenchmarkError):
    """Raised when the two models do not resolve to the frozen precision."""


class MemoryIsolationError(CostBenchmarkError):
    """Raised when a memory figure cannot be attributed to one model."""


# --- the deterministic execution plan -------------------------------------------------


@dataclass(frozen=True)
class ExecutionStep:
    """One timed block: one model, one image, one pass.

    Attributes:
        position: Zero-based position in the executed order.
        execution_pass: Which symmetric pass this block belongs to.
        benchmark_rank: The image's rank in the frozen benchmark order.
        source_image_id: The image's canonical identifier.
        model: The experiment id whose model runs.
    """

    position: int
    execution_pass: str
    benchmark_rank: int
    source_image_id: str
    model: str

    def as_dict(self) -> dict[str, Any]:
        """Serialise the step for fingerprinting and reporting.

        Returns:
            A JSON-serialisable mapping.
        """
        return {
            "position": self.position,
            "execution_pass": self.execution_pass,
            "benchmark_rank": self.benchmark_rank,
            "source_image_id": self.source_image_id,
            "model": self.model,
        }


def build_execution_plan(image_ids: Sequence[str]) -> tuple[ExecutionStep, ...]:
    """Derive the frozen symmetric interleaved execution order.

    Pass A walks the benchmark images in the frozen order and times the
    detector then the segmenter on each. Pass B walks the same images in the
    same order and times the segmenter then the detector. Both passes feed the
    reported distribution, so residual thermal or ordering drift falls on both
    models rather than on whichever happened to run second.

    Args:
        image_ids: The benchmark subset, in the frozen benchmark order.

    Returns:
        Every timed block, in the order it will be executed.

    Raises:
        CostBenchmarkError: If the membership is the wrong size or carries a
            duplicate.
    """
    if len(image_ids) != BENCHMARK_IMAGE_COUNT:
        msg = f"the benchmark subset must hold exactly {BENCHMARK_IMAGE_COUNT} images"
        raise CostBenchmarkError(msg)
    if len(set(image_ids)) != len(image_ids):
        msg = "the benchmark subset carries a duplicate image id"
        raise CostBenchmarkError(msg)

    plan: list[ExecutionStep] = []
    for execution_pass in EXECUTION_PASSES:
        for rank, image_id in enumerate(image_ids):
            for model in PASS_ORDER[execution_pass]:
                plan.append(
                    ExecutionStep(
                        position=len(plan),
                        execution_pass=execution_pass,
                        benchmark_rank=rank,
                        source_image_id=str(image_id),
                        model=model,
                    )
                )
    return tuple(plan)


def execution_plan_fingerprint(plan: Sequence[ExecutionStep]) -> str:
    """Fingerprint an execution plan as an ordering.

    Args:
        plan: The timed blocks, in execution order.

    Returns:
        A SHA-256 hex digest that changes if any block, or the order of any
        two blocks, changes.
    """
    return ordered_fingerprint(
        [
            f"{step.position}|{step.execution_pass}|{step.benchmark_rank}|"
            f"{step.source_image_id}|{step.model}"
            for step in plan
        ]
    )


def expected_observation_count(*, images: int = BENCHMARK_IMAGE_COUNT) -> int:
    """How many timed observations a complete benchmark produces.

    Args:
        images: The benchmark image count.

    Returns:
        Blocks times boundaries times repetitions.
    """
    blocks = images * len(EXECUTION_PASSES) * len(MODELS)
    return blocks * len(TIMING_BOUNDARIES) * TIMED_ITERATIONS_PER_IMAGE


# --- the frozen latency statistics ----------------------------------------------------


def describe_latency(values: Sequence[float]) -> dict[str, Any]:
    """Summarise a latency distribution with exactly the frozen statistic set.

    One convention throughout: ``numpy.percentile`` with linear interpolation,
    and the sample standard deviation. No statistic outside the frozen set is
    computed, and none inside it is omitted, so neither model can be described
    by a more flattering summary than the other.

    Args:
        values: Latency observations, in milliseconds.

    Returns:
        The frozen statistics, rounded to the reported precision.

    Raises:
        CostBenchmarkError: If the sample is empty.
    """
    array = np.asarray([float(value) for value in values], dtype=float)
    if array.size == 0:
        msg = "a latency distribution with no observations cannot be summarised"
        raise CostBenchmarkError(msg)

    def value(number: float) -> float:
        return round(float(number), METRIC_PRECISION)

    return {
        "count": int(array.size),
        "mean": value(array.mean()),
        "median": value(np.median(array)),
        "std": value(array.std(ddof=1)) if array.size > 1 else None,
        "p50": value(np.percentile(array, 50)),
        "p90": value(np.percentile(array, 90)),
        "p95": value(np.percentile(array, 95)),
        "p99": value(np.percentile(array, 99)),
        "min": value(array.min()),
        "max": value(array.max()),
    }


def images_per_second_from_mean(mean_latency_ms: float) -> float:
    """Derive batch-1 throughput from a mean latency.

    From the mean, never from the fastest repetition: one lucky iteration
    describes a scheduling accident, not a rate the model sustains.

    Args:
        mean_latency_ms: Mean latency in milliseconds.

    Returns:
        Images per second.

    Raises:
        CostBenchmarkError: If the mean is not positive.
    """
    if mean_latency_ms <= 0:
        msg = "a non-positive mean latency cannot be converted to a throughput"
        raise CostBenchmarkError(msg)
    return round(1000.0 / float(mean_latency_ms), METRIC_PRECISION)


def absolute_latency_delta_ms(detector_mean: float, segmenter_mean: float) -> float:
    """The segmenter's mean latency minus the detector's, at one boundary.

    Args:
        detector_mean: The detector's mean latency in milliseconds.
        segmenter_mean: The segmenter's mean latency in milliseconds.

    Returns:
        The signed difference in milliseconds.
    """
    return round(float(segmenter_mean) - float(detector_mean), METRIC_PRECISION)


def relative_latency_cost(detector_mean: float, segmenter_mean: float) -> float:
    """The segmenter's mean latency as a fraction above the detector's.

    Args:
        detector_mean: The detector's mean latency in milliseconds.
        segmenter_mean: The segmenter's mean latency in milliseconds.

    Returns:
        ``(segmenter / detector) - 1``.

    Raises:
        CostBenchmarkError: If the detector's mean is not positive.
    """
    if detector_mean <= 0:
        msg = "a non-positive detector mean cannot be a relative-cost denominator"
        raise CostBenchmarkError(msg)
    return round(float(segmenter_mean) / float(detector_mean) - 1.0, METRIC_PRECISION)


def throughput_ratio(detector_ips: float, segmenter_ips: float) -> float:
    """The segmenter's throughput divided by the detector's.

    Args:
        detector_ips: The detector's images per second.
        segmenter_ips: The segmenter's images per second.

    Returns:
        The ratio.

    Raises:
        CostBenchmarkError: If the detector's throughput is not positive.
    """
    if detector_ips <= 0:
        msg = "a non-positive detector throughput cannot be a ratio denominator"
        raise CostBenchmarkError(msg)
    return round(float(segmenter_ips) / float(detector_ips), METRIC_PRECISION)


def build_cost_deltas(
    detector: Mapping[str, Any], segmenter: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    """Compute the frozen derived quantities for both timing boundaries.

    Args:
        detector: The detector's per-boundary statistics, keyed by boundary.
        segmenter: The segmenter's per-boundary statistics, keyed by boundary.

    Returns:
        One block per boundary, each carrying the absolute delta, the relative
        cost, both throughputs and their ratio.

    Raises:
        CostBenchmarkError: If a boundary is missing from either model.
    """
    deltas: dict[str, dict[str, Any]] = {}
    for boundary in TIMING_BOUNDARIES:
        if boundary not in detector or boundary not in segmenter:
            msg = f"both models must report {boundary}"
            raise CostBenchmarkError(msg)
        detector_mean = float(detector[boundary]["mean"])
        segmenter_mean = float(segmenter[boundary]["mean"])
        detector_ips = images_per_second_from_mean(detector_mean)
        segmenter_ips = images_per_second_from_mean(segmenter_mean)
        deltas[boundary] = {
            "detector_mean_latency_ms": round(detector_mean, METRIC_PRECISION),
            "segmenter_mean_latency_ms": round(segmenter_mean, METRIC_PRECISION),
            "absolute_latency_delta_ms": absolute_latency_delta_ms(detector_mean, segmenter_mean),
            "relative_latency_cost": relative_latency_cost(detector_mean, segmenter_mean),
            "detector_images_per_second_from_mean": detector_ips,
            "segmenter_images_per_second_from_mean": segmenter_ips,
            "throughput_ratio": throughput_ratio(detector_ips, segmenter_ips),
            "derived_from": "mean",
            "combined_latency_score": False,
        }
    return deltas


def memory_delta(detector: Mapping[str, Any], segmenter: Mapping[str, Any]) -> dict[str, Any]:
    """Compare two models' inference memory.

    Args:
        detector: The detector's memory block.
        segmenter: The segmenter's memory block.

    Returns:
        Absolute differences in bytes and GiB, and the ratios.

    Raises:
        CostBenchmarkError: If a detector figure is not positive.
    """
    result: dict[str, Any] = {"measurement": INFERENCE_MEMORY}
    for field in ("peak_memory_allocated_bytes", "peak_memory_reserved_bytes"):
        detector_value = int(detector[field])
        segmenter_value = int(segmenter[field])
        if detector_value <= 0:
            msg = f"the detector's {field} must be positive to form a ratio"
            raise CostBenchmarkError(msg)
        stem = field.removesuffix("_bytes")
        result[f"{stem}_delta_bytes"] = segmenter_value - detector_value
        result[f"{stem}_delta_gib"] = round(
            (segmenter_value - detector_value) / GIB, METRIC_PRECISION
        )
        result[f"{stem}_ratio"] = round(segmenter_value / detector_value, METRIC_PRECISION)
    return result


def as_gib(value: int) -> float:
    """Convert a byte count to GiB at the reported precision.

    Args:
        value: A byte count.

    Returns:
        The value in GiB.
    """
    return round(int(value) / GIB, METRIC_PRECISION)


def describe_distribution_shape(
    observations: Sequence[Mapping[str, Any]], latency: Mapping[str, Any]
) -> dict[str, Any]:
    """Describe the observed latency distribution's shape, descriptively.

    This exists because a mean roughly a third above its own median, with a P90
    near the maximum, is not a tail of stragglers, and a reader given only the
    mean would be misled. Everything here is computed from the same complete set
    of observations the frozen statistics come from - nothing is discarded,
    trimmed, winsorised, normalised or rescaled - and none of it replaces a
    frozen statistic.

    The per-block figures are aggregates of block means, with no threshold and
    no binning: phase 10A declared no split between a "fast" and a "slow" mode,
    and inventing one after seeing the data would be a post-hoc bin.

    Args:
        observations: The complete raw observations.
        latency: The frozen per-model, per-boundary statistics.

    Returns:
        The labelled diagnostic.

    Raises:
        CostBenchmarkError: If an observation is missing a schema field.
    """
    per_block: dict[tuple[str, str, int], list[float]] = {}
    block_pass: dict[int, str] = {}
    for observation in observations:
        missing = [field for field in OBSERVATION_FIELDS if field not in observation]
        if missing:
            msg = f"a raw observation is missing {missing}"
            raise CostBenchmarkError(msg)
        position = int(observation["execution_position"])
        block_pass[position] = str(observation["execution_pass"])
        key = (str(observation["model"]), str(observation["timing_type"]), position)
        per_block.setdefault(key, []).append(float(observation["latency_ms"]))

    block_means: dict[tuple[str, str], list[float]] = {}
    per_pass: dict[tuple[str, str, str], list[float]] = {}
    for (model, boundary, position), values in sorted(per_block.items()):
        mean = float(np.asarray(values, dtype=float).mean())
        block_means.setdefault((model, boundary), []).append(mean)
        per_pass.setdefault((model, boundary, block_pass[position]), []).append(mean)

    evidence: dict[str, Any] = {}
    for model in MODELS:
        model_block: dict[str, Any] = {}
        for boundary in TIMING_BOUNDARIES:
            statistics = latency[model][boundary]
            means = np.asarray(block_means[(model, boundary)], dtype=float)
            model_block[boundary] = {
                "mean": statistics["mean"],
                "median": statistics["median"],
                "mean_to_median_ratio": round(
                    float(statistics["mean"]) / float(statistics["median"]), METRIC_PRECISION
                ),
                "p90": statistics["p90"],
                "min": statistics["min"],
                "max": statistics["max"],
                "min_over_max": round(
                    float(statistics["min"]) / float(statistics["max"]), METRIC_PRECISION
                ),
                "timed_blocks": int(means.size),
                "block_mean_min": round(float(means.min()), METRIC_PRECISION),
                "block_mean_max": round(float(means.max()), METRIC_PRECISION),
                "per_pass_block_mean_of_means": {
                    execution_pass: round(
                        float(
                            np.asarray(
                                per_pass[(model, boundary, execution_pass)], dtype=float
                            ).mean()
                        ),
                        METRIC_PRECISION,
                    )
                    for execution_pass in EXECUTION_PASSES
                },
            }
        evidence[model] = model_block

    return {
        "label": DISTRIBUTION_DIAGNOSTIC_LABEL,
        "status": DISTRIBUTION_DIAGNOSTIC_STATUS,
        "replaces_frozen_statistics": False,
        "is_a_frozen_phase_10a_metric": False,
        "is_a_selection_rule": False,
        "is_a_significance_test": False,
        "changes_any_frozen_number": False,
        "computed_from": "ALL_VALID_OBSERVATIONS",
        "observations_used": len(observations),
        "observations_discarded": 0,
        "outlier_rejection_applied": False,
        "observations_normalized_or_rescaled": False,
        "per_mode_statistics_computed": False,
        "mode_split_threshold_declared_by_protocol": False,
        "per_observation_power_state_telemetry": NO_POWER_TELEMETRY,
        "causal_attribution": CAUSAL_ATTRIBUTION,
        "hypothesis": DVFS_HYPOTHESIS,
        "hypothesis_status": DVFS_HYPOTHESIS_STATUS,
        "telemetry_synchronous_with_timed_blocks": False,
        "telemetry_note": TELEMETRY_NOTE,
        "proportionality_across_models_demonstrated": False,
        "proportionality_note": (
            "Each model shows a mean above its own median and a P90 near its own maximum, and "
            "each model's figures are reported separately rather than as a ratio between the "
            "two. That two distributions share a shape is a description, not a demonstration "
            "that any mechanism scales them proportionally, and no such claim is made."
        ),
        "reading": (
            "Quote the mean together with the median, P90 and the range. The per-block figures "
            "show that the spread separates between blocks rather than within them: a block's "
            "thirty repetitions cluster, while block means span the range given. Which level a "
            "block sits at does not follow the image, the model or the pass."
        ),
        "evidence": evidence,
    }


def build_protocol_stability(*, rerun: bool = False) -> dict[str, Any]:
    """Declare that nothing was bent after the timings were observed.

    Args:
        rerun: Whether the benchmark was executed a second time.

    Returns:
        One boolean per way the protocol could have been adapted.
    """
    declaration = dict.fromkeys(PROTOCOL_STABILITY_FIELDS, False)
    declaration["benchmark_rerun"] = bool(rerun)
    declaration["note"] = (
        "The distribution turned out to be untidy, which is exactly when a frozen protocol "
        "earns its keep. The benchmark subset, the 20 warmup iterations, the 30 timed "
        "repetitions, the symmetric interleaved order, FP32, batch 1, imgsz 768 and both "
        "timing boundaries are all as frozen in phase 10A. No observation was removed, no "
        "outlier rule was introduced, no timing was normalised or rescaled, no power, clock or "
        "fan setting was touched, and the benchmark was not re-run to obtain a tidier result."
    )
    return declaration


# --- fingerprints ---------------------------------------------------------------------


def timing_dataset_fingerprint(observations: Sequence[Mapping[str, Any]]) -> str:
    """Fingerprint the raw timed observations as an ordered dataset.

    Covers every observation's identity and its latency, so the digest moves if
    a timing, a repetition, a block or the execution order changes. It carries
    no absolute path, hostname or rendering timestamp.

    Args:
        observations: The raw observations, in execution order.

    Returns:
        A SHA-256 hex digest.

    Raises:
        CostBenchmarkError: If an observation is missing a schema field.
    """
    rows: list[str] = []
    for observation in observations:
        missing = [field for field in OBSERVATION_FIELDS if field not in observation]
        if missing:
            msg = f"a raw observation is missing {missing}"
            raise CostBenchmarkError(msg)
        rows.append(
            "|".join(
                f"{round(float(observation[field]), METRIC_PRECISION)}"
                if field == "latency_ms"
                else str(observation[field])
                for field in OBSERVATION_FIELDS
            )
        )
    return ordered_fingerprint(rows)


def latency_result_fingerprint(payload: Mapping[str, Any]) -> str:
    """Fingerprint the semantic content of the latency result.

    Args:
        payload: The latency artifact.

    Returns:
        A SHA-256 hex digest over model identity, protocol, membership,
        execution order, precision evidence and every reported statistic.
    """
    return result_fingerprint(
        {
            "protocol_fingerprint": payload.get("protocol_fingerprint"),
            "detector_checkpoint": payload.get("detector", {}).get("checkpoint_sha256"),
            "segmenter_checkpoint": payload.get("segmenter", {}).get("checkpoint_sha256"),
            "benchmark_membership_sha256": payload.get("benchmark", {}).get(
                "benchmark_membership_sha256"
            ),
            "execution_plan_sha256": payload.get("benchmark", {}).get("execution_plan_sha256"),
            "timing_dataset_sha256": payload.get("raw_timings", {}).get("timing_dataset_sha256"),
            "precision": payload.get("precision_preflight", {}).get("evidence"),
            "latency": payload.get("latency"),
            "deltas": payload.get("deltas"),
        }
    )


def memory_result_fingerprint(payload: Mapping[str, Any]) -> str:
    """Fingerprint the semantic content of the memory result.

    Args:
        payload: The memory artifact.

    Returns:
        A SHA-256 hex digest over model identity, protocol, the measurement
        conditions and the recorded peaks.
    """
    return result_fingerprint(
        {
            "protocol_fingerprint": payload.get("protocol_fingerprint"),
            "detector_checkpoint": payload.get("detector", {}).get("checkpoint_sha256"),
            "segmenter_checkpoint": payload.get("segmenter", {}).get("checkpoint_sha256"),
            "conditions": payload.get("conditions"),
            "memory": payload.get("memory"),
            "delta": payload.get("delta"),
        }
    )


# --- the runtime precision probe ------------------------------------------------------


def precision_evidence(
    checkpoint: str | Path, settings: Mapping[str, Any], image: Any
) -> dict[str, Any]:
    """Record what precision the runtime actually used for one model.

    The configuration expresses FP32 through the installed framework's
    precision control, but a configuration value is an intention. This runs one
    prediction with a forward pre-hook attached and reports the backend's
    precision flag, every parameter dtype, the dtype of the tensor that reached
    the network, and whether autocast was active at that moment.

    Args:
        checkpoint: The frozen checkpoint to probe.
        settings: A frozen inference block, supplying imgsz, thresholds and the
            precision argument.
        image: A decoded source image to probe with.

    Returns:
        The captured evidence for that model.
    """
    import torch
    from ultralytics import YOLO

    model = YOLO(str(checkpoint))
    seen: dict[str, Any] = {}

    def hook(_module: Any, args: Any) -> None:
        tensor = args[0]
        seen.setdefault("input_dtype", str(tensor.dtype))
        seen.setdefault("autocast_enabled_during_forward", torch.is_autocast_enabled("cuda"))

    call = {
        "source": image,
        "imgsz": settings["imgsz"],
        "conf": settings["conf"],
        "iou": settings["iou"],
        "max_det": settings["max_det"],
        "augment": settings["augment"],
        settings["precision_argument"]: settings["precision_value"],
        "verbose": False,
        "save": False,
    }
    model.predict(**call)
    backend = model.predictor.model
    handle = backend.model.register_forward_pre_hook(hook)
    model.predict(**call)
    handle.remove()

    evidence = {
        "backend_fp16_flag": bool(getattr(backend, "fp16", False)),
        "resolved_precision_argument": settings["precision_argument"],
        "resolved_precision_value": getattr(
            model.predictor.args, str(settings["precision_argument"]), None
        ),
        "parameter_dtypes": sorted({str(p.dtype) for p in backend.model.parameters()}),
        "float_buffer_dtypes": sorted(
            {str(b.dtype) for b in backend.model.buffers() if b.is_floating_point()}
        ),
        "input_dtype": seen.get("input_dtype"),
        "autocast_enabled_during_forward": seen.get("autocast_enabled_during_forward"),
        "quantization_config_present": getattr(backend.model, "qconfig", None) is not None,
        "device_type": str(next(backend.model.parameters()).device).split(":")[0],
    }
    del model
    torch.cuda.empty_cache()
    return evidence


def assert_precision_parity(captured: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Check both models resolved to the frozen FP32 intent, identically.

    Args:
        captured: Per-model evidence from :func:`precision_evidence`.

    Returns:
        The verified parity declaration.

    Raises:
        PrecisionParityError: If the two differ, or either departs from FP32.
    """
    missing = [model for model in MODELS if model not in captured]
    if missing:
        msg = f"{PRECISION_PROTOCOL_MISMATCH}: no precision evidence for {missing}"
        raise PrecisionParityError(msg)

    detector = dict(captured[DETECTOR_EXPERIMENT])
    segmenter = dict(captured[SEGMENTER_EXPERIMENT])
    if detector != segmenter:
        differing = sorted(key for key in detector if detector[key] != segmenter.get(key))
        msg = (
            f"{PRECISION_PROTOCOL_MISMATCH}: the two models resolve differently in {differing}. "
            "A latency comparison across two precisions measures the precision."
        )
        raise PrecisionParityError(msg)

    if detector["backend_fp16_flag"] is not False:
        msg = f"{PRECISION_PROTOCOL_MISMATCH}: the backend enabled FP16 despite the FP32 intent"
        raise PrecisionParityError(msg)
    if detector["parameter_dtypes"] != ["torch.float32"]:
        msg = (
            f"{PRECISION_PROTOCOL_MISMATCH}: parameters are {detector['parameter_dtypes']}, "
            "not FP32 only"
        )
        raise PrecisionParityError(msg)
    if detector["input_dtype"] != "torch.float32":
        msg = f"{PRECISION_PROTOCOL_MISMATCH}: the input tensor is {detector['input_dtype']}"
        raise PrecisionParityError(msg)
    if detector["autocast_enabled_during_forward"] is not False:
        msg = (
            f"{PRECISION_PROTOCOL_MISMATCH}: autocast was active during the forward pass, which "
            "would run parts of the network at a lower precision"
        )
        raise PrecisionParityError(msg)
    if detector["quantization_config_present"]:
        msg = f"{PRECISION_PROTOCOL_MISMATCH}: a quantization configuration is attached"
        raise PrecisionParityError(msg)
    if int(detector["resolved_precision_value"]) != PRECISION_VALUE:
        msg = (
            f"{PRECISION_PROTOCOL_MISMATCH}: {PRECISION_ARGUMENT} resolved to "
            f"{detector['resolved_precision_value']}, not {PRECISION_VALUE}"
        )
        raise PrecisionParityError(msg)

    return {
        "status": "EFFECTIVE_PRECISION_PARITY_VERIFIED",
        "frozen_intent": PRECISION,
        "precision_argument": PRECISION_ARGUMENT,
        "precision_value": PRECISION_VALUE,
        "identical_across_models": True,
        "evidence": {model: dict(captured[model]) for model in MODELS},
        "probe_method": (
            "One prediction per model with a forward pre-hook on the network, capturing the "
            "dtype of the tensor that actually reached it and the autocast state at that "
            "moment. Configuration values were not trusted."
        ),
        "note": (
            "Phase 10B's runner carries its own equivalent probe. It was not refactored to "
            "call this one, because phase 10B is a reported experiment and its script is left "
            "byte-identical; a test pins the two to the same evidence schema instead."
        ),
    }


# --- validators ------------------------------------------------------------------------


def _statistics_problems(block: Any, *, label: str) -> list[str]:
    """Check one latency distribution's statistics are complete and consistent.

    Args:
        block: The statistics mapping.
        label: Its name, for the message.

    Returns:
        One description per problem.
    """
    problems: list[str] = []
    if not isinstance(block, Mapping):
        return [f"{label}: must be a mapping of statistics"]
    missing = [field for field in STATISTIC_FIELDS if field not in block]
    if missing:
        problems.append(f"{label} is missing {missing}")
        return problems
    if int(block["count"]) != len(EXECUTION_PASSES) * BENCHMARK_IMAGE_COUNT * (
        TIMED_ITERATIONS_PER_IMAGE
    ):
        problems.append(
            f"{label}.count must be "
            f"{len(EXECUTION_PASSES) * BENCHMARK_IMAGE_COUNT * TIMED_ITERATIONS_PER_IMAGE} "
            "observations"
        )
    for field in ("mean", "median", "p50", "p90", "p95", "p99", "min", "max"):
        if not isinstance(block[field], (int, float)) or isinstance(block[field], bool):
            problems.append(f"{label}.{field} is not a number")
    if all(isinstance(block[field], (int, float)) for field in ("min", "max", "mean")):
        if not block["min"] <= block["mean"] <= block["max"]:
            problems.append(f"{label}: the mean does not lie between the minimum and the maximum")
        if block["median"] != block["p50"]:
            problems.append(f"{label}: the median and P50 disagree under one convention")
        ordered = [block["p50"], block["p90"], block["p95"], block["p99"], block["max"]]
        if ordered != sorted(ordered):
            problems.append(f"{label}: the percentiles are not monotonic")
    return problems


def validate_latency_result(
    payload: Mapping[str, Any],
    *,
    protocol_fingerprint: str,
    detector_sha256: str,
    segmenter_sha256: str,
    benchmark_sha256: str,
    execution_plan_sha256: str,
) -> list[str]:
    """Check the committed latency artifact against the frozen protocol.

    Args:
        payload: The parsed artifact.
        protocol_fingerprint: The frozen phase 10A protocol fingerprint.
        detector_sha256: The frozen detector's checkpoint digest.
        segmenter_sha256: The frozen segmenter's checkpoint digest.
        benchmark_sha256: The frozen ordered benchmark membership fingerprint.
        execution_plan_sha256: The derived execution-plan fingerprint.

    Returns:
        One description per problem found, empty when the artifact is valid.
    """
    problems: list[str] = []

    if payload.get("phase") != PHASE:
        problems.append(f"phase must be {PHASE!r}")
    if payload.get("status") != COST_BENCHMARK_COMPLETE:
        problems.append(f"status must be {COST_BENCHMARK_COMPLETE!r}")
    if payload.get("benchmark_status") != BENCHMARK_LABEL:
        problems.append(f"benchmark_status must be {BENCHMARK_LABEL!r}")
    if payload.get("protocol_fingerprint") != protocol_fingerprint:
        problems.append("the protocol fingerprint is not the frozen phase 10A one")

    if payload.get("detector", {}).get("checkpoint_sha256") != detector_sha256:
        problems.append("the detector checkpoint is not the frozen one")
    if payload.get("segmenter", {}).get("checkpoint_sha256") != segmenter_sha256:
        problems.append("the segmenter checkpoint is not the frozen one")
    for name, experiment in (
        ("detector", DETECTOR_EXPERIMENT),
        ("segmenter", SEGMENTER_EXPERIMENT),
    ):
        block = payload.get(name, {})
        if block.get("experiment") != experiment:
            problems.append(f"{name}.experiment must be {experiment}")
        if int(block.get("imgsz", 0)) != 768:
            problems.append(f"{name}.imgsz must be 768")
        if block.get("trained_in_this_phase") is not False:
            problems.append(f"{name}.trained_in_this_phase must be false")

    benchmark = payload.get("benchmark", {})
    if benchmark.get("benchmark_membership_sha256") != benchmark_sha256:
        problems.append("the benchmark membership or its order does not match the frozen subset")
    if benchmark.get("execution_plan_sha256") != execution_plan_sha256:
        problems.append("the executed order does not match the derived execution plan")
    if int(benchmark.get("benchmark_image_count", 0)) != BENCHMARK_IMAGE_COUNT:
        problems.append(f"the benchmark must use exactly {BENCHMARK_IMAGE_COUNT} images")
    if benchmark.get("split") != "validation":
        problems.append("the benchmark subset must come from the validation split")
    if benchmark.get("randomized") is not False:
        problems.append("the execution order must not be randomised")
    if benchmark.get("interleaved") is not True or benchmark.get("symmetric") is not True:
        problems.append("the execution order must be interleaved and symmetric")
    if list(benchmark.get("execution_passes", [])) != list(EXECUTION_PASSES):
        problems.append("both symmetric passes must be recorded, in order")
    if benchmark.get("images_reselected") is not False:
        problems.append("the benchmark subset must not be reselected")

    conditions = payload.get("conditions", {})
    if int(conditions.get("batch", 0)) != LATENCY_BATCH:
        problems.append(f"batch must be {LATENCY_BATCH}")
    if int(conditions.get("imgsz", 0)) != 768:
        problems.append("imgsz must be 768")
    if conditions.get("precision") != PRECISION:
        problems.append(f"both models must run in {PRECISION}")
    if int(conditions.get("warmup_iterations", 0)) != WARMUP_ITERATIONS:
        problems.append(f"warmup must be exactly {WARMUP_ITERATIONS} iterations")
    if int(conditions.get("timed_iterations_per_image", 0)) != TIMED_ITERATIONS_PER_IMAGE:
        problems.append(f"timed repetitions must be exactly {TIMED_ITERATIONS_PER_IMAGE}")
    if conditions.get("augment") is not False or conditions.get("tta") is not False:
        problems.append("no augmentation or TTA is authorised")
    if conditions.get("timing_primitive") != TIMING_PRIMITIVE:
        problems.append(f"the timing primitive must be {TIMING_PRIMITIVE}")
    if conditions.get("synchronization_primitive") != SYNCHRONIZATION_PRIMITIVE:
        problems.append(f"the synchronisation primitive must be {SYNCHRONIZATION_PRIMITIVE}")
    for edge in ("cuda_synchronize_before_timed_region", "cuda_synchronize_after_timed_region"):
        if conditions.get(edge) is not True:
            problems.append(f"conditions.{edge} must be true")
    if conditions.get("same_primitive_for_both_models") is not True:
        problems.append("both models must be timed with the same primitive")
    if conditions.get("percentile_convention") != PERCENTILE_CONVENTION:
        problems.append("one percentile convention must be declared")
    for forbidden in ("torch_compile", "tensorrt", "onnx"):
        if conditions.get(forbidden) is not False:
            problems.append(f"conditions.{forbidden} must be false")
    if conditions.get("model_load_excluded_from_timing") is not True:
        problems.append("model loading must be excluded from every timed region")
    if conditions.get("disk_decode_excluded_from_timing") is not True:
        problems.append("image decode must be hoisted out of every timed region")
    if conditions.get("predictions_saved_during_timing") is not False:
        problems.append("nothing may be written to disk inside a timed region")

    boundaries = payload.get("timing_boundaries", {})
    for boundary in TIMING_BOUNDARIES:
        if boundary not in boundaries:
            problems.append(f"timing_boundaries must define {boundary}")
    inference_boundary = boundaries.get(MODEL_INFERENCE_LATENCY, {})
    end_to_end = boundaries.get(END_TO_END_LATENCY, {})
    if inference_boundary.get("mask_reconstruction_included") is not False:
        problems.append("mask reconstruction is outside the model-inference boundary")
    if inference_boundary.get("preprocessing_included") is not False:
        problems.append("preprocessing is outside the model-inference boundary")
    if end_to_end.get("preprocessing_included") is not True:
        problems.append("the end-to-end boundary must include preprocessing")
    if end_to_end.get("postprocessing_included") is not True:
        problems.append("the end-to-end boundary must include NMS and postprocessing")
    if end_to_end.get("segmenter_mask_reconstruction_included") is not True:
        problems.append(
            "mask reconstruction must be inside the segmenter's end-to-end timing: excluding "
            "it would hide the cost this comparison exists to quantify"
        )
    if end_to_end.get("outputs_materialized_before_end_timestamp") is not True:
        problems.append("usable outputs must be materialised before the end timestamp")

    latency = payload.get("latency", {})
    for model in MODELS:
        block = latency.get(model, {})
        if not block:
            problems.append(f"no latency statistics for {model}")
            continue
        for boundary in TIMING_BOUNDARIES:
            problems.extend(
                _statistics_problems(block.get(boundary), label=f"latency.{model}.{boundary}")
            )
            throughput = block.get("throughput", {}).get(boundary)
            statistics = block.get(boundary, {})
            if isinstance(throughput, (int, float)) and isinstance(
                statistics.get("mean"), (int, float)
            ):
                expected = images_per_second_from_mean(float(statistics["mean"]))
                if round(float(throughput), METRIC_PRECISION) != expected:
                    problems.append(
                        f"latency.{model}.throughput.{boundary} does not recompute from the mean"
                    )
            elif throughput is None:
                problems.append(f"latency.{model}.throughput.{boundary} is missing")

    deltas = payload.get("deltas", {})
    for boundary in TIMING_BOUNDARIES:
        block = deltas.get(boundary, {})
        if not block:
            problems.append(f"no cost delta for {boundary}")
            continue
        detector_mean = latency.get(DETECTOR_EXPERIMENT, {}).get(boundary, {}).get("mean")
        segmenter_mean = latency.get(SEGMENTER_EXPERIMENT, {}).get(boundary, {}).get("mean")
        if not isinstance(detector_mean, (int, float)) or not isinstance(
            segmenter_mean, (int, float)
        ):
            continue
        if block.get("absolute_latency_delta_ms") != absolute_latency_delta_ms(
            detector_mean, segmenter_mean
        ):
            problems.append(f"deltas.{boundary}.absolute_latency_delta_ms does not recompute")
        if block.get("relative_latency_cost") != relative_latency_cost(
            detector_mean, segmenter_mean
        ):
            problems.append(f"deltas.{boundary}.relative_latency_cost does not recompute")
        expected_ratio = throughput_ratio(
            images_per_second_from_mean(detector_mean),
            images_per_second_from_mean(segmenter_mean),
        )
        if block.get("throughput_ratio") != expected_ratio:
            problems.append(f"deltas.{boundary}.throughput_ratio does not recompute")
        if block.get("combined_latency_score") is not False:
            problems.append(f"deltas.{boundary} must not carry a combined latency score")

    if payload.get("combined_latency_score") is not False:
        problems.append("no combined latency score may be reported")
    if payload.get("winner_declared") is not False:
        problems.append("no winner may be declared")

    raw = payload.get("raw_timings", {})
    if list(raw.get("schema", [])) != list(OBSERVATION_FIELDS):
        problems.append("the raw timing schema does not match the frozen observation fields")
    if int(raw.get("observations", 0)) != expected_observation_count():
        problems.append(
            f"the benchmark must record exactly {expected_observation_count()} timed observations"
        )
    if not isinstance(raw.get("timing_dataset_sha256"), str):
        problems.append("the raw timing dataset must carry a fingerprint")

    complexity = payload.get("model_complexity", {})
    if complexity.get("label") != STATIC_MODEL_COMPLEXITY:
        problems.append(f"model complexity must be labelled {STATIC_MODEL_COMPLEXITY}")
    if complexity.get("measured_at_benchmark_input_size") is not False:
        problems.append(
            "the committed GFLOPs figures are not computed at the benchmark input size and "
            "must say so"
        )

    if payload.get("segmentation_cost_label") != SEGMENTATION_COST_LABEL:
        problems.append(f"the latency difference must be labelled {SEGMENTATION_COST_LABEL}")
    if payload.get("pure_mask_reconstruction_cost_isolated") is not False:
        problems.append(
            "the benchmark does not isolate mask reconstruction from the mask branch of the "
            "network and must not claim to"
        )

    diagnostic = payload.get("post_hoc_distribution_diagnostic", {})
    if diagnostic:
        if diagnostic.get("label") != DISTRIBUTION_DIAGNOSTIC_LABEL:
            problems.append(
                f"the distribution diagnostic must carry {DISTRIBUTION_DIAGNOSTIC_LABEL}"
            )
        if diagnostic.get("status") != DISTRIBUTION_DIAGNOSTIC_STATUS:
            problems.append(f"the distribution diagnostic must be {DISTRIBUTION_DIAGNOSTIC_STATUS}")
        for flag in (
            "replaces_frozen_statistics",
            "is_a_frozen_phase_10a_metric",
            "is_a_selection_rule",
            "is_a_significance_test",
            "changes_any_frozen_number",
            "outlier_rejection_applied",
            "observations_normalized_or_rescaled",
            "per_mode_statistics_computed",
            "telemetry_synchronous_with_timed_blocks",
            "proportionality_across_models_demonstrated",
        ):
            if diagnostic.get(flag) is not False:
                problems.append(f"post_hoc_distribution_diagnostic.{flag} must be false")
        if diagnostic.get("observations_discarded") != 0:
            problems.append("no observation may be discarded from the frozen distribution")
        if diagnostic.get("causal_attribution") != CAUSAL_ATTRIBUTION:
            problems.append(
                "the cause of any latency mode must be recorded as "
                f"{CAUSAL_ATTRIBUTION}: no telemetry accompanied the timings"
            )
        if diagnostic.get("per_observation_power_state_telemetry") != NO_POWER_TELEMETRY:
            problems.append(f"the artifact must record {NO_POWER_TELEMETRY}")
        if diagnostic.get("hypothesis_status") != DVFS_HYPOTHESIS_STATUS:
            problems.append("the hardware-behaviour reading must stay an untested hypothesis")
        if diagnostic.get("observations_used") != expected_observation_count():
            problems.append(
                "the diagnostic must be computed from every timed observation, not a subset"
            )

    stability = payload.get("protocol_stability_after_observation", {})
    if stability:
        missing = [field for field in PROTOCOL_STABILITY_FIELDS if field not in stability]
        if missing:
            problems.append(f"protocol_stability_after_observation is missing {missing}")
        for field in PROTOCOL_STABILITY_FIELDS:
            if stability.get(field) is not False:
                problems.append(
                    f"protocol_stability_after_observation.{field} must be false: the frozen "
                    "protocol may not be adapted after its timings are seen"
                )

    if payload.get("models_trained_in_this_phase") != 0:
        problems.append("phase 10C trains nothing")
    if payload.get("ap_metrics_recomputed") is not False:
        problems.append("phase 10C recomputes no average precision")
    if payload.get("spatial_analysis_rerun") is not False:
        problems.append("phase 10C reruns no spatial or association analysis")
    if payload.get("thresholds_tuned") != 0:
        problems.append("phase 10C tunes no threshold")

    problems.extend(_holdout_problems(payload, label="the latency result"))
    return problems


def validate_memory_result(
    payload: Mapping[str, Any],
    *,
    protocol_fingerprint: str,
    detector_sha256: str,
    segmenter_sha256: str,
) -> list[str]:
    """Check the committed inference-memory artifact against the frozen protocol.

    Args:
        payload: The parsed artifact.
        protocol_fingerprint: The frozen phase 10A protocol fingerprint.
        detector_sha256: The frozen detector's checkpoint digest.
        segmenter_sha256: The frozen segmenter's checkpoint digest.

    Returns:
        One description per problem found, empty when the artifact is valid.
    """
    problems: list[str] = []

    if payload.get("phase") != PHASE:
        problems.append(f"phase must be {PHASE!r}")
    if payload.get("measurement") != INFERENCE_MEMORY:
        problems.append(f"measurement must be {INFERENCE_MEMORY!r}, never training memory")
    if payload.get("training_memory_reused") is not False:
        problems.append("training memory may not be substituted")
    if payload.get("training_memory") != TRAINING_MEMORY_LABEL:
        problems.append("the artifact must state that training memory was not measured")
    if payload.get("protocol_fingerprint") != protocol_fingerprint:
        problems.append("the protocol fingerprint is not the frozen phase 10A one")
    if payload.get("detector", {}).get("checkpoint_sha256") != detector_sha256:
        problems.append("the detector checkpoint is not the frozen one")
    if payload.get("segmenter", {}).get("checkpoint_sha256") != segmenter_sha256:
        problems.append("the segmenter checkpoint is not the frozen one")

    conditions = payload.get("conditions", {})
    if int(conditions.get("batch", 0)) != LATENCY_BATCH:
        problems.append(f"batch must be {LATENCY_BATCH}")
    if int(conditions.get("imgsz", 0)) != 768:
        problems.append("imgsz must be 768")
    if conditions.get("precision") != PRECISION:
        problems.append(f"both models must be measured in {PRECISION}")
    if int(conditions.get("warmup_iterations", 0)) != WARMUP_ITERATIONS:
        problems.append(f"warmup must be exactly {WARMUP_ITERATIONS} iterations")
    if conditions.get("peak_stats_reset_after_warmup") is not True:
        problems.append(
            "peak memory statistics must be reset after warmup, or the figure describes the "
            "allocator's warmup high-water mark"
        )
    if conditions.get("isolation_method") != MEMORY_ISOLATION:
        problems.append(f"the isolation method must be {MEMORY_ISOLATION!r}")
    if int(conditions.get("models_resident_during_measurement", 0)) != 1:
        problems.append("exactly one model may be resident while its memory is measured")
    if conditions.get("isolation_diagnostic") != MEMORY_ISOLATION_DIAGNOSTIC:
        problems.append("the artifact must record why in-process measurement was rejected")

    memory = payload.get("memory", {})
    for model in MODELS:
        block = memory.get(model, {})
        if not block:
            problems.append(f"no memory measurement for {model}")
            continue
        missing = [field for field in MEMORY_FIELDS if field not in block]
        if missing:
            problems.append(f"memory.{model} is missing {missing}")
            continue
        for field in ("peak_memory_allocated_bytes", "peak_memory_reserved_bytes"):
            if not isinstance(block[field], int) or block[field] <= 0:
                problems.append(f"memory.{model}.{field} must be a positive byte count")
        allocated = block.get("peak_memory_allocated_bytes")
        reserved = block.get("peak_memory_reserved_bytes")
        if isinstance(allocated, int) and isinstance(reserved, int) and allocated > reserved:
            problems.append(f"memory.{model}: peak allocated exceeds peak reserved")
        if isinstance(allocated, int) and block.get("peak_memory_allocated_gib") != as_gib(
            allocated
        ):
            problems.append(f"memory.{model}.peak_memory_allocated_gib does not recompute")
        if isinstance(reserved, int) and block.get("peak_memory_reserved_gib") != as_gib(reserved):
            problems.append(f"memory.{model}.peak_memory_reserved_gib does not recompute")
        if block.get("other_model_resident") is not False:
            problems.append(f"memory.{model} must be measured with no other model resident")
        if block.get("pre_load_allocated_bytes") != 0:
            problems.append(
                f"memory.{model}: the allocator must be empty before the model is loaded, or the "
                "peak cannot be attributed to this model alone"
            )

    delta = payload.get("delta", {})
    detector_block = memory.get(DETECTOR_EXPERIMENT, {})
    segmenter_block = memory.get(SEGMENTER_EXPERIMENT, {})
    if detector_block and segmenter_block and delta:
        try:
            expected = memory_delta(detector_block, segmenter_block)
        except (CostBenchmarkError, KeyError, TypeError, ValueError):
            problems.append("the memory delta could not be recomputed from the recorded peaks")
        else:
            for field, value in expected.items():
                if delta.get(field) != value:
                    problems.append(f"delta.{field} does not recompute")

    problems.extend(_holdout_problems(payload, label="the memory result"))
    return problems


def _holdout_problems(payload: Mapping[str, Any], *, label: str) -> list[str]:
    """Check an artifact declares and honours the holdout policy.

    Args:
        payload: The artifact.
        label: Its name, for the message.

    Returns:
        One description per problem.
    """
    problems: list[str] = []
    if payload.get("test", {}).get("status") != HOLDOUT_STATUS:
        problems.append(f"{label} does not declare the holdout protected")
    if payload.get("holdout_accessed", True):
        problems.append(f"{label} records holdout access")
    body = {key: value for key, value in payload.items() if key != "test"}
    if _mentions_holdout(body):
        problems.append(f"{label} names the protected split outside its holdout declaration")
    return problems


def _mentions_holdout(value: Any) -> bool:
    """Report whether a payload names the protected split.

    Args:
        value: Any part of an artifact.

    Returns:
        ``True`` when ``"test"`` appears as a string value or a key.
    """
    if isinstance(value, str):
        return value.strip().lower() == "test"
    if isinstance(value, Mapping):
        return any(_mentions_holdout(key) or _mentions_holdout(item) for key, item in value.items())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return any(_mentions_holdout(item) for item in value)
    return False


def render_rows(rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> str:
    """Render tabular rows as deterministic CSV text.

    Args:
        rows: The rows, in order.
        fields: The column order.

    Returns:
        CSV text with a header and Unix line endings.
    """
    lines = [",".join(fields)]
    for row in rows:
        lines.append(
            ",".join("" if row.get(field) is None else str(row[field]) for field in fields)
        )
    return "\n".join(lines) + "\n"


def dumps(payload: Any) -> str:
    """Serialise an artifact deterministically.

    Args:
        payload: JSON-serialisable content.

    Returns:
        Sorted, indented JSON text with a trailing newline.
    """
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
