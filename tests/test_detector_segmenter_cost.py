"""Tests for the phase 10C cost benchmark's protocol, arithmetic and validators.

Everything here runs on synthetic timings. A unit test never loads a
checkpoint, never touches CUDA and never benchmarks a real model: a test that
re-timed the models would be slow, would give a different answer every run, and
would not be testing the code under test.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from construction_safety_vision.detector_segmenter_comparison import (
    BENCHMARK_IMAGE_COUNT,
    END_TO_END_LATENCY,
    LATENCY_BATCH,
    LATENCY_STATISTICS,
    MODEL_INFERENCE_LATENCY,
    PRECISION,
    PRECISION_VALUE,
    TIMED_ITERATIONS_PER_IMAGE,
    WARMUP_ITERATIONS,
    load_comparison_protocol,
    ordered_fingerprint,
    stable_rank,
)
from construction_safety_vision.detector_segmenter_cost import (
    BLOCK_FIELDS,
    CAUSAL_ATTRIBUTION,
    COST_BENCHMARK_COMPLETE,
    DISTRIBUTION_DIAGNOSTIC_LABEL,
    DISTRIBUTION_DIAGNOSTIC_STATUS,
    DVFS_HYPOTHESIS,
    DVFS_HYPOTHESIS_STATUS,
    EXECUTION_PASSES,
    GIB,
    INFERENCE_MEMORY,
    MEMORY_FIELDS,
    MEMORY_ISOLATION,
    MEMORY_ISOLATION_DIAGNOSTIC,
    MODELS,
    NO_POWER_TELEMETRY,
    OBSERVATION_FIELDS,
    PASS_FORWARD,
    PASS_ORDER,
    PASS_REVERSE,
    PERCENTILE_CONVENTION,
    PHASE,
    PROTOCOL_STABILITY_FIELDS,
    SEGMENTATION_COST_LABEL,
    STATIC_MODEL_COMPLEXITY,
    STATISTIC_FIELDS,
    SYNCHRONIZATION_PRIMITIVE,
    TIMING_BOUNDARIES,
    TIMING_PRIMITIVE,
    TRAINING_MEMORY_LABEL,
    CostBenchmarkError,
    PrecisionParityError,
    absolute_latency_delta_ms,
    as_gib,
    assert_precision_parity,
    build_cost_deltas,
    build_execution_plan,
    build_protocol_stability,
    describe_distribution_shape,
    describe_latency,
    execution_plan_fingerprint,
    expected_observation_count,
    images_per_second_from_mean,
    memory_delta,
    relative_latency_cost,
    throughput_ratio,
    timing_dataset_fingerprint,
    validate_latency_result,
    validate_memory_result,
)
from construction_safety_vision.paths import ProjectPaths

DETECTOR, SEGMENTER = MODELS

EXPECTED_PROTOCOL = "d92a157637baf52f9a7248bf24257d5a177e8496b7727ffb5afc874ab96d1c4d"
EXPECTED_BENCHMARK = "45059c2cdda1285b4fa6dbfdcfbeac6fedcad551e551e6810b1369af90f7f162"
DETECTOR_SHA = "0466f872a9de22898c70d8834cf1fcfb3d77f6c9fb27e81ee6248d3d19cbc206"
SEGMENTER_SHA = "29337d671459d0f742fb713613cc41d0eafcb8a21f5846ac0da65b2ffc024f20"


@pytest.fixture(scope="module")
def paths() -> ProjectPaths:
    """The project layout.

    Returns:
        The resolved paths.
    """
    return ProjectPaths.from_root()


@pytest.fixture(scope="module")
def benchmark_ids(paths: ProjectPaths) -> tuple[str, ...]:
    """The frozen benchmark subset, in the frozen order.

    Args:
        paths: Project layout.

    Returns:
        The ordered image ids.
    """
    table = paths.reports / "detector_segmenter_latency_membership.csv"
    with table.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return tuple(row["source_image_id"] for row in rows)


# --- the frozen protocol -------------------------------------------------------------


def test_protocol_fingerprint_is_the_frozen_one(paths: ProjectPaths) -> None:
    """The phase 10A configuration still hashes to the fingerprint 10C requires."""
    protocol = load_comparison_protocol(paths.configs / "detector_segmenter_comparison.yaml")
    assert protocol.fingerprint() == EXPECTED_PROTOCOL


def test_frozen_checkpoint_identities(paths: ProjectPaths) -> None:
    """The protocol still names exactly the two frozen checkpoints."""
    protocol = load_comparison_protocol(paths.configs / "detector_segmenter_comparison.yaml")
    assert protocol["detector"]["checkpoint_sha256"] == DETECTOR_SHA
    assert protocol["segmenter"]["checkpoint_sha256"] == SEGMENTER_SHA
    assert protocol["detector"]["experiment"] == DETECTOR
    assert protocol["segmenter"]["experiment"] == SEGMENTER


def test_frozen_benchmark_subset_fingerprint(benchmark_ids: tuple[str, ...]) -> None:
    """The committed benchmark subset reproduces its frozen ordered fingerprint."""
    assert ordered_fingerprint(list(benchmark_ids)) == EXPECTED_BENCHMARK


def test_benchmark_subset_holds_exactly_twenty_images(benchmark_ids: tuple[str, ...]) -> None:
    """The subset is the frozen size, with no duplicate."""
    assert len(benchmark_ids) == BENCHMARK_IMAGE_COUNT == 20
    assert len(set(benchmark_ids)) == BENCHMARK_IMAGE_COUNT


def test_benchmark_subset_is_ranked_by_its_own_digests(benchmark_ids: tuple[str, ...]) -> None:
    """The order is the digest rank, so no image was opened to choose it."""
    assert list(benchmark_ids) == sorted(benchmark_ids, key=stable_rank)


def test_frozen_latency_conditions(paths: ProjectPaths) -> None:
    """Batch, precision, warmup and repetitions are the frozen values."""
    protocol = load_comparison_protocol(paths.configs / "detector_segmenter_comparison.yaml")
    latency = protocol.latency
    assert int(latency["batch"]) == LATENCY_BATCH == 1
    assert int(latency["imgsz"]) == 768
    assert latency["precision"] == PRECISION == "FP32"
    assert int(latency["precision_value"]) == PRECISION_VALUE == 32
    assert int(latency["warmup_iterations"]) == WARMUP_ITERATIONS == 20
    assert int(latency["timed_iterations_per_image"]) == TIMED_ITERATIONS_PER_IMAGE == 30


def test_frozen_execution_order_is_not_randomized(paths: ProjectPaths) -> None:
    """The order is fixed in advance, interleaved and symmetric."""
    protocol = load_comparison_protocol(paths.configs / "detector_segmenter_comparison.yaml")
    order = protocol.latency["execution_order"]
    assert order["randomized"] is False
    assert order["interleaved"] is True
    assert order["symmetric"] is True


def test_frozen_synchronization_requirement(paths: ProjectPaths) -> None:
    """CUDA synchronisation is required on both edges with one primitive."""
    protocol = load_comparison_protocol(paths.configs / "detector_segmenter_comparison.yaml")
    synchronization = protocol.latency["synchronization"]
    assert synchronization["cuda_synchronize"] is True
    assert synchronization["before_timed_region"] is True
    assert synchronization["after_timed_region"] is True
    assert synchronization["same_primitive_for_both_models"] is True
    assert protocol.latency["timing_primitive"] == TIMING_PRIMITIVE


def test_frozen_boundaries_exclude_load_and_include_mask_reconstruction(
    paths: ProjectPaths,
) -> None:
    """Model load stays out of timing and mask reconstruction stays inside end-to-end."""
    protocol = load_comparison_protocol(paths.configs / "detector_segmenter_comparison.yaml")
    latency = protocol.latency
    assert "model loading" in latency["excluded_from_timed_region"]
    boundaries = latency["timing_boundaries"]
    assert boundaries[END_TO_END_LATENCY]["segmenter_mask_reconstruction_included"] is True
    excluded = boundaries[MODEL_INFERENCE_LATENCY]["excludes"]
    assert "mask reconstruction" in excluded
    assert "preprocessing" in excluded


def test_frozen_memory_protocol_forbids_training_memory(paths: ProjectPaths) -> None:
    """Peak stats reset after warmup, and training memory is never substituted."""
    protocol = load_comparison_protocol(paths.configs / "detector_segmenter_comparison.yaml")
    memory = protocol["memory_protocol"]
    assert memory["reset_peak_stats_after_warmup"] is True
    assert memory["training_memory_reused"] is False
    assert int(memory["batch"]) == LATENCY_BATCH


# --- the execution plan ---------------------------------------------------------------


def test_execution_plan_is_symmetric_in_both_directions() -> None:
    """Pass A leads with the detector and pass B leads with the segmenter."""
    plan = build_execution_plan([f"image{index:02d}" for index in range(BENCHMARK_IMAGE_COUNT)])
    assert len(plan) == BENCHMARK_IMAGE_COUNT * len(EXECUTION_PASSES) * len(MODELS)
    forward = [step for step in plan if step.execution_pass == PASS_FORWARD]
    reverse = [step for step in plan if step.execution_pass == PASS_REVERSE]
    assert [step.model for step in forward[:2]] == list(PASS_ORDER[PASS_FORWARD])
    assert [step.model for step in reverse[:2]] == list(PASS_ORDER[PASS_REVERSE])
    assert PASS_ORDER[PASS_FORWARD] != PASS_ORDER[PASS_REVERSE]


def test_execution_plan_never_runs_one_model_to_completion_first() -> None:
    """Consecutive blocks alternate models, which is what interleaving means."""
    plan = build_execution_plan([f"image{index:02d}" for index in range(BENCHMARK_IMAGE_COUNT)])
    models = [step.model for step in plan]
    # No model ever runs three blocks in a row; a pass boundary allows a pair.
    assert not any(models[i] == models[i + 1] == models[i + 2] for i in range(len(models) - 2))


def test_execution_plan_gives_both_models_the_same_images_equally_often() -> None:
    """Each model is timed once per image per pass."""
    plan = build_execution_plan([f"image{index:02d}" for index in range(BENCHMARK_IMAGE_COUNT)])
    for model in MODELS:
        blocks = [step for step in plan if step.model == model]
        assert len(blocks) == BENCHMARK_IMAGE_COUNT * len(EXECUTION_PASSES)
        assert sorted({step.source_image_id for step in blocks}) == sorted(
            {step.source_image_id for step in plan}
        )


def test_execution_plan_positions_are_contiguous() -> None:
    """The recorded position is the executed order."""
    plan = build_execution_plan([f"image{index:02d}" for index in range(BENCHMARK_IMAGE_COUNT)])
    assert [step.position for step in plan] == list(range(len(plan)))


def test_execution_plan_fingerprint_moves_with_the_order() -> None:
    """Reordering the membership changes the plan fingerprint."""
    ids = [f"image{index:02d}" for index in range(BENCHMARK_IMAGE_COUNT)]
    original = execution_plan_fingerprint(build_execution_plan(ids))
    swapped = list(ids)
    swapped[0], swapped[1] = swapped[1], swapped[0]
    assert execution_plan_fingerprint(build_execution_plan(swapped)) != original
    assert execution_plan_fingerprint(build_execution_plan(ids)) == original


def test_execution_plan_rejects_a_wrong_sized_subset() -> None:
    """A subset that is not the frozen size is refused."""
    with pytest.raises(CostBenchmarkError):
        build_execution_plan(["a", "b", "c"])


def test_execution_plan_rejects_a_duplicate() -> None:
    """A duplicated image would be timed twice and is refused."""
    ids = [f"image{index:02d}" for index in range(BENCHMARK_IMAGE_COUNT - 1)]
    with pytest.raises(CostBenchmarkError):
        build_execution_plan([*ids, ids[0]])


def test_expected_observation_count() -> None:
    """Blocks times boundaries times repetitions."""
    assert expected_observation_count() == 20 * 2 * 2 * 2 * 30 == 4800


# --- the latency statistics -----------------------------------------------------------


def test_describe_latency_reports_exactly_the_frozen_statistics() -> None:
    """No statistic outside the frozen set, and none inside it omitted."""
    statistics = describe_latency([1.0, 2.0, 3.0, 4.0])
    assert set(statistics) == set(STATISTIC_FIELDS)
    assert set(LATENCY_STATISTICS) <= set(statistics)


def test_describe_latency_arithmetic_on_a_synthetic_sample() -> None:
    """Every statistic is the documented convention over a known sample."""
    values = [float(value) for value in range(1, 101)]
    statistics = describe_latency(values)
    assert statistics["count"] == 100
    assert statistics["mean"] == 50.5
    assert statistics["median"] == 50.5
    assert statistics["p50"] == 50.5
    assert statistics["min"] == 1.0
    assert statistics["max"] == 100.0
    # Linear interpolation: p90 of 1..100 sits at index 89.1.
    assert statistics["p90"] == pytest.approx(90.1)
    assert statistics["p95"] == pytest.approx(95.05)
    assert statistics["p99"] == pytest.approx(99.01)


def test_describe_latency_uses_the_sample_standard_deviation() -> None:
    """The declared ddof=1 convention, not the population deviation."""
    statistics = describe_latency([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])
    assert statistics["std"] == pytest.approx(2.13809, abs=1e-5)


def test_describe_latency_median_equals_p50() -> None:
    """One percentile convention means the median and P50 cannot disagree."""
    for sample in ([1.0, 5.0, 9.0], [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0]):
        statistics = describe_latency(sample)
        assert statistics["median"] == statistics["p50"]


def test_describe_latency_percentiles_are_monotonic() -> None:
    """P50 <= P90 <= P95 <= P99 <= max, under one convention."""
    statistics = describe_latency([float(value) for value in range(1, 501)])
    ordered = [statistics[name] for name in ("p50", "p90", "p95", "p99", "max")]
    assert ordered == sorted(ordered)


def test_describe_latency_single_observation_has_no_deviation() -> None:
    """A sample of one has no sample standard deviation to report."""
    assert describe_latency([7.5])["std"] is None


def test_describe_latency_refuses_an_empty_sample() -> None:
    """An empty distribution is a defect, not a zero."""
    with pytest.raises(CostBenchmarkError):
        describe_latency([])


def test_percentile_convention_is_declared() -> None:
    """The convention is a named constant, so it cannot drift silently."""
    assert PERCENTILE_CONVENTION == "NUMPY_PERCENTILE_LINEAR_INTERPOLATION"


# --- the derived quantities -----------------------------------------------------------


def test_images_per_second_comes_from_the_mean() -> None:
    """Batch-1 throughput is 1000 divided by the mean latency."""
    assert images_per_second_from_mean(10.0) == 100.0
    assert images_per_second_from_mean(8.0) == 125.0


def test_images_per_second_is_not_derived_from_the_minimum() -> None:
    """The reciprocal of the fastest repetition is a different, larger number."""
    values = [5.0, 10.0, 15.0]
    statistics = describe_latency(values)
    assert images_per_second_from_mean(statistics["mean"]) == 100.0
    assert images_per_second_from_mean(statistics["min"]) == 200.0


def test_images_per_second_refuses_a_non_positive_mean() -> None:
    """A zero or negative mean is a defect, not an infinite throughput."""
    for mean in (0.0, -1.0):
        with pytest.raises(CostBenchmarkError):
            images_per_second_from_mean(mean)


def test_absolute_delta_is_segmenter_minus_detector() -> None:
    """The sign convention is S1 minus D2."""
    assert absolute_latency_delta_ms(10.0, 13.5) == 3.5
    assert absolute_latency_delta_ms(13.5, 10.0) == -3.5


def test_relative_cost_arithmetic() -> None:
    """The relative cost is the ratio minus one."""
    assert relative_latency_cost(10.0, 13.0) == 0.3
    assert relative_latency_cost(10.0, 10.0) == 0.0
    assert relative_latency_cost(10.0, 8.0) == -0.2


def test_relative_cost_refuses_a_non_positive_denominator() -> None:
    """A zero detector mean cannot be a denominator."""
    with pytest.raises(CostBenchmarkError):
        relative_latency_cost(0.0, 5.0)


def test_throughput_ratio_arithmetic() -> None:
    """The ratio is the segmenter's rate over the detector's."""
    assert throughput_ratio(100.0, 80.0) == 0.8


def test_throughput_ratio_is_the_inverse_of_the_relative_cost() -> None:
    """The two derived quantities are consistent by construction."""
    detector_mean, segmenter_mean = 9.157766, 11.914757
    relative = relative_latency_cost(detector_mean, segmenter_mean)
    ratio = throughput_ratio(
        images_per_second_from_mean(detector_mean),
        images_per_second_from_mean(segmenter_mean),
    )
    assert ratio == pytest.approx(1.0 / (1.0 + relative), abs=1e-6)


def test_build_cost_deltas_covers_both_boundaries_separately() -> None:
    """Each boundary gets its own delta, and they are never combined."""
    detector = {
        MODEL_INFERENCE_LATENCY: {"mean": 6.0},
        END_TO_END_LATENCY: {"mean": 9.0},
    }
    segmenter = {
        MODEL_INFERENCE_LATENCY: {"mean": 7.5},
        END_TO_END_LATENCY: {"mean": 12.0},
    }
    deltas = build_cost_deltas(detector, segmenter)
    assert set(deltas) == set(TIMING_BOUNDARIES)
    assert deltas[MODEL_INFERENCE_LATENCY]["absolute_latency_delta_ms"] == 1.5
    assert deltas[END_TO_END_LATENCY]["absolute_latency_delta_ms"] == 3.0
    assert deltas[MODEL_INFERENCE_LATENCY]["relative_latency_cost"] == 0.25
    assert all(block["combined_latency_score"] is False for block in deltas.values())


def test_build_cost_deltas_requires_both_models_at_a_boundary() -> None:
    """A missing boundary is refused rather than silently skipped."""
    with pytest.raises(CostBenchmarkError):
        build_cost_deltas({MODEL_INFERENCE_LATENCY: {"mean": 6.0}}, {})


# --- memory arithmetic ----------------------------------------------------------------


def test_as_gib_conversion() -> None:
    """Bytes convert to GiB at the reported precision."""
    assert as_gib(GIB) == 1.0
    assert as_gib(GIB // 2) == 0.5


def test_memory_delta_arithmetic() -> None:
    """Deltas and ratios recompute from the recorded peaks."""
    detector = {
        "peak_memory_allocated_bytes": 100 * 1024 * 1024,
        "peak_memory_reserved_bytes": 200 * 1024 * 1024,
    }
    segmenter = {
        "peak_memory_allocated_bytes": 300 * 1024 * 1024,
        "peak_memory_reserved_bytes": 400 * 1024 * 1024,
    }
    delta = memory_delta(detector, segmenter)
    assert delta["measurement"] == INFERENCE_MEMORY
    assert delta["peak_memory_allocated_delta_bytes"] == 200 * 1024 * 1024
    assert delta["peak_memory_allocated_ratio"] == 3.0
    assert delta["peak_memory_reserved_ratio"] == 2.0


def test_memory_delta_refuses_a_zero_denominator() -> None:
    """A zero detector peak cannot form a ratio."""
    with pytest.raises(CostBenchmarkError):
        memory_delta(
            {"peak_memory_allocated_bytes": 0, "peak_memory_reserved_bytes": 0},
            {"peak_memory_allocated_bytes": 1, "peak_memory_reserved_bytes": 1},
        )


# --- the raw timing schema ------------------------------------------------------------


def synthetic_observations() -> list[dict[str, object]]:
    """Build a complete synthetic observation set.

    Returns:
        One observation per model, boundary, block and repetition.
    """
    plan = build_execution_plan([f"image{index:02d}" for index in range(BENCHMARK_IMAGE_COUNT)])
    rows: list[dict[str, object]] = []
    for step in plan:
        for boundary in TIMING_BOUNDARIES:
            base = 6.0 if step.model == DETECTOR else 8.0
            if boundary == END_TO_END_LATENCY:
                base += 3.0
            for repetition in range(TIMED_ITERATIONS_PER_IMAGE):
                rows.append(
                    {
                        "model": step.model,
                        "benchmark_rank": step.benchmark_rank,
                        "source_image_id": step.source_image_id,
                        "execution_pass": step.execution_pass,
                        "execution_position": step.position,
                        "repetition": repetition,
                        "timing_type": boundary,
                        "latency_ms": base + repetition * 0.01,
                    }
                )
    return rows


def test_raw_observation_schema_is_the_frozen_one() -> None:
    """Every observation carries exactly the frozen fields."""
    for observation in synthetic_observations()[:5]:
        assert set(observation) == set(OBSERVATION_FIELDS)


def test_synthetic_observation_count_matches_the_expectation() -> None:
    """A complete benchmark produces the expected number of readings."""
    assert len(synthetic_observations()) == expected_observation_count()


def test_timing_dataset_fingerprint_moves_with_a_single_timing() -> None:
    """Changing one latency changes the raw-dataset digest."""
    rows = synthetic_observations()
    original = timing_dataset_fingerprint(rows)
    rows[17]["latency_ms"] = float(rows[17]["latency_ms"]) + 0.000002
    assert timing_dataset_fingerprint(rows) != original


def test_timing_dataset_fingerprint_moves_with_the_order() -> None:
    """The digest covers the executed order, not just the multiset."""
    rows = synthetic_observations()
    original = timing_dataset_fingerprint(rows)
    reordered = [rows[1], rows[0], *rows[2:]]
    assert timing_dataset_fingerprint(reordered) != original


def test_timing_dataset_fingerprint_rejects_an_incomplete_row() -> None:
    """A row missing a schema field is a defect."""
    rows = synthetic_observations()[:1]
    del rows[0]["repetition"]
    with pytest.raises(CostBenchmarkError):
        timing_dataset_fingerprint(rows)


def test_block_schema_carries_no_holdout_identifier() -> None:
    """The committed per-block schema names no split."""
    assert "split" not in BLOCK_FIELDS
    assert all("test" not in field for field in BLOCK_FIELDS)


# --- precision parity -----------------------------------------------------------------


def fp32_evidence() -> dict[str, object]:
    """Evidence describing a model that resolved to FP32.

    Returns:
        One model's precision evidence.
    """
    return {
        "backend_fp16_flag": False,
        "resolved_precision_argument": "quantize",
        "resolved_precision_value": 32,
        "parameter_dtypes": ["torch.float32"],
        "float_buffer_dtypes": ["torch.float32"],
        "input_dtype": "torch.float32",
        "autocast_enabled_during_forward": False,
        "quantization_config_present": False,
        "device_type": "cuda",
    }


def test_precision_parity_accepts_identical_fp32_evidence() -> None:
    """Two models that both resolved to FP32 pass the parity check."""
    parity = assert_precision_parity({DETECTOR: fp32_evidence(), SEGMENTER: fp32_evidence()})
    assert parity["identical_across_models"] is True
    assert parity["frozen_intent"] == PRECISION


def test_precision_parity_rejects_a_difference_between_the_models() -> None:
    """One model in FP16 would make the benchmark a precision comparison."""
    segmenter = fp32_evidence()
    segmenter["input_dtype"] = "torch.float16"
    with pytest.raises(PrecisionParityError):
        assert_precision_parity({DETECTOR: fp32_evidence(), SEGMENTER: segmenter})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("backend_fp16_flag", True),
        ("parameter_dtypes", ["torch.float16"]),
        ("input_dtype", "torch.float16"),
        ("autocast_enabled_during_forward", True),
        ("quantization_config_present", True),
        ("resolved_precision_value", 16),
    ],
)
def test_precision_parity_rejects_any_departure_from_fp32(field: str, value: object) -> None:
    """Agreeing on the wrong precision is still the wrong precision."""
    evidence = fp32_evidence()
    evidence[field] = value
    with pytest.raises(PrecisionParityError):
        assert_precision_parity({DETECTOR: dict(evidence), SEGMENTER: dict(evidence)})


def test_precision_parity_requires_both_models() -> None:
    """Evidence for one model is not parity."""
    with pytest.raises(PrecisionParityError):
        assert_precision_parity({DETECTOR: fp32_evidence()})


def test_precision_evidence_schema_matches_the_phase_10b_probe(paths: ProjectPaths) -> None:
    """Phase 10C's probe records the same fields phase 10B's did.

    Phase 10B's runner was deliberately left byte-identical rather than
    refactored to share this code, so this pins the duplicate to the committed
    schema instead of trusting that the two stayed in step.
    """
    committed = json.loads(
        (paths.reports / "detector_segmenter_box_comparison.json").read_text(encoding="utf-8")
    )
    for model in MODELS:
        assert set(committed["precision_preflight"]["evidence"][model]) == set(fp32_evidence())


# --- the latency validator ------------------------------------------------------------


def latency_payload() -> dict[str, object]:
    """A minimal latency artifact that validates.

    Returns:
        The synthetic artifact.
    """
    observations = synthetic_observations()
    latency: dict[str, object] = {}
    for model in MODELS:
        block: dict[str, object] = {"throughput": {}}
        for boundary in TIMING_BOUNDARIES:
            values = [
                float(row["latency_ms"])
                for row in observations
                if row["model"] == model and row["timing_type"] == boundary
            ]
            statistics = describe_latency(values)
            block[boundary] = statistics
            block["throughput"][boundary] = images_per_second_from_mean(statistics["mean"])
        latency[model] = block

    return {
        "phase": PHASE,
        "status": COST_BENCHMARK_COMPLETE,
        "benchmark_status": "CONTROLLED_LOCAL_HARDWARE_BENCHMARK",
        "protocol_fingerprint": EXPECTED_PROTOCOL,
        "detector": {
            "experiment": DETECTOR,
            "imgsz": 768,
            "checkpoint_sha256": DETECTOR_SHA,
            "trained_in_this_phase": False,
        },
        "segmenter": {
            "experiment": SEGMENTER,
            "imgsz": 768,
            "checkpoint_sha256": SEGMENTER_SHA,
            "trained_in_this_phase": False,
        },
        "benchmark": {
            "split": "validation",
            "benchmark_image_count": BENCHMARK_IMAGE_COUNT,
            "benchmark_membership_sha256": EXPECTED_BENCHMARK,
            "execution_plan_sha256": "plan",
            "execution_passes": list(EXECUTION_PASSES),
            "interleaved": True,
            "symmetric": True,
            "randomized": False,
            "images_reselected": False,
        },
        "conditions": {
            "batch": 1,
            "imgsz": 768,
            "precision": PRECISION,
            "warmup_iterations": WARMUP_ITERATIONS,
            "timed_iterations_per_image": TIMED_ITERATIONS_PER_IMAGE,
            "augment": False,
            "tta": False,
            "timing_primitive": TIMING_PRIMITIVE,
            "synchronization_primitive": SYNCHRONIZATION_PRIMITIVE,
            "cuda_synchronize_before_timed_region": True,
            "cuda_synchronize_after_timed_region": True,
            "same_primitive_for_both_models": True,
            "percentile_convention": PERCENTILE_CONVENTION,
            "torch_compile": False,
            "tensorrt": False,
            "onnx": False,
            "model_load_excluded_from_timing": True,
            "disk_decode_excluded_from_timing": True,
            "predictions_saved_during_timing": False,
        },
        "timing_boundaries": {
            MODEL_INFERENCE_LATENCY: {
                "preprocessing_included": False,
                "mask_reconstruction_included": False,
            },
            END_TO_END_LATENCY: {
                "preprocessing_included": True,
                "postprocessing_included": True,
                "segmenter_mask_reconstruction_included": True,
                "outputs_materialized_before_end_timestamp": True,
            },
        },
        "latency": latency,
        "deltas": build_cost_deltas(latency[DETECTOR], latency[SEGMENTER]),
        "combined_latency_score": False,
        "winner_declared": False,
        "raw_timings": {
            "schema": list(OBSERVATION_FIELDS),
            "observations": expected_observation_count(),
            "timing_dataset_sha256": timing_dataset_fingerprint(observations),
        },
        "model_complexity": {
            "label": STATIC_MODEL_COMPLEXITY,
            "measured_at_benchmark_input_size": False,
        },
        "segmentation_cost_label": SEGMENTATION_COST_LABEL,
        "pure_mask_reconstruction_cost_isolated": False,
        "models_trained_in_this_phase": 0,
        "thresholds_tuned": 0,
        "ap_metrics_recomputed": False,
        "spatial_analysis_rerun": False,
        "test": {"status": "PROTECTED_NOT_ACCESSED"},
        "holdout_accessed": False,
    }


def validate(payload: dict[str, object]) -> list[str]:
    """Run the latency validator with the frozen identities.

    Args:
        payload: The artifact to check.

    Returns:
        The problems found.
    """
    return validate_latency_result(
        payload,
        protocol_fingerprint=EXPECTED_PROTOCOL,
        detector_sha256=DETECTOR_SHA,
        segmenter_sha256=SEGMENTER_SHA,
        benchmark_sha256=EXPECTED_BENCHMARK,
        execution_plan_sha256="plan",
    )


def test_a_well_formed_latency_artifact_validates() -> None:
    """The synthetic artifact passes, so later failures mean something."""
    assert validate(latency_payload()) == []


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("phase",), "10D"),
        (("status",), "PENDING"),
        (("detector", "checkpoint_sha256"), "0" * 64),
        (("segmenter", "checkpoint_sha256"), "0" * 64),
        (("detector", "imgsz"), 640),
        (("detector", "trained_in_this_phase"), True),
        (("benchmark", "benchmark_membership_sha256"), "0" * 64),
        (("benchmark", "execution_plan_sha256"), "other"),
        (("benchmark", "benchmark_image_count"), 19),
        (("benchmark", "randomized"), True),
        (("benchmark", "interleaved"), False),
        (("benchmark", "images_reselected"), True),
        (("conditions", "batch"), 2),
        (("conditions", "precision"), "FP16"),
        (("conditions", "warmup_iterations"), 10),
        (("conditions", "timed_iterations_per_image"), 50),
        (("conditions", "cuda_synchronize_before_timed_region"), False),
        (("conditions", "cuda_synchronize_after_timed_region"), False),
        (("conditions", "same_primitive_for_both_models"), False),
        (("conditions", "timing_primitive"), "time.time"),
        (("conditions", "model_load_excluded_from_timing"), False),
        (("conditions", "disk_decode_excluded_from_timing"), False),
        (("conditions", "predictions_saved_during_timing"), True),
        (("conditions", "torch_compile"), True),
        (("conditions", "tensorrt"), True),
        (("combined_latency_score",), True),
        (("winner_declared",), True),
        (("models_trained_in_this_phase",), 1),
        (("thresholds_tuned",), 1),
        (("ap_metrics_recomputed",), True),
        (("spatial_analysis_rerun",), True),
        (("pure_mask_reconstruction_cost_isolated",), True),
        (("holdout_accessed",), True),
        (("test", "status"), "EVALUATED"),
        (("model_complexity", "measured_at_benchmark_input_size"), True),
    ],
)
def test_latency_validator_rejects_a_protocol_departure(
    path: tuple[str, ...], value: object
) -> None:
    """Every frozen invariant is enforced, not merely documented."""
    payload = latency_payload()
    target: object = payload
    for key in path[:-1]:
        target = target[key]  # type: ignore[index]
    target[path[-1]] = value  # type: ignore[index]
    assert validate(payload) != []


def test_latency_validator_requires_mask_reconstruction_inside_end_to_end() -> None:
    """Excluding it would hide the cost the comparison exists to quantify."""
    payload = latency_payload()
    payload["timing_boundaries"][END_TO_END_LATENCY][  # type: ignore[index]
        "segmenter_mask_reconstruction_included"
    ] = False
    problems = validate(payload)
    assert any("mask reconstruction" in problem for problem in problems)


def test_latency_validator_requires_mask_reconstruction_outside_model_inference() -> None:
    """The narrower boundary is the model core, not the pipeline."""
    payload = latency_payload()
    payload["timing_boundaries"][MODEL_INFERENCE_LATENCY][  # type: ignore[index]
        "mask_reconstruction_included"
    ] = True
    assert validate(payload) != []


def test_latency_validator_rejects_a_missing_boundary() -> None:
    """Both boundaries must be measured; one is not the protocol."""
    payload = latency_payload()
    del payload["timing_boundaries"][END_TO_END_LATENCY]  # type: ignore[index]
    assert validate(payload) != []


def test_latency_validator_rejects_a_wrong_observation_count() -> None:
    """A short benchmark is not the frozen benchmark."""
    payload = latency_payload()
    payload["raw_timings"]["observations"] = 4000  # type: ignore[index]
    assert validate(payload) != []


def test_latency_validator_rejects_a_changed_raw_schema() -> None:
    """The observation schema is part of the protocol."""
    payload = latency_payload()
    payload["raw_timings"]["schema"] = ["model", "latency_ms"]  # type: ignore[index]
    assert validate(payload) != []


def test_latency_validator_recomputes_every_delta() -> None:
    """A hand-edited delta does not survive validation."""
    for field in ("absolute_latency_delta_ms", "relative_latency_cost", "throughput_ratio"):
        payload = latency_payload()
        payload["deltas"][END_TO_END_LATENCY][field] = 999.0  # type: ignore[index]
        problems = validate(payload)
        assert any(field in problem for problem in problems)


def test_latency_validator_recomputes_throughput_from_the_mean() -> None:
    """A throughput that does not follow from the mean is rejected."""
    payload = latency_payload()
    payload["latency"][DETECTOR]["throughput"][END_TO_END_LATENCY] = 1000.0  # type: ignore[index]
    assert validate(payload) != []


def test_latency_validator_rejects_a_wrong_statistic_count() -> None:
    """Each distribution must hold every timed observation for that model."""
    payload = latency_payload()
    payload["latency"][SEGMENTER][END_TO_END_LATENCY]["count"] = 600  # type: ignore[index]
    assert validate(payload) != []


def test_latency_validator_rejects_a_missing_statistic() -> None:
    """The frozen statistic set is complete or the artifact is invalid."""
    payload = latency_payload()
    del payload["latency"][DETECTOR][MODEL_INFERENCE_LATENCY]["p99"]  # type: ignore[index]
    assert validate(payload) != []


def test_latency_validator_rejects_a_holdout_mention() -> None:
    """The protected split may not appear outside the holdout declaration."""
    payload = latency_payload()
    payload["benchmark"]["split"] = "test"  # type: ignore[index]
    assert validate(payload) != []


# --- the memory validator -------------------------------------------------------------


def memory_payload() -> dict[str, object]:
    """A minimal memory artifact that validates.

    Returns:
        The synthetic artifact.
    """
    memory = {
        DETECTOR: {
            "pre_load_allocated_bytes": 0,
            "pre_load_reserved_bytes": 0,
            "baseline_allocated_bytes": 44005888,
            "baseline_reserved_bytes": 92274688,
            "peak_memory_allocated_bytes": 78815744,
            "peak_memory_reserved_bytes": 134217728,
            "peak_memory_allocated_gib": as_gib(78815744),
            "peak_memory_reserved_gib": as_gib(134217728),
            "other_model_resident": False,
            "images_measured": BENCHMARK_IMAGE_COUNT,
        },
        SEGMENTER: {
            "pre_load_allocated_bytes": 0,
            "pre_load_reserved_bytes": 0,
            "baseline_allocated_bytes": 46727680,
            "baseline_reserved_bytes": 106954752,
            "peak_memory_allocated_bytes": 248700928,
            "peak_memory_reserved_bytes": 318767104,
            "peak_memory_allocated_gib": as_gib(248700928),
            "peak_memory_reserved_gib": as_gib(318767104),
            "other_model_resident": False,
            "images_measured": BENCHMARK_IMAGE_COUNT,
        },
    }
    return {
        "phase": PHASE,
        "measurement": INFERENCE_MEMORY,
        "protocol_fingerprint": EXPECTED_PROTOCOL,
        "detector": {"checkpoint_sha256": DETECTOR_SHA},
        "segmenter": {"checkpoint_sha256": SEGMENTER_SHA},
        "conditions": {
            "batch": 1,
            "imgsz": 768,
            "precision": PRECISION,
            "warmup_iterations": WARMUP_ITERATIONS,
            "peak_stats_reset_after_warmup": True,
            "isolation_method": MEMORY_ISOLATION,
            "isolation_diagnostic": MEMORY_ISOLATION_DIAGNOSTIC,
            "models_resident_during_measurement": 1,
        },
        "memory": memory,
        "delta": memory_delta(memory[DETECTOR], memory[SEGMENTER]),
        "training_memory_reused": False,
        "training_memory": TRAINING_MEMORY_LABEL,
        "test": {"status": "PROTECTED_NOT_ACCESSED"},
        "holdout_accessed": False,
    }


def validate_memory(payload: dict[str, object]) -> list[str]:
    """Run the memory validator with the frozen identities.

    Args:
        payload: The artifact to check.

    Returns:
        The problems found.
    """
    return validate_memory_result(
        payload,
        protocol_fingerprint=EXPECTED_PROTOCOL,
        detector_sha256=DETECTOR_SHA,
        segmenter_sha256=SEGMENTER_SHA,
    )


def test_a_well_formed_memory_artifact_validates() -> None:
    """The synthetic memory artifact passes."""
    assert validate_memory(memory_payload()) == []


def test_memory_schema_is_the_frozen_field_set() -> None:
    """Each model reports exactly the recorded memory fields."""
    payload = memory_payload()
    for model in MODELS:
        assert set(MEMORY_FIELDS) <= set(payload["memory"][model])  # type: ignore[index]


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("phase",), "10B"),
        (("measurement",), "TRAINING_MEMORY"),
        (("training_memory_reused",), True),
        (("training_memory",), "S1_TRAINING_PEAK"),
        (("protocol_fingerprint",), "0" * 64),
        (("detector", "checkpoint_sha256"), "0" * 64),
        (("segmenter", "checkpoint_sha256"), "0" * 64),
        (("conditions", "batch"), 8),
        (("conditions", "imgsz"), 640),
        (("conditions", "precision"), "FP16"),
        (("conditions", "warmup_iterations"), 5),
        (("conditions", "peak_stats_reset_after_warmup"), False),
        (("conditions", "isolation_method"), "BOTH_MODELS_RESIDENT"),
        (("conditions", "models_resident_during_measurement"), 2),
        (("holdout_accessed",), True),
        (("test", "status"), "EVALUATED"),
    ],
)
def test_memory_validator_rejects_a_protocol_departure(
    path: tuple[str, ...], value: object
) -> None:
    """Every frozen memory invariant is enforced."""
    payload = memory_payload()
    target: object = payload
    for key in path[:-1]:
        target = target[key]  # type: ignore[index]
    target[path[-1]] = value  # type: ignore[index]
    assert validate_memory(payload) != []


def test_memory_validator_requires_an_empty_allocator_before_the_load() -> None:
    """A contaminated allocator cannot be attributed to one model."""
    payload = memory_payload()
    payload["memory"][SEGMENTER]["pre_load_allocated_bytes"] = 33554432  # type: ignore[index]
    problems = validate_memory(payload)
    assert any("allocator must be empty" in problem for problem in problems)


def test_memory_validator_rejects_another_resident_model() -> None:
    """A second resident model would be charged to the one being measured."""
    payload = memory_payload()
    payload["memory"][DETECTOR]["other_model_resident"] = True  # type: ignore[index]
    assert validate_memory(payload) != []


def test_memory_validator_rejects_allocated_above_reserved() -> None:
    """The allocator cannot hand out more than it reserved."""
    payload = memory_payload()
    payload["memory"][DETECTOR]["peak_memory_allocated_bytes"] = 10**12  # type: ignore[index]
    assert validate_memory(payload) != []


def test_memory_validator_recomputes_the_gib_conversion() -> None:
    """A hand-written GiB figure does not survive validation."""
    payload = memory_payload()
    payload["memory"][SEGMENTER]["peak_memory_reserved_gib"] = 1.0  # type: ignore[index]
    assert validate_memory(payload) != []


def test_memory_validator_recomputes_the_delta() -> None:
    """A hand-edited memory delta is rejected."""
    payload = memory_payload()
    payload["delta"]["peak_memory_reserved_ratio"] = 1.0  # type: ignore[index]
    assert validate_memory(payload) != []


def test_memory_validator_rejects_a_missing_model() -> None:
    """Both models must be measured."""
    payload = memory_payload()
    del payload["memory"][SEGMENTER]  # type: ignore[index]
    assert validate_memory(payload) != []


# --- what phase 10C must not do -------------------------------------------------------


def test_the_cost_module_imports_no_model_framework() -> None:
    """The arithmetic module is importable without torch or ultralytics.

    Its precision probe imports them lazily, inside the function, so a test
    suite can exercise every statistic and validator without a GPU.
    """
    source = Path("src/construction_safety_vision/detector_segmenter_cost.py").read_text(
        encoding="utf-8"
    )
    for line in source.splitlines():
        stripped = line.strip()
        if stripped.startswith(("import ", "from ")) and not line.startswith((" ", "\t")):
            assert "torch" not in stripped
            assert "ultralytics" not in stripped


def test_the_benchmark_runner_contains_no_training_call() -> None:
    """Phase 10C trains nothing, and that is asserted rather than promised."""
    source = Path("scripts/benchmark_detector_segmenter.py").read_text(encoding="utf-8")
    for forbidden in (
        ".train(",
        "model.train",
        "trainer",
        "DetectionTrainer",
        "SegmentationTrainer",
    ):
        assert forbidden not in source


def test_the_benchmark_runner_recomputes_no_average_precision() -> None:
    """No evaluator is invoked or imported: recognition belongs to phase 10B.

    The runner does name a committed canonical-evaluation artifact, in the list
    of files it must leave byte-identical. Naming a file whose digest is checked
    is the opposite of recomputing it, so the assertion targets the import and
    call paths rather than the string.
    """
    source = Path("scripts/benchmark_detector_segmenter.py").read_text(encoding="utf-8")
    for forbidden in (
        "COCOeval",
        "pycocotools",
        "import canonical_evaluation",
        "canonical_evaluation import",
        "mask_iou_evaluation import",
        "evaluate_boxes",
        "summarize()",
    ):
        assert forbidden not in source


def test_the_benchmark_runner_reruns_no_spatial_analysis() -> None:
    """No spatial or association measurement is computed or imported."""
    source = Path("scripts/benchmark_detector_segmenter.py").read_text(encoding="utf-8")
    for forbidden in (
        "associate(",
        "association_category",
        "mask_containment",
        "mask_intersection",
        "shape_extent",
        "centroid_displacement",
        "detector_segmenter_analysis import Instance",
    ):
        assert forbidden not in source


def test_the_benchmark_runner_names_no_holdout_artifact() -> None:
    """No holdout path, document or identifier appears in the runner."""
    source = Path("scripts/benchmark_detector_segmenter.py").read_text(encoding="utf-8")
    for forbidden in ("images/test", "_test.coco.json", "detection_test", "segmentation_test"):
        assert forbidden not in source


def test_the_benchmark_runner_never_unlocks_the_holdout() -> None:
    """The runner reads the unlock variable to refuse, never to set it."""
    source = Path("scripts/benchmark_detector_segmenter.py").read_text(encoding="utf-8")
    assert "holdout_unlocked()" in source
    assert "CSVISION_ALLOW_TEST_SPLIT=1" not in source
    assert "environ[" not in source


def test_the_benchmark_runner_synchronizes_around_every_timed_region() -> None:
    """Every perf_counter reading is bracketed by a CUDA synchronise."""
    source = Path("scripts/benchmark_detector_segmenter.py").read_text(encoding="utf-8")
    lines = source.splitlines()
    starts = [index for index, line in enumerate(lines) if "start = time.perf_counter()" in line]
    assert starts, "the runner takes no timing"
    for index in starts:
        assert "torch.cuda.synchronize()" in lines[index - 1]
    stops = [index for index, line in enumerate(lines) if "time.perf_counter() - start" in line]
    assert len(stops) == len(starts)
    for index in stops:
        assert "torch.cuda.synchronize()" in lines[index - 1]


def test_the_benchmark_runner_uses_one_timing_primitive() -> None:
    """Both models are timed with perf_counter and nothing else."""
    source = Path("scripts/benchmark_detector_segmenter.py").read_text(encoding="utf-8")
    assert "time.perf_counter" in source
    for forbidden in ("time.time(", "time.monotonic(", "cuda.Event", "elapsed_time("):
        assert forbidden not in source


def test_the_benchmark_runner_resets_peak_memory_stats() -> None:
    """The memory measurement resets the peak after warmup, as frozen."""
    source = Path("scripts/benchmark_detector_segmenter.py").read_text(encoding="utf-8")
    assert "torch.cuda.reset_peak_memory_stats()" in source
    assert "torch.cuda.max_memory_allocated()" in source
    assert "torch.cuda.max_memory_reserved()" in source


def test_the_benchmark_runner_does_not_export_or_compile_a_model() -> None:
    """No export or compilation path exists, for either model.

    The artifact records ``torch_compile``, ``tensorrt`` and ``onnx`` as false;
    this asserts there is no call that could make any of them true.
    """
    source = Path("scripts/benchmark_detector_segmenter.py").read_text(encoding="utf-8")
    for forbidden in ("torch.compile(", ".export(", ".engine", "to_onnx", "trt_"):
        assert forbidden not in source


# --- the post-hoc distribution diagnostic ---------------------------------------------


def latency_statistics_from(observations: list[dict[str, object]]) -> dict[str, object]:
    """Summarise synthetic observations the way the artifact does.

    Args:
        observations: The raw observations.

    Returns:
        Per-model, per-boundary statistics.
    """
    latency: dict[str, object] = {}
    for model in MODELS:
        block: dict[str, object] = {}
        for boundary in TIMING_BOUNDARIES:
            block[boundary] = describe_latency(
                [
                    float(row["latency_ms"])
                    for row in observations
                    if row["model"] == model and row["timing_type"] == boundary
                ]
            )
        latency[model] = block
    return latency


def test_distribution_diagnostic_is_labelled_post_hoc() -> None:
    """It was written after the run, so it says so and decides nothing."""
    observations = synthetic_observations()
    diagnostic = describe_distribution_shape(observations, latency_statistics_from(observations))
    assert diagnostic["label"] == DISTRIBUTION_DIAGNOSTIC_LABEL
    assert diagnostic["status"] == DISTRIBUTION_DIAGNOSTIC_STATUS
    for flag in (
        "replaces_frozen_statistics",
        "is_a_frozen_phase_10a_metric",
        "is_a_selection_rule",
        "is_a_significance_test",
        "changes_any_frozen_number",
    ):
        assert diagnostic[flag] is False


def test_distribution_diagnostic_discards_no_observation() -> None:
    """The complete distribution is preserved; nothing is trimmed or rescaled."""
    observations = synthetic_observations()
    diagnostic = describe_distribution_shape(observations, latency_statistics_from(observations))
    assert diagnostic["observations_used"] == expected_observation_count()
    assert diagnostic["observations_discarded"] == 0
    assert diagnostic["outlier_rejection_applied"] is False
    assert diagnostic["observations_normalized_or_rescaled"] is False
    assert diagnostic["computed_from"] == "ALL_VALID_OBSERVATIONS"


def test_distribution_diagnostic_computes_no_per_mode_statistics() -> None:
    """No mode split was declared by the protocol, so none is invented."""
    observations = synthetic_observations()
    diagnostic = describe_distribution_shape(observations, latency_statistics_from(observations))
    assert diagnostic["per_mode_statistics_computed"] is False
    assert diagnostic["mode_split_threshold_declared_by_protocol"] is False


def test_distribution_diagnostic_records_the_absent_telemetry() -> None:
    """No device-state reading accompanied the timings, and that is recorded."""
    observations = synthetic_observations()
    diagnostic = describe_distribution_shape(observations, latency_statistics_from(observations))
    assert diagnostic["per_observation_power_state_telemetry"] == NO_POWER_TELEMETRY
    assert diagnostic["telemetry_synchronous_with_timed_blocks"] is False
    assert diagnostic["causal_attribution"] == CAUSAL_ATTRIBUTION == "UNKNOWN"
    assert diagnostic["hypothesis_status"] == DVFS_HYPOTHESIS_STATUS == "UNTESTED_HYPOTHESIS"


def test_distribution_diagnostic_claims_no_proportionality() -> None:
    """Two models sharing a shape is not a demonstration that one scales the other."""
    observations = synthetic_observations()
    diagnostic = describe_distribution_shape(observations, latency_statistics_from(observations))
    assert diagnostic["proportionality_across_models_demonstrated"] is False


def test_distribution_diagnostic_hypothesis_is_worded_as_a_hypothesis() -> None:
    """The wording may not assert a cause the benchmark cannot establish."""
    lowered = DVFS_HYPOTHESIS.lower()
    assert "consistent with" in lowered
    for forbidden in ("caused by", "definitively", "proves", "demonstrates that"):
        assert forbidden not in lowered


def test_distribution_diagnostic_arithmetic_is_exact() -> None:
    """The reported ratios and block aggregates recompute from the observations."""
    observations = synthetic_observations()
    latency = latency_statistics_from(observations)
    diagnostic = describe_distribution_shape(observations, latency)
    for model in MODELS:
        for boundary in TIMING_BOUNDARIES:
            statistics = latency[model][boundary]
            block = diagnostic["evidence"][model][boundary]
            assert block["mean_to_median_ratio"] == pytest.approx(
                statistics["mean"] / statistics["median"], abs=1e-6
            )
            assert block["min_over_max"] == pytest.approx(
                statistics["min"] / statistics["max"], abs=1e-6
            )
            assert block["timed_blocks"] == BENCHMARK_IMAGE_COUNT * len(EXECUTION_PASSES)
            assert block["block_mean_min"] <= block["block_mean_max"]
            assert set(block["per_pass_block_mean_of_means"]) == set(EXECUTION_PASSES)


def test_distribution_diagnostic_rejects_an_incomplete_observation() -> None:
    """A row missing a schema field is a defect, not a gap to work around."""
    observations = synthetic_observations()
    latency = latency_statistics_from(observations)
    del observations[0]["execution_pass"]
    with pytest.raises(CostBenchmarkError):
        describe_distribution_shape(observations, latency)


# --- protocol stability after observation ---------------------------------------------


def test_protocol_stability_declares_every_way_it_could_have_been_bent() -> None:
    """All fourteen fields are present and all are false."""
    stability = build_protocol_stability()
    for field in PROTOCOL_STABILITY_FIELDS:
        assert stability[field] is False, field
    assert len(PROTOCOL_STABILITY_FIELDS) == 14


def test_protocol_stability_records_a_rerun_honestly() -> None:
    """A second execution would be recorded, not hidden."""
    assert build_protocol_stability(rerun=True)["benchmark_rerun"] is True


def test_latency_validator_rejects_a_bent_protocol_declaration() -> None:
    """Any true stability field fails validation."""
    for field in PROTOCOL_STABILITY_FIELDS:
        payload = latency_payload()
        payload["protocol_stability_after_observation"] = build_protocol_stability()
        payload["protocol_stability_after_observation"][field] = True
        assert validate(payload) != [], field


def test_latency_validator_rejects_a_missing_stability_field() -> None:
    """An incomplete declaration is not a declaration."""
    payload = latency_payload()
    stability = build_protocol_stability()
    del stability["observations_removed"]
    payload["protocol_stability_after_observation"] = stability
    assert validate(payload) != []


def test_latency_validator_rejects_a_causal_claim_in_the_diagnostic() -> None:
    """The cause must stay UNKNOWN while no telemetry exists."""
    observations = synthetic_observations()
    payload = latency_payload()
    payload["post_hoc_distribution_diagnostic"] = describe_distribution_shape(
        observations, latency_statistics_from(observations)
    )
    assert validate(payload) == []
    payload["post_hoc_distribution_diagnostic"]["causal_attribution"] = "GPU_POWER_STATE"
    assert validate(payload) != []


def test_latency_validator_rejects_a_discarded_observation() -> None:
    """Trimming the distribution fails validation."""
    observations = synthetic_observations()
    payload = latency_payload()
    payload["post_hoc_distribution_diagnostic"] = describe_distribution_shape(
        observations, latency_statistics_from(observations)
    )
    payload["post_hoc_distribution_diagnostic"]["observations_discarded"] = 4
    assert validate(payload) != []


def test_latency_validator_rejects_a_promoted_diagnostic() -> None:
    """The diagnostic may never claim to replace a frozen statistic."""
    observations = synthetic_observations()
    payload = latency_payload()
    payload["post_hoc_distribution_diagnostic"] = describe_distribution_shape(
        observations, latency_statistics_from(observations)
    )
    payload["post_hoc_distribution_diagnostic"]["replaces_frozen_statistics"] = True
    assert validate(payload) != []


def test_latency_validator_rejects_a_diagnostic_on_a_subset() -> None:
    """A diagnostic computed from part of the data is refused."""
    observations = synthetic_observations()
    payload = latency_payload()
    payload["post_hoc_distribution_diagnostic"] = describe_distribution_shape(
        observations, latency_statistics_from(observations)
    )
    payload["post_hoc_distribution_diagnostic"]["observations_used"] = 1200
    assert validate(payload) != []


# --- the no-execution rebuild ---------------------------------------------------------


def test_rebuild_results_executes_no_model_and_takes_no_timing() -> None:
    """The rebuild path contains no inference, no timing and no memory call."""
    source = Path("scripts/benchmark_detector_segmenter.py").read_text(encoding="utf-8")
    start = source.index("def rebuild_results()")
    body = source[start : source.index("\ndef rebuild_report()")]
    for forbidden in (
        "prepare_model",
        "predictor.inference",
        "predictor.preprocess",
        "predictor.postprocess",
        "time.perf_counter",
        "torch.cuda",
        "benchmark_latency",
        "measure_memory",
        "memory_in_child",
        "precision_evidence",
        "run_block",
        "YOLO(",
    ):
        assert forbidden not in body, forbidden


def test_rebuild_results_consumes_the_persisted_timings() -> None:
    """It reads the recorded observations rather than producing new ones."""
    source = Path("scripts/benchmark_detector_segmenter.py").read_text(encoding="utf-8")
    start = source.index("def rebuild_results()")
    body = source[start : source.index("\ndef rebuild_report()")]
    assert "load_persisted_observations(paths)" in body
    assert "timing_dataset_fingerprint(observations)" in body


def test_rebuild_results_asserts_the_frozen_numbers_did_not_move() -> None:
    """The rebuild refuses to write if a measured value changed."""
    source = Path("scripts/benchmark_detector_segmenter.py").read_text(encoding="utf-8")
    start = source.index("def rebuild_results()")
    body = source[start : source.index("\ndef rebuild_report()")]
    assert 'unchanged = ("latency", "deltas")' in body
    assert "A rebuild may not alter a measured value." in body


def test_rebuild_report_executes_no_model_and_takes_no_timing() -> None:
    """The report rebuild only re-renders prose from committed artifacts."""
    source = Path("scripts/benchmark_detector_segmenter.py").read_text(encoding="utf-8")
    start = source.index("def rebuild_report()")
    body = source[start : source.index("\ndef run_memory_child(")]
    for forbidden in (
        "prepare_model",
        "time.perf_counter",
        "torch.cuda",
        "benchmark_latency",
        "measure_memory",
        "describe_latency",
        "build_cost_deltas",
        "YOLO(",
    ):
        assert forbidden not in body, forbidden
    assert "if before != after:" in body


def test_the_percentile_definition_has_one_implementation() -> None:
    """Neither rebuild path may redefine a statistic.

    Every reported statistic comes from ``describe_latency``, so a rebuild
    cannot quietly change a percentile convention.
    """
    source = Path("src/construction_safety_vision/detector_segmenter_cost.py").read_text(
        encoding="utf-8"
    )
    assert source.count("np.percentile") == 4  # p50, p90, p95, p99, in one function
    assert source.count("def describe_latency") == 1
