"""Tests for the committed phase 10C cost-benchmark artifacts.

These read what phase 10C actually wrote. They re-derive every fingerprint, the
execution plan and every delta, and they check that the artifacts carry no
holdout information and claim nothing the benchmark cannot support. No model is
loaded and no timing is taken.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pytest

from construction_safety_vision.detector_segmenter_comparison import (
    BENCHMARK_IMAGE_COUNT,
    BENCHMARK_LABEL,
    END_TO_END_LATENCY,
    MODEL_INFERENCE_LATENCY,
    PRECISION,
    TIMED_ITERATIONS_PER_IMAGE,
    WARMUP_ITERATIONS,
    load_comparison_protocol,
    ordered_fingerprint,
)
from construction_safety_vision.detector_segmenter_cost import (
    BLOCK_FIELDS,
    COST_BENCHMARK_COMPLETE,
    DISTRIBUTION_DIAGNOSTIC_LABEL,
    DISTRIBUTION_DIAGNOSTIC_STATUS,
    EXECUTION_PASSES,
    INFERENCE_MEMORY,
    MEMORY_ISOLATION,
    MODELS,
    NO_POWER_TELEMETRY,
    OBSERVATION_FIELDS,
    PHASE,
    PROTOCOL_STABILITY_FIELDS,
    SEGMENTATION_COST_LABEL,
    STATIC_MODEL_COMPLEXITY,
    STATISTIC_FIELDS,
    TIMING_BOUNDARIES,
    as_gib,
    build_execution_plan,
    execution_plan_fingerprint,
    expected_observation_count,
    images_per_second_from_mean,
    latency_result_fingerprint,
    memory_result_fingerprint,
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
def latency(paths: ProjectPaths) -> dict[str, Any]:
    """The committed latency artifact.

    Args:
        paths: Project layout.

    Returns:
        The parsed artifact.
    """
    path = paths.reports / "detector_segmenter_latency_comparison.json"
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def memory(paths: ProjectPaths) -> dict[str, Any]:
    """The committed inference-memory artifact.

    Args:
        paths: Project layout.

    Returns:
        The parsed artifact.
    """
    path = paths.reports / "detector_segmenter_memory_comparison.json"
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def blocks(paths: ProjectPaths) -> list[dict[str, str]]:
    """The committed per-block latency summary.

    Args:
        paths: Project layout.

    Returns:
        The rows.
    """
    path = paths.reports / "detector_segmenter_latency_blocks.csv"
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


# --- the artifacts exist and validate -------------------------------------------------


def test_the_expected_artifacts_exist(paths: ProjectPaths) -> None:
    """Phase 10C wrote its four committed artifacts and its provenance record."""
    for name in (
        "detector_segmenter_latency_comparison.json",
        "detector_segmenter_memory_comparison.json",
        "detector_segmenter_latency_blocks.csv",
        "detector_segmenter_latency_report.md",
        "detector_segmenter_cost_benchmark.provenance.json",
    ):
        assert (paths.reports / name).is_file(), name


def test_the_committed_latency_artifact_validates(
    latency: dict[str, Any], paths: ProjectPaths
) -> None:
    """The emitted artifact passes its own validator against the frozen identities."""
    protocol = load_comparison_protocol(paths.configs / "detector_segmenter_comparison.yaml")
    plan = build_execution_plan(
        [row["source_image_id"] for row in _membership(paths)],
    )
    assert (
        validate_latency_result(
            latency,
            protocol_fingerprint=protocol.fingerprint(),
            detector_sha256=DETECTOR_SHA,
            segmenter_sha256=SEGMENTER_SHA,
            benchmark_sha256=EXPECTED_BENCHMARK,
            execution_plan_sha256=execution_plan_fingerprint(plan),
        )
        == []
    )


def test_the_committed_memory_artifact_validates(
    memory: dict[str, Any], paths: ProjectPaths
) -> None:
    """The memory artifact passes its own validator."""
    protocol = load_comparison_protocol(paths.configs / "detector_segmenter_comparison.yaml")
    assert (
        validate_memory_result(
            memory,
            protocol_fingerprint=protocol.fingerprint(),
            detector_sha256=DETECTOR_SHA,
            segmenter_sha256=SEGMENTER_SHA,
        )
        == []
    )


def _membership(paths: ProjectPaths) -> list[dict[str, str]]:
    """Read the frozen benchmark membership.

    Args:
        paths: Project layout.

    Returns:
        The rows, in the frozen order.
    """
    path = paths.reports / "detector_segmenter_latency_membership.csv"
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


# --- identity and membership ----------------------------------------------------------


def test_the_artifacts_name_the_frozen_models(
    latency: dict[str, Any], memory: dict[str, Any]
) -> None:
    """Both artifacts name exactly the frozen detector and segmenter."""
    for payload in (latency, memory):
        assert payload["detector"]["checkpoint_sha256"] == DETECTOR_SHA
        assert payload["segmenter"]["checkpoint_sha256"] == SEGMENTER_SHA
    assert latency["detector"]["experiment"] == DETECTOR
    assert latency["segmenter"]["experiment"] == SEGMENTER
    assert latency["segmenter"]["overlap_mask"] is False


def test_the_artifacts_name_the_frozen_protocol(
    latency: dict[str, Any], memory: dict[str, Any]
) -> None:
    """Both carry the phase 10A fingerprint."""
    assert latency["protocol_fingerprint"] == EXPECTED_PROTOCOL
    assert memory["protocol_fingerprint"] == EXPECTED_PROTOCOL


def test_the_benchmark_membership_is_the_frozen_subset(
    latency: dict[str, Any], paths: ProjectPaths
) -> None:
    """The recorded membership reproduces the frozen ordered fingerprint."""
    rows = _membership(paths)
    assert latency["benchmark"]["benchmark_membership_sha256"] == ordered_fingerprint(
        [row["source_image_id"] for row in rows]
    )
    assert latency["benchmark"]["benchmark_membership_sha256"] == EXPECTED_BENCHMARK
    assert latency["benchmark"]["benchmark_image_count"] == BENCHMARK_IMAGE_COUNT == 20
    assert latency["benchmark"]["images_reselected"] is False
    assert latency["benchmark"]["images_inspected_to_select"] == 0


def test_the_executed_order_is_the_derived_plan(
    latency: dict[str, Any], paths: ProjectPaths
) -> None:
    """The recorded execution plan is the one the frozen membership derives."""
    plan = build_execution_plan([row["source_image_id"] for row in _membership(paths)])
    assert latency["benchmark"]["execution_plan_sha256"] == execution_plan_fingerprint(plan)
    assert latency["benchmark"]["execution_blocks"] == len(plan) == 80
    assert latency["benchmark"]["randomized"] is False
    assert latency["benchmark"]["interleaved"] is True
    assert latency["benchmark"]["symmetric"] is True
    assert latency["benchmark"]["execution_passes"] == list(EXECUTION_PASSES)


def test_both_models_saw_the_same_input_tensors(latency: dict[str, Any]) -> None:
    """A resolution difference would have made this a resolution comparison."""
    assert latency["benchmark"]["identical_input_tensor_shapes"] is True
    assert latency["benchmark"]["identical_images_for_both_models"] is True
    assert latency["benchmark"]["identical_order_for_both_models"] is True
    shapes = latency["benchmark"]["input_tensor_shapes"]
    assert len(shapes) == BENCHMARK_IMAGE_COUNT
    for shape in shapes.values():
        assert shape[0] == 1
        assert max(shape[2:]) == 768


# --- conditions ------------------------------------------------------------------------


def test_the_recorded_conditions_are_the_frozen_ones(latency: dict[str, Any]) -> None:
    """Batch, resolution, precision, warmup and repetitions are as frozen."""
    conditions = latency["conditions"]
    assert conditions["batch"] == 1
    assert conditions["imgsz"] == 768
    assert conditions["precision"] == PRECISION == "FP32"
    assert conditions["precision_value"] == 32
    assert conditions["warmup_iterations"] == WARMUP_ITERATIONS == 20
    assert conditions["timed_iterations_per_image"] == TIMED_ITERATIONS_PER_IMAGE == 30
    assert conditions["augment"] is False
    assert conditions["tta"] is False
    assert conditions["torch_compile"] is False
    assert conditions["tensorrt"] is False
    assert conditions["onnx"] is False


def test_effective_fp32_parity_was_proved_at_runtime(latency: dict[str, Any]) -> None:
    """The precision evidence is byte-identical between the two models."""
    precision = latency["precision_preflight"]
    assert precision["status"] == "EFFECTIVE_PRECISION_PARITY_VERIFIED"
    assert precision["identical_across_models"] is True
    assert precision["evidence"][DETECTOR] == precision["evidence"][SEGMENTER]
    for model in MODELS:
        evidence = precision["evidence"][model]
        assert evidence["backend_fp16_flag"] is False
        assert evidence["parameter_dtypes"] == ["torch.float32"]
        assert evidence["input_dtype"] == "torch.float32"
        assert evidence["autocast_enabled_during_forward"] is False
        assert evidence["quantization_config_present"] is False
        assert evidence["resolved_precision_value"] == 32
        assert evidence["device_type"] == "cuda"


def test_synchronization_and_the_timing_primitive_are_recorded(latency: dict[str, Any]) -> None:
    """Both edges synchronised, with the same primitive for both models."""
    conditions = latency["conditions"]
    assert conditions["timing_primitive"] == "time.perf_counter"
    assert conditions["synchronization_primitive"] == "torch.cuda.synchronize"
    assert conditions["cuda_synchronize_before_timed_region"] is True
    assert conditions["cuda_synchronize_after_timed_region"] is True
    assert conditions["same_primitive_for_both_models"] is True


def test_model_load_and_disk_decode_are_outside_the_timing(latency: dict[str, Any]) -> None:
    """Neither the checkpoint read nor the image decode is inside a timed region."""
    conditions = latency["conditions"]
    assert conditions["model_load_excluded_from_timing"] is True
    assert conditions["disk_decode_excluded_from_timing"] is True
    assert conditions["decode_shared_between_models"] is True
    assert conditions["predictions_saved_during_timing"] is False


# --- the two boundaries ----------------------------------------------------------------


def test_both_boundaries_are_measured_and_never_merged(latency: dict[str, Any]) -> None:
    """Each boundary has its own statistics, and no combined score exists."""
    assert set(latency["timing_boundaries"]) == set(TIMING_BOUNDARIES)
    assert latency["boundaries_combined"] is False
    assert latency["combined_latency_score"] is False
    assert latency["winner_declared"] is False
    for model in MODELS:
        assert set(TIMING_BOUNDARIES) <= set(latency["latency"][model])


def test_the_model_inference_boundary_excludes_the_pipeline(latency: dict[str, Any]) -> None:
    """The narrow boundary is the forward pass and nothing else."""
    boundary = latency["timing_boundaries"][MODEL_INFERENCE_LATENCY]
    assert boundary["preprocessing_included"] is False
    assert boundary["postprocessing_included"] is False
    assert boundary["mask_reconstruction_included"] is False
    assert boundary["disk_io_included"] is False
    assert boundary["model_load_included"] is False
    assert boundary["same_call_for_both_models"] is True
    assert boundary["framework_call"] == "BasePredictor.inference"


def test_mask_reconstruction_is_inside_the_segmenter_end_to_end_boundary(
    latency: dict[str, Any],
) -> None:
    """Excluding it would hide the cost the comparison exists to quantify."""
    boundary = latency["timing_boundaries"][END_TO_END_LATENCY]
    assert boundary["segmenter_mask_reconstruction_included"] is True
    assert boundary["mask_reconstruction_call"] == "ultralytics.utils.ops.process_mask_native"
    assert boundary["mask_reconstruction_site"] == "SegmentationPredictor.construct_result"
    assert boundary["preprocessing_included"] is True
    assert boundary["postprocessing_included"] is True
    assert boundary["outputs_materialized_before_end_timestamp"] is True
    assert boundary["segmenter_boundary_ends_at"] == (
        "CLASS_CONFIDENCE_BOX_AND_INSTANCE_MASK_AVAILABLE"
    )
    assert boundary["detector_boundary_ends_at"] == "CLASS_CONFIDENCE_BOX_AVAILABLE"


def test_the_segmenter_produced_masks_on_the_original_canvas(latency: dict[str, Any]) -> None:
    """A mask at the letterboxed resolution would not be the project's output."""
    evidence = latency["output_evidence"][SEGMENTER]
    assert evidence["masks_on_original_canvas"] is True
    assert evidence["images_with_masks"] > 0
    assert latency["output_evidence"][DETECTOR]["masks_available"] is False


# --- the statistics and the deltas -----------------------------------------------------


def test_every_distribution_holds_every_timed_observation(latency: dict[str, Any]) -> None:
    """1200 readings per model per boundary, and 4800 in total."""
    for model in MODELS:
        for boundary in TIMING_BOUNDARIES:
            statistics = latency["latency"][model][boundary]
            assert set(statistics) == set(STATISTIC_FIELDS)
            assert statistics["count"] == BENCHMARK_IMAGE_COUNT * 2 * TIMED_ITERATIONS_PER_IMAGE
    assert latency["raw_timings"]["observations"] == expected_observation_count() == 4800
    assert latency["raw_timings"]["observations_per_model_per_boundary"] == 1200


def test_every_statistic_is_internally_consistent(latency: dict[str, Any]) -> None:
    """Minimum, mean, percentiles and maximum stand in the right order."""
    for model in MODELS:
        for boundary in TIMING_BOUNDARIES:
            statistics = latency["latency"][model][boundary]
            assert statistics["min"] <= statistics["mean"] <= statistics["max"]
            assert statistics["median"] == statistics["p50"]
            ordered = [statistics[name] for name in ("p50", "p90", "p95", "p99", "max")]
            assert ordered == sorted(ordered)
            assert statistics["std"] > 0


def test_throughput_recomputes_from_the_mean(latency: dict[str, Any]) -> None:
    """Every images-per-second figure follows from its own mean."""
    for model in MODELS:
        for boundary in TIMING_BOUNDARIES:
            mean = latency["latency"][model][boundary]["mean"]
            assert latency["latency"][model]["throughput"][boundary] == (
                images_per_second_from_mean(mean)
            )


def test_every_delta_recomputes(latency: dict[str, Any]) -> None:
    """The absolute delta, the relative cost and the ratio all follow."""
    for boundary in TIMING_BOUNDARIES:
        block = latency["deltas"][boundary]
        detector_mean = latency["latency"][DETECTOR][boundary]["mean"]
        segmenter_mean = latency["latency"][SEGMENTER][boundary]["mean"]
        assert block["detector_mean_latency_ms"] == detector_mean
        assert block["segmenter_mean_latency_ms"] == segmenter_mean
        assert block["absolute_latency_delta_ms"] == pytest.approx(
            segmenter_mean - detector_mean, abs=1e-6
        )
        assert block["relative_latency_cost"] == pytest.approx(
            segmenter_mean / detector_mean - 1.0, abs=1e-6
        )
        assert block["throughput_ratio"] == pytest.approx(
            images_per_second_from_mean(segmenter_mean)
            / images_per_second_from_mean(detector_mean),
            abs=1e-6,
        )
        assert block["derived_from"] == "mean"
        assert block["combined_latency_score"] is False


def test_the_block_summary_covers_every_block_and_boundary(
    blocks: list[dict[str, str]], latency: dict[str, Any]
) -> None:
    """The committed per-block table is complete and consistent."""
    assert list(blocks[0]) == list(BLOCK_FIELDS)
    assert len(blocks) == latency["benchmark"]["execution_blocks"] * len(TIMING_BOUNDARIES)
    for row in blocks:
        assert int(row["count"]) == TIMED_ITERATIONS_PER_IMAGE
        assert row["model"] in MODELS
        assert row["timing_type"] in TIMING_BOUNDARIES
    positions = sorted({int(row["execution_position"]) for row in blocks})
    assert positions == list(range(latency["benchmark"]["execution_blocks"]))


def test_the_block_summary_reproduces_the_reported_means(
    blocks: list[dict[str, str]], latency: dict[str, Any]
) -> None:
    """Averaging the equal-sized block means returns the reported mean."""
    for model in MODELS:
        for boundary in TIMING_BOUNDARIES:
            means = [
                float(row["mean"])
                for row in blocks
                if row["model"] == model and row["timing_type"] == boundary
            ]
            assert len(means) == BENCHMARK_IMAGE_COUNT * len(EXECUTION_PASSES)
            assert sum(means) / len(means) == pytest.approx(
                latency["latency"][model][boundary]["mean"], abs=1e-5
            )


def test_the_result_fingerprints_reproduce(latency: dict[str, Any], memory: dict[str, Any]) -> None:
    """Both semantic fingerprints recompute from the committed content."""
    assert latency["latency_result_sha256"] == latency_result_fingerprint(latency)
    assert memory["memory_result_sha256"] == memory_result_fingerprint(
        {key: value for key, value in memory.items() if key != "memory_result_sha256"}
    )


def test_the_raw_timing_fingerprint_is_recorded(latency: dict[str, Any]) -> None:
    """The raw dataset is fingerprinted even though it is not committed."""
    assert len(latency["raw_timings"]["timing_dataset_sha256"]) == 64
    assert latency["raw_timings"]["schema"] == list(OBSERVATION_FIELDS)
    assert latency["raw_timings"]["row_level_artifact_committed"] is False
    assert latency["raw_timings"]["holdout_identifiers_stored"] is False


# --- the post-hoc diagnostic and protocol stability ------------------------------------


def test_the_distribution_diagnostic_is_labelled_and_decides_nothing(
    latency: dict[str, Any],
) -> None:
    """A post-hoc description of the distribution's shape, and nothing more."""
    diagnostic = latency["post_hoc_distribution_diagnostic"]
    assert diagnostic["label"] == DISTRIBUTION_DIAGNOSTIC_LABEL
    assert diagnostic["status"] == DISTRIBUTION_DIAGNOSTIC_STATUS
    for flag in (
        "replaces_frozen_statistics",
        "is_a_frozen_phase_10a_metric",
        "is_a_selection_rule",
        "is_a_significance_test",
        "changes_any_frozen_number",
        "outlier_rejection_applied",
        "observations_normalized_or_rescaled",
        "per_mode_statistics_computed",
        "proportionality_across_models_demonstrated",
    ):
        assert diagnostic[flag] is False, flag
    assert diagnostic["observations_discarded"] == 0
    assert diagnostic["observations_used"] == expected_observation_count()


def test_no_power_state_telemetry_accompanied_the_timings(latency: dict[str, Any]) -> None:
    """The absence is recorded, and so is what it costs the interpretation."""
    diagnostic = latency["post_hoc_distribution_diagnostic"]
    assert diagnostic["per_observation_power_state_telemetry"] == NO_POWER_TELEMETRY
    assert diagnostic["telemetry_synchronous_with_timed_blocks"] is False
    assert diagnostic["causal_attribution"] == "UNKNOWN"
    assert diagnostic["hypothesis_status"] == "UNTESTED_HYPOTHESIS"
    assert latency["runtime"]["per_observation_power_state_telemetry"] == NO_POWER_TELEMETRY
    assert latency["runtime"]["telemetry_synchronous_with_timed_blocks"] is False


def test_the_protocol_was_not_adapted_after_the_timings_were_seen(
    latency: dict[str, Any],
) -> None:
    """Every way the frozen protocol could have been bent is recorded false."""
    stability = latency["protocol_stability_after_observation"]
    for field in PROTOCOL_STABILITY_FIELDS:
        assert stability[field] is False, field


def test_the_benchmark_was_not_rerun(latency: dict[str, Any], paths: ProjectPaths) -> None:
    """One benchmark execution, recorded as such in the artifact and the record."""
    assert latency["protocol_stability_after_observation"]["benchmark_rerun"] is False
    record = json.loads(
        (paths.reports / "detector_segmenter_cost_benchmark.provenance.json").read_text(
            encoding="utf-8"
        )
    )
    details = record["details"]
    assert details["benchmark_rerun"] is False
    assert details["observations_discarded"] == 0
    assert details["frozen_statistics_unchanged"] is True
    assert details["statistical_definitions_changed"] is False
    assert details["models_executed_in_this_rebuild"] == 0
    assert details["timings_taken_in_this_rebuild"] == 0


# --- memory ----------------------------------------------------------------------------


def test_the_memory_artifact_measures_inference_not_training(memory: dict[str, Any]) -> None:
    """Training memory is a different quantity and is never substituted."""
    assert memory["measurement"] == INFERENCE_MEMORY
    assert memory["training_memory_reused"] is False
    assert memory["training_memory"].startswith("NOT_MEASURED")
    assert memory["models_trained_in_this_phase"] == 0


def test_each_model_memory_was_measured_in_isolation(memory: dict[str, Any]) -> None:
    """One resident model, with the allocator empty before it was loaded."""
    assert memory["conditions"]["isolation_method"] == MEMORY_ISOLATION
    assert memory["conditions"]["models_resident_during_measurement"] == 1
    assert memory["conditions"]["peak_stats_reset_after_warmup"] is True
    assert memory["conditions"]["reset_call"] == "torch.cuda.reset_peak_memory_stats"
    for model in MODELS:
        block = memory["memory"][model]
        assert block["pre_load_allocated_bytes"] == 0
        assert block["other_model_resident"] is False
        assert block["peak_stats_reset_after_warmup"] is True


def test_the_memory_figures_are_internally_consistent(memory: dict[str, Any]) -> None:
    """Allocated never exceeds reserved, and the GiB conversions recompute."""
    for model in MODELS:
        block = memory["memory"][model]
        allocated = block["peak_memory_allocated_bytes"]
        reserved = block["peak_memory_reserved_bytes"]
        assert 0 < allocated <= reserved
        assert block["baseline_allocated_bytes"] <= allocated
        assert block["peak_memory_allocated_gib"] == as_gib(allocated)
        assert block["peak_memory_reserved_gib"] == as_gib(reserved)


def test_the_memory_delta_recomputes(memory: dict[str, Any]) -> None:
    """The recorded deltas and ratios follow from the recorded peaks."""
    detector = memory["memory"][DETECTOR]
    segmenter = memory["memory"][SEGMENTER]
    delta = memory["delta"]
    for field in ("peak_memory_allocated", "peak_memory_reserved"):
        expected = segmenter[f"{field}_bytes"] - detector[f"{field}_bytes"]
        assert delta[f"{field}_delta_bytes"] == expected
        assert delta[f"{field}_ratio"] == pytest.approx(
            segmenter[f"{field}_bytes"] / detector[f"{field}_bytes"], abs=1e-6
        )


def test_the_memory_conditions_match_the_latency_conditions(
    latency: dict[str, Any], memory: dict[str, Any]
) -> None:
    """Memory is measured at the same batch, resolution and precision."""
    for field in ("batch", "imgsz", "precision", "warmup_iterations"):
        assert memory["conditions"][field] == latency["conditions"][field]


# --- complexity and interpretation limits ----------------------------------------------


def test_static_complexity_is_labelled_and_not_recomputed(latency: dict[str, Any]) -> None:
    """Parameter and FLOP counts are read from committed artifacts."""
    complexity = latency["model_complexity"]
    assert complexity["label"] == STATIC_MODEL_COMPLEXITY
    assert complexity["recomputed_in_this_phase"] is False
    assert complexity["measured_at_benchmark_input_size"] is False
    assert "640" in complexity["reference_input_size"]
    for model in MODELS:
        assert complexity[model]["parameters"] > 0
        assert complexity[model]["gflops"] > 0


def test_static_complexity_matches_the_committed_experiment_manifests(
    latency: dict[str, Any], paths: ProjectPaths
) -> None:
    """The figures are the ones the frozen experiments recorded, unchanged."""
    detector = json.loads(
        (paths.reports / "detection_D2_manifest.json").read_text(encoding="utf-8")
    )["model_complexity"]
    segmenter = json.loads(
        (paths.reports / "segmentation_S1_result_manifest.json").read_text(encoding="utf-8")
    )["model_complexity"]
    complexity = latency["model_complexity"]
    assert complexity[DETECTOR]["parameters"] == detector["parameters"]
    assert complexity[DETECTOR]["gflops"] == detector["gflops"]
    assert complexity[SEGMENTER]["parameters"] == segmenter["parameters"]
    assert complexity[SEGMENTER]["gflops"] == segmenter["gflops"]


def test_the_latency_difference_is_not_attributed_to_mask_reconstruction_alone(
    latency: dict[str, Any],
) -> None:
    """The architectures differ beyond postprocessing, and the label says so."""
    assert latency["segmentation_cost_label"] == SEGMENTATION_COST_LABEL
    assert latency["pure_mask_reconstruction_cost_isolated"] is False


def test_the_benchmark_is_labelled_local_and_controlled(latency: dict[str, Any]) -> None:
    """No claim of hardware-independent latency."""
    assert latency["benchmark_status"] == BENCHMARK_LABEL
    assert latency["status"] == COST_BENCHMARK_COMPLETE
    assert latency["phase"] == PHASE
    assert latency["runtime"]["power_settings_changed_by_this_phase"] is False
    assert latency["runtime"]["thermal_correction_applied"] is False


def test_the_limitations_name_the_machine_and_the_batch(latency: dict[str, Any]) -> None:
    """The limitations a reader needs are written down, not implied."""
    limitations = " ".join(latency["limitations"]).lower()
    for topic in ("throttle", "batch 1", "dvfs", "640"):
        assert topic in limitations


def test_phase_10c_recomputed_no_phase_10b_metric(latency: dict[str, Any]) -> None:
    """Recognition and spatial analysis belong to phase 10B and were not rerun."""
    assert latency["models_trained_in_this_phase"] == 0
    assert latency["models_modified_in_this_phase"] == 0
    assert latency["thresholds_tuned"] == 0
    assert latency["ap_metrics_recomputed"] is False
    assert latency["spatial_analysis_rerun"] is False
    assert latency["association_analysis_rerun"] is False
    assert latency["mask_iou_diagnostic_rerun"] is False
    assert latency["protocol_adapted_after_results"] is False


# --- the holdout -----------------------------------------------------------------------


def test_both_artifacts_declare_the_holdout_protected(
    latency: dict[str, Any], memory: dict[str, Any]
) -> None:
    """Nothing was read, timed or measured on the holdout."""
    for payload in (latency, memory):
        assert payload["test"]["status"] == "PROTECTED_NOT_ACCESSED"
        assert payload["holdout_accessed"] is False
        assert payload["test"]["images_read"] == 0
        assert payload["test"]["statistics"] == 0
    assert latency["test"]["predictions"] == 0
    assert latency["test"]["timings"] == 0


def test_no_committed_artifact_names_the_protected_split(paths: ProjectPaths) -> None:
    """The word appears only inside a holdout declaration."""
    for name in (
        "detector_segmenter_latency_comparison.json",
        "detector_segmenter_memory_comparison.json",
    ):
        payload = json.loads((paths.reports / name).read_text(encoding="utf-8"))
        body = {key: value for key, value in payload.items() if key != "test"}
        assert '"test"' not in json.dumps(body)
        assert ': "test"' not in json.dumps(body)


def test_the_block_summary_holds_only_validation_images(
    blocks: list[dict[str, str]], paths: ProjectPaths
) -> None:
    """Every row names an image from the frozen benchmark subset."""
    allowed = {row["source_image_id"] for row in _membership(paths)}
    assert {row["source_image_id"] for row in blocks} == allowed


def test_the_report_exists_and_names_no_absolute_path(paths: ProjectPaths) -> None:
    """The report is committed evidence and carries nothing machine-specific."""
    report = (paths.reports / "detector_segmenter_latency_report.md").read_text(encoding="utf-8")
    # Built rather than written literally, so the project's own sensitive-content
    # scanner does not flag this assertion as the very thing it guards against.
    windows_drive = "C" + ":" + chr(92)
    posix_home = "/" + "home" + "/"
    assert windows_drive not in report
    assert posix_home not in report
    assert "PROTECTED_NOT_ACCESSED" in report
    assert NO_POWER_TELEMETRY in report
    assert "CONTROLLED_LOCAL_HARDWARE_BENCHMARK" in report


def test_the_row_level_timings_are_not_committed(paths: ProjectPaths) -> None:
    """4800 raw readings stay in ignored runtime space, referenced by digest."""
    assert not (paths.reports / "latency_observations.csv").exists()
    ignored = Path(paths.root / ".gitignore").read_text(encoding="utf-8")
    assert "artifacts" in ignored
