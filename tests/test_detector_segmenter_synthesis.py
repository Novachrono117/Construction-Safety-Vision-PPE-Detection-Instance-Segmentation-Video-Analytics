"""Tests for the phase 10D synthesis logic.

These exercise the builders and the validator against the committed phase 10B
and 10C evidence. No model is loaded, no inference is run and no holdout data is
read - a test below asserts that the runner script cannot do any of those.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest

from construction_safety_vision.detector_segmenter_analysis import (
    NO_BOX_PROXY,
    RARE_CLASS,
    RARE_CLASS_STATUS,
)
from construction_safety_vision.detector_segmenter_comparison import (
    ASSOCIATION_CATEGORIES,
    DETECTOR_EXPERIMENT,
    END_TO_END_LATENCY,
    MODEL_INFERENCE_LATENCY,
    SEGMENTER_EXPERIMENT,
)
from construction_safety_vision.detector_segmenter_synthesis import (
    AXES,
    CLAIM_FIELDS,
    CLAIM_IDS,
    HOLDOUT_STATUS,
    LATENCY_SCOPE,
    MEAN_DERIVED_THROUGHPUT,
    OPERATIONAL_CONF,
    PHASE,
    PROTOCOL_GAP_RESOLUTION,
    SOURCES,
    SYNTHESIS_COMPLETE,
    TRADEOFF_FIELDS,
    SynthesisError,
    build_synthesis,
    load_sources,
    render_markdown,
    tradeoff_rows,
    validate_synthesis,
)
from construction_safety_vision.paths import ProjectPaths

ROOT = ProjectPaths.from_root().root

SCRIPT = ROOT / "scripts" / "synthesize_detector_segmenter.py"
MODULE = ROOT / "src" / "construction_safety_vision" / "detector_segmenter_synthesis.py"


@pytest.fixture(scope="module")
def sources() -> dict[str, dict[str, Any]]:
    return load_sources(ROOT)


@pytest.fixture(scope="module")
def synthesis(sources: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return build_synthesis(ROOT, sources)


# --- sources -----------------------------------------------------------------------------


def test_every_declared_source_is_committed_and_read() -> None:
    for relative in SOURCES.values():
        assert (ROOT / relative).is_file(), relative


def test_synthesis_reads_the_committed_10b_and_10c_artifacts(
    synthesis: dict[str, Any], sources: dict[str, dict[str, Any]]
) -> None:
    recorded = synthesis["source_artifacts"]
    assert set(recorded) == set(SOURCES)
    assert (
        recorded["box_comparison"]["recorded_fingerprint"]
        == sources["box_comparison"]["box_comparison_sha256"]
    )
    assert (
        recorded["spatial_comparison"]["recorded_fingerprint"]
        == sources["spatial_comparison"]["spatial_comparison_sha256"]
    )
    assert (
        recorded["latency_benchmark"]["recorded_fingerprint"]
        == sources["latency_benchmark"]["latency_result_sha256"]
    )
    assert (
        recorded["memory_benchmark"]["recorded_fingerprint"]
        == sources["memory_benchmark"]["memory_result_sha256"]
    )


def test_missing_source_is_a_synthesis_error(tmp_path: Path) -> None:
    with pytest.raises(SynthesisError):
        load_sources(tmp_path)


# --- model identity ----------------------------------------------------------------------


def test_exact_detector_and_segmenter_identities(synthesis: dict[str, Any]) -> None:
    detector = synthesis["models"][DETECTOR_EXPERIMENT]
    segmenter = synthesis["models"][SEGMENTER_EXPERIMENT]
    assert detector["experiment"] == "D2"
    assert detector["model"] == "YOLO11n"
    assert detector["imgsz"] == 768
    assert (
        detector["checkpoint_sha256"]
        == "0466f872a9de22898c70d8834cf1fcfb3d77f6c9fb27e81ee6248d3d19cbc206"
    )
    assert segmenter["experiment"] == "S1"
    assert segmenter["model"] == "YOLO11n-seg"
    assert segmenter["imgsz"] == 768
    assert segmenter["mask_ratio"] == 4
    assert segmenter["overlap_mask"] is False
    assert (
        segmenter["checkpoint_sha256"]
        == "29337d671459d0f742fb713613cc41d0eafcb8a21f5846ac0da65b2ffc024f20"
    )


def test_no_architectural_identity_or_drop_in_replacement_claim(
    synthesis: dict[str, Any],
) -> None:
    models = synthesis["models"]
    assert models["architectural_identity_claimed"] is False
    assert models["drop_in_replacement"] is False
    assert models["same_output_task"] is False


# --- recognition -------------------------------------------------------------------------


def test_all_class_localization_delta_preserved(synthesis: dict[str, Any]) -> None:
    all_class = synthesis["recognition"]["all_class"]["CANONICAL_BOX_MAP50_95"]
    assert all_class["D2"] == 0.48539
    assert all_class["S1"] == 0.505682
    assert all_class["delta"] == 0.020292


def test_supported_sensitivity_preserved_and_marked_descriptive(
    synthesis: dict[str, Any],
) -> None:
    sensitivity = synthesis["recognition"]["supported_class_sensitivity"]
    assert sensitivity["D2_supported_macro"] == 0.589729
    assert sensitivity["S1_supported_macro"] == 0.581689
    assert sensitivity["delta"] == -0.00804
    assert sensitivity["label"] == "POST_HOC_DESCRIPTIVE_SUPPORT_SENSITIVITY"
    assert sensitivity["is_a_frozen_phase_10a_metric"] is False
    assert sensitivity["is_a_selection_rule"] is False
    assert sensitivity["is_a_significance_test"] is False


def test_vest_loose_uncertainty_preserved(synthesis: dict[str, Any]) -> None:
    rare = synthesis["recognition"]["rare_class"]
    assert rare["name"] == RARE_CLASS == "vest_loose"
    assert rare["status"] == RARE_CLASS_STATUS == "DESCRIPTIVE_HIGH_UNCERTAINTY"
    assert rare["contribution"] == 0.026724
    assert rare["contribution_exceeds_total"] is True
    per_class = synthesis["recognition"]["per_class"][RARE_CLASS]
    assert per_class["status"] == RARE_CLASS_STATUS


def test_per_class_trade_off_keeps_both_directions(synthesis: dict[str, Any]) -> None:
    recognition = synthesis["recognition"]
    assert recognition["improved_classes"]
    assert recognition["declined_classes"]
    assert "helmet_loose" in recognition["declined_classes"]
    assert "vest_on_body" in recognition["declined_classes"]
    assert recognition["delta_excluding_rare_class"] == -0.00804


def test_no_superiority_or_significance_claim(synthesis: dict[str, Any]) -> None:
    recognition = synthesis["recognition"]
    assert recognition["superiority_claimed_for_either_model"] is False
    assert recognition["significance_test_performed"] is False
    assert recognition["why_any_class_moved"] == "UNKNOWN"


# --- spatial -----------------------------------------------------------------------------


def test_mask_fill_ratio_read_exactly(synthesis: dict[str, Any]) -> None:
    fill = synthesis["spatial_representation"]["MASK_TO_BOX_FILL_RATIO"]
    assert fill["median"] == 0.664433
    assert fill["mean"] == 0.649956
    assert fill["count"] == 336


def test_no_box_only_features_preserved(synthesis: dict[str, Any]) -> None:
    spatial = synthesis["spatial_representation"]
    assert spatial["mask_only_quantities"] == ["MASK_TO_BOX_FILL_RATIO", "SHAPE_EXTENT"]
    assert spatial["mask_only_label"] == NO_BOX_PROXY == "NO_BOX_ONLY_EQUIVALENT"
    assert spatial["SHAPE_EXTENT"]["median"] == 0.672173


def test_fill_ratio_is_not_a_background_error_rate(synthesis: dict[str, Any]) -> None:
    spatial = synthesis["spatial_representation"]
    assert spatial["background_error_rate_claimed"] is False
    assert "background" in spatial["fill_ratio_reading"]


def test_box_proxy_inflation_is_refinement_not_error(synthesis: dict[str, Any]) -> None:
    refinement = synthesis["spatial_representation"]["proxy_refinement"]
    assert refinement["label"] == "PROXY_REFINEMENT"
    assert refinement["box_error_claimed"] is False
    assert refinement["accuracy_claim"] is False
    area = refinement["features"]["INSTANCE_AREA_PIXELS"]
    assert area["mask_measurement"]["mean"] == 144562.997024
    assert area["box_proxy_statistics"]["mean"] == 237205.836504
    intersection = refinement["features"]["PERSON_PPE_MASK_INTERSECTION"]
    assert intersection["mask_measurement"]["mean"] == 26828.34384
    assert intersection["box_proxy_statistics"]["mean"] == 47280.877637


def test_proxy_refinement_compares_like_with_like(synthesis: dict[str, Any]) -> None:
    """Every tabulated pair holds the same statistic on both sides."""
    for entry in synthesis["spatial_representation"]["proxy_refinement"]["features"].values():
        assert set(entry["mask_measurement"]) == set(entry["box_proxy_statistics"])


def test_centroid_displacement_read_exactly_and_not_called_true_centre(
    synthesis: dict[str, Any],
) -> None:
    centroid = synthesis["spatial_representation"]["centroid"]
    assert centroid["displacement_pixels"]["median"] == 16.048565
    assert centroid["displacement_pixels"]["p95"] == 126.933417
    assert centroid["displacement_pixels"]["max"] == 262.773275
    assert centroid["true_center_claimed"] is False


# --- association -------------------------------------------------------------------------


def test_geometry_isolating_association_counts(synthesis: dict[str, Any]) -> None:
    geometry = synthesis["association"]["geometry_isolating"]
    assert geometry["total_relationships"] == 173
    assert geometry["classified_relationships"] == 172
    assert geometry["taxonomy_exceptions"] == 1
    assert geometry["frozen_category_counts"] == {
        "BOX_AND_MASK_AGREE": 103,
        "BOX_ONLY_ASSOCIATION": 3,
        "MASK_ONLY_ASSOCIATION": 0,
        "NEITHER_ASSOCIATION": 66,
    }


def test_zero_geometry_isolating_mask_only_association_preserved(
    synthesis: dict[str, Any],
) -> None:
    assert synthesis["association"]["mask_only_associations"] == 0


def test_association_taxonomy_exception_preserved(synthesis: dict[str, Any]) -> None:
    limitation = synthesis["association"]["taxonomy_limitation"]
    assert limitation["status"] == "FROZEN_TAXONOMY_NON_EXHAUSTIVE_FOR_OBSERVED_DATA"
    assert limitation["exception_type"] == "BOTH_RULES_ASSOCIATE_DIFFERENT_PERSON"
    assert limitation["exception_status"] == "UNCLASSIFIED_BY_FROZEN_FOUR_CATEGORY_TAXONOMY"
    assert limitation["fifth_peer_category_added"] is False
    assert limitation["frozen_categories_unchanged"] is True
    assert limitation["phase_10a_protocol_modified"] is False
    assert synthesis["association"]["frozen_categories"] == list(ASSOCIATION_CATEGORIES)


def test_pipeline_level_is_not_attributed_to_geometry_alone(synthesis: dict[str, Any]) -> None:
    pipeline = synthesis["association"]["pipeline_level"]
    assert pipeline["attributable_to_geometry_alone"] is False
    assert pipeline["isolates_geometry"] is False
    assert pipeline["taxonomy_exceptions"] == 17


def test_no_compliance_accuracy_claim(synthesis: dict[str, Any]) -> None:
    association = synthesis["association"]
    assert association["compliance_accuracy_claimed"] is False
    coverage = association["visible_ppe_coverage_proxy"]
    assert coverage["label"] == "INTERPRETIVE_OPERATIONAL_PROXY"
    assert coverage["is_not_a_compliance_measure"] is True
    assert coverage["primary_scientific_conclusion"] is False


def test_no_compliance_accuracy_wording_anywhere(synthesis: dict[str, Any]) -> None:
    text = render_markdown(synthesis).lower()
    for phrase in (
        "compliance accuracy of",
        "safety violation accuracy",
        "correct wearing accuracy",
    ):
        assert phrase not in text


# --- cost --------------------------------------------------------------------------------


def test_exact_model_inference_latency_values(synthesis: dict[str, Any]) -> None:
    boundary = synthesis["cost"]["latency"]["boundaries"][MODEL_INFERENCE_LATENCY]
    assert boundary[DETECTOR_EXPERIMENT]["mean"] == 6.05574
    assert boundary[SEGMENTER_EXPERIMENT]["mean"] == 7.777487
    assert boundary["absolute_latency_delta_ms"] == 1.721747
    assert boundary["relative_latency_cost"] == 0.284317


def test_exact_end_to_end_latency_values(synthesis: dict[str, Any]) -> None:
    boundary = synthesis["cost"]["latency"]["boundaries"][END_TO_END_LATENCY]
    assert boundary[DETECTOR_EXPERIMENT]["mean"] == 9.157766
    assert boundary[SEGMENTER_EXPERIMENT]["mean"] == 11.914757
    assert boundary["absolute_latency_delta_ms"] == 2.756991
    assert boundary["relative_latency_cost"] == 0.301055


def test_latency_distribution_reports_median_and_p95(synthesis: dict[str, Any]) -> None:
    boundary = synthesis["cost"]["latency"]["boundaries"][END_TO_END_LATENCY]
    assert boundary[DETECTOR_EXPERIMENT]["median"] == 7.30475
    assert boundary[SEGMENTER_EXPERIMENT]["median"] == 9.96245
    assert boundary[DETECTOR_EXPERIMENT]["p95"] == 14.145295
    assert boundary[SEGMENTER_EXPERIMENT]["p95"] == 17.240505
    distribution = synthesis["cost"]["latency"]["distribution"]
    assert distribution["observations_used"] == 4800
    assert distribution["observations_discarded"] == 0
    assert distribution["outlier_rejection_applied"] is False
    assert distribution["headline_statistic"] == "mean"


def test_throughput_uses_mean_derived_values(synthesis: dict[str, Any]) -> None:
    throughput = synthesis["cost"]["latency"]["boundaries"][END_TO_END_LATENCY]["throughput"]
    assert throughput["label"] == MEAN_DERIVED_THROUGHPUT
    assert throughput[DETECTOR_EXPERIMENT] == 109.196937
    assert throughput[SEGMENTER_EXPERIMENT] == 83.929534
    assert throughput["throughput_ratio"] == 0.768607
    assert throughput["from_fastest_iteration"] is False
    assert throughput["is_batched_throughput"] is False
    assert throughput["is_application_video_fps"] is False


def test_latency_result_is_scoped_to_conf_0_25(synthesis: dict[str, Any]) -> None:
    latency = synthesis["cost"]["latency"]
    assert latency["scope"] == LATENCY_SCOPE == "OPERATIONAL_OUTPUT_LATENCY_AT_CONF_0_25"
    assert latency["conditions"]["conf"] == OPERATIONAL_CONF == 0.25


def test_protocol_gap_not_falsely_attributed_to_phase_10a(synthesis: dict[str, Any]) -> None:
    gap = synthesis["cost"]["latency"]["confidence_protocol_gap"]
    assert gap["label"] == PROTOCOL_GAP_RESOLUTION
    assert gap["phase_10a_froze_latency_confidence"] is False
    assert gap["resolved_to"] == 0.25
    assert gap["resolved_before_any_timing_existed"] is True
    assert gap["applied_equally_to_both_models"] is True
    assert gap["latency_measured_at_ap_confidence"] is False
    assert gap["benchmark_invalidated"] is False


def test_dvfs_causal_attribution_remains_unknown(synthesis: dict[str, Any]) -> None:
    dvfs = synthesis["cost"]["latency"]["dvfs"]
    assert dvfs["causal_attribution"] == "UNKNOWN"
    assert dvfs["hypothesis_status"] == "UNTESTED_HYPOTHESIS"
    assert dvfs["dvfs_asserted_as_cause"] is False
    assert dvfs["proportionality_across_models_demonstrated"] is False


def test_no_synchronized_telemetry_claim(synthesis: dict[str, Any]) -> None:
    dvfs = synthesis["cost"]["latency"]["dvfs"]
    assert (
        dvfs["per_observation_power_state_telemetry"]
        == "NO_SYNCHRONIZED_PER_OBSERVATION_POWER_STATE_TELEMETRY"
    )
    assert dvfs["telemetry_synchronous_with_timed_blocks"] is False


def test_latency_difference_is_pipeline_cost_not_isolated_mask_cost(
    synthesis: dict[str, Any],
) -> None:
    latency = synthesis["cost"]["latency"]
    assert latency["cost_label"] == "ADDITIONAL_SEGMENTATION_PIPELINE_COST"
    assert latency["pure_mask_reconstruction_cost_isolated"] is False
    assert latency["hardware_independent_claim"] is False


def test_memory_values_and_ratios(synthesis: dict[str, Any]) -> None:
    memory = synthesis["cost"]["memory"]
    assert memory["peak_allocated"][DETECTOR_EXPERIMENT]["gib"] == 0.073403
    assert memory["peak_allocated"][SEGMENTER_EXPERIMENT]["gib"] == 0.231621
    assert memory["peak_allocated"]["ratio"] == 3.155473
    assert memory["peak_reserved"][DETECTOR_EXPERIMENT]["gib"] == 0.125
    assert memory["peak_reserved"][SEGMENTER_EXPERIMENT]["gib"] == 0.296875
    assert memory["peak_reserved"]["ratio"] == 2.375
    assert memory["measurement"] == "INFERENCE_MEMORY"
    assert memory["comparable_with_training_memory"] is False


def test_memory_states_both_relative_and_absolute(synthesis: dict[str, Any]) -> None:
    memory = synthesis["cost"]["memory"]
    assert memory["relative_overhead_substantial"] is True
    assert memory["absolute_footprint_low_on_measured_device"] is True
    assert memory["memory_heavy_in_absolute_terms"] is False


def test_gflops_marked_as_640_reference_static_complexity(synthesis: dict[str, Any]) -> None:
    complexity = synthesis["cost"]["static_complexity"]
    assert complexity["label"] == "STATIC_MODEL_COMPLEXITY"
    assert complexity["reference_input_size"] == "FRAMEWORK_DEFAULT_640_NOT_THE_BENCHMARK_768"
    assert complexity["measured_at_benchmark_input_size"] is False
    assert complexity["derived_at_benchmark_input_size"] is False
    assert complexity["recomputed_in_this_phase"] is False
    assert complexity[DETECTOR_EXPERIMENT]["parameters"] == 2624080
    assert complexity[DETECTOR_EXPERIMENT]["gflops_at_reference_input"] == 6.673
    assert complexity[SEGMENTER_EXPERIMENT]["parameters"] == 2843583
    assert complexity[SEGMENTER_EXPERIMENT]["gflops_at_reference_input"] == 9.8
    assert complexity["explains_the_latency"] is False


# --- interpretation ----------------------------------------------------------------------


def test_no_weighted_or_composite_score(synthesis: dict[str, Any]) -> None:
    assert synthesis["aggregate_benefit_score"] is False
    assert synthesis["cost_benefit_index"] is False
    assert synthesis["weighted_score"] is False
    assert synthesis["winner_declared"] is False
    assert synthesis["axes_combined"] is False
    assert list(synthesis["axes"]) == list(AXES)


def test_use_case_conditional_interpretation(synthesis: dict[str, Any]) -> None:
    use_case = synthesis["use_case_conditional"]
    assert use_case["label"] == "USE_CASE_CONDITIONAL"
    assert use_case["universally_superior_model"] is None
    assert use_case["both_remain_frozen_final_models"] is True
    assert len(use_case["detector_preferred_when"]) >= 4
    assert len(use_case["segmenter_preferred_when"]) >= 4


def test_claim_register_covers_every_declared_claim(synthesis: dict[str, Any]) -> None:
    register = synthesis["claim_register"]
    assert tuple(entry["claim_id"] for entry in register) == CLAIM_IDS
    for entry in register:
        for field in CLAIM_FIELDS:
            assert entry[field], (entry["claim_id"], field)


def test_central_answer_scope_is_bounded(synthesis: dict[str, Any]) -> None:
    scope = synthesis["central_scientific_answer"]["scope"]
    assert scope["split"] == "validation"
    assert scope["generalises_to_test"] is False
    assert scope["generalises_to_other_hardware"] is False
    assert scope["generalises_beyond_frozen_association_rule"] is False
    assert scope["significance_tested"] is False


def test_remaining_work_is_identified_and_nothing_marked_complete(
    synthesis: dict[str, Any],
) -> None:
    items = {entry["item"]: entry for entry in synthesis["remaining_work"]}
    assert "FINAL_ONE_SHOT_HOLDOUT_EVALUATION" in items
    assert "REAL_VIDEO_INFERENCE_AT_LEAST_30_SECONDS" in items
    assert "EXECUTABLE_COLAB_NOTEBOOK" in items
    assert "ACADEMIC_REPORT_AND_PDF" in items
    assert "PITCH_SCRIPT_AND_RECORDING" in items
    for entry in synthesis["remaining_work"]:
        assert "COMPLETE" not in entry["status"]


def test_assignment_coverage_marks_pending_items_separately(synthesis: dict[str, Any]) -> None:
    coverage = synthesis["assignment_coverage"]
    assert coverage["delivered"]
    assert coverage["pending"]
    assert set(coverage["delivered"]) & set(coverage["pending"]) == set()
    assert "holdout_metrics_for_both_tasks" in coverage["pending"]
    for entry in coverage["delivered"].values():
        assert entry["split"] == "validation"


# --- holdout and execution ---------------------------------------------------------------


def test_holdout_protected_and_no_test_result(synthesis: dict[str, Any]) -> None:
    assert synthesis["holdout_policy"]["status"] == HOLDOUT_STATUS == "PROTECTED_NOT_ACCESSED"
    assert synthesis["holdout_policy"]["final_test_claims"] is False
    assert synthesis["test"]["status"] == HOLDOUT_STATUS
    assert synthesis["test"]["images_read"] == 0
    assert synthesis["test"]["predictions"] == 0
    assert synthesis["test"]["statistics"] == 0
    assert synthesis["execution"]["test_accessed"] is False


def test_no_model_was_executed(synthesis: dict[str, Any]) -> None:
    execution = synthesis["execution"]
    assert execution["models_executed"] == 0
    assert execution["models_trained"] == 0
    assert execution["latency_measurements_taken"] == 0
    assert execution["memory_measurements_taken"] == 0
    assert execution["AP_recomputed"] is False
    assert execution["spatial_analysis_rerun"] is False
    assert execution["association_analysis_rerun"] is False
    assert execution["thresholds_tuned"] == 0
    assert execution["new_metrics_introduced"] == 0
    assert execution["post_hoc_bins_created"] is False
    assert execution["significance_tests_run"] == 0
    assert execution["confidence_intervals_computed"] == 0


def test_no_model_execution_path_in_the_runner_or_module() -> None:
    """Neither file can train, infer, time or unlock the holdout.

    The forbidden strings are call paths, not words: `allow_test=True` appears
    in the module as prose describing the holdout policy, so what is asserted
    here is that no split-access function is ever *called*.
    """
    for path in (SCRIPT, MODULE):
        text = path.read_text(encoding="utf-8")
        for forbidden in (
            "import torch",
            "from torch",
            "import ultralytics",
            "from ultralytics",
            "YOLO(",
            "perf_counter",
            "torch.cuda",
            "assert_split_allowed(",
            "load_frozen_splits(",
            "FrozenSplits(",
            "os.environ[",
        ):
            assert forbidden not in text, (path.name, forbidden)


# --- validator ---------------------------------------------------------------------------


def test_the_committed_synthesis_validates(
    synthesis: dict[str, Any], sources: dict[str, dict[str, Any]]
) -> None:
    assert validate_synthesis(synthesis, sources) == []
    assert synthesis["status"] == SYNTHESIS_COMPLETE
    assert synthesis["phase"] == PHASE == "10D"


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("aggregate_benefit_score",), True),
        (("winner_declared",), True),
        (("axes_combined",), True),
        (("execution", "models_executed"), 1),
        (("execution", "AP_recomputed"), True),
        (("execution", "test_accessed"), True),
        (("holdout_policy", "status"), "READ"),
        (("recognition", "superiority_claimed_for_either_model"), True),
        (("association", "compliance_accuracy_claimed"), True),
        (("association", "mask_only_associations"), 4),
    ],
)
def test_validator_rejects_a_tampered_synthesis(
    synthesis: dict[str, Any],
    sources: dict[str, dict[str, Any]],
    path: tuple[str, ...],
    value: Any,
) -> None:
    tampered = copy.deepcopy(synthesis)
    target = tampered
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    assert validate_synthesis(tampered, sources)


def test_validator_rejects_an_edited_headline_number(
    synthesis: dict[str, Any], sources: dict[str, dict[str, Any]]
) -> None:
    tampered = copy.deepcopy(synthesis)
    tampered["cost"]["latency"]["boundaries"][END_TO_END_LATENCY]["absolute_latency_delta_ms"] = 1.0
    assert validate_synthesis(tampered, sources)


def test_validator_rejects_a_claim_that_phase_10a_froze_the_latency_confidence(
    synthesis: dict[str, Any], sources: dict[str, dict[str, Any]]
) -> None:
    tampered = copy.deepcopy(synthesis)
    tampered["cost"]["latency"]["confidence_protocol_gap"]["phase_10a_froze_latency_confidence"] = (
        True
    )
    assert validate_synthesis(tampered, sources)


def test_validator_rejects_an_asserted_dvfs_cause(
    synthesis: dict[str, Any], sources: dict[str, dict[str, Any]]
) -> None:
    tampered = copy.deepcopy(synthesis)
    tampered["cost"]["latency"]["dvfs"]["causal_attribution"] = "MOBILE_GPU_DVFS"
    assert validate_synthesis(tampered, sources)


def test_validator_rejects_a_nested_composite_score(
    synthesis: dict[str, Any], sources: dict[str, dict[str, Any]]
) -> None:
    tampered = copy.deepcopy(synthesis)
    tampered["cost"]["cost_benefit_index"] = 0.42
    assert validate_synthesis(tampered, sources)


# --- renderers ---------------------------------------------------------------------------


def test_tradeoff_rows_carry_the_declared_fields_and_no_score(
    synthesis: dict[str, Any],
) -> None:
    rows = tradeoff_rows(synthesis)
    assert rows
    for row in rows:
        assert set(row) == set(TRADEOFF_FIELDS)
        assert "score" not in row["dimension"].lower()
    dimensions = " ".join(row["dimension"] for row in rows)
    for expected in (
        "Recognition / localisation",
        "Spatial geometry",
        "Person-PPE association",
        MODEL_INFERENCE_LATENCY,
        END_TO_END_LATENCY,
        MEAN_DERIVED_THROUGHPUT,
        "Peak allocated inference memory",
        "Peak reserved inference memory",
        "Static parameters",
        "Static reference GFLOPs",
    ):
        assert expected in dimensions


def test_markdown_renders_every_declared_section(synthesis: dict[str, Any]) -> None:
    text = render_markdown(synthesis)
    for heading in (
        "## 1. Research question",
        "## 4. Recognition and localisation",
        "## 6. What masks represent beyond boxes",
        "## 9. Person-PPE association",
        "## 13. End-to-end output cost",
        "## 18. Confidence-protocol-gap disclosure",
        "## 21. Central scientific answer",
        "## 25. Next phase",
    ):
        assert heading in text


def test_markdown_has_no_uninterpolated_placeholders(synthesis: dict[str, Any]) -> None:
    text = render_markdown(synthesis)
    for placeholder in ("{latency[", "{payload[", "{spatial[", "{memory[", "{recognition["):
        assert placeholder not in text
