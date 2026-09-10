"""Tests for the frozen detector-versus-segmenter comparison protocol.

Two halves. The first exercises the parser and the deterministic benchmark
selection with synthetic fixtures; the second checks the committed artifacts
describe the protocol that was actually frozen.

Most of the parser tests are refusals, because the protocol's job is to make
certain later moves impossible: swapping the two confidence thresholds,
deriving the segmenter's boxes from its masks to make them look tidier,
leaving a mask quantity without a declared box counterpart, benchmarking one
model in a different precision, shortening the warmup after seeing variance,
hiding mask reconstruction outside the segmenter's end-to-end timing, or
collapsing benefit and cost into one flattering number.

No test here runs a model, reads an image or touches the holdout.
"""

from __future__ import annotations

import csv
import json
from typing import Any

import pytest
import yaml

from construction_safety_vision.data.canonical import scan_for_sensitive
from construction_safety_vision.detector_segmenter_comparison import (
    AP_CONFIDENCE,
    ASSOCIATION_CATEGORIES,
    ASSOCIATION_LABEL,
    BENCHMARK_IMAGE_COUNT,
    BENCHMARK_LABEL,
    CANONICAL_BOX_METRIC,
    CANONICAL_BOX_METRIC_50,
    COMPARISON_IMGSZ,
    DETECTOR_EXPERIMENT,
    END_TO_END_LATENCY,
    EVALUATION_SPLIT,
    HOLDOUT_STATUS,
    IOU_THRESHOLDS,
    IOU_TYPE,
    LATENCY_BATCH,
    LATENCY_STATISTICS,
    MAX_DET,
    MAX_DETS,
    MODEL_INFERENCE_LATENCY,
    NMS_IOU,
    OPERATIONAL_CONFIDENCE,
    PRECISION,
    PRECISION_VALUE,
    PROTOCOL_NAME,
    SEGMENTER_EXPERIMENT,
    SELECTION_RULE,
    SPATIAL_FEATURES,
    STATUS_FROZEN_NOT_EXECUTED,
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
from construction_safety_vision.paths import ProjectPaths
from construction_safety_vision.provenance import sha256_file

CONFIG_NAME = "detector_segmenter_comparison.yaml"
PROTOCOL_JSON = "detector_segmenter_comparison_protocol.json"
PROTOCOL_MD = "detector_segmenter_comparison_protocol.md"
MEMBERSHIP_CSV = "detector_segmenter_comparison_membership.csv"
LATENCY_CSV = "detector_segmenter_latency_membership.csv"

DETECTOR_SHA = "0466f872a9de22898c70d8834cf1fcfb3d77f6c9fb27e81ee6248d3d19cbc206"
SEGMENTER_SHA = "29337d671459d0f742fb713613cc41d0eafcb8a21f5846ac0da65b2ffc024f20"


@pytest.fixture(scope="module")
def paths() -> ProjectPaths:
    """Resolve the project layout.

    Returns:
        The project paths.
    """
    return ProjectPaths.from_root()


def _write(tmp_path: Any, raw: dict[str, Any]) -> Any:
    """Write a protocol mapping to a temporary YAML file.

    Args:
        tmp_path: Temporary directory.
        raw: The protocol content.

    Returns:
        The written path.
    """
    path = tmp_path / CONFIG_NAME
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return path


@pytest.fixture
def raw(paths: ProjectPaths) -> dict[str, Any]:
    """Read the committed protocol as a plain mapping, for mutation.

    Args:
        paths: Project layout.

    Returns:
        The parsed YAML.
    """
    return yaml.safe_load((paths.configs / CONFIG_NAME).read_text(encoding="utf-8"))


# --- deterministic benchmark selection --------------------------------------------


def test_selection_depends_only_on_the_identifier() -> None:
    """The subset is reproducible from ids alone - no image is ever opened."""
    ids = [f"image-{index:03d}" for index in range(60)]

    assert select_benchmark_images(ids, 20) == select_benchmark_images(list(reversed(ids)), 20)


def test_selection_is_the_stable_hash_ranking() -> None:
    """The rule is exactly 'rank by SHA-256 of the id, take the first N'."""
    ids = [f"image-{index:03d}" for index in range(60)]

    assert list(select_benchmark_images(ids, 20)) == sorted(ids, key=stable_rank)[:20]


def test_selection_is_not_the_natural_order() -> None:
    """A hash ranking must not coincide with taking the first N alphabetically."""
    ids = [f"image-{index:03d}" for index in range(60)]

    assert list(select_benchmark_images(ids, 20)) != sorted(ids)[:20]


def test_selection_refuses_a_duplicate_membership() -> None:
    """A duplicated image would be benchmarked twice and weighted twice."""
    with pytest.raises(ComparisonProtocolError, match="duplicate"):
        select_benchmark_images(["a", "b", "a"], 2)


def test_selection_refuses_to_overdraw() -> None:
    """Asking for more images than exist is an error, not a short list."""
    with pytest.raises(ComparisonProtocolError, match="cannot select"):
        select_benchmark_images(["a", "b"], 20)


def test_membership_fingerprint_is_order_independent() -> None:
    """Membership is a set; its digest must not depend on how it was listed."""
    assert membership_fingerprint(["b", "a"]) == membership_fingerprint(["a", "b"])


def test_ordered_fingerprint_is_order_dependent() -> None:
    """Benchmark order is part of the protocol, so its digest must track it."""
    assert ordered_fingerprint(["b", "a"]) != ordered_fingerprint(["a", "b"])


# --- the parser ---------------------------------------------------------------------


def test_the_committed_protocol_parses(paths: ProjectPaths) -> None:
    """The fixture is accepted, so every refusal below is about its own change."""
    protocol = load_comparison_protocol(paths.configs / CONFIG_NAME)

    assert protocol["protocol"] == PROTOCOL_NAME
    assert protocol["status"] == STATUS_FROZEN_NOT_EXECUTED
    assert protocol["comparison_input_imgsz"] == COMPARISON_IMGSZ


def test_both_models_are_the_frozen_ones(paths: ProjectPaths) -> None:
    """The protocol names D2 and S1 by digest, at the same input size."""
    protocol = load_comparison_protocol(paths.configs / CONFIG_NAME)

    assert protocol["detector"]["experiment"] == DETECTOR_EXPERIMENT
    assert protocol["segmenter"]["experiment"] == SEGMENTER_EXPERIMENT
    assert protocol["detector"]["checkpoint_sha256"] == DETECTOR_SHA
    assert protocol["segmenter"]["checkpoint_sha256"] == SEGMENTER_SHA
    assert protocol["detector"]["imgsz"] == protocol["segmenter"]["imgsz"] == COMPARISON_IMGSZ


def test_a_mismatched_input_size_is_refused(tmp_path: Any, raw: dict[str, Any]) -> None:
    """Comparing at two resolutions would make it a resolution study."""
    raw["segmenter"]["imgsz"] = 640

    with pytest.raises(ComparisonProtocolError, match="resolution"):
        load_comparison_protocol(_write(tmp_path, raw))


def test_swapping_the_two_confidences_is_refused(tmp_path: Any, raw: dict[str, Any]) -> None:
    """AP needs the low-scoring tail; the operational protocol needs a working point."""
    raw["ap_inference"]["conf"] = OPERATIONAL_CONFIDENCE

    with pytest.raises(ComparisonProtocolError, match="never mixed"):
        load_comparison_protocol(_write(tmp_path, raw))


def test_an_operational_confidence_of_0_001_is_refused(tmp_path: Any, raw: dict[str, Any]) -> None:
    """The reverse swap is refused too."""
    raw["operational_inference"]["conf"] = AP_CONFIDENCE

    with pytest.raises(ComparisonProtocolError, match="never mixed"):
        load_comparison_protocol(_write(tmp_path, raw))


@pytest.mark.parametrize("block", ["ap_inference", "operational_inference"])
def test_test_time_augmentation_is_refused(tmp_path: Any, raw: dict[str, Any], block: str) -> None:
    """No TTA, in either protocol."""
    raw[block]["tta"] = True

    with pytest.raises(ComparisonProtocolError, match="augmentation"):
        load_comparison_protocol(_write(tmp_path, raw))


@pytest.mark.parametrize("block", ["ap_inference", "operational_inference"])
def test_a_different_precision_is_refused(tmp_path: Any, raw: dict[str, Any], block: str) -> None:
    """Both models are measured in one precision, pinned explicitly."""
    raw[block]["precision"] = "FP16"

    with pytest.raises(ComparisonProtocolError, match="precision"):
        load_comparison_protocol(_write(tmp_path, raw))


def test_deriving_segmenter_boxes_from_masks_is_refused(tmp_path: Any, raw: dict[str, Any]) -> None:
    """That would measure a post-processing choice this project invented."""
    raw["box_comparability"]["segmenter_boxes_derived_from_masks"] = True

    with pytest.raises(ComparisonProtocolError, match="actual predicted boxes"):
        load_comparison_protocol(_write(tmp_path, raw))


def test_an_unpaired_spatial_feature_is_refused(tmp_path: Any, raw: dict[str, Any]) -> None:
    """A mask quantity with no declared box counterpart cannot answer the question."""
    del raw["box_proxies"]["MASK_CENTROID"]

    with pytest.raises(ComparisonProtocolError, match="box proxy"):
        load_comparison_protocol(_write(tmp_path, raw))


def test_a_missing_spatial_feature_is_refused(tmp_path: Any, raw: dict[str, Any]) -> None:
    """The frozen set of quantities may not shrink."""
    del raw["spatial_features"]["SHAPE_EXTENT"]

    with pytest.raises(ComparisonProtocolError, match="spatial_features"):
        load_comparison_protocol(_write(tmp_path, raw))


def test_an_accuracy_framing_for_association_is_refused(tmp_path: Any, raw: dict[str, Any]) -> None:
    """There is no compliance ground truth to be accurate against."""
    raw["association"]["label"] = "COMPLIANCE_CLASSIFICATION_ACCURACY"

    with pytest.raises(ComparisonProtocolError, match="compliance ground truth"):
        load_comparison_protocol(_write(tmp_path, raw))


@pytest.mark.parametrize(
    "field", ["appearance_embeddings", "tracking", "learned_association", "classes_collapsed"]
)
def test_the_association_stays_geometric_and_class_aware(
    tmp_path: Any, raw: dict[str, Any], field: str
) -> None:
    """No embeddings, no tracking, no learned rule, no collapsed classes."""
    raw["association"][field] = True

    with pytest.raises(ComparisonProtocolError, match=field):
        load_comparison_protocol(_write(tmp_path, raw))


def test_the_disagreement_categories_are_exact(tmp_path: Any, raw: dict[str, Any]) -> None:
    """The four outcomes partition, and may not be extended after the fact."""
    raw["association"]["disagreement_categories"] = [*ASSOCIATION_CATEGORIES, "MOSTLY_AGREE"]

    with pytest.raises(ComparisonProtocolError, match="disagreement_categories"):
        load_comparison_protocol(_write(tmp_path, raw))


def test_the_box_evaluator_semantics_are_pinned(tmp_path: Any, raw: dict[str, Any]) -> None:
    """Standard COCO thresholds and caps, checked rather than assumed."""
    raw["canonical_box_evaluator"]["iou_thresholds"] = [0.50, 0.75]

    with pytest.raises(ComparisonProtocolError, match="iou_thresholds"):
        load_comparison_protocol(_write(tmp_path, raw))


def test_a_mask_iou_evaluator_is_refused(tmp_path: Any, raw: dict[str, Any]) -> None:
    """The recognition axis is a box comparison."""
    raw["canonical_box_evaluator"]["iou_type"] = "segm"

    with pytest.raises(ComparisonProtocolError, match="iou_type"):
        load_comparison_protocol(_write(tmp_path, raw))


def test_evaluating_against_an_adapter_is_refused(tmp_path: Any, raw: dict[str, Any]) -> None:
    """Its measured conversion loss would be folded into both models' scores."""
    raw["canonical_box_evaluator"]["ground_truth_document"] = (
        "data/processed/adapters/yolo_detection/labels.json"
    )

    with pytest.raises(ComparisonProtocolError, match="adapter"):
        load_comparison_protocol(_write(tmp_path, raw))


# --- the latency protocol -------------------------------------------------------------


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("batch", 8),
        ("warmup_iterations", 5),
        ("timed_iterations_per_image", 10),
        ("benchmark_image_count", 5),
        ("precision", "FP16"),
        ("imgsz", 640),
        ("label", "UNIVERSAL_MODEL_LATENCY"),
        ("benchmark_selection_rule", "HAND_PICKED"),
        ("abort_on_deviation", False),
    ],
)
def test_latency_settings_are_pinned(
    tmp_path: Any, raw: dict[str, Any], field: str, value: Any
) -> None:
    """Every knob a later phase might quietly turn is checked."""
    raw["latency_protocol"][field] = value

    with pytest.raises(ComparisonProtocolError):
        load_comparison_protocol(_write(tmp_path, raw))


def test_cuda_synchronization_is_required(tmp_path: Any, raw: dict[str, Any]) -> None:
    """Without it a wall-clock reading measures dispatch, not execution."""
    raw["latency_protocol"]["synchronization"]["cuda_synchronize"] = False

    with pytest.raises(ComparisonProtocolError, match="asynchronous"):
        load_comparison_protocol(_write(tmp_path, raw))


@pytest.mark.parametrize("edge", ["before_timed_region", "after_timed_region"])
def test_both_synchronization_edges_are_required(
    tmp_path: Any, raw: dict[str, Any], edge: str
) -> None:
    """Synchronising only one edge leaves the other measuring queue time."""
    raw["latency_protocol"]["synchronization"][edge] = False

    with pytest.raises(ComparisonProtocolError, match=edge):
        load_comparison_protocol(_write(tmp_path, raw))


def test_mask_reconstruction_must_be_inside_the_end_to_end_boundary(
    tmp_path: Any, raw: dict[str, Any]
) -> None:
    """Excluding it would hide exactly the cost this comparison quantifies."""
    raw["latency_protocol"]["timing_boundaries"][END_TO_END_LATENCY][
        "segmenter_mask_reconstruction_included"
    ] = False

    with pytest.raises(ComparisonProtocolError, match="mask reconstruction"):
        load_comparison_protocol(_write(tmp_path, raw))


def test_both_timing_boundaries_must_exist(tmp_path: Any, raw: dict[str, Any]) -> None:
    """Model-core and end-to-end are different questions and both are asked."""
    del raw["latency_protocol"]["timing_boundaries"][MODEL_INFERENCE_LATENCY]

    with pytest.raises(ComparisonProtocolError, match=MODEL_INFERENCE_LATENCY):
        load_comparison_protocol(_write(tmp_path, raw))


def test_a_missing_statistic_is_refused(tmp_path: Any, raw: dict[str, Any]) -> None:
    """The distribution is reported in full, not just a flattering summary."""
    raw["latency_protocol"]["statistics"] = ["mean"]

    with pytest.raises(ComparisonProtocolError, match="statistics"):
        load_comparison_protocol(_write(tmp_path, raw))


@pytest.mark.parametrize(
    ("field", "value"), [("interleaved", False), ("symmetric", False), ("randomized", True)]
)
def test_the_execution_order_stays_symmetric(
    tmp_path: Any, raw: dict[str, Any], field: str, value: Any
) -> None:
    """Running one model to completion first measures thermal state too."""
    raw["latency_protocol"]["execution_order"][field] = value

    with pytest.raises(ComparisonProtocolError, match=r"execution_order|thermal"):
        load_comparison_protocol(_write(tmp_path, raw))


def test_training_memory_may_not_be_reused(tmp_path: Any, raw: dict[str, Any]) -> None:
    """Inference memory is a different quantity."""
    raw["memory_protocol"]["training_memory_reused"] = True

    with pytest.raises(ComparisonProtocolError, match="training_memory_reused"):
        load_comparison_protocol(_write(tmp_path, raw))


def test_peak_stats_must_be_reset_after_warmup(tmp_path: Any, raw: dict[str, Any]) -> None:
    """Otherwise the figure is the allocator's warmup high-water mark."""
    raw["memory_protocol"]["reset_peak_stats_after_warmup"] = False

    with pytest.raises(ComparisonProtocolError, match="reset_peak_stats_after_warmup"):
        load_comparison_protocol(_write(tmp_path, raw))


# --- refusals about what may be concluded --------------------------------------------


def test_an_aggregate_benefit_score_is_refused(tmp_path: Any, raw: dict[str, Any]) -> None:
    """Collapsing benefit and cost would hide the trade-off."""
    raw["reporting"]["aggregate_benefit_score"] = True

    with pytest.raises(ComparisonProtocolError, match="aggregate_benefit_score"):
        load_comparison_protocol(_write(tmp_path, raw))


def test_the_required_box_metrics_may_not_be_dropped(tmp_path: Any, raw: dict[str, Any]) -> None:
    """Both headline box figures are mandatory."""
    raw["reporting"]["box_metrics"] = [CANONICAL_BOX_METRIC]

    with pytest.raises(ComparisonProtocolError, match=CANONICAL_BOX_METRIC_50):
        load_comparison_protocol(_write(tmp_path, raw))


def test_a_protocol_carrying_a_result_is_refused(tmp_path: Any, raw: dict[str, Any]) -> None:
    """Nothing has been executed, so nothing may be recorded."""
    raw["results"] = {"detector_map": 0.49}

    with pytest.raises(ComparisonProtocolError, match=r"unknown key|no results"):
        load_comparison_protocol(_write(tmp_path, raw))


def test_an_executed_status_is_refused(tmp_path: Any, raw: dict[str, Any]) -> None:
    """The protocol is frozen and not executed."""
    raw["status"] = "EXECUTED"

    with pytest.raises(ComparisonProtocolError, match=STATUS_FROZEN_NOT_EXECUTED):
        load_comparison_protocol(_write(tmp_path, raw))


def test_a_protocol_naming_the_holdout_is_refused(tmp_path: Any, raw: dict[str, Any]) -> None:
    """Validation only, structurally."""
    raw["population"]["split"] = "test"

    with pytest.raises(ComparisonProtocolError, match="protected split"):
        load_comparison_protocol(_write(tmp_path, raw))


def test_a_non_validation_population_is_refused(tmp_path: Any, raw: dict[str, Any]) -> None:
    """The comparison population is the validation split."""
    raw["population"]["split"] = "train"

    with pytest.raises(ComparisonProtocolError, match=r"population\.split"):
        load_comparison_protocol(_write(tmp_path, raw))


def test_models_must_see_identical_images(tmp_path: Any, raw: dict[str, Any]) -> None:
    """Different inputs would make every delta uninterpretable."""
    raw["population"]["identical_images_for_both_models"] = False

    with pytest.raises(ComparisonProtocolError, match="identical_images_for_both_models"):
        load_comparison_protocol(_write(tmp_path, raw))


# --- the manifest validator ------------------------------------------------------------


def _manifest(paths: ProjectPaths) -> dict[str, Any]:
    """Read the committed protocol manifest.

    Args:
        paths: Project layout.

    Returns:
        The manifest.
    """
    return json.loads((paths.reports / PROTOCOL_JSON).read_text(encoding="utf-8"))


def test_the_committed_manifest_validates(paths: ProjectPaths) -> None:
    """The validator accepts what the freeze wrote."""
    protocol = load_comparison_protocol(paths.configs / CONFIG_NAME)
    manifest = _manifest(paths)

    assert (
        validate_protocol_manifest(
            manifest,
            protocol_fingerprint=protocol.fingerprint(),
            detector_sha256=DETECTOR_SHA,
            segmenter_sha256=SEGMENTER_SHA,
            membership_sha256=manifest["population"]["membership_sha256"],
            benchmark_sha256=manifest["latency_protocol"]["benchmark_membership_sha256"],
        )
        == []
    )


@pytest.mark.parametrize(
    "kwarg",
    ["detector_sha256", "segmenter_sha256", "membership_sha256", "benchmark_sha256"],
)
def test_the_validator_rejects_a_wrong_fingerprint(paths: ProjectPaths, kwarg: str) -> None:
    """A wrong checkpoint, membership or benchmark order is caught."""
    protocol = load_comparison_protocol(paths.configs / CONFIG_NAME)
    manifest = _manifest(paths)
    arguments = {
        "protocol_fingerprint": protocol.fingerprint(),
        "detector_sha256": DETECTOR_SHA,
        "segmenter_sha256": SEGMENTER_SHA,
        "membership_sha256": manifest["population"]["membership_sha256"],
        "benchmark_sha256": manifest["latency_protocol"]["benchmark_membership_sha256"],
    }
    arguments[kwarg] = "9" * 64

    assert validate_protocol_manifest(manifest, **arguments)


def test_the_validator_rejects_a_drifted_protocol(paths: ProjectPaths) -> None:
    """A manifest naming a configuration other than the committed one is caught."""
    manifest = _manifest(paths)

    problems = validate_protocol_manifest(
        manifest,
        protocol_fingerprint="0" * 64,
        detector_sha256=DETECTOR_SHA,
        segmenter_sha256=SEGMENTER_SHA,
        membership_sha256=manifest["population"]["membership_sha256"],
        benchmark_sha256=manifest["latency_protocol"]["benchmark_membership_sha256"],
    )

    assert any("protocol_fingerprint" in problem for problem in problems)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("ap_inference", "conf"), 0.25),
        (("operational_inference", "conf"), 0.001),
        (("latency_protocol", "batch"), 8),
        (("latency_protocol", "warmup_iterations"), 3),
        (("latency_protocol", "timed_iterations_per_image"), 3),
        (("latency_protocol", "precision"), "FP16"),
        (("population", "split"), "train"),
    ],
)
def test_the_validator_rejects_a_changed_setting(
    paths: ProjectPaths, path: tuple[str, str], value: Any
) -> None:
    """Every frozen knob is re-checked against the manifest."""
    protocol = load_comparison_protocol(paths.configs / CONFIG_NAME)
    manifest = _manifest(paths)
    manifest[path[0]] = {**manifest[path[0]], path[1]: value}

    assert validate_protocol_manifest(
        manifest,
        protocol_fingerprint=protocol.fingerprint(),
        detector_sha256=DETECTOR_SHA,
        segmenter_sha256=SEGMENTER_SHA,
        membership_sha256=manifest["population"]["membership_sha256"],
        benchmark_sha256=manifest["latency_protocol"]["benchmark_membership_sha256"],
    )


def test_the_validator_rejects_a_result(paths: ProjectPaths) -> None:
    """Phase 10A produces no measurement."""
    protocol = load_comparison_protocol(paths.configs / CONFIG_NAME)
    manifest = {**_manifest(paths), "latency_results": {"D2": 12.0}}

    problems = validate_protocol_manifest(
        manifest,
        protocol_fingerprint=protocol.fingerprint(),
        detector_sha256=DETECTOR_SHA,
        segmenter_sha256=SEGMENTER_SHA,
        membership_sha256=manifest["population"]["membership_sha256"],
        benchmark_sha256=manifest["latency_protocol"]["benchmark_membership_sha256"],
    )

    assert any("latency_results" in problem for problem in problems)


# --- the committed artifacts -------------------------------------------------------------


def test_the_manifest_matches_the_committed_configuration(paths: ProjectPaths) -> None:
    """The manifest names the configuration file it was frozen from."""
    manifest = _manifest(paths)
    protocol = load_comparison_protocol(paths.configs / CONFIG_NAME)

    assert manifest["protocol_config_sha256"] == sha256_file(paths.configs / CONFIG_NAME)
    assert manifest["protocol_fingerprint"] == protocol.fingerprint()
    assert manifest["status"] == STATUS_FROZEN_NOT_EXECUTED


def test_nothing_was_executed(paths: ProjectPaths) -> None:
    """Counts, not prose: the phase ran no model and read no image."""
    manifest = _manifest(paths)

    assert manifest["models_executed_in_this_phase"] == 0
    assert manifest["predictions_produced"] == 0
    assert manifest["latency_measurements_taken"] == 0
    assert manifest["memory_measurements_taken"] == 0
    assert manifest["images_read"] == 0
    assert manifest["thresholds_tuned"] == 0
    assert manifest["models_trained_in_this_phase"] == 0
    assert manifest["detector"]["executed_in_this_phase"] is False
    assert manifest["segmenter"]["executed_in_this_phase"] is False


def test_the_membership_artifact_matches_its_fingerprint(paths: ProjectPaths) -> None:
    """The committed ids reproduce the digest the protocol names."""
    manifest = _manifest(paths)
    with (paths.reports / MEMBERSHIP_CSV).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    ids = [row["source_image_id"] for row in rows]

    assert len(ids) == manifest["population"]["images"] == 65
    assert membership_fingerprint(ids) == manifest["population"]["membership_sha256"]
    assert {row["split"] for row in rows} == {EVALUATION_SPLIT}


def test_the_benchmark_artifact_reproduces_the_selection(paths: ProjectPaths) -> None:
    """The committed subset is exactly what the frozen rule selects."""
    manifest = _manifest(paths)
    with (paths.reports / MEMBERSHIP_CSV).open(encoding="utf-8", newline="") as handle:
        population = [row["source_image_id"] for row in csv.DictReader(handle)]
    with (paths.reports / LATENCY_CSV).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    selected = [row["source_image_id"] for row in rows]

    assert len(selected) == BENCHMARK_IMAGE_COUNT
    assert selected == list(select_benchmark_images(population, BENCHMARK_IMAGE_COUNT))
    assert [int(row["benchmark_order"]) for row in rows] == list(range(BENCHMARK_IMAGE_COUNT))
    assert (
        ordered_fingerprint(selected) == manifest["latency_protocol"]["benchmark_membership_sha256"]
    )
    assert set(selected) <= set(population)


def test_the_benchmark_subset_is_a_strict_subset(paths: ProjectPaths) -> None:
    """20 of 65: a subset chosen mechanically, not the whole split."""
    with (paths.reports / LATENCY_CSV).open(encoding="utf-8", newline="") as handle:
        selected = [row["source_image_id"] for row in csv.DictReader(handle)]

    assert len(set(selected)) == BENCHMARK_IMAGE_COUNT < 65


def test_the_frozen_settings_are_recorded(paths: ProjectPaths) -> None:
    """Every headline number a later phase must honour is in the manifest."""
    manifest = _manifest(paths)
    latency = manifest["latency_protocol"]

    assert manifest["ap_inference"]["conf"] == AP_CONFIDENCE
    assert manifest["operational_inference"]["conf"] == OPERATIONAL_CONFIDENCE
    assert manifest["ap_inference"]["iou"] == manifest["operational_inference"]["iou"] == NMS_IOU
    assert manifest["ap_inference"]["max_det"] == MAX_DET
    assert manifest["comparison_input_imgsz"] == COMPARISON_IMGSZ
    assert latency["batch"] == LATENCY_BATCH
    assert latency["warmup_iterations"] == WARMUP_ITERATIONS
    assert latency["timed_iterations_per_image"] == TIMED_ITERATIONS_PER_IMAGE
    assert latency["benchmark_image_count"] == BENCHMARK_IMAGE_COUNT
    assert latency["precision"] == PRECISION
    assert latency["precision_value"] == PRECISION_VALUE
    assert latency["label"] == BENCHMARK_LABEL
    assert latency["benchmark_selection_rule"] == SELECTION_RULE
    assert set(LATENCY_STATISTICS) <= set(latency["statistics"])


def test_the_box_evaluator_is_recorded_exactly(paths: ProjectPaths) -> None:
    """Standard COCO bbox semantics, written down before any run."""
    evaluator = _manifest(paths)["canonical_box_evaluator"]

    assert evaluator["iou_type"] == IOU_TYPE
    assert tuple(round(float(v), 2) for v in evaluator["iou_thresholds"]) == IOU_THRESHOLDS
    assert tuple(int(v) for v in evaluator["max_dets"]) == MAX_DETS
    assert "adapter" not in evaluator["ground_truth_document"].lower()


def test_every_spatial_feature_has_a_proxy_decision(paths: ProjectPaths) -> None:
    """The pairing that answers 'what does segmentation add'."""
    manifest = _manifest(paths)

    assert set(manifest["spatial_features"]) == set(SPATIAL_FEATURES)
    assert set(manifest["box_proxies"]) == set(SPATIAL_FEATURES)
    assert "NO_BOX_ONLY_EQUIVALENT" in manifest["box_proxies"].values()


def test_the_association_analysis_is_descriptive(paths: ProjectPaths) -> None:
    """No compliance accuracy is claimed, because no such ground truth exists."""
    association = _manifest(paths)["association"]

    assert association["label"] == ASSOCIATION_LABEL
    assert tuple(association["disagreement_categories"]) == ASSOCIATION_CATEGORIES
    assert association["deterministic"] is True
    assert association["classes_collapsed"] is False
    assert "no canonical compliance ground truth" in association["semantics"]


def test_no_winner_and_no_aggregate_score(paths: ProjectPaths) -> None:
    """The comparison describes; it does not rank."""
    manifest = _manifest(paths)

    assert manifest["reporting"]["aggregate_benefit_score"] is False
    assert manifest["reporting"]["winner_declared"] is False
    assert manifest["interpretation"]["single_aggregate_score"] is False
    assert manifest["interpretation"]["weighted_cost_benefit_index"] is False


def test_the_historical_artifacts_are_byte_identical(paths: ProjectPaths) -> None:
    """Both freezes and every experiment artifact are untouched."""
    manifest = _manifest(paths)

    assert manifest["historical_artifacts_unchanged"] is True
    for name, expected in manifest["historical_artifact_digests"].items():
        assert sha256_file(paths.root / name) == expected, name


def test_no_artifact_mentions_the_holdout(paths: ProjectPaths) -> None:
    """Structurally absent, not merely unmentioned."""
    manifest = _manifest(paths)
    body = json.dumps({k: v for k, v in manifest.items() if k != "test"})

    assert manifest["test"]["status"] == HOLDOUT_STATUS
    assert manifest["holdout_accessed"] is False
    assert '"test"' not in body.replace('"test_policy"', "")

    for name in (MEMBERSHIP_CSV, LATENCY_CSV):
        text = (paths.reports / name).read_text(encoding="utf-8")
        assert "test" not in text


def test_the_report_carries_no_results_and_no_sensitive_content(paths: ProjectPaths) -> None:
    """A protocol document with a number in it would not be a protocol."""
    text = (paths.reports / PROTOCOL_MD).read_text(encoding="utf-8")

    assert scan_for_sensitive(text) == []
    assert "contains no results" in text
    for fragment in (
        "PREDECLARED_PROTOCOL",
        "FROZEN_MODEL",
        "CANONICAL_EVALUATION",
        "SPATIAL_INFORMATION",
        "BOX_PROXY",
        "OPERATIONAL_ANALYSIS",
        "CONTROLLED_HARDWARE_BENCHMARK",
        "LIMITATION",
        "HOLDOUT_POLICY",
    ):
        assert fragment in text, fragment


def test_the_freeze_script_has_no_inference_path(paths: ProjectPaths) -> None:
    """A protocol freeze that could run a model is a freeze that might."""
    source = (paths.root / "scripts" / "freeze_detector_segmenter_comparison.py").read_text(
        encoding="utf-8"
    )

    for forbidden in (
        "from ultralytics",
        "import ultralytics",
        "import torch",
        "model.predict(",
        "model.val(",
        "model.train(",
        "perf_counter(",
    ):
        assert forbidden not in source, forbidden
