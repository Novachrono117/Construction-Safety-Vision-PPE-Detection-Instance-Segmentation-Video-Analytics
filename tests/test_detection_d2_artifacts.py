"""Tests that the committed D2 result describes the run that actually happened.

Same discipline as the D1 tests: structure, provenance and arithmetic, not
expected metric values. The D2-specific invariants are that the *only* declared
difference from D0 is the input resolution, that D2 started from the **same
pretrained bytes** D0 started from, and that both its comparisons - against D0
and against D1 - are recorded without any winner being declared.
"""

from __future__ import annotations

import json
from decimal import Decimal

import pytest

from construction_safety_vision.data.canonical import scan_for_sensitive
from construction_safety_vision.data.materialization import sha256_bytes
from construction_safety_vision.detection_comparison import (
    DESCRIPTIVE_HIGH_UNCERTAINTY,
    OFFICIAL_ALL_CLASS_METRIC,
    PRIMARY_SELECTION_METRIC,
    class_support,
    classify_delta,
    descriptive_class_names,
    load_experiment_matrix,
    supported_class_names,
    supported_macro,
)
from construction_safety_vision.detection_results import (
    EXPERIMENT_COMPLETE,
    PER_CLASS_METRICS,
    TEST_PROTECTED,
    validate_result_manifest,
)
from construction_safety_vision.detection_run import FRAMEWORK_LOG_LINE
from construction_safety_vision.paths import ProjectPaths

CLASS_NAMES = ("helmet_loose", "helmet_on_head", "person", "vest_loose", "vest_on_body")
UNSELECTED = "UNSELECTED_PENDING_REVIEW"


@pytest.fixture(scope="module")
def paths() -> ProjectPaths:
    return ProjectPaths.from_root()


@pytest.fixture(scope="module")
def manifest(paths: ProjectPaths) -> dict:
    path = paths.reports / "detection_D2_manifest.json"
    if not path.is_file():
        pytest.skip("detection_D2_manifest.json not present; run the D2 experiment")
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def report(paths: ProjectPaths) -> str:
    path = paths.reports / "detection_D2_report.md"
    if not path.is_file():
        pytest.skip("detection_D2_report.md not present")
    return path.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def results(paths: ProjectPaths) -> dict:
    path = paths.reports / "detection_experiment_results.json"
    if not path.is_file():
        pytest.skip("detection_experiment_results.json not present")
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def matrix(paths: ProjectPaths):
    return load_experiment_matrix(paths.configs / "detection_experiments.yaml")


@pytest.fixture(scope="module")
def d0_manifest(paths: ProjectPaths) -> dict:
    return json.loads((paths.reports / "detection_D0_manifest.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def d1_manifest(paths: ProjectPaths) -> dict:
    path = paths.reports / "detection_D1_manifest.json"
    if not path.is_file():
        pytest.skip("detection_D1_manifest.json not present")
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def support(paths: ProjectPaths, matrix) -> tuple:
    split = json.loads((paths.reports / "split_manifest.json").read_text(encoding="utf-8"))
    return class_support(
        split, dict(zip(CLASS_NAMES, range(5), strict=True)), rule=matrix.support_rule
    )


# --- manifest validity --------------------------------------------------------


def test_manifest_is_valid(manifest):
    assert validate_result_manifest(manifest, class_names=CLASS_NAMES) == []


def test_manifest_declares_the_frozen_experiment(manifest):
    assert manifest["experiment_id"] == "D2"
    assert manifest["task"] == "detection"
    assert manifest["model"] == "YOLO11n"
    assert manifest["status"] == EXPERIMENT_COMPLETE
    assert manifest["weights_committed"] is False


def test_the_intentional_variable_is_input_resolution(manifest):
    assert manifest["intentional_variable"] == "INPUT_RESOLUTION"
    assert manifest["intentional_fields"] == ["training.imgsz"]
    assert manifest["consequential_fields"] == []
    assert manifest["overrides"] == {"training.imgsz": 768}
    assert manifest["inherits"] == "D0"
    assert manifest["reference_experiment"] == "D0"


def test_the_manifest_matches_the_frozen_matrix_and_policy(manifest, paths, matrix):
    assert manifest["experiment_matrix_sha256"] == sha256_bytes(
        paths.configs / "detection_experiments.yaml"
    )
    assert manifest["phase7_policy_sha256"] == sha256_bytes(
        paths.reports / "detection_comparison_policy.json"
    )
    assert manifest["overrides"] == matrix.declaration("D2").overrides


def test_the_manifest_binds_to_the_same_frozen_data_as_the_reference(manifest, d0_manifest):
    assert manifest["adapter_manifest_sha256"] == d0_manifest["adapter_manifest_sha256"]
    assert manifest["dataset_fingerprints"] == d0_manifest["dataset_fingerprints"]
    assert manifest["class_map"] == d0_manifest["class_map"]
    assert (
        manifest["split_reference"]["split_assignment_sha256"]
        == (d0_manifest["split_reference"]["split_assignment_sha256"])
    )


# --- one-variable contract ----------------------------------------------------


def test_only_the_resolution_differs_from_the_reference(manifest):
    verdict = manifest["protocol_compatibility"]
    assert verdict["compatible"] is True
    assert verdict["observed_differences"] == ["training.imgsz"]
    assert verdict["undeclared_differences"] == []
    assert verdict["unapplied_declarations"] == []
    assert verdict["reference_experiment"] == "D0"


def test_capacity_and_weights_did_not_move(manifest, paths):
    from construction_safety_vision.experiment import load_detection_baseline_config

    baseline = load_detection_baseline_config(paths.configs / "detection_baseline.yaml")
    resolved = manifest["resolved_protocol"]
    assert resolved["model"] == baseline.model == "YOLO11n"
    assert resolved["weight_identifier"] == baseline.weight_identifier == "yolo11n.pt"
    assert resolved["training.batch"] == baseline.training["batch"] == 16
    assert resolved["training.epochs"] == baseline.training["epochs"] == 100
    assert resolved["training.optimizer"] == baseline.training["optimizer"] == "auto"
    assert resolved["training.patience"] == baseline.training["patience"]
    assert resolved["training.augmentation"] == baseline.training["augmentation"]
    assert resolved["training.deterministic"] is True
    assert resolved["training.amp"] is True
    assert resolved["seed"] == baseline.seed == 42
    assert resolved["dataset_config"] == baseline.dataset_config
    assert resolved["checkpoint_selection"] == baseline.checkpoint_selection
    # And the one declared difference really did change.
    assert resolved["training.imgsz"] == 768 != baseline.training["imgsz"]


def test_the_effective_run_used_768_at_the_frozen_batch(manifest):
    resolved = manifest["resolved_training_arguments"]
    assert resolved["imgsz"] == 768
    assert resolved["batch"] == 16
    assert resolved["seed"] == 42
    assert resolved["epochs"] == 100
    assert resolved["deterministic"] is True
    assert resolved["amp"] is True


def test_the_pretrained_bytes_are_identical_to_the_reference(manifest, d0_manifest):
    # The point of this experiment is resolution. Starting from a different
    # yolo11n.pt would add an undeclared variable, so the digests must match.
    weights = manifest["pretrained_weights"]
    reference = d0_manifest["pretrained_weights"]
    assert weights["identifier"] == reference["identifier"] == "yolo11n.pt"
    assert weights["sha256"] == reference["sha256"]
    assert weights["size_bytes"] == reference["size_bytes"]
    assert weights["committed"] is False


def test_the_weight_identity_relationship_is_recorded(manifest):
    assert manifest["pretrained_weight_identity"] == "IDENTICAL_TO_REFERENCE_VERIFIED_BY_DIGEST"


def test_the_pretrained_bytes_match_the_phase_6a_provenance(manifest, paths):
    runtime = json.loads(
        (paths.reports / "detection_runtime.provenance.json").read_text(encoding="utf-8")
    )
    recorded = runtime["details"]["pretrained_weights"]
    assert manifest["pretrained_weights"]["sha256"] == recorded["sha256"]
    assert manifest["pretrained_weights"]["source_mechanism"] == recorded["source_mechanism"]


def test_the_weights_on_disk_still_match_the_manifest(manifest, paths):
    path = paths.root / manifest["pretrained_weights"]["relative_path"]
    if not path.is_file():
        pytest.skip("weights are git-ignored and not present")
    assert sha256_bytes(path) == manifest["pretrained_weights"]["sha256"]


# --- optimizer evidence -------------------------------------------------------


def test_the_actual_optimizer_is_recorded_not_the_policy(manifest):
    assert manifest["declared_optimizer_policy"] == "auto"
    determination = manifest["resolved_optimizer_determination"]
    assert manifest["resolved_optimizer"] == determination["optimizer"]
    assert manifest["resolved_optimizer"] != "auto"
    assert manifest["resolved_optimizer"] not in (None, "", "NOT_EXPOSED_RELIABLY")
    assert isinstance(determination["effective_lr0"], (int, float))
    assert "inferred" in determination


def test_a_directly_captured_optimizer_carries_its_log_evidence(manifest):
    determination = manifest["resolved_optimizer_determination"]
    if determination["source"] != FRAMEWORK_LOG_LINE:
        pytest.skip("this run established its optimizer by inference")
    assert determination["inferred"] is False
    assert "optimizer:" in determination["evidence"]
    assert "\x1b" not in determination["evidence"]


# --- checkpoints --------------------------------------------------------------


def test_checkpoint_hashes_are_recorded_and_not_committed(manifest):
    for key in ("best_checkpoint", "last_checkpoint"):
        record = manifest[key]
        assert len(record["sha256"]) == 64
        assert record["size_bytes"] > 0
        assert record["committed"] is False
        assert record["relative_path"].startswith("artifacts/detection/D2/weights/")


def test_the_checkpoint_rule_is_the_frozen_one(manifest, d0_manifest):
    assert manifest["checkpoint_selection"] == d0_manifest["checkpoint_selection"]
    assert isinstance(manifest["best_epoch"], int)
    assert 1 <= manifest["best_epoch"] <= manifest["epochs_completed"]


def test_the_headline_metric_belongs_to_the_selected_checkpoint(manifest):
    cross = manifest["headline_metric_cross_check"]
    if not isinstance(cross["delta"], (int, float)):
        pytest.skip("the epoch history did not expose a comparable metric")
    assert cross["delta"] <= cross["tolerance"]


def test_no_diverted_run_directory_exists(paths):
    root = paths.root / "artifacts" / "detection"
    if not root.is_dir():
        pytest.skip("run directories are git-ignored and not present")
    assert [path.name for path in root.iterdir() if path.name.startswith("D2-")] == []


# --- validation protocol ------------------------------------------------------


def test_validation_ran_at_the_experiment_resolution(manifest, d0_manifest):
    configuration = manifest["validation_configuration"]
    assert configuration["imgsz"] == 768
    assert configuration["batch"] == 16
    assert configuration["split"] == "val"
    # Everything except the resolution matches the reference's evaluation.
    reference = d0_manifest["validation_configuration"]
    for key in ("batch", "conf", "iou", "max_det", "split"):
        assert configuration[key] == reference[key]


# --- the phase 7 comparison ---------------------------------------------------


def test_the_supported_classes_are_mechanically_derived(manifest, support):
    comparison = manifest["phase7_comparison"]
    assert comparison["selection_metric_classes"] == list(supported_class_names(support))
    assert comparison["descriptive_classes"] == list(descriptive_class_names(support))
    assert comparison["selection_metric_classes"] == [
        "helmet_loose",
        "helmet_on_head",
        "person",
        "vest_on_body",
    ]


def test_the_rare_class_remains_descriptive_and_fully_reported(manifest):
    comparison = manifest["phase7_comparison"]
    assert comparison["descriptive_classes"] == ["vest_loose"]
    assert "vest_loose" not in comparison["selection_metric_classes"]
    assert manifest["rare_class"]["classification"] == DESCRIPTIVE_HIGH_UNCERTAINTY
    table = manifest["per_class_metrics"]["vest_loose"]
    for metric in PER_CLASS_METRICS:
        assert metric in table
    assert set(manifest["per_class_metrics"]) == set(CLASS_NAMES)


def test_the_supported_macro_is_the_mean_of_the_supported_per_class_values(manifest):
    comparison = manifest["phase7_comparison"]
    recomputed = supported_macro(
        manifest["per_class_metrics"], comparison["selection_metric_classes"]
    )
    assert round(float(recomputed), 6) == comparison["supported_macro_map50_95"]
    assert Decimal(comparison["supported_macro_exact"]) == recomputed


def test_the_delta_against_the_reference_is_the_difference_of_the_macros(manifest):
    comparison = manifest["phase7_comparison"]
    expected = round(
        comparison["supported_macro_map50_95"] - comparison["reference_supported_macro_map50_95"],
        6,
    )
    assert comparison["delta_supported_macro_vs_reference"] == pytest.approx(expected, abs=1e-6)
    assert comparison["reference_supported_macro_map50_95"] == 0.570142


def test_the_delta_against_the_other_candidate_is_recorded(manifest, d1_manifest):
    peers = manifest["phase7_comparison"]["deltas_vs_peers"]
    assert "D1" in peers
    d1_macro = d1_manifest["phase7_comparison"]["supported_macro_map50_95"]
    expected = round(manifest["phase7_comparison"]["supported_macro_map50_95"] - d1_macro, 6)
    assert peers["D1"][PRIMARY_SELECTION_METRIC] == pytest.approx(expected, abs=1e-6)
    assert peers["D1"]["reference_value"] == d1_macro


def test_the_all_class_delta_against_the_other_candidate_is_recorded(manifest, d1_manifest):
    peers = manifest["phase7_comparison"]["deltas_vs_peers"]
    expected = round(
        manifest["validation_metrics"][OFFICIAL_ALL_CLASS_METRIC]
        - d1_manifest["validation_metrics"][OFFICIAL_ALL_CLASS_METRIC],
        6,
    )
    assert peers["D1"][OFFICIAL_ALL_CLASS_METRIC] == pytest.approx(expected, abs=1e-6)


def test_the_margin_status_follows_the_frozen_margin(manifest):
    comparison = manifest["phase7_comparison"]
    assert comparison["practical_equivalence_margin"] == "0.005"
    delta = Decimal(str(comparison["delta_supported_macro_vs_reference"]))
    expected = {
        "IMPROVED_BEYOND_MARGIN": "IMPROVES_D0_BEYOND_MARGIN",
        "PRACTICALLY_EQUIVALENT_ON_THIS_VALIDATION_SET": "PRACTICALLY_EQUIVALENT_TO_D0",
        "REGRESSED_BEYOND_MARGIN": "BELOW_D0",
    }[classify_delta(delta)]
    assert comparison["margin_status"] == expected


def test_the_all_class_metric_is_preserved(manifest, d0_manifest):
    comparison = manifest["phase7_comparison"]
    assert comparison["official_all_class_metric"] == OFFICIAL_ALL_CLASS_METRIC
    deltas = comparison["deltas_vs_reference"]
    for name in (OFFICIAL_ALL_CLASS_METRIC, "mAP@0.50", "precision", "recall"):
        assert name in manifest["validation_metrics"]
        expected = round(
            manifest["validation_metrics"][name] - d0_manifest["validation_metrics"][name], 6
        )
        assert deltas[name] == pytest.approx(expected, abs=1e-6)


# --- no winner ----------------------------------------------------------------


def test_no_final_detector_is_selected(manifest, report):
    # As in the D1 tests, the live results artifact is deliberately not asserted
    # here: Phase 7D owns its selection state. The invariant these historical D2
    # artifacts protect is that the *experiment* phase froze no winner.
    #
    # Affirmative selection phrasing only. "no final detector is declared" is a
    # legitimate sentence, so matching on "final detector is" would flag the very
    # disclaimer the test wants to see.
    lowered = report.lower()
    for phrase in ("is the phase 7 winner", "the final detector is", "we select d2"):
        assert phrase not in lowered
    assert "no final detector is declared" in lowered or "unselected" in lowered
    assert manifest["phase7_comparison"]["selection_pending"] is True


def test_the_experiment_manifest_reports_a_comparison_without_deciding_it(manifest):
    comparison = manifest["phase7_comparison"]
    # The experiment records every number the frozen rule needs and stops there.
    assert comparison["margin_status"].startswith(("IMPROVES", "BELOW", "PRACTICALLY"))
    assert comparison["selection_pending"] is True
    assert "not applied by an experiment phase" in comparison["selection_pending_reason"]
    assert UNSELECTED not in json.dumps(comparison)
    for key in ("final_selected_detector", "preferred_experiment", "winner"):
        assert key not in comparison


def test_the_live_results_record_all_three_as_complete(results):
    rows = {row["experiment_id"]: row for row in results["experiments"]}
    assert set(rows) == {"D0", "D1", "D2"}
    for experiment_id in ("D0", "D1", "D2"):
        assert rows[experiment_id]["status"] == "COMPLETE"
        assert rows[experiment_id]["metrics"] is not None
    assert results["pending_experiments"] == []


def test_the_frozen_policy_artifact_was_not_mutated(paths, manifest):
    # The policy records the statuses it was frozen with; result manifests
    # reference its digest, so updating execution status there would invalidate
    # their provenance.
    policy = json.loads(
        (paths.reports / "detection_comparison_policy.json").read_text(encoding="utf-8")
    )
    statuses = {
        item["experiment_id"]: item["status"]
        for item in policy["experiments"]
        if item["role"] == "CONTROLLED_EXPERIMENT"
    }
    assert statuses == {"D1": "FROZEN_NOT_EXECUTED", "D2": "FROZEN_NOT_EXECUTED"}
    assert manifest["phase7_policy_sha256"] == sha256_bytes(
        paths.reports / "detection_comparison_policy.json"
    )


# --- fingerprint --------------------------------------------------------------


def test_the_experiment_fingerprint_is_recorded_and_distinct(manifest, d0_manifest, d1_manifest):
    assert len(manifest["experiment_sha256"]) == 64
    assert manifest["experiment_sha256"] != d0_manifest["d0_experiment_sha256"]
    assert manifest["experiment_sha256"] != d1_manifest["experiment_sha256"]


def test_the_fingerprint_covers_every_critical_input(manifest):
    from construction_safety_vision.detection_results import critical_arguments, digest

    recomputed = digest(
        {
            "phase7_policy_sha256": manifest["phase7_policy_sha256"],
            "experiment_matrix_sha256": manifest["experiment_matrix_sha256"],
            "resolved_config_sha256": manifest["resolved_config_sha256"],
            "pretrained_weights_sha256": manifest["pretrained_weights"]["sha256"],
            "adapter_manifest_sha256": manifest["adapter_manifest_sha256"],
            "split_assignment_sha256": manifest["split_reference"]["split_assignment_sha256"],
            "class_map_sha256": manifest["dataset_fingerprints"]["class_map_sha256"],
            "critical_arguments": critical_arguments(manifest["resolved_training_arguments"]),
            "best_checkpoint_sha256": manifest["best_checkpoint"]["sha256"],
        }
    )
    assert recomputed == manifest["experiment_sha256"]


def test_the_fingerprint_is_deterministic(manifest, paths):
    reloaded = json.loads(
        (paths.reports / "detection_D2_manifest.json").read_text(encoding="utf-8")
    )
    assert reloaded["experiment_sha256"] == manifest["experiment_sha256"]


# --- holdout protection -------------------------------------------------------


def test_the_holdout_was_not_accessed(manifest):
    assert manifest["test"]["status"] == TEST_PROTECTED
    assert set(manifest["test"]) == {"status", "reason"}


def test_no_holdout_section_exists_outside_the_status_entry(manifest):
    from construction_safety_vision.detection_results import contains_forbidden_split

    without_test = {key: value for key, value in manifest.items() if key != "test"}
    assert not contains_forbidden_split(without_test)


def test_no_holdout_artifact_was_produced(paths):
    assert not (paths.data_processed / "canonical" / "images" / "test").exists()
    adapters = paths.data_processed / "adapters" / "yolo_detection"
    for kind in ("images", "labels"):
        assert not (adapters / kind / "test").exists()
    annotations = paths.data_processed / "canonical" / "annotations"
    if annotations.is_dir():
        assert not list(annotations.glob("*test*"))
    assert not (paths.root / "artifacts" / "detection" / "D2_test").exists()


def test_the_split_counts_are_the_frozen_development_ones(manifest):
    assert manifest["train_images"] == 303
    assert manifest["train_annotations"] == 1422
    assert manifest["validation_images"] == 65
    assert manifest["validation_annotations"] == 304


# --- artifacts fit to commit --------------------------------------------------


def test_the_artifacts_carry_no_sensitive_content(manifest, report, results):
    assert scan_for_sensitive(json.dumps(manifest, ensure_ascii=False)) == []
    assert scan_for_sensitive(json.dumps(results, ensure_ascii=False)) == []
    assert scan_for_sensitive(report) == []
    for text in (json.dumps(manifest), report):
        assert "ROBOFLOW_API_KEY" not in text


def test_only_metric_figures_are_committed(manifest, paths):
    for relative in manifest["committed_figures"]:
        assert relative.startswith("reports/figures/detection/D2/")
        assert relative.endswith(".png")
        name = relative.rsplit("/", 1)[-1]
        assert "batch" not in name
        assert "labels" not in name
    directory = paths.figures / "detection" / "D2"
    if directory.is_dir():
        names = {path.name for path in directory.iterdir()}
        assert not any("batch" in name for name in names)
        assert "labels.jpg" not in names


def test_the_report_labels_its_claims(report):
    for label in (
        "PREDECLARED_PROTOCOL",
        "COMPUTED_RESULT",
        "CONTROLLED_COMPARISON",
        "OBSERVATION",
        "LIMITATION",
    ):
        assert label in report


def test_the_report_compares_against_both_earlier_experiments(report):
    assert "## 10." in report
    assert "## 11." in report
    assert "D1" in report
    assert "D0" in report


def test_the_report_addresses_the_small_object_hypothesis(report):
    assert "small" in report.lower()
    assert "eda_source.json" in report or "0.01 relative box area" in report


def test_the_report_states_selection_is_pending(report):
    lowered = report.lower()
    assert "pending" in lowered
    assert UNSELECTED.lower() in lowered or "unselected" in lowered


def test_the_provenance_records_one_trained_model(paths):
    path = paths.reports / "detection_D2.provenance.json"
    if not path.is_file():
        pytest.skip("detection_D2.provenance.json not present")
    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["phase"] == 7
    details = record["details"]
    assert details["models_trained_in_this_phase"] == 1
    assert details["holdout_accessed"] is False
    assert details["selection_pending"] is True


def test_a_prerun_record_precedes_the_result(paths, manifest):
    path = paths.reports / "detection_D2_prerun.provenance.json"
    if not path.is_file():
        pytest.skip("detection_D2_prerun.provenance.json not present")
    prerun = json.loads(path.read_text(encoding="utf-8"))
    assert prerun["details"]["stage"] == "PRE_RUN"
    assert prerun["details"]["experiment_id"] == "D2"
    assert prerun["details"]["imgsz"] == 768
    assert prerun["config"]["resolved_config_sha256"] == manifest["resolved_config_sha256"]
    assert prerun["config"]["phase7_policy_sha256"] == manifest["phase7_policy_sha256"]
    assert (
        prerun["details"]["pretrained_weights"]["sha256"]
        == (manifest["pretrained_weights"]["sha256"])
    )
