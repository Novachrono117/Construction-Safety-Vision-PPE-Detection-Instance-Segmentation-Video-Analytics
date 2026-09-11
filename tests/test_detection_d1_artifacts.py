"""Tests that the committed D1 result describes the run that actually happened.

They read the committed manifest and check it against the frozen artifacts it
claims to derive from. They assert structure, provenance and arithmetic rather
than particular metric values: hard-coding an expected mAP would turn a test
into a claim about how well a larger model *should* do, which is exactly the
kind of assertion this project refuses to make.

Two things they do assert about numbers, because both are checkable facts rather
than expectations: that the selection metric is the mean of the per-class values
the frozen support rule admits, and that the delta against D0 is the difference
of the two recorded macros.
"""

from __future__ import annotations

import json
from decimal import Decimal

import pytest

from conftest import holdout_has_been_evaluated
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
from construction_safety_vision.detection_run import (
    DECLARED_EXPLICITLY,
    FRAMEWORK_LOG_LINE,
    REDERIVED_CORROBORATED,
    REDERIVED_UNCORROBORATED,
)
from construction_safety_vision.paths import ProjectPaths

CLASS_NAMES = ("helmet_loose", "helmet_on_head", "person", "vest_loose", "vest_on_body")
OPTIMIZER_SOURCES = (
    FRAMEWORK_LOG_LINE,
    DECLARED_EXPLICITLY,
    REDERIVED_CORROBORATED,
    REDERIVED_UNCORROBORATED,
)


@pytest.fixture(scope="module")
def paths() -> ProjectPaths:
    return ProjectPaths.from_root()


@pytest.fixture(scope="module")
def manifest(paths: ProjectPaths) -> dict:
    path = paths.reports / "detection_D1_manifest.json"
    if not path.is_file():
        pytest.skip("detection_D1_manifest.json not present; run the D1 experiment")
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def report(paths: ProjectPaths) -> str:
    path = paths.reports / "detection_D1_report.md"
    if not path.is_file():
        pytest.skip("detection_D1_report.md not present")
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
def support(paths: ProjectPaths, matrix) -> tuple:
    split = json.loads((paths.reports / "split_manifest.json").read_text(encoding="utf-8"))
    return class_support(
        split, dict(zip(CLASS_NAMES, range(5), strict=True)), rule=matrix.support_rule
    )


# --- manifest validity --------------------------------------------------------


def test_manifest_is_valid(manifest):
    assert validate_result_manifest(manifest, class_names=CLASS_NAMES) == []


def test_manifest_declares_the_frozen_experiment(manifest):
    assert manifest["experiment_id"] == "D1"
    assert manifest["task"] == "detection"
    assert manifest["phase"] == "7B"
    assert manifest["model"] == "YOLO11s"
    assert manifest["status"] == EXPERIMENT_COMPLETE
    assert manifest["weights_committed"] is False


def test_the_intentional_variable_is_model_capacity(manifest):
    assert manifest["intentional_variable"] == "MODEL_CAPACITY"
    assert manifest["intentional_fields"] == ["model"]
    assert manifest["consequential_fields"] == ["weight_identifier"]
    assert manifest["overrides"] == {"model": "YOLO11s", "weight_identifier": "yolo11s.pt"}
    assert manifest["inherits"] == "D0"
    assert manifest["reference_experiment"] == "D0"


def test_the_manifest_matches_the_frozen_matrix(manifest, paths, matrix):
    assert manifest["experiment_matrix_sha256"] == sha256_bytes(
        paths.configs / "detection_experiments.yaml"
    )
    declaration = matrix.declaration("D1")
    assert manifest["overrides"] == declaration.overrides
    assert manifest["intentional_variable"] == declaration.intentional_variable


def test_the_manifest_matches_the_frozen_policy(manifest, paths):
    assert manifest["phase7_policy_sha256"] == sha256_bytes(
        paths.reports / "detection_comparison_policy.json"
    )


def test_the_manifest_binds_to_the_frozen_data(manifest, paths, d0_manifest):
    split = json.loads((paths.reports / "split_manifest.json").read_text(encoding="utf-8"))
    assert (
        manifest["split_reference"]["split_assignment_sha256"] == (split["split_assignment_sha256"])
    )
    # The comparison is only meaningful if both runs saw the same data.
    assert manifest["adapter_manifest_sha256"] == d0_manifest["adapter_manifest_sha256"]
    assert manifest["dataset_fingerprints"] == d0_manifest["dataset_fingerprints"]
    assert manifest["class_map"] == d0_manifest["class_map"]


# --- one-variable contract ----------------------------------------------------


def test_the_protocol_compatibility_verdict_is_recorded_and_passing(manifest):
    verdict = manifest["protocol_compatibility"]
    assert verdict["compatible"] is True
    assert verdict["observed_differences"] == ["model", "weight_identifier"]
    assert verdict["undeclared_differences"] == []
    assert verdict["unapplied_declarations"] == []
    assert verdict["reference_experiment"] == "D0"


def test_every_frozen_field_matches_the_reference_protocol(manifest, paths):
    from construction_safety_vision.experiment import load_detection_baseline_config

    baseline = load_detection_baseline_config(paths.configs / "detection_baseline.yaml")
    resolved = manifest["resolved_protocol"]
    assert resolved["training.imgsz"] == baseline.training["imgsz"] == 640
    assert resolved["training.batch"] == baseline.training["batch"] == 16
    assert resolved["training.epochs"] == baseline.training["epochs"] == 100
    assert resolved["training.patience"] == baseline.training["patience"]
    assert resolved["training.optimizer"] == baseline.training["optimizer"] == "auto"
    assert resolved["training.augmentation"] == baseline.training["augmentation"]
    assert resolved["training.deterministic"] is baseline.training["deterministic"] is True
    assert resolved["training.amp"] is baseline.training["amp"] is True
    assert resolved["seed"] == baseline.seed == 42
    assert resolved["dataset_config"] == baseline.dataset_config
    assert resolved["checkpoint_selection"] == baseline.checkpoint_selection
    # And the two declared differences really did change.
    assert resolved["model"] == "YOLO11s" != baseline.model
    assert resolved["weight_identifier"] == "yolo11s.pt" != baseline.weight_identifier


def test_the_effective_run_used_the_frozen_batch_and_image_size(manifest):
    resolved = manifest["resolved_training_arguments"]
    assert resolved["imgsz"] == 640
    assert resolved["batch"] == 16
    assert resolved["seed"] == 42
    assert resolved["deterministic"] is True
    assert resolved["amp"] is True
    assert resolved["epochs"] == 100


# --- optimizer evidence -------------------------------------------------------


def test_the_actual_optimizer_evidence_is_recorded(manifest):
    assert manifest["declared_optimizer_policy"] == "auto"
    determination = manifest["resolved_optimizer_determination"]
    assert determination["source"] in OPTIMIZER_SOURCES
    assert manifest["resolved_optimizer"] == determination["optimizer"]
    assert manifest["resolved_optimizer"] not in (None, "", "NOT_EXPOSED_RELIABLY")
    assert "inferred" in determination
    assert isinstance(determination["effective_lr0"], (int, float))


def test_a_directly_captured_optimizer_carries_its_log_evidence(manifest):
    determination = manifest["resolved_optimizer_determination"]
    if determination["source"] != FRAMEWORK_LOG_LINE:
        pytest.skip("this run established its optimizer by inference")
    assert determination["inferred"] is False
    assert "optimizer:" in determination["evidence"]
    assert "\x1b" not in determination["evidence"]
    assert determination["effective_momentum"] is not None


def test_an_inferred_optimizer_is_labelled_as_inferred(manifest):
    determination = manifest["resolved_optimizer_determination"]
    if determination["source"] == FRAMEWORK_LOG_LINE:
        pytest.skip("this run captured its optimizer directly")
    assert determination["inferred"] is True
    assert "inference, not a reading" in determination["note"]


# --- checkpoints --------------------------------------------------------------


def test_checkpoint_hashes_are_recorded_and_not_committed(manifest):
    for key in ("best_checkpoint", "last_checkpoint"):
        record = manifest[key]
        assert len(record["sha256"]) == 64
        assert record["size_bytes"] > 0
        assert record["committed"] is False
        assert record["relative_path"].startswith("artifacts/detection/D1/weights/")
    assert manifest["best_checkpoint"]["sha256"] != manifest["last_checkpoint"]["sha256"] or (
        manifest["best_epoch"] == manifest["epochs_completed"]
    )


def test_the_checkpoint_rule_is_the_frozen_one(manifest, d0_manifest):
    assert manifest["checkpoint_selection"] == d0_manifest["checkpoint_selection"]
    assert "ULTRALYTICS_BEST_ON_VALIDATION_FITNESS" in manifest["checkpoint_selection"]
    assert isinstance(manifest["best_epoch"], int)
    assert 1 <= manifest["best_epoch"] <= manifest["epochs_completed"]


def test_the_headline_metric_belongs_to_the_selected_checkpoint(manifest):
    cross = manifest["headline_metric_cross_check"]
    if not isinstance(cross["delta"], (int, float)):
        pytest.skip("the epoch history did not expose a comparable metric")
    assert cross["delta"] <= cross["tolerance"]


def test_exactly_one_run_and_no_diverted_directory(paths, manifest):
    root = paths.root / "artifacts" / "detection"
    if not root.is_dir():
        pytest.skip("run directories are git-ignored and not present")
    diverted = [path.name for path in root.iterdir() if path.name.startswith("D1-")]
    assert diverted == []


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


def test_the_rare_class_remains_descriptive(manifest):
    comparison = manifest["phase7_comparison"]
    assert comparison["descriptive_classes"] == ["vest_loose"]
    assert "vest_loose" not in comparison["selection_metric_classes"]
    rare = manifest["rare_class"]
    assert rare["classification"] == DESCRIPTIVE_HIGH_UNCERTAINTY
    assert rare["classes"] == ["vest_loose"]


def test_the_rare_class_is_still_reported_in_full(manifest):
    table = manifest["per_class_metrics"]["vest_loose"]
    for metric in PER_CLASS_METRICS:
        assert metric in table


def test_every_class_is_reported(manifest):
    assert set(manifest["per_class_metrics"]) == set(CLASS_NAMES)


def test_the_supported_macro_is_the_mean_of_the_supported_per_class_values(manifest):
    comparison = manifest["phase7_comparison"]
    recomputed = supported_macro(
        manifest["per_class_metrics"], comparison["selection_metric_classes"]
    )
    assert round(float(recomputed), 6) == comparison["supported_macro_map50_95"]
    assert Decimal(comparison["supported_macro_exact"]) == recomputed
    # And it genuinely ignores the descriptive class.
    assert "vest_loose" not in comparison["selection_metric_classes"]


def test_the_delta_is_the_difference_of_the_two_macros(manifest):
    comparison = manifest["phase7_comparison"]
    expected = round(
        comparison["supported_macro_map50_95"] - comparison["reference_supported_macro_map50_95"],
        6,
    )
    assert comparison["delta_supported_macro_vs_reference"] == pytest.approx(expected, abs=1e-6)


def test_the_reference_macro_is_the_frozen_d0_value(manifest):
    assert manifest["phase7_comparison"]["reference_supported_macro_map50_95"] == 0.570142


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


def test_the_all_class_metric_is_preserved_alongside_the_selection_metric(manifest):
    comparison = manifest["phase7_comparison"]
    assert comparison["official_all_class_metric"] == OFFICIAL_ALL_CLASS_METRIC
    assert OFFICIAL_ALL_CLASS_METRIC in manifest["validation_metrics"]
    assert OFFICIAL_ALL_CLASS_METRIC in comparison["deltas_vs_reference"]
    for name in ("mAP@0.50", "precision", "recall"):
        assert name in manifest["validation_metrics"]
        assert name in comparison["deltas_vs_reference"]


def test_the_all_class_delta_matches_the_two_manifests(manifest, d0_manifest):
    deltas = manifest["phase7_comparison"]["deltas_vs_reference"]
    for name in (OFFICIAL_ALL_CLASS_METRIC, "mAP@0.50", "precision", "recall"):
        expected = round(
            manifest["validation_metrics"][name] - d0_manifest["validation_metrics"][name], 6
        )
        assert deltas[name] == pytest.approx(expected, abs=1e-6)


# --- no winner yet ------------------------------------------------------------


def test_the_manifest_declares_selection_still_pending(manifest):
    comparison = manifest["phase7_comparison"]
    assert comparison["selection_pending"] is True
    assert "D2 has not been executed" in comparison["selection_pending_reason"]


def test_no_final_winner_is_declared_anywhere(manifest, report):
    # The live results artifact is not asserted here. Phase 7D owns its
    # selection state and moved it to FINAL_SELECTED; what these historical D1
    # artifacts must keep saying is that *this experiment phase* declared no
    # winner, which is a property of the manifest and the report alone.
    #
    # Affirmative selection phrasing only. "no final detector is declared" is a
    # legitimate sentence, so matching on "final detector is" would flag the very
    # disclaimer the test wants to see.
    lowered = report.lower()
    for phrase in ("phase 7 winner is", "the final detector is d1", "d1 is the winner"):
        assert phrase not in lowered
    assert manifest["phase7_comparison"]["selection_pending"] is True


def test_the_results_artifact_covers_every_declared_experiment(results):
    rows = {row["experiment_id"]: row for row in results["experiments"]}
    assert set(rows) == {"D0", "D1", "D2"}
    assert rows["D0"]["status"] == "COMPLETE"
    assert rows["D1"]["status"] == "COMPLETE"
    assert rows["D0"]["margin_status"] == "REFERENCE"
    assert rows["D0"]["metrics"][PRIMARY_SELECTION_METRIC] == 0.570142


def test_no_placeholder_metric_was_invented_for_a_pending_experiment(results, paths):
    # Whichever candidates are still outstanding, they carry no numbers and no
    # artifacts. When the matrix is complete this asserts nothing, by design.
    for row in results["experiments"]:
        if row["status"] == "FROZEN_NOT_EXECUTED":
            assert row["metrics"] is None
            assert row["per_class_metrics"] is None
            assert row["experiment_sha256"] is None
            assert row["experiment_id"] in results["pending_experiments"]
    for experiment_id in results["pending_experiments"]:
        assert not (paths.reports / f"detection_{experiment_id}_manifest.json").exists()
        assert not (paths.root / "artifacts" / "detection" / experiment_id).exists()


# --- the experiment fingerprint ----------------------------------------------


def test_the_experiment_fingerprint_is_recorded(manifest):
    assert len(manifest["experiment_sha256"]) == 64
    assert manifest["experiment_sha256"] != manifest["best_checkpoint"]["sha256"]


def test_the_experiment_fingerprint_differs_from_the_reference(manifest, d0_manifest):
    assert manifest["experiment_sha256"] != d0_manifest["d0_experiment_sha256"]


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


def test_the_pretrained_weights_are_fingerprinted_and_not_committed(manifest, d0_manifest):
    weights = manifest["pretrained_weights"]
    assert weights["identifier"] == "yolo11s.pt"
    assert len(weights["sha256"]) == 64
    assert weights["size_bytes"] > 0
    assert weights["committed"] is False
    # A different model family must start from different bytes.
    assert weights["sha256"] != d0_manifest["pretrained_weights"]["sha256"]


def test_both_checkpoints_were_obtained_the_same_way(paths, manifest):
    # D0's manifest does not carry the acquisition mechanism; phase 6A recorded
    # it in the runtime provenance instead. Read it from there rather than
    # asserting against a key D0 never had, so the comparison is real.
    runtime = json.loads(
        (paths.reports / "detection_runtime.provenance.json").read_text(encoding="utf-8")
    )
    reference_weights = runtime["details"]["pretrained_weights"]
    assert reference_weights["identifier"] == "yolo11n.pt"
    assert (
        manifest["pretrained_weights"]["source_mechanism"]
        == (reference_weights["source_mechanism"])
    )
    assert (
        manifest["pretrained_weights"]["ultralytics_version"]
        == (reference_weights["ultralytics_version"])
    )


# --- holdout protection -------------------------------------------------------


def test_the_holdout_was_not_accessed(manifest):
    assert manifest["test"]["status"] == TEST_PROTECTED
    assert set(manifest["test"]) == {"status", "reason"}


def test_no_holdout_section_exists_outside_the_status_entry(manifest):
    from construction_safety_vision.detection_results import contains_forbidden_split

    without_test = {key: value for key, value in manifest.items() if key != "test"}
    assert not contains_forbidden_split(without_test)


def test_no_holdout_artifact_was_produced(paths):
    # Before phase 11B the holdout has no on-disk presence at all. Phase 11B
    # materialises it, through the phase 5D function and under both
    # authorisation gates, so from then on these paths legitimately exist.
    evaluated = holdout_has_been_evaluated(paths)
    assert evaluated or not (paths.data_processed / "canonical" / "images" / "test").exists()
    # No phase has ever built a holdout YOLO adapter, phase 11B included: it
    # scores against canonical COCO and never writes a derived label view.
    adapters = paths.data_processed / "adapters" / "yolo_detection"
    for kind in ("images", "labels"):
        assert not (adapters / kind / "test").exists()
    annotations = paths.data_processed / "canonical" / "annotations"
    if annotations.is_dir() and not evaluated:
        assert not list(annotations.glob("*test*"))


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
        assert relative.startswith("reports/figures/detection/D1/")
        assert relative.endswith(".png")
        name = relative.rsplit("/", 1)[-1]
        assert "batch" not in name
        assert "labels" not in name
    directory = paths.figures / "detection" / "D1"
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
        "PENDING_EXPERIMENT",
    ):
        assert label in report


def test_the_report_shows_the_descriptive_class_and_both_metrics(report, manifest):
    assert "| `vest_loose` |" in report
    assert str(manifest["phase7_comparison"]["supported_macro_map50_95"]) in report
    assert str(manifest["validation_metrics"][OFFICIAL_ALL_CLASS_METRIC]) in report


def test_the_report_states_that_selection_is_pending(report):
    assert "Final Phase 7 comparison still pending" in report
    assert "UNSELECTED_PENDING_REVIEW" in report or "not executed" in report


def test_the_provenance_records_one_trained_model(paths):
    path = paths.reports / "detection_D1.provenance.json"
    if not path.is_file():
        pytest.skip("detection_D1.provenance.json not present")
    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["phase"] == 7
    details = record["details"]
    assert details["models_trained_in_this_phase"] == 1
    assert details["holdout_accessed"] is False
    assert details["selection_pending"] is True


def test_a_prerun_record_exists_and_precedes_the_result(paths, manifest):
    path = paths.reports / "detection_D1_prerun.provenance.json"
    if not path.is_file():
        pytest.skip("detection_D1_prerun.provenance.json not present")
    prerun = json.loads(path.read_text(encoding="utf-8"))
    assert prerun["details"]["stage"] == "PRE_RUN"
    assert prerun["details"]["experiment_id"] == "D1"
    # The pre-run record must name the same protocol the result was produced
    # under, or it is not evidence that the protocol preceded the result.
    assert prerun["config"]["resolved_config_sha256"] == manifest["resolved_config_sha256"]
    assert prerun["config"]["phase7_policy_sha256"] == manifest["phase7_policy_sha256"]
    assert (
        prerun["details"]["pretrained_weights"]["sha256"]
        == (manifest["pretrained_weights"]["sha256"])
    )
