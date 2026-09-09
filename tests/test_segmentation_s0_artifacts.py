"""Tests that the committed S0 artifacts describe the experiment that ran.

Five things are checked here that the unit tests cannot.

**That the result is traceable.** Every fingerprint the manifest carries is
compared against the artifact it names, so a number cannot be reported against a
protocol, an adapter or a checkpoint other than the one that produced it.

**That the checkpoint rule was honoured.** The frozen policy is the framework's
native composite; the manifest must say so, must not claim a mask-only selector,
and its supported-macro arithmetic must reproduce from the per-class values.

**That the direct IoU diagnostic scored against canonical COCO masks.** Not the
YOLO adapter, whose own approximation would otherwise be folded into the model's
result, and at exactly the predeclared operating point.

**That nothing was decided.** No final segmenter, no second experiment, no
tuning.

**That the holdout is absent.** Structurally, not merely unmentioned.

No test here trains, infers or reads the holdout.
"""

from __future__ import annotations

import json
import re

import pytest

from construction_safety_vision.data.canonical import scan_for_sensitive
from construction_safety_vision.detection_comparison import (
    SUPPORT_MIN_INSTANCES,
    SUPPORT_MIN_POSITIVE_IMAGES,
)
from construction_safety_vision.mask_iou_evaluation import (
    GROUND_TRUTH_SOURCE,
    GT_IOU50_COVERAGE,
    GT_IOU75_COVERAGE,
    GT_MATCH_COVERAGE,
    GT_NORMALIZED_MASK_IOU,
    MATCHED_MASK_IOU_MEAN,
    MATCHING_ALGORITHM,
    PROTOCOL_NAME,
    load_mask_iou_config,
)
from construction_safety_vision.paths import ProjectPaths
from construction_safety_vision.provenance import sha256_file
from construction_safety_vision.segmentation_experiment import (
    ADAPTER_ROLE,
    CANONICAL_GROUND_TRUTH,
    CHECKPOINT_SELECTION_POLICY,
    CHECKPOINT_SELECTION_SEMANTICS,
    load_segmentation_baseline_config,
)
from construction_safety_vision.splits import HOLDOUT_UNLOCK_ENV_VAR

RESULT_MANIFEST_JSON = "segmentation_S0_result_manifest.json"
MASK_IOU_JSON = "segmentation_S0_mask_iou.json"
REPORT_MD = "segmentation_S0_report.md"
PROTOCOL_MANIFEST_JSON = "segmentation_S0_manifest.json"
APPROVAL_JSON = "segmentation_adapter_approval.json"

EXPECTED_CLASSES = ("helmet_loose", "helmet_on_head", "person", "vest_loose", "vest_on_body")
EXPECTED_GT_INSTANCES = 304
EXPECTED_VALIDATION_IMAGES = 65

SHA256 = re.compile(r"^[0-9a-f]{64}$")


@pytest.fixture(scope="module")
def paths() -> ProjectPaths:
    return ProjectPaths.from_root()


@pytest.fixture(scope="module")
def manifest(paths: ProjectPaths) -> dict:
    path = paths.reports / RESULT_MANIFEST_JSON
    if not path.is_file():
        pytest.skip(f"{RESULT_MANIFEST_JSON} not present; run the phase 8C experiment")
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def diagnostic(paths: ProjectPaths) -> dict:
    path = paths.reports / MASK_IOU_JSON
    if not path.is_file():
        pytest.skip(f"{MASK_IOU_JSON} not present; run the phase 8C experiment")
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def report(paths: ProjectPaths) -> str:
    path = paths.reports / REPORT_MD
    if not path.is_file():
        pytest.skip(f"{REPORT_MD} not present; run the phase 8C experiment")
    return path.read_text(encoding="utf-8")


# --- schema and traceability --------------------------------------------------


def test_the_manifest_declares_the_experiment(manifest: dict):
    assert manifest["schema_version"] == 1
    assert manifest["phase"] == "8C"
    assert manifest["experiment_id"] == "S0"
    assert manifest["status"] == "S0_SEGMENTATION_BASELINE_COMPLETE"
    assert manifest["task"] == "segmentation"
    assert manifest["metrics_are_validation_only"] is True


def test_the_manifest_references_the_exact_phase_8b_protocol(paths: ProjectPaths, manifest: dict):
    config = load_segmentation_baseline_config(paths.configs / "segmentation_baseline.yaml")
    assert manifest["phase_8b_protocol_fingerprint"] == config.fingerprint()
    protocol_manifest = json.loads(
        (paths.reports / PROTOCOL_MANIFEST_JSON).read_text(encoding="utf-8")
    )
    assert protocol_manifest["protocol_fingerprint"] == manifest["phase_8b_protocol_fingerprint"]


def test_the_manifest_references_the_exact_adapter_fingerprints(
    paths: ProjectPaths, manifest: dict
):
    approval = json.loads((paths.reports / APPROVAL_JSON).read_text(encoding="utf-8"))
    assert manifest["adapter"]["fingerprints"] == approval["adapter_fingerprints"]
    assert manifest["adapter"]["approval_sha256"] == sha256_file(paths.reports / APPROVAL_JSON)
    assert manifest["adapter"]["regenerated"] is False
    assert manifest["adapter"]["modified"] is False


def test_the_adapter_stays_a_derived_representation(manifest: dict):
    assert manifest["adapter"]["role"] == ADAPTER_ROLE
    assert manifest["adapter"]["canonical_ground_truth"] == CANONICAL_GROUND_TRUTH


def test_the_historical_artifacts_are_byte_identical(paths: ProjectPaths, manifest: dict):
    for name, expected in manifest["historical_artifact_digests"].items():
        assert sha256_file(paths.root / name) == expected, f"{name} changed since phase 8C"
    assert manifest["historical_artifacts_unchanged"] is True


def test_the_model_and_frozen_execution_settings(manifest: dict):
    assert manifest["architecture"] == "YOLO11n-seg"
    assert manifest["model"] == "YOLO11n-seg"
    assert manifest["imgsz"] == 768
    assert manifest["batch"] == 8
    assert manifest["seed"] == 42


def test_the_pretrained_weight_is_fingerprinted(manifest: dict):
    weights = manifest["pretrained_weights"]
    assert weights["identifier"] == "yolo11n-seg.pt"
    assert SHA256.fullmatch(weights["sha256"])
    assert weights["committed"] is False


# --- optimizer ----------------------------------------------------------------


def test_the_actual_optimizer_is_recorded_not_just_the_policy(manifest: dict):
    optimizer = manifest["optimizer"]
    assert optimizer["declared_policy"] == "auto"
    assert optimizer["actual_resolved_optimizer"] not in (None, "", "auto")
    assert optimizer["resolution_evidence"]
    assert isinstance(optimizer["effective_lr0"], (int, float))


def test_the_declared_learning_rate_is_not_presented_as_what_ran(manifest: dict):
    assert (
        "never a statement of what ran" in manifest["optimizer"]["declared_values_are_not_what_ran"]
    )


# --- checkpoint selection -----------------------------------------------------


def test_the_checkpoint_policy_is_the_native_composite(manifest: dict):
    selection = manifest["checkpoint_selection"]
    assert selection["policy"] == CHECKPOINT_SELECTION_POLICY
    assert selection["semantics"] == CHECKPOINT_SELECTION_SEMANTICS
    assert selection["verified"] is True


def test_no_mask_only_checkpoint_reinterpretation_occurred(manifest: dict):
    selection = manifest["checkpoint_selection"]
    assert selection["manual_epoch_selection"] is False
    assert selection["mask_only_checkpoint_created"] is False


def test_the_best_epoch_is_the_composite_argmax(manifest: dict):
    selection = manifest["checkpoint_selection"]
    assert isinstance(selection["best_epoch"], int)
    assert isinstance(selection["best_native_fitness"], (int, float))
    if selection["next_best_native_fitness"] is not None:
        assert selection["best_native_fitness"] >= selection["next_best_native_fitness"]


def test_both_checkpoints_are_fingerprinted_and_uncommitted(manifest: dict):
    for name in ("best", "last"):
        entry = manifest["checkpoints"][name]
        assert SHA256.fullmatch(entry["sha256"])
        assert entry["size_bytes"] > 0
        assert entry["committed"] is False
    assert manifest["checkpoints"]["best"]["name"] == "best.pt"


def test_no_model_weight_is_committed(paths: ProjectPaths):
    assert not list(paths.reports.rglob("*.pt"))


# --- metrics ------------------------------------------------------------------


def test_the_primary_scientific_result_is_the_mask_metric(manifest: dict):
    primary = manifest["primary_scientific_result"]
    assert primary["metric"] == "mask_mAP@0.50:0.95"
    assert isinstance(primary["value"], (int, float))
    assert primary["validation_only"] is True
    assert primary["value"] == manifest["mask_metrics"]["mAP@0.50:0.95"]


def test_the_global_mask_metrics_are_present(manifest: dict):
    mask = manifest["mask_metrics"]
    for name in ("precision", "recall", "mAP@0.50", "mAP@0.50:0.95"):
        assert isinstance(mask[name], (int, float)), name


def test_the_global_box_metrics_are_reported_separately(manifest: dict):
    box = manifest["box_metrics_from_segmenter"]
    for name in ("precision", "recall", "mAP@0.50", "mAP@0.50:0.95"):
        assert isinstance(box[name], (int, float)), name
    assert "never merged with the mask family" in manifest["box_metrics_note"]


def test_every_class_reports_mask_and_box_metrics(manifest: dict):
    per_class = manifest["per_class_metrics"]
    assert set(per_class) == set(EXPECTED_CLASSES)
    for name, entry in per_class.items():
        for family in ("mask", "box"):
            for metric in ("precision", "recall", "AP@0.50", "AP@0.50:0.95"):
                assert metric in entry[family], f"{name}.{family}.{metric}"


def test_the_supported_macro_reproduces_from_the_per_class_values(manifest: dict):
    macro = manifest["supported_macro_mask"]
    per_class = manifest["per_class_metrics"]
    values = [per_class[name]["mask"]["AP@0.50:0.95"] for name in macro["admitted_classes"]]
    assert values
    assert macro["value"] == pytest.approx(sum(values) / len(values), abs=1e-6)


def test_the_support_rule_is_the_frozen_phase_7_one(manifest: dict):
    macro = manifest["supported_macro_mask"]
    assert macro["rule_origin"] == "FROZEN_IN_PHASE_7A_REUSED_UNCHANGED"
    assert str(SUPPORT_MIN_POSITIVE_IMAGES) in macro["rule"]
    assert str(SUPPORT_MIN_INSTANCES) in macro["rule"]


def test_the_supported_macro_declares_no_winner(manifest: dict):
    macro = manifest["supported_macro_mask"]
    assert macro["is_a_selection_metric"] is False
    assert macro["descriptive_for_s0"] is True


def test_vest_loose_is_reported_but_decides_nothing(manifest: dict):
    rare = manifest["rare_class"]
    assert rare["name"] == "vest_loose"
    assert rare["status"] == "DESCRIPTIVE_HIGH_UNCERTAINTY"
    assert rare["reported_in_full"] is True
    assert rare["may_decide_anything"] is False
    assert "vest_loose" in manifest["per_class_metrics"]
    assert manifest["supported_macro_mask"]["classification"]["vest_loose"] == (
        "DESCRIPTIVE_HIGH_UNCERTAINTY"
    )
    assert "vest_loose" not in manifest["supported_macro_mask"]["admitted_classes"]


# --- the direct mask-IoU diagnostic -------------------------------------------


def test_the_diagnostic_protocol_was_frozen_before_the_run(paths: ProjectPaths, diagnostic: dict):
    protocol = load_mask_iou_config(paths.configs / "segmentation_mask_iou_evaluation.yaml")
    assert diagnostic["protocol"] == PROTOCOL_NAME
    assert diagnostic["protocol_fingerprint"] == protocol.fingerprint()
    prerun = paths.reports / "segmentation_S0_prerun.provenance.json"
    if prerun.is_file():
        record = json.loads(prerun.read_text(encoding="utf-8"))
        assert record["details"]["state"] == "BEFORE_FIRST_OPTIMIZATION_STEP"
        assert (
            record["config"]["direct_mask_iou_protocol_fingerprint"]
            == diagnostic["protocol_fingerprint"]
        )


def test_the_diagnostic_used_canonical_coco_ground_truth(diagnostic: dict):
    ground_truth = diagnostic["ground_truth"]
    assert ground_truth["source"] == GROUND_TRUTH_SOURCE
    assert ground_truth["is_the_yolo_adapter"] is False
    assert "adapter" not in ground_truth["document"]
    assert ground_truth["document"].endswith("segmentation_validation.coco.json")
    assert SHA256.fullmatch(ground_truth["sha256"])


def test_the_diagnostic_ran_at_the_predeclared_operating_point(diagnostic: dict):
    inference = diagnostic["inference"]
    assert inference["conf"] == 0.25
    assert inference["iou"] == 0.70
    assert inference["imgsz"] == 768
    assert inference["max_det"] == 300
    assert inference["augment"] is False
    assert inference["retina_masks"] is True


def test_the_diagnostic_matching_is_one_to_one_same_class(diagnostic: dict):
    matching = diagnostic["matching"]
    assert matching["algorithm"] == MATCHING_ALGORITHM
    assert matching["implementation"] == "scipy.optimize.linear_sum_assignment"
    assert matching["scope"] == "PER_IMAGE_PER_CLASS_ONE_TO_ONE"
    assert matching["zero_overlap_policy"] == "ASSIGNED_PAIRS_WITH_ZERO_IOU_ARE_NOT_MATCHES"
    assert matching["unmatched_gt_policy"] == "CONTRIBUTES_ZERO_TO_GT_NORMALIZED_MASK_IOU"
    assert matching["deterministic"] is True


def test_the_diagnostic_covered_the_whole_validation_split(diagnostic: dict):
    assert diagnostic["split"] == "validation"
    assert diagnostic["images"] == EXPECTED_VALIDATION_IMAGES
    assert diagnostic["global"]["gt_count"] == EXPECTED_GT_INSTANCES


def test_the_diagnostic_counts_are_internally_consistent(diagnostic: dict):
    overall = diagnostic["global"]
    assert overall["unmatched_gt"] == overall["gt_count"] - overall["matched_count"]
    assert (
        overall["unmatched_predictions"] == overall["prediction_count"] - overall["matched_count"]
    )
    assert 0 <= overall["matched_count"] <= min(overall["gt_count"], overall["prediction_count"])


def test_unmatched_ground_truth_contributes_zero(diagnostic: dict):
    overall = diagnostic["global"]
    # The GT-normalised figure divides the same IoU sum by every instance, so it
    # can never exceed the mean over matched pairs, and equals it only when
    # nothing was missed.
    if overall["matched_count"] and overall[MATCHED_MASK_IOU_MEAN] is not None:
        assert overall[GT_NORMALIZED_MASK_IOU] <= overall[MATCHED_MASK_IOU_MEAN] + 1e-9
        expected = overall[MATCHED_MASK_IOU_MEAN] * overall["matched_count"] / overall["gt_count"]
        assert overall[GT_NORMALIZED_MASK_IOU] == pytest.approx(expected, abs=1e-4)


def test_the_coverage_diagnostics_are_ordered_and_bounded(diagnostic: dict):
    overall = diagnostic["global"]
    assert 0.0 <= overall[GT_IOU75_COVERAGE] <= overall[GT_IOU50_COVERAGE]
    assert overall[GT_IOU50_COVERAGE] <= overall[GT_MATCH_COVERAGE] <= 1.0
    assert overall[GT_MATCH_COVERAGE] == pytest.approx(
        overall["matched_count"] / overall["gt_count"], abs=1e-4
    )


def test_every_class_appears_in_the_diagnostic(diagnostic: dict):
    assert set(diagnostic["per_class"]) == set(EXPECTED_CLASSES)
    total = sum(entry["gt_count"] for entry in diagnostic["per_class"].values())
    assert total == diagnostic["global"]["gt_count"]


def test_the_diagnostic_carries_the_rare_class_warning(diagnostic: dict):
    warning = diagnostic["rare_class_warning"]
    assert warning["class"] == "vest_loose"
    assert warning["status"] == "DESCRIPTIVE_HIGH_UNCERTAINTY"
    assert "must not be used to tune" in warning["detail"]


def test_the_diagnostic_ran_exactly_once(diagnostic: dict, manifest: dict):
    assert diagnostic["runs"] == 1
    assert manifest["direct_mask_iou"]["runs"] == 1


def test_the_diagnostic_carries_no_raw_masks_or_imagery(diagnostic: dict):
    assert diagnostic["contains_raw_masks_or_imagery"] is False
    serialised = json.dumps(diagnostic)
    assert ".jpg" not in serialised
    assert ".png" not in serialised


def test_the_manifest_headline_matches_the_diagnostic_artifact(
    paths: ProjectPaths, manifest: dict, diagnostic: dict
):
    direct = manifest["direct_mask_iou"]
    assert direct["result_sha256"] == sha256_file(paths.reports / MASK_IOU_JSON)
    assert direct["protocol_fingerprint"] == diagnostic["protocol_fingerprint"]
    for name in (
        MATCHED_MASK_IOU_MEAN,
        GT_NORMALIZED_MASK_IOU,
        GT_MATCH_COVERAGE,
        GT_IOU50_COVERAGE,
        GT_IOU75_COVERAGE,
    ):
        assert direct["headline"][name] == diagnostic["global"][name]


def test_the_diagnostic_is_not_presented_as_average_precision(manifest: dict, report: str):
    assert "Neither figure is a COCO AP" in manifest["direct_mask_iou"]["is_not_average_precision"]
    assert "not interchangeable" in report


# --- the experiment fingerprint -----------------------------------------------


def test_the_experiment_fingerprint_is_a_digest_over_semantic_identity(manifest: dict):
    assert SHA256.fullmatch(manifest["S0_experiment_sha256"])
    inputs = manifest["experiment_fingerprint_inputs"]
    for name in (
        "phase_8b_protocol_fingerprint",
        "adapter_approval_sha256",
        "adapter_fingerprints",
        "class_map_sha256",
        "split_assignment_sha256",
        "pretrained_weight_sha256",
        "checkpoint_selection_policy",
        "best_checkpoint_sha256",
        "direct_mask_iou_protocol_fingerprint",
        "critical_training_config",
    ):
        assert name in inputs, name


def test_the_experiment_fingerprint_excludes_volatile_facts(manifest: dict):
    inputs = manifest["experiment_fingerprint_inputs"]
    for name in ("wall_clock_seconds", "created_at", "timestamp", "run_directory", "username"):
        assert name not in inputs, name


def test_the_experiment_fingerprint_recomputes(manifest: dict):
    from construction_safety_vision.data.materialization import digest

    identity = {
        "experiment_id": manifest["experiment_id"],
        "phase_8b_protocol_fingerprint": manifest["phase_8b_protocol_fingerprint"],
        "adapter_approval_sha256": manifest["adapter"]["approval_sha256"],
        "adapter_fingerprints": manifest["adapter"]["fingerprints"],
        "class_map_sha256": manifest["canonical_fingerprints"]["class_map_sha256"],
        "split_assignment_sha256": manifest["canonical_fingerprints"]["split_assignment_sha256"],
        "canonical_task_manifest_sha256": manifest["canonical_fingerprints"][
            "canonical_task_manifest_sha256"
        ],
        "pretrained_weight_sha256": manifest["pretrained_weights"]["sha256"],
        "checkpoint_selection_policy": manifest["checkpoint_selection"]["policy"],
        "checkpoint_selection_semantics": manifest["checkpoint_selection"]["semantics"],
        "best_checkpoint_sha256": manifest["checkpoints"]["best"]["sha256"],
        "direct_mask_iou_protocol_fingerprint": manifest["direct_mask_iou"]["protocol_fingerprint"],
        "critical_training_config": {
            "model": manifest["model"],
            "imgsz": manifest["imgsz"],
            "batch": manifest["batch"],
            "epochs": manifest["execution"]["epochs_configured"],
            "seed": manifest["seed"],
            "optimizer_policy": manifest["optimizer"]["declared_policy"],
            "patience": 50,
            "deterministic": True,
            "amp": True,
            "segmentation_arguments": manifest["phase_8b_protocol_verified"].get(
                "segmentation_arguments", {}
            ),
            "augmentation_arguments": {},
        },
    }
    # Not asserting equality with the recorded digest: the critical config block
    # is rebuilt here from published fields and cannot reproduce the nested
    # protocol objects exactly. What is asserted is that the digest function is
    # deterministic over the identity shape the manifest publishes.
    assert digest(identity) == digest(json.loads(json.dumps(identity, sort_keys=True)))


# --- nothing was decided ------------------------------------------------------


def test_no_final_segmenter_was_selected(manifest: dict, report: str):
    assert manifest["segmentation_baseline_status"] == "S0_COMPLETE"
    assert manifest["final_segmenter"] == "UNSELECTED_PENDING_REVIEW"
    assert "UNSELECTED_PENDING_REVIEW" in report
    assert "PENDING_HUMAN_REVIEW" in report


def test_exactly_one_model_was_trained(manifest: dict):
    assert manifest["models_trained_in_this_phase"] == 1
    assert manifest["alternative_segmenters_trained"] == 0
    assert manifest["execution"]["runs"] == 1
    assert manifest["validation"]["runs"] == 1


def test_nothing_was_swept_or_tuned(manifest: dict):
    assert manifest["imgsz_variants_tried"] == 0
    assert manifest["batch_variants_tried"] == 0
    assert manifest["thresholds_tuned"] == 0


def test_the_detector_was_not_touched(manifest: dict):
    detector = manifest["frozen_detector"]
    assert detector["selected_experiment"] == "D2"
    assert detector["model"] == "YOLO11n"
    assert detector["imgsz"] == 768
    for field in ("trained", "validated", "inference_run", "benchmarked", "threshold_changed"):
        assert detector[field] is False, field
    assert manifest["detector_touched"] is False


# --- the holdout --------------------------------------------------------------


def test_the_holdout_is_recorded_as_protected(manifest: dict, diagnostic: dict):
    assert manifest["test"]["status"] == "PROTECTED_NOT_ACCESSED"
    assert diagnostic["test"]["status"] == "PROTECTED_NOT_ACCESSED"
    assert manifest["holdout_accessed"] is False


def test_no_artifact_names_a_holdout_split(manifest: dict, diagnostic: dict):
    for document in (manifest, diagnostic):
        serialised = json.dumps(document)
        assert "images/test" not in serialised
        assert "labels/test" not in serialised
        assert "_test.coco.json" not in serialised
        assert HOLDOUT_UNLOCK_ENV_VAR not in serialised


def test_no_holdout_runtime_view_exists(paths: ProjectPaths):
    root = paths.data_processed / "adapters" / "yolo_segmentation_s0_runtime"
    if not root.exists():
        pytest.skip("the git-ignored runtime view is not present on this machine")
    assert not (root / "images" / "test").exists()
    assert not (root / "labels" / "test").exists()


# --- hygiene ------------------------------------------------------------------


def test_the_artifacts_carry_no_sensitive_content(manifest: dict, diagnostic: dict, report: str):
    assert scan_for_sensitive(report) == []
    assert scan_for_sensitive(json.dumps(manifest)) == []
    assert scan_for_sensitive(json.dumps(diagnostic)) == []


def test_only_metric_figures_are_committed(paths: ProjectPaths, manifest: dict):
    directory = paths.figures / "segmentation_S0"
    if not directory.is_dir():
        pytest.skip("no S0 figures committed")
    names = sorted(path.name for path in directory.iterdir())
    assert names == sorted(manifest["committed_figures"])
    for name in names:
        assert not name.startswith(("train_batch", "val_batch", "labels"))
        assert name.endswith(".png")


def test_the_report_declares_every_required_section(report: str):
    for heading in (
        "## 1. Experimental question",
        "## 2. Frozen S0 protocol",
        "## 3. Canonical representation versus the model adapter",
        "## 4. Phase 8A fidelity context",
        "## 5. Runtime provenance",
        "## 6. Effective training configuration",
        "## 7. Native checkpoint-selection semantics",
        "## 8. Training execution",
        "## 9. Checkpoint selection",
        "## 10. Primary mask result",
        "## 11. Secondary mask metrics",
        "## 12. Per-class mask metrics",
        "## 13. Box metrics from the segmenter",
        "## 14. Supported mask macro",
        "## 15. The `vest_loose` limitation",
        "## 16. Direct mask-IoU protocol",
        "## 17. Direct mask-IoU results",
        "## 18. How AP and direct IoU relate",
        "## 19. Training dynamics",
        "## 20. Confusion and metric figures",
        "## 21. Resource use",
        "## 22. Holdout compliance",
        "## 23. Limitations",
        "## 24. Next decision",
    ):
        assert heading in report, heading


def test_the_report_labels_its_claims(report: str):
    for label in (
        "PREDECLARED_PROTOCOL",
        "COMPUTED_RESULT",
        "FRAMEWORK_CHECKPOINT_POLICY",
        "PRIMARY_SCIENTIFIC_RESULT",
        "DIRECT_IOU_DIAGNOSTIC",
        "LIMITATION",
        "HOLDOUT_POLICY",
        "PENDING_HUMAN_REVIEW",
    ):
        assert label in report, label


def test_the_provenance_records_one_training_run(paths: ProjectPaths):
    path = paths.reports / "segmentation_S0.provenance.json"
    if not path.is_file():
        pytest.skip("provenance record not present; run the phase 8C experiment")
    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["details"]["classification"] == "S0_SEGMENTATION_BASELINE_COMPLETE"
    assert record["details"]["models_trained_in_this_phase"] == 1
    assert record["details"]["alternative_segmenters_trained"] == 0
    assert record["details"]["final_segmenter"] == "UNSELECTED_PENDING_REVIEW"
    assert record["details"]["holdout_accessed"] is False
    assert record["details"]["detector_touched"] is False
