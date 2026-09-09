"""Tests that the committed phase 8B artifacts say what phase 8B was allowed to say.

Four things are checked here that the schema tests cannot.

**That phase 8A did not move.** The approval references the audit by digest, and
the digests it records are compared against the files on disk. If a later session
"clarified" the fidelity report, this fails.

**That the approval attaches to the audited bytes.** Every fingerprint in the
approval must equal the one phase 8A recorded - not merely be a digest, and not
merely be self-consistent.

**That nothing was decided that phase 8B was not authorised to decide.** No S0
result, no filtered instance population, no promoted adapter, no retuned
detector.

**That the holdout is absent.** Not merely unmentioned: structurally absent from
the artifacts, the adapter and the descriptor.

No test here trains a model, and none reads the holdout.
"""

from __future__ import annotations

import json
import re

import pytest

from construction_safety_vision.data.canonical import scan_for_sensitive
from construction_safety_vision.paths import ProjectPaths
from construction_safety_vision.provenance import sha256_file
from construction_safety_vision.segmentation_experiment import (
    ADAPTER_ROLE,
    CANONICAL_GROUND_TRUTH,
    CHECKPOINT_RULE,
    PRIMARY_METRIC,
    SUPPORTED_MACRO_MASK_METRIC,
    load_segmentation_baseline_config,
)
from construction_safety_vision.splits import HOLDOUT_UNLOCK_ENV_VAR

APPROVAL_JSON = "segmentation_adapter_approval.json"
S0_MANIFEST_JSON = "segmentation_S0_manifest.json"
S0_REPORT_MD = "segmentation_S0_protocol.md"
AUDIT_MANIFEST_JSON = "segmentation_adapter_audit_manifest.json"

EXPECTED_INSTANCES = 1726
EXPECTED_TRAIN_INSTANCES = 1422
EXPECTED_VALIDATION_INSTANCES = 304
EXPECTED_TRAIN_IMAGES = 303
EXPECTED_VALIDATION_IMAGES = 65
EXPECTED_NEGATIVES = 12

PHASE_8A_ARTIFACTS = (
    "reports/segmentation_adapter_audit_manifest.json",
    "reports/segmentation_adapter_fidelity_report.md",
    "reports/segmentation_adapter_fidelity.csv",
    "reports/segmentation_adapter_audit.provenance.json",
    "configs/segmentation_adapter_audit.yaml",
)

DETECTION_ARTIFACTS = (
    "reports/final_detector_manifest.json",
    "reports/detection_selection_report.md",
    "reports/detection_experiment_comparison.csv",
    "reports/detection_experiment_results.json",
    "reports/final_detector.provenance.json",
)


@pytest.fixture(scope="module")
def paths() -> ProjectPaths:
    return ProjectPaths.from_root()


@pytest.fixture(scope="module")
def approval(paths: ProjectPaths) -> dict:
    path = paths.reports / APPROVAL_JSON
    if not path.is_file():
        pytest.skip(f"{APPROVAL_JSON} not present; run the phase 8B freeze")
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def manifest(paths: ProjectPaths) -> dict:
    path = paths.reports / S0_MANIFEST_JSON
    if not path.is_file():
        pytest.skip(f"{S0_MANIFEST_JSON} not present; run the phase 8B freeze")
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def report(paths: ProjectPaths) -> str:
    path = paths.reports / S0_REPORT_MD
    if not path.is_file():
        pytest.skip(f"{S0_REPORT_MD} not present; run the phase 8B freeze")
    return path.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def audit(paths: ProjectPaths) -> dict:
    path = paths.reports / AUDIT_MANIFEST_JSON
    if not path.is_file():
        pytest.skip(f"{AUDIT_MANIFEST_JSON} not present; run the phase 8A audit")
    return json.loads(path.read_text(encoding="utf-8"))


# --- phase 8A is unchanged ----------------------------------------------------


def test_the_phase_8a_artifacts_are_byte_identical_to_the_recorded_digests(
    paths: ProjectPaths, approval: dict
):
    recorded = approval["phase_8a_artifact_digests"]
    assert sorted(recorded) == sorted(PHASE_8A_ARTIFACTS)
    for name, expected in recorded.items():
        assert sha256_file(paths.root / name) == expected, f"{name} changed since phase 8B"


def test_the_approval_references_the_phase_8a_audit_manifest_by_digest(
    paths: ProjectPaths, approval: dict
):
    observed = sha256_file(paths.reports / AUDIT_MANIFEST_JSON)
    assert approval["phase_8a_audit_manifest_sha256"] == observed


def test_the_phase_8a_manifest_still_records_that_it_selected_nothing(audit: dict):
    assert audit["segmentation_architecture_selection"] == "UNSELECTED_PENDING_FIDELITY_REVIEW"
    assert audit["adapter_status"]["status"] == "AUDIT_ONLY"
    assert audit["phase"] == "8A"


# --- the approval attaches to the audited bytes -------------------------------


def test_the_approval_reuses_the_exact_phase_8a_label_fingerprints(approval: dict, audit: dict):
    recorded = audit["adapter_fingerprints"]
    approved = approval["adapter_fingerprints"]
    for name in (
        "labels_train_sha256",
        "labels_validation_sha256",
        "labels_development_sha256",
        "image_membership_sha256",
    ):
        assert approved[name] == recorded[name], name


def test_the_protocol_declares_the_same_fingerprints(paths: ProjectPaths, audit: dict):
    config = load_segmentation_baseline_config(paths.configs / "segmentation_baseline.yaml")
    recorded = audit["adapter_fingerprints"]
    for name in (
        "labels_train_sha256",
        "labels_validation_sha256",
        "labels_development_sha256",
        "image_membership_sha256",
    ):
        assert config.adapter[name] == recorded[name], name


def test_the_fingerprints_were_verified_against_the_bytes_on_disk(approval: dict):
    assert approval["adapter_fingerprints_verified_on_disk"] is True
    assert approval["adapter_bytes_regenerated"] is False
    assert approval["conversion_algorithm_changed"] is False


def test_the_approval_status_and_role(approval: dict):
    assert approval["status"] == "APPROVED_FOR_CONTROLLED_TRAINING"
    assert approval["adapter_review_status"] == "APPROVED_FOR_CONTROLLED_TRAINING"
    assert approval["adapter_role"] == ADAPTER_ROLE
    assert approval["canonical_ground_truth"] == CANONICAL_GROUND_TRUTH
    assert approval["derived_adapter"] == "YOLO_SEGMENTATION"


def test_the_conversion_is_never_called_lossless(approval: dict, report: str):
    assert approval["conversion_characterisation"] == "ACCEPTED_WITH_QUANTIFIED_APPROXIMATION"
    assert approval["fidelity_summary"]["mask_iou_exact_instances"] == 0
    assert "lossless" not in report.lower().replace("never as lossless", "")


def test_the_architecture_selection_basis_is_audit_plus_review(approval: dict):
    assert (
        approval["architecture_selection_basis"]
        == "PHASE_8A_QUANTITATIVE_FIDELITY_AUDIT_PLUS_HUMAN_REVIEW"
    )


# --- cardinality --------------------------------------------------------------


@pytest.mark.parametrize(
    ("field", "expected"),
    [
        ("instances", EXPECTED_INSTANCES),
        ("train_instances", EXPECTED_TRAIN_INSTANCES),
        ("validation_instances", EXPECTED_VALIDATION_INSTANCES),
        ("train_images", EXPECTED_TRAIN_IMAGES),
        ("validation_images", EXPECTED_VALIDATION_IMAGES),
        ("negative_images", EXPECTED_NEGATIVES),
    ],
)
def test_the_approved_cardinality(approval: dict, field: str, expected: int):
    assert approval["cardinality"][field] == expected


def test_the_manifest_cardinality_agrees_with_the_approval(manifest: dict, approval: dict):
    assert manifest["adapter"]["counts"] == approval["cardinality"]


def test_no_instance_was_filtered_on_fidelity(approval: dict, manifest: dict):
    assert approval["all_instances_retained"] is True
    assert approval["fidelity_based_filtering"] == "NONE"
    assert manifest["adapter"]["all_instances_retained"] is True
    assert manifest["adapter"]["fidelity_based_filtering"] == "NONE"
    assert approval["fidelity_summary"]["instances_iou_lt_0.90"] == 47


def test_the_cardinality_matches_the_phase_8a_audit(approval: dict, audit: dict):
    cardinality = audit["instance_cardinality"]
    assert approval["cardinality"]["instances"] == cardinality["model_instance_rows"]
    assert audit["instance_cardinality"]["instances_dropped"] == 0


# --- the architecture decision ------------------------------------------------


def test_the_selected_architecture(approval: dict, manifest: dict):
    decision = approval["architecture_decision"]
    assert decision["status"] == "FINAL_SELECTED_FOR_S0"
    assert decision["selected_architecture"] == "YOLO11n-seg"
    assert decision["weight_identifier"] == "yolo11n-seg.pt"
    assert manifest["architecture"]["selected"] == "YOLO11n-seg"


def test_the_mask_native_fallback_is_recorded_not_selected(approval: dict, manifest: dict):
    assert approval["architecture_decision"]["fallback_status"] == "NOT_SELECTED_FALLBACK"
    assert manifest["architecture"]["fallback_status"] == "NOT_SELECTED_FALLBACK"
    assert "Mask R-CNN" in approval["architecture_decision"]["fallback_architecture"]


def test_no_universal_superiority_is_claimed(approval: dict, report: str):
    assert "not_a_claim_of_superiority" in approval["architecture_decision"]
    assert "nothing here says YOLO11n-seg is better" in report


# --- the frozen protocol ------------------------------------------------------


def test_the_manifest_records_the_frozen_execution_settings(manifest: dict):
    assert manifest["training"]["imgsz"] == 768
    assert manifest["training"]["batch"] == 8
    assert manifest["training"]["epochs"] == 100
    assert manifest["training"]["patience"] == 50
    assert manifest["training"]["optimizer"] == "auto"
    assert manifest["training"]["amp"] is True
    assert manifest["training"]["deterministic"] is True
    assert manifest["argument_classification"]["seed"]["declared"] == 42


def test_the_protocol_fingerprint_matches_the_committed_configuration(
    paths: ProjectPaths, manifest: dict
):
    config = load_segmentation_baseline_config(paths.configs / "segmentation_baseline.yaml")
    assert manifest["protocol_fingerprint"] == config.fingerprint()
    assert manifest["protocol_config_sha256"] == sha256_file(
        paths.configs / "segmentation_baseline.yaml"
    )


def test_the_segmentation_specific_arguments_are_present(manifest: dict):
    arguments = manifest["segmentation_arguments"]
    for name in ("overlap_mask", "mask_ratio", "retina_masks", "dropout", "max_det"):
        assert name in arguments
    assert manifest["training"]["workers"] == 8
    assert manifest["training"]["close_mosaic"] == 10
    assert manifest["installed_defaults_evidence"] == "INSTALLED_EFFECTIVE_CONFIGURATION"


def test_the_augmentation_policy_is_the_framework_default(manifest: dict):
    assert manifest["augmentation_policy"] == "ULTRALYTICS_DEFAULT_SEGMENTATION_TRAINING_POLICY"
    assert manifest["augmentation_arguments"]["mosaic"] == 1.0
    assert manifest["augmentation_arguments"]["fliplr"] == 0.5


def test_the_batch_decision_is_predeclared(manifest: dict):
    assert manifest["batch_decision"].startswith("PREDECLARED_EXECUTION_DECISION")
    assert "MEMORY_CONSTRAINT_REVIEW_REQUIRED" in manifest["batch_decision"]


def test_the_checkpoint_rule_is_defined_and_its_behaviour_established(manifest: dict):
    assert CHECKPOINT_RULE in manifest["checkpoint_selection"]
    fitness = manifest["checkpoint_fitness_behaviour"]
    assert fitness["classification"] == "COMBINED_MASK_AND_BOX_MAP50_95_UNWEIGHTED_SUM"
    assert fitness["driven_by"] == "BOTH_MASK_AND_BOX"
    assert fitness["evidence"] == "INSTALLED_PACKAGE_SOURCE_INSPECTION"
    assert fitness["component_weights"] == [0.0, 0.0, 0.0, 1.0]


def test_the_checkpoint_selection_policy_is_the_native_composite(manifest: dict):
    policy = manifest["checkpoint_selection_policy"]
    assert policy["checkpoint_selection_policy"] == "ULTRALYTICS_NATIVE_SEGMENTATION_FITNESS"
    assert policy["checkpoint_selection_semantics"] == "BOX_MAP50_95_PLUS_MASK_MAP50_95"
    assert policy["checkpoint_selection_box_component_weight"] == 1.0
    assert policy["checkpoint_selection_mask_component_weight"] == 1.0
    assert policy["established_from"] == "INSTALLED_PACKAGE_SOURCE_INSPECTION"


def test_the_checkpoint_decision_was_human_reviewed_before_s0(manifest: dict):
    policy = manifest["checkpoint_selection_policy"]
    assert policy["checkpoint_selection_review_status"] == "HUMAN_REVIEWED_AND_ACCEPTED_BEFORE_S0"
    # Reviewed before S0 means S0 must still be unrun in this same manifest.
    assert manifest["s0_execution_status"] == "NOT_EXECUTED_PROTOCOL_ONLY"
    assert manifest["s0_result_metrics"] is None


def test_the_selector_is_recorded_as_differing_from_the_reporting_metric(manifest: dict):
    policy = manifest["checkpoint_selection_policy"]
    assert policy["primary_scientific_reporting_metric"] == "MASK_MAP50_95"
    assert policy["selection_metric_equals_primary_reporting_metric"] is False
    assert policy["divergence_is_intentional"] is True


def test_the_framework_fitness_is_not_presented_as_the_primary_metric(manifest: dict):
    policy = manifest["checkpoint_selection_policy"]
    assert policy["framework_fitness_is_a_reporting_metric"] is False
    # The reported hierarchy is untouched by the checkpoint decision.
    metrics = manifest["metrics"]
    assert metrics["primary"] == PRIMARY_METRIC
    assert metrics["composite_box_mask_score"] is False
    for family in ("secondary_mask", "per_class_mask"):
        assert all(name.startswith("mask_") for name in metrics[family])


def test_no_mask_only_post_hoc_checkpoint_selection_is_authorized(manifest: dict):
    policy = manifest["checkpoint_selection_policy"]
    assert policy["custom_mask_only_selector_authorized"] is False
    assert policy["retrospective_reinterpretation_allowed"] is False


def test_future_comparisons_inherit_the_checkpoint_policy(manifest: dict):
    constraint = manifest["checkpoint_selection_policy"]["future_comparison_constraint"]
    assert "ULTRALYTICS_NATIVE_SEGMENTATION_FITNESS" in constraint
    assert "frozen BEFORE any affected experiment is run" in constraint


def test_no_superiority_claim_is_made_for_the_composite(manifest: dict, report: str):
    policy = manifest["checkpoint_selection_policy"]
    assert policy["superiority_claim"].startswith("NONE.")
    assert "NOT a claim that the composite fitness is scientifically superior" in " ".join(
        policy["rationale"]
    )
    assert "superior to mask-only" not in report.replace(
        "NOT a claim that the composite fitness is scientifically superior to mask-only", ""
    )


def test_the_report_documents_the_checkpoint_decision(report: str):
    for fragment in (
        "### 13.1 What the framework actually optimises",
        "### 13.2 The reviewed decision",
        "### 13.3 Why the native fitness was kept",
        "### 13.4 Protocol invariant for future comparisons",
        "`ULTRALYTICS_NATIVE_SEGMENTATION_FITNESS`",
        "`HUMAN_REVIEWED_AND_ACCEPTED_BEFORE_S0`",
        "`BOX_MAP50_95_PLUS_MASK_MAP50_95`",
        "**False** - intentional",
        "checkpoint-selection mechanism, not a scientific headline",
    ):
        assert fragment in report, fragment


def test_the_metric_hierarchy_is_recorded(manifest: dict):
    metrics = manifest["metrics"]
    assert metrics["primary"] == PRIMARY_METRIC
    assert metrics["supported_macro"] == SUPPORTED_MACRO_MASK_METRIC
    assert metrics["composite_box_mask_score"] is False
    assert "mask_mAP@0.50" in metrics["secondary_mask"]
    assert "box_mAP@0.50:0.95" in metrics["box_from_segmenter"]


def test_the_supported_macro_is_not_yet_a_selection_metric(manifest: dict):
    policy = manifest["supported_macro_mask_policy"]
    assert policy["s0_reports_it"] is True
    assert policy["is_a_selection_metric"] is False


def test_the_rare_class_policy_is_the_frozen_phase_7_rule(manifest: dict):
    rare = manifest["rare_class"]
    assert rare["name"] == "vest_loose"
    assert rare["classification"] == "DESCRIPTIVE_HIGH_UNCERTAINTY"
    assert rare["support_rule_origin"] == "FROZEN_IN_PHASE_7A_REUSED_UNCHANGED"
    assert rare["validation_images"] == 1
    assert rare["validation_instances"] == 8
    assert rare["reported_in_full"] is True
    assert rare["may_decide_a_winner"] is False


def test_the_direct_mask_iou_requirement_is_recorded_but_not_executed(manifest: dict):
    requirement = manifest["direct_mask_iou_requirement"]
    assert requirement["status"] == "REQUIRED_FUTURE_PREDECLARED_EVALUATION"
    assert requirement["defined_here"] is False
    assert requirement["executed_here"] is False


# --- the pretrained weights ---------------------------------------------------


def test_the_pretrained_weight_is_fingerprinted(manifest: dict):
    weights = manifest["pretrained_weights"]
    assert weights["identifier"] == "yolo11n-seg.pt"
    assert re.fullmatch(r"[0-9a-f]{64}", weights["sha256"])
    assert weights["size_bytes"] > 0
    assert weights["committed"] is False
    assert weights["source_mechanism"] == "ULTRALYTICS_ASSET_DOWNLOAD"


def test_no_model_binary_is_committed(paths: ProjectPaths):
    assert not (paths.reports / "yolo11n-seg.pt").exists()
    assert not list(paths.reports.glob("*.pt"))


# --- the smoke test -----------------------------------------------------------


def test_the_smoke_test_is_marked_non_experimental(manifest: dict):
    smoke = manifest["smoke_test"]
    assert smoke["classification"] == "NON_EXPERIMENTAL"
    assert smoke["reporting_ban"] == "DO_NOT_REPORT_AS_MODEL_RESULT"
    assert smoke["epochs"] == 1
    assert smoke["run_directory_committed"] is False


def test_the_smoke_test_recorded_no_accuracy_metric(manifest: dict):
    smoke = manifest["smoke_test"]
    assert smoke["metrics_recorded"] is False
    serialised = json.dumps(smoke)
    for banned in ("mAP", "mask_map", "precision", "recall", "AP@"):
        assert banned not in serialised, f"the smoke record leaks {banned}"


def test_the_smoke_test_ran_at_the_frozen_batch_and_resolution(manifest: dict):
    smoke = manifest["smoke_test"]
    assert smoke["batch"] == manifest["training"]["batch"]
    assert smoke["imgsz"] == manifest["training"]["imgsz"]
    assert smoke["out_of_memory"] is False


# --- nothing was decided that this phase could not decide ---------------------


def test_s0_was_not_trained_and_no_result_exists(manifest: dict):
    assert manifest["s0_execution_status"] == "NOT_EXECUTED_PROTOCOL_ONLY"
    assert manifest["s0_result_metrics"] is None
    assert manifest["models_trained_in_this_phase"] == 0


def test_the_report_carries_no_s0_performance_number(report: str):
    assert "no S0 performance result" in report
    assert "NOT_EXECUTED_PROTOCOL_ONLY" in report
    assert "supported_macro_mask_map50_95 0." not in report


def test_the_detector_was_not_touched(manifest: dict):
    detector = manifest["frozen_detector"]
    assert detector["selected_experiment"] == "D2"
    assert detector["model"] == "YOLO11n"
    assert detector["imgsz"] == 768
    assert detector["retrained"] is False
    assert detector["revalidated"] is False
    assert detector["inference_run"] is False
    assert detector["threshold_changed"] is False
    assert manifest["detector_touched"] is False


def test_the_detection_artifacts_are_byte_identical(paths: ProjectPaths, manifest: dict):
    recorded = manifest["detection_artifact_digests"]
    assert sorted(recorded) == sorted(DETECTION_ARTIFACTS)
    for name, expected in recorded.items():
        assert sha256_file(paths.root / name) == expected, f"{name} changed since phase 8B"


# --- the holdout --------------------------------------------------------------


def test_the_holdout_is_recorded_as_protected(approval: dict, manifest: dict):
    assert approval["test"]["status"] == "PROTECTED_NOT_ACCESSED"
    assert manifest["test"]["status"] == "PROTECTED_NOT_ACCESSED"
    assert manifest["holdout_accessed"] is False


def test_no_artifact_names_a_holdout_split(approval: dict, manifest: dict):
    for document in (approval, manifest):
        serialised = json.dumps(document)
        assert '"test": "' not in serialised.replace('"test": {', "")
        assert "images/test" not in serialised
        assert "labels/test" not in serialised
        assert "_test.coco.json" not in serialised


def test_no_holdout_segmentation_adapter_exists(paths: ProjectPaths):
    root = paths.data_processed / "adapters" / "yolo_segmentation_audit"
    assert not (root / "images" / "test").exists()
    assert not (root / "labels" / "test").exists()


def test_the_adapter_descriptor_exposes_no_holdout_split(manifest: dict):
    assert "test" not in manifest["adapter"]["descriptor_keys"]
    assert set(manifest["adapter"]["descriptor_keys"]) == {"names", "path", "train", "val"}


def test_the_unlock_variable_is_named_but_not_required(manifest: dict):
    assert HOLDOUT_UNLOCK_ENV_VAR == "CSVISION_ALLOW_TEST_SPLIT"
    assert HOLDOUT_UNLOCK_ENV_VAR not in json.dumps(manifest)


# --- hygiene ------------------------------------------------------------------


def test_the_artifacts_carry_no_sensitive_content(approval: dict, manifest: dict, report: str):
    assert scan_for_sensitive(report) == []
    assert scan_for_sensitive(json.dumps(manifest)) == []
    assert scan_for_sensitive(json.dumps(approval)) == []


def test_the_report_declares_every_required_section(report: str):
    for heading in (
        "## 1. Architecture decision",
        "## 2. Phase 8A evidence",
        "## 3. Why YOLO11n-seg was accepted",
        "## 4. Why canonical COCO remains authoritative",
        "## 5. Known adapter limitations",
        "## 6. Why no instance filtering occurred",
        "## 7. S0 experimental question",
        "## 8. Frozen data",
        "## 9. Frozen training configuration",
        "## 10. Segmentation-specific framework arguments",
        "## 11. Metric hierarchy",
        "## 12. The `vest_loose` limitation",
        "## 13. Checkpoint-selection behaviour",
        "## 14. Future direct mask-IoU requirement",
        "## 15. Smoke-test result",
        "## 16. Holdout policy",
        "## 17. Fallback status",
        "## 18. Next phase",
    ):
        assert heading in report, heading


def test_the_report_labels_its_claims(report: str):
    for label in (
        "HUMAN_ARCHITECTURE_DECISION",
        "AUDIT_EVIDENCE",
        "PREDECLARED_PROTOCOL",
        "MODEL_ADAPTER_LIMITATION",
        "HOLDOUT_POLICY",
        "FUTURE_EVALUATION_REQUIREMENT",
    ):
        assert label in report, label


def test_the_provenance_record_exists(paths: ProjectPaths):
    path = paths.reports / "segmentation_S0_protocol.provenance.json"
    if not path.is_file():
        pytest.skip("provenance record not present; run the phase 8B freeze")
    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["details"]["classification"] == "S0_PROTOCOL_FROZEN"
    assert record["details"]["models_trained_in_this_phase"] == 0
    assert record["details"]["holdout_accessed"] is False
    assert record["details"]["smoke_metrics_recorded"] is False
