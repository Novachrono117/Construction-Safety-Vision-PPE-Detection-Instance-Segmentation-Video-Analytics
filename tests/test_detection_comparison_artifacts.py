"""Tests that the committed Phase 7 comparison policy says what it should.

They read the emitted artifacts and check them against the frozen inputs they
claim to be derived from, so a policy edited by hand - or regenerated from a
changed configuration without the configuration being committed - fails here.

They also check the two things this phase must be able to prove: that the
holdout appears in the artifacts only as a statement that it was not touched,
and that no D1 or D2 result exists.
"""

from __future__ import annotations

import json

import pytest

from conftest import holdout_has_been_evaluated
from construction_safety_vision.data.canonical import scan_for_sensitive
from construction_safety_vision.detection_comparison import (
    CASE_A,
    CASE_B,
    CASE_C,
    CASE_D,
    DESCRIPTIVE_HIGH_UNCERTAINTY,
    OFFICIAL_ALL_CLASS_METRIC,
    PRIMARY_SELECTION_METRIC,
    build_experiment_record,
    class_support,
    load_experiment_matrix,
)
from construction_safety_vision.paths import ProjectPaths
from construction_safety_vision.splits import HOLDOUT_UNLOCK_ENV_VAR

CLASS_MAP = {
    "helmet_loose": 0,
    "helmet_on_head": 1,
    "person": 2,
    "vest_loose": 3,
    "vest_on_body": 4,
}
HOLDOUT_STATUS = "PROTECTED_NOT_ACCESSED"


@pytest.fixture(scope="module")
def paths() -> ProjectPaths:
    return ProjectPaths.from_root()


@pytest.fixture(scope="module")
def policy(paths: ProjectPaths) -> dict:
    path = paths.reports / "detection_comparison_policy.json"
    if not path.is_file():
        pytest.skip("run scripts/freeze_detection_experiments.py")
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def reference(paths: ProjectPaths) -> dict:
    path = paths.reports / "detection_comparison_reference.json"
    if not path.is_file():
        pytest.skip("run scripts/freeze_detection_experiments.py")
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def report(paths: ProjectPaths) -> str:
    path = paths.reports / "detection_comparison_policy.md"
    if not path.is_file():
        pytest.skip("run scripts/freeze_detection_experiments.py")
    return path.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def matrix(paths: ProjectPaths):
    return load_experiment_matrix(paths.configs / "detection_experiments.yaml")


@pytest.fixture(scope="module")
def d0_manifest(paths: ProjectPaths) -> dict:
    path = paths.reports / "detection_D0_manifest.json"
    if not path.is_file():
        pytest.skip("detection_D0_manifest.json not present")
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def split_manifest(paths: ProjectPaths) -> dict:
    return json.loads((paths.reports / "split_manifest.json").read_text(encoding="utf-8"))


# --- policy structure ---------------------------------------------------------


def test_the_policy_is_classified_frozen(policy):
    assert policy["classification"] == "DETECTION_EXPERIMENT_PROTOCOL_FROZEN"
    assert policy["phase"] == "7A"
    assert policy["reference_experiment"] == "D0"


def test_the_policy_matches_the_committed_configuration(policy, paths):
    import hashlib

    digest = hashlib.sha256((paths.configs / "detection_experiments.yaml").read_bytes()).hexdigest()
    assert policy["experiment_matrix_sha256"] == digest


def test_the_policy_names_the_frozen_support_rule(policy):
    rule = policy["support_rule"]
    assert rule["min_validation_positive_images"] == 5
    assert rule["min_validation_instances"] == 20
    assert rule["split"] == "validation"


def test_the_policy_support_table_matches_the_frozen_split(policy, split_manifest):
    images = split_manifest["images_with_class"]["validation"]
    instances = split_manifest["instances_by_class"]["validation"]
    for row in policy["class_support"]:
        name = row["class_name"]
        assert row["validation_positive_images"] == images[name]
        assert row["validation_instances"] == instances[name]


def test_the_policy_classifies_the_expected_classes(policy):
    assert policy["supported_classes"] == [
        "helmet_loose",
        "helmet_on_head",
        "person",
        "vest_on_body",
    ]
    assert policy["descriptive_high_uncertainty_classes"] == ["vest_loose"]


def test_the_policy_keeps_the_all_class_metric_alongside_the_selection_metric(policy):
    metrics = policy["metrics"]
    assert metrics["primary_selection"] == PRIMARY_SELECTION_METRIC
    assert metrics["official_all_class"] == OFFICIAL_ALL_CLASS_METRIC
    assert metrics["official_all_class_role"] == "OFFICIAL_ALL_CLASS_REPORTING_METRIC"
    assert metrics["all_class_is_never_hidden"] is True
    assert metrics["secondary"] == ["mAP@0.50", "precision", "recall"]


def test_the_policy_freezes_the_margin_and_disclaims_significance(policy):
    margin = policy["practical_equivalence_margin"]
    assert margin["value"] == "0.005"
    assert margin["not_a_significance_test"] is True


def test_the_policy_declares_all_four_selection_cases(policy):
    cases = policy["selection_logic"]["cases"]
    assert set(cases) == {CASE_A, CASE_B, CASE_C, CASE_D}
    assert policy["selection_logic"]["metric"] == PRIMARY_SELECTION_METRIC
    assert policy["selection_logic"]["margin"] == "0.005"


def test_the_policy_forbids_the_descriptive_class_overriding_selection(policy):
    forbidden = " ".join(policy["selection_logic"]["overrides_forbidden"])
    assert DESCRIPTIVE_HIGH_UNCERTAINTY in forbidden
    assert "recall" in forbidden
    assert "holdout" in forbidden


def test_the_policy_keeps_the_descriptive_class_as_a_required_class(policy):
    rare = policy["rare_class_policy"]
    assert rare["classes"] == ["vest_loose"]
    assert rare["still_required"] is True
    assert set(rare["must_report"]) == {"precision", "recall", "AP@0.50", "AP@0.50:0.95"}
    assert "vest_loose" in rare["reported_for_every_experiment"]
    forbidden = " ".join(rare["forbidden"])
    assert "tuning" in forbidden
    assert "holdout" in forbidden


def test_the_policy_declares_the_batch_and_oom_rule(policy):
    memory = policy["memory_policy"]
    assert memory["controlled_batch"] == 16
    assert memory["on_cuda_oom"] == "STOP_AND_CLASSIFY"
    assert memory["classification"] == "MEMORY_CONSTRAINT_REVIEW_REQUIRED"


def test_the_policy_forbids_post_hoc_experiments(policy):
    post_hoc = policy["post_hoc_policy"]
    assert post_hoc["authorised_experiments"] == ["D1", "D2"]
    assert post_hoc["no_further_experiment_without_review"] is True
    joined = " ".join(post_hoc["forbidden_automatic_followups"]).lower()
    for forbidden in ("yolo11m", "imgsz 896", "learning rate", "mosaic", "class weighting"):
        assert forbidden in joined


def test_the_policy_records_the_revalidation_audit_honestly(policy):
    audit = policy["revalidation_audit"]
    assert audit["classification"] == "NON_SELECTION_REVALIDATION"
    assert (
        audit["equivalent_classification"]
        == "PROTOCOL_DEVIATION_WITHOUT_SELECTION_DEGREE_OF_FREEDOM"
    )
    assert audit["selection_degrees_of_freedom_introduced"] == 0
    assert audit["d0_metrics_modified"] is False
    assert "a second independent experiment" in audit["was_not"]
    # The distinction that keeps the note honest: what the repository can check
    # is recorded separately from what it cannot, and neither list is empty.
    assert audit["evidence_basis"] == "MAINTAINER_DECLARED_PARTIALLY_CORROBORATED_ON_DISK"
    assert audit["corroborated_on_disk"]
    assert audit["not_independently_verified_here"]
    unverified = " ".join(audit["not_independently_verified_here"])
    assert "nothing was varied" in unverified


def test_the_revalidation_corroboration_is_true_on_disk(policy, paths):
    # The audit note claims the committed D0 figures come from a separate
    # validation run. That claim is checkable, so it is checked rather than
    # trusted - and skipped rather than faked when the ignored run is absent.
    run_dir = paths.root / "artifacts" / "detection" / "D0_val"
    if not run_dir.is_dir():
        pytest.skip("artifacts/detection/D0_val is git-ignored and not present")
    committed = paths.figures / "detection" / "D0"
    matched = 0
    for figure in sorted(run_dir.glob("*.png")):
        target = committed / figure.name
        if target.is_file():
            assert figure.read_bytes() == target.read_bytes(), figure.name
            matched += 1
    assert matched >= 6
    assert "D0_val" in " ".join(policy["revalidation_audit"]["corroborated_on_disk"])


def test_no_dataset_or_prediction_image_is_committed(paths):
    committed = paths.figures / "detection" / "D0"
    if not committed.is_dir():
        pytest.skip("no committed D0 figures")
    names = {path.name for path in committed.iterdir()}
    assert not any("batch" in name for name in names)
    assert "labels.jpg" not in names


def test_the_policy_states_its_limitations(policy):
    ids = {item["id"] for item in policy["limitations"]}
    assert {
        "SMALL_VALIDATION_SPLIT",
        "RUN_TO_RUN_VARIANCE_NOT_MEASURED",
        "DESCRIPTIVE_CLASS_EXCLUDED_FROM_SELECTION",
    } <= ids
    assert all(item["label"] == "LIMITATION" for item in policy["limitations"])


# --- the D0 reference ---------------------------------------------------------


def test_the_reference_reproduces_from_the_committed_d0_result(
    reference, d0_manifest, split_manifest, matrix
):
    support = class_support(split_manifest, CLASS_MAP, rule=matrix.support_rule)
    record = build_experiment_record(d0_manifest, support)
    assert reference["metrics"][PRIMARY_SELECTION_METRIC] == round(float(record.supported_macro), 6)
    assert reference["metrics"][OFFICIAL_ALL_CLASS_METRIC] == round(
        float(record.all_class_map50_95), 6
    )
    assert reference["experiment_sha256"] == d0_manifest["d0_experiment_sha256"]
    assert reference["best_checkpoint_sha256"] == d0_manifest["best_checkpoint"]["sha256"]


def test_the_reference_holds_the_expected_selection_metric(reference):
    assert reference["metrics"][PRIMARY_SELECTION_METRIC] == 0.570142


def test_the_reference_carries_every_required_field(reference):
    assert reference["model"] == "YOLO11n"
    assert reference["imgsz"] == 640
    assert "parameters" in reference
    for name in (
        PRIMARY_SELECTION_METRIC,
        OFFICIAL_ALL_CLASS_METRIC,
        "mAP@0.50",
        "precision",
        "recall",
    ):
        assert name in reference["metrics"]
    assert set(reference["per_class_metrics"]) == set(CLASS_MAP)
    assert reference["metrics_are_validation_only"] is True


def test_the_reference_per_class_metrics_are_the_committed_ones(reference, d0_manifest):
    assert reference["per_class_metrics"] == {
        name: dict(table) for name, table in sorted(d0_manifest["per_class_metrics"].items())
    }


def test_the_reference_states_the_gap_between_the_two_metrics(reference):
    metrics = reference["metrics"]
    assert reference["supported_macro_minus_all_class"] == round(
        metrics[PRIMARY_SELECTION_METRIC] - metrics[OFFICIAL_ALL_CLASS_METRIC], 6
    )


def test_the_reference_forbids_recomputing_d0(reference):
    assert "must not be recomputed" in reference["reference_freeze_rule"]


# --- one-variable discipline --------------------------------------------------


def test_the_policy_proves_both_comparisons_are_one_variable(policy):
    verdicts = {
        item["experiment_id"]: item for item in policy["one_variable_discipline"]["verdicts"]
    }
    assert set(verdicts) == {"D1", "D2"}
    assert all(item["compatible"] for item in verdicts.values())
    assert verdicts["D1"]["intentional_variable"] == "MODEL_CAPACITY"
    assert verdicts["D1"]["observed_differences"] == ["model", "weight_identifier"]
    assert verdicts["D2"]["intentional_variable"] == "INPUT_RESOLUTION"
    assert verdicts["D2"]["observed_differences"] == ["training.imgsz"]


def test_the_resolved_protocols_share_every_frozen_field(policy):
    protocols = {item["experiment_id"]: item["resolved_protocol"] for item in policy["experiments"]}
    d0 = protocols["D0"]
    for experiment_id in ("D1", "D2"):
        candidate = protocols[experiment_id]
        for field in (
            "seed",
            "training.epochs",
            "training.batch",
            "training.optimizer",
            "training.patience",
            "training.deterministic",
            "training.augmentation",
            "checkpoint_selection",
            "dataset_config",
        ):
            assert candidate[field] == d0[field], f"{experiment_id} moved {field}"


def test_d1_resolves_to_yolo11s_at_the_baseline_resolution(policy):
    protocols = {item["experiment_id"]: item["resolved_protocol"] for item in policy["experiments"]}
    assert protocols["D1"]["model"] == "YOLO11s"
    assert protocols["D1"]["weight_identifier"] == "yolo11s.pt"
    assert protocols["D1"]["training.imgsz"] == protocols["D0"]["training.imgsz"] == 640
    assert protocols["D1"]["training.batch"] == 16


def test_d2_resolves_to_768_on_the_baseline_model(policy):
    protocols = {item["experiment_id"]: item["resolved_protocol"] for item in policy["experiments"]}
    assert protocols["D2"]["training.imgsz"] == 768
    assert protocols["D2"]["model"] == "YOLO11n"
    assert protocols["D2"]["weight_identifier"] == "yolo11n.pt"
    assert protocols["D2"]["training.batch"] == 16


# --- no result exists yet -----------------------------------------------------


def test_the_policy_still_records_the_statuses_it_was_frozen_with(policy):
    # The policy is a phase 7A artifact and its `status` fields say what was true
    # when it was frozen, not what is true now. It must NOT be regenerated as
    # experiments run: later result manifests reference its digest, so rewriting
    # it would invalidate their recorded provenance. Live status lives in
    # reports/detection_experiment_results.json instead.
    statuses = {
        item["experiment_id"]: item["status"]
        for item in policy["experiments"]
        if item["role"] == "CONTROLLED_EXPERIMENT"
    }
    assert statuses == {"D1": "FROZEN_NOT_EXECUTED", "D2": "FROZEN_NOT_EXECUTED"}


def test_the_frozen_policy_digest_still_matches_what_results_reference(policy, paths):
    import hashlib

    digest = hashlib.sha256(
        (paths.reports / "detection_comparison_policy.json").read_bytes()
    ).hexdigest()
    for experiment_id in ("D1", "D2"):
        manifest = paths.reports / f"detection_{experiment_id}_manifest.json"
        if not manifest.is_file():
            continue
        recorded = json.loads(manifest.read_text(encoding="utf-8"))["phase7_policy_sha256"]
        assert recorded == digest, (
            f"{experiment_id} was judged under a different policy file than the one committed"
        )


def test_no_result_exists_for_an_unexecuted_candidate(paths):
    results_path = paths.reports / "detection_experiment_results.json"
    if not results_path.is_file():
        pending = {"D1", "D2"}
    else:
        results = json.loads(results_path.read_text(encoding="utf-8"))
        pending = set(results["pending_experiments"])
    for experiment_id in sorted(pending):
        assert not (paths.reports / f"detection_{experiment_id}_manifest.json").exists()
        assert not (paths.reports / f"detection_{experiment_id}_report.md").exists()
        assert not (paths.root / "artifacts" / "detection" / experiment_id).exists()


def test_the_frozen_policy_carries_no_candidate_metric(policy):
    for item in policy["experiments"]:
        if item["role"] != "CONTROLLED_EXPERIMENT":
            continue
        assert "metrics" not in item
        assert item["pretrained_weights"] is None or (
            item["pretrained_weights"]["fingerprint_status"] == "NOT_YET_FINGERPRINTED"
        )


def test_the_provenance_records_that_nothing_was_trained(paths):
    path = paths.reports / "detection_experiments.provenance.json"
    if not path.is_file():
        pytest.skip("run scripts/freeze_detection_experiments.py")
    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["phase"] == 7
    details = record["details"]
    assert details["models_trained_in_this_phase"] == 0
    assert details["experiments_executed_in_this_phase"] == []
    assert details["holdout_accessed"] is False
    assert details["frozen_experiments"] == ["D1", "D2"]


# --- holdout protection -------------------------------------------------------


def _holdout_leaks(payload, path=""):
    leaks = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            here = f"{path}.{key}" if path else str(key)
            if isinstance(key, str) and key.strip().lower() == "test":
                allowed = (
                    isinstance(value, dict)
                    and set(value) == {"status", "reason"}
                    and value.get("status") == HOLDOUT_STATUS
                )
                if not allowed:
                    leaks.append(here)
                continue
            leaks.extend(_holdout_leaks(value, here))
    elif isinstance(payload, list):
        for index, item in enumerate(payload):
            leaks.extend(_holdout_leaks(item, f"{path}[{index}]"))
    elif isinstance(payload, str) and payload.strip().lower() == "test":
        leaks.append(path)
    return leaks


def test_the_artifacts_name_the_holdout_only_as_untouched(policy, reference):
    assert _holdout_leaks(policy) == []
    assert _holdout_leaks(reference) == []
    assert policy["holdout_policy"]["test"]["status"] == HOLDOUT_STATUS
    assert reference["test"]["status"] == HOLDOUT_STATUS


def test_the_policy_declares_the_holdout_locked_for_this_phase(policy):
    guard = policy["holdout_policy"]["guard"]
    assert guard["env_var"] == HOLDOUT_UNLOCK_ENV_VAR
    assert guard["requires_code_opt_in"] is True
    assert guard["requires_environment_opt_in"] is True
    assert guard["state_during_this_phase"] == "LOCKED"


def test_no_holdout_dataset_or_adapter_was_produced(paths):
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


def test_the_artifacts_carry_no_sensitive_content(policy, reference, report):
    assert scan_for_sensitive(json.dumps(policy, ensure_ascii=False)) == []
    assert scan_for_sensitive(json.dumps(reference, ensure_ascii=False)) == []
    assert scan_for_sensitive(report) == []
    for text in (json.dumps(policy), report):
        assert "ROBOFLOW_API_KEY" not in text


# --- the report ---------------------------------------------------------------


def test_the_report_states_the_sequencing_up_front(report):
    assert "**D0 exists. D1 and D2 do not.**" in report


def test_the_report_labels_its_claims(report):
    for label in (
        "PREDECLARED_PHASE7_POLICY",
        "FROZEN_DATASET_FACT",
        "D0_REFERENCE_RESULT",
        "SELECTION_RULE",
        "LIMITATION",
    ):
        assert label in report


def test_the_report_shows_the_descriptive_class_in_the_per_class_table(report):
    assert "| `vest_loose` |" in report


def test_the_report_shows_both_headline_metrics(report):
    assert "0.570142" in report
    assert "0.464429" in report


def test_the_report_states_what_the_phase_did_not_do(report):
    assert "D1 was not trained. D2 was not trained." in report
    assert "was not fetched or fingerprinted" in report
