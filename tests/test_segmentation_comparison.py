"""Tests for the frozen S0-versus-S1 comparison protocol and its artifacts.

The schema tests are mostly refusals, because the protocol's job is to make
certain later moves impossible: promoting the framework's own metric, adding a
second override, inventing a composite once the numbers disagree, or recording a
result for an experiment that has not run.

The artifact tests check that the committed S0 canonical reference describes the
frozen checkpoint under the frozen evaluator, and that nothing about S1 has been
decided.

No test here trains, infers on real data, or touches the holdout.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest
import yaml

from construction_safety_vision.canonical_evaluation import (
    ALL_CLASS_METRIC,
    ALL_CLASS_METRIC_50,
    GROUND_TRUTH_SOURCE,
    IOU_THRESHOLDS,
    MAX_DETS,
    NATIVE_METRIC_STATUS,
    PRACTICAL_EQUIVALENCE_MARGIN,
    PRIMARY_METRIC,
    load_canonical_evaluation_config,
)
from construction_safety_vision.data.canonical import scan_for_sensitive
from construction_safety_vision.paths import ProjectPaths
from construction_safety_vision.provenance import sha256_file
from construction_safety_vision.segmentation_comparison import (
    CANDIDATE,
    CHECKPOINT_POLICY,
    ONE_VARIABLE_FIELD,
    REFERENCE,
    S1_STATUS,
    TIMING,
    ComparisonConfigError,
    load_comparison_config,
    verify_one_variable_contract,
)
from construction_safety_vision.segmentation_experiment import load_segmentation_baseline_config
from construction_safety_vision.splits import HOLDOUT_UNLOCK_ENV_VAR

CONFIG_NAME = "segmentation_comparison.yaml"
CANONICAL_JSON = "segmentation_S0_canonical_evaluation.json"
POLICY_JSON = "segmentation_comparison_policy.json"
POLICY_MD = "segmentation_comparison_policy.md"
S1_MANIFEST_JSON = "segmentation_S1_protocol_manifest.json"
S1_MD = "segmentation_S1_protocol.md"
REFERENCE_MD = "segmentation_canonical_comparison_reference.md"

EXPECTED_S0_CHECKPOINT = "d7b512b95fafdc8658fd802459d2c87d9ad3f11f922162a774dacf8ca75b87a3"
EXPECTED_PRETRAINED = "55ed65c56c91713d23e8402371c6c49a6fd84f257f7dce452e8d70e41dcbe152"
SHA256 = re.compile(r"^[0-9a-f]{64}$")


@pytest.fixture(scope="module")
def paths() -> ProjectPaths:
    return ProjectPaths.from_root()


@pytest.fixture(scope="module")
def raw(paths: ProjectPaths) -> dict[str, Any]:
    return yaml.safe_load((paths.configs / CONFIG_NAME).read_text(encoding="utf-8"))


def write(tmp_path: Path, document: dict[str, Any]) -> Path:
    destination = tmp_path / CONFIG_NAME
    destination.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return destination


# --- the committed protocol ---------------------------------------------------


def test_the_committed_protocol_parses(paths: ProjectPaths):
    config = load_comparison_config(paths.configs / CONFIG_NAME)
    assert config["reference_experiment"] == REFERENCE
    assert config.candidate["experiment_id"] == CANDIDATE


def test_the_protocol_records_its_own_timing_honestly(paths: ProjectPaths):
    config = load_comparison_config(paths.configs / CONFIG_NAME)
    assert config["protocol_timing"] == TIMING
    assert "POST_S0" in TIMING and "PRE_S1" in TIMING


def test_s1_differs_from_s0_only_by_overlap_mask(paths: ProjectPaths):
    config = load_comparison_config(paths.configs / CONFIG_NAME)
    assert set(config.candidate["overrides"]) == {ONE_VARIABLE_FIELD}
    assert config.candidate["overrides"][ONE_VARIABLE_FIELD] is False
    assert config.candidate["intentional_field"] == ONE_VARIABLE_FIELD
    assert config.candidate["inherits"] == REFERENCE


def test_s1_inherits_every_other_frozen_field(paths: ProjectPaths):
    baseline = load_segmentation_baseline_config(paths.configs / "segmentation_baseline.yaml")
    config = load_comparison_config(paths.configs / CONFIG_NAME)
    contract = verify_one_variable_contract(
        baseline.training_arguments(), config.candidate["overrides"]
    )
    assert contract["one_variable"] is True
    assert contract["reference_value"] is True
    assert contract["candidate_value"] is False
    # The fields the comparison must hold constant are genuinely inherited.
    for name in ("imgsz", "batch", "epochs", "seed", "mask_ratio", "optimizer", "patience"):
        assert name in contract["fields_inherited_unchanged"], name


def test_s1_keeps_mask_ratio_four_and_the_native_checkpoint_policy(paths: ProjectPaths):
    baseline = load_segmentation_baseline_config(paths.configs / "segmentation_baseline.yaml")
    config = load_comparison_config(paths.configs / CONFIG_NAME)
    # mask_ratio is explicitly NOT the intervention: phase 8D assessed it as
    # weakly motivated, and S1 must inherit S0's value unchanged.
    assert baseline.segmentation_arguments["mask_ratio"] == 4
    assert baseline.segmentation_arguments[ONE_VARIABLE_FIELD] is True
    assert config.candidate["overrides"][ONE_VARIABLE_FIELD] is False
    assert config.candidate["checkpoint_policy"] == CHECKPOINT_POLICY


def test_the_primary_metric_and_margin_are_frozen(paths: ProjectPaths):
    config = load_comparison_config(paths.configs / CONFIG_NAME)
    assert config["metrics"]["primary"] == PRIMARY_METRIC
    assert float(config["margin"]) == PRACTICAL_EQUIVALENCE_MARGIN


def test_the_all_class_metrics_are_required(paths: ProjectPaths):
    metrics = load_comparison_config(paths.configs / CONFIG_NAME)["metrics"]
    assert ALL_CLASS_METRIC in metrics["all_class"]
    assert ALL_CLASS_METRIC_50 in metrics["all_class"]


def test_the_native_metric_is_demoted_not_suppressed(paths: ProjectPaths):
    metrics = load_comparison_config(paths.configs / CONFIG_NAME)["metrics"]
    assert metrics["native_framework_status"] == NATIVE_METRIC_STATUS
    assert metrics["native_framework"]
    assert any("native mask mAP@0.50:0.95" in entry for entry in metrics["native_framework"])


def test_the_direct_iou_diagnostics_remain_secondary(paths: ProjectPaths):
    metrics = load_comparison_config(paths.configs / CONFIG_NAME)["metrics"]
    assert "GT_NORMALIZED_MASK_IOU" in metrics["secondary_direct_iou"]
    assert metrics["primary"] != "GT_NORMALIZED_MASK_IOU"


def test_no_composite_metric_exists(paths: ProjectPaths):
    config = load_comparison_config(paths.configs / CONFIG_NAME)
    assert config["metrics"]["composite_score"] is False
    assert config["disagreement_policy"]["create_composite"] is False
    assert config["disagreement_policy"]["invent_weights_after_the_fact"] is False


def test_the_candidate_carries_no_result(paths: ProjectPaths):
    config = load_comparison_config(paths.configs / CONFIG_NAME)
    assert config.candidate["status"] == S1_STATUS
    assert "result" not in config.candidate
    assert config["final_segmenter"] == "UNSELECTED"


def test_the_protocol_references_the_committed_evaluator(paths: ProjectPaths):
    config = load_comparison_config(paths.configs / CONFIG_NAME)
    protocol = load_canonical_evaluation_config(
        paths.configs / "segmentation_canonical_evaluation.yaml"
    )
    assert config["canonical_evaluation_sha256"] == protocol.fingerprint()


# --- refusals -----------------------------------------------------------------


def test_a_second_override_is_refused(tmp_path: Path, raw: dict[str, Any]):
    document = dict(raw)
    document["candidate"] = dict(raw["candidate"])
    document["candidate"]["overrides"] = {"overlap_mask": False, "mask_ratio": 2}
    with pytest.raises(ComparisonConfigError, match="must contain exactly"):
        load_comparison_config(write(tmp_path, document))


def test_an_override_that_changes_nothing_is_refused():
    with pytest.raises(ComparisonConfigError, match="declared but never applied"):
        verify_one_variable_contract({ONE_VARIABLE_FIELD: False}, {ONE_VARIABLE_FIELD: False})


def test_an_undeclared_override_is_refused_by_the_contract():
    with pytest.raises(ComparisonConfigError, match="overrides fields beyond"):
        verify_one_variable_contract(
            {ONE_VARIABLE_FIELD: True, "mask_ratio": 4},
            {ONE_VARIABLE_FIELD: False, "mask_ratio": 2},
        )


def test_promoting_the_native_metric_is_refused(tmp_path: Path, raw: dict[str, Any]):
    document = dict(raw)
    document["metrics"] = dict(raw["metrics"])
    document["metrics"]["native_framework_status"] = "PRIMARY_SELECTION_METRIC"
    with pytest.raises(ComparisonConfigError, match="native_framework_status"):
        load_comparison_config(write(tmp_path, document))


def test_a_relaxed_margin_is_refused(tmp_path: Path, raw: dict[str, Any]):
    document = dict(raw)
    document["margin"] = 0.02
    with pytest.raises(ComparisonConfigError, match="margin must be"):
        load_comparison_config(write(tmp_path, document))


def test_a_composite_score_is_refused(tmp_path: Path, raw: dict[str, Any]):
    document = dict(raw)
    document["metrics"] = dict(raw["metrics"])
    document["metrics"]["composite_score"] = True
    with pytest.raises(ComparisonConfigError, match="composite_score must be false"):
        load_comparison_config(write(tmp_path, document))


def test_a_candidate_result_is_refused(tmp_path: Path, raw: dict[str, Any]):
    document = dict(raw)
    document["candidate"] = dict(raw["candidate"])
    document["candidate"]["result"] = {"canonical_supported_macro": 0.9}
    with pytest.raises(ComparisonConfigError, match="unknown key"):
        load_comparison_config(write(tmp_path, document))


def test_claiming_the_protocol_predated_s0_is_refused(tmp_path: Path, raw: dict[str, Any]):
    document = dict(raw)
    document["protocol_timing"] = "PRE_S0_PROTOCOL_FREEZE"
    with pytest.raises(ComparisonConfigError, match="protocol_timing must be"):
        load_comparison_config(write(tmp_path, document))


def test_a_changed_checkpoint_policy_is_refused(tmp_path: Path, raw: dict[str, Any]):
    document = dict(raw)
    document["candidate"] = dict(raw["candidate"])
    document["candidate"]["checkpoint_policy"] = "MASK_ONLY_BEST_EPOCH"
    with pytest.raises(ComparisonConfigError, match="checkpoint_policy must be"):
        load_comparison_config(write(tmp_path, document))


def test_selecting_a_segmenter_here_is_refused(tmp_path: Path, raw: dict[str, Any]):
    document = dict(raw)
    document["final_segmenter"] = "S1"
    with pytest.raises(ComparisonConfigError, match="final_segmenter must be"):
        load_comparison_config(write(tmp_path, document))


def test_naming_the_holdout_is_refused(tmp_path: Path, raw: dict[str, Any]):
    document = dict(raw)
    document["test_policy"] = "test"
    with pytest.raises(ComparisonConfigError, match="protected split"):
        load_comparison_config(write(tmp_path, document))


# --- the committed artifacts --------------------------------------------------


@pytest.fixture(scope="module")
def canonical(paths: ProjectPaths) -> dict:
    path = paths.reports / CANONICAL_JSON
    if not path.is_file():
        pytest.skip(f"{CANONICAL_JSON} not present; run the phase 8E freeze")
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def policy(paths: ProjectPaths) -> dict:
    path = paths.reports / POLICY_JSON
    if not path.is_file():
        pytest.skip(f"{POLICY_JSON} not present; run the phase 8E freeze")
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def s1(paths: ProjectPaths) -> dict:
    path = paths.reports / S1_MANIFEST_JSON
    if not path.is_file():
        pytest.skip(f"{S1_MANIFEST_JSON} not present; run the phase 8E freeze")
    return json.loads(path.read_text(encoding="utf-8"))


def test_the_canonical_reference_used_the_frozen_checkpoint(canonical: dict):
    assert canonical["checkpoint"]["sha256"] == EXPECTED_S0_CHECKPOINT
    assert canonical["checkpoint"]["experiment"] == REFERENCE
    assert canonical["checkpoint"]["last_pt_used"] is False


def test_the_canonical_reference_used_canonical_ground_truth(canonical: dict):
    truth = canonical["ground_truth"]
    assert truth["source"] == GROUND_TRUTH_SOURCE
    assert truth["is_the_yolo_adapter"] is False
    assert "adapter" not in truth["document"]
    assert truth["images"] == 65
    assert truth["annotations"] == 304


def test_the_canonical_reference_used_the_frozen_settings(canonical: dict):
    assert canonical["inference"]["conf"] == 0.001
    assert canonical["inference"]["imgsz"] == 768
    assert canonical["inference"]["iou"] == 0.70
    assert canonical["inference"]["max_det"] == 300
    assert canonical["inference"]["augment"] is False
    assert tuple(canonical["cocoeval"]["iou_thresholds"]) == IOU_THRESHOLDS
    assert tuple(canonical["cocoeval"]["max_dets"]) == MAX_DETS
    assert canonical["cocoeval"]["iou_type"] == "segm"


def test_the_prediction_cap_and_metric_cap_are_distinguished(canonical: dict):
    assert canonical["inference"]["max_det"] == 300
    assert 300 not in canonical["cocoeval"]["max_dets"]
    assert "conventional cap" in canonical["prediction_cap_versus_metric_cap"]


def test_the_canonical_reference_ran_exactly_once(canonical: dict):
    assert canonical["runs"] == 1
    assert canonical["timing"].startswith("POST_S0_PRE_S1")


def test_the_canonical_metrics_are_present_and_per_class(canonical: dict):
    result = canonical["canonical"]
    assert isinstance(result["all_class_map50_95"], float)
    assert isinstance(result["all_class_map50"], float)
    expected = {"helmet_loose", "helmet_on_head", "person", "vest_loose", "vest_on_body"}
    assert set(result["per_class"]) == expected
    for entry in result["per_class"].values():
        assert "AP@0.50:0.95" in entry
        assert "AP@0.50" in entry


def test_the_supported_macro_reproduces_from_the_per_class_values(canonical: dict):
    macro = canonical["supported_macro"]
    per_class = canonical["canonical"]["per_class"]
    values = [per_class[name]["AP@0.50:0.95"] for name in macro["admitted_classes"]]
    assert values
    assert macro["value"] == pytest.approx(sum(values) / len(values), abs=1e-6)
    assert macro["metric"] == PRIMARY_METRIC


def test_vest_loose_is_excluded_mechanically(canonical: dict):
    assert canonical["support"]["vest_loose"]["supported"] is False
    assert "vest_loose" not in canonical["supported_macro"]["admitted_classes"]
    # Still reported in full.
    assert "vest_loose" in canonical["canonical"]["per_class"]
    assert canonical["supported_macro"]["names_no_class"] is True


def test_the_s0_native_and_direct_results_are_preserved(paths: ProjectPaths, canonical: dict):
    manifest = json.loads(
        (paths.reports / "segmentation_S0_result_manifest.json").read_text(encoding="utf-8")
    )
    direct = json.loads(
        (paths.reports / "segmentation_S0_mask_iou.json").read_text(encoding="utf-8")
    )
    assert (
        canonical["native_reference"]["mask_map50_95"] == manifest["mask_metrics"]["mAP@0.50:0.95"]
    )
    assert canonical["native_reference"]["still_valid"] is True
    assert (
        canonical["direct_iou_reference"]["gt_normalized_mask_iou"]
        == direct["global"]["gt_normalized_mask_iou"]
    )
    assert canonical["direct_iou_reference"]["still_valid"] is True
    assert canonical["direct_iou_reference"]["rerun_in_this_phase"] is False
    assert canonical["s0_retrained"] is False
    assert canonical["s0_native_metrics_regenerated"] is False


def test_the_phase_8d_rescore_is_not_promoted(canonical: dict):
    assert canonical["phase_8d_rescore_status"] == "HYPOTHESIS_GENERATING_DIAGNOSTIC_ONLY"


def test_the_policy_records_its_timing_and_reference_values(policy: dict):
    assert policy["frozen_before_s1"] is True
    assert policy["frozen_before_s0"] is False
    assert policy["candidate_values"] is None
    assert policy["reference_values"]["canonical_supported_macro_mask_map50_95"] is not None
    assert policy["final_segmenter"] == "UNSELECTED"


def test_the_policy_declares_all_three_cases(policy: dict):
    for case in ("improves", "practically_equivalent", "below"):
        assert case in policy["selection_logic"]
    assert policy["margin_is_not_a_significance_test"] is True


def test_the_s1_manifest_is_frozen_not_executed(s1: dict):
    assert s1["experiment_id"] == CANDIDATE
    assert s1["status"] == S1_STATUS
    assert s1["models_trained_in_this_phase"] == 0
    assert s1["final_segmenter"] == "UNSELECTED"
    assert "result" not in s1
    assert "metrics" not in s1


def test_the_s1_manifest_names_the_same_pretrained_binary(s1: dict):
    weights = s1["pretrained_weights"]
    assert weights["sha256"] == EXPECTED_PRETRAINED
    assert weights["size_bytes"] == 6182636
    assert weights["identical_to_s0"] is True
    assert weights["committed"] is False


def test_the_s1_manifest_keeps_the_adapter_fingerprints(paths: ProjectPaths, s1: dict):
    approval = json.loads(
        (paths.reports / "segmentation_adapter_approval.json").read_text(encoding="utf-8")
    )
    assert s1["adapter"]["fingerprints"] == approval["adapter_fingerprints"]
    assert s1["adapter"]["regenerated"] is False


def test_the_s1_manifest_inherits_mask_ratio_four(s1: dict):
    assert s1["inherited_protocol"]["mask_ratio"] == 4
    assert s1["inherited_protocol"]["imgsz"] == 768
    assert s1["inherited_protocol"]["batch"] == 8
    assert s1["inherited_protocol"]["epochs"] == 100
    assert s1["inherited_protocol"]["seed"] == 42
    assert ONE_VARIABLE_FIELD not in s1["inherited_protocol"]


def test_the_s1_checkpoint_policy_is_the_native_composite(s1: dict):
    assert s1["checkpoint_policy"] == CHECKPOINT_POLICY
    assert (
        "mask-only checkpoint selector for either experiment is not authorised"
        in s1["checkpoint_semantics"]
    )


def test_the_feasibility_check_recorded_no_accuracy_metric(s1: dict):
    feasibility = s1["memory_feasibility"]
    assert feasibility["classification"] == "NON_EXPERIMENTAL"
    assert feasibility["metrics_recorded"] is False
    assert feasibility["optimizer_step_taken"] is False
    assert feasibility["validation_run"] is False
    assert feasibility["checkpoint_written"] is False
    assert feasibility["out_of_memory"] is False
    assert feasibility["batch"] == 8
    assert feasibility["overlap_mask"] is False
    serialised = json.dumps(feasibility)
    for banned in ("mAP", "AP@", "precision", "recall", "IoU"):
        assert banned not in serialised, f"the feasibility record leaks {banned}"


def test_the_artifacts_carry_no_sensitive_content(paths: ProjectPaths, canonical, policy, s1):
    for document in (canonical, policy, s1):
        assert scan_for_sensitive(json.dumps(document)) == []
    for name in (REFERENCE_MD, POLICY_MD, S1_MD):
        path = paths.reports / name
        if path.is_file():
            assert scan_for_sensitive(path.read_text(encoding="utf-8")) == []


def test_no_artifact_names_the_holdout(canonical: dict, policy: dict, s1: dict):
    for document in (canonical, policy, s1):
        serialised = json.dumps(document)
        assert "images/test" not in serialised
        assert "labels/test" not in serialised
        assert HOLDOUT_UNLOCK_ENV_VAR not in serialised
        assert document["test"]["status"] == "PROTECTED_NOT_ACCESSED"


def test_the_historical_artifacts_are_unchanged(paths: ProjectPaths):
    # The phase 8C S0 report and manifest in particular must be byte-identical:
    # this phase adds a measurement beside them and revises nothing.
    expected = {
        "reports/segmentation_S0_result_manifest.json": (
            "c3378ce47cecd53c4d7b1df433581d201a554d9667b78e4ab4aafa2deb4a31aa"
        ),
        "reports/segmentation_S0_mask_iou.json": (
            "867fe4441a7097974ead9d2df788379c9f72a612b2a7a5852a23fa3c18a049ba"
        ),
        "reports/segmentation_S0_report.md": (
            "6170b7dbe6ea99c8fbd170abbac430311110f5871bcab9327ca1fed0860191d8"
        ),
    }
    for name, digest in expected.items():
        assert sha256_file(paths.root / name) == digest, f"{name} changed"


def test_the_frozen_s1_protocol_still_records_that_it_was_not_executed(paths: ProjectPaths):
    """The phase 8E protocol artifact is historical and never absorbs a result.

    Until phase 8F ran, this was asserted by the absence of any S1 result file.
    That assertion has served its purpose and would now be false by design, so
    what it protected is stated directly instead: phase 8E's own artifacts must
    still describe a protocol that had not been executed when they were written,
    and S1's result must live in separate phase 8F artifacts rather than being
    written back into them.
    """
    manifest = json.loads((paths.reports / S1_MANIFEST_JSON).read_text(encoding="utf-8"))

    assert manifest["status"] == S1_STATUS
    assert manifest["phase"] == "8E"
    assert manifest["models_trained_in_this_phase"] == 0
    assert manifest["final_segmenter"] == "UNSELECTED"
    for forbidden in ("result", "metrics", "canonical_metrics", "primary_delta"):
        assert forbidden not in manifest, forbidden
