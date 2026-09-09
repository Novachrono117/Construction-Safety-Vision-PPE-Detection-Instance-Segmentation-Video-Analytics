"""Tests for the S0 segmentation protocol schema.

The loader's job is to refuse protocols that would let a later phase cheat, so
the tests are mostly about refusal: a holdout reference, a promoted box metric, a
composite score, a filtered instance population, a digest that is not a digest.

The committed configuration is also parsed here, because a schema nothing valid
satisfies is not a schema.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
import yaml

from construction_safety_vision.paths import ProjectPaths
from construction_safety_vision.segmentation_experiment import (
    ADAPTER_ROLE,
    CANONICAL_GROUND_TRUTH,
    CHECKPOINT_RULE,
    CHECKPOINT_SELECTION_POLICY,
    CHECKPOINT_SELECTION_REVIEW_STATUS,
    CHECKPOINT_SELECTION_SEMANTICS,
    PRIMARY_METRIC,
    PRIMARY_SCIENTIFIC_REPORTING_METRIC,
    SUPPORTED_MACRO_MASK_METRIC,
    SegmentationExperimentConfigError,
    load_segmentation_baseline_config,
)

CONFIG_NAME = "segmentation_baseline.yaml"


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
    config = load_segmentation_baseline_config(paths.configs / CONFIG_NAME)
    assert config.experiment_id == "S0"
    assert config.task == "segmentation"


def test_the_selected_architecture_is_yolo11n_seg(paths: ProjectPaths):
    config = load_segmentation_baseline_config(paths.configs / CONFIG_NAME)
    assert config.architecture == "YOLO11n-seg"
    assert config.model == "YOLO11n-seg"
    assert config.weight_identifier == "yolo11n-seg.pt"


def test_the_fallback_is_recorded_as_not_selected(paths: ProjectPaths):
    config = load_segmentation_baseline_config(paths.configs / CONFIG_NAME)
    assert config.fallback_status == "NOT_SELECTED_FALLBACK"
    assert "Mask R-CNN" in config.fallback_architecture


def test_the_frozen_execution_settings(paths: ProjectPaths):
    config = load_segmentation_baseline_config(paths.configs / CONFIG_NAME)
    assert config.training["imgsz"] == 768
    assert config.training["batch"] == 8
    assert config.training["epochs"] == 100
    assert config.training["patience"] == 50
    assert config.training["optimizer"] == "auto"
    assert config.training["amp"] is True
    assert config.training["deterministic"] is True
    assert config.seed == 42


def test_the_canonical_ground_truth_is_coco(paths: ProjectPaths):
    config = load_segmentation_baseline_config(paths.configs / CONFIG_NAME)
    assert config.canonical_ground_truth == CANONICAL_GROUND_TRUTH


def test_the_yolo_adapter_is_marked_derived(paths: ProjectPaths):
    config = load_segmentation_baseline_config(paths.configs / CONFIG_NAME)
    assert config.adapter["role"] == ADAPTER_ROLE
    assert config.adapter["format"] == "YOLO_SEGMENTATION"


def test_every_development_instance_is_retained(paths: ProjectPaths):
    config = load_segmentation_baseline_config(paths.configs / CONFIG_NAME)
    assert config.adapter["instances"] == 1726
    assert config.adapter["train_instances"] == 1422
    assert config.adapter["validation_instances"] == 304
    assert config.adapter["train_images"] == 303
    assert config.adapter["validation_images"] == 65
    assert config.adapter["negative_images"] == 12
    assert config.adapter["all_instances_retained"] is True
    assert config.adapter["fidelity_based_filtering"] == "NONE"


def test_the_protocol_names_only_development_splits(paths: ProjectPaths):
    config = load_segmentation_baseline_config(paths.configs / CONFIG_NAME)
    assert config.splits == ("train", "validation")
    assert config.test_policy == "PROTECTED_NOT_ACCESSED"


def test_the_checkpoint_rule_is_named(paths: ProjectPaths):
    config = load_segmentation_baseline_config(paths.configs / CONFIG_NAME)
    assert CHECKPOINT_RULE in config.checkpoint_selection


def test_the_metric_hierarchy_separates_mask_from_box(paths: ProjectPaths):
    config = load_segmentation_baseline_config(paths.configs / CONFIG_NAME)
    assert config.metrics.primary == PRIMARY_METRIC
    assert config.metrics.supported_macro == SUPPORTED_MACRO_MASK_METRIC
    assert all(name.startswith("mask_") for name in config.metrics.secondary_mask)
    assert all(name.startswith("mask_") for name in config.metrics.per_class_mask)
    assert all(name.startswith("box_") for name in config.metrics.box_from_segmenter)
    assert all(name.startswith("box_") for name in config.metrics.per_class_box)
    assert config.metrics.as_dict()["composite_box_mask_score"] is False


def test_the_segmentation_arguments_are_declared_explicitly(paths: ProjectPaths):
    config = load_segmentation_baseline_config(paths.configs / CONFIG_NAME)
    for name in ("overlap_mask", "mask_ratio", "retina_masks", "dropout", "max_det", "workers"):
        assert name in config.segmentation_arguments or name in config.training


def test_the_rare_class_policy_is_the_frozen_one(paths: ProjectPaths):
    config = load_segmentation_baseline_config(paths.configs / CONFIG_NAME)
    assert config.rare_class == "vest_loose"
    assert "DESCRIPTIVE_HIGH_UNCERTAINTY" in config.rare_class_limitation
    assert "reported in full" in config.rare_class_limitation


def test_the_dataset_descriptor_is_a_relative_path(paths: ProjectPaths):
    config = load_segmentation_baseline_config(paths.configs / CONFIG_NAME)
    assert not Path(config.dataset_config).is_absolute()
    assert ":" not in config.dataset_config


def test_the_protocol_fingerprint_is_stable(paths: ProjectPaths):
    first = load_segmentation_baseline_config(paths.configs / CONFIG_NAME)
    second = load_segmentation_baseline_config(paths.configs / CONFIG_NAME)
    assert first.fingerprint() == second.fingerprint()


def test_the_assembled_arguments_merge_all_three_blocks(paths: ProjectPaths):
    config = load_segmentation_baseline_config(paths.configs / CONFIG_NAME)
    arguments = config.training_arguments()
    assert arguments["imgsz"] == 768
    assert arguments["overlap_mask"] is True
    assert arguments["mosaic"] == 1.0
    assert arguments["seed"] == 42
    assert "augmentation" not in arguments


# --- the reviewed checkpoint-selection decision -------------------------------


def test_the_checkpoint_policy_is_the_native_ultralytics_segmentation_fitness(
    paths: ProjectPaths,
):
    config = load_segmentation_baseline_config(paths.configs / CONFIG_NAME)
    assert config.checkpoint_selection_policy == CHECKPOINT_SELECTION_POLICY
    assert config.checkpoint_selection_policy == "ULTRALYTICS_NATIVE_SEGMENTATION_FITNESS"


def test_the_checkpoint_semantics_name_both_box_and_mask(paths: ProjectPaths):
    config = load_segmentation_baseline_config(paths.configs / CONFIG_NAME)
    assert config.checkpoint_selection_semantics == CHECKPOINT_SELECTION_SEMANTICS
    assert config.checkpoint_selection_semantics == "BOX_MAP50_95_PLUS_MASK_MAP50_95"
    assert config.checkpoint_selection_box_component_weight == 1.0
    assert config.checkpoint_selection_mask_component_weight == 1.0


def test_the_decision_was_reviewed_before_s0(paths: ProjectPaths):
    config = load_segmentation_baseline_config(paths.configs / CONFIG_NAME)
    assert config.checkpoint_selection_review_status == CHECKPOINT_SELECTION_REVIEW_STATUS
    assert config.checkpoint_selection_review_status == "HUMAN_REVIEWED_AND_ACCEPTED_BEFORE_S0"


def test_the_primary_reporting_metric_is_mask_only_and_differs_from_the_selector(
    paths: ProjectPaths,
):
    config = load_segmentation_baseline_config(paths.configs / CONFIG_NAME)
    assert config.primary_scientific_reporting_metric == PRIMARY_SCIENTIFIC_REPORTING_METRIC
    assert config.primary_scientific_reporting_metric == "MASK_MAP50_95"
    assert config.selection_metric_equals_primary_reporting_metric is False
    # The headline metric in the hierarchy is unchanged by the decision.
    assert config.metrics.primary == PRIMARY_METRIC


def test_the_rationale_claims_no_superiority(paths: ProjectPaths):
    config = load_segmentation_baseline_config(paths.configs / CONFIG_NAME)
    assert config.checkpoint_selection_rationale
    joined = " ".join(config.checkpoint_selection_rationale)
    assert "NOT a claim that the composite fitness is scientifically superior" in joined


def test_future_comparisons_inherit_the_same_checkpoint_policy(paths: ProjectPaths):
    config = load_segmentation_baseline_config(paths.configs / CONFIG_NAME)
    constraint = config.checkpoint_selection_future_constraint
    assert CHECKPOINT_SELECTION_POLICY in constraint
    assert "frozen BEFORE any affected experiment is run" in constraint


def test_relabelling_the_checkpoint_policy_is_refused(tmp_path: Path, raw: dict[str, Any]):
    document = dict(raw)
    document["checkpoint_selection_policy"] = "MASK_ONLY_BEST_EPOCH"
    with pytest.raises(SegmentationExperimentConfigError, match="checkpoint_selection_policy"):
        load_segmentation_baseline_config(write(tmp_path, document))


def test_dropping_the_box_component_is_refused(tmp_path: Path, raw: dict[str, Any]):
    document = dict(raw)
    document["checkpoint_selection_box_component_weight"] = 0.0
    with pytest.raises(SegmentationExperimentConfigError, match=re.escape("must be 1.0")):
        load_segmentation_baseline_config(write(tmp_path, document))


def test_claiming_the_selector_equals_the_reporting_metric_is_refused(
    tmp_path: Path, raw: dict[str, Any]
):
    document = dict(raw)
    document["selection_metric_equals_primary_reporting_metric"] = True
    message = re.escape("selection_metric_equals_primary_reporting_metric must be")
    with pytest.raises(SegmentationExperimentConfigError, match=message):
        load_segmentation_baseline_config(write(tmp_path, document))


def test_changing_the_primary_reporting_metric_is_refused(tmp_path: Path, raw: dict[str, Any]):
    document = dict(raw)
    document["primary_scientific_reporting_metric"] = "BOX_PLUS_MASK_MAP50_95"
    with pytest.raises(
        SegmentationExperimentConfigError, match="primary_scientific_reporting_metric"
    ):
        load_segmentation_baseline_config(write(tmp_path, document))


def test_a_superiority_claim_in_the_rationale_is_refused(tmp_path: Path, raw: dict[str, Any]):
    document = dict(raw)
    document["checkpoint_selection_rationale"] = [
        "The composite fitness is superior to mask-only selection."
    ]
    with pytest.raises(SegmentationExperimentConfigError, match="claims the composite"):
        load_segmentation_baseline_config(write(tmp_path, document))


def test_dropping_the_future_comparison_constraint_is_refused(tmp_path: Path, raw: dict[str, Any]):
    document = dict(raw)
    document["checkpoint_selection_future_constraint"] = "decide later"
    with pytest.raises(SegmentationExperimentConfigError, match="must name"):
        load_segmentation_baseline_config(write(tmp_path, document))


# --- refusals -----------------------------------------------------------------


def test_a_missing_file_is_refused(tmp_path: Path):
    with pytest.raises(SegmentationExperimentConfigError, match="not found"):
        load_segmentation_baseline_config(tmp_path / "absent.yaml")


def test_an_unknown_top_level_key_is_refused(tmp_path: Path, raw: dict[str, Any]):
    document = dict(raw)
    document["surprise"] = True
    with pytest.raises(SegmentationExperimentConfigError, match="unknown key"):
        load_segmentation_baseline_config(write(tmp_path, document))


def test_a_missing_hyperparameter_is_refused(tmp_path: Path, raw: dict[str, Any]):
    document = dict(raw)
    document["training"] = {
        key: value for key, value in raw["training"].items() if key != "weight_decay"
    }
    with pytest.raises(SegmentationExperimentConfigError, match="weight_decay"):
        load_segmentation_baseline_config(write(tmp_path, document))


def test_a_missing_segmentation_argument_is_refused(tmp_path: Path, raw: dict[str, Any]):
    document = dict(raw)
    document["segmentation_arguments"] = {
        key: value for key, value in raw["segmentation_arguments"].items() if key != "overlap_mask"
    }
    with pytest.raises(SegmentationExperimentConfigError, match="overlap_mask"):
        load_segmentation_baseline_config(write(tmp_path, document))


def test_a_missing_augmentation_argument_is_refused(tmp_path: Path, raw: dict[str, Any]):
    document = dict(raw)
    document["augmentation_arguments"] = {
        key: value for key, value in raw["augmentation_arguments"].items() if key != "mosaic"
    }
    with pytest.raises(SegmentationExperimentConfigError, match="mosaic"):
        load_segmentation_baseline_config(write(tmp_path, document))


def test_naming_the_holdout_in_the_splits_is_refused(tmp_path: Path, raw: dict[str, Any]):
    document = dict(raw)
    document["splits"] = ["train", "validation", "test"]
    with pytest.raises(SegmentationExperimentConfigError, match="splits must be exactly"):
        load_segmentation_baseline_config(write(tmp_path, document))


def test_naming_the_holdout_anywhere_is_refused(tmp_path: Path, raw: dict[str, Any]):
    document = dict(raw)
    document["test_policy"] = "test"
    with pytest.raises(SegmentationExperimentConfigError, match="protected split"):
        load_segmentation_baseline_config(write(tmp_path, document))


def test_a_promoted_box_metric_is_refused(tmp_path: Path, raw: dict[str, Any]):
    document = dict(raw)
    document["metrics"] = dict(raw["metrics"])
    document["metrics"]["secondary_mask"] = [*raw["metrics"]["secondary_mask"], "box_mAP@0.50"]
    with pytest.raises(SegmentationExperimentConfigError, match="mask metrics only"):
        load_segmentation_baseline_config(write(tmp_path, document))


def test_a_box_primary_metric_is_refused(tmp_path: Path, raw: dict[str, Any]):
    document = dict(raw)
    document["metrics"] = dict(raw["metrics"])
    document["metrics"]["primary"] = "box_mAP@0.50:0.95"
    message = re.escape("metrics.primary must be")
    with pytest.raises(SegmentationExperimentConfigError, match=message):
        load_segmentation_baseline_config(write(tmp_path, document))


def test_a_composite_box_mask_score_is_refused(tmp_path: Path, raw: dict[str, Any]):
    document = dict(raw)
    document["metrics"] = dict(raw["metrics"])
    document["metrics"]["composite_box_mask_score"] = True
    with pytest.raises(SegmentationExperimentConfigError, match="must be false"):
        load_segmentation_baseline_config(write(tmp_path, document))


def test_a_renamed_supported_macro_is_refused(tmp_path: Path, raw: dict[str, Any]):
    document = dict(raw)
    document["metrics"] = dict(raw["metrics"])
    document["metrics"]["supported_macro"] = "macro_mask_map"
    with pytest.raises(SegmentationExperimentConfigError, match="supported_macro must be"):
        load_segmentation_baseline_config(write(tmp_path, document))


def test_filtering_instances_is_refused(tmp_path: Path, raw: dict[str, Any]):
    document = dict(raw)
    document["adapter"] = dict(raw["adapter"])
    document["adapter"]["all_instances_retained"] = False
    with pytest.raises(SegmentationExperimentConfigError, match="all_instances_retained"):
        load_segmentation_baseline_config(write(tmp_path, document))


def test_declaring_a_fidelity_filter_is_refused(tmp_path: Path, raw: dict[str, Any]):
    document = dict(raw)
    document["adapter"] = dict(raw["adapter"])
    document["adapter"]["fidelity_based_filtering"] = "IOU_BELOW_0.90"
    with pytest.raises(SegmentationExperimentConfigError, match="fidelity_based_filtering"):
        load_segmentation_baseline_config(write(tmp_path, document))


def test_inconsistent_adapter_cardinality_is_refused(tmp_path: Path, raw: dict[str, Any]):
    document = dict(raw)
    document["adapter"] = dict(raw["adapter"])
    document["adapter"]["instances"] = 1700
    with pytest.raises(SegmentationExperimentConfigError, match="do not sum"):
        load_segmentation_baseline_config(write(tmp_path, document))


def test_a_non_digest_fingerprint_is_refused(tmp_path: Path, raw: dict[str, Any]):
    document = dict(raw)
    document["adapter"] = dict(raw["adapter"])
    document["adapter"]["labels_train_sha256"] = "the audited labels"
    with pytest.raises(SegmentationExperimentConfigError, match="not a SHA-256 digest"):
        load_segmentation_baseline_config(write(tmp_path, document))


def test_promoting_the_adapter_to_canonical_is_refused(tmp_path: Path, raw: dict[str, Any]):
    document = dict(raw)
    document["adapter"] = dict(raw["adapter"])
    document["adapter"]["role"] = "CANONICAL"
    with pytest.raises(SegmentationExperimentConfigError, match=re.escape("adapter.role must be")):
        load_segmentation_baseline_config(write(tmp_path, document))


def test_changing_the_canonical_ground_truth_is_refused(tmp_path: Path, raw: dict[str, Any]):
    document = dict(raw)
    document["canonical_ground_truth"] = "YOLO_SEGMENTATION"
    with pytest.raises(SegmentationExperimentConfigError, match="canonical_ground_truth"):
        load_segmentation_baseline_config(write(tmp_path, document))


def test_dropping_the_checkpoint_rule_is_refused(tmp_path: Path, raw: dict[str, Any]):
    document = dict(raw)
    document["checkpoint_selection"] = "whichever epoch looks best"
    with pytest.raises(SegmentationExperimentConfigError, match="checkpoint_selection must name"):
        load_segmentation_baseline_config(write(tmp_path, document))


def test_a_tuned_augmentation_policy_name_is_refused(tmp_path: Path, raw: dict[str, Any]):
    document = dict(raw)
    document["augmentation_policy"] = "TUNED_FOR_SEGMENTATION"
    with pytest.raises(SegmentationExperimentConfigError, match="augmentation_policy must be"):
        load_segmentation_baseline_config(write(tmp_path, document))


def test_a_duplicated_framework_argument_is_refused(tmp_path: Path, raw: dict[str, Any]):
    document = dict(raw)
    document["segmentation_arguments"] = dict(raw["segmentation_arguments"])
    document["segmentation_arguments"]["imgsz"] = 768
    with pytest.raises(SegmentationExperimentConfigError, match="unknown key"):
        load_segmentation_baseline_config(write(tmp_path, document))


def test_a_bad_experiment_id_is_refused(tmp_path: Path, raw: dict[str, Any]):
    document = dict(raw)
    document["experiment_id"] = "segmentation baseline"
    with pytest.raises(SegmentationExperimentConfigError, match="not a valid identifier"):
        load_segmentation_baseline_config(write(tmp_path, document))


def test_a_wrong_task_is_refused(tmp_path: Path, raw: dict[str, Any]):
    document = dict(raw)
    document["task"] = "detection"
    with pytest.raises(SegmentationExperimentConfigError, match="task must be"):
        load_segmentation_baseline_config(write(tmp_path, document))
