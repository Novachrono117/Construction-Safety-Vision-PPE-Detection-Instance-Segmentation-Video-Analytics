"""Tests for strict experiment-configuration parsing."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from construction_safety_vision.config import (
    ConfigError,
    DatasetCandidate,
    ExperimentConfig,
    SplitRatios,
    load_experiment_config,
)
from construction_safety_vision.paths import ProjectPaths

VALID_CONFIG = textwrap.dedent(
    """
    project_name: unit-test
    seed: 7
    split_ratios:
      train: 0.7
      val: 0.15
      test: 0.15
    dataset:
      name: Example dataset
      source: roboflow_universe
      canonical_task: instance_segmentation
      expected_classes: [person, helmet_on_head]
    """
).strip()


def write_config(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(body, encoding="utf-8", newline="\n")
    return path


def test_loads_a_valid_configuration(tmp_path: Path) -> None:
    config = load_experiment_config(write_config(tmp_path, VALID_CONFIG))

    assert config.project_name == "unit-test"
    assert config.seed == 7
    assert config.holdout_split == "test"
    assert config.split_ratios.as_dict() == {"train": 0.7, "val": 0.15, "test": 0.15}
    assert config.dataset.expected_classes == ("person", "helmet_on_head")
    # Dataset claims are hypotheses until the audit phase says otherwise.
    assert config.dataset.verified is False


def test_unknown_keys_are_rejected(tmp_path: Path) -> None:
    # A silently ignored typo would mean the report describes a different
    # experiment than the one that ran.
    body = f"{VALID_CONFIG}\nseeed: 13\n"
    with pytest.raises(ConfigError, match="unknown key"):
        load_experiment_config(write_config(tmp_path, body))


def test_missing_keys_are_rejected(tmp_path: Path) -> None:
    body = VALID_CONFIG.replace("seed: 7\n", "")
    with pytest.raises(ConfigError, match="missing required key"):
        load_experiment_config(write_config(tmp_path, body))


def test_missing_file_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_experiment_config(tmp_path / "absent.yaml")


def test_non_mapping_document_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="top-level mapping"):
        load_experiment_config(write_config(tmp_path, "- just\n- a list\n"))


def test_invalid_yaml_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not valid YAML"):
        load_experiment_config(write_config(tmp_path, "project_name: [unclosed\n"))


@pytest.mark.parametrize(
    "ratios",
    [
        {"train": 0.7, "val": 0.2, "test": 0.2},
        {"train": 0.6, "val": 0.2, "test": 0.1},
    ],
)
def test_split_ratios_must_sum_to_one(ratios: dict[str, float]) -> None:
    with pytest.raises(ConfigError, match=r"sum to 1\.0"):
        SplitRatios.from_mapping(ratios)


@pytest.mark.parametrize(
    "ratios",
    [
        {"train": 0.0, "val": 0.5, "test": 0.5},
        {"train": 1.0, "val": 0.0, "test": 0.0},
        {"train": -0.2, "val": 0.6, "test": 0.6},
    ],
)
def test_every_split_must_be_non_empty(ratios: dict[str, float]) -> None:
    with pytest.raises(ConfigError, match="open interval"):
        SplitRatios.from_mapping(ratios)


def test_split_ratios_reject_non_numeric_values() -> None:
    with pytest.raises(ConfigError, match="numeric"):
        SplitRatios.from_mapping({"train": "most", "val": 0.15, "test": 0.15})


def test_unknown_canonical_task_is_rejected() -> None:
    with pytest.raises(ConfigError, match="canonical_task"):
        DatasetCandidate.from_mapping(
            {
                "name": "Example",
                "source": "roboflow_universe",
                "canonical_task": "classification",
                "expected_classes": ["person"],
            }
        )


def test_duplicate_class_names_are_rejected() -> None:
    with pytest.raises(ConfigError, match="duplicates"):
        DatasetCandidate.from_mapping(
            {
                "name": "Example",
                "source": "roboflow_universe",
                "canonical_task": "instance_segmentation",
                "expected_classes": ["person", "person"],
            }
        )


def test_empty_class_list_is_rejected() -> None:
    with pytest.raises(ConfigError, match="must not be empty"):
        DatasetCandidate.from_mapping(
            {
                "name": "Example",
                "source": "roboflow_universe",
                "canonical_task": "instance_segmentation",
                "expected_classes": [],
            }
        )


def test_class_list_given_as_string_is_rejected() -> None:
    with pytest.raises(ConfigError, match="list of class names"):
        DatasetCandidate.from_mapping(
            {
                "name": "Example",
                "source": "roboflow_universe",
                "canonical_task": "instance_segmentation",
                "expected_classes": "person",
            }
        )


def test_to_dict_is_json_serialisable(tmp_path: Path) -> None:
    import json

    config = load_experiment_config(write_config(tmp_path, VALID_CONFIG))
    payload = json.loads(json.dumps(config.to_dict()))

    assert payload["dataset"]["expected_classes"] == ["person", "helmet_on_head"]
    assert payload["source_path"].endswith("config.yaml")


def test_shipped_project_config_is_valid_and_consistent() -> None:
    # The configuration the repository actually ships must load, and its holdout
    # name must match the split ratios it declares.
    paths = ProjectPaths.from_root()
    config = load_experiment_config(paths.configs / "project.yaml")

    assert isinstance(config, ExperimentConfig)
    assert config.holdout_split in config.split_ratios.as_dict()
    assert config.dataset.canonical_task == "instance_segmentation"
    assert config.dataset.verified is False, "no dataset has been audited yet"
