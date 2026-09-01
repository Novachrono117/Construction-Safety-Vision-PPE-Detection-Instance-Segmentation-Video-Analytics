"""Tests for the source-versus-generated version analysis.

This is the logic behind the phase's central conclusion - how many independent
images the project actually has - so it is tested against the real numbers and
against the cases where it must refuse to conclude.
"""

from __future__ import annotations

import pytest

from construction_safety_vision.data.versioning import analyse_source_vs_generated

# The real figures reported by the provider for this dataset.
PROJECT_SPLITS = {"train": 306, "valid": 87, "test": 43}
VERSION_SPLITS = {"train": 612, "valid": 87, "test": 43}


def test_real_dataset_is_classified_as_source_plus_generated() -> None:
    result = analyse_source_vs_generated(PROJECT_SPLITS, VERSION_SPLITS, 2)

    assert result["conclusion"] == "B_source_plus_generated"
    assert result["generated_splits"] == ["train"]
    assert result["unchanged_splits"] == ["test", "valid"]
    assert result["independent_source_images"] == 436
    assert result["version_images"] == 742


def test_independent_population_is_never_the_version_count() -> None:
    # The whole point: 742 must not be reported as 742 independent samples.
    result = analyse_source_vs_generated(PROJECT_SPLITS, VERSION_SPLITS, 2)
    assert result["independent_source_images"] != result["version_images"]
    assert "NOT" in result["summary"]


def test_unaugmented_version_is_classified_as_source_only() -> None:
    result = analyse_source_vs_generated(PROJECT_SPLITS, dict(PROJECT_SPLITS), None)

    assert result["conclusion"] == "A_source_only"
    assert result["generated_splits"] == []
    assert result["independent_source_images"] == 436


def test_unexplained_ratio_refuses_to_conclude() -> None:
    # A count the declared multiplier cannot explain must not be rationalised.
    weird = {"train": 700, "valid": 87, "test": 43}
    result = analyse_source_vs_generated(PROJECT_SPLITS, weird, 2)

    assert result["conclusion"] == "C_evidence_insufficient"
    assert result["per_split"]["train"]["status"] == "unexplained"
    assert result["per_split"]["train"]["ratio"] == pytest.approx(700 / 306)


def test_missing_multiplier_cannot_justify_a_larger_split() -> None:
    result = analyse_source_vs_generated(PROJECT_SPLITS, VERSION_SPLITS, None)
    assert result["conclusion"] == "C_evidence_insufficient"


def test_multiplier_of_three_is_recognised() -> None:
    tripled = {"train": 918, "valid": 87, "test": 43}
    result = analyse_source_vs_generated(PROJECT_SPLITS, tripled, 3)

    assert result["conclusion"] == "B_source_plus_generated"
    assert result["per_split"]["train"]["multiplier"] == 3


def test_only_shared_splits_are_compared() -> None:
    result = analyse_source_vs_generated({"train": 10}, {"train": 20, "extra": 5}, 2)
    assert set(result["per_split"]) == {"train"}


def test_empty_metadata_is_reported_as_source_only_without_inventing_numbers() -> None:
    result = analyse_source_vs_generated({}, {}, None)
    assert result["independent_source_images"] == 0
    assert result["version_images"] == 0
