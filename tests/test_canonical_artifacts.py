"""Checks on the phase 5A artifacts that are actually committed.

The other test modules exercise the logic on synthetic inputs. These check the
real committed files, because that is what a reader will read and what the later
phases will build on: the population must still be 436, the annotations must
still add up, and nothing unpublishable must have crept into a report.

Each test skips when its artifact is absent, so a fresh clone that has not run
the phase 5A scripts is not reported as broken.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from construction_safety_vision.data.canonical import (
    CURRENT_COMPLETE_GEOMETRY,
    FROZEN_V4_SNAPSHOT,
    PHASE_CLASSIFICATION,
    scan_for_sensitive,
    validate_manifest,
)
from construction_safety_vision.data.sourcegeometry import GEOMETRY_KINDS, UNSUPPORTED
from construction_safety_vision.paths import ProjectPaths

SOURCE_IMAGES = 436
"""The independent source population every phase must preserve."""

REPORTS = ProjectPaths.from_root().reports
"""Directory holding the committed evidence."""


def load(name: str) -> dict:
    """Load a committed JSON report, skipping the test when it is absent.

    Args:
        name: Filename under ``reports``.

    Returns:
        The parsed document.
    """
    path = REPORTS / name
    if not path.is_file():
        pytest.skip(f"{name} not generated yet")
    return json.loads(path.read_text(encoding="utf-8"))


def load_rows(name: str) -> list[dict]:
    """Load a committed CSV report, skipping the test when it is absent.

    Args:
        name: Filename under ``reports``.

    Returns:
        The parsed rows.
    """
    path = REPORTS / name
    if not path.is_file():
        pytest.skip(f"{name} not generated yet")
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_the_committed_manifest_is_valid() -> None:
    manifest = load("canonical_annotation_manifest.json")
    assert validate_manifest(manifest) == []


def test_the_committed_manifest_carries_no_secret_or_local_path() -> None:
    path = REPORTS / "canonical_annotation_manifest.json"
    if not path.is_file():
        pytest.skip("manifest not generated yet")
    assert scan_for_sensitive(path.read_text(encoding="utf-8")) == []


@pytest.mark.parametrize(
    "name",
    [
        "canonical_annotation_decision.md",
        "annotation_drift_report.md",
        "annotation_drift.csv",
        "v4_source_mapping.csv",
        "source_geometry_summary.json",
    ],
)
def test_committed_phase_5a_reports_carry_no_secret_or_local_path(name: str) -> None:
    path = REPORTS / name
    if not path.is_file():
        pytest.skip(f"{name} not generated yet")
    assert scan_for_sensitive(path.read_text(encoding="utf-8")) == []


def test_the_decision_and_its_classification_agree() -> None:
    manifest = load("canonical_annotation_manifest.json")
    assert manifest["decision"] in (CURRENT_COMPLETE_GEOMETRY, FROZEN_V4_SNAPSHOT)
    assert manifest["phase_classification"] == PHASE_CLASSIFICATION[manifest["decision"]]


def test_the_source_population_is_still_436() -> None:
    manifest = load("canonical_annotation_manifest.json")
    assert manifest["source_image_count"] == SOURCE_IMAGES
    assert len(load_rows("v4_source_mapping.csv")) == SOURCE_IMAGES
    assert len(load_rows("annotation_drift.csv")) == SOURCE_IMAGES


def test_every_source_image_has_one_non_augmented_v4_representation() -> None:
    rows = load_rows("v4_source_mapping.csv")
    selected = [row for row in rows if row["v4_image_id_or_filename"]]
    assert len(selected) == SOURCE_IMAGES
    assert all(row["is_augmented"] == "false" for row in selected)
    assert len({row["v4_image_id_or_filename"] for row in selected}) == SOURCE_IMAGES


def test_geometry_counts_account_for_every_annotation() -> None:
    manifest = load("canonical_annotation_manifest.json")
    counts = manifest["geometry_representation_counts"]
    assert set(counts) == set(GEOMETRY_KINDS)
    assert sum(counts.values()) == manifest["annotation_count"]


def test_unsupported_annotations_are_listed_and_classified_not_dropped() -> None:
    manifest = load("canonical_annotation_manifest.json")
    declared = manifest["geometry_representation_counts"][UNSUPPORTED]
    listed = manifest["unsupported_annotations"]
    assert len(listed) == declared
    assert all(entry.get("classification") for entry in listed)
    assert all(entry.get("classification_basis") for entry in listed)


def test_the_drift_table_reconciles_with_the_manifest() -> None:
    manifest = load("canonical_annotation_manifest.json")
    rows = load_rows("annotation_drift.csv")
    assert sum(int(row["current_count"]) for row in rows) == manifest["annotation_count"]
    drift = manifest["drift_summary"]
    assert drift["net_delta"] == drift["current_annotations"] - drift["v4_annotations"]
    # The net delta must never stand in for the amount of change.
    assert drift["net_delta"] == drift["gross_added"] - drift["gross_removed"]


def test_the_recovery_summary_agrees_with_the_manifest() -> None:
    summary = load("source_geometry_summary.json")
    manifest = load("canonical_annotation_manifest.json")
    assert summary["source_images"] == manifest["source_image_count"]
    assert summary["annotations"] == manifest["annotation_count"]
    assert summary["geometry_kind_counts"] == manifest["geometry_representation_counts"]


def test_the_category_map_excludes_the_placeholder_category() -> None:
    manifest = load("canonical_annotation_manifest.json")
    assert "object" not in manifest["category_map"]
    assert manifest["excluded_provider_category"]["name"] == "object"
    assert set(manifest["category_map"]) == set(manifest["class_counts"])


def test_the_manifest_records_limitations_rather_than_claiming_none() -> None:
    manifest = load("canonical_annotation_manifest.json")
    assert manifest["limitations"]
    assert all(isinstance(entry, str) and entry.strip() for entry in manifest["limitations"])


def test_no_committed_figure_is_unreasonably_large() -> None:
    figures = sorted(Path(REPORTS / "figures").glob("*.jpg"))
    if not figures:
        pytest.skip("no figures generated yet")
    oversized = [f.name for f in figures if f.stat().st_size > 512 * 1024]
    assert not oversized, f"figures too large for the repository: {oversized}"
