"""Tests that the built detection adapter matches the canonical dataset it derives from.

These read the committed adapter manifest and, when the bulk adapter data is
present, the label files themselves. They never read the holdout: the only thing
asserted about it is that nothing was produced for it.
"""

from __future__ import annotations

import json

import pytest

from construction_safety_vision.data.canonical import scan_for_sensitive
from construction_safety_vision.data.yolo_detection_adapter import (
    ADAPTER_TYPE,
    NOT_MATERIALIZED,
    load_adapter_config,
    parse_label_line,
    validate_normalised,
)
from construction_safety_vision.paths import ProjectPaths

DEVELOPMENT_SPLITS = ("train", "validation")
EXPECTED_IMAGES = {"train": 303, "validation": 65}
EXPECTED_ANNOTATIONS = {"train": 1422, "validation": 304}
EXPECTED_NEGATIVES = {"train": 10, "validation": 2}
DEVELOPMENT_IMAGES = 368
DEVELOPMENT_ANNOTATIONS = 1726
CLASSES = ("helmet_loose", "helmet_on_head", "person", "vest_loose", "vest_on_body")
FROZEN_CLASS_MAP_SHA256 = "596dab5b5756e3d4253924f84bf46d37a87d80b6ec830da781a08a93cd823753"


@pytest.fixture(scope="module")
def paths() -> ProjectPaths:
    return ProjectPaths.from_root()


@pytest.fixture(scope="module")
def manifest(paths: ProjectPaths) -> dict:
    path = paths.reports / "detection_adapter_manifest.json"
    if not path.is_file():
        pytest.skip("detection_adapter_manifest.json not present; run build_detection_adapter.py")
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def adapter_root(paths: ProjectPaths):
    config = load_adapter_config(paths.configs / "detection_adapter.yaml")
    return paths.root / config.output_root


def label_directory(adapter_root, manifest, split):
    directory = adapter_root / "labels" / manifest[split]["adapter_directory"]
    if not directory.is_dir():
        pytest.skip("adapter bulk data not present; it is git-ignored and re-derivable")
    return directory


# --- manifest ---------------------------------------------------------------------


def test_manifest_declares_a_derived_detection_adapter(manifest):
    assert manifest["adapter_type"] == ADAPTER_TYPE
    assert manifest["derived_representation"] is True
    assert manifest["canonical_detection_format"] == "COCO"
    assert manifest["bulk_data_committed"] is False


def test_manifest_records_that_the_provider_bbox_was_not_used(manifest):
    assert manifest["provider_bbox_used"] is False
    assert manifest["bbox_source"] == "CANONICAL_COCO_DETECTION_BBOX"


def test_no_segmentation_labels_were_generated(manifest):
    assert manifest["segmentation_labels_generated"] is False


def test_manifest_binds_to_the_frozen_protocol(paths, manifest):
    task = json.loads((paths.reports / "task_dataset_manifest.json").read_text(encoding="utf-8"))
    assert manifest["split_assignment_sha256"] == task["split_assignment_sha256"]
    assert manifest["modeling_population_sha256"] == task["modeling_population_sha256"]
    assert manifest["class_map_sha256"] == task["class_map_sha256"]


def test_class_map_is_the_frozen_one(manifest):
    assert manifest["class_map_sha256"] == FROZEN_CLASS_MAP_SHA256
    assert set(manifest["class_map"]) == set(CLASSES)
    assert sorted(manifest["class_map"].values()) == [0, 1, 2, 3, 4]
    assert "object" not in manifest["class_map"]


def test_counts_match_the_canonical_datasets(paths, manifest):
    task = json.loads((paths.reports / "task_dataset_manifest.json").read_text(encoding="utf-8"))
    for split in DEVELOPMENT_SPLITS:
        assert manifest[split]["image_count"] == EXPECTED_IMAGES[split]
        assert manifest[split]["annotation_count"] == EXPECTED_ANNOTATIONS[split]
        assert manifest[split]["image_count"] == task[split]["image_count"]
        assert manifest[split]["annotation_count"] == task[split]["annotation_count"]
    assert manifest["development_totals"]["images"] == DEVELOPMENT_IMAGES
    assert manifest["development_totals"]["annotations"] == DEVELOPMENT_ANNOTATIONS


def test_negatives_are_preserved(manifest):
    for split in DEVELOPMENT_SPLITS:
        assert manifest[split]["negative_images"] == EXPECTED_NEGATIVES[split]
    assert manifest["development_totals"]["negative_images"] == 12


def test_per_class_instances_match_the_canonical_datasets(paths, manifest):
    task = json.loads((paths.reports / "task_dataset_manifest.json").read_text(encoding="utf-8"))
    for split in DEVELOPMENT_SPLITS:
        assert manifest[split]["instances_by_class"] == task[split]["instances_by_class"]


def test_adapter_images_are_byte_identical_to_the_canonical_images(manifest):
    for split in DEVELOPMENT_SPLITS:
        entry = manifest[split]
        assert entry["image_content_sha256"] == entry["canonical_image_content_sha256"]
        assert entry["image_content_sha256"]


def test_round_trip_had_no_mismatch(manifest):
    totals = manifest["round_trip_totals"]
    assert totals["annotations_checked"] == DEVELOPMENT_ANNOTATIONS
    assert totals["within_tolerance"] == DEVELOPMENT_ANNOTATIONS
    assert totals["mismatches"] == 0
    assert totals["max_delta_px"] < totals["tolerance_px"]


def test_manifest_carries_no_sensitive_content(paths):
    text = (paths.reports / "detection_adapter_manifest.json").read_text(encoding="utf-8")
    assert scan_for_sensitive(text) == []


def test_report_carries_no_sensitive_content(paths):
    text = (paths.reports / "detection_adapter_report.md").read_text(encoding="utf-8")
    assert scan_for_sensitive(text) == []


def test_runtime_report_carries_no_sensitive_content(paths):
    path = paths.reports / "detection_runtime_report.md"
    if not path.is_file():
        pytest.skip("runtime report not present; run detection_runtime_check.py")
    assert scan_for_sensitive(path.read_text(encoding="utf-8")) == []


# --- the protected holdout ----------------------------------------------------------


def test_manifest_records_the_holdout_as_unadapted(manifest):
    assert manifest["test"]["status"] == NOT_MATERIALIZED
    assert set(manifest["test"]) == {"status", "reason"}


def test_manifest_leaks_no_holdout_identifier(paths, manifest):
    split_manifest = json.loads((paths.reports / "split_manifest.json").read_text(encoding="utf-8"))
    holdout = {
        image_id
        for group in split_manifest["test"]["groups"]
        for image_id in group["source_image_ids"]
    }
    text = (paths.reports / "detection_adapter_manifest.json").read_text(encoding="utf-8")
    assert not [image_id for image_id in holdout if image_id in text]


def test_no_holdout_adapter_exists(adapter_root):
    assert not (adapter_root / "images" / "test").exists()
    assert not (adapter_root / "labels" / "test").exists()


def test_dataset_descriptor_has_no_test_key(adapter_root):
    descriptor = adapter_root / "dataset.yaml"
    if not descriptor.is_file():
        pytest.skip("adapter bulk data not present; it is git-ignored and re-derivable")
    text = descriptor.read_text(encoding="utf-8")
    assert "\ntest:" not in text
    assert "\ntrain:" in text
    assert "\nval:" in text


# --- the label files -----------------------------------------------------------------


@pytest.mark.parametrize("split", DEVELOPMENT_SPLITS)
def test_label_files_match_the_image_count(adapter_root, manifest, split):
    directory = label_directory(adapter_root, manifest, split)
    assert len(list(directory.glob("*.txt"))) == EXPECTED_IMAGES[split]


@pytest.mark.parametrize("split", DEVELOPMENT_SPLITS)
def test_label_lines_match_the_annotation_count(adapter_root, manifest, split):
    directory = label_directory(adapter_root, manifest, split)
    lines = sum(
        len([line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()])
        for path in directory.glob("*.txt")
    )
    assert lines == EXPECTED_ANNOTATIONS[split]


@pytest.mark.parametrize("split", DEVELOPMENT_SPLITS)
def test_empty_label_files_are_exactly_the_negatives(adapter_root, manifest, split):
    directory = label_directory(adapter_root, manifest, split)
    empty = [path for path in directory.glob("*.txt") if path.stat().st_size == 0]
    assert len(empty) == EXPECTED_NEGATIVES[split]


@pytest.mark.parametrize("split", DEVELOPMENT_SPLITS)
def test_every_label_is_a_valid_normalised_box(adapter_root, manifest, split):
    config = load_adapter_config(ProjectPaths.from_root().configs / "detection_adapter.yaml")
    directory = label_directory(adapter_root, manifest, split)
    valid_classes = set(manifest["class_map"].values())
    for path in sorted(directory.glob("*.txt")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            box = parse_label_line(line)
            assert box.class_id in valid_classes
            assert (
                validate_normalised(box, epsilon=config.tolerances.normalised_bound_epsilon) == []
            )


@pytest.mark.parametrize("split", DEVELOPMENT_SPLITS)
def test_label_stems_are_the_frozen_split_membership(paths, adapter_root, manifest, split):
    split_manifest = json.loads((paths.reports / "split_manifest.json").read_text(encoding="utf-8"))
    frozen = {
        image_id
        for group in split_manifest[split]["groups"]
        for image_id in group["source_image_ids"]
    }
    directory = label_directory(adapter_root, manifest, split)
    assert {path.stem for path in directory.glob("*.txt")} == frozen


@pytest.mark.parametrize("split", DEVELOPMENT_SPLITS)
def test_images_and_labels_pair_one_to_one(adapter_root, manifest, split):
    directory = manifest[split]["adapter_directory"]
    images = adapter_root / "images" / directory
    labels = adapter_root / "labels" / directory
    if not images.is_dir():
        pytest.skip("adapter bulk data not present; it is git-ignored and re-derivable")
    assert {path.stem for path in images.iterdir()} == {path.stem for path in labels.glob("*.txt")}


def test_no_yolo_segmentation_label_was_written(adapter_root, manifest):
    directory = label_directory(adapter_root, manifest, "train")
    for path in sorted(directory.glob("*.txt")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                # A segmentation label carries a polygon: many more than 5 fields.
                assert len(line.split()) == 5
