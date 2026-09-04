"""Tests that the committed task-dataset evidence describes what is on disk.

These read the real manifest under ``reports/`` and, where the bulk data is
present, the emitted COCO files. They assert on the development splits only. The
holdout is checked in exactly one way - that nothing about it was produced - and
never by reading it.
"""

from __future__ import annotations

import json

import pytest

from construction_safety_vision.data.canonical import scan_for_sensitive
from construction_safety_vision.data.coco_materialization import (
    DETECTION,
    SEGMENTATION,
    alignment_report,
    document_fingerprint,
)
from construction_safety_vision.data.materialization import (
    DEVELOPMENT_SPLITS,
    TEST_DISABLED,
    load_materialization_config,
)
from construction_safety_vision.paths import ProjectPaths

EXPECTED_IMAGES = {"train": 303, "validation": 65}
EXPECTED_ANNOTATIONS = {"train": 1422, "validation": 304}
DEVELOPMENT_IMAGES = 368
DEVELOPMENT_ANNOTATIONS = 1726
CLASSES = ("helmet_loose", "helmet_on_head", "person", "vest_loose", "vest_on_body")
NOT_MATERIALIZED = "NOT_MATERIALIZED_PROTECTED_HOLDOUT"


@pytest.fixture(scope="module")
def paths() -> ProjectPaths:
    return ProjectPaths.from_root()


@pytest.fixture(scope="module")
def manifest(paths: ProjectPaths) -> dict:
    path = paths.reports / "task_dataset_manifest.json"
    if not path.is_file():
        pytest.skip("task_dataset_manifest.json not present; run materialize_task_datasets.py")
    return json.loads(path.read_text(encoding="utf-8"))


def coco_path(paths: ProjectPaths, task: str, split: str):
    config = load_materialization_config(paths.configs / "task_materialization.yaml")
    return paths.root / config.output_root / "annotations" / f"{task}_{split}.coco.json"


def load_coco(paths: ProjectPaths, task: str, split: str) -> dict:
    path = coco_path(paths, task, split)
    if not path.is_file():
        pytest.skip(f"{path.name} not present; bulk data is git-ignored and re-derivable")
    return json.loads(path.read_text(encoding="utf-8"))


# --- manifest -----------------------------------------------------------------


def test_manifest_records_the_canonical_formats(manifest):
    assert manifest["canonical_detection_format"] == "COCO"
    assert manifest["canonical_segmentation_format"] == "COCO_INSTANCE_SEGMENTATION"
    assert manifest["model_specific_adapter"] == "NOT_YET_SELECTED"
    assert manifest["bulk_data_committed"] is False


def test_manifest_binds_to_the_frozen_split_and_population(paths, manifest):
    split_manifest = json.loads((paths.reports / "split_manifest.json").read_text(encoding="utf-8"))
    population = json.loads(
        (paths.reports / "canonical_modeling_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["split_assignment_sha256"] == split_manifest["split_assignment_sha256"]
    assert (
        manifest["modeling_population_sha256"]
        == population["fingerprints"]["modeling_population_sha256"]
    )
    assert manifest["groups_sha256"] == population["fingerprints"]["groups_sha256"]


def test_class_map_is_the_frozen_one(paths, manifest):
    population = json.loads(
        (paths.reports / "canonical_modeling_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["class_map"] == population["class_map"]
    assert manifest["class_map_sha256"] == population["fingerprints"]["class_map_sha256"]
    assert set(manifest["class_map"]) == set(CLASSES)
    assert "object" not in manifest["class_map"]


def test_materialization_config_fingerprint_matches_the_committed_config(paths, manifest):
    config = load_materialization_config(paths.configs / "task_materialization.yaml")
    assert manifest["materialization_config_sha256"] == config.fingerprint()
    assert config.test_materialization == TEST_DISABLED


def test_development_counts_are_exact(manifest):
    for split in DEVELOPMENT_SPLITS:
        assert manifest[split]["image_count"] == EXPECTED_IMAGES[split]
        assert manifest[split]["annotation_count"] == EXPECTED_ANNOTATIONS[split]
    assert manifest["development_totals"] == {
        "images": DEVELOPMENT_IMAGES,
        "annotations": DEVELOPMENT_ANNOTATIONS,
    }


def test_zero_instance_development_images_are_retained(manifest):
    # 14 negatives in the population; 2 sit in the holdout, so 12 are here.
    total = sum(manifest[split]["zero_instance_images"] for split in DEVELOPMENT_SPLITS)
    assert manifest["train"]["zero_instance_images"] == 10
    assert manifest["validation"]["zero_instance_images"] == 2
    assert total == 12


def test_geometry_round_trip_had_no_mismatch(manifest):
    for split in DEVELOPMENT_SPLITS:
        trip = manifest[split]["geometry_round_trip"]
        assert trip["mismatches"] == 0
        assert trip["matches"] == trip["annotations_checked"]
        assert trip["annotations_checked"] == EXPECTED_ANNOTATIONS[split]
        assert trip["max_polygon_coordinate_delta_px"] == 0.0


def test_every_geometry_kind_was_actually_exercised(manifest):
    trip = manifest["train"]["geometry_round_trip"]
    assert trip["polygon_annotations_checked"] > 0
    assert trip["rle_annotations_checked"] > 0
    # The two SYNTHETIC_FROM_PROVIDER_BBOX records live on a train image.
    assert trip["synthetic_rectangles_checked"] == 2


def test_boxes_agree_with_the_independent_phase_5a_measurement(manifest):
    for split in DEVELOPMENT_SPLITS:
        agreement = manifest[split]["bbox_agreement"]
        assert agreement["within_tolerance"]
        assert agreement["annotations_compared"] == EXPECTED_ANNOTATIONS[split]


def test_cross_task_alignment_holds(manifest):
    for split in DEVELOPMENT_SPLITS:
        alignment = manifest[split]["alignment"]
        assert alignment["aligned"]
        assert alignment["image_ids_match"]
        assert alignment["annotation_ids_match"]
        assert alignment["categories_match"]
        assert alignment["boxes_match"]


def test_manifest_carries_no_sensitive_content(paths):
    text = (paths.reports / "task_dataset_manifest.json").read_text(encoding="utf-8")
    assert scan_for_sensitive(text) == []


def test_report_carries_no_sensitive_content(paths):
    text = (paths.reports / "task_materialization_report.md").read_text(encoding="utf-8")
    assert scan_for_sensitive(text) == []


# --- the protected holdout ------------------------------------------------------


def test_manifest_records_the_holdout_as_unmaterialized(manifest):
    assert manifest["test"]["status"] == NOT_MATERIALIZED
    assert set(manifest["test"]) == {"status", "reason"}


def test_manifest_records_no_holdout_identifier_or_statistic(paths, manifest):
    split_manifest = json.loads((paths.reports / "split_manifest.json").read_text(encoding="utf-8"))
    holdout_ids = {
        image_id
        for group in split_manifest["test"]["groups"]
        for image_id in group["source_image_ids"]
    }
    text = (paths.reports / "task_dataset_manifest.json").read_text(encoding="utf-8")
    assert not [image_id for image_id in holdout_ids if image_id in text]
    for key in ("image_count", "annotation_count", "instances_by_class"):
        assert key not in manifest["test"]


def test_no_holdout_dataset_was_written(paths):
    config = load_materialization_config(paths.configs / "task_materialization.yaml")
    root = paths.root / config.output_root
    assert not (root / "images" / "test").exists()
    for task in (DETECTION, SEGMENTATION):
        assert not (root / "annotations" / f"{task}_test.coco.json").exists()


def test_no_yolo_labels_were_written(paths):
    config = load_materialization_config(paths.configs / "task_materialization.yaml")
    root = paths.root / config.output_root
    if not root.exists():
        pytest.skip("bulk data not present; it is git-ignored and re-derivable")
    assert not list(root.rglob("*.txt"))
    assert not list(root.rglob("*.yaml"))


# --- the emitted datasets --------------------------------------------------------


@pytest.mark.parametrize("split", DEVELOPMENT_SPLITS)
def test_emitted_documents_match_the_manifest_fingerprints(paths, manifest, split):
    for task, key in (
        (DETECTION, "detection_document_sha256"),
        (SEGMENTATION, "segmentation_document_sha256"),
    ):
        document = load_coco(paths, task, split)
        assert document_fingerprint(document) == manifest[split][key]


@pytest.mark.parametrize("split", DEVELOPMENT_SPLITS)
def test_emitted_documents_are_aligned_on_disk(paths, split):
    report = alignment_report(
        load_coco(paths, DETECTION, split), load_coco(paths, SEGMENTATION, split), split
    )
    assert report["aligned"], report["problems"]


@pytest.mark.parametrize("split", DEVELOPMENT_SPLITS)
def test_emitted_counts_match_the_frozen_membership(paths, split):
    for task in (DETECTION, SEGMENTATION):
        document = load_coco(paths, task, split)
        assert len(document["images"]) == EXPECTED_IMAGES[split]
        assert len(document["annotations"]) == EXPECTED_ANNOTATIONS[split]


@pytest.mark.parametrize("split", DEVELOPMENT_SPLITS)
def test_emitted_image_ids_are_the_frozen_membership(paths, split):
    split_manifest = json.loads((paths.reports / "split_manifest.json").read_text(encoding="utf-8"))
    frozen = {
        image_id
        for group in split_manifest[split]["groups"]
        for image_id in group["source_image_ids"]
    }
    for task in (DETECTION, SEGMENTATION):
        document = load_coco(paths, task, split)
        assert {record["source_image_id"] for record in document["images"]} == frozen


def test_the_same_image_carries_the_same_id_in_both_views(paths):
    ids: dict[str, set[int]] = {}
    for split in DEVELOPMENT_SPLITS:
        for task in (DETECTION, SEGMENTATION):
            for record in load_coco(paths, task, split)["images"]:
                ids.setdefault(record["source_image_id"], set()).add(record["id"])
    assert all(len(values) == 1 for values in ids.values())
    # Global assignment: no numeric id is reused across the two splits.
    flat = [next(iter(values)) for values in ids.values()]
    assert len(set(flat)) == len(flat)


def test_detection_documents_carry_no_masks(paths):
    for split in DEVELOPMENT_SPLITS:
        document = load_coco(paths, DETECTION, split)
        assert all("segmentation" not in record for record in document["annotations"])
        assert all(
            record["bbox_source"] == "DERIVED_FROM_CANONICAL_SEGMENTATION"
            for record in document["annotations"]
        )


def test_segmentation_documents_keep_both_geometry_representations(paths):
    document = load_coco(paths, SEGMENTATION, "train")
    kinds = {
        "rle": sum(1 for r in document["annotations"] if isinstance(r["segmentation"], dict)),
        "polygon": sum(1 for r in document["annotations"] if isinstance(r["segmentation"], list)),
    }
    assert kinds["rle"] > 0
    assert kinds["polygon"] > 0
    assert kinds["rle"] + kinds["polygon"] == EXPECTED_ANNOTATIONS["train"]


def test_synthetic_rectangles_are_labelled_as_synthetic(paths):
    document = load_coco(paths, SEGMENTATION, "train")
    synthetic = [
        record
        for record in document["annotations"]
        if record["geometry_origin"] == "SYNTHETIC_FROM_PROVIDER_BBOX"
    ]
    assert len(synthetic) == 2
    assert all(isinstance(record["segmentation"], list) for record in synthetic)


def test_emitted_categories_are_the_frozen_class_map(paths, manifest):
    for split in DEVELOPMENT_SPLITS:
        for task in (DETECTION, SEGMENTATION):
            document = load_coco(paths, task, split)
            emitted = {record["name"]: record["id"] for record in document["categories"]}
            assert emitted == manifest["class_map"]


def test_emitted_documents_carry_no_timestamp(paths):
    for split in DEVELOPMENT_SPLITS:
        for task in (DETECTION, SEGMENTATION):
            info = load_coco(paths, task, split)["info"]
            assert "date_created" not in info
            assert info["bbox_source"] == "DERIVED_FROM_CANONICAL_SEGMENTATION"


def test_emitted_documents_load_through_pycocotools(paths):
    import contextlib
    import io

    from pycocotools.coco import COCO

    path = coco_path(paths, SEGMENTATION, "validation")
    if not path.is_file():
        pytest.skip("bulk data not present; it is git-ignored and re-derivable")
    with contextlib.redirect_stdout(io.StringIO()):
        coco = COCO(str(path))
    assert len(coco.imgs) == EXPECTED_IMAGES["validation"]
    assert len(coco.anns) == EXPECTED_ANNOTATIONS["validation"]
    # Every mask decodes to something non-empty through the reference API.
    assert all(coco.annToMask(annotation).sum() > 0 for annotation in coco.anns.values())
