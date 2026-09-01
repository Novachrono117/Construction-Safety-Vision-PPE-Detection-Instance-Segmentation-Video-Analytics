"""Tests for COCO structural inspection.

Fixtures are built locally; the real export is never required, so the suite runs
on a clone with no data.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from construction_safety_vision.data.coco import (
    ANNOTATION_FILENAME,
    CocoValidationError,
    classify_segmentation,
    find_split_dirs,
    inspect_export,
    inspect_split,
    load_coco_document,
)

SQUARE = [[0.0, 0.0, 10.0, 0.0, 10.0, 10.0, 0.0, 10.0]]


def write_split(
    root: Path,
    split: str,
    *,
    images: list[dict[str, Any]],
    annotations: list[dict[str, Any]],
    categories: list[dict[str, Any]],
    image_files: list[str] | None = None,
) -> Path:
    """Create a split directory with a COCO document and image files."""
    directory = root / split
    directory.mkdir(parents=True, exist_ok=True)
    for name in image_files if image_files is not None else [i["file_name"] for i in images]:
        (directory / name).write_bytes(b"\xff\xd8\xff")
    document = {"images": images, "annotations": annotations, "categories": categories}
    (directory / ANNOTATION_FILENAME).write_text(
        json.dumps(document), encoding="utf-8", newline="\n"
    )
    return directory


def minimal_split(root: Path, split: str = "train") -> Path:
    """Create a small, fully consistent split."""
    return write_split(
        root,
        split,
        images=[
            {"id": 1, "file_name": "a.jpg"},
            {"id": 2, "file_name": "b.jpg"},
        ],
        annotations=[
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "segmentation": SQUARE,
                "bbox": [0, 0, 10, 10],
            },
            {
                "id": 2,
                "image_id": 1,
                "category_id": 2,
                "segmentation": {"counts": "abc", "size": [10, 10]},
                "bbox": [1, 1, 2, 2],
            },
        ],
        categories=[{"id": 1, "name": "person"}, {"id": 2, "name": "helmet_on_head"}],
    )


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, "absent"),
        ([], "empty"),
        (SQUARE, "polygon"),
        ({"counts": "xyz", "size": [4, 4]}, "rle"),
        ({"something": 1}, "unknown"),
        ([[0.0, 0.0, 1.0, 1.0]], "degenerate_polygon"),
        ([[0.0, 0.0, 1.0, 1.0, 2.0]], "degenerate_polygon"),
        ("nonsense", "unknown"),
    ],
)
def test_segmentation_classification(value: Any, expected: str) -> None:
    assert classify_segmentation(value) == expected


def test_inspects_a_consistent_split(tmp_path: Path) -> None:
    split_dir = minimal_split(tmp_path)

    report = inspect_split(split_dir, export_root=tmp_path)

    assert report.split == "train"
    assert report.image_records == 2
    assert report.annotation_records == 2
    assert report.image_files_on_disk == 2
    assert report.categories == {1: "person", 2: "helmet_on_head"}
    assert report.annotations_per_category == {"helmet_on_head": 1, "person": 1}
    assert report.segmentation_types == {"polygon": 1, "rle": 1}
    assert report.annotations_with_bbox == 2
    assert report.annotations_with_segmentation == 2
    assert report.images_without_annotations == 1
    assert report.is_structurally_sound


def test_detects_a_dangling_image_reference(tmp_path: Path) -> None:
    split_dir = write_split(
        tmp_path,
        "train",
        images=[{"id": 1, "file_name": "a.jpg"}],
        annotations=[{"id": 1, "image_id": 99, "category_id": 1, "segmentation": SQUARE}],
        categories=[{"id": 1, "name": "person"}],
    )

    report = inspect_split(split_dir, export_root=tmp_path)

    assert report.dangling_image_ids == [99]
    assert not report.is_structurally_sound


def test_detects_an_undefined_category_reference(tmp_path: Path) -> None:
    split_dir = write_split(
        tmp_path,
        "train",
        images=[{"id": 1, "file_name": "a.jpg"}],
        annotations=[{"id": 1, "image_id": 1, "category_id": 7, "segmentation": SQUARE}],
        categories=[{"id": 1, "name": "person"}],
    )

    report = inspect_split(split_dir, export_root=tmp_path)

    assert report.undefined_category_ids == [7]
    assert not report.is_structurally_sound


def test_detects_a_missing_image_file(tmp_path: Path) -> None:
    split_dir = write_split(
        tmp_path,
        "train",
        images=[{"id": 1, "file_name": "a.jpg"}, {"id": 2, "file_name": "gone.jpg"}],
        annotations=[],
        categories=[{"id": 1, "name": "person"}],
        image_files=["a.jpg"],
    )

    report = inspect_split(split_dir, export_root=tmp_path)

    assert report.missing_image_files == ["gone.jpg"]
    assert not report.is_structurally_sound


def test_detects_an_orphan_image_file(tmp_path: Path) -> None:
    split_dir = write_split(
        tmp_path,
        "train",
        images=[{"id": 1, "file_name": "a.jpg"}],
        annotations=[],
        categories=[{"id": 1, "name": "person"}],
        image_files=["a.jpg", "stray.jpg"],
    )

    report = inspect_split(split_dir, export_root=tmp_path)

    assert report.orphan_image_files == ["stray.jpg"]


def test_detects_duplicate_ids(tmp_path: Path) -> None:
    split_dir = write_split(
        tmp_path,
        "train",
        images=[{"id": 1, "file_name": "a.jpg"}, {"id": 1, "file_name": "a.jpg"}],
        annotations=[
            {"id": 5, "image_id": 1, "category_id": 1, "segmentation": SQUARE},
            {"id": 5, "image_id": 1, "category_id": 1, "segmentation": SQUARE},
        ],
        categories=[{"id": 1, "name": "person"}],
    )

    report = inspect_split(split_dir, export_root=tmp_path)

    assert report.duplicate_image_ids == [1]
    assert report.duplicate_annotation_ids == [5]
    assert not report.is_structurally_sound


def test_missing_annotation_file_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "train").mkdir()
    with pytest.raises(CocoValidationError, match="not found"):
        inspect_split(tmp_path / "train", export_root=tmp_path)


def test_invalid_json_is_rejected(tmp_path: Path) -> None:
    directory = tmp_path / "train"
    directory.mkdir()
    (directory / ANNOTATION_FILENAME).write_text("{ broken", encoding="utf-8", newline="\n")
    with pytest.raises(CocoValidationError, match="not valid JSON"):
        load_coco_document(directory / ANNOTATION_FILENAME)


def test_non_object_document_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / ANNOTATION_FILENAME
    path.write_text("[1, 2, 3]", encoding="utf-8", newline="\n")
    with pytest.raises(CocoValidationError, match="must contain a JSON object"):
        load_coco_document(path)


@pytest.mark.parametrize("missing", ["images", "annotations", "categories"])
def test_missing_required_key_is_rejected(tmp_path: Path, missing: str) -> None:
    document = {"images": [], "annotations": [], "categories": []}
    del document[missing]
    path = tmp_path / ANNOTATION_FILENAME
    path.write_text(json.dumps(document), encoding="utf-8", newline="\n")

    with pytest.raises(CocoValidationError, match=f"missing the required key '{missing}'"):
        load_coco_document(path)


@pytest.mark.parametrize("wrong", ["images", "annotations", "categories"])
def test_required_key_must_be_a_list(tmp_path: Path, wrong: str) -> None:
    document: dict[str, Any] = {"images": [], "annotations": [], "categories": []}
    document[wrong] = {"not": "a list"}
    path = tmp_path / ANNOTATION_FILENAME
    path.write_text(json.dumps(document), encoding="utf-8", newline="\n")

    with pytest.raises(CocoValidationError, match="must be a list"):
        load_coco_document(path)


def test_find_split_dirs_ignores_directories_without_annotations(tmp_path: Path) -> None:
    minimal_split(tmp_path, "train")
    (tmp_path / "not_a_split").mkdir()

    assert [p.name for p in find_split_dirs(tmp_path)] == ["train"]


def test_inspect_export_covers_every_split(tmp_path: Path) -> None:
    minimal_split(tmp_path, "train")
    minimal_split(tmp_path, "valid")

    reports = inspect_export(tmp_path)

    assert [r.split for r in reports] == ["train", "valid"]


def test_inspect_export_rejects_an_unexpected_layout(tmp_path: Path) -> None:
    (tmp_path / "images").mkdir()
    with pytest.raises(CocoValidationError, match="not the expected COCO one"):
        inspect_export(tmp_path)
