"""Tests for the source-image inventory.

The provider's offset pagination was measured to be unstable under load: a sweep
can repeat records and miss others. These tests pin the behaviour that protects
the audit from that - the walk converges on the full population, and refuses to
return a short one.
"""

from __future__ import annotations

from typing import Any

import pytest

from construction_safety_vision.data.roboflow import DatasetCoordinates
from construction_safety_vision.data.source import (
    InventoryError,
    SourceAnnotation,
    SourceImageRecord,
    collect_source_images,
    split_counts,
)

COORDS = DatasetCoordinates("ws", "proj", 4, "coco-segmentation")


def result(image_id: str, split: str = "train", **overrides: Any) -> dict[str, Any]:
    """Build one search-API result."""
    payload = {
        "id": image_id,
        "name": f"{image_id}.jpg",
        "split": split,
        "width": 1600,
        "height": 1067,
        "created": 1778349021898,
        "tags": [],
        "owner": "owner-1",
        "annotations": {"count": 2, "classes": {"person": 1, "helmet_on_head": 1}},
    }
    payload.update(overrides)
    return payload


class FakeClient:
    """Client stub returning scripted search pages."""

    def __init__(self, pages: list[list[dict[str, Any]]], total: int) -> None:
        self._pages = pages
        self._total = total
        self.calls = 0

    def search_images_page(
        self, coordinates: DatasetCoordinates, *, limit: int = 250, offset: int = 0
    ) -> dict[str, Any]:
        page = self._pages[self.calls] if self.calls < len(self._pages) else []
        self.calls += 1
        return {"total": self._total, "offset": offset, "results": page}


def test_record_parses_a_search_result() -> None:
    record = SourceImageRecord.from_search_result(result("abc", "valid"))

    assert record.image_id == "abc"
    assert record.split == "valid"
    assert record.width == 1600
    assert record.pixels == 1600 * 1067
    assert record.aspect_ratio == pytest.approx(1600 / 1067)
    assert record.class_counts == {"person": 1, "helmet_on_head": 1}


def test_record_without_an_id_is_rejected() -> None:
    with pytest.raises(InventoryError, match="no image id"):
        SourceImageRecord.from_search_result({"name": "x.jpg"})


def test_aspect_ratio_is_none_without_a_height() -> None:
    assert SourceImageRecord.from_search_result(result("a", height=0)).aspect_ratio is None


def test_annotation_parses_a_provider_box() -> None:
    annotation = SourceAnnotation.from_box(
        {
            "id": "1",
            "type": "polygon",
            "label": "person",
            "x": "10",
            "y": "20",
            "width": "30",
            "height": "40",
            "points": [[1, 2], [3, 4], [5, 6]],
        }
    )

    assert annotation.label == "person"
    assert annotation.geometry_type == "polygon"
    assert annotation.point_count == 3
    assert annotation.width == 30.0


def test_mask_annotations_report_no_vertices() -> None:
    # Measured: the provider exposes vertices only for polygon instances.
    annotation = SourceAnnotation.from_box({"id": "2", "type": "mask", "label": "person"})
    assert annotation.geometry_type == "mask"
    assert annotation.point_count == 0


def test_attach_details_counts_geometry_types() -> None:
    record = SourceImageRecord.from_search_result(result("abc"))
    record.attach_details(
        {
            "annotation": {
                "boxes": [
                    {"id": "1", "type": "polygon", "label": "person", "points": [[0, 0]]},
                    {"id": "2", "type": "mask", "label": "vest_on_body"},
                    {"id": "3", "type": "mask", "label": "person"},
                ]
            }
        }
    )

    assert record.geometry_types == {"mask": 2, "polygon": 1}
    assert len(record.annotations) == 3


def test_attach_details_tolerates_an_unannotated_image() -> None:
    record = SourceImageRecord.from_search_result(result("abc", annotations={"count": 0}))
    record.attach_details({"annotation": {}})

    assert record.annotations == []
    assert record.geometry_types == {}


def test_manifest_entry_carries_no_geometry_or_account_identifier() -> None:
    # The manifest is committed: it must not leak the account id or bulk vertices.
    record = SourceImageRecord.from_search_result(result("abc"))
    record.attach_details(
        {"annotation": {"boxes": [{"id": "1", "type": "polygon", "points": [[0, 0], [1, 1]]}]}}
    )

    entry = record.manifest_entry()

    assert "owner" not in entry
    assert "points" not in json_flatten(entry)
    assert entry["image_id"] == "abc"
    assert entry["detail_annotation_count"] == 1


def json_flatten(payload: Any) -> str:
    """Flatten a mapping to a string for substring assertions."""
    import json

    return json.dumps(payload)


def test_walk_collects_a_single_clean_sweep() -> None:
    client = FakeClient([[result("a"), result("b"), result("c")]], total=3)

    walk = collect_source_images(client, COORDS, page_size=3)

    assert [r.image_id for r in walk.records] == ["a", "b", "c"]
    assert walk.repeated_ids == []
    assert walk.is_complete


def test_walk_converges_when_pages_repeat_and_miss_records() -> None:
    # This is the observed provider behaviour: unstable ordering across requests.
    client = FakeClient(
        [
            [result("a"), result("b")],
            [result("a"), result("b")],  # a repeated sweep that adds nothing
            [result("c"), result("d")],
        ],
        total=4,
    )

    walk = collect_source_images(client, COORDS, page_size=2)

    assert [r.image_id for r in walk.records] == ["a", "b", "c", "d"]
    assert walk.repeated_ids == ["a", "b"]
    assert walk.is_complete


def test_walk_refuses_an_incomplete_population() -> None:
    # Silently returning 2 of 4 images would corrupt every later statistic.
    client = FakeClient([[result("a"), result("b")]], total=4)

    with pytest.raises(InventoryError, match="incomplete population"):
        collect_source_images(client, COORDS, page_size=2, max_pages=3)


def test_walk_is_bounded_when_the_provider_never_completes() -> None:
    client = FakeClient([[result("a")]] * 50, total=99)

    with pytest.raises(InventoryError):
        collect_source_images(client, COORDS, page_size=1, max_pages=5)
    assert client.calls <= 5, "the walk must respect its page budget"


def test_records_are_returned_sorted_by_id() -> None:
    client = FakeClient([[result("c"), result("a"), result("b")]], total=3)
    assert [r.image_id for r in collect_source_images(client, COORDS).records] == ["a", "b", "c"]


def test_split_counts_are_sorted() -> None:
    records = [
        SourceImageRecord.from_search_result(result("a", "train")),
        SourceImageRecord.from_search_result(result("b", "valid")),
        SourceImageRecord.from_search_result(result("c", "train")),
    ]
    assert split_counts(records) == {"train": 2, "valid": 1}
