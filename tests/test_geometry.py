"""Tests for bounding boxes derived from COCO segmentation geometry.

Both representations present in the export are exercised. RLE fixtures are built
by encoding synthetic masks with the reference COCO implementation, so the tests
depend on no downloaded data and no network.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from pycocotools import mask as coco_mask

from construction_safety_vision.data.geometry import (
    BBox,
    GeometryError,
    bbox_from_segmentation,
    compare_to_segmentation,
    polygon_bbox,
    rle_bbox,
)

SQUARE = [[10.0, 20.0, 40.0, 20.0, 40.0, 60.0, 10.0, 60.0]]
"""A rectangle spanning x 10..40 and y 20..60."""


def encode_rect(x: int, y: int, width: int, height: int, *, canvas: int = 100) -> dict[str, Any]:
    """Encode a filled rectangle as compressed COCO RLE."""
    mask = np.zeros((canvas, canvas), dtype=np.uint8, order="F")
    mask[y : y + height, x : x + width] = 1
    return coco_mask.encode(mask)


def test_polygon_bbox_is_the_coordinate_extrema() -> None:
    box = polygon_bbox(SQUARE)
    assert (box.x, box.y, box.width, box.height) == (10.0, 20.0, 30.0, 40.0)


def test_polygon_bbox_spans_every_ring() -> None:
    # A multi-part instance must be enclosed by one box covering all parts.
    two_rings = [SQUARE[0], [80.0, 5.0, 90.0, 5.0, 90.0, 15.0, 80.0, 15.0]]
    box = polygon_bbox(two_rings)
    assert box.corners() == (10.0, 5.0, 90.0, 60.0)


def test_polygon_bbox_ignores_unusable_rings() -> None:
    box = polygon_bbox([[], [1.0], SQUARE[0]])
    assert box.corners() == (10.0, 20.0, 40.0, 60.0)


def test_polygon_bbox_rejects_a_ring_with_no_coordinates() -> None:
    with pytest.raises(GeometryError, match="no usable coordinates"):
        polygon_bbox([[], []])


def test_rle_bbox_matches_the_encoded_rectangle() -> None:
    box = rle_bbox(encode_rect(12, 30, 25, 40))
    assert box.as_list() == [12.0, 30.0, 25.0, 40.0]


def test_rle_bbox_accepts_a_string_counts_field() -> None:
    # Exports carry `counts` as text after a JSON round trip; pycocotools wants bytes.
    rle = encode_rect(5, 5, 10, 10)
    as_text = {"size": rle["size"], "counts": rle["counts"].decode("utf-8")}
    assert rle_bbox(as_text).as_list() == rle_bbox(rle).as_list()


def test_rle_bbox_does_not_mutate_its_input() -> None:
    rle = encode_rect(5, 5, 10, 10)
    as_text = {"size": rle["size"], "counts": rle["counts"].decode("utf-8")}
    rle_bbox(as_text)
    assert isinstance(as_text["counts"], str), "caller's mapping must be left alone"


def test_rle_bbox_rejects_a_mapping_without_counts() -> None:
    with pytest.raises(GeometryError, match="'counts' and 'size'"):
        rle_bbox({"size": [10, 10]})


@pytest.mark.parametrize("counts", ["!!! not a valid rle !!!", "AAAA"])
def test_rle_bbox_rejects_a_box_that_escapes_its_own_canvas(counts: str) -> None:
    # Measured: pycocotools does not raise on malformed counts, it returns
    # nonsense - one of these yields a box 115 million pixels wide on a 10x10
    # canvas. Silent nonsense in the project's source of truth is unacceptable.
    with pytest.raises(GeometryError, match="exceeds its declared canvas"):
        rle_bbox({"size": [10, 10], "counts": counts})


def test_rle_bbox_rejects_a_malformed_size() -> None:
    # A bad `size` trips pycocotools itself, which raises a bare TypeError; the
    # wrapper must turn that into a GeometryError rather than let it escape.
    rle = encode_rect(1, 1, 2, 2, canvas=10)
    with pytest.raises(GeometryError, match="Could not decode"):
        rle_bbox({"size": "10x10", "counts": rle["counts"]})


def test_rle_bbox_accepts_a_mask_touching_the_canvas_edge() -> None:
    # The guard must not reject a legitimate mask that reaches the border.
    box = rle_bbox(encode_rect(0, 0, 10, 10, canvas=10))
    assert box.as_list() == [0.0, 0.0, 10.0, 10.0]


def test_dispatch_handles_both_representations() -> None:
    assert bbox_from_segmentation(SQUARE).as_list() == [10.0, 20.0, 30.0, 40.0]
    assert bbox_from_segmentation(encode_rect(1, 2, 3, 4)).as_list() == [1.0, 2.0, 3.0, 4.0]


@pytest.mark.parametrize("value", [None, "polygon", 42, []])
def test_dispatch_rejects_unsupported_segmentations(value: Any) -> None:
    with pytest.raises(GeometryError):
        bbox_from_segmentation(value)


def test_bbox_from_sequence_rejects_malformed_input() -> None:
    with pytest.raises(GeometryError, match="four numeric"):
        BBox.from_sequence([1, 2, 3])


def test_bbox_geometry_helpers() -> None:
    box = BBox(10.0, 20.0, 30.0, 40.0)
    assert box.area == 1200.0
    assert box.corners() == (10.0, 20.0, 40.0, 60.0)
    assert box.as_list() == [10.0, 20.0, 30.0, 40.0]


def test_comparison_of_an_exact_match() -> None:
    result = compare_to_segmentation([10.0, 20.0, 30.0, 40.0], SQUARE, representation="polygon")
    assert result.deltas == (0.0, 0.0, 0.0, 0.0)
    assert result.max_delta == 0.0
    assert result.within(0.0)


def test_comparison_reports_per_corner_deltas() -> None:
    # Supplied box is shifted right by 2 and taller by 3 at the bottom.
    result = compare_to_segmentation([12.0, 20.0, 30.0, 43.0], SQUARE, representation="polygon")
    assert result.deltas == (2.0, 0.0, 2.0, 3.0)
    assert result.max_delta == 3.0
    assert not result.within(1.0)
    assert result.within(3.0)


def test_comparison_tolerance_boundary_is_inclusive() -> None:
    result = compare_to_segmentation([11.0, 20.0, 29.0, 40.0], SQUARE, representation="polygon")
    assert result.max_delta == 1.0
    assert result.within(1.0)
    assert not result.within(0.5)


def test_comparison_is_deterministic() -> None:
    first = compare_to_segmentation([10.0, 20.0, 30.0, 40.0], SQUARE, representation="polygon")
    second = compare_to_segmentation([10.0, 20.0, 30.0, 40.0], SQUARE, representation="polygon")
    assert first == second


def test_comparison_detects_the_rle_quantisation_pattern() -> None:
    # A fractional supplied box against a rasterised mask: sub-pixel disagreement
    # is expected and must not be reported as an exact match.
    rle = encode_rect(20, 30, 40, 50)
    result = compare_to_segmentation([19.75, 30.0, 40.5, 50.0], rle, representation="rle")
    assert not result.within(0.0)
    assert result.within(1.0)
    assert result.representation == "rle"
