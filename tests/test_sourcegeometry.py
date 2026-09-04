"""Tests for recovering complete geometry from the provider's live annotations.

Provider payloads are built here rather than downloaded, so nothing needs the
network or a credential. The RLE fixtures are produced by the reference COCO
implementation and then wrapped exactly as the provider wraps them, which is what
makes these tests a real check on the decoder rather than a restatement of it.
"""

from __future__ import annotations

import base64
import zlib
from typing import Any

import numpy as np
import pytest
from pycocotools import mask as coco_mask

from construction_safety_vision.data.sourcegeometry import (
    POLYGON,
    RLE,
    UNSUPPORTED,
    GeometryRecoveryError,
    centre_box_to_coco,
    decode_mask_counts,
    measure_geometry,
    polygon_ring,
    recover_annotation,
    recover_image,
)

CANVAS = 100
"""Side of the synthetic image used throughout."""


def provider_mask(x: int, y: int, width: int, height: int, *, canvas: int = CANVAS) -> str:
    """Encode a filled rectangle the way the provider encodes a mask annotation.

    Args:
        x: Left edge.
        y: Top edge.
        width: Extent along x.
        height: Extent along y.
        canvas: Side of the square image.

    Returns:
        Base64 text of the zlib-compressed COCO counts string.
    """
    mask = np.zeros((canvas, canvas), dtype=np.uint8, order="F")
    mask[y : y + height, x : x + width] = 1
    counts = coco_mask.encode(mask)["counts"]
    return base64.b64encode(zlib.compress(counts)).decode("ascii")


def mask_box(x: int, y: int, width: int, height: int, **overrides: Any) -> dict[str, Any]:
    """Build a provider ``mask`` box for a rectangle.

    Args:
        x: Left edge.
        y: Top edge.
        width: Extent along x.
        height: Extent along y.
        **overrides: Fields to replace.

    Returns:
        A provider ``boxes`` entry.
    """
    box = {
        "id": "1",
        "label": "person",
        "type": "mask",
        "x": x + width / 2,
        "y": y + height / 2,
        "width": width,
        "height": height,
        "area": width * height,
        "mask": provider_mask(x, y, width, height),
    }
    box.update(overrides)
    return box


def polygon_box(**overrides: Any) -> dict[str, Any]:
    """Build a provider ``polygon`` box for a triangle-topped shape.

    Args:
        **overrides: Fields to replace.

    Returns:
        A provider ``boxes`` entry.
    """
    box = {
        "id": "2",
        "label": "vest_on_body",
        "type": "polygon",
        "points": [[10, 20], [40, 20], [40, 60], [10, 60]],
        "x": 25.0,
        "y": 40.0,
        "width": 30,
        "height": 40,
    }
    box.update(overrides)
    return box


def test_centre_box_converts_to_coco_top_left() -> None:
    assert centre_box_to_coco(50.0, 40.0, 20.0, 10.0) == [40.0, 35.0, 20.0, 10.0]


def test_decode_mask_counts_round_trips_the_provider_encoding() -> None:
    counts = decode_mask_counts(provider_mask(10, 10, 5, 5))
    rle = {"size": [CANVAS, CANVAS], "counts": counts}
    assert float(coco_mask.area(rle)) == 25.0


def test_decode_mask_counts_rejects_non_base64() -> None:
    with pytest.raises(GeometryRecoveryError, match="not valid base64"):
        decode_mask_counts("not base64 !!!")


def test_decode_mask_counts_rejects_uncompressed_payload() -> None:
    plain = base64.b64encode(b"not zlib data at all").decode("ascii")
    with pytest.raises(GeometryRecoveryError, match="not zlib-compressed"):
        decode_mask_counts(plain)


def test_decode_mask_counts_rejects_empty() -> None:
    with pytest.raises(GeometryRecoveryError, match="empty"):
        decode_mask_counts("   ")


def test_mask_annotation_recovers_rle_geometry_in_image_coordinates() -> None:
    recovered = recover_annotation(
        mask_box(30, 40, 12, 8), image_id="img", width=CANVAS, height=CANVAS
    )
    assert recovered.geometry_kind == RLE
    assert recovered.segmentation["size"] == [CANVAS, CANVAS]
    assert recovered.area == 96.0
    assert recovered.bbox == [30.0, 40.0, 12.0, 8.0]
    assert recovered.has_mask_geometry


def test_polygon_annotation_uses_coordinate_extrema_not_rasterisation() -> None:
    # Rasterising a polygon first can shave a thin spike off the box; the
    # provider's own value is the extrema, so that is what must be reproduced.
    recovered = recover_annotation(polygon_box(), image_id="img", width=CANVAS, height=CANVAS)
    assert recovered.geometry_kind == POLYGON
    assert recovered.bbox == [10.0, 20.0, 30.0, 40.0]
    assert recovered.bbox_delta_px == 0.0
    assert recovered.point_count == 4


def test_area_disagreement_is_fatal_because_it_means_a_bad_decode() -> None:
    box = mask_box(30, 40, 12, 8, area=9999)
    with pytest.raises(GeometryRecoveryError, match="disagrees with the provider's declared area"):
        recover_annotation(box, image_id="img", width=CANVAS, height=CANVAS)


def test_box_disagreement_is_recorded_rather_than_fatal() -> None:
    # Phase 4B measured the provider's stored boxes drifting from its own
    # geometry. Segmentation is canonical, so a stale box is data, not a crash.
    shifted = polygon_box(x=35.0, width=40)
    recovered = recover_annotation(shifted, image_id="img", width=CANVAS, height=CANVAS)
    assert recovered.geometry_kind == POLYGON
    assert recovered.bbox_delta_px is not None
    assert recovered.bbox_delta_px > 1.0


def test_annotation_without_geometry_is_classified_never_dropped() -> None:
    box = {
        "id": "I",
        "label": "helmet_on_head",
        "type": None,
        "x": 5,
        "y": 5,
        "width": 4,
        "height": 4,
    }
    recovered = recover_annotation(box, image_id="img", width=CANVAS, height=CANVAS)
    assert recovered.geometry_kind == UNSUPPORTED
    assert not recovered.has_mask_geometry
    assert recovered.bbox == [3.0, 3.0, 4.0, 4.0]
    assert recovered.notes


def test_recover_image_keeps_every_annotation_including_unsupported() -> None:
    image = {
        "id": "img",
        "name": "a.jpg",
        "split": "train",
        "annotation": {
            "width": CANVAS,
            "height": CANVAS,
            "boxes": [
                mask_box(30, 40, 12, 8),
                polygon_box(),
                {
                    "id": "J",
                    "label": "person",
                    "type": None,
                    "x": 5,
                    "y": 5,
                    "width": 4,
                    "height": 4,
                },
            ],
        },
    }
    recovered = recover_image(image)
    assert len(recovered.annotations) == 3
    assert recovered.geometry_counts == {POLYGON: 1, RLE: 1, UNSUPPORTED: 1}
    assert sum(recovered.geometry_counts.values()) == len(recovered.annotations)
    # The geometry-less record still contributes to its class count: dropping it
    # here is exactly the silent loss this project must not have.
    assert recovered.class_counts == {"person": 2, "vest_on_body": 1}


def test_recover_image_rejects_a_payload_without_a_canvas() -> None:
    with pytest.raises(GeometryRecoveryError, match="no usable canvas"):
        recover_image({"id": "img", "annotation": {"width": 0, "height": 0, "boxes": []}})


def test_measure_geometry_rejects_an_rle_that_escapes_its_canvas() -> None:
    # pycocotools does not validate counts; it silently returns nonsense, which
    # would otherwise become a confidently wrong mask.
    with pytest.raises(GeometryRecoveryError, match="escapes its declared canvas"):
        measure_geometry({"size": [10, 10], "counts": "garbagegarbage"}, height=10, width=10)


def test_polygon_ring_accepts_mapping_vertices() -> None:
    ring = polygon_ring([{"x": 1, "y": 2}, {"x": 3, "y": 4}, {"x": 5, "y": 6}])
    assert ring == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]


def test_polygon_ring_rejects_a_degenerate_ring() -> None:
    with pytest.raises(GeometryRecoveryError, match="at least 3 vertices"):
        polygon_ring([[1, 2], [3, 4]])


def test_provider_numeric_strings_are_accepted() -> None:
    # The provider types the same field as int, float or decimal string.
    recovered = recover_annotation(
        polygon_box(x="25.0000", y="40.0000", width="30.0000", height="40.0000"),
        image_id="img",
        width=CANVAS,
        height=CANVAS,
    )
    assert recovered.bbox == [10.0, 20.0, 30.0, 40.0]


def test_summary_entry_carries_no_bulk_geometry() -> None:
    recovered = recover_annotation(
        mask_box(30, 40, 12, 8), image_id="img", width=CANVAS, height=CANVAS
    )
    entry = recovered.summary_entry()
    assert "segmentation" not in entry
    assert "segmentation" in recovered.geometry_entry()
