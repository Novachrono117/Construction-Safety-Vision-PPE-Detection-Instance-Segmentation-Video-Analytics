"""Tests for the fragment-rule features and their evaluation.

The property that matters most is what the features are *not* allowed to see. A
rule that reproduced the version-4 diff by consulting the provider split, or the
export, or a list of identifiers, would score perfectly and be worthless.
"""

from __future__ import annotations

import numpy as np
import pytest
from pycocotools import mask as coco_mask

from construction_safety_vision.data.fragments import (
    AnnotationShape,
    FragmentFeatures,
    box_containment,
    describe_image,
    encode_mask,
    evaluate_rule,
    mask_containment,
)

CANVAS = 100
"""Side of the synthetic image used throughout."""


def rect_polygon(x: float, y: float, w: float, h: float) -> list[list[float]]:
    """Build a rectangular polygon ring.

    Args:
        x: Left edge.
        y: Top edge.
        w: Width.
        h: Height.

    Returns:
        A COCO polygon segmentation.
    """
    return [[x, y, x + w, y, x + w, y + h, x, y + h]]


def shape(
    annotation_id: str,
    label: str,
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    with_mask: bool = True,
) -> AnnotationShape:
    """Build an annotation shape backed by a rectangle.

    Args:
        annotation_id: Identifier.
        label: Class name.
        x: Left edge.
        y: Top edge.
        w: Width.
        h: Height.
        with_mask: Whether to attach an encoded mask.

    Returns:
        The shape.
    """
    polygon = rect_polygon(x, y, w, h)
    return AnnotationShape(
        image_id="img",
        annotation_id=annotation_id,
        label=label,
        geometry_kind="POLYGON",
        bbox=(x, y, w, h),
        area=w * h,
        rle=encode_mask(polygon, height=CANVAS, width=CANVAS) if with_mask else None,
    )


def test_box_containment_is_one_when_fully_inside() -> None:
    assert box_containment((10, 10, 5, 5), (0, 0, 50, 50)) == pytest.approx(1.0)


def test_box_containment_is_zero_when_disjoint() -> None:
    assert box_containment((60, 60, 5, 5), (0, 0, 50, 50)) == 0.0


def test_box_containment_is_asymmetric() -> None:
    inner, outer = (10, 10, 5, 5), (0, 0, 50, 50)
    assert box_containment(inner, outer) > box_containment(outer, inner)


def test_mask_containment_matches_box_containment_for_rectangles() -> None:
    inner = encode_mask(rect_polygon(10, 10, 5, 5), height=CANVAS, width=CANVAS)
    outer = encode_mask(rect_polygon(0, 0, 50, 50), height=CANVAS, width=CANVAS)
    assert mask_containment(inner, outer, 25.0) == pytest.approx(1.0, abs=0.05)


def test_mask_containment_is_lower_than_box_containment_for_an_l_shape() -> None:
    # The reason masks are measured at all: a bounding box says an annotation
    # tucked into the notch of an L is enclosed, and the mask says it is not.
    l_shape = [[0.0, 0.0, 40.0, 0.0, 40.0, 10.0, 10.0, 10.0, 10.0, 40.0, 0.0, 40.0]]
    outer = encode_mask(l_shape, height=CANVAS, width=CANVAS)
    inner_polygon = rect_polygon(20, 20, 10, 10)
    inner = encode_mask(inner_polygon, height=CANVAS, width=CANVAS)
    inner_area = float(coco_mask.area(inner))

    outer_box = (0.0, 0.0, 40.0, 40.0)
    assert box_containment((20, 20, 10, 10), outer_box) == pytest.approx(1.0)
    assert mask_containment(inner, outer, inner_area) == pytest.approx(0.0)


def test_a_nested_same_class_annotation_is_described_as_enclosed() -> None:
    shapes = [shape("big", "person", 0, 0, 60, 60), shape("small", "person", 10, 10, 4, 4)]
    features = {f.annotation_id: f for f in describe_image(shapes, height=CANVAS, width=CANVAS)}
    small = features["small"]
    assert small.enclosed_bbox == pytest.approx(1.0)
    assert small.enclosed_mask == pytest.approx(1.0, abs=0.05)
    assert small.area_ratio == pytest.approx((4 * 4) / (60 * 60))
    assert small.container_id == "big"


def test_a_different_class_never_encloses() -> None:
    # A helmet inside a person is the normal case, not a fragment.
    shapes = [shape("p", "person", 0, 0, 60, 60), shape("h", "helmet_on_head", 10, 10, 4, 4)]
    features = {f.annotation_id: f for f in describe_image(shapes, height=CANVAS, width=CANVAS)}
    assert features["h"].enclosed_bbox == 0.0
    assert features["h"].container_id == ""


def test_a_lone_annotation_has_no_container() -> None:
    features = describe_image([shape("only", "person", 0, 0, 10, 10)], height=CANVAS, width=CANVAS)
    assert features[0].enclosed_bbox == 0.0
    assert features[0].area_ratio == 1.0


def test_relative_and_absolute_area_are_both_reported() -> None:
    features = describe_image([shape("a", "person", 0, 0, 10, 20)], height=CANVAS, width=CANVAS)
    assert features[0].absolute_area_px == pytest.approx(200.0, abs=1.0)
    assert features[0].relative_image_area == pytest.approx(200.0 / (CANVAS * CANVAS), abs=1e-4)


def test_an_annotation_without_a_mask_still_gets_box_features() -> None:
    # An annotation with no geometry must not vanish from the analysis.
    shapes = [
        shape("big", "person", 0, 0, 60, 60),
        shape("nogeo", "person", 10, 10, 4, 4, with_mask=False),
    ]
    features = {f.annotation_id: f for f in describe_image(shapes, height=CANVAS, width=CANVAS)}
    assert features["nogeo"].enclosed_bbox == pytest.approx(1.0)
    assert features["nogeo"].enclosed_mask == 0.0


def test_features_expose_no_provider_split_or_snapshot_membership() -> None:
    # The rule must be computable from the current state alone.
    fields = set(FragmentFeatures.__dataclass_fields__)
    assert not {field for field in fields if "split" in field or "v4" in field}


def test_evaluate_rule_scores_a_perfect_rule() -> None:
    features = describe_image(
        [shape("big", "person", 0, 0, 60, 60), shape("small", "person", 10, 10, 4, 4)],
        height=CANVAS,
        width=CANVAS,
    )
    outcome = evaluate_rule(
        "enclosed", features, lambda f: f.enclosed_bbox >= 0.95, {("img", "small")}
    )
    assert outcome.true_positives == 1
    assert outcome.false_positives == 0
    assert outcome.false_negatives == 0
    assert outcome.precision == 1.0
    assert outcome.recall == 1.0


def test_evaluate_rule_reports_a_false_positive() -> None:
    features = describe_image(
        [shape("big", "person", 0, 0, 60, 60), shape("small", "person", 10, 10, 4, 4)],
        height=CANVAS,
        width=CANVAS,
    )
    outcome = evaluate_rule("all", features, lambda f: True, {("img", "small")})
    assert outcome.selected == 2
    assert outcome.false_positives == 1
    assert outcome.precision == pytest.approx(0.5)


def test_evaluate_rule_reports_a_false_negative() -> None:
    features = describe_image([shape("a", "person", 0, 0, 10, 10)], height=CANVAS, width=CANVAS)
    outcome = evaluate_rule("none", features, lambda f: False, {("img", "a")})
    assert outcome.false_negatives == 1
    assert outcome.recall == 0.0
    assert outcome.precision == 0.0


def test_describing_an_image_is_order_independent() -> None:
    shapes = [shape("big", "person", 0, 0, 60, 60), shape("small", "person", 10, 10, 4, 4)]
    forward = {
        f.annotation_id: f.enclosed_bbox
        for f in describe_image(shapes, height=CANVAS, width=CANVAS)
    }
    backward = {
        f.annotation_id: f.enclosed_bbox
        for f in describe_image(list(reversed(shapes)), height=CANVAS, width=CANVAS)
    }
    assert forward == backward


def test_encode_mask_accepts_an_rle_mapping() -> None:
    array = np.zeros((CANVAS, CANVAS), dtype=np.uint8, order="F")
    array[10:20, 10:20] = 1
    encoded = coco_mask.encode(array)
    canonical = {"size": [CANVAS, CANVAS], "counts": encoded["counts"].decode("ascii")}
    assert float(coco_mask.area(encode_mask(canonical, height=CANVAS, width=CANVAS))) == 100.0
