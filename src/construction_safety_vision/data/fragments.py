"""Geometric features for judging whether an annotation is a redundant fragment.

Phase 5A observed that the annotations added since the version-4 snapshot sit
inside annotations of their own class. Phase 5B has to turn that observation into
something a pipeline can apply, which means a rule computed from the **canonical
current state alone**: no provider split, no version-4 membership, no memorised
list of identifiers. Version 4 may be used to *evaluate* a candidate rule; it may
never be an input to it.

Everything here is per image and per class. An annotation is described by where
it sits relative to the other annotations of the same class on the same image:

* how much of it is swallowed by the same-class annotation that best encloses it,
  measured on bounding boxes and again on the actual masks - a bounding box
  overstates enclosure badly for the elongated, L-shaped regions vests occupy;
* how large it is relative to that enclosing annotation;
* how large it is in absolute pixels and as a fraction of the image.

The separation of *fragment* from *legitimate small object* is a judgement, so
this module computes the evidence and applies a rule that is handed to it. It
does not decide the thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pycocotools import mask as coco_mask


@dataclass(frozen=True)
class AnnotationShape:
    """One annotation reduced to what the fragment analysis needs.

    Attributes:
        image_id: Provider source-image identifier.
        annotation_id: Provider annotation identifier, unique within the image.
        label: Class name.
        geometry_kind: Geometry classification from phase 5A.
        bbox: Box in COCO ``[x, y, w, h]`` original-image pixels.
        area: Mask area in pixels, ``None`` when there is no geometry.
        rle: Encoded mask, ``None`` when there is no geometry.
    """

    image_id: str
    annotation_id: str
    label: str
    geometry_kind: str
    bbox: tuple[float, float, float, float]
    area: float | None
    rle: Any = None


@dataclass(frozen=True)
class FragmentFeatures:
    """Where one annotation sits relative to its own class on its own image.

    Attributes:
        image_id: Provider source-image identifier.
        annotation_id: Provider annotation identifier.
        label: Class name.
        geometry_kind: Geometry classification.
        enclosed_bbox: Largest fraction of this annotation's box lying inside a
            same-class annotation's box. Zero when the class is alone.
        enclosed_mask: The same measured on masks rather than boxes.
        area_ratio: This annotation's area over the area of the same-class
            annotation that encloses it best. One when nothing encloses it.
        relative_image_area: Mask area over image area.
        absolute_area_px: Mask area in pixels.
        container_id: Identifier of the best-enclosing same-class annotation.
    """

    image_id: str
    annotation_id: str
    label: str
    geometry_kind: str
    enclosed_bbox: float
    enclosed_mask: float
    area_ratio: float
    relative_image_area: float
    absolute_area_px: float
    container_id: str

    def as_row(self) -> dict[str, Any]:
        """Serialise for the committed analysis table.

        Returns:
            A mapping of column name to value.
        """
        return {
            "image_id": self.image_id,
            "annotation_id": self.annotation_id,
            "label": self.label,
            "geometry_kind": self.geometry_kind,
            "enclosed_bbox": round(self.enclosed_bbox, 6),
            "enclosed_mask": round(self.enclosed_mask, 6),
            "area_ratio": round(self.area_ratio, 8),
            "relative_image_area": round(self.relative_image_area, 8),
            "absolute_area_px": round(self.absolute_area_px, 2),
            "container_id": self.container_id,
        }


def box_containment(
    inner: tuple[float, float, float, float], outer: tuple[float, float, float, float]
) -> float:
    """Fraction of one box lying inside another.

    Args:
        inner: Box being tested for enclosure, as ``(x, y, w, h)``.
        outer: Box that might enclose it.

    Returns:
        The enclosed fraction of ``inner``, zero when it has no area.
    """
    ix = max(0.0, min(inner[0] + inner[2], outer[0] + outer[2]) - max(inner[0], outer[0]))
    iy = max(0.0, min(inner[1] + inner[3], outer[1] + outer[3]) - max(inner[1], outer[1]))
    area = inner[2] * inner[3]
    return (ix * iy) / area if area > 0 else 0.0


def encode_mask(segmentation: Any, *, height: int, width: int) -> Any:
    """Encode a canonical segmentation as a single COCO RLE.

    Args:
        segmentation: Polygon ring list, or an RLE mapping whose ``counts`` is an
            ASCII string.
        height: Image height in pixels.
        width: Image width in pixels.

    Returns:
        An RLE suitable for :mod:`pycocotools` set operations.
    """
    if isinstance(segmentation, dict):
        return {
            "size": list(segmentation["size"]),
            "counts": segmentation["counts"].encode("ascii"),
        }
    return coco_mask.merge(coco_mask.frPyObjects(segmentation, height, width))


def mask_containment(inner: Any, outer: Any, inner_area: float) -> float:
    """Fraction of one mask lying inside another.

    Args:
        inner: Encoded mask being tested for enclosure.
        outer: Encoded mask that might enclose it.
        inner_area: Area of ``inner`` in pixels.

    Returns:
        The enclosed fraction of ``inner``, zero when it has no area.
    """
    if inner_area <= 0:
        return 0.0
    intersection = float(coco_mask.area(coco_mask.merge([inner, outer], intersect=1)))
    return intersection / inner_area


def describe_image(
    shapes: list[AnnotationShape], *, height: int, width: int
) -> list[FragmentFeatures]:
    """Compute fragment features for every annotation on one image.

    Args:
        shapes: Every annotation on the image.
        height: Image height in pixels.
        width: Image width in pixels.

    Returns:
        One feature record per annotation, in the given order.
    """
    image_area = float(height * width)
    features: list[FragmentFeatures] = []

    for shape in shapes:
        peers = [other for other in shapes if other is not shape and other.label == shape.label]
        own_area = float(shape.area or 0.0)

        best_box, best_mask, best_ratio, container = 0.0, 0.0, 1.0, ""
        for peer in peers:
            enclosed_box = box_containment(shape.bbox, peer.bbox)
            if enclosed_box <= best_box:
                continue
            best_box = enclosed_box
            container = peer.annotation_id
            peer_area = float(peer.area or 0.0)
            best_ratio = own_area / peer_area if peer_area > 0 else 1.0
            best_mask = (
                mask_containment(shape.rle, peer.rle, own_area)
                if shape.rle is not None and peer.rle is not None
                else 0.0
            )

        features.append(
            FragmentFeatures(
                image_id=shape.image_id,
                annotation_id=shape.annotation_id,
                label=shape.label,
                geometry_kind=shape.geometry_kind,
                enclosed_bbox=best_box,
                enclosed_mask=best_mask,
                area_ratio=best_ratio,
                relative_image_area=own_area / image_area if image_area > 0 else 0.0,
                absolute_area_px=own_area,
                container_id=container,
            )
        )
    return features


@dataclass(frozen=True)
class RuleOutcome:
    """How well a candidate rule reproduces a reference set.

    Attributes:
        name: Human-readable rule description.
        selected: Annotations the rule matched.
        true_positives: Matched annotations that are in the reference set.
        false_positives: Matched annotations that are not.
        false_negatives: Reference annotations the rule did not match.
    """

    name: str
    selected: int
    true_positives: int
    false_positives: int
    false_negatives: int

    @property
    def precision(self) -> float:
        """Share of matches that are in the reference set.

        Returns:
            Precision, or zero when nothing was selected.
        """
        return self.true_positives / self.selected if self.selected else 0.0

    @property
    def recall(self) -> float:
        """Share of the reference set the rule matched.

        Returns:
            Recall, or zero when the reference set is empty.
        """
        total = self.true_positives + self.false_negatives
        return self.true_positives / total if total else 0.0

    def as_row(self) -> dict[str, Any]:
        """Serialise for the committed rule table.

        Returns:
            A mapping of column name to value.
        """
        return {
            "rule": self.name,
            "selected": self.selected,
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
        }


def evaluate_rule(
    name: str,
    features: list[FragmentFeatures],
    predicate: Any,
    reference: set[tuple[str, str]],
) -> RuleOutcome:
    """Score a candidate rule against a reference set.

    The reference set is a development-time yardstick, never a rule input.

    Args:
        name: Human-readable rule description.
        features: Feature records for every annotation.
        predicate: Callable taking a feature record and returning whether the
            rule matches it.
        reference: ``(image id, annotation id)`` pairs the rule is compared to.

    Returns:
        The scored outcome.
    """
    selected = {(f.image_id, f.annotation_id) for f in features if predicate(f)}
    true_positives = selected & reference
    return RuleOutcome(
        name=name,
        selected=len(selected),
        true_positives=len(true_positives),
        false_positives=len(selected - reference),
        false_negatives=len(reference - selected),
    )
