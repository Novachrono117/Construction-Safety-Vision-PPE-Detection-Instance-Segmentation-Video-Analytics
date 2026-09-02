"""Bounding boxes derived from COCO segmentation geometry.

The project treats instance segmentation as the single source of truth and
derives detection boxes from it. Deriving a box is therefore not a convenience -
it is the operation that keeps the two tasks describing the same objects, so it
must handle **both** representations present in this export: polygon lists and
compressed COCO RLE.

Nothing here writes to the dataset. Phase 4 uses these functions to audit how far
the provider's supplied ``bbox`` values are from the geometry they claim to
enclose; phase 5 uses them to produce the detection view.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pycocotools import mask as coco_mask

DEFAULT_TOLERANCE_PX = 1.0
"""Default absolute tolerance, in pixels, when comparing two boxes.

Chosen to match the coarsest representation in play: a compressed RLE is a
rasterised binary mask, so a box derived from it is quantised to whole pixels
and can legitimately differ from a fractional polygon-derived box by up to one
pixel per edge. A tolerance below this would flag arithmetic that is correct.
"""


RLE_CANVAS_TOLERANCE_PX = 1.0
"""Slack allowed when checking a decoded RLE box against its declared canvas."""


class GeometryError(ValueError):
    """Raised when a segmentation cannot be converted to a bounding box."""


@dataclass(frozen=True)
class BBox:
    """An axis-aligned box in COCO ``[x, y, width, height]`` convention.

    Attributes:
        x: Left coordinate.
        y: Top coordinate.
        width: Extent along x.
        height: Extent along y.
    """

    x: float
    y: float
    width: float
    height: float

    @classmethod
    def from_sequence(cls, values: Any) -> BBox:
        """Build a box from a four-element sequence.

        Args:
            values: Sequence of ``[x, y, width, height]``.

        Returns:
            The parsed box.

        Raises:
            GeometryError: If the sequence is not four numbers.
        """
        try:
            x, y, width, height = (float(v) for v in values)
        except (TypeError, ValueError) as exc:
            msg = f"Expected four numeric bbox values, got {values!r}"
            raise GeometryError(msg) from exc
        return cls(x, y, width, height)

    @property
    def area(self) -> float:
        """Box area.

        Returns:
            ``width * height``.
        """
        return self.width * self.height

    def as_list(self) -> list[float]:
        """Return the box in COCO order.

        Returns:
            ``[x, y, width, height]``.
        """
        return [self.x, self.y, self.width, self.height]

    def corners(self) -> tuple[float, float, float, float]:
        """Return the box as ``(x_min, y_min, x_max, y_max)``.

        Returns:
            The corner representation.
        """
        return (self.x, self.y, self.x + self.width, self.y + self.height)


def polygon_bbox(segmentation: list[Any]) -> BBox:
    """Compute the tight box enclosing a polygon segmentation.

    Args:
        segmentation: COCO polygon segmentation: a list of rings, each a flat
            list of alternating x and y coordinates.

    Returns:
        The enclosing box.

    Raises:
        GeometryError: If no ring contains a usable coordinate pair.
    """
    xs: list[float] = []
    ys: list[float] = []
    for ring in segmentation:
        if not isinstance(ring, (list, tuple)) or len(ring) < 2:
            continue
        coordinates = [float(value) for value in ring]
        xs.extend(coordinates[0::2])
        ys.extend(coordinates[1::2])
    if not xs or not ys:
        msg = "Polygon segmentation contains no usable coordinates"
        raise GeometryError(msg)
    return BBox(min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys))


def rle_bbox(segmentation: dict[str, Any]) -> BBox:
    """Compute the box enclosing a compressed COCO RLE segmentation.

    Delegates to the reference COCO implementation rather than decoding the
    compressed run lengths by hand.

    Args:
        segmentation: RLE mapping with ``counts`` and ``size``.

    Returns:
        The enclosing box, quantised to whole pixels by the rasterised mask.

    Raises:
        GeometryError: If the RLE cannot be decoded.
    """
    if "counts" not in segmentation or "size" not in segmentation:
        msg = f"RLE segmentation needs 'counts' and 'size', got keys {sorted(segmentation)}"
        raise GeometryError(msg)
    payload = dict(segmentation)
    counts = payload["counts"]
    if isinstance(counts, str):
        payload["counts"] = counts.encode("utf-8")
    try:
        values = coco_mask.toBbox(payload)
    except Exception as exc:  # pycocotools raises bare exceptions on bad input
        msg = f"Could not decode RLE segmentation: {type(exc).__name__}"
        raise GeometryError(msg) from None

    box = BBox.from_sequence(values.tolist())
    # Measured: pycocotools does not reject malformed `counts`. It silently
    # returns nonsense - a garbage string on a 10x10 canvas yields a box over
    # 115 million pixels wide. Since the segmentation is this project's source of
    # truth for every derived detection box, that must fail loudly instead.
    try:
        height, width = (float(v) for v in segmentation["size"])
    except (TypeError, ValueError) as exc:
        msg = f"RLE 'size' must be [height, width], got {segmentation['size']!r}"
        raise GeometryError(msg) from exc
    x_min, y_min, x_max, y_max = box.corners()
    if x_min < -RLE_CANVAS_TOLERANCE_PX or y_min < -RLE_CANVAS_TOLERANCE_PX:
        msg = f"Decoded RLE box starts outside its own canvas: {box.as_list()}"
        raise GeometryError(msg)
    if x_max > width + RLE_CANVAS_TOLERANCE_PX or y_max > height + RLE_CANVAS_TOLERANCE_PX:
        msg = (
            f"Decoded RLE box {box.as_list()} exceeds its declared canvas "
            f"{[width, height]}; the segmentation is malformed"
        )
        raise GeometryError(msg)
    return box


def bbox_from_segmentation(segmentation: Any) -> BBox:
    """Derive the enclosing box from either segmentation representation.

    Args:
        segmentation: A COCO ``segmentation`` value.

    Returns:
        The enclosing box.

    Raises:
        GeometryError: If the representation is unsupported or empty.
    """
    if isinstance(segmentation, dict):
        return rle_bbox(segmentation)
    if isinstance(segmentation, list):
        if not segmentation:
            msg = "Empty segmentation has no bounding box"
            raise GeometryError(msg)
        return polygon_bbox(segmentation)
    msg = f"Unsupported segmentation type: {type(segmentation).__name__}"
    raise GeometryError(msg)


@dataclass(frozen=True)
class BBoxComparison:
    """Difference between a supplied box and one derived from geometry.

    Attributes:
        supplied: The box the export provides.
        derived: The box computed from the segmentation.
        deltas: Absolute per-corner differences ``(x_min, y_min, x_max, y_max)``.
        representation: ``"polygon"`` or ``"rle"``.
    """

    supplied: BBox
    derived: BBox
    deltas: tuple[float, float, float, float]
    representation: str

    @property
    def max_delta(self) -> float:
        """Largest absolute corner difference, in pixels.

        Returns:
            The maximum of :attr:`deltas`.
        """
        return max(self.deltas)

    def within(self, tolerance: float) -> bool:
        """Report whether every corner agrees within a tolerance.

        Args:
            tolerance: Absolute tolerance in pixels.

        Returns:
            ``True`` when the largest difference does not exceed the tolerance.
        """
        return self.max_delta <= tolerance


def compare_to_segmentation(
    supplied: Any,
    segmentation: Any,
    *,
    representation: str,
) -> BBoxComparison:
    """Compare a supplied bbox against the box its segmentation implies.

    Corners are compared rather than ``[x, y, w, h]`` so that a shifted edge is
    reported once rather than twice.

    Args:
        supplied: The bbox from the export.
        segmentation: The segmentation the bbox claims to enclose.
        representation: Representation label recorded on the result.

    Returns:
        The comparison.

    Raises:
        GeometryError: If either box cannot be built.
    """
    supplied_box = BBox.from_sequence(supplied)
    derived_box = bbox_from_segmentation(segmentation)
    deltas = tuple(
        abs(a - b) for a, b in zip(supplied_box.corners(), derived_box.corners(), strict=True)
    )
    return BBoxComparison(
        supplied=supplied_box,
        derived=derived_box,
        deltas=deltas,  # type: ignore[arg-type]
        representation=representation,
    )
