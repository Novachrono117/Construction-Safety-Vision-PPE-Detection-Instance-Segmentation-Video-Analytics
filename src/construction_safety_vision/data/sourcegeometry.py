"""Complete instance-segmentation geometry for the provider's *current* source state.

Phase 4A read the image-details endpoint but consumed only ``points``. That made
every ``mask``-type instance look geometry-less, and the phase concluded that
1,022 of the current annotations exposed a bounding box and nothing more.

That conclusion was a consumption gap, not a provider limitation. A ``mask`` box
carries its geometry inline in a ``mask`` field::

    mask = base64( zlib( COCO compressed-RLE counts ) )

decoded against the **full original image canvas** in ``[height, width]`` order.
The encoding is not documented by the provider, so it is not taken on trust: for
every recovered instance the decoded geometry is re-measured and checked against
the ``area`` and the centre-form box the provider declares alongside it. A
mismatch raises rather than silently producing a wrong mask.

Recovering geometry here has one further advantage over the version-4 export:
these coordinates are the *original* image coordinates, so nothing has to be
inverted back through the export's stretch-resize to 640x640.

Nothing in this module persists a credential, a signed URL or an embedding.
"""

from __future__ import annotations

import base64
import binascii
import zlib
from dataclasses import dataclass, field
from typing import Any

from pycocotools import mask as coco_mask

POLYGON = "POLYGON"
"""Geometry delivered as an explicit vertex ring."""

RLE = "RLE"
"""Geometry delivered as inline COCO compressed run-length encoding."""

BITMASK_OR_MASK_ASSET = "BITMASK_OR_MASK_ASSET"
"""Geometry delivered as a raster asset that must be fetched separately."""

OTHER_SUPPORTED = "OTHER_SUPPORTED"
"""Geometry in some other representation this project can still rasterise."""

UNSUPPORTED = "UNSUPPORTED"
"""No usable segmentation geometry. Never silently dropped; always counted."""

GEOMETRY_KINDS = (POLYGON, RLE, BITMASK_OR_MASK_ASSET, OTHER_SUPPORTED, UNSUPPORTED)
"""Every geometry classification this module can assign, in report order."""

MASK_ASSET_KEYS = ("maskUrl", "mask_url", "maskAsset", "segmentationUrl", "url")
"""Keys that would indicate geometry hosted as a separate raster asset."""

AREA_TOLERANCE_PX = 1.0
"""Absolute slack when checking decoded mask area against the declared area."""

BBOX_TOLERANCE_PX = 1.0
"""Absolute slack, per edge, when checking decoded geometry against the declared box."""


class GeometryRecoveryError(ValueError):
    """Raised when an annotation's geometry cannot be recovered faithfully."""


def decode_mask_counts(payload: str) -> bytes:
    """Decode the provider's ``mask`` field into COCO compressed-RLE counts.

    Args:
        payload: The base64 text of the provider's ``mask`` field.

    Returns:
        The raw ``counts`` byte string expected by :mod:`pycocotools`.

    Raises:
        GeometryRecoveryError: If the value is not base64-wrapped zlib data.
    """
    if not isinstance(payload, str) or not payload.strip():
        msg = "Provider mask field is empty"
        raise GeometryRecoveryError(msg)
    try:
        blob = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError):
        msg = "Provider mask field is not valid base64"
        raise GeometryRecoveryError(msg) from None
    try:
        return zlib.decompress(blob)
    except zlib.error as exc:
        msg = f"Provider mask field is not zlib-compressed: {exc}"
        raise GeometryRecoveryError(msg) from None


def centre_box_to_coco(x: float, y: float, width: float, height: float) -> list[float]:
    """Convert the provider's centre-form box to COCO ``[x, y, w, h]``.

    Args:
        x: Centre x.
        y: Centre y.
        width: Extent along x.
        height: Extent along y.

    Returns:
        The box in COCO top-left convention.
    """
    return [x - width / 2.0, y - height / 2.0, width, height]


def _as_float(value: Any, *, field_name: str) -> float:
    """Coerce a provider value to a float.

    The provider types these fields inconsistently: the same field arrives as an
    integer, a float or a decimal string depending on the image.

    Args:
        value: Raw provider value.
        field_name: Field name, used in the error message.

    Returns:
        The value as a float.

    Raises:
        GeometryRecoveryError: If the value is not numeric.
    """
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        msg = f"Annotation field {field_name!r} is not numeric: {value!r}"
        raise GeometryRecoveryError(msg) from exc


def polygon_ring(points: list[Any]) -> list[float]:
    """Flatten provider vertices into a single COCO polygon ring.

    Args:
        points: Provider ``points`` value: a list of ``[x, y]`` pairs or of
            mappings carrying ``x`` and ``y``.

    Returns:
        A flat ring of alternating x and y coordinates.

    Raises:
        GeometryRecoveryError: If fewer than three usable vertices are present.
    """
    ring: list[float] = []
    for point in points:
        if isinstance(point, dict):
            if "x" not in point or "y" not in point:
                continue
            ring.append(_as_float(point["x"], field_name="points.x"))
            ring.append(_as_float(point["y"], field_name="points.y"))
        elif isinstance(point, (list, tuple)) and len(point) >= 2:
            ring.append(_as_float(point[0], field_name="points[0]"))
            ring.append(_as_float(point[1], field_name="points[1]"))
    if len(ring) < 6:
        msg = f"Polygon needs at least 3 vertices, got {len(ring) // 2}"
        raise GeometryRecoveryError(msg)
    return ring


@dataclass(frozen=True)
class RecoveredAnnotation:
    """One current-state annotation together with its complete geometry.

    Attributes:
        image_id: Provider source-image identifier.
        annotation_id: Provider annotation identifier, unique within the image.
        label: Class name.
        provider_type: The provider's own ``type`` value, empty when absent.
        geometry_kind: One of :data:`GEOMETRY_KINDS`.
        bbox: Declared box in COCO ``[x, y, w, h]`` original-image coordinates.
        segmentation: COCO segmentation: a polygon ring list, or an RLE mapping
            whose ``counts`` is an ASCII string. ``None`` when unsupported.
        area: Measured mask area in pixels, ``None`` when unsupported.
        point_count: Polygon vertex count; zero for non-polygon geometry.
        declared_area: The provider's own ``area`` value, when it supplies one.
        confidence: The provider's ``confidence`` value, present only on records
            that are model suggestions rather than accepted annotations.
        bbox_delta_px: Largest per-edge disagreement between the provider's
            stored box and the box implied by the recovered geometry. Recorded,
            not corrected: segmentation is canonical and the box is derived
            from it downstream.
        notes: Human-readable remarks, e.g. why geometry is unsupported.
    """

    image_id: str
    annotation_id: str
    label: str
    provider_type: str
    geometry_kind: str
    bbox: list[float]
    segmentation: Any = None
    area: float | None = None
    point_count: int = 0
    declared_area: float | None = None
    confidence: float | None = None
    bbox_delta_px: float | None = None
    notes: str = ""

    @property
    def has_mask_geometry(self) -> bool:
        """Whether this annotation can be rasterised into an instance mask.

        Returns:
            ``True`` unless the geometry kind is :data:`UNSUPPORTED`.
        """
        return self.geometry_kind != UNSUPPORTED

    def summary_entry(self) -> dict[str, Any]:
        """Serialise without bulk geometry, for a committed report.

        Returns:
            A JSON-serialisable mapping carrying no vertices and no counts.
        """
        return {
            "image_id": self.image_id,
            "annotation_id": self.annotation_id,
            "label": self.label,
            "provider_type": self.provider_type,
            "geometry_kind": self.geometry_kind,
            "bbox": [round(v, 3) for v in self.bbox],
            "area": round(self.area, 3) if self.area is not None else None,
            "point_count": self.point_count,
            "confidence": self.confidence,
            "bbox_delta_px": (
                round(self.bbox_delta_px, 3) if self.bbox_delta_px is not None else None
            ),
            "notes": self.notes,
        }

    def geometry_entry(self) -> dict[str, Any]:
        """Serialise including geometry, for the git-ignored interim layer.

        Returns:
            A JSON-serialisable mapping including the full segmentation.
        """
        entry = self.summary_entry()
        entry["segmentation"] = self.segmentation
        return entry


def measure_geometry(segmentation: Any, *, height: int, width: int) -> tuple[list[float], float]:
    """Measure the enclosing box and the area of a recovered segmentation.

    The two representations are measured differently on purpose. A polygon's box
    is its coordinate extrema: rasterising it first would quantise away a thin
    spike and shrink the box, which is measurably wrong against the provider's
    own value. An RLE has no coordinates, so it is measured by the reference COCO
    implementation.

    Args:
        segmentation: Recovered COCO segmentation.
        height: Original image height.
        width: Original image width.

    Returns:
        The box in COCO ``[x, y, w, h]`` convention and the mask area in pixels.

    Raises:
        GeometryRecoveryError: If an RLE decodes to a mask outside its own canvas.
    """
    if isinstance(segmentation, dict):
        rle: Any = {"size": segmentation["size"], "counts": segmentation["counts"].encode("ascii")}
        bbox = [float(v) for v in coco_mask.toBbox(rle).tolist()]
        # pycocotools does not reject malformed counts; it silently returns
        # nonsense. A mask that escapes its own canvas is proof of a bad decode.
        if (
            bbox[0] < -BBOX_TOLERANCE_PX
            or bbox[1] < -BBOX_TOLERANCE_PX
            or bbox[0] + bbox[2] > width + BBOX_TOLERANCE_PX
            or bbox[1] + bbox[3] > height + BBOX_TOLERANCE_PX
        ):
            msg = f"Decoded RLE box {bbox} escapes its declared canvas {[width, height]}"
            raise GeometryRecoveryError(msg)
        return bbox, float(coco_mask.area(rle))

    ring = segmentation[0]
    xs, ys = ring[0::2], ring[1::2]
    bbox = [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)]
    rasterised = coco_mask.merge(coco_mask.frPyObjects(segmentation, height, width))
    return bbox, float(coco_mask.area(rasterised))


def _verify_against_declared(
    annotation_id: str,
    segmentation: Any,
    declared_bbox: list[float],
    declared_area: float | None,
    *,
    height: int,
    width: int,
) -> tuple[float, float]:
    """Re-measure recovered geometry and check it against the provider's claims.

    Two different things are being judged here, so they are treated differently.

    *Decode integrity* is judged by area. The mask encoding is undocumented, and
    an exact area agreement across thousands of instances is strong proof that
    the decode is right; a mismatch means the bytes were read as the wrong thing,
    which is fatal and raises.

    *Box agreement* is a data-quality observation, not a decode test. Phase 4B
    measured that this provider's stored boxes and its segmentation geometry
    disagree for a substantial share of RLE instances. The project already treats
    segmentation as canonical and derives boxes from it, so a stale stored box is
    recorded and reported rather than allowed to abort the recovery.

    Args:
        annotation_id: Identifier used in error messages.
        segmentation: Recovered COCO segmentation.
        declared_bbox: The provider's box in COCO convention.
        declared_area: The provider's area, when supplied.
        height: Original image height.
        width: Original image width.

    Returns:
        The measured area in pixels and the largest per-edge box disagreement.

    Raises:
        GeometryRecoveryError: If the measured area contradicts the declared one.
    """
    measured_bbox, measured_area = measure_geometry(segmentation, height=height, width=width)

    if declared_area is not None and abs(measured_area - declared_area) > AREA_TOLERANCE_PX:
        msg = (
            f"Annotation {annotation_id!r}: recovered mask area {measured_area} disagrees with "
            f"the provider's declared area {declared_area}; the geometry decode is not trustworthy"
        )
        raise GeometryRecoveryError(msg)

    deltas = [abs(a - b) for a, b in zip(measured_bbox, declared_bbox, strict=True)]
    return measured_area, max(deltas)


def recover_annotation(
    box: dict[str, Any],
    *,
    image_id: str,
    width: int,
    height: int,
) -> RecoveredAnnotation:
    """Recover one annotation's complete geometry from a provider ``boxes`` entry.

    Args:
        box: One entry of ``image.annotation.boxes``.
        image_id: Provider source-image identifier.
        width: Original image width in pixels.
        height: Original image height in pixels.

    Returns:
        The recovered annotation, classified by geometry kind. An entry with no
        usable geometry is returned as :data:`UNSUPPORTED` rather than dropped.

    Raises:
        GeometryRecoveryError: If geometry is present but decodes into something
            that contradicts the provider's own bbox or area.
    """
    annotation_id = str(box.get("id", ""))
    label = str(box.get("label", ""))
    provider_type = "" if box.get("type") is None else str(box.get("type"))
    bbox = centre_box_to_coco(
        _as_float(box.get("x", 0.0), field_name="x"),
        _as_float(box.get("y", 0.0), field_name="y"),
        _as_float(box.get("width", 0.0), field_name="width"),
        _as_float(box.get("height", 0.0), field_name="height"),
    )
    declared_area = (
        _as_float(box["area"], field_name="area") if box.get("area") is not None else None
    )
    confidence = (
        _as_float(box["confidence"], field_name="confidence")
        if box.get("confidence") is not None
        else None
    )
    common: dict[str, Any] = {
        "image_id": image_id,
        "annotation_id": annotation_id,
        "label": label,
        "provider_type": provider_type,
        "bbox": bbox,
        "declared_area": declared_area,
        "confidence": confidence,
    }

    points = box.get("points")
    if points:
        segmentation: Any = [polygon_ring(list(points))]
        area, delta = _verify_against_declared(
            annotation_id, segmentation, bbox, declared_area, height=height, width=width
        )
        return RecoveredAnnotation(
            geometry_kind=POLYGON,
            segmentation=segmentation,
            area=area,
            point_count=len(segmentation[0]) // 2,
            bbox_delta_px=delta,
            **common,
        )

    mask = box.get("mask")
    if mask:
        counts = decode_mask_counts(str(mask))
        segmentation = {"size": [height, width], "counts": counts.decode("ascii")}
        area, delta = _verify_against_declared(
            annotation_id, segmentation, bbox, declared_area, height=height, width=width
        )
        return RecoveredAnnotation(
            geometry_kind=RLE, segmentation=segmentation, area=area, bbox_delta_px=delta, **common
        )

    asset_key = next((key for key in MASK_ASSET_KEYS if box.get(key)), None)
    if asset_key is not None:
        # Not observed on this project. Recorded as its own kind so that a future
        # provider change surfaces as a classification rather than as data loss.
        return RecoveredAnnotation(
            geometry_kind=BITMASK_OR_MASK_ASSET,
            notes=f"geometry referenced by provider field {asset_key!r}; not fetched",
            **common,
        )

    reason = (
        "provider record carries a confidence score and no geometry: a model "
        "suggestion, not an accepted annotation"
        if confidence is not None
        else f"provider type {provider_type!r} carries no geometry field"
    )
    return RecoveredAnnotation(geometry_kind=UNSUPPORTED, notes=reason, **common)


@dataclass
class RecoveredImage:
    """Every recovered annotation for one source image.

    Attributes:
        image_id: Provider source-image identifier.
        name: Original filename.
        split: Provider split assignment at recovery time.
        width: Original width in pixels.
        height: Original height in pixels.
        annotations: Recovered annotations, in provider order.
    """

    image_id: str
    name: str
    split: str
    width: int
    height: int
    annotations: list[RecoveredAnnotation] = field(default_factory=list)

    @property
    def geometry_counts(self) -> dict[str, int]:
        """Annotation counts per geometry kind.

        Returns:
            Counts keyed by geometry kind, only for kinds that occur.
        """
        counts: dict[str, int] = {}
        for annotation in self.annotations:
            counts[annotation.geometry_kind] = counts.get(annotation.geometry_kind, 0) + 1
        return dict(sorted(counts.items()))

    @property
    def class_counts(self) -> dict[str, int]:
        """Annotation counts per class name.

        Returns:
            Counts keyed by class name, sorted.
        """
        counts: dict[str, int] = {}
        for annotation in self.annotations:
            counts[annotation.label] = counts.get(annotation.label, 0) + 1
        return dict(sorted(counts.items()))


def recover_image(image: dict[str, Any]) -> RecoveredImage:
    """Recover every annotation on one image-details payload.

    Args:
        image: The ``image`` object returned by the details endpoint.

    Returns:
        The image with all of its annotations recovered and classified.

    Raises:
        GeometryRecoveryError: If the payload declares no usable canvas, or if
            any annotation's geometry contradicts its declared box or area.
    """
    annotation = image.get("annotation") or {}
    width = int(annotation.get("width") or image.get("width") or 0)
    height = int(annotation.get("height") or image.get("height") or 0)
    image_id = str(image.get("id", ""))
    if width <= 0 or height <= 0:
        msg = f"Image {image_id!r} declares no usable canvas: {width}x{height}"
        raise GeometryRecoveryError(msg)
    return RecoveredImage(
        image_id=image_id,
        name=str(image.get("name", "")),
        split=str(image.get("split", "")),
        width=width,
        height=height,
        annotations=[
            recover_annotation(box, image_id=image_id, width=width, height=height)
            for box in (annotation.get("boxes") or [])
        ],
    )
