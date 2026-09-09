"""Measure what a canonical instance mask loses in the YOLO segmentation format.

Phase 8A. Every number here comes from comparing two binary masks on the
original source-image canvas: the canonical one, decoded by pycocotools from the
phase 5D COCO document, and a reconstruction rasterised from the label row as it
was **written to disk and read back**. Comparing against an in-memory object
would skip serialisation, float32 parsing and the integer snap, which are three
of the places fidelity is actually lost.

One methodological point decides whether this audit is honest.

**pycocotools and OpenCV do not rasterise identical geometry into identical
pixels.** They differ on boundary-pixel inclusion. So even a polygon carried
through unchanged, coordinate for coordinate, will not reproduce the canonical
mask exactly - and reporting that gap as "YOLO format loss" would blame the
format for a rasteriser convention.

Every instance is therefore measured at **three** levels, so that each stage
gets charged only for what it costs:

1. ``control_iou`` - canonical mask against an OpenCV rasterisation of the
   component rings *before* any merging or serialisation. This isolates contour
   extraction and the rasteriser convention. No conversion can beat it.
2. ``merged_iou`` - the same, after components have been bridged into the single
   ring the format allows, still at full float64. The drop from ``control_iou``
   is what **component joining** costs.
3. ``mask_iou`` - the ring as written to disk and read back: normalised, rounded
   to the configured precision, parsed as float32, denormalised, and rasterised
   through the framework's own ``int32`` cast. The drop from ``merged_iou`` is
   what **serialisation and coordinate quantisation** cost.

That third stage turned out to matter more than the second, which is not what a
reader would guess: an integer contour coordinate that survives normalisation as
99.99993 truncates to 99, shifting a boundary pixel inward. Without the
decomposition, that loss would have been misattributed to component joining.

The metrics are deliberately several. A single mean IoU over 1726 instances
would let 843 near-exact polygons absorb whatever the 881 RLE masks lose, and
would hide entirely the two structural approximations - filling a hole and
bridging a gap - that add area rather than nibbling at a boundary.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = [
    "AREA_ERROR_BANDS",
    "IOU_BANDS",
    "FidelityRow",
    "MaskComparison",
    "area_error_band_counts",
    "band_counts",
    "compare_masks",
    "describe",
    "mask_bbox",
    "quartile_edges",
    "rasterise_ring",
    "summarise",
]

#: Descriptive IoU bands, in report order. Boundaries are half-open upwards.
IOU_BANDS: tuple[tuple[str, float, float], ...] = (
    ("iou_exact", 1.0, float("inf")),
    ("iou_ge_0.99", 0.99, 1.0),
    ("iou_0.95_to_0.99", 0.95, 0.99),
    ("iou_0.90_to_0.95", 0.90, 0.95),
    ("iou_lt_0.90", float("-inf"), 0.90),
)

#: Absolute relative area-error thresholds reported as exceedance counts.
AREA_ERROR_BANDS: tuple[float, ...] = (0.01, 0.02, 0.05, 0.10)

#: Percentiles reported for every distribution.
PERCENTILES: tuple[int, ...] = (1, 5, 25, 50, 75, 95, 99)


def rasterise_ring(
    ring: np.ndarray, *, height: int, width: int, downsample_ratio: int = 1
) -> np.ndarray:
    """Rasterise one ring the way Ultralytics does.

    Delegates to ``ultralytics.data.utils.polygon2mask`` rather than calling
    OpenCV directly, so the reconstruction is the framework's own - including
    its cast of coordinates to ``int32``, which truncates toward zero and is
    itself a source of sub-pixel loss worth measuring rather than avoiding.

    Args:
        ring: An ``(N, 2)`` array of xy pixel coordinates.
        height: Canvas height.
        width: Canvas width.
        downsample_ratio: Passed through; 1 keeps the original canvas.

    Returns:
        A ``(height, width)`` array of 0 and 1, dtype ``uint8``.
    """
    from ultralytics.data.utils import polygon2mask

    flat = np.asarray(ring, dtype=np.float64).reshape(-1)
    mask = polygon2mask((height, width), [flat], color=1, downsample_ratio=downsample_ratio)
    return (np.asarray(mask) > 0).astype(np.uint8)


def mask_bbox(mask: np.ndarray) -> tuple[float, float, float, float] | None:
    """Compute a mask's tight bounding box.

    Args:
        mask: A binary mask.

    Returns:
        ``(x_min, y_min, x_max, y_max)`` in pixel coordinates, or ``None`` when
        the mask is empty.
    """
    rows = np.flatnonzero(mask.any(axis=1))
    columns = np.flatnonzero(mask.any(axis=0))
    if rows.size == 0 or columns.size == 0:
        return None
    return (
        float(columns[0]),
        float(rows[0]),
        float(columns[-1] + 1),
        float(rows[-1] + 1),
    )


@dataclass(frozen=True)
class MaskComparison:
    """One canonical mask against one reconstruction.

    Attributes:
        intersection: Pixels present in both.
        union: Pixels present in either.
        canonical_area: Canonical mask area in pixels.
        reconstructed_area: Reconstructed mask area in pixels.
        false_positive: Pixels the reconstruction adds.
        false_negative: Pixels the reconstruction drops.
        bbox_max_delta: Largest per-edge bounding-box difference in pixels, or
            ``None`` when either mask is empty.
    """

    intersection: int
    union: int
    canonical_area: int
    reconstructed_area: int
    false_positive: int
    false_negative: int
    bbox_max_delta: float | None

    @property
    def iou(self) -> float:
        """Mask intersection over union.

        Returns:
            The IoU, or 1.0 when both masks are empty.
        """
        return 1.0 if self.union == 0 else self.intersection / self.union

    @property
    def dice(self) -> float:
        """Dice coefficient.

        Returns:
            The coefficient, or 1.0 when both masks are empty.
        """
        total = self.canonical_area + self.reconstructed_area
        return 1.0 if total == 0 else (2.0 * self.intersection) / total

    @property
    def absolute_area_error(self) -> int:
        """Absolute difference in mask area, in pixels.

        Returns:
            The difference.
        """
        return abs(self.reconstructed_area - self.canonical_area)

    @property
    def relative_area_error(self) -> float:
        """Absolute area error as a fraction of the canonical area.

        Returns:
            The fraction, or 0.0 when the canonical mask is empty.
        """
        if self.canonical_area == 0:
            return 0.0
        return self.absolute_area_error / self.canonical_area

    @property
    def signed_relative_area_error(self) -> float:
        """Signed area change as a fraction of the canonical area.

        Positive means the reconstruction is larger, which is what filling a
        hole or bridging a gap does. The sign is kept because it separates a
        structural approximation from boundary noise.

        Returns:
            The signed fraction, or 0.0 when the canonical mask is empty.
        """
        if self.canonical_area == 0:
            return 0.0
        return (self.reconstructed_area - self.canonical_area) / self.canonical_area


def compare_masks(canonical: np.ndarray, reconstructed: np.ndarray) -> MaskComparison:
    """Compare two binary masks on the same canvas.

    Args:
        canonical: The canonical mask.
        reconstructed: The mask rebuilt from the written label.

    Returns:
        The comparison.

    Raises:
        ValueError: If the masks are not the same shape. Comparing across
            canvases would silently produce a meaningless number.
    """
    if canonical.shape != reconstructed.shape:
        msg = (
            f"masks have different shapes {canonical.shape} and {reconstructed.shape}; a "
            "fidelity comparison across canvases is meaningless"
        )
        raise ValueError(msg)
    left = canonical > 0
    right = reconstructed > 0
    intersection = int(np.logical_and(left, right).sum())
    union = int(np.logical_or(left, right).sum())
    canonical_area = int(left.sum())
    reconstructed_area = int(right.sum())

    left_box = mask_bbox(left)
    right_box = mask_bbox(right)
    if left_box is None or right_box is None:
        delta = None
    else:
        delta = max(abs(a - b) for a, b in zip(left_box, right_box, strict=True))

    return MaskComparison(
        intersection=intersection,
        union=union,
        canonical_area=canonical_area,
        reconstructed_area=reconstructed_area,
        false_positive=int(np.logical_and(right, ~left).sum()),
        false_negative=int(np.logical_and(left, ~right).sum()),
        bbox_max_delta=delta,
    )


@dataclass
class FidelityRow:
    """One development annotation's complete audit record.

    Attributes:
        source_image_id: The frozen source image identifier.
        annotation_id: The canonical annotation identifier.
        split: ``train`` or ``validation``.
        class_name: The frozen class name.
        canonical_geometry_type: Polygon, RLE or synthetic rectangle.
        canonical_area_px: Canonical mask area.
        reconstructed_area_px: Reconstructed mask area.
        connected_components: Components in the canonical mask.
        hole_count: Interior rings in the canonical mask.
        hole_pixels: Area of those interior rings.
        thinness: Perimeter over the square root of area.
        adapter_point_count: Points written in the label row.
        component_rings: Rings before merging.
        joined: Whether components were bridged.
        mask_iou: Canonical against reconstruction.
        dice: Dice coefficient.
        control_iou: Canonical against an OpenCV rasterisation of the
            unconverted component rings. The conversion cannot beat this.
        merged_iou: Canonical against the merged single ring at full float64,
            before serialisation. Its gap below ``control_iou`` is the cost of
            component joining; the gap from it down to ``mask_iou`` is the cost
            of serialisation and coordinate quantisation.
        absolute_area_error_px: Absolute area difference.
        relative_area_error: Absolute area difference over canonical area.
        signed_relative_area_error: Signed area difference over canonical area.
        fp_pixels: Pixels added.
        fn_pixels: Pixels dropped.
        bbox_max_delta_px: Largest bounding-box edge difference.
        reason_flags: Deterministic attribution of any material deviation.
    """

    source_image_id: str
    annotation_id: str
    split: str
    class_name: str
    canonical_geometry_type: str
    canonical_area_px: int
    reconstructed_area_px: int
    connected_components: int
    hole_count: int
    hole_pixels: int
    thinness: float
    adapter_point_count: int
    component_rings: int
    joined: bool
    mask_iou: float
    dice: float
    control_iou: float
    merged_iou: float
    absolute_area_error_px: int
    relative_area_error: float
    signed_relative_area_error: float
    fp_pixels: int
    fn_pixels: int
    bbox_max_delta_px: float | None
    reason_flags: tuple[str, ...]

    def as_row(self) -> dict[str, Any]:
        """Render the record for the machine-readable table.

        Returns:
            A mapping of column name to value.
        """
        return {
            "source_image_id": self.source_image_id,
            "annotation_id": self.annotation_id,
            "split": self.split,
            "class": self.class_name,
            "canonical_geometry_type": self.canonical_geometry_type,
            "canonical_area_px": self.canonical_area_px,
            "reconstructed_area_px": self.reconstructed_area_px,
            "connected_components": self.connected_components,
            "component_rings": self.component_rings,
            "joined": int(self.joined),
            "hole_count": self.hole_count,
            "hole_pixels": self.hole_pixels,
            "thinness": f"{self.thinness:.6f}",
            "adapter_point_count": self.adapter_point_count,
            "mask_iou": f"{self.mask_iou:.9f}",
            "dice": f"{self.dice:.9f}",
            "control_iou": f"{self.control_iou:.9f}",
            "merged_iou": f"{self.merged_iou:.9f}",
            "join_loss": f"{self.control_iou - self.merged_iou:.9f}",
            "serialization_loss": f"{self.merged_iou - self.mask_iou:.9f}",
            "absolute_area_error_px": self.absolute_area_error_px,
            "relative_area_error": f"{self.relative_area_error:.9f}",
            "signed_relative_area_error": f"{self.signed_relative_area_error:.9f}",
            "fp_pixels": self.fp_pixels,
            "fn_pixels": self.fn_pixels,
            "bbox_max_delta_px": (
                "" if self.bbox_max_delta_px is None else f"{self.bbox_max_delta_px:.3f}"
            ),
            "reason_flags": "|".join(self.reason_flags),
        }


def describe(values: Sequence[float]) -> dict[str, float]:
    """Summarise a distribution with the percentiles the audit reports.

    Args:
        values: The sample.

    Returns:
        Count, minimum, mean, percentiles and maximum. Percentiles use linear
        interpolation, stated because a different convention would move the
        reported figures.
    """
    if not values:
        return {"count": 0}
    array = np.asarray(values, dtype=np.float64)
    summary: dict[str, float] = {
        "count": int(array.size),
        "min": float(array.min()),
        "mean": float(array.mean()),
        "max": float(array.max()),
    }
    for percentile in PERCENTILES:
        key = "median" if percentile == 50 else f"p{percentile:02d}"
        summary[key] = float(np.percentile(array, percentile, method="linear"))
    return summary


def band_counts(values: Sequence[float]) -> dict[str, dict[str, float]]:
    """Bucket IoU values into the descriptive bands.

    ``iou_exact`` is exact equality of the two pixel sets, not a tolerance.

    Args:
        values: Mask IoU values.

    Returns:
        Count and percentage per band, in report order.
    """
    array = np.asarray(values, dtype=np.float64)
    total = int(array.size)
    result: dict[str, dict[str, float]] = {}
    for name, low, high in IOU_BANDS:
        if name == "iou_exact":
            selected = array >= 1.0
        elif high == float("inf"):
            selected = array >= low
        elif low == float("-inf"):
            selected = array < high
        else:
            selected = (array >= low) & (array < high)
        count = int(selected.sum())
        result[name] = {
            "count": count,
            "percent": 0.0 if total == 0 else round(100.0 * count / total, 4),
        }
    return result


def area_error_band_counts(values: Sequence[float]) -> dict[str, dict[str, float]]:
    """Count instances whose relative area error exceeds each threshold.

    Args:
        values: Absolute relative area-error values.

    Returns:
        Count and percentage above each threshold.
    """
    array = np.asarray(values, dtype=np.float64)
    total = int(array.size)
    result: dict[str, dict[str, float]] = {}
    for threshold in AREA_ERROR_BANDS:
        count = int((array > threshold).sum())
        result[f"above_{threshold:g}"] = {
            "count": count,
            "percent": 0.0 if total == 0 else round(100.0 * count / total, 4),
        }
    return result


def quartile_edges(values: Sequence[float]) -> tuple[float, float, float]:
    """Compute the quartile boundaries of a distribution.

    The bins are derived from the development data itself and recorded with
    their boundaries, so a later reader can tell what "smallest quartile" meant
    here. They are diagnostic bins for this audit and deliberately do not reuse
    the project's small-object EDA threshold, which is a different measurement.

    Args:
        values: The sample.

    Returns:
        The 25th, 50th and 75th percentiles.
    """
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return (0.0, 0.0, 0.0)
    return tuple(float(np.percentile(array, q, method="linear")) for q in (25, 50, 75))  # type: ignore[return-value]


def summarise(rows: Iterable[FidelityRow]) -> dict[str, Any]:
    """Summarise a group of instances.

    Args:
        rows: The instances to summarise.

    Returns:
        A JSON-serialisable mapping of the group's fidelity.
    """
    collected = list(rows)
    ious = [row.mask_iou for row in collected]
    controls = [row.control_iou for row in collected]
    merged = [row.merged_iou for row in collected]
    errors = [row.relative_area_error for row in collected]
    return {
        "instances": len(collected),
        "mask_iou": describe(ious),
        "control_iou": describe(controls),
        "merged_iou": describe(merged),
        "join_loss": describe([row.control_iou - row.merged_iou for row in collected]),
        "serialization_loss": describe([row.merged_iou - row.mask_iou for row in collected]),
        "relative_area_error": describe(errors),
        "iou_bands": band_counts(ious),
        "area_error_bands": area_error_band_counts(errors),
        "canonical_area_px": describe([float(row.canonical_area_px) for row in collected]),
        "total_fp_pixels": int(sum(row.fp_pixels for row in collected)),
        "total_fn_pixels": int(sum(row.fn_pixels for row in collected)),
        "total_canonical_area_px": int(sum(row.canonical_area_px for row in collected)),
    }


def group_by(rows: Sequence[FidelityRow], key: str) -> dict[str, list[FidelityRow]]:
    """Partition instances by one attribute.

    Args:
        rows: The instances.
        key: Attribute name to group on.

    Returns:
        Instances keyed by the attribute's string value, in sorted key order.
    """
    groups: dict[str, list[FidelityRow]] = {}
    for row in rows:
        groups.setdefault(str(getattr(row, key)), []).append(row)
    return {name: groups[name] for name in sorted(groups)}


def fingerprint_rows(rows: Sequence[FidelityRow]) -> str:
    """Fingerprint the row-level results deterministically.

    Args:
        rows: The instances, in any order.

    Returns:
        A SHA-256 hex digest independent of iteration order.
    """
    import hashlib

    payload = sorted(
        (row.split, row.source_image_id, row.annotation_id, row.as_row()) for row in rows
    )
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def worst(
    rows: Sequence[FidelityRow], key: str, count: int, *, ascending: bool
) -> list[FidelityRow]:
    """Select the worst instances by one metric, deterministically.

    Ties are broken on split, image and annotation id so that the selection does
    not depend on iteration order.

    Args:
        rows: The instances.
        key: Attribute to rank on.
        count: How many to return.
        ascending: ``True`` to take the smallest values.

    Returns:
        The selected instances.
    """

    def sort_key(row: FidelityRow) -> tuple[Any, ...]:
        value = getattr(row, key)
        return (value if ascending else -value, row.split, row.source_image_id, row.annotation_id)

    return sorted(rows, key=sort_key)[:count]


def attribute_reasons(
    row_data: Mapping[str, Any],
    *,
    material_iou: float,
    small_area_px: int,
    stage_threshold: float = 1e-6,
) -> tuple[str, ...]:
    """Assign deterministic reason flags to a materially non-exact instance.

    Flags are assigned only below ``material_iou``. Above it the disagreement is
    at the scale of individual boundary pixels, and labelling that a "reason"
    would drown the genuinely approximated instances in noise.

    Attribution is driven by the three-level decomposition rather than by
    guesswork, so each stage is charged only for the IoU it actually consumed.
    ``SERIALIZATION_ONLY`` means what it says: it is assigned when serialisation
    is the *sole* material contributor. When serialisation contributes alongside
    another mechanism, the other mechanism is named and the magnitude is carried
    by the numeric ``serialization_loss`` column instead of by an inaccurate
    flag. An instance no stage and no recorded topology explains is
    ``UNATTRIBUTED`` rather than assigned the most plausible-sounding cause.

    Args:
        row_data: Mapping carrying ``mask_iou``, ``merged_iou``, ``control_iou``,
            ``hole_count``, ``connected_components``, ``component_rings``,
            ``joined`` and ``canonical_area_px``.
        material_iou: IoU at or above which the deviation is not material.
        small_area_px: Mask area below which boundary noise dominates.
        stage_threshold: IoU a stage must consume before it is named.

    Returns:
        The flags, in :data:`REASON_FLAGS` order. Empty when the instance is not
        materially non-exact.
    """
    from construction_safety_vision.data.segmentation_adapter import (
        COMPONENT_JOIN_APPROXIMATION,
        HOLE_FILL_APPROXIMATION,
        MULTI_COMPONENT_APPROXIMATION,
        OTHER_GEOMETRIC_APPROXIMATION,
        RASTER_BOUNDARY_DIFFERENCE,
        REASON_FLAGS,
        SERIALIZATION_ONLY,
        SMALL_MASK_SENSITIVITY,
        UNATTRIBUTED,
    )

    iou = float(row_data["mask_iou"])
    if iou >= material_iou:
        return ()

    merged = float(row_data["merged_iou"])
    control = float(row_data["control_iou"])
    raster_gap = 1.0 - control
    join_gap = control - merged
    serialisation_gap = merged - iou

    flags: set[str] = set()
    if raster_gap > stage_threshold:
        flags.add(RASTER_BOUNDARY_DIFFERENCE)
    if join_gap > stage_threshold:
        flags.add(COMPONENT_JOIN_APPROXIMATION)
    if int(row_data["hole_count"]) > 0:
        flags.add(HOLE_FILL_APPROXIMATION)
    if int(row_data["connected_components"]) > 1:
        flags.add(MULTI_COMPONENT_APPROXIMATION)

    # The flag's name is a claim, so it is only made when it is true.
    if serialisation_gap > stage_threshold and not flags:
        flags.add(SERIALIZATION_ONLY)

    if int(row_data["canonical_area_px"]) < small_area_px:
        flags.add(SMALL_MASK_SENSITIVITY)

    if not flags:
        flags.add(UNATTRIBUTED)
    elif flags == {SMALL_MASK_SENSITIVITY}:
        # Small, materially off, and no stage consumed enough to explain it.
        flags.add(OTHER_GEOMETRIC_APPROXIMATION)

    return tuple(flag for flag in REASON_FLAGS if flag in flags)
