"""Convert canonical COCO instance masks into Ultralytics YOLO segmentation rows.

Phase 8A. This module builds a **derived, audit-only** representation. The
canonical COCO instance segmentation frozen in phase 5D stays the ground truth;
if the two ever disagree, the COCO document is right and this adapter is broken.

The format it targets is narrower than the data it is given, and that gap is the
whole reason the audit exists. Read from the installed Ultralytics 8.4.138
source rather than from documentation:

* ``verify_image_label`` parses a row as ``np.array(fields[1:]).reshape(-1, 2)``.
  One row is therefore **one class and one flat sequence of xy pairs** - a single
  closed ring. There is no separator between rings and no ring-role marker.
* A row counts as a segment only when it has more than six fields, i.e. a class
  plus at least three points. Rows of exactly five fields are detection boxes,
  and mixing the two in one file raises.
* Coordinates are normalised and bounds-checked into ``[-0.01, 1.01]``.
* Because a row is one ring, **an interior hole cannot be expressed**. Whatever
  the outer boundary encloses is filled. Ultralytics' own mask converter uses
  ``cv2.RETR_EXTERNAL``, which discards interior contours by construction.
* Because a row is one ring, **a disconnected mask cannot be expressed either**.
  Ultralytics' own COCO converter bridges components with
  ``merge_multi_segment``, joining them along their nearest points with a
  zero-width connector. This module uses that same primitive, so the audit
  measures the framework's behaviour rather than a strategy invented here.

Two consequences of that design are worth stating plainly, because both are easy
to lose in an aggregate number.

**Filling a hole and bridging a gap both add area.** Neither is a rounding
error; they are structural properties of a one-ring format meeting a mask that
needs more than one ring. The audit quantifies each separately.

**One canonical annotation always becomes exactly one row.** Splitting a
disconnected mask into several rows would raise every fidelity metric and would
quietly redefine what an instance is - changing per-image object counts and
invalidating any later detection-versus-segmentation comparison. That invariant
is enforced here, not left to the caller.

A separate hazard, found in ``verify_image_label`` and checked by the audit
rather than assumed away: after parsing, Ultralytics computes each row's
bounding box and drops duplicate ``(class, box)`` rows with ``np.unique``. Two
distinct instances of the same class sharing a bounding box would therefore
collapse into one, changing instance cardinality without any error being raised.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from construction_safety_vision.config import ConfigError, check_keys

__all__ = [
    "ADAPTER_TYPE",
    "CANONICAL_POLYGON",
    "CANONICAL_RLE",
    "REASON_FLAGS",
    "SYNTHETIC_RECTANGLE",
    "AuditAdapterConfig",
    "InstanceGeometry",
    "SegmentationAdapterError",
    "Topology",
    "analyse_topology",
    "canonical_mask",
    "classify_geometry",
    "denormalise_ring",
    "digest",
    "format_label_line",
    "label_text",
    "load_audit_config",
    "merge_rings",
    "normalise_ring",
    "parse_label_line",
    "rings_for_annotation",
]

#: What this adapter produces.
ADAPTER_TYPE = "YOLO_SEGMENTATION_AUDIT"

#: Canonical geometry classifications.
CANONICAL_POLYGON = "CANONICAL_POLYGON"
CANONICAL_RLE = "CANONICAL_RLE"
SYNTHETIC_RECTANGLE = "SYNTHETIC_RECTANGLE"

GEOMETRY_TYPES: tuple[str, ...] = (CANONICAL_POLYGON, CANONICAL_RLE, SYNTHETIC_RECTANGLE)

#: The canonical marker phase 5B wrote for the two materialised rectangles.
SYNTHETIC_ORIGIN = "SYNTHETIC_FROM_PROVIDER_BBOX"

#: Reason flags an instance's deviation may carry.
SERIALIZATION_ONLY = "SERIALIZATION_ONLY"
RASTER_BOUNDARY_DIFFERENCE = "RASTER_BOUNDARY_DIFFERENCE"
MULTI_COMPONENT_APPROXIMATION = "MULTI_COMPONENT_APPROXIMATION"
COMPONENT_JOIN_APPROXIMATION = "COMPONENT_JOIN_APPROXIMATION"
HOLE_FILL_APPROXIMATION = "HOLE_FILL_APPROXIMATION"
SMALL_MASK_SENSITIVITY = "SMALL_MASK_SENSITIVITY"
OTHER_GEOMETRIC_APPROXIMATION = "OTHER_GEOMETRIC_APPROXIMATION"
UNATTRIBUTED = "UNATTRIBUTED"

REASON_FLAGS: tuple[str, ...] = (
    SERIALIZATION_ONLY,
    RASTER_BOUNDARY_DIFFERENCE,
    MULTI_COMPONENT_APPROXIMATION,
    COMPONENT_JOIN_APPROXIMATION,
    HOLE_FILL_APPROXIMATION,
    SMALL_MASK_SENSITIVITY,
    OTHER_GEOMETRIC_APPROXIMATION,
    UNATTRIBUTED,
)

#: The protected split. It has no part in this audit.
FORBIDDEN_SPLIT = "test"

#: Ultralytics parses a row as a segment only above this field count.
SEGMENT_FIELD_THRESHOLD = 6


class SegmentationAdapterError(RuntimeError):
    """Raised when a canonical annotation cannot be represented or verified."""


def digest(payload: Any) -> str:
    """Hash a JSON-serialisable payload deterministically.

    Args:
        payload: Content to hash.

    Returns:
        Its SHA-256 hex digest.
    """
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# --- configuration ------------------------------------------------------------


_CONFIG_KEYS: tuple[str, ...] = (
    "schema_version",
    "adapter_type",
    "status",
    "canonical_status",
    "training_status",
    "canonical_task_manifest",
    "canonical_annotations_root",
    "canonical_images_root",
    "canonical_source",
    "canonical_source_phase",
    "split_directories",
    "allowed_splits",
    "test_policy",
    "class_map_reference",
    "class_map_sha256",
    "placeholder_category_excluded",
    "instance_cardinality_policy",
    "instance_split_allowed",
    "instance_merge_allowed",
    "rle_decode",
    "contour_retrieval_mode",
    "contour_approximation_mode",
    "contour_epsilon_simplification",
    "contour_source",
    "multi_component_policy",
    "multi_component_classification",
    "hole_policy",
    "hole_classification",
    "serialization_precision",
    "minimum_points_per_instance",
    "rasterization",
    "rasterization_canvas",
    "rasterization_downsample_ratio",
    "connectivity",
    "image_transfer_mode",
    "image_transformation",
    "zero_instance_policy",
    "output_root",
    "fidelity_metrics",
    "raster_convention_control",
    "fidelity_bands",
    "fidelity_percentiles",
    "area_error_bands",
    "fidelity_strata",
    "mask_area_binning",
    "thin_structure_proxy",
    "reason_flags",
    "material_deviation_iou",
    "small_mask_area_px",
    "architecture_decision",
)


@dataclass(frozen=True)
class AuditAdapterConfig:
    """Every methodological choice the audit makes, declared in a file.

    Attributes:
        raw: The validated configuration mapping, exactly as parsed.
    """

    raw: dict[str, Any]

    def __getitem__(self, key: str) -> Any:
        """Read a configuration value.

        Args:
            key: Configuration key.

        Returns:
            The value.
        """
        return self.raw[key]

    @property
    def precision(self) -> int:
        """Decimal places written per normalised coordinate.

        Returns:
            The precision.
        """
        return int(self.raw["serialization_precision"])

    @property
    def connectivity(self) -> int:
        """Connectivity used for connected-component analysis.

        Returns:
            Either 4 or 8.
        """
        return int(self.raw["connectivity"])

    @property
    def output_root(self) -> str:
        """Repository-relative destination root.

        Returns:
            The path.
        """
        return str(self.raw["output_root"])

    def as_dict(self) -> dict[str, Any]:
        """Serialise the protocol for hashing and reporting.

        Returns:
            A JSON-serialisable mapping.
        """
        return json.loads(json.dumps(self.raw, sort_keys=True))

    def fingerprint(self) -> str:
        """Hash the protocol, so results can name the rules that produced them.

        Returns:
            A SHA-256 hex digest over the configuration content only.
        """
        return digest(self.as_dict())


def load_audit_config(path: str | Path) -> AuditAdapterConfig:
    """Load and validate the segmentation-audit protocol.

    Parsing is strict: an unknown key raises rather than being ignored, so a
    typo cannot silently disable a policy the audit is documented as following.

    Args:
        path: Configuration file.

    Returns:
        The parsed protocol.

    Raises:
        ConfigError: If the file is missing, is not valid YAML, is malformed, or
            names the protected split anywhere.
    """
    config_path = Path(path).expanduser()
    if not config_path.is_file():
        msg = f"Segmentation-audit configuration not found: {config_path.name}"
        raise ConfigError(msg)
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        msg = f"Segmentation-audit configuration is not valid YAML: {config_path.name} ({exc})"
        raise ConfigError(msg) from exc
    if not isinstance(raw, Mapping):
        msg = f"Segmentation-audit configuration must be a mapping: {config_path.name}"
        raise ConfigError(msg)

    check_keys(raw, required=_CONFIG_KEYS, optional=(), context=config_path.name)

    if raw["adapter_type"] != ADAPTER_TYPE:
        msg = f"adapter_type must be {ADAPTER_TYPE!r}, got {raw['adapter_type']!r}"
        raise ConfigError(msg)

    splits = raw["split_directories"]
    if not isinstance(splits, Mapping) or FORBIDDEN_SPLIT in splits:
        msg = (
            "split_directories must be a mapping and must not carry a "
            f"{FORBIDDEN_SPLIT!r} entry: the holdout gets no adapter"
        )
        raise ConfigError(msg)
    allowed = raw["allowed_splits"]
    if not isinstance(allowed, Sequence) or FORBIDDEN_SPLIT in allowed:
        msg = f"allowed_splits must not contain {FORBIDDEN_SPLIT!r}"
        raise ConfigError(msg)
    if set(allowed) != set(splits):
        msg = f"allowed_splits {sorted(allowed)} disagrees with split_directories {sorted(splits)}"
        raise ConfigError(msg)

    if raw["instance_split_allowed"] is not False or raw["instance_merge_allowed"] is not False:
        msg = (
            "instance_split_allowed and instance_merge_allowed must both be false: the audit's "
            "one-annotation-one-row invariant is not optional"
        )
        raise ConfigError(msg)

    precision = raw["serialization_precision"]
    if isinstance(precision, bool) or not isinstance(precision, int) or not 1 <= precision <= 12:
        msg = f"serialization_precision {precision!r} must be an integer in [1, 12]"
        raise ConfigError(msg)

    if raw["connectivity"] not in (4, 8):
        msg = f"connectivity {raw['connectivity']!r} must be 4 or 8"
        raise ConfigError(msg)

    if tuple(raw["reason_flags"]) != REASON_FLAGS:
        msg = f"reason_flags must be exactly {list(REASON_FLAGS)}"
        raise ConfigError(msg)

    return AuditAdapterConfig(raw=json.loads(json.dumps(raw)))


# --- canonical geometry -------------------------------------------------------


def classify_geometry(annotation: Mapping[str, Any]) -> str:
    """Classify a canonical annotation's geometry.

    The two synthetic rectangles phase 5B materialised are stored as polygons but
    are not human-drawn segmentation, and reporting them inside the polygon
    stratum would let two rectangles vouch for 843 real outlines. They get their
    own class.

    Args:
        annotation: A canonical COCO annotation.

    Returns:
        One of :data:`GEOMETRY_TYPES`.

    Raises:
        SegmentationAdapterError: If the segmentation field has no known shape.
    """
    if annotation.get("geometry_origin") == SYNTHETIC_ORIGIN:
        return SYNTHETIC_RECTANGLE
    segmentation = annotation.get("segmentation")
    if isinstance(segmentation, Mapping):
        return CANONICAL_RLE
    if isinstance(segmentation, list):
        return CANONICAL_POLYGON
    identifier = annotation.get("id", "<unknown>")
    msg = f"annotation {identifier}: segmentation is neither a polygon list nor an RLE mapping"
    raise SegmentationAdapterError(msg)


def canonical_mask(annotation: Mapping[str, Any], *, height: int, width: int) -> np.ndarray:
    """Decode a canonical annotation into its binary mask.

    Decoding goes through pycocotools, the reference implementation the canonical
    masks were built with in phases 5A-5D. Nothing is reimplemented here: a
    private decoder that disagreed with the canonical one would make every
    fidelity number a measurement of the disagreement.

    Args:
        annotation: A canonical COCO annotation.
        height: Source image height.
        width: Source image width.

    Returns:
        A ``(height, width)`` array of 0 and 1, dtype ``uint8``.

    Raises:
        SegmentationAdapterError: If the geometry cannot be decoded.
    """
    from pycocotools import mask as mask_utils

    segmentation = annotation.get("segmentation")
    try:
        if isinstance(segmentation, Mapping):
            rle = dict(segmentation)
            counts = rle.get("counts")
            if isinstance(counts, str):
                rle["counts"] = counts.encode("utf-8")
            decoded = mask_utils.decode(rle)
        elif isinstance(segmentation, list):
            rles = mask_utils.frPyObjects(segmentation, height, width)
            decoded = mask_utils.decode(mask_utils.merge(rles))
        else:
            raise TypeError(type(segmentation))
    except Exception as exc:
        identifier = annotation.get("id", "<unknown>")
        msg = f"annotation {identifier}: canonical geometry could not be decoded ({exc})"
        raise SegmentationAdapterError(msg) from exc

    if decoded.ndim == 3:
        decoded = decoded[:, :, 0]
    if decoded.shape != (height, width):
        identifier = annotation.get("id", "<unknown>")
        msg = (
            f"annotation {identifier}: decoded mask is {decoded.shape}, expected "
            f"{(height, width)}; the geometry does not belong to this canvas"
        )
        raise SegmentationAdapterError(msg)
    return decoded.astype(np.uint8)


@dataclass(frozen=True)
class Topology:
    """What a canonical mask's shape costs a one-ring format.

    Attributes:
        component_count: Connected components at the configured connectivity.
        component_areas: Pixel area of each component, largest first.
        hole_count: Interior rings enclosed by any component.
        hole_pixels: Total pixel area of those interior rings.
        area_px: Mask area in pixels.
        perimeter_px: Total external contour perimeter in pixels.
        thinness: Perimeter over the square root of area, or 0.0 for an empty
            mask. Scale-free, and rises for elongated or ragged shapes.
    """

    component_count: int
    component_areas: tuple[int, ...]
    hole_count: int
    hole_pixels: int
    area_px: int
    perimeter_px: float
    thinness: float

    @property
    def hole_area_fraction(self) -> float:
        """Hole area as a fraction of the filled mask.

        Returns:
            The fraction, or 0.0 when the mask is empty.
        """
        filled = self.area_px + self.hole_pixels
        return 0.0 if filled == 0 else self.hole_pixels / filled

    def as_dict(self) -> dict[str, Any]:
        """Serialise the topology.

        Returns:
            A JSON-serialisable mapping.
        """
        return {
            "component_count": self.component_count,
            "component_areas": list(self.component_areas),
            "hole_count": self.hole_count,
            "hole_pixels": self.hole_pixels,
            "hole_area_fraction": self.hole_area_fraction,
            "area_px": self.area_px,
            "perimeter_px": self.perimeter_px,
            "thinness": self.thinness,
        }


def analyse_topology(mask: np.ndarray, *, connectivity: int = 8) -> Topology:
    """Measure the topology a one-ring format would have to flatten.

    Holes are counted from the contour hierarchy rather than by comparing filled
    areas, so that a mask with several separate holes reports several rather
    than one aggregate.

    Args:
        mask: A binary mask of 0 and 1.
        connectivity: 4 or 8. Eight is used by default because four would count
            diagonally-touching pixels as separate components.

    Returns:
        The topology record.
    """
    import cv2

    # C-contiguous on purpose: pycocotools returns Fortran-ordered arrays, and
    # OpenCV refuses an output buffer whose layout it cannot map onto cv::Mat.
    binary = np.ascontiguousarray((mask > 0).astype(np.uint8))
    area = int(binary.sum())
    if area == 0:
        return Topology(
            component_count=0,
            component_areas=(),
            hole_count=0,
            hole_pixels=0,
            area_px=0,
            perimeter_px=0.0,
            thinness=0.0,
        )

    count, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=connectivity)
    # Label 0 is the background.
    areas = sorted((int(stats[index, cv2.CC_STAT_AREA]) for index in range(1, count)), reverse=True)

    contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    hole_count = 0
    perimeter = 0.0
    outer: list[np.ndarray] = []
    if hierarchy is not None:
        for index, contour in enumerate(contours):
            if int(hierarchy[0][index][3]) == -1:
                perimeter += float(cv2.arcLength(contour, True))
                outer.append(contour)
            else:
                hole_count += 1

    # Hole area is counted in pixels, not with `cv2.contourArea`. That function
    # measures the area enclosed by the contour *path*, which runs along the
    # outer edge of the boundary pixels and overstates a 20x20 hole as 441. The
    # filled-minus-actual difference is exact, and this figure is reported as a
    # pixel count.
    hole_pixels = 0
    if hole_count:
        filled = np.zeros(binary.shape, dtype=np.uint8)
        cv2.drawContours(filled, outer, -1, color=1, thickness=cv2.FILLED)
        hole_pixels = int(np.logical_and(filled > 0, binary == 0).sum())

    return Topology(
        component_count=len(areas),
        component_areas=tuple(areas),
        hole_count=hole_count,
        hole_pixels=hole_pixels,
        area_px=area,
        perimeter_px=perimeter,
        thinness=perimeter / float(np.sqrt(area)),
    )


# --- ring extraction ----------------------------------------------------------


@dataclass
class InstanceGeometry:
    """One canonical annotation on its way to one YOLO row.

    Attributes:
        rings: The component rings in pixel coordinates, before merging.
        merged: The single ring the row will carry, in pixel coordinates.
        joined: Whether components had to be bridged into one path.
        ring_source: How the rings were obtained.
        dropped_rings: Component rings discarded for having fewer than three
            points, which no YOLO row can express.
    """

    rings: list[np.ndarray]
    merged: np.ndarray
    joined: bool
    ring_source: str
    dropped_rings: int = 0
    notes: list[str] = field(default_factory=list)


def rings_for_annotation(
    annotation: Mapping[str, Any],
    mask: np.ndarray,
    *,
    geometry_type: str,
    minimum_points: int = 3,
) -> list[np.ndarray]:
    """Extract an annotation's component rings in pixel coordinates.

    A canonical polygon keeps its own coordinates: re-tracing it from a raster
    would throw away precision the canonical file already holds. A canonical RLE
    has no polygon to keep, so its boundary is traced with the settings
    Ultralytics uses in its own mask converter - ``RETR_EXTERNAL`` and
    ``CHAIN_APPROX_SIMPLE``, with no ``approxPolyDP`` epsilon, because
    simplifying to shrink the file would discard real boundary detail.

    Args:
        annotation: A canonical COCO annotation.
        mask: The annotation's decoded binary mask.
        geometry_type: One of :data:`GEOMETRY_TYPES`.
        minimum_points: Fewest points a ring must have to be expressible.

    Returns:
        The rings, each an ``(N, 2)`` float array of xy pixel coordinates.

    Raises:
        SegmentationAdapterError: If no expressible ring can be produced.
    """
    import cv2

    rings: list[np.ndarray] = []
    if geometry_type in (CANONICAL_POLYGON, SYNTHETIC_RECTANGLE):
        segmentation = annotation.get("segmentation")
        for component in segmentation or []:
            if not isinstance(component, list) or len(component) < minimum_points * 2:
                continue
            if len(component) % 2:
                identifier = annotation.get("id", "<unknown>")
                msg = f"annotation {identifier}: polygon component has an odd coordinate count"
                raise SegmentationAdapterError(msg)
            rings.append(np.asarray(component, dtype=np.float64).reshape(-1, 2))
    else:
        contours, _ = cv2.findContours(
            (mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        for contour in contours:
            points = contour.reshape(-1, 2)
            if len(points) < minimum_points:
                continue
            rings.append(points.astype(np.float64))

    if not rings:
        identifier = annotation.get("id", "<unknown>")
        msg = (
            f"annotation {identifier}: no ring of at least {minimum_points} points could be "
            "produced, so the instance is not expressible as a YOLO segmentation row"
        )
        raise SegmentationAdapterError(msg)
    return rings


def merge_rings(rings: Sequence[np.ndarray]) -> tuple[np.ndarray, bool]:
    """Reduce component rings to the single ring a YOLO row can carry.

    Delegates to Ultralytics' ``merge_multi_segment``, the primitive its own COCO
    converter uses: components are bridged along their closest points with a
    zero-width connector. Using the framework's function keeps the audit about
    the format rather than about a joining strategy invented here.

    Args:
        rings: One or more ``(N, 2)`` pixel-coordinate rings.

    Returns:
        The merged ring and whether any bridging happened.

    Raises:
        SegmentationAdapterError: If no ring was supplied.
    """
    if not rings:
        msg = "cannot merge an empty ring list"
        raise SegmentationAdapterError(msg)
    if len(rings) == 1:
        return np.asarray(rings[0], dtype=np.float64), False

    from ultralytics.data.converter import merge_multi_segment

    flattened = [np.asarray(ring, dtype=np.float64).reshape(-1).tolist() for ring in rings]
    merged = merge_multi_segment(flattened)
    return np.concatenate(merged, axis=0).astype(np.float64), True


# --- serialisation ------------------------------------------------------------


def normalise_ring(ring: np.ndarray, *, image_width: int, image_height: int) -> np.ndarray:
    """Normalise pixel coordinates to the unit square.

    Args:
        ring: An ``(N, 2)`` array of xy pixel coordinates.
        image_width: Source image width.
        image_height: Source image height.

    Returns:
        An ``(N, 2)`` array of normalised coordinates.

    Raises:
        SegmentationAdapterError: If the canvas is degenerate.
    """
    if image_width <= 0 or image_height <= 0:
        msg = f"degenerate canvas {image_width}x{image_height}"
        raise SegmentationAdapterError(msg)
    return np.asarray(ring, dtype=np.float64) / np.array(
        [image_width, image_height], dtype=np.float64
    )


def denormalise_ring(ring: np.ndarray, *, image_width: int, image_height: int) -> np.ndarray:
    """Return normalised coordinates to the pixel grid.

    Args:
        ring: An ``(N, 2)`` array of normalised coordinates.
        image_width: Source image width.
        image_height: Source image height.

    Returns:
        An ``(N, 2)`` array of xy pixel coordinates.
    """
    return np.asarray(ring, dtype=np.float64) * np.array(
        [image_width, image_height], dtype=np.float64
    )


def format_label_line(class_index: int, ring: np.ndarray, *, precision: int) -> str:
    """Render one instance as one YOLO segmentation row.

    Fixed-point rather than a shortest-representation float, so that the file is
    byte-identical between runs and between machines.

    Args:
        class_index: The frozen class index.
        ring: An ``(N, 2)`` array of normalised coordinates.
        precision: Decimal places per coordinate.

    Returns:
        The row, without a trailing newline.

    Raises:
        SegmentationAdapterError: If the row would not parse as a segment.
    """
    points = np.asarray(ring, dtype=np.float64).reshape(-1, 2)
    if len(points) < 3:
        msg = f"a segmentation row needs at least 3 points, got {len(points)}"
        raise SegmentationAdapterError(msg)
    coordinates = " ".join(f"{value:.{precision}f}" for value in points.reshape(-1))
    line = f"{int(class_index)} {coordinates}"
    if len(line.split()) <= SEGMENT_FIELD_THRESHOLD:
        msg = (
            f"row has {len(line.split())} fields; Ultralytics parses a row as a segment only "
            f"above {SEGMENT_FIELD_THRESHOLD}"
        )
        raise SegmentationAdapterError(msg)
    return line


def parse_label_line(line: str) -> tuple[int, np.ndarray]:
    """Parse one written row exactly as Ultralytics parses it.

    Mirrors ``verify_image_label``: fields are split on whitespace, the tail is
    read as ``float32`` and reshaped to ``(-1, 2)``. The float32 cast is
    deliberate and not an oversight - it is what the framework does, and reading
    the file back at float64 would measure a precision the model never sees.

    Args:
        line: One label row.

    Returns:
        The class index and an ``(N, 2)`` array of normalised coordinates.

    Raises:
        SegmentationAdapterError: If the row is malformed or is not a segment.
    """
    fields = line.split()
    if len(fields) <= SEGMENT_FIELD_THRESHOLD:
        msg = (
            f"row has {len(fields)} fields; Ultralytics would not parse it as a segment "
            f"(needs more than {SEGMENT_FIELD_THRESHOLD})"
        )
        raise SegmentationAdapterError(msg)
    if (len(fields) - 1) % 2:
        msg = f"row has {len(fields) - 1} coordinate fields, which is not an even number"
        raise SegmentationAdapterError(msg)
    try:
        class_index = int(float(fields[0]))
        points = np.array(fields[1:], dtype=np.float32).reshape(-1, 2)
    except ValueError as exc:
        msg = f"row is not numeric ({exc})"
        raise SegmentationAdapterError(msg) from exc
    return class_index, points


def label_text(rows: Sequence[str]) -> str:
    """Assemble a label file's contents.

    An image with no annotation gets an empty file rather than no file, which is
    how Ultralytics represents a background image; a missing file is counted as
    a *missing label* instead and the negative would be silently lost.

    Args:
        rows: Rendered rows, in canonical annotation order.

    Returns:
        The file contents.
    """
    if not rows:
        return ""
    return "\n".join(rows) + "\n"


def label_fingerprint(labels: Mapping[str, str]) -> str:
    """Fingerprint a set of label files by name and content.

    Args:
        labels: Label file contents keyed by file name.

    Returns:
        A SHA-256 hex digest, independent of iteration order.
    """
    return digest({name: labels[name] for name in sorted(labels)})


def bounds_violations(ring: np.ndarray, *, low: float = -0.01, high: float = 1.01) -> int:
    """Count normalised coordinates Ultralytics' bounds check would reject.

    Args:
        ring: An ``(N, 2)`` array of normalised coordinates.
        low: Lower bound the framework enforces.
        high: Upper bound the framework enforces.

    Returns:
        The number of offending coordinate values.
    """
    values = np.asarray(ring, dtype=np.float64).reshape(-1)
    return int(((values < low) | (values > high)).sum())
