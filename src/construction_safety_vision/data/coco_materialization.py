"""Canonical COCO detection and instance-segmentation views of one dataset.

Two task representations, one ground truth. The detection view and the
segmentation view describe **the same images and the same objects**; the only
intended difference between them is how each object's extent is expressed. That
is what makes a later comparison between a detector and a segmenter a statement
about the models rather than about their labels, so it is enforced by a validator
rather than left as an intention.

Why COCO for both, and why not YOLO yet. The canonical annotation state holds
polygon geometry *and* compressed RLE. COCO carries both natively; YOLO's
segmentation format carries polygons only, so writing it would mean rasterising
and re-polygonising every RLE mask - an approximation, applied to the ground
truth, before any model has been chosen. That conversion belongs to a
model-specific adapter that measures what it loses, not to the canonical dataset.

Detection boxes are **derived from the segmentation**, never copied from the
provider. Phase 4A measured the provider's stored boxes disagreeing with their
own geometry by up to 123.5 px inside the v4 export, so the provider's box is not
ground truth here; the polygon or mask is.

Nothing here decodes an image, trains, evaluates, or writes a model-specific
format.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pycocotools import mask as coco_mask

from construction_safety_vision.data.geometry import GeometryError, bbox_from_segmentation
from construction_safety_vision.data.materialization import CanonicalIds, Tolerances, digest
from construction_safety_vision.data.population import SYNTHETIC_GEOMETRY

DETECTION = "detection"
"""Task name of the object-detection view."""

SEGMENTATION = "segmentation"
"""Task name of the instance-segmentation view."""

TASKS: tuple[str, ...] = (DETECTION, SEGMENTATION)
"""The two canonical task views, in a fixed order for deterministic output."""

POLYGON = "POLYGON"
"""Canonical geometry expressed as coordinate rings."""

RLE = "RLE"
"""Canonical geometry expressed as compressed COCO run lengths."""

SYNTHETIC_RECTANGLE = "SYNTHETIC_RECTANGLE"
"""Canonical geometry materialised from the provider's box in phase 5B."""

COCO_LICENSE = {
    "id": 1,
    "name": "CC BY 4.0",
    "url": "https://creativecommons.org/licenses/by/4.0/",
}
"""The source dataset's licence, carried into every emitted file."""


class CocoMaterializationError(RuntimeError):
    """Raised when a task representation cannot be built as specified."""


@dataclass(frozen=True)
class CanonicalImage:
    """One modelling image, as the materialiser needs it.

    Attributes:
        source_image_id: Canonical source image id.
        file_name: Destination file name inside the split directory.
        width: Original width in pixels.
        height: Original height in pixels.
        group_id: The indivisible split unit it belongs to.
    """

    source_image_id: str
    file_name: str
    width: int
    height: int
    group_id: str


@dataclass(frozen=True)
class CanonicalAnnotation:
    """One canonical annotation, with the geometry that is the ground truth.

    Attributes:
        source_image_id: Canonical source image id.
        annotation_id: Provider annotation id, unique within its image.
        label: Canonical class name.
        class_index: Frozen class index.
        segmentation: The canonical COCO segmentation, polygon or RLE.
        area: Canonical mask area in pixels.
        canonical_bbox: The box phase 5A measured from this geometry, kept only
            to cross-check the box derived here.
        geometry_origin: Provider segmentation or synthetic rectangle.
    """

    source_image_id: str
    annotation_id: str
    label: str
    class_index: int
    segmentation: Any
    area: float
    canonical_bbox: list[float]
    geometry_origin: str

    @property
    def geometry_kind(self) -> str:
        """Classify the geometry for reporting and round-trip validation.

        Returns:
            :data:`SYNTHETIC_RECTANGLE`, :data:`RLE` or :data:`POLYGON`.
        """
        if self.geometry_origin == SYNTHETIC_GEOMETRY:
            return SYNTHETIC_RECTANGLE
        return RLE if isinstance(self.segmentation, dict) else POLYGON


def categories(class_map: Mapping[str, int]) -> list[dict[str, Any]]:
    """Build the COCO category table from the frozen class map.

    The category id **is** the frozen class index. Introducing a second numbering
    would create two orderings that must be kept in step, and the frozen
    ``class_map_sha256`` would stop describing what the datasets actually
    contain.

    Args:
        class_map: The frozen class-name-to-index map.

    Returns:
        The category table, ordered by index.
    """
    return [
        {"id": index, "name": name, "supercategory": "ppe"}
        for name, index in sorted(class_map.items(), key=lambda item: item[1])
    ]


def derive_bbox(annotation: CanonicalAnnotation) -> list[float]:
    """Derive an annotation's detection box from its canonical segmentation.

    Args:
        annotation: The canonical annotation.

    Returns:
        The box in COCO ``[x, y, width, height]``.

    Raises:
        CocoMaterializationError: If the geometry yields no usable box.
    """
    try:
        box = bbox_from_segmentation(annotation.segmentation)
    except GeometryError as exc:
        msg = (
            f"annotation {annotation.annotation_id!r} on "
            f"{annotation.source_image_id!r} has geometry no box can be derived from: {exc}"
        )
        raise CocoMaterializationError(msg) from exc
    return [round(value, 4) for value in box.as_list()]


def _image_records(
    images: Sequence[CanonicalImage], ids: CanonicalIds, split: str
) -> list[dict[str, Any]]:
    """Build the COCO image table, identical for both task views.

    Args:
        images: The split's images.
        ids: The canonical id assignment.
        split: Split name, recorded on each record for traceability.

    Returns:
        The image table, ordered by canonical image id.
    """
    return [
        {
            "id": ids.image_ids[image.source_image_id],
            "file_name": image.file_name,
            "width": image.width,
            "height": image.height,
            "license": COCO_LICENSE["id"],
            "source_image_id": image.source_image_id,
            "group_id": image.group_id,
            "split": split,
        }
        for image in sorted(images, key=lambda i: ids.image_ids[i.source_image_id])
    ]


def _info(split: str, task: str, config_fingerprint: str, split_sha256: str) -> dict[str, Any]:
    """Build the COCO ``info`` block.

    No timestamp: the file must be byte-identical between runs, so the only
    identity it carries is content-derived.

    Args:
        split: Split name.
        task: Task view name.
        config_fingerprint: Digest of the materialisation protocol.
        split_sha256: Digest of the frozen split membership.

    Returns:
        The info mapping.
    """
    return {
        "description": (
            f"Construction Safety Vision - canonical {task} view of the frozen {split} split"
        ),
        "version": "1",
        "phase": "5D",
        "task": task,
        "split": split,
        "source_dataset": "agis-workspace-8gs52/construction-ppe-compliance-detection v4",
        "license": COCO_LICENSE["name"],
        "annotation_source": "CURRENT_COMPLETE_GEOMETRY",
        "bbox_source": "DERIVED_FROM_CANONICAL_SEGMENTATION",
        "split_assignment_sha256": split_sha256,
        "materialization_config_sha256": config_fingerprint,
    }


def build_segmentation_coco(
    images: Sequence[CanonicalImage],
    annotations: Sequence[CanonicalAnnotation],
    *,
    ids: CanonicalIds,
    class_map: Mapping[str, int],
    split: str,
    config_fingerprint: str,
    split_sha256: str,
) -> dict[str, Any]:
    """Build the canonical instance-segmentation view of one split.

    Canonical geometry is carried through unchanged: a polygon stays a polygon
    and an RLE stays an RLE. Converting between them would be the materialiser
    editing the ground truth.

    Args:
        images: The split's images.
        annotations: The split's canonical annotations.
        ids: The canonical id assignment.
        class_map: The frozen class map.
        split: Split name.
        config_fingerprint: Digest of the materialisation protocol.
        split_sha256: Digest of the frozen split membership.

    Returns:
        The COCO document.
    """
    records = []
    for annotation in sorted(
        annotations, key=lambda a: ids.annotation_ids[(a.source_image_id, a.annotation_id)]
    ):
        records.append(
            {
                "id": ids.annotation_ids[(annotation.source_image_id, annotation.annotation_id)],
                "image_id": ids.image_ids[annotation.source_image_id],
                "category_id": annotation.class_index,
                "segmentation": annotation.segmentation,
                "area": round(float(annotation.area), 4),
                "bbox": derive_bbox(annotation),
                "iscrowd": 0,
                "source_image_id": annotation.source_image_id,
                "source_annotation_id": annotation.annotation_id,
                "geometry_origin": annotation.geometry_origin,
            }
        )
    return {
        "info": _info(split, SEGMENTATION, config_fingerprint, split_sha256),
        "licenses": [COCO_LICENSE],
        "categories": categories(class_map),
        "images": _image_records(images, ids, split),
        "annotations": records,
    }


def build_detection_coco(
    images: Sequence[CanonicalImage],
    annotations: Sequence[CanonicalAnnotation],
    *,
    ids: CanonicalIds,
    class_map: Mapping[str, int],
    split: str,
    config_fingerprint: str,
    split_sha256: str,
) -> dict[str, Any]:
    """Build the canonical object-detection view of one split.

    The detection view carries no mask. It keeps the segmentation-derived
    ``area`` rather than recomputing it from the box, because that is COCO's own
    convention - the reference dataset's detection annotations carry mask area -
    and because the size buckets a detection metric reports must be the same ones
    the segmentation metric reports, or the two tasks are not comparable.

    Args:
        images: The split's images.
        annotations: The split's canonical annotations.
        ids: The canonical id assignment.
        class_map: The frozen class map.
        split: Split name.
        config_fingerprint: Digest of the materialisation protocol.
        split_sha256: Digest of the frozen split membership.

    Returns:
        The COCO document.
    """
    records = []
    for annotation in sorted(
        annotations, key=lambda a: ids.annotation_ids[(a.source_image_id, a.annotation_id)]
    ):
        bbox = derive_bbox(annotation)
        records.append(
            {
                "id": ids.annotation_ids[(annotation.source_image_id, annotation.annotation_id)],
                "image_id": ids.image_ids[annotation.source_image_id],
                "category_id": annotation.class_index,
                "bbox": bbox,
                "area": round(float(annotation.area), 4),
                "bbox_area": round(bbox[2] * bbox[3], 4),
                "iscrowd": 0,
                "source_image_id": annotation.source_image_id,
                "source_annotation_id": annotation.annotation_id,
                "bbox_source": "DERIVED_FROM_CANONICAL_SEGMENTATION",
            }
        )
    return {
        "info": _info(split, DETECTION, config_fingerprint, split_sha256),
        "licenses": [COCO_LICENSE],
        "categories": categories(class_map),
        "images": _image_records(images, ids, split),
        "annotations": records,
    }


def validate_coco(
    document: Mapping[str, Any],
    *,
    task: str,
    split: str,
    image_directory: Path,
    expected_images: int,
    expected_annotations: int,
    class_map: Mapping[str, int],
    tolerances: Tolerances,
) -> list[str]:
    """Check one emitted task representation for structural defects.

    Args:
        document: The COCO document.
        task: :data:`DETECTION` or :data:`SEGMENTATION`.
        split: Split name.
        image_directory: Directory the image records must resolve into.
        expected_images: Images the frozen membership requires.
        expected_annotations: Annotations the canonical population requires.
        class_map: The frozen class map.
        tolerances: Numeric slack allowed.

    Returns:
        One description per problem found, empty when the document is sound.
    """
    problems: list[str] = []
    images = document["images"]
    annotations = document["annotations"]

    problems.extend(_validate_images(images, split, image_directory, expected_images))
    problems.extend(_validate_categories(document["categories"], class_map))
    problems.extend(_validate_annotation_keys(annotations, images, class_map, expected_annotations))
    canvases = {record["id"]: (record["width"], record["height"]) for record in images}
    for record in annotations:
        problems.extend(_validate_geometry(record, task, canvases, tolerances))
    return problems


def _validate_images(
    images: Sequence[Mapping[str, Any]],
    split: str,
    image_directory: Path,
    expected_images: int,
) -> list[str]:
    """Check the image table.

    Args:
        images: The COCO image records.
        split: Split name each record must declare.
        image_directory: Directory the file names must resolve into.
        expected_images: Images the frozen membership requires.

    Returns:
        One description per problem found.
    """
    from construction_safety_vision.paths import long_path

    problems: list[str] = []
    if len(images) != expected_images:
        problems.append(
            f"{len(images)} image record(s), frozen membership requires {expected_images}"
        )
    ids = [record["id"] for record in images]
    if len(set(ids)) != len(ids):
        problems.append("image ids are not unique")
    source_ids = [record["source_image_id"] for record in images]
    if len(set(source_ids)) != len(source_ids):
        problems.append("source image ids are not unique")
    for record in images:
        if record["split"] != split:
            problems.append(
                f"image {record['source_image_id']!r} declares split {record['split']!r}"
            )
        if record["width"] <= 0 or record["height"] <= 0:
            problems.append(f"image {record['source_image_id']!r} has a non-positive dimension")
        if not Path(long_path(image_directory / record["file_name"])).is_file():
            problems.append(f"image file {record['file_name']!r} is missing from disk")
    return problems


def _validate_categories(
    category_records: Sequence[Mapping[str, Any]], class_map: Mapping[str, int]
) -> list[str]:
    """Check the category table against the frozen class map.

    Args:
        category_records: The COCO category records.
        class_map: The frozen class map.

    Returns:
        One description per problem found.
    """
    problems: list[str] = []
    emitted = {record["name"]: record["id"] for record in category_records}
    if emitted != dict(class_map):
        problems.append(f"category table {emitted} does not match the frozen class map")
    if "object" in emitted:
        problems.append("the provider's placeholder category 'object' entered the dataset")
    return problems


def _validate_annotation_keys(
    annotations: Sequence[Mapping[str, Any]],
    images: Sequence[Mapping[str, Any]],
    class_map: Mapping[str, int],
    expected_annotations: int,
) -> list[str]:
    """Check annotation identity and foreign keys.

    Args:
        annotations: The COCO annotation records.
        images: The COCO image records.
        class_map: The frozen class map.
        expected_annotations: Annotations the canonical population requires.

    Returns:
        One description per problem found.
    """
    problems: list[str] = []
    if len(annotations) != expected_annotations:
        problems.append(
            f"{len(annotations)} annotation(s), the canonical population requires "
            f"{expected_annotations}"
        )
    ids = [record["id"] for record in annotations]
    if len(set(ids)) != len(ids):
        problems.append("annotation ids are not unique")
    known_images = {record["id"] for record in images}
    valid_categories = set(class_map.values())
    for record in annotations:
        if record["image_id"] not in known_images:
            problems.append(
                f"annotation {record['id']} references image {record['image_id']}, which is "
                "not in this split"
            )
        if record["category_id"] not in valid_categories:
            problems.append(
                f"annotation {record['id']} carries category {record['category_id']!r}, which "
                "is not in the frozen class map"
            )
    return problems


def _validate_geometry(
    record: Mapping[str, Any],
    task: str,
    canvases: Mapping[int, tuple[int, int]],
    tolerances: Tolerances,
) -> list[str]:
    """Check one annotation's box and, for segmentation, its mask.

    Args:
        record: The COCO annotation record.
        task: :data:`DETECTION` or :data:`SEGMENTATION`.
        canvases: Image ``(width, height)`` keyed by COCO image id.
        tolerances: Numeric slack allowed.

    Returns:
        One description per problem found.
    """
    problems: list[str] = []
    x, y, width, height = record["bbox"]
    if width <= 0 or height <= 0:
        problems.append(f"annotation {record['id']} has a non-positive bbox extent")
    canvas = canvases.get(record["image_id"])
    if canvas is not None:
        canvas_width, canvas_height = canvas
        slack = tolerances.bbox_canvas_px
        if (
            x < -slack
            or y < -slack
            or x + width > canvas_width + slack
            or y + height > canvas_height + slack
        ):
            problems.append(
                f"annotation {record['id']} has bbox {record['bbox']} outside its "
                f"{canvas_width}x{canvas_height} canvas"
            )
    if task == SEGMENTATION:
        problems.extend(_validate_mask(record))
    elif "segmentation" in record:
        problems.append(
            f"annotation {record['id']} carries segmentation geometry in the detection view"
        )
    return problems


def _validate_mask(record: Mapping[str, Any]) -> list[str]:
    """Check that a segmentation record carries usable geometry.

    Args:
        record: The COCO annotation record.

    Returns:
        One description per problem found.
    """
    problems: list[str] = []
    segmentation = record.get("segmentation")
    if not segmentation:
        problems.append(f"annotation {record['id']} has no segmentation geometry")
        return problems
    if record["area"] <= 0:
        problems.append(f"annotation {record['id']} has non-positive area {record['area']}")
    if isinstance(segmentation, dict):
        if "counts" not in segmentation or "size" not in segmentation:
            problems.append(f"annotation {record['id']} has an RLE without 'counts' or 'size'")
        return problems
    for ring in segmentation:
        if not isinstance(ring, list) or len(ring) < 6 or len(ring) % 2:
            problems.append(
                f"annotation {record['id']} has a polygon ring of length {len(ring)}, which "
                "cannot describe a closed shape"
            )
    return problems


def decode_mask(segmentation: Any, *, height: int, width: int) -> Any:
    """Rasterise a COCO segmentation to a binary mask.

    Args:
        segmentation: Polygon rings or a compressed RLE.
        height: Image height.
        width: Image width.

    Returns:
        The decoded mask array.

    Raises:
        CocoMaterializationError: If the geometry cannot be decoded.
    """
    try:
        if isinstance(segmentation, dict):
            counts = segmentation["counts"]
            payload = {
                "size": list(segmentation["size"]),
                "counts": counts.encode("ascii") if isinstance(counts, str) else counts,
            }
            return coco_mask.decode(payload)
        rles = coco_mask.frPyObjects(segmentation, height, width)
        return coco_mask.decode(coco_mask.merge(rles))
    except Exception as exc:  # pycocotools raises bare exceptions on bad input
        msg = f"could not decode segmentation: {type(exc).__name__}"
        raise CocoMaterializationError(msg) from None


@dataclass
class RoundTrip:
    """What the geometry round-trip measured.

    Attributes:
        polygons_checked: Polygon annotations compared coordinate by coordinate.
        rle_checked: RLE annotations compared by decoded mask.
        synthetic_checked: Synthetic rectangles compared coordinate by coordinate.
        matches: Annotations whose emitted geometry equals the canonical one.
        mismatches: One description per annotation that did not match.
        max_polygon_delta_px: Largest coordinate difference observed.
    """

    polygons_checked: int = 0
    rle_checked: int = 0
    synthetic_checked: int = 0
    matches: int = 0
    mismatches: list[str] = field(default_factory=list)
    max_polygon_delta_px: float = 0.0

    @property
    def checked(self) -> int:
        """Total annotations compared.

        Returns:
            The sum of the three kinds.
        """
        return self.polygons_checked + self.rle_checked + self.synthetic_checked

    def as_dict(self) -> dict[str, Any]:
        """Serialise for the manifest and the report.

        Returns:
            A JSON-serialisable mapping.
        """
        return {
            "annotations_checked": self.checked,
            "polygon_annotations_checked": self.polygons_checked,
            "rle_annotations_checked": self.rle_checked,
            "synthetic_rectangles_checked": self.synthetic_checked,
            "matches": self.matches,
            "mismatches": len(self.mismatches),
            "max_polygon_coordinate_delta_px": self.max_polygon_delta_px,
        }


def round_trip_geometry(
    canonical: Sequence[CanonicalAnnotation],
    emitted: Mapping[int, Mapping[str, Any]],
    ids: CanonicalIds,
    canvases: Mapping[str, tuple[int, int]],
    tolerances: Tolerances,
) -> RoundTrip:
    """Verify that what was written back is what the canonical state holds.

    The emitted records are read from the file on disk rather than from the
    in-memory document, so this measures the serialisation too: an RLE that
    survived ``json.dumps`` and ``json.loads`` intact, and a polygon whose
    coordinates were not silently reformatted.

    RLE is compared by **decoded mask**, not by comparing the ``counts`` strings.
    Identical strings would prove only that a copy happened; decoding both sides
    proves the emitted file describes the same pixels.

    Args:
        canonical: The canonical annotations of one split.
        emitted: Emitted segmentation records keyed by COCO annotation id.
        ids: The canonical id assignment.
        canvases: Image ``(width, height)`` keyed by source image id.
        tolerances: Numeric slack allowed.

    Returns:
        The round-trip result.
    """
    result = RoundTrip()
    for annotation in canonical:
        key = ids.annotation_ids[(annotation.source_image_id, annotation.annotation_id)]
        record = emitted.get(key)
        label = f"{annotation.source_image_id}/{annotation.annotation_id}"
        if record is None:
            result.mismatches.append(f"{label}: absent from the emitted segmentation view")
            continue
        kind = annotation.geometry_kind
        if kind == RLE:
            result.rle_checked += 1
        elif kind == SYNTHETIC_RECTANGLE:
            result.synthetic_checked += 1
        else:
            result.polygons_checked += 1

        width, height = canvases[annotation.source_image_id]
        problem = _compare_geometry(
            annotation, record["segmentation"], width=width, height=height, result=result
        )
        if problem is None:
            result.matches += 1
        else:
            result.mismatches.append(f"{label}: {problem}")
    _ = tolerances
    return result


def _compare_geometry(
    annotation: CanonicalAnnotation,
    emitted: Any,
    *,
    width: int,
    height: int,
    result: RoundTrip,
) -> str | None:
    """Compare one emitted geometry against its canonical form.

    Args:
        annotation: The canonical annotation.
        emitted: The geometry read back from the emitted file.
        width: Image width.
        height: Image height.
        result: Round-trip accumulator, updated with the largest delta seen.

    Returns:
        A description of the difference, or ``None`` when they agree.
    """
    canonical = annotation.segmentation
    if isinstance(canonical, dict) != isinstance(emitted, dict):
        return "the emitted geometry changed representation"

    if isinstance(canonical, dict):
        try:
            left = decode_mask(canonical, height=height, width=width)
            right = decode_mask(emitted, height=height, width=width)
        except CocoMaterializationError as exc:
            return str(exc)
        if left.shape != right.shape:
            return f"decoded mask shape {right.shape} != canonical {left.shape}"
        if not bool((left == right).all()):
            return "decoded masks differ"
        return None

    if len(canonical) != len(emitted):
        return f"{len(emitted)} polygon ring(s) emitted, canonical holds {len(canonical)}"
    for index, (canonical_ring, emitted_ring) in enumerate(zip(canonical, emitted, strict=True)):
        if len(canonical_ring) != len(emitted_ring):
            return (
                f"ring {index} has {len(emitted_ring)} coordinate(s), canonical has "
                f"{len(canonical_ring)}"
            )
        for left_value, right_value in zip(canonical_ring, emitted_ring, strict=True):
            delta = abs(float(left_value) - float(right_value))
            result.max_polygon_delta_px = max(result.max_polygon_delta_px, delta)
            if delta > 0.0:
                return f"ring {index} coordinate differs by {delta}"
    return None


def alignment_report(
    detection: Mapping[str, Any], segmentation: Mapping[str, Any], split: str
) -> dict[str, Any]:
    """Compare the two task views of one split, field by field.

    Args:
        detection: The detection document.
        segmentation: The segmentation document.
        split: Split name.

    Returns:
        A machine-readable alignment record, carrying ``problems`` when the two
        views disagree.
    """
    problems: list[str] = []

    detection_images = {record["id"]: record for record in detection["images"]}
    segmentation_images = {record["id"]: record for record in segmentation["images"]}
    if set(detection_images) != set(segmentation_images):
        problems.append("the two views do not hold the same COCO image ids")
    else:
        for image_id, record in detection_images.items():
            other = segmentation_images[image_id]
            if record["source_image_id"] != other["source_image_id"]:
                problems.append(f"image {image_id} maps to different source images")
            if (record["width"], record["height"]) != (other["width"], other["height"]):
                problems.append(f"image {image_id} has different dimensions between views")
            if record["file_name"] != other["file_name"]:
                problems.append(f"image {image_id} points at different files between views")

    detection_annotations = {record["id"]: record for record in detection["annotations"]}
    segmentation_annotations = {record["id"]: record for record in segmentation["annotations"]}
    if set(detection_annotations) != set(segmentation_annotations):
        problems.append("the two views do not hold the same COCO annotation ids")
    else:
        for annotation_id, record in detection_annotations.items():
            other = segmentation_annotations[annotation_id]
            for key in ("image_id", "category_id", "source_image_id", "source_annotation_id"):
                if record[key] != other[key]:
                    problems.append(f"annotation {annotation_id} disagrees on {key!r}")
            if record["bbox"] != other["bbox"]:
                problems.append(
                    f"annotation {annotation_id} has different boxes in the two views; both "
                    "must derive from the same segmentation"
                )

    if detection["categories"] != segmentation["categories"]:
        problems.append("the two views declare different category tables")

    return {
        "split": split,
        "images": len(detection["images"]),
        "annotations": len(detection["annotations"]),
        "image_ids_match": set(detection_images) == set(segmentation_images),
        "annotation_ids_match": set(detection_annotations) == set(segmentation_annotations),
        "categories_match": detection["categories"] == segmentation["categories"],
        "boxes_match": not any("different boxes" in problem for problem in problems),
        "aligned": not problems,
        "problems": problems,
    }


def bbox_agreement(annotations: Sequence[CanonicalAnnotation], tolerance: float) -> dict[str, Any]:
    """Compare boxes derived here against the boxes phase 5A measured.

    Two independent implementations reading the same geometry should agree. When
    they do, the derivation is corroborated; when they do not, one of them is
    wrong and the difference is worth seeing rather than averaging away.

    Args:
        annotations: The canonical annotations of a split.
        tolerance: Largest per-coordinate difference accepted.

    Returns:
        The comparison summary, carrying ``problems`` for anything over
        tolerance.
    """
    problems: list[str] = []
    max_delta = 0.0
    for annotation in annotations:
        derived = derive_bbox(annotation)
        canonical = [float(value) for value in annotation.canonical_bbox]
        delta = max(abs(a - b) for a, b in zip(derived, canonical, strict=True))
        max_delta = max(max_delta, delta)
        if delta > tolerance:
            problems.append(
                f"{annotation.source_image_id}/{annotation.annotation_id}: derived {derived} "
                f"differs from the phase 5A measurement {canonical} by {delta:.4f} px"
            )
    return {
        "annotations_compared": len(annotations),
        "max_delta_px": round(max_delta, 6),
        "tolerance_px": tolerance,
        "within_tolerance": not problems,
        "problems": problems,
    }


def document_fingerprint(document: Mapping[str, Any]) -> str:
    """Fingerprint a task document by the ground truth it carries.

    Excludes ``info``, which names the split and the protocol rather than the
    data, so the digest moves when an image, a category or an annotation moves
    and not when a description is reworded.

    Args:
        document: The COCO document.

    Returns:
        A SHA-256 hex digest.
    """
    return digest(
        {
            "categories": document["categories"],
            "images": document["images"],
            "annotations": document["annotations"],
        }
    )
