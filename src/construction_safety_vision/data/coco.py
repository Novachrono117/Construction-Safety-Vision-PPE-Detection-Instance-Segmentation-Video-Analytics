"""Structural inspection of a COCO instance-segmentation export.

This module answers "is the export internally consistent and does it really
contain segmentation geometry?". It deliberately stops short of exploratory data
analysis: no image is opened, no distribution is computed, no duplicate
detection is performed. Those belong to the audit phase.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"})
"""File suffixes counted as image files on disk."""

ANNOTATION_FILENAME = "_annotations.coco.json"
"""Filename used by the provider for per-split COCO annotations."""

REQUIRED_TOP_LEVEL_KEYS = ("images", "annotations", "categories")
"""Keys a COCO document must contain."""


class CocoValidationError(ValueError):
    """Raised when a COCO document is missing, malformed or structurally invalid."""


@dataclass
class SplitInspection:
    """Structural facts about one exported split.

    Every field is computed from the exported files, never taken from provider
    metadata.

    Attributes:
        split: Split directory name as exported.
        annotations_path: Path of the COCO document, relative to the export root.
        image_records: Number of entries in ``images``.
        annotation_records: Number of entries in ``annotations``.
        image_files_on_disk: Number of image files present in the split directory.
        categories: Mapping of category id to name.
        annotations_per_category: Annotation count keyed by category name.
        segmentation_types: Counter of segmentation representations encountered.
        annotations_with_bbox: Annotations that also carry a ``bbox`` field.
        annotations_with_segmentation: Annotations carrying a non-empty
            ``segmentation`` field.
        missing_image_files: Referenced file names absent from disk.
        orphan_image_files: Image files on disk not referenced by any record.
        dangling_image_ids: Annotation ``image_id`` values with no image record.
        undefined_category_ids: Annotation ``category_id`` values with no category.
        duplicate_image_ids: Image ids appearing more than once.
        duplicate_annotation_ids: Annotation ids appearing more than once.
        images_without_annotations: Image records with no annotation.
    """

    split: str
    annotations_path: str
    image_records: int = 0
    annotation_records: int = 0
    image_files_on_disk: int = 0
    categories: dict[int, str] = field(default_factory=dict)
    annotations_per_category: dict[str, int] = field(default_factory=dict)
    segmentation_types: dict[str, int] = field(default_factory=dict)
    annotations_with_bbox: int = 0
    annotations_with_segmentation: int = 0
    missing_image_files: list[str] = field(default_factory=list)
    orphan_image_files: list[str] = field(default_factory=list)
    dangling_image_ids: list[int] = field(default_factory=list)
    undefined_category_ids: list[int] = field(default_factory=list)
    duplicate_image_ids: list[int] = field(default_factory=list)
    duplicate_annotation_ids: list[int] = field(default_factory=list)
    images_without_annotations: int = 0

    @property
    def is_structurally_sound(self) -> bool:
        """Report whether no referential-integrity problem was found.

        Returns:
            ``True`` when there are no dangling references, undefined
            categories, duplicate ids or missing image files.
        """
        return not (
            self.missing_image_files
            or self.dangling_image_ids
            or self.undefined_category_ids
            or self.duplicate_image_ids
            or self.duplicate_annotation_ids
        )


def classify_segmentation(value: Any) -> str:
    """Classify the representation of a COCO ``segmentation`` field.

    Args:
        value: The raw ``segmentation`` value.

    Returns:
        One of ``"absent"``, ``"empty"``, ``"polygon"``, ``"rle"``,
        ``"degenerate_polygon"`` or ``"unknown"``.
    """
    if value is None:
        return "absent"
    if isinstance(value, dict):
        return "rle" if "counts" in value else "unknown"
    if isinstance(value, list):
        if not value:
            return "empty"
        if all(isinstance(part, list) for part in value):
            # A valid polygon ring needs at least 3 points, i.e. 6 coordinates.
            if any(len(part) < 6 or len(part) % 2 for part in value):
                return "degenerate_polygon"
            return "polygon"
        return "unknown"
    return "unknown"


def load_coco_document(path: Path) -> dict[str, Any]:
    """Load and structurally validate a COCO JSON document.

    Args:
        path: Path to the COCO document.

    Returns:
        The decoded document.

    Raises:
        CocoValidationError: If the file is missing, is not valid JSON, is not a
            JSON object, or lacks a required top-level key or list type.
    """
    if not path.is_file():
        msg = f"COCO annotation file not found: {path}"
        raise CocoValidationError(msg)
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        msg = f"COCO annotation file is not valid JSON: {path.name} ({exc})"
        raise CocoValidationError(msg) from None
    if not isinstance(document, dict):
        msg = f"COCO annotation file must contain a JSON object: {path.name}"
        raise CocoValidationError(msg)
    for key in REQUIRED_TOP_LEVEL_KEYS:
        if key not in document:
            msg = f"COCO document {path.name} is missing the required key {key!r}"
            raise CocoValidationError(msg)
        if not isinstance(document[key], list):
            msg = f"COCO document {path.name} key {key!r} must be a list"
            raise CocoValidationError(msg)
    return document


def inspect_split(split_dir: Path, *, export_root: Path | None = None) -> SplitInspection:
    """Inspect one exported split directory.

    Args:
        split_dir: Directory containing images and the COCO document.
        export_root: Root used to shorten reported paths.

    Returns:
        The structural inspection of the split.

    Raises:
        CocoValidationError: If the COCO document is missing or malformed.
    """
    annotations_path = split_dir / ANNOTATION_FILENAME
    document = load_coco_document(annotations_path)
    base = export_root or split_dir.parent
    try:
        relative = annotations_path.relative_to(base).as_posix()
    except ValueError:
        relative = annotations_path.name

    report = SplitInspection(split=split_dir.name, annotations_path=relative)

    categories = {}
    for entry in document["categories"]:
        if isinstance(entry, dict) and "id" in entry:
            categories[int(entry["id"])] = str(entry.get("name", ""))
    report.categories = categories

    image_ids: Counter[int] = Counter()
    file_names: dict[int, str] = {}
    for entry in document["images"]:
        if not isinstance(entry, dict) or "id" not in entry:
            continue
        image_id = int(entry["id"])
        image_ids[image_id] += 1
        file_names[image_id] = str(entry.get("file_name", ""))
    report.image_records = len(document["images"])
    report.duplicate_image_ids = sorted(i for i, n in image_ids.items() if n > 1)

    on_disk = {path.name for path in split_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES}
    report.image_files_on_disk = len(on_disk)
    referenced = set(file_names.values())
    report.missing_image_files = sorted(referenced - on_disk)
    report.orphan_image_files = sorted(on_disk - referenced)

    annotation_ids: Counter[int] = Counter()
    per_category: Counter[str] = Counter()
    seg_types: Counter[str] = Counter()
    dangling: set[int] = set()
    undefined: set[int] = set()
    annotated_images: set[int] = set()

    report.annotation_records = len(document["annotations"])
    for entry in document["annotations"]:
        if not isinstance(entry, dict):
            continue
        if "id" in entry:
            annotation_ids[int(entry["id"])] += 1
        image_id = entry.get("image_id")
        if image_id is not None:
            image_id = int(image_id)
            annotated_images.add(image_id)
            if image_id not in image_ids:
                dangling.add(image_id)
        category_id = entry.get("category_id")
        if category_id is not None:
            category_id = int(category_id)
            if category_id in categories:
                per_category[categories[category_id]] += 1
            else:
                undefined.add(category_id)
        seg_kind = classify_segmentation(entry.get("segmentation"))
        seg_types[seg_kind] += 1
        if seg_kind not in ("absent", "empty"):
            report.annotations_with_segmentation += 1
        if entry.get("bbox"):
            report.annotations_with_bbox += 1

    report.duplicate_annotation_ids = sorted(i for i, n in annotation_ids.items() if n > 1)
    report.annotations_per_category = dict(sorted(per_category.items()))
    report.segmentation_types = dict(sorted(seg_types.items()))
    report.dangling_image_ids = sorted(dangling)
    report.undefined_category_ids = sorted(undefined)
    report.images_without_annotations = len(set(image_ids) - annotated_images)
    return report


def find_split_dirs(export_root: Path) -> list[Path]:
    """Locate exported split directories.

    A split directory is any directory directly under the export root that
    contains a COCO annotation file.

    Args:
        export_root: Root of the extracted export.

    Returns:
        Split directories, sorted by name.
    """
    if not export_root.is_dir():
        return []
    return sorted(
        path
        for path in export_root.iterdir()
        if path.is_dir() and (path / ANNOTATION_FILENAME).is_file()
    )


def inspect_export(export_root: Path) -> list[SplitInspection]:
    """Inspect every split of an extracted export.

    Args:
        export_root: Root of the extracted export.

    Returns:
        One inspection per split, sorted by split name.

    Raises:
        CocoValidationError: If no split with annotations is found.
    """
    split_dirs = find_split_dirs(export_root)
    if not split_dirs:
        msg = (
            f"No split directory containing {ANNOTATION_FILENAME!r} was found under "
            f"{export_root}. The export layout is not the expected COCO one."
        )
        raise CocoValidationError(msg)
    return [inspect_split(path, export_root=export_root) for path in split_dirs]
