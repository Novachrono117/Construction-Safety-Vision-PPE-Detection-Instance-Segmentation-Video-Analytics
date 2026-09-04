"""Annotation drift between the live source project and the version-4 snapshot.

The two annotation states are not comparable as raw numbers. The export's images
are stretched to 640x640, its train split holds an augmented copy of every source
image alongside the pristine one, and its stored boxes are known to disagree with
its own segmentation geometry. So the comparison is built on three deliberate
choices.

*Population.* Only the non-augmented export representation of each source image
participates, resolved by :mod:`construction_safety_vision.data.v4mapping`. The
augmented copies are not a second observation of anything.

*Coordinates.* Boxes are compared in a normalised frame - divided by each side's
own image dimensions. The export's preprocessing is a pure stretch resize, so a
normalised box is invariant under it and nothing has to be inverted.

*Geometry.* An export box is derived from its segmentation rather than read from
its stored ``bbox`` field, because phase 4B measured those two disagreeing by up
to 123 px. Segmentation is this project's source of truth on both sides.

A net count difference can hide equal numbers of additions and removals, so
additions and removals are counted separately and never inferred from the net.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

MATCH_IOU = 0.5
"""Lowest box overlap at which two annotations are taken to describe one object."""

GEOMETRY_STABLE_IOU = 0.95
"""Overlap above which a matched pair's geometry is treated as unchanged."""

UNCHANGED = "unchanged"
"""Same annotations on both sides, same classes, geometry stable."""

MODIFIED = "modified_in_place"
"""Same annotation count, but a class or a shape changed."""

ADDED_ONLY = "added_only"
"""Annotations present now that the snapshot does not have."""

REMOVED_ONLY = "removed_only"
"""Annotations in the snapshot that are no longer present."""

ADDED_AND_REMOVED = "added_and_removed"
"""Both directions on the same image; the net delta alone would hide one."""

UNMAPPED = "unmapped"
"""The source image has no resolved snapshot representation to compare against."""


@dataclass(frozen=True)
class DriftInstance:
    """One annotation reduced to what the comparison needs.

    Attributes:
        key: Identifier of the annotation on its own side.
        label: Class name.
        bbox: Box as ``(x, y, w, h)`` normalised to the unit square.
    """

    key: str
    label: str
    bbox: tuple[float, float, float, float]


def normalise_bbox(
    bbox: list[float] | tuple[float, ...], *, width: float, height: float
) -> tuple[float, float, float, float]:
    """Scale a pixel box into the unit square.

    Args:
        bbox: Box in COCO ``[x, y, w, h]`` pixel convention.
        width: Image width the box is expressed in.
        height: Image height the box is expressed in.

    Returns:
        The box with every component divided by its own axis length.

    Raises:
        ValueError: If either dimension is not positive.
    """
    if width <= 0 or height <= 0:
        msg = f"Cannot normalise a box against dimensions {width}x{height}"
        raise ValueError(msg)
    x, y, w, h = (float(v) for v in bbox)
    return (x / width, y / height, w / width, h / height)


def bbox_iou(
    left: tuple[float, float, float, float], right: tuple[float, float, float, float]
) -> float:
    """Intersection over union of two boxes.

    Args:
        left: First box as ``(x, y, w, h)``.
        right: Second box as ``(x, y, w, h)``.

    Returns:
        The overlap ratio, zero when the boxes are disjoint or degenerate.
    """
    lx, ly, lw, lh = left
    rx, ry, rw, rh = right
    ix = max(0.0, min(lx + lw, rx + rw) - max(lx, rx))
    iy = max(0.0, min(ly + lh, ry + rh) - max(ly, ry))
    intersection = ix * iy
    union = lw * lh + rw * rh - intersection
    return intersection / union if union > 0 else 0.0


def containment(
    inner: tuple[float, float, float, float], outer: tuple[float, float, float, float]
) -> float:
    """Fraction of one box that lies inside another.

    Unlike IoU this is asymmetric, which is the point: it answers "is this small
    new annotation sitting inside something already annotated?" without the
    answer being dragged to zero by the size difference.

    Args:
        inner: The box being tested for enclosure, as ``(x, y, w, h)``.
        outer: The box that might enclose it.

    Returns:
        The enclosed fraction of ``inner``, zero when it has no area.
    """
    ix = max(0.0, min(inner[0] + inner[2], outer[0] + outer[2]) - max(inner[0], outer[0]))
    iy = max(0.0, min(inner[1] + inner[3], outer[1] + outer[3]) - max(inner[1], outer[1]))
    area = inner[2] * inner[3]
    return (ix * iy) / area if area > 0 else 0.0


def describe_addition(added: DriftInstance, snapshot: list[DriftInstance]) -> tuple[float, float]:
    """Locate a new annotation relative to what the snapshot already had.

    An annotation that sits wholly inside an existing annotation of its own
    class, while being a small fraction of it, cannot be a newly covered object:
    it is a fragment or a duplicate of something already labelled. Reporting both
    numbers keeps that inference checkable instead of asserted.

    Args:
        added: The annotation with no snapshot counterpart.
        snapshot: Every snapshot annotation on the same image.

    Returns:
        How much of the addition lies inside the best-enclosing same-class
        snapshot annotation, and the addition's area as a fraction of that
        annotation's area. ``(0.0, 1.0)`` when the class is absent.
    """
    same_class = [
        (containment(added.bbox, other.bbox), other)
        for other in snapshot
        if other.label == added.label
    ]
    if not same_class:
        return 0.0, 1.0
    enclosed, container = max(same_class, key=lambda entry: entry[0])
    container_area = container.bbox[2] * container.bbox[3]
    added_area = added.bbox[2] * added.bbox[3]
    ratio = added_area / container_area if container_area > 0 else 1.0
    return enclosed, ratio


def match_instances(
    current: list[DriftInstance],
    snapshot: list[DriftInstance],
    *,
    threshold: float = MATCH_IOU,
) -> tuple[
    list[tuple[DriftInstance, DriftInstance, float]],
    list[DriftInstance],
    list[DriftInstance],
]:
    """Pair annotations across the two states.

    Same-class pairs are resolved first so that a genuine relabel is not masked
    by a greedy cross-class match on the same object. Remaining annotations are
    then matched across classes, which is what exposes a relabel.

    Args:
        current: Annotations in the live source state.
        snapshot: Annotations in the version-4 snapshot.
        threshold: Lowest overlap accepted as a match.

    Returns:
        The matched triples ``(current, snapshot, iou)``, the unmatched current
        annotations, and the unmatched snapshot annotations.
    """
    matches: list[tuple[DriftInstance, DriftInstance, float]] = []
    open_current = list(current)
    open_snapshot = list(snapshot)

    for same_class_only in (True, False):
        candidates = [
            (bbox_iou(c.bbox, s.bbox), c, s)
            for c in open_current
            for s in open_snapshot
            if not same_class_only or c.label == s.label
        ]
        candidates = [entry for entry in candidates if entry[0] >= threshold]
        candidates.sort(key=lambda entry: (-entry[0], entry[1].key, entry[2].key))
        taken_current: set[int] = set()
        taken_snapshot: set[int] = set()
        for iou, c, s in candidates:
            if id(c) in taken_current or id(s) in taken_snapshot:
                continue
            taken_current.add(id(c))
            taken_snapshot.add(id(s))
            matches.append((c, s, iou))
        open_current = [c for c in open_current if id(c) not in taken_current]
        open_snapshot = [s for s in open_snapshot if id(s) not in taken_snapshot]

    return matches, open_current, open_snapshot


def _class_counts(instances: list[DriftInstance]) -> dict[str, int]:
    """Count instances per class.

    Args:
        instances: Annotations to count.

    Returns:
        Counts keyed by class name, sorted.
    """
    counts: dict[str, int] = {}
    for instance in instances:
        counts[instance.label] = counts.get(instance.label, 0) + 1
    return dict(sorted(counts.items()))


@dataclass
class ImageDrift:
    """How one source image's annotations changed since the snapshot.

    Attributes:
        source_image_id: Provider source-image identifier.
        provider_split: The provider's split assignment for this image.
        current_count: Annotations in the live source state.
        v4_count: Annotations on the non-augmented snapshot representation.
        current_classes: Live class counts.
        v4_classes: Snapshot class counts.
        added: Annotations with no snapshot counterpart.
        removed: Snapshot annotations with no live counterpart.
        relabelled: Matched pairs whose class differs.
        reshaped: Matched pairs whose overlap falls below the stability bound.
        matched: One ``(iou, snapshot box area)`` pair per match, where the area
            is in the unit square. Kept so the report can show that low overlap
            tracks object size, which is what separates rasterisation noise from
            a real edit.
        addition_placement: One ``(enclosed fraction, area ratio)`` pair per
            added annotation, describing where it sits relative to the
            same-class annotation the snapshot already had there.
        match_status: Summary classification of this image.
        notes: Remarks.
    """

    source_image_id: str
    provider_split: str
    current_count: int
    v4_count: int
    current_classes: dict[str, int] = field(default_factory=dict)
    v4_classes: dict[str, int] = field(default_factory=dict)
    added: list[DriftInstance] = field(default_factory=list)
    removed: list[DriftInstance] = field(default_factory=list)
    relabelled: list[tuple[str, str]] = field(default_factory=list)
    reshaped: int = 0
    matched: list[tuple[float, float]] = field(default_factory=list)
    addition_placement: list[tuple[float, float]] = field(default_factory=list)
    match_status: str = UNCHANGED
    notes: str = ""

    @property
    def delta(self) -> int:
        """Net annotation change.

        Returns:
            Live count minus snapshot count.
        """
        return self.current_count - self.v4_count

    @property
    def added_class_counts(self) -> dict[str, int]:
        """Class counts of the annotations that appeared.

        Returns:
            Counts keyed by class name.
        """
        return _class_counts(self.added)

    @property
    def removed_class_counts(self) -> dict[str, int]:
        """Class counts of the annotations that disappeared.

        Returns:
            Counts keyed by class name.
        """
        return _class_counts(self.removed)

    def csv_row(self) -> dict[str, Any]:
        """Serialise for the committed drift table.

        Returns:
            A mapping of column name to value.
        """
        return {
            "source_image_id": self.source_image_id,
            "provider_split": self.provider_split,
            "current_count": self.current_count,
            "v4_count": self.v4_count,
            "delta": self.delta,
            "current_classes": _format_counts(self.current_classes),
            "v4_classes": _format_counts(self.v4_classes),
            "added_class_counts": _format_counts(self.added_class_counts),
            "removed_class_counts": _format_counts(self.removed_class_counts),
            "relabelled": ";".join(f"{old}->{new}" for old, new in self.relabelled),
            "reshaped_matches": self.reshaped,
            "match_status": self.match_status,
            "notes": self.notes,
        }


def _format_counts(counts: dict[str, int]) -> str:
    """Render class counts as a stable, spreadsheet-safe string.

    Args:
        counts: Counts keyed by class name.

    Returns:
        A semicolon-separated rendering, empty when there is nothing to show.
    """
    return ";".join(f"{name}={count}" for name, count in sorted(counts.items()))


def compare_image(
    source_image_id: str,
    provider_split: str,
    current: list[DriftInstance],
    snapshot: list[DriftInstance],
    *,
    threshold: float = MATCH_IOU,
    stable_iou: float = GEOMETRY_STABLE_IOU,
) -> ImageDrift:
    """Compare one image's annotations across the two states.

    Args:
        source_image_id: Provider source-image identifier.
        provider_split: The provider's split assignment.
        current: Live annotations, already normalised.
        snapshot: Snapshot annotations, already normalised.
        threshold: Lowest overlap accepted as a match.
        stable_iou: Overlap above which geometry counts as unchanged.

    Returns:
        The drift record for this image.
    """
    matches, added, removed = match_instances(current, snapshot, threshold=threshold)
    relabelled = [(s.label, c.label) for c, s, _ in matches if c.label != s.label]
    reshaped = sum(1 for _, _, iou in matches if iou < stable_iou)

    if added and removed:
        status = ADDED_AND_REMOVED
    elif added:
        status = ADDED_ONLY
    elif removed:
        status = REMOVED_ONLY
    elif relabelled or reshaped:
        status = MODIFIED
    else:
        status = UNCHANGED

    return ImageDrift(
        source_image_id=source_image_id,
        provider_split=provider_split,
        current_count=len(current),
        v4_count=len(snapshot),
        current_classes=_class_counts(current),
        v4_classes=_class_counts(snapshot),
        added=added,
        removed=removed,
        relabelled=relabelled,
        reshaped=reshaped,
        matched=[(iou, s.bbox[2] * s.bbox[3]) for _, s, iou in matches],
        addition_placement=[describe_addition(instance, snapshot) for instance in added],
        match_status=status,
    )


def summarise(drifts: list[ImageDrift]) -> dict[str, Any]:
    """Aggregate per-image drift into the figures the report needs.

    Gross additions and removals are summed independently of the net delta,
    because an image that gained one annotation and lost another nets to zero
    while having changed twice.

    Args:
        drifts: Per-image drift records.

    Returns:
        A JSON-serialisable summary.
    """
    added_by_class: dict[str, int] = {}
    removed_by_class: dict[str, int] = {}
    by_split: dict[str, dict[str, int]] = {}
    by_status: dict[str, int] = {}

    for drift in drifts:
        by_status[drift.match_status] = by_status.get(drift.match_status, 0) + 1
        split = by_split.setdefault(
            drift.provider_split,
            {"images": 0, "changed_images": 0, "added": 0, "removed": 0, "net": 0},
        )
        split["images"] += 1
        split["added"] += len(drift.added)
        split["removed"] += len(drift.removed)
        split["net"] += drift.delta
        if drift.match_status != UNCHANGED:
            split["changed_images"] += 1
        for name, count in drift.added_class_counts.items():
            added_by_class[name] = added_by_class.get(name, 0) + count
        for name, count in drift.removed_class_counts.items():
            removed_by_class[name] = removed_by_class.get(name, 0) + count

    gross_added = sum(len(d.added) for d in drifts)
    gross_removed = sum(len(d.removed) for d in drifts)
    classes = sorted(set(added_by_class) | set(removed_by_class))

    return {
        "images_compared": len(drifts),
        "current_annotations": sum(d.current_count for d in drifts),
        "v4_annotations": sum(d.v4_count for d in drifts),
        "net_delta": sum(d.delta for d in drifts),
        "gross_added": gross_added,
        "gross_removed": gross_removed,
        "images_unchanged_count": sum(1 for d in drifts if d.delta == 0),
        "images_positive_delta": sum(1 for d in drifts if d.delta > 0),
        "images_negative_delta": sum(1 for d in drifts if d.delta < 0),
        "images_touched": sum(1 for d in drifts if d.match_status != UNCHANGED),
        "match_status_counts": dict(sorted(by_status.items())),
        "added_by_class": dict(sorted(added_by_class.items())),
        "removed_by_class": dict(sorted(removed_by_class.items())),
        "net_by_class": {
            name: added_by_class.get(name, 0) - removed_by_class.get(name, 0) for name in classes
        },
        "by_provider_split": dict(sorted(by_split.items())),
        "relabelled_pairs": sum(len(d.relabelled) for d in drifts),
        "reshaped_matches": sum(d.reshaped for d in drifts),
        **_overlap_evidence([pair for d in drifts for pair in d.matched]),
        "addition_placement": _addition_evidence(
            [pair for d in drifts for pair in d.addition_placement]
        ),
    }


ENCLOSED_THRESHOLD = 0.80
"""Enclosed fraction above which an addition lies inside an existing annotation."""

CONTAINER_RATIO_THRESHOLDS = (0.01, 0.05, 0.10, 0.25, 0.50)
"""Reported sizes of an addition relative to the annotation enclosing it."""


def _addition_evidence(placements: list[tuple[float, float]]) -> dict[str, Any]:
    """Describe where the new annotations sit relative to the old ones.

    Args:
        placements: One ``(enclosed fraction, area ratio)`` per addition.

    Returns:
        Counts that show whether the additions cover new objects or subdivide
        objects that were already labelled.
    """
    if not placements:
        return {"additions": 0}
    enclosed = sum(1 for value, _ in placements if value >= ENCLOSED_THRESHOLD)
    ratios = sorted(ratio for _, ratio in placements)
    return {
        "additions": len(placements),
        "inside_same_class_annotation": enclosed,
        "covering_a_new_object": len(placements) - enclosed,
        "median_area_fraction_of_container": round(ratios[len(ratios) // 2], 4),
        "smaller_than_container_by": {
            f"{threshold:.0%}": sum(1 for ratio in ratios if ratio < threshold)
            for threshold in CONTAINER_RATIO_THRESHOLDS
        },
    }


SNAPSHOT_SIDE_PX = 640
"""Side length the snapshot's geometry is rasterised at, used to express areas."""

AREA_BUCKETS_PX = ((0, 256), (256, 1024), (1024, 4096), (4096, 16384))
"""Object-size buckets, in snapshot pixels squared, with an open top bucket."""

RESHAPE_STRICT_IOU = 0.80
"""Overlap below which a *large* object's difference cannot be quantisation."""


def _overlap_evidence(matched: list[tuple[float, float]]) -> dict[str, Any]:
    """Describe how matched-pair overlap depends on object size.

    A matched pair is compared across two rasterisations of different resolution:
    the live geometry in full original pixels, the snapshot's in 640x640. A single
    pixel of quantisation is a large fraction of a small object and a negligible
    one of a large object, so an overlap threshold applied without regard to size
    measures the export's resolution as much as it measures editing. Reporting the
    size dependence is what lets a reader tell the two apart.

    Args:
        matched: One ``(iou, snapshot box area in the unit square)`` per match.

    Returns:
        Overlap counts by threshold and by object-size bucket.
    """
    if not matched:
        return {"matched_pairs": 0, "iou_below": {}, "iou_by_object_size": {}, "reshaped_large": 0}

    scale = SNAPSHOT_SIDE_PX * SNAPSHOT_SIDE_PX
    sized = [(iou, area * scale) for iou, area in matched]
    buckets: dict[str, dict[str, Any]] = {}
    for low, high in (*AREA_BUCKETS_PX, (AREA_BUCKETS_PX[-1][1], float("inf"))):
        group = [iou for iou, area in sized if low <= area < high]
        if not group:
            continue
        label = f"{low}-{int(high)}" if high != float("inf") else f"{low}+"
        buckets[label] = {
            "pairs": len(group),
            "below_0.95": sum(1 for iou in group if iou < GEOMETRY_STABLE_IOU),
            "median_iou": round(sorted(group)[len(group) // 2], 4),
        }
    return {
        "matched_pairs": len(sized),
        "iou_below": {
            f"{t:.2f}": sum(1 for iou, _ in sized if iou < t)
            for t in (0.99, 0.95, 0.90, 0.80, 0.70, 0.50)
        },
        "iou_by_object_size": buckets,
        "reshaped_large": sum(
            1 for iou, area in sized if iou < RESHAPE_STRICT_IOU and area >= AREA_BUCKETS_PX[-1][0]
        ),
    }
