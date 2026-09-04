"""The canonical modelling population: which images and annotations may be modelled.

Phase 5A decided *which annotation state* is authoritative. This module decides
*which of it is eligible*, and which images must stay together when a split is
eventually designed. It does not create a split, and it deliberately cannot: no
function here knows about train, validation or test.

Three ideas keep the result honest.

**Exclusion is logical, never physical.** An excluded image stays in the source
provenance population of 436 and stays on disk; it is marked ineligible with a
reason and a decision source. Nothing is deleted, and the difference between "the
data we obtained" and "the data we model" is written down rather than lost.

**Every action carries its provenance.** A human judgement from phase 4B and a
geometric derivation are both recorded, but never as the same kind of evidence.

**Grouping is conservative.** Only visually confirmed semantic duplicates are
merged. A perceptual-hash candidate that no person has reviewed stays a singleton
and is recorded separately, because merging on a hash alone would silently shrink
the pool a future split can draw from.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

CLASS_ORDER: tuple[str, ...] = (
    "helmet_loose",
    "helmet_on_head",
    "person",
    "vest_loose",
    "vest_on_body",
)
"""The canonical class set, in a fixed order.

Alphabetical, and written out rather than derived, so the index a class receives
can never depend on dictionary iteration order, on the provider's category ids,
or on which classes happen to appear in a given run. The provider's placeholder
``object`` category is absent by construction.
"""

ELIGIBLE = "ELIGIBLE"
"""Image belongs to the modelling population."""

EXCLUDED = "EXCLUDED"
"""Image stays in the source population but is not modelled."""

KEEP_IMAGE = "KEEP_IMAGE"
"""Action recorded for an image that enters the modelling population."""

EXCLUDE_IMAGE = "EXCLUDE_IMAGE"
"""Action recorded for an image held out of the modelling population."""

KEEP_ANNOTATION = "KEEP"
"""Annotation enters the modelling population unchanged."""

MATERIALISE_RECTANGLE = "MATERIALIZE_RECTANGLE_FROM_BBOX"
"""Annotation enters with geometry synthesised from its bounding box."""

EXCLUDE_WITH_IMAGE = "EXCLUDE_WITH_IMAGE"
"""Annotation leaves because its image is not in the modelling population."""

SYNTHETIC_GEOMETRY = "SYNTHETIC_FROM_PROVIDER_BBOX"
"""Geometry origin marker for a rectangle this pipeline generated."""

PROVIDER_GEOMETRY = "PROVIDER_SEGMENTATION"
"""Geometry origin marker for geometry the provider actually holds."""

NESTED_SAME_CLASS_CANDIDATE = "NESTED_SAME_CLASS_CANDIDATE"
"""Flag for an annotation that lies inside another of its own class.

Descriptive, not a verdict. Phase 5B measured that nesting alone does not
separate an annotation error from a legitimate instance split or a geometry
refinement, so no annotation is excluded on this basis and the wording avoids
calling anything a fragment that nobody has established to be one.
"""

SINGLETON = "SINGLETON"
"""A group of one image."""

SEMANTIC_DUPLICATE = "SEMANTIC_DUPLICATE"
"""A group of images a person confirmed to show the same content."""

UNCONFIRMED_CANDIDATE = "UNCONFIRMED_GROUP_CANDIDATE"
"""A perceptual-hash relation nobody has reviewed. Never merges a group."""


class PopulationError(ValueError):
    """Raised when the modelling population cannot be built as specified."""


def build_class_map(classes: tuple[str, ...] = CLASS_ORDER) -> dict[str, int]:
    """Assign a stable index to each canonical class.

    Args:
        classes: Class names in their canonical order.

    Returns:
        A mapping of class name to zero-based index.

    Raises:
        PopulationError: If a class name repeats.
    """
    if len(set(classes)) != len(classes):
        msg = f"Canonical class order contains a duplicate: {classes}"
        raise PopulationError(msg)
    return {name: index for index, name in enumerate(classes)}


def rectangle_polygon(
    bbox: list[float] | tuple[float, ...], *, width: int, height: int
) -> list[list[float]]:
    """Build a closed rectangle polygon from a bounding box.

    Used for the annotations the provider stores as a class and a box with no
    segmentation. The result is a synthetic mask and is labelled as one wherever
    it is recorded; it is not evidence that anyone drew this outline.

    Corners are emitted in a single consistent winding, so the ring cannot
    self-intersect. The box is clipped to the image canvas first.

    Args:
        bbox: Box in COCO ``[x, y, w, h]`` original-image pixels.
        width: Image width in pixels.
        height: Image height in pixels.

    Returns:
        A COCO polygon segmentation: one ring of four corners.

    Raises:
        PopulationError: If the box has no area inside the canvas.
    """
    x, y, box_width, box_height = (float(v) for v in bbox)
    left = min(max(x, 0.0), float(width))
    top = min(max(y, 0.0), float(height))
    right = min(max(x + box_width, 0.0), float(width))
    bottom = min(max(y + box_height, 0.0), float(height))
    if right - left <= 0.0 or bottom - top <= 0.0:
        msg = (
            f"Box {list(bbox)} has no area inside a {width}x{height} canvas; "
            "it cannot be materialised as a rectangle"
        )
        raise PopulationError(msg)
    return [[left, top, right, top, right, bottom, left, bottom]]


@dataclass
class ImageRecord:
    """One source image and its status in the modelling population.

    Attributes:
        source_image_id: Provider source-image identifier. The stable key.
        name: Original filename. Never used as identity.
        width: Original width in pixels.
        height: Original height in pixels.
        status: :data:`ELIGIBLE` or :data:`EXCLUDED`.
        action: :data:`KEEP_IMAGE` or :data:`EXCLUDE_IMAGE`.
        reason: Why the action was taken.
        decision_source: Where the decision came from.
        annotation_count: Annotations the canonical snapshot holds for it.
        eligible_annotation_count: Annotations entering the modelling population.
        group_id: The split unit it belongs to, empty when excluded.
    """

    source_image_id: str
    name: str
    width: int
    height: int
    status: str
    action: str
    reason: str
    decision_source: str
    annotation_count: int = 0
    eligible_annotation_count: int = 0
    group_id: str = ""

    @property
    def is_zero_instance(self) -> bool:
        """Whether the image carries no eligible annotation.

        Returns:
            ``True`` when nothing is annotated on it.
        """
        return self.eligible_annotation_count == 0

    def csv_row(self) -> dict[str, Any]:
        """Serialise for the committed population table.

        Returns:
            A mapping of column name to value.
        """
        return {
            "source_image_id": self.source_image_id,
            "name": self.name,
            "width": self.width,
            "height": self.height,
            "status": self.status,
            "action": self.action,
            "reason": self.reason,
            "decision_source": self.decision_source,
            "annotation_count": self.annotation_count,
            "eligible_annotation_count": self.eligible_annotation_count,
            "zero_instance": "true" if self.is_zero_instance else "false",
            "group_id": self.group_id,
        }


@dataclass
class AnnotationRecord:
    """One canonical annotation and what the modelling layer does with it.

    Attributes:
        source_image_id: Provider source-image identifier.
        annotation_id: Provider annotation identifier, unique within the image.
        label: Canonical class name.
        geometry_kind: Geometry classification from phase 5A.
        action: What the modelling layer does with it.
        reason: Why.
        decision_source: Where the decision came from.
        geometry_origin: Whether the geometry is the provider's or synthesised.
        review_flag: Non-empty when an unresolved question is attached to it.
        segmentation: The canonical segmentation entering the modelling layer.
        bbox: Box in COCO ``[x, y, w, h]`` original-image pixels.
        area: Mask area in pixels.
    """

    source_image_id: str
    annotation_id: str
    label: str
    geometry_kind: str
    action: str
    reason: str
    decision_source: str
    geometry_origin: str
    review_flag: str = ""
    segmentation: Any = None
    bbox: list[float] = field(default_factory=list)
    area: float | None = None

    @property
    def is_eligible(self) -> bool:
        """Whether this annotation enters the modelling population.

        Returns:
            ``True`` unless it was excluded.
        """
        return self.action in (KEEP_ANNOTATION, MATERIALISE_RECTANGLE)

    def csv_row(self) -> dict[str, Any]:
        """Serialise for the committed action table.

        Geometry is excluded: it is bulk data and lives in the interim layer.

        Returns:
            A mapping of column name to value.
        """
        return {
            "source_image_id": self.source_image_id,
            "annotation_id": self.annotation_id,
            "label": self.label,
            "geometry_kind": self.geometry_kind,
            "action": self.action,
            "reason": self.reason,
            "decision_source": self.decision_source,
            "geometry_origin": self.geometry_origin,
            "review_flag": self.review_flag,
            "bbox": ";".join(f"{v:.3f}" for v in self.bbox),
            "area": f"{self.area:.3f}" if self.area is not None else "",
        }


@dataclass
class Group:
    """An indivisible unit for a future split.

    Attributes:
        group_id: Deterministic identifier.
        members: Source image ids, sorted.
        group_type: :data:`SINGLETON` or :data:`SEMANTIC_DUPLICATE`.
        group_origin: What established the grouping.
        notes: Remarks.
    """

    group_id: str
    members: list[str]
    group_type: str
    group_origin: str
    notes: str = ""

    @property
    def split_indivisible(self) -> bool:
        """Whether the members must land in the same split.

        Returns:
            ``True`` always: a group is the unit a split assigns.
        """
        return True

    def csv_rows(self) -> list[dict[str, Any]]:
        """Serialise one row per member.

        Returns:
            A list of mappings of column name to value.
        """
        return [
            {
                "group_id": self.group_id,
                "source_image_id": member,
                "group_type": self.group_type,
                "group_origin": self.group_origin,
                "split_indivisible": "true",
                "notes": self.notes,
            }
            for member in self.members
        ]


def build_groups(
    eligible_image_ids: list[str],
    confirmed_duplicates: dict[str, list[str]],
) -> list[Group]:
    """Partition the eligible images into indivisible split units.

    Args:
        eligible_image_ids: Every image in the modelling population.
        confirmed_duplicates: Manually confirmed groups, keyed by their phase 4B
            group identifier, each listing its member image ids.

    Returns:
        Every group, ordered by group id. Each eligible image appears in exactly
        one group.

    Raises:
        PopulationError: If a confirmed group is degenerate, names an ineligible
            image, or claims an image another group already claims.
    """
    eligible = set(eligible_image_ids)
    if len(eligible) != len(eligible_image_ids):
        msg = "Eligible image list contains a duplicate identifier"
        raise PopulationError(msg)

    groups: list[Group] = []
    claimed: dict[str, str] = {}
    for group_id in sorted(confirmed_duplicates):
        members = sorted(set(confirmed_duplicates[group_id]))
        if len(members) < 2:
            msg = f"Confirmed duplicate group {group_id!r} has {len(members)} member(s); need >= 2"
            raise PopulationError(msg)
        for member in members:
            if member not in eligible:
                msg = (
                    f"Confirmed duplicate group {group_id!r} names {member!r}, which is not "
                    "in the modelling population"
                )
                raise PopulationError(msg)
            if member in claimed:
                msg = f"Image {member!r} is claimed by both {claimed[member]!r} and {group_id!r}"
                raise PopulationError(msg)
            claimed[member] = group_id
        groups.append(
            Group(
                group_id=group_id,
                members=members,
                group_type=SEMANTIC_DUPLICATE,
                group_origin="PHASE_4B_HUMAN_AUDIT",
                notes="visually confirmed as the same content; must not be split apart",
            )
        )

    for image_id in sorted(eligible - set(claimed)):
        groups.append(
            Group(
                group_id=f"singleton-{image_id}",
                members=[image_id],
                group_type=SINGLETON,
                group_origin="DEFAULT_ONE_IMAGE_PER_GROUP",
            )
        )
    return sorted(groups, key=lambda group: group.group_id)


def group_features(
    group: Group,
    annotations_by_image: dict[str, list[AnnotationRecord]],
    classes: tuple[str, ...] = CLASS_ORDER,
) -> dict[str, Any]:
    """Summarise a group for the split design that comes next.

    Deliberately carries no provider split. The phase 4B audit rejected that
    split, and leaving the column in would let it bias an optimiser that has no
    reason to consult it.

    Args:
        group: The group to summarise.
        annotations_by_image: Eligible annotations keyed by source image id.
        classes: Canonical class order.

    Returns:
        One row of the group feature table.
    """
    instances: dict[str, int] = dict.fromkeys(classes, 0)
    zero_instance_images = 0
    for member in group.members:
        member_annotations = annotations_by_image.get(member, [])
        if not member_annotations:
            zero_instance_images += 1
        for annotation in member_annotations:
            if annotation.label in instances:
                instances[annotation.label] += 1

    row: dict[str, Any] = {
        "group_id": group.group_id,
        "image_count": len(group.members),
        "zero_instance_image_count": zero_instance_images,
        "group_type": group.group_type,
    }
    for name in classes:
        row[f"has_{name}"] = "true" if instances[name] else "false"
    for name in classes:
        row[f"instances_{name}"] = instances[name]
    row["total_instances"] = sum(instances.values())
    return row


def _digest(payload: Any) -> str:
    """Hash a JSON-serialisable payload deterministically.

    Args:
        payload: Content to hash.

    Returns:
        Its SHA-256 hex digest.
    """
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def fingerprint_population(
    images: list[ImageRecord],
    annotations: list[AnnotationRecord],
    groups: list[Group],
    class_map: dict[str, int],
) -> dict[str, str]:
    """Fingerprint the semantic content of the modelling population.

    Timestamps, file paths and run metadata are excluded on purpose: two runs of
    the same pipeline over the same data must produce the same fingerprint, and a
    change to any decision must change it.

    Args:
        images: Every source image record.
        annotations: Every annotation record.
        groups: Every group.
        class_map: The canonical class-to-index map.

    Returns:
        Component digests and the overall ``modeling_population_sha256``.
    """
    image_payload = [
        [record.source_image_id, record.status, record.action, record.reason, record.group_id]
        for record in sorted(images, key=lambda r: r.source_image_id)
    ]
    annotation_payload = [
        [
            record.source_image_id,
            record.annotation_id,
            record.label,
            record.action,
            record.geometry_origin,
            record.geometry_kind,
            _digest(record.segmentation),
            [round(v, 4) for v in record.bbox],
        ]
        for record in sorted(annotations, key=lambda r: (r.source_image_id, r.annotation_id))
    ]
    group_payload = [
        [group.group_id, group.group_type, group.members]
        for group in sorted(groups, key=lambda g: g.group_id)
    ]

    components = {
        "images_sha256": _digest(image_payload),
        "annotations_sha256": _digest(annotation_payload),
        "groups_sha256": _digest(group_payload),
        "class_map_sha256": _digest(class_map),
    }
    components["modeling_population_sha256"] = _digest(components)
    return components


def validate_population(
    images: list[ImageRecord],
    annotations: list[AnnotationRecord],
    groups: list[Group],
) -> list[str]:
    """Check the invariants a split design will rely on.

    Args:
        images: Every source image record.
        annotations: Every annotation record.
        groups: Every group.

    Returns:
        One description per broken invariant, empty when the population is sound.
    """
    problems: list[str] = []
    eligible = {record.source_image_id for record in images if record.status == ELIGIBLE}

    membership: dict[str, list[str]] = {}
    for group in groups:
        if not group.members:
            problems.append(f"group {group.group_id!r} has no members")
        if group.group_type == SEMANTIC_DUPLICATE and len(group.members) < 2:
            problems.append(f"semantic duplicate group {group.group_id!r} has fewer than 2 members")
        if group.group_type == SINGLETON and len(group.members) != 1:
            problems.append(f"singleton group {group.group_id!r} has {len(group.members)} members")
        for member in group.members:
            membership.setdefault(member, []).append(group.group_id)

    for image_id, owners in sorted(membership.items()):
        if len(owners) > 1:
            problems.append(f"image {image_id!r} belongs to groups {owners}")
        if image_id not in eligible:
            problems.append(f"group member {image_id!r} is not an eligible image")
    missing = sorted(eligible - set(membership))
    if missing:
        problems.append(f"{len(missing)} eligible image(s) belong to no group, e.g. {missing[:3]}")

    for record in annotations:
        if not record.is_eligible:
            continue
        if record.source_image_id not in eligible:
            problems.append(
                f"annotation {record.annotation_id!r} is eligible but its image "
                f"{record.source_image_id!r} is not"
            )
        # Any falsy geometry counts: None, an empty ring list and an empty RLE
        # would all reach the materialisation step and fail there instead.
        if not record.segmentation:
            problems.append(
                f"eligible annotation {record.annotation_id!r} on {record.source_image_id!r} "
                "has no segmentation geometry"
            )
        if record.label not in CLASS_ORDER:
            problems.append(
                f"annotation {record.annotation_id!r} carries class {record.label!r}, "
                "which is not in the canonical class set"
            )
        if record.area is not None and record.area <= 0:
            problems.append(
                f"eligible annotation {record.annotation_id!r} on {record.source_image_id!r} "
                f"has non-positive area {record.area}"
            )
    return problems
