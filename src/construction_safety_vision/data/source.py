"""The provider's source-image population: inventory, records and annotations.

Phase 3 established that the version-4 export contains 742 images but only 436
independent source images. Everything in this module works on the **436**, which
is the population every dataset statistic and every future split must be built
from.

Two provider surfaces are used:

* the search API, which lists source images with their split, original
  dimensions and per-class annotation counts;
* the image-details API, which returns the authoritative annotation geometry in
  *original image coordinates* - unlike the export, which is resized to 640x640
  and, for the train split, augmented.

Nothing here persists a credential, a URL or an embedding.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from typing import Any

from construction_safety_vision.data.roboflow import (
    SEARCH_MAX_PAGE_SIZE,
    SEARCH_PAGE_SIZE,
    DatasetCoordinates,
    RoboflowClient,
    RoboflowError,
)

GEOMETRY_TYPES = ("polygon", "mask")
"""Annotation geometry types the provider is known to emit."""


class InventoryError(RoboflowError):
    """Raised when the recovered inventory is not internally consistent."""


@dataclass(frozen=True)
class SourceAnnotation:
    """One annotated instance on a source image, in original coordinates.

    Attributes:
        annotation_id: Provider annotation identifier, unique within the image.
        label: Class name.
        geometry_type: ``polygon`` or ``mask``.
        x: Centre x reported by the provider.
        y: Centre y reported by the provider.
        width: Width reported by the provider.
        height: Height reported by the provider.
        point_count: Number of geometry vertices.
    """

    annotation_id: str
    label: str
    geometry_type: str
    x: float
    y: float
    width: float
    height: float
    point_count: int

    @classmethod
    def from_box(cls, box: dict[str, Any]) -> SourceAnnotation:
        """Build a record from a provider ``boxes`` entry.

        Args:
            box: One entry of ``image.annotation.boxes``.

        Returns:
            The parsed annotation.
        """
        points = box.get("points") or []
        return cls(
            annotation_id=str(box.get("id", "")),
            label=str(box.get("label", "")),
            geometry_type=str(box.get("type", "unknown")),
            x=float(box.get("x", 0.0)),
            y=float(box.get("y", 0.0)),
            width=float(box.get("width", 0.0)),
            height=float(box.get("height", 0.0)),
            point_count=len(points),
        )


@dataclass
class SourceImageRecord:
    """One source image as the provider describes it.

    Attributes:
        image_id: Provider image identifier. The stable key for everything.
        name: Original filename.
        split: Provider split assignment.
        width: Original width in pixels.
        height: Original height in pixels.
        created: Provider creation timestamp, milliseconds since the epoch.
        tags: Provider tags.
        annotation_count: Instances on this image.
        class_counts: Instances keyed by class name.
        owner: Provider owner identifier, needed to address the original image.
        geometry_types: Instance counts keyed by geometry type.
        annotations: Per-instance records, when details were fetched.
    """

    image_id: str
    name: str
    split: str
    width: int
    height: int
    created: int
    tags: list[str] = field(default_factory=list)
    annotation_count: int = 0
    class_counts: dict[str, int] = field(default_factory=dict)
    owner: str = ""
    geometry_types: dict[str, int] = field(default_factory=dict)
    annotations: list[SourceAnnotation] = field(default_factory=list)

    @property
    def aspect_ratio(self) -> float | None:
        """Width divided by height.

        Returns:
            The aspect ratio, or ``None`` when the height is unknown.
        """
        return self.width / self.height if self.height else None

    @property
    def pixels(self) -> int:
        """Total pixel count.

        Returns:
            ``width * height``.
        """
        return self.width * self.height

    @classmethod
    def from_search_result(cls, result: dict[str, Any]) -> SourceImageRecord:
        """Build a record from one search-API result.

        Args:
            result: One entry of the search response ``results`` list.

        Returns:
            The parsed record.

        Raises:
            InventoryError: If the result carries no image identifier.
        """
        image_id = str(result.get("id", "")).strip()
        if not image_id:
            msg = f"Search result has no image id: keys={sorted(result)}"
            raise InventoryError(msg)
        annotations = result.get("annotations") or {}
        return cls(
            image_id=image_id,
            name=str(result.get("name", "")),
            split=str(result.get("split", "")),
            width=int(result.get("width", 0) or 0),
            height=int(result.get("height", 0) or 0),
            created=int(result.get("created", 0) or 0),
            tags=[str(t) for t in (result.get("tags") or [])],
            annotation_count=int(annotations.get("count", 0) or 0),
            class_counts={str(k): int(v) for k, v in (annotations.get("classes") or {}).items()},
            owner=str(result.get("owner", "")),
        )

    def attach_details(self, image: dict[str, Any]) -> None:
        """Attach authoritative annotation geometry from the details API.

        Args:
            image: The ``image`` object returned by the details endpoint.
        """
        boxes = (image.get("annotation") or {}).get("boxes") or []
        self.annotations = [SourceAnnotation.from_box(box) for box in boxes]
        counts: dict[str, int] = {}
        for annotation in self.annotations:
            counts[annotation.geometry_type] = counts.get(annotation.geometry_type, 0) + 1
        self.geometry_types = dict(sorted(counts.items()))

    def manifest_entry(self) -> dict[str, Any]:
        """Serialise the record for the committed manifest.

        Geometry vertices are deliberately excluded: they are bulk data and live
        in the git-ignored interim layer. No URL, credential or local path is
        included.

        Returns:
            A JSON-serialisable mapping.
        """
        return {
            "image_id": self.image_id,
            "name": self.name,
            "split": self.split,
            "width": self.width,
            "height": self.height,
            "aspect_ratio": round(self.aspect_ratio, 6) if self.aspect_ratio else None,
            "pixels": self.pixels,
            "created": self.created,
            "tags": self.tags,
            "annotation_count": self.annotation_count,
            "class_counts": dict(sorted(self.class_counts.items())),
            "geometry_types": self.geometry_types,
            "detail_annotation_count": len(self.annotations),
        }

    def annotation_entries(self) -> list[dict[str, Any]]:
        """Serialise the per-instance geometry summary.

        Returns:
            One mapping per annotated instance.
        """
        return [{"image_id": self.image_id, **asdict(a)} for a in self.annotations]


@dataclass
class InventoryWalk:
    """Outcome of walking the provider's source inventory.

    Attributes:
        records: One record per unique source image, ordered by image id.
        declared_total: The total the provider reported.
        pages_fetched: Number of search pages requested.
        repeated_ids: Image ids the provider returned more than once across
            pages. Observed once in practice under heavy request load; recorded
            rather than hidden, because a repeat means offset pagination was
            momentarily inconsistent and something else may have been skipped.
    """

    records: list[SourceImageRecord] = field(default_factory=list)
    declared_total: int = 0
    pages_fetched: int = 0
    repeated_ids: list[str] = field(default_factory=list)

    @property
    def is_complete(self) -> bool:
        """Whether the walk recovered exactly the declared population.

        Returns:
            ``True`` when unique records equal the provider's declared total.
        """
        return bool(self.declared_total) and len(self.records) == self.declared_total


def collect_source_images(
    client: RoboflowClient,
    coordinates: DatasetCoordinates,
    *,
    page_size: int = SEARCH_MAX_PAGE_SIZE,
    max_pages: int = 100,
) -> InventoryWalk:
    """Walk the whole source inventory, page by page.

    Offset pagination on this provider has been observed to return a repeated id
    under load. Repeats are therefore tolerated and counted rather than treated
    as fatal, but the walk still fails if the number of *unique* records does not
    match the declared total - a repeat may mean another record was skipped, and
    silently returning 435 of 436 images would corrupt every later statistic.

    Args:
        client: Authenticated provider client.
        coordinates: Dataset identity.
        page_size: Records requested per page.
        max_pages: Safety bound so a misbehaving endpoint cannot loop forever.

    Returns:
        The walk outcome, including any repeated ids observed.

    Raises:
        InventoryError: If the unique record count does not match the total the
            provider declared.
    """
    walk = InventoryWalk()
    seen: dict[str, SourceImageRecord] = {}
    offset = 0
    for _ in range(max_pages):
        page = client.search_images_page(coordinates, limit=page_size, offset=offset)
        walk.pages_fetched += 1
        if not walk.declared_total:
            walk.declared_total = int(page.get("total", 0) or 0)
        results = page.get("results") or []
        if not results:
            # An empty page mid-sweep means the offset ran past the end; restart
            # the sweep rather than stopping, unless nothing is left to find.
            if walk.declared_total and len(seen) >= walk.declared_total:
                break
            if offset == 0:
                break
            offset = 0
            continue
        for result in results:
            record = SourceImageRecord.from_search_result(result)
            if record.image_id in seen:
                walk.repeated_ids.append(record.image_id)
                continue
            seen[record.image_id] = record
        if walk.declared_total and len(seen) >= walk.declared_total:
            break
        offset += len(results)
        if walk.declared_total and offset >= walk.declared_total:
            # Ordering is not stable across requests on this provider, so a
            # single sweep can miss records while repeating others. Sweep again
            # from the start and keep only what is new; the loop terminates as
            # soon as the declared population is complete.
            offset = 0

    walk.records = [seen[key] for key in sorted(seen)]
    if not walk.is_complete:
        msg = (
            f"Recovered {len(walk.records)} unique source records but the provider declared "
            f"{walk.declared_total} after {walk.pages_fetched} pages (repeats seen: "
            f"{len(walk.repeated_ids)}). Do not proceed with an incomplete population."
        )
        raise InventoryError(msg)
    return walk


def iter_source_images(
    client: RoboflowClient,
    coordinates: DatasetCoordinates,
    *,
    page_size: int = SEARCH_PAGE_SIZE,
) -> Iterator[SourceImageRecord]:
    """Yield every source image record.

    Args:
        client: Authenticated provider client.
        coordinates: Dataset identity.
        page_size: Records requested per page.

    Yields:
        One record per source image, ordered by image id.
    """
    yield from collect_source_images(client, coordinates, page_size=page_size).records


def split_counts(records: list[SourceImageRecord]) -> dict[str, int]:
    """Count records per provider split.

    Args:
        records: Source records.

    Returns:
        Counts keyed by split name, sorted.
    """
    counts: dict[str, int] = {}
    for record in records:
        counts[record.split] = counts.get(record.split, 0) + 1
    return dict(sorted(counts.items()))
