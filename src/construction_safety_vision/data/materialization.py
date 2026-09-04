"""Turning the frozen membership into task-ready data, without changing the data.

Phase 5D materialises what phase 5C.2 froze. The one property that matters is
that it adds nothing: the same source bytes, the same annotations, the same
class order, arranged into the layout a training pipeline expects. Anything this
step *changes* about the data is a defect, not a feature.

Three rules follow from that.

**Images are copied, never processed.** Byte-for-byte, verified by hashing both
sides. No resize, no crop, no re-encode, no EXIF rotation, no colour conversion.
A destination that already exists with different bytes is a hard failure, never a
silent overwrite, because it means two runs disagree about what the data is.

**Identifiers are global and content-derived.** COCO numeric ids come from a
deterministic ordering of the whole modelling population, not from the split
being written and not from filesystem traversal. That is what lets the holdout be
materialised later, by the same code, without renumbering anything that already
exists - and it is why the same image carries the same id in the detection view
and the segmentation view.

**The holdout is reachable only through the guard.** The split being materialised
is resolved by
:meth:`construction_safety_vision.data.split_freeze.FrozenSplits.image_ids`, so
``test`` needs both opt-ins here exactly as it does everywhere else. There is no
separate code path for it: the future final-evaluation run executes this same
function with the same configuration.

Nothing here trains, evaluates, converts geometry to a model-specific format, or
reads the provider's rejected split.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from construction_safety_vision.config import ConfigError, check_keys
from construction_safety_vision.data.split_freeze import (
    TEST,
    TRAIN,
    VALIDATION,
    FrozenSplits,
)
from construction_safety_vision.paths import long_path

CONFIG_SCHEMA_VERSION = 1
"""Schema version of ``configs/task_materialization.yaml``."""

DEVELOPMENT_SPLITS: tuple[str, ...] = (TRAIN, VALIDATION)
"""The splits a development run may materialise."""

TEST_DISABLED = "disabled_during_development"
"""The only test-materialisation policy accepted while models are unfrozen."""

COPIED = "COPIED"
"""A destination file that this run created."""

ALREADY_PRESENT = "ALREADY_PRESENT"
"""A destination file that already held the correct bytes."""

CHUNK_SIZE = 1 << 20
"""Read size used when hashing and copying image files."""


class MaterializationError(RuntimeError):
    """Raised when materialisation cannot proceed or would alter the data."""


@dataclass(frozen=True)
class Tolerances:
    """Numeric slack allowed by the validators, in pixels.

    Attributes:
        bbox_canvas_px: How far a derived box may fall outside the image canvas
            before it is rejected. A box derived from a rasterised RLE is
            quantised to whole pixels, so one pixel of slack is arithmetic, not
            error.
        polygon_coordinate_px: How far an emitted polygon coordinate may differ
            from the canonical one. Zero: polygons are copied, not recomputed, so
            any difference at all is a serialisation defect.
        bbox_agreement_px: How far the box derived here may differ from the box
            phase 5A measured from the same geometry.
    """

    bbox_canvas_px: float
    polygon_coordinate_px: float
    bbox_agreement_px: float

    def as_dict(self) -> dict[str, float]:
        """Serialise for the manifest and the configuration fingerprint.

        Returns:
            A JSON-serialisable mapping.
        """
        return {
            "bbox_canvas_px": self.bbox_canvas_px,
            "polygon_coordinate_px": self.polygon_coordinate_px,
            "bbox_agreement_px": self.bbox_agreement_px,
        }


@dataclass(frozen=True)
class MaterializationConfig:
    """Every choice the materialisation makes, declared in a file.

    None of these belong only in code: a reader auditing the datasets needs to
    see that boxes come from segmentation rather than from the provider, that
    images were copied rather than processed, and that the holdout was excluded
    by policy rather than by accident.

    Attributes:
        schema_version: Version of this configuration schema.
        source_representation: Which annotation state is being materialised.
        canonical_split_manifest: Repository-relative path to the frozen split.
        category_map_source: Repository-relative path to the frozen class map.
        development_splits: Splits a development run may write.
        test_materialization: Policy for the protected split.
        image_copy_mode: How image bytes are transferred.
        image_naming: How destination file names are derived.
        detection_bbox_source: Where detection boxes come from.
        segmentation_geometry_policy: What happens to canonical geometry.
        category_id_policy: How COCO category ids relate to the class map.
        output_format: Canonical task format.
        output_root: Repository-relative destination root.
        tolerances: Numeric slack allowed by the validators.
    """

    schema_version: int
    source_representation: str
    canonical_split_manifest: str
    category_map_source: str
    development_splits: tuple[str, ...]
    test_materialization: str
    image_copy_mode: str
    image_naming: str
    detection_bbox_source: str
    segmentation_geometry_policy: str
    category_id_policy: str
    output_format: str
    output_root: str
    tolerances: Tolerances

    def as_dict(self) -> dict[str, Any]:
        """Serialise the protocol for hashing and reporting.

        Returns:
            A JSON-serialisable mapping of the configuration content.
        """
        return {
            "schema_version": self.schema_version,
            "source_representation": self.source_representation,
            "canonical_split_manifest": self.canonical_split_manifest,
            "category_map_source": self.category_map_source,
            "development_splits": list(self.development_splits),
            "test_materialization": self.test_materialization,
            "image_copy_mode": self.image_copy_mode,
            "image_naming": self.image_naming,
            "detection_bbox_source": self.detection_bbox_source,
            "segmentation_geometry_policy": self.segmentation_geometry_policy,
            "category_id_policy": self.category_id_policy,
            "output_format": self.output_format,
            "output_root": self.output_root,
            "tolerances": self.tolerances.as_dict(),
        }

    def fingerprint(self) -> str:
        """Hash the protocol, so a dataset can name the rules that produced it.

        Returns:
            A SHA-256 hex digest over the configuration content only.
        """
        return digest(self.as_dict())


def digest(payload: Any) -> str:
    """Hash a JSON-serialisable payload deterministically.

    Args:
        payload: Content to hash.

    Returns:
        Its SHA-256 hex digest.
    """
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_materialization_config(path: str | Path) -> MaterializationConfig:
    """Load and validate the materialisation protocol.

    Parsing is strict: an unknown key raises rather than being ignored, so a typo
    cannot silently disable a policy the datasets are documented as following.

    Args:
        path: Configuration file.

    Returns:
        The parsed protocol.

    Raises:
        ConfigError: If the file is missing, is not valid YAML, or is malformed.
    """
    config_path = Path(path).expanduser()
    if not config_path.is_file():
        msg = f"Task-materialisation configuration not found: {config_path.name}"
        raise ConfigError(msg)
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        msg = f"Task-materialisation configuration is not valid YAML: {config_path.name} ({exc})"
        raise ConfigError(msg) from exc
    if not isinstance(raw, Mapping):
        msg = f"Task-materialisation configuration must be a mapping: {config_path.name}"
        raise ConfigError(msg)

    check_keys(
        raw,
        required=(
            "schema_version",
            "source_representation",
            "canonical_split_manifest",
            "category_map_source",
            "development_splits",
            "test_materialization",
            "image_copy_mode",
            "image_naming",
            "detection_bbox_source",
            "segmentation_geometry_policy",
            "category_id_policy",
            "output_format",
            "output_root",
            "tolerances",
        ),
        context="task_materialization",
    )
    splits = raw["development_splits"]
    if not isinstance(splits, list) or tuple(splits) != DEVELOPMENT_SPLITS:
        msg = (
            f"task_materialization.development_splits must be {list(DEVELOPMENT_SPLITS)}, "
            f"got {splits!r}. The protected split is never a development split."
        )
        raise ConfigError(msg)
    if raw["test_materialization"] != TEST_DISABLED:
        msg = (
            f"task_materialization.test_materialization must be {TEST_DISABLED!r} while the "
            f"models are unfrozen, got {raw['test_materialization']!r}"
        )
        raise ConfigError(msg)

    tolerances = raw["tolerances"]
    if not isinstance(tolerances, Mapping):
        msg = "task_materialization.tolerances: must be a mapping"
        raise ConfigError(msg)
    check_keys(
        tolerances,
        required=("bbox_canvas_px", "polygon_coordinate_px", "bbox_agreement_px"),
        context="task_materialization.tolerances",
    )
    try:
        parsed_tolerances = Tolerances(
            bbox_canvas_px=float(tolerances["bbox_canvas_px"]),
            polygon_coordinate_px=float(tolerances["polygon_coordinate_px"]),
            bbox_agreement_px=float(tolerances["bbox_agreement_px"]),
        )
    except (TypeError, ValueError) as exc:
        msg = f"task_materialization.tolerances: values must be numeric ({exc})"
        raise ConfigError(msg) from exc

    return MaterializationConfig(
        schema_version=int(raw["schema_version"]),
        source_representation=str(raw["source_representation"]),
        canonical_split_manifest=str(raw["canonical_split_manifest"]),
        category_map_source=str(raw["category_map_source"]),
        development_splits=DEVELOPMENT_SPLITS,
        test_materialization=str(raw["test_materialization"]),
        image_copy_mode=str(raw["image_copy_mode"]),
        image_naming=str(raw["image_naming"]),
        detection_bbox_source=str(raw["detection_bbox_source"]),
        segmentation_geometry_policy=str(raw["segmentation_geometry_policy"]),
        category_id_policy=str(raw["category_id_policy"]),
        output_format=str(raw["output_format"]),
        output_root=str(raw["output_root"]),
        tolerances=parsed_tolerances,
    )


@dataclass(frozen=True)
class CanonicalIds:
    """Deterministic COCO numeric ids over the whole modelling population.

    The ids are global rather than per split, which is what makes the holdout
    materialisable later without renumbering: the id an image carries in the
    training set is the id it would carry anywhere. They come from a sorted
    canonical ordering, never from directory traversal, so two machines agree.

    Attributes:
        image_ids: COCO image id keyed by source image id.
        annotation_ids: COCO annotation id keyed by
            ``(source_image_id, annotation_id)``.
    """

    image_ids: dict[str, int]
    annotation_ids: dict[tuple[str, str], int]

    def fingerprint(self) -> str:
        """Hash the id assignment.

        Returns:
            A SHA-256 hex digest over the sorted id mappings.
        """
        return digest(
            {
                "image_ids": [[key, self.image_ids[key]] for key in sorted(self.image_ids)],
                "annotation_ids": [
                    [key[0], key[1], self.annotation_ids[key]]
                    for key in sorted(self.annotation_ids)
                ],
            }
        )


def build_canonical_ids(
    source_image_ids: Iterable[str], annotation_keys: Iterable[tuple[str, str]]
) -> CanonicalIds:
    """Assign COCO numeric ids from a canonical sorted ordering.

    Ids start at 1 because a zero id is ambiguous in several COCO consumers.

    Args:
        source_image_ids: Every eligible source image id in the modelling
            population, in any order.
        annotation_keys: Every ``(source_image_id, annotation_id)`` pair in the
            modelling population, in any order.

    Returns:
        The id assignment.

    Raises:
        MaterializationError: If an id appears more than once.
    """
    images = sorted(source_image_ids)
    if len(images) != len(set(images)):
        msg = "the modelling population lists a source image id more than once"
        raise MaterializationError(msg)
    keys = sorted(annotation_keys)
    if len(keys) != len(set(keys)):
        msg = "the modelling population lists an annotation key more than once"
        raise MaterializationError(msg)
    return CanonicalIds(
        image_ids={image_id: index for index, image_id in enumerate(images, start=1)},
        annotation_ids={key: index for index, key in enumerate(keys, start=1)},
    )


@dataclass(frozen=True)
class PopulationView:
    """What the materialiser needs to know about the modelling population.

    Attributes:
        eligible: Every eligible source image id, sorted.
        dimensions: Declared ``(width, height)`` keyed by source image id.
        groups: Indivisible split unit keyed by source image id.
    """

    eligible: tuple[str, ...]
    dimensions: dict[str, tuple[int, int]]
    groups: dict[str, str]


POPULATION_FIELDS: tuple[str, ...] = (
    "source_image_id",
    "status",
    "width",
    "height",
    "group_id",
)
"""The only columns the materialiser reads from the population table.

Named exhaustively rather than consumed loosely. Any other column the table
carries - including a provider-split provenance column, should one ever be added
- is inert here by construction: split membership comes from the frozen manifest
and from nothing else.
"""


def load_population(rows: Iterable[Mapping[str, str]], *, eligible_status: str) -> PopulationView:
    """Read the modelling population's geometry and grouping metadata.

    Args:
        rows: Rows of ``reports/canonical_modeling_population.csv``.
        eligible_status: The status value marking an image as eligible.

    Returns:
        The parsed view.

    Raises:
        MaterializationError: If a row is malformed or an image repeats.
    """
    eligible: list[str] = []
    dimensions: dict[str, tuple[int, int]] = {}
    groups: dict[str, str] = {}
    for row in rows:
        if row["status"] != eligible_status:
            continue
        image_id = row["source_image_id"]
        if image_id in dimensions:
            msg = f"source image {image_id!r} appears more than once in the population table"
            raise MaterializationError(msg)
        try:
            dimensions[image_id] = (int(row["width"]), int(row["height"]))
        except (KeyError, TypeError, ValueError) as exc:
            msg = f"source image {image_id!r} has no usable width/height ({exc})"
            raise MaterializationError(msg) from exc
        groups[image_id] = row["group_id"]
        eligible.append(image_id)
    return PopulationView(eligible=tuple(sorted(eligible)), dimensions=dimensions, groups=groups)


def materialized_name(source_image_id: str, source_path: Path) -> str:
    """Return the destination file name for one source image.

    The canonical source image id is used rather than the provider's filename.
    Provider filenames are descriptive, long enough to hit the Windows path limit
    on this machine, and are not guaranteed unique across the population; the id
    is short, unique by construction and already the key every artifact joins on.

    Args:
        source_image_id: Canonical source image id.
        source_path: The file being copied, read only for its extension.

    Returns:
        The destination file name.

    Raises:
        MaterializationError: If the source file has no extension.
    """
    suffix = source_path.suffix.lower()
    if not suffix:
        msg = (
            f"source image {source_path.name!r} has no extension; cannot name it deterministically"
        )
        raise MaterializationError(msg)
    return f"{source_image_id}{suffix}"


def sha256_bytes(path: Path) -> str:
    """Hash a file's bytes.

    Args:
        path: File to hash.

    Returns:
        Its SHA-256 hex digest.
    """
    hasher = hashlib.sha256()
    with Path(long_path(path)).open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_SIZE), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


@dataclass(frozen=True)
class ImageCopy:
    """The outcome of materialising one image.

    Attributes:
        source_image_id: Canonical source image id.
        file_name: Destination file name, relative to its split directory.
        source_sha256: Digest of the source bytes.
        materialized_sha256: Digest of the destination bytes.
        action: :data:`COPIED` or :data:`ALREADY_PRESENT`.
        size_bytes: File size.
    """

    source_image_id: str
    file_name: str
    source_sha256: str
    materialized_sha256: str
    action: str
    size_bytes: int

    @property
    def byte_identical(self) -> bool:
        """Whether the destination holds exactly the source bytes.

        Returns:
            ``True`` when both digests agree.
        """
        return self.source_sha256 == self.materialized_sha256


def copy_image(source: Path, destination: Path, *, source_image_id: str) -> ImageCopy:
    """Copy one image byte-for-byte and prove that it was not altered.

    Both sides are hashed after the copy rather than trusting the copy call. The
    cost is one extra read; the benefit is that "byte-identical" is measured
    rather than assumed, which is the entire claim this phase makes about images.

    Args:
        source: Canonical source original.
        destination: Destination file.
        source_image_id: Canonical source image id, recorded in the result.

    Returns:
        The copy outcome.

    Raises:
        MaterializationError: If the source is missing, or a destination already
            exists holding different bytes, or the copy did not reproduce the
            source exactly.
    """
    if not Path(long_path(source)).is_file():
        msg = f"source image for {source_image_id!r} not found at {source.name}"
        raise MaterializationError(msg)
    source_hash = sha256_bytes(source)

    if Path(long_path(destination)).exists():
        existing = sha256_bytes(destination)
        if existing != source_hash:
            msg = (
                f"{destination.name} already exists with different bytes "
                f"({existing[:16]} != {source_hash[:16]}). Refusing to overwrite: two runs "
                "disagree about what this image is."
            )
            raise MaterializationError(msg)
        return ImageCopy(
            source_image_id=source_image_id,
            file_name=destination.name,
            source_sha256=source_hash,
            materialized_sha256=existing,
            action=ALREADY_PRESENT,
            size_bytes=Path(long_path(destination)).stat().st_size,
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    # copyfile, not copy2: metadata is not part of the data, and copying mtime
    # would make the destination differ between runs for no reason.
    shutil.copyfile(long_path(source), long_path(destination))
    written = sha256_bytes(destination)
    if written != source_hash:
        msg = (
            f"copying {source_image_id!r} did not reproduce the source bytes "
            f"({written[:16]} != {source_hash[:16]})"
        )
        raise MaterializationError(msg)
    return ImageCopy(
        source_image_id=source_image_id,
        file_name=destination.name,
        source_sha256=source_hash,
        materialized_sha256=written,
        action=COPIED,
        size_bytes=Path(long_path(destination)).stat().st_size,
    )


def image_content_fingerprint(copies: Sequence[ImageCopy]) -> str:
    """Fingerprint the image content of one split.

    Covers the bytes, not the filenames' order on disk, so it moves when an image
    changes and not when a directory is re-listed.

    Args:
        copies: The copies made for one split.

    Returns:
        A SHA-256 hex digest over the sorted ``(image id, digest)`` pairs.
    """
    return digest(sorted([copy.source_image_id, copy.materialized_sha256] for copy in copies))


def resolve_split_images(
    splits: FrozenSplits,
    split: str,
    *,
    purpose: str,
    allow_test: bool = False,
    env: dict[str, str] | None = None,
) -> tuple[str, ...]:
    """Resolve one split's image ids through the holdout guard.

    Every materialisation - development now, holdout later - goes through this
    one function, so the protected split cannot acquire a code path with weaker
    checks than the one the development splits use.

    Args:
        splits: The frozen split.
        split: Split being materialised.
        purpose: Short reason, recorded in the guard's error message.
        allow_test: Explicit in-code opt-in for the holdout.
        env: Environment mapping to read. Defaults to ``os.environ``.

    Returns:
        The split's source image ids, sorted.

    Raises:
        HoldoutViolationError: If the holdout is requested without both opt-ins.
    """
    return splits.image_ids(split, purpose=purpose, allow_test=allow_test, env=env)


def assert_development_split(split: str) -> str:
    """Reject a split that a development run must not materialise.

    Args:
        split: Split name.

    Returns:
        The split name.

    Raises:
        MaterializationError: If the split is the protected holdout or unknown.
    """
    if split == TEST:
        msg = (
            "test is the protected holdout and is not materialised by a development run. "
            "The final-evaluation phase materialises it, through the same code, with both "
            "holdout opt-ins present."
        )
        raise MaterializationError(msg)
    if split not in DEVELOPMENT_SPLITS:
        msg = f"unknown split {split!r}; expected one of {list(DEVELOPMENT_SPLITS)}"
        raise MaterializationError(msg)
    return split
